"""Transform seed paragraphs into 5 English varieties + an SAE control chain.

Reads data/wiki/seeds.jsonl (written by `pa seeds`; override with --seeds-in)
and appends to data/wiki/variants.jsonl (--out): one record per (seed, variety) with the
Multi-VALUE output text and `executed_rules` feature ground truth. Must be
run from this directory's own venv (`uv run python transform_seeds.py`) --
see README.md for why Multi-VALUE is isolated here. See the experimental design
"Design" for the variety list, the SAE control chain, and the >=3-fired-
features-per-variety filtering criterion.

Each seed's 5-variety transform is independent of every other seed, so
--workers > 1 shards seeds across worker processes (each loads its own
spaCy/stanza pipeline once -- see _patch_shared_pipelines -- then processes
only its shard). --gpus round-robins workers across GPU ids (stanza's
coref/BERT model is the bottleneck and is much faster on GPU per doc; see
envs/multivalue/README.md "GPU usage" for the license note on NVIDIA's
proprietary CUDA runtime this pulls in -- approved 2026-07-12).
"""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "wiki"

# the design doc "Varieties (5) + SAE control" -- name -> Multi-VALUE class name.
VARIETY_CLASS_NAMES = {
    "Indian English": "IndianDialect",
    "Nigerian English": "NigerianDialect",
    "Colloquial Singapore English": "ColloquialSingaporeDialect",
    "African American Vernacular English": "AfricanAmericanVernacular",
    "Australian English": "AustralianDialect",
}

MIN_FIRED_FEATURES = 3


def _set_thread_limits(n_threads: int) -> None:
    """Must run before torch/spaCy/stanza are imported in this process --
    otherwise BLAS/torch already picked a thread count. With N worker
    processes on a shared machine, uncapped per-process thread pools
    oversubscribe the CPU and each other."""
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(n_threads)


def _patch_shared_pipelines() -> None:
    """Each Dialects subclass builds its own spaCy/stanza pipeline in
    __init__ (~expensive BERT-based coref model load); nothing about those
    pipelines is dialect-specific (they're only ever called to parse text),
    so cache and share a single instance across all 5 varieties within this
    process instead of loading it 5 times."""
    import spacy
    import stanza
    from multivalue import BaseDialect as base_dialect_module

    nlp_cache: dict[str, object] = {}
    orig_spacy_load = spacy.load

    def cached_spacy_load(name, *a, **kw):
        if name not in nlp_cache:
            nlp_cache[name] = orig_spacy_load(name, *a, **kw)
        return nlp_cache[name]

    coref_cache: dict[tuple, object] = {}
    orig_stanza_pipeline = stanza.Pipeline

    def cached_stanza_pipeline(*a, **kw):
        key = (a, tuple(sorted(kw.items())))
        if key not in coref_cache:
            coref_cache[key] = orig_stanza_pipeline(*a, **kw)
        return coref_cache[key]

    base_dialect_module.spacy.load = cached_spacy_load
    base_dialect_module.stanza.Pipeline = cached_stanza_pipeline


def load_seeds(path: Path) -> list[dict]:
    seeds = []
    with path.open() as f:
        for line in f:
            seeds.append(json.loads(line))
    return seeds


def load_done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            done.add((record["seed_id"], record["variety"]))
    return done


def transform(seeds: list[dict], seed: int, done: set[tuple[str, str]], out_f) -> None:
    from multivalue import Dialects

    _patch_shared_pipelines()
    dialects = {
        name: getattr(Dialects, cls_name)(seed=seed)
        for name, cls_name in VARIETY_CLASS_NAMES.items()
    }

    for seed_record in seeds:
        seed_id = seed_record["seed_id"]
        abstract = seed_record["abstract"]

        if (seed_id, "SAE") not in done:
            out_f.write(
                json.dumps(
                    {
                        "seed_id": seed_id,
                        "variety": "SAE",
                        "text": abstract,
                        "executed_rules": {},
                        "n_features_fired": 0,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out_f.flush()

        for variety_name, dialect in dialects.items():
            if (seed_id, variety_name) in done:
                continue
            try:
                transformed = dialect.convert_sae_to_dialect(abstract)
                executed_rules = dict(dialect.executed_rules)
            except Exception as exc:
                # raise on specific inputs (e.g. abbreviation periods causing
                # a spaCy/stanza sentence-boundary mismatch); one bad seed
                # must not abort the whole run.
                record = {
                    "seed_id": seed_id,
                    "variety": variety_name,
                    "text": None,
                    "executed_rules": {},
                    "n_features_fired": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"FAILED {seed_id}/{variety_name}: {record['error']}")
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                continue
            record = {
                "seed_id": seed_id,
                "variety": variety_name,
                "text": transformed,
                "executed_rules": {
                    f"{k[0]}-{k[1]}": v for k, v in executed_rules.items()
                },
                "n_features_fired": len(executed_rules),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()


def _worker(
    shard_seeds: list[dict],
    seed_value: int,
    done: set,
    n_threads: int,
    part_path: Path,
    gpu_id: int | None,
) -> Path:
    _set_thread_limits(n_threads)
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    with part_path.open("w") as out_f:
        transform(shard_seeds, seed=seed_value, done=done, out_f=out_f)
    return part_path


def _chunk(items: list, n: int) -> list[list]:
    n = max(1, min(n, len(items))) if items else 1
    chunks = [[] for _ in range(n)]
    for i, item in enumerate(items):
        chunks[i % n].append(item)
    return [c for c in chunks if c]


def run(
    seeds: list[dict],
    seed_value: int,
    done: set,
    out_path: Path,
    workers: int,
    threads_per_worker: int,
    gpu_ids: list[int] | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if workers <= 1:
        gpu_id = gpu_ids[0] if gpu_ids else None
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        with out_path.open("a") as out_f:
            transform(seeds, seed=seed_value, done=done, out_f=out_f)
        return

    shards = _chunk(seeds, workers)
    part_paths = [out_path.with_suffix(f".part{i}.jsonl") for i in range(len(shards))]
    # round-robin assign shards across the given GPUs (or None => CPU-only)
    shard_gpu_ids = (
        [gpu_ids[i % len(gpu_ids)] for i in range(len(shards))]
        if gpu_ids
        else [None] * len(shards)
    )
    print(
        f"running {len(shards)} workers ({threads_per_worker} threads each) "
        f"over {len(seeds)} seeds, GPUs: {shard_gpu_ids}"
    )

    ctx = mp.get_context("spawn")  # spawn: fresh interpreter per worker, avoids
    # re-using a parent process's already-initialized torch/BLAS thread pool
    with ctx.Pool(processes=len(shards)) as pool:
        results = pool.starmap(
            _worker,
            [
                (shard, seed_value, done, threads_per_worker, part_path, gpu_id)
                for shard, part_path, gpu_id in zip(
                    shards, part_paths, shard_gpu_ids, strict=True
                )
            ],
        )

    with out_path.open("a") as out_f:
        for part_path in results:
            with part_path.open() as part_f:
                out_f.write(part_f.read())
            part_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds-in", type=Path, default=DATA_DIR / "seeds.jsonl")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "variants.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel worker processes, sharded by seed (each independent)",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=4,
        help="BLAS/torch thread cap per worker, to avoid oversubscribing the "
        "machine when --workers > 1",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="comma-separated GPU ids to round-robin workers across (e.g. "
        "'0,1,2,3'); omit for CPU-only. stanza's coref/BERT model is the "
        "bottleneck and is ~15-20x faster on GPU than CPU per doc.",
    )
    args = parser.parse_args()
    gpu_ids = [int(g) for g in args.gpus.split(",")] if args.gpus else None

    seeds = load_seeds(args.seeds_in)
    done = load_done_keys(args.out)
    pending = [
        s
        for s in seeds
        if not all((s["seed_id"], v) in done for v in VARIETY_CLASS_NAMES)
    ]
    print(
        f"{len(seeds)} seeds loaded; {len(done)} (seed, variety) pairs already "
        f"done; {len(pending)} seeds have pending work"
    )

    if pending:
        run(
            pending,
            seed_value=args.seed,
            done=done,
            out_path=args.out,
            workers=args.workers,
            threads_per_worker=args.threads_per_worker,
            gpu_ids=gpu_ids,
        )

    below_threshold, errored = _seeds_needing_attention(
        args.out, seeds, MIN_FIRED_FEATURES
    )
    if errored:
        print(
            f"{len(errored)}/{len(seeds)} seeds hit a transform error: "
            f"{sorted(errored)}"
        )
    if below_threshold:
        print(
            f"{len(below_threshold)}/{len(seeds)} seeds have a variety with "
            f"<{MIN_FIRED_FEATURES} features fired (the design doc's resample "
            f"threshold): {sorted(below_threshold)}"
        )
    if not errored and not below_threshold:
        print(f"all seeds meet the >={MIN_FIRED_FEATURES}-features-fired threshold")


def _seeds_needing_attention(
    out_path: Path, seeds: list[dict], min_fired: int
) -> tuple[set[str], set[str]]:
    fired_counts: dict[str, list[int]] = {s["seed_id"]: [] for s in seeds}
    errored: set[str] = set()
    with out_path.open() as f:
        for line in f:
            record = json.loads(line)
            if record["variety"] == "SAE":
                continue
            if record["n_features_fired"] is None:
                errored.add(record["seed_id"])
                continue
            fired_counts.setdefault(record["seed_id"], []).append(
                record["n_features_fired"]
            )
    below_threshold = {
        seed_id
        for seed_id, counts in fired_counts.items()
        if counts and min(counts) < min_fired
    }
    return below_threshold, errored


if __name__ == "__main__":
    main()
