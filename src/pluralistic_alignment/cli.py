"""`pa` command line: seeds, edit, report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from pluralistic_alignment.paths import DATA_DIR, RESULTS_DIR, WIKI_DATA_DIR


def _cmd_seeds(args: argparse.Namespace) -> None:
    from pluralistic_alignment.data.variants import iter_jsonl  # noqa: PLC0415
    from pluralistic_alignment.data.wikipedia import SampleSettings, sample_seeds  # noqa: PLC0415
    from pluralistic_alignment.schema import item_sort_key  # noqa: PLC0415

    existing: list[dict] = list(iter_jsonl(args.out)) if args.out.exists() else []
    if existing and not args.append:
        raise SystemExit(f"{args.out} exists; pass --append to add new seeds")
    ids = [item_sort_key(r["seed_id"])[1] for r in existing]
    seeds = sample_seeds(
        n_seeds=args.n,
        settings=SampleSettings(seed=args.seed),
        existing_titles=frozenset(r["title"] for r in existing),
        id_start=max(ids, default=-1) + 1,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in seeds)
    print(f"wrote {len(seeds)} seeds to {args.out}", flush=True)
    # The HF streaming reader leaves native prefetch threads running; letting the
    # interpreter finalize can abort (SIGABRT, "PyGILState_Release ... finalizing")
    # after the file is written. Everything is flushed and closed, so exit now.
    sys.stderr.flush()
    os._exit(0)


def _cmd_edit(args: argparse.Namespace) -> None:
    from pluralistic_alignment.data.selection import select_items  # noqa: PLC0415
    from pluralistic_alignment.data.variants import load_variants  # noqa: PLC0415
    from pluralistic_alignment.editing.arms import build_arm  # noqa: PLC0415
    from pluralistic_alignment.editing.clients import CallSettings, build_models  # noqa: PLC0415
    from pluralistic_alignment.editing.loop import RunPlan, run  # noqa: PLC0415
    from pluralistic_alignment.settings import load_models_config  # noqa: PLC0415

    items = set(select_items(args.variants, args.n_items, args.min_markers))
    texts, _ = load_variants(args.variants)
    chains = {k: v for k, v in texts.items() if (k[0], k[1]) in items}
    models = build_models(load_models_config())
    if args.models:
        wanted = set(args.models.split(","))
        models = [m for m in models if m.name in wanted]
        if missing := wanted - {m.name for m in models}:
            raise SystemExit(f"unknown model(s): {sorted(missing)}")
    plan = RunPlan(chains, models, [build_arm(a) for a in args.arms.split(",")])
    settings = CallSettings(args.temperature, args.max_retries, args.sleep_between)
    run(plan, args.out, settings)


def _cmd_report(args: argparse.Namespace) -> None:
    from pluralistic_alignment.analysis.report import (  # noqa: PLC0415
        ReportInputs,
        write_report,
    )
    from pluralistic_alignment.metrics.semantic import load_embedder  # noqa: PLC0415

    inputs = ReportInputs(
        args.seeds, args.variants, args.rewrites, args.retention_matcher
    )
    write_report(inputs, args.results_dir, load_embedder(args.embedder))


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with one subcommand per pipeline stage."""
    parser = argparse.ArgumentParser(prog="pa", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    seeds = sub.add_parser("seeds", help="sample Wikipedia seed paragraphs")
    seeds.add_argument("--n", type=int, default=50)
    seeds.add_argument("--seed", type=int, default=42)
    seeds.add_argument(
        "--out", type=Path, default=DATA_DIR / "items" / "wikipedia_seeds.jsonl"
    )
    seeds.add_argument("--append", action="store_true")
    seeds.set_defaults(func=_cmd_seeds)

    edit = sub.add_parser("edit", help="iterated editing over selected items")
    edit.add_argument("--variants", type=Path, default=WIKI_DATA_DIR / "variants.jsonl")
    edit.add_argument("--out", type=Path, required=True)
    edit.add_argument("--arms", default="A0", help="comma-separated, e.g. A0,A1")
    edit.add_argument("--n-items", type=int, default=30)
    edit.add_argument("--min-markers", type=int, default=3)
    edit.add_argument("--models", default=None, help="comma-separated subset")
    edit.add_argument("--temperature", type=float, default=0.7)
    edit.add_argument("--max-retries", type=int, default=3)
    edit.add_argument("--sleep-between", type=float, default=0.0)
    edit.set_defaults(func=_cmd_edit)

    report = sub.add_parser("report", help="metrics from cached edits")
    report.add_argument("--seeds", type=Path, default=WIKI_DATA_DIR / "seeds.jsonl")
    report.add_argument(
        "--variants", type=Path, default=WIKI_DATA_DIR / "variants.jsonl"
    )
    report.add_argument("--rewrites", type=Path, nargs="+", required=True)
    report.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "report")
    report.add_argument("--embedder", default="Qwen/Qwen3-Embedding-0.6B")
    report.add_argument(
        "--retention-matcher",
        choices=["v3", "v0"],
        default="v3",
        help="v3: whole-token, excess over SAE, surface+deletion markers; "
        "v0: legacy substring+fuzzy (July 2026 numbers)",
    )
    report.set_defaults(func=_cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the `pa` script."""
    args = build_parser().parse_args(argv)
    args.func(args)
