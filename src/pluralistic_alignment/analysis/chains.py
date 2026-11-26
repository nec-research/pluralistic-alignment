"""Assemble complete chains [t0, t1, ..., t_max] from variants + cached edits."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from pluralistic_alignment.data.variants import iter_jsonl
from pluralistic_alignment.schema import MAX_T, ChainKey, VariantKey, normalize_record


def load_rewrites(paths: Iterable[Path]) -> dict[ChainKey, dict[int, str]]:
    """{chain: {t: text}} from one or more edit caches; error rows skipped."""
    by_chain: dict[ChainKey, dict[int, str]] = defaultdict(dict)
    for path in paths:
        for raw in iter_jsonl(path):
            record = normalize_record(raw)
            if record["text"] is None:
                continue
            key = ChainKey(
                record["block"],
                record["item_id"],
                record["variety"],
                record["model"],
                record["arm"],
            )
            by_chain[key][record["t"]] = record["text"]
    return dict(by_chain)


def build_full_chains(
    t0_texts: Mapping[VariantKey, str],
    rewrites: Mapping[ChainKey, dict[int, str]],
    max_t: int = MAX_T,
) -> dict[ChainKey, list[str]]:
    """Chains complete through `max_t`; incomplete ones are dropped and counted."""
    complete: dict[ChainKey, list[str]] = {}
    n_dropped = 0
    for key, by_t in rewrites.items():
        if key.variant not in t0_texts:
            continue
        if not all(t in by_t for t in range(1, max_t + 1)):
            n_dropped += 1
            continue
        complete[key] = [t0_texts[key.variant]] + [by_t[t] for t in range(1, max_t + 1)]
    if n_dropped:
        print(f"dropped {n_dropped} incomplete chains (missing a t in 1..{max_t})")
    return complete
