"""Item selection: keep an item only if every variety fires enough trackable markers.

Selection rule: each of the five non-control varieties must fire at least three
trackable markers; items are taken in id order (the seeded sampling order), so a
failing item is replaced by the next one rather than truncating the set.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection
from pathlib import Path

from pluralistic_alignment.data.variants import iter_jsonl, trackable_markers
from pluralistic_alignment.schema import (
    CONTROL_VARIETY,
    item_sort_key,
    normalize_record,
)

MIN_TRACKABLE_MARKERS = 3
ENGLISH_VARIETIES = (
    "Indian English",
    "Nigerian English",
    "Colloquial Singapore English",
    "African American Vernacular English",
    "Australian English",
)


def select_items(
    variants_path: Path,
    n_items: int,
    min_markers: int = MIN_TRACKABLE_MARKERS,
    varieties: Collection[str] = ENGLISH_VARIETIES,
) -> list[tuple[str, str]]:
    """First `n_items` qualifying (block, item_id) pairs in id order."""
    counts: dict[tuple[str, str], dict[str, int | None]] = defaultdict(dict)
    for raw in iter_jsonl(variants_path):
        record = normalize_record(raw)
        if record["variety"] == CONTROL_VARIETY:
            continue
        n_trackable = (
            len(trackable_markers(record))
            if record.get("n_features_fired") is not None
            else None
        )
        counts[(record["block"], record["item_id"])][record["variety"]] = n_trackable

    qualifying = [
        key
        for key in sorted(counts, key=lambda k: (k[0], *item_sort_key(k[1])))
        if all(
            (c := counts[key].get(v)) is not None and c >= min_markers
            for v in varieties
        )
    ]
    if len(qualifying) < n_items:
        print(
            f"WARNING: only {len(qualifying)} items have >={min_markers} trackable "
            f"markers in every variety, fewer than the {n_items} requested"
        )
    return qualifying[:n_items]
