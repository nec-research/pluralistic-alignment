"""Append-only JSONL cache of edits, keyed by (block, item, variety, model, arm, t).

Reads legacy files too (via `normalize_record`). Error rows (`text: null`) are
not treated as done, so a failed call is retried on resume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Any

from pluralistic_alignment.data.variants import iter_jsonl
from pluralistic_alignment.schema import CacheKey, normalize_record


def record_key(record: dict[str, Any]) -> CacheKey:
    """Cache key of a normalized record."""
    return (
        record["block"],
        record["item_id"],
        record["variety"],
        record["model"],
        record["arm"],
        record["t"],
    )


def load_done(path: Path) -> dict[CacheKey, str]:
    """Cached successful outputs by key; empty if `path` does not exist."""
    if not path.exists():
        return {}
    done: dict[CacheKey, str] = {}
    for raw in iter_jsonl(path):
        record = normalize_record(raw)
        if record["text"] is not None:
            done[record_key(record)] = record["text"]
    return done


def append_record(out: IO[str], record: dict[str, Any]) -> None:
    """Write one record and flush, so an interrupted run loses nothing."""
    out.write(json.dumps(record, ensure_ascii=False) + "\n")
    out.flush()
