"""Wikipedia seed paragraphs from `wikimedia/structured-wikipedia` (CC BY-SA 4.0).

Streams the dataset (no dump download), keeps prose-only abstracts of 30 to 120
words, and samples in a fixed shuffled order (seed 42 in the experiment). Calling
again with `existing_titles` walks the same order and returns the next batch,
which is how seeds 31 to 50 extend the original 1 to 30.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

DATASET_NAME = "wikimedia/structured-wikipedia"
DATASET_CONFIG = "enwiki_namespace_0"
# Hub revision of the July 2026 sampling (data/wiki/seeds.jsonl). The shuffled
# order also depends on the `datasets` version, pinned in pyproject.toml.
DATASET_REVISION = "417c267bb457fa645c22eb3b5c77764963194c70"
DATASET_LICENSE = "CC BY-SA 4.0"

_MARKUP_ARTIFACT_RE = re.compile(r"\{\{|\}\}|\[\[|\]\]|\|\||=\s*\{")
_LIST_TITLE_RE = re.compile(
    r"^(list of|index of|timeline of|glossary of)", re.IGNORECASE
)
_DISAMBIGUATION_RE = re.compile(r"may (also |often )?refer to", re.IGNORECASE)

StreamFactory = Callable[[int, int], Iterable[dict[str, Any]]]


@dataclass(frozen=True)
class SampleSettings:
    """Filter and sampling knobs (defaults)."""

    min_words: int = 30
    max_words: int = 120
    seed: int = 42
    shuffle_buffer_size: int = 10_000
    max_scanned: int = 200_000


def is_prose_seed(name: str, abstract: str, min_words: int, max_words: int) -> bool:
    """True if `abstract` is prose (no list/disambiguation/markup) within bounds."""
    if not abstract:
        return False
    if _LIST_TITLE_RE.match(name.strip()):
        return False
    if _DISAMBIGUATION_RE.search(abstract):
        return False
    if _MARKUP_ARTIFACT_RE.search(abstract):
        return False
    return min_words <= len(abstract.split()) <= max_words


def stream_wikipedia(seed: int, shuffle_buffer_size: int) -> Iterable[dict[str, Any]]:
    """Seeded, shuffled stream over English Wikipedia articles."""
    from datasets import load_dataset  # noqa: PLC0415 - heavy import, only when sampling

    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        streaming=True,
        revision=DATASET_REVISION,
    )
    return ds.shuffle(seed=seed, buffer_size=shuffle_buffer_size)


def _seed_record(
    row: dict[str, Any], seed_id: str, retrieved_at: str
) -> dict[str, Any]:
    abstract = (row.get("abstract") or "").strip()
    date_modified = row.get("date_modified")
    return {
        "seed_id": seed_id,
        "title": row.get("name") or "",
        "url": row.get("url"),
        "abstract": abstract,
        "word_count": len(abstract.split()),
        "date_modified": (
            date_modified.isoformat()
            if isinstance(date_modified, datetime)
            else date_modified
        ),
        "source_dataset": DATASET_NAME,
        "source_config": DATASET_CONFIG,
        "source_revision": DATASET_REVISION,
        "license": DATASET_LICENSE,
        "retrieved_at": retrieved_at,
    }


def sample_seeds(
    n_seeds: int,
    settings: SampleSettings,
    existing_titles: frozenset[str] = frozenset(),
    id_start: int = 0,
    stream: StreamFactory = stream_wikipedia,
) -> list[dict[str, Any]]:
    """Sample `n_seeds` new prose seeds, ids continuing from `id_start`."""
    retrieved_at = datetime.now(UTC).isoformat()
    collected: list[dict[str, Any]] = []
    scanned = 0
    for row in stream(settings.seed, settings.shuffle_buffer_size):
        scanned += 1
        if scanned > settings.max_scanned:
            break
        name = row.get("name") or ""
        if name in existing_titles:
            continue
        abstract = (row.get("abstract") or "").strip()
        if not is_prose_seed(name, abstract, settings.min_words, settings.max_words):
            continue
        seed_id = f"wiki-{id_start + len(collected):03d}"
        collected.append(_seed_record(row, seed_id, retrieved_at))
        if len(collected) >= n_seeds:
            break

    if len(collected) < n_seeds:
        raise RuntimeError(
            f"only found {len(collected)}/{n_seeds} new seeds after scanning "
            f"{scanned} rows; raise max_scanned or relax the word-count filter"
        )
    return collected
