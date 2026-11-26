"""Round-0 variants: one record per (block, item, variety) with marker ground truth.

Written by `envs/multivalue/transform_seeds.py` for the English blocks. Records
whose transform failed have `text: null` and are skipped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from pluralistic_alignment.schema import (
    CONTROL_VARIETY,
    Marker,
    VariantKey,
    normalize_record,
)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Parsed JSON objects, one per non-empty line."""
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def trackable_markers(record: dict[str, Any]) -> list[str]:
    """Surface markers of the fired rules that have a non-empty value.

    Deletion rules (e.g. zero copula) fire with an empty value and are covered
    separately by `deletion_markers`.
    """
    return [
        rule["value"] for rule in record["executed_rules"].values() if rule.get("value")
    ]


_WORD = re.compile(r"\S+")


def _span(key: str) -> tuple[int, int]:
    start, end = key.split("-")
    return int(start), int(end)


def deletion_markers(rules: Mapping[str, Any], sae_text: str) -> list[Marker]:
    """One context marker per deletion rule (empty value, non-empty SAE span).

    Rule keys are `start-end` character spans in the SAE text. The marker is the
    SAE text from the nearest surviving word before the span to the nearest
    surviving word after it, with all deleted spans of the record removed
    ("serving in the Legislature" -> "serving Legislature"). Deletions at the
    text edge have one-sided context only and are skipped. Rules with an empty
    value that are not true deletions (e.g. reorderings) yield markers absent
    from the round-0 text; retention scoring then treats them as unmeasurable.
    """
    deletions = [
        (_span(key), rule.get("type", ""))
        for key, rule in rules.items()
        if not rule.get("value")
    ]
    deletions = [
        ((a, b), typ)
        for (a, b), typ in sorted(deletions)
        if b > a and sae_text[a:b].strip()
    ]
    gaps = [span for span, _ in deletions]
    words = [
        (m.start(), m.end())
        for m in _WORD.finditer(sae_text)
        if not any(a <= m.start() and m.end() <= b for a, b in gaps)
    ]
    out: list[Marker] = []
    for (a, b), typ in deletions:
        left = [w for w in words if w[1] <= a]
        right = [w for w in words if w[0] >= b]
        if not left or not right:
            continue
        lo, hi = left[-1][0], right[0][1]
        form = " ".join(sae_text[s:e] for s, e in words if lo <= s and e <= hi)
        out.append(Marker(form, "deletion", typ))
    return out


def load_markers(path: Path) -> dict[VariantKey, list[Marker]]:
    """Surface + deletion markers per (block, item_id, variety); SAE gets none."""
    records = [normalize_record(raw) for raw in iter_jsonl(path)]
    records = [r for r in records if r["text"] is not None]
    sae = {
        (r["block"], r["item_id"]): r["text"]
        for r in records
        if r["variety"] == CONTROL_VARIETY
    }
    markers: dict[VariantKey, list[Marker]] = {}
    for record in records:
        key = (record["block"], record["item_id"], record["variety"])
        rules = record.get("executed_rules") or {}
        surface = [
            Marker(rule["value"], "surface", rule.get("type", ""))
            for rule in rules.values()
            if rule.get("value")
        ]
        sae_text = sae.get((record["block"], record["item_id"]))
        deleted = deletion_markers(rules, sae_text) if sae_text else []
        markers[key] = [] if record["variety"] == CONTROL_VARIETY else surface + deleted
    return markers


def load_variants(
    path: Path,
) -> tuple[dict[VariantKey, str], dict[VariantKey, list[str]]]:
    """(round-0 text, trackable markers), both keyed by (block, item_id, variety)."""
    texts: dict[VariantKey, str] = {}
    markers: dict[VariantKey, list[str]] = {}
    for raw in iter_jsonl(path):
        record = normalize_record(raw)
        if record["text"] is None:
            continue
        key = (record["block"], record["item_id"], record["variety"])
        texts[key] = record["text"]
        markers[key] = trackable_markers(record)
    return texts, markers


def load_seed_texts(path: Path, block: str = "wikipedia") -> dict[tuple[str, str], str]:
    """(block, item_id) -> original standard text, from a seeds file."""
    seeds: dict[tuple[str, str], str] = {}
    for raw in iter_jsonl(path):
        record = normalize_record(raw)
        seeds[(record.get("block", block), record["item_id"])] = record["abstract"]
    return seeds
