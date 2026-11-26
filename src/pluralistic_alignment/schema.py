"""Record identity shared by data, editing and analysis.

A chain is one (block, item, variety, model, arm) editing sequence; a cache key
adds the round t. Legacy records (`seed_id`, `condition`, no `block`) are mapped
onto this schema by `normalize_record`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

MAX_T = 5
CONTROL_VARIETY = "SAE"
DEFAULT_BLOCK = "wikipedia"
CONDITION_TO_ARM = {"neutral": "A0", "steering": "A1"}

CacheKey = tuple[str, str, str, str, str, int]
VariantKey = tuple[str, str, str]


class Marker(NamedTuple):
    """One dialect-feature marker of a round-0 variant.

    `surface`: the rule's inserted/substituted text (e.g. "becomed").
    `deletion`: the SAE words either side of a deleted span, joined across the
    gap (e.g. "in the Legislature" -> "in Legislature").
    """

    form: str
    kind: str = "surface"
    rule_type: str = ""


@dataclass(frozen=True, order=True)
class ChainKey:
    """One editing chain: (block, item_id, variety, model, arm)."""

    block: str
    item_id: str
    variety: str
    model: str
    arm: str

    @property
    def variant(self) -> VariantKey:
        """(block, item_id, variety): the round-0 text this chain starts from."""
        return (self.block, self.item_id, self.variety)

    def at(self, t: int) -> CacheKey:
        """Cache key for round t of this chain."""
        return (self.block, self.item_id, self.variety, self.model, self.arm, t)

    def fields(self) -> dict[str, str]:
        """Identity columns for a result row."""
        return {
            "block": self.block,
            "item_id": self.item_id,
            "variety": self.variety,
            "model": self.model,
            "arm": self.arm,
        }


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Copy of `raw` in the current schema (legacy field names mapped)."""
    record = dict(raw)
    if "item_id" not in record and "seed_id" in record:
        record["item_id"] = record.pop("seed_id")
    if "arm" not in record and "condition" in record:
        condition = record.pop("condition")
        record["arm"] = CONDITION_TO_ARM.get(condition, condition)
    record.setdefault("block", DEFAULT_BLOCK)
    return record


def item_sort_key(item_id: str) -> tuple[str, int]:
    """Sort `wiki-007` before `wiki-010`: prefix, then numeric suffix."""
    prefix, _, number = item_id.rpartition("-")
    return (prefix, int(number)) if number.isdigit() else (item_id, -1)
