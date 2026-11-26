"""Tests for schema: record identity, normalization, sorting."""

from pluralistic_alignment.schema import (
    CONDITION_TO_ARM,
    ChainKey,
    Marker,
    item_sort_key,
    normalize_record,
)


def test_chain_key_variant() -> None:
    """ChainKey.variant returns (block, item_id, variety)."""
    k = ChainKey("wikipedia", "wiki-001", "AAE", "gpt-5.4-mini", "A0")
    assert k.variant == ("wikipedia", "wiki-001", "AAE")


def test_chain_key_at() -> None:
    """ChainKey.at returns a 6-tuple cache key."""
    k = ChainKey("wikipedia", "wiki-001", "AAE", "gpt-5.4-mini", "A0")
    assert k.at(3) == ("wikipedia", "wiki-001", "AAE", "gpt-5.4-mini", "A0", 3)


def test_chain_key_fields() -> None:
    """ChainKey.fields returns identity dict."""
    k = ChainKey("wikipedia", "wiki-002", "IndE", "gemma-3-4b-it", "A1")
    assert k.fields() == {
        "block": "wikipedia",
        "item_id": "wiki-002",
        "variety": "IndE",
        "model": "gemma-3-4b-it",
        "arm": "A1",
    }


def test_normalize_record_legacy_fields() -> None:
    """Legacy records (seed_id, condition) are mapped to current schema."""
    raw = {"seed_id": "wiki-005", "condition": "neutral", "text": "hello", "t": 1}
    result = normalize_record(raw)
    assert result["item_id"] == "wiki-005"
    assert result["arm"] == "A0"
    assert result["block"] == "wikipedia"
    assert "seed_id" not in result


def test_normalize_record_already_current() -> None:
    """Records already in current schema pass through unchanged."""
    raw = {"item_id": "wiki-005", "arm": "A1", "block": "pmc", "text": "hi", "t": 2}
    result = normalize_record(raw)
    assert result == raw


def test_normalize_record_unknown_condition() -> None:
    """Unknown condition strings are kept as-is for the arm."""
    raw = {"seed_id": "wiki-001", "condition": "custom_arm", "text": "x", "t": 1}
    result = normalize_record(raw)
    assert result["arm"] == "custom_arm"


def test_item_sort_key_numeric() -> None:
    """wiki-007 sorts before wiki-010."""
    assert item_sort_key("wiki-007") < item_sort_key("wiki-010")


def test_item_sort_key_non_numeric() -> None:
    """Non-numeric suffixes get -1 as number."""
    prefix, num = item_sort_key("misc-abc")
    assert prefix == "misc-abc"
    assert num == -1


def test_marker_defaults() -> None:
    """Marker has sensible defaults for kind and rule_type."""
    m = Marker(form="becomed")
    assert m.kind == "surface"
    assert m.rule_type == ""


def test_condition_mapping() -> None:
    """CONDITION_TO_ARM maps neutral->A0, steering->A1."""
    assert CONDITION_TO_ARM == {"neutral": "A0", "steering": "A1"}
