"""Feature retention: V3 matcher (default), deletion markers, legacy V0."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pluralistic_alignment.data.variants import deletion_markers, load_markers
from pluralistic_alignment.metrics.variety.markers import (
    compute_retention,
    marker_count,
    normalize_tokens,
    retention_v3,
)
from pluralistic_alignment.schema import ChainKey, Marker

SAE = "He started his career and remained for several years. It is one of the films."
T0 = "He start his career and remain for several year. It is one of the films."
MARKERS = ["start", "remain", "year"]
V0_OVERCOUNT_FLOOR = 0.5


# --- token matching -------------------------------------------------------


def test_normalize_tokens_drops_case_and_punctuation() -> None:
    """Normalize tokens drops case and punctuation."""
    assert (
        normalize_tokens("Assembly, NA\N{NON-BREAKING HYPHEN}92  (2008)!")
        == "assembly na-92 2008"
    )
    assert normalize_tokens("It\N{RIGHT SINGLE QUOTATION MARK}s  here") == "it's here"


def test_marker_count_is_whole_word() -> None:
    """Marker count is whole word."""
    assert marker_count("direct", "He directed it") == 0
    assert marker_count("one", "it was done") == 0
    assert marker_count("year", "Several YEAR, one year") == len(["YEAR", "year"])
    assert marker_count("several year", "for  several\nyear") == 1


def test_marker_count_ignores_inserted_punctuation() -> None:
    """Marker count ignores inserted punctuation."""
    assert marker_count("assembly na-92", "National Assembly, NA-92 Bhakkar") == 1
    assert marker_count("2021. owner", "since 2021. Owner of") == 1


def test_marker_count_does_not_split_hyphenated_words() -> None:
    """Marker count does not split hyphenated words."""
    assert marker_count("na", "NA-92") == 0


# --- V3 scoring -----------------------------------------------------------


def test_v3_recall_at_t0_is_one() -> None:
    """V3 recall at t0 is one."""
    assert retention_v3(MARKERS, T0, SAE, T0) == 1.0


def test_v3_standard_text_scores_zero() -> None:
    """V3 standard text scores zero."""
    standardized = (
        "He began his career and stayed for several years. It is one of the films."
    )
    assert retention_v3(MARKERS, standardized, SAE, T0) == 0.0


def test_v3_partial_retention() -> None:
    """V3 partial retention."""
    half = "He start his career and stayed for several years."
    assert retention_v3(MARKERS, half, SAE, T0) == pytest.approx(1 / 3)


def test_v3_ignores_words_already_in_sae() -> None:
    """V3 ignores words already in sae."""
    sae = "It is a film. It is one of the best."
    t0 = "It is one film. It is one of the best."
    assert retention_v3(["one"], t0, sae, t0) == 1.0
    assert retention_v3(["one"], sae, sae, t0) == 0.0


def test_v3_unmeasurable_markers_give_nan() -> None:
    """V3 unmeasurable markers give nan."""
    # rule inserted "the" but another rule removed one: no excess over SAE at t=0
    sae = "the cat saw the dog"
    t0 = "cat saw the the dog"
    assert math.isnan(retention_v3(["the"], t0, sae, t0))


def test_v3_retained_capped_at_t0_count() -> None:
    """V3 retained capped at t0 count."""
    assert retention_v3(["year"], "year year year", SAE, T0) == 1.0


# --- deletion markers -----------------------------------------------------

DEL_SAE = "Since 2021. The owner is serving in the Legislature of Texas."


def test_deletion_marker_is_context_across_gap() -> None:
    """Deletion marker is context across gap."""
    a = DEL_SAE.index("the Legislature")
    rules = {f"{a}-{a + 3}": {"value": "", "type": "remove_det_definite"}}
    assert deletion_markers(rules, DEL_SAE) == [
        Marker("in Legislature", "deletion", "remove_det_definite")
    ]


def test_deletion_markers_merge_adjacent_deletions() -> None:
    """Deletion markers merge adjacent deletions."""
    a_in, a_the = DEL_SAE.index("in the"), DEL_SAE.index("the Legislature")
    rules = {
        f"{a_in}-{a_in + 2}": {"value": "", "type": "null_prepositions"},
        f"{a_the}-{a_the + 3}": {"value": "", "type": "remove_det_definite"},
    }
    assert [m.form for m in deletion_markers(rules, DEL_SAE)] == [
        "serving Legislature",
        "serving Legislature",
    ]


def test_deletion_marker_keeps_sentence_punctuation() -> None:
    """Deletion marker keeps sentence punctuation."""
    a = DEL_SAE.index("The owner")
    rules = {f"{a}-{a + 3}": {"value": "", "type": "remove_det_definite"}}
    assert [m.form for m in deletion_markers(rules, DEL_SAE)] == ["2021. owner"]


def test_deletion_marker_skipped_at_text_edge_and_for_surface_rules() -> None:
    """Deletion marker skipped at text edge and for surface rules."""
    rules = {
        "0-5": {"value": "", "type": "remove_det_definite"},  # "Since": no left context
        "12-15": {"value": "the", "type": "x"},  # surface rule: not a deletion
        "20-20": {"value": "", "type": "x"},  # empty span
    }
    assert deletion_markers(rules, DEL_SAE) == []


def test_load_markers_surface_and_deletion(tmp_path: Path) -> None:
    """Load markers surface and deletion."""
    a = DEL_SAE.index("the Legislature")
    t0 = "Since 2021. The owners is serving in Legislature of Texas."
    path = tmp_path / "variants.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "seed_id": "w-1",
                    "variety": "SAE",
                    "text": DEL_SAE,
                    "executed_rules": {},
                },
                {
                    "seed_id": "w-1",
                    "variety": "Indian English",
                    "text": t0,
                    "executed_rules": {
                        "16-21": {"value": "owners", "type": "plural"},
                        f"{a}-{a + 3}": {"value": "", "type": "remove_det_definite"},
                    },
                },
            ]
        )
    )
    markers = load_markers(path)
    assert markers[("wikipedia", "w-1", "Indian English")] == [
        Marker("owners", "surface", "plural"),
        Marker("in Legislature", "deletion", "remove_det_definite"),
    ]
    assert markers[("wikipedia", "w-1", "SAE")] == []


# --- chain-level retention ------------------------------------------------

SAE_KEY = ("wikipedia", "wiki-1", "SAE")


def _chain(variety: str = "Indian English") -> ChainKey:
    return ChainKey("wikipedia", "wiki-1", variety, "m", "A0")


def test_compute_retention_v3_rows_and_losses() -> None:
    """Compute retention v3 rows and losses."""
    key = _chain()
    texts = [T0, "He start his career and stayed for several years.", SAE]
    df, losses = compute_retention(
        {key: texts}, {key.variant: MARKERS}, {SAE_KEY: SAE, key.variant: T0}
    )
    assert df["retention"].tolist() == pytest.approx([1.0, 1 / 3, 0.0])
    assert {(e["t"], e["marker"]) for e in losses} == {
        (1, "remain"),
        (1, "year"),
        (2, "start"),
    }


def test_compute_retention_splits_surface_and_deletion() -> None:
    """Compute retention splits surface and deletion."""
    sae = "He is serving in the Legislature and remained there."
    t0 = "He is serving in Legislature and remain there."
    key = _chain()
    markers = [Marker("remain", "surface"), Marker("in Legislature", "deletion")]
    texts = [t0, "He is serving in the Legislature and remain there."]
    df, _ = compute_retention(
        {key: texts}, {key.variant: markers}, {SAE_KEY: sae, key.variant: t0}
    )
    row = df[df["t"] == 1].iloc[0]
    assert row["retention"] == pytest.approx(1 / 2)
    assert row["retention_surface"] == 1.0
    assert row["retention_deletion"] == 0.0


def test_compute_retention_skips_sae_and_unmeasurable_chains() -> None:
    """Compute retention skips sae and unmeasurable chains."""
    key = _chain()
    sae_chain = _chain("SAE")
    df, _ = compute_retention(
        {key: ["cat saw the the dog"], sae_chain: ["the cat saw the dog"]},
        {key.variant: ["the"], SAE_KEY: []},
        {SAE_KEY: "the cat saw the dog", key.variant: "cat saw the the dog"},
    )
    assert df.empty


def test_compute_retention_v0_uses_surface_markers_only() -> None:
    """Compute retention v0 uses surface markers only."""
    key = _chain()
    markers = [*MARKERS, Marker("in Legislature", "deletion")]
    df, _ = compute_retention(
        {key: [T0, SAE]},
        {key.variant: markers},
        {SAE_KEY: SAE, key.variant: T0},
        matcher="v0",
    )
    # V0 substring+fuzzy: "start" in "started", "year" in "years" -> over-counts
    assert df["retention"].iloc[0] == 1.0
    assert df["retention"].iloc[1] > V0_OVERCOUNT_FLOOR


def test_compute_retention_rejects_unknown_matcher() -> None:
    """Compute retention rejects unknown matcher."""
    with pytest.raises(ValueError, match="matcher"):
        compute_retention({}, {}, {}, matcher="v9")
