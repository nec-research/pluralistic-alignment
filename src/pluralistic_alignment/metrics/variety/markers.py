"""Feature retention: share of injected Multi-VALUE features still present at round t.

Two matchers:

- **V3** (default). Text is reduced to lowercase word tokens (punctuation
  ignored, hyphen/apostrophe kept inside words); a marker is counted as a
  whole-token sequence. Only occurrences *in excess of the SAE original* count,
  capped at the round-0 excess; markers with no round-0 excess are not
  measurable and are excluded. Covers surface markers and deletion context
  markers (see `pluralistic_alignment.data.variants.deletion_markers`).
- **V0** (legacy, legacy July 2026). Lowercase substring or rapidfuzz
  partial_ratio >= 85, surface markers only. Counts standard English as
  retained dialect (false-positive rate ~0.3 on standard text); kept for
  comparison with earlier numbers only.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from pluralistic_alignment.schema import CONTROL_VARIETY, ChainKey, Marker, VariantKey

FUZZY_MATCH_THRESHOLD = 85.0  # V0: rapidfuzz partial_ratio, 0-100
MATCHERS = ("v3", "v0")
KINDS = ("surface", "deletion")

_TOKEN = re.compile(r"\w+(?:['-]\w+)*")
_CHAR_MAP = str.maketrans(
    {
        "\N{HYPHEN}": "-",
        "\N{NON-BREAKING HYPHEN}": "-",
        "\N{LEFT SINGLE QUOTATION MARK}": "'",
        "\N{RIGHT SINGLE QUOTATION MARK}": "'",
    }
)
_pattern_cache: dict[str, re.Pattern[str]] = {}


# --- V0 (legacy) ------------------------------------------------------------


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace (V0)."""
    return " ".join(text.lower().split())


def marker_present(marker: str, text: str) -> bool:
    """V0: exact normalized substring match, with a fuzzy fallback."""
    norm_marker, norm_text = normalize(marker), normalize(text)
    if norm_marker in norm_text:
        return True
    return fuzz.partial_ratio(norm_marker, norm_text) >= FUZZY_MATCH_THRESHOLD


# --- V3 ---------------------------------------------------------------------


def normalize_tokens(text: str) -> str:
    """Lowercase word tokens joined by single spaces; punctuation dropped."""
    return " ".join(_TOKEN.findall(text.lower().translate(_CHAR_MAP)))


def _pattern(norm_marker: str) -> re.Pattern[str]:
    if norm_marker not in _pattern_cache:
        body = re.escape(norm_marker)
        _pattern_cache[norm_marker] = re.compile(rf"(?<![\w'-]){body}(?![\w'-])")
    return _pattern_cache[norm_marker]


def _count_normalized(norm_marker: str, norm_text: str) -> int:
    return len(_pattern(norm_marker).findall(norm_text)) if norm_marker else 0


def marker_count(marker: str, text: str) -> int:
    """Whole-token occurrences of `marker` in `text` (case/punctuation-insensitive)."""
    return _count_normalized(normalize_tokens(marker), normalize_tokens(text))


def _as_marker(marker: Marker | str) -> Marker:
    return marker if isinstance(marker, Marker) else Marker(marker)


def measurable_counts(
    markers: Sequence[Marker | str], sae_text: str, t0_text: str
) -> dict[tuple[str, str], int]:
    """(normalized form, kind) -> instances measurable at round 0.

    Measurable = min(times injected, round-0 count - SAE count); zero entries
    are dropped.
    """
    sae_norm, t0_norm = normalize_tokens(sae_text), normalize_tokens(t0_text)
    injected = Counter(
        (normalize_tokens(m.form), m.kind) for m in map(_as_marker, markers)
    )
    out: dict[tuple[str, str], int] = {}
    for (form, kind), n in injected.items():
        excess = _count_normalized(form, t0_norm) - _count_normalized(form, sae_norm)
        if (eff := min(n, max(0, excess))) > 0:
            out[(form, kind)] = eff
    return out


def kept_counts(
    measurable: Mapping[tuple[str, str], int], text: str, sae_text: str
) -> dict[tuple[str, str], int]:
    """(form, kind) -> measurable instances still present in `text`."""
    sae_norm, text_norm = normalize_tokens(sae_text), normalize_tokens(text)
    return {
        key: min(
            eff,
            max(
                0,
                _count_normalized(key[0], text_norm)
                - _count_normalized(key[0], sae_norm),
            ),
        )
        for key, eff in measurable.items()
    }


def _share(
    kept: Mapping[Any, int], measurable: Mapping[Any, int], kind: str | None = None
) -> float:
    keys = [k for k in measurable if kind is None or k[1] == kind]
    total = sum(measurable[k] for k in keys)
    return sum(kept[k] for k in keys) / total if total else math.nan


def retention_v3(
    markers: Sequence[Marker | str], text: str, sae_text: str, t0_text: str
) -> float:
    """V3 retention of `text` (NaN if no marker is measurable at round 0)."""
    measurable = measurable_counts(markers, sae_text, t0_text)
    return _share(kept_counts(measurable, text, sae_text), measurable)


# --- per-chain table ----------------------------------------------------------


def _retention_v3_chain(
    key: ChainKey,
    texts: Sequence[str],
    markers: Sequence[Marker | str],
    sae_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    measurable = measurable_counts(markers, sae_text, texts[0])
    if not measurable:
        return [], []
    rows: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    previous = dict(measurable)
    for t, text in enumerate(texts):
        kept = kept_counts(measurable, text, sae_text)
        rows.append(
            {
                **key.fields(),
                "t": t,
                "retention": _share(kept, measurable),
                **{
                    f"retention_{kind}": _share(kept, measurable, kind)
                    for kind in KINDS
                },
            }
        )
        if t > 0:
            losses += [
                {
                    **key.fields(),
                    "t": t,
                    "marker": form,
                    "marker_kind": kind,
                    "text_before": texts[t - 1],
                    "text_after": text,
                }
                for (form, kind), n in kept.items()
                if n < previous[(form, kind)]
            ]
        previous = kept
    return rows, losses


def _retention_v0_chain(
    key: ChainKey, texts: Sequence[str], markers: Sequence[Marker | str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    forms = [m.form for m in map(_as_marker, markers) if m.kind == "surface"]
    if not forms:
        return [], []
    rows: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    present_prev = dict.fromkeys(forms, True)
    for t, text in enumerate(texts):
        n_present = 0
        for marker in forms:
            is_present = marker_present(marker, text)
            if is_present:
                n_present += 1
            elif t > 0 and present_prev[marker]:
                losses.append(
                    {
                        **key.fields(),
                        "t": t,
                        "marker": marker,
                        "marker_kind": "surface",
                        "text_before": texts[t - 1],
                        "text_after": text,
                    }
                )
            present_prev[marker] = is_present
        share = n_present / len(forms)
        rows.append(
            {
                **key.fields(),
                "t": t,
                "retention": share,
                "retention_surface": share,
                "retention_deletion": math.nan,
            }
        )
    return rows, losses


def compute_retention(
    chains: Mapping[ChainKey, list[str]],
    markers: Mapping[VariantKey, Sequence[Marker | str]],
    t0_texts: Mapping[VariantKey, str],
    matcher: str = "v3",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Per-chain retention at each t, plus (marker, before, after) loss events.

    `t0_texts` must include the SAE control text of each item (V3 subtracts it).
    Columns: `retention` (all markers), `retention_surface`, `retention_deletion`.
    Chains of the SAE control or with no measurable marker are skipped.
    """
    if matcher not in MATCHERS:
        raise ValueError(
            f"unknown retention matcher {matcher!r}; expected one of {MATCHERS}"
        )
    rows: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    for key, texts in chains.items():
        chain_markers = markers.get(key.variant, [])
        if key.variety == CONTROL_VARIETY or not chain_markers:
            continue
        if matcher == "v0":
            chain_rows, chain_losses = _retention_v0_chain(key, texts, chain_markers)
        else:
            sae_text = t0_texts[(key.block, key.item_id, CONTROL_VARIETY)]
            chain_rows, chain_losses = _retention_v3_chain(
                key, texts, chain_markers, sae_text
            )
        rows += chain_rows
        losses += chain_losses
    return pd.DataFrame(rows), losses
