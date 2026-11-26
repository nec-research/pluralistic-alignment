"""Lexical diversity per chain and round: MATTR and MTLD."""

from __future__ import annotations

import sys
import types
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

import pandas as pd

from pluralistic_alignment.schema import ChainKey


def _import_lex_div() -> types.ModuleType:
    """Import `lexical_diversity.lex_div` without setuptools' `pkg_resources`.

    lex_div only calls `pkg_resources.resource_filename` once, at import time,
    to locate its bundled lemma list. setuptools >= 81 removed `pkg_resources`,
    so a one-function stand-in is provided for the duration of the import only.
    """
    shim = types.ModuleType("pkg_resources")
    shim.resource_filename = lambda package, name: str(files(package) / name)  # type: ignore[attr-defined]
    saved = sys.modules.get("pkg_resources")
    sys.modules["pkg_resources"] = shim
    try:
        from lexical_diversity import lex_div  # noqa: PLC0415

        return lex_div
    finally:
        if saved is None:
            del sys.modules["pkg_resources"]
        else:
            sys.modules["pkg_resources"] = saved


ld = _import_lex_div()

MIN_TOKENS_FOR_MATTR = 25
MIN_TOKENS_FOR_MTLD = 10


def lexical_diversity(text: str) -> tuple[float | None, float | None]:
    """(MATTR, MTLD); None when the text is too short for the measure."""
    tokens = ld.tokenize(text)
    mattr = ld.mattr(tokens) if len(tokens) >= MIN_TOKENS_FOR_MATTR else None
    mtld = ld.mtld(tokens) if len(tokens) >= MIN_TOKENS_FOR_MTLD else None
    return mattr, mtld


def compute_lexical_diversity(chains: Mapping[ChainKey, list[str]]) -> pd.DataFrame:
    """One row per chain and round with `mattr` and `mtld`."""
    rows: list[dict[str, Any]] = []
    for key, texts in chains.items():
        for t, text in enumerate(texts):
            mattr, mtld = lexical_diversity(text)
            rows.append({**key.fields(), "t": t, "mattr": mattr, "mtld": mtld})
    return pd.DataFrame(rows)
