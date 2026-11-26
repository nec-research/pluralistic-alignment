"""Cross-variety convergence within a seed, and convergence to the control chain.

Embedding similarity sat near ceiling in the experiment, so the token-set Jaccard
version is the one with dynamic range.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from pluralistic_alignment.metrics.variety.markers import normalize
from pluralistic_alignment.schema import CONTROL_VARIETY, ChainKey

MIN_VARIETIES_FOR_CONVERGENCE = 2

GroupKey = tuple[str, str, str, str]  # block, item_id, model, arm


def group_by_seed(
    chains: Mapping[ChainKey, Any],
) -> dict[GroupKey, dict[str, ChainKey]]:
    """{(block, item_id, model, arm): {variety: chain key}}."""
    groups: dict[GroupKey, dict[str, ChainKey]] = defaultdict(dict)
    for key in chains:
        groups[(key.block, key.item_id, key.model, key.arm)][key.variety] = key
    return groups


def jaccard(a: set[str], b: set[str]) -> float:
    """Token-set Jaccard; two empty sets count as identical."""
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _convergence_rows(
    chains: Mapping[ChainKey, Any],
    n_rounds: Callable[[ChainKey], int],
    round_values: Callable[[list[ChainKey], ChainKey | None, int], dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (block, item_id, model, arm), by_variety in group_by_seed(chains).items():
        varieties = [k for v, k in by_variety.items() if v != CONTROL_VARIETY]
        if len(varieties) < MIN_VARIETIES_FOR_CONVERGENCE:
            continue
        control = by_variety.get(CONTROL_VARIETY)
        for t in range(n_rounds(varieties[0])):
            rows.append(
                {
                    "block": block,
                    "item_id": item_id,
                    "model": model,
                    "arm": arm,
                    "t": t,
                    **round_values(varieties, control, t),
                }
            )
    return pd.DataFrame(rows)


def compute_convergence(embeddings: Mapping[ChainKey, np.ndarray]) -> pd.DataFrame:
    """Mean pairwise cosine between variety chains, and mean cosine to the control."""

    def values(varieties: list[ChainKey], control: ChainKey | None, t: int) -> dict:
        vecs = np.stack([embeddings[k][t] for k in varieties])
        upper = np.triu_indices(len(vecs), k=1)
        return {
            "pairwise_variety_similarity": float((vecs @ vecs.T)[upper].mean()),
            "similarity_to_sae_chain": (
                float((vecs @ embeddings[control][t]).mean()) if control else None
            ),
        }

    return _convergence_rows(embeddings, lambda k: len(embeddings[k]), values)


def compute_lexical_convergence(chains: Mapping[ChainKey, list[str]]) -> pd.DataFrame:
    """Mean pairwise token-set Jaccard between variety chains, and to the control."""

    def tokens(key: ChainKey, t: int) -> set[str]:
        return set(normalize(chains[key][t]).split())

    def values(varieties: list[ChainKey], control: ChainKey | None, t: int) -> dict:
        sets = [tokens(k, t) for k in varieties]
        pairwise = [
            jaccard(sets[i], sets[j])
            for i in range(len(sets))
            for j in range(i + 1, len(sets))
        ]
        control_tokens = tokens(control, t) if control else None
        return {
            "pairwise_variety_jaccard": float(np.mean(pairwise)),
            "jaccard_to_sae_chain": (
                float(np.mean([jaccard(s, control_tokens) for s in sets]))
                if control_tokens is not None
                else None
            ),
        }

    return _convergence_rows(chains, lambda k: len(chains[k]), values)
