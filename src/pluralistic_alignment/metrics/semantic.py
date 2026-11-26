"""Embedding similarity (Qwen3-Embedding-0.6B) and meaning preservation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from pluralistic_alignment.schema import ChainKey

EMBEDDER_NAME = "Qwen/Qwen3-Embedding-0.6B"


class Embedder(Protocol):
    """The `SentenceTransformer.encode` subset used here."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> Any:
        """Return one vector per sentence."""
        ...


def load_embedder(name: str = EMBEDDER_NAME) -> Embedder:
    """Load a sentence-transformers model (GPU if available)."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - heavy

    # encode() is overloaded, so it does not match the Protocol structurally.
    return cast("Embedder", SentenceTransformer(name))


def encode(embedder: Embedder, texts: Sequence[str]) -> np.ndarray:
    """Unit-normalized embeddings, batched."""
    return np.asarray(
        embedder.encode(
            list(texts),
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
    )


def compute_embeddings(
    chains: Mapping[ChainKey, list[str]], embedder: Embedder
) -> dict[ChainKey, np.ndarray]:
    """One (rounds, dim) matrix per chain."""
    keys = list(chains)
    vectors = encode(embedder, [text for key in keys for text in chains[key]])
    out: dict[ChainKey, np.ndarray] = {}
    offset = 0
    for key in keys:
        n = len(chains[key])
        out[key] = vectors[offset : offset + n]
        offset += n
    return out


def compute_semantic_preservation(
    embeddings: Mapping[ChainKey, np.ndarray],
    seeds: Mapping[tuple[str, str], str],
    embedder: Embedder,
) -> pd.DataFrame:
    """Cosine similarity of every round to the original standard seed text."""
    seed_keys = sorted(seeds)
    seed_vecs = dict(
        zip(seed_keys, encode(embedder, [seeds[k] for k in seed_keys]), strict=True)
    )
    rows: list[dict[str, Any]] = []
    for key, vecs in embeddings.items():
        seed_vec = seed_vecs.get((key.block, key.item_id))
        if seed_vec is None:
            continue
        rows.extend(
            {**key.fields(), "t": t, "semantic_preservation": float(vec @ seed_vec)}
            for t, vec in enumerate(vecs)
        )
    return pd.DataFrame(rows)
