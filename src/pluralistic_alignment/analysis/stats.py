"""Bootstrap CIs over items, the original compounding statistic, and the sign test."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from pluralistic_alignment.schema import MAX_T

MIN_BOOTSTRAP_N = 2
BOOTSTRAP_SEED = 42


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 2000
) -> tuple[float, float, float]:
    """(mean, lo, hi), 95% BCa over items; degenerate CI if BCa is undefined."""
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) < MIN_BOOTSTRAP_N or np.allclose(arr, arr[0]):
        return mean, mean, mean
    res = stats.bootstrap(
        (arr,),
        np.mean,
        n_resamples=n_resamples,
        confidence_level=0.95,
        method="BCa",
        rng=BOOTSTRAP_SEED,
    )
    return mean, float(res.confidence_interval.low), float(res.confidence_interval.high)


def aggregate_with_ci(
    df: pd.DataFrame, value_col: str, group_cols: list[str]
) -> pd.DataFrame:
    """Mean of `value_col` per group with a bootstrap CI over items."""
    rows: list[dict[str, Any]] = []
    for group_key, sub in df.groupby(group_cols):
        values = sub[value_col].dropna().tolist()
        if not values:
            continue
        mean, lo, hi = bootstrap_ci(values)
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        rows.append(
            {
                **dict(zip(group_cols, keys, strict=True)),
                f"{value_col}_mean": mean,
                f"{value_col}_ci_lo": lo,
                f"{value_col}_ci_hi": hi,
                "n_items": len(values),
            }
        )
    return pd.DataFrame(rows)


def compounding_stat(retention: pd.DataFrame, max_t: int = MAX_T) -> pd.DataFrame:
    """Share of total t=max_t loss that happens after t=1, macro over items."""
    pivot = retention.pivot_table(
        index=["block", "item_id", "variety", "model", "arm"],
        columns="t",
        values="retention",
    )
    if 1 not in pivot.columns or max_t not in pivot.columns:
        return pd.DataFrame()
    total = 1.0 - pivot[max_t]
    after_t1 = pivot[1] - pivot[max_t]
    frac = (after_t1 / total.replace(0, np.nan)).rename("frac_loss_after_t1")
    return (
        frac.reset_index()
        .groupby(["block", "variety", "model", "arm"])["frac_loss_after_t1"]
        .mean()
        .reset_index()
    )


def sign_test_slopes(retention: pd.DataFrame, max_t: int = MAX_T) -> dict[str, Any]:
    """One-sided sign test that per-item retention falls from t=1 to t=max_t."""
    pivot = retention.pivot_table(
        index=["block", "item_id"], columns="t", values="retention", aggfunc="mean"
    )
    if 1 not in pivot.columns or max_t not in pivot.columns:
        return {"n_items": 0, "n_negative": 0, "p_value": None}
    slopes = (pivot[max_t] - pivot[1]).to_numpy()
    n_negative = int((slopes < 0).sum())
    n_nonzero = int((slopes != 0).sum())
    result = (
        stats.binomtest(n_negative, n_nonzero, p=0.5, alternative="greater")
        if n_nonzero
        else None
    )
    return {
        "n_items": len(slopes),
        "n_negative": n_negative,
        "n_nonzero": n_nonzero,
        "p_value": float(result.pvalue) if result else None,
    }
