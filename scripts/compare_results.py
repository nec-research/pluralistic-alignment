"""Compare re-derived results with the originals copied from Pluralistic_OOD.

Usage: uv run python scripts/compare_results.py RUN [RUN ...]
Reads results/wiki/runs/<RUN>/ (original) and results/repro/<RUN>/ (re-derived).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "results" / "wiki" / "runs"
REPRO = ROOT / "results" / "repro"
KEY_COLS = ["metric", "lexical_metric", "variety", "model", "condition", "t"]
EXACT_FILES = ("results_table.tex", "audit_sample.jsonl")


def _metric_label(row: pd.Series) -> str:
    lexical = row.get("lexical_metric")
    return str(lexical) if isinstance(lexical, str) else str(row["metric"])


def compare_results_csv(orig_path: Path, new_path: Path) -> list[str]:
    """Max |diff| per metric over the numeric columns, after aligning rows on keys."""
    orig, new = pd.read_csv(orig_path), pd.read_csv(new_path)
    lines = [f"  results.csv rows: original {len(orig)}, repro {len(new)}"]
    if list(orig.columns) != list(new.columns):
        lines.append(
            "  COLUMNS DIFFER: only original "
            f"{sorted(map(str, set(orig) - set(new)))}, "
            f"only repro {sorted(map(str, set(new) - set(orig)))}"
        )
    keys = [c for c in KEY_COLS if c in orig.columns and c in new.columns]
    merged = orig.merge(
        new, on=keys, how="outer", suffixes=("_o", "_n"), indicator=True
    )
    unmatched = merged[merged["_merge"] != "both"]
    if len(unmatched):
        lines.append(f"  UNMATCHED ROWS: {len(unmatched)}")
    both = merged[merged["_merge"] == "both"].copy()
    both["label"] = both.apply(_metric_label, axis=1)
    value_cols = [
        c[:-2]
        for c in both.columns
        if c.endswith("_o") and f"{c[:-2]}_n" in both.columns
    ]
    for label, group in both.groupby("label"):
        diffs = [
            np.nanmax((group[f"{c}_o"] - group[f"{c}_n"]).abs().to_numpy(), initial=0.0)
            for c in value_cols
            if pd.api.types.is_numeric_dtype(group[f"{c}_o"])
        ]
        lines.append(f"  {label:<28} max |diff| = {max(diffs, default=0.0):.3g}")
    return lines


def compare_run(run: str) -> list[str]:
    """Report lines for one run."""
    orig_dir, new_dir = ORIGINAL / run, REPRO / run
    lines = [f"== {run}"]
    lines += compare_results_csv(orig_dir / "results.csv", new_dir / "results.csv")
    orig_c = pd.read_csv(orig_dir / "compounding_stat.csv")
    new_c = pd.read_csv(new_dir / "compounding_stat.csv")
    diff = np.nanmax(np.abs(orig_c["frac_loss_after_t1"] - new_c["frac_loss_after_t1"]))
    lines.append(f"  compounding_stat.csv        max |diff| = {diff:.3g}")
    same = json.loads((orig_dir / "sign_test.json").read_text()) == json.loads(
        (new_dir / "sign_test.json").read_text()
    )
    lines.append(f"  sign_test.json              {'identical' if same else 'DIFFERS'}")
    for name in EXACT_FILES:
        same = (orig_dir / name).read_bytes() == (new_dir / name).read_bytes()
        lines.append(f"  {name:<27} {'identical' if same else 'DIFFERS'}")
    return lines


def main(runs: list[str]) -> None:
    """Print a comparison for each run."""
    for run in runs:
        print("\n".join(compare_run(run)))


if __name__ == "__main__":
    main(sys.argv[1:])
