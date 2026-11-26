"""Per (model, arm) retention summary: V0 vs V3, stats, steering effect.

CPU only (no embeddings). Uses `pluralistic_alignment.metrics.variety.markers`.
Writes `results/wiki/retention_summary.md` and `results/wiki/retention_summary.csv`.

Usage (repo root):
  CUDA_VISIBLE_DEVICES= PYTHONPATH=src python scripts/retention_summary.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz
from scipy import stats

from pluralistic_alignment.analysis.chains import build_full_chains, load_rewrites
from pluralistic_alignment.analysis.stats import compounding_stat, sign_test_slopes
from pluralistic_alignment.data.variants import load_markers, load_variants
from pluralistic_alignment.metrics.variety.markers import compute_retention
from pluralistic_alignment.schema import CONDITION_TO_ARM, CONTROL_VARIETY

VARIANTS = Path("data/wiki/variants.jsonl")
RUNS = {  # source label -> rewrite caches (one or more models/arms each)
    "nov": [
        "sc_gemma3_4b_full30",
        "sc_gemma3_4b_steering",
        "sc_gptoss20b_full30",
        "sc_gptoss20b_steering",
        "sc_qwen35b_full30",
        "sc_qwen35b_steering",
    ],
    "jul": ["full_30seeds_v2", "gemma3_4b_v1", "gptoss20b_v1", "vllm_qwen35b_v1"],
}
OUT = Path("results/wiki")
COND = {arm: cond for cond, arm in CONDITION_TO_ARM.items()}
KEYS = ["source", "model", "arm"]
COLS = ["retention", "retention_surface", "retention_deletion"]
MIN_PAIRS = 2
P_SCI = 0.01  # below: scientific notation
NA = "n/a"
LABEL = {
    "retention": "v3_all",
    "retention_surface": "v3_surface",
    "retention_deletion": "v3_deletion",
}


def load_source(runs: list[str]) -> dict:
    """Merged rewrite caches of the given run directories."""
    return load_rewrites([Path(f"data/wiki/runs/{r}/rewrites.jsonl") for r in runs])


def edit_rows(chains: dict) -> list[dict]:
    """Per-round edit size and no-op flag for every dialect chain."""
    rows = []
    for key, texts in chains.items():
        if key.variety == CONTROL_VARIETY:
            continue
        for t in range(1, len(texts)):
            prev, cur = texts[t - 1], texts[t]
            rows.append(
                {
                    "model": key.model,
                    "arm": key.arm,
                    "t": t,
                    "change": 1 - fuzz.ratio(prev, cur) / 100,
                    "noop": prev.strip() == cur.strip(),
                }
            )
    return rows


def macro(df: pd.DataFrame, col: str, t: int) -> float:
    """Mean over varieties of the per-variety mean (as in results.csv)."""
    d = df[df["t"] == t]
    return d.groupby("variety")[col].mean().mean()


def main() -> None:
    """Compute and write the summary tables."""
    t0, _ = load_variants(VARIANTS)
    markers = load_markers(VARIANTS)
    per_chain, edits = [], []
    for source, runs in RUNS.items():
        chains = build_full_chains(t0, load_source(runs))
        for matcher in ("v3", "v0"):
            df, _ = compute_retention(chains, markers, t0, matcher=matcher)
            per_chain.append(df.assign(source=source, matcher=matcher))
        edits.append(pd.DataFrame(edit_rows(chains)).assign(source=source))
    ret, ed = pd.concat(per_chain), pd.concat(edits)

    rows = []
    for (source, model, arm), d in ret[ret.matcher == "v3"].groupby(KEYS):
        d0 = ret[
            (ret.matcher == "v0")
            & (ret.source == source)
            & (ret.model == model)
            & (ret.arm == arm)
        ]
        e = ed[(ed.source == source) & (ed.model == model) & (ed.arm == arm)]
        st = sign_test_slopes(d)
        row = {
            "source": source,
            "model": model,
            "cond": COND.get(arm, arm),
            "v0_t1": macro(d0, "retention", 1),
            "v0_t5": macro(d0, "retention", 5),
            **{f"{LABEL[c]}_t{t}": macro(d, c, t) for c in COLS for t in (1, 5)},
            "sign_neg": st["n_negative"],
            "sign_nonzero": st["n_nonzero"],
            "sign_p": st["p_value"],
            "loss_after_t1": compounding_stat(d)["frac_loss_after_t1"].mean(),
            "edit_t1": e[e.t == 1].change.mean(),
            "noop_t1": e[e.t == 1].noop.mean(),
            "noop_t2_5": e[e.t > 1].noop.mean(),
        }
        rows.append(row)
    summary = pd.DataFrame(rows)

    effects = []
    v3 = ret[ret.matcher == "v3"]
    for (source, model), d in v3.groupby(["source", "model"]):
        for col in COLS:
            for t in (1, 5):
                j = (
                    d[d.t == t]
                    .pivot_table(
                        index=["item_id", "variety"], columns="arm", values=col
                    )
                    .dropna()
                )
                if not {"A0", "A1"} <= set(j.columns) or len(j) < MIN_PAIRS:
                    continue
                diff = j.A1 - j.A0
                p = (
                    stats.wilcoxon(j.A1, j.A0).pvalue
                    if (diff != 0).any()
                    else float("nan")
                )
                effects.append(
                    {
                        "source": source,
                        "model": model,
                        "measure": LABEL[col],
                        "t": t,
                        "neutral": j.A0.mean(),
                        "steering": j.A1.mean(),
                        "n_pairs": len(j),
                        "n_higher": int((diff > 0).sum()),
                        "wilcoxon_p": p,
                    }
                )
    eff = pd.DataFrame(effects)

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "retention_summary.csv", index=False)
    eff.to_csv(OUT / "retention_steering_effect.csv", index=False)
    write_markdown(summary, eff, OUT / "retention_summary.md")
    print((OUT / "retention_summary.md").read_text())


def f3(x: Any) -> str:
    """Three decimals, or n/a."""
    return NA if pd.isna(x) else f"{x:.3f}"


def fp(x: Any) -> str:
    """p-value: scientific below P_SCI."""
    return NA if pd.isna(x) else f"{x:.0e}" if x < P_SCI else f"{x:.2f}"


def pair(r: object, col: str) -> str:
    """'t1 / t5' of a summary row."""
    return f"{f3(getattr(r, f'{col}_t1'))} / {f3(getattr(r, f'{col}_t5'))}"


HEADER = [
    "# Retention summary",
    "",
    "Generated by `scripts/retention_summary.py`; do not edit.",
    "Macro over 5 varieties, n=30 items. V3 = default matcher",
    "(surface + deletion markers); V0 = legacy. Sign test: items whose",
    "retention falls t1 to t5 (one-sided). Edit = 1 - char similarity to",
    "the previous round.",
    "",
    "## Retention t1 / t5",
    "",
    "| Source | Model | Cond | V0 | **V3 all** | V3 surface | V3 deletion "
    "| Sign test t1-t5 | Loss after t1 | Edit t1 | No-op t1 / t2-5 |",
    "|---|---|---|---|---|---|---|---|---|---|---|",
]


def write_markdown(s: pd.DataFrame, eff: pd.DataFrame, path: Path) -> None:
    """Both tables as markdown."""
    lines = list(HEADER)
    order = ["source", "model", "cond"]
    for r in s.sort_values(order, ascending=[False, True, True]).itertuples():
        cells = [
            r.source,
            r.model,
            r.cond,
            pair(r, "v0"),
            f"**{pair(r, 'v3_all')}**",
            pair(r, "v3_surface"),
            pair(r, "v3_deletion"),
            f"{r.sign_neg}/{r.sign_nonzero}, p={fp(r.sign_p)}",
            f"{r.loss_after_t1:.1%}",
            f3(r.edit_t1),
            f"{r.noop_t1:.0%} / {r.noop_t2_5:.0%}",
        ]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    lines += [
        "",
        "## Steering effect (V3, paired by item and variety, Wilcoxon)",
        "",
        "| Source | Model | Measure | t | Neutral | Steering | Pairs higher | p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = ["source", "model", "measure", "t"]
    for r in eff.sort_values(order, ascending=[False, True, True, True]).itertuples():
        cells = [
            r.source,
            r.model,
            r.measure,
            r.t,
            f3(r.neutral),
            f3(r.steering),
            f"{r.n_higher}/{r.n_pairs}",
            fp(r.wilcoxon_p),
        ]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
