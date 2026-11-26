"""Metric report generation.

Re-derivable from cached JSONL with no API calls. Writes `results.csv`,
`compounding_stat.csv`, `sign_test.json`, `audit_sample.jsonl`,
`results_table.tex` and `results.pdf` to the results directory.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from pluralistic_alignment.analysis.chains import build_full_chains, load_rewrites
from pluralistic_alignment.analysis.stats import (
    aggregate_with_ci,
    compounding_stat,
    sign_test_slopes,
)
from pluralistic_alignment.data.variants import (
    load_markers,
    load_seed_texts,
    load_variants,
)
from pluralistic_alignment.editing.postprocess import looks_wrapped
from pluralistic_alignment.metrics.diversity import (
    compute_convergence,
    compute_lexical_convergence,
)
from pluralistic_alignment.metrics.lexical import compute_lexical_diversity
from pluralistic_alignment.metrics.semantic import (
    Embedder,
    compute_embeddings,
    compute_semantic_preservation,
)
from pluralistic_alignment.metrics.variety.markers import compute_retention
from pluralistic_alignment.paths import check_output_path
from pluralistic_alignment.schema import MAX_T, ChainKey

AUDIT_SAMPLE_SIZE = 30
AUDIT_SEED = 42
CHAIN_COLS = ["block", "variety", "model", "arm", "t"]
SEED_COLS = ["block", "model", "arm", "t"]
RETENTION_COLS = ["retention", "retention_surface", "retention_deletion"]


@dataclass(frozen=True)
class ReportInputs:
    """Seeds, round-0 variants and one or more edit caches."""

    seeds: Path
    variants: Path
    rewrites: Sequence[Path]
    retention_matcher: str = "v3"


@dataclass(frozen=True)
class PerChainMetrics:
    """Unaggregated metric tables plus loss events for the audit sample."""

    retention: pd.DataFrame
    convergence: pd.DataFrame
    lexical_convergence: pd.DataFrame
    lexical: pd.DataFrame
    semantic: pd.DataFrame
    losses: list[dict[str, Any]]


def wrapper_count(chains: Mapping[ChainKey, list[str]]) -> tuple[int, int]:
    """(flagged, total) edited texts with chat-wrapper boilerplate (t >= 1)."""
    edited = [text for texts in chains.values() for text in texts[1:]]
    return sum(looks_wrapped(text) for text in edited), len(edited)


def compute_metrics(inputs: ReportInputs, embedder: Embedder) -> PerChainMetrics:
    """All metrics for every complete chain."""
    seeds = load_seed_texts(inputs.seeds)
    t0_texts, _ = load_variants(inputs.variants)
    markers = load_markers(inputs.variants)
    chains = build_full_chains(t0_texts, load_rewrites(inputs.rewrites))
    print(f"{len(chains)} complete chains (t=0..{MAX_T})")
    flagged, total = wrapper_count(chains)
    if flagged:
        print(f"WARNING: {flagged}/{total} edited texts look like chat-wrapper output")
    retention, losses = compute_retention(
        chains, markers, t0_texts, matcher=inputs.retention_matcher
    )
    print(f"retention matcher: {inputs.retention_matcher}")
    embeddings = compute_embeddings(chains, embedder)
    return PerChainMetrics(
        retention=retention,
        convergence=compute_convergence(embeddings),
        lexical_convergence=compute_lexical_convergence(chains),
        lexical=compute_lexical_diversity(chains),
        semantic=compute_semantic_preservation(embeddings, seeds, embedder),
        losses=losses,
    )


def sanity_warnings(metrics: PerChainMetrics) -> list[str]:
    """Sanity checks: t=0 retention is 1.0; convergence at t=0 is below t=MAX_T."""
    warnings: list[str] = []
    retention, convergence = metrics.retention, metrics.convergence
    t0 = retention.loc[retention["t"] == 0, "retention"]
    if not t0.empty and not np.allclose(t0, 1.0):
        warnings.append(f"t=0 retention is not 1.0 for all chains (min={t0.min():.3f})")
    if not convergence.empty:
        sim = convergence.groupby("t")["pairwise_variety_similarity"].mean()
        if 0 in sim and MAX_T in sim and sim[0] >= sim[MAX_T]:
            warnings.append(
                f"t=0 cross-variety similarity ({sim[0]:.3f}) is not below "
                f"t={MAX_T} ({sim[MAX_T]:.3f})"
            )
    return warnings


def aggregate_retention(retention: pd.DataFrame, column: str) -> pd.DataFrame:
    """One retention column as `retention_mean/ci` rows tagged `metric=column`."""
    agg = aggregate_with_ci(retention, column, CHAIN_COLS)
    renamed = {c: c.replace(column, "retention", 1) for c in agg.columns}
    return agg.rename(columns=renamed).assign(metric=column)


def aggregate(metrics: PerChainMetrics) -> pd.DataFrame:
    """The combined `results.csv` table."""
    lexical_long = metrics.lexical.melt(
        id_vars=["block", "item_id", "variety", "model", "arm", "t"],
        value_vars=["mattr", "mtld"],
        var_name="metric",
        value_name="value",
    )
    parts = [
        *(
            aggregate_retention(metrics.retention, col)
            for col in RETENTION_COLS
            if col in metrics.retention
        ),
        aggregate_with_ci(
            metrics.convergence, "pairwise_variety_similarity", SEED_COLS
        ).assign(metric="convergence"),
        aggregate_with_ci(
            metrics.lexical_convergence, "pairwise_variety_jaccard", SEED_COLS
        ).assign(metric="lexical_convergence"),
        aggregate_with_ci(
            metrics.convergence, "similarity_to_sae_chain", SEED_COLS
        ).assign(metric="convergence_to_sae"),
        aggregate_with_ci(
            metrics.lexical_convergence, "jaccard_to_sae_chain", SEED_COLS
        ).assign(metric="lexical_convergence_to_sae"),
        aggregate_with_ci(lexical_long, "value", [*CHAIN_COLS, "metric"]).rename(
            columns={"metric": "lexical_metric"}
        ),
        aggregate_with_ci(metrics.semantic, "semantic_preservation", CHAIN_COLS).assign(
            metric="semantic_preservation"
        ),
    ]
    return pd.concat(parts, ignore_index=True)


def write_audit_sample(losses: list[dict[str, Any]], out_path: Path) -> None:
    """Seeded sample of loss events for the manual 3-way audit."""
    rng = random.Random(AUDIT_SEED)  # noqa: S311 - reproducible sampling
    sample = rng.sample(losses, min(AUDIT_SAMPLE_SIZE, len(losses)))
    with out_path.open("w") as f:
        for record in sample:
            # filled in by hand: loss | harmless_paraphrase | fluency_repair
            f.write(json.dumps({**record, "label": None}, ensure_ascii=False) + "\n")


def write_latex_table(retention_agg: pd.DataFrame, out_path: Path) -> None:
    """Retention at t=1 and t=MAX_T per variety, averaged over models and arms."""
    summary = (
        retention_agg.groupby(["variety", "t"])["retention_mean"]
        .mean()
        .reset_index()
        .pivot(index="variety", columns="t", values="retention_mean")
    )

    def fmt(value: Any) -> str:
        return (
            f"{value:.2f}"
            if isinstance(value, float) and not math.isnan(value)
            else "--"
        )

    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        f"Variety & $t=1$ & $t={MAX_T}$ \\\\",
        "\\midrule",
    ]
    lines += [
        f"{variety} & {fmt(row.get(1))} & {fmt(row.get(MAX_T))} \\\\"
        for variety, row in summary.iterrows()
    ]
    lines += ["\\bottomrule", "\\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n")


def write_figure(results: pd.DataFrame, out_path: Path) -> None:
    """(a) retention over t per variety, (b) cross-variety convergence over t."""
    retention = results[results["metric"] == "retention"]
    convergence = results[results["metric"] == "convergence"].sort_values("t")
    fig = Figure(figsize=(9, 3.5))
    ax_ret, ax_conv = fig.subplots(1, 2)
    for variety, unsorted in retention.groupby("variety"):
        group = unsorted.sort_values("t")
        ax_ret.plot(group["t"], group["retention_mean"], marker="o", label=variety)
        ax_ret.fill_between(
            group["t"], group["retention_ci_lo"], group["retention_ci_hi"], alpha=0.15
        )
    ax_ret.set(xlabel="iteration $t$", ylabel="feature retention")
    ax_ret.set_title("(a) feature retention")
    ax_ret.legend(fontsize=6, loc="lower left")
    if not convergence.empty:
        ax_conv.plot(
            convergence["t"],
            convergence["pairwise_variety_similarity_mean"],
            marker="o",
            color="black",
        )
        ax_conv.fill_between(
            convergence["t"],
            convergence["pairwise_variety_similarity_ci_lo"],
            convergence["pairwise_variety_similarity_ci_hi"],
            alpha=0.15,
        )
    ax_conv.set(xlabel="iteration $t$", ylabel="mean pairwise similarity")
    ax_conv.set_title("(b) cross-variety convergence")
    fig.tight_layout()
    fig.savefig(out_path)


def write_report(inputs: ReportInputs, results_dir: Path, embedder: Embedder) -> None:
    """Compute everything and write the six output files."""
    metrics = compute_metrics(inputs, embedder)
    check_output_path(results_dir, set(metrics.retention.get("block", [])))
    for warning in sanity_warnings(metrics):
        print(f"WARNING: {warning}")
    results = aggregate(metrics)
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "results.csv", index=False)
    compounding_stat(metrics.retention).to_csv(
        results_dir / "compounding_stat.csv", index=False
    )
    (results_dir / "sign_test.json").write_text(
        json.dumps(sign_test_slopes(metrics.retention), indent=2)
    )
    write_audit_sample(metrics.losses, results_dir / "audit_sample.jsonl")
    write_latex_table(
        results[results["metric"] == "retention"], results_dir / "results_table.tex"
    )
    write_figure(results, results_dir / "results.pdf")
    print(f"wrote report to {results_dir}")
