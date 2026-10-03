"""Statistical enhancement of the defense comparisons (C7).

Adds multiple-comparison control (Holm, Benjamini-Hochberg), paired effect
sizes and bootstrap confidence intervals on top of the raw paired t / Wilcoxon
tests that each backbone run already writes.  With ten seeds the two-sided
Wilcoxon cannot go below ``2 / 2**10 ≈ 0.00195``, so the adjusted p-values and
the effect sizes are what the manuscript tables should report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def holm_adjust(pvalues) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment (FWER control)."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, index in enumerate(order, start=1):
        running = max(running, (n - rank + 1) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def benjamini_hochberg(pvalues) -> np.ndarray:
    """Benjamini-Hochberg step-up adjustment (FDR control)."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for rank in range(n, 0, -1):
        index = order[rank - 1]
        running = min(running, p[index] * n / rank)
        adjusted[index] = running
    return adjusted


def _adjust_with_nan(pvalues, adjust) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    mask = ~np.isnan(p)
    if mask.any():
        out[mask] = adjust(p[mask])
    return out


def paired_cohens_dz(a, b) -> float:
    """Cohen's d_z for paired samples (mean difference / sd of differences)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diff.size < 2:
        return float("nan")
    sd = diff.std(ddof=1)
    if sd == 0.0:
        return float("nan")
    return float(diff.mean() / sd)


def bootstrap_mean_diff_ci(a, b, *, n_boot: int = 10000, alpha: float = 0.05,
                           seed: int = 2026) -> tuple[float, float]:
    """Percentile bootstrap CI for mean(a) - mean(b), paired by seed."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diff.size < 2 or n_boot < 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, diff.size, size=(n_boot, diff.size))
    samples = diff[indices].mean(axis=1)
    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def enhance_significance(results_df: pd.DataFrame, epsilon_list=None, comparisons=None, *,
                         n_boot: int = 10000, alpha: float = 0.05,
                         seed: int = 2026) -> pd.DataFrame:
    """Paired tests plus Holm/BH, effect size and bootstrap CI.

    Families for the multiplicity adjustment are the comparisons inside one
    (attack, epsilon, metric) cell, which is the natural family for the
    manuscript's "is defense A better than defense B in this scenario" claims.
    """
    from .reporting import DEFAULT_COMPARISONS, compute_significance_tests

    if comparisons is None:
        comparisons = DEFAULT_COMPARISONS
    if epsilon_list is None:
        epsilon_list = tuple(sorted({
            float(value) for value in results_df["epsilon"].unique() if value > 0
        }))
    base = compute_significance_tests(results_df, tuple(epsilon_list), comparisons)
    if base.empty:
        return base

    enhanced = []
    for (attack, epsilon, metric), group in base.groupby(
            ["attack", "epsilon", "metric"], sort=False):
        group = group.copy()
        group["holm_t_pvalue"] = _adjust_with_nan(group["t_pvalue"], holm_adjust)
        group["bh_t_pvalue"] = _adjust_with_nan(group["t_pvalue"], benjamini_hochberg)
        group["holm_wilcoxon_pvalue"] = _adjust_with_nan(
            group["wilcoxon_pvalue"], holm_adjust)
        group["bh_wilcoxon_pvalue"] = _adjust_with_nan(
            group["wilcoxon_pvalue"], benjamini_hochberg)

        effects = []
        for row in group.itertuples():
            model_a, _, model_b = row.comparison.partition("_vs_")
            cell = results_df[(results_df["attack"] == attack) &
                              np.isclose(results_df["epsilon"], epsilon)]
            a = cell[cell["model"] == model_a].sort_values("seed")[metric].to_numpy(float)
            b = cell[cell["model"] == model_b].sort_values("seed")[metric].to_numpy(float)
            mean_diff = float(np.mean(a - b)) if len(a) == len(b) and len(a) else float("nan")
            low, high = bootstrap_mean_diff_ci(a, b, n_boot=n_boot, alpha=alpha, seed=seed)
            effects.append((mean_diff, paired_cohens_dz(a, b), low, high))
        group[["mean_diff", "cohens_dz", "ci_low", "ci_high"]] = effects
        group["significant_holm_t"] = group["holm_t_pvalue"] < alpha
        group["significant_holm_wilcoxon"] = group["holm_wilcoxon_pvalue"] < alpha
        enhanced.append(group)
    return pd.concat(enhanced, ignore_index=True)
