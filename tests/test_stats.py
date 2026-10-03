"""Multiple-comparison control, effect sizes and bootstrap CIs (C7)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ids_defense_selection.stats import (
    benjamini_hochberg,
    bootstrap_mean_diff_ci,
    enhance_significance,
    holm_adjust,
    paired_cohens_dz,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "enhance_significance.py"


def test_holm_matches_the_known_example() -> None:
    pvalues = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(holm_adjust(pvalues), [0.03, 0.06, 0.06, 0.02])


def test_benjamini_hochberg_is_monotone_and_at_least_raw() -> None:
    pvalues = np.array([0.001, 0.02, 0.04, 0.2])
    adjusted = benjamini_hochberg(pvalues)
    assert np.all(adjusted >= pvalues - 1e-12)
    assert np.all(adjusted <= 1.0)
    assert np.all(np.diff(adjusted) >= -1e-12)


def test_effect_size_and_bootstrap_interval() -> None:
    a = np.array([0.90, 0.91, 0.89, 0.92, 0.88])
    b = np.array([0.85, 0.87, 0.84, 0.88, 0.83])
    assert paired_cohens_dz(a, b) > 3
    low, high = bootstrap_mean_diff_ci(a, b, n_boot=2000, seed=7)
    assert low < (a - b).mean() < high


def _raw_results() -> pd.DataFrame:
    models = ["standard", "adv_training", "constrained_adv", "trades",
              "free_at", "class_aware_constrained"]
    rows = []
    for seed in (7, 13, 21, 42, 100):
        for index, model in enumerate(models):
            rows.append({"seed": seed, "model": model, "attack": "pgd",
                         "epsilon": 0.10, "f1": 0.90 - 0.01 * index,
                         "attack_success_rate": 0.05 * index})
    return pd.DataFrame(rows)


def test_enhance_significance_adds_corrected_columns() -> None:
    table = enhance_significance(_raw_results(), epsilon_list=(0.10,), n_boot=500)
    assert {"holm_t_pvalue", "bh_t_pvalue", "holm_wilcoxon_pvalue",
            "cohens_dz", "mean_diff", "ci_low", "ci_high",
            "significant_holm_t"} <= set(table.columns)
    assert (table["holm_t_pvalue"] >= table["t_pvalue"] - 1e-12).all()
    assert table["holm_t_pvalue"].between(0, 1).all()


def test_script_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--n-boot" in completed.stdout
