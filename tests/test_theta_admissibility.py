"""Preference sweep (C4) and admissibility sweep (C5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ids_defense_selection.selection import (
    OBJECTIVE_COLUMNS,
    THETA_PRESETS,
    admissibility_sweep_table,
    switching_boundaries,
    theta_grid,
    theta_summary_table,
    theta_sweep_table,
)


def _front() -> pd.DataFrame:
    # Three mutually non-dominated trade-off points plus a dominated one.
    return pd.DataFrame(
        [[0.90, 0.30, 0.50, 0.50],
         [0.55, 0.60, 0.50, 0.50],
         [0.20, 0.85, 0.50, 0.50],
         [0.30, 0.30, 0.20, 0.20]],
        index=["cleanish", "middle", "robustish", "dominated"],
        columns=OBJECTIVE_COLUMNS,
    )


def _stds(means: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=means.index, columns=means.columns)


def test_canonical_presets_match_the_manuscript_table() -> None:
    assert THETA_PRESETS == {
        "Robust": (0.10, 0.60, 0.10, 0.20),
        "Balanced": (0.25, 0.35, 0.20, 0.20),
        "Clean": (0.50, 0.15, 0.15, 0.20),
        "Cost": (0.20, 0.20, 0.50, 0.10),
    }


def test_theta_grid_covers_the_simplex() -> None:
    coarse = theta_grid(0.25)
    assert coarse.shape == (35, 4)                  # C(4+3, 3)
    assert np.allclose(coarse.sum(axis=1), 1.0)
    assert theta_grid(0.1).shape[0] == 286
    with pytest.raises(ValueError, match="divide 1.0"):
        theta_grid(0.3)


def test_theta_sweep_and_summary() -> None:
    means, stds = _front(), _stds(_front())
    sweep = theta_sweep_table(means, stds, margin=0.0, step=0.05)
    assert len(sweep) == 1771
    assert set(sweep["selected"]) <= {"cleanish", "middle", "robustish"}

    # A vertex of the simplex isolates one objective.
    clean_vertex = sweep[(sweep.theta1 == 1.0) & (sweep.theta2 == 0.0)]
    assert set(clean_vertex["selected"]) == {"cleanish"}
    robust_vertex = sweep[(sweep.theta1 == 0.0) & (sweep.theta2 == 1.0)]
    assert set(robust_vertex["selected"]) == {"robustish"}

    summary = theta_summary_table(sweep)
    assert np.isclose(summary["share"].sum(), 1.0)
    assert set(summary["defense"]) <= set(sweep["selected"])


def test_switching_boundaries_report_selected_pairs() -> None:
    means, stds = _front(), _stds(_front())
    sweep = theta_sweep_table(means, stds, margin=0.0, step=0.1)
    payload = switching_boundaries(means, stds, 0.0, sweep, step=0.1)
    pairs = {tuple(item["pair"]) for item in payload["boundaries"]}
    assert ("cleanish", "middle") in pairs
    assert ("cleanish", "robustish") in pairs
    assert len(payload["boundaries"][0]["coef"]) == 4


def test_admissibility_sweep_flags_empty_and_full_regions() -> None:
    means, stds = _front(), _stds(_front())
    table = admissibility_sweep_table(means, stds, margin=0.0, step=0.1)
    assert len(table) == 121
    full = table[(table.tau2 == 0.0) & (table.tau4 == 0.0)].iloc[0]
    assert full["n_admissible"] == 4
    assert not bool(full["no_candidate"])
    strict = table[(table.tau2 == 1.0) & (table.tau4 == 1.0)].iloc[0]
    assert strict["n_admissible"] == 0
    assert bool(strict["no_candidate"])
    assert strict["selected_robust"] == "none"
    # Thresholds at the observed level keep the candidate.
    keep = table[(table.tau2 == 0.3) & (table.tau4 == 0.5)].iloc[0]
    assert keep["n_admissible"] == 3
