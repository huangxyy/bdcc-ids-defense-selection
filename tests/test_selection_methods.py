"""Decision-method ablation: Pareto/weighted vs weighted-only vs TOPSIS."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ids_defense_selection.selection import (
    OBJECTIVE_COLUMNS,
    THETA_PRESETS,
    candidate_dependence_table,
    compare_decision_methods,
    select_defense,
    select_topsis,
)

THETA = (0.5, 0.5, 0.0, 0.0)


def _objectives() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.90, 0.30, 0.50, 0.50],
         [0.55, 0.35, 0.50, 0.50],
         [0.10, 0.40, 0.50, 0.50]],
        index=["a", "b", "c"], columns=OBJECTIVE_COLUMNS,
    )


def test_topsis_prefers_a_dominating_candidate() -> None:
    matrix = np.array([[0.9, 0.9, 0.9, 0.9], [0.5, 0.5, 0.5, 0.5]])
    assert select_topsis(matrix, ["a", "b"], THETA) == "a"


def test_fixed01_and_minmax_normalisation_can_disagree() -> None:
    """Min-max amplifies narrow objective ranges; the fixed [0,1] rule does not."""
    matrix = _objectives().values
    names = ["a", "b", "c"]
    assert select_defense(matrix, names, THETA) == "b"
    assert select_defense(matrix, names, THETA, normalise="fixed01") == "a"


def test_compare_decision_methods_reports_every_rule() -> None:
    means = _objectives()
    stds = pd.DataFrame(0.0, index=means.index, columns=means.columns)
    table = compare_decision_methods(means, stds, margin=0.0,
                                     theta_presets={"t": THETA})
    assert list(table["theta_name"]) == ["t"]
    assert set(table.columns) >= {
        "pareto_weighted", "weighted_no_pareto", "weighted_fixed01",
        "topsis_no_pareto", "topsis_pareto", "n_candidates", "pareto_size",
    }
    assert table.loc[0, "n_candidates"] == 3
    assert table.loc[0, "pareto_size"] == 3


def test_candidate_dependence_detects_a_flip() -> None:
    means = _objectives()
    stds = pd.DataFrame(0.0, index=means.index, columns=means.columns)
    table = candidate_dependence_table(means, stds, margin=0.0,
                                       theta_presets={"t": THETA})
    assert len(table) == len(means)  # one row per removed candidate
    row = table[table["removed"] == "b"].iloc[0]
    assert row["selection"] == "b"
    assert row["selection_without"] == "a"
    assert bool(row["changed"]) is True


def test_default_presets_are_used_when_none_is_given() -> None:
    means = _objectives()
    stds = pd.DataFrame(0.0, index=means.index, columns=means.columns)
    table = compare_decision_methods(means, stds, margin=0.0)
    assert len(table) == len(THETA_PRESETS)
