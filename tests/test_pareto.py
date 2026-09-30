"""Tests for the Pareto filtering and preference-weighted selection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pareto_selection import build_objective_matrix, is_pareto_optimal, select_defense


def _toy_objectives() -> np.ndarray:
    # maximise every objective: row 0 is dominated by row 2, row 1 is optimal
    return np.array([
        [0.80, 0.50, 0.10],
        [0.90, 0.90, 0.30],
        [0.85, 0.60, 0.40],
    ])


def test_is_pareto_optimal_marks_dominated_points() -> None:
    mask = is_pareto_optimal(_toy_objectives())
    assert not mask[0]
    assert mask[1]
    assert mask[2]


def test_select_defense_follows_the_preference_vector() -> None:
    objectives = _toy_objectives()
    names = ["a", "b", "c"]
    assert select_defense(objectives, names, (0.0, 1.0, 0.0)) == "b"
    # maximising the third (cost-efficiency) column favours c
    assert select_defense(objectives, names, (0.0, 0.0, 1.0)) == "c"


def test_build_objective_matrix_aligns_the_four_objectives() -> None:
    mean_results = pd.DataFrame({
        "model": ["standard", "adv_training"],
        "attack": ["clean", "clean"],
        "epsilon": [0.0, 0.0],
        "f1": [0.90, 0.88],
        "attack_success_rate": [0.0, 0.0],
    })
    robust_rows = pd.DataFrame({
        "model": ["standard", "adv_training"],
        "attack": ["pgd", "pgd"],
        "epsilon": [0.10, 0.10],
        "f1": [0.60, 0.80],
        "attack_success_rate": [0.40, 0.20],
    })
    mean_results = pd.concat([mean_results, robust_rows], ignore_index=True)
    efficiency = pd.DataFrame({
        "model": ["standard", "adv_training"],
        "relative_train_cost_vs_standard": [1.0, 4.0],
    })
    categories = pd.DataFrame({
        "model": ["standard", "adv_training"],
        "attack": ["pgd", "pgd"],
        "epsilon": [0.10, 0.10],
        "adv_recall": [0.50, 0.70],
    })

    objectives = build_objective_matrix(
        {"mean_results": mean_results, "efficiency_mean": efficiency,
         "category_mean_results": categories},
        ref_attack="pgd", ref_epsilon=0.10,
    )
    assert list(objectives.index) == ["standard", "adv_training"]
    assert objectives.loc["standard", "phi1"] == 0.90
    assert objectives.loc["adv_training", "phi2"] == 0.80
    assert objectives.loc["adv_training", "phi3"] == 0.25
    assert objectives.loc["standard", "phi4"] == 0.50
