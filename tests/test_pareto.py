"""Tests for the Pareto filtering and preference-weighted selection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids_defense_selection.paths import PROJECT_ROOT
from ids_defense_selection.selection import (
    BACKBONE_CLI_KEYS,
    BACKBONE_DIRS,
    BACKBONE_LABELS,
    BACKBONE_SELECTION_COLUMNS,
    THETA_PRESETS,
    apply_admissibility,
    build_objective_matrices,
    build_objective_matrix,
    is_pareto_optimal,
    is_pareto_optimal_uncertain,
    resolve_outputs_root,
    select_defense,
    write_selection_csv,
)


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


def _matrix(rows: list[list[float]], models: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=models, columns=["phi1", "phi2", "phi3", "phi4"])


def test_uncertainty_aware_dominance_matches_point_estimates_at_margin_zero() -> None:
    means = _matrix([[0.80, 0.50, 0.10, 0.05],
                     [0.90, 0.90, 0.30, 0.20],
                     [0.85, 0.60, 0.40, 0.10]], ["a", "b", "c"])
    zero_std = pd.DataFrame(0.0, index=means.index, columns=means.columns)
    assert (is_pareto_optimal_uncertain(means, zero_std, 0.0)
            == is_pareto_optimal(means.values)).all()


def test_uncertainty_aware_dominance_keeps_statistically_close_candidates() -> None:
    # point estimates say "a dominates b"; one std of noise makes it a tie
    means = _matrix([[0.90, 0.80, 0.50, 0.20], [0.89, 0.79, 0.50, 0.20]], ["a", "b"])
    stds = pd.DataFrame(0.02, index=means.index, columns=means.columns)

    assert is_pareto_optimal(means.values).tolist() == [True, False]
    assert is_pareto_optimal_uncertain(means, stds, margin=1.0).tolist() == [True, True]
    # a candidate that is clearly better on every objective still dominates
    strong = _matrix([[0.95, 0.90, 0.60, 0.30], [0.89, 0.79, 0.50, 0.20]], ["a", "b"])
    assert is_pareto_optimal_uncertain(strong, stds, margin=1.0).tolist() == [True, False]


def test_admissibility_thresholds_filter_candidates() -> None:
    means = _matrix([[0.90, 0.60, 1.00, 0.05],
                     [0.88, 0.90, 0.40, 0.20]], ["standard", "trades"])
    mask = apply_admissibility(means, min_phi2=0.80, min_phi4=0.10)
    assert mask.tolist() == [False, True]
    assert apply_admissibility(means).all()          # thresholds off by default


def test_objective_matrices_report_per_seed_dispersion() -> None:
    mean_results = pd.DataFrame({
        "model": ["standard", "standard", "trades", "trades"],
        "attack": ["clean", "pgd", "clean", "pgd"],
        "epsilon": [0.0, 0.10, 0.0, 0.10],
        "f1": [0.90, 0.60, 0.88, 0.84],
        "attack_success_rate": [0.0, 0.40, 0.0, 0.16],
    })
    std_results = pd.DataFrame({
        "model": ["standard", "standard", "trades", "trades"],
        "attack": ["clean", "pgd", "clean", "pgd"],
        "epsilon": [0.0, 0.10, 0.0, 0.10],
        "f1": [0.010, 0.020, 0.008, 0.012],
        "attack_success_rate": [0.0, 0.050, 0.0, 0.020],
    })
    efficiency_raw = pd.DataFrame({
        "seed": [1, 1, 2, 2],
        "model": ["standard", "trades", "standard", "trades"],
        "train_seconds": [100.0, 400.0, 110.0, 330.0],
    })
    category_raw = pd.DataFrame({
        "seed": [1, 1, 2, 2],
        "model": ["standard", "trades", "standard", "trades"],
        "attack": ["pgd"] * 4,
        "epsilon": [0.10] * 4,
        "attack_cat": ["A", "A", "B", "B"],
        "adv_recall": [0.30, 0.70, 0.10, 0.80],
    })
    data = {"mean_results": mean_results, "std_results": std_results,
            "efficiency_mean": pd.DataFrame({"model": ["standard", "trades"],
                                             "relative_train_cost_vs_standard": [1.0, 4.0]}),
            "efficiency_raw": efficiency_raw,
            "category_mean_results": pd.DataFrame({
                "model": ["standard", "trades"], "attack": ["pgd", "pgd"],
                "epsilon": [0.10, 0.10], "adv_recall": [0.20, 0.75]}),
            "category_raw_results": category_raw}

    means, stds = build_objective_matrices(data, ref_attack="pgd", ref_epsilon=0.10)
    assert means.loc["trades", "phi2"] == 0.84
    assert np.isclose(stds.loc["standard", "phi2"], 0.05)
    assert np.isclose(stds.loc["trades", "phi1"], 0.008)
    # phi3 = 1 / relative cost (per-seed mean of 4.0 and 3.0 -> 3.5)
    assert np.isclose(means.loc["trades", "phi3"], 1 / 3.5)
    # phi4: per-seed min over categories, then mean  (0.30 and 0.10 -> 0.20)
    assert np.isclose(means.loc["standard", "phi4"], 0.20)
    assert np.isclose(stds.loc["standard", "phi4"], np.std([0.30, 0.10], ddof=1))


def test_outputs_root_is_resolved_against_the_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--outputs-root must not follow the current working directory."""
    monkeypatch.chdir(tmp_path)
    assert resolve_outputs_root("outputs") == PROJECT_ROOT / "outputs"

    absolute = tmp_path / "somewhere"
    assert resolve_outputs_root(str(absolute)) == absolute


def test_backbone_constants_cover_the_decision_layer() -> None:
    """FT-Transformer must be part of the decision layer, CIC-IDS2017 must not."""
    assert set(BACKBONE_LABELS) == {"MLP", "CNN", "FT-Trans"}
    assert set(BACKBONE_LABELS) == set(BACKBONE_DIRS) - {"CICIDS"}
    assert set(BACKBONE_CLI_KEYS) == set(BACKBONE_LABELS)
    assert set(BACKBONE_SELECTION_COLUMNS) == set(BACKBONE_LABELS)
    assert set(BACKBONE_CLI_KEYS.values()) == {"mlp", "cnn", "ft"}


def test_selection_csv_has_one_column_per_backbone(tmp_path: Path) -> None:
    """The selection table must cover MLP, 1D-CNN and FT-Transformer."""
    objectives = {
        key: pd.DataFrame(
            [[0.90 - 0.01 * index, 0.80 - 0.02 * index, 0.50, 0.40 + 0.05 * index]
             for index in range(3)],
            index=["standard", "trades", "free_at"],
            columns=["phi1", "phi2", "phi3", "phi4"],
        )
        for key in BACKBONE_LABELS
    }
    path = tmp_path / "pareto_selection_results.csv"
    write_selection_csv(objectives, str(path))

    frame = pd.read_csv(path)
    assert len(frame) == len(THETA_PRESETS)
    assert set(frame.columns) == {
        "theta_name", "theta_values",
        "mlp_selected", "cnn_selected", "ft_selected",
    }
    for column in ("mlp_selected", "cnn_selected", "ft_selected"):
        assert not (frame[column] == "N/A").any()
