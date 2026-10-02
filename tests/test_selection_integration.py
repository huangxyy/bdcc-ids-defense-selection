"""Integration test: the decision script must cover all three backbones."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from ids_defense_selection.selection import main

BACKBONE_DIRS = ("mlp", "cnn1d", "ft_transformer")
MODELS = ("standard", "trades", "free_at")


def _write_backbone(root: Path, subdir: str, offset: float) -> None:
    """Minimal but complete output tree for one backbone."""
    out = root / subdir
    out.mkdir(parents=True)
    mean_rows, std_rows = [], []
    for index, model in enumerate(MODELS):
        mean_rows.append({"model": model, "attack": "clean", "epsilon": 0.0,
                          "f1": 0.90 - 0.02 * index - offset,
                          "attack_success_rate": 0.0})
        std_rows.append({"model": model, "attack": "clean", "epsilon": 0.0,
                         "f1": 0.010, "attack_success_rate": 0.0})
        for attack, epsilon, base_asr in (("pgd", 0.02, 0.20),
                                          ("pgd", 0.05, 0.30),
                                          ("pgd", 0.10, 0.40),
                                          ("fgsm", 0.05, 0.35),
                                          ("cw", 0.10, 0.45),
                                          ("apgd", 0.10, 0.50)):
            mean_rows.append({"model": model, "attack": attack, "epsilon": epsilon,
                              "f1": 0.60 - 0.02 * index,
                              "attack_success_rate": base_asr - 0.10 * index})
            std_rows.append({"model": model, "attack": attack, "epsilon": epsilon,
                             "f1": 0.020, "attack_success_rate": 0.030})
    pd.DataFrame(mean_rows).to_csv(out / "mean_results.csv", index=False)
    pd.DataFrame(std_rows).to_csv(out / "std_results.csv", index=False)
    pd.DataFrame({"model": MODELS,
                  "relative_train_cost_vs_standard": [1.0, 4.0, 3.0]}) \
        .to_csv(out / "efficiency_mean.csv", index=False)
    pd.DataFrame({"seed": [1, 1, 1, 2, 2, 2],
                  "model": list(MODELS) * 2,
                  "train_seconds": [100.0, 400.0, 300.0, 100.0, 380.0, 290.0]}) \
        .to_csv(out / "efficiency_raw.csv", index=False)
    category_rows = []
    for seed in (1, 2):
        for index, model in enumerate(MODELS):
            for category, value in (("A", 0.20 + 0.10 * index),
                                    ("B", 0.75 - 0.05 * index)):
                category_rows.append({"seed": seed, "model": model,
                                      "attack": "pgd", "epsilon": 0.10,
                                      "attack_cat": category,
                                      "adv_recall": value - offset})
    pd.DataFrame(category_rows).to_csv(out / "category_raw_results.csv", index=False)
    pd.DataFrame(category_rows).groupby(
        ["model", "attack", "epsilon"], as_index=False)["adv_recall"].min() \
        .to_csv(out / "category_mean_results.csv", index=False)


def test_main_covers_every_backbone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for subdir, offset in zip(BACKBONE_DIRS, (0.0, 0.005, 0.010)):
        _write_backbone(tmp_path, subdir, offset)
    monkeypatch.setattr(sys, "argv", [
        "pareto_selection.py", "--outputs-root", str(tmp_path),
        "--confidence-margin", "1.0",
    ])

    assert main() == 0

    for subdir in BACKBONE_DIRS:
        assert (tmp_path / subdir / "risk_profile_4d.csv").is_file()
        comparison = pd.read_csv(tmp_path / subdir / "decision_comparators.csv")
        assert {"pareto_weighted", "weighted_no_pareto", "weighted_fixed01",
                "topsis_no_pareto", "topsis_pareto"} <= set(comparison.columns)
        dependence = pd.read_csv(tmp_path / subdir / "candidate_dependence.csv")
        assert {"theta_name", "removed", "selection", "selection_without",
                "changed"} <= set(dependence.columns)
        assert (tmp_path / subdir / "theta_sweep.csv").is_file()
        summary = pd.read_csv(tmp_path / subdir / "theta_summary.csv")
        assert {"defense", "n_regions", "share"} <= set(summary.columns)
        assert (tmp_path / subdir / "switching_regions.json").is_file()
        admissibility = pd.read_csv(tmp_path / subdir / "admissibility_sweep.csv")
        assert {"tau2", "tau4", "n_admissible", "pareto_size",
                "no_candidate"} <= set(admissibility.columns)
    settings = json.loads((tmp_path / "decision_settings.json").read_text(encoding="utf-8"))
    assert set(settings["backbones"]) == {"MLP", "CNN", "FT-Trans"}
    selection = pd.read_csv(tmp_path / "mlp" / "pareto_selection_results.csv")
    assert {"mlp_selected", "cnn_selected", "ft_selected"} <= set(selection.columns)
    figures = tmp_path / "figures"
    for name in ("pareto_front_comparison.png", "risk_surface_heatmap.png",
                 "epsilon_pareto_evolution.png", "theta_sensitivity.png"):
        assert (figures / name).stat().st_size > 0
