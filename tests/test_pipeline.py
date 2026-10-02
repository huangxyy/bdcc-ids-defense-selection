"""End-to-end test of the training/evaluation pipeline on synthetic data.

The dataset is tiny (a few hundred rows), so the test runs in seconds on CPU
while still exercising feature engineering, the six defenses, the attack suite,
the evaluation loop and the output writer.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ids_defense_selection import (
    DEFENSE_ORDER,
    EvaluationSet,
    ExperimentConfig,
    MLPBackbone,
    build_features,
    compute_significance_tests,
    evaluate_defenses,
    load_unsw_nb15,
    make_dataloader,
    stratified_subset_indices,
    train_all_defenses,
)
from ids_defense_selection.reporting import BackboneRunFrames, write_backbone_outputs


def _synthetic_split(n_train: int = 320, n_test: int = 160) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)

    def frame(n: int) -> pd.DataFrame:
        labels = rng.integers(0, 2, size=n)
        data = {
            "id": np.arange(n),
            "dur": rng.random(n),
            "spkts": rng.integers(1, 100, size=n),
            "dpkts": rng.integers(1, 100, size=n),
            "sbytes": rng.integers(0, 10000, size=n),
            "dbytes": rng.integers(0, 10000, size=n),
            "rate": rng.random(n) * 1000,
            "proto": rng.choice(["tcp", "udp", "arp"], size=n),
            "service": rng.choice(["-", "http", "dns"], size=n),
            "state": rng.choice(["FIN", "INT", "CON"], size=n),
            "attack_cat": rng.choice(["Normal", "Exploits", "Fuzzers", "Reconnaissance"],
                                     size=n, p=[0.4, 0.25, 0.2, 0.15]),
            "label": labels,
        }
        return pd.DataFrame(data)

    return frame(n_train), frame(n_test)


@pytest.fixture(scope="module")
def synthetic_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict:
    data_dir = tmp_path_factory.mktemp("data")
    train_df, test_df = _synthetic_split()
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return {"train_path": str(train_path), "test_path": str(test_path),
            "train_df": train_df, "test_df": test_df}


def _config(synthetic_dataset: dict, output_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        train_path=synthetic_dataset["train_path"],
        test_path=synthetic_dataset["test_path"],
        output_dir=str(output_dir),
        training_budget_mode="matched_continuation",
        batch_size=128,
        baseline_epochs=1,
        adv_epochs=1,
        hidden_dims=(16, 8, 4),
        dropout=0.1,
        eval_attack_rows=64,
        adv_steps=2,
        eval_pgd_steps=2,
        epsilon_list=(0.05,),
        seeds=(1,),
        sensitivity_batches=1,
        transfer_attack_settings=(),
        validity_attack_settings=(("fgsm", 0.05),),
        reference_models=(),
        cw_steps=2,
        apgd_steps=3,
    )


def test_training_evaluation_and_outputs(synthetic_dataset: dict, tmp_path: Path) -> None:
    config = _config(synthetic_dataset, tmp_path / "run")
    device = torch.device("cpu")

    train_df, test_df = load_unsw_nb15(config.train_path, config.test_path)
    x_train, y_train, x_test, y_test, metadata = build_features(train_df, test_df)
    eval_indices = stratified_subset_indices(y_test, config.eval_attack_rows, seed=2026)

    train_loader = make_dataloader(x_train, y_train, batch_size=config.batch_size, shuffle=True)
    trained = train_all_defenses(
        model_factory=lambda: MLPBackbone(x_train.shape[1], config.hidden_dims, config.dropout),
        config=config,
        train_loader=train_loader,
        device=device,
        metadata=metadata,
        seed=1,
    )
    assert set(DEFENSE_ORDER) <= set(trained.models)
    assert set(trained.train_cost) >= set(DEFENSE_ORDER)

    eval_set = EvaluationSet(
        x_eval=x_test[eval_indices],
        y_eval=y_test[eval_indices],
        x_test=x_test,
        y_test=y_test,
        attack_mask=metadata["numeric_mask"],
        numeric_mins=metadata["numeric_mins"],
        numeric_maxs=metadata["numeric_maxs"],
    )
    evaluation = evaluate_defenses(
        trained, config, eval_set, device, seed=1,
        include_categories=False,
        full_test_attack_settings=(("pgd", 0.05),),
    )
    assert {"clean", "fgsm", "pgd", "cw", "apgd"} <= set(evaluation.metrics["attack"])
    assert len(evaluation.efficiency) == len(DEFENSE_ORDER)
    assert not evaluation.validity.empty
    assert not evaluation.full_test_clean.empty
    assert not evaluation.full_test_attack.empty

    aggregated = write_backbone_outputs(
        Path(config.output_dir),
        config,
        BackboneRunFrames(
            results=evaluation.metrics,
            sensitivity=trained.sensitivity_table,
            efficiency=evaluation.efficiency,
            validity=evaluation.validity,
            full_test_clean=evaluation.full_test_clean,
            full_test_attack=evaluation.full_test_attack,
        ),
        summary={"architecture": "test"},
    )
    for name in ("mean_results.csv", "raw_results.csv", "efficiency_mean.csv",
                 "run_summary.json", "f1_curve.png"):
        assert (Path(config.output_dir) / name).exists(), f"missing output {name}"
    assert "standard" in set(aggregated["mean_results"]["model"])


def test_significance_tests_compare_defense_pairs() -> None:
    rows = []
    # the per-seed difference varies so the paired tests are well conditioned
    for seed, (base, gain) in enumerate([(0.80, 0.05), (0.84, 0.03), (0.82, 0.06)], start=1):
        for model, offset in (("constrained_adv", gain), ("standard", 0.0)):
            rows.append({
                "seed": seed, "model": model, "attack": "clean", "epsilon": 0.0,
                "f1": base + offset, "attack_success_rate": 0.2 - gain,
            })
    significance = compute_significance_tests(pd.DataFrame(rows), epsilon_list=(0.05,))
    assert not significance.empty
    assert set(significance["comparison"]) == {"constrained_adv_vs_standard"}


def test_significance_tests_survive_degenerate_samples() -> None:
    """Two seeds with identical defences must not warn or crash the run."""
    rows = [
        {"seed": seed, "model": model, "attack": "clean", "epsilon": 0.0,
         "f1": 0.8, "attack_success_rate": 0.0}
        for seed in (1, 2)
        for model in ("constrained_adv", "standard")
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        significance = compute_significance_tests(pd.DataFrame(rows), epsilon_list=(0.05,))
    assert not significance.empty
    clean = significance[significance["metric"] == "f1"]
    assert set(clean["n_seeds"]) == {2}
    assert set(clean["note"]) == {"identical paired samples (no effect to test)"}
