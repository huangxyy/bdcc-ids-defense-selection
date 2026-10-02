"""Checkpoint bundles: train once, evaluate many times."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ids_defense_selection import (
    ExperimentConfig,
    MLPBackbone,
    load_checkpoint,
    rebuild_trained_defenses,
    save_checkpoint,
)
from ids_defense_selection.defenses import TrainedDefenses

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_checkpoints.py"


def _config() -> ExperimentConfig:
    return ExperimentConfig(train_path="train.csv", test_path="test.csv",
                            hidden_dims=(4, 3, 2), dropout=0.15)


def _trained(input_dim: int = 5) -> TrainedDefenses:
    torch.manual_seed(0)
    model = MLPBackbone(input_dim, (4, 3, 2), 0.15)
    other = MLPBackbone(input_dim, (4, 3, 2), 0.15)
    other.load_state_dict(model.state_dict())
    with torch.no_grad():
        other.net[0].weight.add_(0.01)
    return TrainedDefenses(
        baseline_model=model,
        models={"standard": model, "adv_training": other},
        train_cost={"standard": {"train_seconds": 1.0},
                    "adv_training": {"train_seconds": 2.0}},
        perturb_ratio={"standard": 0.0, "adv_training": 1.0},
        perturb_features={"standard": 0, "adv_training": input_dim},
        selected_mask=np.array([True, True, False, False, False]),
        class_aware_mask=np.array([False, True, False, False, False]),
        sensitivity_table=pd.DataFrame({"feature": list(range(input_dim)),
                                        "score": np.linspace(0, 1, input_dim)}),
    )


def test_checkpoint_roundtrip_restores_weights_and_masks(tmp_path: Path) -> None:
    trained = _trained()
    directory = save_checkpoint(
        tmp_path / "run", trained, _config(), seed=7, backbone="mlp", input_dim=5,
        dataset_split={"train_rows": 175341, "test_rows": 82332,
                       "official_direction": True},
    )
    assert (directory / "trained_defenses.pt").is_file()
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["backbone"] == "mlp"
    assert metadata["seed"] == 7
    assert set(metadata["defenses"]) == {"standard", "adv_training"}

    bundle = load_checkpoint(directory)
    rebuilt = rebuild_trained_defenses(bundle, "cpu")
    for name, model in trained.models.items():
        for key, value in model.state_dict().items():
            assert torch.allclose(value, rebuilt.models[name].state_dict()[key])
    assert np.array_equal(rebuilt.selected_mask, trained.selected_mask)
    assert rebuilt.sensitivity_table.shape == trained.sensitivity_table.shape
    assert rebuilt.baseline_model is not None


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does-not-exist")


def test_eval_only_script_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT), "--help"],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--checkpoint" in completed.stdout
    assert "--adaptive-eval" in completed.stdout
