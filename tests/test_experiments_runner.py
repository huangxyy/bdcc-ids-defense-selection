"""The orchestrator must forward large-GPU scaling flags to every backbone."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_experiments.py"


def _dry_run(*extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), "--dry-run", *extra],
        capture_output=True, text=True, timeout=300, check=False,
    )


def test_dry_run_forwards_scaling_flags() -> None:
    completed = _dry_run("--backbones", "mlp,cnn", "--seeds", "7,13",
                         "--epsilon-list", "0.02,0.05",
                         "--eval-attack-rows", "82332", "--adaptive-eval")
    assert completed.returncode == 0, completed.stderr
    for flag in ("--seeds 7,13", "--epsilon-list 0.02,0.05",
                 "--eval-attack-rows 82332", "--adaptive-eval"):
        assert completed.stdout.count(flag) == 2, completed.stdout


def test_ft_capacity_preset_only_touches_the_ft_command() -> None:
    completed = _dry_run("--backbones", "mlp,ft", "--ft-capacity", "medium")
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if ".py " in line]
    mlp_line = next(line for line in lines if "run_mlp.py" in line)
    ft_line = next(line for line in lines if "run_ft_transformer.py" in line)
    assert "--ft-d-token" not in mlp_line
    assert "--ft-d-token 64 --ft-n-heads 4 --ft-n-layers 3 --ft-d-ffn 128" in ft_line


def test_set_overrides_any_config_field() -> None:
    completed = _dry_run("--backbones", "ft",
                         "--set", "baseline_epochs=20",
                         "--set", "ft_d_token=96", "--set", "ft-n-heads=6")
    assert completed.returncode == 0, completed.stderr
    ft_line = next(line for line in completed.stdout.splitlines()
                   if "run_ft_transformer.py" in line)
    assert "--baseline-epochs 20" in ft_line
    assert "--ft-d-token 96" in ft_line
    assert "--ft-n-heads 6" in ft_line


def test_set_wins_over_the_capacity_preset() -> None:
    completed = _dry_run("--backbones", "ft", "--ft-capacity", "medium",
                         "--set", "ft_d_token=96", "--set", "ft_n_heads=8")
    assert completed.returncode == 0, completed.stderr
    ft_line = next(line for line in completed.stdout.splitlines()
                   if "run_ft_transformer.py" in line)
    assert "--ft-d-token 96" in ft_line and "--ft-d-token 64" not in ft_line
    assert "--ft-n-heads 8" in ft_line and "--ft-n-heads 4" not in ft_line


def test_unknown_set_field_is_rejected() -> None:
    completed = _dry_run("--set", "not_a_field=1")
    assert completed.returncode == 2
    assert "unknown config field" in completed.stderr


def test_set_rejects_orchestrator_owned_fields() -> None:
    completed = _dry_run("--set", "output_dir=/tmp/clobber")
    assert completed.returncode == 2
    assert "not allowed" in completed.stderr


def test_set_rejects_invalid_boolean_and_empty_values() -> None:
    invalid_bool = _dry_run("--set", "adaptive_eval=maybe")
    assert invalid_bool.returncode == 2
    assert "adaptive_eval" in invalid_bool.stderr

    empty = _dry_run("--set", "dropout=")
    assert empty.returncode == 2
    assert "needs a value" in empty.stderr


def test_adaptive_tuning_without_eval_warns_and_is_not_forwarded_alone() -> None:
    completed = _dry_run("--backbones", "mlp", "--adaptive-steps", "100")
    assert completed.returncode == 0
    assert "no effect" in completed.stderr


def test_ft_capacity_warns_when_ft_is_not_selected() -> None:
    completed = _dry_run("--backbones", "mlp", "--ft-capacity", "large")
    assert completed.returncode == 0
    assert "--ft-capacity is ignored" in completed.stderr


def test_forwarded_values_are_accepted_by_a_backbone(tmp_path: Path) -> None:
    """Whatever the orchestrator prints must parse in the backbone config CLI."""
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_mlp.py"),
         "--train-path", str(PROJECT_ROOT / "data" / "train.csv"),
         "--test-path", str(PROJECT_ROOT / "data" / "test.csv"),
         "--output-dir", str(tmp_path), "--device", "cpu",
         "--training-budget-mode", "matched_continuation",
         "--seeds", "7,13", "--epsilon-list", "0.02,0.05",
         "--eval-attack-rows", "82332",
         "--adaptive-eval", "--adaptive-steps", "100", "--adaptive-restarts", "10",
         "--baseline-epochs", "20", "--batch-size", "2048",
         "--hidden-dims", "512,256,128",
         "--transfer-attack-settings", "fgsm:0.05,pgd:0.1",
         "--ft-d-token", "96", "--ft-n-heads", "6", "--print-config"],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["evaluation"]["seeds"] == [7, 13]
    assert payload["evaluation"]["eval_attack_rows"] == 82332
    assert payload["evaluation"]["adaptive_eval"] is True
    assert payload["evaluation"]["transfer_attack_settings"] == [["fgsm", 0.05], ["pgd", 0.1]]
    assert payload["training"]["batch_size"] == 2048
    assert payload["training"]["hidden_dims"] == [512, 256, 128]
    assert payload["training"]["ft_d_token"] == 96
