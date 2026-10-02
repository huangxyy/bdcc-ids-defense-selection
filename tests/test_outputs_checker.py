"""Tests for scripts/check_outputs.py: the artefact gate must fail loudly."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKER = PROJECT_ROOT / "scripts" / "check_outputs.py"


def _write_minimal_run(root: Path) -> Path:
    """Create the smallest outputs/mlp tree the checker accepts."""
    out = root / "mlp"
    out.mkdir(parents=True)
    models = ["standard", "adv_training", "constrained_adv",
              "trades", "free_at", "class_aware_constrained"]
    rows = []
    for seed in (7, 13, 21, 42, 100):
        for index, model in enumerate(models):
            rows.append({"seed": seed, "model": model, "attack": "pgd",
                         "epsilon": 0.10, "f1": 0.50 + 0.01 * index,
                         "attack_success_rate": 0.20 - 0.01 * index})
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "raw_results.csv", index=False)
    group = ["model", "attack", "epsilon"]
    raw.groupby(group, as_index=False)[["f1", "attack_success_rate"]].mean() \
        .to_csv(out / "mean_results.csv", index=False)
    raw.groupby(group, as_index=False)[["f1", "attack_success_rate"]].std() \
        .to_csv(out / "std_results.csv", index=False)
    pd.DataFrame({"model": models, "train_seconds": 10.0,
                  "inference_ms_per_sample": 0.1}).to_csv(out / "efficiency_raw.csv", index=False)
    pd.DataFrame({"model": models, "train_seconds": 10.0,
                  "inference_ms_per_sample": 0.1}).to_csv(out / "efficiency_mean.csv", index=False)
    pd.DataFrame({"comparison": ["trades_vs_adv_training"], "attack": ["pgd"],
                  "epsilon": [0.10], "metric": ["f1"], "n_seeds": [5],
                  "t_pvalue": [0.01], "wilcoxon_pvalue": [0.06]}) \
        .to_csv(out / "significance_tests.csv", index=False)
    pd.DataFrame({"group": ["training"], "field": ["batch_size"],
                  "value": ["1024"], "help": ["batch size"]}) \
        .to_csv(out / "hyperparameters.csv", index=False)
    pd.DataFrame({"model": models, "attack": ["fgsm"] * len(models),
                  "epsilon": [0.10] * len(models), "f1": [0.5] * len(models),
                  "attack_success_rate": [0.2] * len(models),
                  "seen_in_training": [False] * len(models),
                  "asr_gap_vs_pgd": [0.0] * len(models)}) \
        .to_csv(out / "attack_generalization.csv", index=False)
    (out / "run_summary.json").write_text(json.dumps({
        "resolved_device": "cpu",
        "dataset_split": {"train_rows": 175341, "test_rows": 82332,
                          "official_direction": True},
    }), encoding="utf-8")
    return out


def _run_checker(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(root), "--backbones", "mlp"],
        capture_output=True, text=True, timeout=300, check=False,
    )


def _run_checker_require_checkpoints(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(root), "--backbones", "mlp",
         "--require-checkpoints"],
        capture_output=True, text=True, timeout=300, check=False,
    )


def test_checker_accepts_a_complete_run(tmp_path: Path) -> None:
    _write_minimal_run(tmp_path)
    completed = _run_checker(tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "All checks passed" in completed.stdout


def test_checker_rejects_a_missing_artefact(tmp_path: Path) -> None:
    out = _write_minimal_run(tmp_path)
    (out / "mean_results.csv").unlink()
    completed = _run_checker(tmp_path)
    assert completed.returncode == 1
    assert "missing mean_results.csv" in completed.stdout


def test_checker_requires_checkpoints_when_asked(tmp_path: Path) -> None:
    out = _write_minimal_run(tmp_path)
    completed = _run_checker_require_checkpoints(tmp_path)
    assert completed.returncode == 1
    assert "no checkpoints" in completed.stdout

    checkpoint = out / "checkpoints" / "seed7"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trained_defenses.pt").write_bytes(b"stub")
    (checkpoint / "metadata.json").write_text("{}", encoding="utf-8")
    assert _run_checker_require_checkpoints(tmp_path).returncode == 0
