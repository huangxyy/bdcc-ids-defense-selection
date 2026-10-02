"""The experiment logger must turn a run directory into a traceable record."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = PROJECT_ROOT / "scripts" / "log_experiment.py"


def _write_run(root: Path) -> Path:
    run = root / "ft_probe"
    run.mkdir(parents=True)
    models = ["standard", "adv_training", "constrained_adv",
              "trades", "free_at", "class_aware_constrained"]
    rows = []
    for seed in (7, 13):
        for index, model in enumerate(models):
            rows.append({"seed": seed, "model": model, "attack": "pgd",
                         "epsilon": 0.10, "f1": 0.85 - 0.01 * index,
                         "attack_success_rate": 0.05 - 0.005 * index})
    raw = pd.DataFrame(rows)
    raw.to_csv(run / "raw_results.csv", index=False)
    group = ["model", "attack", "epsilon"]
    raw.groupby(group, as_index=False)[["f1", "attack_success_rate"]].mean() \
        .to_csv(run / "mean_results.csv", index=False)
    raw.groupby(group, as_index=False)[["f1", "attack_success_rate"]].std() \
        .to_csv(run / "std_results.csv", index=False)
    pd.DataFrame({"model": models, "train_seconds": [100.0] * len(models),
                  "inference_ms_per_sample": [0.006] * len(models)}) \
        .to_csv(run / "efficiency_mean.csv", index=False)
    (run / "run_summary.json").write_text(json.dumps({
        "resolved_device": "cuda",
        "dataset_split": {"train_rows": 175341, "test_rows": 82332,
                          "official_direction": True},
        "config": {"evaluation": {"eval_attack_rows": 82332,
                                  "epsilon_list": [0.05, 0.10],
                                  "adaptive_eval": True},
                   "training": {"baseline_epochs": 10, "adv_epochs": 8,
                                "batch_size": 512, "learning_rate": 0.001,
                                "ft_d_token": 32, "ft_n_heads": 2,
                                "ft_n_layers": 2, "ft_d_ffn": 64}},
    }), encoding="utf-8")
    return run


def test_logger_writes_record_and_index(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    records = tmp_path / "records"
    completed = subprocess.run(
        [sys.executable, str(LOGGER), "--outputs-dir", str(run),
         "--record-dir", str(records), "--title", "FT probe test",
         "--status", "done", "--command", "uv run python scripts/run_ft_transformer.py",
         "--note", "synthetic fixture"],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert completed.returncode == 0, completed.stderr

    record = next(path for path in records.glob("*.md") if path.name != "INDEX.md")
    text = record.read_text(encoding="utf-8")
    assert "# 实验记录：FT probe test" in text
    assert "## 关键结果（pgd，ε=0.1）" in text
    assert "adv_training" in text and "class_aware_constrained" in text
    assert "## 分析" in text and "## 结论 / 下一步" in text
    assert "synthetic fixture" in text
    assert "official_direction" in text

    index = (records / "INDEX.md").read_text(encoding="utf-8")
    assert "FT probe test" in index and record.name in index

    # Logging the same title twice must not overwrite the first record.
    subprocess.run(
        [sys.executable, str(LOGGER), "--outputs-dir", str(run),
         "--record-dir", str(records), "--title", "FT probe test"],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert len(list(records.glob("*.md"))) == 3  # two records + INDEX.md
