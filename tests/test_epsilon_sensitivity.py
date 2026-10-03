"""The epsilon-sensitivity exporter must run on an outputs tree."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "export_epsilon_sensitivity.py"


def _write_backbone(root: Path, subdir: str) -> None:
    run = root / subdir
    run.mkdir(parents=True)
    rows = []
    for epsilon in (0.02, 0.05, 0.10, 0.20):
        for index, model in enumerate(("standard", "trades")):
            rows.append({"model": model, "attack": "pgd", "epsilon": epsilon,
                         "f1": 0.85, "attack_success_rate": 0.02 * (index + 1) * epsilon})
    mean = pd.DataFrame(rows)
    mean.to_csv(run / "mean_results.csv", index=False)
    std = mean.copy()
    std["attack_success_rate"] = 0.001
    std.to_csv(run / "std_results.csv", index=False)


def test_exporter_writes_table_and_figure(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    for subdir in ("mlp", "cnn1d", "ft_transformer"):
        _write_backbone(root, subdir)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--outputs-root", str(root), "--attack", "pgd"],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    tables = root / "tables"
    assert (tables / "epsilon_sensitivity.csv").is_file()
    markdown = (tables / "epsilon_sensitivity.md").read_text(encoding="utf-8")
    assert "ε=0.1" in markdown and "Standard" in markdown
    assert (tables / "epsilon_sensitivity_pgd.png").is_file()
