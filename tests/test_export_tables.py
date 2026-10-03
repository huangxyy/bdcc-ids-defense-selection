"""The manuscript-table exporter must run on an outputs tree."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "export_paper_tables.py"


def _write_backbone(root: Path, subdir: str) -> None:
    run = root / subdir
    run.mkdir(parents=True)
    pd.DataFrame({
        "model": ["standard", "trades"],
        "phi1_clean_f1": [0.87, 0.85], "phi2_resilience": [0.84, 0.99],
        "phi3_cost_eff": [1.0, 0.3], "phi4_fairness": [0.36, 0.96],
        "phi1_std": [0.01, 0.01], "phi2_std": [0.02, 0.001],
        "phi3_std": [0.0, 0.01], "phi4_std": [0.03, 0.01],
        "is_pareto_optimal": [True, True],
        "is_pareto_optimal_deterministic": [True, True],
        "admissible": [False, True],
    }).to_csv(run / "risk_profile_4d.csv", index=False)
    pd.DataFrame({"model": ["trades"], "supported": [True], "margin": [0.01],
                  "theta1": [0.25], "theta2": [0.25], "theta3": [0.25],
                  "theta4": [0.25]}).to_csv(run / "supportedness.csv", index=False)
    pd.DataFrame({
        "comparison": ["trades_vs_standard"], "attack": ["pgd"], "epsilon": [0.10],
        "metric": ["f1"], "n_seeds": [5], "t_pvalue": [0.01],
        "wilcoxon_pvalue": [0.06], "note": [""], "holm_t_pvalue": [0.02],
        "bh_t_pvalue": [0.02], "holm_wilcoxon_pvalue": [0.06],
        "bh_wilcoxon_pvalue": [0.06], "mean_diff": [0.02], "cohens_dz": [1.5],
        "ci_low": [0.01], "ci_high": [0.03],
        "significant_holm_t": [True], "significant_holm_wilcoxon": [False],
    }).to_csv(run / "significance_enhanced.csv", index=False)


def test_exporter_writes_tables(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    for subdir in ("mlp", "cnn1d", "ft_transformer"):
        _write_backbone(root, subdir)
    pd.DataFrame({
        "theta_name": ["Robust", "Balanced", "Clean", "Cost", "extra_fair"],
        "theta_values": ["(0.1, 0.6, 0.1, 0.2)"] * 5,
        "mlp_selected": ["PGD-AT"] * 5,
        "cnn_selected": ["TRADES"] * 5,
        "ft_selected": ["Class-Aware"] * 5,
    }).to_csv(root / "mlp" / "pareto_selection_results.csv", index=False)
    (root / "decision_settings.json").write_text(
        json.dumps({"confidence_margin": 1.0, "min_phi2": 0.9, "min_phi4": 0.6}),
        encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--outputs-root", str(root)],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    tables = root / "tables"
    assert (tables / "paper_tables.md").is_file()
    assert (tables / "table3_mlp.csv").is_file()
    document = (tables / "paper_tables.md").read_text(encoding="utf-8")
    assert "Table 3." in document
    assert "Table 6." in document
    assert "Table 7." in document
    assert "Significance" in document
