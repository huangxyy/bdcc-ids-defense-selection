"""Supported vs unsupported efficient solutions (C6)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from ids_defense_selection.supportedness import (
    supported_solutions,
    tchebycheff_reachability,
    tchebycheff_selection,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "check_supportedness.py"


def test_classic_non_convex_front_has_one_unsupported_point() -> None:
    """A, B, C lie on x+y=1; D=(0.4, 0.45) is efficient but below the hull."""
    objectives = pd.DataFrame(
        [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.4, 0.45]],
        index=["A", "B", "C", "D"], columns=["phi1", "phi2"],
    )
    table = supported_solutions(objectives).set_index("model")
    assert bool(table.loc["A", "supported"])
    assert bool(table.loc["B", "supported"])
    assert bool(table.loc["C", "supported"])
    assert not bool(table.loc["D", "supported"])
    assert table.loc["D", "margin"] < 0


def test_supported_solution_reports_a_preference_vector() -> None:
    objectives = pd.DataFrame(
        [[0.9, 0.2, 0.5, 0.5], [0.3, 0.9, 0.5, 0.5]],
        index=["clean", "robust"],
        columns=["phi1", "phi2", "phi3", "phi4"],
    )
    table = supported_solutions(objectives).set_index("model")
    assert bool(table.loc["clean", "supported"]) and bool(table.loc["robust", "supported"])
    assert abs(table.loc["clean", ["theta1", "theta2", "theta3", "theta4"]].sum() - 1) < 1e-6


def test_tchebycheff_reaches_an_unsupported_point() -> None:
    """D=(0.55,0.42) is efficient but not a maximiser of any weighted sum."""
    objectives = pd.DataFrame(
        [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.55, 0.42]],
        index=["A", "B", "C", "D"], columns=["phi1", "phi2"],
    )
    assert not bool(supported_solutions(objectives).set_index("model")
                    .loc["D", "supported"])
    assert tchebycheff_selection(objectives, (0.6, 0.4)) == "D"
    reach = tchebycheff_reachability(objectives, step=0.1).set_index("model")
    assert bool(reach.loc["D", "reachable_tchebycheff"])
    assert reach.loc["D", "theta_tchebycheff"] != ""


def test_script_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--outputs-root" in completed.stdout
