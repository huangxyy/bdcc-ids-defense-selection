"""Structure tests: the src/scripts split must stay consistent and runnable."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"


def test_standard_layout() -> None:
    """The library lives in src/; the old grab-bag code/ directory is gone."""
    assert (SRC / "ids_defense_selection" / "__init__.py").is_file()
    assert (PROJECT_ROOT / "docs" / "structure.md").is_file()
    assert (PROJECT_ROOT / "docs" / "verification.md").is_file()
    assert not (PROJECT_ROOT / "code").exists()


def test_every_script_bootstraps_the_library() -> None:
    """A script that imports the library must first put src/ on sys.path."""
    missing: list[str] = []
    for script in sorted(SCRIPTS.glob("*.py")):
        if script.name == "_bootstrap.py":
            continue
        text = script.read_text(encoding="utf-8")
        if "ids_defense_selection" in text and "import _bootstrap" not in text:
            missing.append(script.name)
    assert not missing, f"scripts without _bootstrap: {missing}"


def test_entry_point_runs_from_any_working_directory(tmp_path: Path) -> None:
    """`python <repo>/scripts/run_mlp.py --help` must work outside the repo."""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_mlp.py"), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--train-path" in completed.stdout
    assert "--seeds" in completed.stdout
