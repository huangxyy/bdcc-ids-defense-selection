"""Put the in-repository library (``src/``) on ``sys.path``.

Every entry point in ``scripts/`` is executed by *file path*, so Python puts
``scripts/`` on ``sys.path`` -- not ``src/``.  Importing this module once at the
top of a script makes ``import ids_defense_selection`` work no matter which
working directory the script is started from, without requiring the project to
be built or installed:

    uv run python scripts/run_mlp.py --help

Tests do not use this module: ``pytest`` is pointed at ``src/`` through
``[tool.pytest.ini_options] pythonpath`` in ``pyproject.toml``.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
