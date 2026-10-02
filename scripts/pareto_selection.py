#!/usr/bin/env python
"""CLI entry point for the Pareto filtering and preference-aware selection.

The analysis itself lives in :mod:`ids_defense_selection.selection`, so it can
be imported and tested without running a script; this file only wires up the
command line.

    uv run python scripts/pareto_selection.py --ref-attack pgd --ref-epsilon 0.10
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path when run by path)

from ids_defense_selection.selection import main

if __name__ == "__main__":
    raise SystemExit(main())
