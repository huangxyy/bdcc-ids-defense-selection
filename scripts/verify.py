#!/usr/bin/env python
"""One-command verification of the repository.

Steps, fastest first:

  1. dataset layout        scripts/prepare_data.py
  2. static analysis       ruff check
  3. unit/integration      pytest -q
  4. smoke test            scripts/smoke_test.py   (~1 minute, skipped by --quick)
  5. reproduction dry run  scripts/run_experiments.py --dry-run (skipped by --quick)

The exit code is non-zero if any selected step fails, so this is safe to use
before every commit and before handing the artifact to anyone:

    uv run python scripts/verify.py
    uv run python scripts/verify.py --quick
    uv run python scripts/verify.py --device cuda
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path when run by path)

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent


def run_step(label: str, cmd: list[str]) -> bool:
    """Run one verification step and report PASS/FAIL with its wall time."""
    print(f"\n=== [{label}] " + "-" * max(0, 60 - len(label)))
    print("    " + " ".join(str(part) for part in cmd), flush=True)
    started = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    elapsed = time.perf_counter() - started
    status = "PASS" if result.returncode == 0 else f"FAIL (exit {result.returncode})"
    print(f"=== [{label}] {status} in {elapsed:.1f}s", flush=True)
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="skip the smoke test and the reproduction dry run")
    parser.add_argument("--device", default="auto",
                        help="device forwarded to the smoke test (auto, cpu, cuda, ...)")
    args = parser.parse_args()

    steps: list[tuple[str, list[str], bool]] = [
        ("dataset", [sys.executable, str(SCRIPTS / "prepare_data.py")], True),
        ("lint", [sys.executable, "-m", "ruff", "check"], True),
        ("tests", [sys.executable, "-m", "pytest", "-q"], True),
        ("smoke", [sys.executable, str(SCRIPTS / "smoke_test.py"), "--device", args.device], not args.quick),
        ("dry-run", [sys.executable, str(SCRIPTS / "run_experiments.py"), "--dry-run"], not args.quick),
    ]

    selected = [(label, cmd) for label, cmd, enabled in steps if enabled]
    print("=" * 78)
    print("Repository verification")
    print("=" * 78)
    print(f"  root   : {ROOT}")
    print(f"  steps  : {', '.join(label for label, _ in selected)}")
    print(f"  profile: {'quick' if args.quick else 'full'}")

    results = {label: run_step(label, cmd) for label, cmd in selected}

    print("\n" + "=" * 78)
    print("Verification summary")
    print("=" * 78)
    for label, _ in selected:
        print(f"  [{'PASS' if results[label] else 'FAIL'}] {label}")
    failed = [label for label, ok in results.items() if not ok]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        print("Fix the failing step and run the command again.")
        return 1
    print("\nAll selected checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
