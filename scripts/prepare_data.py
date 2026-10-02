#!/usr/bin/env python
"""Validate (and if necessary repair) the dataset layout before running anything.

This repository follows the **official UNSW-NB15 split**:

    data/train.csv   175,341 records   <- official training partition
    data/test.csv     82,332 records   <- official testing partition

A very common mistake is to download the two official files
(UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv) and save them under the
names train.csv / test.csv without checking -- which silently swaps the split and
makes the experiment train on the small partition and test on the large one.
This script detects that case and can fix it.

Usage
-----
    python scripts/prepare_data.py                 # check only
    python scripts/prepare_data.py --fix-swap      # swap the two files if reversed
    python scripts/prepare_data.py --data-dir data
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path when run by path)

import argparse
import os
import sys
from pathlib import Path

from ids_defense_selection.paths import DEFAULT_DATA_DIR, resolve_path
from ids_defense_selection.spec import (
    REQUIRED_COLUMNS,
    UNSW_NB15_TEST_ROWS,
    UNSW_NB15_TRAIN_ROWS,
)

# Backwards-compatible aliases (smoke_test.py imports these names).
TRAIN_ROWS = UNSW_NB15_TRAIN_ROWS
TEST_ROWS = UNSW_NB15_TEST_ROWS


def count_records(path: Path) -> int:
    """Number of data records (excluding the header). Streaming, so 200 MB is fine."""
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for _ in fh:
            n += 1
    return max(0, n - 1)


def read_csv_header(path: Path) -> list[str]:
    # utf-8-sig strips the byte-order mark that the distributed files carry,
    # so the first column is reported as "id" instead of "\ufeffid".
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        return [c.strip() for c in fh.readline().strip().split(",")]


def check_partition(path: Path, expected: int, role: str) -> tuple[bool, int]:
    if not path.exists():
        print(f"  [MISSING] {path}  (expected {expected:,} records -- the official {role} partition)")
        return False, -1
    n = count_records(path)
    hdr = read_csv_header(path)
    ok = n == expected
    mark = "OK " if ok else "!! "
    print(f"  [{mark}] {path.name:<12} {n:>9,} records   (expected {expected:,} -- official {role})")
    missing = [c for c in REQUIRED_COLUMNS if c not in hdr]
    if missing:
        print(f"          missing required column(s): {missing}")
        ok = False
    return ok, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                    help="directory holding train.csv and test.csv "
                         "(relative paths resolve against the repository root)")
    ap.add_argument("--fix-swap", action="store_true",
                    help="if the two files are reversed, swap them back")
    args = ap.parse_args()

    d = resolve_path(args.data_dir)
    train, test = d / "train.csv", d / "test.csv"

    print("=" * 78)
    print("Dataset check -- official UNSW-NB15 split")
    print("=" * 78)
    ok_train, n_train = check_partition(train, TRAIN_ROWS, "training")
    ok_test, n_test = check_partition(test, TEST_ROWS, "testing")
    print()

    if n_train == TEST_ROWS and n_test == TRAIN_ROWS:
        print("  The two files are REVERSED with respect to the official split.")
        print(f"    {train.name} holds the official testing partition ({TEST_ROWS:,})")
        print(f"    {test.name}  holds the official training partition ({TRAIN_ROWS:,})")
        if args.fix_swap:
            tmp = d / "_swap_tmp.csv"
            os.replace(train, tmp)
            os.replace(test, train)
            os.replace(tmp, test)
            print("\n  Swapped. Re-run this script to confirm.")
            return verify_layout(d)
        print("\n  Re-run with --fix-swap to exchange them, or rename the files by hand.")
        return 1

    if ok_train and ok_test:
        print("  Dataset layout is correct -- you can run the experiments.")
        print()
        print("  Next steps:")
        print("    uv run python scripts/smoke_test.py                   # ~1 minute sanity check")
        print("    uv run python scripts/run_experiments.py                   # full reproduction (--device auto)")
        return 0

    print("  Dataset is not ready. Download UNSW-NB15 from")
    print("    https://research.unsw.edu.au/projects/unsw-nb15-dataset")
    print("  and place the official partitions as:")
    print(f"    {d}/train.csv   ({TRAIN_ROWS:,} records)")
    print(f"    {d}/test.csv    ({TEST_ROWS:,} records)")
    print("  See data/README.md for details.")
    return 1


def verify_layout(d: Path) -> int:
    ok_t, n_t = check_partition(d / "train.csv", TRAIN_ROWS, "training")
    ok_e, n_e = check_partition(d / "test.csv", TEST_ROWS, "testing")
    return 0 if (ok_t and ok_e) else 1


if __name__ == "__main__":
    sys.exit(main())
