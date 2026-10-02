"""Dataset specification shared by the validator and the experiment code.

Standard library only, so ``prepare_data.py`` can stay lightweight.
The partition sizes are the *official* UNSW-NB15 direction: the larger file is
the training partition.  ``split_summary`` records that evidence inside every
``run_summary.json`` so a reversed split can never pass silently.
"""
from __future__ import annotations

#: Official UNSW-NB15 partition sizes (records).
UNSW_NB15_TRAIN_ROWS = 175_341
UNSW_NB15_TEST_ROWS = 82_332

#: Columns every partition must provide.
REQUIRED_COLUMNS = ("label", "attack_cat")


def split_summary(train_rows: int, test_rows: int) -> dict:
    """Summarise the split direction actually used by a run."""
    return {
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "official_direction": bool(
            train_rows == UNSW_NB15_TRAIN_ROWS and test_rows == UNSW_NB15_TEST_ROWS
        ),
    }
