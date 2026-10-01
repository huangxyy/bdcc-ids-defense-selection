"""Tests for dataset loading (byte-order mark handling and feature selection)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ids_defense_selection.data import build_features, load_unsw_nb15


def _frame(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "id": np.arange(1, n + 1),
        "dur": np.linspace(0.0, 1.0, n),
        "sbytes": np.arange(n) * 10,
        "proto": ["tcp", "udp"] * (n // 2),
        "attack_cat": ["Normal", "Exploits"] * (n // 2),
        "label": [0, 1] * (n // 2),
    })


def _write(path: Path, frame: pd.DataFrame, *, bom: bool) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig" if bom else "utf-8")


def test_load_unsw_nb15_strips_byte_order_mark(tmp_path: Path) -> None:
    train_path, test_path = tmp_path / "train.csv", tmp_path / "test.csv"
    _write(train_path, _frame(), bom=True)
    _write(test_path, _frame(), bom=True)
    assert train_path.read_bytes().startswith(b"\xef\xbb\xbf")

    train_df, test_df = load_unsw_nb15(str(train_path), str(test_path))
    assert list(train_df.columns)[0] == "id"
    assert "\ufeffid" not in train_df.columns
    assert list(test_df.columns)[0] == "id"


def test_id_column_is_never_a_feature(tmp_path: Path) -> None:
    train_path, test_path = tmp_path / "train.csv", tmp_path / "test.csv"
    _write(train_path, _frame(10), bom=True)
    _write(test_path, _frame(6), bom=True)

    train_df, test_df = load_unsw_nb15(str(train_path), str(test_path))
    x_train, _, x_test, _, metadata = build_features(train_df, test_df)

    assert metadata["numeric_cols"] == ["dur", "sbytes"]
    assert "id" not in metadata["numeric_cols"]
    assert "attack_cat" not in metadata["feature_names"]
    assert int(metadata["numeric_mask"].sum()) == 2
    # two continuous features + two one-hot levels for `proto`
    assert x_train.shape == (10, 4)
    assert x_test.shape == (6, 4)


def test_plain_utf8_files_still_load(tmp_path: Path) -> None:
    train_path, test_path = tmp_path / "train.csv", tmp_path / "test.csv"
    _write(train_path, _frame(), bom=False)
    _write(test_path, _frame(), bom=False)

    train_df, test_df = load_unsw_nb15(str(train_path), str(test_path))
    assert list(train_df.columns) == ["id", "dur", "sbytes", "proto", "attack_cat", "label"]
    assert list(test_df.columns) == list(train_df.columns)
