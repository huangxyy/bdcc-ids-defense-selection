"""Reproducibility helpers, the UNSW-NB15 feature pipeline and data loading.

The one-hot width depends on how many categorical levels the training partition
contains, so it is a property of the split rather than a fixed constant.  With
the official split this module produces 194 transformed features
(39 continuous + 155 one-hot); the manuscript reports 190, which corresponds to
the reversed split used in the submitted experiments.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    """Seed every stochastic component AND make GPU kernels deterministic.

    cuDNN selects reduction orders non-deterministically by default, so two runs
    with the same seed can differ in the third decimal place. That matters here:
    several Pareto decisions hinge on margins of that size. With the settings
    below, repeated runs are bit-identical.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_unsw_nb15(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two official UNSW-NB15 partitions.

    ``utf-8-sig`` transparently strips a leading byte-order mark, which the
    distributed CSV files carry.  Without it the first column would be named
    ``"\\ufeffid"``, survive the ``id`` drop and leak the row number into the
    feature matrix as an extra continuous feature.
    """
    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    test_df = pd.read_csv(test_path, encoding="utf-8-sig")
    return train_df, test_df


def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Standardise the continuous features and one-hot encode the categorical ones.

    The scaler and the one-hot vocabulary are fitted on the *training* partition
    only.  The returned metadata carries everything the attack constraints need:
    the numeric mask and the [min, max] box of the standardised continuous
    features observed on the training set.
    """
    target_col = "label"
    drop_cols = [col for col in ["id", "attack_cat"] if col in train_df.columns]
    feature_cols = [col for col in train_df.columns if col not in drop_cols + [target_col]]

    categorical_cols = [col for col in feature_cols if train_df[col].dtype.kind == "O"]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    encoder = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )

    x_train = encoder.fit_transform(train_df[feature_cols])
    x_test = encoder.transform(test_df[feature_cols])
    y_train = train_df[target_col].astype(np.float32).to_numpy()
    y_test = test_df[target_col].astype(np.float32).to_numpy()

    numeric_dim = len(numeric_cols)
    total_dim = x_train.shape[1]
    numeric_mask = np.zeros(total_dim, dtype=np.float32)
    numeric_mask[:numeric_dim] = 1.0

    numeric_mins = x_train[:, :numeric_dim].min(axis=0)
    numeric_maxs = x_train[:, :numeric_dim].max(axis=0)
    feature_names = list(numeric_cols)
    if categorical_cols:
        one_hot = encoder.named_transformers_["cat"]
        feature_names.extend(one_hot.get_feature_names_out(categorical_cols).tolist())

    metadata = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_names": feature_names,
        "numeric_mask": numeric_mask,
        "numeric_mins": numeric_mins,
        "numeric_maxs": numeric_maxs,
    }
    return x_train.astype(np.float32), y_train, x_test.astype(np.float32), y_test, metadata


def stratified_subset_indices(y: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Indices of a stratified subset of `size` rows, sorted ascending."""
    indices = np.arange(len(y))
    if len(y) <= size:
        return indices
    _, subset_indices = train_test_split(
        indices,
        test_size=size,
        random_state=seed,
        stratify=y,
    )
    return np.sort(subset_indices)


def numpy_to_torch(array: np.ndarray) -> torch.Tensor:
    """Zero-copy conversion that always yields a writable tensor."""
    contiguous = np.ascontiguousarray(array)
    if not contiguous.flags.writeable:
        contiguous = contiguous.copy()
    return torch.from_numpy(contiguous)


def make_dataloader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Wrap feature/label arrays in a TensorDataset DataLoader."""
    dataset = TensorDataset(numpy_to_torch(x), numpy_to_torch(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
