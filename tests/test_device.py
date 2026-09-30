"""Tests for device resolution and diagnostics."""
from __future__ import annotations

import torch
import pytest

from ids_defense_selection.device import describe_device, device_report, resolve_device


def test_resolve_cpu_is_always_available() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert describe_device(torch.device("cpu")).startswith("cpu")


def test_auto_picks_an_available_device() -> None:
    device = resolve_device("auto")
    assert device.type in {"cpu", "cuda", "mps"}
    # on a CPU-only machine auto must fall back to CPU
    if not torch.cuda.is_available():
        assert device.type == "cpu"


def test_unknown_device_spec_is_rejected() -> None:
    with pytest.raises(Exception):
        resolve_device("quantum")


def test_explicit_cuda_requires_a_gpu() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine")
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda:0")


def test_device_report_mentions_the_main_sections() -> None:
    report = device_report()
    for section in ("environment", "CUDA", "MPS", "suggestion"):
        assert section in report
