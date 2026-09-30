"""Device selection and diagnostics (CPU / CUDA / MPS).

Every runner resolves its ``--device`` value through :func:`resolve_device`, so
``auto`` works on any machine and an explicit ``cuda`` request on a CPU-only
machine fails with a clear message instead of a deep torch error.
"""
from __future__ import annotations

import os
import platform

import torch


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device specification to a ``torch.device``.

    Accepted values: ``auto``, ``cpu``, ``cuda``, ``cuda:N`` and ``mps``.
    ``auto`` prefers CUDA, then Apple MPS, then CPU.  Explicit requests are
    validated up front so an unavailable device fails early and clearly.
    """
    normalized = str(spec or "auto").strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(normalized)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"--device {spec!r} was requested, but CUDA is not available "
                f"(torch {torch.__version__}, CUDA build: {torch.version.cuda or 'none'}). "
                "Use --device cpu or --device auto."
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"--device {spec!r} was requested, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
    if device.type == "mps" and not _mps_available():
        raise RuntimeError(
            f"--device {spec!r} was requested, but MPS is not available on this machine."
        )
    return device


def describe_device(device: torch.device) -> str:
    """One-line human-readable description of a resolved device."""
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        return (f"{device} ({props.name}, {props.total_memory / 1024 ** 3:.1f} GiB, "
                f"compute capability {props.major}.{props.minor})")
    if device.type == "mps":
        return "mps (Apple Metal)"
    return f"cpu ({os.cpu_count()} logical cores, torch threads={torch.get_num_threads()})"


def log_device(requested: str, resolved: torch.device) -> None:
    """Print the resolved device in a uniform format."""
    print(f"[device] requested={requested} -> {describe_device(resolved)}", flush=True)


def device_report() -> str:
    """Multi-line environment report used by ``code/check_devices.py``."""
    cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    lines = [
        "== environment ==",
        f"python       : {platform.python_version()} "
        f"({platform.system()} {platform.machine()})",
        f"torch        : {torch.__version__} (CUDA build: {torch.version.cuda or 'none'})",
        f"torch threads: {torch.get_num_threads()}",
        f"cuDNN        : {cudnn_version if cudnn_version is not None else 'unavailable'}",
        f"determinism  : cudnn.deterministic={torch.backends.cudnn.deterministic}, "
        f"cudnn.benchmark={torch.backends.cudnn.benchmark}",
        "",
        "== CUDA ==",
        f"available    : {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"devices      : {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            lines.append(
                f"  cuda:{index}     {props.name}  {props.total_memory / 1024 ** 3:.1f} GiB  "
                f"cc {props.major}.{props.minor}"
            )
    lines += [
        "",
        "== MPS ==",
        f"available    : {_mps_available()}",
        "",
        "== suggestion ==",
        f"--device auto -> {resolve_device('auto')}",
    ]
    return "\n".join(lines)
