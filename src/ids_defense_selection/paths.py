"""Central filesystem layout: the single source of truth for every default path.

Scripts resolve user-supplied paths against :data:`PROJECT_ROOT` instead of the
current working directory, so every documented command behaves identically no
matter where it is started from.
"""
from __future__ import annotations

from pathlib import Path

#: Repository root: the directory that contains pyproject.toml, src/ and scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Default dataset directory and default root of the output tree.
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"

#: Backbone CLI key -> canonical output subdirectory under the output root.
BACKBONE_OUTPUT_SUBDIRS: dict[str, str] = {
    "mlp": "mlp",
    "cnn": "cnn1d",
    "ft": "ft_transformer",
}

#: Backbone CLI key -> experiment runner script inside ``scripts/``.
BACKBONE_RUNNERS: dict[str, str] = {
    "mlp": "run_mlp.py",
    "cnn": "run_cnn1d.py",
    "ft": "run_ft_transformer.py",
}

#: Backbone CLI key -> phi4 evaluation output subdirectory.
PHI4_OUTPUT_SUBDIRS: dict[str, str] = {
    "cnn": "phi4_cnn",
    "ft": "phi4_ft",
}


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    """Resolve *value* against *base* unless it is already absolute."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def default_output_dir(backbone: str, root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Canonical output directory of a backbone experiment."""
    return root / BACKBONE_OUTPUT_SUBDIRS[backbone]


def default_phi4_output_dir(backbone: str, root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Canonical output directory of a phi4 evaluation."""
    return root / PHI4_OUTPUT_SUBDIRS[backbone]
