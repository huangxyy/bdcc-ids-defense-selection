"""Shared matplotlib style and the canonical defense palette."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#: Journal-grade style used by every figure script.
STYLE = {
    # Font
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "mathtext.fontset": "stix",
    # Figure
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # Axes
    "axes.linewidth": 0.6,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.unicode_minus": False,
    # Ticks
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    # Legend
    "legend.fontsize": 7,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7",
    "legend.borderpad": 0.3,
    "legend.handlelength": 1.5,
    # Lines
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
}


def apply_style() -> None:
    """Apply the shared rcParams."""
    plt.rcParams.update(STYLE)


#: Canonical defense order (matches ids_defense_selection.DEFENSE_ORDER).
DEFENSE_ORDER = (
    "standard", "adv_training", "constrained_adv",
    "trades", "free_at", "class_aware_constrained",
)
DEFENSE_LABELS = {
    "standard": "Standard",
    "adv_training": "PGD-AT",
    "constrained_adv": "Constrained",
    "trades": "TRADES",
    "free_at": "Free AT",
    "class_aware_constrained": "Class-Aware",
}
#: Muted, grayscale-friendly palette (colorbrewer Set2-like).
DEFENSE_COLORS = {
    "standard": "#1f77b4",
    "adv_training": "#ff7f0e",
    "constrained_adv": "#2ca02c",
    "trades": "#d62728",
    "free_at": "#9467bd",
    "class_aware_constrained": "#8c564b",
}
#: Markers for line plots.
DEFENSE_MARKERS = {
    "standard": "o",
    "adv_training": "s",
    "constrained_adv": "D",
    "trades": "^",
    "free_at": "v",
    "class_aware_constrained": "P",
}


def get_label(defense: str) -> str:
    """Display label of a defense key."""
    return DEFENSE_LABELS.get(defense, defense)


def get_color(defense: str) -> str:
    """Plot colour of a defense key."""
    return DEFENSE_COLORS.get(defense, "#666666")


def get_marker(defense: str) -> str:
    """Plot marker of a defense key."""
    return DEFENSE_MARKERS.get(defense, "o")
