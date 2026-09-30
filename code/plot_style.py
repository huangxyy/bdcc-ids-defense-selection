"""
Shared matplotlib style used by the figure scripts of this repository.
Import this module in all figure scripts.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Journal-grade style ──
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
    "axes.grid": False,        # no default grid — add manually where needed
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

def apply_style():
    plt.rcParams.update(STYLE)

# ── Grayscale-friendly palette (6 models) ──
# Distinct in both color and grayscale; print-safe
MODEL_ORDER = [
    "standard", "adv_training", "constrained_adv",
    "trades", "free_at", "class_aware_constrained",
]
MODEL_LABELS = {
    "standard": "Standard",
    "adv_training": "PGD-AT",
    "constrained_adv": "Constrained",
    "trades": "TRADES",
    "free_at": "Free AT",
    "class_aware_constrained": "Class-Aware",
}
# Muted, distinguishable colors (colorbrewer Set2-like)
MODEL_COLORS = {
    "standard":                  "#1f77b4",  # steel blue
    "adv_training":              "#ff7f0e",  # orange
    "constrained_adv":           "#2ca02c",  # green
    "trades":                    "#d62728",  # red
    "free_at":                   "#9467bd",  # purple
    "class_aware_constrained":   "#8c564b",  # brown
}
# Hatching patterns for grayscale differentiation
MODEL_HATCHES = {
    "standard":                  "",
    "adv_training":              "//",
    "constrained_adv":           "\\\\",
    "trades":                    "xx",
    "free_at":                   "..",
    "class_aware_constrained":   "oo",
}
# Markers for line plots
MODEL_MARKERS = {
    "standard":                  "o",
    "adv_training":              "s",
    "constrained_adv":           "D",
    "trades":                    "^",
    "free_at":                   "v",
    "class_aware_constrained":   "P",
}

def get_label(m):
    return MODEL_LABELS.get(m, m)

def get_color(m):
    return MODEL_COLORS.get(m, "#666666")


def get_marker(m):
    return MODEL_MARKERS.get(m, "o")




