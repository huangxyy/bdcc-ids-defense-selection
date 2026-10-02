"""
Pareto-Optimal Defense Selection Analysis
==========================================
Computes 4-dimensional risk profiles (phi1..phi4) for each defense strategy
across the MLP, 1D-CNN and FT-Transformer backbones, finds the Pareto front,
and generates publication-quality figures and summary CSV tables.

Usage:
    uv run python scripts/pareto_selection.py
    uv run python scripts/pareto_selection.py --ref-attack pgd --ref-epsilon 0.10
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from . import style as FS
from .config import DEFAULT_EPSILON_LIST
from .paths import BACKBONE_OUTPUT_SUBDIRS, DEFAULT_OUTPUT_ROOT, resolve_path

warnings.filterwarnings("ignore", category=FutureWarning)

FS.apply_style()

# Canonical output directories written by scripts/run_experiments.py (and the
# CIC-IDS2017 script).  --outputs-root shifts the whole tree.
BACKBONE_DIRS = {
    "MLP": BACKBONE_OUTPUT_SUBDIRS["mlp"],
    "CNN": BACKBONE_OUTPUT_SUBDIRS["cnn"],
    "FT-Trans": BACKBONE_OUTPUT_SUBDIRS["ft"],
    "CICIDS": "cicids2017_strict_matched_budget_run",
}

#: Decision-layer backbones: key -> display label.
BACKBONE_LABELS = {
    "MLP": "MLP",
    "CNN": "1D-CNN",
    "FT-Trans": "FT-Transformer",
}

#: Decision-layer backbone -> CLI key of ``scripts/run_experiments.py``.
BACKBONE_CLI_KEYS = {"MLP": "mlp", "CNN": "cnn", "FT-Trans": "ft"}

#: Decision-layer backbone -> column in ``pareto_selection_results.csv``.
BACKBONE_SELECTION_COLUMNS = {
    "MLP": "mlp_selected",
    "CNN": "cnn_selected",
    "FT-Trans": "ft_selected",
}

DEFENSE_ORDER = FS.DEFENSE_ORDER  # 6 defenses in canonical order


def resolve_outputs_root(value: str | Path) -> Path:
    """Resolve ``--outputs-root`` against the repository root, never the cwd.

    Every other script funnels user-supplied paths through
    :func:`ids_defense_selection.paths.resolve_path`; this analysis reads the
    same output tree, so it must follow the same rule.  Otherwise
    ``--outputs-root outputs`` silently means ``$PWD/outputs`` and running the
    script from a different directory would analyse an empty tree.
    """
    return resolve_path(value)

# Theta presets: (w1_clean, w2_resilience, w3_cost, w4_fairness)
THETA_PRESETS = {
    "Robust":   (0.10, 0.60, 0.10, 0.20),
    "Balanced": (0.25, 0.35, 0.20, 0.20),
    "Clean":    (0.50, 0.15, 0.15, 0.20),
    # Intermediate sweeps
    "w_rob0.7": (0.10, 0.70, 0.10, 0.10),
    "w_bal2":   (0.30, 0.30, 0.20, 0.20),
    "w_cln2":   (0.60, 0.10, 0.15, 0.15),
    "w_fair":   (0.20, 0.30, 0.10, 0.40),
    "w_cost":   (0.20, 0.30, 0.40, 0.10),
}

# ── Data loading helpers ─────────────────────────────────────────────────────

def _load_csv_safe(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_backbone_data(backbone: str, dirs: dict[str, str]) -> dict:
    """Load all relevant CSVs for a given backbone key."""
    d = dirs.get(backbone)
    if d is None or not os.path.isdir(d):
        return {}
    data = {}
    for name in ("mean_results", "std_results", "efficiency_mean", "efficiency_raw",
                 "category_mean_results", "category_raw_results"):
        df = _load_csv_safe(os.path.join(d, f"{name}.csv"))
        if df is not None:
            data[name] = df
    return data


# ── Phi computation ──────────────────────────────────────────────────────────

def compute_phi1(mean_results: pd.DataFrame) -> pd.Series:
    """phi1: clean F1 (attack='clean', epsilon=0.0)."""
    clean = mean_results[(mean_results["attack"] == "clean") &
                         (mean_results["epsilon"] == 0.0)]
    return clean.set_index("model")["f1"]


def compute_phi2(mean_results: pd.DataFrame, ref_attack: str,
                 ref_epsilon: float) -> pd.Series:
    """phi2: 1 - ASR at reference attack / epsilon."""
    subset = mean_results[(mean_results["attack"] == ref_attack) &
                          (np.isclose(mean_results["epsilon"], ref_epsilon))]
    if subset.empty:
        return pd.Series(dtype=float)
    return (1.0 - subset.set_index("model")["attack_success_rate"])


def compute_phi2_scenario(mean_results: pd.DataFrame, attack: str,
                           epsilon: float) -> pd.Series:
    """phi2 for arbitrary (attack, epsilon) scenario — used for heatmap."""
    subset = mean_results[(mean_results["attack"] == attack) &
                          (np.isclose(mean_results["epsilon"], epsilon))]
    if subset.empty:
        return pd.Series(dtype=float)
    return (1.0 - subset.set_index("model")["attack_success_rate"])


def compute_phi3(efficiency: pd.DataFrame) -> pd.Series:
    """phi3: 1 / relative_train_cost_vs_standard."""
    eff = efficiency.set_index("model")["relative_train_cost_vs_standard"]
    return 1.0 / eff.clip(lower=1e-6)


def compute_phi4(category_results: pd.DataFrame | None, ref_attack: str,
                 ref_epsilon: float) -> pd.Series | None:
    """phi4: min adv_recall across attack categories (category fairness)."""
    if category_results is None:
        return None
    subset = category_results[
        (category_results["attack"] == ref_attack) &
        (np.isclose(category_results["epsilon"], ref_epsilon))
    ]
    if subset.empty:
        return None
    return subset.groupby("model")["adv_recall"].min()


def compute_phi12_stds(std_results: pd.DataFrame | None, ref_attack: str,
                       ref_epsilon: float) -> tuple[pd.Series | None, pd.Series | None]:
    """Per-seed dispersion of phi1 and phi2, read from std_results.csv."""
    if std_results is None or std_results.empty:
        return None, None

    def std_at(attack: str, epsilon: float, column: str) -> pd.Series | None:
        subset = std_results[
            (std_results["attack"] == attack) & np.isclose(std_results["epsilon"], epsilon)]
        if subset.empty:
            return None
        return subset.set_index("model")[column]

    return std_at("clean", 0.0, "f1"), std_at(ref_attack, ref_epsilon, "attack_success_rate")


def compute_phi3_with_std(efficiency_raw: pd.DataFrame | None) -> tuple[pd.Series | None, pd.Series | None]:
    """phi3 from per-seed training times, with the dispersion propagated.

    relative cost r = train_seconds / standard.train_seconds (per seed);
    phi3 = 1 / r, and std(phi3) ~ std(r) / mean(r)^2.
    """
    if efficiency_raw is None or efficiency_raw.empty:
        return None, None
    if not {"seed", "model", "train_seconds"}.issubset(efficiency_raw.columns):
        return None, None
    standard = (efficiency_raw[efficiency_raw["model"] == "standard"]
                .set_index("seed")["train_seconds"])
    if standard.empty:
        return None, None
    frame = efficiency_raw.copy()
    frame["relative_cost"] = [
        row["train_seconds"] / standard.get(row["seed"], np.nan)
        for _, row in frame.iterrows()
    ]
    mean = frame.groupby("model")["relative_cost"].mean()
    std = frame.groupby("model")["relative_cost"].std().fillna(0.0)
    mean_clipped = mean.clip(lower=1e-6)
    return 1.0 / mean_clipped, std / (mean_clipped ** 2)


def compute_phi4_with_std(category_raw: pd.DataFrame | None, ref_attack: str,
                          ref_epsilon: float) -> tuple[pd.Series | None, pd.Series | None]:
    """phi4 from per-seed per-category recall (min per seed, then mean/std)."""
    if category_raw is None or category_raw.empty:
        return None, None
    subset = category_raw[
        (category_raw["attack"] == ref_attack)
        & np.isclose(category_raw["epsilon"], ref_epsilon)]
    if subset.empty:
        return None, None
    per_seed = subset.groupby(["model", "seed"])["adv_recall"].min().reset_index()
    mean = per_seed.groupby("model")["adv_recall"].mean()
    std = per_seed.groupby("model")["adv_recall"].std().fillna(0.0)
    return mean, std


def build_objective_matrices(data: dict, ref_attack: str,
                             ref_epsilon: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (means, stds): the 6x4 objective matrix and its per-seed dispersion.

    Std columns are zero-filled when the per-seed files are unavailable, so an
    old output directory still works (with a deterministic decision, i.e. the
    confidence margin has no effect).
    """
    means = build_objective_matrix(data, ref_attack, ref_epsilon)
    stds = pd.DataFrame(0.0, index=means.index, columns=means.columns, dtype=float)
    if means.empty:
        return means, stds

    phi1_std, phi2_std = compute_phi12_stds(data.get("std_results"), ref_attack, ref_epsilon)
    phi3_mean, phi3_std = compute_phi3_with_std(data.get("efficiency_raw"))
    phi4_mean, phi4_std = compute_phi4_with_std(data.get("category_raw_results"),
                                                ref_attack, ref_epsilon)
    # phi3/phi4 are per-seed quantities (cost ratio, worst-category recall):
    # when the raw files exist, take mean over seeds of the per-seed value
    # instead of the aggregate-then-threshold shortcut used by older runs.
    for column, series_mean, series_std in (("phi1", None, phi1_std),
                                            ("phi2", None, phi2_std),
                                            ("phi3", phi3_mean, phi3_std),
                                            ("phi4", phi4_mean, phi4_std)):
        for model in means.index:
            if series_mean is not None and model in series_mean.index \
                    and np.isfinite(series_mean[model]):
                means.loc[model, column] = float(series_mean[model])
            if series_std is not None and model in series_std.index \
                    and np.isfinite(series_std[model]):
                stds.loc[model, column] = float(series_std[model])
    return means, stds


def build_objective_matrix(data: dict, ref_attack: str,
                            ref_epsilon: float) -> pd.DataFrame:
    """
    Assemble a DataFrame of shape (n_defenses, 4) with phi1..phi4.
    Missing phi4 is filled with phi1 (conservative fallback).
    """
    mr  = data.get("mean_results")
    eff = data.get("efficiency_mean")
    cat = data.get("category_mean_results")

    if mr is None or eff is None:
        return pd.DataFrame()

    phi1 = compute_phi1(mr)
    phi2 = compute_phi2(mr, ref_attack, ref_epsilon)
    phi3 = compute_phi3(eff)
    phi4 = compute_phi4(cat, ref_attack, ref_epsilon)

    # Align on DEFENSE_ORDER; keep only models present in all phis
    models = [m for m in DEFENSE_ORDER if m in phi1.index and
              m in phi2.index and m in phi3.index]

    if not models:
        # e.g. the requested (attack, epsilon) is not part of the stored results
        return pd.DataFrame(columns=["phi1", "phi2", "phi3", "phi4"]).rename_axis("model")

    rows = []
    for m in models:
        p4 = phi4[m] if (phi4 is not None and m in phi4.index) else phi1[m]
        rows.append({
            "model":    m,
            "phi1":     phi1[m],
            "phi2":     phi2[m],
            "phi3":     phi3[m],
            "phi4":     p4,
        })

    df = pd.DataFrame(rows).set_index("model")
    return df


# ── Pareto front ─────────────────────────────────────────────────────────────

def is_pareto_optimal(objectives_matrix: np.ndarray) -> np.ndarray:
    """
    Return boolean array: True if row i is Pareto-optimal (all-maximize).
    j dominates i iff j >= i in every objective and j > i in at least one.
    """
    n = len(objectives_matrix)
    optimal = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (np.all(objectives_matrix[j] >= objectives_matrix[i]) and
                    np.any(objectives_matrix[j] > objectives_matrix[i])):
                optimal[i] = False
                break
    return optimal


# ── Parameterised defense selection ─────────────────────────────────────────

def select_defense(objectives: np.ndarray, defense_names: list,
                   theta: tuple) -> str:
    """Min-max normalise then pick defense maximising theta-weighted score."""
    lo = objectives.min(axis=0)
    hi = objectives.max(axis=0)
    norm = (objectives - lo) / (hi - lo + 1e-10)
    scores = norm @ np.array(theta)
    return defense_names[int(np.argmax(scores))]


def is_pareto_optimal_uncertain(means: pd.DataFrame, stds: pd.DataFrame,
                                margin: float = 1.0) -> np.ndarray:
    """Dominance test that accounts for run-to-run variation (paper Eq. 4).

    Candidate j dominates i when its *lower* confidence bound is at least i's
    *upper* confidence bound on every objective (mean - margin*std >=
    mean + margin*std), with at least one strict improvement.  ``margin=0``
    recovers the deterministic criterion exactly.
    """
    columns = ["phi1", "phi2", "phi3", "phi4"]
    m = means[columns].to_numpy(dtype=float)
    s = stds.reindex(means.index)[columns].fillna(0.0).to_numpy(dtype=float)
    n = len(m)
    optimal = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lower_j = m[j] - margin * s[j]
            upper_i = m[i] + margin * s[i]
            if np.all(lower_j >= upper_i) and np.any(lower_j > upper_i):
                optimal[i] = False
                break
    return optimal


def apply_admissibility(means: pd.DataFrame, min_phi2: float = 0.0,
                        min_phi4: float = 0.0) -> pd.Series:
    """Boolean mask of candidates that clear the minimum-acceptable thresholds.

    Applied *before* Pareto filtering and scoring, so a fully compensatory
    weighted sum can never return a candidate that is unacceptable on security
    grounds (e.g. an undefended baseline).
    """
    if means.empty:
        return pd.Series(dtype=bool)
    return (means["phi2"] >= min_phi2) & (means["phi4"] >= min_phi4)


# ── Figure 1: Pareto front comparison ────────────────────────────────────────

def _pareto_front_path(objectives_2d: np.ndarray,
                       pareto_mask: np.ndarray) -> np.ndarray | None:
    """Extract Pareto points sorted by phi1 for dashed-line drawing."""
    pts = objectives_2d[pareto_mask]
    if len(pts) == 0:
        return None
    return pts[np.argsort(pts[:, 0])]


def plot_pareto_front_comparison(objectives: dict[str, pd.DataFrame],
                                 out_path: str) -> None:
    """One Pareto panel per backbone (MLP, 1D-CNN, FT-Transformer)."""
    items = list(objectives.items())
    fig, axes = plt.subplots(1, len(items), figsize=(3.35 * len(items), 3.0),
                             squeeze=False)

    for index, (ax, (key, obj)) in enumerate(zip(axes[0], items)):
        title = f"({chr(ord('a') + index)}) {BACKBONE_LABELS.get(key, key)} Backbone"
        if obj.empty:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)
            continue

        models = obj.index.tolist()
        mat    = obj[["phi1", "phi2", "phi3", "phi4"]].values
        pareto = is_pareto_optimal(mat)

        # Size proportional to phi3 (cost efficiency)
        phi3 = mat[:, 2]
        phi3_norm = (phi3 - phi3.min()) / (phi3.max() - phi3.min() + 1e-10)
        sizes = 30 + phi3_norm * 120  # range [30, 150]

        for idx, m in enumerate(models):
            x, y = mat[idx, 0], mat[idx, 1]
            col = FS.get_color(m)
            mk  = FS.get_marker(m)
            if pareto[idx]:
                ax.scatter(x, y, s=sizes[idx], color=col, marker=mk,
                           zorder=4, linewidths=0.8, edgecolors="black")
            else:
                ax.scatter(x, y, s=sizes[idx] * 0.55, color=col, marker=mk,
                           zorder=3, linewidths=0.5, edgecolors="0.5",
                           alpha=0.6)

        # Dashed Pareto frontier line
        front_pts = _pareto_front_path(mat[:, :2], pareto)
        if front_pts is not None and len(front_pts) > 1:
            ax.plot(front_pts[:, 0], front_pts[:, 1],
                    linestyle="--", color="0.3", linewidth=0.8,
                    zorder=2, alpha=0.7)

        ax.set_xlabel(r"$\phi_1$ Clean F1", fontsize=8)
        ax.set_ylabel(r"$\phi_2$ Adversarial Resilience (1$-$ASR)", fontsize=8)
        ax.set_title(title, fontsize=9, pad=4)

        # Legend: defence names
        handles = [
            mpatches.Patch(color=FS.get_color(m), label=FS.get_label(m))
            for m in models
        ]
        ax.legend(handles=handles, fontsize=6, loc="lower right",
                  handlelength=1.0, handletextpad=0.4, borderpad=0.4)

        # Annotation: pareto symbol
        for idx, m in enumerate(models):
            if pareto[idx]:
                ax.annotate("*", (mat[idx, 0], mat[idx, 1]),
                            textcoords="offset points", xytext=(3, 3),
                            fontsize=7, color="black")

    plt.tight_layout(pad=0.8)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 2: Risk surface heatmap ───────────────────────────────────────────

# Attack scenarios shown in the risk-surface heatmap.
# "adaptive" is deliberately absent: the evaluation code no longer emits that
# column (it was a duplicate of PGD), so referencing it would raise KeyError.
HEATMAP_SCENARIOS = [
    ("fgsm", 0.05),
    ("pgd",  0.10),
    ("cw",   0.10),
    ("apgd", 0.10),
]
W1_BALANCED = 0.25
W2_BALANCED = 0.35


def _risk_matrix(obj_phi1: pd.Series, mr: pd.DataFrame,
                 models: list) -> np.ndarray:
    """
    Build |models| x |scenarios| risk matrix.
    R = 1 - (w1*phi1 + w2*phi2_scenario)  (lower is safer).
    """
    mat = np.full((len(models), len(HEATMAP_SCENARIOS)), np.nan)
    for col_idx, (atk, eps) in enumerate(HEATMAP_SCENARIOS):
        phi2 = compute_phi2_scenario(mr, atk, eps)
        for row_idx, m in enumerate(models):
            if m in obj_phi1.index and m in phi2.index:
                r = 1.0 - (W1_BALANCED * obj_phi1[m] +
                            W2_BALANCED * phi2[m])
                mat[row_idx, col_idx] = r
    return mat


def plot_risk_surface_heatmap(data: dict[str, dict], out_path: str) -> None:
    """One risk-surface panel per backbone."""
    items = list(data.items())
    fig, axes = plt.subplots(1, len(items), figsize=(3.35 * len(items), 3.2),
                             squeeze=False)
    scenario_labels = [f"{a}\n$\\varepsilon$={e:.2f}"
                       for a, e in HEATMAP_SCENARIOS]

    for index, (ax, (key, entry)) in enumerate(zip(axes[0], items)):
        title = f"({chr(ord('a') + index)}) {BACKBONE_LABELS.get(key, key)}"
        mr = entry.get("mean_results")
        if mr is None:
            ax.set_visible(False)
            continue

        phi1 = compute_phi1(mr)
        models = [m for m in DEFENSE_ORDER if m in phi1.index]
        mat = _risk_matrix(phi1, mr, models)

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                       vmin=0.0, vmax=1.0)

        ax.set_xticks(range(len(HEATMAP_SCENARIOS)))
        ax.set_xticklabels(scenario_labels, fontsize=6)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([FS.get_label(m) for m in models], fontsize=7)
        ax.set_title(title, fontsize=9, pad=4)

        # Cell text
        for r in range(len(models)):
            for c in range(len(HEATMAP_SCENARIOS)):
                v = mat[r, c]
                if not np.isnan(v):
                    txt_col = "white" if v > 0.65 else "black"
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color=txt_col)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Composite Risk R")

    plt.tight_layout(pad=0.8)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 3: Epsilon Pareto evolution ───────────────────────────────────────

EPSILON_LEVELS = list(DEFAULT_EPSILON_LIST)
EPS_STYLES = {0.02: ("o", "solid",  0.9),
              0.05: ("s", "dashed", 0.75),
              0.10: ("D", "dotted", 0.6)}
DEFAULT_EPS_STYLE = ("o", "solid", 0.8)


def plot_epsilon_pareto_evolution(data: dict[str, dict], out_path: str) -> None:
    """One epsilon-evolution panel per backbone."""
    items = list(data.items())
    fig, axes = plt.subplots(1, len(items), figsize=(3.35 * len(items), 2.8),
                             squeeze=False)

    for index, (ax, (key, entry)) in enumerate(zip(axes[0], items)):
        label = BACKBONE_LABELS.get(key, key)
        mr = entry.get("mean_results")
        eff = entry.get("efficiency_mean")
        if mr is None or eff is None:
            ax.set_visible(False)
            continue

        phi1 = compute_phi1(mr)
        phi3 = compute_phi3(eff)
        models = [m for m in DEFENSE_ORDER if m in phi1.index and m in phi3.index]

        for eps in EPSILON_LEVELS:
            mk, ls, alp = EPS_STYLES.get(eps, DEFAULT_EPS_STYLE)
            phi2 = compute_phi2(mr, "pgd", eps)
            if phi2.empty:
                continue

            pts = np.array([[phi1[m], phi2[m]] for m in models
                            if m in phi2.index])
            mods = [m for m in models if m in phi2.index]

            if len(pts) == 0:
                continue

            phi34 = np.array([[phi3[m], phi1[m]] for m in mods])
            full_mat = np.hstack([pts, phi34])
            pareto = is_pareto_optimal(full_mat)

            phi3v = np.array([phi3[m] for m in mods])
            phi3_norm = (phi3v - phi3v.min()) / (phi3v.max() - phi3v.min() + 1e-10)
            sizes = 25 + phi3_norm * 80

            for idx, m in enumerate(mods):
                col = FS.get_color(m)
                edge = "black" if pareto[idx] else "0.7"
                lw = 0.8 if pareto[idx] else 0.4
                ax.scatter(pts[idx, 0], pts[idx, 1],
                           s=sizes[idx], marker=mk, color=col,
                           edgecolors=edge, linewidths=lw,
                           alpha=alp, zorder=3 + int(pareto[idx]))

            front = _pareto_front_path(pts, pareto)
            if front is not None and len(front) > 1:
                ax.plot(front[:, 0], front[:, 1], linestyle=ls,
                        color="0.4", linewidth=0.7, alpha=0.8)

        ax.set_xlabel(r"$\phi_1$ Clean F1", fontsize=8)
        ax.set_ylabel(r"$\phi_2$ Adversarial Resilience", fontsize=8)
        ax.set_title(f"({chr(ord('a') + index)}) {label}, PGD", fontsize=9, pad=4)

        if models:
            def_handles = [mpatches.Patch(color=FS.get_color(m), label=FS.get_label(m))
                           for m in models]
            leg1 = ax.legend(handles=def_handles, fontsize=6,
                             loc="lower left", handlelength=1.0)
            ax.add_artist(leg1)
            from matplotlib.lines import Line2D
            eps_handles = [
                Line2D([0], [0], marker=EPS_STYLES.get(eps, DEFAULT_EPS_STYLE)[0],
                       color="0.4", linestyle=EPS_STYLES.get(eps, DEFAULT_EPS_STYLE)[1],
                       linewidth=0.8, markersize=5, label=f"$\\varepsilon$={eps:.2f}")
                for eps in EPSILON_LEVELS
            ]
            ax.legend(handles=eps_handles, fontsize=6,
                      loc="lower right", handlelength=1.5)

    plt.tight_layout(pad=0.6)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 4: Theta sensitivity ──────────────────────────────────────────────

def plot_theta_sensitivity(objectives: dict[str, pd.DataFrame], out_path: str) -> None:
    """One preference-sensitivity panel per backbone."""
    theta_names  = list(THETA_PRESETS.keys())
    theta_vals   = list(THETA_PRESETS.values())

    items = list(objectives.items())
    fig, axes = plt.subplots(1, len(items), figsize=(3.35 * len(items), 2.8),
                             squeeze=False)

    for index, (ax, (key, obj)) in enumerate(zip(axes[0], items)):
        title = f"({chr(ord('a') + index)}) {BACKBONE_LABELS.get(key, key)} Backbone"
        if obj.empty:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)
            continue

        models = obj.index.tolist()
        mat    = obj[["phi1", "phi2", "phi3", "phi4"]].values

        selected = [select_defense(mat, models, th) for th in theta_vals]
        colors   = [FS.get_color(s) for s in selected]
        labels_sel = [FS.get_label(s) for s in selected]

        x = np.arange(len(theta_names))
        bars = ax.bar(x, np.ones(len(theta_names)), color=colors,
                      edgecolor="white", linewidth=0.4, width=0.7)

        # Defense label inside bar
        for i, (bar, lbl) in enumerate(zip(bars, labels_sel)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    0.5, lbl, ha="center", va="center",
                    fontsize=6, color="white", fontweight="bold",
                    rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(theta_names, fontsize=6, rotation=40, ha="right")
        ax.set_yticks([])
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel("Preference Weight Scenario ($\\theta$)", fontsize=7)
        ax.set_ylim(0, 1.15)

        # Legend
        seen = {}
        for m, lbl in zip(selected, labels_sel):
            if m not in seen:
                seen[m] = lbl
        handles = [mpatches.Patch(color=FS.get_color(m), label=lbl)
                   for m, lbl in seen.items()]
        ax.legend(handles=handles, fontsize=6, loc="upper right",
                  ncol=2, handlelength=0.8)

    plt.tight_layout(pad=0.8)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── CSV output helpers ────────────────────────────────────────────────────────

def write_risk_profile_csv(obj: pd.DataFrame, pareto_mask: np.ndarray,
                           out_path: str, *,
                           stds: pd.DataFrame | None = None,
                           pareto_deterministic: np.ndarray | None = None,
                           admissible: pd.Series | None = None) -> None:
    """Write the 4-D risk profile with dispersion and both Pareto verdicts.

    ``is_pareto_optimal`` uses the uncertainty-aware criterion (confidence
    margin); ``is_pareto_optimal_deterministic`` keeps the point-estimate
    verdict so a reviewer can see which decisions the margin changes.
    """
    df = obj.copy().reset_index()
    df.columns = ["model", "phi1_clean_f1", "phi2_resilience",
                  "phi3_cost_eff", "phi4_fairness"]
    if stds is not None:
        std_view = stds.reindex(obj.index)[["phi1", "phi2", "phi3", "phi4"]].reset_index(drop=True)
        for source, target in zip(("phi1", "phi2", "phi3", "phi4"),
                                  ("phi1_std", "phi2_std", "phi3_std", "phi4_std")):
            df[target] = std_view[source].to_numpy()
    df["is_pareto_optimal"] = pareto_mask
    if pareto_deterministic is not None:
        df["is_pareto_optimal_deterministic"] = pareto_deterministic
    if admissible is not None:
        df["admissible"] = admissible.reindex(obj.index).to_numpy()
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"  Saved: {out_path}")


def write_selection_csv(objectives: dict[str, pd.DataFrame], out_path: str) -> None:
    """Write the preference-preset selection for every available backbone."""
    rows = []
    for name, theta in THETA_PRESETS.items():
        row: dict[str, str] = {"theta_name": name, "theta_values": str(theta)}
        for key in BACKBONE_LABELS:
            objective = objectives.get(key, pd.DataFrame())
            selected = "N/A"
            if not objective.empty:
                models = objective.index.tolist()
                matrix = objective[["phi1", "phi2", "phi3", "phi4"]].values
                selected = FS.get_label(select_defense(matrix, models, theta))
            row[BACKBONE_SELECTION_COLUMNS[key]] = selected
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


# ── Stdout summary ────────────────────────────────────────────────────────────

def print_summary(backbone: str, obj: pd.DataFrame, *,
                  stds: pd.DataFrame | None = None,
                  pareto: np.ndarray | None = None,
                  admissible: pd.Series | None = None,
                  margin: float = 1.0) -> None:
    if obj.empty:
        print(f"\n[{backbone}] No data available.")
        return

    mat    = obj[["phi1", "phi2", "phi3", "phi4"]].values
    models = obj.index.tolist()
    if pareto is None:
        pareto = is_pareto_optimal(mat)
    elif isinstance(pareto, pd.Series):
        pareto = pareto.reindex(models).fillna(False).to_numpy(dtype=bool)
    else:
        pareto = np.asarray(pareto, dtype=bool)

    print(f"\n{'='*60}")
    print(f"  {backbone} Backbone — 4D Risk Profile")
    if margin:
        print(f"  (Pareto uses mean ± {margin:g}·std; admissible thresholds applied)")
    print(f"{'='*60}")
    header = f"{'Defense':<25}  phi1   phi2   phi3   phi4  Pareto  Admissible"
    print(header)
    print("-" * len(header))
    for i, m in enumerate(models):
        p1, p2, p3, p4 = mat[i]
        star = "*" if pareto[i] else " "
        ok = "yes" if (admissible is None or bool(admissible.get(m, True))) else "NO"
        print(f"  {FS.get_label(m):<23}  {p1:.3f}  {p2:.3f}  {p3:.3f}"
              f"  {p4:.3f}  {star}       {ok}")

    if stds is not None and not stds.empty:
        print("\n  per-seed dispersion (std):")
        for m in models:
            if m not in stds.index:
                continue
            row = stds.loc[m]
            print(f"    {FS.get_label(m):<23}  ±{row['phi1']:.4f}  ±{row['phi2']:.4f}"
                  f"  ±{row['phi3']:.4f}  ±{row['phi4']:.4f}")

    print(f"\n  Pareto-optimal defenses: "
          f"{', '.join(FS.get_label(m) for m, ok in zip(models, pareto) if ok)}")

    print("\n  Defense selection by theta preset:")
    for name, theta in list(THETA_PRESETS.items())[:3]:   # show top 3
        sel = select_defense(mat, models, theta)
        print(f"    theta_{name:<12} -> {FS.get_label(sel)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pareto-optimal defense selection analysis")
    p.add_argument("--ref-attack",   default="pgd",
                   help="Reference attack for phi2/phi4 (default: pgd)")
    p.add_argument("--ref-epsilon",  type=float, default=0.10,
                   help="Reference epsilon (default: 0.10)")
    p.add_argument("--outputs-root", default=str(DEFAULT_OUTPUT_ROOT),
                   help="root directory containing mlp/, cnn1d/, ft_transformer/ "
                        "(default: <repo>/outputs)")
    p.add_argument("--confidence-margin", type=float, default=1.0,
                   help="std multiples for the uncertainty-aware dominance test "
                        "(0 = deterministic point estimates)")
    p.add_argument("--min-phi2", type=float, default=0.0,
                   help="admissibility threshold: drop candidates with resilience below this")
    p.add_argument("--min-phi4", type=float, default=0.0,
                   help="admissibility threshold: drop candidates with worst-class recall below this")
    p.add_argument("--no-figs",      action="store_true",
                   help="Skip figure generation")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ref_atk = args.ref_attack
    ref_eps = args.ref_epsilon
    outputs_root = resolve_outputs_root(args.outputs_root)
    dirs = {key: os.path.join(outputs_root, sub) for key, sub in BACKBONE_DIRS.items()}
    fig_dir = os.path.join(outputs_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── Load data ──
    print(f"\nLoading data  (ref: {ref_atk}, eps={ref_eps:.2f})")
    data = {key: load_backbone_data(key, dirs) for key in BACKBONE_LABELS}
    missing = [key for key in BACKBONE_LABELS if not data[key]]
    for key in missing:
        print(f"  [missing] {dirs[key]}/mean_results.csv -- run "
              f"'uv run python scripts/run_experiments.py "
              f"--backbones {BACKBONE_CLI_KEYS[key]}' first")
    if not any(data.values()):
        print("\nNo backbone results found; nothing to analyse.")
        return 1

    # ── Build objective matrices (means + per-seed dispersion) ──
    print(f"\nDecision settings: margin={args.confidence_margin:g}·std, "
          f"min_phi2={args.min_phi2:g}, min_phi4={args.min_phi4:g}")
    prepared: dict[str, dict] = {}
    for key, label in BACKBONE_LABELS.items():
        means, stds = build_objective_matrices(data[key], ref_atk, ref_eps)
        entry = {"means": means, "stds": stds, "admissible": pd.Series(dtype=bool),
                 "dropped": [], "pareto_uncertain": [], "pareto_deterministic": []}
        if means.empty:
            if data[key]:
                print(f"  [warning] {label} results exist but contain no "
                      f"attack={ref_atk!r} at epsilon={ref_eps:.2f}; "
                      "check --ref-attack/--ref-epsilon against mean_results.csv")
            prepared[key] = entry
            continue

        if means["phi4"].equals(means["phi1"]):
            print(f"  [warning] {label}: phi4 (worst-class recall) is missing for this "
                  "reference scenario and fell back to phi1; make sure the category "
                  "evaluation ran at the same (attack, epsilon) "
                  "(e.g. --category-epsilon matches --ref-epsilon).")

        admissible = apply_admissibility(means, args.min_phi2, args.min_phi4)
        entry["admissible"] = admissible
        entry["dropped"] = [m for m in means.index if not bool(admissible.get(m, True))]
        if entry["dropped"]:
            print(f"  [{label}] inadmissible (phi2 < {args.min_phi2:g} or "
                  f"phi4 < {args.min_phi4:g}): {', '.join(entry['dropped'])}")

        kept = means[admissible]
        if kept.empty:
            print(f"  [{label}] no candidate passes the admissibility thresholds.")
            prepared[key] = entry
            continue
        kept_stds = stds.reindex(kept.index).fillna(0.0)
        pareto_uncertain = is_pareto_optimal_uncertain(kept, kept_stds, args.confidence_margin)
        pareto_det = is_pareto_optimal(kept[["phi1", "phi2", "phi3", "phi4"]].values)
        entry["kept"] = kept
        entry["kept_stds"] = kept_stds
        entry["pareto_uncertain"] = [m for m, ok in zip(kept.index, pareto_uncertain) if ok]
        entry["pareto_deterministic"] = [m for m, ok in zip(kept.index, pareto_det) if ok]
        # full-length masks for the CSV / terminal table
        full_u = pd.Series(False, index=means.index)
        full_u.loc[kept.index] = pareto_uncertain
        full_d = pd.Series(False, index=means.index)
        full_d.loc[kept.index] = pareto_det
        entry["mask_uncertain"] = full_u
        entry["mask_deterministic"] = full_d
        prepared[key] = entry

    # ── Stdout summaries ──
    for key, label in BACKBONE_LABELS.items():
        entry = prepared[key]
        if entry["means"].empty:
            print_summary(label, entry["means"])
            continue
        print_summary(label, entry["means"], stds=entry["stds"],
                      pareto=entry.get("mask_uncertain"),
                      admissible=entry["admissible"],
                      margin=args.confidence_margin)
        print(f"    pareto (margin={args.confidence_margin:g}): "
              f"{', '.join(FS.get_label(m) for m in entry['pareto_uncertain'])}")
        if entry["pareto_deterministic"] != entry["pareto_uncertain"]:
            print(f"    point-estimate front differs: "
                  f"{', '.join(FS.get_label(m) for m in entry['pareto_deterministic'])}")

    # ── CSV outputs ──
    print("\nWriting CSV tables...")
    for key in BACKBONE_LABELS:
        base_dir = dirs[key]
        entry = prepared[key]
        if entry["means"].empty:
            continue
        write_risk_profile_csv(
            entry["means"], entry["mask_uncertain"],
            os.path.join(base_dir, "risk_profile_4d.csv"),
            stds=entry["stds"],
            pareto_deterministic=entry.get("mask_deterministic"),
            admissible=entry["admissible"])

    objectives = {
        key: prepared[key].get("kept", prepared[key]["means"].iloc[0:0])
        for key in BACKBONE_LABELS
    }
    target_dir = next(
        (dirs[key] for key in BACKBONE_LABELS if not prepared[key]["means"].empty),
        dirs["MLP"],
    )
    os.makedirs(target_dir, exist_ok=True)
    write_selection_csv(objectives, os.path.join(target_dir, "pareto_selection_results.csv"))

    settings = {
        "reference_attack": ref_atk,
        "reference_epsilon": ref_eps,
        "confidence_margin": args.confidence_margin,
        "min_phi2": args.min_phi2,
        "min_phi4": args.min_phi4,
        "backbones": {
            key: {
                "n_candidates": int(len(prepared[key]["means"])),
                "inadmissible": prepared[key]["dropped"],
                "pareto_uncertain": prepared[key]["pareto_uncertain"],
                "pareto_deterministic": prepared[key]["pareto_deterministic"],
                "dispersion_available": bool(
                    not prepared[key]["stds"].empty and prepared[key]["stds"].to_numpy().sum() > 0),
            }
            for key in BACKBONE_LABELS
        },
    }
    settings_path = os.path.join(outputs_root, "decision_settings.json")
    with open(settings_path, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2, ensure_ascii=False)
    print(f"  Saved: {settings_path}")

    if args.no_figs:
        print("\nFigure generation skipped (--no-figs).")
        return 0

    # ── Figures ──
    print("\nGenerating figures...")
    backbone_data = {key: data[key] for key in BACKBONE_LABELS}

    plot_pareto_front_comparison(
        objectives, os.path.join(fig_dir, "pareto_front_comparison.png"))
    plot_risk_surface_heatmap(
        backbone_data, os.path.join(fig_dir, "risk_surface_heatmap.png"))
    plot_epsilon_pareto_evolution(
        backbone_data, os.path.join(fig_dir, "epsilon_pareto_evolution.png"))
    plot_theta_sensitivity(
        objectives, os.path.join(fig_dir, "theta_sensitivity.png"))

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
