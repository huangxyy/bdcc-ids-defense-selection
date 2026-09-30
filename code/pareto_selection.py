"""
Pareto-Optimal Defense Selection Analysis
==========================================
Computes 4-dimensional risk profiles (phi1..phi4) for each defense strategy
across MLP and 1D-CNN backbones, finds the Pareto front, and generates
publication-quality figures and summary CSV tables.

Usage:
    python code/pareto_selection.py
    python code/pareto_selection.py --ref-attack pgd --ref-epsilon 0.10
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from ids_defense_selection import style as FS

warnings.filterwarnings("ignore", category=FutureWarning)

FS.apply_style()

# Canonical output directories written by run_experiments.py (and the
# CIC-IDS2017 script).  --outputs-root shifts the whole tree.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKBONE_DIRS = {
    "MLP": "mlp",
    "CNN": "cnn1d",
    "FT-Trans": "ft_transformer",
    "CICIDS": "cicids2017_strict_matched_budget_run",
}

DEFENSE_ORDER = FS.DEFENSE_ORDER  # 6 defenses in canonical order

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
    for name in ("mean_results", "efficiency_mean", "category_mean_results"):
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


# ── Figure 1: Pareto front comparison ────────────────────────────────────────

def _pareto_front_path(objectives_2d: np.ndarray,
                       pareto_mask: np.ndarray) -> np.ndarray | None:
    """Extract Pareto points sorted by phi1 for dashed-line drawing."""
    pts = objectives_2d[pareto_mask]
    if len(pts) == 0:
        return None
    return pts[np.argsort(pts[:, 0])]


def plot_pareto_front_comparison(obj_mlp: pd.DataFrame,
                                 obj_cnn: pd.DataFrame,
                                 out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.69, 3.0))

    for ax, obj, title in zip(axes,
                               [obj_mlp, obj_cnn],
                               ["(a) MLP Backbone", "(b) 1D-CNN Backbone"]):
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


def plot_risk_surface_heatmap(data_mlp: dict, data_cnn: dict,
                               out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.69, 3.2))
    scenario_labels = [f"{a}\n$\\varepsilon$={e:.2f}"
                       for a, e in HEATMAP_SCENARIOS]

    for ax, data, title in zip(axes,
                                [data_mlp, data_cnn],
                                ["(a) MLP", "(b) 1D-CNN"]):
        mr = data.get("mean_results")
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


# ── Figure 3: Epsilon Pareto evolution (MLP only) ────────────────────────────

EPSILON_LEVELS = [0.02, 0.05, 0.10]
EPS_STYLES = {0.02: ("o", "solid",  0.9),
              0.05: ("s", "dashed", 0.75),
              0.10: ("D", "dotted", 0.6)}


def plot_epsilon_pareto_evolution(data_mlp: dict, out_path: str) -> None:
    mr  = data_mlp.get("mean_results")
    eff = data_mlp.get("efficiency_mean")
    if mr is None or eff is None:
        print("  Skipping epsilon_pareto_evolution: MLP data unavailable.")
        return

    phi1 = compute_phi1(mr)
    phi3 = compute_phi3(eff)
    models = [m for m in DEFENSE_ORDER if m in phi1.index and m in phi3.index]

    fig, ax = plt.subplots(figsize=(3.35, 2.8))

    legend_eps = []
    for eps in EPSILON_LEVELS:
        mk, ls, alp = EPS_STYLES[eps]
        phi2 = compute_phi2(mr, "pgd", eps)
        if phi2.empty:
            continue

        pts  = np.array([[phi1[m], phi2[m]] for m in models
                         if m in phi2.index])
        mods = [m for m in models if m in phi2.index]

        if len(pts) == 0:
            continue

        phi34 = np.array([[phi3[m], phi1[m]] for m in mods])
        full_mat = np.hstack([pts, phi34])
        pareto   = is_pareto_optimal(full_mat)

        phi3v = np.array([phi3[m] for m in mods])
        phi3_norm = (phi3v - phi3v.min()) / (phi3v.max() - phi3v.min() + 1e-10)
        sizes = 25 + phi3_norm * 80

        for idx, m in enumerate(mods):
            col = FS.get_color(m)
            edge = "black" if pareto[idx] else "0.7"
            lw   = 0.8 if pareto[idx] else 0.4
            ax.scatter(pts[idx, 0], pts[idx, 1],
                       s=sizes[idx], marker=mk, color=col,
                       edgecolors=edge, linewidths=lw,
                       alpha=alp, zorder=3 + int(pareto[idx]))

        # Pareto frontier line for this epsilon
        front = _pareto_front_path(pts, pareto)
        if front is not None and len(front) > 1:
            ax.plot(front[:, 0], front[:, 1], linestyle=ls,
                    color="0.4", linewidth=0.7, alpha=0.8)

        legend_eps.append(mpatches.Patch(
            color="0.5", linestyle=ls, fill=False,
            label=f"$\\varepsilon$={eps:.2f}"))

    ax.set_xlabel(r"$\phi_1$ Clean F1", fontsize=8)
    ax.set_ylabel(r"$\phi_2$ Adversarial Resilience", fontsize=8)
    ax.set_title("Pareto Front Evolution (MLP, PGD)", fontsize=9, pad=4)

    # Defense color legend
    def_handles = [mpatches.Patch(color=FS.get_color(m), label=FS.get_label(m))
                   for m in models]
    leg1 = ax.legend(handles=def_handles, fontsize=6,
                     loc="lower left", handlelength=1.0)
    ax.add_artist(leg1)
    # Epsilon style legend using Line2D proxies
    from matplotlib.lines import Line2D
    eps_handles = [
        Line2D([0], [0], marker=EPS_STYLES[eps][0], color="0.4",
               linestyle=EPS_STYLES[eps][1], linewidth=0.8,
               markersize=5, label=f"$\\varepsilon$={eps:.2f}")
        for eps in EPSILON_LEVELS
    ]
    ax.legend(handles=eps_handles, fontsize=6,
              loc="lower right", handlelength=1.5)

    plt.tight_layout(pad=0.6)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 4: Theta sensitivity ──────────────────────────────────────────────

def plot_theta_sensitivity(obj_mlp: pd.DataFrame, obj_cnn: pd.DataFrame,
                           out_path: str) -> None:
    theta_names  = list(THETA_PRESETS.keys())
    theta_vals   = list(THETA_PRESETS.values())

    fig, axes = plt.subplots(1, 2, figsize=(6.69, 2.8))

    for ax, obj, title in zip(axes,
                               [obj_mlp, obj_cnn],
                               ["(a) MLP Backbone", "(b) 1D-CNN Backbone"]):
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
                            out_path: str) -> None:
    df = obj.copy().reset_index()
    df.columns = ["model", "phi1_clean_f1", "phi2_resilience",
                  "phi3_cost_eff", "phi4_fairness"]
    df["is_pareto_optimal"] = pareto_mask
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"  Saved: {out_path}")


def write_selection_csv(obj_mlp: pd.DataFrame, obj_cnn: pd.DataFrame,
                         out_path: str) -> None:
    rows = []
    for name, theta in THETA_PRESETS.items():
        mlp_sel = cnn_sel = "N/A"
        if not obj_mlp.empty:
            models_m = obj_mlp.index.tolist()
            mat_m    = obj_mlp[["phi1", "phi2", "phi3", "phi4"]].values
            mlp_sel  = FS.get_label(select_defense(mat_m, models_m, theta))
        if not obj_cnn.empty:
            models_c = obj_cnn.index.tolist()
            mat_c    = obj_cnn[["phi1", "phi2", "phi3", "phi4"]].values
            cnn_sel  = FS.get_label(select_defense(mat_c, models_c, theta))
        rows.append({
            "theta_name":   name,
            "theta_values": str(theta),
            "mlp_selected": mlp_sel,
            "cnn_selected": cnn_sel,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


# ── Stdout summary ────────────────────────────────────────────────────────────

def print_summary(backbone: str, obj: pd.DataFrame) -> None:
    if obj.empty:
        print(f"\n[{backbone}] No data available.")
        return

    mat    = obj[["phi1", "phi2", "phi3", "phi4"]].values
    models = obj.index.tolist()
    pareto = is_pareto_optimal(mat)

    print(f"\n{'='*60}")
    print(f"  {backbone} Backbone — 4D Risk Profile")
    print(f"{'='*60}")
    header = f"{'Defense':<25}  phi1   phi2   phi3   phi4  Pareto"
    print(header)
    print("-" * len(header))
    for i, m in enumerate(models):
        p1, p2, p3, p4 = mat[i]
        star = "*" if pareto[i] else " "
        print(f"  {FS.get_label(m):<23}  {p1:.3f}  {p2:.3f}  {p3:.3f}"
              f"  {p4:.3f}  {star}")

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
    p.add_argument("--outputs-root", default=os.path.join(PROJECT_ROOT, "outputs"),
                   help="root directory containing mlp/, cnn1d/, ft_transformer/ "
                        "(default: <repo>/outputs)")
    p.add_argument("--no-figs",      action="store_true",
                   help="Skip figure generation")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ref_atk = args.ref_attack
    ref_eps = args.ref_epsilon
    dirs = {key: os.path.join(args.outputs_root, sub) for key, sub in BACKBONE_DIRS.items()}
    fig_dir = os.path.join(args.outputs_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── Load data ──
    print(f"\nLoading data  (ref: {ref_atk}, eps={ref_eps:.2f})")
    data = {k: load_backbone_data(k, dirs) for k in ("MLP", "CNN", "CICIDS")}
    missing = [k for k in ("MLP", "CNN") if not data[k]]
    if missing:
        for key in missing:
            print(f"  [missing] {dirs[key]}/mean_results.csv -- "
                  f"run 'uv run python run_experiments.py --backbones {key.lower()}' first")
        if len(missing) == len(("MLP", "CNN")):
            print("\nNo backbone results found; nothing to analyse.")
            return 1

    # ── Build objective matrices ──
    obj_mlp  = build_objective_matrix(data["MLP"],  ref_atk, ref_eps)
    obj_cnn  = build_objective_matrix(data["CNN"],  ref_atk, ref_eps)

    # ── Stdout summaries ──
    print_summary("MLP",  obj_mlp)
    print_summary("1D-CNN", obj_cnn)

    # ── CSV outputs ──
    print("\nWriting CSV tables...")
    for backbone, obj, base_dir in [
        ("MLP",    obj_mlp,  dirs["MLP"]),
        ("CNN",    obj_cnn,  dirs["CNN"]),
    ]:
        if obj.empty:
            continue
        mat    = obj[["phi1", "phi2", "phi3", "phi4"]].values
        pareto = is_pareto_optimal(mat)
        write_risk_profile_csv(
            obj, pareto,
            os.path.join(base_dir, "risk_profile_4d.csv"))

    target_dir = dirs["MLP"] if not obj_mlp.empty else dirs["CNN"]
    os.makedirs(target_dir, exist_ok=True)
    write_selection_csv(obj_mlp, obj_cnn,
                        os.path.join(target_dir, "pareto_selection_results.csv"))

    if args.no_figs:
        print("\nFigure generation skipped (--no-figs).")
        return 0

    # ── Figures ──
    print("\nGenerating figures...")

    plot_pareto_front_comparison(
        obj_mlp, obj_cnn,
        os.path.join(fig_dir, "pareto_front_comparison.png"))

    plot_risk_surface_heatmap(
        data["MLP"], data["CNN"],
        os.path.join(fig_dir, "risk_surface_heatmap.png"))

    plot_epsilon_pareto_evolution(
        data["MLP"],
        os.path.join(fig_dir, "epsilon_pareto_evolution.png"))

    plot_theta_sensitivity(
        obj_mlp, obj_cnn,
        os.path.join(fig_dir, "theta_sensitivity.png"))

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
