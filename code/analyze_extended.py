"""
Extended robustness analysis.
Trains all six defenses per seed, then runs:
  - Fine-grained epsilon sweep (10 PGD points)
  - ROC curves (clean + attacked)
  - Gradient masking detection
  - Full test set PGD attack evaluation

Usage:
  uv run python code/analyze_extended.py --device cuda --output-dir outputs/extended_analysis
"""
from __future__ import annotations

import argparse
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score

import ids_defense_selection as idsds
from ids_defense_selection import style as FS
from ids_defense_selection.paths import DEFAULT_OUTPUT_ROOT, resolve_path

for fn in ["SimHei", "Microsoft YaHei"]:
    try:
        matplotlib.font_manager.findfont(fn, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [fn, "DejaVu Sans"]
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

EPSILON_SWEEP = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
SEEDS = idsds.DEFAULT_SEEDS
MODEL_NAMES = list(idsds.DEFENSE_ORDER)
MODEL_LABELS = {name: FS.get_label(name) for name in MODEL_NAMES}
COLORS = [FS.get_color(name) for name in MODEL_NAMES]


def train_all_models(config, train_loader, baseline_model, device, metadata):
    """Train all 6 models and return dict {name: model}."""
    numeric_mask = metadata["numeric_mask"]
    numeric_mask_t = torch.from_numpy(numeric_mask.astype(np.float32)).to(device)
    mins_t = torch.from_numpy(metadata["numeric_mins"].astype(np.float32)).to(device)
    maxs_t = torch.from_numpy(metadata["numeric_maxs"].astype(np.float32)).to(device)

    models = {}
    models["standard"] = copy.deepcopy(baseline_model)

    def adv_builder(m, xb, yb):
        return idsds.pgd_attack(m, xb, yb, config.adv_epsilon, config.adv_alpha,
                                config.adv_steps, numeric_mask_t, mins_t, maxs_t)

    adv_model = copy.deepcopy(baseline_model)
    models["adv_training"] = idsds.fit_supervised(
        adv_model, train_loader, config.adv_epochs, device, config,
        adversarial=True, attack_builder=adv_builder)

    sel_mask, _ = idsds.compute_sensitivity_mask(baseline_model, train_loader, device, numeric_mask, metadata["feature_names"], config.sensitivity_top_ratio, config.sensitivity_batches)
    sel_mask_t = torch.from_numpy(sel_mask.astype(np.float32)).to(device)

    def constrained_builder(m, xb, yb):
        return idsds.pgd_attack(m, xb, yb, config.adv_epsilon, config.adv_alpha,
                                config.adv_steps, sel_mask_t, mins_t, maxs_t)

    c_model = copy.deepcopy(baseline_model)
    models["constrained_adv"] = idsds.fit_supervised(
        c_model, train_loader, config.adv_epochs, device, config,
        adversarial=True, attack_builder=constrained_builder)

    trades_model = copy.deepcopy(baseline_model)
    models["trades"] = idsds.fit_trades(trades_model, train_loader, config.adv_epochs, device, config, numeric_mask_t, mins_t, maxs_t)

    free_model = copy.deepcopy(baseline_model)
    models["free_at"] = idsds.fit_free_at(free_model, train_loader, config.adv_epochs, device, config, numeric_mask_t, mins_t, maxs_t)

    ca_mask, _ = idsds.compute_class_aware_sensitivity_mask(baseline_model, train_loader, device, numeric_mask, metadata["feature_names"], config.sensitivity_top_ratio, config.sensitivity_batches)
    ca_mask_t = torch.from_numpy(ca_mask.astype(np.float32)).to(device)
    ca_model = copy.deepcopy(baseline_model)
    models["class_aware_constrained"] = idsds.fit_class_aware_constrained(ca_model, train_loader, config.adv_epochs, device, config, ca_mask_t, mins_t, maxs_t, config.class_aware_minority_weight)

    return models, numeric_mask_t, mins_t, maxs_t


# ---------------------------------------------------------------- epsilon sweep
def epsilon_sweep_eval(models, eval_x, eval_y, device, config, numeric_mask_t, mins_t, maxs_t):
    """Evaluate all models at many epsilon values under PGD."""
    rows = []
    for name, model in models.items():
        model.eval()
        x_t = torch.from_numpy(eval_x.astype(np.float32)).to(device)
        y_t = torch.from_numpy(eval_y.astype(np.float32)).to(device)
        with torch.no_grad():
            clean_logits = model(x_t)
            clean_probs = torch.sigmoid(clean_logits).cpu().numpy()
        clean_pred = (clean_probs >= 0.5).astype(np.int32)
        for eps in EPSILON_SWEEP:
            x_adv = idsds.pgd_attack(model, x_t, y_t, eps, config.adv_alpha, config.adv_steps, numeric_mask_t, mins_t, maxs_t)
            with torch.no_grad():
                logits = model(x_adv)
                probs = torch.sigmoid(logits).cpu().numpy()
            metrics = idsds.classification_metrics(eval_y, probs, clean_pred=clean_pred)
            rows.append({"model": name, "epsilon": eps, **metrics})
    return pd.DataFrame(rows)


def roc_eval(models, eval_x, eval_y, device, config, numeric_mask_t, mins_t, maxs_t):
    """Collect ROC curve data for clean and PGD eps=0.10."""
    roc_data = {}
    for name, model in models.items():
        model.eval()
        x_t = torch.from_numpy(eval_x.astype(np.float32)).to(device)
        y_t = torch.from_numpy(eval_y.astype(np.float32)).to(device)
        with torch.no_grad():
            probs_clean = torch.sigmoid(model(x_t)).cpu().numpy()
        x_adv = idsds.pgd_attack(model, x_t, y_t, 0.10, config.adv_alpha, config.adv_steps, numeric_mask_t, mins_t, maxs_t)
        with torch.no_grad():
            probs_adv = torch.sigmoid(model(x_adv)).cpu().numpy()
        roc_data[name] = {"clean": probs_clean, "attacked": probs_adv}
    return roc_data


def gradient_masking_eval(models, eval_x, eval_y, device, numeric_mask_t):
    """Compute input gradient L2 norms for gradient masking detection."""
    rows = []
    for name, model in models.items():
        model.eval()
        x_t = torch.from_numpy(eval_x[:2000].astype(np.float32)).to(device).requires_grad_(True)
        y_t = torch.from_numpy(eval_y[:2000].astype(np.float32)).to(device)
        logits = model(x_t)
        loss = F.binary_cross_entropy_with_logits(logits, y_t)
        loss.backward()
        grads = x_t.grad.detach()
        masked_grads = grads * numeric_mask_t
        l2_norms = masked_grads.norm(dim=1).cpu().numpy()
        rows.append({
            "model": name,
            "grad_l2_mean": float(l2_norms.mean()),
            "grad_l2_std": float(l2_norms.std()),
            "grad_l2_median": float(np.median(l2_norms)),
            "loss_value": float(loss.item()),
        })
        model.zero_grad(set_to_none=True)
    return pd.DataFrame(rows)


def full_test_attack(models, x_test, y_test, device, config, numeric_mask_t, mins_t, maxs_t):
    """PGD eps=0.10 on the full test set (batched to avoid OOM)."""
    batch_size = 4096
    rows = []
    for name, model in models.items():
        model.eval()
        all_probs = []
        clean_preds = []
        n = len(x_test)
        with torch.no_grad():
            for i in range(0, n, batch_size):
                xb = torch.from_numpy(x_test[i:i+batch_size].astype(np.float32)).to(device)
                logits = model(xb)
                clean_preds.append((torch.sigmoid(logits) >= 0.5).int().cpu().numpy())
        clean_pred = np.concatenate(clean_preds)

        for i in range(0, n, batch_size):
            xb = torch.from_numpy(x_test[i:i+batch_size].astype(np.float32)).to(device)
            yb = torch.from_numpy(y_test[i:i+batch_size].astype(np.float32)).to(device)
            x_adv = idsds.pgd_attack(model, xb, yb, 0.10, config.adv_alpha, config.adv_steps, numeric_mask_t, mins_t, maxs_t)
            with torch.no_grad():
                probs = torch.sigmoid(model(x_adv)).cpu().numpy()
            all_probs.append(probs)
        y_prob = np.concatenate(all_probs)
        metrics = idsds.classification_metrics(y_test, y_prob, clean_pred)
        rows.append({"model": name, "attack": "pgd", "epsilon": 0.10, "test_rows": n, **metrics})
        print(f"  Full-test PGD eps=0.10 {name}: F1={metrics['f1']:.4f} ASR={metrics['attack_success_rate']:.4f}")
    return pd.DataFrame(rows)


def plot_epsilon_sweep(df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, (metric, ylabel) in enumerate([("f1", "F1"), ("attack_success_rate", "ASR")]):
        ax = axes[i]
        for j, name in enumerate(MODEL_NAMES):
            sub = df[df["model"] == name].sort_values("epsilon")
            ax.plot(sub["epsilon"], sub[metric], marker="o", color=COLORS[j], label=MODEL_LABELS[name], linewidth=2, markersize=5)
        ax.set_xlabel("PGD 扰动强度 ε", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_title("F1 vs 扰动强度（细粒度扫描）", fontsize=13)
    axes[1].set_title("ASR vs 扰动强度（细粒度扫描）", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc(roc_data, eval_y, output_path_clean, output_path_attacked):
    for condition, out_path, title in [
        ("clean", output_path_clean, "ROC 曲线：Clean 条件"),
        ("attacked", output_path_attacked, "ROC 曲线：PGD ε=0.10 攻击后"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 7))
        for j, name in enumerate(MODEL_NAMES):
            probs = roc_data[name][condition]
            fpr, tpr, _ = roc_curve(eval_y, probs)
            auc_val = roc_auc_score(eval_y, probs)
            ax.plot(fpr, tpr, color=COLORS[j], linewidth=2, label=f"{MODEL_LABELS[name]} (AUC={auc_val:.4f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_gradient_masking(grad_df, output_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(grad_df))
    bars = ax.bar(x, grad_df["grad_l2_mean"], yerr=grad_df["grad_l2_std"],
                  color=COLORS[:len(grad_df)], edgecolor="white", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in grad_df["model"]], fontsize=10, rotation=15)
    ax.set_ylabel("输入梯度 L2 范数（均值±标准差）", fontsize=12)
    ax.set_title("梯度遮蔽检测：各防御模型的输入梯度范数", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, grad_df["grad_l2_mean"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001, f"{v:.4f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------- entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "extended_analysis"),
                        help="output directory (relative paths resolve against the repository root)")
    args = parser.parse_args()

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    config = idsds.ExperimentConfig(
        train_path=str(idsds.DEFAULT_DATA_DIR / "train.csv"),
        test_path=str(idsds.DEFAULT_DATA_DIR / "test.csv"),
        output_dir=str(out_dir),
        device=args.device,
    )

    train_df, test_df = idsds.load_unsw_nb15(config.train_path, config.test_path)
    x_train, y_train, x_test, y_test, metadata = idsds.build_features(train_df, test_df)

    eval_indices = idsds.stratified_subset_indices(y_test, config.eval_attack_rows,
                                                   seed=config.eval_subset_seed)
    eval_x, eval_y = x_test[eval_indices], y_test[eval_indices]

    all_eps_sweep = []
    all_roc = {name: {"clean": [], "attacked": []} for name in MODEL_NAMES}
    all_grad = []
    all_full_test = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")
        idsds.set_seed(seed)

        train_loader = idsds.make_dataloader(x_train, y_train, config.batch_size, shuffle=True)
        input_dim = x_train.shape[1]

        print("  Training baseline model...")
        baseline = idsds.MLPBackbone(input_dim, config.hidden_dims, config.dropout)
        baseline = idsds.fit_supervised(baseline, train_loader, config.baseline_epochs, device, config)

        print("  Training 6 defense models...")
        models, nmask_t, mins_t, maxs_t = train_all_models(config, train_loader, baseline, device, metadata)

        print("  [2.1] Epsilon sweep...")
        eps_df = epsilon_sweep_eval(models, eval_x, eval_y, device, config, nmask_t, mins_t, maxs_t)
        eps_df["seed"] = seed
        all_eps_sweep.append(eps_df)

        print("  [2.2] ROC curves...")
        roc = roc_eval(models, eval_x, eval_y, device, config, nmask_t, mins_t, maxs_t)
        for name in MODEL_NAMES:
            all_roc[name]["clean"].append(roc[name]["clean"])
            all_roc[name]["attacked"].append(roc[name]["attacked"])

        print("  [2.3] Gradient masking detection...")
        grad_df = gradient_masking_eval(models, eval_x, eval_y, device, nmask_t)
        grad_df["seed"] = seed
        all_grad.append(grad_df)

        print("  [2.4] Full test set attack...")
        ft_df = full_test_attack(models, x_test, y_test, device, config, nmask_t, mins_t, maxs_t)
        ft_df["seed"] = seed
        all_full_test.append(ft_df)

    # Aggregate and save
    eps_raw = pd.concat(all_eps_sweep, ignore_index=True)
    eps_mean = eps_raw.groupby(["model", "epsilon"]).mean(numeric_only=True).reset_index()
    eps_mean = eps_mean.drop(columns=["seed"], errors="ignore")
    eps_raw.to_csv(out_dir / "epsilon_sweep_raw.csv", index=False)
    eps_mean.to_csv(out_dir / "epsilon_sweep_mean.csv", index=False)
    plot_epsilon_sweep(eps_mean, out_dir / "epsilon_sweep.png")
    print(f"\nSaved epsilon sweep: {out_dir / 'epsilon_sweep.png'}")

    avg_roc = {}
    for name in MODEL_NAMES:
        avg_roc[name] = {
            "clean": np.mean(np.stack(all_roc[name]["clean"]), axis=0),
            "attacked": np.mean(np.stack(all_roc[name]["attacked"]), axis=0),
        }
    plot_roc(avg_roc, eval_y, out_dir / "roc_clean.png", out_dir / "roc_attacked.png")
    print(f"Saved ROC curves: {out_dir / 'roc_clean.png'}, {out_dir / 'roc_attacked.png'}")

    grad_raw = pd.concat(all_grad, ignore_index=True)
    grad_mean = grad_raw.groupby("model").mean(numeric_only=True).reset_index()
    grad_mean = grad_mean.drop(columns=["seed"], errors="ignore")
    grad_raw.to_csv(out_dir / "gradient_norms_raw.csv", index=False)
    grad_mean.to_csv(out_dir / "gradient_norms_mean.csv", index=False)
    plot_gradient_masking(grad_mean, out_dir / "gradient_masking_check.png")
    print(f"Saved gradient masking: {out_dir / 'gradient_masking_check.png'}")

    ft_raw = pd.concat(all_full_test, ignore_index=True)
    ft_mean = ft_raw.groupby(["model", "attack", "epsilon"]).mean(numeric_only=True).reset_index()
    ft_mean = ft_mean.drop(columns=["seed"], errors="ignore")
    ft_raw.to_csv(out_dir / "full_test_attack_raw.csv", index=False)
    ft_mean.to_csv(out_dir / "full_test_attack_mean.csv", index=False)
    print(f"Saved full test attack: {out_dir / 'full_test_attack_mean.csv'}")

    print("\n=== Phase 2 Complete ===")


if __name__ == "__main__":
    main()
