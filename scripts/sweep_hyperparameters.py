"""Hyperparameter sensitivity sweep.

Varies the training-time perturbation budget and the defense-specific
coefficients (TRADES beta, Free AT replays, sensitivity ratio) and records how
the four-dimensional profiles and the resulting selections respond.

    python scripts/sweep_hyperparameters.py --device cuda --out-dir outputs/sweep
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path when run by path)

import argparse
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import ids_defense_selection as idsds

for fn in ["SimHei", "Microsoft YaHei"]:
    try:
        matplotlib.font_manager.findfont(fn, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [fn, "DejaVu Sans"]
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

SEEDS = idsds.DEFAULT_SEEDS
TRADES_BETAS = [1.0, 3.0, 6.0, 10.0, 15.0]
CLASS_AWARE_WEIGHTS = [1.0, 2.0, 3.0, 5.0, 10.0]


def evaluate_pgd(model, eval_x, eval_y, device, config, mask_t, mins_t, maxs_t, epsilon=0.10):
    model.eval()
    x_t = torch.from_numpy(eval_x.astype(np.float32)).to(device)
    y_t = torch.from_numpy(eval_y.astype(np.float32)).to(device)
    x_adv = idsds.pgd_attack(model, x_t, y_t, epsilon, config.adv_alpha, config.adv_steps, mask_t, mins_t, maxs_t)

    with torch.no_grad():
        clean_probs = torch.sigmoid(model(x_t)).cpu().numpy()
        adv_probs = torch.sigmoid(model(x_adv)).cpu().numpy()

    clean_pred = (clean_probs >= 0.5).astype(np.int32)
    clean_metrics = idsds.classification_metrics(eval_y, clean_probs)
    adv_metrics = idsds.classification_metrics(eval_y, adv_probs, clean_pred)
    return clean_metrics, adv_metrics


def sweep_trades_beta(train_loader, baseline_model, eval_x, eval_y, device, config, mask_t, mins_t, maxs_t):
    rows = []
    for beta in TRADES_BETAS:
        for seed in SEEDS:
            idsds.set_seed(seed)
            model = copy.deepcopy(baseline_model)
            config_copy = copy.copy(config)
            config_copy.trades_beta = beta
            model = idsds.fit_trades(model, train_loader, config.adv_epochs, device, config_copy, mask_t, mins_t, maxs_t)
            clean_m, adv_m = evaluate_pgd(model, eval_x, eval_y, device, config, mask_t, mins_t, maxs_t)
            rows.append({"param": "trades_beta", "value": beta, "seed": seed,
                         "clean_f1": clean_m["f1"], "clean_auc": clean_m["auc"],
                         "robust_f1": adv_m["f1"], "asr": adv_m["attack_success_rate"]})
            print(f"  TRADES beta={beta} seed={seed}: clean_f1={clean_m['f1']:.4f} robust_f1={adv_m['f1']:.4f} asr={adv_m['attack_success_rate']:.4f}")
    return pd.DataFrame(rows)


def sweep_class_aware_weight(train_loader, baseline_model, eval_x, eval_y, device, config, mask_t, mins_t, maxs_t, metadata):
    numeric_mask = metadata["numeric_mask"]
    ca_mask, _ = idsds.compute_class_aware_sensitivity_mask(
        baseline_model, train_loader, device, numeric_mask,
        metadata["feature_names"], config.sensitivity_top_ratio, config.sensitivity_batches
    )
    ca_mask_t = torch.from_numpy(ca_mask.astype(np.float32)).to(device)
    rows = []
    for w in CLASS_AWARE_WEIGHTS:
        for seed in SEEDS:
            idsds.set_seed(seed)
            model = copy.deepcopy(baseline_model)
            model = idsds.fit_class_aware_constrained(model, train_loader, config.adv_epochs, device, config, ca_mask_t, mins_t, maxs_t, w)
            clean_m, adv_m = evaluate_pgd(model, eval_x, eval_y, device, config, ca_mask_t, mins_t, maxs_t)
            rows.append({"param": "class_aware_weight", "value": w, "seed": seed,
                         "clean_f1": clean_m["f1"], "clean_auc": clean_m["auc"],
                         "robust_f1": adv_m["f1"], "asr": adv_m["attack_success_rate"]})
            print(f"  CA weight={w} seed={seed}: clean_f1={clean_m['f1']:.4f} robust_f1={adv_m['f1']:.4f} asr={adv_m['attack_success_rate']:.4f}")
    return pd.DataFrame(rows)


def plot_sweep(df, param_name, param_label, output_path):
    means = df.groupby("value").mean(numeric_only=True).reset_index()
    stds = df.groupby("value").std(numeric_only=True).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric, label in zip(axes, ["clean_f1", "robust_f1", "asr"], ["Clean F1", "Robust F1 (PGD ε=0.10)", "ASR"]):
        ax.errorbar(means["value"], means[metric], yerr=stds[metric], marker="o", linewidth=2, capsize=4, color="#e74c3c")
        ax.set_xlabel(param_label, fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f"{label} vs {param_label}", fontsize=13)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------- sweep driver
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto",
                        help="torch device: auto, cpu, cuda, cuda:N or mps")
    parser.add_argument("--output-dir", default=str(idsds.DEFAULT_OUTPUT_ROOT / "sweep"),
                        help="directory for the sweep tables and figures")
    args = parser.parse_args()

    out_dir = idsds.resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = idsds.resolve_device(args.device)
    idsds.log_device(args.device, device)

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

    train_loader = idsds.make_dataloader(x_train, y_train, config.batch_size, shuffle=True)
    input_dim = x_train.shape[1]
    mins_t = torch.from_numpy(metadata["numeric_mins"].astype(np.float32)).to(device)
    maxs_t = torch.from_numpy(metadata["numeric_maxs"].astype(np.float32)).to(device)
    numeric_mask_t = torch.from_numpy(metadata["numeric_mask"].astype(np.float32)).to(device)

    idsds.set_seed(42)
    baseline = idsds.MLPBackbone(input_dim, config.hidden_dims, config.dropout)
    baseline = idsds.fit_supervised(baseline, train_loader, config.baseline_epochs, device, config)

    print("=== TRADES beta sweep ===")
    beta_df = sweep_trades_beta(train_loader, baseline, eval_x, eval_y, device, config, numeric_mask_t, mins_t, maxs_t)
    beta_df.to_csv(out_dir / "trades_beta_sweep.csv", index=False)
    plot_sweep(beta_df, "trades_beta", "TRADES β", out_dir / "trades_beta_sweep.png")
    print(f"Saved: {out_dir / 'trades_beta_sweep.png'}")

    beta_summary = beta_df.groupby("value").agg(
        clean_f1_mean=("clean_f1", "mean"), clean_f1_std=("clean_f1", "std"),
        robust_f1_mean=("robust_f1", "mean"), robust_f1_std=("robust_f1", "std"),
        asr_mean=("asr", "mean"), asr_std=("asr", "std"),
    ).reset_index()
    print("\nTRADES beta summary:")
    print(beta_summary.round(4).to_string(index=False))

    print("\n=== Class-aware weight sweep ===")
    ca_df = sweep_class_aware_weight(train_loader, baseline, eval_x, eval_y, device, config, numeric_mask_t, mins_t, maxs_t, metadata)
    ca_df.to_csv(out_dir / "class_aware_weight_sweep.csv", index=False)
    plot_sweep(ca_df, "class_aware_weight", "Class-aware minority weight",
               out_dir / "class_aware_weight_sweep.png")
    print(f"Saved: {out_dir / 'class_aware_weight_sweep.png'}")

    ca_summary = ca_df.groupby("value").agg(
        clean_f1_mean=("clean_f1", "mean"), clean_f1_std=("clean_f1", "std"),
        robust_f1_mean=("robust_f1", "mean"), robust_f1_std=("robust_f1", "std"),
        asr_mean=("asr", "mean"), asr_std=("asr", "std"),
    ).reset_index()
    print("\nClass-aware weight summary:")
    print(ca_summary.round(4).to_string(index=False))

    print("\n=== Phase 3.1 Complete ===")


if __name__ == "__main__":
    main()
