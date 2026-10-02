#!/usr/bin/env python
"""Write a markdown record for one experiment run and register it in an index.

The record is generated from the artefacts in ``--outputs-dir`` (the same
directory that ``check_outputs.py`` validates), so every logged number is
traceable to a CSV file:

    uv run python scripts/log_experiment.py \
        --outputs-dir outputs/ft_probe \
        --record-dir /home/mx/tmp/IDS/实验记录 \
        --title "FT probe alpha=0.10, eps<=0.20, adaptive 50x5" \
        --status done --command "uv run python scripts/run_ft_transformer.py ..."

The generated file has ``## 分析`` and ``## 结论 / 下一步`` placeholders for the
interpretation; the index row is appended to ``INDEX.md`` in the same folder.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path when run by path)

import argparse
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_FILES = (
    "raw_results.csv", "mean_results.csv", "std_results.csv",
    "significance_tests.csv", "efficiency_raw.csv", "efficiency_mean.csv",
    "category_raw_results.csv", "category_mean_results.csv",
    "attack_generalization.csv", "adaptive_attack_raw.csv",
    "adaptive_attack_mean.csv", "hyperparameters.csv", "run_summary.json",
)

STATUS_CHOICES = ("planned", "running", "done", "failed")


def git_revision() -> str:
    """Return ``<short-sha>[-dirty]`` of the current checkout, or ``unknown``."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001 - logging must never fail on git
        return "unknown"


def slugify(text: str) -> str:
    keep = [ch if (ch.isalnum() or ch in "-_") else "-" for ch in text.strip()]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:60] or "run"


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def key_results(mean: pd.DataFrame | None, std: pd.DataFrame | None,
                ref_attack: str, ref_epsilon: float,
                efficiency: pd.DataFrame | None) -> str:
    if mean is None or mean.empty:
        return "_no mean_results.csv; fill in manually_"
    subset = mean[(mean["attack"] == ref_attack) &
                  ((mean["epsilon"] - ref_epsilon).abs() < 1e-9)]
    if subset.empty:
        return f"_no rows for attack={ref_attack!r}, epsilon={ref_epsilon}_"
    train_seconds = {}
    if efficiency is not None and {"model", "train_seconds"}.issubset(efficiency.columns):
        train_seconds = dict(zip(efficiency["model"], efficiency["train_seconds"]))
    std_view = {}
    if std is not None and not std.empty:
        std_subset = std[(std["attack"] == ref_attack) &
                         ((std["epsilon"] - ref_epsilon).abs() < 1e-9)]
        std_view = dict(zip(std_subset["model"], std_subset["attack_success_rate"]))
    rows = []
    for _, row in subset.sort_values("attack_success_rate").iterrows():
        asr = float(row["attack_success_rate"])
        asr_std = std_view.get(row["model"])
        rows.append({
            "defense": row["model"],
            "clean F1": "",
            "ASR": f"{asr:.5f}" + (f" ± {asr_std:.5f}" if asr_std is not None else ""),
            "phi2 = 1-ASR": f"{1.0 - asr:.5f}",
            "train_seconds": f"{train_seconds.get(row['model'], float('nan')):.1f}"
                             if row["model"] in train_seconds else "",
        })
    return _markdown_table(rows, ["defense", "ASR", "phi2 = 1-ASR",
                                  "train_seconds"])


def build_record(outputs_dir: Path, title: str, status: str, command: str,
                 ref_attack: str, ref_epsilon: float, notes: list[str]) -> str:
    summary_path = outputs_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    config = summary.get("config", {})
    evaluation = config.get("evaluation", {})
    training = config.get("training", {})

    def read(name: str) -> pd.DataFrame | None:
        path = outputs_dir / name
        return pd.read_csv(path) if path.is_file() else None

    raw = read("raw_results.csv")
    seeds = sorted(raw["seed"].unique().tolist()) if raw is not None and "seed" in raw.columns else []
    mean, std = read("mean_results.csv"), read("std_results.csv")
    efficiency = read("efficiency_mean.csv")
    inventory = []
    for name in EXPECTED_FILES:
        path = outputs_dir / name
        size = f"{path.stat().st_size / 1024:.0f} KB" if path.is_file() else "missing"
        inventory.append({"file": name, "status": "ok" if path.is_file() else "missing", "size": size})

    lines = [
        f"# 实验记录：{title}",
        "",
        f"> 记录时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}　状态：**{status}**　"
        f"代码版本：`{git_revision()}`",
        f"> 产物目录：`{outputs_dir}`　设备：`{summary.get('resolved_device', 'unknown')}`",
        "",
        "## 运行配置",
        "",
        _markdown_table([{
            "seeds": ",".join(str(s) for s in seeds) or evaluation.get("seeds", ""),
            "eval rows": evaluation.get("eval_attack_rows", ""),
            "epsilon": evaluation.get("epsilon_list", ""),
            "adaptive": evaluation.get("adaptive_eval", ""),
            "epochs": f"{training.get('baseline_epochs', '')}+{training.get('adv_epochs', '')}",
            "batch": training.get("batch_size", ""),
            "lr": training.get("learning_rate", ""),
            "FT capacity": (f"{training.get('ft_d_token')}/{training.get('ft_n_heads')}/"
                            f"{training.get('ft_n_layers')}/{training.get('ft_d_ffn')}"),
        }], ["seeds", "eval rows", "epsilon", "adaptive", "epochs", "batch", "lr", "FT capacity"]),
        "",
        f"数据集划分：`{summary.get('dataset_split', {})}`",
        "",
        "## 运行命令",
        "",
        "```bash",
        command or "# 待补充：实际执行的命令",
        "```",
    ]
    if notes:
        lines += ["", "## 备注", ""] + [f"- {note}" for note in notes]
    lines += [
        "",
        f"## 关键结果（{ref_attack}，ε={ref_epsilon}）",
        "",
        key_results(mean, std, ref_attack, ref_epsilon, efficiency),
        "",
        "## 产物清单",
        "",
        _markdown_table(inventory, ["file", "status", "size"]),
        "",
        "## 分析",
        "",
        "<!-- 人工填写：与基线/上次实验的差异、原因假设、需要补的实验 -->",
        "",
        "## 结论 / 下一步",
        "",
        "<!-- 人工填写：采纳/否定的假设、下一步命令 -->",
        "",
    ]
    return "\n".join(lines)


def update_index(record_dir: Path, filename: str, title: str, status: str) -> None:
    index = record_dir / "INDEX.md"
    header = ("# 实验总账\n\n"
              "| 日期 | 记录 | 状态 | 代码版本 | 关键结论 |\n"
              "|---|---|---|---|---|\n")
    row = (f"| {date.today().isoformat()} | [{title}]({filename}) | {status} | "
           f"`{git_revision()}` | 待补充 |\n")
    if index.is_file():
        text = index.read_text(encoding="utf-8")
        if filename in text:
            return
        if not text.startswith("# 实验总账"):
            text = header + text
        index.write_text(text.rstrip("\n") + "\n" + row, encoding="utf-8")
    else:
        index.write_text(header + row, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-dir", required=True,
                        help="run directory, e.g. outputs/ft_probe")
    parser.add_argument("--record-dir", required=True,
                        help="where the markdown record and INDEX.md are written")
    parser.add_argument("--title", required=True, help="human-readable run title")
    parser.add_argument("--status", choices=STATUS_CHOICES, default="done")
    parser.add_argument("--command", default="", help="exact command that was executed")
    parser.add_argument("--ref-attack", default="pgd")
    parser.add_argument("--ref-epsilon", type=float, default=0.10)
    parser.add_argument("--note", action="append", default=[],
                        help="extra note line (repeatable)")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    if not outputs_dir.is_dir():
        print(f"error: {outputs_dir} is not a directory", file=sys.stderr)
        return 2
    record_dir = Path(args.record_dir).expanduser()
    record_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{date.today().isoformat()}_{slugify(args.title)}.md"
    path = record_dir / filename
    suffix = 2
    while path.exists():
        path = record_dir / f"{date.today().isoformat()}_{slugify(args.title)}_v{suffix}.md"
        suffix += 1

    path.write_text(build_record(outputs_dir, args.title, args.status,
                                 args.command, args.ref_attack,
                                 args.ref_epsilon, args.note),
                    encoding="utf-8")
    update_index(record_dir, path.name, args.title, args.status)
    print(f"record : {path}")
    print(f"index  : {record_dir / 'INDEX.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
