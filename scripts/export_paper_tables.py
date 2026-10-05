#!/usr/bin/env python
"""Export the manuscript tables (3-7 plus the significance table) as CSV/Markdown.

Reads the decision artefacts in ``outputs/`` and writes ready-to-paste tables
under ``outputs/tables/``:

    uv run python scripts/export_paper_tables.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path when run by path)

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ids_defense_selection import style as FS

BACKBONES = (
    ("mlp", "mlp", "MLP", "Table 3"),
    ("cnn1d", "cnn1d", "1D-CNN", "Table 4"),
    ("ft", "ft_transformer", "FT-Transformer", "Table 5"),
)
OBJECTIVES = (
    ("phi1_clean_f1", "phi1_std", r"$\phi_1$ Clean F1"),
    ("phi2_resilience", "phi2_std", r"$\phi_2$ Resilience"),
    ("phi3_cost_eff", "phi3_std", r"$\phi_3$ Cost efficiency"),
    ("phi4_worst_class_recall", "phi4_std", r"$\phi_4$ Worst-class recall"),
)
CANONICAL_PRESETS = ("Robust", "Balanced", "Clean", "Cost")


def _fmt(mean: float, std: float | None) -> str:
    if std is None or pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def _markdown(rows: list[dict], columns: list[str], title: str) -> str:
    lines = [f"### {title}", "",
             "| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def _load(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.is_file() else None


def _profile_rows(profile: pd.DataFrame, supported: pd.DataFrame | None) -> list[dict]:
    support_map = {}
    if supported is not None:
        support_map = dict(zip(supported["model"], supported["supported"]))
    order = {name: index for index, name in enumerate(FS.DEFENSE_ORDER)}
    profile = profile.sort_values("model", key=lambda col: col.map(order).fillna(99))
    rows = []
    for row in profile.itertuples():
        entry = {"Defense": FS.get_label(row.model)}
        for mean_col, std_col, header in OBJECTIVES:
            entry[header] = _fmt(getattr(row, mean_col), getattr(row, std_col, None))
        entry["Pareto (Eq. 4)"] = "yes" if bool(row.is_pareto_optimal) else "no"
        if hasattr(row, "is_pareto_optimal_deterministic"):
            entry["Pareto (point)"] = "yes" if bool(row.is_pareto_optimal_deterministic) else "no"
        if hasattr(row, "admissible"):
            entry["Admissible"] = "yes" if bool(row.admissible) else "no"
        entry["Supported"] = ("yes" if support_map.get(row.model) else "no") \
            if row.model in support_map else "-"
        rows.append(entry)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--out-dir", default=None,
                        help="default: <outputs-root>/tables")
    parser.add_argument("--ref-attack", default="pgd")
    parser.add_argument("--ref-epsilon", type=float, default=0.10)
    args = parser.parse_args()

    root = Path(args.outputs_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = {}
    settings_path = root / "decision_settings.json"
    if settings_path.is_file():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))

    sections = []
    columns = ["Defense", *(header for _, _, header in OBJECTIVES),
               "Pareto (Eq. 4)", "Pareto (point)", "Admissible", "Supported"]

    for key, subdir, label, table_name in BACKBONES:
        profile = _load(root / subdir / "risk_profile_4d.csv")
        if profile is None:
            print(f"[skip] {root / subdir / 'risk_profile_4d.csv'} missing")
            continue
        supported = _load(root / subdir / "supportedness.csv")
        rows = _profile_rows(profile, supported)
        table = pd.DataFrame(rows)
        table.to_csv(out_dir / f"{table_name.replace(' ', '').lower()}_{key}.csv", index=False)
        sections.append(_markdown(rows, columns, f"{table_name}. Four-dimensional evaluation ({label})"))

    selection = _load(root / "mlp" / "pareto_selection_results.csv")
    if selection is not None:
        canonical = selection[selection["theta_name"].isin(CANONICAL_PRESETS)]
        rows = []
        for row in canonical.itertuples():
            rows.append({
                "Preset": row.theta_name,
                "theta": row.theta_values,
                "MLP": row.mlp_selected,
                "1D-CNN": row.cnn_selected,
                "FT-Transformer": row.ft_selected,
            })
        canonical.to_csv(out_dir / "table6_preference_selection.csv", index=False)
        sections.append(_markdown(rows, ["Preset", "theta", "MLP", "1D-CNN",
                                         "FT-Transformer"],
                                  "Table 6. Preference-aware defense selection"))

        single_rows = []
        for key, subdir, label, _ in BACKBONES:
            profile = _load(root / subdir / "risk_profile_4d.csv")
            if profile is None:
                continue
            if "admissible" in profile.columns:
                profile = profile[profile["admissible"].astype(bool)]
            for mean_col, _, header in OBJECTIVES:
                winner = profile.loc[profile[mean_col].idxmax(), "model"]
                single_rows.append({"Criterion": header, "Backbone": label,
                                    "Selected": FS.get_label(winner)})
        if selection is not None:
            for row in selection[selection["theta_name"].isin(CANONICAL_PRESETS)].itertuples():
                for label, column in (("MLP", "mlp_selected"),
                                      ("1D-CNN", "cnn_selected"),
                                      ("FT-Transformer", "ft_selected")):
                    single_rows.append({"Criterion": f"Preference: {row.theta_name}",
                                        "Backbone": label,
                                        "Selected": getattr(row, column)})
        table7 = pd.DataFrame(single_rows)
        table7.to_csv(out_dir / "table7_single_vs_preference.csv", index=False)
        pivot = table7.pivot_table(index="Criterion", columns="Backbone",
                                   values="Selected", aggfunc="first")
        row_order = [header for _, _, header in OBJECTIVES] + [
            f"Preference: {preset}" for preset in CANONICAL_PRESETS]
        pivot = pivot.reindex([row for row in row_order if row in pivot.index])
        pivot = pivot.reindex(columns=["MLP", "1D-CNN", "FT-Transformer"])
        markdown_rows = []
        for criterion in pivot.index:
            row = {"Criterion": criterion}
            for backbone in pivot.columns:
                row[backbone] = pivot.loc[criterion, backbone]
            markdown_rows.append(row)
        sections.append(_markdown(markdown_rows,
                                  ["Criterion", *list(pivot.columns)],
                                  "Table 7. Single-metric vs preference-aware selection"))

    significance_sections = []
    for key, subdir, label, _ in BACKBONES:
        table = _load(root / subdir / "significance_enhanced.csv")
        if table is None:
            continue
        subset = table[(table["attack"] == args.ref_attack) &
                       ((table["epsilon"] - args.ref_epsilon).abs() < 1e-9) &
                       (table["metric"] == "f1")].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("holm_t_pvalue")
        subset.to_csv(out_dir / f"table_significance_{key}.csv", index=False)
        markdown_rows = [{
            "Comparison": " vs ".join(
                FS.get_label(part) for part in row.comparison.split("_vs_")),
            "mean diff": f"{row.mean_diff:+.4f}",
            "Cohen's dz": f"{row.cohens_dz:+.2f}",
            "95% CI": f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}]",
            "p (Holm)": f"{row.holm_t_pvalue:.4f}",
            "Wilcoxon p (Holm)": f"{row.holm_wilcoxon_pvalue:.4f}",
        } for row in subset.itertuples()]
        significance_sections.append(_markdown(
            markdown_rows,
            ["Comparison", "mean diff", "Cohen's dz", "95% CI",
             "p (Holm)", "Wilcoxon p (Holm)"],
            f"Significance ({label}, {args.ref_attack} ε={args.ref_epsilon:g}, F1)"))

    header = ["# Manuscript tables (generated)", "",
              f"> generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}　"
              f"decision margin={settings.get('confidence_margin')}　"
              f"τ2={settings.get('min_phi2')}　τ4={settings.get('min_phi4')}",
              ""]
    document = "\n".join(header + sections + significance_sections)
    (out_dir / "paper_tables.md").write_text(document, encoding="utf-8")
    (out_dir / "tables_summary.json").write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "outputs_root": str(root.resolve()),
        "ref_attack": args.ref_attack,
        "ref_epsilon": args.ref_epsilon,
        "decision_settings": settings,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"tables written to {out_dir}")
    print(document[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
