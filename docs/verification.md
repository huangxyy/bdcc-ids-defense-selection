# 验证指南

验证分十一层,从秒级到小时级逐层加深。提交前至少跑完第 1–6 层(即 `make verify`),
正式实验跑完后加跑第 8 层。

## 0. 环境

```bash
uv sync --group dev      # 首次执行;国内服务器可用 scripts/setup_server.sh
```

## 1. 一条命令

```bash
make help                # 列出所有目标
make verify              # 全量:数据 + lint + 测试 + 冒烟 + dry-run
make quick               # 快检:数据 + lint + 测试(约半分钟)
```

等价于:

```bash
uv run python scripts/verify.py
uv run python scripts/verify.py --quick
```

## 2. 分层验证

| 层级 | 命令 | 检查内容 | 耗时 |
|---|---|---|---|
| 1 结构 | `uv run pytest tests/test_structure.py` | `src/`/`scripts/` 布局、脚本引导、任意目录可运行 | 秒 |
| 2 数据 | `uv run python scripts/prepare_data.py` | 官方划分方向、列、行数;`--fix-swap` 可修复反向 | 秒 |
| 3 静态 | `uv run ruff check` | 未定义名称、语法、导入错误 | 秒 |
| 4 单元/集成 | `uv run pytest -q` | 合成数据上的端到端流程(CPU,121 个测试) | 约 20 秒 |
| 5 冒烟 | `uv run python scripts/smoke_test.py` | 真实数据加载、三个骨干前向、PGD 约束、自适应攻击 | 约 1 分钟 |
| 6 复现计划 | `uv run python scripts/run_experiments.py --dry-run` | 三个骨干的命令、路径、设备参数 | 秒 |
| 7 真实运行 | `uv run python scripts/run_experiments.py --device cuda` | 表 3–5 的原始数据 | 数小时 |
| 8 产物校验 | `uv run python scripts/check_outputs.py --require-analysis` | 必需文件/列、官方划分、种子数、离散度 | 秒 |
| 9 实验留痕 | `uv run python scripts/log_experiment.py --outputs-dir outputs/<run> --record-dir <dir> --title "..."` | 生成 markdown 记录(数值自动来自 CSV)并登记 INDEX.md | 秒 |
| 10 权重复用 | `uv run python scripts/evaluate_checkpoints.py --checkpoint outputs/<run>/checkpoints/seed7 --output-dir outputs/<eval>` | 加载已训练权重做评测变体(ε/步长/自适应),不重训 | 分钟 |
| 11 统计与有效解 | `uv run python scripts/enhance_significance.py`、`uv run python scripts/check_supportedness.py` | Holm/BH 校正、效应量、bootstrap CI;supported/unsupported 有效解判定 | 秒 |

## 3. 论文内容 → 命令 → 产物

| 论文内容 | 命令 | 产物 |
|---|---|---|
| 表 3–5 四维画像 | `scripts/run_mlp.py` / `run_cnn1d.py` / `run_ft_transformer.py` | `outputs/<backbone>/{raw,mean,std}_results.csv`、`efficiency_*.csv`、`category_*.csv` |
| 表 6–7 帕累托与偏好选择 | `scripts/pareto_selection.py` | `risk_profile_4d.csv`、`decision_comparators.csv`、`candidate_dependence.csv`、`theta_sweep.csv`、`theta_summary.csv`、`switching_regions.json`、`admissibility_sweep.csv`、`pareto_selection_results.csv`、`decision_settings.json`、`outputs/figures/*` |
| phi4 最差类别召回 | `scripts/evaluate_phi4_cnn.py`、`evaluate_phi4_ft.py` | `outputs/phi4_*/category_*.csv` |
| 攻击泛化 | 主运行内置 | `attack_generalization.csv` |
| 自适应攻击 | `run_* --adaptive-eval` | `adaptive_attack_{raw,mean}.csv` |
| epsilon 扫描 / ROC / 梯度遮蔽 | `scripts/analyze_extended.py` | `outputs/extended_analysis/*` |
| 跨骨干迁移 | `scripts/evaluate_cross_backbone_transfer.py` | `outputs/transfer_matrix/*` |
| 六种防御超参数 | 主运行内置 | `hyperparameters.csv` |
| supported/unsupported 有效解 | `scripts/check_supportedness.py --tchebycheff-rho 0.1` | `<backbone>/supportedness.csv` |
| Holm/BH 校正与效应量 | `scripts/enhance_significance.py` | `<backbone>/significance_enhanced.csv` |
| bootstrap 前沿稳定性 | `scripts/bootstrap_dominance.py` | `outputs/tables/bootstrap_*.csv` |
| 均值 vs 中位数 | `scripts/check_aggregation_robustness.py` | `outputs/tables/aggregation_robustness.csv` |
| epsilon 敏感性 | `scripts/export_epsilon_sensitivity.py --attack pgd` | `outputs/tables/epsilon_sensitivity*` |
| 论文表 3–7 导出 | `scripts/export_paper_tables.py` | `outputs/tables/paper_tables.md` + CSV |

修订版主实验协议(官方划分 + 10 seeds + 50 步评测 PGD,三个骨干同参):

```bash
uv run python scripts/run_mlp.py --device cuda --save-checkpoints \
  --seeds 7,13,21,42,100,11,23,37,59,89 --eval-attack-rows 20000 \
  --epsilon-list 0.02,0.05,0.10,0.20 --eval-pgd-alpha-ratio 0.10 --eval-pgd-steps 50
# run_cnn1d.py / run_ft_transformer.py 使用同样的参数
uv run python scripts/pareto_selection.py --ref-attack pgd --ref-epsilon 0.10 \
  --confidence-margin 1.0 --min-phi2 0.90 --min-phi4 0.60
```

修订版论文表格的数字快照与新旧对照见 [docs/results.md](results.md)。

## 4. 修订版验收标准

- `make verify` 全绿;
- 主运行 `run_summary.json` 中 `dataset_split.official_direction = true`;
- `raw_results.csv` 至少有 5 个 seed(修订版为 10 个),`std_results.csv` 非零(确实做了跨种子聚合);
- `check_outputs.py --require-analysis` 通过;
- 论文表格中的每个数字都能在上述 CSV 中找到来源,不手工改写。

## 5. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `ModuleNotFoundError: ids_defense_selection` | 不要用 `python -m scripts.xxx`;按文档用 `uv run python scripts/xxx.py`,脚本会自行引导 `src/` |
| `uv sync` 下载慢 | 用 `scripts/setup_server.sh`(默认清华镜像)或设置 `UV_DEFAULT_INDEX` |
| matplotlib 提示 cache 目录不可写 | 无害;可设置 `MPLCONFIGDIR=/tmp/matplotlib` |
| `check_outputs` 报缺少文件 | 先跑对应主实验;`--quick` 的验证不产生产物 |
| 划分方向告警 | `uv run python scripts/prepare_data.py --fix-swap`,然后重跑全部实验 |
| FT-Transformer 缺少决策产物 | 确认 `outputs/ft_transformer/` 已由 `run_ft_transformer.py` 生成;缺失的骨干会被决策脚本跳过,`--require-analysis` 会报出缺哪个文件 |
