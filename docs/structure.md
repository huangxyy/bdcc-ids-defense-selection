# 仓库结构

仓库按"库 / 入口 / 测试 / 文档 / 数据 / 产物"六层组织。只有三条约定:

1. **可导入、可测试的逻辑只放 `src/ids_defense_selection/`。**
   指标、攻击、防御、帕累托过滤、偏好加权、统计检验都属于这里。
   典型例子:`scripts/pareto_selection.py` 只是 CLI 壳,真正的实现是
   `src/ids_defense_selection/selection.py`,因此可以直接被 pytest 导入。
2. **`scripts/` 只放入口。**
   允许参数解析、日志打印、串联调用;任何需要被测试或被复用的算法都必须下沉到
   `src/`,再从脚本里调用。
3. **路径统一由 `src/ids_defense_selection/paths.py` 解析。**
   脚本通过 `scripts/_bootstrap.py` 把 `src/` 加入 `sys.path`,所以
   `uv run python scripts/<name>.py` 可以在任意工作目录运行,也不需要安装项目。

## 目录

```
src/ids_defense_selection/     库:配置、数据、攻击、防御、评测、决策、报告
scripts/                       入口:_bootstrap + 实验/分析/工具脚本
tests/                         单元、集成、结构测试
docs/                          结构说明、验证指南与修订版结果快照(docs/results.md)
data/                          UNSW-NB15 官方划分(随仓库提供)
outputs/                       实验产物(不入版本库,规范见 outputs/README.md)
Makefile                       make verify / quick / test / lint / smoke / dry-run
```

## 各目录职责

| 路径 | 放什么 | 不放什么 |
|---|---|---|
| `src/ids_defense_selection/` | 可导入的算法与流程;每个模块都能被 `import` | CLI 参数解析、`sys.exit`、实验目录硬编码 |
| `scripts/` | 薄入口:参数解析、打印、调用库、写 `outputs/` | 新的算法逻辑(应先加进 `src/` 并写测试) |
| `tests/` | 单元/集成/结构回归;使用合成数据,CPU 数秒跑完 | 依赖真实训练时长或 GPU 的测试 |
| `docs/` | 结构、验证、实验产物映射、修订版结果快照 | 会随代码漂移的重复文档(定稿后同步 README) |
| `data/` | 官方划分的 `train.csv` / `test.csv`;CIC-IDS2017 单独下载 | 中间缓存、实验产物 |
| `outputs/` | 训练、评测、决策、图表产物 | 手工编辑的源文件;除 `README.md` 外均被 gitignore |

## 运行机制

- **不构建包**:`pyproject.toml` 里 `[tool.uv] package = false`,`uv sync` 只安装依赖。
- **脚本导入**:每个脚本顶部 `import _bootstrap`,由它把 `src/` 放到 `sys.path` 最前。
  新增脚本必须包含这一行,`tests/test_structure.py` 会检查。
- **测试导入**:pytest 通过 `[tool.pytest.ini_options] pythonpath = ["src"]` 找到库。
- **路径解析**:脚本的默认数据/输出路径都来自 `paths.py`;自定义路径经 `resolve_path()`
  相对仓库根解析,因此从任何目录启动结果一致。

## 新增文件放哪里

| 场景 | 位置 | 是否补测试 |
|---|---|---|
| 新指标 / 新判定 / 新聚合 | `src/ids_defense_selection/` | 是,`tests/` |
| 新实验入口(训练、评测) | `scripts/run_*.py` 或 `scripts/evaluate_*.py` | 至少保证 `--help` 与 `--dry-run` |
| 新分析/图表 | 逻辑进 `src/`,画图与 CLI 进 `scripts/` | 逻辑要测,图表不测 |
| 一次性工具 | `scripts/`(名字自解释) | 视情况 |
| 数据/产物说明 | `data/README.md`、`outputs/README.md` | 否 |
