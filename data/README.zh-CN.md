# 数据目录

[English](README.md) | **简体中文**

本仓库**不包含**数据集。请自行下载并把文件放到这里。

## UNSW-NB15（必需）

来源：<https://research.unsw.edu.au/projects/unsw-nb15-dataset>

把两个**官方**划分按代码期望的文件名放置：

```
data/train.csv    175,341 条    <- 官方 UNSW_NB15_training-set.csv
data/test.csv      82,332 条    <- 官方 UNSW_NB15_testing-set.csv
```

两个文件都必须包含代码使用的列，尤其是 `attack_cat`（多分类攻击标签）和 `label`（二分类标签）。

### 校验目录

```bash
uv run python code/prepare_data.py
```

该命令会统计每个文件的记录数并与官方划分大小比较，同时检查必需的列是否存在。

### 如果两个文件被弄反了

很容易出现这样的情况：下载了 `UNSW_NB15_training-set.csv` 和 `UNSW_NB15_testing-set.csv`，
**没有确认哪个是哪个**就保存成 `train.csv` / `test.csv`。这会悄悄变成在小划分上训练、在大划分上评测。
`prepare_data.py` 能检测出这种情况并修复：

```bash
uv run python code/prepare_data.py --fix-swap
```

交换之后，`train.csv` 有 175,341 条记录，`test.csv` 有 82,332 条——即官方方向。

## CIC-IDS2017（可选）

只有 `code/run_cicids2017.py` 和 `code/run_cicids2017_source_disjoint.py` 需要。

来源：<https://www.unb.ca/cic/datasets/ids-2017.html>

```
data/cicids2017/monday.csv
data/cicids2017/tuesday.csv
data/cicids2017/wednesday.csv
data/cicids2017/thursday.csv
data/cicids2017/friday.csv
```
