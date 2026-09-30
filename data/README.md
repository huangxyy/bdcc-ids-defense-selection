# Data

The datasets are **not** included in this repository. Download them and place the files here.

## UNSW-NB15 (required)

Source: <https://research.unsw.edu.au/projects/unsw-nb15-dataset>

Place the two **official** partitions under the names this code expects:

```
data/train.csv    175,341 records    <- official UNSW_NB15_training-set.csv
data/test.csv      82,332 records    <- official UNSW_NB15_testing-set.csv
```

Both files must contain the columns used by the code, in particular
`attack_cat` (multi-class attack label) and `label` (binary label).

### Verify the layout

```bash
uv run python code/prepare_data.py
```

This counts the records in each file and compares them against the official partition sizes.
It also checks that the required columns are present.

### If the two files are reversed

It is easy to download `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` and save them as
`train.csv` / `test.csv` **without noticing which is which**. That silently makes the experiment train on
the small partition and evaluate on the large one. `prepare_data.py` detects this and can repair it:

```bash
uv run python code/prepare_data.py --fix-swap
```

After the swap, `train.csv` holds 175,341 records and `test.csv` holds 82,332 -- the official direction.

## CIC-IDS2017 (optional)

Only needed by `code/run_cicids2017.py` and `code/run_cicids2017_source_disjoint.py`.

Source: <https://www.unb.ca/cic/datasets/ids-2017.html>

```
data/cicids2017/monday.csv
data/cicids2017/tuesday.csv
data/cicids2017/wednesday.csv
data/cicids2017/thursday.csv
data/cicids2017/friday.csv
```
