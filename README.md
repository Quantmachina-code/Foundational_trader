# Foundational_trader
Use foundational models to estimate weekly portfolio.

## Split workflow (download once, train many times)

### 1) Download and prepare data to parquet
```bash
python data_prepare.py --output panel_data_weekly.parquet
```
This downloads weekly data, computes engineered features, and writes a reusable parquet panel.

### 2) Run TabPFN on prepared parquet
Script mode:
```bash
python run_tabpfn.py
```
Notebook mode:
```python
from run_tabpfn import run_tabpfn_from_parquet
preds_df, metrics = run_tabpfn_from_parquet(
    data_path="panel_data_weekly.parquet",
    target_col="target_1",
)
```

### 3) Run TabICL on prepared parquet
Script mode:
```bash
python run_tabicl.py
```
Notebook mode:
```python
from run_tabicl import run_tabicl_from_parquet
preds_df, metrics = run_tabicl_from_parquet(
    data_path="panel_data_weekly.parquet",
    target_col="target_1",
)
```

Each model script reads parquet directly, so data is not downloaded again.
