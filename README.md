# Foundational_trader
Use foundational models to estimate weekly portfolio.

## Split workflow (download once, train many times)

### 1) Download and prepare data to parquet
```bash
python data_prepare.py --output panel_data_weekly.parquet
```
This downloads weekly data, computes engineered features, and writes a reusable parquet panel.

### 2) Run TabPFN on prepared parquet
```bash
python run_tabpfn.py --data panel_data_weekly.parquet --target target_1
```

### 3) Run TabICL on prepared parquet
```bash
python run_tabicl.py --data panel_data_weekly.parquet --target target_1
```

Each model script reads the parquet directly, so data is not downloaded again.
