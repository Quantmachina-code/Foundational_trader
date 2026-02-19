"""Notebook-friendly TabICL runner using prepared parquet panel data.

Usage patterns:
1) Jupyter notebook:
    from run_tabicl import run_tabicl_from_parquet
    preds_df, metrics = run_tabicl_from_parquet(target_col="target_1")

2) Script execution:
    python run_tabicl.py
    # edit the CONFIG block below for quick parameter changes
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from main_weekly import (
    HAS_TABICL,
    TEST_YEARS,
    TRAIN_YEARS,
    get_feature_cols,
    prepare_arrays,
    run_tabicl,
    temporal_split,
)

# ── Notebook-style config block (edit directly in notebook/script) ────────────
CONFIG = {
    "data_path": "panel_data_weekly.parquet",
    "target_col": "target_1",  # target_1 or target_2
    "output_path": "outputs/tabicl_predictions.csv",
    "train_years": TRAIN_YEARS,
    "test_years": TEST_YEARS,
}


def run_tabicl_from_parquet(
    data_path: str = CONFIG["data_path"],
    target_col: str = CONFIG["target_col"],
    output_path: str = CONFIG["output_path"],
    train_years: int = CONFIG["train_years"],
    test_years: int = CONFIG["test_years"],
) -> tuple[pd.DataFrame, dict]:
    """Load parquet panel, run TabICL, save predictions, and return results."""
    if target_col not in {"target_1", "target_2"}:
        raise ValueError("target_col must be one of: 'target_1', 'target_2'.")

    if not HAS_TABICL:
        raise RuntimeError(
            "TabICL is not available. Install tabicl and place tabicl*.ckpt in repo root."
        )

    panel = pd.read_parquet(data_path)
    panel["date"] = pd.to_datetime(panel["date"])

    train_df, test_df = temporal_split(
        panel,
        train_years=train_years,
        test_years=test_years,
    )
    feature_cols = get_feature_cols(panel)

    X_train, y_train, X_test, y_test, test_index = prepare_arrays(
        train_df,
        test_df,
        feature_cols,
        target_col=target_col,
    )

    preds, proba, metrics = run_tabicl(
        X_train,
        y_train,
        X_test,
        y_test,
        label=f"TabICL-{target_col}",
    )

    out_df = test_df.iloc[test_index][["date", "ticker", target_col]].copy()
    out_df["y_pred"] = preds
    if proba is not None:
        out_df["y_prob"] = proba

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output, index=False)

    print(f"Saved predictions to {output} ({len(out_df)} rows)")
    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    return out_df, metrics


if __name__ == "__main__":
    run_tabicl_from_parquet(**CONFIG)
