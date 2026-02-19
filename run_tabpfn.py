"""Run TabPFN on prebuilt panel parquet data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from main_weekly import (
    HAS_TABPFN,
    TRAIN_YEARS,
    TEST_YEARS,
    get_feature_cols,
    prepare_arrays,
    run_tabpfn,
    temporal_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TabPFN using prepared parquet panel.")
    parser.add_argument("--data", default="panel_data_weekly.parquet", help="Input panel parquet path.")
    parser.add_argument("--target", default="target_1", choices=["target_1", "target_2"], help="Binary target column.")
    parser.add_argument("--output", default="outputs/tabpfn_predictions.csv", help="Prediction output csv path.")
    parser.add_argument("--train-years", type=int, default=TRAIN_YEARS, help="Train years.")
    parser.add_argument("--test-years", type=int, default=TEST_YEARS, help="Test years.")
    args = parser.parse_args()

    if not HAS_TABPFN:
        raise RuntimeError("TabPFN is not available. Install tabpfn and place tabpfn*.ckpt in repo root.")

    panel = pd.read_parquet(args.data)
    panel["date"] = pd.to_datetime(panel["date"])

    train_df, test_df = temporal_split(panel, train_years=args.train_years, test_years=args.test_years)
    feature_cols = get_feature_cols(panel)

    X_train, y_train, X_test, y_test, test_index = prepare_arrays(
        train_df,
        test_df,
        feature_cols,
        target_col=args.target,
    )

    preds, proba, metrics = run_tabpfn(
        X_train,
        y_train,
        X_test,
        y_test,
        label=f"TabPFN-{args.target}",
    )

    out_df = test_df.iloc[test_index][["date", "ticker", args.target]].copy()
    out_df["y_pred"] = preds
    if proba is not None:
        out_df["y_prob"] = proba

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"Saved predictions to {output_path} ({len(out_df)} rows)")
    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
