"""Self-contained, notebook-friendly TabICL runner.

All code needed to run TabICL from prepared parquet data lives in this file.
"""

from __future__ import annotations

import glob
import importlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)

TRAIN_YEARS = 6
TEST_YEARS = 4

CONFIG = {
    "data_path": "panel_data_weekly.parquet",
    "target_col": "target_1",  # target_1 or target_2
    "output_path": "outputs/tabicl_predictions.csv",
    "train_years": TRAIN_YEARS,
    "test_years": TEST_YEARS,
}


def _find_ckpt(patterns: list[str]) -> str | None:
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def _load_tabicl_classifier() -> callable:
    ckpt = _find_ckpt(["tabicl*.ckpt", "TabICL*.ckpt"])
    if ckpt is None:
        raise RuntimeError("TabICL checkpoint not found (expected tabicl*.ckpt in repo root).")

    module = None
    for mod in ["tabicl", "tabicl.classifier", "tabicl.model"]:
        try:
            m = importlib.import_module(mod)
            if hasattr(m, "TabICLClassifier"):
                module = m
                break
        except ImportError:
            continue

    if module is None:
        raise RuntimeError("tabicl package is not importable.")

    cls = module.TabICLClassifier
    sig = inspect.signature(cls.__init__).parameters
    if "ckpt_path" in sig:
        def _factory(**kwargs):
            kwargs.setdefault("ckpt_path", ckpt)
            return cls(**kwargs)
    elif "model_path" in sig:
        def _factory(**kwargs):
            kwargs.setdefault("model_path", ckpt)
            return cls(**kwargs)
    elif "checkpoint" in sig:
        def _factory(**kwargs):
            kwargs.setdefault("checkpoint", ckpt)
            return cls(**kwargs)
    else:
        def _factory(**kwargs):
            return cls(ckpt, **kwargs)

    print(f"Using TabICL checkpoint: {ckpt}")
    return _factory


def temporal_split(panel: pd.DataFrame, train_years: int, test_years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = panel.sort_values("date")
    min_date = panel["date"].min()
    train_end = min_date + pd.DateOffset(years=train_years)
    test_end = train_end + pd.DateOffset(years=test_years)

    train = panel[panel["date"] < train_end]
    test = panel[(panel["date"] >= train_end) & (panel["date"] <= test_end)]
    return train, test


def get_feature_cols(panel: pd.DataFrame) -> list[str]:
    exclude = {
        "date", "ticker", "week", "Open", "High", "Low", "Close", "Volume",
        "Adj Close", "target_1", "target_2", "target_quantile", "fwd_open_to_open",
    }
    numeric = {np.float64, np.float32, np.int64, np.int32}
    return [c for c in panel.columns if c not in exclude and panel[c].dtype in numeric]


def prepare_arrays(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    X_train = np.clip(imputer.fit_transform(train[feature_cols]), -100, 100)
    X_test = np.clip(imputer.transform(test[feature_cols]), -100, 100)
    y_train = train[target_col].values.astype(int)
    y_test = test[target_col].values.astype(int)
    idx = test.index.to_numpy()
    return X_train, y_train, X_test, y_test, idx


def _evaluate(model, X_test: np.ndarray, y_test: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray, dict]:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "brier": brier_score_loss(y_test, proba[:, 1]),
        "log_loss": log_loss(y_test, proba),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, proba[:, 1])
    except Exception:
        metrics["roc_auc"] = np.nan

    print(f"\n[{label}]")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, (float, np.floating)) else f"  {k}: {v}")
    print(classification_report(y_test, preds, digits=3))
    return preds, proba[:, 1], metrics


def run_tabicl_from_parquet(
    data_path: str = CONFIG["data_path"],
    target_col: str = CONFIG["target_col"],
    output_path: str = CONFIG["output_path"],
    train_years: int = CONFIG["train_years"],
    test_years: int = CONFIG["test_years"],
) -> tuple[pd.DataFrame, dict]:
    if target_col not in {"target_1", "target_2"}:
        raise ValueError("target_col must be one of: 'target_1', 'target_2'.")

    panel = pd.read_parquet(data_path)
    panel["date"] = pd.to_datetime(panel["date"])

    train_df, test_df = temporal_split(panel, train_years=train_years, test_years=test_years)
    feature_cols = get_feature_cols(panel)
    X_train, y_train, X_test, y_test, test_index = prepare_arrays(train_df, test_df, feature_cols, target_col)

    tabicl_factory = _load_tabicl_classifier()
    model = tabicl_factory()
    model.fit(X_train, y_train)

    preds, proba_1, metrics = _evaluate(model, X_test, y_test, label=f"TabICL-{target_col}")

    out_df = test_df.loc[test_index, ["date", "ticker", target_col]].copy()
    out_df["y_pred"] = preds
    out_df["y_prob"] = proba_1

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output, index=False)
    print(f"Saved predictions to {output} ({len(out_df)} rows)")

    return out_df, metrics


if __name__ == "__main__":
    run_tabicl_from_parquet(**CONFIG)
