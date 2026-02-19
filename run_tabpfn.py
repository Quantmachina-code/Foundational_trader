"""Self-contained, notebook-friendly TabPFN runner.

All code needed to run TabPFN from prepared parquet data lives in this file.
"""

from __future__ import annotations

import glob
import inspect
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
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
    "output_path": "outputs/tabpfn_predictions.csv",
    "train_years": TRAIN_YEARS,
    "test_years": TEST_YEARS,
    "max_train": 3000,
    "max_features": 100,
    "random_seed": 42,
}


def _find_ckpt(patterns: list[str]) -> str | None:
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def _load_tabpfn_classifier() -> callable:
    ckpt = _find_ckpt(["tabpfn*.ckpt", "TabPFN*.ckpt"])
    if ckpt is None:
        raise RuntimeError("TabPFN checkpoint not found (expected tabpfn*.ckpt in repo root).")

    os.environ["TABPFN_MODEL_PATH"] = ckpt
    from tabpfn import TabPFNClassifier as _TabPFNClassifier

    sig = inspect.signature(_TabPFNClassifier.__init__).parameters
    if "model_path" in sig:
        def _factory(**kwargs):
            kwargs.setdefault("model_path", ckpt)
            kwargs.setdefault("device", "cpu")
            return _TabPFNClassifier(**kwargs)
    else:
        def _factory(**kwargs):
            kwargs.setdefault("device", "cpu")
            return _TabPFNClassifier(**kwargs)

    print(f"Using TabPFN checkpoint: {ckpt}")
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
    numeric_cols = panel.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


def prepare_arrays(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not feature_cols:
        raise ValueError(
            "No numeric feature columns found. Ensure parquet contains engineered numeric features."
        )

    train_features = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    test_features = test[feature_cols].apply(pd.to_numeric, errors="coerce")

    imputer = SimpleImputer(strategy="median")
    X_train = np.clip(imputer.fit_transform(train_features), -100, 100)
    X_test = np.clip(imputer.transform(test_features), -100, 100)
    y_train = train[target_col].values.astype(int)
    y_test = test[target_col].values.astype(int)
    idx = test.index.to_numpy()
    return X_train, y_train, X_test, y_test, idx


def _select_features(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    selector = SelectKBest(f_classif, k=min(k, X_tr.shape[1]))
    return selector.fit_transform(X_tr, y_tr), selector.transform(X_te)


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


def run_tabpfn_from_parquet(
    data_path: str = CONFIG["data_path"],
    target_col: str = CONFIG["target_col"],
    output_path: str = CONFIG["output_path"],
    train_years: int = CONFIG["train_years"],
    test_years: int = CONFIG["test_years"],
    max_train: int = CONFIG["max_train"],
    max_features: int = CONFIG["max_features"],
    random_seed: int = CONFIG["random_seed"],
) -> tuple[pd.DataFrame, dict]:
    if target_col not in {"target_1", "target_2"}:
        raise ValueError("target_col must be one of: 'target_1', 'target_2'.")

    panel = pd.read_parquet(data_path)
    panel["date"] = pd.to_datetime(panel["date"])

    train_df, test_df = temporal_split(panel, train_years=train_years, test_years=test_years)
    if train_df.empty or test_df.empty:
        raise ValueError(
            "Temporal split produced empty train/test set. Check date range and split years."
        )

    feature_cols = get_feature_cols(panel)
    X_train, y_train, X_test, y_test, test_index = prepare_arrays(train_df, test_df, feature_cols, target_col)

    if X_train.shape[1] > max_features:
        X_train, X_test = _select_features(X_train, y_train, X_test, k=max_features)

    rng = np.random.default_rng(random_seed)
    if len(X_train) > max_train:
        idx = rng.choice(len(X_train), max_train, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    tabpfn_factory = _load_tabpfn_classifier()
    model = tabpfn_factory(N_ensemble_configurations=16)
    model.fit(X_train, y_train)

    preds, proba_1, metrics = _evaluate(model, X_test, y_test, label=f"TabPFN-{target_col}")

    out_df = test_df.loc[test_index, ["date", "ticker", target_col]].copy()
    out_df["y_pred"] = preds
    out_df["y_prob"] = proba_1

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output, index=False)
    print(f"Saved predictions to {output} ({len(out_df)} rows)")

    return out_df, metrics


if __name__ == "__main__":
    run_tabpfn_from_parquet(**CONFIG)
