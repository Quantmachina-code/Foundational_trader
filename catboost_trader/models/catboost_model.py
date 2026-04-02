"""CatBoost ranking model with monthly retraining and exponential sample weights.

Design
------
* One ``CatBoostTrader`` instance per (horizon, feature_set) combination.
* ``fit()`` accepts the full expanding training window and a reference date;
  it computes exponentially decaying sample weights so recent observations
  contribute more to the objective.
* ``predict()`` returns a raw score Series (higher = stronger buy signal).
* The caller (``simulation/engine.py``) triggers monthly retraining by calling
  ``fit()`` at the start of each new month.
* Feature selection via ``SelectKBest(f_regression)`` caps the dimensionality
  at ``MAX_FEATURES`` without removing the imputation step.
* A ``SimpleImputer(median)`` fills NaN values; features are then clipped to
  ±100 to reduce outlier influence.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer

from catboost_trader.config import CATBOOST_PARAMS, HALF_LIFE_DAYS, MAX_FEATURES


class CatBoostTrader:
    """CatBoost regressor that predicts cross-sectional quantile rank of forward returns.

    Parameters
    ----------
    horizon:
        Forward return horizon in trading days (1, 3, or 5).
    feature_set:
        ``"original"`` or ``"alpha"`` — controls which columns are used.
    params:
        CatBoost parameters dict.  Defaults to ``CATBOOST_PARAMS`` from config.
    """

    def __init__(
        self,
        horizon: int,
        feature_set: str,
        params: dict | None = None,
    ) -> None:
        self.horizon = horizon
        self.feature_set = feature_set
        self.params = params or dict(CATBOOST_PARAMS)
        self._model: CatBoostRegressor | None = None
        self._imputer: SimpleImputer | None = None
        self._selector: SelectKBest | None = None
        self._feature_cols: list[str] = []
        self._selected_cols: list[str] = []
        self.is_fitted: bool = False
        self.last_retrain_date: pd.Timestamp | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _exponential_weights(dates: pd.Series, ref_date: pd.Timestamp) -> np.ndarray:
        """Return normalised exponential sample weights.

        ``w[t] = exp(-ln(2) / half_life * (ref_date - t).days)``
        Weights sum to 1.
        """
        decay = np.log(2) / HALF_LIFE_DAYS
        days_back = (ref_date - pd.to_datetime(dates)).dt.days.clip(lower=0).values
        weights = np.exp(-decay * days_back)
        return weights / weights.sum()

    def _preprocess(
        self,
        X: pd.DataFrame,
        fit: bool = False,
    ) -> np.ndarray:
        """Impute → clip → (optionally fit and) select features."""
        Xv = X[self._feature_cols].values.astype(np.float64)

        if fit:
            self._imputer = SimpleImputer(strategy="median")
            Xv = self._imputer.fit_transform(Xv)
        else:
            Xv = self._imputer.transform(Xv)

        Xv = np.clip(Xv, -100.0, 100.0)
        return Xv

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ref_date: pd.Timestamp,
        date_col: str = "date",
    ) -> "CatBoostTrader":
        """Train (or retrain) the model on an expanding window.

        Parameters
        ----------
        X:
            Feature DataFrame (multi-ticker panel rows) including a ``date``
            column (or the column named by *date_col*).
        y:
            Target Series aligned to X — cross-sectional rank of forward return.
        ref_date:
            The date at which training is being performed.  Used as the
            reference for exponential weight decay (most recent data → highest
            weight).
        date_col:
            Name of the date column in X used for weight computation.
        """
        # Identify feature columns
        from catboost_trader.models.feature_sets import (
            ALPHA_FACTORY_FEATURES,
            ORIGINAL_FEATURES,
        )
        from catboost_trader.config import FEATURE_SET_ORIGINAL

        all_features = (
            ORIGINAL_FEATURES if self.feature_set == FEATURE_SET_ORIGINAL
            else ALPHA_FACTORY_FEATURES
        )
        self._feature_cols = [c for c in all_features if c in X.columns]

        # Drop rows with missing target
        mask = y.notna()
        X_fit = X.loc[mask].reset_index(drop=True)
        y_fit = y.loc[mask].reset_index(drop=True)

        if len(X_fit) < 100:
            raise ValueError(
                f"Too few training samples ({len(X_fit)}) for horizon={self.horizon}."
            )

        # Sample weights
        if date_col in X_fit.columns:
            weights = self._exponential_weights(X_fit[date_col], ref_date)
        else:
            weights = np.ones(len(X_fit)) / len(X_fit)

        # Preprocess
        Xv = self._preprocess(X_fit, fit=True)
        yv = y_fit.values

        # Feature selection
        k = min(MAX_FEATURES, Xv.shape[1])
        self._selector = SelectKBest(f_regression, k=k)
        Xv_sel = self._selector.fit_transform(Xv, yv)

        mask_sel = self._selector.get_support()
        self._selected_cols = [c for c, m in zip(self._feature_cols, mask_sel) if m]

        # Train CatBoost
        self._model = CatBoostRegressor(**self.params)
        self._model.fit(Xv_sel, yv, sample_weight=weights)

        self.is_fitted = True
        self.last_retrain_date = ref_date
        print(
            f"[CatBoostTrader h={self.horizon} {self.feature_set}] "
            f"Trained on {len(X_fit):,} samples up to {ref_date.date()}  "
            f"| {len(self._selected_cols)} features selected"
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict cross-sectional score for ranking.

        Parameters
        ----------
        X:
            Feature DataFrame for one date (or multiple dates) with the same
            columns used during training.

        Returns
        -------
        pd.Series
            Raw predicted scores indexed like X.  Higher = stronger buy signal.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        Xv = self._preprocess(X, fit=False)
        Xv_sel = self._selector.transform(Xv)
        preds = self._model.predict(Xv_sel)
        return pd.Series(preds, index=X.index, name="score")

    def needs_retrain(self, current_date: pd.Timestamp) -> bool:
        """True if no training has happened or a new calendar month has begun."""
        if not self.is_fitted or self.last_retrain_date is None:
            return True
        return current_date.month != self.last_retrain_date.month

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the entire trainer (model + preprocessors + metadata)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[CatBoostTrader] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "CatBoostTrader":
        """Load a pickled trainer from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected CatBoostTrader, got {type(obj)}")
        return obj

    def __repr__(self) -> str:
        status = f"fitted up to {self.last_retrain_date.date()}" if self.is_fitted else "not fitted"
        return f"CatBoostTrader(horizon={self.horizon}, feature_set={self.feature_set!r}, {status})"
