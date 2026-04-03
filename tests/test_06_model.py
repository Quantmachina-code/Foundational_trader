"""Test 06 — CatBoostTrader model.

Tests fit, predict, save/load, and needs_retrain on synthetic data.
Uses a minimal CatBoost config (50 iterations) so the test runs fast.

Run:
    pytest tests/test_06_model.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from catboost_trader.models.catboost_model import CatBoostTrader
from catboost_trader.data.features import build_feature_matrix
from catboost_trader.data.targets import build_targets
from catboost_trader.config import FEATURE_SET_ORIGINAL, FEATURE_SET_ALPHA


# Fast CatBoost params for tests
_FAST_PARAMS = dict(
    iterations=50,
    learning_rate=0.1,
    depth=4,
    verbose=0,
    random_seed=42,
    loss_function="RMSE",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def training_data(synthetic_panel):
    """Pre-built feature panel + targets for the ORIGINAL feature set."""
    panel = build_feature_matrix(synthetic_panel, feature_set=FEATURE_SET_ORIGINAL)
    panel = build_targets(panel, horizon=1)
    panel = panel.dropna(subset=["target"]).reset_index(drop=True)
    return panel


@pytest.fixture(scope="module")
def fitted_model(training_data):
    """A single fitted CatBoostTrader (ORIGINAL, h=1)."""
    model = CatBoostTrader(horizon=1, feature_set=FEATURE_SET_ORIGINAL, params=_FAST_PARAMS)
    ref_date = training_data["date"].max()
    model.fit(training_data, training_data["target"], ref_date=ref_date)
    return model


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_not_fitted_at_init(self):
        m = CatBoostTrader(horizon=1, feature_set=FEATURE_SET_ORIGINAL, params=_FAST_PARAMS)
        assert not m.is_fitted

    def test_repr_not_fitted(self):
        m = CatBoostTrader(horizon=3, feature_set=FEATURE_SET_ALPHA, params=_FAST_PARAMS)
        assert "not fitted" in repr(m)

    def test_predict_before_fit_raises(self):
        m = CatBoostTrader(horizon=1, feature_set=FEATURE_SET_ORIGINAL, params=_FAST_PARAMS)
        with pytest.raises(RuntimeError, match="fitted"):
            m.predict(pd.DataFrame({"d_rsi_14": [0.5]}))


# ---------------------------------------------------------------------------
# Exponential sample weights
# ---------------------------------------------------------------------------

class TestExponentialWeights:
    def test_recent_dates_have_higher_weight(self):
        dates = pd.Series(pd.date_range("2015-01-01", "2017-12-31", freq="D"))
        ref = pd.Timestamp("2017-12-31")
        weights = CatBoostTrader._exponential_weights(dates, ref)
        # The last weight should be the highest
        assert weights[-1] == weights.max()

    def test_weights_sum_to_one(self):
        dates = pd.Series(pd.date_range("2015-01-01", "2017-12-31", freq="D"))
        ref = pd.Timestamp("2017-12-31")
        weights = CatBoostTrader._exponential_weights(dates, ref)
        assert abs(weights.sum() - 1.0) < 1e-9

    def test_weights_all_positive(self):
        dates = pd.Series(pd.date_range("2015-01-01", "2017-12-31", freq="D"))
        ref = pd.Timestamp("2017-12-31")
        weights = CatBoostTrader._exponential_weights(dates, ref)
        assert (weights > 0).all()


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

class TestFit:
    def test_is_fitted_after_fit(self, fitted_model):
        assert fitted_model.is_fitted

    def test_last_retrain_date_set(self, fitted_model, training_data):
        assert fitted_model.last_retrain_date == training_data["date"].max()

    def test_selected_cols_non_empty(self, fitted_model):
        assert len(fitted_model._selected_cols) > 0

    def test_too_few_samples_raises(self):
        small = pd.DataFrame({
            "date": pd.bdate_range("2020-01-01", periods=5),
            "ticker": "A",
            "d_rsi_14": np.random.rand(5),
            "target": np.random.rand(5),
        })
        m = CatBoostTrader(horizon=1, feature_set=FEATURE_SET_ORIGINAL, params=_FAST_PARAMS)
        with pytest.raises(ValueError, match="Too few"):
            m.fit(small, small["target"], ref_date=pd.Timestamp("2020-01-10"))

    def test_fit_returns_self(self, training_data):
        m = CatBoostTrader(horizon=1, feature_set=FEATURE_SET_ORIGINAL, params=_FAST_PARAMS)
        result = m.fit(training_data, training_data["target"],
                       ref_date=training_data["date"].max())
        assert result is m


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_returns_series(self, fitted_model, training_data):
        X = training_data.iloc[:20]
        preds = fitted_model.predict(X)
        assert isinstance(preds, pd.Series)

    def test_predict_length_matches_input(self, fitted_model, training_data):
        X = training_data.iloc[:20]
        preds = fitted_model.predict(X)
        assert len(preds) == 20

    def test_predict_no_nans(self, fitted_model, training_data):
        preds = fitted_model.predict(training_data.iloc[:50])
        assert preds.notna().all()

    def test_predict_finite(self, fitted_model, training_data):
        preds = fitted_model.predict(training_data.iloc[:50])
        assert np.isfinite(preds.values).all()


# ---------------------------------------------------------------------------
# needs_retrain
# ---------------------------------------------------------------------------

class TestNeedsRetrain:
    def test_needs_retrain_when_not_fitted(self):
        m = CatBoostTrader(horizon=1, feature_set=FEATURE_SET_ORIGINAL, params=_FAST_PARAMS)
        assert m.needs_retrain(pd.Timestamp("2020-01-01"))

    def test_no_retrain_same_month(self, fitted_model):
        same_month = fitted_model.last_retrain_date + pd.Timedelta(days=5)
        # Stay within the same month
        if same_month.month != fitted_model.last_retrain_date.month:
            pytest.skip("Dates crossed month boundary — skip")
        assert not fitted_model.needs_retrain(same_month)

    def test_needs_retrain_next_month(self, fitted_model):
        next_month = fitted_model.last_retrain_date + pd.DateOffset(months=1)
        assert fitted_model.needs_retrain(next_month)


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_and_load(self, fitted_model, training_data, tmp_path):
        save_path = tmp_path / "model.pkl"
        fitted_model.save(save_path)
        assert save_path.exists()

        loaded = CatBoostTrader.load(save_path)
        assert loaded.is_fitted
        assert loaded.horizon == fitted_model.horizon
        assert loaded.feature_set == fitted_model.feature_set

        # Predictions should match
        X = training_data.iloc[:10]
        p1 = fitted_model.predict(X).values
        p2 = loaded.predict(X).values
        np.testing.assert_allclose(p1, p2, rtol=1e-6)

    def test_load_wrong_type_raises(self, tmp_path):
        import pickle
        bad_path = tmp_path / "bad.pkl"
        with open(bad_path, "wb") as f:
            pickle.dump({"not": "a model"}, f)
        with pytest.raises(TypeError):
            CatBoostTrader.load(bad_path)
