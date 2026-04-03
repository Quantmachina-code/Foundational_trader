"""Test 04 — Feature engineering.

Tests build_feature_matrix() and get_feature_columns() for both the
ORIGINAL and ALPHA feature sets using the synthetic panel fixture.

Run:
    pytest tests/test_04_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from catboost_trader.data.features import build_feature_matrix, get_feature_columns
from catboost_trader.config import FEATURE_SET_ORIGINAL, FEATURE_SET_ALPHA
from catboost_trader.models.feature_sets import ORIGINAL_FEATURES, ALPHA_FACTORY_FEATURES


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _trim_panel(panel: pd.DataFrame, n_days: int = 300) -> pd.DataFrame:
    """Use only the last n_days to keep tests fast."""
    dates = panel["date"].sort_values().unique()[-n_days:]
    return panel[panel["date"].isin(dates)].copy()


# ---------------------------------------------------------------------------
# ORIGINAL feature set
# ---------------------------------------------------------------------------

class TestOriginalFeatures:
    @pytest.fixture(scope="class")
    def feat_panel(self, synthetic_panel):
        return build_feature_matrix(_trim_panel(synthetic_panel), feature_set=FEATURE_SET_ORIGINAL)

    def test_returns_dataframe(self, feat_panel):
        assert isinstance(feat_panel, pd.DataFrame)

    def test_preserves_ticker_and_date(self, feat_panel):
        assert "date" in feat_panel.columns
        assert "ticker" in feat_panel.columns

    def test_original_feature_columns_present(self, feat_panel):
        present = [c for c in ORIGINAL_FEATURES if c in feat_panel.columns]
        assert len(present) > 0, "No original d_* features found"

    def test_no_af_columns_in_original(self, feat_panel):
        af_cols = [c for c in feat_panel.columns if c.startswith("af_")]
        assert len(af_cols) == 0, f"Alpha columns leaked into original set: {af_cols}"

    def test_rank_scaled_to_unit_interval(self, feat_panel):
        """Cross-sectionally ranked features should be in [0, 1]."""
        feat_cols = [c for c in ORIGINAL_FEATURES if c in feat_panel.columns]
        for col in feat_cols[:5]:   # spot-check first 5
            valid = feat_panel[col].dropna()
            assert (valid >= 0.0).all(), f"{col} below 0 after rank-scaling"
            assert (valid <= 1.0).all(), f"{col} above 1 after rank-scaling"

    def test_ohlcv_columns_not_rank_scaled(self, feat_panel):
        """Raw price/volume columns must NOT be rank-scaled (they should stay > 1)."""
        assert feat_panel["close"].max() > 1.0

    def test_row_count_unchanged(self, synthetic_panel, feat_panel):
        trimmed = _trim_panel(synthetic_panel)
        assert len(feat_panel) == len(trimmed)


# ---------------------------------------------------------------------------
# ALPHA feature set
# ---------------------------------------------------------------------------

class TestAlphaFeatures:
    @pytest.fixture(scope="class")
    def feat_panel(self, synthetic_panel):
        return build_feature_matrix(_trim_panel(synthetic_panel), feature_set=FEATURE_SET_ALPHA)

    def test_af_columns_present(self, feat_panel):
        af_cols = [c for c in feat_panel.columns if c.startswith("af_")]
        assert len(af_cols) > 20, f"Expected >20 af_ columns, got {len(af_cols)}"

    def test_original_columns_still_present(self, feat_panel):
        d_cols = [c for c in ORIGINAL_FEATURES if c in feat_panel.columns]
        assert len(d_cols) > 0

    def test_alpha_rank_scaled(self, feat_panel):
        af_cols = [c for c in ALPHA_FACTORY_FEATURES if c in feat_panel.columns]
        for col in af_cols[:5]:
            valid = feat_panel[col].dropna()
            assert (valid >= 0.0).all(), f"{col} below 0"
            assert (valid <= 1.0).all(), f"{col} above 1"

    def test_more_columns_than_original(self, synthetic_panel):
        trimmed = _trim_panel(synthetic_panel)
        orig  = build_feature_matrix(trimmed, feature_set=FEATURE_SET_ORIGINAL)
        alpha = build_feature_matrix(trimmed, feature_set=FEATURE_SET_ALPHA)
        assert alpha.shape[1] > orig.shape[1]


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------

class TestFeatureErrors:
    def test_invalid_feature_set_raises(self, synthetic_panel):
        with pytest.raises(ValueError, match="feature_set"):
            build_feature_matrix(synthetic_panel, feature_set="bogus")


# ---------------------------------------------------------------------------
# get_feature_columns
# ---------------------------------------------------------------------------

class TestGetFeatureColumns:
    def test_original_returns_d_cols(self, synthetic_panel):
        trimmed = _trim_panel(synthetic_panel)
        feat_panel = build_feature_matrix(trimmed, feature_set=FEATURE_SET_ORIGINAL)
        cols = get_feature_columns(feat_panel, FEATURE_SET_ORIGINAL)
        assert all(c.startswith("d_") for c in cols)
        assert len(cols) > 0

    def test_alpha_returns_af_and_d_cols(self, synthetic_panel):
        trimmed = _trim_panel(synthetic_panel)
        feat_panel = build_feature_matrix(trimmed, feature_set=FEATURE_SET_ALPHA)
        cols = get_feature_columns(feat_panel, FEATURE_SET_ALPHA)
        has_d  = any(c.startswith("d_")  for c in cols)
        has_af = any(c.startswith("af_") for c in cols)
        assert has_d and has_af

    def test_returns_list(self, synthetic_panel):
        trimmed = _trim_panel(synthetic_panel)
        feat_panel = build_feature_matrix(trimmed, feature_set=FEATURE_SET_ORIGINAL)
        cols = get_feature_columns(feat_panel, FEATURE_SET_ORIGINAL)
        assert isinstance(cols, list)
