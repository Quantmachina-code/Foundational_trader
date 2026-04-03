"""Test 03 — Regime detection.

Tests build_regime_signals(), classify_regime(), and get_leverage()
using the synthetic panel fixture.  No FMP calls, no model required.

Run:
    pytest tests/test_03_regime.py -v
"""

import numpy as np
import pandas as pd
import pytest

from catboost_trader.simulation.regime import (
    build_regime_signals,
    classify_regime,
    get_leverage,
)
from catboost_trader.config import MAX_LEVERAGE


# ---------------------------------------------------------------------------
# build_regime_signals
# ---------------------------------------------------------------------------

class TestBuildRegimeSignals:
    def test_returns_dataframe(self, synthetic_panel):
        df = build_regime_signals(synthetic_panel)
        assert isinstance(df, pd.DataFrame)

    def test_has_score_column(self, synthetic_panel):
        df = build_regime_signals(synthetic_panel)
        assert "score" in df.columns

    def test_has_signal_columns(self, synthetic_panel):
        df = build_regime_signals(synthetic_panel)
        expected = {"sig_mom5", "sig_mom10", "sig_mom20",
                    "sig_breadth", "sig_vol", "sig_rsi", "sig_disp"}
        assert expected <= set(df.columns)

    def test_score_in_unit_interval(self, synthetic_panel):
        df = build_regime_signals(synthetic_panel)
        valid = df["score"].dropna()
        assert (valid >= 0.0).all(), "Score below 0"
        assert (valid <= 1.0).all(), "Score above 1"

    def test_date_indexed(self, synthetic_panel):
        df = build_regime_signals(synthetic_panel)
        assert df.index.name == "date"

    def test_no_look_ahead_in_signals(self, synthetic_panel):
        """Signals should be NaN at the very start (warm-up period), not all populated."""
        df = build_regime_signals(synthetic_panel)
        # First row should be NaN (not enough history for rolling rank)
        assert df["sig_mom5"].iloc[0] is np.nan or np.isnan(df["sig_mom5"].iloc[0])

    def test_fallback_without_spy(self, synthetic_panel):
        """When SPY is absent the function should use equal-weight fallback."""
        panel_no_spy = synthetic_panel[synthetic_panel["ticker"] != "SPY"].copy()
        df = build_regime_signals(panel_no_spy, market_ticker="SPY")
        assert "score" in df.columns
        assert df["score"].dropna().shape[0] > 0

    def test_breadth_uses_d_c_vs_sma_50(self, synthetic_panel):
        """Panel already has d_c_vs_sma_50 — check breadth signal is computed."""
        df = build_regime_signals(synthetic_panel)
        assert "sig_breadth" in df.columns
        assert df["sig_breadth"].dropna().shape[0] > 0


# ---------------------------------------------------------------------------
# classify_regime
# ---------------------------------------------------------------------------

class TestClassifyRegime:
    def _history(self, n: int = 200, seed: int = 0) -> pd.Series:
        rng = np.random.default_rng(seed)
        return pd.Series(rng.uniform(0, 1, n))

    def test_returns_valid_label(self):
        h = self._history()
        label = classify_regime(float(h.median()), h)
        assert label in ("BEAR", "NEUTRAL", "BULL")

    def test_very_low_score_is_bear(self):
        h = self._history()
        label = classify_regime(0.0, h)   # absolute minimum → BEAR
        assert label == "BEAR"

    def test_very_high_score_is_bull(self):
        h = self._history()
        label = classify_regime(1.0, h)   # absolute maximum → BULL
        assert label == "BULL"

    def test_insufficient_history_returns_neutral(self):
        short_history = pd.Series([0.1, 0.2, 0.3])   # < 50 obs
        label = classify_regime(0.5, short_history)
        assert label == "NEUTRAL"

    def test_nan_history_ignored(self):
        h = pd.Series([np.nan] * 30 + [0.5] * 100)
        label = classify_regime(0.5, h)
        assert label in ("BEAR", "NEUTRAL", "BULL")


# ---------------------------------------------------------------------------
# get_leverage
# ---------------------------------------------------------------------------

class TestGetLeverage:
    def test_bear_returns_zero(self):
        assert get_leverage("BEAR", realised_vol=0.15) == 0.0

    def test_neutral_returns_one(self):
        lev = get_leverage("NEUTRAL", realised_vol=0.15)
        assert lev == pytest.approx(1.0, abs=1e-9)

    def test_bull_returns_positive(self):
        lev = get_leverage("BULL", realised_vol=0.15)
        assert lev > 0.0

    def test_bull_capped_at_max_leverage(self):
        # Very low vol → vol-targeted leverage would be huge → should be capped
        lev = get_leverage("BULL", realised_vol=0.001)
        assert lev <= MAX_LEVERAGE + 1e-9

    def test_drawdown_delever_reduces_leverage(self):
        lev_no_dd  = get_leverage("BULL", realised_vol=0.15, portfolio_dd=0.0)
        lev_with_dd = get_leverage("BULL", realised_vol=0.15, portfolio_dd=0.25)
        assert lev_with_dd < lev_no_dd

    def test_drawdown_delever_not_applied_below_threshold(self):
        """DD below 20 % should not delever."""
        lev_base   = get_leverage("BULL", realised_vol=0.15, portfolio_dd=0.0)
        lev_low_dd = get_leverage("BULL", realised_vol=0.15, portfolio_dd=0.10)
        assert lev_low_dd == pytest.approx(lev_base, rel=1e-6)

    def test_leverage_non_negative(self):
        for regime in ("BEAR", "NEUTRAL", "BULL"):
            for vol in (0.05, 0.15, 0.50):
                for dd in (0.0, 0.15, 0.30, 0.80):
                    lev = get_leverage(regime, realised_vol=vol, portfolio_dd=dd)
                    assert lev >= 0.0, f"Negative leverage: regime={regime} vol={vol} dd={dd}"
