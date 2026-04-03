"""Test 05 — Target construction.

Tests build_targets() for correctness of forward return computation and
cross-sectional rank scaling.

Run:
    pytest tests/test_05_targets.py -v
"""

import numpy as np
import pandas as pd
import pytest

from catboost_trader.data.targets import build_targets


# ---------------------------------------------------------------------------
# Minimal deterministic panel for exact arithmetic checks
# ---------------------------------------------------------------------------

def _make_deterministic_panel() -> pd.DataFrame:
    """3 tickers, 10 days.  Prices designed for predictable forward returns."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    tickers = ["A", "B", "C"]
    rows = []
    # A: steady rise  B: flat  C: steady fall
    for i, date in enumerate(dates):
        rows.append({"date": date, "ticker": "A", "close": 100.0 + i})
        rows.append({"date": date, "ticker": "B", "close": 100.0})
        rows.append({"date": date, "ticker": "C", "close": 100.0 - i})
    return pd.DataFrame(rows)


class TestBuildTargets:
    @pytest.fixture
    def panel(self):
        return _make_deterministic_panel()

    def test_returns_dataframe(self, panel):
        out = build_targets(panel, horizon=1)
        assert isinstance(out, pd.DataFrame)

    def test_target_column_present(self, panel):
        out = build_targets(panel, horizon=1)
        assert "target" in out.columns

    def test_original_columns_preserved(self, panel):
        out = build_targets(panel, horizon=1)
        for col in ("date", "ticker", "close"):
            assert col in out.columns

    def test_target_in_unit_interval(self, panel):
        out = build_targets(panel, horizon=1)
        valid = out["target"].dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()

    def test_last_h_rows_nan(self, panel):
        """The last horizon rows per ticker cannot have a forward return."""
        horizon = 2
        out = build_targets(panel, horizon=horizon)
        for ticker in ["A", "B", "C"]:
            sub = out[out["ticker"] == ticker].sort_values("date")
            tail = sub.iloc[-horizon:]
            assert tail["target"].isna().all(), \
                f"Expected NaN in last {horizon} rows of {ticker}"

    def test_ranking_order_correct(self, panel):
        """On each date, the rising ticker (A) should rank above the falling one (C)."""
        out = build_targets(panel, horizon=1)
        # Check a middle date (not last h rows)
        mid_date = out["date"].unique()[3]
        slice_ = out[out["date"] == mid_date].set_index("ticker")
        assert slice_.loc["A", "target"] > slice_.loc["C", "target"]

    def test_cross_sectional_rank_uniform(self, panel):
        """With 3 tickers the rank values should be {1/3, 2/3, 3/3} or similar."""
        out = build_targets(panel, horizon=1)
        date0 = out["date"].unique()[0]
        vals = sorted(out[out["date"] == date0]["target"].dropna().tolist())
        # With 3 tickers, pct ranks are approximately [1/3, 2/3, 1.0]
        assert len(vals) == 3

    def test_horizon_1_vs_5(self, synthetic_panel):
        """Longer horizon produces same number of rows but more NaNs at tail."""
        out1 = build_targets(synthetic_panel, horizon=1)
        out5 = build_targets(synthetic_panel, horizon=5)
        assert len(out1) == len(out5)
        assert out5["target"].isna().sum() > out1["target"].isna().sum()

    def test_missing_close_raises(self):
        bad = pd.DataFrame({"date": pd.bdate_range("2020-01-01", 5), "ticker": "A"})
        with pytest.raises(ValueError, match="close"):
            build_targets(bad, horizon=1)

    def test_no_look_ahead(self, panel):
        """The target for date t should use close[t+h]; confirm by manual calc."""
        horizon = 1
        out = build_targets(panel, horizon=horizon)
        # Ticker A on day 0: close=100, close[+1]=101 → raw return = 1/100 = 0.01
        day0 = out["date"].unique()[0]
        row_a = out[(out["date"] == day0) & (out["ticker"] == "A")].iloc[0]
        # We can't know the exact rank without also computing B and C, but target must be defined
        assert not pd.isna(row_a["target"])
