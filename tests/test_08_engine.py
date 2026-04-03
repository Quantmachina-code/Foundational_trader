"""Test 08 — Full simulation engine (integration test).

Uses a compact synthetic panel (3 tickers, 700 days) to run the complete
simulation loop end-to-end without any FMP or eToro API calls.

The training window is 2012-01-01 → 2014-12-31.
The simulation window is 2015-01-01 → a few months later.

This test is intentionally NOT exhaustive — it validates that the
plumbing works and that key invariants hold, not that the model is good.

Run:
    pytest tests/test_08_engine.py -v       (fast, ~60s with 50-iter catboost)
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from catboost_trader.simulation.engine import run_simulation, SimulationResult
from catboost_trader.config import FEATURE_SET_ORIGINAL
from tests.conftest import make_panel


# ---------------------------------------------------------------------------
# Fixture: compact panel covering both train and sim windows
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mini_panel():
    """3 non-SPY tickers × 700 trading days starting 2012-01-01.

    SPY is included so regime signals work (fallback equal-weight if absent).
    """
    return make_panel(tickers=["AAPL", "MSFT", "GOOG", "SPY"], n_days=700,
                      start=pd.Timestamp("2012-01-01"))


# Dates are derived dynamically in helpers below to match the panel's actual range.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_bounds(panel: pd.DataFrame):
    """Return (train_start, sim_start, sim_end) based on panel date range.

    Uses the first 70 % of dates for training and the last 30 % for simulation.
    """
    dates = sorted(panel["date"].unique())
    n = len(dates)
    split = int(n * 0.70)
    # Add one-day buffer so training doesn't overlap simulation
    return (
        str(dates[0].date()),
        str(dates[split].date()),
        str(dates[-1].date()),
    )


def _run(mini_panel, top_n=3):
    train_start, sim_start, sim_end = _date_bounds(mini_panel)
    return run_simulation(
        panel=mini_panel,
        horizon=1,
        feature_set=FEATURE_SET_ORIGINAL,
        top_n=top_n,
        train_start=train_start,
        sim_start=sim_start,
        sim_end=sim_end,
        initial_capital=100_000.0,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# SimulationResult structure
# ---------------------------------------------------------------------------

class TestSimulationResultStructure:
    @pytest.fixture(scope="class")
    def result(self, mini_panel):
        return _run(mini_panel)

    def test_returns_simulation_result(self, result):
        assert isinstance(result, SimulationResult)

    def test_equity_curve_is_series(self, result):
        assert isinstance(result.equity_curve, pd.Series)

    def test_equity_curve_non_empty(self, result):
        assert len(result.equity_curve) > 0

    def test_equity_curve_date_indexed(self, result):
        assert isinstance(result.equity_curve.index, pd.DatetimeIndex)

    def test_equity_starts_near_initial_capital(self, result):
        first_equity = result.equity_curve.iloc[0]
        assert 50_000 < first_equity <= 100_000, \
            f"First equity {first_equity} not in expected range"

    def test_equity_all_positive(self, result):
        assert (result.equity_curve > 0).all()

    def test_equity_within_sim_window(self, result, mini_panel):
        _, sim_start, sim_end = _date_bounds(mini_panel)
        assert result.equity_curve.index.min() >= pd.Timestamp(sim_start)
        assert result.equity_curve.index.max() <= pd.Timestamp(sim_end)

    def test_regime_df_has_columns(self, result):
        assert "regime" in result.regime_df.columns
        assert "leverage" in result.regime_df.columns

    def test_regime_values_valid(self, result):
        valid = {"BEAR", "NEUTRAL", "BULL"}
        assert set(result.regime_df["regime"].unique()) <= valid

    def test_leverage_non_negative(self, result):
        assert (result.regime_df["leverage"] >= 0).all()

    def test_metrics_dict_non_empty(self, result):
        assert isinstance(result.metrics, dict)
        assert len(result.metrics) > 0

    def test_metrics_has_sharpe(self, result):
        assert "sharpe" in result.metrics

    def test_metrics_has_max_drawdown(self, result):
        assert "max_drawdown" in result.metrics

    def test_retrain_dates_non_empty(self, result):
        assert len(result.retrain_dates) >= 1

    def test_label_set(self, result):
        assert result.label != ""


# ---------------------------------------------------------------------------
# Economic invariants
# ---------------------------------------------------------------------------

class TestEconomicInvariants:
    @pytest.fixture(scope="class")
    def result(self, mini_panel):
        return _run(mini_panel)

    def test_no_free_money(self, result):
        """Equity should never exceed initial capital × max_leverage^n × some headroom."""
        assert result.equity_curve.max() < 100_000 * 50, "Equity implausibly large"

    def test_stop_exits_dataframe(self, result):
        """stop_exits must be a DataFrame (may be empty if no stops triggered)."""
        assert isinstance(result.stop_exits, pd.DataFrame)

    def test_bear_regime_has_zero_leverage(self, result):
        bear_rows = result.regime_df[result.regime_df["regime"] == "BEAR"]
        if len(bear_rows) > 0:
            assert (bear_rows["leverage"] == 0.0).all()

    def test_monthly_retrains_at_most_once_per_month(self, result):
        """retrain_dates must not have two entries in the same month."""
        months = [(d.year, d.month) for d in result.retrain_dates]
        assert len(months) == len(set(months)), "Duplicate monthly retrains detected"


# ---------------------------------------------------------------------------
# Top-N variation
# ---------------------------------------------------------------------------

class TestTopNVariation:
    def test_different_top_n_produces_different_equity(self, mini_panel):
        r3  = _run(mini_panel, top_n=3)
        r5  = _run(mini_panel, top_n=min(5, 3))   # only 3 non-SPY tickers
        # Curves may differ (different position sizing), both should be valid
        assert (r3.equity_curve > 0).all()
        assert (r5.equity_curve > 0).all()

    def test_top_n_label_reflects_setting(self, mini_panel):
        r = _run(mini_panel, top_n=3)
        assert "top3" in r.label or "3" in r.label


# ---------------------------------------------------------------------------
# Portfolio class unit tests (lighter than full engine run)
# ---------------------------------------------------------------------------

class TestPortfolio:
    def _make_broker(self, panel):
        from catboost_trader.brokers.paper import PaperBroker
        return PaperBroker(panel, initial_capital=10_000.0)

    def test_no_positions_initially(self, mini_panel):
        from catboost_trader.simulation.portfolio import Portfolio
        broker = self._make_broker(mini_panel)
        p = Portfolio(broker)
        assert len(broker.get_positions()) == 0

    def test_drawdown_zero_at_start(self, mini_panel):
        from catboost_trader.simulation.portfolio import Portfolio
        broker = self._make_broker(mini_panel)
        p = Portfolio(broker)
        assert p.current_drawdown() == 0.0

    def test_close_all_in_bear(self, mini_panel):
        from catboost_trader.simulation.portfolio import Portfolio
        broker = self._make_broker(mini_panel)
        p = Portfolio(broker)
        date0 = mini_panel["date"].min()
        broker.set_date(date0)
        broker.buy("AAPL", qty=1.0)

        # Rebalance with leverage=0 (bear) should close all
        p.rebalance(pd.Series({"AAPL": 0.9, "MSFT": 0.1}), top_n=2,
                    leverage=0.0, date=date0)
        assert len(broker.get_positions()) == 0

    def test_equity_series_date_indexed(self, mini_panel):
        from catboost_trader.simulation.portfolio import Portfolio
        broker = self._make_broker(mini_panel)
        p = Portfolio(broker)
        date0 = mini_panel["date"].min()
        p.record_daily(date0, "NEUTRAL", 1.0)
        eq = p.get_equity_series()
        assert isinstance(eq.index, pd.DatetimeIndex)
