"""Test 07 — PaperBroker.

Tests the in-memory broker in full isolation: no model, no FMP data.
Uses a tiny hand-crafted panel so every dollar amount is verifiable.

Run:
    pytest tests/test_07_paper_broker.py -v
"""

import pytest
import pandas as pd

from catboost_trader.brokers.paper import PaperBroker
from catboost_trader.brokers.interface import Order, Position


# ---------------------------------------------------------------------------
# Minimal 2-ticker, 5-day price panel
# ---------------------------------------------------------------------------

def _make_panel() -> pd.DataFrame:
    """AAPL: $100 → $150 over 5 days.  MSFT: flat $200."""
    dates = pd.bdate_range("2022-01-03", periods=5)
    rows = []
    aapl_prices = [100.0, 110.0, 120.0, 130.0, 150.0]
    for date, price in zip(dates, aapl_prices):
        rows.append({"date": date, "ticker": "AAPL", "close": price})
        rows.append({"date": date, "ticker": "MSFT", "close": 200.0})
    return pd.DataFrame(rows)


@pytest.fixture
def broker():
    """Fresh PaperBroker with $10,000 capital, no fees."""
    panel = _make_panel()
    b = PaperBroker(panel, initial_capital=10_000.0)
    b.set_date(pd.Timestamp("2022-01-03"))
    return b


@pytest.fixture
def broker_with_commission():
    panel = _make_panel()
    b = PaperBroker(panel, initial_capital=10_000.0, commission=5.0, slippage_pct=0.001)
    b.set_date(pd.Timestamp("2022-01-03"))
    return b


# ---------------------------------------------------------------------------
# Price lookup
# ---------------------------------------------------------------------------

class TestGetPrice:
    def test_known_ticker(self, broker):
        assert broker.get_price("AAPL") == 100.0

    def test_unknown_ticker_raises(self, broker):
        with pytest.raises(KeyError):
            broker.get_price("GOOG")

    def test_price_changes_with_date(self, broker):
        broker.set_date(pd.Timestamp("2022-01-07"))
        assert broker.get_price("AAPL") == 150.0

    def test_price_before_set_date_raises(self):
        panel = _make_panel()
        b = PaperBroker(panel, initial_capital=10_000.0)
        with pytest.raises(RuntimeError, match="set_date"):
            b.get_price("AAPL")


# ---------------------------------------------------------------------------
# Cash & balance
# ---------------------------------------------------------------------------

class TestBalance:
    def test_initial_cash(self, broker):
        assert broker.get_cash() == 10_000.0

    def test_balance_equals_cash_when_no_positions(self, broker):
        assert broker.get_balance() == 10_000.0

    def test_balance_after_buy(self, broker):
        broker.buy("AAPL", qty=10.0)   # 10 × $100 = $1,000
        assert broker.get_cash() == pytest.approx(9_000.0)
        assert broker.get_balance() == pytest.approx(10_000.0)   # equity unchanged

    def test_balance_reflects_price_increase(self, broker):
        broker.buy("AAPL", qty=10.0)   # buy at $100
        broker.set_date(pd.Timestamp("2022-01-07"))  # now $150
        assert broker.get_balance() == pytest.approx(9_000.0 + 10 * 150.0)


# ---------------------------------------------------------------------------
# Buy orders
# ---------------------------------------------------------------------------

class TestBuyOrder:
    def test_buy_creates_position(self, broker):
        broker.buy("AAPL", qty=5.0)
        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.shares == pytest.approx(5.0)

    def test_buy_reduces_cash(self, broker):
        broker.buy("MSFT", qty=3.0)   # 3 × $200 = $600
        assert broker.get_cash() == pytest.approx(9_400.0)

    def test_buy_sets_avg_cost(self, broker):
        broker.buy("AAPL", qty=10.0)
        pos = broker.get_position("AAPL")
        assert pos.avg_cost == pytest.approx(100.0)

    def test_buy_adds_to_existing_position(self, broker):
        broker.buy("AAPL", qty=5.0)   # at $100
        broker.set_date(pd.Timestamp("2022-01-04"))
        broker.buy("AAPL", qty=5.0)   # at $110
        pos = broker.get_position("AAPL")
        assert pos.shares == pytest.approx(10.0)
        assert pos.avg_cost == pytest.approx(105.0)   # (500+550)/10

    def test_buy_capped_at_available_cash(self, broker):
        # Try to buy 200 shares at $100 = $20,000 but only $10,000 available
        broker.buy("AAPL", qty=200.0)
        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.shares <= 100.0   # can't exceed $10,000 / $100

    def test_returns_order_id(self, broker):
        order_id = broker.buy("AAPL", qty=1.0)
        assert order_id.startswith("PAPER-")


# ---------------------------------------------------------------------------
# Sell orders
# ---------------------------------------------------------------------------

class TestSellOrder:
    def test_sell_removes_position(self, broker):
        broker.buy("AAPL", qty=5.0)
        broker.sell("AAPL", qty=5.0)
        assert broker.get_position("AAPL") is None

    def test_sell_partial(self, broker):
        broker.buy("AAPL", qty=10.0)
        broker.sell("AAPL", qty=4.0)
        pos = broker.get_position("AAPL")
        assert pos.shares == pytest.approx(6.0)

    def test_sell_increases_cash(self, broker):
        broker.buy("AAPL", qty=10.0)   # spend $1,000
        initial_cash = broker.get_cash()
        broker.sell("AAPL", qty=10.0)   # get back $1,000
        assert broker.get_cash() == pytest.approx(initial_cash + 1_000.0)

    def test_sell_with_no_position_rejected(self, broker):
        order_id = broker.sell("GOOG", qty=1.0)
        assert "REJECTED" in order_id

    def test_sell_capped_at_held_shares(self, broker):
        broker.buy("AAPL", qty=3.0)
        broker.sell("AAPL", qty=10.0)   # only 3 held
        assert broker.get_position("AAPL") is None


# ---------------------------------------------------------------------------
# close_position
# ---------------------------------------------------------------------------

class TestClosePosition:
    def test_close_existing(self, broker):
        broker.buy("AAPL", qty=5.0)
        broker.close_position("AAPL")
        assert broker.get_position("AAPL") is None

    def test_close_non_existent_returns_none(self, broker):
        result = broker.close_position("GOOG")
        assert result is None


# ---------------------------------------------------------------------------
# Commission & slippage
# ---------------------------------------------------------------------------

class TestCommissionSlippage:
    def test_commission_deducted_on_buy(self, broker_with_commission):
        broker_with_commission.buy("AAPL", qty=10.0)
        # 10 × $100 × 1.001 (slippage) + $5 commission
        expected_cost = 10 * 100.0 * 1.001 + 5.0
        assert broker_with_commission.get_cash() == pytest.approx(10_000.0 - expected_cost, rel=1e-4)

    def test_slippage_on_sell(self, broker_with_commission):
        broker_with_commission.buy("AAPL", qty=1.0)
        cash_after_buy = broker_with_commission.get_cash()
        broker_with_commission.sell("AAPL", qty=1.0)
        # Sell proceeds = $100 × 0.999 - $5 commission
        expected_proceeds = 100.0 * 0.999 - 5.0
        assert broker_with_commission.get_cash() == pytest.approx(cash_after_buy + expected_proceeds, rel=1e-4)


# ---------------------------------------------------------------------------
# mark_to_market / update_peak_prices
# ---------------------------------------------------------------------------

class TestMarkToMarket:
    def test_mtm_advances_date_and_returns_equity(self, broker):
        broker.buy("AAPL", qty=10.0)
        equity = broker.mark_to_market(pd.Timestamp("2022-01-07"))
        # cash(9000) + 10 × 150 = 10,500
        assert equity == pytest.approx(10_500.0)

    def test_update_peak_prices(self, broker):
        broker.buy("AAPL", qty=5.0)   # peak = $100
        broker.set_date(pd.Timestamp("2022-01-07"))   # price = $150
        broker.update_peak_prices()
        pos = broker.get_position("AAPL")
        assert pos.peak_price == pytest.approx(150.0)
