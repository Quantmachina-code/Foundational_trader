"""Test 02 — PositionRisk (trailing stop-loss).

Tests the core risk logic in isolation: no data, no model, no broker.

Run:
    pytest tests/test_02_risk.py -v
"""

import pytest
from catboost_trader.simulation.risk import PositionRisk


class TestPositionRiskInit:
    def test_initial_peak_equals_entry(self):
        pr = PositionRisk("AAPL", entry_price=100.0)
        assert pr.peak_price == 100.0

    def test_initial_stop_price(self):
        pr = PositionRisk("AAPL", entry_price=100.0, stop_pct=0.07)
        assert pr.stop_price == pytest.approx(93.0, rel=1e-6)

    def test_custom_stop_pct(self):
        pr = PositionRisk("MSFT", entry_price=200.0, stop_pct=0.10)
        assert pr.stop_price == pytest.approx(180.0, rel=1e-6)


class TestPositionRiskUpdate:
    def test_no_trigger_when_low_above_stop(self):
        pr = PositionRisk("AAPL", entry_price=100.0, stop_pct=0.07)
        # high=105 → peak advances to 105, stop = 105*0.93 = 97.65
        # low=99.0 > 97.65 → not triggered
        assert pr.update(high=105.0, low=99.0) is False

    def test_trigger_when_low_equals_stop(self):
        pr = PositionRisk("AAPL", entry_price=100.0, stop_pct=0.07)
        assert pr.update(high=100.0, low=93.0) is True    # low == stop exactly

    def test_trigger_when_low_below_stop(self):
        pr = PositionRisk("AAPL", entry_price=100.0, stop_pct=0.07)
        assert pr.update(high=100.0, low=80.0) is True

    def test_peak_advances_on_new_high(self):
        pr = PositionRisk("AAPL", entry_price=100.0, stop_pct=0.07)
        pr.update(high=120.0, low=105.0)
        assert pr.peak_price == 120.0
        assert pr.stop_price == pytest.approx(120.0 * 0.93, rel=1e-6)

    def test_peak_does_not_retreat(self):
        pr = PositionRisk("AAPL", entry_price=100.0, stop_pct=0.07)
        pr.update(high=120.0, low=105.0)
        pr.update(high=110.0, low=108.0)   # lower high, peak should stay at 120
        assert pr.peak_price == 120.0

    def test_stop_rises_as_price_rises(self):
        pr = PositionRisk("AAPL", entry_price=100.0, stop_pct=0.07)
        pr.update(high=110.0, low=105.0)
        assert pr.stop_price == pytest.approx(110.0 * 0.93, rel=1e-6)
        pr.update(high=130.0, low=115.0)
        assert pr.stop_price == pytest.approx(130.0 * 0.93, rel=1e-6)

    def test_multi_day_no_trigger(self):
        """Simulate 5 steady up-days — no stop should fire."""
        pr = PositionRisk("AAPL", entry_price=100.0, stop_pct=0.07)
        prices = [102, 104, 106, 108, 110]
        for p in prices:
            triggered = pr.update(high=p, low=p * 0.98)
            assert triggered is False
        assert pr.peak_price == 110.0

    def test_stop_fires_after_big_drop(self):
        """Price rises to 150 then crashes 20% intra-day."""
        pr = PositionRisk("AAPL", entry_price=100.0, stop_pct=0.07)
        pr.update(high=150.0, low=145.0)
        # Now low = 150 * 0.80 = 120.0; stop = 150*0.93 = 139.5 → triggered
        triggered = pr.update(high=145.0, low=120.0)
        assert triggered is True
        assert pr.stop_price == pytest.approx(139.5, rel=1e-4)


class TestPositionRiskReset:
    def test_reset_raises_peak(self):
        pr = PositionRisk("AAPL", entry_price=100.0, stop_pct=0.07)
        pr.reset(new_price=150.0)
        assert pr.peak_price == 150.0
        assert pr.stop_price == pytest.approx(150.0 * 0.93, rel=1e-6)

    def test_reset_does_not_lower_peak(self):
        pr = PositionRisk("AAPL", entry_price=100.0, stop_pct=0.07)
        pr.update(high=200.0, low=190.0)
        pr.reset(new_price=150.0)   # new_price < current peak → peak stays
        assert pr.peak_price == 200.0


class TestPositionRiskRepr:
    def test_repr_contains_ticker(self):
        pr = PositionRisk("AAPL", entry_price=100.0)
        assert "AAPL" in repr(pr)
