"""Paper (simulation) broker.

``PaperBroker`` implements ``BrokerInterface`` entirely in memory using
historical price data.  It is the default broker for back-testing.

Design decisions
----------------
* Prices are assumed to be end-of-day close prices (already available in the
  daily panel).  The caller sets the date via ``set_date()`` before placing
  any orders; subsequent ``get_price()`` calls look up that date's close.
* Fractional shares are allowed (realistic for dollar-based position sizing).
* No commissions by default — add a per-trade cost by setting ``commission``
  in the constructor.
* Slippage is modelled as a simple percentage of trade value (default 0%).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from catboost_trader.brokers.interface import (
    BrokerInterface,
    Order,
    Position,
)


class PaperBroker(BrokerInterface):
    """In-memory broker for back-testing.

    Parameters
    ----------
    panel:
        Multi-ticker daily panel with at minimum ``date``, ``ticker``,
        and ``close`` columns.
    initial_capital:
        Starting cash balance.
    commission:
        Flat per-trade commission in dollars (default 0).
    slippage_pct:
        One-way slippage as a fraction of trade value (default 0).
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        initial_capital: float = 100_000.0,
        commission: float = 0.0,
        slippage_pct: float = 0.0,
    ) -> None:
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._commission = commission
        self._slippage_pct = slippage_pct
        self._positions: dict[str, Position] = {}
        self._order_counter = 0
        self._current_date: Optional[pd.Timestamp] = None

        # Build a fast price lookup: {date → {ticker → close}}
        panel = panel.copy()
        panel["date"] = pd.to_datetime(panel["date"])
        self._prices: dict[pd.Timestamp, dict[str, float]] = {}
        for date, grp in panel.groupby("date"):
            self._prices[date] = dict(zip(grp["ticker"], grp["close"]))

    # ------------------------------------------------------------------
    # Date management
    # ------------------------------------------------------------------

    def set_date(self, date: pd.Timestamp) -> None:
        """Advance the broker to *date* so price lookups use that day's close."""
        self._current_date = pd.Timestamp(date)

    # ------------------------------------------------------------------
    # BrokerInterface implementation
    # ------------------------------------------------------------------

    def get_price(self, ticker: str) -> float:
        if self._current_date is None:
            raise RuntimeError("Call set_date() before get_price().")
        day_prices = self._prices.get(self._current_date, {})
        price = day_prices.get(ticker)
        if price is None or pd.isna(price):
            raise KeyError(f"No price for {ticker} on {self._current_date.date()}")
        return float(price)

    def get_cash(self) -> float:
        return self._cash

    def get_balance(self) -> float:
        equity = self._cash
        for ticker, pos in self._positions.items():
            try:
                equity += pos.shares * self.get_price(ticker)
            except KeyError:
                equity += pos.shares * pos.avg_cost
        return equity

    def get_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def get_position(self, ticker: str) -> Optional[Position]:
        return self._positions.get(ticker)

    def place_order(self, order: Order) -> str:
        self._order_counter += 1
        order_id = f"PAPER-{self._order_counter:06d}"

        try:
            price = self.get_price(order.ticker)
        except KeyError:
            return f"{order_id}-REJECTED-NO-PRICE"

        # Apply slippage
        if order.side == "BUY":
            exec_price = price * (1 + self._slippage_pct)
        else:
            exec_price = price * (1 - self._slippage_pct)

        trade_value = order.qty * exec_price
        cost = trade_value + self._commission

        if order.side == "BUY":
            if cost > self._cash:
                # Scale down to available cash
                order.qty = max(0.0, (self._cash - self._commission) / exec_price)
                trade_value = order.qty * exec_price
                cost = trade_value + self._commission
                if order.qty <= 0:
                    return f"{order_id}-REJECTED-NO-CASH"

            self._cash -= cost

            pos = self._positions.get(order.ticker)
            if pos is None:
                self._positions[order.ticker] = Position(
                    ticker=order.ticker,
                    shares=order.qty,
                    avg_cost=exec_price,
                    peak_price=exec_price,
                )
            else:
                total_shares = pos.shares + order.qty
                pos.avg_cost = (pos.shares * pos.avg_cost + order.qty * exec_price) / total_shares
                pos.shares = total_shares
                pos.peak_price = max(pos.peak_price, exec_price)

        else:  # SELL
            pos = self._positions.get(order.ticker)
            if pos is None or pos.shares == 0:
                return f"{order_id}-REJECTED-NO-POSITION"

            qty = min(order.qty, pos.shares)
            proceeds = qty * exec_price - self._commission
            self._cash += proceeds

            pos.shares -= qty
            if pos.shares < 1e-6:
                del self._positions[order.ticker]

        return order_id

    # ------------------------------------------------------------------
    # Extra helpers for the simulation engine
    # ------------------------------------------------------------------

    def mark_to_market(self, date: pd.Timestamp) -> float:
        """Set date and return current total equity."""
        self.set_date(date)
        return self.get_balance()

    def update_peak_prices(self) -> None:
        """Update peak_price for all positions using today's close."""
        for ticker, pos in self._positions.items():
            try:
                price = self.get_price(ticker)
                pos.peak_price = max(pos.peak_price, price)
            except KeyError:
                pass
