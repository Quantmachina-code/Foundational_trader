"""Portfolio management: position sizing, rebalancing, and equity tracking.

``Portfolio`` orchestrates position sizing and rebalancing.  It is broker-agnostic
(works with any ``BrokerInterface`` implementation) and communicates with the
risk layer (``PositionRisk``) to enforce trailing stops.

Rebalancing logic
-----------------
Each trading day the simulation engine calls::

    exits = portfolio.check_stops(today_ohlcv)   # 1. risk exits first
    portfolio.rebalance(scores, top_n, leverage)  # 2. then selection
    portfolio.record_daily(date, close_prices)    # 3. mark-to-market

Inside ``rebalance``:
    a. Target set = top_n tickers by predicted score.
    b. Exit any held ticker NOT in target set.
    c. Compute target position size = (equity × leverage) / top_n per stock.
    d. Enter or increase positions toward target size.
    e. Equal-weight within the portfolio (no individual stock weighting).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from catboost_trader.brokers.interface import BrokerInterface
from catboost_trader.simulation.risk import PositionRisk


class Portfolio:
    """Manages position sizing, rebalancing, and equity tracking.

    Parameters
    ----------
    broker:
        Any ``BrokerInterface`` implementation.
    trailing_stop_pct:
        Trailing stop fraction (default from config).
    """

    def __init__(
        self,
        broker: BrokerInterface,
        trailing_stop_pct: float = 0.07,
    ) -> None:
        self._broker = broker
        self._trailing_stop_pct = trailing_stop_pct
        self._risks: dict[str, PositionRisk] = {}   # ticker → stop tracker
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        self.regime_log: list[tuple[pd.Timestamp, str, float]] = []
        self._peak_equity: float = broker.get_balance()
        self._stop_exits: list[dict] = []    # audit trail of stop-triggered exits

    # ------------------------------------------------------------------
    # Stop-loss management
    # ------------------------------------------------------------------

    def check_stops(
        self,
        ohlcv_today: pd.DataFrame,
        date: pd.Timestamp,
    ) -> list[str]:
        """Check trailing stops against today's high/low.

        Parameters
        ----------
        ohlcv_today:
            Slice of the panel for *date* with columns: ticker, high, low, close.
        date:
            Current simulation date.

        Returns
        -------
        list[str]
            Tickers whose stops were triggered (already exited by this call).
        """
        if ohlcv_today.empty:
            return []

        price_map = ohlcv_today.set_index("ticker")
        triggered: list[str] = []

        for ticker in list(self._risks.keys()):
            if ticker not in price_map.index:
                continue
            row = price_map.loc[ticker]
            high = float(row.get("high", row.get("close", np.nan)))
            low = float(row.get("low", row.get("close", np.nan)))
            close = float(row.get("close", np.nan))

            if pd.isna(high) or pd.isna(low):
                continue

            risk = self._risks[ticker]
            if risk.update(high, low):
                # Stop triggered — exit at stop price (conservative)
                exit_price = risk.stop_price
                pos = self._broker.get_position(ticker)
                if pos is not None and pos.shares > 0:
                    # Temporarily override broker date if it supports it
                    if hasattr(self._broker, "set_date"):
                        self._broker.set_date(date)
                    self._broker.sell(ticker, pos.shares)
                    self._stop_exits.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "exit_price": exit_price,
                            "close": close,
                            "reason": "trailing_stop",
                        }
                    )
                del self._risks[ticker]
                triggered.append(ticker)

        return triggered

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def rebalance(
        self,
        scores: pd.Series,
        top_n: int,
        leverage: float,
        date: pd.Timestamp,
        ohlcv_today: pd.DataFrame | None = None,
    ) -> None:
        """Select top_n stocks and rebalance toward equal-weight target.

        Parameters
        ----------
        scores:
            Series of predicted scores indexed by ticker.  Higher = stronger.
        top_n:
            Number of positions to hold.
        leverage:
            Gross leverage to apply to total equity.
        date:
            Current simulation date.
        ohlcv_today:
            Optional OHLCV slice for today (used for price lookups).
        """
        if hasattr(self._broker, "set_date"):
            self._broker.set_date(date)

        if leverage == 0.0:
            # Bear market — close everything
            self._close_all(date)
            return

        # ── Target portfolio ─────────────────────────────────────────────
        valid_scores = scores.dropna()
        if len(valid_scores) == 0:
            return

        n = min(top_n, len(valid_scores))
        target_tickers = set(valid_scores.nlargest(n).index.tolist())

        # ── Exit positions no longer in target ───────────────────────────
        current_tickers = set(self._broker.get_positions().keys())
        to_exit = current_tickers - target_tickers
        for ticker in to_exit:
            self._broker.close_position(ticker)
            self._risks.pop(ticker, None)

        # ── Compute target position size ─────────────────────────────────
        equity = self._broker.get_balance()
        self._peak_equity = max(self._peak_equity, equity)
        gross_exposure = equity * leverage
        target_value_per_stock = gross_exposure / n  # equal-weight

        # ── Enter / adjust positions ──────────────────────────────────────
        for ticker in target_tickers:
            try:
                price = self._broker.get_price(ticker)
            except (KeyError, Exception):
                continue

            if price <= 0 or pd.isna(price):
                continue

            target_shares = target_value_per_stock / price
            pos = self._broker.get_position(ticker)
            current_shares = pos.shares if pos is not None else 0.0

            diff = target_shares - current_shares
            if abs(diff) < 0.01:    # ignore tiny rounding adjustments
                continue

            if diff > 0:
                order_id = self._broker.buy(ticker, diff)
            else:
                order_id = self._broker.sell(ticker, abs(diff))

            # Initialise or update trailing stop
            if ticker not in self._risks:
                self._risks[ticker] = PositionRisk(
                    ticker=ticker,
                    entry_price=price,
                    stop_pct=self._trailing_stop_pct,
                )
            else:
                self._risks[ticker].reset(price)

    # ------------------------------------------------------------------
    # Mark-to-market
    # ------------------------------------------------------------------

    def record_daily(self, date: pd.Timestamp, regime: str, leverage: float) -> float:
        """Record today's equity and return it.

        Parameters
        ----------
        date:
            Current simulation date.
        regime:
            Current regime string (``"BEAR"`` / ``"NEUTRAL"`` / ``"BULL"``).
        leverage:
            Applied leverage today.

        Returns
        -------
        float
            End-of-day equity.
        """
        if hasattr(self._broker, "set_date"):
            self._broker.set_date(date)

        equity = self._broker.get_balance()
        self._peak_equity = max(self._peak_equity, equity)
        self.equity_curve.append((date, equity))
        self.regime_log.append((date, regime, leverage))
        return equity

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _close_all(self, date: pd.Timestamp) -> None:
        """Exit all positions (called when entering BEAR regime)."""
        if hasattr(self._broker, "set_date"):
            self._broker.set_date(date)
        for ticker in list(self._broker.get_positions().keys()):
            self._broker.close_position(ticker)
        self._risks.clear()

    def current_drawdown(self) -> float:
        """Return current drawdown as a positive fraction (0 = at peak)."""
        equity = self._broker.get_balance()
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - equity / self._peak_equity)

    def get_equity_series(self) -> pd.Series:
        """Return the recorded equity curve as a date-indexed Series."""
        if not self.equity_curve:
            return pd.Series(dtype=float)
        dates, values = zip(*self.equity_curve)
        return pd.Series(values, index=pd.DatetimeIndex(dates), name="equity")

    def get_regime_series(self) -> pd.DataFrame:
        """Return regime / leverage log as a DataFrame."""
        if not self.regime_log:
            return pd.DataFrame(columns=["date", "regime", "leverage"])
        df = pd.DataFrame(self.regime_log, columns=["date", "regime", "leverage"])
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    def get_stop_exit_log(self) -> pd.DataFrame:
        """Return audit trail of all trailing-stop exits."""
        return pd.DataFrame(self._stop_exits)
