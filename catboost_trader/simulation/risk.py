"""Position-level risk management: trailing stop-loss.

``PositionRisk`` tracks a 7 % trailing stop for a single position.
Each day the simulation engine calls ``update(high, low)`` which:

1. Advances the peak price if today's high is a new high.
2. Recomputes the stop price as ``peak × (1 - stop_pct)``.
3. Returns True (→ exit) if today's LOW is at or below the stop price.

Using the daily low to check the stop (rather than the close) is conservative
and realistic — intra-day the stop could well have been hit even if the stock
closed above it.  The exit price used in the portfolio is the stop price itself
(not the close), which is the most realistic assumption given we only have OHLCV.
"""

from __future__ import annotations

from dataclasses import dataclass

from catboost_trader.config import TRAILING_STOP_PCT


@dataclass
class PositionRisk:
    """Trailing-stop tracker for a single position.

    Parameters
    ----------
    ticker:
        Ticker symbol (informational).
    entry_price:
        Price at which the position was entered.
    stop_pct:
        Trailing stop as a fraction of peak price (default: 7 %).
    """

    ticker: str
    entry_price: float
    stop_pct: float = TRAILING_STOP_PCT

    def __post_init__(self) -> None:
        self.peak_price: float = self.entry_price
        self.stop_price: float = self.entry_price * (1.0 - self.stop_pct)

    def update(self, high: float, low: float) -> bool:
        """Update peak and check stop.

        Parameters
        ----------
        high:
            Today's intraday high (used to advance peak).
        low:
            Today's intraday low (checked against stop price).

        Returns
        -------
        bool
            True if the trailing stop was triggered (→ position should be exited
            at ``self.stop_price``).
        """
        # Advance peak
        if high > self.peak_price:
            self.peak_price = high
            self.stop_price = self.peak_price * (1.0 - self.stop_pct)

        # Check if low breached the stop
        return low <= self.stop_price

    def reset(self, new_price: float) -> None:
        """Reset the stop tracker (e.g. after adding to a position)."""
        self.peak_price = max(self.peak_price, new_price)
        self.stop_price = self.peak_price * (1.0 - self.stop_pct)

    def __repr__(self) -> str:
        return (
            f"PositionRisk({self.ticker}, peak={self.peak_price:.2f}, "
            f"stop={self.stop_price:.2f}, pct={self.stop_pct:.1%})"
        )
