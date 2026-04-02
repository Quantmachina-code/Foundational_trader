"""Abstract broker interface.

All broker implementations (paper, eToro, …) must subclass ``BrokerInterface``
and implement every abstract method.  The simulation engine only calls these
methods, so switching from back-test to live trading is one line change in the
entry-point scripts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Order:
    """Represents a single order instruction."""

    ticker: str
    qty: float           # positive = buy, negative = sell (fractional shares allowed)
    side: str            # "BUY" or "SELL"
    order_type: str = "MARKET"
    limit_price: Optional[float] = None
    meta: dict = field(default_factory=dict)


@dataclass
class Position:
    """A currently held position."""

    ticker: str
    shares: float
    avg_cost: float
    peak_price: float    # used by risk manager for trailing stop


class BrokerInterface(ABC):
    """Abstract base class for all broker backends.

    Implement this interface to add support for any brokerage.  The paper
    broker (``PaperBroker``) is used for back-testing; the eToro broker stub
    (``EtoroBroker``) shows where live-trading calls would go.
    """

    # ------------------------------------------------------------------
    # Account information
    # ------------------------------------------------------------------

    @abstractmethod
    def get_balance(self) -> float:
        """Return total account equity (cash + market value of positions)."""

    @abstractmethod
    def get_cash(self) -> float:
        """Return uninvested cash balance."""

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        """Return a mapping of ticker → Position for all open positions."""

    @abstractmethod
    def get_position(self, ticker: str) -> Optional[Position]:
        """Return the Position for *ticker*, or None if not held."""

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    @abstractmethod
    def place_order(self, order: Order) -> str:
        """Submit an order and return an order-ID string."""

    @abstractmethod
    def get_price(self, ticker: str) -> float:
        """Return the current (or last-known) price for *ticker*."""

    # ------------------------------------------------------------------
    # Convenience helpers (not abstract — default implementations provided)
    # ------------------------------------------------------------------

    def buy(self, ticker: str, qty: float) -> str:
        """Place a market buy order for *qty* shares of *ticker*."""
        return self.place_order(Order(ticker=ticker, qty=qty, side="BUY"))

    def sell(self, ticker: str, qty: float) -> str:
        """Place a market sell order for *qty* shares of *ticker*."""
        return self.place_order(Order(ticker=ticker, qty=abs(qty), side="SELL"))

    def close_position(self, ticker: str) -> Optional[str]:
        """Fully exit the position in *ticker* if one exists."""
        pos = self.get_position(ticker)
        if pos is None or pos.shares == 0:
            return None
        return self.sell(ticker, pos.shares)
