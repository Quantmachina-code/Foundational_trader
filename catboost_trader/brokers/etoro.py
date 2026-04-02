"""eToro live-trading broker stub.

Replace every ``TODO`` method body with real eToro API calls once you have:
  1. An eToro API key (``api_key``)
  2. Your account ID (``account_id``)
  3. Your current portfolio balance (``balance``)

Switching from paper trading to live trading requires changing only one line
in ``main_trader.py``::

    # Paper (back-test):
    broker = PaperBroker(panel, initial_capital=100_000)

    # Live (eToro):
    broker = EtoroBroker(api_key="…", account_id="…")

No changes are needed anywhere else in the codebase.
"""

from __future__ import annotations

from typing import Optional

from catboost_trader.brokers.interface import BrokerInterface, Order, Position


class EtoroBroker(BrokerInterface):
    """eToro live-trading broker.

    Parameters
    ----------
    api_key:
        eToro OpenAPI key.
    account_id:
        eToro account identifier.
    """

    def __init__(self, api_key: str, account_id: str) -> None:
        self._api_key = api_key
        self._account_id = account_id
        # TODO: initialise the eToro API client here
        #   from etoro_sdk import EToroClient
        #   self._client = EToroClient(api_key=api_key)

    # ------------------------------------------------------------------
    # Account information
    # ------------------------------------------------------------------

    def get_balance(self) -> float:
        """Return total portfolio equity from eToro."""
        # TODO: return self._client.get_account_equity(self._account_id)
        raise NotImplementedError("EtoroBroker.get_balance() not yet implemented.")

    def get_cash(self) -> float:
        """Return uninvested cash from eToro."""
        # TODO: return self._client.get_available_cash(self._account_id)
        raise NotImplementedError("EtoroBroker.get_cash() not yet implemented.")

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> dict[str, Position]:
        """Return all open positions from eToro as Position objects."""
        # TODO:
        #   raw = self._client.get_positions(self._account_id)
        #   return {p["symbol"]: Position(ticker=p["symbol"], shares=p["units"],
        #                                  avg_cost=p["avg_price"], peak_price=p["avg_price"])
        #           for p in raw}
        raise NotImplementedError("EtoroBroker.get_positions() not yet implemented.")

    def get_position(self, ticker: str) -> Optional[Position]:
        """Return the Position for *ticker*, or None if not held."""
        positions = self.get_positions()
        return positions.get(ticker)

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def place_order(self, order: Order) -> str:
        """Submit a market order to eToro and return the order ID."""
        # TODO:
        #   response = self._client.place_order(
        #       account_id=self._account_id,
        #       symbol=order.ticker,
        #       units=order.qty,
        #       side=order.side,          # "BUY" or "SELL"
        #       order_type="MARKET",
        #   )
        #   return response["orderId"]
        raise NotImplementedError("EtoroBroker.place_order() not yet implemented.")

    def get_price(self, ticker: str) -> float:
        """Fetch the current market price for *ticker* from eToro."""
        # TODO:
        #   quote = self._client.get_quote(ticker)
        #   return float(quote["ask"])    # or "last" depending on API
        raise NotImplementedError("EtoroBroker.get_price() not yet implemented.")
