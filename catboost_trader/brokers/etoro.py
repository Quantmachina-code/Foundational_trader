"""eToro live-trading broker — REST API implementation.

Authentication
--------------
eToro's partner API uses a Bearer token passed in the Authorization header.
Obtain credentials by applying for eToro's OpenAPI partnership programme:
  https://www.etoro.com/partners/

Required environment variables (or pass explicitly):
  ETORO_API_KEY     — your API key / Bearer token
  ETORO_ACCOUNT_ID  — your numeric account ID

Leverage & shorts
-----------------
eToro does NOT accept a continuous leverage value.  Positions are sized by
``investedAmount`` (USD), and leverage is chosen from a discrete set:
  [1, 2, 5, 10, 25, 50, 100, 200]  — availability depends on the instrument
  and your regulatory jurisdiction.

For stocks (most common):
  EU/UK regulated accounts (ESMA): max 5×
  Non-EU accounts: up to 10× (sometimes higher via CFD)

The ``_quantize_leverage`` helper maps a continuous computed leverage value
to the nearest lower discrete level so positions are never over-leveraged.

Short positions are opened by setting ``direction="Sell"``.  On eToro these
are CFD positions — the broker closes them via the normal DELETE endpoint.

Instrument IDs
--------------
eToro uses numeric instrument IDs (not ticker symbols) for order placement.
``_get_instrument_id`` looks up the mapping via the instruments endpoint and
caches the result for the process lifetime.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

from catboost_trader.brokers.interface import BrokerInterface, Order, Position

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ETORO_BASE = "https://api.etoro.com/sapi/v1"

# Discrete leverage levels eToro supports (ascending).
# The bot picks the largest value ≤ requested leverage.
_LEVERAGE_LEVELS = [1, 2, 5, 10, 25, 50]

# Max leverage cap enforced by the bot regardless of what eToro allows.
# EU ESMA rules cap retail stocks at 5×.  Adjust for your jurisdiction.
MAX_STOCK_LEVERAGE = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quantize_leverage(continuous_lev: float, max_lev: int = MAX_STOCK_LEVERAGE) -> int:
    """Map a continuous leverage to the nearest lower discrete eToro level.

    >>> _quantize_leverage(3.7)
    2
    >>> _quantize_leverage(6.0)
    5
    >>> _quantize_leverage(0.4)
    1
    """
    capped = max(1.0, min(float(continuous_lev), float(max_lev)))
    # Largest level that does not exceed the continuous value
    chosen = 1
    for lvl in _LEVERAGE_LEVELS:
        if lvl <= capped:
            chosen = lvl
        else:
            break
    return min(chosen, max_lev)


# ---------------------------------------------------------------------------
# eToro broker
# ---------------------------------------------------------------------------

class EtoroBroker(BrokerInterface):
    """eToro live-trading broker via the OpenAPI REST endpoints.

    Parameters
    ----------
    api_key:
        Bearer token (eToro OpenAPI partner key).
    account_id:
        Numeric eToro account ID (string or int).
    demo:
        If True, prefix all requests with the paper-trading base URL.
        eToro's demo environment mirrors the live API.
    max_retries:
        Number of HTTP retries on transient errors (429 / 5xx).
    retry_delay:
        Base delay (seconds) between retries; doubles each attempt.
    """

    def __init__(
        self,
        api_key: str | None = None,
        account_id: str | None = None,
        *,
        demo: bool = False,
        max_retries: int = 4,
        retry_delay: float = 2.0,
    ) -> None:
        self._api_key    = api_key    or os.environ.get("ETORO_API_KEY", "")
        self._account_id = str(account_id or os.environ.get("ETORO_ACCOUNT_ID", ""))

        if not self._api_key:
            raise ValueError(
                "eToro API key missing.  Pass api_key= or set ETORO_API_KEY."
            )
        if not self._account_id:
            raise ValueError(
                "eToro account ID missing.  Pass account_id= or set ETORO_ACCOUNT_ID."
            )

        self._base       = _ETORO_BASE.replace("api.", "demo-api.") if demo else _ETORO_BASE
        self._demo       = demo
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        # Instrument ID cache:  ticker -> int
        self._instrument_cache: dict[str, int] = {}

        logger.info(
            "EtoroBroker initialised  account=%s  demo=%s  base=%s",
            self._account_id, demo, self._base,
        )

    # ------------------------------------------------------------------
    # Private: HTTP layer
    # ------------------------------------------------------------------

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        """Perform an authenticated HTTP request with exponential-backoff retry."""
        url = f"{self._base}/{path.lstrip('/')}"
        delay = self._retry_delay

        for attempt in range(1, self._max_retries + 2):
            try:
                resp = requests.request(
                    method, url,
                    headers=self._headers,
                    json=json,
                    params=params,
                    timeout=30,
                )
                if resp.status_code == 429:          # rate limit — always retry
                    wait = float(resp.headers.get("Retry-After", delay))
                    logger.warning("Rate-limited; sleeping %.1fs", wait)
                    time.sleep(wait)
                    continue
                if resp.status_code in (500, 502, 503, 504):
                    if attempt <= self._max_retries:
                        logger.warning(
                            "HTTP %d on %s; retry %d/%d in %.0fs",
                            resp.status_code, path, attempt, self._max_retries, delay,
                        )
                        time.sleep(delay)
                        delay *= 2
                        continue
                resp.raise_for_status()
                # 204 No Content (e.g. DELETE) returns empty body
                return resp.json() if resp.content else {}

            except requests.exceptions.ConnectionError as exc:
                if attempt <= self._max_retries:
                    logger.warning("Connection error; retry %d/%d: %s", attempt, self._max_retries, exc)
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise

        raise RuntimeError(f"eToro API request failed after {self._max_retries} retries: {method} {url}")

    # ------------------------------------------------------------------
    # Private: Instrument ID lookup
    # ------------------------------------------------------------------

    def _get_instrument_id(self, ticker: str) -> int:
        """Return the eToro numeric instrument ID for *ticker*.

        Results are cached for the lifetime of this broker instance.
        eToro uses the ``InstrumentDisplayName`` or ``Ticker`` field returned
        by the instruments endpoint to match a symbol.
        """
        if ticker in self._instrument_cache:
            return self._instrument_cache[ticker]

        # eToro instruments search endpoint
        data = self._request("GET", "instruments", params={"SearchText": ticker})

        instruments = data if isinstance(data, list) else data.get("instruments", [])
        for inst in instruments:
            # Match by exact symbol / ticker field
            sym = inst.get("Ticker") or inst.get("symbol") or inst.get("displayName") or ""
            if sym.upper() == ticker.upper():
                iid = int(inst["InstrumentID"] if "InstrumentID" in inst else inst["instrumentId"])
                self._instrument_cache[ticker] = iid
                return iid

        raise ValueError(
            f"Instrument '{ticker}' not found in eToro instruments endpoint.  "
            "Check that the ticker is listed on eToro and your API has access."
        )

    # ------------------------------------------------------------------
    # Account information
    # ------------------------------------------------------------------

    def get_balance(self) -> float:
        """Return total portfolio equity (invested + available cash)."""
        data = self._request("GET", f"accounts/{self._account_id}/balance")
        # eToro response shape:  {"realizedEquity": ..., "availableToInvest": ...}
        return float(
            data.get("realizedEquity")
            or data.get("equity")
            or data.get("totalBalance")
            or 0.0
        )

    def get_cash(self) -> float:
        """Return available (uninvested) cash."""
        data = self._request("GET", f"accounts/{self._account_id}/balance")
        return float(
            data.get("availableToInvest")
            or data.get("availableCash")
            or data.get("cash")
            or 0.0
        )

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> dict[str, Position]:
        """Return all open positions keyed by ticker symbol."""
        data = self._request(
            "GET", f"accounts/{self._account_id}/portfolio/positions"
        )
        raw_positions = data if isinstance(data, list) else data.get("positions", [])

        result: dict[str, Position] = {}
        for p in raw_positions:
            ticker = (
                p.get("Ticker") or p.get("ticker")
                or p.get("symbol") or p.get("instrumentTicker") or ""
            ).upper()
            if not ticker:
                continue

            # eToro stores the invested amount; shares approximated as
            # invested / avg-price  (useful for sizing new orders)
            avg_price  = float(p.get("OpenRate") or p.get("avgOpenRate") or p.get("avgPrice") or 0)
            invested   = float(p.get("InvestedAmount") or p.get("investedAmount") or 0)
            leverage   = float(p.get("Leverage") or p.get("leverage") or 1)
            units      = invested * leverage / avg_price if avg_price > 0 else 0.0
            direction  = (p.get("Direction") or p.get("direction") or "Buy").upper()

            position = Position(
                ticker    = ticker,
                shares    = units if direction == "BUY" else -units,
                avg_cost  = avg_price,
                peak_price= avg_price,   # will be updated as price moves
            )
            # Attach raw eToro position ID so we can close it later
            position_id = str(p.get("PositionID") or p.get("positionId") or "")
            # Store extra metadata in a side dict on the object
            object.__setattr__(position, "_etoro_id",       position_id)
            object.__setattr__(position, "_etoro_direction", direction)
            object.__setattr__(position, "_etoro_leverage",  int(leverage))

            result[ticker] = position

        return result

    def get_position(self, ticker: str) -> Optional[Position]:
        positions = self.get_positions()
        return positions.get(ticker.upper())

    def get_position_ids(self) -> dict[str, str]:
        """Return {ticker: etoro_position_id} for all open positions.

        Used by the tranche manager to close specific positions by ID.
        """
        data = self._request(
            "GET", f"accounts/{self._account_id}/portfolio/positions"
        )
        raw = data if isinstance(data, list) else data.get("positions", [])
        result: dict[str, str] = {}
        for p in raw:
            ticker = (
                p.get("Ticker") or p.get("ticker")
                or p.get("symbol") or p.get("instrumentTicker") or ""
            ).upper()
            pid = str(p.get("PositionID") or p.get("positionId") or "")
            if ticker and pid:
                result[ticker] = pid
        return result

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def place_order(self, order: Order) -> str:
        """Submit a market order and return the eToro position ID.

        The ``order.meta`` dict may contain:
          leverage  (int)  — discrete leverage level (default 1)
          amount    (float)— USD invested amount; if absent, qty × price is used
        """
        instrument_id = self._get_instrument_id(order.ticker)

        # Determine USD invested amount
        invested_amount: float
        if "amount" in order.meta:
            invested_amount = float(order.meta["amount"])
        else:
            # qty is fractional shares; get current price to compute USD amount
            price = self.get_price(order.ticker)
            invested_amount = abs(order.qty) * price

        leverage = int(order.meta.get("leverage", 1))

        body = {
            "InstrumentID":   instrument_id,
            "Direction":      order.side.capitalize(),   # "Buy" or "Sell"
            "Amount":         round(invested_amount, 2),
            "Leverage":       leverage,
            "IsTslEnabled":   False,
            "StopLossAmount": None,
            "TakeProfitAmount": None,
        }

        logger.info(
            "place_order  %s  %s  $%.2f  %dx",
            order.ticker, order.side, invested_amount, leverage,
        )
        resp = self._request(
            "POST",
            f"accounts/{self._account_id}/portfolio/positions",
            json=body,
        )

        position_id = str(
            resp.get("PositionID")
            or resp.get("positionId")
            or resp.get("id")
            or ""
        )
        logger.info("Order filled  positionId=%s", position_id)
        return position_id

    def close_position_by_id(self, position_id: str) -> bool:
        """Close a specific eToro position by its numeric position ID.

        Returns True on success, False if the position was not found.
        """
        try:
            self._request(
                "DELETE",
                f"accounts/{self._account_id}/portfolio/positions/{position_id}",
            )
            logger.info("Closed position %s", position_id)
            return True
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.warning("Position %s not found (already closed?)", position_id)
                return False
            raise

    def get_price(self, ticker: str) -> float:
        """Return the current ask price for *ticker*."""
        instrument_id = self._get_instrument_id(ticker)
        data = self._request("GET", f"instruments/{instrument_id}/rates")
        return float(
            data.get("Ask") or data.get("ask")
            or data.get("last") or data.get("close") or 0.0
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def open_long(
        self,
        ticker: str,
        usd_amount: float,
        leverage: int = 1,
    ) -> str:
        """Open a leveraged long CFD position.

        Parameters
        ----------
        ticker:
            Stock symbol (e.g. "AAPL").
        usd_amount:
            USD to invest (margin).  Notional exposure = usd_amount × leverage.
        leverage:
            Discrete leverage level.  Must be in _LEVERAGE_LEVELS and ≤ MAX_STOCK_LEVERAGE.

        Returns
        -------
        eToro position ID string.
        """
        return self.place_order(Order(
            ticker=ticker,
            qty=usd_amount / max(self.get_price(ticker), 1e-6),
            side="BUY",
            meta={"amount": usd_amount, "leverage": leverage},
        ))

    def open_short(
        self,
        ticker: str,
        usd_amount: float,
        leverage: int = 1,
    ) -> str:
        """Open a leveraged short (Sell) CFD position."""
        return self.place_order(Order(
            ticker=ticker,
            qty=usd_amount / max(self.get_price(ticker), 1e-6),
            side="SELL",
            meta={"amount": usd_amount, "leverage": leverage},
        ))
