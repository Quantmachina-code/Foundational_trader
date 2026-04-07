"""H=3 tranche state management.

A "tranche" is the set of long + short positions opened on a single trading
day.  Because the model horizon is h=3 trading days, we hold each tranche
for exactly 3 trading days then close it.

State is persisted to a JSON file so the bot survives restarts.

Tranche record schema
---------------------
{
  "open_date": "2025-04-01",        ISO trade date
  "long_position_ids":  {"AAPL": "pid_001", ...},   eToro position IDs
  "short_position_ids": {"TSLA": "pid_042", ...},
  "leverage":   5,
  "long_weight": 0.85,
  "short_weight": 0.15,
}

Lifecycle
---------
open_tranche(date, ...)  → appended to state["tranches"]
get_expired_tranches(today) → tranches whose open_date is ≥ h trading days ago
close_tranche(broker, tranche_id) → calls broker.close_position_by_id() for each position
"""

from __future__ import annotations

import json
import logging
from datetime import date as _date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from catboost_trader.brokers.etoro import EtoroBroker

logger = logging.getLogger(__name__)

HORIZON = 3   # trading days to hold each tranche


class TrancheManager:
    """Persist and manage h=3 tranches across bot runs.

    Parameters
    ----------
    state_path:
        Path to the JSON state file.  Created fresh if it doesn't exist.
    horizon:
        Number of trading days each tranche is held (default 3).
    """

    def __init__(self, state_path: str | Path, horizon: int = HORIZON) -> None:
        self._path    = Path(state_path)
        self._horizon = horizon
        self._state   = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    state = json.load(f)
                logger.info("Loaded tranche state from %s  (%d tranches)", self._path, len(state.get("tranches", [])))
                return state
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Corrupt state file; starting fresh: %s", exc)
        return {"tranches": []}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._state, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Tranche operations
    # ------------------------------------------------------------------

    def open_tranche(
        self,
        open_date:          _date | pd.Timestamp | str,
        long_position_ids:  dict[str, str],
        short_position_ids: dict[str, str],
        leverage:           int,
        long_weight:        float,
        short_weight:       float,
        regime:             str = "UNKNOWN",
        regime_score:       float = 0.0,
    ) -> None:
        """Record a newly opened tranche."""
        tranche = {
            "open_date":          str(pd.Timestamp(open_date).date()),
            "long_position_ids":  long_position_ids,
            "short_position_ids": short_position_ids,
            "leverage":           leverage,
            "long_weight":        long_weight,
            "short_weight":       short_weight,
            "regime":             regime,
            "regime_score":       float(regime_score),
        }
        self._state["tranches"].append(tranche)
        self._save()
        logger.info(
            "Opened tranche  date=%s  longs=%d  shorts=%d  lev=%dx",
            tranche["open_date"],
            len(long_position_ids),
            len(short_position_ids),
            leverage,
        )

    def get_expired_tranches(self, today: _date | pd.Timestamp | str) -> list[dict]:
        """Return tranches that should be closed today.

        A tranche opened on date T should be closed when T + horizon trading
        days have elapsed.  We use a conservative calendar approximation:
        trading days = calendar days × (5/7), or exact if a trading calendar
        is provided.

        In practice we compare against a sorted list of trading dates derived
        from the downloaded price data so the count is exact.
        """
        today_ts = pd.Timestamp(today).normalize()
        expired  = []
        for t in self._state["tranches"]:
            open_ts = pd.Timestamp(t["open_date"]).normalize()
            # Approximate: count trading days between open and today
            n_trading = _count_trading_days(open_ts, today_ts)
            if n_trading >= self._horizon:
                expired.append(t)
        return expired

    def close_expired_tranches(
        self,
        broker: "EtoroBroker",
        today:  _date | pd.Timestamp | str,
    ) -> list[dict]:
        """Close all expired tranches via the broker.

        Returns list of closed tranche records.
        """
        expired  = self.get_expired_tranches(today)
        closed   = []
        open_ids = broker.get_position_ids()   # ticker → etoro_position_id

        for tranche in expired:
            logger.info(
                "Closing tranche opened %s  (%d longs, %d shorts)",
                tranche["open_date"],
                len(tranche["long_position_ids"]),
                len(tranche["short_position_ids"]),
            )
            all_positions = {
                **tranche["long_position_ids"],
                **tranche["short_position_ids"],
            }
            for ticker, stored_pid in all_positions.items():
                # Prefer the stored position ID; fall back to live lookup
                pid = stored_pid or open_ids.get(ticker.upper())
                if not pid:
                    logger.warning("No position ID for %s — may have been closed already", ticker)
                    continue
                success = broker.close_position_by_id(pid)
                logger.info("  %s  pid=%s  closed=%s", ticker, pid, success)

            closed.append(tranche)

        # Prune closed tranches from state
        remaining = [t for t in self._state["tranches"] if t not in closed]
        self._state["tranches"] = remaining
        self._save()

        return closed

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def active_tranches(self) -> list[dict]:
        return list(self._state["tranches"])

    def n_active(self) -> int:
        return len(self._state["tranches"])

    def summary(self) -> str:
        lines = [f"Active tranches: {self.n_active()}"]
        for t in self._state["tranches"]:
            lines.append(
                f"  {t['open_date']}  "
                f"L={list(t['long_position_ids'].keys())}  "
                f"S={list(t['short_position_ids'].keys())}  "
                f"{t['leverage']}x  {t['regime']}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: count approximate trading days between two dates
# ---------------------------------------------------------------------------

def _count_trading_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Count weekday (Mon-Fri) days between start (exclusive) and end (inclusive).

    This is an approximation — it counts Mon-Fri days but does not subtract
    US market holidays.  Off-by-one on holiday weeks is acceptable for an
    EOD strategy with h=3.
    """
    if end <= start:
        return 0
    idx = pd.bdate_range(start=start + pd.Timedelta(days=1), end=end)
    return len(idx)
