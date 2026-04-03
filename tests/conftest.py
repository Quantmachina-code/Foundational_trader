"""Shared fixtures for catboost_trader test suite.

All fixtures use purely synthetic data so tests run without FMP API keys
or pre-built parquet files.  The synthetic panel mimics the real panel
schema (date, ticker, open, high, low, close, adjClose, volume, d_* cols).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "GOOG", "SPY"]          # SPY needed for regime
N_DAYS   = 800                                       # > REGIME_WINDOW(500) warmup
START    = pd.Timestamp("2012-01-01")


# ---------------------------------------------------------------------------
# Synthetic panel builder
# ---------------------------------------------------------------------------

def make_panel(
    tickers: list[str] = TICKERS,
    n_days: int        = N_DAYS,
    start: pd.Timestamp = START,
    seed: int           = 42,
) -> pd.DataFrame:
    """Return a synthetic multi-ticker daily panel with d_* indicator columns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)

    rows = []
    for ticker in tickers:
        # Simulate a random-walk price series
        log_rets = rng.normal(0.0003, 0.015, n_days)
        close = 100.0 * np.exp(np.cumsum(log_rets))
        high  = close * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        low   = close * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        open_ = close * (1 + rng.normal(0, 0.003, n_days))
        vol   = rng.integers(1_000_000, 10_000_000, n_days).astype(float)

        s = pd.Series(close)
        sma20  = s.rolling(20).mean().values
        sma50  = s.rolling(50).mean().values
        sma200 = s.rolling(200).mean().values
        ema12  = s.ewm(span=12, adjust=False).mean().values
        ema20  = s.ewm(span=20, adjust=False).mean().values
        ema26  = s.ewm(span=26, adjust=False).mean().values
        ema50  = s.ewm(span=50, adjust=False).mean().values
        ema200 = s.ewm(span=200, adjust=False).mean().values

        delta = s.diff()
        gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi   = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).values

        ema12s = pd.Series(ema12)
        ema26s = pd.Series(ema26)
        macd   = (ema12s - ema26s).values
        macd_signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
        macd_hist   = macd - macd_signal

        log_ret1 = np.concatenate([[np.nan], np.diff(np.log(close))])
        vol20 = pd.Series(log_ret1).rolling(20).std().values * np.sqrt(252)
        vol60 = pd.Series(log_ret1).rolling(60).std().values * np.sqrt(252)

        for i, date in enumerate(dates):
            rows.append({
                "date":           date,
                "ticker":         ticker,
                "open":           open_[i],
                "high":           high[i],
                "low":            low[i],
                "close":          close[i],
                "adjClose":       close[i],
                "volume":         vol[i],
                "d_sma_20":       sma20[i],
                "d_sma_50":       sma50[i],
                "d_sma_200":      sma200[i],
                "d_ema_12":       ema12[i],
                "d_ema_20":       ema20[i],
                "d_ema_26":       ema26[i],
                "d_ema_50":       ema50[i],
                "d_ema_200":      ema200[i],
                "d_c_vs_sma_20":  (close[i] / sma20[i] - 1) if sma20[i] else np.nan,
                "d_c_vs_sma_50":  (close[i] / sma50[i] - 1) if sma50[i] else np.nan,
                "d_c_vs_sma_200": (close[i] / sma200[i] - 1) if sma200[i] else np.nan,
                "d_c_vs_ema_12":  (close[i] / ema12[i] - 1),
                "d_c_vs_ema_20":  (close[i] / ema20[i] - 1),
                "d_c_vs_ema_26":  (close[i] / ema26[i] - 1),
                "d_c_vs_ema_50":  (close[i] / ema50[i] - 1),
                "d_c_vs_ema_200": (close[i] / ema200[i] - 1),
                "d_rsi_14":       rsi[i],
                "d_macd":         macd[i],
                "d_macd_signal":  macd_signal[i],
                "d_macd_hist":    macd_hist[i],
                "d_macd_norm":    macd[i] / close[i],
                "d_log_ret_1":    log_ret1[i],
                "d_vol_20":       vol20[i],
                "d_vol_60":       vol60[i],
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_panel() -> pd.DataFrame:
    """Full 800-day × 4-ticker synthetic panel."""
    return make_panel()


@pytest.fixture(scope="session")
def small_panel() -> pd.DataFrame:
    """Smaller 600-day × 3-ticker panel (no SPY) for faster tests."""
    return make_panel(tickers=["AAPL", "MSFT", "GOOG"], n_days=600)
