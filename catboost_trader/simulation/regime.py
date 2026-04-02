"""Bull / bear market regime detection with adaptive thresholds.

Algorithm
---------
1. Compute a composite regime score ∈ [0, 1] each day from seven signals:
      - SPY 5-day momentum  (rolling percentile rank in 500-day window)
      - SPY 10-day momentum
      - SPY 20-day momentum
      - Market breadth: fraction of stocks above their 50-day SMA
      - Vol regime:     inverse pct-rank of SPY 20-day realised vol
      - RSI signal:     (RSI - 50) / 50, pct-ranked
      - Dispersion:     inverse pct-rank of cross-sectional 1-day return dispersion
   Each signal is computed as a rolling percentile rank (``rolling_pct_rank``)
   over the last ``REGIME_WINDOW`` calendar days so there is NO look-ahead.

2. Classify the regime using adaptive thresholds derived from the rolling
   distribution of the composite score itself:
      score < P33(history)   → BEAR   (fully out of market)
      P33 ≤ score ≤ P67      → NEUTRAL (1× leverage)
      score > P67(history)   → BULL   (vol-targeted leverage up to MAX_LEVERAGE)

3. Compute dynamic leverage:
      BEAR:    0.0×
      NEUTRAL: 1.0×
      BULL:    blend of regime leverage and vol-targeting:
               lev = 0.5 × BULL_BASE_LEVERAGE + 0.5 × (VOL_TARGET / realised_vol)
               clamped to [0, MAX_LEVERAGE]
   Additionally, drawdown-deleveraging is applied:
               if portfolio_dd > DRAWDOWN_DELEVER_THRESHOLD:
                   lev × (1 − portfolio_dd / max_dd)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from catboost_trader.config import (
    BULL_BASE_LEVERAGE,
    DRAWDOWN_DELEVER_THRESHOLD,
    MARKET_TICKER,
    MAX_LEVERAGE,
    REGIME_PERCENTILE_BEAR,
    REGIME_PERCENTILE_BULL,
    REGIME_WINDOW,
    VOL_TARGET,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_pct_rank(series: pd.Series, window: int, min_periods: int = 50) -> pd.Series:
    """Rolling percentile rank (no look-ahead).

    For each position t, returns the fraction of the prior ``window`` values
    that are ≤ series[t].  Values before ``min_periods`` are NaN.
    """
    def _rank_last(arr: np.ndarray) -> float:
        if len(arr) < min_periods:
            return np.nan
        return float(np.mean(arr[:-1] <= arr[-1]))

    return series.rolling(window, min_periods=min_periods).apply(_rank_last, raw=True)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100.0 - 100.0 / (1.0 + rs)


# ---------------------------------------------------------------------------
# Regime score computation
# ---------------------------------------------------------------------------

def build_regime_signals(
    panel: pd.DataFrame,
    market_ticker: str = MARKET_TICKER,
    window: int = REGIME_WINDOW,
) -> pd.DataFrame:
    """Pre-compute all regime signals for the full date range.

    Returns a date-indexed DataFrame with columns:
        score, regime_raw, signal_*, breadth, dispersion
    to be used inside the simulation loop (zero look-ahead since we always
    use the row for date t to make decisions after t's close).

    Parameters
    ----------
    panel:
        Multi-ticker daily panel with columns: date, ticker, close, d_sma_50.
    market_ticker:
        Ticker to use as the market proxy for momentum / vol signals (SPY).
    window:
        Rolling window in bars for percentile ranking.
    """
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    # ── Market-proxy series (SPY or fallback to equal-weight index) ──────
    spy = panel[panel["ticker"] == market_ticker].set_index("date").sort_index()
    if spy.empty:
        # Fallback: equal-weight daily return of all tickers
        spy = (
            panel.pivot_table(values="close", index="date", columns="ticker")
            .mean(axis=1)
            .rename("close")
            .to_frame()
        )

    close_spy = spy["close"]
    log_ret_spy = np.log(close_spy / close_spy.shift(1))

    # ── Signal 1-3: momentum percentile ranks ─────────────────────────────
    mom_5d = close_spy.pct_change(5)
    mom_10d = close_spy.pct_change(10)
    mom_20d = close_spy.pct_change(20)
    sig_mom5 = _rolling_pct_rank(mom_5d, window)
    sig_mom10 = _rolling_pct_rank(mom_10d, window)
    sig_mom20 = _rolling_pct_rank(mom_20d, window)

    # ── Signal 4: market breadth (% stocks above 50-day SMA) ─────────────
    if "d_sma_50" in panel.columns and "d_c_vs_sma_50" in panel.columns:
        above_50 = (
            panel.assign(above=(panel["d_c_vs_sma_50"] > 0).astype(float))
            .groupby("date")["above"]
            .mean()
        )
    else:
        # Compute from close vs 50-day rolling mean
        close_wide = panel.pivot_table(values="close", index="date", columns="ticker")
        sma50_wide = close_wide.rolling(50).mean()
        above_50 = (close_wide > sma50_wide).mean(axis=1)

    above_50 = above_50.reindex(close_spy.index, method="ffill")
    sig_breadth = _rolling_pct_rank(above_50, window)

    # ── Signal 5: volatility regime (inverse — low vol = bullish) ─────────
    vol_20 = log_ret_spy.rolling(20).std() * np.sqrt(252)
    sig_vol = 1.0 - _rolling_pct_rank(vol_20, window)

    # ── Signal 6: RSI signal ──────────────────────────────────────────────
    rsi = _rsi(close_spy, 14)
    rsi_centered = (rsi - 50) / 50  # maps RSI=50 → 0, RSI=100 → 1
    sig_rsi = _rolling_pct_rank(rsi_centered, window)

    # ── Signal 7: cross-sectional return dispersion (inverse) ─────────────
    dispersion = (
        panel.groupby("date")["close"]
        .apply(lambda x: x.pct_change().std() if len(x) > 1 else np.nan)
    )
    dispersion = dispersion.reindex(close_spy.index, method="ffill")
    sig_disp = 1.0 - _rolling_pct_rank(dispersion, window)

    # ── Composite score ───────────────────────────────────────────────────
    signals = pd.DataFrame(
        {
            "sig_mom5": sig_mom5,
            "sig_mom10": sig_mom10,
            "sig_mom20": sig_mom20,
            "sig_breadth": sig_breadth,
            "sig_vol": sig_vol,
            "sig_rsi": sig_rsi,
            "sig_disp": sig_disp,
        }
    )
    score = signals.mean(axis=1)  # equal-weight composite

    result = signals.copy()
    result["score"] = score
    result.index.name = "date"
    return result


def classify_regime(score: float, score_history: pd.Series) -> str:
    """Classify regime using adaptive P33/P67 thresholds from rolling history.

    Parameters
    ----------
    score:
        Today's composite regime score.
    score_history:
        Series of past scores (today excluded) used to compute thresholds.

    Returns
    -------
    ``"BEAR"``, ``"NEUTRAL"``, or ``"BULL"``.
    """
    valid = score_history.dropna()
    if len(valid) < 50:
        return "NEUTRAL"

    p33 = float(np.percentile(valid, REGIME_PERCENTILE_BEAR))
    p67 = float(np.percentile(valid, REGIME_PERCENTILE_BULL))

    if score <= p33:
        return "BEAR"
    if score >= p67:
        return "BULL"
    return "NEUTRAL"


def get_leverage(
    regime: str,
    realised_vol: float,
    portfolio_dd: float = 0.0,
) -> float:
    """Compute dynamic leverage based on regime, volatility, and drawdown.

    Parameters
    ----------
    regime:
        ``"BEAR"``, ``"NEUTRAL"``, or ``"BULL"``.
    realised_vol:
        Recent annualised realised volatility of the portfolio (e.g. 20-day).
        Must be > 0.
    portfolio_dd:
        Current portfolio drawdown as a positive fraction (0 = no drawdown,
        0.2 = 20 % below peak).

    Returns
    -------
    float
        Gross leverage to apply, clamped to [0, MAX_LEVERAGE].
    """
    if regime == "BEAR":
        return 0.0

    if regime == "NEUTRAL":
        base_lev = 1.0
    else:  # BULL
        # Blend regime baseline with vol-targeted leverage
        if realised_vol > 0:
            vol_target_lev = VOL_TARGET / realised_vol
        else:
            vol_target_lev = MAX_LEVERAGE
        base_lev = 0.5 * BULL_BASE_LEVERAGE + 0.5 * vol_target_lev

    leverage = min(base_lev, MAX_LEVERAGE)

    # Drawdown deleveraging
    if portfolio_dd > DRAWDOWN_DELEVER_THRESHOLD and portfolio_dd < 1.0:
        scale = 1.0 - (portfolio_dd / 1.0)   # linear scale down to zero
        leverage *= max(scale, 0.0)

    return max(0.0, min(leverage, MAX_LEVERAGE))
