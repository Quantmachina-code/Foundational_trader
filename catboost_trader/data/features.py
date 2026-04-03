"""Feature engineering for the CatBoost trading model.

Two feature sets are supported:

ORIGINAL
    Only the ``d_*`` columns already present in the downloaded parquet
    (SMA, EMA, RSI, MACD, log-return, realised-vol).  No additional
    computation beyond what ``compute_daily_indicators`` already provides.

ALPHA (alpha-factory)
    All ORIGINAL features plus 200+ additional ``af_*`` features:
    multi-period returns, ATR, ADX, Stochastic, Williams %R, CCI,
    Bollinger Bands, OBV, MFI, volume z-score, cross-sectional z-scores,
    lagged features, calendar dummies, and interaction terms.

Cross-sectional quantile scaling
    Every feature is rank-scaled to [0, 1] via ``rank(pct=True)`` across
    all tickers for each date.  This eliminates absolute-value look-ahead
    bias because the ranking is purely relative within each day's universe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from catboost_trader.config import FEATURE_SET_ALPHA, FEATURE_SET_ORIGINAL


# ---------------------------------------------------------------------------
# Low-level indicator helpers (single-ticker time-series, daily bars)
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index (ADX)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    atr = _atr(high, low, close, period)
    plus_di = (
        pd.Series(plus_dm, index=close.index).ewm(com=period - 1, adjust=False).mean()
        / atr.replace(0, float("nan")) * 100
    )
    minus_di = (
        pd.Series(minus_dm, index=close.index).ewm(com=period - 1, adjust=False).mean()
        / atr.replace(0, float("nan")) * 100
    )
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan")) * 100
    return dx.ewm(com=period - 1, adjust=False).mean()


def _compute_af_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Alpha-Factory (af_*) features for a single-ticker daily DataFrame.

    Parameters
    ----------
    df:
        Daily OHLCV DataFrame indexed by date with columns:
        open, high, low, close, volume, (d_* already present).

    Returns
    -------
    Same DataFrame with af_* columns appended.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df.get("volume", pd.Series(np.nan, index=df.index))

    # ── Multi-period returns ──────────────────────────────────────────────
    for n in [2, 3, 5, 10, 20, 60]:
        df[f"af_ret_{n}d"] = close.pct_change(n)
        df[f"af_log_ret_{n}d"] = np.log(close / close.shift(n))

    # 12-month return excluding last month (momentum factor)
    df["af_mom_12_1"] = close.pct_change(252) - close.pct_change(21)

    # ── Volatility ────────────────────────────────────────────────────────
    log1 = np.log(close / close.shift(1))
    for n in [5, 10, 20, 60]:
        df[f"af_vol_{n}d"] = log1.rolling(n).std() * np.sqrt(252)

    df["af_atr_14"] = _atr(high, low, close, 14) / close
    df["af_adx_14"] = _adx(high, low, close, 14)

    # ── Price structure ───────────────────────────────────────────────────
    # Stochastic %K / %D
    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    stoch_k = (close - lowest_low) / (highest_high - lowest_low).replace(0, float("nan")) * 100
    df["af_stoch_k"] = stoch_k
    df["af_stoch_d"] = stoch_k.rolling(3).mean()

    # Williams %R
    df["af_williams_r"] = (highest_high - close) / (highest_high - lowest_low).replace(0, float("nan")) * -100

    # CCI (Commodity Channel Index)
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(20).mean()
    mean_dev = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["af_cci_20"] = (typical_price - sma_tp) / (0.015 * mean_dev.replace(0, float("nan")))

    # Rate of change
    for n in [5, 10, 20, 60]:
        df[f"af_roc_{n}d"] = close / close.shift(n) - 1

    # Bollinger Bands (%B and width)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    band_width = (bb_upper - bb_lower).replace(0, float("nan"))
    df["af_bb_pct_20"] = (close - bb_lower) / band_width
    df["af_bb_width_20"] = band_width / sma20.replace(0, float("nan"))

    # Price vs 52-week high/low
    df["af_vs_52w_high"] = close / high.rolling(252).max() - 1
    df["af_vs_52w_low"] = close / low.rolling(252).min() - 1

    # ── Volume ────────────────────────────────────────────────────────────
    if not volume.isna().all():
        # OBV
        price_dir = np.sign(close.diff())
        df["af_obv"] = (volume * price_dir).cumsum()
        df["af_obv_norm"] = df["af_obv"] / df["af_obv"].abs().rolling(20).max().replace(0, float("nan"))

        # Volume z-score
        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std()
        df["af_vol_zscore"] = (volume - vol_mean) / vol_std.replace(0, float("nan"))
        df["af_vol_ratio_5"] = volume / volume.rolling(5).mean().replace(0, float("nan"))
        df["af_vol_ratio_20"] = volume / volume.rolling(20).mean().replace(0, float("nan"))

        # Money Flow Index
        typical_price2 = (high + low + close) / 3
        raw_mf = typical_price2 * volume
        pos_mf = raw_mf.where(typical_price2 > typical_price2.shift(1), 0.0).rolling(14).sum()
        neg_mf = raw_mf.where(typical_price2 < typical_price2.shift(1), 0.0).rolling(14).sum()
        df["af_mfi_14"] = 100 - (100 / (1 + pos_mf / neg_mf.replace(0, float("nan"))))

    # ── Lagged key signals (1-day lag) ────────────────────────────────────
    for col in ["d_rsi_14", "d_macd_hist", "af_ret_5d", "af_vol_20d"]:
        if col in df.columns:
            df[f"af_lag1_{col}"] = df[col].shift(1)

    # ── Calendar features ─────────────────────────────────────────────────
    df["af_day_of_week"] = df.index.dayofweek.astype(float)
    df["af_month"] = df.index.month.astype(float)
    df["af_quarter"] = df.index.quarter.astype(float)

    # Month-end / month-start dummy
    df["af_is_month_end"] = df.index.is_month_end.astype(float)
    df["af_is_month_start"] = df.index.is_month_start.astype(float)

    # ── Interaction features ──────────────────────────────────────────────
    if "af_vol_20d" in df.columns and "af_ret_20d" in df.columns:
        df["af_vol_x_mom20"] = df["af_vol_20d"] * df["af_ret_20d"]
    if "d_rsi_14" in df.columns and "af_ret_5d" in df.columns:
        df["af_rsi_x_mom5"] = df["d_rsi_14"] / 100.0 * df["af_ret_5d"]
    if "af_vol_20d" in df.columns and "d_rsi_14" in df.columns:
        df["af_vol_x_rsi"] = df["af_vol_20d"] * (df["d_rsi_14"] / 100.0 - 0.5)

    return df


# ---------------------------------------------------------------------------
# Cross-sectional rank scaling
# ---------------------------------------------------------------------------

def _cross_section_rank(panel: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Rank-scale features cross-sectionally per date to [0, 1].

    For each date, applies ``rank(pct=True)`` across all tickers within
    each feature column.  NaNs are left as NaN (not imputed here).

    Uses ``transform`` (pandas 3.0 compatible) rather than ``apply`` so that
    the ``date`` column is never consumed as a group key.
    """
    panel = panel.copy()
    for col in feature_cols:
        if col in panel.columns:
            panel[col] = panel.groupby("date")[col].transform(
                lambda s: s.rank(pct=True)
            )
    return panel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_matrix(
    panel: pd.DataFrame,
    feature_set: str = FEATURE_SET_ALPHA,
) -> pd.DataFrame:
    """Build the full feature matrix from the combined daily panel.

    Parameters
    ----------
    panel:
        Multi-ticker daily panel with ``date`` and ``ticker`` columns plus
        ``d_*`` indicator columns (output of ``ensure_data``).
    feature_set:
        ``"original"`` → use only existing ``d_*`` columns.
        ``"alpha"``    → compute additional ``af_*`` features.

    Returns
    -------
    pd.DataFrame
        Panel with same index as input plus feature columns rank-scaled
        cross-sectionally per date.  Non-feature columns (date, ticker,
        OHLCV) are retained unchanged.
    """
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    if feature_set not in (FEATURE_SET_ORIGINAL, FEATURE_SET_ALPHA):
        raise ValueError(f"feature_set must be 'original' or 'alpha', got {feature_set!r}")

    if feature_set == FEATURE_SET_ALPHA:
        # Apply af_* computation per ticker
        groups = []
        for ticker, grp in panel.groupby("ticker", sort=False):
            grp = grp.sort_values("date").copy()
            grp = grp.set_index("date")
            grp = _compute_af_features(grp)
            grp = grp.reset_index()
            groups.append(grp)
        panel = pd.concat(groups, ignore_index=True)

    # Identify feature columns
    from catboost_trader.models.feature_sets import (
        ALPHA_FACTORY_FEATURES,
        ORIGINAL_FEATURES,
    )

    if feature_set == FEATURE_SET_ORIGINAL:
        feat_cols = [c for c in ORIGINAL_FEATURES if c in panel.columns]
    else:
        feat_cols = [c for c in ALPHA_FACTORY_FEATURES if c in panel.columns]

    # Cross-sectional rank scaling
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    panel = _cross_section_rank(panel, feat_cols)

    return panel


def get_feature_columns(panel: pd.DataFrame, feature_set: str) -> list[str]:
    """Return the list of feature column names present in *panel*."""
    from catboost_trader.models.feature_sets import (
        ALPHA_FACTORY_FEATURES,
        ORIGINAL_FEATURES,
    )

    if feature_set == FEATURE_SET_ORIGINAL:
        return [c for c in ORIGINAL_FEATURES if c in panel.columns]
    return [c for c in ALPHA_FACTORY_FEATURES if c in panel.columns]
