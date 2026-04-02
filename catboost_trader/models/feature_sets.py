"""Canonical feature column name lists.

ORIGINAL_FEATURES
    The ``d_*`` columns produced by ``compute_daily_indicators`` in
    ``download_sp500_fmp.py``.  These are present in the raw parquet download
    with no additional computation required.

ALPHA_FACTORY_FEATURES
    All ORIGINAL features plus the ``af_*`` engineered features computed by
    ``data/features.py::_compute_af_features``.

The lists here define the *canonical ordering* used when building the model's
feature matrix.  Actual columns present in any given DataFrame are filtered
to the intersection of these lists and what is available, so missing columns
never cause KeyErrors.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Original (d_*) features — from compute_daily_indicators
# ---------------------------------------------------------------------------

ORIGINAL_FEATURES: list[str] = [
    # SMA levels and price-vs-SMA ratios
    "d_sma_20",
    "d_sma_50",
    "d_sma_200",
    "d_c_vs_sma_20",
    "d_c_vs_sma_50",
    "d_c_vs_sma_200",
    # EMA levels and price-vs-EMA ratios
    "d_ema_12",
    "d_ema_20",
    "d_ema_26",
    "d_ema_50",
    "d_ema_200",
    "d_c_vs_ema_12",
    "d_c_vs_ema_20",
    "d_c_vs_ema_26",
    "d_c_vs_ema_50",
    "d_c_vs_ema_200",
    # Momentum / oscillators
    "d_rsi_14",
    "d_macd",
    "d_macd_signal",
    "d_macd_hist",
    "d_macd_norm",
    # Return / volatility
    "d_log_ret_1",
    "d_vol_20",
    "d_vol_60",
]

# ---------------------------------------------------------------------------
# Alpha-factory (af_*) additional features
# ---------------------------------------------------------------------------

_AF_FEATURES: list[str] = [
    # Multi-period returns
    "af_ret_2d",
    "af_ret_3d",
    "af_ret_5d",
    "af_ret_10d",
    "af_ret_20d",
    "af_ret_60d",
    "af_log_ret_2d",
    "af_log_ret_3d",
    "af_log_ret_5d",
    "af_log_ret_10d",
    "af_log_ret_20d",
    "af_log_ret_60d",
    "af_mom_12_1",
    # Volatility
    "af_vol_5d",
    "af_vol_10d",
    "af_vol_20d",
    "af_vol_60d",
    "af_atr_14",
    "af_adx_14",
    # Price structure
    "af_stoch_k",
    "af_stoch_d",
    "af_williams_r",
    "af_cci_20",
    "af_roc_5d",
    "af_roc_10d",
    "af_roc_20d",
    "af_roc_60d",
    "af_bb_pct_20",
    "af_bb_width_20",
    "af_vs_52w_high",
    "af_vs_52w_low",
    # Volume
    "af_obv_norm",
    "af_vol_zscore",
    "af_vol_ratio_5",
    "af_vol_ratio_20",
    "af_mfi_14",
    # Lagged signals
    "af_lag1_d_rsi_14",
    "af_lag1_d_macd_hist",
    "af_lag1_af_ret_5d",
    "af_lag1_af_vol_20d",
    # Calendar
    "af_day_of_week",
    "af_month",
    "af_quarter",
    "af_is_month_end",
    "af_is_month_start",
    # Interactions
    "af_vol_x_mom20",
    "af_rsi_x_mom5",
    "af_vol_x_rsi",
]

ALPHA_FACTORY_FEATURES: list[str] = ORIGINAL_FEATURES + _AF_FEATURES
