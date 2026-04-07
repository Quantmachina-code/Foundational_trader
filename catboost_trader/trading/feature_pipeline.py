"""Exact replica of the Jupyter notebook's feature engineering pipeline.

This module replicates Cells 3-5 of the notebook (data cleaning + Alpha
Factory feature engineering + cross-sectional quantile scaling) so that
the EOD bot produces feature vectors identical to those used for training.

Usage
-----
    from catboost_trader.trading.feature_pipeline import build_features_for_prediction

    # df_raw: raw sp500_daily.parquet loaded as a DataFrame
    df_scaled = build_features_for_prediction(df_raw)

    # Extract latest date's feature matrix
    last_date = df_scaled["date"].max()
    today_df  = df_scaled[df_scaled["date"] == last_date]

The returned df_scaled has the same columns as the notebook's df_scaled,
ready to pass to the trained imputer/selector/model pipeline.

Notes
-----
- Graph/correlation-neighbourhood features (Section L of notebook Cell 4)
  are expensive and SKIPPED here; the fitted imputer handles the resulting
  NaNs via median imputation.
- The function loads the FULL historical panel so rolling features are
  computed correctly (252-day windows require 1+ year of history).
"""

from __future__ import annotations

import logging
import time as _time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_features_for_prediction(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply notebook Cell 3+4+5 pipeline to raw sp500_daily.parquet data.

    Parameters
    ----------
    df_raw:
        Raw panel as returned by the FMP downloader: columns include
        date, ticker, open, high, low, close, adjClose, volume,
        d_* indicators, and optional fundamental columns.

    Returns
    -------
    pd.DataFrame
        Same rows as df_raw (minus duplicates/bad prices) with:
        - All original columns preserved
        - af_* feature columns added
        - All feature columns cross-sectionally quantile-scaled to [0, 1]
    """
    t0 = _time.time()

    # ── Cell 3: Clean ─────────────────────────────────────────────────────
    df = _clean(df_raw)

    # ── Cell 4: Alpha Factory ─────────────────────────────────────────────
    df = _alpha_factory(df)

    # ── Cell 5: Quantile scaling ──────────────────────────────────────────
    af_cols   = sorted(c for c in df.columns if c.startswith("af_"))
    tech_cols = [c for c in df.columns if c.startswith("d_")]
    fund_cols = _fund_cols_present(df)
    all_feat  = tech_cols + fund_cols + af_cols

    df_scaled = _quantile_scale(df, all_feat)

    logger.info(
        "feature_pipeline: %d rows | %d tickers | %d features | %.0fs",
        len(df_scaled),
        df_scaled["ticker"].nunique(),
        len(all_feat),
        _time.time() - t0,
    )
    return df_scaled


# ---------------------------------------------------------------------------
# Cell 3: Data cleaning
# ---------------------------------------------------------------------------

def _clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Drop duplicate (date, ticker) — keep last
    df = (
        df.sort_values(["date", "ticker"])
          .drop_duplicates(subset=["date", "ticker"], keep="last")
    )

    # Remove zero / negative adjusted close prices
    price_col = "adjClose" if "adjClose" in df.columns else "close"
    df = df[df[price_col] > 0]

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Cell 4: Alpha Factory
# ---------------------------------------------------------------------------

def _alpha_factory(df: pd.DataFrame) -> pd.DataFrame:
    price_col = "adjClose" if "adjClose" in df.columns else "close"
    g = df.groupby("ticker")

    # ── A. Multi-horizon momentum ─────────────────────────────────────────
    for d in [1, 2, 3, 5, 10, 20, 60, 120, 252]:
        df[f"af_ret_{d}d"] = g[price_col].transform(lambda x: x.pct_change(d))

    df["af_mom_accel_5_20"]   = df["af_ret_5d"]  - df["af_ret_20d"]
    df["af_mom_accel_20_60"]  = df["af_ret_20d"] - df["af_ret_60d"]
    df["af_mom_accel_60_120"] = df["af_ret_60d"] - df["af_ret_120d"]
    df["af_ret_1d_vs_20d"]    = df["af_ret_1d"]  - df["af_ret_20d"]
    df["af_ret_5d_vs_60d"]    = df["af_ret_5d"]  - df["af_ret_60d"]

    for d in [1, 2, 3, 5, 10, 20, 60]:
        df[f"af_logret_{d}d"] = g[price_col].transform(
            lambda x: np.log(x / x.shift(d)))

    # ── B. Volatility structure ───────────────────────────────────────────
    log_ret_col = "d_log_ret_1"
    if log_ret_col not in df.columns:
        df[log_ret_col] = g[price_col].transform(lambda x: np.log(x / x.shift(1)))

    for w in [5, 10, 20, 60, 120]:
        df[f"af_vol_{w}d"] = g[log_ret_col].transform(
            lambda x: x.rolling(w, min_periods=max(w // 2, 2)).std())

    df["af_vol_ratio_5_20"]   = df["af_vol_5d"]  / (df["af_vol_20d"]  + 1e-10)
    df["af_vol_ratio_10_60"]  = df["af_vol_10d"] / (df["af_vol_60d"]  + 1e-10)
    df["af_vol_ratio_20_60"]  = df["af_vol_20d"] / (df["af_vol_60d"]  + 1e-10)
    df["af_vol_ratio_20_120"] = df["af_vol_20d"] / (df["af_vol_120d"] + 1e-10)

    df["_neg_ret"] = df[log_ret_col].clip(upper=0)
    df["_pos_ret"] = df[log_ret_col].clip(lower=0)
    df["af_downvol_20"] = g["_neg_ret"].transform(lambda x: x.rolling(20, min_periods=10).std())
    df["af_upvol_20"]   = g["_pos_ret"].transform(lambda x: x.rolling(20, min_periods=10).std())
    df["af_vol_skew"]   = df["af_downvol_20"] / (df["af_upvol_20"] + 1e-10)
    df.drop(columns=["_neg_ret", "_pos_ret"], inplace=True)

    for w in [20, 60]:
        df[f"af_skew_{w}d"] = g[log_ret_col].transform(
            lambda x: x.rolling(w, min_periods=w // 2).skew())
        df[f"af_kurt_{w}d"] = g[log_ret_col].transform(
            lambda x: x.rolling(w, min_periods=w // 2).kurt())

    # ── C. Price structure ────────────────────────────────────────────────
    for w in [20, 60, 120, 252]:
        hi = g["high"].transform(lambda x: x.rolling(w, min_periods=w // 2).max())
        lo = g["low"].transform(lambda x: x.rolling(w, min_periods=w // 2).min())
        df[f"af_pct_from_high_{w}d"] = df[price_col] / (hi + 1e-10) - 1
        df[f"af_pct_from_low_{w}d"]  = df[price_col] / (lo + 1e-10) - 1
        df[f"af_williams_{w}d"]      = (hi - df[price_col]) / (hi - lo + 1e-10)

    df["af_intraday_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-10)
    df["af_body_range"]     = (df["close"] - df["open"]).abs() / (df["high"] - df["low"] + 1e-10)
    df["af_upper_shadow"]   = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["high"] - df["low"] + 1e-10)
    df["af_lower_shadow"]   = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-10)

    df["af_gap"] = (
        g.apply(lambda x: x["open"] / x["close"].shift(1) - 1)
         .reset_index(level=0, drop=True)
    )
    df["af_avg_range_20"] = g["af_intraday_range"].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    df["af_range_vs_avg"] = df["af_intraday_range"] / (df["af_avg_range_20"] + 1e-10)

    # ── D. Volume & liquidity ─────────────────────────────────────────────
    for w in [5, 10, 20, 60]:
        df[f"_vol_ma_{w}d"] = g["volume"].transform(
            lambda x: x.rolling(w, min_periods=w // 2).mean())

    df["af_vol_spike_5_20"]  = df["volume"] / (df["_vol_ma_20d"] + 1)
    df["af_vol_ratio_5_60"]  = df["_vol_ma_5d"] / (df["_vol_ma_60d"] + 1)
    df["af_dollar_vol"]      = df[price_col] * df["volume"]
    df["af_dollar_vol_20"]   = g["af_dollar_vol"].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    df["af_illiquidity"]     = df[log_ret_col].abs() / (df["af_dollar_vol"] + 1)
    df["af_illiquidity_20"]  = g["af_illiquidity"].transform(
        lambda x: x.rolling(20, min_periods=10).mean())

    df["af_vwap_ratio"] = df[price_col] / (
        g.apply(
            lambda x: (x[price_col] * x["volume"]).rolling(20, min_periods=10).sum()
            / x["volume"].rolling(20, min_periods=10).sum()
        ).reset_index(level=0, drop=True) + 1e-10
    )

    df["_obv"] = (
        g.apply(lambda x: (x["volume"] * np.sign(x[log_ret_col])).cumsum())
         .reset_index(level=0, drop=True)
    )
    df["af_obv_slope_20"] = g["_obv"].transform(
        lambda x: x.diff(20) / (x.rolling(20).std() + 1e-10))
    df.drop(columns=["_obv"], inplace=True)

    df["af_vol_chg_5"]  = g["volume"].transform(lambda x: x.pct_change(5))
    df["af_vol_chg_20"] = g["volume"].transform(lambda x: x.pct_change(20))

    # drop intermediates
    df.drop(columns=[c for c in df.columns if c.startswith("_vol_ma_") or c == "af_dollar_vol"],
            inplace=True)

    # ── E. Lagged & rolling features ─────────────────────────────────────
    g = df.groupby("ticker")
    lag_feats = [c for c in ["d_rsi_14", "d_macd_norm", "d_vol_20",
                              "af_ret_1d", "af_ret_5d", "af_vol_spike_5_20"]
                 if c in df.columns]
    for feat in lag_feats:
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f"af_lag_{feat}_L{lag}"] = g[feat].transform(lambda x: x.shift(lag))

    roll_feats = [c for c in ["d_rsi_14", "d_macd_norm", "af_ret_1d"] if c in df.columns]
    for feat in roll_feats:
        for w in [5, 10, 20]:
            df[f"af_rmean_{feat}_{w}d"] = g[feat].transform(
                lambda x: x.rolling(w, min_periods=w // 2).mean())
            df[f"af_rstd_{feat}_{w}d"]  = g[feat].transform(
                lambda x: x.rolling(w, min_periods=w // 2).std())
        for w in [20, 60]:
            mean = g[feat].transform(lambda x: x.rolling(w, min_periods=w // 2).mean())
            std  = g[feat].transform(lambda x: x.rolling(w, min_periods=w // 2).std())
            df[f"af_zscore_{feat}_{w}d"] = (df[feat] - mean) / (std + 1e-10)

    # ── F. Fundamental transformations ────────────────────────────────────
    if "netIncome" in df.columns and "totalAssets" in df.columns:
        df["af_roa"] = df["netIncome"] / (df["totalAssets"] + 1e-10)
    if "netIncome" in df.columns and "totalStockholdersEquity" in df.columns:
        df["af_roe"] = df["netIncome"] / (df["totalStockholdersEquity"] + 1e-10)
    if "ebitda" in df.columns and "totalAssets" in df.columns:
        df["af_rota"] = df["ebitda"] / (df["totalAssets"] + 1e-10)
    if "revenue" in df.columns and "totalAssets" in df.columns:
        df["af_asset_turnover"] = df["revenue"] / (df["totalAssets"] + 1e-10)
    if "operatingCashFlow" in df.columns and "revenue" in df.columns:
        df["af_cash_conversion"] = df["operatingCashFlow"] / (df["revenue"] + 1e-10)
    if "totalDebt" in df.columns and "ebitda" in df.columns:
        df["af_debt_to_ebitda"] = df["totalDebt"] / (df["ebitda"] + 1e-10)
    if "freeCashFlow" in df.columns and "totalDebt" in df.columns:
        df["af_fcf_to_debt"] = df["freeCashFlow"] / (df["totalDebt"] + 1e-10)
    if "eps" in df.columns:
        df["af_ep_ratio"] = df["eps"] / (df[price_col] + 1e-10)

    g = df.groupby("ticker")
    growth_cols = [c for c in ["revenue", "netIncome", "grossProfit", "ebitda",
                                "operatingCashFlow", "freeCashFlow", "eps"]
                   if c in df.columns]
    for col in growth_cols:
        shifted = g[col].transform(lambda x: x.shift(252))
        df[f"af_growth_{col}"] = (df[col] - shifted) / (shifted.abs() + 1e-10)

    chg_cols = [c for c in ["netMargin", "grossMargin", "cfMargin",
                              "debtToEquity", "debtToAssets"] if c in df.columns]
    for col in chg_cols:
        df[f"af_chg_q_{col}"] = g[col].transform(lambda x: x.diff(63))
        df[f"af_chg_y_{col}"] = g[col].transform(lambda x: x.diff(252))

    if all(c in df.columns for c in ["netIncome", "operatingCashFlow", "totalAssets"]):
        df["af_accruals"] = (
            (df["netIncome"] - df["operatingCashFlow"]) / (df["totalAssets"] + 1e-10)
        )

    # ── G. Market / regime features ───────────────────────────────────────
    for col in ["mkt_ret_1d", "_stock_x_mkt", "_mkt_sq"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    market_daily = df.groupby("date").agg(
        mkt_ret_1d  = (log_ret_col, "mean"),
        mkt_ret_std = (log_ret_col, "std"),
        n_stocks    = ("ticker", "count"),
    ).reset_index()

    for w in [5, 10, 20, 60]:
        market_daily[f"af_mkt_ret_{w}d"] = (
            market_daily["mkt_ret_1d"].rolling(w, min_periods=w // 2).sum())
    for w in [10, 20, 60]:
        market_daily[f"af_mkt_vol_{w}d"] = (
            market_daily["mkt_ret_1d"].rolling(w, min_periods=w // 2).std())
    market_daily["af_mkt_vol_regime"] = (
        market_daily["af_mkt_vol_20d"] / (market_daily["af_mkt_vol_60d"] + 1e-10))

    breadth = (
        df.groupby("date")[log_ret_col]
          .apply(lambda x: (x > 0).mean())
          .reset_index()
    )
    breadth.columns = ["date", "af_mkt_breadth"]
    market_daily = market_daily.merge(breadth, on="date", how="left")
    for w in [5, 20]:
        market_daily[f"af_mkt_breadth_{w}d"] = (
            market_daily["af_mkt_breadth"].rolling(w, min_periods=w // 2).mean())

    market_daily["af_mkt_dispersion"]     = market_daily["mkt_ret_std"]
    market_daily["af_mkt_dispersion_20d"] = (
        market_daily["mkt_ret_std"].rolling(20, min_periods=10).mean())

    mkt_feat_cols = [c for c in market_daily.columns if c.startswith("af_mkt_")]
    merge_cols = ["date", "mkt_ret_1d"] + mkt_feat_cols
    df = df.merge(market_daily[merge_cols], on="date", how="left")

    df["af_excess_vs_mkt_1d"] = df[log_ret_col] - df["mkt_ret_1d"]

    # Rolling beta
    df["_stock_x_mkt"] = df[log_ret_col] * df["mkt_ret_1d"]
    df["_mkt_sq"]      = df["mkt_ret_1d"] ** 2
    g = df.groupby("ticker")

    for w in [60, 120]:
        cov_sm  = g["_stock_x_mkt"].transform(lambda x: x.rolling(w, min_periods=w // 2).mean())
        mean_s  = g[log_ret_col].transform(lambda x: x.rolling(w, min_periods=w // 2).mean())
        var_m   = g["_mkt_sq"].transform(lambda x: x.rolling(w, min_periods=w // 2).mean())
        mean_m  = g["mkt_ret_1d"].transform(lambda x: x.rolling(w, min_periods=w // 2).mean())
        cov     = cov_sm - mean_s * mean_m
        var_mkt = var_m - mean_m ** 2
        df[f"af_beta_{w}d"] = cov / (var_mkt + 1e-10)

    df["af_idio_ret_1d"]  = df[log_ret_col] - df["af_beta_60d"] * df["mkt_ret_1d"]
    g = df.groupby("ticker")
    df["af_idio_vol_20d"] = g["af_idio_ret_1d"].transform(
        lambda x: x.rolling(20, min_periods=10).std())

    df.drop(columns=["_stock_x_mkt", "_mkt_sq", "mkt_ret_1d"], inplace=True)
    g = df.groupby("ticker")

    # ── H. Interaction features ───────────────────────────────────────────
    if "af_ret_5d" in df.columns and "af_vol_20d" in df.columns:
        df["af_ix_mom5_vol20"]    = df["af_ret_5d"]  * df["af_vol_20d"]
        df["af_ix_mom20_vol60"]   = df["af_ret_20d"] * df["af_vol_60d"]
        df["af_ix_mom5_volratio"] = df["af_ret_5d"]  * df["af_vol_ratio_5_20"]
    if "af_vol_spike_5_20" in df.columns:
        df["af_ix_mom5_volspike"] = df["af_ret_5d"] * df["af_vol_spike_5_20"]
        df["af_ix_mom1_volspike"] = df["af_ret_1d"] * df["af_vol_spike_5_20"]
    if "d_rsi_14" in df.columns:
        df["af_ix_rsi_mom5"]  = df["d_rsi_14"] * df["af_ret_5d"]
        df["af_ix_rsi_mom20"] = df["d_rsi_14"] * df["af_ret_20d"]
        df["af_rsi_extreme"]  = ((df["d_rsi_14"] > 0.8) | (df["d_rsi_14"] < 0.2)).astype(np.float32)
    if "af_ep_ratio" in df.columns:
        df["af_ix_value_mom20"] = df["af_ep_ratio"] * df["af_ret_20d"]
        df["af_ix_value_mom60"] = df["af_ep_ratio"] * df["af_ret_60d"]
    if "af_roa" in df.columns:
        df["af_ix_quality_mom20"] = df["af_roa"] * df["af_ret_20d"]
    if "af_beta_60d" in df.columns and "af_mkt_ret_20d" in df.columns:
        df["af_ix_beta_mktmom"] = df["af_beta_60d"] * df["af_mkt_ret_20d"]
    if "d_macd_norm" in df.columns and "af_vol_spike_5_20" in df.columns:
        df["af_ix_macd_volspike"] = df["d_macd_norm"] * df["af_vol_spike_5_20"]

    # ── I. TS rank features ───────────────────────────────────────────────
    g = df.groupby("ticker")
    ts_rank_feats = [c for c in ["af_ret_5d", "af_ret_20d", "af_vol_20d", "d_rsi_14"]
                     if c in df.columns]
    for feat in ts_rank_feats:
        for w in [60, 252]:
            df[f"af_tsrank_{feat}_{w}d"] = g[feat].transform(
                lambda x: x.rolling(w, min_periods=w // 2).rank(pct=True))

    # ── J. Calendar features ──────────────────────────────────────────────
    df["af_day_of_week"]       = df["date"].dt.dayofweek.astype(np.float32) / 4.0
    df["af_day_of_month"]      = df["date"].dt.day.astype(np.float32) / 31.0
    df["af_month"]             = df["date"].dt.month.astype(np.float32) / 12.0
    df["af_quarter"]           = df["date"].dt.quarter.astype(np.float32) / 4.0
    df["af_days_to_month_end"] = (
        (df["date"].dt.days_in_month - df["date"].dt.day).astype(np.float32) / 31.0
    )
    df["af_is_january"]   = (df["date"].dt.month == 1).astype(np.float32)
    df["af_turn_of_month"]= (
        (df["date"].dt.day <= 2) | (df["date"].dt.day >= df["date"].dt.days_in_month - 1)
    ).astype(np.float32)

    # ── K. Cross-sectional features ───────────────────────────────────────
    cs_feats = [c for c in [log_ret_col, "d_rsi_14", "d_vol_20", "d_macd_norm",
                              "af_ret_1d", "af_ret_5d", "af_ret_20d", "af_ret_60d",
                              "af_vol_5d", "af_vol_20d", "af_vol_60d"]
                if c in df.columns]
    for feat in cs_feats:
        cs_mean = df.groupby("date")[feat].transform("mean")
        cs_std  = df.groupby("date")[feat].transform("std")
        df[f"af_cs_zscore_{feat}"] = (df[feat] - cs_mean) / (cs_std + 1e-10)
        df[f"af_cs_rank_{feat}"]   = df.groupby("date")[feat].transform(
            lambda x: x.rank(pct=True))

    key_cs = [c for c in [log_ret_col, "d_rsi_14", "af_ret_20d", "af_vol_20d"]
              if c in df.columns][:4]
    for feat in key_cs:
        cs_median = df.groupby("date")[feat].transform("median")
        cs_q25    = df.groupby("date")[feat].transform(lambda x: x.quantile(0.25))
        cs_q75    = df.groupby("date")[feat].transform(lambda x: x.quantile(0.75))
        iqr       = cs_q75 - cs_q25
        df[f"af_cs_dist_median_{feat}"] = df[feat] - cs_median
        df[f"af_cs_iqr_pos_{feat}"]     = (df[feat] - cs_median) / (iqr + 1e-10)

    for cand in ["af_ret_5d", "af_ret_20d"]:
        if cand in df.columns:
            df[f"af_cs_dispersion_{cand}"] = df.groupby("date")[cand].transform("std")

    # ── L. Graph features — SKIPPED (set to NaN; imputer handles them) ────
    # af_graph_nb_ret_1d, af_graph_nb_corr, af_graph_ret_vs_nb_1d
    # af_graph_ret_vs_nb_1d_L1, af_graph_ret_vs_nb_1d_L5, af_graph_nb_corr_chg

    # ── Final cleanup ─────────────────────────────────────────────────────
    af_cols = sorted(c for c in df.columns if c.startswith("af_"))
    # Replace inf with NaN
    for col in af_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    # Winsorise at 0.1% / 99.9%
    for col in af_cols:
        vals = df[col].dropna()
        if len(vals) > 100:
            lo, hi = vals.quantile(0.001), vals.quantile(0.999)
            df[col] = df[col].clip(lo, hi)

    return df


# ---------------------------------------------------------------------------
# Cell 5: Cross-sectional quantile scaling
# ---------------------------------------------------------------------------

def _quantile_scale(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Cross-sectionally rank each feature to [0, 1] per date (NaN preserved)."""
    df_scaled = df.copy()
    valid_cols = [c for c in feature_cols if c in df_scaled.columns]

    # Replace inf before ranking
    for col in valid_cols:
        if df_scaled[col].dtype.kind == "f":
            mask = np.isinf(df_scaled[col])
            if mask.any():
                df_scaled.loc[mask, col] = np.nan

    for col in valid_cols:
        df_scaled[col] = df_scaled.groupby("date")[col].rank(
            method="average", pct=True, na_option="keep"
        )
    return df_scaled


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FUND_COLS_CANDIDATE = [
    "netMargin", "grossMargin", "cfMargin",
    "debtToEquity", "debtToAssets",
    "revenue", "netIncome", "grossProfit",
    "operatingIncome", "ebitda", "eps", "epsDiluted",
    "cashAndCashEquivalents", "totalAssets", "totalDebt",
    "netDebt", "totalStockholdersEquity", "totalLiabilities",
    "operatingCashFlow", "freeCashFlow", "capitalExpenditure",
]


def _fund_cols_present(df: pd.DataFrame) -> list[str]:
    return [c for c in FUND_COLS_CANDIDATE if c in df.columns]
