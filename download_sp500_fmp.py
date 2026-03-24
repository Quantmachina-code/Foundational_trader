"""Download S&P 500 data at three frequencies from the FMP API.

Three parquet files are produced:

  sp500_daily.parquet
      One row per ticker per trading day.
      Columns: date, ticker, OHLCV, adjClose + daily technical indicators
      (d_* prefix) + quarterly fundamentals forward-filled to every day.

  sp500_weekly.parquet
      One row per ticker per week (Friday close).
      Columns: date, ticker, weekly OHLCV + end-of-week snapshot of daily
      indicators (d_* prefix) + weekly indicators computed on weekly bars
      (w_* prefix) + quarterly fundamentals forward-filled to every week.

  sp500_quarterly.parquet
      One row per ticker per quarter (shifted +1 quarter for look-ahead safety).
      Columns: date, ticker + raw income-statement / balance-sheet /
      cash-flow fields.  No forward-filling; sparse by design.

Quarterly shift
---------------
FMP reports Q1 data with date = 2023-03-31.  Because companies don't publish
results until weeks after quarter-end, we shift every quarter's date forward
by exactly 3 months (DateOffset(months=3)).  So Q1-2023 data only appears
in the daily/weekly datasets from 2023-06-30 onward — no look-ahead bias.

Resume support
--------------
Each ticker is cached in three sub-directories under --cache-dir:
  cache_dir/daily/{TICKER}.parquet
  cache_dir/weekly/{TICKER}.parquet
  cache_dir/quarterly/{TICKER}.parquet

A ticker is skipped on re-run only if ALL THREE cache files exist, so a
partial failure is always retried in full.

Usage
-----
  python download_sp500_fmp.py --api-key YOUR_KEY

FMP API docs: https://site.financialmodelingprep.com/developer/docs
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FMP_BASE = "https://financialmodelingprep.com/api/v3"

# ---------------------------------------------------------------------------
# Fundamental column lists (kept lean — drop anything not needed downstream)
# ---------------------------------------------------------------------------
INCOME_COLS = [
    "revenue",
    "costOfRevenue",
    "grossProfit",
    "grossProfitRatio",
    "operatingIncome",
    "operatingIncomeRatio",
    "ebitda",
    "netIncome",
    "netIncomeRatio",
    "eps",
    "epsDiluted",
    "weightedAverageShsOut",
    "weightedAverageShsOutDil",
]

BALANCE_COLS = [
    "cashAndCashEquivalents",
    "shortTermInvestments",
    "totalCurrentAssets",
    "totalAssets",
    "totalCurrentLiabilities",
    "totalLiabilities",
    "totalStockholdersEquity",
    "totalDebt",
    "netDebt",
    "retainedEarnings",
]

CASHFLOW_COLS = [
    "operatingCashFlow",
    "capitalExpenditure",
    "freeCashFlow",
    "dividendsPaid",
    "stockBasedCompensation",
    "changeInWorkingCapital",
]

STATEMENTS: list[tuple[str, list[str]]] = [
    ("income-statement", INCOME_COLS),
    ("balance-sheet-statement", BALANCE_COLS),
    ("cash-flow-statement", CASHFLOW_COLS),
]

# OHLCV columns and their resampling rules for weekly aggregation
_OHLCV_AGG: dict[str, str] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "adjClose": "last",
    "volume": "sum",
}


# ---------------------------------------------------------------------------
# FMP API helpers
# ---------------------------------------------------------------------------

def _fmp_get(endpoint: str, api_key: str, params: dict | None = None) -> list | dict:
    """GET from FMP; raises on non-200 or API error payload."""
    url = f"{FMP_BASE}/{endpoint}"
    p: dict = {"apikey": api_key}
    if params:
        p.update(params)
    resp = requests.get(url, params=p, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "Error Message" in data:
        raise RuntimeError(data["Error Message"])
    return data


def get_sp500_tickers(api_key: str) -> list[str]:
    """Return sorted list of current S&P 500 symbols from FMP."""
    data = _fmp_get("sp500_constituent", api_key)
    return sorted({row["symbol"] for row in data if row.get("symbol")})


# ---------------------------------------------------------------------------
# Raw data download
# ---------------------------------------------------------------------------

def _download_daily_raw(ticker: str, api_key: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV from FMP.  Returns DataFrame indexed by date."""
    data = _fmp_get(
        f"historical-price-full/{ticker}",
        api_key,
        {"from": start, "to": end},
    )
    if not data or "historical" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["historical"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    available = [c for c in _OHLCV_AGG if c in df.columns]
    return df[available].astype(float, errors="ignore")


def download_quarterly_fundamentals(
    ticker: str,
    api_key: str,
    delay: float,
) -> pd.DataFrame:
    """Fetch quarterly income-statement + balance-sheet + cash-flow from FMP.

    Returns a DataFrame indexed by the *shifted* quarter-end date
    (original date + 3 months).  This is the look-ahead-safe date at which
    each quarter's figures first become usable in a model.

    Example: Q1-2023 (2023-03-31) → shifted to 2023-06-30.
    """
    frames: list[pd.DataFrame] = []

    for statement, wanted_cols in STATEMENTS:
        try:
            raw = _fmp_get(
                f"{statement}/{ticker}",
                api_key,
                {"period": "quarter", "limit": 60},
            )
            time.sleep(delay)
        except Exception as exc:
            print(f"    [{statement}] skipped: {exc}")
            continue

        if not raw:
            continue

        df = pd.DataFrame(raw)
        if "date" not in df.columns:
            continue

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        available = [c for c in wanted_cols if c in df.columns]
        if available:
            frames.append(df[available])

    if not frames:
        return pd.DataFrame()

    # Outer-join all three statements on the quarter-end date index
    fundamentals = frames[0]
    for frame in frames[1:]:
        fundamentals = fundamentals.join(frame, how="outer")

    # Shift forward by one full quarter
    fundamentals.index = fundamentals.index + pd.DateOffset(months=3)
    fundamentals.index.name = "date"

    return fundamentals


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100.0 - 100.0 / (1.0 + rs)


def compute_daily_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    """Add daily technical indicators (d_* prefix) to a daily OHLCV DataFrame.

    Indicators
    ----------
    SMA 20 / 50 / 200d          d_sma_{n}, d_c_vs_sma_{n}
    EMA 12 / 20 / 26 / 50 / 200d  d_ema_{n}, d_c_vs_ema_{n}
    RSI 14d                     d_rsi_14
    MACD 12-26-9                d_macd, d_macd_signal, d_macd_hist, d_macd_norm
    Log return 1d               d_log_ret_1
    Volatility 20d / 60d        d_vol_20, d_vol_60  (annualised, √252)
    """
    df = daily.copy()
    close = df["close"]

    for n in [20, 50, 200]:
        sma = close.rolling(n).mean()
        df[f"d_sma_{n}"] = sma
        df[f"d_c_vs_sma_{n}"] = close / sma - 1

    for n in [12, 20, 26, 50, 200]:
        ema = close.ewm(span=n, adjust=False).mean()
        df[f"d_ema_{n}"] = ema
        df[f"d_c_vs_ema_{n}"] = close / ema - 1

    df["d_rsi_14"] = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["d_macd"] = ema12 - ema26
    df["d_macd_signal"] = df["d_macd"].ewm(span=9, adjust=False).mean()
    df["d_macd_hist"] = df["d_macd"] - df["d_macd_signal"]
    df["d_macd_norm"] = df["d_macd"] / close

    log_ret = np.log(close / close.shift(1))
    df["d_log_ret_1"] = log_ret
    df["d_vol_20"] = log_ret.rolling(20).std() * np.sqrt(252)
    df["d_vol_60"] = log_ret.rolling(60).std() * np.sqrt(252)

    return df


def _resample_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample a daily DataFrame (OHLCV + d_* indicators) to week-ending-Friday.

    OHLCV columns use their natural aggregation rule; all other columns
    (indicator columns) take the last value of the week (Friday snapshot).
    Empty weeks (e.g. full-week holidays) are dropped.
    """
    agg = {col: _OHLCV_AGG.get(col, "last") for col in daily.columns}
    weekly = daily.resample("W-FRI").agg(agg)
    ref = "close" if "close" in weekly.columns else weekly.columns[0]
    return weekly.dropna(subset=[ref])


def compute_weekly_indicators(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add weekly technical indicators (w_* prefix) to a weekly OHLCV DataFrame.

    Indicators
    ----------
    Lagged returns 1w/2w/3w/4w/1q    w_ret_*, w_log_ret_*
    SMA 4/8/13/26/52w                w_sma_{n}, w_c_vs_sma_{n}
    EMA 4/8/13/26/52w                w_ema_{n}, w_c_vs_ema_{n}
    MA crossovers 4-13, 8-26         w_sma/ema_{s}_{l}_cross
    RSI 14w / 26w                    w_rsi_14, w_rsi_26
    MACD 12-26-9                     w_macd, w_macd_signal, w_macd_hist, w_macd_norm
    Rate of change 4w/1q/2q/1y       w_roc_*
    Volatility 4w/1q/2q/1y           w_vol_*  (annualised, √52)
    ATR 14w                          w_atr_14, w_atr_pct_14
    Bollinger Bands 20w ×2σ          w_bb_pct_20, w_bb_width_20
    Volume ratio 4w / 13w            w_vol_ratio_4, w_vol_ratio_13
    """
    df = weekly.copy()
    close = df["close"]

    # Lagged returns
    for n, label in [(1, "1w"), (2, "2w"), (3, "3w"), (4, "4w"), (13, "1q")]:
        df[f"w_ret_{label}"] = close.pct_change(n)
        df[f"w_log_ret_{label}"] = np.log(close / close.shift(n))

    # SMA and EMA
    for n in [4, 8, 13, 26, 52]:
        sma = close.rolling(n).mean()
        df[f"w_sma_{n}"] = sma
        df[f"w_c_vs_sma_{n}"] = close / sma - 1

        ema = close.ewm(span=n, adjust=False).mean()
        df[f"w_ema_{n}"] = ema
        df[f"w_c_vs_ema_{n}"] = close / ema - 1

    for s, l in [(4, 13), (8, 26)]:
        df[f"w_sma_{s}_{l}_cross"] = df[f"w_sma_{s}"] / df[f"w_sma_{l}"] - 1
        df[f"w_ema_{s}_{l}_cross"] = df[f"w_ema_{s}"] / df[f"w_ema_{l}"] - 1

    # RSI
    df["w_rsi_14"] = _rsi(close, 14)
    df["w_rsi_26"] = _rsi(close, 26)

    # MACD 12-26-9
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["w_macd"] = ema12 - ema26
    df["w_macd_signal"] = df["w_macd"].ewm(span=9, adjust=False).mean()
    df["w_macd_hist"] = df["w_macd"] - df["w_macd_signal"]
    df["w_macd_norm"] = df["w_macd"] / close

    # Rate of change
    for n, label in [(4, "4w"), (13, "1q"), (26, "2q"), (52, "1y")]:
        df[f"w_roc_{label}"] = close / close.shift(n) - 1

    # Volatility (annualised)
    log_ret_1w = np.log(close / close.shift(1))
    for n, label in [(4, "4w"), (13, "1q"), (26, "2q"), (52, "1y")]:
        df[f"w_vol_{label}"] = log_ret_1w.rolling(n).std() * np.sqrt(52)

    # ATR 14w
    if {"high", "low", "close"}.issubset(df.columns):
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["w_atr_14"] = tr.rolling(14).mean()
        df["w_atr_pct_14"] = df["w_atr_14"] / close

    # Bollinger Bands 20w ×2σ
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    band_width = (bb_upper - bb_lower).replace(0, float("nan"))
    df["w_bb_pct_20"] = (close - bb_lower) / band_width
    df["w_bb_width_20"] = band_width / sma20

    # Volume ratios
    if "volume" in df.columns:
        vol = df["volume"]
        df["w_vol_ratio_4"] = vol / vol.rolling(4).mean()
        df["w_vol_ratio_13"] = vol / vol.rolling(13).mean()

    return df


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def _add_fundamental_ratios(df: pd.DataFrame) -> None:
    """Derive common fundamental ratios in-place where source columns exist."""
    if {"netIncome", "revenue"}.issubset(df.columns):
        df["netMargin"] = df["netIncome"] / df["revenue"].replace(0, float("nan"))
    if {"grossProfit", "revenue"}.issubset(df.columns):
        df["grossMargin"] = df["grossProfit"] / df["revenue"].replace(0, float("nan"))
    if {"operatingCashFlow", "revenue"}.issubset(df.columns):
        df["cfMargin"] = df["operatingCashFlow"] / df["revenue"].replace(0, float("nan"))
    if {"totalDebt", "totalStockholdersEquity"}.issubset(df.columns):
        df["debtToEquity"] = df["totalDebt"] / df["totalStockholdersEquity"].replace(0, float("nan"))
    if {"totalLiabilities", "totalAssets"}.issubset(df.columns):
        df["debtToAssets"] = df["totalLiabilities"] / df["totalAssets"].replace(0, float("nan"))


def _merge_fundamentals(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Left-join shifted quarterly fundamentals onto a price DataFrame and forward-fill.

    Works for both daily and weekly price DataFrames.  Forward-fill propagates
    each quarter's values forward until the next shifted quarter date lands
    within the index.
    """
    if fundamentals.empty:
        return prices.copy()

    merged = prices.join(fundamentals, how="left")
    fund_cols = [c for c in fundamentals.columns if c in merged.columns]
    merged[fund_cols] = merged[fund_cols].ffill()
    _add_fundamental_ratios(merged)
    return merged


# ---------------------------------------------------------------------------
# Per-ticker processing  →  four cache files
# ---------------------------------------------------------------------------
#
# Cache layout
# ─────────────────────────────────────────────────────────────────────────
# cache_dir/
#   raw/{TICKER}.parquet        Pure daily OHLCV — source of truth for updates.
#                               Rolling indicators need the full history to warm
#                               up (e.g. SMA-200), so we always recompute from
#                               the complete raw series after appending new bars.
#   daily/{TICKER}.parquet      Daily OHLCV + d_* indicators + fundamentals ffilled.
#   weekly/{TICKER}.parquet     Weekly OHLCV + d_* (end-of-week) + w_* indicators
#                               + fundamentals ffilled.
#   quarterly/{TICKER}.parquet  Raw shifted quarterly rows — sparse, no ffill.
# ─────────────────────────────────────────────────────────────────────────


def _ticker_is_cached(ticker: str, cache_dir: Path) -> bool:
    """True only when all four cache files (raw + three frequencies) exist."""
    return (
        (cache_dir / "raw" / f"{ticker}.parquet").exists()
        and (cache_dir / "daily" / f"{ticker}.parquet").exists()
        and (cache_dir / "weekly" / f"{ticker}.parquet").exists()
        and (cache_dir / "quarterly" / f"{ticker}.parquet").exists()
    )


def _last_cached_date(ticker: str, cache_dir: Path) -> pd.Timestamp | None:
    """Return the most recent date in the raw OHLCV cache for *ticker*.

    If the raw cache doesn't exist but the daily cache does (i.e. data was
    downloaded before the raw sub-cache was introduced), the OHLCV columns are
    extracted from the daily cache and saved as the raw cache automatically —
    no API call needed.

    Returns None if no cache exists at all.
    """
    raw_path = cache_dir / "raw" / f"{ticker}.parquet"

    if not raw_path.exists():
        daily_path = cache_dir / "daily" / f"{ticker}.parquet"
        if not daily_path.exists():
            return None
        # Migration: reconstruct raw cache from existing daily cache
        daily_df = pd.read_parquet(daily_path)
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        ohlcv_cols = [c for c in _OHLCV_AGG if c in daily_df.columns]
        raw = daily_df[["date"] + ohlcv_cols].set_index("date")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw.reset_index().to_parquet(raw_path, index=False)

    raw_df = pd.read_parquet(raw_path, columns=["date"])
    return pd.to_datetime(raw_df["date"]).max()


def _rebuild_derived_caches(
    ticker: str,
    daily_raw: pd.DataFrame,
    fundamentals: pd.DataFrame,
    cache_dir: Path,
) -> bool:
    """Compute and save daily / weekly / quarterly cache files from raw inputs.

    *daily_raw* must be a date-indexed DataFrame of pure OHLCV columns.
    Returns False if there is too little data to be useful.
    """
    for sub in ("daily", "weekly", "quarterly"):
        (cache_dir / sub).mkdir(parents=True, exist_ok=True)

    daily_with_ind = compute_daily_indicators(daily_raw)

    # Daily
    daily_full = _merge_fundamentals(daily_with_ind, fundamentals)
    daily_full.insert(0, "ticker", ticker)
    daily_full.reset_index().to_parquet(
        cache_dir / "daily" / f"{ticker}.parquet", index=False
    )

    # Weekly
    weekly_raw = _resample_to_weekly(daily_with_ind)
    if len(weekly_raw) < 10:
        return False
    weekly_with_ind = compute_weekly_indicators(weekly_raw)
    weekly_full = _merge_fundamentals(weekly_with_ind, fundamentals)
    weekly_full.insert(0, "ticker", ticker)
    weekly_full.reset_index().to_parquet(
        cache_dir / "weekly" / f"{ticker}.parquet", index=False
    )

    # Quarterly (sparse — raw shifted rows, no ffill)
    if not fundamentals.empty:
        q = fundamentals.copy()
        q.insert(0, "ticker", ticker)
        q.reset_index().to_parquet(
            cache_dir / "quarterly" / f"{ticker}.parquet", index=False
        )

    return True


def process_ticker(
    ticker: str,
    api_key: str,
    start: str,
    end: str,
    cache_dir: Path,
    delay: float,
) -> bool:
    """Full historical download and cache for one ticker.  Returns True on success."""
    if _ticker_is_cached(ticker, cache_dir):
        return True

    (cache_dir / "raw").mkdir(parents=True, exist_ok=True)

    try:
        daily_raw = _download_daily_raw(ticker, api_key, start, end)
        time.sleep(delay)

        if daily_raw.empty or len(daily_raw) < 30:
            return False

        # Save raw OHLCV — source of truth for future incremental updates
        daily_raw.reset_index().to_parquet(
            cache_dir / "raw" / f"{ticker}.parquet", index=False
        )

        fundamentals = download_quarterly_fundamentals(ticker, api_key, delay)
        return _rebuild_derived_caches(ticker, daily_raw, fundamentals, cache_dir)

    except Exception as exc:
        print(f"    ERROR: {exc}")
        return False


def update_ticker(
    ticker: str,
    api_key: str,
    start: str,
    end: str,
    cache_dir: Path,
    delay: float,
) -> str:
    """Incrementally update one ticker's cache with new data up to *end*.

    Behaviour
    ---------
    New ticker (no cache at all)
        Falls back to a full historical download from *start*.

    Existing ticker
        1. Reads the raw OHLCV cache to find the last stored date.
        2. Fetches only new daily bars from (last_date + 1 calendar day) to *end*.
           If the remote returns no new rows the ticker is already up to date.
        3. Always re-fetches quarterly fundamentals (cheap — 3 requests) so that
           newly reported quarters and any restatements are captured.
        4. Appends new bars to the raw cache, then recomputes ALL indicators and
           derived caches from the full raw series so that rolling windows
           (e.g. SMA-200) are correctly initialised.

    Returns one of: ``"NEW"`` | ``"UPDATED"`` | ``"UP_TO_DATE"`` | ``"FAIL"``
    """
    last_date = _last_cached_date(ticker, cache_dir)

    if last_date is None:
        # Brand-new ticker — full download
        ok = process_ticker(ticker, api_key, start, end, cache_dir, delay)
        return "NEW" if ok else "FAIL"

    # Check whether there is anything new to fetch
    fetch_from = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if fetch_from > end:
        return "UP_TO_DATE"

    try:
        # ── 1. Fetch new daily bars ────────────────────────────────────────
        new_bars = _download_daily_raw(ticker, api_key, fetch_from, end)
        time.sleep(delay)

        # ── 2. Load existing raw OHLCV and append ─────────────────────────
        raw_path = cache_dir / "raw" / f"{ticker}.parquet"
        existing_raw = pd.read_parquet(raw_path)
        existing_raw["date"] = pd.to_datetime(existing_raw["date"])
        existing_raw = existing_raw.set_index("date")

        if new_bars.empty:
            # No new price data but we still refresh fundamentals below
            daily_raw = existing_raw
            got_new_prices = False
        else:
            # Deduplicate by keeping the freshly downloaded version of any overlap
            combined = pd.concat([existing_raw, new_bars])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            daily_raw = combined
            got_new_prices = True
            daily_raw.reset_index().to_parquet(raw_path, index=False)

        # ── 3. Re-fetch quarterly fundamentals ────────────────────────────
        fundamentals = download_quarterly_fundamentals(ticker, api_key, delay)

        # ── 4. Recompute all derived caches from full raw series ───────────
        _rebuild_derived_caches(ticker, daily_raw, fundamentals, cache_dir)

        return "UPDATED" if got_new_prices else "UP_TO_DATE"

    except Exception as exc:
        print(f"    ERROR: {exc}")
        return "FAIL"


# ---------------------------------------------------------------------------
# Merge all cached tickers → combined parquets
# ---------------------------------------------------------------------------

def build_combined_parquet(sub_cache: Path, output_path: Path) -> pd.DataFrame:
    """Concatenate all per-ticker parquet files in *sub_cache* into one file."""
    files = sorted(sub_cache.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found in {sub_cache}")

    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception as exc:
            print(f"  Warning: skipping {f.name}: {exc}")

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    return combined


def _print_summary(label: str, df: pd.DataFrame, path: Path) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  Path   : {path}")
    print(f"  Shape  : {df.shape}")
    print(f"  Dates  : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Tickers: {df['ticker'].nunique()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download S&P 500 data at daily, weekly, and quarterly frequencies "
            "from FMP and save three parquet files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-key", required=True, help="FMP API key.")
    parser.add_argument("--start", default="2010-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument(
        "--end",
        default=pd.Timestamp.today().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--out-daily", default="sp500_daily.parquet",
        help="Output path for the daily parquet.",
    )
    parser.add_argument(
        "--out-weekly", default="sp500_weekly.parquet",
        help="Output path for the weekly parquet.",
    )
    parser.add_argument(
        "--out-quarterly", default="sp500_quarterly.parquet",
        help="Output path for the quarterly parquet.",
    )
    parser.add_argument(
        "--cache-dir", default="fmp_ticker_cache",
        help="Root cache directory (sub-dirs: raw/, daily/, weekly/, quarterly/).",
    )
    parser.add_argument(
        "--delay", type=float, default=0.25,
        help=(
            "Seconds to wait between FMP requests. "
            "Raise to 1.0+ on the free plan (250 req/day)."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only the first N tickers (0 = all). Useful for testing.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Skip downloading; just rebuild the three parquets from existing cache.",
    )
    parser.add_argument(
        "--update", action="store_true",
        help=(
            "Incremental update mode: for each ticker fetch only new bars since "
            "the last cached date, always re-fetch quarterly fundamentals, then "
            "recompute all indicators.  New or missing tickers get a full download."
        ),
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    out_daily = Path(args.out_daily)
    out_weekly = Path(args.out_weekly)
    out_quarterly = Path(args.out_quarterly)

    if args.rebuild:
        print("Rebuilding parquets from cache (no API calls)…")
        for sub, out, label in [
            ("daily", out_daily, "Daily"),
            ("weekly", out_weekly, "Weekly"),
            ("quarterly", out_quarterly, "Quarterly"),
        ]:
            sub_cache = cache_dir / sub
            if not any(sub_cache.glob("*.parquet")):
                print(f"  {label}: no cache files found, skipping.")
                continue
            df = build_combined_parquet(sub_cache, out)
            _print_summary(label, df, out)
        return

    # ------------------------------------------------------------------
    # 1. Tickers
    # ------------------------------------------------------------------
    print("Fetching S&P 500 constituents from FMP…")
    tickers = get_sp500_tickers(args.api_key)
    print(f"  {len(tickers)} constituents found.")

    if args.limit > 0:
        tickers = tickers[: args.limit]
        print(f"  Limiting to first {args.limit} tickers.")

    # ------------------------------------------------------------------
    # 2a. UPDATE mode — incremental fetch for each ticker
    # ------------------------------------------------------------------
    if args.update:
        cached_count = sum(1 for t in tickers if _last_cached_date(t, cache_dir) is not None)
        new_count = len(tickers) - cached_count
        print(f"  Update mode — cached: {cached_count}  |  new (full download): {new_count}\n")

        counts: dict[str, int] = {"NEW": 0, "UPDATED": 0, "UP_TO_DATE": 0, "FAIL": 0}
        for idx, ticker in enumerate(tickers, 1):
            print(f"[{idx:3d}/{len(tickers)}] {ticker:<6s}", end="", flush=True)
            status = update_ticker(
                ticker, args.api_key, args.start, args.end, cache_dir, args.delay
            )
            counts[status] += 1
            print(f"  {status}")

        print(
            f"\nUpdate summary: {counts['NEW']} new  |  {counts['UPDATED']} updated  "
            f"|  {counts['UP_TO_DATE']} up-to-date  |  {counts['FAIL']} failed"
        )

    # ------------------------------------------------------------------
    # 2b. FULL DOWNLOAD mode — skip already-complete tickers
    # ------------------------------------------------------------------
    else:
        already_done = sum(1 for t in tickers if _ticker_is_cached(t, cache_dir))
        print(f"  Fully cached: {already_done}/{len(tickers)}  |  To download: {len(tickers) - already_done}\n")

        success = skip = fail = 0
        for idx, ticker in enumerate(tickers, 1):
            cached = _ticker_is_cached(ticker, cache_dir)
            tag = "SKIP" if cached else "…   "
            print(f"[{idx:3d}/{len(tickers)}] {ticker:<6s}  {tag}", end="", flush=True)

            if cached:
                skip += 1
                print()
                continue

            ok = process_ticker(ticker, args.api_key, args.start, args.end, cache_dir, args.delay)
            if ok:
                success += 1
                print("  OK")
            else:
                fail += 1
                print("  FAIL")

        print(f"\nDownload complete: {success} new  |  {skip} cached  |  {fail} failed")

    # ------------------------------------------------------------------
    # 3. Merge all cache files → three final parquets
    # ------------------------------------------------------------------
    print("\nBuilding combined parquets…")

    results: dict[str, pd.DataFrame] = {}
    for sub, out, label in [
        ("daily", out_daily, "Daily"),
        ("weekly", out_weekly, "Weekly"),
        ("quarterly", out_quarterly, "Quarterly"),
    ]:
        sub_cache = cache_dir / sub
        n_files = len(list(sub_cache.glob("*.parquet")))
        if n_files == 0:
            print(f"  {label}: no cache files — skipping.")
            continue
        print(f"  {label}: merging {n_files} ticker files…", end="", flush=True)
        df = build_combined_parquet(sub_cache, out)
        results[label] = df
        print(f"  done  ({df.shape[0]:,} rows × {df.shape[1]} cols)")

    for label, df in results.items():
        out = {"Daily": out_daily, "Weekly": out_weekly, "Quarterly": out_quarterly}[label]
        _print_summary(label, df, out)

    print(f"\n{'─'*60}")
    print("All done.")


if __name__ == "__main__":
    main()
