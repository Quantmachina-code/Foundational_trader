"""Download S&P 500 weekly price and quarterly fundamental data via FMP API.

Price data is downloaded daily from FMP then resampled to weekly bars
(week ending Friday): Open=first day, High=max, Low=min, Close/adjClose=last
day, Volume=sum.

Quarterly fundamentals are shifted forward by one quarter to avoid look-ahead
bias: Q1 data (period ending 2023-03-31) is only treated as known from
2023-06-30 onward, since earnings are typically reported weeks after quarter-end.
The shifted quarterly values are forward-filled into every weekly row until the
next quarter's data becomes available.

Download is gradual — one ticker at a time — and each ticker is cached as its
own parquet file so the script can be interrupted and resumed safely.

Usage:
    python download_sp500_fmp.py --api-key YOUR_KEY

Options:
    --api-key     Financial Modeling Prep API key (required)
    --start       Start date, default 2010-01-01
    --end         End date, default today
    --output      Final merged parquet path, default sp500_fmp.parquet
    --cache-dir   Per-ticker cache directory, default fmp_ticker_cache/
    --delay       Seconds between requests to respect rate limits, default 0.25
    --limit       Cap number of tickers processed (0 = all); useful for testing

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

# Fundamental fields to keep (subset of each statement to keep parquet lean)
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

# Statements to pull and the columns to retain
STATEMENTS: list[tuple[str, list[str]]] = [
    ("income-statement", INCOME_COLS),
    ("balance-sheet-statement", BALANCE_COLS),
    ("cash-flow-statement", CASHFLOW_COLS),
]


# ---------------------------------------------------------------------------
# FMP helpers
# ---------------------------------------------------------------------------

def _fmp_get(endpoint: str, api_key: str, params: dict | None = None) -> list | dict:
    """GET from FMP API; raises on non-200 or error payload."""
    url = f"{FMP_BASE}/{endpoint}"
    p: dict = {"apikey": api_key}
    if params:
        p.update(params)
    resp = requests.get(url, params=p, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # FMP returns {"Error Message": "..."} for invalid keys / limits
    if isinstance(data, dict) and "Error Message" in data:
        raise RuntimeError(data["Error Message"])
    return data


def get_sp500_tickers(api_key: str) -> list[str]:
    """Return current S&P 500 constituent symbols from FMP."""
    data = _fmp_get("sp500_constituent", api_key)
    return sorted({row["symbol"] for row in data if row.get("symbol")})


# ---------------------------------------------------------------------------
# Per-ticker download functions
# ---------------------------------------------------------------------------

def download_prices(ticker: str, api_key: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV, compute indicators on both timeframes, return weekly bars.

    Pipeline
    --------
    1. Download daily OHLCV from FMP.
    2. Compute daily technical indicators (EMA/SMA/RSI/MACD/vol on daily bars).
    3. Resample everything to week-ending-Friday bars:
         OHLCV  → open=first, high=max, low=min, close/adjClose=last, volume=sum
         indicators → last value of the week
    4. Compute weekly technical indicators on the weekly bars
       (lagged returns, weekly EMA/SMA, weekly RSI/MACD/BB/ATR/momentum).
    5. Return the combined weekly DataFrame.
    """
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

    keep = ["open", "high", "low", "close", "adjClose", "volume"]
    daily = df[[c for c in keep if c in df.columns]].astype(float, errors="ignore")

    daily_with_ind = compute_daily_indicators(daily)
    weekly = _resample_to_weekly(daily_with_ind)
    return compute_weekly_indicators(weekly)


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100.0 - 100.0 / (1.0 + rs)


def compute_daily_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators on daily OHLCV bars.

    Indicators are prefixed with ``d_`` to distinguish them from the weekly
    equivalents once both are present in the final row.  All columns are
    resampled to weekly with ``"last"`` (end-of-week snapshot).

    Indicators computed
    -------------------
    SMAs        : 20d, 50d, 200d  (price-relative ratio)
    EMAs        : 12d, 20d, 26d, 50d, 200d
    RSI         : 14d
    MACD        : 12-26-9  (line, signal, histogram, price-normalised)
    Log returns : 1d
    Volatility  : rolling 20d and 60d annualised (√252)
    """
    df = daily.copy()
    close = df["close"]

    # SMAs
    for n in [20, 50, 200]:
        sma = close.rolling(n).mean()
        df[f"d_sma_{n}"] = sma
        df[f"d_c_vs_sma_{n}"] = close / sma - 1

    # EMAs
    for n in [12, 20, 26, 50, 200]:
        ema = close.ewm(span=n, adjust=False).mean()
        df[f"d_ema_{n}"] = ema
        df[f"d_c_vs_ema_{n}"] = close / ema - 1

    # RSI 14d
    df["d_rsi_14"] = _rsi(close, 14)

    # MACD (12-26-9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["d_macd"] = ema12 - ema26
    df["d_macd_signal"] = df["d_macd"].ewm(span=9, adjust=False).mean()
    df["d_macd_hist"] = df["d_macd"] - df["d_macd_signal"]
    df["d_macd_norm"] = df["d_macd"] / close

    # Log return and vol
    log_ret = np.log(close / close.shift(1))
    df["d_log_ret_1"] = log_ret
    df["d_vol_20"] = log_ret.rolling(20).std() * np.sqrt(252)
    df["d_vol_60"] = log_ret.rolling(60).std() * np.sqrt(252)

    return df


def _resample_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample a daily DataFrame (OHLCV + indicators) to week-ending-Friday bars.

    OHLCV columns use their natural aggregation; every other column (indicators)
    takes the last value of the week.
    """
    OHLCV_AGG = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "adjClose": "last",
        "volume": "sum",
    }
    agg: dict[str, str] = {}
    for col in daily.columns:
        agg[col] = OHLCV_AGG.get(col, "last")

    weekly = daily.resample("W-FRI").agg(agg)
    ref = "close" if "close" in weekly.columns else weekly.columns[0]
    weekly = weekly.dropna(subset=[ref])
    return weekly


def compute_weekly_indicators(weekly: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators directly on weekly OHLCV bars.

    All column names are prefixed ``w_``.

    Indicators computed
    -------------------
    Lagged returns  : 1w, 2w, 3w, 4w, 1q (13w)  — simple and log
    SMAs            : 4w, 8w, 13w, 26w, 52w  + price-relative ratio
    EMAs            : 4w, 8w, 13w, 26w, 52w  + price-relative ratio
    MA crossovers   : sma 4/13, 8/26 ; ema 4/13, 8/26
    RSI             : 14w, 26w
    MACD            : 12-26-9  (line, signal, hist, normalised)
    Rate of change  : 4w, 13w, 26w, 52w
    Volatility      : rolling 4w, 13w, 26w, 52w  annualised (√52)
    ATR             : 14w  (and as % of close)
    Bollinger Bands : 20w ×2σ  (%, width)
    Volume ratios   : 4w, 13w  (vs rolling mean)
    """
    df = weekly.copy()
    close = df["close"]

    # ------------------------------------------------------------------
    # Lagged returns
    # ------------------------------------------------------------------
    for n, label in [(1, "1w"), (2, "2w"), (3, "3w"), (4, "4w"), (13, "1q")]:
        df[f"w_ret_{label}"] = close.pct_change(n)
        df[f"w_log_ret_{label}"] = np.log(close / close.shift(n))

    # ------------------------------------------------------------------
    # SMAs and EMAs
    # ------------------------------------------------------------------
    for n in [4, 8, 13, 26, 52]:
        sma = close.rolling(n).mean()
        df[f"w_sma_{n}"] = sma
        df[f"w_c_vs_sma_{n}"] = close / sma - 1

        ema = close.ewm(span=n, adjust=False).mean()
        df[f"w_ema_{n}"] = ema
        df[f"w_c_vs_ema_{n}"] = close / ema - 1

    # MA crossovers (ratio > 0 means shorter MA above longer MA)
    for s, l in [(4, 13), (8, 26)]:
        df[f"w_sma_{s}_{l}_cross"] = df[f"w_sma_{s}"] / df[f"w_sma_{l}"] - 1
        df[f"w_ema_{s}_{l}_cross"] = df[f"w_ema_{s}"] / df[f"w_ema_{l}"] - 1

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------
    df["w_rsi_14"] = _rsi(close, 14)
    df["w_rsi_26"] = _rsi(close, 26)

    # ------------------------------------------------------------------
    # MACD (12-26-9)
    # ------------------------------------------------------------------
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["w_macd"] = ema12 - ema26
    df["w_macd_signal"] = df["w_macd"].ewm(span=9, adjust=False).mean()
    df["w_macd_hist"] = df["w_macd"] - df["w_macd_signal"]
    df["w_macd_norm"] = df["w_macd"] / close

    # ------------------------------------------------------------------
    # Rate of change / momentum
    # ------------------------------------------------------------------
    for n, label in [(4, "4w"), (13, "1q"), (26, "2q"), (52, "1y")]:
        df[f"w_roc_{label}"] = close / close.shift(n) - 1

    # ------------------------------------------------------------------
    # Historical volatility (annualised, √52 weeks/year)
    # ------------------------------------------------------------------
    log_ret_1w = np.log(close / close.shift(1))
    for n, label in [(4, "4w"), (13, "1q"), (26, "2q"), (52, "1y")]:
        df[f"w_vol_{label}"] = log_ret_1w.rolling(n).std() * np.sqrt(52)

    # ------------------------------------------------------------------
    # ATR (14w)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Bollinger Bands (20w, 2σ)
    # ------------------------------------------------------------------
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    band_width = (bb_upper - bb_lower).replace(0, float("nan"))
    df["w_bb_pct_20"] = (close - bb_lower) / band_width
    df["w_bb_width_20"] = band_width / sma20

    # ------------------------------------------------------------------
    # Volume ratios
    # ------------------------------------------------------------------
    if "volume" in df.columns:
        vol = df["volume"]
        df["w_vol_ratio_4"] = vol / vol.rolling(4).mean()
        df["w_vol_ratio_13"] = vol / vol.rolling(13).mean()

    return df


def download_quarterly_fundamentals(
    ticker: str,
    api_key: str,
    delay: float,
) -> pd.DataFrame:
    """Pull income statement + balance sheet + cash flow (quarterly).

    The resulting DataFrame is indexed by the *shifted* date: each quarter's
    data is pushed forward by one quarter (≈ 91 days via DateOffset) so that
    it only appears in time-series after the next quarter begins — preventing
    look-ahead bias.

    Example
    -------
    Q1-2023 data (period ending 2023-03-31) becomes available from 2023-06-30.
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

    # Outer-join all statements on quarter-end date
    fundamentals = frames[0]
    for frame in frames[1:]:
        fundamentals = fundamentals.join(frame, how="outer")

    # --- KEY: shift 1 quarter forward -----------------------------------
    # Reason: at Q-end companies have NOT yet reported; data is only known
    # ~1–2 months later.  We conservatively delay by a full quarter so that
    # Q1 fundamentals only enter the feature set starting from Q2-end.
    # This guarantees no look-ahead bias in any downstream model.
    # --------------------------------------------------------------------
    fundamentals.index = fundamentals.index + pd.DateOffset(months=3)
    fundamentals.index.name = "date"

    return fundamentals


# ---------------------------------------------------------------------------
# Merge & cache
# ---------------------------------------------------------------------------

def merge_price_and_fundamentals(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join fundamentals onto weekly prices and forward-fill quarterly values.

    Quarterly data arrives at most once every ~13 weekly bars.  Forward-fill
    carries the most-recently-available figure forward until the next (shifted)
    quarter date falls within the weekly index.
    """
    if fundamentals.empty:
        return prices.copy()

    merged = prices.join(fundamentals, how="left")

    fund_cols = [c for c in fundamentals.columns if c in merged.columns]
    merged[fund_cols] = merged[fund_cols].ffill()

    # Optionally derive a few simple ratios from merged data
    _add_fundamental_ratios(merged)

    return merged


def _add_fundamental_ratios(df: pd.DataFrame) -> None:
    """Add a handful of common fundamental ratios in-place (if source cols exist)."""
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


def process_ticker(
    ticker: str,
    api_key: str,
    start: str,
    end: str,
    cache_dir: Path,
    delay: float,
) -> bool:
    """Download, merge, and cache a single ticker.  Returns True on success."""
    cache_file = cache_dir / f"{ticker}.parquet"
    if cache_file.exists():
        return True  # Resume: already done

    try:
        prices = download_prices(ticker, api_key, start, end)
        time.sleep(delay)

        if prices.empty or len(prices) < 10:
            # Fewer than 10 weekly bars — not worth caching
            return False

        fundamentals = download_quarterly_fundamentals(ticker, api_key, delay)

        df = merge_price_and_fundamentals(prices, fundamentals)
        df.insert(0, "ticker", ticker)
        df = df.reset_index()  # bring 'date' back as a column

        df.to_parquet(cache_file, index=False)
        return True

    except Exception as exc:
        print(f"    ERROR: {exc}")
        return False


# ---------------------------------------------------------------------------
# Combine all cached tickers → final parquet
# ---------------------------------------------------------------------------

def build_combined_parquet(cache_dir: Path, output_path: Path) -> pd.DataFrame:
    """Concatenate all per-ticker parquet files into one sorted dataset."""
    files = sorted(cache_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No cached parquet files found in {cache_dir}")

    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception as exc:
            print(f"  Warning: skipping corrupted cache {f.name}: {exc}")

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download S&P 500 weekly price + quarterly fundamentals from FMP and save to parquet.",
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
        "--output",
        default="sp500_fmp.parquet",
        help="Output path for the final merged parquet.",
    )
    parser.add_argument(
        "--cache-dir",
        default="fmp_ticker_cache",
        help="Directory to store per-ticker parquet files (used for resume).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help=(
            "Seconds to wait between each FMP request. "
            "Increase to ~0.5–1.0 on the free plan (250 req/day limit)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N tickers (0 = all). Useful for testing.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the combined parquet from cache without re-downloading.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)

    if args.rebuild:
        print("Rebuilding combined parquet from cache…")
        combined = build_combined_parquet(cache_dir, output_path)
        print(f"Saved {output_path}  shape={combined.shape}")
        return

    # ------------------------------------------------------------------
    # 1. Get tickers
    # ------------------------------------------------------------------
    print("Fetching S&P 500 constituents from FMP…")
    tickers = get_sp500_tickers(args.api_key)
    print(f"  {len(tickers)} constituents found.")

    if args.limit > 0:
        tickers = tickers[: args.limit]
        print(f"  Limiting to first {args.limit} tickers.")

    already_cached = sum(1 for t in tickers if (cache_dir / f"{t}.parquet").exists())
    todo = len(tickers) - already_cached
    print(f"  Cached: {already_cached}  |  To download: {todo}\n")

    # ------------------------------------------------------------------
    # 2. Download each ticker
    # ------------------------------------------------------------------
    success = skip = fail = 0

    for idx, ticker in enumerate(tickers, 1):
        is_cached = (cache_dir / f"{ticker}.parquet").exists()

        if is_cached:
            skip += 1
            status = "SKIP"
        else:
            status = "…"

        print(f"[{idx:3d}/{len(tickers)}] {ticker:<6s}  {status}", end="", flush=True)

        if is_cached:
            print()
            continue

        ok = process_ticker(ticker, args.api_key, args.start, args.end, cache_dir, args.delay)
        if ok:
            success += 1
            print("  OK")
        else:
            fail += 1
            print("  FAIL")

    print(f"\nDownload summary: {success} new  |  {skip} cached  |  {fail} failed")

    # ------------------------------------------------------------------
    # 3. Merge into final parquet
    # ------------------------------------------------------------------
    cached_count = sum(1 for t in tickers if (cache_dir / f"{t}.parquet").exists())
    if cached_count == 0:
        print("No data cached — nothing to merge.")
        return

    print(f"\nBuilding combined parquet from {cached_count} ticker files…")
    combined = build_combined_parquet(cache_dir, output_path)
    print(f"Saved → {output_path}")
    print(f"Shape : {combined.shape}")
    print(f"Dates : {combined['date'].min().date()} → {combined['date'].max().date()}")
    print(f"Tickers: {combined['ticker'].nunique()}")

    # Brief column summary
    price_cols = ["open", "high", "low", "close", "adjClose", "volume"]
    fund_cols = [c for c in combined.columns if c not in price_cols + ["date", "ticker"]]
    print(f"Price cols     : {[c for c in price_cols if c in combined.columns]}")
    print(f"Fundamental cols ({len(fund_cols)}): {fund_cols[:10]}{'…' if len(fund_cols) > 10 else ''}")


if __name__ == "__main__":
    main()
