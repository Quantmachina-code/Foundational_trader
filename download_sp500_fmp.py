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
    """Download daily OHLCV from FMP and resample to weekly bars (week-ending Friday).

    Weekly OHLCV convention:
      open     – price on the first trading day of the week
      high     – highest intraday price during the week
      low      – lowest intraday price during the week
      close    – price on the last trading day of the week
      adjClose – adjusted close on the last trading day of the week
      volume   – total shares traded during the week
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

    return _resample_to_weekly(daily)


def _resample_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample a daily OHLCV DataFrame to week-ending-Friday bars."""
    agg: dict[str, str] = {}
    if "open" in daily.columns:
        agg["open"] = "first"
    if "high" in daily.columns:
        agg["high"] = "max"
    if "low" in daily.columns:
        agg["low"] = "min"
    if "close" in daily.columns:
        agg["close"] = "last"
    if "adjClose" in daily.columns:
        agg["adjClose"] = "last"
    if "volume" in daily.columns:
        agg["volume"] = "sum"

    # W-FRI: week label is the Friday that closes the week
    weekly = daily.resample("W-FRI").agg(agg)
    # Drop weeks where no trading occurred (e.g. holiday weeks at boundaries)
    weekly = weekly.dropna(subset=["close"] if "close" in weekly.columns else weekly.columns[:1])
    return weekly


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
