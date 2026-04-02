"""Data download wrapper.

Thin orchestration layer over ``download_sp500_fmp.py``.  All raw HTTP calls
and indicator logic live in that file; this module only handles path resolution
and the download/update loop so the rest of the package can call ``ensure_data``
without caring about FMP internals.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make the repo root importable so download_sp500_fmp can be imported directly.
_REPO_ROOT = Path(__file__).parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from download_sp500_fmp import (  # noqa: E402
    build_combined_parquet,
    get_sp500_tickers,
    process_ticker,
    update_ticker,
)

from catboost_trader.config import (
    CACHE_DIR,
    COMBINED_PARQUET,
    DATA_FETCH_START,
    FMP_REQUEST_DELAY,
    SIM_END,
)


def ensure_data(
    api_key: str,
    *,
    start: str = DATA_FETCH_START,
    end: str = SIM_END,
    update: bool = True,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Download (or update) all S&P 500 daily data and return the combined panel.

    Parameters
    ----------
    api_key:
        FMP API key.
    start:
        Earliest date to fetch (ISO format).  Defaults to ``DATA_FETCH_START``
        from config (2009-01-01 for indicator warm-up).
    end:
        Latest date to fetch.  Defaults to ``SIM_END``.
    update:
        If True, incrementally update existing tickers; if False, skip tickers
        that already have a complete cache.
    force_rebuild:
        Rebuild the combined parquet even if it already exists and is fresh.

    Returns
    -------
    pd.DataFrame
        Multi-ticker daily panel with columns:
        date, ticker, open, high, low, close, adjClose, volume, d_* indicators.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tickers = get_sp500_tickers(api_key)
    print(f"[downloader] {len(tickers)} S&P 500 constituents")

    for idx, ticker in enumerate(tickers, 1):
        print(f"  [{idx:3d}/{len(tickers)}] {ticker:<6s}", end="", flush=True)
        if update:
            status = update_ticker(ticker, api_key, start, end, CACHE_DIR, FMP_REQUEST_DELAY)
        else:
            ok = process_ticker(ticker, api_key, start, end, CACHE_DIR, FMP_REQUEST_DELAY)
            status = "OK" if ok else "FAIL"
        print(f"  {status}")

    # Rebuild or load the combined panel
    daily_cache = CACHE_DIR / "daily"
    if force_rebuild or not COMBINED_PARQUET.exists():
        print("[downloader] Building combined parquet …")
        panel = build_combined_parquet(daily_cache, COMBINED_PARQUET)
    else:
        print(f"[downloader] Loading existing panel from {COMBINED_PARQUET}")
        panel = pd.read_parquet(COMBINED_PARQUET)

    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(
        f"[downloader] Panel shape: {panel.shape}  |  "
        f"dates: {panel['date'].min().date()} → {panel['date'].max().date()}  |  "
        f"tickers: {panel['ticker'].nunique()}"
    )
    return panel


def load_panel(path: Path | None = None) -> pd.DataFrame:
    """Load the pre-built combined daily panel parquet."""
    p = path or COMBINED_PARQUET
    if not p.exists():
        raise FileNotFoundError(
            f"Panel not found at {p}. Run ensure_data() first."
        )
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)
