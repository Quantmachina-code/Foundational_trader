"""Prepare weekly multi-asset panel data and save to parquet.

This script reuses the feature engineering pipeline from ``main_weekly.py`` and
stores the resulting panel so model scripts can run without re-downloading.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from main_weekly import (
    BENCHMARK,
    EXTRA_TICKERS,
    N_QUANTILE_DECILES,
    START_DATE,
    END_DATE,
    build_panel_dataset,
    compute_technical_indicators,
    download_weekly_data,
    get_sp500_tickers,
)


def build_universe(include_sp500: bool = True) -> list[str]:
    """Build deduplicated ticker universe."""
    sp500 = get_sp500_tickers() if include_sp500 else []
    seen = set(sp500)
    extras = [ticker for ticker in EXTRA_TICKERS if ticker not in seen]
    return sp500 + extras


def prepare_panel(start: str, end: str, include_sp500: bool = True) -> pd.DataFrame:
    """Download data, compute indicators, and assemble panel."""
    tickers = build_universe(include_sp500=include_sp500)
    raw_data = download_weekly_data(tickers, start=start, end=end)
    if not raw_data:
        raise RuntimeError("No data downloaded.")

    all_feat: dict[str, pd.DataFrame] = {}
    for ticker, ticker_df in raw_data.items():
        all_feat[ticker] = compute_technical_indicators(ticker_df)

    if BENCHMARK not in all_feat:
        raise RuntimeError(f"Benchmark {BENCHMARK} not found in downloaded data.")

    spy_open = all_feat[BENCHMARK]["Open"]
    macro_refs = {
        k: all_feat[k]
        for k in ["SPY", "BTC-USD", "ETH-USD", "GLD", "SLV", "USO", "GC=F", "CL=F"]
        if k in all_feat
    }
    panel = build_panel_dataset(
        all_feat=all_feat,
        spy_open=spy_open,
        macro_refs=macro_refs,
        n_deciles=N_QUANTILE_DECILES,
    )
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare panel data to parquet.")
    parser.add_argument("--start", default=START_DATE, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=END_DATE, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--output",
        default="panel_data_weekly.parquet",
        help="Output parquet file path.",
    )
    parser.add_argument(
        "--extras-only",
        action="store_true",
        help="Skip S&P 500 constituents and only use EXTRA_TICKERS.",
    )
    args = parser.parse_args()

    panel = prepare_panel(args.start, args.end, include_sp500=not args.extras_only)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)

    print(f"Saved panel to {output_path} with shape={panel.shape}")


if __name__ == "__main__":
    main()
