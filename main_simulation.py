"""Entry point for the full walk-forward simulation sweep.

Usage examples
--------------
# Full sweep of all 24 combinations (3 horizons × 2 feature sets × 4 top-N):
    python main_simulation.py --api-key $FMP_API_KEY

# Single run: horizon=1, original features, top-5:
    python main_simulation.py --api-key $FMP_API_KEY --horizon 1 --features original --top-n 5

# Skip data download (panel parquet already exists):
    python main_simulation.py --no-download --horizon 3 --features alpha --top-n 10

# Quick test run (no download, first 50 tickers only):
    python main_simulation.py --no-download --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure the repo root is importable
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from catboost_trader.config import (
    COMBINED_PARQUET,
    FEATURE_SETS,
    HORIZONS,
    INITIAL_CAPITAL,
    RESULTS_DIR,
    SIM_END,
    SIM_START,
    TOP_N_OPTIONS,
    TRAIN_START,
)
from catboost_trader.data.downloader import ensure_data, load_panel
from catboost_trader.simulation.engine import run_simulation
from catboost_trader.analysis.metrics import metrics_to_df
from catboost_trader.analysis.plots import save_all_plots


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CatBoost trading simulation sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("FMP_API_KEY", ""),
        help="FMP API key (or set env FMP_API_KEY).",
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Skip data download; load existing panel parquet.",
    )
    p.add_argument(
        "--horizon",
        type=int,
        choices=[1, 3, 5],
        default=None,
        help="Single horizon to run (default: sweep 1, 3, 5).",
    )
    p.add_argument(
        "--features",
        choices=["original", "alpha", "both"],
        default="both",
        help="Feature set to use.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        choices=[3, 5, 10, 15],
        default=None,
        help="Single top-N to run (default: sweep 3, 5, 10, 15).",
    )
    p.add_argument(
        "--capital",
        type=float,
        default=INITIAL_CAPITAL,
        help="Starting capital.",
    )
    p.add_argument(
        "--output",
        default=str(RESULTS_DIR),
        help="Directory for results (CSV, parquet, plots).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a minimal simulation (first 50 tickers) to validate the pipeline.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving plots (faster).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────
    if args.no_download:
        print("[main] Loading existing panel …")
        panel = load_panel()
    else:
        if not args.api_key:
            print("ERROR: --api-key or env FMP_API_KEY is required for data download.")
            sys.exit(1)
        print("[main] Downloading / updating data …")
        panel = ensure_data(args.api_key)

    if args.dry_run:
        # Limit to first 50 tickers for a quick sanity check
        tickers = sorted(panel["ticker"].unique())[:50]
        panel = panel[panel["ticker"].isin(tickers)].copy()
        print(f"[main] DRY RUN: limited to {len(tickers)} tickers.")

    # ── 2. Build sweep parameters ─────────────────────────────────────────
    horizons = [args.horizon] if args.horizon else HORIZONS
    if args.features == "both":
        feature_sets = FEATURE_SETS
    else:
        feature_sets = [args.features]
    top_ns = [args.top_n] if args.top_n else TOP_N_OPTIONS

    combos = [
        (h, fs, n)
        for h in horizons
        for fs in feature_sets
        for n in top_ns
    ]
    print(f"\n[main] Running {len(combos)} simulation(s) …\n")

    # ── 3. Run simulations ────────────────────────────────────────────────
    all_results: dict[str, object] = {}
    t0 = time.time()

    for i, (h, fs, n) in enumerate(combos, 1):
        label = f"h{h}_{fs}_top{n}"
        print(f"\n[{i}/{len(combos)}] {label}")
        try:
            result = run_simulation(
                panel=panel,
                horizon=h,
                feature_set=fs,
                top_n=n,
                train_start=TRAIN_START,
                sim_start=SIM_START,
                sim_end=SIM_END,
                initial_capital=args.capital,
                verbose=True,
            )
            all_results[label] = result

            # Save equity curve
            eq_path = out_dir / f"equity_{label}.parquet"
            result.equity_curve.to_frame("equity").reset_index().to_parquet(
                eq_path, index=False
            )

        except Exception as exc:
            print(f"  ERROR in {label}: {exc}")
            import traceback
            traceback.print_exc()

    # ── 4. Summary CSV ────────────────────────────────────────────────────
    if all_results:
        summary = metrics_to_df(all_results)
        csv_path = out_dir / "metrics_summary.csv"
        summary.to_csv(csv_path, index=False)
        print(f"\n[main] Summary saved to {csv_path}")
        print(summary.to_string(index=False))

        # ── 5. Plots ──────────────────────────────────────────────────────
        if not args.no_plots:
            try:
                save_all_plots(all_results, output_dir=str(out_dir / "plots"))
            except Exception as exc:
                print(f"[main] Plot generation failed: {exc}")

    elapsed = time.time() - t0
    print(f"\n[main] Done in {elapsed/60:.1f} min.")


if __name__ == "__main__":
    main()
