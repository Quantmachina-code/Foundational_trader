"""eToro Walk-Forward Simulation — MaxLev_Bear25, h=3.

Simulates the production EOD bot logic on historical data without placing
any real orders.  Controlled by environment variables or CLI flags.

Environment variables
---------------------
    SIMULATION_WALKTHROUGH=True     Enable simulation mode
    SIM_START_DATE=2018-01-01       First trading day (inclusive)
    SIM_END_DATE=2025-12-31         Last trading day (inclusive)
    FMP_API_KEY=...                 FMP key for data download
    SIM_EXPORT_DIR=etoro_model_export   Model export directory

CLI usage
---------
    python etoro_walkforward_sim.py \\
        --export-dir etoro_model_export/ \\
        --fmp-key $FMP_API_KEY \\
        --start 2018-01-01 \\
        --end   2025-12-31

    # Skip download (use existing parquet):
    python etoro_walkforward_sim.py --no-download --data-path sp500_daily.parquet

    # Suppress charts:
    python etoro_walkforward_sim.py --no-plot

Outputs (written to --output-dir, default: sim_results/)
---------
    equity_curve.csv      date, equity, drawdown, regime, leverage
    trade_log.csv         date, ticker, side, fwd_ret_3d, pnl
    metrics.json          Sharpe, Calmar, max_DD, CAGR, hit_rate, ...
    equity_curve.png      (if matplotlib available and --no-plot not set)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from catboost_trader.etoro_scenario import (
    SCENARIO,
    apply_risk_controls,
    get_allocation,
    get_raw_leverage,
    quantize_leverage,
    rescale_score,
)
from catboost_trader.trading.feature_pipeline import build_features_for_prediction
from catboost_trader.trading.live_regime import RegimeTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

HORIZON = 3   # trading days per tranche


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward simulation of eToro MaxLev_Bear25 strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--export-dir",
        default=os.environ.get("SIM_EXPORT_DIR", "etoro_model_export"),
        help="Directory produced by export_notebook_models.py.",
    )
    p.add_argument(
        "--fmp-key",
        default=os.environ.get("FMP_API_KEY", ""),
        help="FMP API key (or set FMP_API_KEY).",
    )
    p.add_argument(
        "--start",
        default=os.environ.get("SIM_START_DATE", "2018-01-01"),
        help="Simulation start date (ISO).",
    )
    p.add_argument(
        "--end",
        default=os.environ.get("SIM_END_DATE", "2025-12-31"),
        help="Simulation end date (ISO).",
    )
    p.add_argument(
        "--data-path",
        default="sp500_daily.parquet",
        help="Path to combined sp500 parquet (created/updated by downloader).",
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Skip FMP data download; use existing parquet.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=SCENARIO["top_n"],
        help="Long positions per tranche.",
    )
    p.add_argument(
        "--bottom-n",
        type=int,
        default=SCENARIO["bottom_n"],
        help="Short positions per tranche.",
    )
    p.add_argument(
        "--initial-capital",
        type=float,
        default=100_000.0,
        help="Starting portfolio equity in USD.",
    )
    p.add_argument(
        "--output-dir",
        default="sim_results",
        help="Directory for output files.",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate matplotlib charts.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data download (reused from etoro_eod_trader)
# ---------------------------------------------------------------------------

def _download_data(fmp_key: str, data_path: str, start: str) -> pd.DataFrame:
    from download_sp500_fmp import (
        build_combined_parquet,
        get_sp500_tickers,
        update_ticker,
    )
    from catboost_trader.config import FMP_REQUEST_DELAY

    data_p    = Path(data_path)
    cache_dir = data_p.parent / "fmp_ticker_cache" / "daily"
    cache_dir.mkdir(parents=True, exist_ok=True)

    today   = pd.Timestamp.today().strftime("%Y-%m-%d")
    tickers = get_sp500_tickers(fmp_key)
    logger.info("Downloading %d tickers [%s → %s] …", len(tickers), start, today)

    for i, ticker in enumerate(tickers, 1):
        try:
            update_ticker(ticker, fmp_key, start, today, cache_dir.parent, FMP_REQUEST_DELAY)
        except Exception as exc:
            logger.warning("[%d/%d] %s failed: %s", i, len(tickers), ticker, exc)
        if i % 50 == 0:
            logger.info("  %d/%d tickers done", i, len(tickers))

    logger.info("Building combined parquet …")
    df = build_combined_parquet(cache_dir, data_p)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------

class _ModelBundle:
    def __init__(self, model, imp, sel, all_feat_cols, sel_feat_cols, label):
        self.model         = model
        self.imp           = imp
        self.sel           = sel
        self.all_feat_cols = all_feat_cols
        self.sel_feat_cols = sel_feat_cols
        self.label         = label

    def predict(self, df_scaled: pd.DataFrame) -> pd.Series:
        valid  = [c for c in self.all_feat_cols if c in df_scaled.columns]
        Xv     = df_scaled[valid].values.astype(np.float32)
        Xv     = self.imp.transform(Xv)
        Xv     = self.sel.transform(Xv)
        scores = self.model.predict(Xv)
        return pd.Series(scores, index=df_scaled.index, name="score")


def _load_models(export_dir: Path, horizon: int = 3) -> tuple[_ModelBundle, _ModelBundle]:
    def _load_bundle(label: str) -> _ModelBundle:
        m = CatBoostRegressor()
        m.load_model(str(export_dir / f"{label}_model_h{horizon}.cbm"))
        with open(export_dir / f"{label}_pipeline_h{horizon}.pkl", "rb") as f:
            pipe = pickle.load(f)
        return _ModelBundle(
            model         = m,
            imp           = pipe["imp"],
            sel           = pipe["sel"],
            all_feat_cols = pipe["all_feat_cols"],
            sel_feat_cols = pipe["sel_feat_cols"],
            label         = label,
        )

    orig = _load_bundle("orig")
    af   = _load_bundle("af")
    logger.info("Models loaded  orig=%d feat  af=%d feat",
                len(orig.sel_feat_cols), len(af.sel_feat_cols))
    return orig, af


# ---------------------------------------------------------------------------
# Forward-return computation
# ---------------------------------------------------------------------------

def _compute_forward_returns(df_raw: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """Add fwd_log_ret_{h}d column: sum of next h log-returns per ticker.

    Uses adjClose to compute log returns.  Only dates where a full h-day
    window of future prices exists will have non-NaN values.
    """
    price_col = "adjClose" if "adjClose" in df_raw.columns else "close"
    df = df_raw[["date", "ticker", price_col]].copy()
    df = df.sort_values(["ticker", "date"])

    log_ret = np.log(df[price_col] / df.groupby("ticker")[price_col].shift(1))
    df["_log_ret"] = log_ret

    # Rolling forward sum over next h periods (shift back by h to align to open date)
    fwd_col = f"fwd_log_ret_{horizon}d"
    df[fwd_col] = (
        df.groupby("ticker")["_log_ret"]
          .transform(lambda s: s.rolling(horizon).sum().shift(-horizon))
    )
    df = df.drop(columns=["_log_ret", price_col])
    return df.set_index(["date", "ticker"])[fwd_col]


# ---------------------------------------------------------------------------
# Tranche dataclass
# ---------------------------------------------------------------------------

class _Tranche(NamedTuple):
    open_date:  pd.Timestamp
    longs:      list[str]   # tickers
    shorts:     list[str]
    leverage:   float
    long_w:     float
    short_w:    float
    regime:     str
    regime_rs:  float


# ---------------------------------------------------------------------------
# Daily return from a tranche
# ---------------------------------------------------------------------------

def _tranche_daily_return(
    tranche:      _Tranche,
    date:         pd.Timestamp,
    fwd_ret_map:  pd.Series,  # index=(date, ticker) → fwd_log_ret
    horizon:      int,
) -> float:
    """Estimate single-day contribution of an open tranche on *date*.

    For simplicity we spread the h-day forward return evenly across the h
    holding days, yielding a daily equivalent.  A tranche opened on T
    contributes on days T, T+1, T+2.
    """
    open_date = tranche.open_date

    # Fetch forward return measured from open_date
    long_ret  = 0.0
    short_ret = 0.0
    n_long    = len(tranche.longs)
    n_short   = len(tranche.shorts)

    for tkr in tranche.longs:
        key = (open_date, tkr)
        if key in fwd_ret_map.index:
            long_ret += fwd_ret_map[key]

    for tkr in tranche.shorts:
        key = (open_date, tkr)
        if key in fwd_ret_map.index:
            short_ret += fwd_ret_map[key]   # shorts: negative of price move profits us

    # Average per-ticker return for each leg
    avg_long  = (long_ret  / n_long)  if n_long  else 0.0
    avg_short = -(short_ret / n_short) if n_short else 0.0   # short = −price move

    # Weighted portfolio return (unleveraged), spread over horizon days
    daily_ret = (
        tranche.long_w  * avg_long  +
        tranche.short_w * avg_short
    ) / horizon

    # Apply leverage
    return daily_ret * tranche.leverage


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def _compute_metrics(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    if len(rets) < 2:
        return {}

    ann   = 252
    cagr  = float((equity.iloc[-1] / equity.iloc[0]) ** (ann / len(rets)) - 1)
    vol   = float(rets.std() * np.sqrt(ann))
    sharpe = float(cagr / vol) if vol > 1e-9 else 0.0

    roll_max = equity.cummax()
    dd       = (equity - roll_max) / roll_max
    max_dd   = float(dd.min())
    calmar   = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0

    return {
        "CAGR":     round(cagr,   4),
        "Ann_Vol":  round(vol,    4),
        "Sharpe":   round(sharpe, 3),
        "Max_DD":   round(max_dd, 4),
        "Calmar":   round(calmar, 3),
        "N_Days":   len(equity),
        "Final_Equity": round(float(equity.iloc[-1]), 2),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(equity_df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not available — skipping chart")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # ── Equity ──
    ax = axes[0]
    ax.plot(equity_df["date"], equity_df["equity"], color="steelblue", linewidth=1.2)
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Walk-Forward Simulation — MaxLev_Bear25 h=3")
    ax.grid(True, alpha=0.3)

    # ── Drawdown ──
    ax = axes[1]
    ax.fill_between(equity_df["date"], equity_df["drawdown"] * 100,
                    0, color="crimson", alpha=0.5)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)

    # ── Leverage ──
    ax = axes[2]
    ax.plot(equity_df["date"], equity_df["leverage"], color="darkorange", linewidth=0.8)
    ax.set_ylabel("Leverage")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = output_dir / "equity_curve.png"
    plt.savefig(chart_path, dpi=150)
    plt.close(fig)
    logger.info("Chart saved → %s", chart_path)


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation(args: argparse.Namespace) -> None:
    logger.info("=" * 60)
    logger.info("Walk-Forward Simulation — MaxLev_Bear25  h=3")
    logger.info("=" * 60)
    logger.info("start=%s  end=%s  top_n=%d  bottom_n=%d  capital=%.0f",
                args.start, args.end, args.top_n, args.bottom_n, args.initial_capital)

    export_dir = Path(args.export_dir)
    if not export_dir.exists():
        logger.error("Export dir not found: %s", export_dir)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load meta ─────────────────────────────────────────────────────────
    meta      = json.loads((export_dir / "meta.json").read_text())
    score_min = float(meta["score_min"])
    score_max = float(meta["score_max"])
    horizon   = int(meta.get("horizon", HORIZON))
    top_n     = args.top_n
    bottom_n  = args.bottom_n
    logger.info("Meta loaded  horizon=%d  score_range=[%.4f, %.4f]",
                horizon, score_min, score_max)

    # ── 2. Download / load data ───────────────────────────────────────────────
    data_path = Path(args.data_path)
    if not args.no_download:
        if not args.fmp_key:
            logger.error("--fmp-key required unless --no-download is set.")
            sys.exit(1)
        warmup_start = (
            pd.Timestamp(args.start) - pd.DateOffset(years=2)
        ).strftime("%Y-%m-%d")
        df_raw = _download_data(args.fmp_key, str(data_path), warmup_start)
    else:
        if not data_path.exists():
            logger.error("Parquet not found: %s — run without --no-download first.", data_path)
            sys.exit(1)
        logger.info("Loading parquet: %s …", data_path)
        df_raw = pd.read_parquet(data_path)
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        df_raw = df_raw.sort_values(["date", "ticker"]).reset_index(drop=True)

    logger.info("Raw data: %d rows, %d tickers, dates %s → %s",
                len(df_raw),
                df_raw["ticker"].nunique(),
                df_raw["date"].min().date(),
                df_raw["date"].max().date())

    # ── 3. Feature pipeline (once on full dataset) ───────────────────────────
    logger.info("Running feature pipeline …")
    df_scaled = build_features_for_prediction(df_raw)
    df_scaled["date"] = pd.to_datetime(df_scaled["date"])
    logger.info("Feature pipeline done: %d rows  %d cols",
                len(df_scaled), len(df_scaled.columns))

    # ── 4. Forward returns for PnL measurement ───────────────────────────────
    logger.info("Computing forward returns …")
    fwd_ret_series = _compute_forward_returns(df_raw, horizon=horizon)

    # ── 5. Load models ────────────────────────────────────────────────────────
    orig_bundle, af_bundle = _load_models(export_dir, horizon=horizon)

    # ── 6. Load regime tracker ────────────────────────────────────────────────
    regime_tracker = RegimeTracker.from_export(export_dir)

    # ── 7. Determine trading dates ────────────────────────────────────────────
    sim_start = pd.Timestamp(args.start).normalize()
    sim_end   = pd.Timestamp(args.end).normalize()

    all_dates = sorted(df_scaled["date"].unique())
    trade_dates = [
        d for d in all_dates
        if sim_start <= pd.Timestamp(d).normalize() <= sim_end
    ]
    logger.info("%d trading dates in simulation window", len(trade_dates))
    if not trade_dates:
        logger.error("No trading dates in [%s, %s]. Aborting.", args.start, args.end)
        sys.exit(1)

    # ── 8. Portfolio state ────────────────────────────────────────────────────
    equity      = args.initial_capital
    peak_equity = equity
    running_rets: list[float] = []
    open_tranches: list[_Tranche] = []   # up to h=3 at a time

    # Tracking lists for output
    equity_rows: list[dict] = []
    trade_rows:  list[dict] = []

    logger.info("Starting walk-forward loop …")

    for step, trade_date in enumerate(trade_dates, 1):
        date_ts = pd.Timestamp(trade_date).normalize()

        # ── a. Apply daily P&L from open tranches ─────────────────────────
        day_ret = 0.0
        n_active = max(len(open_tranches), 1)

        for tranche in open_tranches:
            contrib = _tranche_daily_return(tranche, date_ts, fwd_ret_series, horizon)
            day_ret += contrib / n_active   # equal-weight across tranches

        equity     *= (1.0 + day_ret)
        peak_equity = max(peak_equity, equity)
        running_rets.append(day_ret)

        drawdown = (equity - peak_equity) / peak_equity

        # ── b. Close expired tranches ──────────────────────────────────────
        still_open = []
        for tranche in open_tranches:
            n_td = _count_trading_days(tranche.open_date, date_ts)
            if n_td >= horizon:
                logger.debug("Closed tranche opened %s  L=%s  S=%s",
                             tranche.open_date.date(), tranche.longs, tranche.shorts)
            else:
                still_open.append(tranche)
        open_tranches = still_open

        # ── c. Cross-section features for this date ────────────────────────
        day_df = df_scaled[df_scaled["date"] == trade_date].copy()
        if len(day_df) < top_n + bottom_n:
            logger.debug("Skipping %s — too few tickers (%d)", date_ts.date(), len(day_df))
            equity_rows.append({
                "date": date_ts,
                "equity": equity,
                "drawdown": drawdown,
                "regime": "NEUTRAL",
                "leverage": 1.0,
            })
            continue
        day_df = day_df.set_index("ticker")

        # ── d. Regime update (causal) ──────────────────────────────────────
        try:
            regime, raw_score, rs_score = regime_tracker.update(day_df.reset_index(), date_ts)
        except Exception as exc:
            logger.debug("Regime error on %s: %s — defaulting NEUTRAL", date_ts.date(), exc)
            regime   = "NEUTRAL"
            rs_score = 0.0

        # ── e. Leverage & allocation ───────────────────────────────────────
        long_w, short_w = get_allocation(rs_score)
        raw_lev         = get_raw_leverage(rs_score)
        actual_lev      = apply_risk_controls(
            raw_lev,
            running_rets[-20:],
            equity / args.initial_capital,
            peak_equity / args.initial_capital,
        )
        discrete_lev = quantize_leverage(actual_lev)

        # ── f. Model predictions → top/bottom picks ────────────────────────
        try:
            orig_scores = orig_bundle.predict(day_df)
            af_scores   = af_bundle.predict(day_df)
        except Exception as exc:
            logger.warning("Prediction error on %s: %s — skipping tranche", date_ts.date(), exc)
            equity_rows.append({
                "date": date_ts, "equity": equity,
                "drawdown": drawdown, "regime": regime, "leverage": discrete_lev,
            })
            continue

        longs  = list(orig_scores.nlargest(top_n).index)
        shorts = list(af_scores.nsmallest(bottom_n).index)

        # ── g. Open new tranche ────────────────────────────────────────────
        new_tranche = _Tranche(
            open_date = date_ts,
            longs     = longs,
            shorts    = shorts,
            leverage  = float(discrete_lev),
            long_w    = long_w,
            short_w   = short_w,
            regime    = regime,
            regime_rs = rs_score,
        )
        open_tranches.append(new_tranche)

        # ── h. Log trades ──────────────────────────────────────────────────
        for tkr in longs:
            fwd = fwd_ret_series.get((date_ts, tkr), np.nan)
            trade_rows.append({
                "date":        date_ts,
                "ticker":      tkr,
                "side":        "LONG",
                "fwd_ret_3d":  fwd,
                "leverage":    discrete_lev,
                "regime":      regime,
            })
        for tkr in shorts:
            fwd = fwd_ret_series.get((date_ts, tkr), np.nan)
            trade_rows.append({
                "date":        date_ts,
                "ticker":      tkr,
                "side":        "SHORT",
                "fwd_ret_3d":  -fwd if not np.isnan(fwd) else np.nan,
                "leverage":    discrete_lev,
                "regime":      regime,
            })

        if step % 50 == 0:
            logger.info(
                "[%d/%d] %s  equity=%.0f  dd=%.1f%%  lev=%dx  regime=%s  L=%s  S=%s",
                step, len(trade_dates), date_ts.date(),
                equity, drawdown * 100, discrete_lev, regime,
                longs[:3], shorts[:3],
            )

        equity_rows.append({
            "date":      date_ts,
            "equity":    equity,
            "drawdown":  drawdown,
            "regime":    regime,
            "leverage":  discrete_lev,
        })

    # ── 9. Build output DataFrames ────────────────────────────────────────────
    equity_df = pd.DataFrame(equity_rows)
    trade_df  = pd.DataFrame(trade_rows)

    # ── 10. Compute metrics ───────────────────────────────────────────────────
    metrics = _compute_metrics(equity_df.set_index("date")["equity"])
    logger.info("=== Simulation Results ===")
    for k, v in metrics.items():
        logger.info("  %-20s %s", k, v)

    # ── 11. Write outputs ─────────────────────────────────────────────────────
    eq_path = output_dir / "equity_curve.csv"
    tr_path = output_dir / "trade_log.csv"
    mt_path = output_dir / "metrics.json"

    equity_df.to_csv(eq_path, index=False)
    trade_df.to_csv(tr_path, index=False)
    mt_path.write_text(json.dumps(metrics, indent=2))

    logger.info("Outputs written to %s", output_dir)
    logger.info("  %s", eq_path)
    logger.info("  %s", tr_path)
    logger.info("  %s", mt_path)

    # ── 12. Optional chart ────────────────────────────────────────────────────
    if not args.no_plot:
        _plot(equity_df, output_dir)

    # ── 13. Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Walk-Forward Simulation Summary")
    print("=" * 50)
    print(f"  Period:        {trade_dates[0].date()} → {trade_dates[-1].date()}")
    print(f"  Trading days:  {len(trade_dates)}")
    print(f"  Initial cap:   ${args.initial_capital:,.0f}")
    print(f"  Final equity:  ${equity:,.0f}")
    for k, v in metrics.items():
        print(f"  {k:<20s} {v}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Helper: count trading days (same as tranche_manager)
# ---------------------------------------------------------------------------

def _count_trading_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    if end <= start:
        return 0
    return len(pd.bdate_range(start=start + pd.Timedelta(days=1), end=end))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow env-var override of SIMULATION_WALKTHROUGH as a guard
    sim_flag = os.environ.get("SIMULATION_WALKTHROUGH", "").strip().lower()
    if sim_flag and sim_flag not in ("true", "1", "yes"):
        print("SIMULATION_WALKTHROUGH is set but not 'True' — exiting.")
        sys.exit(0)

    run_simulation(_parse_args())
