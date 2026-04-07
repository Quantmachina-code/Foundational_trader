"""eToro EOD Trading Bot — MaxLev_Bear25 strategy, horizon h=3.

Run this script 10 minutes before market close (3:50 PM ET on trading days).

What it does each run
---------------------
1. Download / update S&P 500 daily data from FMP (incremental).
2. Build the full feature matrix (Cell 4+5 pipeline from the notebook).
3. Compute today's regime score → long/short weights + leverage.
4. Close any tranche opened exactly 3 trading days ago.
5. Predict with the exported CatBoost models:
     - orig_model_h3  → rank all stocks (long leg, top 10)
     - af_model_h3    → rank all stocks (short leg, bottom 10)
6. Open a new tranche: buy top 10 long, sell short bottom 10.
7. Update tranche state.

Usage
-----
    # Live trading (eToro credentials required):
    python etoro_eod_trader.py \
        --export-dir etoro_model_export/ \
        --fmp-key $FMP_API_KEY \
        --etoro-key $ETORO_API_KEY \
        --etoro-account $ETORO_ACCOUNT_ID

    # Dry-run (no orders placed — for testing):
    python etoro_eod_trader.py \
        --export-dir etoro_model_export/ \
        --fmp-key $FMP_API_KEY \
        --dry-run

    # Cron schedule (10 min before NYSE close Mon-Fri):
    50 15 * * 1-5  cd /path/to/repo && python etoro_eod_trader.py >> logs/eod.log 2>&1

Required environment variables (alternative to CLI flags):
    FMP_API_KEY, ETORO_API_KEY, ETORO_ACCOUNT_ID
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from catboost_trader.etoro_scenario import (
    SCENARIO,
    apply_risk_controls,
    compute_position_sizes,
    get_allocation,
    get_raw_leverage,
    quantize_leverage,
    rescale_score,
)
from catboost_trader.trading.feature_pipeline import build_features_for_prediction
from catboost_trader.trading.live_regime import RegimeTracker
from catboost_trader.trading.tranche_manager import TrancheManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="eToro EOD trading bot — MaxLev_Bear25, h=3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--export-dir",
        default="etoro_model_export",
        help="Directory produced by export_notebook_models.py",
    )
    p.add_argument(
        "--fmp-key",
        default=os.environ.get("FMP_API_KEY", ""),
        help="FMP API key (or set FMP_API_KEY).",
    )
    p.add_argument(
        "--etoro-key",
        default=os.environ.get("ETORO_API_KEY", ""),
        help="eToro API key (or set ETORO_API_KEY).",
    )
    p.add_argument(
        "--etoro-account",
        default=os.environ.get("ETORO_ACCOUNT_ID", ""),
        help="eToro account ID (or set ETORO_ACCOUNT_ID).",
    )
    p.add_argument(
        "--state-file",
        default="etoro_tranche_state.json",
        help="JSON file for tranche state persistence.",
    )
    p.add_argument(
        "--portfolio-history",
        default="etoro_portfolio_history.json",
        help="JSON file for cumulative portfolio return tracking.",
    )
    p.add_argument(
        "--data-path",
        default="sp500_daily.parquet",
        help="Path to sp500_daily.parquet (updated by downloader).",
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Skip FMP data download (use existing parquet).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute everything but do NOT place any eToro orders.",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Use eToro demo (paper-trading) API environment.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=SCENARIO["top_n"],
        help="Number of long positions per tranche.",
    )
    p.add_argument(
        "--bottom-n",
        type=int,
        default=SCENARIO["bottom_n"],
        help="Number of short positions per tranche.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Portfolio history tracking
# ---------------------------------------------------------------------------

def _load_portfolio_history(path: str) -> dict:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return {"cum_value": 1.0, "cum_peak": 1.0, "daily_rets": []}


def _save_portfolio_history(path: str, hist: dict) -> None:
    Path(path).write_text(json.dumps(hist, indent=2))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class _ModelBundle:
    """Holds a CatBoost model + its fitted imputer/selector + feature lists."""

    def __init__(
        self,
        model:          CatBoostRegressor,
        imp,            # sklearn SimpleImputer
        sel,            # sklearn SelectKBest
        all_feat_cols:  list[str],
        sel_feat_cols:  list[str],
        label:          str,
    ) -> None:
        self.model         = model
        self.imp           = imp
        self.sel           = sel
        self.all_feat_cols = all_feat_cols
        self.sel_feat_cols = sel_feat_cols
        self.label         = label

    def predict(self, df_scaled: pd.DataFrame) -> pd.Series:
        """Return a score Series indexed by ticker."""
        valid = [c for c in self.all_feat_cols if c in df_scaled.columns]
        Xv    = df_scaled[valid].values.astype(np.float32)
        Xv    = self.imp.transform(Xv)
        Xv    = self.sel.transform(Xv)
        scores = self.model.predict(Xv)
        return pd.Series(scores, index=df_scaled.index, name="score")


def _load_models(export_dir: Path, horizon: int = 3) -> tuple[_ModelBundle, _ModelBundle]:
    """Load orig (long) and af (short) model bundles."""

    def _load_bundle(label: str) -> _ModelBundle:
        model = CatBoostRegressor()
        model.load_model(str(export_dir / f"{label}_model_h{horizon}.cbm"))

        with open(export_dir / f"{label}_pipeline_h{horizon}.pkl", "rb") as f:
            pipe = pickle.load(f)

        return _ModelBundle(
            model         = model,
            imp           = pipe["imp"],
            sel           = pipe["sel"],
            all_feat_cols = pipe["all_feat_cols"],
            sel_feat_cols = pipe["sel_feat_cols"],
            label         = label,
        )

    orig_bundle = _load_bundle("orig")
    af_bundle   = _load_bundle("af")
    logger.info(
        "Models loaded  orig=%d feat  af=%d feat",
        len(orig_bundle.sel_feat_cols),
        len(af_bundle.sel_feat_cols),
    )
    return orig_bundle, af_bundle


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def _update_data(fmp_key: str, data_path: str) -> pd.DataFrame:
    """Incrementally update sp500_daily.parquet via FMP."""
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).parent))

    from download_sp500_fmp import (
        build_combined_parquet,
        get_sp500_tickers,
        update_ticker,
    )
    from catboost_trader.config import FMP_REQUEST_DELAY

    data_p = Path(data_path)
    cache_dir = data_p.parent / "fmp_ticker_cache" / "daily"
    cache_dir.mkdir(parents=True, exist_ok=True)

    today   = pd.Timestamp.today().strftime("%Y-%m-%d")
    start   = "2009-01-01"   # enough history for 252-day rolling features

    tickers = get_sp500_tickers(fmp_key)
    logger.info("Updating %d tickers to %s …", len(tickers), today)

    for i, ticker in enumerate(tickers, 1):
        try:
            update_ticker(ticker, fmp_key, start, today, cache_dir.parent, FMP_REQUEST_DELAY)
        except Exception as exc:
            logger.warning("[%d/%d] %s failed: %s", i, len(tickers), ticker, exc)
        if i % 50 == 0:
            logger.info("  %d/%d done", i, len(tickers))

    logger.info("Building combined parquet …")
    df = build_combined_parquet(cache_dir, data_p)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    logger.info("=" * 60)
    logger.info("eToro EOD Bot — MaxLev_Bear25  h=3")
    logger.info("=" * 60)
    logger.info("dry_run=%s  demo=%s  top_n=%d  bottom_n=%d",
                args.dry_run, args.demo, args.top_n, args.bottom_n)

    export_dir = Path(args.export_dir)
    if not export_dir.exists():
        logger.error("Export dir not found: %s", export_dir)
        sys.exit(1)

    # ── Load meta ─────────────────────────────────────────────────────────
    meta = json.loads((export_dir / "meta.json").read_text())
    score_min = float(meta["score_min"])
    score_max = float(meta["score_max"])
    top_n     = args.top_n
    bottom_n  = args.bottom_n
    horizon   = int(meta["horizon"])

    logger.info("Loaded meta  horizon=%d  score_range=[%.4f, %.4f]",
                horizon, score_min, score_max)

    # ── 1. Download / load data ───────────────────────────────────────────
    if args.no_download:
        logger.info("Loading existing parquet from %s …", args.data_path)
        df_raw = pd.read_parquet(args.data_path)
        df_raw["date"] = pd.to_datetime(df_raw["date"])
    else:
        if not args.fmp_key:
            logger.error("--fmp-key or FMP_API_KEY required for data download.")
            sys.exit(1)
        t0 = time.time()
        df_raw = _update_data(args.fmp_key, args.data_path)
        logger.info("Data download complete in %.0fs  shape=%s", time.time() - t0, df_raw.shape)

    today_date = df_raw["date"].max()
    logger.info("Latest data date: %s", today_date.date())

    # ── 2. Feature engineering (notebook Cell 4 + 5) ─────────────────────
    logger.info("Running feature pipeline …")
    t0 = time.time()
    df_scaled = build_features_for_prediction(df_raw)
    logger.info("Feature pipeline done in %.0fs  shape=%s", time.time() - t0, df_scaled.shape)

    # Extract today's cross-section
    today_df = df_scaled[df_scaled["date"] == today_date].copy()
    today_df = today_df.set_index("ticker")
    logger.info("Today's cross-section: %d tickers", len(today_df))

    if len(today_df) < 50:
        logger.error("Only %d tickers for %s — aborting.", len(today_df), today_date.date())
        sys.exit(1)

    # ── 3. Load models ────────────────────────────────────────────────────
    orig_bundle, af_bundle = _load_models(export_dir, horizon)

    # ── 4. Predict scores ─────────────────────────────────────────────────
    logger.info("Predicting …")
    orig_scores = orig_bundle.predict(today_df)   # for long leg
    af_scores   = af_bundle.predict(today_df)      # for short leg

    long_picks  = orig_scores.nlargest(top_n)
    short_picks = af_scores.nsmallest(bottom_n)

    logger.info("Long picks (top %d by orig model):", top_n)
    for ticker, score in long_picks.items():
        logger.info("  LONG   %-6s  score=%.4f", ticker, score)

    logger.info("Short picks (bottom %d by af model):", bottom_n)
    for ticker, score in short_picks.items():
        logger.info("  SHORT  %-6s  score=%.4f", ticker, score)

    # ── 5. Regime & leverage ──────────────────────────────────────────────
    today_panel = df_scaled[df_scaled["date"] == today_date]
    regime_tracker = RegimeTracker.from_export(export_dir)
    regime, raw_score, rs = regime_tracker.update(today_panel, date=today_date)

    long_w, short_w = get_allocation(rs)
    raw_lev = get_raw_leverage(rs)

    # Portfolio history for risk controls
    port_hist  = _load_portfolio_history(args.portfolio_history)
    daily_rets = port_hist.get("daily_rets", [])
    cum_value  = float(port_hist.get("cum_value", 1.0))
    cum_peak   = float(port_hist.get("cum_peak", 1.0))

    actual_lev_cont = apply_risk_controls(raw_lev, daily_rets, cum_value, cum_peak)
    lev_discrete    = quantize_leverage(actual_lev_cont)

    logger.info(
        "Regime=%s  score_raw=%.4f  score_rs=%.4f  "
        "long_w=%.0f%%  short_w=%.0f%%  "
        "lev_cont=%.2fx  lev_discrete=%dx",
        regime, raw_score, rs,
        long_w * 100, short_w * 100,
        actual_lev_cont, lev_discrete,
    )

    # ── 6. Connect to eToro (or dry-run) ──────────────────────────────────
    if args.dry_run:
        logger.info("[DRY-RUN] No orders will be placed.")
        broker = None
    else:
        if not args.etoro_key or not args.etoro_account:
            logger.error("--etoro-key and --etoro-account required for live trading.")
            sys.exit(1)
        from catboost_trader.brokers.etoro import EtoroBroker
        broker = EtoroBroker(
            api_key    = args.etoro_key,
            account_id = args.etoro_account,
            demo       = args.demo,
        )

    # ── 7. Close expired tranches ─────────────────────────────────────────
    tranche_mgr = TrancheManager(args.state_file, horizon=horizon)
    logger.info(tranche_mgr.summary())

    if broker is not None:
        closed = tranche_mgr.close_expired_tranches(broker, today=today_date)
        if closed:
            logger.info("Closed %d expired tranche(s).", len(closed))
    else:
        expired = tranche_mgr.get_expired_tranches(today=today_date)
        if expired:
            logger.info("[DRY-RUN] Would close %d expired tranche(s):", len(expired))
            for t in expired:
                logger.info("  tranche %s  L=%s  S=%s",
                            t["open_date"],
                            list(t["long_position_ids"].keys()),
                            list(t["short_position_ids"].keys()))

    # ── 8. Size positions ─────────────────────────────────────────────────
    if broker is not None:
        balance = broker.get_balance()
    else:
        balance = float(meta.get("demo_capital", 2000.0))
        logger.info("[DRY-RUN] Using demo capital $%.2f", balance)

    n_active_after_close = tranche_mgr.n_active()   # already updated
    # After we add this new tranche it becomes n_active_after_close + 1
    n_tranches = max(n_active_after_close + 1, 1)

    long_usd_per_pos, short_usd_per_pos = compute_position_sizes(
        capital           = balance,
        actual_lev        = actual_lev_cont,
        long_weight       = long_w,
        short_weight      = short_w,
        n_active_tranches = n_tranches,
    )

    logger.info(
        "Position sizing  balance=$%.2f  n_tranches=%d  "
        "long=$%.2f/pos x%d  short=$%.2f/pos x%d  lev=%dx",
        balance, n_tranches,
        long_usd_per_pos, top_n,
        short_usd_per_pos, bottom_n,
        lev_discrete,
    )

    # ── 9. Open new tranche ───────────────────────────────────────────────
    long_position_ids:  dict[str, str] = {}
    short_position_ids: dict[str, str] = {}

    if broker is not None:
        logger.info("Opening long positions …")
        for ticker in long_picks.index:
            try:
                pid = broker.open_long(ticker, long_usd_per_pos, leverage=lev_discrete)
                long_position_ids[ticker] = pid
                logger.info("  LONG  %-6s  $%.2f  lev=%dx  pid=%s",
                            ticker, long_usd_per_pos, lev_discrete, pid)
                time.sleep(0.5)   # brief pause between orders
            except Exception as exc:
                logger.error("  LONG  %-6s  FAILED: %s", ticker, exc)
                long_position_ids[ticker] = ""

        logger.info("Opening short positions …")
        for ticker in short_picks.index:
            try:
                pid = broker.open_short(ticker, short_usd_per_pos, leverage=lev_discrete)
                short_position_ids[ticker] = pid
                logger.info("  SHORT %-6s  $%.2f  lev=%dx  pid=%s",
                            ticker, short_usd_per_pos, lev_discrete, pid)
                time.sleep(0.5)
            except Exception as exc:
                logger.error("  SHORT %-6s  FAILED: %s", ticker, exc)
                short_position_ids[ticker] = ""
    else:
        logger.info("[DRY-RUN] Would open positions:")
        for ticker in long_picks.index:
            long_position_ids[ticker] = f"DRY_LONG_{ticker}"
            logger.info("  LONG  %-6s  $%.2f  lev=%dx", ticker, long_usd_per_pos, lev_discrete)
        for ticker in short_picks.index:
            short_position_ids[ticker] = f"DRY_SHORT_{ticker}"
            logger.info("  SHORT %-6s  $%.2f  lev=%dx", ticker, short_usd_per_pos, lev_discrete)

    # ── 10. Persist tranche state ──────────────────────────────────────────
    tranche_mgr.open_tranche(
        open_date          = today_date,
        long_position_ids  = long_position_ids,
        short_position_ids = short_position_ids,
        leverage           = lev_discrete,
        long_weight        = long_w,
        short_weight       = short_w,
        regime             = regime,
        regime_score       = raw_score,
    )

    # ── 11. Update portfolio history ──────────────────────────────────────
    # We can't know today's P&L until positions are closed; we record 0
    # for now and update in a future run once we have closing prices.
    # (A more advanced version would query open P&L from eToro each run.)
    _save_portfolio_history(args.portfolio_history, port_hist)

    # ── 12. Final summary ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EOD RUN COMPLETE  %s", today_date.date())
    logger.info("=" * 60)
    logger.info("  Regime     : %s  (score %.4f  →  %.4f)", regime, raw_score, rs)
    logger.info("  Leverage   : %.2fx  →  discrete %dx", actual_lev_cont, lev_discrete)
    logger.info("  Allocation : %.0f%% long / %.0f%% short",
                long_w * 100, short_w * 100)
    logger.info("  New tranche:")
    logger.info("    Long : %s", list(long_picks.index))
    logger.info("    Short: %s", list(short_picks.index))
    logger.info("  Active tranches after: %d", tranche_mgr.n_active())
    logger.info("  Dry-run: %s", args.dry_run)


if __name__ == "__main__":
    main()
