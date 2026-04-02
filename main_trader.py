"""Entry point for live / paper trading.

This script loads a pre-trained CatBoost model and runs a single trading
cycle (predict → rebalance → report).  It is designed to be called on a
daily schedule (e.g. via cron after market close).

Usage
-----
# Paper trading (uses PaperBroker with today's panel data):
    python main_trader.py \\
        --api-key $FMP_API_KEY \\
        --horizon 1 --features alpha --top-n 10 \\
        --model-path models_saved/h1_alpha.pkl \\
        --capital 100000

# Live trading via eToro (once EtoroBroker is implemented):
    python main_trader.py \\
        --api-key $FMP_API_KEY \\
        --etoro-key $ETORO_API_KEY \\
        --etoro-account $ETORO_ACCOUNT_ID \\
        --horizon 1 --features alpha --top-n 10 \\
        --model-path models_saved/h1_alpha.pkl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from catboost_trader.config import (
    MODEL_DIR,
    SIM_END,
    TRAIN_START,
)
from catboost_trader.data.downloader import ensure_data
from catboost_trader.data.features import build_feature_matrix, get_feature_columns
from catboost_trader.models.catboost_model import CatBoostTrader
from catboost_trader.simulation.regime import build_regime_signals, classify_regime, get_leverage
from catboost_trader.simulation.portfolio import Portfolio


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CatBoost live/paper trading cycle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("FMP_API_KEY", ""),
        help="FMP API key.",
    )
    p.add_argument("--horizon", type=int, choices=[1, 3, 5], default=1)
    p.add_argument("--features", choices=["original", "alpha"], default="alpha")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument(
        "--model-path",
        default=None,
        help="Path to a saved CatBoostTrader pickle.  If not provided, trains fresh.",
    )
    p.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Starting capital for paper trading.",
    )
    # eToro credentials (optional — live trading)
    p.add_argument("--etoro-key", default=None, help="eToro API key.")
    p.add_argument("--etoro-account", default=None, help="eToro account ID.")

    p.add_argument(
        "--retrain",
        action="store_true",
        help="Force model retraining before predicting.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.api_key:
        print("ERROR: --api-key or env FMP_API_KEY required.")
        sys.exit(1)

    # ── 1. Update data to latest ──────────────────────────────────────────
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"[trader] Updating data to {today} …")
    panel = ensure_data(args.api_key, end=today, update=True)

    # ── 2. Build features ─────────────────────────────────────────────────
    print("[trader] Building features …")
    feat_panel = build_feature_matrix(panel, args.features)
    feat_cols = get_feature_columns(feat_panel, args.features)

    # ── 3. Load or train model ────────────────────────────────────────────
    model_path = (
        Path(args.model_path) if args.model_path
        else MODEL_DIR / f"h{args.horizon}_{args.features}.pkl"
    )

    if model_path.exists() and not args.retrain:
        print(f"[trader] Loading model from {model_path} …")
        model = CatBoostTrader.load(model_path)
    else:
        print("[trader] Training fresh model …")
        model = CatBoostTrader(
            horizon=args.horizon,
            feature_set=args.features,
        )
        ref_date = pd.Timestamp(today)
        train_mask = feat_panel["date"] < ref_date
        subset = feat_panel[train_mask].dropna(subset=["target"]) if "target" in feat_panel.columns else feat_panel[train_mask]

        if "target" not in feat_panel.columns:
            from catboost_trader.data.targets import build_targets
            feat_panel = build_targets(feat_panel, args.horizon)
            subset = feat_panel[feat_panel["date"] < ref_date].dropna(subset=["target"])

        available_cols = [c for c in feat_cols if c in subset.columns]
        X = subset[["date"] + available_cols]
        y = subset["target"]
        model.fit(X, y, ref_date=ref_date, date_col="date")
        model.save(model_path)

    # ── 4. Regime check ───────────────────────────────────────────────────
    print("[trader] Computing regime …")
    regime_signals = build_regime_signals(feat_panel)
    today_ts = pd.Timestamp(today)
    if today_ts in regime_signals.index:
        score = float(regime_signals.loc[today_ts, "score"])
    else:
        score = float(regime_signals["score"].iloc[-1])
    history = regime_signals.loc[regime_signals.index < today_ts, "score"]
    regime = classify_regime(score, history)
    leverage = get_leverage(regime, realised_vol=0.15)  # default vol if not tracked
    print(f"[trader] Regime={regime}  score={score:.3f}  leverage={leverage:.2f}x")

    if leverage == 0.0:
        print("[trader] BEAR market — no trades. Exit.")
        return

    # ── 5. Predict today's scores ─────────────────────────────────────────
    today_data = feat_panel[feat_panel["date"] == today_ts]
    if today_data.empty:
        # Use last available date
        last_date = feat_panel["date"].max()
        today_data = feat_panel[feat_panel["date"] == last_date]
        print(f"[trader] No data for {today} — using {last_date.date()}")

    today_features = today_data.set_index("ticker")
    available = [c for c in feat_cols if c in today_features.columns]
    X_today = today_features[available]
    scores = model.predict(X_today)

    # ── 6. Select top-N ───────────────────────────────────────────────────
    top_picks = scores.nlargest(args.top_n)
    print(f"\n[trader] Top-{args.top_n} picks (predicted rank score):")
    for ticker, score in top_picks.items():
        print(f"  {ticker:<6s}  {score:.4f}")

    # ── 7. Execute via broker ─────────────────────────────────────────────
    if args.etoro_key and args.etoro_account:
        from catboost_trader.brokers.etoro import EtoroBroker
        broker = EtoroBroker(api_key=args.etoro_key, account_id=args.etoro_account)
        print("\n[trader] Using eToro live broker.")
    else:
        from catboost_trader.brokers.paper import PaperBroker
        broker = PaperBroker(feat_panel, initial_capital=args.capital)
        broker.set_date(today_ts)
        print("\n[trader] Using PaperBroker (simulation mode).")

    portfolio = Portfolio(broker)
    portfolio.rebalance(scores, top_n=args.top_n, leverage=leverage, date=today_ts)

    positions = broker.get_positions()
    equity = broker.get_balance()
    print(f"\n[trader] Portfolio after rebalance:")
    print(f"  Equity:    ${equity:,.2f}")
    print(f"  Positions: {len(positions)}")
    for t, pos in positions.items():
        print(f"    {t:<6s}  {pos.shares:.2f} shares  @${pos.avg_cost:.2f}")


if __name__ == "__main__":
    main()
