"""Main simulation loop: walk-forward back-test with monthly retraining.

``run_simulation`` is the top-level function.  It accepts a pre-built feature
panel and a model configuration (horizon + feature_set + top_n), then:

  1. Trains an initial CatBoost model on the training window.
  2. Steps through every trading day from SIM_START to SIM_END.
  3. On the first day of each new calendar month, retrains the model on the
     expanding window of historical data up to (but not including) that date.
  4. Each day:
        a. Check trailing stops against today's high/low  → possible exits.
        b. Classify the regime; compute leverage.
        c. Predict scores for today's universe.
        d. Rebalance toward top_n stocks at the computed leverage.
        e. Record equity.
  5. Returns a ``SimulationResult`` dataclass containing the equity curve,
     regime series, stop-exit log, and a metrics dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from catboost_trader.config import (
    INITIAL_CAPITAL,
    SIM_END,
    SIM_START,
    TRAILING_STOP_PCT,
    TRAIN_START,
)
from catboost_trader.brokers.interface import BrokerInterface
from catboost_trader.brokers.paper import PaperBroker
from catboost_trader.data.features import build_feature_matrix, get_feature_columns
from catboost_trader.data.targets import build_targets
from catboost_trader.models.catboost_model import CatBoostTrader
from catboost_trader.simulation.portfolio import Portfolio
from catboost_trader.simulation.regime import (
    build_regime_signals,
    classify_regime,
    get_leverage,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Container for all outputs of a single simulation run."""

    horizon: int
    feature_set: str
    top_n: int
    equity_curve: pd.Series = field(default_factory=pd.Series)
    regime_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    stop_exits: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict = field(default_factory=dict)
    retrain_dates: list[pd.Timestamp] = field(default_factory=list)
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"h{self.horizon}_{self.feature_set}_top{self.top_n}"


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def run_simulation(
    panel: pd.DataFrame,
    horizon: int,
    feature_set: str,
    top_n: int,
    *,
    train_start: str = TRAIN_START,
    sim_start: str = SIM_START,
    sim_end: str = SIM_END,
    initial_capital: float = INITIAL_CAPITAL,
    broker: BrokerInterface | None = None,
    verbose: bool = True,
) -> SimulationResult:
    """Run a complete walk-forward simulation.

    Parameters
    ----------
    panel:
        Multi-ticker daily panel with d_* (and optionally af_*) columns.
        Must span at least ``train_start`` through ``sim_end``.
    horizon:
        CatBoost forward-return horizon in trading days (1, 3, or 5).
    feature_set:
        ``"original"`` or ``"alpha"``.
    top_n:
        Number of long positions to hold simultaneously.
    train_start:
        First date of the initial training window.
    sim_start:
        First date of the simulation (live) period.
    sim_end:
        Last date of the simulation.
    initial_capital:
        Starting portfolio value.
    broker:
        Broker backend.  Defaults to ``PaperBroker``.
    verbose:
        Print progress if True.

    Returns
    -------
    SimulationResult
    """
    result = SimulationResult(horizon=horizon, feature_set=feature_set, top_n=top_n)

    # ── 1. Prepare feature matrix and targets ────────────────────────────
    if verbose:
        print(f"\n[engine] Building feature matrix (feature_set={feature_set}) …")

    # Only compute af_* if needed — expensive for 500 tickers × 16 years
    full_panel = build_feature_matrix(panel, feature_set)

    if verbose:
        print(f"[engine] Building targets (horizon={horizon}) …")
    full_panel = build_targets(full_panel, horizon)

    full_panel["date"] = pd.to_datetime(full_panel["date"])
    feat_cols = get_feature_columns(full_panel, feature_set)

    # ── 2. Build broker ───────────────────────────────────────────────────
    if broker is None:
        broker = PaperBroker(full_panel, initial_capital=initial_capital)

    portfolio = Portfolio(broker, trailing_stop_pct=TRAILING_STOP_PCT)

    # ── 3. Pre-compute regime signals over entire date range ─────────────
    if verbose:
        print("[engine] Building regime signals …")
    regime_signals = build_regime_signals(full_panel)

    # ── 4. Get trading dates in simulation window ─────────────────────────
    all_dates = (
        full_panel[
            (full_panel["date"] >= pd.Timestamp(sim_start))
            & (full_panel["date"] <= pd.Timestamp(sim_end))
        ]["date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(all_dates) == 0:
        raise ValueError(f"No trading dates found between {sim_start} and {sim_end}.")

    # ── 5. Initial model training ─────────────────────────────────────────
    model = CatBoostTrader(horizon=horizon, feature_set=feature_set)
    train_mask = (full_panel["date"] >= pd.Timestamp(train_start)) & (
        full_panel["date"] < pd.Timestamp(sim_start)
    )
    _retrain(model, full_panel, feat_cols, train_mask, pd.Timestamp(sim_start), verbose)
    result.retrain_dates.append(pd.Timestamp(sim_start))

    # Realised-vol tracker (20-day rolling on portfolio daily returns)
    _portfolio_returns: list[float] = []

    if verbose:
        print(f"[engine] Simulating {len(all_dates)} trading days …")

    # ── 6. Main daily loop ────────────────────────────────────────────────
    last_retrain_month: int = pd.Timestamp(sim_start).month - 1   # force first retrain
    prev_equity: float = initial_capital

    for i, date in enumerate(all_dates):
        date = pd.Timestamp(date)

        # ── 6a. Monthly retraining check ──────────────────────────────────
        if date.month != last_retrain_month:
            # Retrain on all data from train_start up to yesterday
            train_mask = (full_panel["date"] >= pd.Timestamp(train_start)) & (
                full_panel["date"] < date
            )
            if train_mask.sum() >= 200:
                _retrain(model, full_panel, feat_cols, train_mask, date, verbose)
                result.retrain_dates.append(date)
            last_retrain_month = date.month

        # ── 6b. Regime classification ─────────────────────────────────────
        if date in regime_signals.index:
            score = float(regime_signals.loc[date, "score"])
            history = regime_signals.loc[regime_signals.index < date, "score"]
        else:
            score = 0.5
            history = pd.Series(dtype=float)

        portfolio_dd = portfolio.current_drawdown()
        realised_vol = _realised_vol(_portfolio_returns)
        regime = classify_regime(score, history)
        leverage = get_leverage(regime, realised_vol, portfolio_dd)

        # ── 6c. Trailing stop checks (before rebalance) ──────────────────
        today_ohlcv = full_panel[full_panel["date"] == date]
        portfolio.check_stops(today_ohlcv, date)

        # ── 6d. Predict scores if not in BEAR ────────────────────────────
        if leverage > 0.0 and model.is_fitted:
            today_features = today_ohlcv.set_index("ticker")
            available_feat_cols = [c for c in feat_cols if c in today_features.columns]
            if len(available_feat_cols) > 0:
                X_today = today_features[available_feat_cols]
                scores = model.predict(X_today)
            else:
                scores = pd.Series(dtype=float)
        else:
            scores = pd.Series(dtype=float)

        # ── 6e. Rebalance ────────────────────────────────────────────────
        portfolio.rebalance(scores, top_n, leverage, date, today_ohlcv)

        # ── 6f. Record daily equity ───────────────────────────────────────
        equity = portfolio.record_daily(date, regime, leverage)
        daily_ret = equity / prev_equity - 1.0 if prev_equity > 0 else 0.0
        _portfolio_returns.append(daily_ret)
        if len(_portfolio_returns) > 60:
            _portfolio_returns.pop(0)
        prev_equity = equity

        if verbose and (i % 100 == 0 or i == len(all_dates) - 1):
            n_pos = len(broker.get_positions())
            print(
                f"  {date.date()}  regime={regime:<7s}  lev={leverage:.2f}x  "
                f"equity=${equity:,.0f}  positions={n_pos}"
            )

    # ── 7. Assemble result ────────────────────────────────────────────────
    result.equity_curve = portfolio.get_equity_series()
    result.regime_df = portfolio.get_regime_series()
    result.stop_exits = portfolio.get_stop_exit_log()

    from catboost_trader.analysis.metrics import compute_metrics
    result.metrics = compute_metrics(result.equity_curve)

    if verbose:
        _print_summary(result)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _retrain(
    model: CatBoostTrader,
    panel: pd.DataFrame,
    feat_cols: list[str],
    mask: pd.Series,
    ref_date: pd.Timestamp,
    verbose: bool,
) -> None:
    """Fit the model on the masked subset of the panel."""
    subset = panel[mask].copy()
    if subset.empty or "target" not in subset.columns:
        return
    # Drop rows with no target (last h rows per ticker)
    subset = subset.dropna(subset=["target"])
    if len(subset) < 200:
        return

    available_cols = [c for c in feat_cols if c in subset.columns]
    X = subset[["date"] + available_cols].copy()
    y = subset["target"]

    try:
        model.fit(X, y, ref_date=ref_date, date_col="date")
    except Exception as exc:
        if verbose:
            print(f"  [retrain] FAILED on {ref_date.date()}: {exc}")


def _realised_vol(returns: list[float], annualise: bool = True) -> float:
    """Annualised realised vol from a list of daily returns."""
    if len(returns) < 5:
        return 0.15   # default to 15% until we have enough history
    vol = float(np.std(returns, ddof=1))
    if annualise:
        vol *= np.sqrt(252)
    return max(vol, 0.001)   # floor at 0.1% to avoid division by zero


def _print_summary(result: SimulationResult) -> None:
    m = result.metrics
    print(f"\n{'='*60}")
    print(f"  {result.label}")
    print(f"  Annual return : {m.get('annual_return', 0):.1%}")
    print(f"  Sharpe        : {m.get('sharpe', 0):.2f}")
    print(f"  Max drawdown  : {m.get('max_drawdown', 0):.1%}")
    print(f"  Retrains      : {len(result.retrain_dates)}")
    print(f"  Stop exits    : {len(result.stop_exits)}")
    print(f"{'='*60}")
