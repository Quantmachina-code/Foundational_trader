"""Performance metrics for the simulation equity curve.

All functions accept a date-indexed ``pd.Series`` of portfolio equity values
(not returns) and return scalar metrics or a dict of metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def daily_returns(equity: pd.Series) -> pd.Series:
    """Convert an equity curve to daily returns."""
    return equity.pct_change().dropna()


def annual_return(equity: pd.Series) -> float:
    """Compound annual growth rate (CAGR)."""
    rets = daily_returns(equity)
    if len(rets) == 0:
        return 0.0
    n_years = len(rets) / 252.0
    total = (1 + rets).prod()
    if total <= 0 or n_years <= 0:
        return float("-inf")
    return float(total ** (1 / n_years) - 1)


def sharpe_ratio(equity: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio (×√252)."""
    rets = daily_returns(equity)
    if len(rets) < 2:
        return 0.0
    excess = rets - risk_free / 252
    std = excess.std(ddof=1)
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(252))


def sortino_ratio(equity: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    rets = daily_returns(equity)
    if len(rets) < 2:
        return 0.0
    excess = rets - risk_free / 252
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("inf")
    downside_std = downside.std(ddof=1)
    if downside_std == 0:
        return float("inf")
    return float(excess.mean() / downside_std * np.sqrt(252))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction."""
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()
    dd = (equity - running_peak) / running_peak
    return float(-dd.min())


def calmar_ratio(equity: pd.Series) -> float:
    """CAGR / max drawdown."""
    mdd = max_drawdown(equity)
    if mdd == 0:
        return float("inf")
    return float(annual_return(equity) / mdd)


def volatility(equity: pd.Series) -> float:
    """Annualised daily return volatility."""
    rets = daily_returns(equity)
    if len(rets) < 2:
        return 0.0
    return float(rets.std(ddof=1) * np.sqrt(252))


def win_rate(equity: pd.Series) -> float:
    """Fraction of trading days with positive return."""
    rets = daily_returns(equity)
    if len(rets) == 0:
        return 0.0
    return float((rets > 0).mean())


def avg_win_loss(equity: pd.Series) -> tuple[float, float]:
    """Average winning day return and average losing day return (both as fractions)."""
    rets = daily_returns(equity)
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    return avg_win, avg_loss


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Return the drawdown at every point in time as a positive fraction."""
    running_peak = equity.cummax()
    dd = (equity - running_peak) / running_peak
    return (-dd).rename("drawdown")


# ---------------------------------------------------------------------------
# Summary dict
# ---------------------------------------------------------------------------

def compute_metrics(equity: pd.Series) -> dict:
    """Compute all standard metrics and return as a flat dict.

    Parameters
    ----------
    equity:
        Date-indexed Series of portfolio equity values.

    Returns
    -------
    dict with keys:
        annual_return, sharpe, sortino, calmar, max_drawdown, volatility,
        win_rate, avg_win, avg_loss, n_days, start_equity, end_equity.
    """
    if equity.empty or len(equity) < 2:
        return {k: float("nan") for k in [
            "annual_return", "sharpe", "sortino", "calmar",
            "max_drawdown", "volatility", "win_rate", "avg_win", "avg_loss",
            "n_days", "start_equity", "end_equity",
        ]}

    avg_w, avg_l = avg_win_loss(equity)
    return {
        "annual_return": annual_return(equity),
        "sharpe": sharpe_ratio(equity),
        "sortino": sortino_ratio(equity),
        "calmar": calmar_ratio(equity),
        "max_drawdown": max_drawdown(equity),
        "volatility": volatility(equity),
        "win_rate": win_rate(equity),
        "avg_win": avg_w,
        "avg_loss": avg_l,
        "n_days": len(equity),
        "start_equity": float(equity.iloc[0]),
        "end_equity": float(equity.iloc[-1]),
    }


def metrics_to_df(results: dict[str, "SimulationResult"]) -> pd.DataFrame:  # noqa: F821
    """Build a summary DataFrame from a dict of SimulationResult objects.

    Parameters
    ----------
    results:
        Mapping of label → SimulationResult.

    Returns
    -------
    pd.DataFrame sorted by Sharpe ratio descending.
    """
    rows = []
    for label, res in results.items():
        row = {"label": label, "horizon": res.horizon,
               "feature_set": res.feature_set, "top_n": res.top_n}
        row.update(res.metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    if "sharpe" in df.columns:
        df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    return df
