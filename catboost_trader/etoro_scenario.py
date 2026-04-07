"""MaxLev_Bear25 scenario — allocation & leverage functions.

Exact parameter set from the Jupyter notebook scenario engine
(Cell 11 / Cell 12, scenario "MaxLeverage_Bear25"):

  long_base   = 0.85   base long weight
  bull_long   = 1.00   max long in extreme bull
  bear_long   = 0.25   min long in extreme bear
  lev_bull_max = 10.0  max leverage in extreme bull
  lev_neutral  =  5.0  leverage at regime-neutral
  lev_bear_min =  0.5  min leverage in extreme bear
  vol_target   =  0.20 annualised vol target for risk control
  dd_delever   = -0.15 drawdown threshold for de-leveraging
  max_daily_loss = -0.05 hard stop on single-day loss

Regime score is first rescaled to [-1, +1] using the P2/P98 range observed
during the simulation period:
  score_rescaled = clip((raw - mid) / half_range, -1, 1)
  where mid      = (score_min + score_max) / 2
        half_range = (score_max - score_min) / 2

Allocation  (long_weight, short_weight):
  score ≥ 0  →  long_w = long_base + score*(bull_long  - long_base)
  score <  0  →  long_w = long_base + score*(long_base - bear_long)
  long_w = clip(long_w, bear_long, bull_long)
  short_w = 1 - long_w

Leverage (raw, pre-risk-control):
  score ≥ 0  →  lev = lev_neutral + score*(lev_bull_max - lev_neutral)
  score <  0  →  lev = lev_neutral + score*(lev_neutral - lev_bear_min)
  lev = clip(lev, lev_bear_min, lev_bull_max)

Risk controls (applied on top of raw leverage):
  1. Vol targeting:  lev = 0.5*raw + 0.5*(vol_target/√252 / daily_vol)
  2. DD de-lever:    if drawdown < dd_delever → scale by clip(1 + dd_severity*0.5, 0.3, 1)
  3. Daily stop-loss: if last_daily_ret < max_daily_loss → lev *= 0.5

eToro cap: continuous leverage is quantized to the nearest lower discrete
level supported by eToro (1, 2, 5, 10) and capped at MAX_STOCK_LEVERAGE.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Scenario parameters (exact match to notebook MaxLev_Bear25)
# ---------------------------------------------------------------------------

SCENARIO = dict(
    name          = "MaxLev_Bear25",
    long_base     = 0.85,
    bull_long     = 1.00,
    bear_long     = 0.25,
    lev_bull_max  = 10.0,
    lev_neutral   =  5.0,
    lev_bear_min  =  0.5,
    vol_target    =  0.20,
    dd_delever    = -0.15,
    max_daily_loss= -0.05,
    top_n         = 10,
    bottom_n      = 10,
)

VOL_LOOKBACK = 20   # trading days used to estimate realised daily vol

# eToro discrete leverage levels available for stocks
ETORO_LEVERAGE_LEVELS = [1, 2, 5, 10]
MAX_STOCK_LEVERAGE    = 10   # regulatory cap for your jurisdiction; adjust if needed


# ---------------------------------------------------------------------------
# Core functions (mirror notebook scenario engine, no pandas required)
# ---------------------------------------------------------------------------

def rescale_score(raw_score: float, score_min: float, score_max: float) -> float:
    """Rescale raw regime score to [-1, +1] using observed P2/P98 range."""
    if np.isnan(raw_score):
        return 0.0
    mid        = (score_min + score_max) / 2.0
    half_range = (score_max - score_min) / 2.0 + 1e-10
    return float(np.clip((raw_score - mid) / half_range, -1.0, 1.0))


def get_allocation(score_rescaled: float) -> tuple[float, float]:
    """Return (long_weight, short_weight) for given rescaled score.

    Returns
    -------
    (long_w, short_w)  both in [0, 1], summing to 1.
    """
    p = SCENARIO
    if score_rescaled >= 0:
        long_w = p["long_base"] + score_rescaled * (p["bull_long"] - p["long_base"])
    else:
        long_w = p["long_base"] + score_rescaled * (p["long_base"] - p["bear_long"])
    long_w = float(np.clip(long_w, p["bear_long"], p["bull_long"]))
    return long_w, 1.0 - long_w


def get_raw_leverage(score_rescaled: float) -> float:
    """Return continuous raw leverage for given rescaled score (pre-risk-control)."""
    p = SCENARIO
    if score_rescaled >= 0:
        lev = p["lev_neutral"] + score_rescaled * (p["lev_bull_max"] - p["lev_neutral"])
    else:
        lev = p["lev_neutral"] + score_rescaled * (p["lev_neutral"] - p["lev_bear_min"])
    return float(np.clip(lev, p["lev_bear_min"], p["lev_bull_max"]))


def apply_risk_controls(
    raw_lev:       float,
    running_rets:  list[float],
    cum_value:     float,
    cum_peak:      float,
) -> float:
    """Apply vol-targeting + drawdown de-leveraging + daily stop-loss.

    Parameters
    ----------
    raw_lev:
        Leverage from get_raw_leverage().
    running_rets:
        Recent net daily return history (used for realised vol and last ret).
    cum_value:
        Current portfolio value (normalised; starts at 1.0).
    cum_peak:
        Peak portfolio value so far.

    Returns
    -------
    Actual leverage to use, clamped to [lev_bear_min/2, lev_bull_max].
    """
    p = SCENARIO
    lev = raw_lev

    # ── 1. Vol targeting ──────────────────────────────────────────────────
    if len(running_rets) >= VOL_LOOKBACK:
        daily_vol = float(np.std(running_rets[-VOL_LOOKBACK:]))
        daily_vol_target = p["vol_target"] / np.sqrt(252)
        if daily_vol > 1e-6:
            vol_lev = daily_vol_target / daily_vol
            lev = 0.5 * lev + 0.5 * vol_lev

    # ── 2. Drawdown de-leveraging ─────────────────────────────────────────
    dd = (cum_value - cum_peak) / cum_peak if cum_peak > 0 else 0.0
    if dd < p["dd_delever"]:
        dd_severity = (dd - p["dd_delever"]) / (abs(p["dd_delever"]) + 1e-10)
        reduction   = float(np.clip(1.0 + dd_severity * 0.5, 0.3, 1.0))
        lev *= reduction

    # ── 3. Daily stop-loss ────────────────────────────────────────────────
    if running_rets and running_rets[-1] < p["max_daily_loss"]:
        lev *= 0.5

    return float(np.clip(lev, p["lev_bear_min"] * 0.5, p["lev_bull_max"]))


def quantize_leverage(continuous_lev: float) -> int:
    """Map continuous leverage → nearest lower discrete eToro level.

    Examples
    --------
    3.7 → 2,  6.0 → 5,  11.0 → 10,  0.4 → 1
    """
    capped = max(1.0, min(continuous_lev, float(MAX_STOCK_LEVERAGE)))
    chosen = 1
    for lvl in ETORO_LEVERAGE_LEVELS:
        if lvl <= capped:
            chosen = lvl
        else:
            break
    return chosen


def compute_position_sizes(
    capital:          float,
    actual_lev:       float,
    long_weight:      float,
    short_weight:     float,
    n_active_tranches: int = 3,
) -> tuple[float, float]:
    """Return (usd_per_long_position, usd_per_short_position).

    Capital is divided equally across all active tranches (at most h=3 at
    a time).  Within a tranche the leveraged notional is split between the
    long and short legs proportionally to their weights.

    Parameters
    ----------
    capital:
        Total account equity (USD).
    actual_lev:
        Risk-controlled continuous leverage.
    long_weight, short_weight:
        Allocation fractions (sum to 1).
    n_active_tranches:
        Number of currently active tranches (default = horizon = 3).

    Returns
    -------
    (per_long_usd, per_short_usd)
        USD invested (margin) per individual position.
    """
    p = SCENARIO
    # Allocate 1/n_active_tranches of equity to this new tranche
    tranche_equity = capital / max(n_active_tranches, 1)
    levered_amount = tranche_equity * actual_lev

    long_usd  = levered_amount * long_weight  / p["top_n"]
    short_usd = levered_amount * short_weight / p["bottom_n"]
    return long_usd, short_usd
