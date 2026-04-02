"""Central configuration for the CatBoost trading simulation engine.

All parameters live here so that switching from paper to live trading,
or changing any modelling assumption, requires edits in one place only.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parents[1]
CACHE_DIR = ROOT_DIR / "data" / "fmp_ticker_cache"
DAILY_CACHE_DIR = CACHE_DIR / "daily"
COMBINED_PARQUET = ROOT_DIR / "data" / "sp500_panel.parquet"
RESULTS_DIR = ROOT_DIR / "results"
MODEL_DIR = ROOT_DIR / "models_saved"

# ---------------------------------------------------------------------------
# Date ranges
# ---------------------------------------------------------------------------

TRAIN_START = "2010-01-01"
TRAIN_END = "2017-12-31"
SIM_START = "2018-01-01"
SIM_END = "2026-03-31"
DATA_FETCH_START = "2009-01-01"  # extra warmup for rolling indicators

# ---------------------------------------------------------------------------
# CatBoost model parameters
# ---------------------------------------------------------------------------

CATBOOST_PARAMS: dict = dict(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    subsample=0.8,
    colsample_bylevel=0.8,
    min_data_in_leaf=20,
    l2_leaf_reg=3.0,
    loss_function="RMSE",
    random_seed=42,
    verbose=0,
    thread_count=-1,
)

HORIZONS: list[int] = [1, 3, 5]          # forward-return horizons in trading days
HALF_LIFE_DAYS: int = 252                 # exponential sample-weight half-life
MAX_FEATURES: int = 300                   # SelectKBest cap for feature selection

# ---------------------------------------------------------------------------
# Portfolio / strategy
# ---------------------------------------------------------------------------

TOP_N_OPTIONS: list[int] = [3, 5, 10, 15]   # top-N positions to sweep
INITIAL_CAPITAL: float = 100_000.0

# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------

TRAILING_STOP_PCT: float = 0.07   # 7 % trailing stop from position peak
MAX_LEVERAGE: float = 3.0         # maximum allowed gross leverage
VOL_TARGET: float = 0.15          # 15 % annualised vol target for leverage sizing
DRAWDOWN_DELEVER_THRESHOLD: float = 0.20   # start deleveraging when DD > 20 %

# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

REGIME_WINDOW: int = 500               # rolling window (calendar days) for adaptive thresholds
REGIME_PERCENTILE_BEAR: int = 33       # below P33 → BEAR (fully out)
REGIME_PERCENTILE_BULL: int = 67       # above P67 → BULL (max leverage)
BULL_BASE_LEVERAGE: float = 1.5        # minimum leverage in BULL (before vol-targeting)

# Market index ticker used for regime momentum signals
MARKET_TICKER: str = "SPY"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

FMP_REQUEST_DELAY: float = 0.25   # seconds between API calls (raise to 1.0 on free plan)

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

FEATURE_SET_ORIGINAL = "original"    # only d_* columns from compute_daily_indicators
FEATURE_SET_ALPHA = "alpha"          # d_* + af_* engineered features
FEATURE_SETS = [FEATURE_SET_ORIGINAL, FEATURE_SET_ALPHA]
