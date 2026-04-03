"""Test 01 — Config module.

Verifies that all expected constants exist, have the right types, and satisfy
basic sanity constraints.  No external calls, no data files required.

Run:
    pytest tests/test_01_config.py -v
"""

import pytest
from pathlib import Path
import catboost_trader.config as cfg


def test_paths_are_path_objects():
    assert isinstance(cfg.ROOT_DIR, Path)
    assert isinstance(cfg.CACHE_DIR, Path)
    assert isinstance(cfg.COMBINED_PARQUET, Path)
    assert isinstance(cfg.RESULTS_DIR, Path)
    assert isinstance(cfg.MODEL_DIR, Path)


def test_date_strings_parseable():
    import pandas as pd
    for attr in ("TRAIN_START", "TRAIN_END", "SIM_START", "SIM_END", "DATA_FETCH_START"):
        val = getattr(cfg, attr)
        pd.Timestamp(val)   # raises if not parseable


def test_training_before_simulation():
    import pandas as pd
    assert pd.Timestamp(cfg.TRAIN_END) < pd.Timestamp(cfg.SIM_START)


def test_catboost_params_keys():
    required = {"iterations", "learning_rate", "depth", "subsample",
                "colsample_bylevel", "min_data_in_leaf", "l2_leaf_reg",
                "loss_function", "random_seed", "verbose"}
    assert required <= set(cfg.CATBOOST_PARAMS.keys())


def test_catboost_params_values():
    p = cfg.CATBOOST_PARAMS
    assert p["iterations"] > 0
    assert 0 < p["learning_rate"] < 1
    assert 1 <= p["depth"] <= 16
    assert p["loss_function"] == "RMSE"


def test_horizons():
    assert isinstance(cfg.HORIZONS, list)
    assert all(isinstance(h, int) and h > 0 for h in cfg.HORIZONS)


def test_risk_params():
    assert 0 < cfg.TRAILING_STOP_PCT < 1, "Stop must be between 0 and 100%"
    assert cfg.MAX_LEVERAGE >= 1.0
    assert 0 < cfg.VOL_TARGET < 1


def test_top_n_options():
    assert isinstance(cfg.TOP_N_OPTIONS, list)
    assert len(cfg.TOP_N_OPTIONS) >= 1
    assert all(n > 0 for n in cfg.TOP_N_OPTIONS)


def test_regime_percentiles():
    assert cfg.REGIME_PERCENTILE_BEAR < cfg.REGIME_PERCENTILE_BULL
    assert 0 < cfg.REGIME_PERCENTILE_BEAR < 100
    assert 0 < cfg.REGIME_PERCENTILE_BULL < 100


def test_feature_set_constants():
    assert cfg.FEATURE_SET_ORIGINAL == "original"
    assert cfg.FEATURE_SET_ALPHA == "alpha"
    assert cfg.FEATURE_SET_ORIGINAL in cfg.FEATURE_SETS
    assert cfg.FEATURE_SET_ALPHA in cfg.FEATURE_SETS
