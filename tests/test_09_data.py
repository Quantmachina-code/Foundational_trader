"""Test 09 — Data downloader (requires FMP API key).

These tests make REAL HTTP calls to the FMP API and write files to disk.
They are SKIPPED automatically when the FMP_API_KEY environment variable
is not set.  Run them explicitly once you have a key:

    FMP_API_KEY=your_key pytest tests/test_09_data.py -v -s

What is tested
--------------
* get_sp500_tickers()      → returns ≥ 400 tickers
* ensure_data()            → downloads a 3-ticker subset and returns a DataFrame
* load_panel()             → reloads the parquet from disk
* Panel schema validation  → required columns present, date range correct
"""

import os
import pytest
import pandas as pd

FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
SKIP_REASON = "FMP_API_KEY not set — skipping live data tests"

requires_fmp = pytest.mark.skipif(not FMP_API_KEY, reason=SKIP_REASON)


# ---------------------------------------------------------------------------
# get_sp500_tickers
# ---------------------------------------------------------------------------

@requires_fmp
def test_get_sp500_tickers_returns_list():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from download_sp500_fmp import get_sp500_tickers

    tickers = get_sp500_tickers(FMP_API_KEY)
    assert isinstance(tickers, list)
    assert len(tickers) >= 400, f"Expected ≥400 tickers, got {len(tickers)}"
    assert all(isinstance(t, str) for t in tickers)
    assert "AAPL" in tickers or "MSFT" in tickers   # sanity


@requires_fmp
def test_get_sp500_tickers_no_duplicates():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from download_sp500_fmp import get_sp500_tickers

    tickers = get_sp500_tickers(FMP_API_KEY)
    assert len(tickers) == len(set(tickers)), "Duplicate tickers returned"


# ---------------------------------------------------------------------------
# ensure_data (download a minimal slice)
# ---------------------------------------------------------------------------

@requires_fmp
def test_ensure_data_returns_dataframe(tmp_path, monkeypatch):
    """Download only AAPL and MSFT to keep the test fast."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))

    # Patch get_sp500_tickers to return a tiny universe
    import download_sp500_fmp as fmp_mod
    monkeypatch.setattr(fmp_mod, "get_sp500_tickers",
                        lambda api_key: ["AAPL", "MSFT"])

    # Redirect cache and output to tmp_path
    import catboost_trader.config as cfg
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cfg, "COMBINED_PARQUET", tmp_path / "panel.parquet")

    from catboost_trader.data.downloader import ensure_data
    panel = ensure_data(
        FMP_API_KEY,
        start="2023-01-01",
        end="2023-06-30",
        update=False,
    )
    assert isinstance(panel, pd.DataFrame)
    assert len(panel) > 0


@requires_fmp
def test_ensure_data_panel_schema(tmp_path, monkeypatch):
    """Verify required columns are present after download."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))

    import download_sp500_fmp as fmp_mod
    monkeypatch.setattr(fmp_mod, "get_sp500_tickers",
                        lambda api_key: ["AAPL"])

    import catboost_trader.config as cfg
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cfg, "COMBINED_PARQUET", tmp_path / "panel.parquet")

    from catboost_trader.data.downloader import ensure_data
    panel = ensure_data(FMP_API_KEY, start="2023-01-01", end="2023-03-31", update=False)

    required_cols = [
        "date", "ticker", "open", "high", "low", "close", "volume",
        "d_rsi_14", "d_macd", "d_log_ret_1", "d_vol_20",
    ]
    for col in required_cols:
        assert col in panel.columns, f"Missing column: {col}"


@requires_fmp
def test_ensure_data_date_range(tmp_path, monkeypatch):
    """Downloaded data must cover (at least partially) the requested range."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))

    import download_sp500_fmp as fmp_mod
    monkeypatch.setattr(fmp_mod, "get_sp500_tickers",
                        lambda api_key: ["AAPL"])

    import catboost_trader.config as cfg
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cfg, "COMBINED_PARQUET", tmp_path / "panel.parquet")

    from catboost_trader.data.downloader import ensure_data
    panel = ensure_data(FMP_API_KEY, start="2023-01-01", end="2023-06-30", update=False)

    panel["date"] = pd.to_datetime(panel["date"])
    assert panel["date"].min() <= pd.Timestamp("2023-02-01")
    assert panel["date"].max() >= pd.Timestamp("2023-05-01")


# ---------------------------------------------------------------------------
# load_panel
# ---------------------------------------------------------------------------

def test_load_panel_raises_without_file(tmp_path):
    """load_panel must raise FileNotFoundError if the parquet doesn't exist."""
    from catboost_trader.data.downloader import load_panel
    with pytest.raises(FileNotFoundError):
        load_panel(tmp_path / "nonexistent.parquet")


@requires_fmp
def test_load_panel_matches_ensure_data(tmp_path, monkeypatch):
    """Panel loaded from disk should equal the one returned by ensure_data."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))

    import download_sp500_fmp as fmp_mod
    monkeypatch.setattr(fmp_mod, "get_sp500_tickers",
                        lambda api_key: ["AAPL"])

    import catboost_trader.config as cfg
    combined = tmp_path / "panel.parquet"
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cfg, "COMBINED_PARQUET", combined)

    from catboost_trader.data.downloader import ensure_data, load_panel
    panel1 = ensure_data(FMP_API_KEY, start="2023-01-01", end="2023-03-31", update=False)
    panel2 = load_panel(combined)

    assert panel1.shape == panel2.shape
    pd.testing.assert_frame_equal(
        panel1.reset_index(drop=True),
        panel2.reset_index(drop=True),
        check_like=True,
    )
