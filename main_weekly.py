"""
Weekly Multi-Asset Foundational Model Classification Pipeline
============================================================
Assets: S&P500 stocks + SPY, Gold (GLD/GC=F), Silver (SLV/SI=F),
        BTC-USD, ETH-USD, Oil (USO/CL=F)
Targets:
  1. Binary: next week open-to-open return > 0
  2. Binary: next week open-to-open return > S&P500 return (excess return)
  3. Quantile regression: 50 quantile deciles of all weekly data
Models: TabPFN, TabICL — loaded from local .ckpt files in the same directory
Train: first 6 years (prompt context) | Test/Backtest: last 4 years

LOCAL MODEL SETUP
─────────────────
Place checkpoint files in the same directory as this script:
  TabPFN  → any file matching  tabpfn*.ckpt  (e.g. tabpfn_v2.ckpt)
  TabICL  → any file matching  tabicl*.ckpt  (e.g. tabicl.ckpt)

TabPFN ckpt loading:
  TabPFNClassifier(model_path="./tabpfn.ckpt", ...)
  (tabpfn>=2.0 supports model_path; for v1 set TABPFN_MODEL_PATH env var)

TabICL ckpt loading:
  TabICLClassifier(ckpt_path="./tabicl.ckpt", ...)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys
import json
import glob
from pathlib import Path

# ─── Local .ckpt file discovery ───────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()

def _find_ckpt(pattern: str) -> str | None:
    """Return first matching .ckpt path relative to script directory, or None."""
    matches = sorted(SCRIPT_DIR.glob(pattern))
    if matches:
        return str(matches[0])
    return None

TABPFN_CKPT = _find_ckpt("tabpfn*.ckpt") or _find_ckpt("TabPFN*.ckpt")
TABICL_CKPT = _find_ckpt("tabicl*.ckpt") or _find_ckpt("TabICL*.ckpt")

print(f"TabPFN ckpt : {TABPFN_CKPT or '⚠ NOT FOUND (place tabpfn*.ckpt here)'}")
print(f"TabICL ckpt : {TABICL_CKPT or '⚠ NOT FOUND (place tabicl*.ckpt here)'}")

# ─── TabPFN loading ───────────────────────────────────────────────────────────
HAS_TABPFN = False
TabPFNClassifier = None

if TABPFN_CKPT:
    # Set env var so TabPFN v1/v2 both find the local weights
    os.environ["TABPFN_MODEL_PATH"] = TABPFN_CKPT
    try:
        from tabpfn import TabPFNClassifier as _TabPFN

        # Test whether this version accepts model_path kwarg (v2+)
        import inspect
        _pfn_sig = inspect.signature(_TabPFN.__init__).parameters
        if "model_path" in _pfn_sig:
            # v2 API — pass path explicitly
            def TabPFNClassifier(**kwargs):
                kwargs.setdefault("model_path", TABPFN_CKPT)
                kwargs.setdefault("device", "cpu")
                return _TabPFN(**kwargs)
        else:
            # v1 API — relies on env var; just wrap
            def TabPFNClassifier(**kwargs):
                kwargs.setdefault("device", "cpu")
                return _TabPFN(**kwargs)

        HAS_TABPFN = True
        print(f"✓ TabPFN loaded from: {TABPFN_CKPT}")
    except ImportError:
        print("✗ tabpfn package not installed — run: pip install tabpfn")
    except Exception as e:
        print(f"✗ TabPFN load error: {e}")
else:
    print("  Skipping TabPFN (no ckpt file found)")

# ─── TabICL loading ───────────────────────────────────────────────────────────
HAS_TABICL = False
TabICLClassifier = None

if TABICL_CKPT:
    try:
        # Try multiple import paths used across TabICL versions
        _tabicl_mod = None
        for _mod_path in ["tabicl", "tabicl.classifier", "tabicl.model"]:
            try:
                import importlib
                _m = importlib.import_module(_mod_path)
                if hasattr(_m, "TabICLClassifier"):
                    _tabicl_mod = _m
                    break
            except ImportError:
                continue

        if _tabicl_mod is None:
            raise ImportError("tabicl package not found in any expected location")

        _TabICL = _tabicl_mod.TabICLClassifier

        import inspect
        _icl_sig = inspect.signature(_TabICL.__init__).parameters

        if "ckpt_path" in _icl_sig:
            def TabICLClassifier(**kwargs):
                kwargs.setdefault("ckpt_path", TABICL_CKPT)
                return _TabICL(**kwargs)
        elif "model_path" in _icl_sig:
            def TabICLClassifier(**kwargs):
                kwargs.setdefault("model_path", TABICL_CKPT)
                return _TabICL(**kwargs)
        elif "checkpoint" in _icl_sig:
            def TabICLClassifier(**kwargs):
                kwargs.setdefault("checkpoint", TABICL_CKPT)
                return _TabICL(**kwargs)
        else:
            # Fallback: try passing as first positional arg
            def TabICLClassifier(**kwargs):
                return _TabICL(TABICL_CKPT, **kwargs)

        HAS_TABICL = True
        print(f"✓ TabICL loaded from: {TABICL_CKPT}")
    except ImportError:
        print("✗ tabicl package not installed — run: pip install tabicl")
    except Exception as e:
        print(f"✗ TabICL load error: {e}")
else:
    print("  Skipping TabICL (no ckpt file found)")

from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                              brier_score_loss, log_loss)
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ─── Config ──────────────────────────────────────────────────────────────────
START_DATE = "2014-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")
TRAIN_YEARS = 6
TEST_YEARS  = 4
N_QUANTILE_DECILES = 50
OUTPUT_DIR = Path("/mnt/user-data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Ticker assembly ─────────────────────────────────────────────────────────

EXTRA_TICKERS = [
    # Crypto
    "BTC-USD", "ETH-USD", "SOL-USD",
    # Commodity ETFs
    "GLD",   # Gold
    "SLV",   # Silver
    "CPER",  # Copper
    "PALL",  # Palladium
    "USO",   # WTI Oil
    # Commodity futures (continuous front-month)
    "GC=F",  # Gold futures
    "SI=F",  # Silver futures
    "HG=F",  # Copper futures
    "PA=F",  # Palladium futures
    "CL=F",  # WTI Oil futures
    # Market benchmark (must be present)
    "SPY",
]

BENCHMARK = "SPY"


def get_sp500_tickers() -> list:
    """
    Fetch current S&P 500 tickers from Wikipedia.
    Converts dots to hyphens for Yahoo Finance compatibility (e.g. BRK.B -> BRK-B).
    Falls back to a built-in ~100 ticker list if the request fails.
    """
    import urllib.request, html

    FALLBACK = [
        "AAPL","MSFT","GOOGL","AMZN","NVDA","META","BRK-B","LLY","AVGO","JPM",
        "V","UNH","XOM","TSLA","MA","PG","JNJ","COST","HD","MRK",
        "ABBV","CVX","CRM","BAC","NFLX","AMD","PEP","KO","TMO","ORCL",
        "WMT","ACN","ABT","MCD","LIN","TXN","CSCO","ADBE","DHR","NEE",
        "PM","INTC","QCOM","BMY","RTX","HON","UNP","AMGN","IBM","GS",
        "CAT","SPGI","BLK","ISRG","DE","SYK","MDLZ","PLD","CI","C",
        "ELV","T","GILD","ZTS","AXP","NOW","ADI","VRTX","REGN","MMC",
        "PGR","CB","BDX","AON","MO","ETN","PANW","AMAT","LRCX","SO",
        "DUK","GE","ITW","SBUX","CME","KLAC","SNPS","BSX","MCO","ICE",
        "WM","FIS","TJX","APH","ECL","ROP","NOC","CSX","EMR","CDNS",
        "PYPL","NXPI","MCHP","SWKS","QRVO","TER","MPWR","ON","STX","WDC",
        "HPQ","HPE","DELL","NTAP","FFIV","JNPR","ZBRA","CTSH","CDW","FSLR",
    ]

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    print("  Fetching S&P 500 tickers from Wikipedia...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")

        # Parse first wikitable — ticker symbols are in the first <td> of each row
        import re
        # Find the constituents table
        table_match = re.search(r'<table[^>]*wikitable[^>]*>(.*?)</table>', raw, re.DOTALL)
        if not table_match:
            raise ValueError("Table not found")
        table_html = table_match.group(1)

        tickers = []
        for row in re.findall(r'<tr>(.*?)</tr>', table_html, re.DOTALL):
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
            if cells:
                # First cell is the ticker symbol
                raw_ticker = re.sub(r'<[^>]+>', '', cells[0]).strip()
                raw_ticker = html.unescape(raw_ticker)
                # Yahoo Finance uses hyphens not dots
                ticker = raw_ticker.replace('.', '-').upper()
                if ticker and 1 <= len(ticker) <= 6:
                    tickers.append(ticker)

        if len(tickers) < 400:
            raise ValueError(f"Only parsed {len(tickers)} tickers, expected ~500")

        print(f"OK  ({len(tickers)} tickers)")
        return tickers

    except Exception as e:
        print(f"FAILED ({e}) — using fallback list ({len(FALLBACK)} tickers)")
        return FALLBACK

# ─── 1. DATA DOWNLOAD ────────────────────────────────────────────────────────

def download_weekly_data(tickers, start=START_DATE, end=END_DATE):
    """Download weekly OHLCV data for all tickers."""
    print(f"\n{'='*60}")
    print("DOWNLOADING WEEKLY DATA")
    print(f"{'='*60}")
    print(f"Tickers: {len(tickers)} | Period: {start} → {end}")

    all_data = {}
    failed = []

    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, start=start, end=end,
                             interval="1wk", progress=False, auto_adjust=True)
            if len(df) > 52:  # at least 1 year of weekly data
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df.index = pd.to_datetime(df.index)
                all_data[ticker] = df
                if (i+1) % 20 == 0:
                    print(f"  Downloaded {i+1}/{len(tickers)} tickers...")
        except Exception as e:
            failed.append(ticker)

    print(f"  ✓ Success: {len(all_data)} | ✗ Failed: {len(failed)}")
    if failed:
        print(f"  Failed tickers: {failed[:10]}{'...' if len(failed)>10 else ''}")
    return all_data

# ─── 2. TECHNICAL INDICATORS ─────────────────────────────────────────────────

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive technical indicators on OHLCV weekly data.
    Returns original df + feature columns.
    """
    feat = df.copy()
    # Normalise index — yfinance may return MultiIndex or tz-aware DatetimeIndex
    if isinstance(feat.index, pd.MultiIndex):
        feat = feat.reset_index(level=0, drop=True)
    feat.index = pd.to_datetime(feat.index).tz_localize(None)
    feat.columns = [c[0] if isinstance(c, tuple) else c for c in feat.columns]

    o, h, l, c, v = feat['Open'], feat['High'], feat['Low'], feat['Close'], feat['Volume']

    # ── Returns & price transforms ────────────────────────────────────────────
    for p in [1, 2, 3, 4, 8, 13, 26, 52]:
        feat[f'ret_{p}w']     = c.pct_change(p)
        feat[f'log_ret_{p}w'] = np.log(c / c.shift(p))

    feat['open_to_close']  = (c - o) / o
    feat['high_to_low_rng']= (h - l) / l
    feat['gap_open']       = (o - c.shift(1)) / c.shift(1)

    # ── Moving averages ────────────────────────────────────────────────────────
    for w in [4, 8, 13, 20, 26, 52]:
        feat[f'sma_{w}']     = c.rolling(w).mean()
        feat[f'ema_{w}']     = c.ewm(span=w, adjust=False).mean()
        feat[f'c_vs_sma{w}'] = c / feat[f'sma_{w}'] - 1
        feat[f'c_vs_ema{w}'] = c / feat[f'ema_{w}'] - 1

    # MA crossovers
    feat['ma_cross_4_13']  = feat['sma_4']  / feat['sma_13']  - 1
    feat['ma_cross_8_26']  = feat['sma_8']  / feat['sma_26']  - 1
    feat['ma_cross_13_52'] = feat['sma_13'] / feat['sma_52']  - 1
    feat['ema_cross_4_13'] = feat['ema_4']  / feat['ema_13']  - 1
    feat['ema_cross_8_26'] = feat['ema_8']  / feat['ema_26']  - 1

    # ── Volatility ────────────────────────────────────────────────────────────
    log_ret = np.log(c / c.shift(1))
    for w in [4, 8, 13, 26, 52]:
        feat[f'vol_{w}w']     = log_ret.rolling(w).std() * np.sqrt(52)
        feat[f'vol_ratio_{w}']= feat['vol_4w'] / feat[f'vol_{w}w']

    feat['atr_4']  = _atr(h, l, c, 4)
    feat['atr_13'] = _atr(h, l, c, 13)
    feat['atr_26'] = _atr(h, l, c, 26)
    feat['atr_pct']= feat['atr_13'] / c

    # Parkinson vol
    feat['park_vol'] = np.sqrt(1/(4*np.log(2)) * np.log(h/l)**2)

    # ── Momentum / Oscillators ────────────────────────────────────────────────
    # RSI
    for w in [7, 14, 21]:
        feat[f'rsi_{w}'] = _rsi(c, w)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    feat['macd']        = ema12 - ema26
    feat['macd_signal'] = feat['macd'].ewm(span=9, adjust=False).mean()
    feat['macd_hist']   = feat['macd'] - feat['macd_signal']
    feat['macd_norm']   = feat['macd'] / c

    # Stochastic
    for w in [5, 14]:
        low_min  = l.rolling(w).min()
        high_max = h.rolling(w).max()
        feat[f'stoch_k_{w}'] = 100 * (c - low_min) / (high_max - low_min + 1e-10)
        feat[f'stoch_d_{w}'] = feat[f'stoch_k_{w}'].rolling(3).mean()

    # Williams %R
    feat['willr_14'] = -100 * (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min() + 1e-10)

    # CCI
    tp = (h + l + c) / 3
    feat['cci_14'] = (tp - tp.rolling(14).mean()) / (0.015 * tp.rolling(14).std())
    feat['cci_26'] = (tp - tp.rolling(26).mean()) / (0.015 * tp.rolling(26).std())

    # ROC
    for p in [4, 8, 13, 26]:
        feat[f'roc_{p}'] = (c - c.shift(p)) / c.shift(p) * 100

    # PPO (% Price Oscillator)
    feat['ppo']        = (ema12 - ema26) / ema26 * 100
    feat['ppo_signal'] = feat['ppo'].ewm(span=9, adjust=False).mean()

    # TSI
    diff1 = c.diff(1)
    feat['tsi'] = (diff1.ewm(25).mean().ewm(13).mean() /
                   diff1.abs().ewm(25).mean().ewm(13).mean() * 100)

    # ── Trend indicators ─────────────────────────────────────────────────────
    # ADX
    feat['adx_14'] = _adx(h, l, c, 14)
    feat['adx_26'] = _adx(h, l, c, 26)

    # Aroon
    for w in [14, 25]:
        feat[f'aroon_up_{w}']   = h.rolling(w+1).apply(lambda x: (np.argmax(x)+1)/(w)*100, raw=True)
        feat[f'aroon_down_{w}'] = l.rolling(w+1).apply(lambda x: (np.argmin(x)+1)/(w)*100, raw=True)
        feat[f'aroon_osc_{w}']  = feat[f'aroon_up_{w}'] - feat[f'aroon_down_{w}']

    # Donchian channel position
    for w in [4, 13, 26]:
        feat[f'dc_pos_{w}'] = (c - l.rolling(w).min()) / (h.rolling(w).max() - l.rolling(w).min() + 1e-10)

    # ── Volume indicators ─────────────────────────────────────────────────────
    if v.sum() > 0:
        feat['vol_sma_4']    = v.rolling(4).mean()
        feat['vol_ratio_4']  = v / feat['vol_sma_4']
        feat['vol_sma_13']   = v.rolling(13).mean()
        feat['vol_ratio_13'] = v / feat['vol_sma_13']

        # OBV
        obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
        feat['obv']          = obv
        feat['obv_ema']      = obv.ewm(span=13).mean()
        feat['obv_signal']   = obv / obv.ewm(span=13).mean() - 1

        # Force Index
        feat['force_13'] = (c.diff(1) * v).ewm(span=13).mean()

        # MFI
        tp_vol = tp * v
        pos_flow = tp_vol.where(tp > tp.shift(1), 0)
        neg_flow = tp_vol.where(tp < tp.shift(1), 0)
        feat['mfi_14'] = 100 - 100/(1 + pos_flow.rolling(14).sum()/
                                       (neg_flow.rolling(14).sum()+1e-10))

        # VWAP proxy (rolling)
        feat['vwap_13'] = (tp * v).rolling(13).sum() / v.rolling(13).sum()
        feat['c_vs_vwap'] = c / feat['vwap_13'] - 1

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    for w, k in [(13, 2), (26, 2), (13, 1.5)]:
        sma = c.rolling(w).mean()
        std = c.rolling(w).std()
        feat[f'bb_upper_{w}_{k}'] = sma + k*std
        feat[f'bb_lower_{w}_{k}'] = sma - k*std
        feat[f'bb_pct_{w}_{k}']   = (c - feat[f'bb_lower_{w}_{k}']) / (feat[f'bb_upper_{w}_{k}'] - feat[f'bb_lower_{w}_{k}'] + 1e-10)
        feat[f'bb_width_{w}_{k}'] = (feat[f'bb_upper_{w}_{k}'] - feat[f'bb_lower_{w}_{k}']) / sma

    # ── Keltner Channel ───────────────────────────────────────────────────────
    kc_mid  = c.ewm(span=20).mean()
    kc_atr  = _atr(h, l, c, 10)
    feat['kc_pos'] = (c - (kc_mid - 1.5*kc_atr)) / (3*kc_atr + 1e-10)

    # ── Candlestick patterns (manual, no talib needed) ────────────────────────
    body     = (c - o).abs()
    rng      = (h - l).abs() + 1e-10
    up_shadow= h - c.where(c > o, o)
    dn_shadow= c.where(c < o, o) - l

    feat['candle_body_pct']    = body / rng
    feat['candle_up_shadow']   = up_shadow / rng
    feat['candle_dn_shadow']   = dn_shadow / rng
    feat['candle_is_bull']     = (c > o).astype(int)
    feat['candle_doji']        = (body / rng < 0.1).astype(int)
    feat['candle_hammer']      = ((dn_shadow > 2*body) & (up_shadow < 0.1*rng)).astype(int)
    feat['candle_shooting_star']= ((up_shadow > 2*body) & (dn_shadow < 0.1*rng)).astype(int)
    feat['candle_marubozu']    = (body / rng > 0.9).astype(int)
    feat['candle_spinning_top']= ((body/rng < 0.3) & (up_shadow/rng > 0.3) & (dn_shadow/rng > 0.3)).astype(int)

    # Engulfing patterns
    prev_body = (c.shift(1) - o.shift(1)).abs()
    feat['candle_bull_engulf'] = ((c > o) & (c.shift(1) < o.shift(1)) &
                                  (o < c.shift(1)) & (c > o.shift(1))).astype(int)
    feat['candle_bear_engulf'] = ((c < o) & (c.shift(1) > o.shift(1)) &
                                  (o > c.shift(1)) & (c < o.shift(1))).astype(int)

    # Inside/outside bar
    feat['inside_bar']  = ((h < h.shift(1)) & (l > l.shift(1))).astype(int)
    feat['outside_bar'] = ((h > h.shift(1)) & (l < l.shift(1))).astype(int)

    # 3-candle patterns
    feat['morning_star'] = (
        (c.shift(2) < o.shift(2)) &
        (body.shift(1) / rng.shift(1) < 0.2) &
        (c > o) &
        (c > (o.shift(2) + c.shift(2)) / 2)
    ).astype(int)

    feat['evening_star'] = (
        (c.shift(2) > o.shift(2)) &
        (body.shift(1) / rng.shift(1) < 0.2) &
        (c < o) &
        (c < (o.shift(2) + c.shift(2)) / 2)
    ).astype(int)

    # ── Pivot points ─────────────────────────────────────────────────────────
    pivot = (h.shift(1) + l.shift(1) + c.shift(1)) / 3
    feat['pivot']      = pivot
    feat['r1']         = 2*pivot - l.shift(1)
    feat['s1']         = 2*pivot - h.shift(1)
    feat['c_vs_pivot'] = (c - pivot) / (feat['atr_13'] + 1e-10)

    # ── High/Low relative to recent range ─────────────────────────────────────
    for w in [4, 8, 13, 26, 52]:
        feat[f'pct_from_high_{w}'] = c / h.rolling(w).max() - 1
        feat[f'pct_from_low_{w}']  = c / l.rolling(w).min() - 1
        feat[f'range_pct_{w}']     = (h.rolling(w).max() - l.rolling(w).min()) / l.rolling(w).min()

    # ── Lags of key features ─────────────────────────────────────────────────
    lag_cols = ['ret_1w', 'ret_2w', 'ret_4w', 'rsi_14', 'macd_norm',
                'open_to_close', 'vol_13w', 'bb_pct_13_2', 'adx_14']
    for col in lag_cols:
        if col in feat.columns:
            for lag in [1, 2, 3, 4, 8]:
                feat[f'{col}_lag{lag}'] = feat[col].shift(lag)

    # ── Price ratios ──────────────────────────────────────────────────────────
    feat['hl_ratio']  = h / (l + 1e-10)
    feat['co_ratio']  = c / (o + 1e-10)
    feat['hc_ratio']  = (h - c) / (h - l + 1e-10)
    feat['lc_ratio']  = (c - l) / (h - l + 1e-10)

    # Skewness, kurtosis of returns
    for w in [8, 13, 26]:
        feat[f'ret_skew_{w}'] = log_ret.rolling(w).skew()
        feat[f'ret_kurt_{w}'] = log_ret.rolling(w).kurt()

    return feat

def _rsi(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)

def _atr(high, low, close, period=14):
    tr = pd.concat([high-low,
                    (high - close.shift(1)).abs(),
                    (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def _adx(high, low, close, period=14):
    tr   = pd.concat([high-low,
                      (high - close.shift(1)).abs(),
                      (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    atr  = tr.ewm(alpha=1/period, adjust=False).mean()
    up   = high - high.shift(1)
    down = low.shift(1) - low
    dm_p = up.where((up > down) & (up > 0), 0)
    dm_n = down.where((down > up) & (down > 0), 0)
    di_p = 100 * dm_p.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10)
    di_n = 100 * dm_n.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10)
    dx   = 100 * (di_p - di_n).abs() / (di_p + di_n + 1e-10)
    return dx.ewm(alpha=1/period, adjust=False).mean()

# ─── 3. TARGETS ──────────────────────────────────────────────────────────────

def add_targets(df: pd.DataFrame, benchmark_open: pd.Series = None) -> pd.DataFrame:
    """
    Add three target variables:
      target_1: next week open-to-open return > 0  (binary)
      target_2: next week open-to-open return > S&P500 return (binary)
      fwd_open_to_open: raw forward return (quantile targets computed globally)
    """
    # Normalise index to plain DatetimeIndex (yfinance sometimes gives MultiIndex)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)

    fwd_ret = df['Open'].pct_change(1).shift(-1)   # next week open-to-open
    df['fwd_open_to_open'] = fwd_ret

    # Target 1: next week return > 0
    df['target_1'] = (fwd_ret > 0).astype(int)

    # Target 2: outperform S&P500 next week
    if benchmark_open is not None:
        bench = pd.to_datetime(benchmark_open.index).tz_localize(None)
        sp_series = pd.Series(benchmark_open.values, index=bench)
        sp_fwd = sp_series.pct_change(1).shift(-1)
        sp_fwd = sp_fwd.reindex(df.index)
        df['target_2'] = (fwd_ret > sp_fwd).astype(int)
    else:
        df['target_2'] = np.nan

    return df

def compute_global_quantile_targets(all_feat: pd.DataFrame,
                                    n_deciles: int = 50) -> pd.DataFrame:
    """Assign quantile decile labels based on global distribution of fwd returns."""
    valid = all_feat['fwd_open_to_open'].dropna()
    quantile_bins = pd.qcut(valid, q=n_deciles, labels=False, duplicates='drop')
    all_feat['target_quantile'] = quantile_bins
    return all_feat

# ─── 4. CROSS-ASSET FEATURES ─────────────────────────────────────────────────

def add_cross_asset_features(ticker_df: pd.DataFrame,
                              ref_dfs: dict) -> pd.DataFrame:
    """Add returns of reference assets as features (aligned by date)."""
    df = ticker_df.copy()
    # Ensure tz-naive DatetimeIndex
    df.index = pd.to_datetime(df.index).tz_localize(None)

    for name, ref in ref_dfs.items():
        try:
            ref_idx = pd.to_datetime(ref.index).tz_localize(None)
            if 'ret_1w' in ref.columns:
                col = pd.Series(ref['ret_1w'].values, index=ref_idx, name=f'xret_{name}')
            else:
                closes = pd.Series(ref['Close'].values, index=ref_idx)
                col = closes.pct_change(1).rename(f'xret_{name}')
            df = df.join(col, how='left')
            df[f'xret_{name}'] = df[f'xret_{name}'].ffill()
        except Exception:
            df[f'xret_{name}'] = np.nan
    return df

# ─── 5. DATASET ASSEMBLY ─────────────────────────────────────────────────────

def build_panel_dataset(all_feat: dict,
                        spy_open: pd.Series,
                        macro_refs: dict,
                        n_deciles: int = 50) -> pd.DataFrame:
    """
    Assemble panel dataset from all tickers.
    Returns long-format DataFrame with columns: date, ticker, features..., targets...
    """
    print(f"\n{'='*60}")
    print("BUILDING PANEL DATASET")
    print(f"{'='*60}")

    panels = []
    for ticker, df in all_feat.items():
        df = add_targets(df, spy_open)
        df = add_cross_asset_features(df, macro_refs)
        df['ticker'] = ticker
        # Capture the DatetimeIndex as an explicit column BEFORE concat
        df = df.copy()
        df['date'] = df.index
        panels.append(df)

    # ignore_index=True avoids duplicate index issues across tickers
    panel = pd.concat(panels, ignore_index=True)
    panel['date'] = pd.to_datetime(panel['date'])

    # Drop any stray index columns that yfinance/pandas may have injected
    for _col in ['Date', 'Datetime', 'Price', 'index', 'level_0']:
        if _col in panel.columns:
            panel = panel.drop(columns=[_col])

    # Global quantile targets
    panel = compute_global_quantile_targets(panel, n_deciles)

    # Drop rows missing targets or Open
    panel = panel.dropna(subset=['target_1', 'fwd_open_to_open'])

    print(f"  Panel shape: {panel.shape}")
    print(f"  Tickers: {panel['ticker'].nunique()}")
    print(f"  Date range: {panel['date'].min()} → {panel['date'].max()}")
    return panel

# ─── 6. TRAIN/TEST SPLIT ─────────────────────────────────────────────────────

def temporal_split(panel: pd.DataFrame,
                   train_years: int = 6,
                   test_years: int = 4):
    """Time-based split: first N years train, last M years test."""
    panel = panel.sort_values('date')
    min_date = panel['date'].min()

    train_end = min_date + pd.DateOffset(years=train_years)
    test_start = train_end
    test_end   = test_start + pd.DateOffset(years=test_years)

    train = panel[panel['date'] < train_end]
    test  = panel[(panel['date'] >= test_start) & (panel['date'] <= test_end)]

    print(f"\n  Train: {train['date'].min().date()} → {train['date'].max().date()} | n={len(train):,}")
    print(f"  Test:  {test['date'].min().date()}  → {test['date'].max().date()} | n={len(test):,}")
    return train, test

def get_feature_cols(panel: pd.DataFrame):
    """Return feature columns (exclude metadata and target cols)."""
    exclude = {'date', 'ticker', 'week', 'Open', 'High', 'Low', 'Close', 'Volume',
               'Adj Close', 'target_1', 'target_2', 'target_quantile',
               'fwd_open_to_open'}
    return [c for c in panel.columns if c not in exclude and panel[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

# ─── 7. MODEL TRAINING & EVALUATION ─────────────────────────────────────────

def prepare_arrays(train, test, feature_cols, target_col):
    """Impute, clip, and return X_train, y_train, X_test, y_test."""
    imp = SimpleImputer(strategy='median')
    X_train = imp.fit_transform(train[feature_cols])
    X_test  = imp.transform(test[feature_cols])

    # Clip extreme values
    X_train = np.clip(X_train, -100, 100)
    X_test  = np.clip(X_test,  -100, 100)

    y_train = train[target_col].values.astype(int)
    y_test  = test[target_col].values.astype(int)

    return X_train, y_train, X_test, y_test

def evaluate_classifier(model, X_test, y_test, label=""):
    """Evaluate binary classifier, return metrics dict."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    metrics = {
        'accuracy':  accuracy_score(y_test, preds),
        'brier':     brier_score_loss(y_test, proba[:,1]),
        'log_loss':  log_loss(y_test, proba),
    }
    try:
        metrics['roc_auc'] = roc_auc_score(y_test, proba[:,1])
    except:
        metrics['roc_auc'] = np.nan

    print(f"\n  [{label}]")
    for k, v in metrics.items():
        print(f"    {k:12s}: {v:.4f}")
    print(f"\n{classification_report(y_test, preds, digits=3)}")
    return metrics, preds, proba

def _select_features(X_tr, y_tr, X_te, k=100):
    """SelectKBest wrapper — returns (X_tr_sel, X_te_sel, selector)."""
    from sklearn.feature_selection import SelectKBest, f_classif
    sel = SelectKBest(f_classif, k=min(k, X_tr.shape[1]))
    X_tr_sel = sel.fit_transform(X_tr, y_tr)
    X_te_sel = sel.transform(X_te)
    return X_tr_sel, X_te_sel, sel


def run_tabpfn(X_train, y_train, X_test, y_test,
               max_train=3000, max_features=100, label="TabPFN"):
    """
    Run TabPFN from local ckpt.
    - Subsamples training set to max_train (TabPFN context limit).
    - Reduces features to max_features via SelectKBest if needed.
    """
    if not HAS_TABPFN:
        print(f"\n  ⚠ TabPFN not available (no ckpt or package missing), skipping.")
        return None, None, None

    print(f"\n  Running {label}  [ckpt: {Path(TABPFN_CKPT).name}]")

    # Feature selection
    X_tr, X_te = X_train, X_test
    if X_tr.shape[1] > max_features:
        X_tr, X_te, _ = _select_features(X_tr, y_train, X_te, k=max_features)
        print(f"    Features reduced: {X_train.shape[1]} → {X_tr.shape[1]}")

    # Subsample
    y_tr = y_train
    if len(X_tr) > max_train:
        idx  = np.random.choice(len(X_tr), max_train, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
        print(f"    Train subsampled: {len(X_train)} → {max_train}")

    try:
        clf = TabPFNClassifier(N_ensemble_configurations=16)
        clf.fit(X_tr, y_tr)
        return evaluate_classifier(clf, X_te, y_test, label)
    except Exception as e:
        print(f"    TabPFN error: {e}")
        return None, None, None


def run_tabicl(X_train, y_train, X_test, y_test, label="TabICL"):
    """
    Run TabICL from local ckpt.
    TabICL handles large feature counts natively; no hard subsample limit.
    """
    if not HAS_TABICL:
        print(f"\n  ⚠ TabICL not available (no ckpt or package missing), skipping.")
        return None, None, None

    print(f"\n  Running {label}  [ckpt: {Path(TABICL_CKPT).name}]")
    try:
        clf = TabICLClassifier()
        clf.fit(X_train, y_train)
        return evaluate_classifier(clf, X_test, y_test, label)
    except Exception as e:
        print(f"    TabICL error: {e}")
        return None, None, None

# ─── 8. BACKTESTING ──────────────────────────────────────────────────────────

def rolling_backtest(panel: pd.DataFrame,
                     feature_cols: list,
                     target_col: str,
                     model_fn,
                     train_years: int = 6,
                     step_weeks: int = 4,
                     window: str = 'expanding'):
    """
    Walk-forward backtest.
    window: 'expanding' (all past) or 'rolling' (fixed window).
    Returns DataFrame with predictions per week.
    """
    print(f"\n  Walk-forward backtest | step={step_weeks}w | window={window}")
    panel = panel.sort_values('date').copy()
    dates = sorted(panel['date'].unique())

    min_date  = dates[0]
    start_idx = sum(1 for d in dates if d < min_date + pd.DateOffset(years=train_years))

    all_preds = []

    for i in range(start_idx, len(dates), step_weeks):
        test_dates  = dates[i:i+step_weeks]

        if window == 'expanding':
            train_dates = dates[:i]
        else:
            lookback = train_years * 52
            train_dates = dates[max(0, i-lookback):i]

        tr = panel[panel['date'].isin(train_dates)].dropna(subset=[target_col])
        te = panel[panel['date'].isin(test_dates)].dropna(subset=[target_col])

        if len(tr) < 50 or len(te) == 0:
            continue

        try:
            X_tr, y_tr, X_te, y_te = prepare_arrays(tr, te, feature_cols, target_col)
            result = model_fn(X_tr, y_tr)
            if result is None or result[0] is None:
                continue
            model, transform_fn = result
            # Apply feature transform if factory returned one (e.g. feature selector)
            X_te_final = transform_fn(X_te) if transform_fn is not None else X_te
            proba = model.predict_proba(X_te_final)[:,1]
            preds = (proba > 0.5).astype(int)

            for j, (_, row) in enumerate(te.iterrows()):
                if j < len(preds):
                    all_preds.append({
                        'date': row['date'], 'ticker': row['ticker'],
                        'y_true': y_te[j], 'y_pred': preds[j],
                        'proba': proba[j],
                        'fwd_ret': row['fwd_open_to_open']
                    })
        except:
            continue

    return pd.DataFrame(all_preds)

def compute_backtest_metrics(bt_df: pd.DataFrame, label="") -> dict:
    """Compute rolling backtest performance metrics."""
    if len(bt_df) == 0:
        return {}

    metrics = {
        'accuracy':    accuracy_score(bt_df['y_true'], bt_df['y_pred']),
        'roc_auc':     roc_auc_score(bt_df['y_true'], bt_df['proba']),
        'n_samples':   len(bt_df),
    }

    # Long-short strategy: go long when pred=1, short when pred=0
    bt_df = bt_df.copy()
    bt_df['strategy_ret'] = bt_df['fwd_ret'] * (2*bt_df['y_pred'] - 1)

    weekly_strat = bt_df.groupby('date')['strategy_ret'].mean()
    metrics['strategy_ann_ret']    = weekly_strat.mean() * 52
    metrics['strategy_ann_sharpe'] = (weekly_strat.mean() / (weekly_strat.std()+1e-10)) * np.sqrt(52)
    metrics['strategy_max_dd']     = _max_drawdown(weekly_strat)

    print(f"\n  Backtest [{label}]")
    for k, v in metrics.items():
        print(f"    {k:25s}: {v:.4f}" if isinstance(v, float) else f"    {k:25s}: {v}")
    return metrics

def _max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

# ─── 9. VISUALIZATION ────────────────────────────────────────────────────────

def plot_results(panel, backtest_dfs, all_metrics, feature_cols):
    """Generate comprehensive results visualizations."""
    print("\n  Generating plots...")

    fig = plt.figure(figsize=(24, 32))
    fig.suptitle('Weekly Multi-Asset Foundational Model Pipeline\n(TabPFN / TabICL)',
                 fontsize=16, y=0.98)
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.45, wspace=0.3)

    # 1. Asset price history
    ax1 = fig.add_subplot(gs[0, :])
    key_assets = ['SPY','BTC-USD','ETH-USD','GLD','SLV','USO']
    for asset in key_assets:
        sub = panel[panel['ticker']==asset].set_index('date')['Close']
        if len(sub) > 0:
            norm = sub / sub.iloc[0] * 100
            ax1.plot(norm.index, norm.values, label=asset, linewidth=1.2)
    ax1.axvline(panel['date'].min() + pd.DateOffset(years=6),
                color='red', linestyle='--', linewidth=1.5, label='Train/Test Split')
    ax1.set_title('Normalized Price History (base=100)')
    ax1.legend(fontsize=8, ncol=7)
    ax1.set_ylabel('Normalized Price')

    # 2. Target distribution
    ax2 = fig.add_subplot(gs[1, 0])
    panel['target_1'].value_counts().plot.bar(ax=ax2, color=['#e74c3c','#2ecc71'])
    ax2.set_title('Target 1: Next Week Return > 0')
    ax2.set_xticklabels(['Negative','Positive'], rotation=0)

    ax3 = fig.add_subplot(gs[1, 1])
    panel['target_2'].dropna().astype(int).value_counts().plot.bar(ax=ax3, color=['#e74c3c','#2ecc71'])
    ax3.set_title('Target 2: Beat S&P500')
    ax3.set_xticklabels(['Under','Beats'], rotation=0)

    ax4 = fig.add_subplot(gs[1, 2])
    panel['fwd_open_to_open'].clip(-0.3, 0.3).hist(ax=ax4, bins=60, color='steelblue', edgecolor='white')
    ax4.set_title('Forward Return Distribution')
    ax4.set_xlabel('Weekly Open-to-Open Return')

    # 3. Quantile target distribution
    ax5 = fig.add_subplot(gs[2, :2])
    panel['target_quantile'].dropna().hist(ax=ax5, bins=50, color='mediumpurple', edgecolor='white')
    ax5.set_title(f'Quantile Target Distribution ({N_QUANTILE_DECILES} deciles)')
    ax5.set_xlabel('Quantile Decile')

    # 4. Feature correlation heatmap (top features)
    ax6 = fig.add_subplot(gs[2, 2])
    top_feat = feature_cols[:20] if len(feature_cols) >= 20 else feature_cols
    corr = panel[top_feat].corr()
    sns.heatmap(corr, ax=ax6, cmap='RdBu_r', center=0,
                xticklabels=False, yticklabels=False, cbar_kws={'shrink':0.8})
    ax6.set_title('Feature Correlation (top 20)')

    # 5. Backtest results
    row = 3
    for name, bt in backtest_dfs.items():
        if bt is None or len(bt) == 0:
            continue
        ax = fig.add_subplot(gs[row, :])
        bt_agg = bt.groupby('date').agg(
            acc=('y_true', lambda x: accuracy_score(x, bt.loc[x.index, 'y_pred'])),
            n=('y_true', 'count')
        ).reset_index()
        ax.plot(bt_agg['date'], bt_agg['acc'].rolling(4).mean(), label='Rolling Accuracy (4w MA)')
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Baseline (50%)')
        ax.fill_between(bt_agg['date'], 0.5, bt_agg['acc'].rolling(4).mean(), alpha=0.2)
        ax.set_title(f'Walk-Forward Backtest Accuracy: {name}')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.set_ylim(0.3, 0.75)
        row += 1
        if row >= 5:
            break

    plt.savefig(OUTPUT_DIR / 'pipeline_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved pipeline_results.png")

# ─── 10. MAIN PIPELINE ───────────────────────────────────────────────────────

def main():
    print("="*60)
    print("WEEKLY MULTI-ASSET FOUNDATIONAL ML PIPELINE")
    print("="*60)
    print(f"Period: {START_DATE} → {END_DATE}")
    print(f"Train: {TRAIN_YEARS} years | Backtest: {TEST_YEARS} years")
    print(f"TabPFN available: {HAS_TABPFN} | TabICL available: {HAS_TABICL}")

    # ── Build ticker universe ─────────────────────────────────────────────────
    sp500 = get_sp500_tickers()
    # Deduplicate, preserving order; extras go at the end
    seen = set(sp500)
    extras = [t for t in EXTRA_TICKERS if t not in seen]
    all_tickers = sp500 + extras
    print(f"  Universe: {len(sp500)} S&P500 + {len(extras)} extra = {len(all_tickers)} total")

    # ── Download ──────────────────────────────────────────────────────────────
    raw_data = download_weekly_data(all_tickers)

    if not raw_data:
        print("ERROR: No data downloaded. Check internet connection.")
        return

    # ── Feature engineering ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPUTING TECHNICAL INDICATORS")
    print(f"{'='*60}")

    all_feat = {}
    for i, (ticker, df) in enumerate(raw_data.items()):
        try:
            feat_df = compute_technical_indicators(df)
            all_feat[ticker] = feat_df
            if (i+1) % 20 == 0:
                print(f"  Processed {i+1}/{len(raw_data)} tickers | features: {feat_df.shape[1]}")
        except Exception as e:
            print(f"  Warning: {ticker} failed: {e}")

    print(f"  ✓ Features computed for {len(all_feat)} tickers")
    if all_feat:
        sample_ticker = next(iter(all_feat))
        n_feat = all_feat[sample_ticker].shape[1]
        print(f"  Feature columns per ticker: ~{n_feat}")

    # ── Reference assets for cross-asset features ─────────────────────────────
    macro_refs = {}
    for ref_name in ['SPY', 'GLD', 'BTC-USD', 'USO']:
        if ref_name in all_feat:
            macro_refs[ref_name.replace('-','_').replace('=','')] = all_feat[ref_name]

    spy_open = all_feat['SPY']['Open'] if 'SPY' in all_feat else None

    # ── Build panel ───────────────────────────────────────────────────────────
    panel = build_panel_dataset(all_feat, spy_open, macro_refs, N_QUANTILE_DECILES)

    # Save panel (sample for inspection)
    panel_sample = panel.sample(min(10000, len(panel))).sort_values('date')
    panel_sample.to_csv(OUTPUT_DIR / 'panel_sample.csv', index=False)
    print(f"\n  ✓ Panel sample saved (10k rows)")

    # ── Train/test split ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TRAIN / TEST SPLIT")
    print(f"{'='*60}")
    train, test = temporal_split(panel, TRAIN_YEARS, TEST_YEARS)
    feature_cols = get_feature_cols(panel)
    print(f"  Feature columns: {len(feature_cols)}")

    # ── Model training ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("MODEL TRAINING & EVALUATION")
    print(f"{'='*60}")

    all_metrics    = {}
    backtest_dfs   = {}
    results_summary = []

    targets = [
        ('target_1', 'Next Week Return > 0'),
        ('target_2', 'Beat S&P500 Return'),
    ]

    for target_col, target_name in targets:
        print(f"\n{'─'*50}")
        print(f"TARGET: {target_name}")
        print(f"{'─'*50}")

        # Filter valid samples
        tr = train.dropna(subset=[target_col])
        te = test.dropna(subset=[target_col])

        if len(tr) < 100:
            print(f"  Insufficient training data: {len(tr)}")
            continue

        X_train, y_train, X_test, y_test = prepare_arrays(tr, te, feature_cols, target_col)
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Class balance (train): {np.bincount(y_train)/len(y_train)}")

        key = f"{target_col}"

        # TabPFN
        pfn_metrics, pfn_preds, pfn_proba = run_tabpfn(
            X_train, y_train, X_test, y_test,
            label=f"TabPFN | {target_name}")
        if pfn_metrics:
            all_metrics[f'{key}_tabpfn'] = pfn_metrics
            results_summary.append({
                'target': target_name, 'model': 'TabPFN', **pfn_metrics})

        # TabICL
        icl_metrics, icl_preds, icl_proba = run_tabicl(
            X_train, y_train, X_test, y_test,
            label=f"TabICL | {target_name}")
        if icl_metrics:
            all_metrics[f'{key}_tabicl'] = icl_metrics
            results_summary.append({
                'target': target_name, 'model': 'TabICL', **icl_metrics})

    # ── Walk-forward backtest ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("WALK-FORWARD BACKTESTING")
    print(f"{'='*60}")

    if HAS_TABPFN:
        # Simplified backtest on a ticker subset for speed
        bt_panel = panel[panel['ticker'].isin(['SPY','AAPL','MSFT','BTC-USD','GLD'])].copy()

        def _pfn_factory(X_tr, y_tr):
            """Factory for walk-forward: feature-select then fit TabPFN."""
            X_tr_s, _, sel = _select_features(X_tr, y_tr, X_tr, k=100)
            n = min(1000, len(X_tr_s))
            clf = TabPFNClassifier(N_ensemble_configurations=8)
            clf.fit(X_tr_s[:n], y_tr[:n])
            # Return (clf, transform_fn) so rolling_backtest can apply transform
            return clf, sel.transform

        try:
            bt_df = rolling_backtest(bt_panel, feature_cols, 'target_1',
                                     _pfn_factory,
                                     train_years=TRAIN_YEARS, step_weeks=4)
            if len(bt_df) > 0:
                bt_metrics = compute_backtest_metrics(bt_df, "TabPFN | Target 1")
                backtest_dfs['TabPFN_target1'] = bt_df
                bt_df.to_csv(OUTPUT_DIR / 'backtest_tabpfn.csv', index=False)
        except Exception as e:
            print(f"  Backtest error: {e}")
            backtest_dfs['TabPFN_target1'] = pd.DataFrame()

    # ── Quantile regression via classification ────────────────────────────────
    print(f"\n{'='*60}")
    print("QUANTILE CLASSIFICATION (50 DECILES)")
    print(f"{'='*60}")

    panel_q = panel.dropna(subset=['target_quantile'])
    tr_q = panel_q[panel_q['date'] < panel_q['date'].min() + pd.DateOffset(years=TRAIN_YEARS)]
    te_q = panel_q[panel_q['date'] >= panel_q['date'].min() + pd.DateOffset(years=TRAIN_YEARS)]

    if len(tr_q) > 100 and len(te_q) > 0:
        X_tr_q = SimpleImputer(strategy='median').fit_transform(tr_q[feature_cols].fillna(0))
        X_te_q = SimpleImputer(strategy='median').fit_transform(te_q[feature_cols].fillna(0))
        y_tr_q = tr_q['target_quantile'].fillna(0).astype(int).values
        y_te_q = te_q['target_quantile'].fillna(0).astype(int).values
        X_tr_q = np.clip(X_tr_q, -100, 100)
        X_te_q = np.clip(X_te_q, -100, 100)

        print(f"  Quantile classification: {len(np.unique(y_tr_q))} classes")

        if HAS_TABPFN and len(np.unique(y_tr_q)) <= 10:
            # TabPFN supports limited classes — use coarser 5-bin quantiles
            y_coarse_tr = (y_tr_q // 10)
            y_coarse_te = (y_te_q // 10)

            X_tr_q100, X_te_q100, _ = _select_features(X_tr_q[:2000], y_coarse_tr[:2000], X_te_q, k=100)

            try:
                pfn_q = TabPFNClassifier(N_ensemble_configurations=8)
                pfn_q.fit(X_tr_q100, y_coarse_tr[:2000])
                preds_q  = pfn_q.predict(X_te_q100)
                acc_q    = accuracy_score(y_coarse_te, preds_q)
                print(f"\n  [TabPFN | Quantile 5-bin] Accuracy: {acc_q:.4f}")
                results_summary.append({
                    'target': 'Quantile (5 bins)', 'model': 'TabPFN',
                    'accuracy': acc_q, 'roc_auc': np.nan,
                    'brier': np.nan, 'log_loss': np.nan
                })
            except Exception as e:
                print(f"  TabPFN quantile error: {e}")

    # ── Save results ──────────────────────────────────────────────────────────
    if results_summary:
        results_df = pd.DataFrame(results_summary)
        results_df.to_csv(OUTPUT_DIR / 'model_results.csv', index=False)
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        print(results_df.to_string(index=False))

    # Save metrics JSON
    with open(OUTPUT_DIR / 'all_metrics.json', 'w') as f:
        json.dump({k: {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                       for kk, vv in v.items()}
                   for k, v in all_metrics.items()}, f, indent=2)

    # ── Visualization ──────────────────────────────────────────────────────────
    try:
        plot_results(panel, backtest_dfs, all_metrics, feature_cols)
    except Exception as e:
        print(f"  Plotting error: {e}")

    # ── Feature importance (correlation-based) ────────────────────────────────
    print(f"\n{'='*60}")
    print("TOP PREDICTIVE FEATURES (correlation with target_1)")
    print(f"{'='*60}")

    corr_with_target = panel[feature_cols + ['target_1']].corr()['target_1'].drop('target_1')
    top_features = corr_with_target.abs().nlargest(20)
    print(top_features.to_string())
    top_features.to_csv(OUTPUT_DIR / 'top_features.csv')

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Output files in: {OUTPUT_DIR}")
    print("  - pipeline_results.png  (visualization)")
    print("  - panel_sample.csv      (panel data sample)")
    print("  - model_results.csv     (accuracy, AUC, etc.)")
    print("  - all_metrics.json      (detailed metrics)")
    print("  - top_features.csv      (feature importance)")
    if HAS_TABPFN:
        print("  - backtest_tabpfn.csv   (walk-forward backtest)")


if __name__ == "__main__":
    np.random.seed(42)
    main()