"""Live regime computation for the eToro EOD bot.

Extends the historical ``mkt_full`` DataFrame (exported from the notebook)
with today's new data point and computes the updated regime score using the
same 7-signal, 252-day rolling-percentile algorithm from notebook Cell 9.

The key constraint is strict causality: the regime for date T uses only
signal history strictly before T.

Usage
-----
    from catboost_trader.trading.live_regime import RegimeTracker

    tracker = RegimeTracker.from_export("etoro_model_export/")
    regime, score_raw, score_rescaled = tracker.update(df_today_panel)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Signal weights matching notebook Cell 9 SIGNAL_CONFIG
_SIGNAL_WEIGHTS = {
    "mom_20d":    0.15,
    "mom_10d":    0.10,
    "mom_5d":     0.10,
    "breadth":    0.25,
    "vol":        0.15,   # inverted: high vol = bearish
    "rsi":        0.10,   # inverted: high RSI = overbought
    "dispersion": 0.15,
}


class RegimeTracker:
    """Maintains a live extension of the notebook's mkt_full DataFrame.

    Parameters
    ----------
    mkt_full_path:
        Path to ``regime_history.parquet`` exported by the notebook.
    meta_path:
        Path to ``meta.json`` exported by the notebook.
    """

    def __init__(
        self,
        mkt_full:  pd.DataFrame,
        score_min: float,
        score_max: float,
        lookback:  int = 252,
        bull_pct:  int = 67,
        bear_pct:  int = 33,
        min_thresh_hist: int = 60,
    ) -> None:
        self._mkt        = mkt_full.copy()
        self._score_min  = score_min
        self._score_max  = score_max
        self._lookback   = lookback
        self._bull_pct   = bull_pct
        self._bear_pct   = bear_pct
        self._min_thresh = min_thresh_hist

        # Ensure the index is a sorted DatetimeIndex
        self._mkt.index = pd.to_datetime(self._mkt.index)
        self._mkt = self._mkt.sort_index()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_export(cls, export_dir: str | Path) -> "RegimeTracker":
        """Load from the directory produced by export_notebook_models.py."""
        d    = Path(export_dir)
        hist = pd.read_parquet(d / "regime_history.parquet")
        meta = json.loads((d / "meta.json").read_text())

        return cls(
            mkt_full  = hist,
            score_min = float(meta["score_min"]),
            score_max = float(meta["score_max"]),
            lookback  = meta["regime"]["lookback"],
            bull_pct  = meta["regime"]["bull_pct"],
            bear_pct  = meta["regime"]["bear_pct"],
            min_thresh_hist = meta["regime"]["min_thresh_hist"],
        )

    # ------------------------------------------------------------------
    # Main: update with today's cross-sectional data
    # ------------------------------------------------------------------

    def update(
        self,
        df_today: pd.DataFrame,
        date:     pd.Timestamp | None = None,
    ) -> tuple[str, float, float]:
        """Compute today's regime from the latest panel slice.

        Parameters
        ----------
        df_today:
            Cross-sectional panel for ONE date (all tickers).
            Must contain columns: d_log_ret_1, (optionally d_rsi_14).
        date:
            Trading date.  Inferred from df_today["date"] if not provided.

        Returns
        -------
        (regime, raw_score, rescaled_score)
          regime: "BULL", "NEUTRAL", or "BEAR"
          raw_score: composite score before rescaling (range ~ [-1, +1])
          rescaled_score: score mapped to [-1, +1] via P2/P98 of history
        """
        if date is None:
            date = pd.Timestamp(df_today["date"].max())
        date = pd.Timestamp(date).normalize()

        # Build today's market signals (same as notebook Cell 9)
        today_signals = self._compute_day_signals(df_today, date)

        # Append to history and re-score (causal: uses only past rows)
        if date not in self._mkt.index:
            new_row = pd.DataFrame([today_signals], index=[date])
            self._mkt = pd.concat([self._mkt, new_row]).sort_index()

        raw_score   = self._composite_score(date)
        regime      = self._classify(date, raw_score)
        rescaled    = self._rescale(raw_score)

        logger.info(
            "Regime  date=%s  score_raw=%.4f  score_rs=%.4f  regime=%s",
            date.date(), raw_score, rescaled, regime,
        )
        return regime, raw_score, rescaled

    # ------------------------------------------------------------------
    # Private: signal computation for a single day
    # ------------------------------------------------------------------

    def _compute_day_signals(
        self,
        df_day: pd.DataFrame,
        date:   pd.Timestamp,
    ) -> dict:
        """Compute raw market signals for date from cross-section df_day."""
        log_ret_col = "d_log_ret_1"

        if log_ret_col not in df_day.columns:
            raise ValueError(f"'{log_ret_col}' column required in df_today")

        mkt_ret = float(df_day[log_ret_col].mean())
        breadth = float((df_day[log_ret_col] > 0).mean())
        dispersion = float(df_day[log_ret_col].std())
        avg_rsi = (
            float(df_day["d_rsi_14"].mean())
            if "d_rsi_14" in df_day.columns else 50.0
        )

        return {
            "mkt_ret":    mkt_ret,
            "breadth":    breadth,
            "dispersion": dispersion,
            "avg_rsi":    avg_rsi,
        }

    # ------------------------------------------------------------------
    # Private: composite score & classification
    # ------------------------------------------------------------------

    def _rolling_pct_rank(self, col: str, date: pd.Timestamp) -> float:
        """Percentile rank of col[date] vs prior LOOKBACK values."""
        hist_idx = self._mkt.index[self._mkt.index < date]
        n = len(hist_idx)
        if n < 30:
            return np.nan
        start_idx = max(0, n - self._lookback)
        hist_vals = self._mkt[col].iloc[start_idx:n].dropna().values
        if len(hist_vals) < 30:
            return np.nan
        current = float(self._mkt.loc[date, col])
        if np.isnan(current):
            return np.nan
        return float(np.mean(hist_vals <= current))

    def _composite_score(self, date: pd.Timestamp) -> float:
        """Compute weighted composite regime score for *date*."""
        # Derive rolling signals
        mkt = self._mkt

        # Rolling returns (approximate — sum of mkt_ret over window)
        hist_before = mkt[mkt.index < date]["mkt_ret"].dropna()
        hist_before_plus = mkt[mkt.index <= date]["mkt_ret"].dropna()

        def _roll_ret(n: int) -> float:
            return float(hist_before_plus.tail(n).sum()) if len(hist_before_plus) >= n else np.nan

        signals_raw = {
            "mkt_ret_20d": _roll_ret(20),
            "mkt_ret_10d": _roll_ret(10),
            "mkt_ret_5d":  _roll_ret(5),
            "breadth_20d": (
                float(mkt.loc[mkt.index <= date, "breadth"].tail(20).mean())
                if "breadth" in mkt.columns else np.nan
            ),
            "vol_regime": self._vol_regime(date),
            "avg_rsi":    float(mkt.loc[date, "avg_rsi"]) if "avg_rsi" in mkt.columns else 50.0,
            "dispersion_20d": (
                float(mkt.loc[mkt.index <= date, "dispersion"].tail(20).mean())
                if "dispersion" in mkt.columns else np.nan
            ),
        }

        # Percentile-rank each signal vs history (causal)
        signal_keys = {
            "mom_20d":    "mkt_ret_20d",
            "mom_10d":    "mkt_ret_10d",
            "mom_5d":     "mkt_ret_5d",
            "breadth":    "breadth_20d",
            "vol":        "vol_regime",
            "rsi":        "avg_rsi",
            "dispersion": "dispersion_20d",
        }
        inverted = {"vol", "rsi"}   # high vol / high RSI = bearish → invert

        composite = 0.0
        weight_sum = 0.0

        for sig_name, raw_key in signal_keys.items():
            raw_val = signals_raw.get(raw_key, np.nan)
            if np.isnan(raw_val):
                continue

            # Rank against stored history for this derived signal
            hist_series = self._build_rolling_history(raw_key, date)
            if len(hist_series) < 30:
                continue
            pct_rank = float(np.mean(hist_series <= raw_val))
            scaled = pct_rank * 2 - 1   # → [-1, +1]
            if sig_name in inverted:
                scaled = -scaled

            w = _SIGNAL_WEIGHTS[sig_name]
            composite  += scaled * w
            weight_sum += w

        if weight_sum < 0.3:
            return np.nan
        return composite / weight_sum

    def _build_rolling_history(self, key: str, date: pd.Timestamp) -> np.ndarray:
        """Build a history array for a derived rolling signal."""
        mkt = self._mkt
        past = mkt[mkt.index < date]
        n = min(self._lookback, len(past))
        past = past.tail(n)

        if key == "mkt_ret_20d":
            return self._rolling_sum_history(past["mkt_ret"], 20)
        if key == "mkt_ret_10d":
            return self._rolling_sum_history(past["mkt_ret"], 10)
        if key == "mkt_ret_5d":
            return self._rolling_sum_history(past["mkt_ret"], 5)
        if key == "breadth_20d":
            return past["breadth"].rolling(20, min_periods=10).mean().dropna().values if "breadth" in past.columns else np.array([])
        if key == "vol_regime":
            return self._vol_regime_history(past)
        if key == "avg_rsi":
            return past["avg_rsi"].dropna().values if "avg_rsi" in past.columns else np.array([])
        if key == "dispersion_20d":
            return past["dispersion"].rolling(20, min_periods=10).mean().dropna().values if "dispersion" in past.columns else np.array([])
        return np.array([])

    @staticmethod
    def _rolling_sum_history(series: pd.Series, n: int) -> np.ndarray:
        return series.rolling(n, min_periods=max(n // 2, 3)).sum().dropna().values

    def _vol_regime(self, date: pd.Timestamp) -> float:
        """vol_20d / vol_60d ratio at date."""
        if "mkt_ret" not in self._mkt.columns:
            return np.nan
        past = self._mkt[self._mkt.index <= date]["mkt_ret"].dropna()
        vol20 = float(past.tail(20).std()) if len(past) >= 20 else np.nan
        vol60 = float(past.tail(60).std()) if len(past) >= 60 else np.nan
        if np.isnan(vol20) or np.isnan(vol60) or vol60 < 1e-10:
            return np.nan
        return vol20 / vol60

    def _vol_regime_history(self, past: pd.DataFrame) -> np.ndarray:
        if "mkt_ret" not in past.columns:
            return np.array([])
        ret = past["mkt_ret"].dropna()
        v20 = ret.rolling(20, min_periods=10).std()
        v60 = ret.rolling(60, min_periods=30).std()
        ratio = v20 / (v60 + 1e-10)
        return ratio.dropna().values

    def _classify(self, date: pd.Timestamp, raw_score: float) -> str:
        """BULL / NEUTRAL / BEAR using adaptive rolling thresholds."""
        if np.isnan(raw_score):
            return "NEUTRAL"

        # Score history strictly before today
        past_scores = []
        for d in self._mkt.index:
            if d >= date:
                break
            s = self._composite_score_from_history(d)
            if not np.isnan(s):
                past_scores.append(s)

        past_scores_arr = np.array(past_scores[-self._lookback:])
        if len(past_scores_arr) < self._min_thresh:
            return "NEUTRAL"

        bull_t = np.percentile(past_scores_arr, self._bull_pct)
        bear_t = np.percentile(past_scores_arr, self._bear_pct)

        if raw_score > bull_t:
            return "BULL"
        if raw_score < bear_t:
            return "BEAR"
        return "NEUTRAL"

    def _composite_score_from_history(self, date: pd.Timestamp) -> float:
        """Fast composite score from stored mkt_full columns if available."""
        if "regime_score_smooth" in self._mkt.columns:
            val = self._mkt.loc[date, "regime_score_smooth"] if date in self._mkt.index else np.nan
            return float(val) if not pd.isna(val) else np.nan
        if "regime_score" in self._mkt.columns:
            val = self._mkt.loc[date, "regime_score"] if date in self._mkt.index else np.nan
            return float(val) if not pd.isna(val) else np.nan
        return np.nan

    def _rescale(self, raw_score: float) -> float:
        """Rescale raw score to [-1, +1] using P2/P98 range from export."""
        if np.isnan(raw_score):
            return 0.0
        mid        = (self._score_min + self._score_max) / 2.0
        half_range = (self._score_max - self._score_min) / 2.0 + 1e-10
        return float(np.clip((raw_score - mid) / half_range, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Latest regime (no new data — use last known)
    # ------------------------------------------------------------------

    def latest_regime(self) -> tuple[str, float, float]:
        """Return regime for the most recently seen date."""
        last_date = self._mkt.index.max()
        raw_score = self._composite_score_from_history(last_date)
        if np.isnan(raw_score):
            raw_score = 0.0
        regime   = self._classify(last_date, raw_score)
        rescaled = self._rescale(raw_score)
        return regime, raw_score, rescaled
