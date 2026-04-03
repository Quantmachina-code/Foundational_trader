"""Forward-return target construction.

For each horizon h, the target for ticker i on date t is:

    raw_return[i, t] = close[t+h] / close[t] - 1

Then cross-sectional quantile rank per date:

    target[i, t] = rank(raw_return[:, t], pct=True)

This produces a uniform [0, 1] target that focuses the CatBoost model on
*relative* stock selection rather than absolute return magnitude.
"""

from __future__ import annotations

import pandas as pd


def build_targets(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Compute h-day forward-return cross-sectional rank targets.

    Parameters
    ----------
    panel:
        Multi-ticker daily panel with ``date``, ``ticker``, and ``close``
        columns (sorted by date, ticker).
    horizon:
        Number of trading days ahead to measure the return.

    Returns
    -------
    pd.DataFrame
        Same index as *panel* with one extra column ``target`` containing
        the cross-sectional rank (0-1) of the h-day forward return.
        Rows where the forward return cannot be computed (last h rows per
        ticker) will have ``NaN`` in the target column.
    """
    if "close" not in panel.columns:
        raise ValueError("panel must contain a 'close' column")

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Compute raw forward return per ticker using transform (pandas 3.0 compatible)
    panel["_fwd_ret"] = panel.groupby("ticker")["close"].transform(
        lambda s: s.shift(-horizon) / s - 1
    )

    # Cross-sectional rank per date using transform
    panel["target"] = panel.groupby("date")["_fwd_ret"].transform(
        lambda s: s.rank(pct=True)
    )
    panel = panel.drop(columns=["_fwd_ret"])

    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)
