"""Visualisation utilities for simulation results.

All functions return a ``matplotlib.figure.Figure`` so the caller can either
show it interactively or save it to disk.  No ``plt.show()`` calls are made
inside this module — that is the caller's responsibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import matplotlib.figure


def _get_plt():
    """Lazy import of matplotlib to avoid hard dependency at import time."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plots.  Install with: pip install matplotlib"
        ) from exc


# ---------------------------------------------------------------------------
# Individual charts
# ---------------------------------------------------------------------------

def plot_equity_curve(
    equity: pd.Series,
    label: str = "",
    title: str = "Equity Curve",
) -> "matplotlib.figure.Figure":
    """Plot a single equity curve normalised to 100 at start."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(12, 5))
    norm = equity / equity.iloc[0] * 100
    ax.plot(norm.index, norm.values, label=label or "Strategy", linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("Indexed equity (start = 100)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_equity_comparison(
    results: dict[str, "SimulationResult"],  # noqa: F821
    title: str = "Strategy Comparison — Equity Curves",
) -> "matplotlib.figure.Figure":
    """Overlay normalised equity curves for multiple simulation results."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(14, 6))

    for label, res in results.items():
        eq = res.equity_curve
        if eq.empty:
            continue
        norm = eq / eq.iloc[0] * 100
        ax.plot(norm.index, norm.values, label=label, linewidth=1.2)

    ax.set_title(title)
    ax.set_ylabel("Indexed equity (start = 100)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_drawdown(
    result: "SimulationResult",  # noqa: F821
    title: str | None = None,
) -> "matplotlib.figure.Figure":
    """Plot the drawdown series over time."""
    from catboost_trader.analysis.metrics import drawdown_series

    plt = _get_plt()
    eq = result.equity_curve
    if eq.empty:
        fig, ax = plt.subplots()
        ax.set_title("No data")
        return fig

    dd = drawdown_series(eq) * 100  # as percentage
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color="red", label="Drawdown")
    ax.plot(dd.index, dd.values, color="red", linewidth=0.8)
    ax.set_title(title or f"Drawdown — {result.label}")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_regime_overlay(
    result: "SimulationResult",  # noqa: F821
    title: str | None = None,
) -> "matplotlib.figure.Figure":
    """Plot equity curve with BEAR / NEUTRAL / BULL regime shading."""
    plt = _get_plt()
    eq = result.equity_curve
    regime_df = result.regime_df

    if eq.empty:
        fig, ax = plt.subplots()
        ax.set_title("No data")
        return fig

    fig, ax = plt.subplots(figsize=(14, 6))
    norm = eq / eq.iloc[0] * 100
    ax.plot(norm.index, norm.values, color="steelblue", linewidth=1.5, label="Equity (indexed)")

    # Shade regime periods
    _REGIME_COLORS = {"BEAR": "salmon", "NEUTRAL": "lightyellow", "BULL": "lightgreen"}
    if not regime_df.empty and "regime" in regime_df.columns:
        prev_date = None
        prev_regime = None
        for date, row in regime_df.iterrows():
            regime = row["regime"]
            if regime != prev_regime and prev_regime is not None and prev_date is not None:
                color = _REGIME_COLORS.get(prev_regime, "white")
                ax.axvspan(prev_date, date, alpha=0.2, color=color, linewidth=0)
            if regime != prev_regime:
                prev_regime = regime
                prev_date = date

    ax.set_title(title or f"Equity + Regime — {result.label}")
    ax.set_ylabel("Indexed equity (start = 100)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)

    # Legend patches
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color="lightgreen", alpha=0.5, label="BULL"),
        mpatches.Patch(color="lightyellow", alpha=0.5, label="NEUTRAL"),
        mpatches.Patch(color="salmon", alpha=0.5, label="BEAR"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + patches, labels + ["BULL", "NEUTRAL", "BEAR"], fontsize=8)
    fig.tight_layout()
    return fig


def plot_top_n_comparison(
    results_by_n: dict[int, "SimulationResult"],  # noqa: F821
    feature_set: str = "",
    horizon: int = 0,
) -> "matplotlib.figure.Figure":
    """Compare equity curves across different top-N values."""
    plt = _get_plt()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax_eq, ax_bar = axes

    # Left: equity curves
    for n, res in sorted(results_by_n.items()):
        eq = res.equity_curve
        if eq.empty:
            continue
        norm = eq / eq.iloc[0] * 100
        ax_eq.plot(norm.index, norm.values, label=f"Top-{n}", linewidth=1.3)

    ax_eq.set_title(f"Top-N Comparison  |  h={horizon}  |  {feature_set}")
    ax_eq.set_ylabel("Indexed equity")
    ax_eq.grid(True, alpha=0.3)
    ax_eq.legend()

    # Right: bar chart of Sharpe ratios
    ns = sorted(results_by_n.keys())
    sharpes = [results_by_n[n].metrics.get("sharpe", 0) for n in ns]
    bars = ax_bar.bar([str(n) for n in ns], sharpes, color="steelblue", alpha=0.7)
    ax_bar.bar_label(bars, fmt="%.2f", padding=2)
    ax_bar.set_title("Sharpe Ratio by Top-N")
    ax_bar.set_xlabel("Top-N positions")
    ax_bar.set_ylabel("Sharpe Ratio")
    ax_bar.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


def plot_metrics_heatmap(
    summary_df: pd.DataFrame,
    metric: str = "sharpe",
) -> "matplotlib.figure.Figure":
    """Heatmap of a single metric across (horizon, feature_set) x top_n."""
    plt = _get_plt()
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
    except Exception:
        pass

    pivot = summary_df.pivot_table(
        values=metric,
        index=["horizon", "feature_set"],
        columns="top_n",
        aggfunc="first",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label=metric)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"Top-{c}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"h={h} {fs}" for h, fs in pivot.index])

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)

    ax.set_title(f"Metric: {metric}  —  all model combinations")
    fig.tight_layout()
    return fig


def save_all_plots(
    results: dict[str, "SimulationResult"],  # noqa: F821
    output_dir: str = "results/plots",
) -> None:
    """Save standard plots for every result to *output_dir*."""
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Combined comparison
    fig = plot_equity_comparison(results)
    fig.savefig(out / "equity_comparison.png", dpi=120, bbox_inches="tight")

    for label, res in results.items():
        safe_label = label.replace("/", "_").replace(" ", "_")

        fig = plot_drawdown(res)
        fig.savefig(out / f"drawdown_{safe_label}.png", dpi=100, bbox_inches="tight")

        fig = plot_regime_overlay(res)
        fig.savefig(out / f"regime_{safe_label}.png", dpi=100, bbox_inches="tight")

    _get_plt().close("all")
    print(f"[plots] Saved {2 * len(results) + 1} plots to {out}/")
