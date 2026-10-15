"""Figures in one consistent style, designed to stay readable at thumbnail size."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1b1f24"
MUTED = "#6b7280"
GRID = "#e5e7eb"
INDEX_COLOR = "#0b3c5d"
MASI_COLOR = "#9aa3ad"
ELEVATED_COLOR = "#f2a541"
STRESS_COLOR = "#d1495b"
COMPONENT_COLORS = {                      # colour-blind-safe (Okabe-Ito)
    "equity_momentum": "#0072b2",
    "equity_volatility": "#56b4e9",
    "bond_volatility": "#009e73",
    "market_breadth": "#e69f00",
    "news_sentiment": "#cc79a7",
    "oil_volatility": "#d55e00",
}
COMPONENT_LABELS = {
    "equity_momentum": "Equity momentum",
    "equity_volatility": "Equity volatility",
    "bond_volatility": "Bond volatility",
    "market_breadth": "Market breadth",
    "news_sentiment": "News sentiment",
    "oil_volatility": "Oil volatility",
}


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 13,
        "axes.titlesize": 17, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 13, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.frameon": False, "legend.fontsize": 12,
        "lines.linewidth": 1.8,
    })


def _year_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0.01)


def stress_periods(flag: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous runs of True in a date-indexed boolean series, as (start, end) pairs."""
    runs, start, prev = [], None, None
    for d, f in flag.items():
        if f and start is None:
            start = d
        if not f and start is not None:
            runs.append((start, prev))
            start = None
        prev = d
    if start is not None:
        runs.append((start, prev))
    return runs


def _shade(ax, periods, color, alpha) -> None:
    for a, b in periods:
        ax.axvspan(a, b + pd.Timedelta(days=1), color=color, alpha=alpha, lw=0)


def with_gaps(series: pd.Series, max_gap_days: int = 10) -> pd.Series:
    """Insert a missing value inside data holes so lines are not drawn across them."""
    s = series.dropna()
    gap = s.index.to_series().diff().dt.days > max_gap_days
    holes = pd.Series(np.nan, index=s.index[gap] - pd.Timedelta(days=1))
    return pd.concat([s, holes]).sort_index()


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- thumbnail
def plot_thumbnail(index: pd.Series, masi: pd.Series, stress_flag: pd.Series,
                   title: str, subtitle: str, path: Path, masi_label: str = "MASI") -> Path:
    """Index over time against the MASI, stress periods shaded. Landscape 16:9."""
    set_style()
    fig, ax = plt.subplots(figsize=(12, 6.75))
    _shade(ax, stress_periods(stress_flag), STRESS_COLOR, 0.18)
    line = with_gaps(index)
    ax.plot(line.index, line.values, color=INDEX_COLOR, lw=2.0, zorder=3)
    ax.set_ylabel("Stress index (0-100)", color=INDEX_COLOR)
    ax.set_ylim(0, max(100, float(index.max()) * 1.05))
    _year_axis(ax)

    ax2 = ax.twinx()
    m = with_gaps(masi.loc[index.index.min():index.index.max()])
    ax2.plot(m.index, m.values, color=MASI_COLOR, lw=1.6, zorder=2)
    ax2.set_ylabel(masi_label, color=MUTED)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    fig.suptitle(title, x=0.06, y=0.99, ha="left", fontsize=26, fontweight="bold", color=INK)
    fig.text(0.06, 0.915, subtitle, ha="left", fontsize=14, color=MUTED)
    handles = [plt.Line2D([], [], color=INDEX_COLOR, lw=2.5, label="Stress index"),
               plt.Line2D([], [], color=MASI_COLOR, lw=2.5, label="MASI equity index (right axis)"),
               plt.Rectangle((0, 0), 1, 1, color=STRESS_COLOR, alpha=0.3, label="Stress periods")]
    ax.legend(handles=handles, loc="upper left", ncol=3)
    fig.subplots_adjust(top=0.86)
    return _save(fig, path)


# ----------------------------------------------------------------------------- method diagram
def plot_method_diagram(n_articles: int, n_stocks: int, path: Path) -> Path:
    """Data sources -> sentiment scoring -> components -> composite index."""
    set_style()
    fig, ax = plt.subplots(figsize=(14, 6.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, head, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=color, ec="none", alpha=0.14))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="none", ec=color, lw=2))
        ax.text(x + w / 2, y + h - 0.32, head, ha="center", va="top", fontsize=14,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.85, body, ha="center", va="top", fontsize=11.5,
                color=INK, linespacing=1.45)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=22,
                                     lw=2, color=MUTED))

    ax.text(1.55, 5.95, "1  Data", ha="center", fontsize=13, color=MUTED, fontweight="bold")
    ax.text(5.15, 5.95, "2  Signals", ha="center", fontsize=13, color=MUTED, fontweight="bold")
    ax.text(8.75, 5.95, "3  Components", ha="center", fontsize=13, color=MUTED, fontweight="bold")
    ax.text(12.35, 5.95, "4  Composite", ha="center", fontsize=13, color=MUTED, fontweight="bold")

    box(0.1, 3.3, 2.9, 2.3, "Financial news",
        f"{n_articles:,} articles\nfrom Moroccan financial\nnews websites", COMPONENT_COLORS["news_sentiment"])
    box(0.1, 0.4, 2.9, 2.5, "Market data",
        f"MASI equity index\nMBI bond index\n{n_stocks} listed stocks\nBrent crude oil",
        COMPONENT_COLORS["equity_momentum"])

    box(3.7, 3.3, 2.9, 2.3, "Sentiment scoring",
        "French transformer model\nscores each article\nDaily average: 90-day\nwindow, 14-day half-life",
        COMPONENT_COLORS["news_sentiment"])
    box(3.7, 0.4, 2.9, 2.5, "Market signals",
        "Momentum vs 100-day MA\n20-day realised volatility\n(equity, bonds, oil)\nShare of stocks above MA",
        COMPONENT_COLORS["equity_momentum"])

    box(7.3, 0.4, 2.9, 5.2, "Six components",
        "News sentiment\nEquity momentum\nEquity volatility\nBond volatility\nMarket breadth\nOil volatility\n\n"
        "Each ranked against\nits own past year\n(0 = calm, 1 = stressed)", COMPONENT_COLORS["market_breadth"])

    box(10.9, 0.4, 2.9, 5.2, "Stress index",
        "CISS aggregation\n(European Central\nBank method)\n\nStress counts more\nwhen components\nmove together\n\n"
        "Daily, 0-100 scale\n\nRegimes: calm,\nelevated, stress", STRESS_COLOR)

    arrow(3.0, 4.45, 3.7, 4.45)
    arrow(3.0, 1.65, 3.7, 1.65)
    arrow(6.6, 4.45, 7.3, 3.6)
    arrow(6.6, 1.65, 7.3, 2.4)
    arrow(10.2, 3.0, 10.9, 3.0)
    return _save(fig, path)


# ----------------------------------------------------------------------------- components
def plot_components(contrib: pd.DataFrame, title: str, subtitle: str, path: Path) -> Path:
    """Monthly average contribution of each component, stacked.

    Contributions are additive (they sum to the index). Negative contributions,
    which occur when a component is negatively correlated with the others, are
    stacked below zero.
    """
    set_style()
    monthly = contrib.resample("MS").mean()
    x = monthly.index + pd.Timedelta(days=14)
    fig, ax = plt.subplots(figsize=(12, 6.2))
    pos_base = np.zeros(len(monthly))
    neg_base = np.zeros(len(monthly))
    for c in monthly.columns:
        v = monthly[c].to_numpy()
        pos, neg = np.clip(v, 0, None), np.clip(v, None, 0)
        ax.bar(x, pos, bottom=pos_base, width=27, color=COMPONENT_COLORS[c], label=COMPONENT_LABELS[c], lw=0)
        ax.bar(x, neg, bottom=neg_base, width=27, color=COMPONENT_COLORS[c], lw=0)
        pos_base += pos
        neg_base += neg
    ax.plot(x, monthly.sum(axis=1), color=INK, lw=1.4, label="Stress index (monthly mean)")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("Index points")
    _year_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=11.5)
    fig.suptitle(title, x=0.06, y=0.99, ha="left", fontsize=20, fontweight="bold", color=INK)
    fig.text(0.06, 0.915, subtitle, ha="left", fontsize=13, color=MUTED)
    fig.subplots_adjust(top=0.86)
    return _save(fig, path)


# ----------------------------------------------------------------------------- validation
def plot_validation(index: pd.Series, targets: pd.DataFrame, buckets: pd.DataFrame,
                    masi_drawdown: pd.Series, title: str, subtitle: str, path: Path) -> Path:
    """Left: index over time above the MASI drawdown. Right: forward volatility and
    forward worst fall by index quintile."""
    set_style()
    fig = plt.figure(figsize=(14, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.8, 1], height_ratios=[1.6, 1],
                          wspace=0.22, hspace=0.12)

    ax = fig.add_subplot(gs[0, 0])
    line = with_gaps(index)
    ax.plot(line.index, line.values, color=INDEX_COLOR, lw=1.8)
    ax.set_ylabel("Stress index")
    ax.set_ylim(0, 100)
    ax.set_title("Index (top) and MASI drawdown from its peak (bottom)", fontsize=14)
    ax.tick_params(labelbottom=False)

    bx = fig.add_subplot(gs[1, 0], sharex=ax)
    dd = with_gaps(masi_drawdown.loc[index.index.min():index.index.max()] * 100)
    bx.fill_between(dd.index, dd.values, 0, color=STRESS_COLOR, alpha=0.55, lw=0)
    bx.set_ylabel("Drawdown (%)")
    _year_axis(bx)

    cx = fig.add_subplot(gs[:, 1])
    x = np.arange(len(buckets))
    cx.bar(x - 0.2, buckets["fwd_realised_vol"] * 100, width=0.4, color=INDEX_COLOR,
           label="Volatility (%, annualised)")
    cx.bar(x + 0.2, -buckets["fwd_max_drawdown"] * 100, width=0.4, color=STRESS_COLOR,
           label="Worst fall (%)")
    cx.set_xticks(x, [f"{b}" for b in buckets.index])
    cx.set_xlabel("Index quintile (Q1 calm, Q5 stressed)")
    cx.set_title("MASI over the next 20 trading days", fontsize=14)
    cx.set_ylim(0, buckets["fwd_realised_vol"].max() * 100 * 1.3)   # headroom for the legend
    cx.legend(loc="upper left", ncol=1, fontsize=11.5)
    cx.grid(axis="x", visible=False)

    fig.suptitle(title, x=0.06, y=1.03, ha="left", fontsize=20, fontweight="bold", color=INK)
    fig.text(0.06, 0.955, subtitle, ha="left", fontsize=13, color=MUTED)
    fig.subplots_adjust(top=0.85)
    return _save(fig, path)
