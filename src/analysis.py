"""
analysis.py
-----------
All analysis functions + chart generation.
Saves every plot to outputs/plots/.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
PALETTE = {
    "Extreme Fear":  "#d62728",
    "Fear":          "#ff7f0e",
    "Neutral":       "#bcbd22",
    "Greed":         "#2ca02c",
    "Extreme Greed": "#17becf",
}
SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#ddd",
    "text.color":       "#ddd",
    "xtick.color":      "#aaa",
    "ytick.color":      "#aaa",
    "grid.color":       "#2a2a3e",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "monospace",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})

COLORS = list(PALETTE.values())


def _save(fig, name: str):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {name}")


# ══════════════════════════════════════════════════════════════════════════
# 1. PnL vs Sentiment
# ══════════════════════════════════════════════════════════════════════════
def plot_pnl_vs_sentiment(daily: pd.DataFrame):
    """Bar chart — average daily PnL per sentiment category."""
    agg = (
        daily.groupby("sentiment", observed=True)["total_pnl"]
        .agg(["mean", "sem"])
        .reindex(SENTIMENT_ORDER)
        .dropna()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        agg["sentiment"], agg["mean"],
        color=[PALETTE.get(s, "#888") for s in agg["sentiment"]],
        yerr=agg["sem"], capsize=4, error_kw={"color": "#aaa", "lw": 1.5},
        width=0.6, edgecolor="#000", linewidth=0.8
    )
    ax.axhline(0, color="#888", lw=1, linestyle="--")
    ax.set_title("Average Daily PnL by Market Sentiment", pad=14)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Avg PnL (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    for bar, val in zip(bars, agg["mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (abs(bar.get_height()) * 0.05),
                f"${val:,.1f}", ha="center", va="bottom", fontsize=9, color="#eee")

    fig.tight_layout()
    _save(fig, "01_pnl_vs_sentiment.png")
    return agg


# ══════════════════════════════════════════════════════════════════════════
# 2. Win Rate vs Sentiment
# ══════════════════════════════════════════════════════════════════════════
def plot_winrate_vs_sentiment(daily: pd.DataFrame):
    agg = (
        daily.groupby("sentiment", observed=True)["win_rate"]
        .mean()
        .reindex(SENTIMENT_ORDER)
        .dropna()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(agg["sentiment"], agg["win_rate"] * 100,
           color=[PALETTE.get(s, "#888") for s in agg["sentiment"]],
           edgecolor="#000", linewidth=0.8, width=0.6)
    ax.axhline(50, color="#ff9900", lw=1.2, linestyle="--", label="50 % line")
    ax.set_title("Average Win Rate by Market Sentiment")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Win Rate (%)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "02_winrate_vs_sentiment.png")


# ══════════════════════════════════════════════════════════════════════════
# 3. Leverage vs Sentiment (Box Plot)
# ══════════════════════════════════════════════════════════════════════════
def plot_leverage_distribution(daily: pd.DataFrame):
    data = [
        daily[daily["sentiment"] == s]["avg_leverage"].dropna().values
        for s in SENTIMENT_ORDER
        if s in daily["sentiment"].values
    ]
    labels = [s for s in SENTIMENT_ORDER if s in daily["sentiment"].values]

    fig, ax = plt.subplots(figsize=(11, 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops={"color": "#fff", "lw": 2},
                    whiskerprops={"color": "#888"},
                    capprops={"color": "#888"},
                    flierprops={"marker": "o", "markersize": 3,
                                "markerfacecolor": "#555", "linestyle": "none"})

    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(PALETTE.get(label, "#555"))
        patch.set_alpha(0.75)

    ax.set_title("Leverage Distribution by Sentiment")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Average Leverage (x)")
    ax.yaxis.grid(True)
    fig.tight_layout()
    _save(fig, "03_leverage_distribution.png")


# ══════════════════════════════════════════════════════════════════════════
# 4. Trade Frequency vs Sentiment
# ══════════════════════════════════════════════════════════════════════════
def plot_trade_frequency(daily: pd.DataFrame):
    agg = (
        daily.groupby("sentiment", observed=True)["trade_count"]
        .mean()
        .reindex(SENTIMENT_ORDER)
        .dropna()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(agg["sentiment"], agg["trade_count"],
           color=[PALETTE.get(s, "#888") for s in agg["sentiment"]],
           edgecolor="#000", linewidth=0.8, width=0.6)
    ax.set_title("Average Trade Frequency by Sentiment")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Avg Trades per Trader per Day")
    fig.tight_layout()
    _save(fig, "04_trade_frequency.png")


# ══════════════════════════════════════════════════════════════════════════
# 5. Cumulative PnL Over Time (Line Chart)
# ══════════════════════════════════════════════════════════════════════════
def plot_cumulative_pnl(daily: pd.DataFrame):
    ts = (
        daily.groupby("date")["total_pnl"]
        .sum()
        .sort_index()
        .reset_index()
    )
    ts["cum_pnl"] = ts["total_pnl"].cumsum()

    # Colour background by dominant sentiment
    sent_day = (
        daily.groupby("date")["sentiment"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
    )
    ts = ts.merge(sent_day, on="date", how="left")

    fig, ax = plt.subplots(figsize=(14, 6))

    prev_date = ts["date"].iloc[0]
    prev_sent = ts["sentiment"].iloc[0]
    for _, row in ts.iterrows():
        if row["sentiment"] != prev_sent or row.name == len(ts) - 1:
            ax.axvspan(prev_date, row["date"],
                       color=PALETTE.get(prev_sent, "#555"), alpha=0.12)
            prev_date = row["date"]
            prev_sent = row["sentiment"]

    ax.plot(ts["date"], ts["cum_pnl"], color="#00e5ff", lw=2, label="Cumulative PnL")
    ax.fill_between(ts["date"], ts["cum_pnl"], alpha=0.15, color="#00e5ff")
    ax.set_title("Cumulative PnL Over Time (shaded by Sentiment)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    fig.tight_layout()
    _save(fig, "05_cumulative_pnl_timeline.png")


# ══════════════════════════════════════════════════════════════════════════
# 6. Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════
def plot_correlation_heatmap(daily: pd.DataFrame):
    cols = ["total_pnl", "win_rate", "avg_leverage", "trade_count",
            "avg_size", "is_fear", "total_volume"]
    cols = [c for c in cols if c in daily.columns]

    corr = daily[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, linewidths=0.5,
                annot_kws={"size": 10},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    _save(fig, "06_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════
# 7. Trader Tier Performance
# ══════════════════════════════════════════════════════════════════════════
def plot_trader_tier_performance(daily: pd.DataFrame):
    if "trader_tier" not in daily.columns:
        return

    agg = (
        daily.groupby(["trader_tier", "sentiment"], observed=True)["total_pnl"]
        .mean()
        .reset_index()
    )

    tier_order = ["Top", "Mid", "Bottom"]
    tier_colors = {"Top": "#2ca02c", "Mid": "#ff7f0e", "Bottom": "#d62728"}

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(SENTIMENT_ORDER))
    width = 0.25

    for i, tier in enumerate(tier_order):
        subset = agg[agg["trader_tier"] == tier].set_index("sentiment")["total_pnl"]
        vals = [subset.get(s, 0) for s in SENTIMENT_ORDER]
        ax.bar(x + i * width, vals, width,
               label=f"{tier} Traders", color=tier_colors[tier],
               alpha=0.85, edgecolor="#000")

    ax.set_title("Trader Tier Performance by Sentiment")
    ax.set_xticks(x + width)
    ax.set_xticklabels(SENTIMENT_ORDER)
    ax.set_ylabel("Avg PnL (USD)")
    ax.axhline(0, color="#888", lw=1, linestyle="--")
    ax.legend()
    fig.tight_layout()
    _save(fig, "07_trader_tier_vs_sentiment.png")


# ══════════════════════════════════════════════════════════════════════════
# 8. PnL During Sentiment Transitions
# ══════════════════════════════════════════════════════════════════════════
def plot_sentiment_transitions(daily: pd.DataFrame):
    if "fear_to_greed" not in daily.columns:
        return

    groups = {
        "Fear → Greed": daily[daily["fear_to_greed"] == True]["total_pnl"],
        "Greed → Fear": daily[daily["greed_to_fear"] == True]["total_pnl"],
        "No Shift":     daily[(daily["fear_to_greed"] == False) &
                               (daily["greed_to_fear"] == False)]["total_pnl"],
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(groups.keys(), [g.mean() for g in groups.values()],
           color=["#2ca02c", "#d62728", "#888"],
           edgecolor="#000", width=0.5)
    ax.axhline(0, color="#aaa", lw=1, linestyle="--")
    ax.set_title("Avg PnL During Sentiment Transitions")
    ax.set_ylabel("Avg PnL (USD)")
    fig.tight_layout()
    _save(fig, "08_sentiment_transitions.png")


# ══════════════════════════════════════════════════════════════════════════
# 9. Over-Leverage Risk Analysis
# ══════════════════════════════════════════════════════════════════════════
def plot_overleveraged_risk(daily: pd.DataFrame):
    if "is_overleveraged" not in daily.columns:
        return

    agg = (
        daily.groupby(["sentiment", "is_overleveraged"], observed=True)["total_pnl"]
        .mean()
        .unstack("is_overleveraged")
        .reindex(SENTIMENT_ORDER)
        .dropna(how="all")
        .rename(columns={0: "Normal Leverage", 1: "Over-Leveraged"})
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    agg.plot(kind="bar", ax=ax, color=["#17becf", "#d62728"],
             edgecolor="#000", width=0.65)
    ax.axhline(0, color="#aaa", lw=1, linestyle="--")
    ax.set_title("PnL: Normal vs Over-Leveraged Traders by Sentiment")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Avg PnL (USD)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    _save(fig, "09_overleveraged_risk.png")


# ══════════════════════════════════════════════════════════════════════════
# Statistical Tests
# ══════════════════════════════════════════════════════════════════════════
def run_statistical_tests(daily: pd.DataFrame) -> dict:
    """Mann-Whitney U: Fear PnL vs Greed PnL."""
    fear_pnl  = daily[daily["is_fear"] == 1]["total_pnl"].dropna()
    greed_pnl = daily[daily["is_fear"] == 0]["total_pnl"].dropna()

    u_stat, p_val = stats.mannwhitneyu(fear_pnl, greed_pnl, alternative="two-sided")

    results = {
        "fear_mean_pnl":    fear_pnl.mean(),
        "greed_mean_pnl":   greed_pnl.mean(),
        "fear_win_rate":    daily[daily["is_fear"] == 1]["win_rate"].mean(),
        "greed_win_rate":   daily[daily["is_fear"] == 0]["win_rate"].mean(),
        "fear_avg_lev":     daily[daily["is_fear"] == 1]["avg_leverage"].mean(),
        "greed_avg_lev":    daily[daily["is_fear"] == 0]["avg_leverage"].mean(),
        "mwu_stat":         u_stat,
        "mwu_p_value":      p_val,
        "significant":      p_val < 0.05,
    }
    return results


# ══════════════════════════════════════════════════════════════════════════
# Master runner
# ══════════════════════════════════════════════════════════════════════════
def run_all_analysis(daily: pd.DataFrame) -> dict:
    print("\n📊 Generating plots …")
    pnl_agg = plot_pnl_vs_sentiment(daily)
    plot_winrate_vs_sentiment(daily)
    plot_leverage_distribution(daily)
    plot_trade_frequency(daily)
    plot_cumulative_pnl(daily)
    plot_correlation_heatmap(daily)
    plot_trader_tier_performance(daily)
    plot_sentiment_transitions(daily)
    plot_overleveraged_risk(daily)

    print("\n📐 Running statistical tests …")
    stats_results = run_statistical_tests(daily)

    print("\n── Statistical Results ──")
    for k, v in stats_results.items():
        print(f"  {k:<22}: {v}")

    return {"pnl_summary": pnl_agg, "stats": stats_results}