import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────
PALETTE = {
    "Extreme Fear": "#d62728",
    "Fear": "#ff7f0e",
    "Neutral": "#bcbd22",
    "Greed": "#2ca02c",
    "Extreme Greed": "#17becf",
}
SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {name}")


# ─────────────────────────────────────────────────────
# 1. PnL Distribution (Box Plot)
# ─────────────────────────────────────────────────────
def plot_box(daily):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=daily,
        x="sentiment",
        y="total_pnl",
        order=SENTIMENT_ORDER,
        palette=PALETTE,
        ax=ax
    )
    ax.set_title("PnL Distribution by Market Sentiment")
    ax.set_ylabel("Total PnL")
    _save(fig, "01_box_pnl.png")


# ─────────────────────────────────────────────────────
# 2. Leverage Distribution (IMPORTANT)
# ─────────────────────────────────────────────────────
def plot_leverage(daily):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=daily,
        x="sentiment",
        y="avg_leverage",
        order=SENTIMENT_ORDER,
        palette=PALETTE,
        ax=ax
    )
    ax.set_title("Leverage Usage by Market Sentiment")
    ax.set_ylabel("Average Leverage")
    _save(fig, "02_leverage.png")


# ─────────────────────────────────────────────────────
# 3. Win Rate Distribution
# ─────────────────────────────────────────────────────
def plot_winrate(daily):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=daily,
        x="sentiment",
        y="win_rate",
        order=SENTIMENT_ORDER,
        palette=PALETTE,
        ax=ax
    )
    ax.set_title("Win Rate by Market Sentiment")
    ax.set_ylabel("Win Rate")
    _save(fig, "03_winrate.png")


# ─────────────────────────────────────────────────────
# 4. Combined Plot (Box + Individual Points)
# ─────────────────────────────────────────────────────
def plot_combined(daily):
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=daily,
        x="sentiment",
        y="total_pnl",
        order=SENTIMENT_ORDER,
        palette=PALETTE,
        ax=ax
    )

    sns.stripplot(
        data=daily,
        x="sentiment",
        y="total_pnl",
        order=SENTIMENT_ORDER,
        color="white",
        alpha=0.3,
        jitter=True,
        ax=ax
    )

    ax.set_title("PnL Distribution with Individual Trades")
    _save(fig, "04_combined.png")


# ─────────────────────────────────────────────────────
# 5. Correlation Heatmap
# ─────────────────────────────────────────────────────
def plot_heatmap(daily):
    cols = ["total_pnl", "win_rate", "avg_leverage", "trade_count"]
    cols = [c for c in cols if c in daily.columns]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(daily[cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    _save(fig, "05_heatmap.png")


# ─────────────────────────────────────────────────────
# Statistical Test
# ─────────────────────────────────────────────────────
def run_stats(daily):
    fear = daily[daily["is_fear"] == 1]["total_pnl"]
    greed = daily[daily["is_fear"] == 0]["total_pnl"]

    if len(fear) == 0 or len(greed) == 0:
        return {"error": "Not enough data for statistical test"}

    u, p = stats.mannwhitneyu(fear, greed)

    return {
        "fear_mean": fear.mean(),
        "greed_mean": greed.mean(),
        "p_value": p,
        "significant": p < 0.05
    }


# ─────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────
def run_all_analysis(daily):
    print("\n📊 Generating 5 key graphs...")

    plot_box(daily)
    plot_leverage(daily)
    plot_winrate(daily)
    plot_combined(daily)
    plot_heatmap(daily)

    print("\n📐 Running statistical test...")
    stats_res = run_stats(daily)

    print("\nResults:")
    for k, v in stats_res.items():
        print(f"{k}: {v}")

    return stats_res