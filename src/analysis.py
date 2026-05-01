import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

PALETTE = {
    "Extreme Fear": "#d62728",
    "Fear": "#ff7f0e",
    "Neutral": "#bcbd22",
    "Greed": "#2ca02c",
    "Extreme Greed": "#17becf",
}
SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


def _save(fig, name):
    fig.savefig(os.path.join(PLOTS_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {name}")


# 1. BOX PLOT (summary)
def plot_box(daily):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=daily, x="sentiment", y="total_pnl",
                order=SENTIMENT_ORDER, palette=PALETTE, ax=ax)
    ax.set_title("PnL Distribution (Boxplot)")
    _save(fig, "01_box.png")


# 2. VIOLIN PLOT (shape)
def plot_violin(daily):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=daily, x="sentiment", y="total_pnl",
                   order=SENTIMENT_ORDER, palette=PALETTE,
                   inner="quartile", ax=ax)
    ax.set_title("PnL Distribution Shape (Violin)")
    _save(fig, "02_violin.png")


# 3. SCATTER (relationship)
def plot_scatter(daily):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=daily, x="avg_leverage", y="total_pnl",
                    hue="sentiment", palette=PALETTE,
                    alpha=0.6, ax=ax)
    ax.set_title("Leverage vs PnL")
    _save(fig, "03_scatter.png")


# 4. COMBINED (box + strip)
def plot_combined(daily):
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(data=daily, x="sentiment", y="total_pnl",
                order=SENTIMENT_ORDER, palette=PALETTE, ax=ax)

    sns.stripplot(data=daily, x="sentiment", y="total_pnl",
                  order=SENTIMENT_ORDER, color="white",
                  alpha=0.3, ax=ax)

    ax.set_title("PnL Distribution + Individual Points")
    _save(fig, "04_combined.png")


# 5. HEATMAP (correlation)
def plot_heatmap(daily):
    cols = ["total_pnl", "win_rate", "avg_leverage", "trade_count"]
    cols = [c for c in cols if c in daily.columns]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(daily[cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    _save(fig, "05_heatmap.png")


# ── STATS ─────────────────────────
def run_stats(daily):
    fear = daily[daily["is_fear"] == 1]["total_pnl"]
    greed = daily[daily["is_fear"] == 0]["total_pnl"]

    u, p = stats.mannwhitneyu(fear, greed)

    return {
        "fear_mean": fear.mean(),
        "greed_mean": greed.mean(),
        "p_value": p,
        "significant": p < 0.05
    }


# ── MAIN ─────────────────────────
def run_all_analysis(daily):
    print("\n📊 Generating 5 key graphs...")

    plot_box(daily)
    plot_violin(daily)
    plot_scatter(daily)
    plot_combined(daily)
    plot_heatmap(daily)

    print("\n📐 Running stats...")
    stats_res = run_stats(daily)

    print("\nResults:")
    for k, v in stats_res.items():
        print(f"{k}: {v}")

    return stats_res