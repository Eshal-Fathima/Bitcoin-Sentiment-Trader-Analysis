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
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {name}")


# ── FIXED PLOTS ─────────────────────────────────────────

def plot_pnl_vs_sentiment(daily):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=daily, x="sentiment", y="total_pnl",
                order=SENTIMENT_ORDER, palette=PALETTE, ax=ax)
    ax.set_title("PnL Distribution by Sentiment")
    _save(fig, "01_pnl_vs_sentiment.png")


def plot_winrate_vs_sentiment(daily):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=daily, x="sentiment", y="win_rate",
                order=SENTIMENT_ORDER, palette=PALETTE, ax=ax)
    ax.set_title("Win Rate Distribution by Sentiment")
    _save(fig, "02_winrate_vs_sentiment.png")


def plot_trade_frequency(daily):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=daily, x="sentiment", y="trade_count",
                order=SENTIMENT_ORDER, palette=PALETTE, ax=ax)
    ax.set_title("Trade Count Distribution")
    _save(fig, "03_trade_frequency.png")


def plot_cumulative_pnl(daily):
    agg = daily.groupby("account")["total_pnl"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(agg.index, agg["total_pnl"].cumsum())
    ax.set_title("Cumulative PnL Across Traders")
    _save(fig, "04_cumulative_pnl.png")


def plot_trader_tier(daily):
    if "trader_tier" not in daily.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=daily, x="trader_tier", y="total_pnl", ax=ax)
    ax.set_title("Trader Tier Performance")
    _save(fig, "05_trader_tier.png")


# ── STATS ─────────────────────────────────────────

def run_statistical_tests(daily):
    fear = daily[daily["is_fear"] == 1]["total_pnl"]
    greed = daily[daily["is_fear"] == 0]["total_pnl"]

    u, p = stats.mannwhitneyu(fear, greed)

    return {
        "fear_mean": fear.mean(),
        "greed_mean": greed.mean(),
        "p_value": p,
        "significant": p < 0.05
    }


# ── MAIN FUNCTION (THIS WAS MISSING) ───────────────────

def run_all_analysis(daily):
    print("Running analysis...")

    plot_pnl_vs_sentiment(daily)
    plot_winrate_vs_sentiment(daily)
    plot_trade_frequency(daily)
    plot_cumulative_pnl(daily)
    plot_trader_tier(daily)

    stats_res = run_statistical_tests(daily)

    print("\nStatistical Results:")
    for k, v in stats_res.items():
        print(f"{k}: {v}")

    return stats_res