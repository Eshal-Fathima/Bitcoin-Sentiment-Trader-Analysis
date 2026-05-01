import pandas as pd
import numpy as np


SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


def add_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-trader per-day features."""

    grp = df.groupby(["date", "account"])

    agg = grp.agg(
        total_pnl        = ("pnl",       "sum"),
        mean_pnl         = ("pnl",       "mean"),
        trade_count      = ("pnl",       "count"),
        win_count        = ("is_win",    "sum"),
        avg_leverage     = ("leverage",  "mean"),
        max_leverage     = ("leverage",  "max"),
        avg_size         = ("size",      "mean"),
        total_volume     = ("size",      "sum"),
        sentiment        = ("sentiment", "first"),
        is_fear          = ("is_fear",   "first"),
    ).reset_index()

    agg["win_rate"]          = agg["win_count"] / agg["trade_count"]
    agg["sentiment"]         = pd.Categorical(agg["sentiment"],
                                              categories=SENTIMENT_ORDER, ordered=True)
    return agg


def add_sentiment_shift(daily: pd.DataFrame) -> pd.DataFrame:
    """Flag days where sentiment changed from previous day."""
    sent_by_day = (
        daily[["date", "sentiment"]]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    sent_by_day["prev_sentiment"] = sent_by_day["sentiment"].shift(1)
    sent_by_day["sentiment_shift"] = sent_by_day["sentiment"] != sent_by_day["prev_sentiment"]

    # Fear → Greed transition
    fear_set  = {"Extreme Fear", "Fear"}
    greed_set = {"Greed", "Extreme Greed"}
    sent_by_day["fear_to_greed"] = (
        sent_by_day["prev_sentiment"].isin(fear_set) &
        sent_by_day["sentiment"].isin(greed_set)
    )
    sent_by_day["greed_to_fear"] = (
        sent_by_day["prev_sentiment"].isin(greed_set) &
        sent_by_day["sentiment"].isin(fear_set)
    )

    daily = daily.merge(
        sent_by_day[["date", "prev_sentiment", "sentiment_shift",
                      "fear_to_greed", "greed_to_fear"]],
        on="date", how="left"
    )
    return daily


def add_trader_profile(daily: pd.DataFrame) -> pd.DataFrame:
    """Classify traders into performance tiers."""
    trader_stats = (
        daily.groupby("account")
        .agg(lifetime_pnl=("total_pnl", "sum"),
             avg_win_rate=("win_rate",   "mean"),
             total_trades=("trade_count","sum"))
        .reset_index()
    )

    # Percentile-based tiers
    p33 = trader_stats["lifetime_pnl"].quantile(0.33)
    p66 = trader_stats["lifetime_pnl"].quantile(0.66)

    def tier(pnl):
        if pnl >= p66:  return "Top"
        if pnl >= p33:  return "Mid"
        return "Bottom"

    trader_stats["trader_tier"] = trader_stats["lifetime_pnl"].apply(tier)

    daily = daily.merge(trader_stats[["account", "trader_tier",
                                       "lifetime_pnl", "avg_win_rate",
                                       "total_trades"]],
                         on="account", how="left")
    return daily


def add_risk_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add risk-adjusted metrics."""
    # Sharpe-like ratio per trader (across days)
    def sharpe(s):
        if s.std() == 0 or len(s) < 2:
            return np.nan
        return s.mean() / s.std()

    trader_sharpe = (
        daily.groupby("account")["total_pnl"]
        .apply(sharpe)
        .reset_index()
        .rename(columns={"total_pnl": "sharpe_ratio"})
    )
    daily = daily.merge(trader_sharpe, on="account", how="left")

    # Over-leverage flag: top-quartile leverage
    lev_q75 = daily["avg_leverage"].quantile(0.75)
    daily["is_overleveraged"] = (daily["avg_leverage"] >= lev_q75).astype(int)

    return daily


def build_features(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline."""
    print("Building daily features …")
    daily = add_daily_features(merged_df)

    print("Adding sentiment shift flags …")
    daily = add_sentiment_shift(daily)

    print("Classifying trader profiles …")
    daily = add_trader_profile(daily)

    print("Computing risk features …")
    daily = add_risk_features(daily)

    print(f"Feature set ready: {daily.shape[0]} rows × {daily.shape[1]} cols")
    return daily