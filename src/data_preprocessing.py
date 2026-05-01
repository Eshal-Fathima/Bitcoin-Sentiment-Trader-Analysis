"""
data_preprocessing.py
---------------------
Loads, cleans, and merges the Bitcoin Fear/Greed sentiment data
with the Hyperliquid trader data.
"""

import pandas as pd
import numpy as np
import os


# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ── Loaders ────────────────────────────────────────────────────────────────
def load_sentiment(path: str | None = None) -> pd.DataFrame:
    """Load and clean the Fear & Greed sentiment CSV."""
    path = path or os.path.join(RAW_DIR, "bitcoin_sentiment.csv")
    df = pd.read_csv(path)

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Rename to standard names
    rename = {}
    for col in df.columns:
        if "date" in col:
            rename[col] = "date"
        elif "class" in col or "sentiment" in col or "value_classification" in col:
            rename[col] = "sentiment"
        elif "value" in col and "class" not in col:
            rename[col] = "fear_greed_value"
    df.rename(columns=rename, inplace=True)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df["date"] = df["date"].dt.normalize()          # strip time component

    # Standardise sentiment labels
    label_map = {
        "extreme fear":  "Extreme Fear",
        "fear":          "Fear",
        "neutral":       "Neutral",
        "greed":         "Greed",
        "extreme greed": "Extreme Greed",
    }
    if "sentiment" in df.columns:
        df["sentiment"] = (
            df["sentiment"].str.strip().str.lower().map(label_map).fillna(df["sentiment"])
        )

    # Binary label: Fear-side vs Greed-side
    fear_labels = {"Extreme Fear", "Fear"}
    df["is_fear"] = df["sentiment"].isin(fear_labels).astype(int)

    df.sort_values("date", inplace=True)
    df.drop_duplicates(subset="date", keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_trades(path: str | None = None) -> pd.DataFrame:
    """Load and clean the Hyperliquid trader CSV."""
    path = path or os.path.join(RAW_DIR, "hyperliquid_trades.csv")
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Explicit rename map — priority order matters, first match wins per target
    # We build target→source mapping to ensure no two cols map to the same name
    TARGET_PRIORITY = {
        "time":     ["timestamp_ist", "time", "timestamp", "datetime"],
        "account":  ["account", "trader", "wallet", "user"],
        "symbol":   ["coin", "symbol", "asset"],
        "price":    ["execution_price", "price", "px"],
        "size":     ["size_usd", "sz", "size", "qty", "amount"],
        "side":     ["side", "direction"],
        "leverage": ["leverage", "lev"],
        "pnl":      ["closed_pnl", "pnl", "profit"],
        "event":    ["event", "type"],
    }

    rename = {}
    used_targets = set()
    for target, candidates in TARGET_PRIORITY.items():
        for candidate in candidates:
            if candidate in df.columns and target not in used_targets:
                rename[candidate] = target
                used_targets.add(target)
                break

    # Any remaining unmatched size/price columns get renamed to avoid conflicts
    for col in df.columns:
        if col not in rename and col not in rename.values():
            if "size" in col:
                rename[col] = f"size_{col}"   # e.g. size_tokens stays unique
            elif "price" in col or "exec" in col:
                rename[col] = f"extra_{col}"

    df.rename(columns=rename, inplace=True)

    # Parse time — handle both unix timestamps and date strings
    if "time" in df.columns:
        # Try standard datetime parse first
        parsed = pd.to_datetime(df["time"], errors="coerce")
        # If most values are NaT, try unix timestamp (seconds)
        if parsed.isna().sum() > len(parsed) * 0.5:
            parsed = pd.to_datetime(df["time"], unit="s", errors="coerce")
        df["time"] = parsed
        df.dropna(subset=["time"], inplace=True)
        df["date"] = df["time"].dt.normalize()

    # Numeric coercions
    for col in ("price", "size", "leverage", "pnl"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing leverage with 1
    if "leverage" in df.columns:
        df["leverage"].fillna(1, inplace=True)

    # Winning trade flag
    if "pnl" in df.columns:
        df["is_win"] = (df["pnl"] > 0).astype(int)

    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Merge ──────────────────────────────────────────────────────────────────
def merge_datasets(sentiment: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Left-join trades with daily sentiment on date."""
    merged = trades.merge(sentiment[["date", "sentiment", "is_fear", "fear_greed_value"]
                                     if "fear_greed_value" in sentiment.columns
                                     else ["date", "sentiment", "is_fear"]],
                          on="date", how="left")
    # Drop rows where sentiment couldn't be matched
    before = len(merged)
    merged.dropna(subset=["sentiment"], inplace=True)
    after = len(merged)
    print(f"[merge] Kept {after}/{before} rows after sentiment join.")
    return merged


# ── Save ───────────────────────────────────────────────────────────────────
def save_processed(df: pd.DataFrame, name: str = "merged_data.csv"):
    out = os.path.join(PROCESSED_DIR, name)
    df.to_csv(out, index=False)
    print(f"[save] Written → {out}")


# ── Pipeline ───────────────────────────────────────────────────────────────
def run_pipeline() -> pd.DataFrame:
    print("Loading sentiment …")
    sentiment = load_sentiment()
    print(f"  → {len(sentiment)} rows | {sentiment['date'].min().date()} to {sentiment['date'].max().date()}")

    print("Loading trades …")
    trades = load_trades()
    print(f"  → {len(trades)} rows")

    print("Merging …")
    merged = merge_datasets(sentiment, trades)

    save_processed(merged)
    return merged


if __name__ == "__main__":
    run_pipeline()