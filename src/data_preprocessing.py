"""
data_preprocessing.py
---------------------
Loads, cleans, and merges the Bitcoin Fear/Greed sentiment data
with the Hyperliquid trader data.
"""

import pandas as pd
import os


# ── Paths ────────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ── SENTIMENT ────────────────────────────────────────────────────────────
def load_sentiment(path=None):
    path = path or os.path.join(RAW_DIR, "bitcoin_sentiment.csv")
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    rename = {}
    for col in df.columns:
        if "date" in col:
            rename[col] = "date"
        elif "class" in col or "sentiment" in col:
            rename[col] = "sentiment"
        elif "value" in col and "class" not in col:
            rename[col] = "fear_greed_value"

    df.rename(columns=rename, inplace=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]
    df["date"] = df["date"].dt.normalize()

    label_map = {
        "extreme fear":  "Extreme Fear",
        "fear":          "Fear",
        "neutral":       "Neutral",
        "greed":         "Greed",
        "extreme greed": "Extreme Greed",
    }

    df["sentiment"] = (
        df["sentiment"].astype(str).str.lower().map(label_map).fillna(df["sentiment"])
    )

    df["is_fear"] = df["sentiment"].isin(["Fear", "Extreme Fear"]).astype(int)

    df = df.sort_values("date").drop_duplicates("date")
    df.reset_index(drop=True, inplace=True)

    return df


# ── TRADES ───────────────────────────────────────────────────────────────
def load_trades(path=None):
    path = path or os.path.join(RAW_DIR, "hyperliquid_trades.csv")
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Rename based on your dataset
    rename = {
        "account": "account",
        "coin": "symbol",
        "execution_price": "price",
        "size_usd": "size",
        "side": "side",
        "timestamp": "time",        # IMPORTANT: use this column
        "closed_pnl": "pnl",
    }

    df.rename(columns=rename, inplace=True)

    # 🔥 FIXED TIMESTAMP PARSING (milliseconds)
    if pd.api.types.is_numeric_dtype(df["time"]):
        print("[info] Detected numeric timestamp → using milliseconds")
        df["time"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
    else:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # Debug
    print("[DEBUG] Sample timestamps:")
    print(df["time"].head())

    # Remove invalid rows safely
    before = len(df)
    df = df[df["time"].notna()]
    after = len(df)
    print(f"[time parsing] kept {after}/{before} rows")

    # Remove timezone
    df["time"] = df["time"].dt.tz_localize(None)

    # Extract date
    df["date"] = df["time"].dt.normalize()

    # Convert numeric
    for col in ["price", "size", "pnl"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add missing columns
    df["leverage"] = 1
    df["is_win"] = (df["pnl"] > 0).astype(int)

    df = df.sort_values("time").reset_index(drop=True)

    return df


# ── MERGE ────────────────────────────────────────────────────────────────
def merge_datasets(sentiment, trades):

    sentiment = sentiment.sort_values("date")
    trades = trades.sort_values("date")

    # 🔥 Use nearest match (important)
    merged = pd.merge_asof(
        trades,
        sentiment,
        on="date",
        direction="backward"
    )

    merged["sentiment"].fillna("Neutral", inplace=True)
    merged["is_fear"].fillna(0, inplace=True)

    print("\n[merge] Sentiment distribution:")
    print(merged["sentiment"].value_counts())

    return merged


# ── SAVE ─────────────────────────────────────────────────────────────────
def save_processed(df):
    out = os.path.join(PROCESSED_DIR, "merged_data.csv")
    df.to_csv(out, index=False)
    print(f"[save] Written → {out}")


# ── PIPELINE ─────────────────────────────────────────────────────────────
def run_pipeline():
    print("Loading sentiment …")
    sentiment = load_sentiment()
    print(f"  → {len(sentiment)} rows")

    print("\nLoading trades …")
    trades = load_trades()
    print(f"  → {len(trades)} rows")

    print("\nMerging …")
    merged = merge_datasets(sentiment, trades)

    save_processed(merged)

    return merged


if __name__ == "__main__":
    run_pipeline()