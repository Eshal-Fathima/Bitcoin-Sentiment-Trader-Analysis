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
        df["sentiment"].str.strip().str.lower().map(label_map).fillna(df["sentiment"])
    )

    df["is_fear"] = df["sentiment"].isin({"Extreme Fear", "Fear"}).astype(int)

    df = df.sort_values("date").drop_duplicates("date")
    df.reset_index(drop=True, inplace=True)

    return df


def load_trades(path: str | None = None) -> pd.DataFrame:
    path = path or os.path.join(RAW_DIR, "hyperliquid_trades.csv")
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    TARGET_PRIORITY = {
        "time": ["timestamp", "timestamp_ist", "time"],
        "account": ["account"],
        "symbol": ["coin"],
        "price": ["execution_price"],
        "size": ["size_usd"],
        "side": ["side"],
        "pnl": ["closed_pnl"],
    }

    rename = {}
    for target, candidates in TARGET_PRIORITY.items():
        for c in candidates:
            if c in df.columns:
                rename[c] = target
                break

    df.rename(columns=rename, inplace=True)

    # 🔥 FORCE leverage (since not present)
    df["leverage"] = 1

    # 🔥 FIXED TIME PARSING (IMPORTANT)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # Try unix timestamp if failed
    if df["time"].isna().sum() > len(df) * 0.5:
        print("[info] Trying unix timestamp parsing...")
        df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")

    # KEEP only valid rows (safe filtering)
    before = len(df)
    df = df[df["time"].notna()]
    after = len(df)
    print(f"[time parsing] kept {after}/{before} rows")

    df["time"] = df["time"].dt.tz_localize(None)
    df["date"] = df["time"].dt.normalize()

    # Convert numeric
    for col in ["price", "size", "pnl", "leverage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_win"] = (df["pnl"] > 0).astype(int)

    df = df.sort_values("time").reset_index(drop=True)

    return df


# ── Merge ──────────────────────────────────────────────────────────────────
def merge_datasets(sentiment, trades):

    sentiment = sentiment.sort_values("date")
    trades = trades.sort_values("date")

    merged = pd.merge_asof(
        trades,
        sentiment,
        on="date",
        direction="backward"
    )

    merged["sentiment"].fillna("Neutral", inplace=True)
    merged["is_fear"].fillna(0, inplace=True)

    print(f"[merge] Final rows: {len(merged)}")

    return merged

# ── Save ───────────────────────────────────────────────────────────────────
def save_processed(df: pd.DataFrame):
    out = os.path.join(PROCESSED_DIR, "merged_data.csv")
    df.to_csv(out, index=False)
    print(f"[save] Written → {out}")


# ── Pipeline ───────────────────────────────────────────────────────────────
def run_pipeline():
    print("Loading sentiment …")
    sentiment = load_sentiment()
    print(f"  → {len(sentiment)} rows")

    print("Loading trades …")
    trades = load_trades()
    print(f"  → {len(trades)} rows")

    print("Merging …")
    merged = merge_datasets(sentiment, trades)

    save_processed(merged)
    return merged