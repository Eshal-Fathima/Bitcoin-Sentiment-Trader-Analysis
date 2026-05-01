# Bitcoin Sentiment vs Trader Performance Analysis

## Overview

This project analyzes how **Bitcoin market sentiment (Fear & Greed Index)** impacts **trader behavior, risk exposure, and profitability**. By combining sentiment data with real trading activity, the project uncovers how emotions like fear and greed influence decision-making and outcomes in financial markets.

---

## 📂 Project Structure

```text
bitcoin-sentiment-trader-analysis/
│
├── data/
│   ├── raw/                  # Original datasets (not pushed to GitHub)
│   │   ├── bitcoin_sentiment.csv
│   │   └── hyperliquid_trades.csv
│   │
│   └── processed/            # Cleaned & merged dataset
│       └── merged_data.csv
│
├── notebooks/
│   └── analysis.ipynb        # Main notebook for running the analysis
│
├── src/
│   ├── data_preprocessing.py # Data loading, cleaning, merging
│   ├── feature_engineering.py# Feature creation (PnL, win rate, leverage, etc.)
│   └── analysis.py           # Visualization + statistical analysis
│
outputs/
├── plots/                     # Generated visualizations (included in repo)
│   ├── 01_box_pnl.png
│   ├── 02_leverage.png
│   ├── 03_winrate.png
│   ├── 04_combined.png
│   └── 05_heatmap.png
│
└── results.csv               # Statistical results         # Summary/statistical outputs
│
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Workflow

The project follows a structured data pipeline:

1. **Data Preprocessing**
   * Load sentiment and trade datasets
   * Clean column formats and timestamps
   * Merge datasets based on date alignment

2. **Feature Engineering**
   * Compute key metrics:
     * Total PnL
     * Win rate
     * Average leverage
     * Trade frequency
   * Create trader profiles and risk indicators

3. **Analysis & Visualization**
   * Generate meaningful plots
   * Analyze relationships between sentiment and performance
   * Perform statistical testing

---

## 📊 Key Visualizations

This project focuses on 5 essential, non-redundant graphs, each designed to capture a different aspect of trader behavior under varying market sentiment:

* 📦 **PnL Distribution (Box Plot)** → Shows how profitability varies across sentiment
  → Highlights median performance, spread, and outliers, helping identify whether traders perform better during Fear or Greed phases.

* 📈 **Leverage Usage by Sentiment** → Highlights risk-taking behavior
  → Reveals how traders adjust leverage based on market emotions, indicating whether risk appetite increases during Greed or decreases during Fear.

* 🎯 **Win Rate Distribution** → Compares trading success rates
  → Evaluates consistency of trader performance and whether sentiment affects the probability of winning trades.

* 🔥 **Combined Plot (Box + Individual Trades)** → Displays both distribution and raw data
  → Combines statistical summary with individual trade points to uncover hidden patterns, clusters, and variability that averages alone cannot show.

* 📊 **Correlation Heatmap** → Reveals relationships between trading metrics
  → Identifies how key variables (PnL, leverage, trade count, win rate) interact, helping understand which factors most influence trading outcomes.


---

## Insights
* Trader performance varies across market sentiment phases
* High leverage does not consistently improve profitability
* Market sentiment influences risk-taking behavior
* Distribution-based analysis provides deeper insights than averages

---

##  Notes
* The dataset contains **limited unique timestamps**,
  so the analysis focuses on **behavioral patterns** rather than time-series trends
* Raw datasets are excluded from GitHub via `.gitignore`

---

## Tech Stack
* Python
* pandas, numpy
* matplotlib, seaborn
* scipy (statistical testing)

---

This project demonstrates how **market psychology directly impacts trading behavior and outcomes**, emphasizing the importance of incorporating sentiment into trading strategies.
