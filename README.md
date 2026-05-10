<div align="center">

# 📊 Bitcoin Sentiment vs Trader Performance Analysis

**Does market psychology actually drive trading outcomes — or just behavior?**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)
[![pandas](https://img.shields.io/badge/pandas-EDA-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![scipy](https://img.shields.io/badge/scipy-Statistical%20Testing-8CAAE6?style=flat-square)](https://scipy.org)

*End-to-end data pipeline · 5 targeted visualizations · Statistical validation · Real Hyperliquid trade data*

</div>

---

## The Question

Markets run on emotion as much as logic. The Fear & Greed Index captures the collective mood of Bitcoin traders at any given moment — but does that mood actually change whether they win or lose? Or does it just change how recklessly they bet?

This project investigates exactly that, combining the **Bitcoin Fear & Greed Index** with **real trader activity from Hyperliquid** to measure how sentiment shapes profitability, leverage, and win rates across thousands of trades.

> **Key Finding:** Sentiment affects *behavior* more than *outcomes*. Traders take on measurably more risk during Greed phases — but higher leverage does not consistently translate into better PnL or win rates. Market psychology moves the dial on risk appetite, not on results.

---

## Project Structure

```
bitcoin-sentiment-trader-analysis/
│
├── data/
│   ├── raw/                        # Original datasets (excluded via .gitignore)
│   │   ├── bitcoin_sentiment.csv   # Fear & Greed Index over time
│   │   └── hyperliquid_trades.csv  # Real trader activity
│   └── processed/
│       └── merged_data.csv         # Cleaned & merged dataset
│
├── notebooks/
│   └── analysis.ipynb              # Full walkthrough — EDA to insights
│
├── src/
│   ├── data_preprocessing.py       # Loading, cleaning, timestamp alignment, merging
│   ├── feature_engineering.py      # PnL, win rate, leverage, trade frequency metrics
│   └── analysis.py                 # Visualization generation + statistical testing
│
├── outputs/
│   ├── plots/                      # All 5 generated charts (included in repo)
│   │   ├── 01_box_pnl.png
│   │   ├── 02_leverage.png
│   │   ├── 03_winrate.png
│   │   ├── 04_combined.png
│   │   └── 05_heatmap.png
│   └── results.csv                 # Statistical test outputs
│
├── requirements.txt
└── README.md
```

---

## Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| Fear & Greed Index | Bitcoin Sentiment API | Daily sentiment score (0–100) mapped to labels: Extreme Fear → Extreme Greed |
| Trade Data | Hyperliquid | Real perpetual futures trades including PnL, leverage, size, and timestamps |

The two datasets are merged on date — each trade is tagged with the sentiment label that was active on that trading day, enabling direct comparison of behavior across sentiment phases.

> **Note on data scope:** The dataset contains limited unique timestamps, so the analysis focuses on behavioral and distributional patterns rather than time-series trends. All insights are framed accordingly.

---

## Pipeline

### 1. Data Preprocessing (`src/data_preprocessing.py`)
- Load both CSVs and standardize column formats
- Parse and align timestamps across datasets
- Merge trade records with their corresponding daily sentiment label
- Handle missing values and remove malformed rows

### 2. Feature Engineering (`src/feature_engineering.py`)
Raw trade data becomes analysis-ready by computing:

| Feature | Description |
|---------|-------------|
| **Total PnL** | Net profit/loss per trade in USD |
| **Win Rate** | Proportion of profitable trades per trader/sentiment group |
| **Average Leverage** | Mean leverage used, grouped by sentiment phase |
| **Trade Frequency** | Number of trades executed per sentiment period |
| **Trader Risk Profile** | Composite indicator combining leverage and trade size |

### 3. Analysis & Visualization (`src/analysis.py`)
Five targeted charts, each built to answer a distinct question — no redundant plots.

---

## Visualizations

### 01 · PnL Distribution by Sentiment — Box Plot
**What it shows:** The spread of profit and loss across each sentiment category (Extreme Fear, Fear, Neutral, Greed, Extreme Greed). The box captures where most traders landed; the median line shows the typical outcome; outlier dots reveal unusually large wins or losses.

**What it answers:** Do traders actually make more money when the market feels greedy?

---

### 02 · Leverage Usage by Sentiment
**What it shows:** How much risk traders took on — measured by leverage — during each sentiment phase. Higher leverage means trading with more borrowed capital relative to your balance.

**What it answers:** Does market optimism push traders to take on more risk? And by how much?

---

### 03 · Win Rate Distribution by Sentiment
**What it shows:** The percentage of trades that were profitable, compared across sentiment phases.

**What it answers:** Does sentiment affect *how often* traders win — or just *how much* they bet? (Spoiler: mostly the latter.)

---

### 04 · Combined Plot — Box + Individual Trade Points
**What it shows:** The box plot summary overlaid with every individual trade as a scatter point. This is the most information-dense chart in the project.

**Why it matters:** Averages and medians can mask a lot. Clustering, bimodal distributions, and extreme outliers all become visible when you plot every data point alongside the statistical summary. This chart is where the most nuanced patterns emerge.

---

### 05 · Correlation Heatmap
**What it shows:** A color-coded grid comparing every trading metric against every other — PnL, leverage, win rate, trade frequency, and trade size. Darker cells indicate stronger relationships; lighter cells indicate weak or no correlation.

**What it answers:** Which factors actually drive outcomes? Is leverage correlated with PnL? Does trading more frequently improve win rate? The heatmap makes multivariate relationships scannable at a glance.

---

## Key Findings

**Sentiment shapes risk appetite, not returns.**
Traders consistently increase leverage during Greed phases — risk-taking behavior is clearly sentiment-driven. But the PnL distribution and win rates do not improve proportionally. The market takes the other side.

**High leverage ≠ high profitability.**
The correlation heatmap reveals a weak relationship between leverage and PnL. Traders who bet bigger during Greed do not systematically outperform those who don't.

**Distribution-based analysis outperforms average-based analysis.**
Mean PnL figures looked similar across sentiment phases — but the combined overlay chart revealed meaningful differences in the spread and clustering of outcomes. Averages here were actively misleading.

**Win rates are more stable than PnL across sentiment phases.**
While raw profitability fluctuated, the proportion of winning trades remained relatively consistent — suggesting that sentiment changes *how* traders trade more than *whether* their trade logic is sound.

---

## Statistical Validation

All visual findings are backed by statistical testing using **scipy**:
- Group comparisons across sentiment phases use appropriate tests based on distribution shape
- Results are exported to `outputs/results.csv` for reproducibility
- Effect sizes are reported alongside p-values to distinguish statistical from practical significance

---

## Quickstart

```bash
git clone https://github.com/Eshal-Fathima/Bitcoin-Sentiment-Trader-Analysis
cd Bitcoin-Sentiment-Trader-Analysis
pip install -r requirements.txt
```

**Run the full pipeline via notebook:**
```bash
jupyter notebook notebooks/analysis.ipynb
```

**Or run scripts individually:**
```bash
python src/data_preprocessing.py   # Clean and merge data
python src/feature_engineering.py  # Compute trading metrics
python src/analysis.py             # Generate plots + stats
```

Output charts will appear in `outputs/plots/`. Statistical results in `outputs/results.csv`.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **pandas** | Data loading, cleaning, merging, groupby aggregations |
| **NumPy** | Numerical operations and array manipulation |
| **matplotlib** | Base plotting layer for all visualizations |
| **seaborn** | Statistical chart styling — box plots, heatmaps, distributions |
| **scipy** | Hypothesis testing and statistical validation |
| **Jupyter Notebook** | End-to-end analysis walkthrough with inline outputs |

---

## Skills Demonstrated

- **Exploratory Data Analysis (EDA)** — Systematic investigation of a real financial dataset from raw CSV to insight
- **Feature engineering** — Deriving meaningful metrics (win rate, leverage profiles, PnL) from raw trade logs
- **Multi-dataset merging** — Aligning time-series sentiment data with event-level trade records
- **Statistical thinking** — Going beyond visuals with hypothesis testing and effect size reporting
- **Visualization design** — 5 non-redundant charts, each answering a distinct analytical question
- **Insight communication** — Translating statistical findings into plain-language conclusions

---

## About

Built to explore how market psychology intersects with trading behavior and outcomes in the Bitcoin perpetual futures market. The analysis deliberately avoids overclaiming — where data limitations exist (limited unique timestamps), the scope of conclusions is adjusted accordingly.

**Author:** [Eshal Fathima](https://github.com/Eshal-Fathima) · CS Undergrad, Big Data Analytics · SRM University
