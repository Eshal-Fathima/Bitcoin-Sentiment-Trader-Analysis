# 📊 Bitcoin Sentiment vs Trader Performance Analysis

## 🚀 Overview

This project explores how **market sentiment (Fear & Greed Index)** influences **trader behavior, risk-taking, and profitability** using real trading data.

By combining sentiment data with trade-level metrics, the analysis reveals how emotional market phases affect trading outcomes.

---

## 🎯 Objectives

* Analyze the impact of sentiment on trader profitability (PnL)
* Examine leverage usage across different market conditions
* Compare win rates under Fear vs Greed phases
* Identify relationships between trading features using correlation analysis

---

## 📂 Project Structure

```
bitcoin-sentiment-trader-analysis/
│
├── notebooks/
│   └── analysis.ipynb        # Interactive analysis notebook
│
├── src/
│   └── analysis.py           # Visualization & statistical analysis
│
├── outputs/
│   └── plots/                # Generated visualizations
│
└── README.md
```

---

## 📊 Key Visualizations

This project focuses on **5 meaningful, non-redundant visualizations**:

### 📦 1. PnL Distribution (Box Plot)

Shows how trader profitability varies across sentiment categories.

### 📈 2. Leverage Usage by Sentiment

Highlights how traders adjust risk exposure under Fear vs Greed.

### 🎯 3. Win Rate Distribution

Compares trading success rates across sentiment conditions.

### 🔥 4. Combined Plot (Box + Individual Points)

Reveals both distribution and individual trade behavior.

### 📊 5. Correlation Heatmap

Displays relationships between PnL, leverage, trade count, and win rate.

---

## 🧠 Key Insights

* Trader performance varies significantly across sentiment phases
* Increased leverage does not guarantee higher profitability
* Market sentiment influences risk-taking behavior
* Distribution-based analysis reveals patterns hidden by averages

---

## ⚙️ Tech Stack

* **Python**
* pandas, numpy
* matplotlib, seaborn
* scipy (statistical testing)

---

## 📈 Methodology

1. Data preprocessing and cleaning
2. Feature engineering (PnL, win rate, leverage, etc.)
3. Sentiment alignment using Fear & Greed Index
4. Visualization using multiple plot types
5. Statistical validation (Mann-Whitney U test)

---

## ⚠️ Note

The dataset contains limited unique timestamps.
Therefore, the analysis focuses on **behavioral patterns and distributions** rather than time-series trends.

---

## 📌 Conclusion

This project demonstrates that **market psychology plays a critical role in trading performance**, emphasizing the importance of sentiment-aware strategies in financial decision-making.

---

## 👤 Author

Your Name
