# Real-World Data Project — Finance (AAPL Stock Analysis & Prediction)

Internship task: work on a domain-specific dataset for applied learning.

## Domain: Finance
This project performs end-to-end analysis and prediction on **real historical Apple Inc. (AAPL) stock data** — not synthetic data — covering 506 trading days from February 2015 to February 2017, sourced from a public market-data repository.

## Project Structure
```
real-world-finance-project/
├── data/
│   ├── aapl_raw.csv              # Real AAPL OHLCV data (source data)
│   └── aapl_clean.csv            # Cleaned + feature-engineered data
├── notebooks/
│   ├── 01_data_preparation.py    # Cleaning & feature engineering
│   ├── 02_analysis_and_prediction.py  # Full analysis + prediction pipeline
│   └── finance_analysis.ipynb    # Notebook version with all outputs
├── outputs/
│   ├── price_trend.png
│   ├── volume_and_returns.png
│   ├── rsi_indicator.png
│   ├── feature_correlation.png
│   ├── confusion_matrices.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── PROJECT_REPORT.md         # Structured findings & conclusions
├── requirements.txt
└── README.md
```

## Approach
1. **Data preparation** — cleaned real AAPL price/volume data, engineered financial features: moving averages (10/20/50-day), 10-day rolling volatility, RSI (14-day momentum indicator), daily return %, volume change %
2. **Exploratory analysis** — price trend with moving averages, volume patterns, return distribution, RSI overbought/oversold zones
3. **Prediction task** — binary classification: will tomorrow's close be higher than today's? Trained Logistic Regression and Random Forest on a **time-ordered** 80/20 split (critical for time series — no shuffling, so the model never sees the future during training)
4. **Evaluation** — accuracy, ROC-AUC, confusion matrices, feature importance, compared against a naive baseline

## Key Findings
- AAPL returned **+10.1%** over the analyzed period, with a clear uptrend visible in both raw price and moving averages
- **Next-day direction is hard to predict**: both models performed at or below the naive "always predict up" baseline (55.4% accuracy). This is a realistic, honest result — it reflects market efficiency, not a modeling mistake. If short-term direction were easily predictable from public price history alone, that edge wouldn't persist.
- RSI and short-term moving averages were the most informative features, consistent with their real-world use in technical analysis
- Full narrative in `outputs/PROJECT_REPORT.md`

## How to Run
```bash
pip install -r requirements.txt
python notebooks/01_data_preparation.py       # clean + engineer features
python notebooks/02_analysis_and_prediction.py # analysis, prediction, visuals
```
Or open `notebooks/finance_analysis.ipynb` in Jupyter to run interactively.

## Tools Used
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Expected Outcome
This project demonstrates applying data science skills in a real-world context: working with real (not synthetic) market data, engineering domain-relevant features, framing a genuine prediction problem correctly (time-ordered split for time series), and — importantly — reporting an honest negative-ish result rather than overstating what the model can do.
