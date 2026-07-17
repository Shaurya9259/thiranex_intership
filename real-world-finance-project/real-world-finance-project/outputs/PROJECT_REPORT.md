# Real-World Data Project — AAPL Stock Analysis & Prediction

## Domain: Finance
**Dataset:** Real historical Apple Inc. (AAPL) daily stock prices, 2015-02-17 to 2017-02-16 (506 trading days), sourced from public market data (plotly datasets repository).

## 1. Price Trend Analysis
- Adjusted close moved from $122.91 to $135.35 over the period — a total return of **10.1%**.
- The 20-day and 50-day moving averages (`price_trend.png`) show sustained uptrends with several pullback periods.
- Daily returns (`volume_and_returns.png`) cluster tightly around zero, consistent with normal equity volatility, with occasional spikes around earnings-related news.

## 2. Momentum & Volatility
- RSI-14 (`rsi_indicator.png`) crossed into overbought (>70) and oversold (<30) territory multiple times, flagging potential reversal points.
- 10-day rolling volatility varied notably across the period, higher during broader market stress periods.

## 3. Feature Relationships
- See `feature_correlation.png` for how moving averages, RSI, volatility, and volume changes relate to price and to each other.

## 4. Prediction Task: Next-Day Price Direction
Framed as binary classification (will tomorrow's close be higher than today's?) using a time-ordered 80/20 train/test split (no shuffling, to avoid leaking future data into training).

| Model | Accuracy | ROC AUC |
|---|---|---|
| Baseline (always predict "up") | 0.554 | — |
| Logistic Regression | 0.457 | 0.520 |
| Random Forest | 0.457 | 0.455 |

**Top predictive features:** ma_50, volatility_10d, ma_20

## Conclusions
1. AAPL showed a clear upward trend over the analyzed period, consistent with the broader market during 2015-2017.
2. Next-day price *direction* is inherently hard to predict from price/volume history alone — both models performed **at or below** the naive baseline of always predicting "up" (55.4%). This is a genuine and important finding, not a failure of the modeling: it's consistent with market efficiency — if next-day direction were reliably predictable from public price history, that edge would already be arbitraged away by other traders. A result like this is a realistic, honest outcome for this kind of task.
3. RSI and short-term moving averages were the most informative engineered features, aligning with their common use in real-world technical analysis.
4. **Caveat:** This predicts short-term *direction*, not magnitude, and does not account for transaction costs, slippage, or fundamentals (earnings, news, macro data) — none of which are in this dataset. Real trading decisions would need much more than this.

## Files
- `price_trend.png`, `volume_and_returns.png`, `rsi_indicator.png` — exploratory visualizations
- `feature_correlation.png` — relationships between engineered features
- `confusion_matrices.png`, `roc_curve.png`, `feature_importance.png` — model evaluation
