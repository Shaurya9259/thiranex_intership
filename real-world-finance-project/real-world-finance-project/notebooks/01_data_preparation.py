"""
01_data_preparation.py
------------------------
Loads real historical Apple (AAPL) stock data (2015-2017, sourced from
plotly's public datasets repo) and prepares it for analysis: renames
columns, engineers financial features used throughout the project.
"""
import pandas as pd
import numpy as np

RAW_PATH = "/home/claude/real-world-finance-project/data/aapl_raw.csv"
CLEAN_PATH = "/home/claude/real-world-finance-project/data/aapl_clean.csv"

df = pd.read_csv(RAW_PATH)

# Clean column names
df.columns = [c.replace("AAPL.", "").lower() for c in df.columns]
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print("Shape:", df.shape)
print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
print("Missing values:\n", df.isna().sum())

# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------
df["daily_return_pct"] = df["adjusted"].pct_change() * 100
df["ma_10"] = df["adjusted"].rolling(10).mean()
df["ma_20"] = df["adjusted"].rolling(20).mean()
df["ma_50"] = df["adjusted"].rolling(50).mean()
df["volatility_10d"] = df["daily_return_pct"].rolling(10).std()
df["volume_change_pct"] = df["volume"].pct_change() * 100

# Relative Strength Index (RSI, 14-day) - classic momentum indicator
delta = df["adjusted"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df["rsi_14"] = 100 - (100 / (1 + rs))

# Prediction target: will tomorrow's price close higher than today's? (1 = up, 0 = down)
df["next_day_up"] = (df["adjusted"].shift(-1) > df["adjusted"]).astype(int)

df.to_csv(CLEAN_PATH, index=False)
print(f"\nSaved cleaned + feature-engineered data to {CLEAN_PATH}")
print("Final shape:", df.shape)
