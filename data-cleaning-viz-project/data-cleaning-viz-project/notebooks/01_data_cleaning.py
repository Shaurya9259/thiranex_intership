"""
01_data_cleaning.py
--------------------
Cleans the raw retail sales dataset:
 - Standardizes date formats
 - Fixes text casing/whitespace issues
 - Handles missing values
 - Removes duplicate rows
 - Detects and treats outliers using the IQR method
 - Saves a clean dataset for the visualization step
"""
import pandas as pd
import numpy as np

RAW_PATH = "/home/claude/data-cleaning-viz-project/data/raw_sales_data.csv"
CLEAN_PATH = "/home/claude/data-cleaning-viz-project/data/clean_sales_data.csv"

df = pd.read_csv(RAW_PATH)
print("Initial shape:", df.shape)

# ---------------------------------------------------------
# 1. Remove exact duplicate rows
# ---------------------------------------------------------
before = len(df)
df = df.drop_duplicates(subset=["order_id"], keep="first")
print(f"Removed {before - len(df)} duplicate rows")

# ---------------------------------------------------------
# 2. Standardize text columns (strip whitespace, fix casing)
# ---------------------------------------------------------
for col in ["category", "region", "payment_mode"]:
    df[col] = df[col].astype(str).str.strip().str.title()
    df[col] = df[col].replace("Nan", np.nan)

# ---------------------------------------------------------
# 3. Parse mixed date formats into a single datetime type
# ---------------------------------------------------------
# The raw data mixes several date formats (YYYY-MM-DD, DD/MM/YYYY,
# DD-Mon-YYYY, MM/DD/YYYY). A single pd.to_datetime call can't safely
# resolve all of these at once, so we try each known format in turn
# and keep the first one that parses successfully for each row.
KNOWN_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y", "%m/%d/%Y"]

def parse_mixed_dates(series):
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining = series.copy()
    for fmt in KNOWN_FORMATS:
        parsed = pd.to_datetime(remaining, format=fmt, errors="coerce")
        result = result.fillna(parsed)
        remaining = remaining.where(parsed.isna())
    return result

df["order_date"] = parse_mixed_dates(df["order_date"].astype(str))

# ---------------------------------------------------------
# 4. Handle missing values
# ---------------------------------------------------------
# Numeric: fill unit_price/units_sold with the median of their category
df["unit_price"] = df.groupby("category")["unit_price"].transform(
    lambda x: x.fillna(x.median())
)
df["units_sold"] = df.groupby("category")["units_sold"].transform(
    lambda x: x.fillna(x.median())
)
# Categorical: fill with mode (most frequent value)
for col in ["region", "payment_mode"]:
    df[col] = df[col].fillna(df[col].mode()[0])

# Drop rows where the date couldn't be parsed at all (rare, unrecoverable)
df = df.dropna(subset=["order_date"])

# ---------------------------------------------------------
# 5. Outlier treatment using IQR (Interquartile Range) method
# ---------------------------------------------------------
def cap_outliers_iqr(series, factor=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return series.clip(lower=max(lower, 0), upper=upper)

for col in ["units_sold", "unit_price"]:
    n_outliers = ((df[col] < df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))) |
                  (df[col] > df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))).sum()
    print(f"Outliers capped in {col}: {n_outliers}")
    df[col] = cap_outliers_iqr(df[col])

# ---------------------------------------------------------
# 6. Recompute derived column after cleaning
# ---------------------------------------------------------
df["revenue"] = (df["units_sold"] * df["unit_price"]).round(2)
df["month"] = df["order_date"].dt.month_name()
df["order_id"] = df["order_id"].astype(int)
df["units_sold"] = df["units_sold"].round().astype(int)

df = df.sort_values("order_date").reset_index(drop=True)
df.to_csv(CLEAN_PATH, index=False)

print("\nFinal shape:", df.shape)
print("Missing values remaining:\n", df.isna().sum())
print(f"\nClean data saved to {CLEAN_PATH}")
