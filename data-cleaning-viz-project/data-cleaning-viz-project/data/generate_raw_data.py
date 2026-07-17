"""
Generates a realistic 'messy' raw retail sales dataset with the kinds
of problems real-world data actually has: missing values, duplicates,
outliers, inconsistent text casing, and mixed date formats.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

n = 1200
categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports", "Toys"]
regions = ["North", "South", "East", "West"]
payment_modes = ["Credit Card", "UPI", "Cash", "Debit Card", "Net Banking"]

dates = pd.date_range("2025-01-01", "2025-12-31", periods=n)

df = pd.DataFrame({
    "order_id": np.arange(1001, 1001 + n),
    "order_date": dates,
    "category": np.random.choice(categories, n),
    "region": np.random.choice(regions, n),
    "payment_mode": np.random.choice(payment_modes, n),
    "units_sold": np.random.randint(1, 25, n),
    "unit_price": np.round(np.random.uniform(5, 500, n), 2),
})
df["revenue"] = np.round(df["units_sold"] * df["unit_price"], 2)

# --- Inject messiness ---

# 1. Missing values (unit_price, region, payment_mode)
for col, frac in [("unit_price", 0.06), ("region", 0.04), ("payment_mode", 0.03), ("units_sold", 0.02)]:
    idx = np.random.choice(df.index, int(len(df) * frac), replace=False)
    df.loc[idx, col] = np.nan

# 2. Outliers in units_sold and unit_price (data entry errors)
outlier_idx = np.random.choice(df.index, 15, replace=False)
df.loc[outlier_idx[:8], "units_sold"] = np.random.randint(500, 2000, 8)
df.loc[outlier_idx[8:], "unit_price"] = np.random.uniform(9000, 20000, 7)

# 3. Duplicate rows
dupes = df.sample(30, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

# 4. Inconsistent casing / whitespace in text columns
def mess_text(x):
    if pd.isna(x):
        return x
    choice = np.random.rand()
    if choice < 0.2:
        return x.upper()
    elif choice < 0.4:
        return x.lower()
    elif choice < 0.5:
        return f"  {x}  "
    return x

df["category"] = df["category"].apply(mess_text)
df["region"] = df["region"].apply(mess_text)

# 5. Recompute revenue as raw (so it may not match after cleaning - intentional, we'll recompute in cleaning step)
df["revenue"] = np.round(df["units_sold"] * df["unit_price"], 2)

# 6. Mixed date formats (string mess) - convert some to different string formats
date_strs = []
for i, d in enumerate(df["order_date"]):
    if pd.isna(d):
        date_strs.append(np.nan)
        continue
    r = i % 4
    if r == 0:
        date_strs.append(d.strftime("%Y-%m-%d"))
    elif r == 1:
        date_strs.append(d.strftime("%d/%m/%Y"))
    elif r == 2:
        date_strs.append(d.strftime("%d-%b-%Y"))
    else:
        date_strs.append(d.strftime("%m/%d/%Y"))
df["order_date"] = date_strs

df = df.sample(frac=1, random_state=7).reset_index(drop=True)  # shuffle rows
df.to_csv("/home/claude/data-cleaning-viz-project/data/raw_sales_data.csv", index=False)
print("Raw dataset created:", df.shape)
