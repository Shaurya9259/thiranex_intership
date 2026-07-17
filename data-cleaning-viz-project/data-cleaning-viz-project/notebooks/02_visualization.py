"""
02_visualization.py
--------------------
Loads the cleaned dataset and produces a set of visual insights,
saved individually and as a combined dashboard image.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
CLEAN_PATH = "/home/claude/data-cleaning-viz-project/data/clean_sales_data.csv"
OUT_DIR = "/home/claude/data-cleaning-viz-project/outputs"

df = pd.read_csv(CLEAN_PATH, parse_dates=["order_date"])

month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
df["month"] = pd.Categorical(df["month"], categories=month_order, ordered=True)

# ---------------------------------------------------------
# Combined dashboard: 2x2 grid of key insights
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Retail Sales — Cleaned Data Dashboard", fontsize=16, fontweight="bold")

# 1. Revenue by category
cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
sns.barplot(x=cat_rev.values, y=cat_rev.index, ax=axes[0, 0], palette="viridis", hue=cat_rev.index, legend=False)
axes[0, 0].set_title("Total Revenue by Category")
axes[0, 0].set_xlabel("Revenue")
axes[0, 0].set_ylabel("")

# 2. Monthly revenue trend
monthly_rev = df.groupby("month", observed=True)["revenue"].sum()
axes[0, 1].plot(monthly_rev.index, monthly_rev.values, marker="o", color="#2c7fb8")
axes[0, 1].set_title("Monthly Revenue Trend")
axes[0, 1].set_ylabel("Revenue")
axes[0, 1].tick_params(axis="x", rotation=45)

# 3. Revenue share by region
region_rev = df.groupby("region")["revenue"].sum()
axes[1, 0].pie(region_rev.values, labels=region_rev.index, autopct="%1.1f%%",
               colors=sns.color_palette("pastel"))
axes[1, 0].set_title("Revenue Share by Region")

# 4. Payment mode distribution
sns.countplot(data=df, y="payment_mode", order=df["payment_mode"].value_counts().index,
              ax=axes[1, 1], palette="mako", hue="payment_mode", legend=False)
axes[1, 1].set_title("Orders by Payment Mode")
axes[1, 1].set_xlabel("Number of Orders")
axes[1, 1].set_ylabel("")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{OUT_DIR}/dashboard.png", dpi=150)
plt.close()
print("Saved dashboard.png")

# ---------------------------------------------------------
# Individual chart: unit_price distribution (before/after outlier context)
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["unit_price"], bins=30, kde=True, color="#e07a5f")
plt.title("Distribution of Unit Price (After Outlier Capping)")
plt.xlabel("Unit Price")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/unit_price_distribution.png", dpi=150)
plt.close()
print("Saved unit_price_distribution.png")

# ---------------------------------------------------------
# Individual chart: correlation heatmap
# ---------------------------------------------------------
plt.figure(figsize=(6, 5))
corr = df[["units_sold", "unit_price", "revenue"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Numeric Features")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=150)
plt.close()
print("Saved correlation_heatmap.png")

# ---------------------------------------------------------
# Summary stats table (for README / report)
# ---------------------------------------------------------
summary = df.groupby("category").agg(
    total_orders=("order_id", "count"),
    total_units=("units_sold", "sum"),
    total_revenue=("revenue", "sum"),
    avg_order_value=("revenue", "mean"),
).round(2).sort_values("total_revenue", ascending=False)
summary.to_csv(f"{OUT_DIR}/category_summary.csv")
print("\nCategory Summary:\n", summary)
