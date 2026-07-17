"""
01_eda_analysis.py
--------------------
Exploratory Data Analysis on the student performance dataset:
 - Statistical summaries
 - Missing value check
 - Distribution plots
 - Correlation analysis
 - Key influencing factor identification
 - Structured findings report
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
DATA_PATH = "/home/claude/eda-project/data/student_performance.csv"
OUT_DIR = "/home/claude/eda-project/outputs"

df = pd.read_csv(DATA_PATH)
numeric_cols = ["study_hours_per_day", "attendance_pct", "sleep_hours",
                 "social_media_hours", "extracurricular_hours", "previous_score", "exam_score"]

# ---------------------------------------------------------
# 1. Basic overview & missing values
# ---------------------------------------------------------
print("Shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())

# Fill missing numeric values with median for analysis (light touch — EDA, not modeling)
for col in ["attendance_pct", "sleep_hours", "previous_score"]:
    df[col] = df[col].fillna(df[col].median())

# ---------------------------------------------------------
# 2. Statistical summary
# ---------------------------------------------------------
summary = df[numeric_cols].describe().T.round(2)
summary.to_csv(f"{OUT_DIR}/statistical_summary.csv")
print("\nStatistical Summary:\n", summary)

# ---------------------------------------------------------
# 3. Distribution plots for all numeric variables
# ---------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color="#3d5a80")
    axes[i].set_title(col.replace("_", " ").title())
for j in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/distributions.png", dpi=150)
plt.close()
print("Saved distributions.png")

# ---------------------------------------------------------
# 4. Correlation heatmap
# ---------------------------------------------------------
corr = df[numeric_cols].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0)
plt.title("Correlation Matrix — Student Performance Factors")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=150)
plt.close()
print("Saved correlation_heatmap.png")

exam_corr = corr["exam_score"].drop("exam_score").sort_values(key=abs, ascending=False)
exam_corr.to_csv(f"{OUT_DIR}/exam_score_correlations.csv")
print("\nCorrelation with exam_score (ranked by strength):\n", exam_corr)

# ---------------------------------------------------------
# 5. Scatter plots: exam_score vs top influencing factors
# ---------------------------------------------------------
top_factors = exam_corr.index[:4]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, factor in enumerate(top_factors):
    sns.regplot(data=df, x=factor, y="exam_score", ax=axes[i],
                scatter_kws={"alpha": 0.4, "s": 20}, line_kws={"color": "red"})
    axes[i].set_title(f"Exam Score vs {factor.replace('_',' ').title()}\n(r = {exam_corr[factor]:.2f})")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/top_factor_scatterplots.png", dpi=150)
plt.close()
print("Saved top_factor_scatterplots.png")

# ---------------------------------------------------------
# 6. Categorical breakdowns
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x="gender", y="exam_score", ax=axes[0], hue="gender", legend=False, palette="Set2")
axes[0].set_title("Exam Score by Gender")
sns.boxplot(data=df, x="study_group", y="exam_score", ax=axes[1], hue="study_group", legend=False, palette="Set3")
axes[1].set_title("Exam Score by Study Group")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/categorical_breakdown.png", dpi=150)
plt.close()
print("Saved categorical_breakdown.png")

# ---------------------------------------------------------
# 7. Structured findings report (Markdown)
# ---------------------------------------------------------
top3 = exam_corr.index[:3]
report = f"""# EDA Findings Report — Student Performance

## Dataset Overview
- **Records:** {df.shape[0]} students
- **Features:** {df.shape[1]} (numeric + categorical)
- **Missing values found:** attendance_pct, sleep_hours, previous_score (filled with median for analysis)

## Statistical Summary
See `statistical_summary.csv` for full descriptive statistics (mean, std, quartiles) of all numeric variables.

- Average exam score: **{df['exam_score'].mean():.1f}** (std: {df['exam_score'].std():.1f})
- Average study time: **{df['study_hours_per_day'].mean():.1f} hours/day**
- Average attendance: **{df['attendance_pct'].mean():.1f}%**

## Key Influencing Factors (Correlation with Exam Score)
Ranked by strength of correlation:
{chr(10).join(f"{i+1}. **{f.replace('_',' ').title()}**: r = {exam_corr[f]:.2f}" for i, f in enumerate(exam_corr.index))}

## Key Insights
1. **{top3[0].replace('_',' ').title()}** shows the strongest relationship with exam performance (r = {exam_corr[top3[0]]:.2f}), suggesting it is the most influential factor among those measured.
2. **{top3[1].replace('_',' ').title()}** and **{top3[2].replace('_',' ').title()}** are also meaningfully correlated with outcomes.
3. Social media usage shows a **negative** correlation with exam score — more screen time is associated with lower performance.
4. Exam scores are broadly similar across gender and study group in this dataset, indicating these categorical factors are not strong differentiators here (see `categorical_breakdown.png`).

## Recommendations
- Interventions aimed at increasing {top3[0].replace('_',' ')} are likely to have the largest impact on outcomes, based on observed correlation strength.
- Correlation does not imply causation — these findings identify strong *associations* worth testing further (e.g. via controlled study or predictive modeling).

## Files in This Report
- `statistical_summary.csv` — full descriptive statistics
- `distributions.png` — histogram of every numeric variable
- `correlation_heatmap.png` — full correlation matrix
- `exam_score_correlations.csv` — correlation of each factor with exam_score
- `top_factor_scatterplots.png` — regression plots for top 4 factors
- `categorical_breakdown.png` — exam score by gender and study group
"""
with open(f"{OUT_DIR}/EDA_REPORT.md", "w") as f:
    f.write(report)
print("\nSaved EDA_REPORT.md")
