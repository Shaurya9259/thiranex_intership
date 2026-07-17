# EDA Findings Report — Student Performance

## Dataset Overview
- **Records:** 600 students
- **Features:** 10 (numeric + categorical)
- **Missing values found:** attendance_pct, sleep_hours, previous_score (filled with median for analysis)

## Statistical Summary
See `statistical_summary.csv` for full descriptive statistics (mean, std, quartiles) of all numeric variables.

- Average exam score: **85.6** (std: 10.7)
- Average study time: **3.4 hours/day**
- Average attendance: **81.6%**

## Key Influencing Factors (Correlation with Exam Score)
Ranked by strength of correlation:
1. **Study Hours Per Day**: r = 0.75
2. **Previous Score**: r = 0.25
3. **Social Media Hours**: r = -0.25
4. **Attendance Pct**: r = 0.14
5. **Sleep Hours**: r = 0.11
6. **Extracurricular Hours**: r = -0.07

## Key Insights
1. **Study Hours Per Day** shows the strongest relationship with exam performance (r = 0.75), suggesting it is the most influential factor among those measured.
2. **Previous Score** and **Social Media Hours** are also meaningfully correlated with outcomes.
3. Social media usage shows a **negative** correlation with exam score — more screen time is associated with lower performance.
4. Exam scores are broadly similar across gender and study group in this dataset, indicating these categorical factors are not strong differentiators here (see `categorical_breakdown.png`).

## Recommendations
- Interventions aimed at increasing study hours per day are likely to have the largest impact on outcomes, based on observed correlation strength.
- Correlation does not imply causation — these findings identify strong *associations* worth testing further (e.g. via controlled study or predictive modeling).

## Files in This Report
- `statistical_summary.csv` — full descriptive statistics
- `distributions.png` — histogram of every numeric variable
- `correlation_heatmap.png` — full correlation matrix
- `exam_score_correlations.csv` — correlation of each factor with exam_score
- `top_factor_scatterplots.png` — regression plots for top 4 factors
- `categorical_breakdown.png` — exam score by gender and study group
