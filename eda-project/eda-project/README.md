# Exploratory Data Analysis (EDA) Project

Internship task: analyze a dataset to uncover patterns and trends.

## Overview
This project performs a full EDA on a student exam performance dataset (600 students), using statistical summaries and visualizations to identify which factors most strongly influence exam outcomes.

## Project Structure
```
eda-project/
├── data/
│   ├── generate_data.py          # Creates the synthetic dataset
│   └── student_performance.csv   # Dataset (600 students)
├── notebooks/
│   ├── 01_eda_analysis.py        # Full EDA pipeline
│   └── eda_analysis.ipynb        # Notebook version with all outputs
├── outputs/
│   ├── statistical_summary.csv
│   ├── distributions.png
│   ├── correlation_heatmap.png
│   ├── exam_score_correlations.csv
│   ├── top_factor_scatterplots.png
│   ├── categorical_breakdown.png
│   └── EDA_REPORT.md             # Structured findings report
├── requirements.txt
└── README.md
```

## Dataset
600 student records with: study hours/day, attendance %, sleep hours, social media hours, extracurricular hours, previous exam score, gender, study group, and the target — final exam score.

## Approach
1. **Data quality check** — identified missing values in attendance, sleep, and previous score; filled with median
2. **Statistical summary** — mean, std, quartiles for every numeric variable
3. **Distribution analysis** — histograms for all numeric variables
4. **Correlation analysis** — full correlation matrix + ranked correlation with exam score
5. **Key factor visualization** — regression scatter plots for the top 4 correlated factors
6. **Categorical breakdown** — exam score across gender and study group

## Key Findings
| Factor | Correlation with Exam Score |
|---|---|
| Study hours/day | **+0.75** (strongest) |
| Previous score | +0.25 |
| Social media hours | **−0.25** |
| Attendance % | +0.14 |
| Sleep hours | +0.11 |
| Extracurricular hours | −0.07 |

- **Study hours per day is the dominant factor** — far stronger than any other variable measured.
- Social media usage is the clearest negative influence on performance.
- Gender and study group show no meaningful difference in outcomes in this dataset.
- Full narrative write-up in `outputs/EDA_REPORT.md`.

## How to Run
```bash
pip install -r requirements.txt
python data/generate_data.py           # (optional) regenerate the dataset
python notebooks/01_eda_analysis.py    # run full EDA and generate outputs
```
Or open `notebooks/eda_analysis.ipynb` in Jupyter to run interactively.

## Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn

## Expected Outcome
This project demonstrates core EDA skills: profiling a dataset, checking data quality, summarizing distributions, quantifying relationships between variables, and communicating findings as a structured, decision-useful report rather than just a pile of charts.
