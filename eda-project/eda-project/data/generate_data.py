"""
Generates a realistic student exam performance dataset for
exploratory data analysis — with genuine underlying relationships
between study habits and exam scores, plus natural noise.
"""
import numpy as np
import pandas as pd

np.random.seed(7)
n = 600

study_hours_per_day = np.round(np.clip(np.random.normal(3.5, 1.5, n), 0, 10), 1)
attendance_pct = np.round(np.clip(np.random.normal(82, 12, n), 40, 100), 1)
sleep_hours = np.round(np.clip(np.random.normal(6.5, 1.3, n), 3, 10), 1)
social_media_hours = np.round(np.clip(np.random.normal(2.8, 1.4, n), 0, 8), 1)
previous_score = np.round(np.clip(np.random.normal(65, 15, n), 20, 100), 1)
extracurricular_hours = np.round(np.clip(np.random.exponential(1.5, n), 0, 6), 1)
gender = np.random.choice(["Male", "Female"], n)
study_group = np.random.choice(["Group A", "Group B", "Group C"], n)

# Exam score driven by a believable combination of factors + noise
exam_score = (
    35
    + 6.2 * study_hours_per_day
    + 0.18 * attendance_pct
    + 1.1 * sleep_hours
    - 1.8 * social_media_hours
    + 0.22 * previous_score
    - 0.5 * extracurricular_hours
    + np.random.normal(0, 6, n)
)
exam_score = np.clip(exam_score, 0, 100).round(1)

df = pd.DataFrame({
    "student_id": np.arange(1, n + 1),
    "gender": gender,
    "study_group": study_group,
    "study_hours_per_day": study_hours_per_day,
    "attendance_pct": attendance_pct,
    "sleep_hours": sleep_hours,
    "social_media_hours": social_media_hours,
    "extracurricular_hours": extracurricular_hours,
    "previous_score": previous_score,
    "exam_score": exam_score,
})

# Inject a small amount of realistic missingness (EDA should surface this)
for col, frac in [("sleep_hours", 0.03), ("attendance_pct", 0.02), ("previous_score", 0.02)]:
    idx = np.random.choice(df.index, int(len(df) * frac), replace=False)
    df.loc[idx, col] = np.nan

df.to_csv("/home/claude/eda-project/data/student_performance.csv", index=False)
print("Dataset created:", df.shape)
print(df.isna().sum())
