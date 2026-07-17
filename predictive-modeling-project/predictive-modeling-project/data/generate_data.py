"""
Generates a realistic customer churn dataset for a supervised
classification task (predict whether a customer will churn).
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

tenure_months = np.random.randint(1, 72, n)
monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
total_charges = np.round(monthly_charges * tenure_months * np.random.uniform(0.85, 1.05, n), 2)
contract_type = np.random.choice(["Month-to-Month", "One Year", "Two Year"], n, p=[0.55, 0.25, 0.20])
support_calls = np.random.poisson(1.5, n)
internet_addon = np.random.choice(["Yes", "No"], n, p=[0.65, 0.35])
age = np.random.randint(18, 75, n)
payment_delay_days = np.random.exponential(3, n).astype(int)
tech_issues_reported = np.random.poisson(0.8, n)

# Build churn probability from a realistic underlying logic + noise
churn_score = (
    -0.09 * tenure_months
    + 0.022 * monthly_charges
    + (contract_type == "Month-to-Month") * 2.2
    + 0.55 * support_calls
    + 0.45 * tech_issues_reported
    + 0.15 * payment_delay_days
    - 0.02 * age
    + np.random.normal(0, 1.0, n)
)
churn_prob = 1 / (1 + np.exp(-(churn_score - churn_score.mean()) / (churn_score.std() * 0.6)))
churned = (churn_prob > np.random.uniform(0, 1, n)).astype(int)

df = pd.DataFrame({
    "customer_id": np.arange(10001, 10001 + n),
    "age": age,
    "tenure_months": tenure_months,
    "contract_type": contract_type,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "internet_addon": internet_addon,
    "support_calls": support_calls,
    "tech_issues_reported": tech_issues_reported,
    "payment_delay_days": payment_delay_days,
    "churned": churned,
})

df.to_csv("/home/claude/predictive-modeling-project/data/customer_churn.csv", index=False)
print("Dataset created:", df.shape)
print("Churn rate:", df["churned"].mean().round(3))
