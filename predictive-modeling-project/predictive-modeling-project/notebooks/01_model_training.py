"""
01_model_training.py
---------------------
Trains and evaluates three supervised classification models
(Logistic Regression, Decision Tree, Random Forest) to predict
customer churn, then compares them on accuracy, confusion matrices,
and ROC curves.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)

sns.set_theme(style="whitegrid")
DATA_PATH = "/home/claude/predictive-modeling-project/data/customer_churn.csv"
OUT_DIR = "/home/claude/predictive-modeling-project/outputs"

# ---------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)

le_contract = LabelEncoder()
le_addon = LabelEncoder()
df["contract_type_enc"] = le_contract.fit_transform(df["contract_type"])
df["internet_addon_enc"] = le_addon.fit_transform(df["internet_addon"])

feature_cols = [
    "age", "tenure_months", "contract_type_enc", "monthly_charges",
    "total_charges", "internet_addon_enc", "support_calls",
    "tech_issues_reported", "payment_delay_days"
]
X = df[feature_cols]
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (helps Logistic Regression converge cleanly)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Churn rate — train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

# ---------------------------------------------------------
# 2. Train models
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
}

results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
    }
    print(f"\n{name}")
    print(f"  Accuracy:  {results[name]['accuracy']:.3f}")
    print(f"  Precision: {results[name]['precision']:.3f}")
    print(f"  Recall:    {results[name]['recall']:.3f}")
    print(f"  F1 Score:  {results[name]['f1']:.3f}")
    print(f"  ROC AUC:   {results[name]['auc']:.3f}")

# ---------------------------------------------------------
# 3. Model comparison table
# ---------------------------------------------------------
comparison = pd.DataFrame({
    name: {k: v for k, v in res.items() if k in ["accuracy", "precision", "recall", "f1", "auc"]}
    for name, res in results.items()
}).T.round(3)
comparison.to_csv(f"{OUT_DIR}/model_comparison.csv")
print("\nModel Comparison:\n", comparison)

# ---------------------------------------------------------
# 4. Confusion matrices (all three models)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    ax.set_title(f"{name}\nAccuracy: {res['accuracy']:.2%}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrices.png", dpi=150)
plt.close()
print("Saved confusion_matrices.png")

# ---------------------------------------------------------
# 5. ROC curves (all three models on one plot)
# ---------------------------------------------------------
plt.figure(figsize=(7, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Model Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curves.png", dpi=150)
plt.close()
print("Saved roc_curves.png")

# ---------------------------------------------------------
# 6. Feature importance (Random Forest)
# ---------------------------------------------------------
rf = results["Random Forest"]["model"]
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index, legend=False, palette="crest")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
plt.close()
print("Saved feature_importance.png")

# ---------------------------------------------------------
# 7. Save classification report for the best model
# ---------------------------------------------------------
best_name = comparison["accuracy"].idxmax()
best_report = classification_report(y_test, results[best_name]["y_pred"])
with open(f"{OUT_DIR}/best_model_report.txt", "w") as f:
    f.write(f"Best model by accuracy: {best_name}\n\n")
    f.write(best_report)
print(f"\nBest model: {best_name}")
print(best_report)
