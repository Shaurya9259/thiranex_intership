# Predictive Modeling Using Machine Learning

Internship task: build a model to predict outcomes based on given data.

## Overview
This project builds and compares three supervised classification models to predict **customer churn** (whether a telecom customer will leave) based on their account and usage data.

## Project Structure
```
predictive-modeling-project/
├── data/
│   ├── generate_data.py         # Creates the synthetic churn dataset
│   └── customer_churn.csv       # Dataset (2,000 customers)
├── notebooks/
│   ├── 01_model_training.py     # Full training + evaluation pipeline
│   └── predictive_modeling.ipynb# Notebook version with all outputs
├── outputs/
│   ├── confusion_matrices.png   # Confusion matrix for each model
│   ├── roc_curves.png           # ROC curve comparison
│   ├── feature_importance.png   # Random Forest feature importance
│   ├── model_comparison.csv     # Accuracy/precision/recall/F1/AUC table
│   └── best_model_report.txt    # Classification report for best model
├── requirements.txt
└── README.md
```

## Dataset
2,000 customer records with features: age, tenure, contract type, monthly/total charges, internet add-on, number of support calls, tech issues reported, and payment delay days. Target: `churned` (1 = churned, 0 = retained).

## Approach
1. **Preprocessing** — label-encoded categorical features (`contract_type`, `internet_addon`), scaled numeric features for Logistic Regression
2. **Train/test split** — 80/20, stratified on the target to preserve class balance
3. **Models trained**:
   - Logistic Regression
   - Decision Tree (max depth 6)
   - Random Forest (200 trees, max depth 8)
4. **Evaluation** — accuracy, precision, recall, F1 score, ROC-AUC, confusion matrices, ROC curves

## Results
| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.728 | 0.708 | 0.755 | 0.731 | 0.795 |
| Decision Tree | 0.680 | 0.677 | 0.663 | 0.670 | 0.740 |
| Random Forest | 0.698 | 0.694 | 0.684 | 0.689 | 0.773 |

**Best model: Logistic Regression**, with the strongest ROC-AUC (0.795), suggesting the churn signal in this data is largely linear/additive rather than requiring complex interactions.

Top predictive features (from Random Forest importance): contract type, tenure, and support call frequency — consistent with real-world churn drivers.

## How to Run
```bash
pip install -r requirements.txt
python data/generate_data.py         # (optional) regenerate the dataset
python notebooks/01_model_training.py # train models & generate outputs
```
Or open `notebooks/predictive_modeling.ipynb` in Jupyter to run interactively.

## Tools Used
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Expected Outcome
This project demonstrates the supervised learning workflow end-to-end: feature preparation, training multiple algorithm types, and evaluating/comparing them using accuracy metrics, confusion matrices, and ROC curves rather than relying on a single number.
