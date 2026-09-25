"""
Smart Lender - Model Training Pipeline
----------------------------------------
Loads the loan applicant dataset, preprocesses it, trains four
classification models (Decision Tree, Random Forest, KNN, XGBoost),
compares their performance, and saves the best model (+ preprocessing
objects) for use by the Flask web application.
"""

import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, precision_score,
                              recall_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_PATH = "../dataset/loan_dataset.csv"
RESULTS_DIR = "../results"
MODEL_DIR = "../flask_app/model"

import os
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Loaded dataset:", df.shape)

# --------------------------------------------------------------------------
# 2. Data cleaning / missing value imputation
# --------------------------------------------------------------------------
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

df["Dependents"] = df["Dependents"].replace("3+", "3").astype(int)

# --------------------------------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------------------------------
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["LoanAmount_log"] = np.log1p(df["LoanAmount"])
df["TotalIncome_log"] = np.log1p(df["TotalIncome"])

label_encoders = {}
for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

target_le = LabelEncoder()
df["Loan_Status"] = target_le.fit_transform(df["Loan_Status"])  # N=0, Y=1

feature_cols = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area", "TotalIncome_log", "LoanAmount_log",
]

X = df[feature_cols]
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------------------------------
# 4. Train models
# --------------------------------------------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=9),
    "XGBoost": XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        use_label_encoder=False, eval_metric="logloss", random_state=42
    ),
}

results = []
trained_models = {}

for name, model in models.items():
    if name == "KNN":
        model.fit(X_train_scaled, y_train)
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    prec = precision_score(y_test, test_pred)
    rec = recall_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)

    results.append({
        "Model": name,
        "Train_Accuracy": round(train_acc * 100, 1),
        "Test_Accuracy": round(test_acc * 100, 1),
        "Precision": round(prec * 100, 1),
        "Recall": round(rec * 100, 1),
        "F1_Score": round(f1 * 100, 1),
    })
    trained_models[name] = model
    print(f"{name:15s} | Train Acc: {train_acc*100:5.1f}% | Test Acc: {test_acc*100:5.1f}%")

results_df = pd.DataFrame(results).sort_values("Test_Accuracy", ascending=False)
results_df.to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)
print("\nModel comparison:\n", results_df)

# --------------------------------------------------------------------------
# 5. Select and save the best model (XGBoost, per project spec)
# --------------------------------------------------------------------------
best_model_name = "XGBoost"
best_model = trained_models[best_model_name]

joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(label_encoders, f"{MODEL_DIR}/label_encoders.pkl")
joblib.dump(target_le, f"{MODEL_DIR}/target_encoder.pkl")
joblib.dump(feature_cols, f"{MODEL_DIR}/feature_cols.pkl")

with open(f"{RESULTS_DIR}/best_model_info.json", "w") as f:
    json.dump({
        "best_model": best_model_name,
        "train_accuracy": float(results_df[results_df.Model == best_model_name].Train_Accuracy.values[0]),
        "test_accuracy": float(results_df[results_df.Model == best_model_name].Test_Accuracy.values[0]),
    }, f, indent=2)

# --------------------------------------------------------------------------
# 6. Plots: model comparison + confusion matrix
# --------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="Model", y="Test_Accuracy", palette="viridis")
plt.title("Model Comparison - Test Accuracy (%)")
plt.ylabel("Test Accuracy (%)")
plt.ylim(0, 100)
for i, v in enumerate(results_df["Test_Accuracy"]):
    plt.text(i, v + 1, f"{v}%", ha="center")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/model_comparison.png", dpi=150)
plt.close()

best_test_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_test_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)
plt.close()

report = classification_report(y_test, best_test_pred, target_names=["Rejected", "Approved"])
with open(f"{RESULTS_DIR}/classification_report.txt", "w") as f:
    f.write(f"Best Model: {best_model_name}\n\n")
    f.write(report)

print("\nTraining complete. Model + results saved.")
