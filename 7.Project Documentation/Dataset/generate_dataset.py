"""
Smart Lender - Synthetic Loan Applicant Dataset Generator
-----------------------------------------------------------
Generates a realistic loan-applicant dataset that follows the same schema
as the well-known "Loan Prediction" dataset (Analytics Vidhya / Kaggle):

Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area, Loan_Status

The data is synthetically generated with realistic distributions and a
built-in relationship between the features and Loan_Status so that the
ML models trained on it produce sensible, explainable results.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 614  # match size of the classic Loan Prediction dataset

genders = np.random.choice(["Male", "Female"], size=N, p=[0.8, 0.2]).astype(object)
married = np.random.choice(["Yes", "No"], size=N, p=[0.65, 0.35]).astype(object)
dependents = np.random.choice(["0", "1", "2", "3+"], size=N, p=[0.58, 0.17, 0.17, 0.08]).astype(object)
education = np.random.choice(["Graduate", "Not Graduate"], size=N, p=[0.78, 0.22])
self_employed = np.random.choice(["Yes", "No"], size=N, p=[0.14, 0.86]).astype(object)
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], size=N, p=[0.38, 0.38, 0.24])
loan_term = np.random.choice([360, 300, 240, 180, 120, 84, 60, 36], size=N,
                              p=[0.72, 0.05, 0.05, 0.07, 0.04, 0.03, 0.02, 0.02]).astype(float)

applicant_income = np.round(np.random.lognormal(mean=8.35, sigma=0.55, size=N)).astype(int)
applicant_income = np.clip(applicant_income, 150, 81000)

coapplicant_income = np.where(
    married == "Yes",
    np.round(np.random.lognormal(mean=7.6, sigma=0.9, size=N)).astype(int),
    0
)
coapplicant_income = np.where(np.random.rand(N) < 0.15, 0, coapplicant_income)
coapplicant_income = np.clip(coapplicant_income, 0, 42000)

loan_amount = np.round((applicant_income + coapplicant_income) / np.random.uniform(6, 14, N)).astype(float)
loan_amount = np.clip(loan_amount, 9, 700)

credit_history = np.random.choice([1.0, 0.0], size=N, p=[0.84, 0.16])

# --- Build an underlying "true" approval probability so labels are learnable ---
score = (
    2.4 * credit_history
    + 0.4 * (education == "Graduate")
    + 0.3 * (property_area == "Semiurban")
    - 0.2 * (property_area == "Rural")
    + 0.5 * np.tanh((applicant_income + coapplicant_income - loan_amount * 100) / 20000)
    - 0.6 * (loan_amount > 250)
    + 0.25 * (married == "Yes")
    - 0.35 * (dependents == "3+")
    - 0.95
    + np.random.normal(0, 0.75, N)
)

prob_approve = 1 / (1 + np.exp(-score))
loan_status = np.where(prob_approve > 0.5, "Y", "N")

loan_ids = [f"LP{str(100000 + i)[1:]}" for i in range(1, N + 1)]

df = pd.DataFrame({
    "Loan_ID": loan_ids,
    "Gender": genders,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
    "Loan_Status": loan_status,
})

# --- Introduce a handful of realistic missing values (mimics real-world data) ---
rng = np.random.default_rng(7)


def sprinkle_nan(column, frac):
    idx = rng.choice(N, size=int(N * frac), replace=False)
    df.loc[idx, column] = np.nan


sprinkle_nan("Gender", 0.02)
sprinkle_nan("Married", 0.008)
sprinkle_nan("Dependents", 0.025)
sprinkle_nan("Self_Employed", 0.05)
sprinkle_nan("LoanAmount", 0.035)
sprinkle_nan("Loan_Amount_Term", 0.023)
sprinkle_nan("Credit_History", 0.08)

df.to_csv("loan_dataset.csv", index=False)
print(f"Generated dataset with {len(df)} rows -> loan_dataset.csv")
print(df["Loan_Status"].value_counts())
print(df.isna().sum())
