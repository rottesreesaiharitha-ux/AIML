"""
Smart Lender - Flask Web Application
--------------------------------------
Serves a loan-eligibility prediction web app backed by a trained
XGBoost model (selected as the best of Decision Tree / Random Forest /
KNN / XGBoost during offline model training - see model_training/).
"""

import os

import joblib
import numpy as np
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

# --------------------------------------------------------------------------
# Load model + preprocessing artifacts once at startup
# --------------------------------------------------------------------------
model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
target_encoder = joblib.load(os.path.join(MODEL_DIR, "target_encoder.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))


def build_feature_vector(form):
    """Convert raw form input into the model's expected feature vector."""
    gender = label_encoders["Gender"].transform([form["gender"]])[0]
    married = label_encoders["Married"].transform([form["married"]])[0]
    education = label_encoders["Education"].transform([form["education"]])[0]
    self_employed = label_encoders["Self_Employed"].transform([form["self_employed"]])[0]
    property_area = label_encoders["Property_Area"].transform([form["property_area"]])[0]

    dependents = form["dependents"].replace("3+", "3")
    dependents = int(dependents)

    applicant_income = float(form["applicant_income"])
    coapplicant_income = float(form["coapplicant_income"])
    loan_amount = float(form["loan_amount"])
    loan_term = float(form["loan_term"])
    credit_history = float(form["credit_history"])

    total_income = applicant_income + coapplicant_income
    total_income_log = np.log1p(total_income)
    loan_amount_log = np.log1p(loan_amount)

    row = {
        "Gender": gender,
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
        "TotalIncome_log": total_income_log,
        "LoanAmount_log": loan_amount_log,
    }
    return np.array([[row[c] for c in feature_cols]])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    try:
        X = build_feature_vector(request.form)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        status = target_encoder.inverse_transform([pred])[0]
        confidence = round(float(np.max(proba)) * 100, 1)
        approved = status == "Y"
        risk_level = "Low" if confidence >= 80 else ("Medium" if confidence >= 60 else "High")

        return render_template(
            "result.html",
            approved=approved,
            confidence=confidence,
            risk_level=risk_level,
            applicant_name=request.form.get("applicant_name", "Applicant"),
        )
    except Exception as exc:  # noqa: BLE001
        return render_template("predict.html", error=str(exc))


@app.route("/dashboard")
def dashboard():
    import json
    import csv

    results_dir = os.path.join(BASE_DIR, "..", "results")
    comparison = []
    comp_path = os.path.join(results_dir, "model_comparison.csv")
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            comparison = list(csv.DictReader(f))

    best_info = {}
    info_path = os.path.join(results_dir, "best_model_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            best_info = json.load(f)

    return render_template("dashboard.html", comparison=comparison, best_info=best_info)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
