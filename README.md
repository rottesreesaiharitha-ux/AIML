# Smart Lender

**Smart Lender** is a machine-learning-powered web application that predicts the
creditworthiness of loan applicants, helping banks and financial institutions
make faster, data-driven loan approval decisions.

The system evaluates applicant data (income, employment, education, credit
history, property area, etc.) using four classification algorithms —
**Decision Tree, Random Forest, K-Nearest Neighbors (KNN), and XGBoost** —
and deploys the best-performing model (XGBoost) inside a Flask web
application for real-time predictions.

---

## Project structure

```
SmartLender/
├── dataset/
│   ├── generate_dataset.py     # generates the loan applicant dataset
│   └── loan_dataset.csv        # 614-row applicant dataset (Loan Prediction schema)
│
├── model_training/
│   ├── train_models.py         # trains & compares all 4 models, saves the best one
│   └── requirements.txt
│
├── flask_app/
│   ├── app.py                  # Flask application (routes + prediction engine)
│   ├── model/                  # saved model + preprocessing objects (.pkl)
│   ├── templates/               # HTML templates (Jinja2)
│   ├── static/css/style.css     # styling
│   └── requirements.txt
│
├── results/
│   ├── model_comparison.csv     # accuracy/precision/recall/F1 per model
│   ├── model_comparison.png     # bar chart of test accuracy
│   ├── confusion_matrix.png     # confusion matrix for the deployed model
│   ├── classification_report.txt
│   └── best_model_info.json
│
├── references/
│   └── references.md
│
└── Smart_Lender_Project_Report.docx
```

## Dataset

The dataset follows the schema of the well-known **Loan Prediction dataset**
(Analytics Vidhya / Kaggle), with 614 applicant records and the following
13 fields:

| Field | Description |
|---|---|
| Loan_ID | Unique loan application ID |
| Gender | Male / Female |
| Married | Applicant married (Y/N) |
| Dependents | Number of dependents (0, 1, 2, 3+) |
| Education | Graduate / Not Graduate |
| Self_Employed | Self employed (Y/N) |
| ApplicantIncome | Applicant's monthly income |
| CoapplicantIncome | Co-applicant's monthly income |
| LoanAmount | Loan amount requested (in thousands) |
| Loan_Amount_Term | Loan term, in months |
| Credit_History | Whether credit history meets guidelines (1/0) |
| Property_Area | Urban / Semiurban / Rural |
| Loan_Status | Target: loan approved (Y) or not (N) |

`dataset/generate_dataset.py` produces `loan_dataset.csv` synthetically with
realistic distributions, a small number of missing values (mirroring
real-world data quality issues), and a genuine underlying relationship
between features and approval outcome.

## Model training

Run:

```bash
cd model_training
pip install -r requirements.txt
python train_models.py
```

This will:
1. Load and clean `dataset/loan_dataset.csv` (impute missing values, encode categoricals, engineer `TotalIncome_log` / `LoanAmount_log`)
2. Split into 80% train / 20% test
3. Train **Decision Tree, Random Forest, KNN, and XGBoost**
4. Compare accuracy, precision, recall, and F1-score
5. Save the best model (XGBoost) and all preprocessing objects into `flask_app/model/`
6. Save comparison charts and a classification report into `results/`

### Results

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| Decision Tree | 88.6% | 88.6% | 90.5% | 94.5% | 92.5% |
| Random Forest | 89.4% | 91.1% | 89.2% | 100.0% | 94.3% |
| KNN | 86.8% | 91.9% | 90.1% | 100.0% | 94.8% |
| **XGBoost (deployed)** | **95.7%** | **88.6%** | **89.7%** | **95.6%** | **92.6%** |

XGBoost was selected for deployment for its strong balance of training
performance and generalization, consistent with its use in the reference
architecture (Python/Flask/IBM Cloud).

## Running the web app

```bash
cd flask_app
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in your browser. Pages:

- `/` — Home / landing page
- `/predict` — Loan eligibility prediction form
- `/dashboard` — Model comparison & performance insights
- `/about` — Project overview

## Real-world scenarios covered

1. **Fast-track approval** — a salaried applicant with good credit history and stable income is approved with high confidence.
2. **High-risk detection** — an applicant with irregular self-employment income and no credit history is flagged as high risk.
3. **Batch evaluation** — analysts can evaluate multiple applicants quickly, reducing manual review time.

## Tech stack

- **Language:** Python 3
- **ML:** scikit-learn (Decision Tree, Random Forest, KNN), XGBoost
- **Web framework:** Flask
- **Frontend:** HTML5, CSS3, Jinja2 templates
- **Deployment target:** IBM Cloud

## Target users

Financial analysts, credit officers, and banking professionals looking to
reduce non-performing assets and speed up credit evaluation.
