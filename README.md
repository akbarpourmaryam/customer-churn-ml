# Customer Churn Prediction

End-to-end classical machine learning project for predicting telecom customer churn. It covers validation, cleaning, feature engineering, train/validation/test splitting, cross-validated tuning, model comparison, MLflow tracking, model versioning, interpretation, testing, and basic API serving.

## Final result

XGBoost was selected by validation PR-AUC after comparison with a majority-class baseline, logistic regression, and random forest.

| Final test metric | Score |
|---|---:|
| Accuracy | 0.7764 |
| Balanced accuracy | 0.7556 |
| Precision | 0.5624 |
| Recall | 0.7112 |
| F1 | 0.6281 |
| ROC-AUC | 0.8470 |
| PR-AUC | 0.6655 |

The selected threshold is `0.60`. The model identified 266 churners, missed 108, and incorrectly flagged 207 non-churners on the untouched test set.

Accuracy alone is misleading: the dummy model reaches 73.5% accuracy by predicting that nobody churns, while churn recall and F1 are both zero.

![Validation model comparison](reports/model_comparison.png)

## Workflow

```text
Kaggle CSV → validation → cleaning → feature engineering
→ stratified 60/20/20 split → 5-fold tuning on train
→ threshold selection and comparison on validation
→ best model refit on train + validation → one final test evaluation
→ versioned model + MLflow runs + report + API
```

Engineered inside the saved pipeline:

- `TotalServices`: number of subscribed services
- `TenureGroup`: customer lifecycle band

Class imbalance is handled through stratification, class-weight tuning, XGBoost positive-class weighting, PR-AUC selection, and threshold tuning.

## Models

- Dummy majority-class baseline
- Logistic Regression
- Random Forest
- XGBoost

All settings and search grids live in [`configs/training.yaml`](configs/training.yaml).

## Setup and run

Python 3.11 or 3.12 is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python generate_eda_report.py
python train_model.py
python -m pytest
```

## MLflow

Training logs parameters, cross-validation and validation metrics, threshold tables, and serialized pipelines.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Prediction API

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "customerID":"demo-001", "gender":"Female", "SeniorCitizen":0,
    "Partner":"Yes", "Dependents":"No", "tenure":3,
    "PhoneService":"Yes", "MultipleLines":"No",
    "InternetService":"Fiber optic", "OnlineSecurity":"No",
    "OnlineBackup":"No", "DeviceProtection":"No", "TechSupport":"No",
    "StreamingTV":"Yes", "StreamingMovies":"Yes",
    "Contract":"Month-to-month", "PaperlessBilling":"Yes",
    "PaymentMethod":"Electronic check", "MonthlyCharges":89.10,
    "TotalCharges":267.30
  }'
```

## Structure

```text
configs/                 Training and search configuration
data/                    Source dataset
docs/                    Methodology and presentation guidance
notebooks/               Exploratory notebook
reports/                 Tracked EDA and evaluation outputs
src/                     Validation, features, training, evaluation, API
tests/                   Unit and API tests
train_model.py           End-to-end training entry point
generate_eda_report.py   Reproducible EDA entry point
pyproject.toml           Package and dependency definition
```

Generated model and MLflow state are ignored by Git: `artifacts/`, `mlflow.db`, and `mlruns/`.

## Reports

- [EDA report](reports/eda_report.md)
- [Model evaluation report](reports/evaluation_report.md)
- [Evaluation explanation guide](docs/model_evaluation.md)
- [Complete beginner's project guide](docs/complete_project_guide.md)
- [Project architecture](docs/architecture.md)
- [Interview and stakeholder questions](docs/interview_and_stakeholder_qa.md)
- [Step-by-step running guide](docs/how_to_run.md)
- [EDA notebook](notebooks/eda.ipynb)

## Limitations

- The dataset is static and relatively small.
- Predictive associations do not establish causality.
- The threshold optimizes F1 because business retention costs were not supplied.
- Tree importance can be biased; permutation importance or SHAP is a useful extension.
- Real deployment would need drift monitoring, authentication, logging, and a remote registry.
