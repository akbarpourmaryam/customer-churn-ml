# How to run the project

All commands assume the terminal is in the repository root:

```bash
cd /Users/maryamakbarpour/Projects/Kaggle/customer-churn-ml
```

## 1. Activate the environment

```bash
source .venv/bin/activate
```

The prompt should contain one `(.venv)`. If it shows `((.venv))`, the environment was activated twice; this is harmless.

## 2. Install or refresh dependencies

```bash
python -m pip install -e '.[dev]'
```

This installs the project in editable mode, so source-code changes are immediately available.

## 3. Run tests

```bash
python -m pytest
```

Expected result:

```text
16 passed
```

The current FastAPI/Starlette test stack may print one third-party deprecation warning about `httpx`. It does not indicate a failed test or broken API.

## 4. Generate EDA

```bash
python generate_eda_report.py
```

Outputs:

- `reports/eda_report.md`
- `reports/eda_overview.png`

## 5. Train the latest complete model

Stop MLflow and the API first if they are using the terminal, then run:

```bash
python train_model.py
```

The terminal prints progress for Dummy, Logistic Regression, Random Forest, and XGBoost. A normal run logs warnings that Cloudpickle files should only be loaded from trusted sources. This is expected because the project loads only its own locally generated models.

Do not interrupt the process. Completion is confirmed when the terminal prints:

```text
Selected model: xgboost
Final test metrics:
Best model bundle: .../artifacts/best_model.joblib
Evaluation report: .../reports/evaluation_report.md
```

The latest model replaces:

```text
artifacts/best_model.joblib
```

A timestamped copy is also saved under `artifacts/versions/`.

## 6. View MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000`, choose `customer-churn-classification`, and compare the latest four runs.

Repeated training intentionally creates additional experiment runs. MLflow preserves history so results can be compared over time.

Stop MLflow with `Ctrl+C`.

## 7. Run the prediction API

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open:

- `http://127.0.0.1:8000/docs` for interactive requests.
- `http://127.0.0.1:8000/health` for model status.

The root URL `/` intentionally has no route and returns `404`.

Stop the API with `Ctrl+C`.

## 8. Decide whether retraining is necessary

Retrain when data, cleaning, features, model settings, or code changes. Do not retrain merely to start MLflow or the API; both use existing outputs.

## Generated versus tracked files

Tracked in Git:

- Source code, configuration, tests, documentation, EDA, and final reports.

Generated and ignored:

- `artifacts/`
- `mlflow.db`
- `mlruns/`

The ignored files exist locally and can be regenerated from tracked code and configuration.
