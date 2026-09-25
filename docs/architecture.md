# Project architecture

## Training architecture

```mermaid
flowchart TD
    A["Telco customer CSV"] --> B["Schema and value validation"]
    B --> C["Data cleaning"]
    C --> D["Stratified 60/20/20 split"]
    D --> E["Training set"]
    D --> F["Validation set"]
    D --> G["Untouched test set"]

    E --> H["Saved preprocessing pipeline"]
    H --> H1["Feature engineering: TotalServices and TenureGroup"]
    H1 --> H2["Scale numeric features"]
    H2 --> H3["One-hot encode categories"]

    H3 --> I["Dummy baseline"]
    H3 --> J["Logistic Regression"]
    H3 --> K["Random Forest"]
    H3 --> L["XGBoost"]

    J --> M["5-fold grid search"]
    K --> M
    L --> M
    I --> N["Validation comparison"]
    M --> N
    F --> N
    N --> O["Select by validation PR-AUC"]
    N --> P["Select threshold by validation F1"]
    O --> Q["Refit winner on train + validation"]
    P --> Q
    Q --> R["Evaluate once on test set"]
    G --> R

    R --> S["Versioned best_model.joblib"]
    R --> T["Evaluation reports and plots"]
    M --> U["MLflow experiment runs"]
```

The test set is deliberately isolated until the end. It does not influence feature choices, hyperparameters, model selection, or threshold selection.

## Prediction architecture

```mermaid
sequenceDiagram
    participant Client as Website, CRM, or user
    participant API as FastAPI /predict
    participant Validate as Input validation and cleaning
    participant Bundle as best_model.joblib
    participant Model as Preprocessing + XGBoost

    Client->>API: Customer JSON
    API->>Validate: Validate fields and values
    Validate->>Model: Clean one-row DataFrame
    Bundle-->>Model: Model, threshold, metadata
    Model-->>API: Churn probability
    API-->>Client: Probability, class, threshold, version
```

## Artifact architecture

```mermaid
flowchart LR
    A["Source code and configuration"] --> B["train_model.py"]
    B --> C["MLflow: experiment history"]
    B --> D["artifacts/: deployable model bundles"]
    B --> E["reports/: tracked presentation results"]
    D --> F["FastAPI service"]
    E --> G["GitHub reviewer or stakeholder"]
```

| Location | Purpose | Committed to Git? |
|---|---|---|
| `src/` | Reusable application and ML code | Yes |
| `configs/training.yaml` | Reproducible experiment settings | Yes |
| `tests/` | Automated behavioral checks | Yes |
| `reports/` | Final human-readable evidence | Yes |
| `artifacts/` | Regenerable model binaries and detailed outputs | No |
| `mlflow.db`, `mlruns/` | Local experiment database and artifacts | No |

## Source-code responsibilities

| File | Responsibility |
|---|---|
| `src/cleaning.py` | Validate columns and values; clean numeric and string fields |
| `src/features.py` | Create `TotalServices` and `TenureGroup` inside the model pipeline |
| `src/preprocessing.py` | Scale numeric variables and encode categorical variables |
| `src/train.py` | Split data, construct models, and perform grid search |
| `src/evaluate.py` | Metrics, thresholds, cross-validation summaries, and plots |
| `src/reporting.py` | Generate the final Markdown evaluation report |
| `src/config.py` | Load YAML configuration and resolve project-relative paths |
| `src/api.py` | Load the final bundle and serve `/health` and `/predict` |
| `train_model.py` | Coordinate the complete training lifecycle |
| `generate_eda_report.py` | Generate reproducible EDA output |
