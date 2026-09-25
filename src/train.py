"""Model construction, data splitting, and hyperparameter optimization."""

from copy import deepcopy

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor


MODEL_NAMES = ("dummy", "logistic_regression", "random_forest", "xgboost")


def split_data(data, test_size=0.2, validation_size=0.2, random_state=42):
    """Create stratified train, validation, and final test partitions."""

    X = data.drop(columns=["Churn"])
    y = data["Churn"].map({"No": 0, "Yes": 1})
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    relative_validation_size = validation_size / (1 - test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=y_train_validation,
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(model_name, random_state=42, imbalance_ratio=1.0):
    """Build a complete feature-processing and classification pipeline."""

    classifiers = {
        "dummy": DummyClassifier(strategy="most_frequent", random_state=random_state),
        "logistic_regression": LogisticRegression(max_iter=2_000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=1
        ),
        "xgboost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
            scale_pos_weight=imbalance_ratio,
        ),
    }
    if model_name not in classifiers:
        raise ValueError(f"Unknown model {model_name!r}. Choose from {list(classifiers)}.")
    return Pipeline(
        steps=[("preprocessor", build_preprocessor()), ("classifier", classifiers[model_name])]
    )


def build_models(random_state=42, imbalance_ratio=1.0):
    """Build every required baseline and candidate model."""

    return {
        name: build_model(name, random_state, imbalance_ratio)
        for name in MODEL_NAMES
    }


def tune_model(model_name, model, X_train, y_train, model_config, cv_folds=5, random_state=42):
    """Tune one pipeline with stratified cross-validation."""

    parameter_grid = deepcopy(model_config.get("parameters", {}))
    if model_name == "dummy" or not parameter_grid:
        return model.fit(X_train, y_train), {}, None
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        model,
        parameter_grid,
        scoring=model_config.get("scoring", "average_precision"),
        cv=splitter,
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)
