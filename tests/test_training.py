import numpy as np
import pytest

from src.cleaning import clean_data
from src.evaluate import (
    compare_thresholds,
    cross_validate_model,
    evaluate_model,
    feature_importance_report,
    select_threshold,
)
from src.features import add_features
from src.train import build_model, build_models, split_data, tune_model


def test_feature_engineering_adds_expected_features(valid_data):
    engineered = add_features(valid_data)
    assert {"TotalServices", "TenureGroup"}.issubset(engineered.columns)
    assert engineered["TotalServices"].between(0, 8).all()


def test_build_models_includes_all_required_candidates():
    assert set(build_models()) == {
        "dummy", "logistic_regression", "random_forest", "xgboost"
    }


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("unknown")


def test_split_is_stratified_and_model_can_be_evaluated(valid_data):
    cleaned = clean_data(valid_data)
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(cleaned)
    model = build_model("logistic_regression").fit(X_train, y_train)
    metrics, matrix = evaluate_model(model, X_test, y_test)

    assert all(set(part) == {0, 1} for part in (y_train, y_validation, y_test))
    assert len(X_train) + len(X_validation) + len(X_test) == len(cleaned)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert matrix.shape == (2, 2)


def test_tune_model_returns_best_estimator(valid_data):
    cleaned = clean_data(valid_data)
    X_train, _, _, y_train, _, _ = split_data(cleaned)
    fitted, parameters, score = tune_model(
        "logistic_regression",
        build_model("logistic_regression"),
        X_train,
        y_train,
        {"scoring": "average_precision", "parameters": {"classifier__C": [0.1, 1.0]}},
        cv_folds=2,
    )
    assert parameters["classifier__C"] in {0.1, 1.0}
    assert 0 <= score <= 1
    assert hasattr(fitted, "predict_proba")


def test_cross_validation_returns_metrics_and_oof_probabilities(valid_data):
    cleaned = clean_data(valid_data)
    X_train, _, _, y_train, _, _ = split_data(cleaned)
    summary, probabilities = cross_validate_model(
        build_model("logistic_regression"), X_train, y_train, n_splits=2
    )
    assert set(summary["roc_auc"]) == {"mean", "std"}
    assert len(probabilities) == len(y_train)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_threshold_comparison_and_selection():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.45, 0.9])
    comparison = compare_thresholds(y_true, probabilities, thresholds=[0.3, 0.5])
    assert list(comparison["threshold"]) == [0.3, 0.5]
    assert select_threshold(comparison) == 0.3
    assert comparison.loc[0, "customers_contacted"] == 3


def test_feature_importance_is_available_for_logistic_regression(valid_data):
    cleaned = clean_data(valid_data)
    X_train, _, _, y_train, _, _ = split_data(cleaned)
    model = build_model("logistic_regression").fit(X_train, y_train)
    report = feature_importance_report(model)
    assert not report.empty
    assert report["absolute_importance"].is_monotonic_decreasing
