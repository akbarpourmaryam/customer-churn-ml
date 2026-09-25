"""Model evaluation, threshold analysis, plots, and interpretation utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold


METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
)


def calculate_metrics(y_true, y_probability, threshold=0.5):
    """Calculate classification metrics at a probability threshold."""

    y_prediction = (np.asarray(y_probability) >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_prediction),
        "precision": precision_score(y_true, y_prediction, zero_division=0),
        "recall": recall_score(y_true, y_prediction, zero_division=0),
        "f1": f1_score(y_true, y_prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probability),
        "pr_auc": average_precision_score(y_true, y_probability),
        "positive_rate": float(np.mean(y_true)),
    }
    return metrics, confusion_matrix(y_true, y_prediction)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluate a trained classifier on untouched test data."""

    probability = model.predict_proba(X_test)[:, 1]
    return calculate_metrics(y_test, probability, threshold)


def cross_validate_model(model, X_train, y_train, n_splits=5):
    """Return fold statistics and out-of-fold training probabilities."""

    smallest_class = int(pd.Series(y_train).value_counts().min())
    effective_splits = min(n_splits, smallest_class)
    if effective_splits < 2:
        raise ValueError("Cross-validation requires at least two rows in each class.")
    splitter = StratifiedKFold(
        n_splits=effective_splits, shuffle=True, random_state=42
    )
    out_of_fold_probability = np.zeros(len(y_train), dtype=float)
    fold_metrics = []
    for train_indices, validation_indices in splitter.split(X_train, y_train):
        fold_model = clone(model)
        fold_model.fit(X_train.iloc[train_indices], y_train.iloc[train_indices])
        probability = fold_model.predict_proba(X_train.iloc[validation_indices])[:, 1]
        out_of_fold_probability[validation_indices] = probability
        metrics, _ = calculate_metrics(y_train.iloc[validation_indices], probability)
        fold_metrics.append(metrics)

    summary = {
        name: {
            "mean": float(np.mean([fold[name] for fold in fold_metrics])),
            "std": float(np.std([fold[name] for fold in fold_metrics], ddof=1)),
        }
        for name in METRIC_NAMES
    }
    return summary, out_of_fold_probability


def compare_thresholds(y_true, y_probability, thresholds=None):
    """Compare thresholds using out-of-fold training predictions."""

    if thresholds is None:
        thresholds = np.arange(0.20, 0.61, 0.05)
    rows = []
    for threshold in thresholds:
        metrics, matrix = calculate_metrics(y_true, y_probability, threshold)
        true_negative, false_positive, false_negative, true_positive = matrix.ravel()
        rows.append(
            {
                "threshold": float(round(threshold, 2)),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "customers_contacted": int(false_positive + true_positive),
                "true_positives": int(true_positive),
                "false_positives": int(false_positive),
                "false_negatives": int(false_negative),
                "true_negatives": int(true_negative),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(threshold_results, metric="f1"):
    """Select the best training threshold, preferring the lower threshold on ties."""

    best_row = threshold_results.sort_values(
        [metric, "threshold"], ascending=[False, True]
    ).iloc[0]
    return float(best_row["threshold"])


def save_evaluation_plots(model, X_test, y_test, output_path):
    """Save ROC, precision-recall, calibration, and probability plots."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probability = model.predict_proba(X_test)[:, 1]
    y_array = np.asarray(y_test)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probability)
    axes[0, 0].plot(false_positive_rate, true_positive_rate)
    axes[0, 0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0, 0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")

    precision, recall, _ = precision_recall_curve(y_test, probability)
    axes[0, 1].plot(recall, precision)
    axes[0, 1].axhline(y_test.mean(), linestyle="--", color="gray")
    axes[0, 1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")

    observed, predicted = calibration_curve(y_test, probability, n_bins=10)
    axes[1, 0].plot(predicted, observed, marker="o")
    axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1, 0].set(title="Calibration curve", xlabel="Mean predicted probability", ylabel="Observed churn rate")

    axes[1, 1].hist(probability[y_array == 0], bins=20, alpha=0.6, label="Stayed")
    axes[1, 1].hist(probability[y_array == 1], bins=20, alpha=0.6, label="Churned")
    axes[1, 1].set(title="Predicted probabilities", xlabel="Churn probability", ylabel="Customers")
    axes[1, 1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def logistic_coefficient_report(model):
    """Return coefficients, association directions, and odds ratios."""

    preprocessor = model.named_steps["preprocessor"]
    coefficients = model.named_steps["classifier"].coef_[0]
    report = pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
        }
    )
    report["direction"] = np.where(
        report["coefficient"] >= 0,
        "higher churn association",
        "lower churn association",
    )
    report["absolute_coefficient"] = report["coefficient"].abs()
    return report.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)


def feature_importance_report(model):
    """Return coefficients or impurity-based importance for a fitted pipeline."""

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(classifier, "coef_"):
        values = classifier.coef_[0]
        importance_type = "coefficient"
    elif hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        importance_type = "feature_importance"
    else:
        return pd.DataFrame(columns=["feature", "importance", "absolute_importance", "type"])
    report = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values,
            "absolute_importance": np.abs(values),
            "type": importance_type,
        }
    )
    return report.sort_values("absolute_importance", ascending=False).reset_index(drop=True)


def save_feature_importance_plot(report, output_path, top_n=15):
    """Save a horizontal plot of the most influential transformed features."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top = report.head(top_n).sort_values("absolute_importance")
    figure, axis = plt.subplots(figsize=(9, 6))
    sns.barplot(data=top, x="absolute_importance", y="feature", ax=axis, color="#4C78A8")
    axis.set(title="Top model feature importance", xlabel="Absolute importance", ylabel="")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_model_comparison_plot(comparison, output_path):
    """Plot validation metrics used to compare candidate models."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_columns = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    long = comparison.melt(
        id_vars="model", value_vars=metric_columns, var_name="metric", value_name="score"
    )
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(data=long, x="model", y="score", hue="metric", ax=axis)
    axis.set(title="Validation-set model comparison", xlabel="", ylabel="Score", ylim=(0, 1))
    axis.tick_params(axis="x", rotation=15)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
