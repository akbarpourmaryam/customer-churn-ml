"""Model evaluation utilities."""

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained binary classifier."""

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy"          : accuracy_score(y_test, y_pred),
        "balanced_accuracy" : balanced_accuracy_score(y_test, y_pred),
        "precision"         : precision_score(y_test, y_pred),
        "recall"            : recall_score(y_test, y_pred),
        "f1"                : f1_score(y_test, y_pred),
        "roc_auc"           : roc_auc_score(y_test, y_prob),
        "pr_auc"            : average_precision_score(y_test, y_prob),
        "positive_rate"     : y_test.mean(),
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm
