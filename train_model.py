"""Run the customer churn training pipeline and persist its artifacts."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.cleaning import clean_data
from src.evaluate import evaluate_model
from src.train import build_models, split_data, train_model


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def save_artifacts(model_name, model, metrics, confusion_matrix, artifacts_dir):
    """Save a fitted model and JSON-serializable evaluation results."""

    models_dir = artifacts_dir / "models"
    metrics_dir = artifacts_dir / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{model_name}.joblib"
    metrics_path = metrics_dir / f"{model_name}.json"
    joblib.dump(model, model_path)
    result = {
        "model": model_name,
        "metrics": {name: float(value) for name, value in metrics.items()},
        "confusion_matrix": confusion_matrix.tolist(),
    }
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return model_path, metrics_path


def main(data_path=DEFAULT_DATA_PATH, artifacts_dir=DEFAULT_ARTIFACTS_DIR):
    """Train and evaluate all configured models."""

    data_path = Path(data_path)
    artifacts_dir = Path(artifacts_dir)
    raw_data = pd.read_csv(data_path)
    df_clean = clean_data(raw_data)
    X_train, X_test, y_train, y_test = split_data(df_clean)

    results = {}
    for model_name, model in build_models().items():
        fitted_model = train_model(model, X_train, y_train)
        metrics, confusion_matrix = evaluate_model(fitted_model, X_test, y_test)
        model_path, metrics_path = save_artifacts(
            model_name,
            fitted_model,
            metrics,
            confusion_matrix,
            artifacts_dir,
        )
        results[model_name] = metrics
        print(f"\n{model_name}")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        print("  Confusion matrix:")
        print(confusion_matrix)
        print(f"  Model: {model_path}")
        print(f"  Metrics: {metrics_path}")

    return results


def parse_args():
    """Parse optional command-line paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(data_path=arguments.data, artifacts_dir=arguments.artifacts_dir)
