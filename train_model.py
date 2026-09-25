"""Run the complete, reproducible customer-churn training workflow."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from src.cleaning import clean_data
from src.config import DEFAULT_CONFIG_PATH, load_config
from src.evaluate import (
    compare_thresholds,
    cross_validate_model,
    evaluate_model,
    feature_importance_report,
    save_evaluation_plots,
    save_feature_importance_plot,
    save_model_comparison_plot,
    select_threshold,
)
from src.reporting import write_evaluation_report
from src.train import build_models, split_data, tune_model


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def configure_mlflow(config):
    tracking_directory = Path(config["paths"]["mlruns_dir"])
    tracking_directory.mkdir(parents=True, exist_ok=True)
    database_path = tracking_directory.parent / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{database_path.resolve()}")
    experiment_name = config["mlflow"]["experiment_name"]
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name, artifact_location=tracking_directory.resolve().as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)


def train_project(config_path=DEFAULT_CONFIG_PATH):
    """Tune candidates, select a winner, evaluate it, and save all outputs."""

    config        = load_config(config_path)
    seed          = config["random_seed"]
    paths         = config["paths"]
    artifacts_dir = Path(paths["artifacts_dir"])
    reports_dir   = Path(paths["reports_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "models").mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    configure_mlflow(config)

    data  = clean_data(pd.read_csv(paths["data_path"]))
    split = config["data_split"]
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(
        data,
        test_size=split["test"],
        validation_size=split["validation"],
        random_state=seed,
    )
    imbalance_ratio      = float((y_train == 0).sum() / (y_train == 1).sum())
    models               = build_models(seed, imbalance_ratio)
    evaluation_config    = config["evaluation"]
    candidate_params     = {}
    candidate_thresholds = {}
    comparison_rows      = []
    tracking_data        = X_train.assign(Churn=y_train).copy()
    integer_columns      = tracking_data.select_dtypes(include="integer").columns
    tracking_data[integer_columns] = tracking_data[integer_columns].astype(float)
    mlflow_dataset = mlflow.data.from_pandas(
        tracking_data,
        source=Path(paths["data_path"]).resolve().as_uri(),
        name="telco-churn-training-partition",
        targets="Churn",
    )

    for model_name, model in models.items():
        print(f"Training and evaluating {model_name}...")
        mlflow.start_run(run_name=model_name)
        mlflow.log_input(mlflow_dataset, context="training")
        fitted, best_params, best_cv_score = tune_model(
            model_name,
            model,
            X_train,
            y_train,
            config["models"][model_name],
            cv_folds=evaluation_config["cv_folds"],
            random_state=seed,
        )
        cv_summary, _ = cross_validate_model(
            fitted, X_train, y_train, n_splits=evaluation_config["cv_folds"]
        )
        validation_probability = fitted.predict_proba(X_validation)[:, 1]
        threshold = 0.5
        threshold_table = compare_thresholds(
            y_validation, validation_probability, evaluation_config["thresholds"]
        )
        if model_name != "dummy":
            threshold = select_threshold(
                threshold_table, evaluation_config["threshold_metric"]
            )
        validation_metrics, validation_matrix = evaluate_model(
            fitted, X_validation, y_validation, threshold
        )

        candidate_params[model_name] = best_params
        candidate_thresholds[model_name] = threshold
        joblib.dump(fitted, artifacts_dir / "models" / f"{model_name}.joblib")
        threshold_path = artifacts_dir / "analysis" / f"{model_name}_thresholds.csv"
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        threshold_table.to_csv(threshold_path, index=False)
        write_json(
            artifacts_dir / "metrics" / f"{model_name}.json",
            {
                "model": model_name,
                "best_parameters": best_params,
                "grid_search_best_pr_auc": best_cv_score,
                "cross_validation": cv_summary,
                "selected_threshold": threshold,
                "validation_metrics": validation_metrics,
                "validation_confusion_matrix": validation_matrix.tolist(),
            },
        )
        comparison_rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                **{name: value for name, value in validation_metrics.items() if name != "positive_rate"},
                "cv_pr_auc": cv_summary["pr_auc"]["mean"],
                "cv_pr_auc_std": cv_summary["pr_auc"]["std"],
            }
        )

        mlflow.log_param("model", model_name)
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("imbalance_ratio", imbalance_ratio)
        mlflow.log_params({key.replace("classifier__", ""): value for key, value in best_params.items()})
        mlflow.log_metrics({f"validation_{key}": float(value) for key, value in validation_metrics.items()})
        mlflow.log_metrics({f"cv_{key}_mean": values["mean"] for key, values in cv_summary.items()})
        mlflow.log_artifact(str(threshold_path), artifact_path="analysis")
        mlflow.sklearn.log_model(
            fitted,
            name="model",
            serialization_format="cloudpickle",
        )
        mlflow.end_run()

    comparison = pd.DataFrame(comparison_rows).sort_values(
        evaluation_config["selection_metric"], ascending=False
    ).reset_index(drop=True)
    comparison_path = reports_dir / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    save_model_comparison_plot(comparison, reports_dir / "model_comparison.png")
    best_model_name = str(comparison.loc[0, "model"])
    threshold = candidate_thresholds[best_model_name]

    X_train_validation = pd.concat([X_train, X_validation])
    y_train_validation = pd.concat([y_train, y_validation])
    best_model         = build_models(seed, imbalance_ratio)[best_model_name]
    best_model.set_params(**candidate_params[best_model_name])
    best_model.fit(X_train_validation, y_train_validation)
    test_metrics, test_matrix = evaluate_model(best_model, X_test, y_test, threshold)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_version = f"{created_at}-{git_commit()[:8]}"
    metadata = {
        "model_name": best_model_name,
        "model_version": model_version,
        "created_at_utc": created_at,
        "selected_threshold": threshold,
        "selection_metric": evaluation_config["selection_metric"],
        "random_seed": seed,
        "dataset_rows": len(data),
        "dataset_sha256": file_sha256(paths["data_path"]),
        "git_commit": git_commit(),
        "best_parameters": candidate_params[best_model_name],
        "test_metrics": test_metrics,
        "test_confusion_matrix": test_matrix.tolist(),
    }
    bundle = {"model": best_model, "threshold": threshold, "metadata": metadata}
    joblib.dump(bundle, artifacts_dir / "best_model.joblib")
    versions_dir = artifacts_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, versions_dir / f"{model_version}.joblib")
    write_json(artifacts_dir / "best_model_metadata.json", metadata)

    importance = feature_importance_report(best_model)
    importance.to_csv(reports_dir / "best_model_feature_importance.csv", index=False)
    save_feature_importance_plot(importance, reports_dir / "best_model_feature_importance.png")
    save_evaluation_plots(best_model, X_test, y_test, reports_dir / "best_model_evaluation.png")
    write_evaluation_report(
        reports_dir / "evaluation_report.md",
        comparison,
        best_model_name,
        threshold,
        test_metrics,
        test_matrix,
        metadata,
    )
    write_json(reports_dir / "final_test_metrics.json", metadata)

    print("\nModel comparison (validation set)")
    print(comparison.to_string(index=False))
    print(f"\nSelected model: {best_model_name}")
    print(f"Selected threshold: {threshold:.2f}")
    print("Final test metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"Best model bundle: {artifacts_dir / 'best_model.joblib'}")
    print(f"Evaluation report: {reports_dir / 'evaluation_report.md'}")
    return {"best_model": best_model_name, "metadata": metadata, "comparison": comparison}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def cli():
    arguments = parse_args()
    train_project(arguments.config)


if __name__ == "__main__":
    cli()
