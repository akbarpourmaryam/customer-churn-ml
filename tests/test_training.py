import json

import joblib
import pytest

from src.cleaning import clean_data
from src.evaluate import evaluate_model
from src.train import build_model, build_models, split_data, train_model
from train_model import main


def test_build_models_includes_dummy_and_logistic_regression():
    assert set(build_models()) == {"dummy", "logistic_regression"}


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("unknown")


def test_split_is_stratified_and_model_can_be_evaluated(valid_data):
    cleaned = clean_data(valid_data)
    X_train, X_test, y_train, y_test = split_data(cleaned)
    model = train_model(build_model("logistic_regression"), X_train, y_train)
    metrics, matrix = evaluate_model(model, X_test, y_test)

    assert set(y_train) == {0, 1}
    assert set(y_test) == {0, 1}
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert matrix.shape == (2, 2)


def test_main_saves_loadable_models_and_metrics(valid_data, tmp_path):
    data_path = tmp_path / "input.csv"
    artifacts_dir = tmp_path / "output"
    valid_data.to_csv(data_path, index=False)

    results = main(data_path=data_path, artifacts_dir=artifacts_dir)

    assert set(results) == {"dummy", "logistic_regression"}
    for model_name in results:
        model_path = artifacts_dir / "models" / f"{model_name}.joblib"
        metrics_path = artifacts_dir / "metrics" / f"{model_name}.json"
        assert hasattr(joblib.load(model_path), "predict")
        saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert saved_metrics["model"] == model_name
        assert len(saved_metrics["confusion_matrix"]) == 2
