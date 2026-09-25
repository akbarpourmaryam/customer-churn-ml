import joblib
from fastapi.testclient import TestClient

from src.api import app, load_bundle
from src.cleaning import clean_data
from src.train import build_model, split_data


def test_prediction_endpoint_returns_probability(valid_data, tmp_path, monkeypatch):
    cleaned = clean_data(valid_data)
    X_train, _, _, y_train, _, _ = split_data(cleaned)
    model = build_model("logistic_regression").fit(X_train, y_train)
    bundle_path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "threshold": 0.4,
            "metadata": {"model_name": "logistic_regression", "model_version": "test-v1"},
        },
        bundle_path,
    )
    monkeypatch.setenv("CHURN_MODEL_PATH", str(bundle_path))
    load_bundle.cache_clear()
    payload = valid_data.drop(columns="Churn").iloc[1].to_dict()

    response = TestClient(app).post("/predict", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert 0 <= result["churn_probability"] <= 1
    assert result["threshold"] == 0.4
    assert result["model_version"] == "test-v1"
