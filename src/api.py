"""FastAPI service for predictions from the saved best-model bundle."""

import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.cleaning import clean_prediction_data
from src.config import PROJECT_ROOT


DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_model.joblib"


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customerID: str
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@lru_cache(maxsize=1)
def load_bundle():
    model_path = Path(os.getenv("CHURN_MODEL_PATH", DEFAULT_MODEL_PATH))
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found at {model_path}. Run training first.")
    return joblib.load(model_path)


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")


@app.get("/health")
def health():
    try:
        bundle = load_bundle()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"status": "ok", "model_version": bundle["metadata"]["model_version"]}


@app.post("/predict")
def predict(customer: Customer):
    try:
        bundle = load_bundle()
        data = clean_prediction_data(pd.DataFrame([customer.model_dump()]))
        probability = float(bundle["model"].predict_proba(data)[:, 1][0])
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    threshold = float(bundle["threshold"])
    return {
        "customer_id": customer.customerID,
        "churn_probability": probability,
        "churn_prediction": int(probability >= threshold),
        "threshold": threshold,
        "model_name": bundle["metadata"]["model_name"],
        "model_version": bundle["metadata"]["model_version"],
    }


def run():
    import uvicorn

    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=False)
