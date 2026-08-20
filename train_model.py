"""Run the customer churn training pipeline."""

import pandas as pd

from src.data.cleaning import clean_data
from src.models.evaluate import evaluate_model
from src.models.train import (
    build_model,
    split_data,
    train_model
)

def main():
    raw_data = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_clean = clean_data(raw_data)
    X_train, X_test, y_train, y_test = split_data(df_clean)
    model                            = build_model()
    model                            = train_model(model, X_train, y_train)
    metrics, confusion_matrix        = evaluate_model(model, X_test, y_test)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("\n", "Confusion matrix:", "\n", confusion_matrix)


if __name__ == "__main__":
    main()
    