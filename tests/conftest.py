import pandas as pd
import pytest


@pytest.fixture
def valid_data():
    """Return a small valid dataset containing both target classes."""

    rows = []
    for index in range(10):
        rows.append(
            {
                "customerID": f"customer-{index}",
                "gender": "Female" if index % 2 else "Male",
                "SeniorCitizen": index % 2,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": index,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 50.0 + index,
                "TotalCharges": "" if index == 0 else str((50.0 + index) * index),
                "Churn": "Yes" if index % 2 else "No",
            }
        )
    return pd.DataFrame(rows)
