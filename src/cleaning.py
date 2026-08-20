"""Data cleaning and validation for the Telco Customer Churn dataset."""

import pandas as pd

def validate_data(raw_data: pd.DataFrame) -> None:
    """Validate assumptions about the raw dataset."""

    # customerID should exist for every customer.
    if raw_data["customerID"].isna().any():
        raise ValueError("customerID contains missing values.")

    # Every customer should appear only once.
    if raw_data["customerID"].duplicated().any():
        raise ValueError("Duplicate customer IDs found.")

    # SeniorCitizen should only contain 0 and 1.
    valid_senior_values = {0, 1}
    actual_senior_values = set(raw_data["SeniorCitizen"].dropna().unique())

    if not actual_senior_values.issubset(valid_senior_values):
        raise ValueError(f"Unexpected SeniorCitizen values:"
                         f"{actual_senior_values - valid_senior_values}")


def validate_target(raw_data: pd.DataFrame) -> None:
    """Validate the churn target."""

    if "Churn" not in raw_data.columns:
        raise ValueError("Training data must contain Churn.")
    if raw_data["Churn"].isna().any():
        raise ValueError("Churn contains missing values.")
    allowed_values = {"Yes", "No"}
    actual_values = set(raw_data["Churn"].unique())
    unexpected_values = actual_values - allowed_values
    if unexpected_values:
        raise ValueError(f"Unexpected Churn values: {unexpected_values}")


def clean_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco Customer Churn dataset."""

    raw_data       = raw_data.copy()
    # Remove accidental leading/trailing whitespace from string columns.
    string_columns = raw_data.select_dtypes(include=["object", "string"]).columns
    for column in string_columns:
        raw_data[column] = raw_data[column].str.strip()
    # Validate structural assumptions.
    validate_data(raw_data=raw_data)
    validate_target(raw_data=raw_data)
    # TotalCharges should be numeric.
    raw_data["TotalCharges"] = pd.to_numeric(raw_data["TotalCharges"], errors="coerce")
    # New customers have no accumulated charges.
    new_customer_missing     = (raw_data["TotalCharges"].isna() & raw_data["tenure"].eq(0))
    raw_data.loc[new_customer_missing, "TotalCharges"] = 0.0
    # Missing TotalCharges for an existing customer is unexpected.
    unexpected_missing = (raw_data["TotalCharges"].isna() & raw_data["tenure"].gt(0))
    if unexpected_missing.any():
        raise ValueError(
            "TotalCharges contains missing or invalid values for customers with tenure greater than 0.")
    return raw_data
