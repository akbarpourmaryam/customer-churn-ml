"""Data cleaning and validation for the Telco Customer Churn dataset."""

import pandas as pd

from src.preprocessing import (
    RAW_BINARY_FEATURES,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERIC_FEATURES,
)


ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
FEATURE_COLUMNS = {
    ID_COLUMN,
    *RAW_NUMERIC_FEATURES,
    *RAW_BINARY_FEATURES,
    *RAW_CATEGORICAL_FEATURES,
}
REQUIRED_COLUMNS = {TARGET_COLUMN, *FEATURE_COLUMNS}


def validate_schema(raw_data: pd.DataFrame, require_target: bool = True) -> None:
    """Validate the table shape and required columns before accessing them."""

    if not isinstance(raw_data, pd.DataFrame):
        raise TypeError("Training data must be a pandas DataFrame.")
    if raw_data.empty:
        raise ValueError("Training data must contain at least one row.")

    required_columns = REQUIRED_COLUMNS if require_target else FEATURE_COLUMNS
    missing_columns = required_columns - set(raw_data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

def validate_data(raw_data: pd.DataFrame, require_target: bool = True) -> None:
    """Validate assumptions about the raw dataset."""

    validate_schema(raw_data, require_target=require_target)
    
    # customerID should exist for every customer.
    if raw_data[ID_COLUMN].isna().any() or raw_data[ID_COLUMN].eq("").any():
        raise ValueError("customerID contains missing values.")

    # Every customer should appear only once.
    if raw_data[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate customer IDs found.")

    # SeniorCitizen should only contain 0 and 1.
    valid_senior_values = {0, 1}
    actual_senior_values = set(raw_data["SeniorCitizen"].dropna().unique())

    if not actual_senior_values.issubset(valid_senior_values):
        raise ValueError(
            "Unexpected SeniorCitizen values: "
            f"{actual_senior_values - valid_senior_values}"
        )

    if raw_data["SeniorCitizen"].isna().any():
        raise ValueError("SeniorCitizen contains missing values.")

    # TotalCharges is handled after conversion because blank values are valid
    # for brand-new customers whose tenure is zero.
    for column in ["tenure", "MonthlyCharges", *RAW_CATEGORICAL_FEATURES]:
        if raw_data[column].isna().any():
            raise ValueError(f"{column} contains missing values.")

    for column in ("tenure", "MonthlyCharges"):
        numeric_values = pd.to_numeric(raw_data[column], errors="coerce")
        if numeric_values.isna().any():
            raise ValueError(f"{column} must contain only numeric values.")

    empty_categorical_columns = [
        column for column in RAW_CATEGORICAL_FEATURES if raw_data[column].eq("").any()
    ]
    if empty_categorical_columns:
        raise ValueError(
            "Categorical columns contain empty values: "
            f"{sorted(empty_categorical_columns)}"
        )

    if (pd.to_numeric(raw_data["tenure"], errors="raise") < 0).any():
        raise ValueError("tenure cannot contain negative values.")
    if (pd.to_numeric(raw_data["MonthlyCharges"], errors="raise") < 0).any():
        raise ValueError("MonthlyCharges cannot contain negative values.")


def validate_target(raw_data: pd.DataFrame) -> None:
    """Validate the churn target."""

    if raw_data[TARGET_COLUMN].isna().any():
        raise ValueError("Churn contains missing values.")
    allowed_values = {"Yes", "No"}
    actual_values = set(raw_data[TARGET_COLUMN].unique())
    unexpected_values = actual_values - allowed_values
    if unexpected_values:
        raise ValueError(f"Unexpected Churn values: {unexpected_values}")


def clean_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco Customer Churn dataset."""

    raw_data = raw_data.copy()
    # Remove accidental leading/trailing whitespace from string columns.
    string_columns = raw_data.select_dtypes(include=["object", "string"]).columns
    for column in string_columns:
        raw_data[column] = raw_data[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    # Validate structural assumptions after whitespace normalization.
    validate_data(raw_data=raw_data)
    validate_target(raw_data=raw_data)
    # TotalCharges should be numeric.
    raw_data["TotalCharges"] = pd.to_numeric(raw_data["TotalCharges"], errors="coerce")
    # New customers have no accumulated charges.
    new_customer_missing = raw_data["TotalCharges"].isna() & raw_data["tenure"].eq(0)
    raw_data.loc[new_customer_missing, "TotalCharges"] = 0.0
    # Missing TotalCharges for an existing customer is unexpected.
    unexpected_missing = (raw_data["TotalCharges"].isna() & raw_data["tenure"].gt(0))
    if unexpected_missing.any():
        raise ValueError(
            "TotalCharges contains missing or invalid values for customers with tenure greater than 0.")
    return raw_data


def clean_prediction_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Clean feature-only data using the same rules as training data."""

    data = raw_data.copy()
    string_columns = data.select_dtypes(include=["object", "string"]).columns
    for column in string_columns:
        data[column] = data[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    validate_data(data, require_target=False)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    new_customer_missing = data["TotalCharges"].isna() & data["tenure"].eq(0)
    data.loc[new_customer_missing, "TotalCharges"] = 0.0
    if data["TotalCharges"].isna().any():
        raise ValueError("TotalCharges contains missing or invalid values.")
    return data
