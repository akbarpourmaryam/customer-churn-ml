import pandas as pd
import pytest

from src.cleaning import clean_data, validate_schema


def test_clean_data_converts_total_charges_and_fills_new_customer(valid_data):
    cleaned = clean_data(valid_data)

    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    assert cleaned.loc[0, "TotalCharges"] == 0.0


def test_validate_schema_reports_all_missing_columns(valid_data):
    invalid = valid_data.drop(columns=["customerID", "Churn"])

    with pytest.raises(ValueError, match="customerID.*Churn|Churn.*customerID"):
        validate_schema(invalid)


def test_clean_data_rejects_duplicate_customer_ids(valid_data):
    valid_data.loc[1, "customerID"] = valid_data.loc[0, "customerID"]

    with pytest.raises(ValueError, match="Duplicate customer IDs"):
        clean_data(valid_data)


def test_clean_data_rejects_invalid_target(valid_data):
    valid_data.loc[0, "Churn"] = "Maybe"

    with pytest.raises(ValueError, match="Unexpected Churn values"):
        clean_data(valid_data)


def test_clean_data_rejects_invalid_total_charges_for_existing_customer(valid_data):
    valid_data.loc[1, "TotalCharges"] = "not-a-number"

    with pytest.raises(ValueError, match="TotalCharges contains missing or invalid"):
        clean_data(valid_data)


def test_clean_data_rejects_non_numeric_monthly_charges(valid_data):
    valid_data["MonthlyCharges"] = valid_data["MonthlyCharges"].astype(object)
    valid_data.loc[0, "MonthlyCharges"] = "not-a-number"

    with pytest.raises(ValueError, match="MonthlyCharges must contain only numeric"):
        clean_data(valid_data)


def test_clean_data_rejects_empty_categorical_value(valid_data):
    valid_data.loc[0, "Contract"] = ""

    with pytest.raises(ValueError, match="Categorical columns contain empty values"):
        clean_data(valid_data)
