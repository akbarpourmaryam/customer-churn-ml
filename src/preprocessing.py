"""Feature preprocessing for the Telco Customer Churn models."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import FeatureEngineer


RAW_NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
RAW_BINARY_FEATURES = ["SeniorCitizen"]
RAW_CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERIC_FEATURES = [*RAW_NUMERIC_FEATURES, "TotalServices"]
BINARY_NUMERIC_FEATURES = RAW_BINARY_FEATURES
CATEGORICAL_FEATURES = [*RAW_CATEGORICAL_FEATURES, "TenureGroup"]


def build_preprocessor() -> Pipeline:
    """Build feature engineering, scaling, and encoding transformations."""

    columns = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("binary", "passthrough", BINARY_NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[("feature_engineering", FeatureEngineer()), ("columns", columns)]
    )
