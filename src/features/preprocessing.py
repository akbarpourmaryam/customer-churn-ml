"""Feature preprocessing for the Telco Customer Churn model."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "tenure", 
    "MonthlyCharges",
    "TotalCharges"
]

BINARY_NUMERIC_FEATURES = [
    "SeniorCitizen"
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing transformations for features."""
    # NUMERIC_FEATURES magnitudes are extremely different so we use StandardScaler(). 
    numeric_transformer     = Pipeline(steps=[("scaler", StandardScaler())])
    # handle_unknown="ignore" means if transformation data contains a category that wasn't seen during fitting, don't crash.
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor            = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer,NUMERIC_FEATURES),
            ("binary", "passthrough",BINARY_NUMERIC_FEATURES),
            ("categorical", categorical_transformer,CATEGORICAL_FEATURES),
        ],
        remainder="drop")
    return preprocessor
