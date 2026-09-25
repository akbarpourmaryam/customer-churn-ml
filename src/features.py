"""Domain feature engineering used consistently during training and inference."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create compact, explainable customer-lifecycle features."""

    result = data.copy()
    result["TotalServices"] = result[SERVICE_COLUMNS].eq("Yes").sum(axis=1)
    result["TenureGroup"] = pd.cut(
        result["tenure"],
        bins=[-np.inf, 12, 24, 48, 60, np.inf],
        labels=["0-12", "13-24", "25-48", "49-60", "61+"],
    ).astype(str)
    return result


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer that keeps feature engineering in the model."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        return add_features(X)

    def get_feature_names_out(self, input_features=None):
        input_features = (
            self.feature_names_in_ if input_features is None else input_features
        )
        input_features = list(input_features)
        return np.asarray([*input_features, "TotalServices", "TenureGroup"], dtype=object)
