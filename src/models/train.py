"""Train the Telco Customer Churn model."""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.features.preprocessing import build_preprocessor


def split_data(df_clean):
    """Split cleaned data into training and test sets."""

    X = df_clean.drop(columns=["Churn"])
    y = df_clean["Churn"].map({"No": 0, "Yes": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2, #test_size is set to 0.2, meaning that 20% of the data will be used for testing and 80% for training.
        random_state=42, #random_state is set to 42 to ensure reproducibility of the train-test split. This means that every time we run the code, we will get the same split of data into training and test sets.
        stratify=y, #since our data is imbalanced, we want to make sure that the train and test sets have the same proportion of churned customers as the original dataset.
    )
    return X_train, X_test, y_train, y_test


def build_model() -> Pipeline:
    """Build the preprocessing and modeling pipeline."""

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000)),
              ])
    return model


def train_model(model, X_train, y_train):
    """Fit the model on the training data."""

    model.fit(X_train, y_train)

    return model
