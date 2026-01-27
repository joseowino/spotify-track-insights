"""
modeling.py

Module for training and predicting with Linear Regression on Spotify dataset.

Functions:
- train_linear_regression: Train a Linear Regression model
- predict_model: Make predictions with a trained model
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# ==============================
# Configuration
# ==============================
MODEL_OUTPUT_PATH = os.path.join("models", "linear_regression.joblib")

# ==============================
# Functions
# ==============================

def train_linear_regression(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    save_model: bool = True,
    output_path: str = MODEL_OUTPUT_PATH
) -> tuple:
    """
    Train a Linear Regression model on the provided dataset.

    Args:
        df (pd.DataFrame): Preprocessed dataset
        target (str): Name of the target column
        test_size (float): Fraction of data for testing
        random_state (int): Random seed
        save_model (bool): If True, saves the trained model
        output_path (str): Filepath to save the model

    Returns:
        tuple: (trained model, X_test, y_test, y_pred)
    """

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"[INFO] Linear Regression trained: R²={r2:.4f}, RMSE={rmse:.4f}")

    if save_model:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(model, output_path)
        print(f"[INFO] Model saved to {output_path}")

    return model, X_test, y_test, y_pred


def predict_model(model: LinearRegression, df: pd.DataFrame) -> pd.Series:
    """
    Make predictions using a trained Linear Regression model.

    Args:
        model (LinearRegression): Trained model
        df (pd.DataFrame): Feature data (must match training features)

    Returns:
        pd.Series: Predictions
    """
    predictions = model.predict(df)
    return pd.Series(predictions, index=df.index)


# ==============================
# Script Execution
# ==============================
if __name__ == "__main__":
    from src.preprocessing import run_preprocessing

    df = run_preprocessing()

    TARGET = "energy"

    model, X_test, y_test, y_pred = train_linear_regression(df, target=TARGET)
