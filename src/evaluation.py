"""
evaluation.py

Model evaluation utilities for Linear Regression.

Functions:
- evaluate_regression: Calculate regression metrics
- plot_actual_vs_predicted: Visualize predictions
- plot_residuals: Analyze residual errors
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# ==============================
# Metrics
# ==============================

def evaluate_regression(y_true, y_pred) -> dict:
    """
    Evaluate regression model performance.

    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted values

    Returns:
        dict: Evaluation metrics (R2, RMSE, MAE)
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5  # Version-safe RMSE
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae
    }


# ==============================
# Visualization
# ==============================

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted"):
    """
    Plot actual vs predicted values.

    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted values
        title (str): Plot title
    """
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_residuals(y_true, y_pred, title="Residuals Plot"):
    """
    Plot residuals (errors).

    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted values
        title (str): Plot title
    """
    residuals = y_true - y_pred

    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.grid(True)
    plt.show()


# ==============================
# Script Execution (Optional)
# ==============================

if __name__ == "__main__":
    from src.preprocessing import run_preprocessing
    from src.modeling import train_linear_regression

    TARGET = "energy"

    df = run_preprocessing()
    model, X_test, y_test, y_pred = train_linear_regression(
        df, target=TARGET, save_model=False
    )

    metrics = evaluate_regression(y_test, y_pred)
    print("[INFO] Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    plot_actual_vs_predicted(y_test, y_pred)
    plot_residuals(y_test, y_pred)
