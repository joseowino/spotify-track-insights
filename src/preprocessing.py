"""
preprocessing.py

Data cleaning and preprocessing for the Spotify Tracks dataset.

Responsibilities:
- Load raw dataset
- Select relevant features
- Clean and transform data
- Save processed dataset for modeling
"""

import pandas as pd
import os
from typing import List

from src.data_loader import load_raw_data

# ==============================
# Configuration
# ==============================

OUTPUT_PATH = os.path.join("data", "processed_dataset.csv")

FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "loudness",
    "duration_ms",
    "explicit"
]

# ==============================
# Helper Functions
# ==============================

def convert_duration_to_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert duration from milliseconds to minutes.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with duration_min column
    """

    df = df.copy()
    df["duration_min"] = df["duration_ms"] / 60000
    df.drop(columns=["duration_ms"], inplace=True)
    return df


def clean_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Clean and preprocess Spotify dataset.

    Steps:
    - Select relevant columns
    - Drop missing values
    - Convert duration to minutes
    - Ensure correct data types

    Args:
        df (pd.DataFrame): Raw dataset
        columns (List[str]): Columns to retain

    Returns:
        pd.DataFrame: Cleaned dataset
    """
    df = df[columns].copy()

    df.dropna(inplace=True)

    df = convert_duration_to_minutes(df)

    df["explicit"] = df["explicit"].astype(int)

    return df


def save_processed_data(df: pd.DataFrame, output_path: str = OUTPUT_PATH) -> None:
    """
    Save processed dataset to CSV.

    Args:
        df (pd.DataFrame): Cleaned dataset
        output_path (str): Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Processed data saved to {output_path}")


# ==============================
# Main Pipeline
# ==============================

def run_preprocessing() -> pd.DataFrame:
    """
    Run full preprocessing pipeline.

    Returns:
        pd.DataFrame: Cleaned dataset
    """
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw, FEATURE_COLUMNS)
    save_processed_data(df_clean)
    return df_clean


# ==============================
# Script Execution
# ==============================

if __name__ == "__main__":
    df = run_preprocessing()
    print(f"[INFO] Preprocessing complete: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
