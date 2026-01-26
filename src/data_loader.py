"""
data_loader.py

Module for loading Spotify tracks dataset.

Functions:
- load_raw_data: Load the raw CSV file from data/raw/
- load_processed_data: Load preprocessed CSV file from data/processed/
"""

import pandas as pd
import os

# ==============================
# File Paths
# ==============================
RAW_DATA_PATH = os.path.join("data", "row_dataset.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed_dataset.csv")

# ==============================
# Functions
# ==============================
def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw Spotify tracks dataset from CSV.

    Args:
        filepath (str): Path to raw CSV file.

    Returns:
        pd.DataFrame: Raw dataset
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[INFO] Raw data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        raise
    except Exception as e:
        print(f"[ERROR] Failed to load raw data: {e}")
        raise


def load_processed_data(filepath: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """
    Load preprocessed Spotify dataset from CSV.

    Args:
        filepath (str): Path to processed CSV file.

    Returns:
        pd.DataFrame: Processed dataset ready for modeling
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[INFO] Processed data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        raise
    except Exception as e:
        print(f"[ERROR] Failed to load processed data: {e}")
        raise


# ==============================
# Quick Preview
# ==============================
if __name__ == "__main__":
    df_raw = load_raw_data()
    print(df_raw.head())
