"""
Data loading and basic operations for NYC Taxi dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_data(train_path: str, test_path: str = None):
    """
    Load raw CSV data from disk
    
    Parameters
    ----------
    train_path : str
        Path to training CSV file
    test_path : str, optional
        Path to test CSV file
    
    Returns
    -------
    tuple
        (train_df, test_df) or just train_df if test_path is None
    """
    logger.info(f"Loading training data from {train_path}...")
    train = pd.read_csv(train_path)
    logger.info(f"✓ Loaded {len(train)} training records")
    
    if test_path:
        logger.info(f"Loading test data from {test_path}...")
        test = pd.read_csv(test_path)
        logger.info(f"✓ Loaded {len(test)} test records")
        return train, test
    
    return train


def save_processed_data(df: pd.DataFrame, output_path: str):
    """Save processed dataframe to CSV"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved to {output_path}")


def basic_data_info(df: pd.DataFrame, name: str = "Data"):
    """Print basic information about dataframe"""
    print(f"\n{'='*60}")
    print(f"{name.upper()} SUMMARY")
    print(f"{'='*60}")
    print(f"Shape:           {df.shape}")
    print(f"Memory Usage:    {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nDatatypes:")
    print(df.dtypes)
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nDuplicates:      {df.duplicated().sum()}")
    print(f"{'='*60}\n")


def get_numeric_columns(df: pd.DataFrame) -> list:
    """Get list of numeric columns"""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list:
    """Get list of categorical columns"""
    return df.select_dtypes(include=['object']).columns.tolist()


if __name__ == "__main__":
    # Example usage
    train_df = load_raw_data("data/raw/train.csv")
    basic_data_info(train_df, name="Training Data")
