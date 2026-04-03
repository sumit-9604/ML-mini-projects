"""
Feature engineering and data preprocessing for NYC Taxi dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


def extract_temporal_features(df: pd.DataFrame, datetime_col: str = 'pickup_datetime') -> pd.DataFrame:
    """
    Extract temporal features from datetime column
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    datetime_col : str
        Name of datetime column
    
    Returns
    -------
    pd.DataFrame
        Dataframe with temporal features added
    """
    df = df.copy()
    
    # Convert to datetime
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # Extract features
    df['pickup_hour'] = df[datetime_col].dt.hour          # 0-23
    df['pickup_day'] = df[datetime_col].dt.day            # 1-31
    df['pickup_month'] = df[datetime_col].dt.month        # 1-12
    df['pickup_weekday'] = df[datetime_col].dt.weekday    # 0-6 (Mon-Sun)
    df['pickup_quarter'] = df[datetime_col].dt.quarter    # 1-4
    
    # Drop original datetime column
    df = df.drop(datetime_col, axis=1)
    
    logger.info("✓ Temporal features extracted")
    return df


def calculate_distance(df: pd.DataFrame, 
                       pickup_lon: str = 'pickup_longitude',
                       pickup_lat: str = 'pickup_latitude',
                       dropoff_lon: str = 'dropoff_longitude',
                       dropoff_lat: str = 'dropoff_latitude') -> pd.DataFrame:
    """
    Calculate Euclidean distance between pickup and dropoff
    
    Note: For production, use Haversine (great circle) distance
    This is approximation for this example
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    pickup_lon, pickup_lat : str
        Pickup coordinate column names
    dropoff_lon, dropoff_lat : str
        Dropoff coordinate column names
    
    Returns
    -------
    pd.DataFrame
        Dataframe with distance features added
    """
    df = df.copy()
    
    # Euclidean distance
    df['distance'] = np.sqrt(
        (df[pickup_lon] - df[dropoff_lon])**2 +
        (df[pickup_lat] - df[dropoff_lat])**2
    )
    
    # Log transform for normalization
    df['log_distance'] = np.log1p(df['distance'])
    
    logger.info("✓ Distance features calculated")
    return df


def remove_outliers(df: pd.DataFrame, target_col: str = 'trip_duration', 
                    quantile_high: float = 0.99) -> pd.DataFrame:
    """
    Remove outliers using quantile method
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Column to check for outliers
    quantile_high : float
        Upper quantile threshold (default: 99th percentile)
    
    Returns
    -------
    pd.DataFrame
        Dataframe with outliers removed
    """
    initial_len = len(df)
    
    # Remove very short trips (< 1 second)
    df = df[df[target_col] > 1].copy()
    
    # Remove very long trips (> 99th percentile)
    threshold = df[target_col].quantile(quantile_high)
    df = df[df[target_col] <= threshold].copy()
    
    removed = initial_len - len(df)
    logger.info(f"✓ Removed {removed} outlier records ({removed/initial_len*100:.2f}%)")
    
    return df


def log_transform_target(df: pd.DataFrame, target_col: str = 'trip_duration',
                         new_col: str = 'log_trip_duration') -> pd.DataFrame:
    """
    Apply log transformation to target variable
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Original target column
    new_col : str
        New log-transformed column name
    
    Returns
    -------
    pd.DataFrame
        Dataframe with log-transformed target
    """
    df = df.copy()
    df[new_col] = np.log1p(df[target_col])
    logger.info("✓ Target variable log-transformed")
    return df


def prepare_features(df: pd.DataFrame, feature_cols: list, 
                     target_col: str = None,
                     scaler_type: str = 'standard') -> tuple:
    """
    Prepare features and target for modeling
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names
    target_col : str, optional
        Target column name
    scaler_type : str
        Type of scaler ('standard' or 'robust')
    
    Returns
    -------
    tuple
        (X, y, scaler) or just X if target_col is None
    """
    X = df[feature_cols].copy()
    
    # Scale features
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=feature_cols)
    
    logger.info(f"✓ Features scaled using {scaler_type} scaler")
    
    if target_col:
        y = df[target_col].copy()
        logger.info(f"✓ Target variable prepared: {len(y)} samples")
        return X, y, scaler
    
    return X, scaler


def preprocess_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> dict:
    """
    Complete preprocessing pipeline
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame, optional
        Test data
    
    Returns
    -------
    dict
        Dictionary with processed train/test data and scaler
    """
    logger.info("Starting preprocessing pipeline...")
    
    # Handle training data
    train = train_df.copy()
    train = extract_temporal_features(train)
    train = calculate_distance(train)
    train = remove_outliers(train)
    train = log_transform_target(train)
    
    result = {'train': train}
    
    # Handle test data if provided
    if test_df is not None:
        test = test_df.copy()
        test = extract_temporal_features(test)
        test = calculate_distance(test)
        result['test'] = test
    
    logger.info("✓ Preprocessing pipeline complete")
    return result


if __name__ == "__main__":
    # Example usage
    from data_loader import load_raw_data
    
    train_df, test_df = load_raw_data("data/raw/train.csv", "data/raw/test.csv")
    processed = preprocess_pipeline(train_df, test_df)
    
    print(processed['train'].info())
