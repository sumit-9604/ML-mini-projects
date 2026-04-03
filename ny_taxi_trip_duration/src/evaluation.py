"""
Model evaluation metrics and comparison functions
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    mean_absolute_percentage_error
)
import logging

logger = logging.getLogger(__name__)


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate comprehensive regression metrics
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    dict
        Dictionary of metrics {RMSE, MAE, MAPE, R2}
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R² Score': r2
    }


def compare_models(models_dict: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Train and compare multiple models
    
    Parameters
    ----------
    models_dict : dict
        Dictionary of {model_name: model_object}
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test targets
    
    Returns
    -------
    pd.DataFrame
        Comparison table with metrics for each model
    """
    results = []
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<25} | {'RMSE':<8} | {'MAE':<8} | {'MAPE':<8} | {'R²':<8}")
    print("-"*80)
    
    for name, model in models_dict.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_regression_metrics(y_test, y_pred)
        
        # Store results
        result_row = {'Model': name}
        result_row.update(metrics)
        results.append(result_row)
        
        # Print progress
        print(f"{name:<25} | {metrics['RMSE']:<8.4f} | {metrics['MAE']:<8.4f} | "
              f"{metrics['MAPE']:<8.4f} | {metrics['R² Score']:<8.4f}")
    
    print("="*80 + "\n")
    
    # Create dataframe and sort by RMSE
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSE').reset_index(drop=True)
    
    # Add rank
    results_df.insert(0, 'Rank', range(1, len(results_df) + 1))
    
    return results_df


def print_best_model_summary(results_df: pd.DataFrame):
    """
    Print summary of best performing model
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Comparison dataframe from compare_models()
    """
    best = results_df.iloc[0]
    
    print("\n" + "🏆"*30)
    print(f"\n{'BEST MODEL SUMMARY':^60}")
    print("\n" + "🏆"*30)
    print(f"\nModel:        {best['Model']}")
    print(f"RMSE:         {best['RMSE']:.4f}")
    print(f"MAE:          {best['MAE']:.4f}")
    print(f"MAPE:         {best['MAPE']:.4f}%")
    print(f"R² Score:     {best['R² Score']:.4f}")
    print("\n" + "🏆"*30 + "\n")


def calculate_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate prediction residuals
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    np.ndarray
        Residuals (y_true - y_pred)
    """
    return y_true - y_pred


def residual_summary(residuals: np.ndarray) -> dict:
    """
    Summary statistics for residuals
    
    Parameters
    ----------
    residuals : np.ndarray
        Residual values
    
    Returns
    -------
    dict
        Summary statistics
    """
    return {
        'Mean': np.mean(residuals),
        'Std Dev': np.std(residuals),
        'Min': np.min(residuals),
        'Max': np.max(residuals),
        'Median': np.median(residuals),
        '% Within ±1σ': np.sum(np.abs(residuals) <= np.std(residuals)) / len(residuals) * 100
    }


def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Print detailed model performance analysis
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    """
    metrics = calculate_regression_metrics(y_true, y_pred)
    residuals = calculate_residuals(y_true, y_pred)
    residual_stats = residual_summary(residuals)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    
    print("\n📊 Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric:<15}: {value:.4f}")
    
    print("\n📈 Residual Statistics:")
    for stat, value in residual_stats.items():
        if isinstance(value, float):
            print(f"  {stat:<15}: {value:.4f}")
        else:
            print(f"  {stat:<15}: {value:.2f}%")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    y_true = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    y_pred = np.array([1.6, 2.1, 2.4, 3.1, 3.3])
    
    metrics = calculate_regression_metrics(y_true, y_pred)
    print(metrics)
