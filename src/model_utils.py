"""
Model utilities for FX Volatility Project.
Functions for building, evaluating, and comparing OLS and WLS models.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def run_ols_regression(
    X: np.ndarray, 
    y: np.ndarray, 
    add_constant: bool = True
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run OLS regression.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    add_constant : bool
        Whether to add a constant term
        
    Returns:
    --------
    sm.regression.linear_model.RegressionResultsWrapper
        OLS regression results
    """
    if add_constant:
        X = sm.add_constant(X)
    
    model = sm.OLS(y, X)
    results = model.fit()
    
    return results


def test_heteroskedasticity(
    results: sm.regression.linear_model.RegressionResultsWrapper
) -> Dict[str, Dict[str, float]]:
    """
    Test for heteroskedasticity in regression residuals.
    
    Parameters:
    -----------
    results : sm.regression.linear_model.RegressionResultsWrapper
        OLS regression results
        
    Returns:
    --------
    Dict[str, Dict[str, float]]
        Dictionary containing test results
    """
    residuals = results.resid
    exog = results.model.exog
    
    # Run Breusch-Pagan test
    bp_test = het_breuschpagan(residuals, exog)
    bp_results = {
        'lm_stat': bp_test[0],
        'lm_pvalue': bp_test[1],
        'f_stat': bp_test[2],
        'f_pvalue': bp_test[3]
    }
    
    # Run White test
    white_test = het_white(residuals, exog)
    white_results = {
        'lm_stat': white_test[0],
        'lm_pvalue': white_test[1],
        'f_stat': white_test[2],
        'f_pvalue': white_test[3]
    }
    
    return {
        'breusch_pagan': bp_results,
        'white': white_results
    }


def estimate_variance_function(
    results: sm.regression.linear_model.RegressionResultsWrapper,
    X: np.ndarray,
    method: str = 'log_squared_residuals',
    add_constant: bool = True
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Estimate variance function for WLS regression.
    
    Parameters:
    -----------
    results : sm.regression.linear_model.RegressionResultsWrapper
        OLS regression results
    X : np.ndarray
        Feature matrix
    method : str
        Method for estimating variance function
    add_constant : bool
        Whether to add a constant term
        
    Returns:
    --------
    sm.regression.linear_model.RegressionResultsWrapper
        Variance function regression results
    """
    residuals = results.resid
    
    if method == 'log_squared_residuals':
        # Use log of squared residuals as dependent variable
        log_squared_resid = np.log(residuals**2)
        
        if add_constant:
            X = sm.add_constant(X)
            
        var_model = sm.OLS(log_squared_resid, X).fit()
        return var_model
    
    elif method == 'abs_residuals':
        # Use absolute residuals as dependent variable
        abs_resid = np.abs(residuals)
        
        if add_constant:
            X = sm.add_constant(X)
            
        var_model = sm.OLS(abs_resid, X).fit()
        return var_model
    
    else:
        raise ValueError(f"Unknown method: {method}")


def run_wls_regression(
    X: np.ndarray, 
    y: np.ndarray, 
    weights: np.ndarray,
    add_constant: bool = True
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run WLS regression.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    weights : np.ndarray
        Weights for observations
    add_constant : bool
        Whether to add a constant term
        
    Returns:
    --------
    sm.regression.linear_model.RegressionResultsWrapper
        WLS regression results
    """
    if add_constant:
        X = sm.add_constant(X)
    
    model = sm.WLS(y, X, weights=weights)
    results = model.fit()
    
    return results


def calculate_weights(
    var_model: sm.regression.linear_model.RegressionResultsWrapper,
    X: np.ndarray,
    method: str = 'log_squared_residuals',
    add_constant: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """
    Calculate weights for WLS regression.
    
    Parameters:
    -----------
    var_model : sm.regression.linear_model.RegressionResultsWrapper
        Variance function regression results
    X : np.ndarray
        Feature matrix
    method : str
        Method used for estimating variance function
    add_constant : bool
        Whether to add a constant term
    normalize : bool
        Whether to normalize weights
        
    Returns:
    --------
    np.ndarray
        Weights for WLS regression
    """
    if add_constant:
        X_pred = sm.add_constant(X)
    else:
        X_pred = X
    
    if method == 'log_squared_residuals':
        # Predicted log variance
        log_var_pred = var_model.predict(X_pred)
        # Convert to variance
        var_pred = np.exp(log_var_pred)
        # Weights are inverse of variance
        weights = 1 / var_pred
    
    elif method == 'abs_residuals':
        # Predicted absolute residuals
        abs_resid_pred = var_model.predict(X_pred)
        # Square to get variance estimate
        var_pred = abs_resid_pred**2
        # Weights are inverse of variance
        weights = 1 / var_pred
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights
    if normalize:
        weights = weights / np.mean(weights)
    
    return weights


def compare_models(
    ols_results: sm.regression.linear_model.RegressionResultsWrapper,
    wls_results: sm.regression.linear_model.RegressionResultsWrapper,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Compare OLS and WLS regression results.
    
    Parameters:
    -----------
    ols_results : sm.regression.linear_model.RegressionResultsWrapper
        OLS regression results
    wls_results : sm.regression.linear_model.RegressionResultsWrapper
        WLS regression results
    feature_names : List[str]
        Names of features
        
    Returns:
    --------
    pd.DataFrame
        DataFrame comparing OLS and WLS results
    """
    # Extract coefficients and standard errors
    ols_coef = ols_results.params
    wls_coef = wls_results.params
    ols_se = ols_results.bse
    wls_se = wls_results.bse
    
    # Create comparison DataFrame
    if 'const' in ols_results.params:
        names = ['const'] + feature_names
    else:
        names = feature_names
    
    comparison = pd.DataFrame({
        'OLS_Coef': ols_coef,
        'WLS_Coef': wls_coef,
        'OLS_SE': ols_se,
        'WLS_SE': wls_se,
        'Diff_Coef': wls_coef - ols_coef,
        'Diff_Coef_Pct': (wls_coef - ols_coef) / ols_coef * 100,
        'SE_Ratio': wls_se / ols_se,
        'OLS_PValue': ols_results.pvalues,
        'WLS_PValue': wls_results.pvalues
    }, index=names)
    
    return comparison


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_ols: np.ndarray,
    y_pred_wls: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate OLS and WLS predictions.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True target values
    y_pred_ols : np.ndarray
        OLS predictions
    y_pred_wls : np.ndarray
        WLS predictions
        
    Returns:
    --------
    Dict[str, Dict[str, float]]
        Dictionary containing evaluation metrics
    """
    # Calculate metrics for OLS
    ols_mse = mean_squared_error(y_true, y_pred_ols)
    ols_rmse = np.sqrt(ols_mse)
    ols_mae = mean_absolute_error(y_true, y_pred_ols)
    ols_r2 = r2_score(y_true, y_pred_ols)
    
    # Calculate metrics for WLS
    wls_mse = mean_squared_error(y_true, y_pred_wls)
    wls_rmse = np.sqrt(wls_mse)
    wls_mae = mean_absolute_error(y_true, y_pred_wls)
    wls_r2 = r2_score(y_true, y_pred_wls)
    
    # Calculate improvement percentages
    mse_improvement = (ols_mse - wls_mse) / ols_mse * 100
    rmse_improvement = (ols_rmse - wls_rmse) / ols_rmse * 100
    mae_improvement = (ols_mae - wls_mae) / ols_mae * 100
    r2_improvement = (wls_r2 - ols_r2) / abs(ols_r2) * 100 if ols_r2 != 0 else np.inf
    
    return {
        'OLS': {
            'MSE': ols_mse,
            'RMSE': ols_rmse,
            'MAE': ols_mae,
            'R2': ols_r2
        },
        'WLS': {
            'MSE': wls_mse,
            'RMSE': wls_rmse,
            'MAE': wls_mae,
            'R2': wls_r2
        },
        'Improvement': {
            'MSE': mse_improvement,
            'RMSE': rmse_improvement,
            'MAE': mae_improvement,
            'R2': r2_improvement
        }
    }


def run_regime_specific_models(
    X: np.ndarray,
    y: np.ndarray,
    regimes: np.ndarray,
    n_regimes: int = 2,
    add_constant: bool = True
) -> Dict[int, Tuple[sm.regression.linear_model.RegressionResultsWrapper, sm.regression.linear_model.RegressionResultsWrapper]]:
    """
    Run regime-specific OLS and WLS models.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    regimes : np.ndarray
        Array of regime labels
    n_regimes : int
        Number of regimes
    add_constant : bool
        Whether to add a constant term
        
    Returns:
    --------
    Dict[int, Tuple[sm.regression.linear_model.RegressionResultsWrapper, sm.regression.linear_model.RegressionResultsWrapper]]
        Dictionary mapping regime labels to (OLS, WLS) result tuples
    """
    regime_models = {}
    
    for regime in range(n_regimes):
        # Filter data for this regime
        mask = (regimes == regime)
        X_regime = X[mask]
        y_regime = y[mask]
        
        if len(X_regime) < 30:  # Skip if too few observations
            continue
        
        # Run OLS
        ols_results = run_ols_regression(X_regime, y_regime, add_constant)
        
        # Estimate variance function
        var_model = estimate_variance_function(ols_results, X_regime, add_constant=add_constant)
        
        # Calculate weights
        weights = calculate_weights(var_model, X_regime, add_constant=add_constant)
        
        # Run WLS
        wls_results = run_wls_regression(X_regime, y_regime, weights, add_constant)
        
        # Store results
        regime_models[regime] = (ols_results, wls_results)
    
    return regime_models