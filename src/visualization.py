"""
Visualization utilities for FX Volatility Project.
Functions for creating plots and visualizations for forex analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess


def plot_heteroskedasticity_diagnostics(
    results: sm.regression.linear_model.RegressionResultsWrapper,
    X: np.ndarray,
    feature_names: List[str],
    figsize: Tuple[int, int] = (15, 12)
) -> plt.Figure:
    """
    Create diagnostic plots for heteroskedasticity.
    
    Parameters:
    -----------
    results : sm.regression.linear_model.RegressionResultsWrapper
        Regression results
    X : np.ndarray
        Feature matrix
    feature_names : List[str]
        Names of features
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Figure with diagnostic plots
    """
    residuals = results.resid
    fitted_values = results.fittedvalues
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot residuals vs fitted values
    axes[0, 0].scatter(fitted_values, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='-')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted Values')
    
    # Create scale-location plot
    sqrt_abs_resid = np.sqrt(np.abs(residuals / np.std(residuals)))
    axes[0, 1].scatter(fitted_values, sqrt_abs_resid, alpha=0.5)
    
    # Add LOWESS trend line
    lowess_result = lowess(sqrt_abs_resid, fitted_values, frac=2/3)
    axes[0, 1].plot(lowess_result[:, 0], lowess_result[:, 1], color='red', lw=2)
    
    axes[0, 1].set_xlabel('Fitted Values')
    axes[0, 1].set_ylabel('√|Standardized Residuals|')
    axes[0, 1].set_title('Scale-Location Plot')
    
    # Create QQ plot
    sm.qqplot(residuals, line='45', ax=axes[1, 0], fit=True)
    axes[1, 0].set_title('Q-Q Plot')
    
    # Plot squared residuals vs fitted values
    axes[1, 1].scatter(fitted_values, residuals**2, alpha=0.5)
    axes[1, 1].set_xlabel('Fitted Values')
    axes[1, 1].set_ylabel('Squared Residuals')
    axes[1, 1].set_title('Squared Residuals vs Fitted Values')
    
    plt.tight_layout()
    return fig


def plot_residuals_vs_features(
    results: sm.regression.linear_model.RegressionResultsWrapper,
    X: np.ndarray,
    feature_names: List[str],
    figsize: Tuple[int, int] = (15, 10),
    max_features: int = 6
) -> plt.Figure:
    """
    Plot residuals against each feature.
    
    Parameters:
    -----------
    results : sm.regression.linear_model.RegressionResultsWrapper
        Regression results
    X : np.ndarray
        Feature matrix
    feature_names : List[str]
        Names of features
    figsize : Tuple[int, int]
        Figure size
    max_features : int
        Maximum number of features to plot
        
    Returns:
    --------
    plt.Figure
        Figure with residual plots
    """
    residuals = results.resid
    
    # Limit the number of features to plot
    n_features = min(len(feature_names), max_features)
    n_rows = (n_features + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(n_features):
        axes[i].scatter(X[:, i], residuals, alpha=0.5)
        axes[i].axhline(y=0, color='r', linestyle='-')
        axes[i].set_xlabel(feature_names[i])
        axes[i].set_ylabel('Residuals')
        axes[i].set_title(f'Residuals vs {feature_names[i]}')
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_volatility_regimes(
    returns: pd.Series,
    volatility: pd.Series,
    regimes: pd.Series,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot volatility regimes.
    
    Parameters:
    -----------
    returns : pd.Series
        Returns series
    volatility : pd.Series
        Volatility series
    regimes : pd.Series
        Regime labels
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Figure with volatility regime plots
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot returns
    axes[0].plot(returns.index, returns, alpha=0.7)
    axes[0].set_ylabel('Returns')
    axes[0].set_title('FX Returns')
    
    # Plot volatility
    axes[1].plot(volatility.index, volatility, color='orange')
    axes[1].set_ylabel('Volatility')
    axes[1].set_title('Rolling Volatility')
    
    # Plot regimes
    cmap = plt.cm.get_cmap('viridis', len(regimes.unique()))
    for regime in sorted(regimes.unique()):
        regime_data = volatility[regimes == regime]
        axes[2].scatter(regime_data.index, regime_data, 
                       color=cmap(regime), label=f'Regime {regime}',
                       alpha=0.7)
    
    axes[2].set_ylabel('Volatility')
    axes[2].set_title('Volatility Regimes')
    axes[2].legend()
    
    plt.tight_layout()
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot comparison of OLS and WLS coefficients.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        DataFrame comparing OLS and WLS results
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Figure with coefficient comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot coefficients
    comparison_df[['OLS_Coef', 'WLS_Coef']].plot(kind='bar', ax=axes[0])
    axes[0].set_title('OLS vs WLS Coefficients')
    axes[0].set_ylabel('Coefficient Value')
    axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot standard errors
    comparison_df[['OLS_SE', 'WLS_SE']].plot(kind='bar', ax=axes[1])
    axes[1].set_title('OLS vs WLS Standard Errors')
    axes[1].set_ylabel('Standard Error')
    
    plt.tight_layout()
    return fig


def plot_prediction_comparison(
    y_true: np.ndarray,
    y_pred_ols: np.ndarray,
    y_pred_wls: np.ndarray,
    dates: pd.DatetimeIndex,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot comparison of OLS and WLS predictions.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True target values
    y_pred_ols : np.ndarray
        OLS predictions
    y_pred_wls : np.ndarray
        WLS predictions
    dates : pd.DatetimeIndex
        Dates for x-axis
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Figure with prediction comparison
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Plot predictions over time
    axes[0].plot(dates, y_true, 'k-', label='Actual', alpha=0.7)
    axes[0].plot(dates, y_pred_ols, 'b--', label='OLS Predictions', alpha=0.7)
    axes[0].plot(dates, y_pred_wls, 'r--', label='WLS Predictions', alpha=0.7)
    axes[0].set_title('Actual vs Predicted Values')
    axes[0].set_ylabel('Target Value')
    axes[0].legend()
    
    # Plot prediction errors
    ols_errors = y_true - y_pred_ols
    wls_errors = y_true - y_pred_wls
    
    axes[1].plot(dates, ols_errors, 'b-', label='OLS Errors', alpha=0.7)
    axes[1].plot(dates, wls_errors, 'r-', label='WLS Errors', alpha=0.7)
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1].set_title('Prediction Errors')
    axes[1].set_ylabel('Error')
    axes[1].legend()
    
    plt.tight_layout()
    return fig


def plot_weights_analysis(
    weights: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    dates: pd.DatetimeIndex,
    figsize: Tuple[int, int] = (15, 10),
    max_features: int = 3
) -> plt.Figure:
    """
    Plot analysis of WLS weights.
    
    Parameters:
    -----------
    weights : np.ndarray
        WLS weights
    X : np.ndarray
        Feature matrix
    feature_names : List[str]
        Names of features
    dates : pd.DatetimeIndex
        Dates for x-axis
    figsize : Tuple[int, int]
        Figure size
    max_features : int
        Maximum number of features to plot against weights
        
    Returns:
    --------
    plt.Figure
        Figure with weights analysis
    """
    # Limit the number of features to plot
    n_features = min(len(feature_names), max_features)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot weights over time
    axes[0, 0].plot(dates, weights)
    axes[0, 0].set_title('WLS Weights Over Time')
    axes[0, 0].set_ylabel('Weight')
    
    # Plot weights distribution
    axes[0, 1].hist(weights, bins=30, alpha=0.7)
    axes[0, 1].set_title('Distribution of Weights')
    axes[0, 1].set_xlabel('Weight')
    axes[0, 1].set_ylabel('Frequency')
    
    # Plot weights against selected features
    for i in range(n_features):
        ax = axes[1, 0] if i == 0 else axes[1, 1]
        ax.scatter(X[:, i], weights, alpha=0.5)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('Weight')
        ax.set_title(f'Weights vs {feature_names[i]}')
    
    plt.tight_layout()
    return fig


def plot_strategy_performance(
    returns: pd.Series,
    strategy_returns: pd.Series,
    cumulative_returns: pd.Series,
    benchmark_returns: pd.Series,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot trading strategy performance.
    
    Parameters:
    -----------
    returns : pd.Series
        Asset returns
    strategy_returns : pd.Series
        Strategy returns
    cumulative_returns : pd.Series
        Cumulative strategy returns
    benchmark_returns : pd.Series
        Cumulative benchmark returns
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Figure with strategy performance plots
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot asset returns
    axes[0].plot(returns.index, returns, alpha=0.7)
    axes[0].set_ylabel('Asset Returns')
    axes[0].set_title('Asset Returns')
    
    # Plot strategy returns
    axes[1].plot(strategy_returns.index, strategy_returns, color='green', alpha=0.7)
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1].set_ylabel('Strategy Returns')
    axes[1].set_title('Strategy Returns')
    
    # Plot cumulative returns
    axes[2].plot(cumulative_returns.index, cumulative_returns, 'g-', 
                label='Strategy', linewidth=2)
    axes[2].plot(benchmark_returns.index, benchmark_returns, 'b--', 
                label='Buy & Hold', linewidth=2)
    axes[2].set_ylabel('Cumulative Returns')
    axes[2].set_title('Cumulative Performance')
    axes[2].legend()
    
    plt.tight_layout()
    return fig