"""
Data utilities for FX Volatility Project.
Functions for loading, processing, and preparing forex data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime, timedelta
import ta


def fetch_fx_data(
    pairs: List[str], 
    start_date: str, 
    end_date: str, 
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch historical FX data for specified currency pairs.
    
    Parameters:
    -----------
    pairs : List[str]
        List of currency pairs (e.g., ["EURUSD=X", "GBPUSD=X"])
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    interval : str
        Data interval (default: "1d" for daily)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing historical FX data
    """
    data_frames = {}
    
    for pair in pairs:
        ticker = yf.Ticker(pair)
        data = ticker.history(start=start_date, end=end_date, interval=interval)
        data_frames[pair] = data['Close']
    
    # Combine all pairs into a single DataFrame
    fx_data = pd.concat(data_frames, axis=1)
    fx_data.columns = [pair.replace('=X', '') for pair in pairs]
    
    return fx_data


def fetch_economic_data(
    indicators: List[str], 
    start_date: str, 
    end_date: str
) -> pd.DataFrame:
    """
    Fetch economic indicators from FRED.
    
    Parameters:
    -----------
    indicators : List[str]
        List of FRED indicator codes
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing economic indicators
    """
    try:
        data = web.DataReader(indicators, 'fred', start_date, end_date)
        return data
    except Exception as e:
        print(f"Error fetching economic data: {e}")
        return pd.DataFrame()


def calculate_returns(
    price_data: pd.DataFrame, 
    method: str = 'log'
) -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        DataFrame containing price data
    method : str
        Method for calculating returns ('log' or 'simple')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing returns
    """
    if method == 'log':
        returns = np.log(price_data / price_data.shift(1))
    else:  # simple returns
        returns = price_data.pct_change()
    
    return returns


def create_lagged_features(
    data: pd.DataFrame, 
    columns: List[str], 
    lags: List[int]
) -> pd.DataFrame:
    """
    Create lagged features for specified columns.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame
    columns : List[str]
        Columns for which to create lags
    lags : List[int]
        List of lag periods
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with original and lagged features
    """
    df = data.copy()
    
    for col in columns:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    return df


def calculate_rolling_volatility(
    returns: pd.DataFrame, 
    windows: List[int] = [5, 10, 22, 66]
) -> pd.DataFrame:
    """
    Calculate rolling volatility for different window sizes.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame containing returns
    windows : List[int]
        List of window sizes in days
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing rolling volatility measures
    """
    vol_df = pd.DataFrame(index=returns.index)
    
    for col in returns.columns:
        for window in windows:
            vol_df[f"{col}_vol_{window}d"] = returns[col].rolling(window=window).std() * np.sqrt(252)
    
    return vol_df


def add_technical_indicators(
    price_data: pd.DataFrame, 
    pair_name: str
) -> pd.DataFrame:
    """
    Add technical indicators for a currency pair.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        DataFrame containing OHLCV data for a currency pair
    pair_name : str
        Name of the currency pair
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with technical indicators
    """
    df = price_data.copy()
    
    # Add RSI
    df[f"{pair_name}_rsi_14"] = ta.momentum.RSIIndicator(
        close=df['Close'], window=14
    ).rsi()
    
    # Add Bollinger Bands
    bollinger = ta.volatility.BollingerBands(
        close=df['Close'], window=20, window_dev=2
    )
    df[f"{pair_name}_bb_upper"] = bollinger.bollinger_hband()
    df[f"{pair_name}_bb_lower"] = bollinger.bollinger_lband()
    df[f"{pair_name}_bb_width"] = bollinger.bollinger_wband()
    
    # Add MACD
    macd = ta.trend.MACD(
        close=df['Close'], window_slow=26, window_fast=12, window_sign=9
    )
    df[f"{pair_name}_macd"] = macd.macd()
    df[f"{pair_name}_macd_signal"] = macd.macd_signal()
    df[f"{pair_name}_macd_diff"] = macd.macd_diff()
    
    return df


def prepare_model_data(
    fx_data: pd.DataFrame, 
    economic_data: pd.DataFrame, 
    target_pair: str,
    feature_columns: List[str],
    target_horizon: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """
    Prepare data for modeling, including train/validation/test splits.
    
    Parameters:
    -----------
    fx_data : pd.DataFrame
        DataFrame containing FX data
    economic_data : pd.DataFrame
        DataFrame containing economic indicators
    target_pair : str
        Target currency pair for prediction
    feature_columns : List[str]
        List of feature columns to use
    target_horizon : int
        Forecast horizon in days
    train_ratio : float
        Ratio of data to use for training
    val_ratio : float
        Ratio of data to use for validation
        
    Returns:
    --------
    Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]
        Dictionary of DataFrames and arrays for training, validation, and testing
    """
    # Merge FX and economic data
    merged_data = fx_data.join(economic_data, how='left')
    merged_data = merged_data.dropna()
    
    # Create target variable (future returns)
    target = merged_data[target_pair].shift(-target_horizon)
    features = merged_data[feature_columns]
    
    # Remove rows with NaN due to shifting
    valid_idx = ~target.isna()
    target = target[valid_idx]
    features = features[valid_idx]
    
    # Split data into train, validation, and test sets
    n = len(target)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Create DataFrames dictionary
    data_frames = {
        'train_features': features.iloc[:train_end],
        'train_target': target.iloc[:train_end],
        'val_features': features.iloc[train_end:val_end],
        'val_target': target.iloc[train_end:val_end],
        'test_features': features.iloc[val_end:],
        'test_target': target.iloc[val_end:]
    }
    
    # Create numpy arrays dictionary
    arrays = {
        'train_X': features.iloc[:train_end].values,
        'train_y': target.iloc[:train_end].values,
        'val_X': features.iloc[train_end:val_end].values,
        'val_y': target.iloc[train_end:val_end].values,
        'test_X': features.iloc[val_end:].values,
        'test_y': target.iloc[val_end:].values
    }
    
    return data_frames, arrays


def identify_volatility_regimes(
    returns: pd.DataFrame,
    column: str,
    n_regimes: int = 2,
    window: int = 22
) -> pd.DataFrame:
    """
    Identify volatility regimes using rolling volatility and K-means clustering.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame containing returns
    column : str
        Column name for the returns series
    n_regimes : int
        Number of volatility regimes to identify
    window : int
        Rolling window size for volatility calculation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with volatility and regime labels
    """
    from sklearn.cluster import KMeans
    
    # Calculate rolling volatility
    rolling_vol = returns[column].rolling(window=window).std() * np.sqrt(252)
    
    # Prepare data for clustering
    X = rolling_vol.dropna().values.reshape(-1, 1)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_regimes, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Create result DataFrame
    result = pd.DataFrame(index=rolling_vol.dropna().index)
    result['volatility'] = rolling_vol.dropna()
    result['regime'] = labels
    
    # Sort regimes by volatility level
    regime_volatility = result.groupby('regime')['volatility'].mean()
    regime_mapping = {i: rank for rank, i in enumerate(regime_volatility.sort_values().index)}
    result['regime'] = result['regime'].map(regime_mapping)
    
    return result