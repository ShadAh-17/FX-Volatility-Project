"""
Backtesting framework for FX Volatility Project.
Tools for testing trading strategies with volatility-adjusted position sizing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt


class VolatilityAdjustedStrategy:
    """
    A trading strategy that adjusts position sizes based on volatility regimes.
    """
    
    def __init__(
        self,
        base_position_size: float = 1.0,
        target_volatility: float = 0.10,
        max_position_size: float = 3.0,
        stop_loss_std: float = 2.0,
        take_profit_std: float = 3.0
    ):
        """
        Initialize the strategy.
        
        Parameters:
        -----------
        base_position_size : float
            Base position size (1.0 = 100% of capital)
        target_volatility : float
            Target annualized volatility
        max_position_size : float
            Maximum position size
        stop_loss_std : float
            Stop loss in standard deviations
        take_profit_std : float
            Take profit in standard deviations
        """
        self.base_position_size = base_position_size
        self.target_volatility = target_volatility
        self.max_position_size = max_position_size
        self.stop_loss_std = stop_loss_std
        self.take_profit_std = take_profit_std
        
        # Initialize state variables
        self.position = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
    
    def calculate_position_size(
        self,
        current_volatility: float
    ) -> float:
        """
        Calculate position size based on volatility.
        
        Parameters:
        -----------
        current_volatility : float
            Current annualized volatility
            
        Returns:
        --------
        float
            Position size
        """
        if current_volatility == 0:
            return self.max_position_size
        
        # Scale position size inversely with volatility
        position_size = self.base_position_size * (self.target_volatility / current_volatility)
        
        # Apply limits
        position_size = min(position_size, self.max_position_size)
        position_size = max(position_size, 0)
        
        return position_size
    
    def generate_signal(
        self,
        prediction: float,
        threshold: float = 0.0001
    ) -> int:
        """
        Generate trading signal based on prediction.
        
        Parameters:
        -----------
        prediction : float
            Predicted return
        threshold : float
            Signal threshold
            
        Returns:
        --------
        int
            Signal: 1 (buy), -1 (sell), 0 (neutral)
        """
        if prediction > threshold:
            return 1
        elif prediction < -threshold:
            return -1
        else:
            return 0
    
    def set_stop_loss_take_profit(
        self,
        price: float,
        volatility: float,
        signal: int
    ) -> Tuple[float, float]:
        """
        Set stop loss and take profit levels.
        
        Parameters:
        -----------
        price : float
            Current price
        volatility : float
            Current daily volatility
        signal : int
            Trading signal
            
        Returns:
        --------
        Tuple[float, float]
            Stop loss and take profit levels
        """
        daily_vol = volatility / np.sqrt(252)
        
        if signal > 0:  # Long position
            stop_loss = price * (1 - self.stop_loss_std * daily_vol)
            take_profit = price * (1 + self.take_profit_std * daily_vol)
        else:  # Short position
            stop_loss = price * (1 + self.stop_loss_std * daily_vol)
            take_profit = price * (1 - self.take_profit_std * daily_vol)
        
        return stop_loss, take_profit


def backtest_strategy(
    prices: pd.Series,
    predictions: pd.Series,
    volatility: pd.Series,
    strategy: VolatilityAdjustedStrategy,
    initial_capital: float = 10000,
    transaction_cost: float = 0.0001
) -> pd.DataFrame:
    """
    Backtest a trading strategy.
    
    Parameters:
    -----------
    prices : pd.Series
        Asset prices
    predictions : pd.Series
        Predicted returns
    volatility : pd.Series
        Annualized volatility
    strategy : VolatilityAdjustedStrategy
        Trading strategy
    initial_capital : float
        Initial capital
    transaction_cost : float
        Transaction cost as a fraction of trade value
        
    Returns:
    --------
    pd.DataFrame
        Backtest results
    """
    # Initialize results DataFrame
    results = pd.DataFrame(index=prices.index)
    results['price'] = prices
    results['prediction'] = predictions
    results['volatility'] = volatility
    
    # Initialize state variables
    position = 0
    capital = initial_capital
    shares = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    
    # Initialize results columns
    results['signal'] = 0
    results['position'] = 0
    results['position_size'] = 0
    results['shares'] = 0
    results['capital'] = initial_capital
    results['equity'] = initial_capital
    results['returns'] = 0
    results['strategy_returns'] = 0
    results['trade_count'] = 0
    results['stop_loss'] = 0
    results['take_profit'] = 0
    
    # Loop through each day
    for i in range(1, len(results)):
        # Get current data
        current_price = results['price'].iloc[i]
        previous_price = results['price'].iloc[i-1]
        current_prediction = results['prediction'].iloc[i-1]  # Use previous day's prediction
        current_volatility = results['volatility'].iloc[i-1]  # Use previous day's volatility
        
        # Check for stop loss or take profit
        if position != 0:
            if (position > 0 and previous_price <= stop_loss) or \
               (position < 0 and previous_price >= stop_loss):
                # Stop loss hit
                position = 0
                capital = shares * previous_price + capital
                shares = 0
                results['trade_count'].iloc[i] = results['trade_count'].iloc[i-1] + 1
            elif (position > 0 and previous_price >= take_profit) or \
                 (position < 0 and previous_price <= take_profit):
                # Take profit hit
                position = 0
                capital = shares * previous_price + capital
                shares = 0
                results['trade_count'].iloc[i] = results['trade_count'].iloc[i-1] + 1
        
        # Generate new signal if no position
        if position == 0:
            signal = strategy.generate_signal(current_prediction)
            
            if signal != 0:
                # Calculate position size
                position_size = strategy.calculate_position_size(current_volatility)
                position = signal
                
                # Calculate shares to buy/sell
                trade_value = capital * position_size
                shares = trade_value / current_price * position
                
                # Apply transaction costs
                capital -= abs(shares) * current_price * transaction_cost
                
                # Set stop loss and take profit
                stop_loss, take_profit = strategy.set_stop_loss_take_profit(
                    current_price, current_volatility, position)
                
                entry_price = current_price
                results['trade_count'].iloc[i] = results['trade_count'].iloc[i-1] + 1
            else:
                # Carry forward trade count
                results['trade_count'].iloc[i] = results['trade_count'].iloc[i-1]
        else:
            # Carry forward trade count
            results['trade_count'].iloc[i] = results['trade_count'].iloc[i-1]
        
        # Calculate equity and returns
        equity = capital + shares * current_price
        daily_return = (current_price / previous_price) - 1
        
        if position != 0:
            strategy_return = position * daily_return
        else:
            strategy_return = 0
        
        # Update results
        results['signal'].iloc[i] = signal if 'signal' in locals() else 0
        results['position'].iloc[i] = position
        results['position_size'].iloc[i] = position_size if 'position_size' in locals() else 0
        results['shares'].iloc[i] = shares
        results['capital'].iloc[i] = capital
        results['equity'].iloc[i] = equity
        results['returns'].iloc[i] = daily_return
        results['strategy_returns'].iloc[i] = strategy_return
        results['stop_loss'].iloc[i] = stop_loss
        results['take_profit'].iloc[i] = take_profit
    
    # Calculate cumulative returns
    results['cumulative_returns'] = (1 + results['returns']).cumprod()
    results['strategy_cumulative_returns'] = (1 + results['strategy_returns']).cumprod()
    
    return results


def calculate_performance_metrics(
    results: pd.DataFrame,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Calculate performance metrics for a backtest.
    
    Parameters:
    -----------
    results : pd.DataFrame
        Backtest results
    risk_free_rate : float
        Annualized risk-free rate
        
    Returns:
    --------
    Dict[str, float]
        Performance metrics
    """
    # Extract returns
    returns = results['strategy_returns'].dropna()
    
    # Calculate metrics
    total_return = results['strategy_cumulative_returns'].iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    annualized_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
    
    # Calculate drawdown
    cumulative = results['strategy_cumulative_returns']
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    trades = results['trade_count'].diff().fillna(0)
    trade_returns = results['strategy_returns'][trades > 0]
    win_rate = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0
    
    # Calculate profit factor
    gross_profits = trade_returns[trade_returns > 0].sum()
    gross_losses = abs(trade_returns[trade_returns < 0].sum())
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'num_trades': int(trades.sum())
    }


def compare_strategies(
    results_ols: pd.DataFrame,
    results_wls: pd.DataFrame,
    benchmark_returns: pd.Series
) -> pd.DataFrame:
    """
    Compare performance of OLS and WLS strategies against a benchmark.
    
    Parameters:
    -----------
    results_ols : pd.DataFrame
        OLS strategy backtest results
    results_wls : pd.DataFrame
        WLS strategy backtest results
    benchmark_returns : pd.Series
        Benchmark cumulative returns
        
    Returns:
    --------
    pd.DataFrame
        Performance comparison
    """
    # Calculate metrics
    metrics_ols = calculate_performance_metrics(results_ols)
    metrics_wls = calculate_performance_metrics(results_wls)
    
    # Calculate benchmark metrics
    benchmark_total_return = benchmark_returns.iloc[-1] - 1
    benchmark_returns_series = benchmark_returns.pct_change().dropna()
    benchmark_annualized_return = (1 + benchmark_total_return) ** (252 / len(benchmark_returns_series)) - 1
    benchmark_volatility = benchmark_returns_series.std() * np.sqrt(252)
    benchmark_sharpe = benchmark_annualized_return / benchmark_volatility if benchmark_volatility != 0 else 0
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'OLS_Strategy': [
            metrics_ols['total_return'],
            metrics_ols['annualized_return'],
            metrics_ols['annualized_volatility'],
            metrics_ols['sharpe_ratio'],
            metrics_ols['max_drawdown'],
            metrics_ols['win_rate'],
            metrics_ols['profit_factor'],
            metrics_ols['num_trades']
        ],
        'WLS_Strategy': [
            metrics_wls['total_return'],
            metrics_wls['annualized_return'],
            metrics_wls['annualized_volatility'],
            metrics_wls['sharpe_ratio'],
            metrics_wls['max_drawdown'],
            metrics_wls['win_rate'],
            metrics_wls['profit_factor'],
            metrics_wls['num_trades']
        ],
        'Benchmark': [
            benchmark_total_return,
            benchmark_annualized_return,
            benchmark_volatility,
            benchmark_sharpe,
            (benchmark_returns / benchmark_returns.cummax() - 1).min(),
            None,
            None,
            None
        ]
    }, index=[
        'Total Return',
        'Annualized Return',
        'Annualized Volatility',
        'Sharpe Ratio',
        'Maximum Drawdown',
        'Win Rate',
        'Profit Factor',
        'Number of Trades'
    ])
    
    return comparison