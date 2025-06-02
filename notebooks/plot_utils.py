"""
Advanced plotting utilities for FX Volatility Project
Dark-themed professional visualizations for forex analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Try importing plotly for interactive plots
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def set_dark_theme():
    """Set up dark theme for matplotlib plots"""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#1e1e1e',
        'axes.facecolor': '#1e1e1e',
        'savefig.facecolor': '#1e1e1e',
        'figure.figsize': (14, 8),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': '#555555',
        'axes.labelcolor': '#e0e0e0',
        'text.color': '#e0e0e0',
        'xtick.color': '#e0e0e0',
        'ytick.color': '#e0e0e0',
        'axes.edgecolor': '#555555',
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.prop_cycle': plt.cycler('color', ['#ff9500', '#00b3ff', '#00ff88', 
                                               '#ff3366', '#aa88ff', '#ffcc00', 
                                               '#8ec07c', '#fe8019', '#83a598'])
    })
    
    # Configure seaborn plots
    sns.set_theme(style="darkgrid")
    sns.set_palette("bright")
    sns.set_context("notebook", font_scale=1.2)

def apply_theme():
    """Apply dark theme to all plots automatically"""
    set_dark_theme()
    
    # Patch matplotlib's figure function
    original_plt_figure = plt.figure
    def new_plt_figure(*args, **kwargs):
        fig = original_plt_figure(*args, **kwargs)
        return fig
    plt.figure = new_plt_figure
    
    # Patch seaborn's lineplot function
    original_sns_lineplot = sns.lineplot
    def new_sns_lineplot(*args, **kwargs):
        ax = original_sns_lineplot(*args, **kwargs)
        return ax
    sns.lineplot = new_sns_lineplot
    
    return "Dark theme applied to all plots"

def plot_time_series(data, title="Time Series Plot", figsize=(14, 8)):
    """
    Create a professional time series plot with dark theme
    
    Parameters:
    -----------
    data : DataFrame or Series
        Time series data to plot
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(data, pd.DataFrame):
        for column in data.columns:
            ax.plot(data.index, data[column], linewidth=2, label=column)
        ax.legend(loc='best', frameon=True, facecolor='#2e2e2e', edgecolor='#555555')
    else:
        ax.plot(data.index, data, linewidth=2)
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Date', fontsize=14, labelpad=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    plt.tight_layout()
    return fig, ax

def plot_correlation_matrix(data, title="Correlation Matrix", figsize=(12, 10)):
    """
    Create a professional correlation matrix heatmap
    
    Parameters:
    -----------
    data : DataFrame
        Data to calculate correlations from
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib figure
    """
    corr = data.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f")
    
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    
    return fig

def plot_volatility_regimes(data, regimes, pair_name, figsize=(14, 8)):
    """
    Plot price series with volatility regimes highlighted
    
    Parameters:
    -----------
    data : Series
        Price or return data
    regimes : Series
        Regime classifications (same index as data)
    pair_name : str
        Name of the currency pair
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the price/return data
    ax.plot(data.index, data, color='#00b3ff', linewidth=1.5, alpha=0.7)
    
    # Get unique regimes
    unique_regimes = regimes.unique()
    colors = ['#1e88e5', '#ff9800', '#e53935']  # Blue, Orange, Red for Low, Medium, High
    
    # Plot background colors for different regimes
    for i, regime in enumerate(unique_regimes):
        regime_data = regimes == regime
        regime_ranges = []
        start_idx = None
        
        # Find continuous ranges of the same regime
        for idx, val in regime_data.items():
            if val and start_idx is None:
                start_idx = idx
            elif not val and start_idx is not None:
                regime_ranges.append((start_idx, idx))
                start_idx = None
        
        # Add the last range if it ends with this regime
        if start_idx is not None:
            regime_ranges.append((start_idx, regime_data.index[-1]))
        
        # Highlight each range
        for start, end in regime_ranges:
            ax.axvspan(start, end, alpha=0.3, color=colors[i % len(colors)])
    
    # Add legend for regimes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i % len(colors)], alpha=0.3, 
                            label=f'Regime {regime}')
                      for i, regime in enumerate(unique_regimes)]
    ax.legend(handles=legend_elements, loc='upper left')
    
    ax.set_title(f'{pair_name} with Volatility Regimes', fontsize=16, pad=20)
    ax.set_xlabel('Date', fontsize=14, labelpad=10)
    ax.set_ylabel('Value', fontsize=14, labelpad=10)
    ax.grid(True, alpha=0.3)
    
    fig.autofmt_xdate()
    plt.tight_layout()
    
    return fig

def plot_model_comparison(comparison_df, figsize=(14, 8)):
    """
    Plot comparison of model coefficients (OLS vs WLS)
    
    Parameters:
    -----------
    comparison_df : DataFrame
        DataFrame with model comparison data
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract feature names and coefficients
    features = comparison_df.index
    ols_coefs = comparison_df['OLS']
    wls_coefs = comparison_df['WLS']
    
    # Set positions for bars
    x = np.arange(len(features))
    width = 0.35
    
    # Create bars
    ax.bar(x - width/2, ols_coefs, width, label='OLS', color='#00b3ff', alpha=0.8)
    ax.bar(x + width/2, wls_coefs, width, label='WLS', color='#ff9500', alpha=0.8)
    
    # Add zero line
    ax.axhline(y=0, color='#555555', linestyle='-', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Features', fontsize=14, labelpad=10)
    ax.set_ylabel('Coefficient Value', fontsize=14, labelpad=10)
    ax.set_title('OLS vs WLS Coefficient Comparison', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.legend()
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_3d_scatter(x, y, z, color=None, size=None, title="3D Scatter Plot", 
                   xlabel="X", ylabel="Y", zlabel="Z"):
    """
    Create a professional 3D scatter plot
    
    Parameters:
    -----------
    x, y, z : array-like
        Coordinates for scatter points
    color : array-like, optional
        Values for color mapping
    size : array-like, optional
        Values for size mapping
    title, xlabel, ylabel, zlabel : str
        Plot labels
        
    Returns:
    --------
    fig : plotly figure or matplotlib figure
    """
    if PLOTLY_AVAILABLE:
        if color is None:
            color = z
            
        if size is None:
            size = 5
        elif isinstance(size, (list, np.ndarray)):
            # Normalize size between 5 and 15
            size = 5 + 10 * (size - np.min(size)) / (np.max(size) - np.min(size) + 1e-10)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=size,
                color=color,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Color Scale"),
                line=dict(width=0.5, color='white')
            )
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                zaxis_title=zlabel,
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1)
            ),
            width=1000,
            height=800,
            template='plotly_dark'
        )
        
        return fig
    else:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if color is None:
            color = z
            
        if size is None:
            size = 50
        elif isinstance(size, (list, np.ndarray)):
            # Normalize size between 20 and 200
            size = 20 + 180 * (size - np.min(size)) / (np.max(size) - np.min(size) + 1e-10)
        
        scatter = ax.scatter(x, y, z, c=color, s=size, cmap=cm.viridis, alpha=0.7)
        
        ax.set_xlabel(xlabel, labelpad=10)
        ax.set_ylabel(ylabel, labelpad=10)
        ax.set_zlabel(zlabel, labelpad=10)
        ax.set_title(title, pad=20)
        
        fig.colorbar(scatter, shrink=0.5, aspect=10)
        
        return fig

# Apply dark theme by default when imported
set_dark_theme()