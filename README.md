# FX Volatility Regime Analysis Project

This project demonstrates the practical application of OLS regression assumptions and weighted least squares regression in a quantitative finance context, focusing on foreign exchange markets.

## Project Overview

The FX Volatility Regime Analysis project explores how heteroskedasticity affects foreign exchange markets and demonstrates how weighted least squares (WLS) regression can improve prediction accuracy and trading performance compared to ordinary least squares (OLS) regression.

## Key Features

- Detection and visualization of heteroskedasticity in FX returns
- Identification of distinct volatility regimes using clustering techniques
- Implementation of both OLS and WLS regression models
- Volatility-adjusted trading strategy with dynamic position sizing
- Comprehensive performance analysis across different market regimes

## Project Structure

```
FX_Volatility_Project/
├── data/
│   ├── processed/       # Cleaned and processed data
│   └── raw/             # Raw data files
├── notebooks/
│   ├── 1_data_preparation.ipynb        # Data collection and preprocessing
│   ├── 2_exploratory_analysis.ipynb    # Heteroskedasticity testing and visualization
│   ├── 3_model_development.ipynb       # OLS and WLS model implementation
│   └── 4_strategy_backtesting.ipynb    # Trading strategy backtesting
├── results/
│   ├── figures/         # Generated visualizations
│   ├── models/          # Saved model files
│   └── performance/     # Strategy performance metrics
├── src/
│   ├── backtest.py      # Backtesting framework
│   ├── data_utils.py    # Data processing utilities
│   ├── model_utils.py   # Regression model utilities
│   └── visualization.py # Visualization functions
├── project_details.md   # Detailed project documentation
└── requirements.txt     # Project dependencies
```

## Key Concepts Demonstrated

1. **Heteroskedasticity Detection**: Using visual and statistical methods (Breusch-Pagan and White tests) to identify non-constant error variance in FX returns.

2. **Volatility Regime Identification**: Applying clustering techniques to identify distinct market regimes with different volatility characteristics.

3. **Weighted Least Squares**: Implementing WLS regression to address heteroskedasticity by giving less weight to observations with higher error variance.

4. **Variance Function Estimation**: Modeling the relationship between volatility and prediction error variance to derive optimal weights for WLS.

5. **Volatility-Adjusted Trading**: Developing a trading strategy that dynamically adjusts position sizes based on current market volatility.

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages listed in `requirements.txt`

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fx-volatility-project.git
cd fx-volatility-project

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

The project is organized as a series of Jupyter notebooks that should be run in sequence:

1. `1_data_preparation.ipynb`: Collects and processes FX data and economic indicators
2. `2_exploratory_analysis.ipynb`: Analyzes statistical properties and tests for heteroskedasticity
3. `3_model_development.ipynb`: Implements and compares OLS and WLS regression models
4. `4_strategy_backtesting.ipynb`: Backtests trading strategies based on both models

## Results

The project demonstrates that:

- FX returns exhibit significant heteroskedasticity, with error variance strongly related to market volatility
- WLS regression provides more accurate predictions than OLS, especially during high-volatility periods
- A trading strategy based on WLS predictions outperforms one based on OLS in terms of risk-adjusted returns
- The improvement from WLS is most pronounced during high-volatility market regimes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data provided by Yahoo Finance and FRED
- Inspired by research on heteroskedasticity in financial markets