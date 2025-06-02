# FX Volatility Regime Analysis: Detailed Project Documentation

## Project Background

Financial markets, particularly foreign exchange (FX) markets, are characterized by time-varying volatility. This heteroskedasticity poses challenges for traditional Ordinary Least Squares (OLS) regression, which assumes constant error variance. This project demonstrates how Weighted Least Squares (WLS) regression can address heteroskedasticity and improve both prediction accuracy and trading performance.

## Theoretical Framework

### Heteroskedasticity in Financial Markets

Heteroskedasticity refers to the situation where the variance of errors in a regression model varies across observations. In financial markets, this often manifests as:

- Higher volatility during market stress periods
- Lower volatility during stable economic conditions
- Volatility clustering (periods of high volatility tend to persist)

When heteroskedasticity is present, OLS regression remains unbiased but becomes inefficient, producing suboptimal standard errors and potentially misleading hypothesis tests.

### Weighted Least Squares Approach

WLS addresses heteroskedasticity by assigning different weights to observations based on their error variance:

1. Observations with higher error variance receive lower weights
2. Observations with lower error variance receive higher weights

The WLS estimator is given by:

$$\hat{\beta}_{WLS} = (X'WX)^{-1}X'Wy$$

Where:
- $X$ is the matrix of independent variables
- $y$ is the vector of dependent variables
- $W$ is a diagonal matrix of weights (inverse of error variances)

## Methodology

### 1. Data Collection and Preparation

- **FX Data**: Daily exchange rates for major currency pairs (EUR/USD, GBP/USD, USD/JPY, etc.)
- **Economic Indicators**: Interest rates, yield curves, volatility indices, and other macroeconomic variables
- **Feature Engineering**: Returns, volatility measures, lagged variables, and technical indicators

### 2. Heteroskedasticity Detection

- **Visual Methods**: Residual plots, scale-location plots
- **Statistical Tests**: Breusch-Pagan test, White test
- **Volatility Regime Identification**: K-means clustering of rolling volatility

### 3. Model Development

- **OLS Implementation**: Standard regression model with constant weights
- **Variance Function Estimation**: Modeling the relationship between predictors and error variance
- **WLS Implementation**: Regression with weights derived from the variance function

### 4. Trading Strategy

- **Signal Generation**: Based on OLS and WLS predictions
- **Position Sizing**: Inversely proportional to current volatility
- **Risk Management**: Dynamic stop-loss and take-profit levels based on volatility
- **Performance Evaluation**: Comparison of OLS and WLS strategies across different volatility regimes

## Implementation Details

### Data Processing Pipeline

1. **Data Collection**: Using yfinance for FX data and pandas-datareader for economic indicators
2. **Cleaning**: Handling missing values and outliers
3. **Feature Engineering**:
   - Log returns calculation
   - Rolling volatility estimation
   - Creation of lagged features
   - Interest rate differentials

### Variance Function Specification

The variance function is estimated using a log-linear model:

$$\log(\hat{e}^2) = \gamma_0 + \gamma_1 X_1 + \gamma_2 X_2 + ... + \gamma_k X_k + v$$

Where:
- $\hat{e}^2$ are the squared residuals from the OLS regression
- $X_1, X_2, ..., X_k$ are predictors of error variance (e.g., volatility measures)
- $v$ is the error term of the variance model

### Weight Calculation

Weights for the WLS regression are calculated as:

$$w_i = \frac{1}{\exp(\hat{\gamma}_0 + \hat{\gamma}_1 X_{1i} + \hat{\gamma}_2 X_{2i} + ... + \hat{\gamma}_k X_{ki})}$$

These weights are then normalized to maintain the scale of the regression.

### Trading Strategy Implementation

The volatility-adjusted strategy:

1. Generates signals based on model predictions exceeding a threshold
2. Calculates position size inversely proportional to current volatility
3. Sets stop-loss and take-profit levels based on volatility-adjusted standard deviations
4. Tracks performance metrics including returns, Sharpe ratio, and drawdowns

## Key Findings

### Statistical Analysis

- FX returns exhibit significant heteroskedasticity, with error variance strongly related to market volatility
- The variance function shows that VIX index and rolling volatility are strong predictors of error variance
- WLS provides more efficient coefficient estimates with lower standard errors

### Trading Performance

- WLS-based strategy shows improved risk-adjusted returns compared to OLS
- The improvement is most significant during high-volatility regimes
- Position sizing based on volatility helps manage risk effectively across different market conditions

## Limitations and Future Work

### Limitations

- The model assumes that the variance function is correctly specified
- Transaction costs may impact real-world performance
- The approach does not account for potential regime changes in the relationship between predictors and returns

### Future Extensions

- Incorporate GARCH models for more sophisticated volatility modeling
- Explore Bayesian approaches to parameter estimation
- Implement adaptive variance function estimation
- Extend to multivariate systems with cross-currency effects

## References

1. Greene, W. H. (2003). Econometric Analysis. Pearson Education.
2. Campbell, J. Y., Lo, A. W., & MacKinlay, A. C. (1997). The Econometrics of Financial Markets. Princeton University Press.
3. Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. Econometrica, 50(4), 987-1007.
4. White, H. (1980). A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity. Econometrica, 48(4), 817-838.