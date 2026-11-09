# Lecture 09 — Time-Series Foundations: Stationarity, ARMA, and Forecasting

## Objectives
Define weak stationarity; derive AR(1) moments; distinguish deterministic and stochastic trends; understand ARMA representation, diagnostics, and forecast evaluation.

## 1. Time-series stochastic processes
A process \(\{y_t\}\) is weakly stationary if its mean is constant, variance is finite and constant, and covariance depends only on the lag. Stationarity is a property of the stochastic process, not simply a visual judgment from a plotted series.

## 2. AR(1)
\[
y_t=c+\rho y_{t-1}+\varepsilon_t,
\qquad \varepsilon_t\sim WN(0,\sigma_\varepsilon^2).
\]
For \(|\rho|<1\), the mean is \(\mu=c/(1-\rho)\), and variance is \(\sigma_\varepsilon^2/(1-\rho^2)\). The autocorrelation at lag \(h\) is \(\rho^h\). If \(\rho=1\), the process has a unit root and is not covariance stationary under the usual innovation assumptions.

## 3. Trends and spurious regression
A deterministic trend is modeled explicitly as a time function; a stochastic trend accumulates shocks. Regressing unrelated nonstationary series in levels can produce apparently strong relationships and invalid conventional inference. Cointegration is a distinct case in which a linear combination of nonstationary variables is stationary.

## 4. ARMA and identification
ARMA(p,q) combines autoregressive and moving-average dynamics. ACF and PACF patterns are diagnostic aids, not infallible identification rules. Model selection should consider theory, information criteria, residual whiteness, parameter stability, and forecast performance.

## 5. Forecast evaluation
Use genuine out-of-sample or rolling-origin evaluation. Compare forecasts using MAE, RMSE, and scale-aware measures where appropriate. Forecast accuracy and causal identification are different objectives.

## Problem set
1. Derive the stationary mean and variance of an AR(1).
2. Explain the unit-root case.
3. Distinguish deterministic and stochastic trends.
4. Explain why unrelated random walks can generate spurious regression.
5. Design a rolling forecast evaluation.

## Lab
Simulate stationary AR(1) and random-walk processes; inspect ACF/PACF; estimate AR models; test residual autocorrelation; compare rolling forecasts.

## Reading
Hamilton, *Time Series Analysis*; Enders, *Applied Econometric Time Series*; Brockwell and Davis, *Introduction to Time Series and Forecasting*.