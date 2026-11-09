# Lecture 15 — Empirical Finance and Financial Econometrics

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Estimate return and factor models; model conditional volatility; design event studies; evaluate predictability with careful data controls.

## 1. Return regressions
A factor regression is
\[
R_{it}-R_{ft}=\alpha_i+\beta_i'f_t+\varepsilon_{it}.
\]
Inference should consider heteroskedasticity, serial correlation, cross-sectional dependence, and generated regressors. Rolling estimates reveal parameter instability but introduce window-choice sensitivity.

## 2. Volatility models
ARCH models specify conditional variance as a function of past squared innovations. GARCH(1,1) is
\[
h_t=\omega+\alpha\varepsilon_{t-1}^2+\beta h_{t-1},
\]
with positivity restrictions and covariance-stationarity condition \(\alpha+\beta<1\) in the standard model. Volatility clustering is distinct from predictability of returns.

## 3. Stochastic and realized volatility
Stochastic-volatility models treat latent variance as a stochastic process. Realized variance aggregates high-frequency intraday returns, with market microstructure noise requiring attention at very high sampling frequencies.

## 4. Predictability and cointegration
Predictive regressions relate future returns to instruments but face persistence and small-sample bias. Price levels may be nonstationary; cointegration tests are relevant for linked price or yield series, not a generic substitute for return modeling.

## 5. Research design and data integrity
Avoid look-ahead bias, survivorship bias, delisting omissions, data snooping, stale prices, and non-synchronous trading. Define universe membership as of each date and preserve point-in-time information.

## 6. Forecast evaluation
Use genuine out-of-sample splits, benchmark models, loss functions, and forecast-encompassing tests where appropriate. Report economic performance net of transaction costs, turnover, and implementation constraints.

## 7. Computational lab
Estimate CAPM and multifactor regressions, fit GARCH, produce volatility forecasts, and evaluate out of sample. Include code, data dictionary, software versions, and robustness checks.

## Problem set
1. State GARCH positivity and persistence conditions.
2. Distinguish conditional volatility from expected return.
3. Explain look-ahead and survivorship bias.
4. Design an out-of-sample return-prediction evaluation.

## Further reading
Campbell, Lo and MacKinlay, *The Econometrics of Financial Markets*; Engle (1982); Bollerslev (1986); Tsay, *Analysis of Financial Time Series*.
