# Lecture 13 — Financial Risk Management: VaR, Expected Shortfall, and Stress Testing

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Quantify market and credit risk; estimate VaR and expected shortfall; understand tail dependence, model risk, and stress testing.

## 1. Value at Risk
For loss \(L\), confidence level \(\alpha\), VaR is the loss quantile
\[
VaR_\alpha(L)=\inf\{\ell:P(L\leq\ell)\geq\alpha\}.
\]
It does not describe the severity of losses beyond the quantile.

## 2. Expected shortfall
For continuous distributions,
\[
ES_\alpha(L)=E[L\mid L\geq VaR_\alpha(L)].
\]
For distributions with atoms, use a quantile-integral definition to handle probability mass at VaR. ES summarizes tail severity beyond the threshold.

## 3. Estimation methods
Historical simulation revalues positions under observed changes. Variance-covariance methods impose distributional assumptions, often multivariate normality. Monte Carlo simulation generates risk factors from a specified stochastic model and reprices the portfolio.

## 4. Credit and counterparty risk
Credit portfolio risk depends on default probabilities, loss severity, exposure, and dependence. Counterparty exposure can vary with market values and netting agreements. Wrong-way risk occurs when exposure and counterparty default risk increase together.

## 5. Stress testing and model risk
Stress tests apply coherent adverse scenarios; reverse stress testing identifies conditions that cause failure. Backtesting compares realized losses with risk forecasts. Model risk includes misspecification, parameter uncertainty, implementation error, and unstable correlations.

## 6. Worked example
Given ordered portfolio losses, calculate empirical 95% VaR and average losses in the worst 5%, stating the finite-sample quantile convention.

## 7. Computational lab
Estimate historical and parametric VaR/ES. Backtest exceedances, compare normal and heavy-tailed innovations, and run a correlated market shock scenario.

## Problem set
1. Define VaR precisely.
2. Explain why ES captures tail severity.
3. Compare historical and parametric simulation.
4. Design a stress test for a leveraged portfolio.

## Further reading
Jorion, *Value at Risk*; McNeil, Frey and Embrechts, *Quantitative Risk Management*.
