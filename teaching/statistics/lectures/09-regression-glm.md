# Lecture 09 — Linear Regression and Generalized Linear Models

## Learning objectives
Derive the linear model; interpret conditional mean parameters; specify GLMs; interpret logistic and Poisson regression; assess overdispersion and diagnostics.

## 1. Linear regression
Write \(Y_i=x_i'\beta+\varepsilon_i\), with \(E(\varepsilon_i\mid x_i)=0\). Then \(E(Y_i\mid x_i)=x_i'\beta\). Under full column rank, OLS is \(\widehat\beta=(X'X)^{-1}X'Y\). Regression association is not automatically causal; causal interpretation requires an identification argument.

## 2. GLM structure
A generalized linear model specifies a response distribution from the exponential family, linear predictor \(\eta_i=x_i'\beta\), and link \(g(\mu_i)=\eta_i\), where \(\mu_i=E(Y_i\mid x_i)\). The variance function links conditional variance to the mean.

## 3. Logistic regression
For binary \(Y_i\),
\[
\log\frac{p_i}{1-p_i}=x_i'\beta.
\]
A one-unit increase in \(x_j\) changes log odds by \(\beta_j\); \(e^{\beta_j}\) is the conditional odds ratio. The probability marginal effect is \(\beta_jp_i(1-p_i)\), which depends on covariates.

## 4. Poisson regression
For counts, \(Y_i\mid x_i\sim Poisson(\mu_i)\), with \(\log\mu_i=x_i'\beta\). Then \(e^{\beta_j}\) is the multiplicative change in the conditional mean for a one-unit change in \(x_j\). An exposure variable enters as an offset \(\log t_i\). Assess the Poisson variance restriction; robust covariance does not repair a misspecified mean or dependence structure.

## 5. Estimation and inference
GLM coefficients are usually estimated by maximum likelihood, often using iteratively reweighted least squares. Sandwich covariance can protect inference against some variance misspecification, subject to suitable independence or clustering assumptions.

## 6. Diagnostics
Assess residual patterns, leverage, influence, calibration, discrimination, overdispersion, excess zeros, and separation. Match diagnostics to the outcome distribution and scientific objective.

## Worked example
If a logistic model gives \(\widehat\beta_j=0.4\), the odds ratio is \(e^{0.4}\approx1.49\). This is not a 49-percentage-point probability increase. Transform updated odds to obtain the probability change at a specified baseline risk.

## Problem set
1. Derive OLS and state its rank condition.
2. Derive the logistic probability from the logit link.
3. Interpret a Poisson coefficient and exposure offset.
4. Explain why an odds ratio is not a risk difference.
5. Identify diagnostics for overdispersion and separation.

## Computational lab
Fit OLS, logistic, and Poisson models. Report coefficient scales and marginal effects; compare model-based and sandwich standard errors; assess calibration and residual diagnostics.

## Reading
McCullagh and Nelder, *Generalized Linear Models*; Agresti, *Foundations of Linear and Generalized Linear Models*; Dobson and Barnett, *An Introduction to Generalized Linear Models*.