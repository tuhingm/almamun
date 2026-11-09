# Lecture 09 — Linear Regression and Generalized Linear Models

## Learning objectives
Connect regression to conditional means; derive GLM structure; interpret logistic and Poisson models; assess overdispersion and diagnostics.

## 1. Linear regression
\[
Y_i=x_i'\beta+\varepsilon_i.
\]
OLS estimates the conditional mean under a correctly specified linear conditional expectation. Distinguish prediction, association, and causal interpretation.

## 2. GLM framework
A generalized linear model specifies a random component, systematic component \(\eta_i=x_i'\beta\), and link \(g(\mu_i)=\eta_i\), where \(\mu_i=E(Y_i|x_i)\). The link maps the mean domain to the real line.

## 3. Logistic regression
For binary \(Y_i\),
\[
\log\frac{p_i}{1-p_i}=x_i'\beta.
\]
A one-unit change in \(x_j\) changes log odds by \(\beta_j\); \(e^{\beta_j}\) is the conditional odds ratio holding other covariates fixed. The probability marginal effect depends on covariate values.

## 4. Poisson regression
For count outcomes, the log link gives \(\log E(Y_i|x_i)=x_i'\beta\). Exposure can enter as an offset. Equidispersion is a distributional restriction; robust covariance can address some variance misspecification but does not make the Poisson likelihood fully correct.

## 5. Diagnostics
Assess functional form, influential observations, separation in logistic regression, overdispersion in count models, calibration, and out-of-sample prediction. Report average marginal effects when probability-scale interpretation is required.

## Problem session
1. Interpret a logistic coefficient and odds ratio.
2. Derive the probability marginal effect in a logit model.
3. Explain the role of an exposure offset in Poisson regression.
4. Distinguish overdispersion from heteroskedasticity in a generic linear model.
5. Explain why odds ratios are not risk differences.

## Reading
McCullagh and Nelder, *Generalized Linear Models*; Agresti, *Categorical Data Analysis*; Cameron and Trivedi, *Regression Analysis of Count Data*.