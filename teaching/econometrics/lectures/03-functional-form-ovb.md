# Lecture 03 — Functional Form, Partial Effects, and Omitted-Variable Bias

**Core question:** What does a regression coefficient measure when the model contains transformations, interactions, or omitted determinants?

## Objectives
Interpret common functional forms; derive marginal effects; derive omitted-variable bias; explain the Frisch–Waugh–Lovell (FWL) theorem; select specifications using economic reasoning rather than mechanical fit.

## 1. Functional form is an economic restriction
A linear regression is linear in parameters, not necessarily in variables. Examples include:
- Level-level: \(Y=\beta_0+\beta_1X+u\), where \(\partial E[Y|X]/\partial X=\beta_1\).
- Log-level: \(\log Y=\beta_0+\beta_1X+u\), where a one-unit increase in \(X\) changes \(Y\) by approximately \(100\beta_1\%\) for small \(\beta_1\).
- Level-log: \(Y=\beta_0+\beta_1\log X+u\), where a 1% increase in \(X\) changes \(Y\) by approximately \(0.01\beta_1\) units.
- Log-log: \(\log Y=\beta_0+\beta_1\log X+u\), where \(\beta_1\) is an elasticity.

For large coefficient changes, use exact percentage transformations rather than the small-change approximation.

## 2. Polynomial and interaction models
A quadratic specification \(Y=\beta_0+\beta_1X+\beta_2X^2+u\) has marginal effect \(\beta_1+2\beta_2X\). The turning point is \(-\beta_1/(2\beta_2)\), but it is substantively meaningful only if it lies within the relevant support and the curvature is estimated adequately.

For an interaction,
\[
Y=\beta_0+\beta_1X+\beta_2D+\beta_3(XD)+u,
\]
where \(D\in\{0,1\}\), the slope in group 0 is \(\beta_1\), while the slope in group 1 is \(\beta_1+\beta_3\). The coefficient \(\beta_3\) is the difference in slopes, not the full marginal effect in group 1.

## 3. Omitted-variable bias
Suppose the true model is
\[
Y=\beta_1X_1+\beta_2X_2+u,
\]
but the estimated regression omits \(X_2\). In the scalar case, the probability limit of the short-regression slope is
\[
\operatorname{plim}\widetilde\beta_1
=\beta_1+\beta_2\frac{\operatorname{Cov}(X_1,X_2)}{\operatorname{Var}(X_1)}
\]
when the relevant moments exist and the variables are centered (or an intercept is included). The bias sign depends on both \(\beta_2\) and the covariance. Naming an omitted variable is not enough; one must reason about both signs.

## 4. FWL theorem
To interpret \(\beta_1\) in a multiple regression, residualize both \(Y\) and \(X_1\) with respect to the other controls. The slope from regressing the residualized outcome on the residualized regressor equals the full-regression coefficient. This identifies the variation actually used by OLS.

## 5. Specification diagnostics
Residual plots can reveal nonlinearity, heteroskedasticity, and outliers. RESET tests certain neglected nonlinear combinations but does not identify the correct alternative. Information criteria compare fitted likelihoods with complexity penalties; they do not establish causal validity. Functional form should be motivated by theory, institutional knowledge, and diagnostics jointly.

## Worked illustration
Suppose the true wage equation includes education and experience, with positive returns to experience and positive covariance between education and experience. Omitting experience biases the education coefficient upward in the scalar omitted-variable formula. If the covariance were negative, the direction would reverse, holding the experience effect positive.

## Problem set
1. Derive marginal effects for a quadratic model and identify the turning point.
2. Interpret every coefficient in a model with a binary-group interaction.
3. Derive omitted-variable bias from the two-regressor population projection.
4. Explain why a high \(R^2\) does not establish causal identification.
5. State what RESET can and cannot diagnose.

## Lab
Estimate linear, quadratic, and log-log models; calculate marginal effects at sample percentiles; compare HC3 inference; plot residuals against fitted values. Report the support over which each marginal effect is interpreted.

## Reading
Wooldridge, *Introductory Econometrics*; Angrist and Pischke, *Mostly Harmless Econometrics*; Davidson and MacKinnon, *Econometric Theory and Methods*.