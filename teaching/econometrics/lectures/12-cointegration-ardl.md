# Lecture 12 — Cointegration, VECM, and ARDL Models

## Objectives
Define cointegration; derive the error-correction representation; interpret cointegrating vectors and adjustment speeds; distinguish ARDL bounds testing from unrestricted level regressions.

## 1. Cointegration
If components of \(y_t\) are individually \(I(1)\), a vector \(\beta\ne0\) is cointegrating when \(\beta'y_t\) is \(I(0)\). The stationary combination represents a long-run equilibrium relation, while deviations may be corrected over time.

## 2. VECM representation
For a VAR in levels, the error-correction form is
\[
\Delta y_t=\Pi y_{t-1}+\sum_{j=1}^{p-1}\Gamma_j\Delta y_{t-j}+\varepsilon_t,
\qquad \Pi=\alpha\beta'.
\]
The rank of \(\Pi\) determines the number of linearly independent cointegrating relations under the maintained I(1) framework. \(\beta\) contains long-run vectors; \(\alpha\) contains adjustment coefficients.

## 3. Engle–Granger and Johansen
Engle–Granger estimates a long-run relation and tests residual stationarity, with critical values specific to generated residuals. Johansen's system method uses reduced-rank likelihood methods and trace / maximum-eigenvalue statistics. Deterministic terms and lag structure affect the testing distribution and interpretation.

## 4. ARDL and bounds testing
An ARDL error-correction form can be written with lagged levels and differenced terms. Bounds testing evaluates the joint significance of lagged-level terms under a mixture of I(0)/I(1) regressors. The method does not permit I(2) variables under the standard bounds framework. Deterministic cases, lag selection, sample size, and critical-value choice must be documented.

## 5. Interpretation
A significant error-correction coefficient is not sufficient by itself to establish a credible structural long-run relation. Discuss integration orders, deterministic components, parameter stability, weak exogeneity, and economic theory.

## Problem set
1. Define cointegration and give an example of two I(1) variables with an I(0) spread.
2. Explain the decomposition \(\Pi=\alpha\beta'\).
3. Interpret adjustment coefficient signs in a scalar error-correction model.
4. Compare Engle–Granger and Johansen approaches.
5. State the integration-order restriction in standard ARDL bounds testing.

## Lab
Test integration orders; estimate an Engle–Granger relation, Johansen VECM, and ARDL error-correction model; compare deterministic specifications and residual diagnostics.

## Reading
Engle and Granger (1987); Johansen (1991, 1995); Pesaran, Shin, and Smith (2001); Lütkepohl, *New Introduction to Multiple Time Series Analysis*.