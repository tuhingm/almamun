# Lecture 04 — Heteroskedasticity, Dependence, and Robust Inference

## Objectives
Derive the sandwich covariance; distinguish heteroskedasticity from serial and cluster dependence; understand HC estimators and cluster-robust inference; select covariance estimators from the sampling design.

## 1. General variance expression
With \(\widehat\beta-\beta=(X'X)^{-1}X'u\), conditional covariance is
\[
\operatorname{Var}(\widehat\beta|X)
=(X'X)^{-1}X'\Omega X(X'X)^{-1},
\]
where \(\Omega=\operatorname{Var}(u|X)\). The classical formula follows only when \(\Omega=\sigma^2I\).

## 2. Heteroskedasticity
If observations are independent but have unequal conditional variances, \(\Omega=\operatorname{diag}(\sigma_1^2,\ldots,\sigma_n^2)\). HC0 replaces unknown variances with squared residuals:
\[
\widehat V_{HC0}=(X'X)^{-1}\left(\sum_i x_ix_i'\widehat u_i^2\right)(X'X)^{-1}.
\]
HC1 applies a degrees-of-freedom correction. HC2 divides squared residuals by \(1-h_{ii}\); HC3 divides by \((1-h_{ii})^2\), where \(h_{ii}\) is leverage.

## 3. Dependence changes the covariance structure
For time series or clustered observations, off-diagonal elements of \(\Omega\) need not be zero. HC estimators alone do not handle serial or within-cluster correlation. Cluster-robust covariance aggregates score contributions within clusters. Its validity depends primarily on a sufficiently large number of independent clusters; few-cluster settings require special methods and careful design-based reasoning.

## 4. HAC covariance
For ordered time-series observations, heteroskedasticity-and-autocorrelation-consistent estimators estimate long-run score covariance using weighted sample autocovariances. Bandwidth and kernel choices affect finite-sample behavior. HAC inference requires appropriate weak-dependence and asymptotic conditions; it is not a universal repair for nonstationarity or structural breaks.

## 5. Testing heteroskedasticity
Breusch–Pagan-type tests examine whether conditional variance depends on specified regressors. White-type tests use broader functions of regressors. A failure to reject does not prove homoskedasticity. Robust standard errors are often used as a defensible default when the independent-observation framework applies.

## Problem set
1. Derive the sandwich covariance from the OLS estimation error.
2. Explain why HC3 changes standard errors but not coefficients.
3. Distinguish HC, cluster-robust, and HAC covariance estimators.
4. Explain why 12 clusters may be insufficient for conventional cluster asymptotics.
5. State the dependence structure in a panel where clustering by individual is appropriate.

## Lab
Simulate heteroskedastic errors; compare classical, HC0, HC3, and cluster-robust uncertainty. Report the number of independent clusters and the covariance assumptions.

## Reading
White (1980); MacKinnon and White (1985); Cameron and Miller (2015), “A Practitioner’s Guide to Cluster-Robust Inference”; Newey and West (1987).