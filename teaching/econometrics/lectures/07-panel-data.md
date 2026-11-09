# Lecture 07 — Panel Data: Fixed Effects, Random Effects, and Correlated Effects

## Objectives
Derive within and first-difference estimators; distinguish FE and RE assumptions; understand two-way effects and clustered inference; recognize time-varying confounding.

## 1. Panel model
\[
y_{it}=x_{it}'\beta+\alpha_i+\lambda_t+u_{it},
\]
where \(\alpha_i\) captures time-invariant unit heterogeneity and \(\lambda_t\) common time effects. Panel structure permits control for persistent unobserved heterogeneity, but it does not remove all confounding.

## 2. Within transformation
For a balanced panel, subtract unit means:
\[
y_{it}-\bar y_i=(x_{it}-\bar x_i)'\beta+(u_{it}-\bar u_i).
\]
The time-invariant effect \(\alpha_i\) disappears. Regressors constant within unit are not identified by the within estimator.

## 3. First differences
Taking adjacent differences yields
\[
\Delta y_{it}=\Delta x_{it}'\beta+\Delta u_{it},
\]
which also eliminates \(\alpha_i\). FE and first differences can differ in efficiency depending on the error process and time dimension.

## 4. Random effects and Mundlak
Random effects requires orthogonality between unit effects and regressors across all periods. If this fails, RE is generally inconsistent. Correlated random effects/Mundlak augments the model with unit means of time-varying regressors, allowing a useful decomposition of within and between associations.

## 5. Inference and interpretation
Repeated observations within a unit commonly induce within-unit dependence. Cluster standard errors at the level justified by treatment assignment or error dependence. Fixed effects do not solve time-varying omitted variables, reverse causality, measurement error, or endogenous treatment.

## Problem set
1. Derive the within transformation.
2. Explain why time-invariant regressors are absorbed by unit FE.
3. State the RE orthogonality assumption.
4. Explain the Mundlak device.
5. Identify the correct clustering level in a policy applied at the regional level.

## Lab
Estimate pooled OLS, unit FE, two-way FE, RE, and correlated RE. Compare within and between estimates, cluster inference, and interpret the identifying variation.

## Reading
Wooldridge, *Econometric Analysis of Cross Section and Panel Data*; Baltagi, *Econometric Analysis of Panel Data*; Arellano, *Panel Data Econometrics*.