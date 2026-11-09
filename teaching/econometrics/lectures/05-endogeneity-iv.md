# Lecture 05 — Endogeneity, Instrumental Variables, and Two-Stage Least Squares

## Objectives
Define endogeneity; derive IV and 2SLS; state relevance and exclusion; diagnose weak instruments; interpret overidentification tests; explain LATE under heterogeneous treatment effects.

## 1. Why OLS can fail
In \(y_i=\beta x_i+u_i\), OLS consistency requires \(\operatorname{Cov}(x_i,u_i)=0\) under standard moment conditions. Omitted variables, simultaneity, measurement error, and selection can violate this condition. Increasing sample size does not eliminate asymptotic bias.

## 2. Instrumental-variable conditions
An instrument \(z_i\) must satisfy:
- Relevance: \(\operatorname{Cov}(z_i,x_i)\ne0\).
- Exogeneity/exclusion: \(\operatorname{Cov}(z_i,u_i)=0\), together with the substantive claim that \(z_i\) affects \(y_i\) only through the endogenous regressor under the model.

Relevance is partly assessed statistically; exclusion requires institutional and economic justification.

## 3. Just-identified IV
With one instrument and one endogenous regressor,
\[
\widehat\beta_{IV}=\frac{\sum_i(z_i-\bar z)(y_i-\bar y)}
{\sum_i(z_i-\bar z)(x_i-\bar x)}.
\]
The ratio uses the instrument-induced variation in \(x\). Under valid relevance and exogeneity conditions, it consistently estimates the structural slope.

## 4. 2SLS matrix derivation
Let \(Z\) contain excluded instruments and included exogenous regressors, and let \(P_Z=Z(Z'Z)^{-1}Z'\). Then
\[
\widehat\beta_{2SLS}=(X'P_ZX)^{-1}X'P_Zy.
\]
The first stage projects endogenous regressors onto the instrument set; the second stage uses fitted endogenous variation. Inference must use the IV/2SLS covariance formula, not ordinary OLS standard errors from manually substituting fitted values.

## 5. Weak instruments
When instruments explain little variation in endogenous regressors, 2SLS can be biased in finite samples and conventional confidence intervals can have poor coverage. Report first-stage diagnostics, partial explanatory power, and weak-IV-robust procedures where appropriate. A rule-of-thumb first-stage F threshold is not a universal guarantee.

## 6. Overidentification
When there are more excluded instruments than endogenous regressors, overidentifying restrictions can be tested using Sargan or Hansen-type statistics under their assumptions. Non-rejection does not prove instrument validity; rejection indicates that at least some maintained restrictions are inconsistent with the data/model.

## 7. LATE intuition
With binary treatment and a valid instrument that shifts treatment, the IV estimand may identify a local average treatment effect for compliers under independence, exclusion, relevance, and monotonicity. It need not equal the population ATE when treatment effects are heterogeneous.

## Problem set
1. Derive the probability limit of the OLS slope under endogeneity.
2. Derive the just-identified IV ratio from sample moment conditions.
3. State the rank condition for 2SLS identification.
4. Explain weak-instrument bias and why a large sample does not cure a weak first stage automatically.
5. Explain the complier population in LATE and why IV need not estimate ATE.

## Lab
Estimate OLS and 2SLS on a documented dataset. Report the first-stage coefficient, partial R-squared, weak-instrument diagnostic, reduced form, 2SLS estimate, robust standard errors, and a substantive defense of exclusion.

## Reading
Angrist and Pischke, *Mostly Harmless Econometrics*; Imbens and Rubin, *Causal Inference*; Stock and Yogo (2005); Staiger and Stock (1997).