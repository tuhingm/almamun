# Lecture 11 — VAR and Structural VAR: Dynamic Interactions and Shock Identification

## Objectives
Specify VAR systems; assess stability; interpret Granger predictability; derive impulse responses; distinguish reduced-form innovations from structural shocks.

## 1. Reduced-form VAR
\[
y_t=c+A_1y_{t-1}+\cdots+A_py_{t-p}+u_t,
\qquad E(u_tu_t')=\Sigma_u.
\]
A VAR treats each variable as potentially endogenous and models its dependence on lags of all system variables.

## 2. Stability
The VAR is stable when roots of the characteristic polynomial lie outside the unit circle (equivalently, companion-matrix eigenvalues lie inside it). Stability supports a moving-average representation and conventional impulse-response analysis.

## 3. Granger predictability
Variable \(x\) Granger-causes \(y\) in the predictive sense if lags of \(x\) improve prediction of \(y\), conditional on included information. This is not equivalent to structural causality.

## 4. Impulse responses
The moving-average representation is \(y_t=\mu+\sum_{h=0}^{\infty}\Psi_hu_{t-h}\). Reduced-form innovations may be contemporaneously correlated. Orthogonalized or structural responses require an identification scheme.

## 5. Structural identification
Write \(A_0y_t=\cdots+\varepsilon_t\), with \(u_t=A_0^{-1}\varepsilon_t\). Recovering structural shocks requires restrictions: recursive ordering, short-run zero restrictions, long-run restrictions, sign restrictions, or external instruments. The chosen restrictions are substantive identifying assumptions.

## 6. Reporting
Report variables, transformations, lag order, deterministic terms, stability, residual diagnostics, identification restrictions, confidence intervals, and robustness to ordering/specification.

## Problem set
1. State the stability condition for a VAR.
2. Distinguish Granger predictability from structural causality.
3. Explain why reduced-form innovations are not automatically orthogonal.
4. Compare recursive and sign-restricted identification.
5. Interpret a cumulative impulse response.

## Lab
Estimate a VAR; select lag length; test stability; calculate IRFs and FEVDs; compare recursive orderings and clearly state the identifying assumptions.

## Reading
Lütkepohl, *New Introduction to Multiple Time Series Analysis*; Kilian and Lütkepohl, *Structural Vector Autoregressive Analysis*.