# Lecture 08 — Dynamic Panels, Difference GMM, and System GMM

## Objectives
Explain Nickell bias; derive differenced moment conditions; understand instrument timing; assess instrument proliferation; interpret specification tests.

## 1. Dynamic panel
\[
y_{it}=\rho y_{i,t-1}+x_{it}'\beta+\alpha_i+u_{it}.
\]
With fixed \(T\), the within-transformed lagged dependent variable is correlated with the transformed error. This creates Nickell bias, which does not vanish as the number of units grows while \(T\) remains fixed.

## 2. Difference GMM
First differencing removes \(\alpha_i\):
\[
\Delta y_{it}=\rho\Delta y_{i,t-1}+\Delta x_{it}'\beta+\Delta u_{it}.
\]
For a predetermined or endogenous lagged outcome, deeper lags in levels may serve as instruments under specified serial-correlation and timing assumptions. The moment conditions must be stated explicitly; instrument validity is not automatic.

## 3. System GMM
System GMM combines differenced and levels equations, adding moment restrictions in levels that require additional assumptions about initial conditions and changes in the variables. It may improve relevance when series are persistent, but the extra moments need substantive justification.

## 4. Diagnostics and instrument count
Report AR(1) and AR(2) tests in differenced residuals, Hansen/Sargan tests where appropriate, instrument count, number of groups, and sensitivity to collapsing or restricting lag depth. Too many instruments can overfit endogenous regressors and weaken specification tests.

## Problem set
1. Explain the correlation that generates Nickell bias.
2. Derive the differenced equation.
3. State a valid lagged-level moment condition and its timing assumptions.
4. Explain why system GMM adds assumptions beyond difference GMM.
5. Discuss instrument proliferation and design a parsimonious instrument strategy.

## Lab
Estimate a dynamic panel with restricted lag ranges; report instrument count relative to groups, AR(1)/AR(2), Hansen diagnostics, and sensitivity to collapsed instruments.

## Reading
Arellano and Bond (1991); Blundell and Bond (1998); Roodman (2009); Arellano, *Panel Data Econometrics*.