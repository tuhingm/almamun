# Econometrics — Graduate Problem Sets and Solution Guidance

This document provides selected derivations and answer guidance. Students should show intermediate steps and state assumptions; a numerical answer without an econometric argument is incomplete.

## Problem set A — OLS and projection

### A1. Derive OLS
Minimize \(Q(b)=(y-Xb)'(y-Xb)\). Expanding gives \(Q(b)=y'y-2b'X'y+b'X'Xb\). The FOC is \(-2X'y+2X'Xb=0\). Under full column rank, \(\widehat\beta=(X'X)^{-1}X'y\).

### A2. Prove projection properties
Let \(P_X=X(X'X)^{-1}X'\). Symmetry follows from symmetry of \((X'X)^{-1}\). Idempotence follows by multiplication:
\[
P_X^2=X(X'X)^{-1}X'X(X'X)^{-1}X'=P_X.
\]
Thus \(P_X\) is an orthogonal projection onto the column space of \(X\).

### A3. Residual orthogonality
\[
X'\widehat u=X'(y-X\widehat\beta)=X'y-X'X(X'X)^{-1}X'y=0.
\]
This is a sample normal-equation identity; it does not prove population exogeneity.

## Problem set B — Inference

### B1. OLS covariance
Using \(\widehat\beta-\beta=(X'X)^{-1}X'u\) and \(\operatorname{Var}(u|X)=\sigma^2I\),
\[
\operatorname{Var}(\widehat\beta|X)
=(X'X)^{-1}X'(\sigma^2I)X(X'X)^{-1}
=\sigma^2(X'X)^{-1}.
\]

### B2. Why n-k residual degrees of freedom?
The residual-maker \(M_X=I-P_X\) is idempotent and has rank \(n-k\). Under Gaussian errors, \(\widehat u'\widehat u/\sigma^2\) has a chi-square distribution with \(n-k\) degrees of freedom.

### B3. Single coefficient test
For \(H_0:\beta_j=0\), calculate \(t=\widehat\beta_j/se(\widehat\beta_j)\). The exact reference distribution is \(t_{n-k}\) under the classical normal linear model. Without normality, large-sample approximations require regularity conditions.

## Problem set C — Endogeneity and IV

### C1. Omitted-variable bias
If the true model is \(y=\beta_1x_1+\beta_2x_2+u\), but \(x_2\) is omitted, the short-regression probability limit is
\[
\operatorname{plim}\widetilde\beta_1
=\beta_1+\beta_2\frac{\operatorname{Cov}(x_1,x_2)}{\operatorname{Var}(x_1)}.
\]
The sign requires both the effect of \(x_2\) and its covariance with \(x_1\).

### C2. Just-identified IV
The sample moment condition is \(\sum_i z_i(y_i-\beta x_i)=0\). Solving yields
\[
\widehat\beta_{IV}=\frac{\sum_i z_iy_i}{\sum_i z_ix_i}
\]
for variables centered or with appropriate intercept handling. Consistency requires instrument relevance and orthogonality to the structural error.

### C3. Exclusion defense
A strong answer identifies the institutional mechanism generating instrument variation, explains why it affects the outcome only through the endogenous regressor, and discusses plausible direct channels and sample selection.

## Problem set D — Panel and dynamic panel

### D1. Within estimator
Subtracting individual means from \(y_{it}=x_{it}'\beta+\alpha_i+u_{it}\) removes \(\alpha_i\):
\[
y_{it}-\bar y_i=(x_{it}-\bar x_i)'\beta+(u_{it}-\bar u_i).
\]

### D2. Time-invariant regressors
For a regressor constant within unit, \(x_i-\bar x_i=0\). Its coefficient cannot be separately identified from unit fixed effects.

### D3. Nickell bias
The transformed lagged outcome contains past disturbances, while the transformed error contains current and lagged disturbances. Their correlation persists as \(N\to\infty\) for fixed \(T\).

### D4. GMM validity
A valid moment condition must follow from the timing classification and serial-correlation assumptions. Students must explicitly state which lags instrument which differenced equation and why.

## Problem set E — Time series

### E1. AR(1) moments
For \(y_t=c+\rho y_{t-1}+\varepsilon_t\), with \(|\rho|<1\), the mean solves \(\mu=c+\rho\mu\), hence \(\mu=c/(1-\rho)\). The variance solves \(\gamma_0=\rho^2\gamma_0+\sigma_\varepsilon^2\), hence \(\gamma_0=\sigma_\varepsilon^2/(1-\rho^2)\).

### E2. Cointegration
A vector of I(1) variables is cointegrated if at least one nonzero linear combination is I(0). Cointegration does not mean each series is stationary; it means their long-run deviations are stationary under the specified relation.

### E3. VAR identification
Reduced-form innovations are generally contemporaneously correlated. Structural shocks require additional restrictions that identify a transformation from reduced-form innovations to economically interpretable shocks.

## Problem set F — Causal inference

### F1. Difference-in-differences
The estimator is the treated group's before-after change minus the control group's before-after change. Its causal interpretation requires parallel trends in untreated potential outcomes, together with no anticipation and an appropriate treatment/composition structure.

### F2. Regression discontinuity
A local causal interpretation requires continuity of potential outcomes at the cutoff and no precise manipulation of the running variable that sorts units around the threshold. In fuzzy RD, the discontinuity in treatment receipt instruments actual treatment.

## Grading rubric
- Correct estimand and notation: 20%.
- Derivation and mathematical validity: 30%.
- Identification and assumptions: 20%.
- Interpretation and diagnostics: 20%.
- Reproducibility and presentation: 10%.
