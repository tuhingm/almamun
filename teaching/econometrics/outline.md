# Topic Outline

A condensed map of the core definitions and results in Econometrics. Each topic is developed in full in the [lecture notes](lecture-notes.md).


**Level:** Advanced undergraduate and graduate. **Scope:** Probability foundations, regression, inference, endogeneity, IV, panel data, time series, causal inference, and reproducible computation.

## Learning objectives
Students will formulate estimands, derive estimators, state identification assumptions, distinguish finite-sample from asymptotic results, diagnose model failures, and implement reproducible analyses in R, Python, or Stata.

## 1. Probability and asymptotic foundations
Let (Y,X) be random variables. Conditional expectation E(Y|X) is the minimum mean-square predictor among functions of X. Use iterated expectations, total variance, covariance decompositions, LLN, CLT, Slutsky's theorem, continuous mapping, and the delta method. Distinguish convergence in probability from convergence in distribution.

## 2. Linear regression and projection geometry
For y = Xβ + u, OLS minimizes (y-Xb)'(y-Xb), yielding βhat=(X'X)^(-1)X'y when X has full column rank. Define projection P_X=X(X'X)^(-1)X' and residual maker M_X=I-P_X. Then fitted values are P_Xy, residuals are M_Xy, and X'u_hat=0. Under correct linear conditional mean, full rank, E(u|X)=0, and Var(u|X)=σ²I, OLS is BLUE. Normality is not required for Gauss-Markov.

Variance: Var(βhat|X)=σ²(X'X)^(-1); estimate σ² by RSS/(n-k). Derive t and F tests under normality. Explain partial regression and the Frisch–Waugh–Lovell theorem.

## 3. Functional form and interpretation
Levels, log-level, level-log, log-log, polynomials, interactions, dummy variables, elasticities, marginal effects, and nonlinear transformations. For log(y)=β0+β1log(x)+u, β1 is an elasticity. Interpret coefficients conditionally; a regression coefficient is not automatically causal. Cover RESET, residual plots, leverage, studentized residuals, and Cook's distance.

## 4. Inference and robust covariance
Test H0:Rβ=r using Wald, likelihood-ratio, or score procedures. Explain size, power, confidence coverage, p-values, multiple testing, and economic significance. Under heteroskedasticity, use sandwich covariance (X'X)^(-1)[Σ x_i x_i' uhat_i²](X'X)^(-1). HC0-HC3 differ in finite-sample leverage corrections. Robust standard errors do not fix endogeneity or wrong functional form.

## 5. Endogeneity and instrumental variables
Endogeneity arises from omitted variables, simultaneity, measurement error, or selection. Instrument z must satisfy relevance and exclusion/exogeneity. With instrument matrix Z, P_Z=Z(Z'Z)^(-1)Z' and βhat_2SLS=(X'P_ZX)^(-1)X'P_Zy. Explain first-stage partial R², weak-instrument diagnostics, weak-IV robust inference, overidentification tests, and local average treatment effects. Relevance is empirically assessable; exclusion needs institutional and economic justification.

## 6. Simultaneous equations
Structural system By_t+Γx_t=ε_t and reduced form y_t=Πx_t+v_t. Explain order and rank conditions, exclusion restrictions, structural versus reduced-form parameters, 2SLS and 3SLS. A reduced-form residual does not automatically identify a structural shock.

## 7. Panel data
Model y_it=x_it'β+α_i+λ_t+u_it. Derive within transformation and first differences. Compare pooled OLS, fixed effects, random effects, correlated random effects/Mundlak, two-way effects, and clustered inference. Fixed effects remove time-invariant α_i but not time-varying confounding. Discuss serial correlation, unbalanced panels, and few-cluster inference.

## 8. Dynamic panels
For y_it=ρy_i,t-1+x_it'β+α_i+u_it, explain Nickell bias for fixed T, difference GMM, system GMM, timing-based moment restrictions, instrument proliferation, AR(1)/AR(2), Hansen/Sargan tests, instrument collapsing, and sensitivity to instrument count. Students must state the exact moment conditions used.

## 9. Time-series econometrics
Stationarity, weak dependence, ACF/PACF, white noise, deterministic trends, stochastic trends, unit roots, spurious regression, ARMA, lag selection, residual diagnostics, and forecast evaluation. For AR(1), y_t=ρy_t-1+ε_t, covariance stationarity requires |ρ|<1.

## 10. VAR and structural VAR
VAR(p): y_t=c+Σ A_j y_t-j+ε_t. Cover stability, companion form, lag selection, Granger predictability, impulse responses, forecast-error variance decomposition, and structural identification through contemporaneous restrictions, long-run restrictions, sign restrictions, or external instruments. Reduced-form innovations alone do not identify economic shocks.

## 11. Cointegration, VECM and ARDL
Cointegration is a stationary linear combination of nonstationary I(1) variables. VECM: Δy_t=Πy_t-1+ΣΓ_jΔy_t-j+ε_t, with Π=αβ'. Explain rank, adjustment coefficients, Johansen trace and maximum-eigenvalue tests, Engle-Granger, deterministic components, ARDL bounds testing, and integration-order restrictions.

## 12. Causal inference
Potential outcomes Y(1),Y(0), ATE=E[Y(1)-Y(0)], ignorability, overlap, matching, weighting, regression adjustment, DiD, event studies, synthetic control, and regression discontinuity. DiD estimator: (treated post-treated pre)-(control post-control pre). State parallel trends, inspect pre-trends, and account for staggered adoption and heterogeneous effects.

## Worked problem set
1. Differentiate the OLS objective and derive βhat=(X'X)^(-1)X'y.
2. In y=β1x1+β2x2+u, derive omitted-variable bias when x2 is excluded: β1 probability limit shifts by β2 Cov(x1,x2)/Var(x1).
3. Explain relevance versus exclusion for an instrument and provide a plausible exclusion violation.
4. Show within-demeaning removes α_i and explain why time-invariant regressors are unidentified under individual fixed effects.
5. Explain spurious regression and how cointegration changes the interpretation.

## Computational labs
Python: statsmodels OLS, HC3 covariance, IV/2SLS, panel models, VAR and ARDL. R: lm, sandwich, lmtest, fixest, plm, vars, urca. Stata: regress, ivregress, xtreg, xtabond2, var, ardl. Each lab must report sample construction, estimand, assumptions, covariance estimator, diagnostics, and replication steps.

## Assessment
Problem sets 30%; replication project 25%; midterm 20%; final applied paper 25%. Graduate extension requires asymptotic derivations, identification defense, and Monte Carlo comparison.

## References
Wooldridge, Introductory Econometrics and Econometric Analysis of Cross Section and Panel Data; Hayashi, Econometrics; Greene, Econometric Analysis; Angrist and Pischke, Mostly Harmless Econometrics; Hamilton, Time Series Analysis; Lütkepohl, New Introduction to Multiple Time Series Analysis.

## Semester sequence

| Unit | Lecture block | Main analytical output |
|---|---|---|
| I | Probability, sampling, and asymptotics | LLN, CLT, Slutsky, delta method |
| II | Linear regression and OLS geometry | Projection matrices, FWL, Gauss–Markov |
| III | OLS inference and specification | t/F/Wald tests, robust covariance, functional form |
| IV | Endogeneity and identification | OVB, IV/2SLS, weak instruments, control functions |
| V | Simultaneous equations | Structural form, reduced form, order and rank conditions |
| VI | Cross-sectional and panel methods | FE, RE, Mundlak, clustered inference, dynamic panels |
| VII | Time-series foundations | Stationarity, unit roots, ARMA, forecasting |
| VIII | Multivariate time series | VAR/SVAR, cointegration, VECM, ARDL |
| IX | Causal designs | DiD, event studies, synthetic control, RD |
| X | Research implementation | Replication, Monte Carlo, empirical paper |
