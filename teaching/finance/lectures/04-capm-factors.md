# Lecture 04 — CAPM, Beta, and Empirical Factor Models

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Derive CAPM implications; estimate beta and alpha; interpret factor models and distinguish systematic from idiosyncratic risk.

## 1. CAPM
Under its benchmark assumptions, the CAPM implies
\[
E(R_i)-R_f=\beta_i[E(R_M)-R_f],\quad
\beta_i=\frac{\operatorname{Cov}(R_i,R_M)}{\operatorname{Var}(R_M)}.
\]
Beta measures covariance exposure to the market portfolio, not total volatility.

## 2. Security market line
The security market line maps beta into expected return. Assets above the line have positive model-implied alpha; assets below it have negative alpha. This interpretation is conditional on the benchmark model and measurement choices.

## 3. Time-series regression
Estimate
\[
R_{it}-R_{ft}=\alpha_i+\beta_i(R_{Mt}-R_{ft})+\varepsilon_{it}.
\]
The intercept is Jensen's alpha. Standard errors should account for heteroskedasticity and serial dependence where present. Nonsynchronous trading can bias beta estimates.

## 4. Multifactor models
A linear factor model is
\[
R_{it}-R_{ft}=\alpha_i+\boldsymbol\beta_i'\mathbf f_t+\varepsilon_{it}.
\]
Fama-French factors include market, size, and value; later specifications add profitability and investment. Momentum is another commonly studied factor. Factors can be traded portfolios or constructed mimicking portfolios.

## 5. Cross-sectional tests
Fama-MacBeth procedures estimate factor prices in repeated cross-sectional regressions. Inference must consider generated regressors, beta estimation error, and cross-sectional dependence.

## 6. Worked example
If \(R_f=3\%\), market expected return is 8%, and beta is 1.2, CAPM expected return is 9%. Explain why an asset with low standalone variance can still have high beta.

## 7. Computational lab
Estimate rolling market betas and CAPM/factor regressions. Report confidence intervals, residual diagnostics, factor definitions, and out-of-sample alpha stability.

## Problem set
1. Derive beta from covariance and variance.
2. Interpret Jensen's alpha.
3. Compare CAPM and a multifactor specification.
4. Explain beta measurement error.

## Further reading
Sharpe (1964); Lintner (1965); Fama and French (1993, 2015); Campbell, Lo and MacKinlay.
