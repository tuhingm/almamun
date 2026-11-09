# Lecture 02 — OLS Inference: Sampling Distributions, Hypothesis Tests, and Confidence Intervals

**Course:** Econometrics: Theory, Estimation, Identification and Applications  
**Level:** Advanced undergraduate / graduate  
**Lecture format:** 90-minute theory lecture + 60-minute problem session

## Central question

An estimator is a random variable because it depends on a random sample. How can we quantify its uncertainty, test economic restrictions, and distinguish exact finite-sample inference from large-sample approximation?

## Learning objectives

By the end of this lecture, students should be able to:

1. derive the conditional sampling distribution of the OLS estimator under Gaussian errors;
2. distinguish the Gauss–Markov result from normal-theory inference;
3. derive standard errors, t statistics, and F/Wald tests;
4. construct and interpret confidence intervals;
5. explain the role of asymptotic normality when errors are not Gaussian;
6. identify when conventional standard errors are invalid.

## 1. Sampling uncertainty

Consider the linear model

\[
y=X\beta_0+u,
\]

where \(y\in\mathbb R^n\), \(X\in\mathbb R^{n\times k}\), and \(X\) has full column rank. The OLS estimator is

\[
\widehat\beta=(X'X)^{-1}X'y
=\beta_0+(X'X)^{-1}X'u.
\]

For a fixed design matrix \(X\), the randomness in \(\widehat\beta\) comes from \(u\). Under

\[
\mathbb E[u\mid X]=0,
\qquad \operatorname{Var}(u\mid X)=\sigma^2I_n,
\]

we obtain

\[
\mathbb E[\widehat\beta\mid X]=\beta_0,
\qquad
\operatorname{Var}(\widehat\beta\mid X)
=\sigma^2(X'X)^{-1}.
\]

The diagonal elements of this covariance matrix are the variances of individual coefficient estimators. Their square roots are the conditional standard errors when \(\sigma^2\) is known.

## 2. Estimating the disturbance variance

The residual vector is \(\widehat u=M_Xy\), where

\[
M_X=I_n-X(X'X)^{-1}X'.
\]

The residual sum of squares is

\[
RSS=\widehat u'\widehat u.
\]

Under the classical homoskedastic model, an unbiased estimator of \(\sigma^2\) is

\[
\widehat\sigma^2=\frac{RSS}{n-k}.
\]

Why \(n-k\)? The projection matrix \(P_X\) has rank \(k\), while the residual-maker \(M_X\) has rank and trace \(n-k\). Under the model, the residual degrees of freedom are therefore \(n-k\).

The estimated covariance matrix is

\[
\widehat{\operatorname{Var}}(\widehat\beta\mid X)
=\widehat\sigma^2(X'X)^{-1}.
\]

For coefficient \(j\),

\[
se(\widehat\beta_j)
=\sqrt{\widehat\sigma^2[(X'X)^{-1}]_{jj}}.
\]

## 3. Exact finite-sample distribution under normality

Suppose, in addition, that

\[
u\mid X\sim N(0,\sigma^2I_n).
\]

Then

\[
\widehat\beta\mid X
\sim N\left(\beta_0,\sigma^2(X'X)^{-1}\right).
\]

Moreover, the scaled residual sum of squares satisfies

\[
\frac{RSS}{\sigma^2}\mid X\sim\chi^2_{n-k},
\]

and is independent of \(\widehat\beta\). Consequently, under the null hypothesis \(H_0:\beta_j=\beta_{j,0}\),

\[
t=
\frac{\widehat\beta_j-\beta_{j,0}}
{se(\widehat\beta_j)}
\sim t_{n-k}.
\]

This is an exact finite-sample result conditional on \(X\), under the stated Gaussian, homoskedastic, uncorrelated-error model. Normality is not required for OLS unbiasedness or the Gauss–Markov theorem.

## 4. Testing a single coefficient

For a two-sided test,

\[
H_0:\beta_j=\beta_{j,0},
\qquad
H_1:\beta_j\ne\beta_{j,0},
\]

compute

\[
t_{obs}=\frac{\widehat\beta_j-\beta_{j,0}}{se(\widehat\beta_j)}.
\]

At significance level \(\alpha\), reject when

\[
|t_{obs}|>t_{1-\alpha/2,n-k}.
\]

The p-value is the probability, under the null model, of observing a test statistic at least as extreme as the realized statistic. It is not the probability that the null hypothesis is true.

**Worked example.** Suppose \(\widehat\beta_1=0.42\), \(se(\widehat\beta_1)=0.15\), and the null is \(\beta_1=0\). Then

\[
t_{obs}=0.42/0.15=2.80.
\]

For a sufficiently large residual degrees of freedom, this is evidence against the zero-coefficient null at the 5% two-sided level. The exact p-value must be computed using the appropriate t distribution and degrees of freedom.

## 5. Confidence intervals

A two-sided \(100(1-\alpha)\%\) confidence interval is

\[
\widehat\beta_j
\pm t_{1-\alpha/2,n-k}se(\widehat\beta_j).
\]

A 95% confidence procedure has 95% coverage under repeated sampling when its assumptions hold. It does not mean that the fixed parameter has a 95% probability of lying in the realized interval in the frequentist framework.

The test of \(H_0:\beta_j=0\) at level 5% is equivalent to checking whether the corresponding 95% two-sided confidence interval contains zero, provided the same standard error and reference distribution are used.

## 6. Joint restrictions and the F test

Economic theory often implies multiple restrictions. Write them as

\[
H_0:R\beta=r,
\]

where \(R\in\mathbb R^{q\times k}\) has rank \(q\). The Wald quadratic form is

\[
W=(R\widehat\beta-r)'
\left[R\widehat V R'\right]^{-1}
(R\widehat\beta-r),
\]

where \(\widehat V\) is the estimated covariance matrix of \(\widehat\beta\).

Under the classical normal linear model,

\[
F=\frac{W}{q}\sim F_{q,n-k}.
\]

An equivalent restricted-versus-unrestricted expression is

\[
F=
\frac{(RSS_R-RSS_U)/q}{RSS_U/(n-k)},
\]

when both models are nested and estimated by OLS under the same sample and classical assumptions. Here \(RSS_R\) and \(RSS_U\) are the restricted and unrestricted residual sums of squares.

**Economic application.** To test whether a set of regional indicators has no explanatory contribution, jointly test that all corresponding coefficients equal zero. Individual t tests do not answer this joint question.

## 7. Asymptotic inference

Exact normal-theory distributions rely on stronger assumptions than large-sample approximations. Under suitable regularity conditions, random sampling, finite moments, and exogeneity,

\[
\sqrt n(\widehat\beta-\beta_0)
\xrightarrow{d}N(0,Q^{-1}\Omega Q^{-1}),
\]

where

\[
Q=\operatorname{plim}\frac{X'X}{n},
\qquad
\Omega=\operatorname{plim}\frac{1}{n}\sum_{i=1}^n x_ix_i'u_i^2
\]

in the independent-observation heteroskedastic setting. Under homoskedasticity, \(\Omega=\sigma^2Q\), giving the familiar covariance simplification.

The asymptotic result motivates normal critical values in large samples. It does not automatically validate inference under arbitrary dependence, weak identification, heavy tails, or misspecification.

## 8. Heteroskedasticity and the sandwich covariance

If observations are independent but

\[
\operatorname{Var}(u_i\mid x_i)=\sigma_i^2,
\]

then the conventional homoskedastic covariance formula is generally incorrect. A heteroskedasticity-consistent estimator is

\[
\widehat V_{HC0}
=(X'X)^{-1}
\left(\sum_{i=1}^n x_ix_i'\widehat u_i^2\right)
(X'X)^{-1}.
\]

HC1 applies a degrees-of-freedom correction; HC2 and HC3 additionally adjust for leverage. Robust covariance estimation changes estimated uncertainty, not the OLS coefficient itself. It does not correct endogeneity, omitted-variable bias, sample selection, or an incorrect conditional mean.

## 9. Research interpretation

A complete empirical report should distinguish:
- **Estimate:** the point estimate and its units;
- **Uncertainty:** standard error and confidence interval;
- **Test:** null, alternative, statistic, reference distribution, and p-value;
- **Economic magnitude:** whether the effect is substantively meaningful;
- **Identification:** assumptions that support the intended interpretation;
- **Covariance structure:** why the selected standard error estimator is valid for the sampling design.

Statistical significance is not a substitute for effect-size interpretation or identification.

## 10. Problem session

**Problem 1.** Starting from \(\widehat\beta=\beta_0+(X'X)^{-1}X'u\), derive its conditional expectation and variance under the classical assumptions.

**Problem 2.** Explain why the residual degrees of freedom are \(n-k\), using the rank of the residual-maker matrix.

**Problem 3.** Under Gaussian errors, state the distributions of \(\widehat\beta\), \(RSS/\sigma^2\), and the t statistic. Identify the independence result needed to obtain the t distribution.

**Problem 4.** A coefficient estimate is 0.30 with standard error 0.12. Calculate the t statistic for a zero null and explain what additional information is needed for an exact p-value.

**Problem 5.** Formulate a joint F test for \(H_0:\beta_2=\beta_3=\beta_4=0\). State the dimensions of \(R\) and \(r\).

**Problem 6.** Explain why HC standard errors do not change the OLS point estimates and why they do not solve endogeneity.

**Problem 7.** Distinguish an exact finite-sample t test from an asymptotic z test. Give the assumptions supporting each.

## Computational lab: OLS inference in Python

The following script is fully executable. It estimates OLS, reports classical and HC3 standard errors, constructs confidence intervals, performs a single-coefficient test, and conducts a joint Wald test.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

rng = np.random.default_rng(2026)
n = 500

x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
heteroskedastic_sd = 0.5 + 0.8 * np.abs(x1)
u = rng.normal(size=n) * heteroskedastic_sd
y = 1.0 + 0.4 * x1 - 0.7 * x2 + u

X = pd.DataFrame({"x1": x1, "x2": x2})
X = sm.add_constant(X)

ols = sm.OLS(y, X).fit()
hc3 = ols.get_robustcov_results(cov_type="HC3")

print("CLASSICAL OLS")
print(ols.summary())

print("\\nHC3 ROBUST INFERENCE")
print(hc3.summary())

print("\\nCLASSICAL 95% CONFIDENCE INTERVALS")
print(ols.conf_int(alpha=0.05))

print("\\nHC3 95% CONFIDENCE INTERVALS")
print(hc3.conf_int(alpha=0.05))

# Single restriction: beta_x1 = 0
print("\\nHC3 TEST: beta_x1 = 0")
print(hc3.t_test("x1 = 0"))

# Joint restriction: beta_x1 = beta_x2 = 0
print("\\nHC3 JOINT WALD TEST")
print(hc3.wald_test("x1 = 0, x2 = 0", scalar=True))

# Classical nested-model F test
restricted = sm.OLS(y, X[["const"]]).fit()
print("\\nCLASSICAL NESTED-MODEL F TEST")
print(ols.compare_f_test(restricted))
```

**Lab interpretation.** The disturbance variance in this design depends on \(|x_1|\), so the homoskedastic covariance formula is deliberately misspecified. Compare the classical and HC3 standard errors. The point estimates are identical; the estimated uncertainty can differ.

## Suggested reading

- Wooldridge, *Introductory Econometrics: A Modern Approach*, inference chapters.
- Hayashi, *Econometrics*, linear regression and asymptotic theory.
- Greene, *Econometric Analysis*, linear regression and hypothesis testing.
- White (1980), “A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity.”

**Next lecture:** Functional form, partial effects, omitted-variable bias, and the Frisch–Waugh–Lovell theorem.
