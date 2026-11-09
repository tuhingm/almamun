# Lecture 05 — Confidence Intervals and Hypothesis Testing

## Learning objectives
Construct confidence intervals; formulate tests; derive size and power; interpret p-values; understand likelihood-ratio, Wald, and score tests; distinguish statistical and practical significance.

## 1. Confidence intervals
A \((1-\alpha)\) confidence procedure satisfies
\[
P_\theta\{L(X)\le\theta\le U(X)\}\ge1-\alpha.
\]
The parameter is fixed and the interval is random before sampling. Coverage is a repeated-sampling property, not a posterior probability assigned to a realized interval.

## 2. Testing framework
Specify null \(H_0\), alternative \(H_1\), statistic, rejection region, and level \(\alpha\). Type I error is rejection under the null; Type II error is non-rejection under a specified alternative; power is \(1-\beta\). Power depends on the alternative, sample size, variance, design, and test.

## 3. p-values
A p-value is the probability, under the null model, of a test statistic at least as extreme as the observed statistic under the selected extremeness ordering. It is not the probability that the null is true, nor a measure of effect magnitude.

## 4. Exact mean inference
For i.i.d. normal observations with unknown variance,
\[
T=\frac{\bar X-\mu_0}{S/\sqrt n}\sim t_{n-1}
\]
under the null. The two-sided interval is \(\bar X\pm t_{1-\alpha/2,n-1}S/\sqrt n\).

## 5. Likelihood-based tests
The likelihood-ratio statistic is \(2[\ell(\widehat\theta)-\ell(\widetilde\theta)]\), where the second estimate is restricted under the null. Under regularity and \(q\) restrictions it converges to \(\chi_q^2\). Wald tests use unrestricted estimates and covariance; score tests evaluate the score under the restricted estimate. Their finite-sample behavior can differ.

## 6. Multiple testing
Bonferroni and Holm procedures control family-wise error; false-discovery-rate methods target the expected proportion of false rejections among rejections under their conditions. Report estimates, uncertainty intervals, and substantive scale—not p-values alone.

## Worked example
If \(\bar X=12\), \(S=4\), and \(n=25\), the standard error is 0.8. For \(H_0:\mu=10\), the t statistic is 2.5 with 24 degrees of freedom. Interpret the test together with the estimated difference of two units and its interval.

## Problem set
1. Construct a t interval from a sample summary.
2. Explain 95% repeated-sampling coverage.
3. Distinguish Type I error, Type II error, and power.
4. Compare LR, Wald, and score tests.
5. Explain why a p-value is not the posterior probability of the null.

## Computational lab
Simulate null and alternative samples; estimate empirical size and power; compare LR, Wald, and score tests in finite samples; demonstrate multiplicity under repeated testing.

## Reading
Casella and Berger, *Statistical Inference*; Lehmann and Romano, *Testing Statistical Hypotheses*; Wasserstein and Lazar, “The ASA Statement on p-Values.”