# Lecture 05 — Confidence Intervals and Hypothesis Testing

## Core question
How do sampling distributions support calibrated uncertainty statements and decisions about statistical hypotheses?

## Learning objectives
Construct confidence intervals; formulate tests; derive size and power; interpret p-values; understand likelihood-ratio, Wald, and score tests; distinguish statistical from practical significance.

## 1. Confidence procedures
A \((1-\alpha)\) confidence procedure satisfies
\[
P_\theta\{L(X)\le\theta\le U(X)\}\ge1-\alpha
\]
under the specified model. The parameter is fixed and the interval is random before sampling. A 95% procedure has 95% repeated-sampling coverage, not a 95% frequentist probability that a realized fixed interval contains the parameter.

## 2. Hypothesis testing
Specify \(H_0\), \(H_1\), a test statistic, rejection region, and level \(\alpha\). Type I error is rejection under the null; Type II error is non-rejection under a specified alternative. Power is \(1-\beta\). A test's operating characteristics depend on the alternative and sample size.

## 3. p-values
A p-value is the probability, calculated under the null model, of a test statistic at least as extreme as the observed statistic, using the chosen extremeness ordering. It is not the probability that the null is true and is not an effect-size measure. A small p-value indicates incompatibility with the null model and test assumptions.

## 4. Mean inference
For i.i.d. normal observations with unknown variance,
\[
T=\frac{\bar X-\mu_0}{S/\sqrt n}\sim t_{n-1}
\]
under \(H_0:\mu=\mu_0\). The corresponding two-sided interval is \(\bar X\pm t_{1-\alpha/2,n-1}S/\sqrt n\). For large samples, asymptotic normal procedures may be used under appropriate conditions.

## 5. Likelihood-based tests
The likelihood-ratio statistic is \(LR=2\{\ell(\widehat\theta)-\ell(\widetilde\theta)\}\), where \(\widetilde\theta\) is the restricted MLE. Under regularity and a fixed number \(q\) of restrictions, \(LR\xrightarrow{d}\chi_q^2\). Wald tests use unrestricted estimates and their covariance; score tests evaluate the score under the restricted estimate. Their finite-sample behavior can differ.

## 6. Multiple testing and interpretation
Testing many hypotheses increases the chance of false discoveries. Family-wise error procedures such as Bonferroni control the probability of at least one false rejection; false-discovery-rate procedures control the expected proportion of false discoveries among rejections under their stated conditions. Report estimates, uncertainty intervals, and substantive scale—not p-values alone.

## Worked example
If \(\bar X=12\), \(S=4\), and \(n=25\), then the standard error is \(0.8\). The test statistic for \(H_0:\mu=10\) is \(2.5\) with 24 degrees of freedom. The decision follows from the appropriate t critical value or p-value; the estimated difference of 2 units should also be assessed substantively.

## Problem set
1. Construct a t confidence interval from a numerical sample summary.
2. Explain the repeated-sampling interpretation of 95% coverage.
3. Compare Type I error, Type II error, and power.
4. Derive the likelihood-ratio statistic for a nested normal model.
5. Explain why a p-value is not the posterior probability of the null.

## Computational lab
Simulate null and alternative samples; estimate empirical size and power; compare Wald, score, and likelihood-ratio tests in a small-sample model; demonstrate family-wise error under repeated independent tests.

## Reading
Casella and Berger, *Statistical Inference*; Lehmann and Romano, *Testing Statistical Hypotheses*; Wasserstein and Lazar, “The ASA Statement on p-Values.”