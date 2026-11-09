# Lecture 05 — Confidence Intervals and Hypothesis Testing

## Learning objectives
Construct confidence intervals; formulate tests; understand size, power, p-values, Type I and II errors; distinguish statistical and practical significance.

## 1. Confidence intervals
A confidence procedure has coverage \(P_\theta\{L(X)\le\theta\le U(X)\}\ge1-\alpha\) under the specified model. The parameter is fixed in frequentist inference; the interval is random before sampling. After observing data, the interval either contains the parameter or does not.

## 2. Testing framework
Specify null \(H_0\), alternative \(H_1\), test statistic, rejection region, and significance level \(\alpha\). Type I error is rejection when the null is true; Type II error is failure to reject under a particular alternative. Power is \(1-\beta\).

## 3. p-values
A p-value is computed assuming the null and measures how extreme the observed statistic is relative to the null distribution. It is not the probability that the null is true and is not an effect-size measure.

## 4. Likelihood ratio, Wald, and score
For regular parametric models, LR compares maximized likelihoods under restricted and unrestricted models; Wald uses the estimated distance from the null restriction; score evaluates the slope of the likelihood at the null estimate. They are asymptotically equivalent under standard regularity but can differ in finite samples.

## 5. Multiple testing
Testing many hypotheses increases the chance of at least one false rejection. Family-wise error procedures include Bonferroni and Holm; false discovery rate procedures target the expected proportion of false discoveries among rejections. Choice depends on the inferential objective.

## Problem session
1. Explain repeated-sampling coverage.
2. Define size and power.
3. Interpret a p-value without probability-of-null language.
4. Compare LR, Wald, and score tests.
5. Distinguish family-wise error from false discovery rate.

## Reading
Lehmann and Romano, *Testing Statistical Hypotheses*; Casella and Berger; Wasserman, *All of Statistics*.