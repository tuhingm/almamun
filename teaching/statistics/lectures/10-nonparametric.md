# Lecture 10 — Nonparametric Inference, Rank Tests, and Resampling

## Learning objectives
Understand empirical distributions and quantiles; derive rank procedures; use permutation and bootstrap methods; state what rank tests actually test.

## 1. Empirical distribution
\[
\widehat F_n(x)=n^{-1}\sum_{i=1}^n1(X_i\le x).
\]
The empirical CDF estimates the population distribution and underlies quantile estimation, goodness-of-fit procedures, and resampling.

## 2. Rank-based procedures
Ranks reduce sensitivity to the numerical scale and can provide robust alternatives under particular assumptions. The Wilcoxon signed-rank procedure concerns paired differences under symmetry conditions; Mann–Whitney is related to stochastic ordering and probability-of-superiority interpretations. Neither universally tests equality of means.

## 3. Kruskal–Wallis
The Kruskal–Wallis test compares rank distributions across independent groups. Interpreting it as a test of medians requires additional assumptions about distributional shape.

## 4. Permutation inference
Under a sharp null and a valid exchangeability/randomization scheme, recompute a test statistic over allowable assignments. The reference distribution is design-based. Arbitrary permutation of observational data is not automatically valid.

## 5. Bootstrap
Resampling approximates estimator sampling distributions. Choose resampling units to preserve the design: individuals, clusters, blocks, or time-series dependence structure as appropriate.

## Problem session
1. Define the empirical CDF and sample quantile.
2. State assumptions for a paired signed-rank procedure.
3. Explain the Mann–Whitney estimand.
4. Explain when permutation tests are exact.
5. Design a cluster bootstrap.

## Reading
Hollander, Wolfe and Chicken, *Nonparametric Statistical Methods*; Efron and Tibshirani, *An Introduction to the Bootstrap*; van der Vaart, *Asymptotic Statistics*.