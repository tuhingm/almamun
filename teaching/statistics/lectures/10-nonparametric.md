# Lecture 10 — Nonparametric Inference, Rank Tests, and Resampling

## Learning objectives
Use empirical distributions and quantiles; derive rank procedures; state the hypotheses actually tested; conduct permutation tests and bootstrap inference.

## 1. Empirical distribution
The empirical CDF is
\[
\widehat F_n(x)=n^{-1}\sum_{i=1}^n1(X_i\le x).
\]
It estimates the population distribution and supports quantile estimation, goodness-of-fit procedures, and resampling. Quantile conventions should be stated when finite-sample results depend on interpolation.

## 2. Sign and signed-rank tests
For paired data define \(D_i=X_i-Y_i\). The sign test uses signs of nonzero differences and targets a median-zero condition under its assumptions. The Wilcoxon signed-rank test ranks absolute differences and uses their signs. Its conventional location interpretation requires a symmetric distribution of paired differences; ties and zero differences need explicit treatment.

## 3. Mann–Whitney and Kruskal–Wallis
The Mann–Whitney statistic compares ranks in two independent groups and is related to \(P(X>Y)+\tfrac12P(X=Y)\). It is not universally a test of equal means or medians; a location-shift interpretation needs additional assumptions. Kruskal–Wallis extends rank comparisons to multiple independent groups.

## 4. Permutation inference
A permutation test constructs a reference distribution by reassigning labels according to exchangeability or the actual randomization design. Under a sharp null and valid assignment mechanism, randomization inference can be exact. Arbitrary permutation is not justified for dependent or selected observational data.

## 5. Bootstrap
The nonparametric bootstrap resamples with replacement from the empirical distribution and recomputes a statistic. Percentile, basic, and studentized intervals differ. Clustered or serially dependent data require resampling schemes that preserve the dependence structure.

## 6. Robustness and assumptions
Nonparametric methods still require assumptions such as independence, exchangeability, or symmetry, depending on the procedure. Robustness to outliers does not imply robustness to dependence or selection.

## Worked example
For paired before-after data, calculate differences, rank absolute nonzero differences, and sum positive ranks. The signed-rank statistic uses both direction and rank magnitude; it is not simply a test of the arithmetic mean difference.

## Problem set
1. Define the empirical CDF and prove monotonicity.
2. State the symmetry condition used by the signed-rank test.
3. Interpret the Mann–Whitney probability-of-superiority parameter.
4. Explain exchangeability in permutation inference.
5. Specify a bootstrap scheme for clustered observations.

## Computational lab
Compare t, sign, and signed-rank tests under normal, skewed, and contaminated distributions. Implement a randomization test and a cluster bootstrap.

## Reading
Hollander, Wolfe, and Chicken, *Nonparametric Statistical Methods*; Wasserman, *All of Nonparametric Statistics*; Efron and Tibshirani, *An Introduction to the Bootstrap*.