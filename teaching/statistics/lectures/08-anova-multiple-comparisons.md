# Lecture 08 — ANOVA, Linear Contrasts, and Multiple Comparisons

## Learning objectives
Derive one-way ANOVA; partition sums of squares; formulate contrasts; distinguish fixed and random effects; understand multiplicity adjustments and diagnostics.

## 1. One-way model
For group \(i=1,\ldots,k\),
\[
Y_{ij}=\mu+\tau_i+\varepsilon_{ij},\quad
\varepsilon_{ij}\overset{iid}{\sim}N(0,\sigma^2).
\]
A constraint such as \(\sum_i n_i\tau_i=0\) identifies the intercept and group effects. The omnibus null is equality of all group means.

## 2. Sum-of-squares decomposition
With group means \(\bar Y_i\) and grand mean \(\bar Y\),
\[
SS_T=\sum_i\sum_j(Y_{ij}-\bar Y)^2
=\sum_i n_i(\bar Y_i-\bar Y)^2+
\sum_i\sum_j(Y_{ij}-\bar Y_i)^2.
\]
Between-group and within-group degrees of freedom are \(k-1\) and \(N-k\). The statistic \(F=MS_B/MS_W\) compares explained group variation with residual variation.

## 3. Linear contrasts
A contrast is \(\psi=\sum_i c_i\mu_i\), with \(\sum_i c_i=0\). Its estimator is \(\widehat\psi=\sum_i c_i\bar Y_i\), with variance under the common-variance model
\[
Var(\widehat\psi)=\sigma^2\sum_i c_i^2/n_i.
\]
A planned contrast answers a targeted question; an omnibus rejection does not identify which groups differ.

## 4. Multiple comparisons
Tukey controls family-wise error for all pairwise comparisons under its model. Bonferroni tests each of \(m\) hypotheses at level \(\alpha/m\); Holm's step-down procedure is generally less conservative while controlling family-wise error. False-discovery-rate methods target a different error criterion.

## 5. Fixed and random effects
Fixed effects represent specific levels of direct interest. Random effects treat levels as draws from a population. In a random-intercept model, \(b_j\sim N(0,\tau^2)\), and \(ICC=\tau^2/(\tau^2+\sigma^2)\). This choice changes the estimand and covariance structure.

## 6. Diagnostics
Inspect residuals for unequal variance, outliers, dependence, and non-normality. Welch's ANOVA accommodates unequal group variances in one-way comparisons. Robust and permutation alternatives still require a clearly specified target hypothesis.

## Worked example
For three teaching methods with means 70, 76, and 82, the omnibus test evaluates equality of all three means. The contrast \((-1,0,1)\) compares endpoint methods and is not the same as testing every pair.

## Problem set
1. Derive the one-way sum-of-squares decomposition.
2. State the omnibus F-test hypotheses.
3. Calculate the variance of a specified contrast.
4. Compare Tukey, Bonferroni, and Holm.
5. Distinguish fixed and random treatment levels.

## Computational lab
Fit one-way ANOVA, compute planned contrasts, and compare Tukey and Holm-adjusted pairwise inference. Simulate unequal variances and compare classical and Welch procedures.

## Reading
Montgomery, *Design and Analysis of Experiments*; Hsu, *Multiple Comparisons*; Kutner et al., *Applied Linear Statistical Models*.