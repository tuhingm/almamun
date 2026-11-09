# Lecture 08 — ANOVA, Linear Contrasts, and Multiple Comparisons

## Learning objectives
Derive one-way ANOVA; partition sums of squares; formulate contrasts; distinguish fixed and random effects; control multiplicity.

## 1. One-way model
\[
Y_{ij}=\mu+\tau_i+\varepsilon_{ij},\quad i=1,\ldots,k.
\]
Under independent homoskedastic normal errors, test equality of group means using
\[
F=MS_{between}/MS_{within}.
\]
The null is that all group means are equal. Rejection indicates at least one difference, not which pair differs.

## 2. Sum-of-squares decomposition
Total variation decomposes into between-group and within-group components. Degrees of freedom are \(k-1\), \(N-k\), and \(N-1\), respectively. The F statistic compares variation explained by group membership with residual variation.

## 3. Contrasts
A contrast \(\sum_i c_i\mu_i\) satisfies \(\sum_i c_i=0\). Planned contrasts test scientifically motivated comparisons. Their variance depends on group sample sizes and error variance.

## 4. Multiple comparisons
Tukey controls family-wise error for all pairwise comparisons under its assumptions; Bonferroni and Holm adjust general families of tests; Dunnett is designed for comparisons against a control. Selection of a procedure should reflect the family of hypotheses.

## 5. Fixed and random effects
Fixed effects treat factor levels as the specific levels of interest. Random effects model levels as sampled from a population distribution. The distinction changes the estimand and variance components.

## Problem session
1. Derive the one-way ANOVA F ratio.
2. State the ANOVA degrees of freedom.
3. Construct a contrast comparing the average of two treatments with a control.
4. Compare Tukey, Holm, and Dunnett procedures.
5. Explain fixed versus random factor interpretation.

## Reading
Montgomery, *Design and Analysis of Experiments*; Hsu, *Multiple Comparisons*; Searle, Casella and McCulloch, *Variance Components*.