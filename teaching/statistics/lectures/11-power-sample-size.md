# Lecture 11 — Statistical Power, Sample Size, and Equivalence

## Learning objectives
Define power; plan sample size; understand effect sizes and design effects; distinguish non-significance from equivalence.

## 1. Power
Power is \(P_{\theta_1}(\text{reject }H_0)=1-\beta\) under a specified alternative \(\theta_1\). It depends on effect size, variance, sample size, significance level, design, and test.

## 2. Two-sample mean illustration
For equal group sizes, common variance \(\sigma^2\), and difference \(\delta\), the standard error of the difference is approximately \(\sigma\sqrt{2/n}\). The noncentrality parameter is approximately \(\delta\sqrt{n/(2\sigma^2)}\). Exact planning uses the relevant sampling distribution and target power.

## 3. Design effects
Clustered sampling reduces effective information when within-cluster outcomes are correlated. A common approximation is \(DE=1+(m-1)\rho\), where \(m\) is average cluster size and \(\rho\) is intraclass correlation. Effective sample size is roughly total sample size divided by this design effect under simplifying assumptions.

## 4. Equivalence and non-inferiority
Failure to reject a zero-effect null does not establish equivalence. Equivalence testing defines a substantively acceptable interval and tests whether the effect lies within it, commonly using two one-sided tests.

## 5. Planning discipline
Specify primary outcome, minimum relevant effect, variance source, alpha, target power, allocation, clustering, attrition, and multiplicity before data collection. Avoid using observed post-hoc power as a substitute for confidence intervals.

## Problem session
1. Define power and explain its dependence on effect size.
2. Calculate a two-group standard error.
3. Explain the cluster design effect.
4. Distinguish equivalence from failure to reject a zero null.
5. Prepare a sample-size analysis plan.

## Reading
Cohen, *Statistical Power Analysis for the Behavioral Sciences*; Chow et al., *Sample Size Calculations in Clinical Research*; Lakens (2017), equivalence testing.