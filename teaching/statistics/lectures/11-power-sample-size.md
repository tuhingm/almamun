# Lecture 11 — Statistical Power, Sample Size, and Equivalence

## Learning objectives
Define power and minimum detectable effects; calculate sample sizes; account for clustering and attrition; distinguish non-significance from evidence of equivalence.

## 1. Power
For rejection region \(R\), power at parameter \(\theta\) is \(\pi(\theta)=P_\theta(T\in R)\). At a specified alternative it equals \(1-\beta\). Power depends on effect size, variance, sample size, significance level, design, and test. It is a design property under an assumed model, not a post hoc explanation for a non-significant finding.

## 2. Two independent means
For equal group sizes \(n\), common variance \(\sigma^2\), and difference \(\delta\), the standard error is approximately \(\sigma\sqrt{2/n}\). The standardized effect is \(d=\delta/\sigma\), and the normal-approximation noncentrality parameter is \(d\sqrt{n/2}\). Exact planning uses the noncentral t distribution and the planned sidedness.

## 3. Regression and proportions
Regression power depends on residual variance, the variance of the focal regressor after partialling out controls, and the design matrix. Collinearity reduces information even when total sample size is fixed. Proportion comparisons depend on null and alternative proportions and the selected test.

## 4. Clustered designs
For equal cluster size \(m\) and intraclass correlation \(\rho\), the approximate design effect is
\[
DE=1+(m-1)\rho.
\]
Effective sample size is approximately total sample size divided by this factor. Unequal cluster sizes and few clusters require more careful planning and small-sample corrections.

## 5. Attrition
If the required analyzable sample is \(n_a\) and expected attrition is \(r\), recruit approximately \(n_a/(1-r)\). This inflation does not remove bias from informative attrition; the missing-data mechanism must be considered separately.

## 6. Equivalence and non-inferiority
For equivalence margin \(\Delta>0\), test \(H_0:\theta\le-\Delta\text{ or }\theta\ge\Delta\) against \(H_1:-\Delta<\theta<\Delta\). The two-one-sided-tests procedure assesses whether the confidence interval lies inside the equivalence bounds. Failure to reject a zero-effect null is not evidence of equivalence.

## Worked example
If \(\sigma=10\) and the target difference is 5, then \(d=0.5\). Use the noncentral t distribution to find the group size achieving target power at the chosen alpha, then verify sensitivity to variance and attrition assumptions.

## Problem set
1. Define power as a function of the parameter.
2. Derive the two-group mean standard error.
3. Calculate a design effect for given cluster size and ICC.
4. Inflate an analyzable sample for attrition.
5. Formulate equivalence hypotheses for a chosen margin.

## Computational lab
Calculate sample sizes over a grid of effect sizes and variances. Plot power curves and compare individual versus cluster randomization.

## Reading
Cohen, *Statistical Power Analysis for the Behavioral Sciences*; Chow et al., *Sample Size Calculations in Clinical Research*; Lakens, *Equivalence Tests*.