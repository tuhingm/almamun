# Lecture 03 — Sampling Distributions and Asymptotic Theory

## Learning objectives
Distinguish sampling distributions from population distributions; apply LLN, CLT, Slutsky, continuous mapping, and delta methods; understand bootstrap approximation.

## 1. Sampling distributions
An estimator \(\widehat\theta=T(X_1,\ldots,X_n)\) is random before the sample is observed. Its sampling distribution determines bias, variance, mean squared error, and coverage. The decomposition \(MSE(\widehat\theta)=Var(\widehat\theta)+Bias(\widehat\theta)^2\) shows why unbiasedness alone is not a complete optimality criterion.

## 2. Law of large numbers
For i.i.d. observations with finite first absolute moment, the sample mean converges in probability to its population mean:
\[
\bar X_n\xrightarrow{p}\mu.
\]
This result supports consistency of sample averages and sample moments.

## 3. Central limit theorem
For i.i.d. observations with finite mean \(\mu\) and variance \(0<\sigma^2<\infty\),
\[
\sqrt n(\bar X_n-\mu)\xrightarrow{d}N(0,\sigma^2).
\]
The CLT provides a large-sample approximation; it does not assert that the finite-sample population distribution is normal.

## 4. Slutsky and continuous mapping
If \(X_n\xrightarrow{d}X\) and \(Y_n\xrightarrow{p}c\), then sums and products converge to the corresponding transformations involving \(X\) and \(c\). If \(g\) is continuous, \(g(X_n)\xrightarrow{d}g(X)\). These results justify replacing unknown constants with consistent estimators in asymptotic statistics.

## 5. Delta method
If \(\sqrt n(\widehat\theta-\theta)\xrightarrow{d}N(0,V)\) and \(g\) is differentiable at \(\theta\), then
\[
\sqrt n[g(\widehat\theta)-g(\theta)]\xrightarrow{d}N(0,[g'(\theta)]^2V).
\]
For vector parameters, use the gradient and covariance matrix. The first-order method may fail when the derivative is zero.

## 6. Bootstrap
The nonparametric bootstrap samples with replacement from the empirical distribution and recomputes the estimator. Percentile and studentized intervals are common. Valid resampling must preserve dependence: clustered observations require cluster resampling, while time series require dependence-aware methods.

## Worked example
For the sample mean, replacing \(\sigma\) by the consistent sample standard deviation \(S\) gives \(\sqrt n(\bar X-\mu)/S\xrightarrow{d}N(0,1)\) by Slutsky. Applying the delta method to \(g(\mu)=\log\mu\) gives asymptotic variance \(\sigma^2/\mu^2\).

## Problem set
1. Prove sample-mean consistency using Chebyshev's inequality.
2. State the assumptions and conclusion of the CLT.
3. Use Slutsky to justify studentization.
4. Derive the delta-method variance for a logarithm.
5. Explain why an i.i.d. bootstrap is unsuitable for clustered data.

## Computational lab
Simulate sample means from normal, skewed, and heavy-tailed populations. Compare finite-sample distributions and coverage of normal and bootstrap intervals as \(n\) increases.

## Reading
van der Vaart, *Asymptotic Statistics*; Casella and Berger, *Statistical Inference*; Efron and Tibshirani, *An Introduction to the Bootstrap*.