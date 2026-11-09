# Lecture 03 — Sampling Distributions and Asymptotic Theory

## Core question
Why do estimators have predictable large-sample distributions, and how are those distributions used to quantify uncertainty?

## Learning objectives
Distinguish population and sampling distributions; derive the sampling distribution of a mean; apply LLN, CLT, Slutsky's theorem, continuous mapping, and the delta method; understand bootstrap approximation.

## 1. Sampling distributions and estimator risk
An estimator \(\widehat\theta=T(X_1,\ldots,X_n)\) is random before the sample is observed. Its sampling distribution determines bias, variance, mean squared error, and interval coverage. The decomposition
\[
MSE(\widehat\theta)=Var(\widehat\theta)+Bias(\widehat\theta)^2
\]
shows why unbiasedness alone is not sufficient to rank estimators.

## 2. Law of large numbers
For i.i.d. observations with \(E|X_i|<\infty\), the weak law gives
\[
\bar X_n\xrightarrow{p}\mu.
\]
The strong law gives almost-sure convergence under the standard i.i.d. integrability condition. LLN justifies consistency of sample averages and sample moments.

## 3. Central limit theorem
For i.i.d. observations with finite mean \(\mu\) and variance \(0<\sigma^2<\infty\),
\[
\sqrt n(\bar X_n-\mu)\xrightarrow{d}N(0,\sigma^2).
\]
Equivalently, standardization by the population standard deviation yields a standard normal limit. The CLT is an approximation, not a claim that the finite-sample population distribution is normal.

## 4. Slutsky and continuous mapping
If \(X_n\xrightarrow{d}X\) and \(Y_n\xrightarrow{p}c\), then \(X_n+Y_n\xrightarrow{d}X+c\), \(X_nY_n\xrightarrow{d}cX\), and, for \(c\ne0\), \(X_n/Y_n\xrightarrow{d}X/c\). If \(X_n\xrightarrow{d}X\) and \(g\) is continuous, then \(g(X_n)\xrightarrow{d}g(X)\).

## 5. Delta method
If \(\sqrt n(\widehat\theta-\theta)\xrightarrow{d}N(0,V)\) and \(g\) is differentiable at \(\theta\), then
\[
\sqrt n\{g(\widehat\theta)-g(\theta)\}
\xrightarrow{d}N(0,[g'(\theta)]^2V).
\]
For vector parameters, replace the derivative with the gradient and the variance with \(G'VG\). The method is a first-order Taylor approximation; second-order methods are needed when the first derivative vanishes.

## 6. Bootstrap
The nonparametric bootstrap samples with replacement from the empirical distribution \(\widehat F_n\), recomputes the estimator, and approximates its sampling distribution. Percentile and studentized intervals are common constructions. Validity depends on regularity and sampling structure; clustered or dependent observations require a bootstrap that preserves the relevant dependence.

## Worked example
For a sample mean, estimate \(\sigma\) by the sample standard deviation \(S\). Slutsky's theorem gives
\[
\frac{\sqrt n(\bar X-\mu)}{S}\xrightarrow{d}N(0,1).
\]
For a positive mean, applying the delta method to \(g(\mu)=\log\mu\) gives asymptotic variance \(\sigma^2/\mu^2\).

## Problem set
1. Prove consistency of the sample mean using Chebyshev's inequality.
2. State the conditions and conclusion of the CLT.
3. Use Slutsky's theorem to justify replacing a population variance by a consistent estimator.
4. Derive the delta-method variance for \(\log\widehat\theta\).
5. Explain why an i.i.d. bootstrap is inappropriate for clustered observations.

## Computational lab
Use Monte Carlo simulation to compare finite-sample distributions of sample means under normal, skewed, and heavy-tailed distributions. Evaluate coverage of normal-approximation and bootstrap intervals as sample size increases.

## Reading
van der Vaart, *Asymptotic Statistics*; Casella and Berger, *Statistical Inference*; Efron and Tibshirani, *An Introduction to the Bootstrap*.