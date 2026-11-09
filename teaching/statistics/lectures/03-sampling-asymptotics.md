# Lecture 03 — Sampling Distributions and Asymptotic Theory

## Learning objectives
Distinguish population and sampling distributions; apply LLN and CLT; use Slutsky and continuous mapping; derive delta-method approximations; understand bootstrap logic.

## 1. Sampling distributions
An estimator \(\widehat\theta=T(X_1,\ldots,X_n)\) is a random variable before the sample is observed. Its sampling distribution describes repeated-sample behavior. Bias, variance, mean squared error, and coverage are properties of this distribution.

## 2. Law of large numbers
For i.i.d. observations with finite first moment, the sample mean converges in probability to the population mean:
\[
\bar X_n\xrightarrow{p}\mu.
\]
This is consistency of the sample mean for \(\mu\).

## 3. Central limit theorem
With finite variance \(\sigma^2\),
\[
\sqrt n(\bar X_n-\mu)\xrightarrow{d}N(0,\sigma^2).
\]
The CLT describes a standardized estimation error, not the distribution of the raw observations. Dependence or heavy tails may require alternative conditions and limits.

## 4. Slutsky and continuous mapping
If \(X_n\to_p c\) and \(Y_n\to_d Y\), then \(X_nY_n\to_d cY\). Continuous transformations preserve convergence under appropriate conditions. These tools allow unknown standard deviations to be replaced by consistent estimators.

## 5. Delta method
If \(\sqrt n(\widehat\theta-\theta)\to_dN(0,V)\) and \(g\) is differentiable at \(\theta\), then
\[
\sqrt n\{g(\widehat\theta)-g(\theta)\}
\to_dN(0,g'(\theta)^2V).
\]
For vector parameters, replace the derivative by the gradient and use \(G'VG\).

## 6. Bootstrap
The bootstrap approximates sampling uncertainty by resampling from the empirical distribution. Validity depends on the estimator, sampling structure, and regularity. For clustered or serially dependent data, resampling individual rows independently is generally inappropriate; resample the independent sampling units or use a dependence-aware procedure.

## Problem session
1. State conditions for the LLN and CLT used in this lecture.
2. Apply the delta method to \(\log\bar X\).
3. Explain why the CLT does not imply that the data are normally distributed.
4. Design a bootstrap for a clustered sample.
5. Distinguish consistency from unbiasedness.

## Reading
van der Vaart, *Asymptotic Statistics*; Lehmann and Casella, *Theory of Point Estimation*; Wasserman, *All of Statistics*.