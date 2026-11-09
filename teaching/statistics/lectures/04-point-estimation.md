# Lecture 04 — Point Estimation: Moments, Likelihood, and Optimality

## Learning objectives
Construct method-of-moments and maximum-likelihood estimators; derive score and information; distinguish unbiasedness, consistency, and efficiency; apply invariance and asymptotic theory.

## 1. Estimands and identification
A parameter \(\theta\) indexes a statistical model. An estimand is the target functional, an estimator \(\widehat\theta\) is a data-dependent rule, and an estimate is its realized value. Identification requires distinct parameter values to imply distinct distributions or population moments.

## 2. Method of moments
Equate sample moments to their population counterparts. If \(E_\theta[m(X)]=g(\theta)\), solve
\[
n^{-1}\sum_{i=1}^n m(X_i)=g(\widehat\theta).
\]
For multiple parameters, solve a system of equations. Existence and uniqueness of the solution are identification and parameter-space issues.

## 3. Maximum likelihood
For independent observations, \(L_n(\theta)=\prod_i f(X_i;\theta)\) and \(\ell_n(\theta)=\sum_i\log f(X_i;\theta)\). The MLE maximizes the log-likelihood over the parameter space. Interior differentiable solutions satisfy the score equation \(U_n(\theta)=\partial\ell_n(\theta)/\partial\theta=0\), but boundary solutions must also be checked.

## 4. Score and Fisher information
Under regularity conditions, \(E[U_i(\theta)]=0\) and
\[
I(\theta)=E[U_i(\theta)U_i(\theta)']
=-E\left[\frac{\partial^2\ell_i(\theta)}{\partial\theta\partial\theta'}\right].
\]
The Cramér–Rao lower bound for unbiased estimators is based on inverse information under its regularity conditions; it is not an unconditional guarantee of attainability.

## 5. Large-sample properties
Consistency is \(\widehat\theta\xrightarrow{p}\theta_0\). Asymptotic normality commonly takes the form \(\sqrt n(\widehat\theta-\theta_0)\xrightarrow{d}N(0,V)\). For a regular correctly specified MLE, \(V=I(\theta_0)^{-1}\); under misspecification, a sandwich covariance is generally required.

## 6. Invariance and estimator comparison
A transformation of an MLE is an MLE of the corresponding transformed parameter when well-defined. Compare estimators by bias, variance, MSE, consistency, robustness, and computational stability; finite-sample and asymptotic rankings may differ.

## Worked example: Bernoulli likelihood
For \(X_i\sim Bernoulli(p)\), the log-likelihood is \(\ell(p)=\sum_i[X_i\log p+(1-X_i)\log(1-p)]\). The score equation gives \(\widehat p=\bar X\), which is unbiased and consistent, with variance \(p(1-p)/n\).

## Problem set
1. Derive the method-of-moments estimator for an exponential rate.
2. Derive the Bernoulli MLE and Fisher information.
3. Distinguish identification from consistency.
4. Explain why the inverse-information covariance can fail under misspecification.
5. Compare two estimators using MSE rather than bias alone.

## Computational lab
Implement numerical likelihood maximization for Bernoulli and normal models. Compare numerical and analytic estimates and simulate bias and MSE.

## Reading
Casella and Berger, *Statistical Inference*; Lehmann and Casella, *Theory of Point Estimation*; van der Vaart, *Asymptotic Statistics*.