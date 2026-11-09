# Lecture 04 — Point Estimation: Moments, Likelihood, and Optimality

## Core question
How should unknown parameters be estimated, and which properties determine whether an estimator is useful?

## Learning objectives
Construct method-of-moments and maximum-likelihood estimators; derive score and information; distinguish unbiasedness, consistency, and efficiency; apply invariance; understand likelihood-based asymptotics.

## 1. Estimands, estimators, and identification
A parameter \(\theta\in\Theta\) indexes a statistical model. The estimand is the target functional; an estimator \(\widehat\theta\) is a data-dependent rule; an estimate is its realized value. Identification requires distinct parameter values to imply distinct distributions (or distinct population moments used by the estimator).

## 2. Method of moments
If \(E_\theta[m(X)]=g(\theta)\), equate the population moment to its sample counterpart:
\[
\frac1n\sum_{i=1}^n m(X_i)=g(\widehat\theta).
\]
With multiple parameters, solve a system of moment equations. The solution may fail to be unique or may lie outside the parameter space; these are substantive estimator-construction issues, not merely computational inconveniences.

## 3. Maximum likelihood
For independent observations with density or mass function \(f(x_i;\theta)\), the likelihood and log-likelihood are
\[
L_n(\theta)=\prod_{i=1}^nf(x_i;\theta),\qquad
\ell_n(\theta)=\sum_{i=1}^n\log f(x_i;\theta).
\]
An MLE maximizes \(\ell_n(\theta)\) over \(\Theta\). Interior differentiable solutions satisfy the score equation \(U_n(\theta)=\partial\ell_n(\theta)/\partial\theta=0\), but boundary maxima must also be considered.

## 4. Score and Fisher information
Under regularity conditions, the score has mean zero and Fisher information can be written
\[
I(\theta)=E_\theta[U_i(\theta)^2]
=-E_\theta\left[\frac{\partial^2\ell_i(\theta)}{\partial\theta\partial\theta'}\right].
\]
For a scalar parameter, the Cramér–Rao lower bound for unbiased estimators is \(1/[nI(\theta)]\) under its regularity conditions. It is a bound, not a guarantee that an estimator attains it.

## 5. Large-sample properties
Consistency means \(\widehat\theta\xrightarrow{p}\theta_0\). Asymptotic normality commonly takes the form
\[
\sqrt n(\widehat\theta-\theta_0)\xrightarrow{d}N(0,V).
\]
For a regular correctly specified MLE, \(V=I(\theta_0)^{-1}\). Under misspecification, the sandwich covariance generally replaces inverse information.

## 6. Invariance and estimator comparison
If \(\widehat\theta\) is an MLE, then \(g(\widehat\theta)\) is an MLE of \(g(\theta)\) when the transformed parameter is well-defined. Compare estimators using bias, variance, MSE, consistency, robustness, and computational stability. Finite-sample and asymptotic rankings need not coincide.

## Worked example: Bernoulli likelihood
For \(X_i\sim Bernoulli(p)\), \(\ell(p)=\sum_i[X_i\log p+(1-X_i)\log(1-p)]\). Setting the score to zero gives \(\widehat p=\bar X\). Its variance is \(p(1-p)/n\), and it is unbiased and consistent.

## Problem set
1. Derive the method-of-moments estimator of the rate of an exponential distribution.
2. Derive the Bernoulli MLE and its Fisher information.
3. Show that the MLE of \(\sigma^2\) in a normal model with known mean uses divisor \(n\).
4. Distinguish identification from estimator consistency.
5. Explain why the inverse-information covariance can fail under model misspecification.

## Computational lab
Implement numerical maximization for Bernoulli and normal likelihoods. Compare analytic and numerical estimates; inspect convergence from different starting values; use simulation to compare bias and MSE.

## Reading
Casella and Berger, *Statistical Inference*; Lehmann and Casella, *Theory of Point Estimation*; van der Vaart, *Asymptotic Statistics*.