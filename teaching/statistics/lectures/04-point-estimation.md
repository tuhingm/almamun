# Lecture 04 — Point Estimation: Moments, Likelihood, and Optimality

## Learning objectives
Construct estimators by moments and likelihood; distinguish unbiasedness, consistency, and efficiency; derive score and information; understand invariance and finite-sample limitations.

## 1. Estimands and estimators
A parameter \(\theta\) indexes a statistical model. An estimand is the target functional; an estimator \(\widehat\theta\) is a data-dependent rule; an estimate is its realized value. These terms are not interchangeable.

## 2. Method of moments
Choose sample moments to match population moments. If \(E(X)=\theta\), then \(\widehat\theta=\bar X\). With multiple parameters, solve a system of moment equations. Identification requires that the population moment equations uniquely determine the parameter.

## 3. Maximum likelihood
For independent observations with density \(f(x_i;\theta)\),
\[
L(\theta)=\prod_{i=1}^nf(x_i;\theta),\qquad
\ell(\theta)=\sum_i\log f(x_i;\theta).
\]
The MLE maximizes the likelihood over the parameter space. Interior first-order conditions set the score \(U(\theta)=\partial\ell/\partial\theta\) equal to zero; boundary maxima require separate consideration.

## 4. Bernoulli example
For \(X_i\sim Bernoulli(p)\),
\[
\ell(p)=\sum_i[X_i\log p+(1-X_i)\log(1-p)].
\]
The score is \(\sum_iX_i/p-(n-\sum_iX_i)/(1-p)\). Setting it to zero gives \(\widehat p=\bar X\).

## 5. Properties
Unbiasedness means \(E(\widehat\theta)=\theta\); consistency means convergence in probability; efficiency compares variance among an explicitly defined estimator class. MSE decomposes as variance plus squared bias. An unbiased estimator need not have low MSE.

## 6. Fisher information and asymptotics
Under regularity, the MLE is asymptotically normal:
\[
\sqrt n(\widehat\theta-\theta_0)\to_dN(0,I(\theta_0)^{-1}),
\]
where per-observation Fisher information is \(I(\theta)=E[s_i(\theta)s_i(\theta)']\). Misspecification, boundaries, and weak identification can invalidate standard approximations.

## Problem session
1. Derive the Bernoulli MLE.
2. Compare method-of-moments and MLE for an exponential distribution.
3. Define bias, variance, MSE, and consistency.
4. Explain the invariance property of MLE.
5. Identify a setting where a parameter lies on the boundary.

## Reading
Lehmann and Casella, *Theory of Point Estimation*; Casella and Berger; van der Vaart, *Asymptotic Statistics*.