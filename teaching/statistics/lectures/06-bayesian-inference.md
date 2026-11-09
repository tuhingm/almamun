# Lecture 06 — Bayesian Inference, Conjugacy, and Posterior Computation

## Learning objectives
Derive posterior distributions; work with conjugate models; distinguish credible and confidence intervals; derive posterior predictions; assess prior sensitivity and MCMC computation.

## 1. Bayes' theorem
For parameter \(\theta\) and data \(y\),
\[
p(\theta\mid y)=\frac{p(y\mid\theta)p(\theta)}{p(y)}\propto p(y\mid\theta)p(\theta).
\]
The likelihood describes data conditional on parameters, the prior represents pre-data uncertainty, and the posterior updates that uncertainty. All distributions must be defined on compatible parameter spaces.

## 2. Beta-binomial conjugacy
Let \(Y\mid p\sim Binomial(n,p)\) and \(p\sim Beta(a,b)\). Then
\[
p\mid Y=y\sim Beta(a+y,b+n-y),\quad E(p\mid y)=\frac{a+y}{a+b+n}.
\]
The prior parameters act as pseudo-counts. Posterior variance follows from the beta variance formula.

## 3. Normal-normal conjugacy
Suppose \(Y_i\mid\mu\sim N(\mu,\sigma^2)\), with known variance, and \(\mu\sim N(m_0,s_0^2)\). Completing the square gives
\[
s_n^{-2}=s_0^{-2}+n\sigma^{-2},\quad
m_n=s_n^2(s_0^{-2}m_0+n\sigma^{-2}\bar Y).
\]
The posterior mean is a precision-weighted average of prior mean and sample mean.

## 4. Credible intervals and prediction
A credible set satisfies \(P(\theta\in C(y)\mid y)=1-\alpha\). This is conditional on observed data and the model, unlike frequentist coverage. Posterior prediction integrates parameter uncertainty:
\[
p(y_{new}\mid y)=\int p(y_{new}\mid\theta)p(\theta\mid y)d\theta.
\]

## 5. Prior sensitivity
Use prior predictive checks to determine whether the prior implies plausible data. Examine posterior sensitivity across scientifically defensible prior specifications. An improper prior is acceptable only when the resulting posterior is proper for the intended inference.

## 6. MCMC
Metropolis–Hastings proposes \(\theta'\sim q(\theta'\mid\theta)\) and accepts with probability
\[
\alpha=\min\left(1,\frac{p(y\mid\theta')p(\theta')q(\theta\mid\theta')}{p(y\mid\theta)p(\theta)q(\theta'\mid\theta)}\right).
\]
Assess trace plots, mixing, autocorrelation, effective sample size, and between-chain diagnostics. No single diagnostic proves convergence.

## Worked example
With prior \(p\sim Beta(2,2)\) and 8 successes in 10 trials, the posterior is \(Beta(10,4)\), with mean \(10/14\). Posterior quantiles provide credible intervals.

## Problem set
1. Derive the beta-binomial posterior.
2. Derive the normal-normal posterior by completing the square.
3. Distinguish credible and confidence intervals.
4. Derive a Bernoulli posterior predictive probability.
5. Explain effective sample size and multiple-chain diagnostics.

## Computational lab
Compute conjugate posteriors analytically and numerically. Implement Metropolis–Hastings for a non-conjugate posterior and report acceptance, trace, autocorrelation, and effective sample size.

## Reading
Gelman et al., *Bayesian Data Analysis*; Robert, *The Bayesian Choice*; McElreath, *Statistical Rethinking*.