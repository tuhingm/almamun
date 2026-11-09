# Lecture 06 — Bayesian Inference, Conjugacy, and Posterior Computation

## Core question
How does prior information combine with a likelihood to produce posterior uncertainty and predictions?

## Learning objectives
Derive posterior distributions; work with conjugate models; distinguish credible and confidence intervals; derive posterior predictions; assess prior sensitivity and MCMC computation.

## 1. Bayes' theorem
For parameter \(\theta\) and observed data \(y\),
\[
p(\theta\mid y)=\frac{p(y\mid\theta)p(\theta)}{p(y)}
\propto p(y\mid\theta)p(\theta).
\]
The likelihood describes the data conditional on the parameter; the prior describes uncertainty about the parameter before observing the data; the posterior is the updated distribution. The prior and likelihood must be defined on compatible parameter spaces.

## 2. Beta-binomial conjugacy
Let \(Y\mid p\sim Binomial(n,p)\) and \(p\sim Beta(a,b)\). Then
\[
p\mid Y=y\sim Beta(a+y,b+n-y),\quad
E(p\mid y)=\frac{a+y}{a+b+n}.
\]
The posterior mean combines prior pseudo-counts and observed successes. The posterior variance follows from the beta variance formula.

## 3. Normal-normal conjugacy
Suppose \(Y_i\mid\mu\sim N(\mu,\sigma^2)\), with known \(\sigma^2\), and \(\mu\sim N(m_0,s_0^2)\). The posterior is normal with precision
\[
s_n^{-2}=s_0^{-2}+n\sigma^{-2},\qquad
m_n=s_n^2\left(s_0^{-2}m_0+n\sigma^{-2}\bar Y\right).
\]
Precision weighting makes the influence of prior and sample information explicit.

## 4. Credible intervals and prediction
A posterior credible set \(C(y)\) satisfies \(P(\theta\in C(y)\mid y)=1-\alpha\). Unlike a frequentist confidence interval, this is a probability statement conditional on the observed data and the model. Posterior prediction integrates parameter uncertainty:
\[
p(y_{new}\mid y)=\int p(y_{new}\mid\theta)p(\theta\mid y)d\theta.
\]

## 5. Prior specification and sensitivity
Priors may encode substantive information, regularize weakly identified parameters, or stabilize high-dimensional estimation. Examine prior predictive implications before fitting and posterior sensitivity across plausible specifications. Improper priors require care: the resulting posterior must be proper for standard posterior inference.

## 6. MCMC computation
When the posterior cannot be normalized or sampled directly, MCMC constructs a Markov chain with the posterior as its invariant distribution. For Metropolis–Hastings, propose \(\theta'\sim q(\theta'\mid\theta)\) and accept with probability
\[
\alpha=\min\left\{1,\frac{p(y\mid\theta')p(\theta')q(\theta\mid\theta')}
{p(y\mid\theta)p(\theta)q(\theta'\mid\theta)}\right\}.
\]
Assess trace behavior, mixing, effective sample size, autocorrelation, and convergence diagnostics; diagnostics do not prove convergence.

## Worked example
With a beta prior \(Beta(2,2)\), observing 8 successes in 10 Bernoulli trials gives posterior \(Beta(10,4)\), posterior mean \(10/14\), and a posterior distribution that can be summarized by quantiles or a highest-posterior-density interval.

## Problem set
1. Derive the beta-binomial posterior from Bayes' theorem.
2. Derive the normal-normal posterior mean and variance by completing the square.
3. Distinguish credible and confidence intervals.
4. Derive the posterior predictive distribution for a Bernoulli outcome.
5. Explain why multiple chains and effective sample size are useful in MCMC assessment.

## Computational lab
Compute conjugate posteriors analytically and numerically. Implement a Metropolis–Hastings sampler for a non-conjugate one-parameter posterior and report acceptance rate, trace plots, autocorrelation, and effective sample size.

## Reading
Gelman et al., *Bayesian Data Analysis*; Robert, *The Bayesian Choice*; McElreath, *Statistical Rethinking*.