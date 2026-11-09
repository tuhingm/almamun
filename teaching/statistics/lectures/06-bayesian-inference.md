# Lecture 06 — Bayesian Inference, Conjugacy, and Posterior Computation

## Learning objectives
Derive posterior distributions; understand conjugacy; distinguish credible and confidence intervals; explain posterior prediction, prior sensitivity, and MCMC diagnostics.

## 1. Bayes' theorem
For parameter \(\theta\) and data \(y\),
\[
p(\theta|y)=\frac{p(y|\theta)p(\theta)}{p(y)}\propto p(y|\theta)p(\theta).
\]
The posterior combines the likelihood with the prior. Its interpretation is conditional on the model and prior specification.

## 2. Beta-Binomial conjugacy
Let \(Y|p\sim Binomial(n,p)\) and \(p\sim Beta(a,b)\). Then
\[
p|Y=y\sim Beta(a+y,b+n-y).
\]
The posterior mean is \((a+y)/(a+b+n)\), a weighted combination of prior information and sample evidence.

## 3. Normal-Normal conjugacy
For \(Y_i|\mu\sim N(\mu,\sigma^2)\) with known variance and prior \(\mu\sim N(m_0,s_0^2)\), posterior variance and mean are
\[
s_n^2=(s_0^{-2}+n\sigma^{-2})^{-1},\quad
m_n=s_n^2(m_0s_0^{-2}+n\bar y\sigma^{-2}).
\]

## 4. Credible intervals and prediction
A Bayesian credible interval contains a specified posterior probability mass. Posterior predictive distributions integrate uncertainty in parameters:
\[
p(y_{new}|y)=\int p(y_{new}|\theta)p(\theta|y)d\theta.
\]
Prediction therefore includes both sampling variation and posterior parameter uncertainty.

## 5. MCMC
When posterior integrals cannot be evaluated analytically, simulation methods such as Metropolis-Hastings and Hamiltonian Monte Carlo approximate posterior expectations. Assess trace behavior, effective sample size, autocorrelation, chain mixing, and convergence diagnostics. A single diagnostic does not prove convergence.

## 6. Prior sensitivity
Report prior distributions and scales, explain substantive rationale, and examine whether conclusions change under plausible alternatives. Prior predictive checks can reveal implausible implications before fitting the model.

## Problem session
1. Derive the Beta-Binomial posterior.
2. Derive the Normal-Normal posterior.
3. Distinguish credible and confidence intervals.
4. Explain effective sample size in MCMC.
5. Design a prior sensitivity analysis.

## Reading
Gelman et al., *Bayesian Data Analysis*; Robert and Casella, *Monte Carlo Statistical Methods*; McElreath, *Statistical Rethinking*.