# Lecture 11 — DSGE Estimation: Likelihood, Bayesian Inference, and Model Evaluation

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Connect a solved DSGE model to observables; construct a likelihood; estimate parameters; assess identification and posterior fit.

## 1. Measurement system
A solved linear model can be written
\[
s_t=Ts_{t-1}+R\varepsilon_t,\qquad
y_t^o=Zs_t+D\varepsilon_t+\eta_t,
\]
where \(s_t\) is the state, \(y_t^o\) observed data, and \(\eta_t\) measurement error. Observables require explicit definitions, units, transformations, and sample alignment.

## 2. Kalman filter likelihood
For linear Gaussian systems, the Kalman filter recursively computes the one-step forecast error and covariance. The log likelihood is the sum of Gaussian forecast-error contributions:
\[
\ell(\theta)=-\frac12\sum_t\left[m\log(2\pi)+\log|F_t|+v_t'F_t^{-1}v_t\right],
\]
where \(v_t\) and \(F_t\) are the innovation and its covariance.

## 3. Bayesian estimation
Bayes' rule gives
\[
p(\theta\mid Y)\propto p(Y\mid\theta)p(\theta).
\]
The prior must respect parameter support and be justified. Posterior simulation commonly uses Metropolis–Hastings or other MCMC methods. Report acceptance rates, trace plots, effective sample sizes, convergence diagnostics, and Monte Carlo standard errors where available.

## 4. Identification and sensitivity
A concentrated likelihood or posterior correlation may indicate weak identification. Examine prior-to-posterior learning, likelihood profiles, local sensitivity, and alternative observable sets. Posterior predictive checks evaluate whether simulated observables reproduce salient empirical features.

## 5. Model comparison
Marginal likelihoods and information criteria answer different questions and depend on assumptions. State the criterion, prior sensitivity, and model parameterization. Do not equate a better in-sample fit with a better causal or policy model.

## 6. Computational lab
Estimate an RBC or NK model using simulated data first, then real data. Store data transformations, parameter priors, random seeds, chain settings, and posterior draws. Compare posterior predictive moments with observed moments.

## Problem set
1. Derive the Gaussian innovation log likelihood.
2. Explain the role of the measurement equation.
3. Distinguish prior, likelihood, and posterior.
4. Propose diagnostics for weak identification and MCMC convergence.

## Further reading
An and Schorfheide (2007); Fernández-Villaverde (2010); Herbst and Schorfheide, *Bayesian Estimation of DSGE Models*.
