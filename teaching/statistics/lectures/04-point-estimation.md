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

## Coding exercises

These exercises use **real data** in Python. Install once with
`pip install wooldridge statsmodels scipy numpy pandas`. The `wooldridge` package holds survey, experimental and
administrative datasets; `scipy.stats` and `statsmodels` provide the inference tools. Try the starter code first, then
open the solution. Every printed result below comes from running the solution code exactly as shown.

### Exercise 4.1 — Poisson maximum likelihood for arrest counts, by hand

*Data:* `crime1` — 2,725 young men in California; `narr86` = number of arrests in 1986.

1. Write the Poisson log-likelihood for \(\lambda\) and maximize it numerically with `scipy.optimize`. Confirm \(\widehat\lambda=\bar X\).
2. Compute the standard error from (a) the observed information (numerical second derivative) and (b) the sandwich formula.
3. Compare the sample variance with the mean. What does over-dispersion imply for the two standard errors?
4. Compare observed and Poisson-predicted frequencies of 0, 1, 2 and 3+ arrests.

```python
import numpy as np, wooldridge as wd
from scipy import optimize, stats
x = wd.data("crime1")["narr86"].to_numpy()
negll = lambda lam: -np.sum(stats.poisson.logpmf(x, lam))
# TODO: optimize.minimize_scalar(negll, bounds=(1e-6, 5), method="bounded")
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
from scipy import optimize, stats
x = wd.data("crime1")["narr86"].to_numpy(); n = len(x)
negll = lambda lam: -np.sum(stats.poisson.logpmf(x, lam))
lam = optimize.minimize_scalar(negll, bounds=(1e-6, 5), method="bounded", options={"xatol": 1e-10}).x
print(f"MLE lambda = {lam:.5f};  sample mean = {x.mean():.5f}")
h = 1e-4; info = (negll(lam + h) - 2 * negll(lam) + negll(lam - h)) / h**2
score_var = np.sum((x / lam - 1)**2)
print(f"se from observed information = {np.sqrt(1/info):.5f};  sandwich se = {np.sqrt(score_var)/info:.5f}")
print(f"sample variance {x.var(ddof=1):.4f} vs mean {x.mean():.4f}  (ratio {x.var(ddof=1)/x.mean():.2f})")
obs = [np.mean(x == k) for k in (0, 1, 2)] + [np.mean(x >= 3)]
fit = [stats.poisson.pmf(k, lam) for k in (0, 1, 2)] + [stats.poisson.sf(2, lam)]
print("P(0,1,2,3+) observed:", np.round(obs, 4), " Poisson:", np.round(fit, 4))
```

Output:

```text
MLE lambda = 0.40440;  sample mean = 0.40440
se from observed information = 0.01218;  sandwich se = 0.01645
sample variance 0.7380 vs mean 0.4044  (ratio 1.82)
P(0,1,2,3+) observed: [0.7229 0.2051 0.0444 0.0275]  Poisson: [0.6674 0.2699 0.0546 0.0082]
```

The numerical MLE equals the sample mean, as theory says. The variance is 1.8 times the mean, so the Poisson assumption
\(\operatorname{Var}=E\) fails. The sandwich standard error is about 35% larger than the information-based one, because
the information equality no longer holds. The Poisson model underpredicts zeros and, especially, three or more arrests
(2.8% observed against 0.8% predicted): the classic signature of overdispersion, typically from unobserved heterogeneity
in arrest risk. Lecture 9's negative binomial model addresses this.

</details>

### Exercise 4.2 — Method of moments versus maximum likelihood for a gamma model

*Data:* `wage1` — hourly wages.

For a gamma distribution with shape \(k\) and scale \(\theta\), \(E[X]=k\theta\) and \(\operatorname{Var}(X)=k\theta^2\).

1. Derive and compute the method-of-moments estimators \(\widehat\theta=s^2/\bar x\), \(\widehat k=\bar x^2/s^2\).
2. Compute the MLE with `scipy.stats.gamma.fit`.
3. Bootstrap both estimators (1,000 draws) and compare their standard deviations. Which is more efficient, and is that what theory predicts?

```python
import numpy as np, wooldridge as wd
from scipy import stats
w = wd.data("wage1")["wage"].to_numpy()
# TODO: MoM, MLE, bootstrap both
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
from scipy import stats
w = wd.data("wage1")["wage"].to_numpy(); n = len(w)
mom = lambda x: (x.mean()**2 / x.var(ddof=1), x.var(ddof=1) / x.mean())
mle = lambda x: stats.gamma.fit(x, floc=0)[::2]
print("MoM  (shape, scale):", np.round(mom(w), 4))
print("MLE  (shape, scale):", np.round(mle(w), 4))
rng = np.random.default_rng(3); B = np.array([[*mom(s), *mle(s)] for s in (w[rng.integers(0, n, n)] for _ in range(1000))])
print(f"bootstrap SD of shape:  MoM {B[:,0].std():.4f}   MLE {B[:,2].std():.4f}")
print(f"bootstrap SD of scale:  MoM {B[:,1].std():.4f}   MLE {B[:,3].std():.4f}")
```

Output:

```text
MoM  (shape, scale): [2.5489 2.3132]
MLE  (shape, scale): [3.4686 1.6999]
bootstrap SD of shape:  MoM 0.2104   MLE 0.2187
bootstrap SD of scale:  MoM 0.2154   MLE 0.1351
```

The two estimators give quite different parameters (shape 2.5 against 3.5), which is itself a sign that the gamma model
fits poorly: under correct specification both would be consistent for the same values. MLE is noticeably more precise for
the scale parameter, but not for the shape. Theory says the MLE is asymptotically efficient **if the model is correct**;
when the model is misspecified, the two estimators target different pseudo-true values and the efficiency ranking need not
hold. Compare with the previous exercise, where the gamma lost clearly to the lognormal.

</details>

## Reading
Casella and Berger, *Statistical Inference*; Lehmann and Casella, *Theory of Point Estimation*; van der Vaart, *Asymptotic Statistics*.