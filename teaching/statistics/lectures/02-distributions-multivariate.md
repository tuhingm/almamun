# Lecture 02 — Probability Distributions and Multivariate Models

## Learning objectives
Derive moments of standard distributions; transform random variables using Jacobians; obtain joint, marginal, and conditional distributions; interpret covariance matrices and multivariate normality.

## 1. Discrete distributions
For \(X\sim Bernoulli(p)\), \(E(X)=p\) and \(Var(X)=p(1-p)\). A sum of independent Bernoulli indicators is \(Binomial(n,p)\), with mean \(np\) and variance \(np(1-p)\). The Poisson mass function is \(P(X=x)=e^{-\lambda}\lambda^x/x!\), with mean and variance \(\lambda\). State the support and parameterization whenever using a distribution.

## 2. Continuous distributions
The uniform, exponential, gamma, beta, normal, lognormal, chi-square, Student-t, and F distributions appear in probability modeling and inference. For example, the exponential density is \(f(x)=\lambda e^{-\lambda x}\), \(x\ge0\), with mean \(1/\lambda\) and variance \(1/\lambda^2\). The normal family is closed under linear transformations; chi-square, t, and F distributions arise from independent normal and chi-square quantities.

## 3. Transformations
For a monotone differentiable scalar transformation \(Y=g(X)\),
\[
f_Y(y)=f_X(g^{-1}(y))\left|\frac{d}{dy}g^{-1}(y)\right|.
\]
For a one-to-one vector transformation, multiply the original density by the absolute determinant of the inverse Jacobian. For non-monotone scalar transformations, sum over all inverse branches. The support must be transformed as well as the density.

## 4. Joint and conditional distributions
Marginal densities integrate out other coordinates. The conditional density is \(f_{X|Y}(x|y)=f_{X,Y}(x,y)/f_Y(y)\) where the denominator is positive. Independence is equivalent to factorization of the joint distribution into its marginals. Conditional independence is a separate property.

## 5. Covariance matrices and multivariate normality
For random vector \(X\), \(\Sigma=E[(X-\mu)(X-\mu)']\). It is symmetric positive semidefinite because \(a'\Sigma a=Var(a'X)\ge0\). If \(X\sim N_p(\mu,\Sigma)\), every linear combination is normal. Within the jointly normal family, zero covariance implies independence; this implication is not generally valid outside that family.

## Worked example
If \(X\sim N(\mu,\sigma^2)\), then \(Z=(X-\mu)/\sigma\sim N(0,1)\). For independent standard normals, the sum of \(k\) squared values follows \(\chi_k^2\), which underlies exact variance inference.

## Problem set
1. Derive the binomial mean and variance using indicator variables.
2. Transform an exponential variable using \(Y=\log X\).
3. Derive a marginal density from a bivariate joint density.
4. Prove covariance matrices are positive semidefinite.
5. Explain why zero covariance implies independence for jointly normal variables but not generally.

## Computational lab
Simulate binomial, Poisson, gamma, and multivariate normal data. Compare empirical and theoretical moments, and visualize how covariance changes the geometry of bivariate normal samples.

## Coding exercises

These exercises use **real data** in Python. Install once with
`pip install wooldridge statsmodels scipy numpy pandas`. The `wooldridge` package holds survey, experimental and
administrative datasets; `scipy.stats` and `statsmodels` provide the inference tools. Try the starter code first, then
open the solution. Every printed result below comes from running the solution code exactly as shown.

### Exercise 2.1 — Which distribution fits wages? Lognormal versus gamma by maximum likelihood

*Data:* `wage1` — 526 hourly wages, CPS 1976.

1. Fit lognormal and gamma distributions to `wage` by maximum likelihood (`scipy.stats`, location fixed at 0).
2. Compare them by log-likelihood and AIC.
3. Compare fitted and empirical quantiles (10th, 50th, 90th, 99th percentiles).
4. Run a Kolmogorov–Smirnov test for each. Why are the KS p-values only approximate here?

```python
import numpy as np, wooldridge as wd
from scipy import stats
w = wd.data("wage1")["wage"].to_numpy()
ln = stats.lognorm.fit(w, floc=0); ga = stats.gamma.fit(w, floc=0)
# TODO: log-likelihoods via .logpdf(w, *params).sum(); AIC = 2k - 2 logL; quantiles via .ppf
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
from scipy import stats
w = wd.data("wage1")["wage"].to_numpy()
fits = {"lognormal": (stats.lognorm, stats.lognorm.fit(w, floc=0)), "gamma": (stats.gamma, stats.gamma.fit(w, floc=0))}
q = [0.10, 0.50, 0.90, 0.99]
print("empirical quantiles:      ", np.quantile(w, q).round(2))
for name, (dist, par) in fits.items():
    ll = dist.logpdf(w, *par).sum()
    ks = stats.kstest(w, dist.cdf, args=par)
    print(f"{name:9s}: logL = {ll:8.2f}, AIC = {2*2 - 2*ll:7.2f}, quantiles {dist.ppf(q, *par).round(2)}, KS = {ks.statistic:.3f} (p = {ks.pvalue:.3f})")
```

Output:

```text
empirical quantiles:       [ 2.92  4.65 10.   19.99]
lognormal: logL = -1267.28, AIC = 2538.56, quantiles [ 2.57  5.07 10.01 17.44], KS = 0.089 (p = 0.000)
gamma    : logL = -1298.24, AIC = 2600.47, quantiles [ 2.37  5.34 10.14 15.62], KS = 0.103 (p = 0.000)
```

The lognormal fits much better than the gamma by log-likelihood and AIC (a difference of over 60 AIC points), and it
tracks the middle and upper quantiles more closely. Both understate the 99th percentile, so the right tail of wages is
heavier than either model allows. Both KS tests reject — with 526 observations, even moderate misfit is detected. The KS
p-values are only approximate because the parameters were estimated from the same data (a Lilliefors-type correction, or
a parametric bootstrap, gives exact p-values).

</details>

### Exercise 2.2 — Conditional normal distribution and the regression line

*Data:* `econmath` — 856 students in an introductory economics course: ACT math score `actmth` and college GPA `colgpa`.

If \((X,Y)\) is bivariate normal, \(E[Y\mid X=x]=\mu_Y+\frac{\sigma_{XY}}{\sigma_X^2}(x-\mu_X)\) and \(\operatorname{Var}(Y\mid X)=\sigma_Y^2(1-\rho^2)\).

1. Estimate the mean vector and covariance matrix of `(actmth, colgpa)` (drop missing values).
2. Compute the implied conditional mean and SD of GPA at ACT math scores 20, 25 and 30.
3. Compare the conditional-mean slope with the OLS slope, and the conditional SD with the regression's residual SD.
4. Check the normality assumption for GPA with a skewness statistic. Is a GPA above 4.0 possible under the fitted normal model?

```python
import numpy as np, wooldridge as wd
d = wd.data("econmath")[["actmth", "colgpa"]].dropna()
mu = d.mean().to_numpy(); S = d.cov().to_numpy()
# TODO: slope = S[0,1]/S[0,0]; cond_sd = sqrt(S[1,1]*(1-rho^2))
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
from scipy import stats
d = wd.data("econmath")[["actmth", "colgpa"]].dropna()
mu = d.mean().to_numpy(); S = d.cov().to_numpy(); rho = S[0, 1] / np.sqrt(S[0, 0] * S[1, 1])
slope = S[0, 1] / S[0, 0]; csd = np.sqrt(S[1, 1] * (1 - rho**2))
print(f"n = {len(d)}, means {mu.round(3)}, correlation {rho:.3f}")
for x in (20, 25, 30):
    m = mu[1] + slope * (x - mu[0])
    print(f"  ACT math {x}: E[GPA|x] = {m:.3f}, SD = {csd:.3f}, P(GPA > 4 | x) = {stats.norm.sf(4, m, csd):.4f}")
ols = smf.ols("colgpa ~ actmth", data=d).fit()
print(f"conditional-mean slope {slope:.5f} vs OLS slope {ols.params['actmth']:.5f};  conditional SD {csd:.4f} vs residual SD {np.sqrt(ols.mse_resid):.4f}")
print(f"skewness of GPA: {stats.skew(d['colgpa']):.3f}  (max observed {d['colgpa'].max():.2f})")
```

Output:

```text
n = 814, means [23.211  2.811], correlation 0.300
  ACT math 20: E[GPA|x] = 2.673, SD = 0.514, P(GPA > 4 | x) = 0.0049
  ACT math 25: E[GPA|x] = 2.888, SD = 0.514, P(GPA > 4 | x) = 0.0153
  ACT math 30: E[GPA|x] = 3.102, SD = 0.514, P(GPA > 4 | x) = 0.0404
conditional-mean slope 0.04287 vs OLS slope 0.04287;  conditional SD 0.5143 vs residual SD 0.5147
skewness of GPA: -0.132  (max observed 4.00)
```

The slope implied by the bivariate normal formula, \(\sigma_{XY}/\sigma_X^2\), is identical to the OLS slope: the
population regression of \(Y\) on \(X\) is the normal conditional mean. The conditional SD matches the residual SD
(up to the degrees-of-freedom correction). The correlation of 0.30 means ACT math explains only 9% of GPA variance.
The normal model assigns small but positive probability to a GPA above 4.0, which is impossible — GPA is bounded and
slightly left-skewed, so the normal model is an approximation that fails in the tails.

</details>

## Reading
Casella and Berger, *Statistical Inference*; DeGroot and Schervish, *Probability and Statistics*; Anderson, *An Introduction to Multivariate Statistical Analysis*.