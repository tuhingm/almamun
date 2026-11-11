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

## Coding exercises

These exercises use **real data** in Python. Install once with
`pip install wooldridge statsmodels scipy numpy pandas`. The `wooldridge` package holds survey, experimental and
administrative datasets; `scipy.stats` and `statsmodels` provide the inference tools. Try the starter code first, then
open the solution. Every printed result below comes from running the solution code exactly as shown.

### Exercise 6.1 — Beta–binomial updating: response rates in a charity mail campaign

*Data:* `charity` — 4,268 people who received a charity's mail appeal; `respond` = 1 if they gave.

1. With a uniform \(\text{Beta}(1,1)\) prior, compute the posterior for the response rate after the first 50, 500 and all 4,268 letters (in the data's order). Report posterior means and 95% credible intervals.
2. Repeat with an informative prior \(\text{Beta}(2,18)\) (prior mean 10%, worth 20 letters). When does the prior stop mattering?
3. Compute the posterior predictive probability that at least 400 of the next 1,000 letters get a response.

```python
import numpy as np, wooldridge as wd
from scipy import stats
r = wd.data("charity")["respond"].to_numpy()
# TODO: posterior Beta(a + k, b + n - k); stats.beta.ppf([0.025, 0.975], ...); betabinom for prediction
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
from scipy import stats
r = wd.data("charity")["respond"].to_numpy()
for a, b, lab in ((1, 1, "uniform Beta(1,1)"), (2, 18, "informative Beta(2,18)")):
    print(lab)
    for n in (50, 500, len(r)):
        k = r[:n].sum(); post = stats.beta(a + k, b + n - k)
        lo, hi = post.ppf([0.025, 0.975])
        print(f"   n = {n:5d}, responses {k:5d}: posterior mean {post.mean():.4f}, 95% interval [{lo:.4f}, {hi:.4f}]")
k, n = r.sum(), len(r)
pred = stats.betabinom(1000, 1 + k, 1 + n - k)
print(f"P(at least 400 of next 1000 respond | data, uniform prior) = {pred.sf(399):.4f}")
```

Output:

```text
uniform Beta(1,1)
   n =    50, responses     7: posterior mean 0.1538, 95% interval [0.0702, 0.2626]
   n =   500, responses   202: posterior mean 0.4044, 95% interval [0.3619, 0.4476]
   n =  4268, responses  1707: posterior mean 0.4000, 95% interval [0.3854, 0.4147]
informative Beta(2,18)
   n =    50, responses     7: posterior mean 0.1286, 95% interval [0.0614, 0.2157]
   n =   500, responses   202: posterior mean 0.3923, 95% interval [0.3508, 0.4346]
   n =  4268, responses  1707: posterior mean 0.3986, 95% interval [0.3839, 0.4133]
P(at least 400 of next 1000 respond | data, uniform prior) = 0.5105
```

After 50 letters, the posterior is wide and depends on the prior: the informative prior pulls the estimate toward 10%. By
500 letters, the two posteriors differ by about one percentage point; with all 4,268, they are practically identical —
the data swamp a prior worth 20 observations. The first 50 letters are not representative: only 14% responded, against
40% overall, so the data are sorted in some way. That is a warning that sequential updating assumes exchangeability. The
posterior predictive probability of at least 400 responses in 1,000 is about one half, because 400 is almost exactly the
expected number.

</details>

### Exercise 6.2 — A random-walk Metropolis sampler for a lognormal wage model

*Data:* `wage1` — hourly wages.

Model \(\log wage_i\sim N(\mu,\sigma^2)\) with priors \(\mu\sim N(0,10^2)\) and \(\log\sigma\sim N(0,1)\).

1. Write the log posterior and a random-walk Metropolis sampler on \((\mu,\log\sigma)\). Run 4 chains of 5,000 draws from dispersed starting values; discard the first 1,000 of each.
2. Report the acceptance rate, the posterior means and 95% credible intervals, and the split-\(\widehat R\) for each parameter.
3. Compare with the frequentist estimates. Then report the posterior for the **median** wage \(e^{\mu}\) and the **mean** wage \(e^{\mu+\sigma^2/2}\).

```python
import numpy as np, wooldridge as wd
from scipy import stats
y = np.log(wd.data("wage1")["wage"].to_numpy())
def logpost(m, ls):
    return stats.norm.logpdf(y, m, np.exp(ls)).sum() + stats.norm.logpdf(m, 0, 10) + stats.norm.logpdf(ls, 0, 1)
# TODO: chains; proposal theta + 0.05 * normal; accept with prob exp(new - old)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
from scipy import stats
y = np.log(wd.data("wage1")["wage"].to_numpy())
def logpost(th):
    m, ls = th
    return stats.norm.logpdf(y, m, np.exp(ls)).sum() + stats.norm.logpdf(m, 0, 10) + stats.norm.logpdf(ls, 0, 1)
rng = np.random.default_rng(7); chains, acc = [], 0
for start in ([0.0, 1.0], [3.0, -2.0], [-1.0, 0.0], [2.0, 0.5]):
    th = np.array(start); lp = logpost(th); out = []
    for i in range(5000):
        prop = th + rng.normal(0, [0.04, 0.05]); lpp = logpost(prop)
        if np.log(rng.random()) < lpp - lp:
            th, lp = prop, lpp; acc += 1
        out.append(th.copy())
    chains.append(np.array(out[1000:]))
C = np.array(chains)                                   # 4 x 4000 x 2
print(f"acceptance rate {acc / 20000:.3f}")
def split_rhat(x):
    h = x.shape[1] // 2; s = np.vstack([x[:, :h], x[:, h:]])
    W = s.var(1, ddof=1).mean(); B = h * s.mean(1).var(ddof=1)
    return np.sqrt(((h - 1) / h * W + B / h) / W)
for j, name in enumerate(("mu", "log sigma")):
    d = C[:, :, j].ravel()
    print(f"{name:9s}: mean {d.mean():.4f}, 95% [{np.percentile(d, 2.5):.4f}, {np.percentile(d, 97.5):.4f}], split R-hat {split_rhat(C[:, :, j]):.4f}")
print(f"frequentist: mean log wage {y.mean():.4f} (se {y.std(ddof=1)/np.sqrt(len(y)):.4f}), log sd {np.log(y.std(ddof=0)):.4f}")
mu, sig = C[:, :, 0].ravel(), np.exp(C[:, :, 1].ravel())
for lab, v in (("median wage e^mu", np.exp(mu)), ("mean wage e^(mu+s^2/2)", np.exp(mu + sig**2 / 2))):
    print(f"{lab:23s}: posterior mean ${v.mean():.2f}, 95% [{np.percentile(v, 2.5):.2f}, {np.percentile(v, 97.5):.2f}]")
```

Output:

```text
acceptance rate 0.359
mu       : mean 1.6229, 95% [1.5777, 1.6705], split R-hat 1.0022
log sigma: mean -0.6310, 95% [-0.6882, -0.5689], split R-hat 1.0011
frequentist: mean log wage 1.6233 (se 0.0232), log sd -0.6329
median wage e^mu       : posterior mean $5.07, 95% [4.84, 5.31]
mean wage e^(mu+s^2/2) : posterior mean $5.84, 95% [5.57, 6.14]
```

The acceptance rate of about 36% is in the efficient range for a random-walk sampler. All four chains, from very
different starting points, converge to the same distribution: split-\(\widehat R\) is about 1.00. With 526 observations
and weak priors, the posterior means equal the frequentist estimates to two or three decimals. The posterior of a
**function** of parameters is immediate from the draws: the median wage is about \$5.07 and the mean wage about \$5.84,
reflecting the right skew of a lognormal.

</details>

## Reading
Gelman et al., *Bayesian Data Analysis*; Robert, *The Bayesian Choice*; McElreath, *Statistical Rethinking*.