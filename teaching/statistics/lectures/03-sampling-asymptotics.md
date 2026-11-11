# Lecture 03 — Sampling Distributions and Asymptotic Theory

## Learning objectives
Distinguish sampling distributions from population distributions; apply LLN, CLT, Slutsky, continuous mapping, and delta methods; understand bootstrap approximation.

## 1. Sampling distributions
An estimator \(\widehat\theta=T(X_1,\ldots,X_n)\) is random before the sample is observed. Its sampling distribution determines bias, variance, mean squared error, and coverage. The decomposition \(MSE(\widehat\theta)=Var(\widehat\theta)+Bias(\widehat\theta)^2\) shows why unbiasedness alone is not a complete optimality criterion.

## 2. Law of large numbers
For i.i.d. observations with finite first absolute moment, the sample mean converges in probability to its population mean:
\[
\bar X_n\xrightarrow{p}\mu.
\]
This result supports consistency of sample averages and sample moments.

## 3. Central limit theorem
For i.i.d. observations with finite mean \(\mu\) and variance \(0<\sigma^2<\infty\),
\[
\sqrt n(\bar X_n-\mu)\xrightarrow{d}N(0,\sigma^2).
\]
The CLT provides a large-sample approximation; it does not assert that the finite-sample population distribution is normal.

## 4. Slutsky and continuous mapping
If \(X_n\xrightarrow{d}X\) and \(Y_n\xrightarrow{p}c\), then sums and products converge to the corresponding transformations involving \(X\) and \(c\). If \(g\) is continuous, \(g(X_n)\xrightarrow{d}g(X)\). These results justify replacing unknown constants with consistent estimators in asymptotic statistics.

## 5. Delta method
If \(\sqrt n(\widehat\theta-\theta)\xrightarrow{d}N(0,V)\) and \(g\) is differentiable at \(\theta\), then
\[
\sqrt n[g(\widehat\theta)-g(\theta)]\xrightarrow{d}N(0,[g'(\theta)]^2V).
\]
For vector parameters, use the gradient and covariance matrix. The first-order method may fail when the derivative is zero.

## 6. Bootstrap
The nonparametric bootstrap samples with replacement from the empirical distribution and recomputes the estimator. Percentile and studentized intervals are common. Valid resampling must preserve dependence: clustered observations require cluster resampling, while time series require dependence-aware methods.

## Worked example
For the sample mean, replacing \(\sigma\) by the consistent sample standard deviation \(S\) gives \(\sqrt n(\bar X-\mu)/S\xrightarrow{d}N(0,1)\) by Slutsky. Applying the delta method to \(g(\mu)=\log\mu\) gives asymptotic variance \(\sigma^2/\mu^2\).

## Problem set
1. Prove sample-mean consistency using Chebyshev's inequality.
2. State the assumptions and conclusion of the CLT.
3. Use Slutsky to justify studentization.
4. Derive the delta-method variance for a logarithm.
5. Explain why an i.i.d. bootstrap is unsuitable for clustered data.

## Computational lab
Simulate sample means from normal, skewed, and heavy-tailed populations. Compare finite-sample distributions and coverage of normal and bootstrap intervals as \(n\) increases.

## Coding exercises

These exercises use **real data** in Python. Install once with
`pip install wooldridge statsmodels scipy numpy pandas`. The `wooldridge` package holds survey, experimental and
administrative datasets; `scipy.stats` and `statsmodels` provide the inference tools. Try the starter code first, then
open the solution. Every printed result below comes from running the solution code exactly as shown.

### Exercise 3.1 — The CLT with a heavily skewed real population: CEO salaries

*Data:* `ceosal1` — 1990 salaries (thousands of dollars) of 209 CEOs. We treat these 209 values as a population.

1. Describe the population: mean, median, skewness.
2. Draw 10,000 samples with replacement of size \(n=5, 30, 200\) and record the sample mean. For each \(n\), report the skewness of the sampling distribution and the coverage of the nominal 95% \(t\)-interval for the population mean.
3. Relate the results to the Berry–Esseen idea that the normal approximation error shrinks like \(\text{skewness}/\sqrt n\).

```python
import numpy as np, wooldridge as wd
from scipy import stats
pop = wd.data("ceosal1")["salary"].to_numpy(); mu = pop.mean()
rng = np.random.default_rng(0)
# TODO: for n in (5, 30, 200): samples = rng.choice(pop, (10000, n)); compute means, t-intervals, coverage
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
from scipy import stats
pop = wd.data("ceosal1")["salary"].to_numpy(); mu = pop.mean()
print(f"population: mean {mu:.1f}, median {np.median(pop):.1f}, skewness {stats.skew(pop):.2f}")
rng = np.random.default_rng(0)
for n in (5, 30, 200):
    s = rng.choice(pop, (10000, n))
    m, sd = s.mean(1), s.std(1, ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    cover = np.mean(np.abs(m - mu) <= half)
    print(f"n = {n:3d}: skewness of sample mean {stats.skew(m):5.2f}, t-interval coverage {cover:.3f} (MC se {np.sqrt(cover*(1-cover)/10000):.3f})")
```

Output:

```text
population: mean 1281.1, median 1039.0, skewness 6.85
n =   5: skewness of sample mean  3.20, t-interval coverage 0.856 (MC se 0.004)
n =  30: skewness of sample mean  1.29, t-interval coverage 0.790 (MC se 0.004)
n = 200: skewness of sample mean  0.48, t-interval coverage 0.893 (MC se 0.003)
```

CEO salaries are extremely skewed (skewness 6.9, with one salary above \$14 million). The sampling distribution of the
mean is still visibly skewed at \(n=200\), and the \(t\)-interval **undercovers badly at every sample size** — 79–89%
instead of 95%. Coverage is not even monotone in \(n\): at \(n=5\), most samples miss the extreme salaries entirely,
which produces short intervals that happen to cover the mean more often than at \(n=30\). The Berry–Esseen bound says the
error shrinks like skewness\(/\sqrt n\): with skewness near 7, "large \(n\)" means thousands. Use logs, a bootstrap-t
interval, or report the median.

</details>

### Exercise 3.2 — Delta method and bootstrap for a ratio: the 401(k) participation rate

*Data:* `401k` — 1,534 U.S. 401(k) plans: number of participants `totpart` and eligible employees `totelg`.

The aggregate participation rate is the ratio \(R=E[totpart]/E[totelg]\).

1. Estimate \(R\) and its standard error by the delta method, using the gradient \((1/\mu_e,\,-\mu_p/\mu_e^2)\) and the sample covariance matrix of the means.
2. Estimate the standard error with a nonparametric bootstrap (2,000 draws, seed 1).
3. Compare \(R\) with the average of plan-level rates \(E[prate]/100\). Why do they differ?

```python
import numpy as np, wooldridge as wd
d = wd.data("401k"); p, e = d["totpart"].to_numpy(), d["totelg"].to_numpy(); n = len(d)
R = p.mean() / e.mean()
# TODO: V = np.cov(np.vstack([p, e])) / n; se = sqrt(g' V g)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
d = wd.data("401k"); p, e = d["totpart"].to_numpy(float), d["totelg"].to_numpy(float); n = len(d)
R = p.mean() / e.mean()
V = np.cov(np.vstack([p, e])) / n
g = np.array([1 / e.mean(), -p.mean() / e.mean()**2])
print(f"aggregate participation rate R = {R:.4f}, delta-method se = {np.sqrt(g @ V @ g):.4f}")
rng = np.random.default_rng(1); idx = rng.integers(0, n, (2000, n))
boot = p[idx].mean(1) / e[idx].mean(1)
print(f"bootstrap se = {boot.std(ddof=1):.4f}, 95% percentile CI [{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}]")
print(f"average plan-level rate = {d['prate'].mean()/100:.4f}")
print(f"correlation between plan size (totelg) and plan rate: {np.corrcoef(e, d['prate'])[0,1]:.3f}")
```

Output:

```text
aggregate participation rate R = 0.8316, delta-method se = 0.0158
bootstrap se = 0.0157, 95% percentile CI [0.8001, 0.8600]
average plan-level rate = 0.8736
correlation between plan size (totelg) and plan rate: -0.076
```

The delta-method and bootstrap standard errors agree closely (about 0.016), because the ratio of two means is smooth and
\(n\) is large. The aggregate rate (83%) is lower than the average plan rate (87%): the aggregate weights plans by their
number of eligible employees, and larger plans have slightly lower participation (negative correlation). The two numbers
answer different questions — "what share of eligible workers participate?" versus "what is the typical plan's rate?".

</details>

## Reading
van der Vaart, *Asymptotic Statistics*; Casella and Berger, *Statistical Inference*; Efron and Tibshirani, *An Introduction to the Bootstrap*.