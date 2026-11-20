title: Extreme Value Theory
short: Extreme value theory
when: You care about the far tail of a distribution (large losses, floods, claims) and need probabilities and levels beyond most of the data
data: 2,167 Danish fire insurance claims, 1980-1990, in millions of Danish krone (McNeil 1997; evir, danish)
tools: Python · OES engine · SciPy
oes: evt
lead: Use this guide to model the tail of a distribution with a peaks-over-threshold approach. Above a high threshold the excesses follow a generalised Pareto distribution (Pickands 1975; Balkema and de Haan 1974). The guide covers the model, the choice of the threshold, the Hill estimator, return levels, value at risk and expected shortfall, a goodness-of-fit test, a simulation of the estimates and the write-up.
---

## The question and the model

**How large can the losses be, and how often?** For a high threshold \(u\) the excesses \(y = x - u\) are approximately generalised Pareto with shape \(\xi\) and scale \(\sigma\), \(P(y > t) = (1 + \xi t/\sigma)^{-1/\xi}\). A positive \(\xi\) means a heavy tail: moments of order above \(1/\xi\) do not exist. With \(\zeta\) the share of observations above \(u\), the level exceeded once in \(m\) observations is \(u + (\sigma/\xi)\bigl((m\zeta)^{\xi} - 1\bigr)\). The 99% value at risk is the 1-in-100 level and expected shortfall is the mean loss beyond it. The threshold trades bias (too low) against variance (too high).

<div class="write" markdown="1">
**How to write it: the model**

We fit a generalised Pareto distribution to the excesses of Danish fire claims over a threshold of 10 million krone by maximum likelihood. We check the stability of the shape across thresholds, compare it with the Hill estimator of the tail index, and report return levels with delta-method intervals, value at risk and expected shortfall at 99 and 99.5 per cent. We test the fit and check the estimator in a simulation from a known generalised Pareto distribution.
</div>

## Load the data

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

d = pd.read_csv("danish.csv")
print(len(d), "claims; mean", round(d.claim.mean(), 2), ", median", round(d.claim.median(), 2), ", maximum", round(d.claim.max(), 1))
print("the largest claim is", round(d.claim.max() / d.claim.median()), "times the median; the 10 largest claims make up", round(d.claim.nlargest(10).sum() / d.claim.sum() * 100), "% of the total")
def s(res, label):
    return next(x["value"] for x in res["summary"] if x["label"] == label)
```

<div class="read" markdown="1">
**How to read it**

There are 2,167 claims with a mean of 3.39 and a median of 1.78 million krone and a maximum of 263.3, 148 times the median. The 10 largest claims make up 13% of the total: the risk is in the tail.
</div>

## Fit the tail

```python
spec = dict(x="claim", threshold=10)
r = oes.run("evt", d.to_csv(index=False), spec)
oes.summary(r)
oes.show(r, "Generalised Pareto")
oes.show(r, "Hill estimator")
oes.show(r, "Goodness of fit")
```

<div class="read" markdown="1">
**How to read it**

Above a threshold of 10 there are 109 claims (5.0% of the sample). The generalised Pareto fit gives shape xi = 0.497 (standard error 0.136) and scale sigma = 6.98 (1.11). A positive shape means a heavy power-law tail with a tail index of 2.01: the mean exists, and the variance is at the borderline, because it needs xi below 0.5 and the confidence interval for xi contains 0.5. The Hill estimator gives 0.54 to 0.72 for 20 to 250 top observations, a tail index of 1.4 to 1.8, somewhat heavier than the likelihood estimate and with no flat region. The Kolmogorov-Smirnov test does not reject the fit (p = 0.98; the p-value is approximate because the parameters are estimated).
</div>

## Return levels, value at risk and expected shortfall

```python
oes.show(r, "Return levels")
oes.show(r, "Value at risk")
```

<div class="read" markdown="1">
**How to read it**

The claim exceeded once in 100 is 27.3 (standard error 2.8, 95% interval 21.8 to 32.8): 20 claims exceed it, which matches the 21.7 expected. The 1-in-250 level is 45.4 (32.0 to 58.7, 9 observed) and the 1-in-1,000 level is 94.3 (44.8 to 143.9, 3 observed). The interval for the 1-in-10,000 level, 305, runs from -11 to 621: the delta-method interval is symmetric and is not credible so far out. The 99% value at risk is 27.3 from the tail model and 26.0 from the empirical quantile; at 99.5% the model gives 40.2 and the data 34.8. Expected shortfall at 99% is 58.2 from the model and 61.8 from the data. At 99.9% the empirical quantile (131.6) rests on two observations and differs from the model (94.3).
</div>

## Robustness: the threshold

```python
oes.show(r, "Stability of the shape")
print("threshold    exceedances    xi     sigma   VaR 99%")
for u in (5, 8, 10, 15, 20):
    e = oes.run("evt", d.to_csv(index=False), dict(x="claim", threshold=u))
    print(f"{u:8d} {s(e, 'Exceedances'):12d} {s(e, 'Shape xi'):9.3f} {s(e, 'Scale sigma'):8.2f} {e['tables'][4]['rows'][0][1]:9.1f}")
```

<div class="read" markdown="1">
**How to read it**

The shape is not stable across thresholds: 0.69 at the 85% quantile (325 exceedances), 0.58 at 90%, 0.45 at 92.5%, 0.49 at 95% (our choice), 0.63 at 97.5% and 0.86 at 99% (22 exceedances), but the standard errors grow from 0.10 to 0.42, so most of the movement is noise. For fixed thresholds from 5 to 20 the shape is between 0.41 and 0.68, while the 99% value at risk stays between 25.8 and 28.3. Quantiles inside the range of the data are robust to the threshold; the shape, and with it the far tail, is not.
</div>

## Diagnostics: a simulation with a known tail

```python
from scipy import stats
rng = np.random.default_rng(7)
print("true xi = 0.3 (GPD data, so the excesses over any threshold keep xi = 0.3); threshold at the 90% quantile; 100 samples")
print("sample size   exceedances   mean xi-hat   s.d.   share of 95% intervals covering xi")
for n in (300, 1000, 4000):
    xs = []; cover = 0
    for k in range(100):
        e = stats.genpareto.rvs(0.3, scale=1.0, size=n, random_state=rng)
        out = oes.run("evt", pd.DataFrame(dict(x=e, i=np.arange(n))).to_csv(index=False), dict(x="x", quantile=0.9))
        row = out["tables"][0]["rows"][3]
        xs.append(row[1]); cover += abs(row[1] - 0.3) < 1.96 * row[2]
    print(f"{n:9d} {int(0.1 * n):12d} {np.mean(xs):12.3f} {np.std(xs):8.3f} {cover / 100:22.2f}")
```

<div class="read" markdown="1">
**How to read it**

When the data are generalised Pareto with xi = 0.3, the estimate is biased and noisy with few exceedances: with 30 exceedances the mean is 0.199 (a bias of -0.10), the standard deviation is 0.270 and the 95% interval covers the truth in 88% of samples; with 100 the mean is 0.282, the standard deviation 0.143 and the coverage 94%; with 400 they are 0.298, 0.064 and 94%. With the 109 exceedances of the Danish data the precision is about 0.14, as the standard error says. Report the interval for the shape, not only the point estimate.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

We fit a generalised Pareto distribution to the 109 Danish fire claims above 10 million krone (5.0% of 2,167 claims). The shape is 0.50 (standard error 0.14) and the scale 6.98 (1.11), so the tail is heavy, with a tail index of about 2; the Kolmogorov-Smirnov test does not reject the fit (p = 0.98). The 99% value at risk is 27.3 million (95% interval 21.8 to 32.8) and the expected shortfall 58.2 million; the 1-in-1,000 claim is 94 million, with a wide interval (45 to 144). The Hill estimator gives a somewhat heavier tail (0.54 to 0.72) and the shape estimate moves between 0.45 and 0.86 across thresholds, with large standard errors. The 99% value at risk is stable across thresholds (25.8 to 28.3), but levels beyond the data depend on the shape and are uncertain. In simulations the shape is biased downward with 30 exceedances and nearly unbiased with 100 or more.
</div>

**References.** Pickands (1975), *Annals of Statistics*; Balkema and de Haan (1974), *Annals of Probability*; Hill (1975), *Annals of Statistics*; McNeil (1997), *ASTIN Bulletin*; Coles (2001), *An Introduction to Statistical Modeling of Extreme Values*; McNeil, Frey and Embrechts (2015), *Quantitative Risk Management*.
