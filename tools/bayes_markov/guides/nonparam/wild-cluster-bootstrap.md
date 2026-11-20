title: Cluster-Robust Inference and the Wild Bootstrap
short: Wild cluster bootstrap
when: Errors are correlated within groups, and with few groups the usual cluster-robust test is unreliable
data: Petersen's (2009) simulated panel of 500 firms over 10 years, used to compare standard errors; plus a simulation with few clusters
tools: Python · OES engine · NumPy
oes: wildboot
lead: Use this guide when observations come in groups (firms, schools, villages) and the errors within a group are correlated. Conventional standard errors are then too small. Cluster-robust standard errors (Liang and Zeger 1986) fix this when there are many groups; with few groups the wild cluster bootstrap-t (Cameron, Gelbach and Miller 2008) is more reliable. The guide covers the problem, the three kinds of inference, confidence intervals, a simulation of size with few clusters and the write-up.
---

## The question and the model

**How much do we overstate precision if we ignore the grouping, and how do we test with few groups?** In a regression \(y_{gi} = x_{gi}'\beta + u_{gi}\) with \(u\) correlated within group \(g\), the variance of \(\hat\beta\) is estimated by summing the scores within each group, \(\hat V = (X'X)^{-1}\bigl(\sum_g X_g' \hat u_g \hat u_g' X_g\bigr)(X'X)^{-1}\), with a small-sample factor (CR1). This is valid when the number of groups is large. The wild cluster bootstrap imposes the null, multiplies the residuals of each group by a random sign (or a Webb weight, which has six points and suits fewer than 12 groups) and uses the distribution of the t statistic.

<div class="write" markdown="1">
**How to write it: the model**

We regress y on x in Petersen's (2009) panel of 500 firms and 10 years and report conventional, heteroskedasticity-robust and cluster-robust (CR1) standard errors and a wild cluster bootstrap-t p-value that imposes the null. We then draw a simulation with 8 clusters and no true effect to compare the rejection rates of the tests at the 5 per cent level.
</div>

## Load the data

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

d = pd.read_csv("petersencl.csv")
print(len(d), "observations,", d.firm.nunique(), "firms,", d.year.nunique(), "years")
print(d[["y", "x"]].describe().round(2).loc[["mean", "std"]])
```

<div class="read" markdown="1">
**How to read it**

There are 5,000 observations on 500 firms over 10 years. Every firm is observed in 10 years, so the clusters have equal size. The outcome has a standard deviation of 2.25 and the regressor of 0.99.
</div>

## Three kinds of inference

```python
spec = dict(y="y", x=["x"], cluster="firm", reps=999, seed=1)
r = oes.run("wildboot", d.to_csv(index=False), spec)
oes.summary(r)
oes.show(r, "Inference on")
```

<div class="read" markdown="1">
**How to read it**

The slope is 1.035. The conventional standard error is 0.0286 and the heteroskedasticity-robust one 0.0284, almost the same. The cluster-robust standard error is 0.0506, 1.77 times larger, and the cluster jackknife (CR3) gives 0.0508. The t statistic falls from 36.2 to 20.5. The wild cluster bootstrap-t reports the CR1 standard error and its own p-value. All the p-values are below 0.001: with 500 clusters the inference is not in doubt, only its precision.
</div>

## Confidence intervals and clusters

```python
oes.show(r, "95% confidence intervals")
oes.show(r, "Clusters")
```

<div class="read" markdown="1">
**How to read it**

The conventional 95% interval is 0.979 to 1.091 (width 0.112). The cluster-robust interval is 0.935 to 1.134 (width 0.199) and the wild bootstrap-t interval is 0.932 to 1.138 (width 0.206): ignoring the clusters understates the uncertainty by about 44%. The cross-products of the scores within clusters are 2.2 times as large as the squared scores themselves, which is what makes the conventional variance too small: both the regressor and the error are persistent within firm.
</div>

## Robustness: a different null and weights

```python
for w in ("rademacher", "webb", "mammen"):
    e = oes.run("wildboot", d.to_csv(index=False), dict(spec, weights=w))
    print(f"{w:10s} wild bootstrap-t p-value {e['tables'][0]['rows'][4][3]:.4f}")
e = oes.run("wildboot", d.to_csv(index=False), dict(spec, null=1.0))
print("test of beta = 1:", [(row[0], row[2], row[3]) for row in e["tables"][0]["rows"] if row[0].startswith(("Cluster-robust", "Wild"))])
few = d[d.firm.isin(list(range(1, 9)))]
ef = oes.run("wildboot", few.to_csv(index=False), dict(spec, weights="webb"))
print("first 8 firms only: coefficient", ef["summary"][2]["value"])
for row in ef["tables"][0]["rows"]:
    print(f"  {row[0]:42s} se {row[1]:.4f}  p {row[3]:.4f}")
```

<div class="read" markdown="1">
**How to read it**

With 500 clusters the three weight distributions (Rademacher, Webb and Mammen) agree: the p-value is below 0.0001 for each. A test of beta = 1 is not rejected (cluster-robust p = 0.491, wild bootstrap p = 0.489). Using only the first 8 firms, the coefficient is 1.164 and the methods disagree: the conventional standard error is 0.231 (p < 0.001), the cluster-robust one is 0.445 (p = 0.035), the jackknife gives 0.495 (p = 0.051) and the wild cluster bootstrap-t with Webb weights gives p = 0.141. With 8 clusters, the conclusion at the 5% level depends on the method.
</div>

## Diagnostics: size with few clusters

```python
rng = np.random.default_rng(4)
G, m, R = 8, 25, 100
rej = dict(conv=0, cr1=0, wild=0)
for k in range(R):
    g = np.repeat(np.arange(G), m); xg = rng.normal(size=G)
    x = xg[g]; y = rng.normal(size=G)[g] + rng.normal(size=G * m)       # no effect of x; errors correlated within group
    e = oes.run("wildboot", pd.DataFrame(dict(y=y, x=x, g=g)).to_csv(index=False), dict(y="y", x=["x"], cluster="g", reps=199, seed=k, weights="webb"))
    rows = e["tables"][0]["rows"]
    rej["conv"] += rows[0][3] < 0.05; rej["cr1"] += rows[2][3] < 0.05; rej["wild"] += rows[4][3] < 0.05
print(f"true size 5%, {G} clusters of {m}, {R} samples; rejection rates:")
print("  conventional:", rej["conv"] / R, "  cluster-robust CR1 (t with G-1 df):", rej["cr1"] / R, "  wild cluster bootstrap-t:", rej["wild"] / R)
```

<div class="read" markdown="1">
**How to read it**

In 100 samples with 8 clusters of 25 and no effect of x, with errors correlated within clusters, the conventional test rejects at the 5% level in 61% of the samples, the cluster-robust test with t(G-1) critical values in 14%, and the wild cluster bootstrap-t (Webb weights) in 3%. Conventional standard errors give a test with nearly no meaning, cluster-robust standard errors are too liberal with few clusters, and the wild bootstrap is close to the nominal size (3% with 100 samples, a standard error of about 2 points).
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

We regress y on x in a panel of 500 firms and 10 years. The slope is 1.035. The conventional standard error (0.029) ignores the correlation within firms; the firm-cluster-robust standard error is 0.051, 77% larger, and the 95% interval widens from 0.112 to 0.199. With 500 clusters the wild cluster bootstrap-t gives the same conclusion as the cluster-robust test (p < 0.001), and a test of a slope of 1 is not rejected (p = 0.49). With only 8 firms the cluster-robust p-value (0.035) and the bootstrap p-value (0.141) differ, and in a simulation with 8 clusters the cluster-robust test rejects a true null in 14% of samples at the 5% level against 3% for the wild bootstrap, so we report the bootstrap when the number of clusters is small.
</div>

**References.** Liang and Zeger (1986), *Biometrika*; Cameron, Gelbach and Miller (2008), *Review of Economics and Statistics*; Webb (2014); Petersen (2009), *Review of Financial Studies*; Cameron and Miller (2015), *Journal of Human Resources*; MacKinnon and Webb (2017), *Journal of Applied Econometrics*.
