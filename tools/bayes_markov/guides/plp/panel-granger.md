title: Panel Granger Causality (Dumitrescu-Hurlin)
short: Panel Granger
when: You want to know whether one variable helps predict another in a panel where the relationship may differ across units, such as growth and unemployment across countries
data: Economic performance of 14 OECD countries, 1966-90 (Garrett 1998): GDP growth and unemployment
tools: Python · OES engine · statsmodels
oes: dh_causality
lead: Use this guide when you ask whether past values of one variable predict another in a panel and you do not want to force the same coefficient on every unit. The Dumitrescu-Hurlin (2012) test runs the Granger regression unit by unit and combines the Wald statistics, with a small-sample standardisation that matters when T is short. The guide covers the setup, stationarity, the test in both directions, the unit-by-unit results, sensitivity to the number of lags and the text for the paper. Granger causality is predictive content, not a causal effect.
---

## The question and the test

**Does x help predict y, in at least one unit?** For each unit \(i\) the regression is

\[
y_{it} = \alpha_i + \sum_{k=1}^{K} \gamma_{ik}\, y_{i,t-k} + \sum_{k=1}^{K} \beta_{ik}\, x_{i,t-k} + e_{it},
\]

and every coefficient, including \(\beta_{ik}\), may differ by unit. The null hypothesis is **homogeneous non-causality**: \(\beta_{ik}=0\) for all \(k\) in all units. The alternative allows causality in some units and not in others. Dumitrescu and Hurlin average the unit Wald statistics, \(\bar W = N^{-1}\sum_i W_i\), and standardise them. \(\bar Z\) is for large \(T\) and \(\tilde Z\) is for small \(T\) with fixed \(N\); with \(T\) around 25, as here, use \(\tilde Z\).

<div class="read" markdown="1">
**Three checks before you trust the test**

1. **Are both series stationary?** The test is for stationary series. A variable with a unit root must be differenced first, or the regressions are spurious.
2. **Is T long enough for the number of lags?** Each unit regression loses \(2K+1\) parameters. With \(T = 25\), more than two or three lags leaves few degrees of freedom.
3. **Do you want a causal claim?** Granger causality is about prediction. A third variable that moves both series produces it as well.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We test whether unemployment helps predict GDP growth, and the reverse, in a panel of 14 OECD countries over 1966-90 with the Dumitrescu-Hurlin (2012) Granger non-causality test for heterogeneous panels. Each country has its own regression with one lag of each variable, so the coefficients may differ across countries. GDP growth is already a growth rate and is used in levels; the unemployment rate is persistent and is used in first differences. Because T is short (24 observations per country), we base inference on the small-sample statistic \(\tilde Z\).
</div>

## Load the data and check stationarity

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

data = open("garrett.csv").read()
df = pd.read_csv("garrett.csv")
print(f"{df.country.nunique()} countries, {df.year.min()}-{df.year.max()}, observations per country: {sorted(int(n) for n in df.groupby('country').size().unique())}")
print(df[["gdp_growth", "unem"]].describe().round(2).loc[["mean", "std", "min", "max"]])
for v in ["gdp_growth", "unem"]:
    for form, dd in [("level", df), ("first difference", df.assign(**{v: df.groupby("country")[v].diff()}).dropna())]:
        r0 = oes.run("punitroot", dd.to_csv(index=False), dict(y=v, unit="country", time="year", maxlag=1, sims=200))
        t = {x[0]: x for x in r0["tables"][0]["rows"]}
        print(f"{v:11s} {form:17s} IPS p = {t['Im-Pesaran-Shin t-bar (W)'][3]:.3f}   Fisher Z p = {t['Fisher-type, inverse normal (Z)'][3]:.3f}")
```

<div class="read" markdown="1">
**How to read it**

The panel is balanced: 14 countries with 25 annual observations each, 24 after first differencing. GDP growth has a mean of 3.25% and a standard deviation of 2.43. It is stationary in levels (IPS and Fisher p-values below 0.001), as a growth rate should be, so it enters in levels. The unemployment rate, with a mean of 4.99%, does not reject a unit root in levels (IPS p = 0.64, Fisher p = 0.62) and is stationary in first differences (p < 0.001), so it enters as a change. With 14 series of 25 observations these unit-root tests have limited power, which is one more reason to keep the transformation simple and state it.
</div>

## Run the test in both directions

```python
spec = dict(y="gdp_growth", x="unem", unit="country", time="year", lags=1, diff_x="yes")
r = oes.run("dh_causality", data, spec)
oes.summary(r)
oes.show(r, "Dumitrescu-Hurlin")
```

<div class="read" markdown="1">
**How to read it**

Changes in unemployment do not help predict GDP growth: W-bar is 1.29 (it is expected to be near the number of lags, 1, under the null), Z-tilde is 0.40 and the p-value is 0.69. GDP growth does help predict changes in unemployment: W-bar is 2.29, Z-tilde is 2.59 and the p-value is 0.0097. The large-T statistic Z-bar gives a p-value of 0.0006 for the second test, sixteen times smaller than the 0.0097 from Z-tilde, which shows why the small-sample statistic should be used with 24 observations per country. The predictive relationship runs from growth to unemployment, which fits an Okun-type relationship, and not the other way.
</div>

## Which countries drive the result?

```python
oes.show(r, "Unit-by-unit")
tab = pd.DataFrame(r["tables"][1]["rows"], columns=r["tables"][1]["columns"])
print("countries where unemployment changes predict growth at 5%:", tab[tab["5% decision"] == "rejects"]["Unit"].tolist())
back = oes.run("dh_causality", data, dict(spec, y="unem", x="gdp_growth", diff_y="yes", diff_x="no"))
b = pd.DataFrame(back["tables"][1]["rows"], columns=back["tables"][1]["columns"])
print("countries where growth predicts changes in unemployment at 5%:", b[b["5% decision"] == "rejects"]["Unit"].tolist())
```

<div class="read" markdown="1">
**How to read it**

No country rejects the hypothesis that unemployment changes do not predict growth at 5%: the smallest p-value is Finland's 0.067, followed by Italy (0.100) and Canada (0.117). In the opposite direction, growth predicts changes in unemployment in Italy and the United Kingdom at 5%. With 14 countries, about 0.7 rejections would be expected by chance, so two is not many, and the panel rejection reflects modest evidence in several countries as much as strong evidence in two. The test cannot say which countries have causality, only that it is unlikely to be absent in all of them.
</div>

## How many lags?

```python
out = {}
for K in [1, 2, 3]:
    rr = oes.run("dh_causality", data, dict(spec, lags=K))
    rows = rr["tables"][0]["rows"]
    out[f"K = {K}"] = [rows[0][1], rows[0][3], rows[0][5], rows[1][1], rows[1][3], rows[1][5]]
print(pd.DataFrame(out, index=["unem -> growth: W-bar", "p (Z-bar)", "p (Z-tilde)", "growth -> unem: W-bar", "p (Z-bar)", "p (Z-tilde)"]).T.round(3))
```

<div class="read" markdown="1">
**How to read it**

The result for unemployment to growth is the same for every number of lags (Z-tilde p-values of 0.69, 0.48 and 0.19). The result for growth to unemployment is not: the Z-tilde p-value is 0.010 with one lag, 0.119 with two and 0.432 with three, while the large-T statistic Z-bar gives 0.001, 0.012 and 0.059. Each added lag uses two more parameters in a regression with 24 observations, which lowers power, so the evidence that growth predicts unemployment changes is confined to the one-lag specification. Choose the lag length before looking at the test, with an information criterion or from theory, and report the others.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

The Dumitrescu-Hurlin (2012) test does not reject the null that changes in unemployment do not Granger-cause GDP growth in any of the 14 countries (W-bar = 1.29, Z-tilde = 0.40, p = 0.69, one lag). It rejects the null that GDP growth does not Granger-cause changes in unemployment (W-bar = 2.29, Z-tilde = 2.59, p = 0.010); at the 5% level the country regressions reject for Italy and the United Kingdom. The finding is specific to the one-lag specification: with two lags the p-value is 0.12 and with three it is 0.43. Both variables are treated as stationary after taking first differences of unemployment, and inference uses the small-sample statistic because T = 24. Granger causality measures predictive content and not a causal effect.
</div>

| Report | What | Where |
|---|---|---|
| Model | Unit regressions, lags, transformations, which statistic | Step 1 |
| Data | Stationarity of both variables | Step 2 |
| Test | W-bar, Z-bar, Z-tilde in both directions | Step 3 |
| Units | Which countries reject | Step 4 |
| Robustness | Number of lags | Step 5 |

**References.** Dumitrescu, E.-I. and Hurlin, C. (2012), "Testing for Granger Non-Causality in Heterogeneous Panels," *Economic Modelling* 29. Lopez, L. and Weber, S. (2017), "Testing for Granger Causality in Panel Data," *Stata Journal* 17. Granger, C. (1969), "Investigating Causal Relations by Econometric Models and Cross-Spectral Methods," *Econometrica* 37. Garrett, G. (1998), *Partisan Politics in the Global Economy*, Cambridge University Press.
