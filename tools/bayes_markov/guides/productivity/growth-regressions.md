title: Growth Regressions: Solow and Convergence
short: Growth regressions
when: You have cross-country (or cross-region) data on income, investment, population growth and schooling and want to test the Solow model and the speed of convergence
data: Income per working-age person in 1960 and 1985, investment rate, population growth and schooling for 121 countries (Durlauf and Johnson 1995; Ecdat, GrowthDJ); the Mankiw-Romer-Weil non-oil sample has 98 countries
tools: Python · OES engine · NumPy
oes: growth
lead: Use this guide to test whether the Solow growth model explains differences in income across countries and whether poor countries catch up. The guide follows Mankiw, Romer and Weil (1992): the level regression with the restriction that the coefficients on saving and on population growth are equal and opposite, the implied capital share, the augmented model with human capital, and the regression of growth on initial income for unconditional and conditional convergence. It covers robustness to the sample, a simulation with a known capital share and the write-up.
---

## The question and the model

**Why are some countries rich, and do poor countries grow faster?** In the Solow model, steady-state income per worker depends on the saving rate \(s\) and on \(n + g + \delta\), the sum of population growth, technical progress and depreciation:

\[
\ln y = a + \frac{\alpha}{1-\alpha}\ln s - \frac{\alpha}{1-\alpha}\ln(n+g+\delta) .
\]

The coefficients on \(\ln s\) and \(\ln(n+g+\delta)\) are equal and opposite, and the ratio \(\alpha/(1-\alpha)\) identifies the capital share \(\alpha\). Mankiw, Romer and Weil (1992) add human capital \(h\) (schooling), which adds \(\frac{\beta}{1-\alpha-\beta}\ln h\). Convergence is a statement about growth: with speed \(\lambda\), \(\ln y_T - \ln y_0 = -(1 - e^{-\lambda T})\ln y_0 + \dots\); it is *unconditional* if growth falls with initial income alone, *conditional* if it does so once \(s\), \(n\) and \(h\) are held constant. As usual \(g + \delta = 0.05\).

<div class="write" markdown="1">
**How to write it: the model**

We estimate the Mankiw-Romer-Weil model on 98 non-oil countries: the logarithm of 1985 income per working-age person on the logarithms of the investment rate, of \(n + 0.05\) and of the schooling rate, with robust standard errors; we test the Solow restriction that the first two coefficients sum to zero and compute the implied capital share. We then regress growth from 1960 to 1985 on initial income, alone and with the three controls, and report the speed of convergence and the half-life.
</div>

## Load the data

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

d = pd.read_csv("growthdj.csv")
print(len(d), "countries;", (d.oil == "no").sum(), "non-oil;", (d.inter == "yes").sum(), "intermediate sample;", (d.oecd == "yes").sum(), "OECD")
print(d[["gdp60", "gdp85", "invest", "popgrowth", "school"]].describe().round(2).loc[["count", "mean", "min", "max"]])
d["growth"] = np.log(d.gdp85 / d.gdp60) / 25
print("average annual growth 1960-85: mean", round(d.growth.mean() * 100, 2), "%; the 10 poorest in 1960:", round(d.nsmallest(10, "gdp60").growth.mean() * 100, 2), "%; the 10 richest:", round(d.nlargest(10, "gdp60").growth.mean() * 100, 2), "%")
```

<div class="read" markdown="1">
**How to read it**

There are 121 countries: 98 non-oil, 75 in the intermediate sample and 22 in the OECD sample. Some variables are missing for some countries (116 values of 1960 income, 108 of 1985 income, 107 of population growth, 118 of schooling), and the engine uses the countries with all of them. Income per worker spans a factor of 200 in 1960 (383 to 77,881). The highest 1960 income is three times the highest 1985 income (25,635), which is the mark of an oil exporter; Mankiw, Romer and Weil drop oil producers, and the sample filter does the same here. Average growth is 1.8% a year; the ten poorest countries in 1960 grew at 1.88% and the ten richest at 0.62%, which looks like convergence but is a comparison of two small groups.
</div>

## The Solow regression in levels

```python
spec = dict(y0="gdp60", y1="gdp85", invest="invest", popgrowth="popgrowth", school="school", sample="oil", sample_value="no")
r = oes.run("growth", d.to_csv(index=False), spec)
oes.summary(r)
oes.show(r, "Income per worker in levels")
oes.show(r, "Tests and implied")
oes.show(r, "Restricted regression")
```

<div class="read" markdown="1">
**How to read it**

The coefficients on the saving rate and on \(n + g + \delta\) are 0.70 and -1.75, and the schooling coefficient is 0.65, all with t statistics above 4.6; the R-squared is 0.79. They agree to the second decimal with the values reported by Mankiw, Romer and Weil (0.69, -1.73 and 0.66). The Solow model says the first two coefficients are equal and opposite; their sum is -1.05, and the F test rejects the restriction at 5% (F = 5.21, p = 0.025). The restricted regression gives 0.835 on \(\ln s - \ln(n+g+\delta)\) and 0.635 on schooling, an implied capital share of 0.338 and a human capital share of 0.257, close to the one-third that national accounts suggest. Without schooling the same regression would imply a capital share of about 0.6, which is too large (see below): schooling is correlated with saving, and leaving it out inflates the saving coefficient.
</div>

## Convergence

```python
oes.show(r, "Unconditional convergence")
oes.show(r, "Conditional convergence")
oes.plot(r)
```

<div class="read" markdown="1">
**How to read it**

Unconditional convergence fails: the coefficient on initial income is +0.094 (t = 2.16), so richer countries grew slightly faster, and the model implies no convergence. Conditional on the saving rate, population growth and schooling it is -0.288 (t = -5.3): countries converge to their own steady states. The implied speed is 1.36% a year, a half-life of 50.9 years, slower than the 2% a year that Barro and Sala-i-Martin found for U.S. states. Saving (+0.52) and schooling (+0.23) raise growth, and population growth (-0.51) lowers it. The chart plots growth against initial income with the unconditional fit, which slopes up.
</div>

## Robustness: samples and specification

```python
for name, sp in [("intermediate sample", dict(spec, sample="inter", sample_value="yes")), ("all countries (oil included)", {k: v for k, v in spec.items() if k not in ("sample", "sample_value")})]:
    rr = oes.run("growth", d.to_csv(index=False), sp)
    t = rr["tables"]
    print(f"--- {name}: {rr['nobs']} countries")
    print("  level coefficients (s, n+g+d, h):", [round(row[1], 3) for row in t[0]["rows"][1:]], ", Solow restriction p =", t[1]["rows"][0][2])
    print("  capital share (augmented):", rr["tables"][1]["rows"][1][1], ", unconditional beta", rr["summary"][2]["value"], ", conditional beta", rr["summary"][3]["value"])
rt = oes.run("growth", d.to_csv(index=False), {k: v for k, v in spec.items() if k != "school"})
print("--- textbook Solow (no schooling):", [round(row[1], 3) for row in rt["tables"][0]["rows"][1:]], ", implied alpha", rt["tables"][1]["rows"][1][1])
rg = oes.run("growth", d.to_csv(index=False), dict(spec, gd=0.03))
print("--- g + delta = 0.03: capital share", rg["summary"][1]["value"])
```

<div class="read" markdown="1">
**How to read it**

*Sample.* In the intermediate sample (75 countries) the level coefficients are 0.70, -1.50 and 0.73, the capital share is 0.326, the Solow restriction is not rejected at 5% (p = 0.085), and the unconditional coefficient is -0.004 (no convergence). With all 104 countries with data, oil producers included, the population-growth coefficient shrinks to -0.75, the capital share to 0.277 and the restriction is not rejected (p = 0.82); conditional convergence remains (-0.297). *No schooling.* In the textbook model the coefficients are 1.42 and -1.99 and the implied capital share is 0.598. *Depreciation and growth.* With \(g + \delta = 0.03\) instead of 0.05 the capital share is 0.332 (against 0.338). The qualitative results, no unconditional but strong conditional convergence and a capital share near one third, are stable; the rejection of the Solow restriction is not, and depends on the sample.
</div>

## Diagnostics: does the test recover a known capital share?

```python
rng = np.random.default_rng(0)
n = 400
s = rng.uniform(5, 30, n); nn = rng.uniform(0.5, 3, n); h = rng.uniform(2, 15, n)
for a, b in [(1/3, 1/3), (0.6, 0.2)]:
    ly = 8 + a/(1-a-b)*(np.log(s/100) - np.log(nn/100 + .05)) + b/(1-a-b)*np.log(h/100) + rng.normal(0, .2, n)
    sim = pd.DataFrame(dict(y0=np.exp(ly - 0.3), y1=np.exp(ly), s=s, n=nn, h=h))
    rs = oes.run("growth", sim.to_csv(index=False), dict(y0="y0", y1="y1", invest="s", popgrowth="n", school="h"))
    print(f"true alpha {a:.3f}: estimated {rs['tables'][1]['rows'][1][1]:.3f}; Solow restriction p = {rs['tables'][1]['rows'][0][2]}")
# violating the restriction: income depends on saving and population growth with unequal weights
ly = 8 + 0.8*np.log(s/100) - 0.3*np.log(nn/100 + .05) + 0.5*np.log(h/100) + rng.normal(0, .2, n)
sim = pd.DataFrame(dict(y0=np.exp(ly - 0.3), y1=np.exp(ly), s=s, n=nn, h=h))
rs = oes.run("growth", sim.to_csv(index=False), dict(y0="y0", y1="y1", invest="s", popgrowth="n", school="h"))
print("restriction violated: p =", rs["tables"][1]["rows"][0][2])
```

<div class="read" markdown="1">
**How to read it**

With a known capital share of one third and a human capital share of one third (400 countries), the estimated capital share is 0.343 and the Solow restriction is not rejected (p = 0.80). With a capital share of 0.6 and 0.2 for human capital, the estimate is 0.602 (p = 0.24). When we build income from saving and population growth with unequal weights (0.8 on saving, -0.3 on population growth), the test rejects with a p-value of 0.0. The test suite checks the coefficients and robust standard errors against statsmodels and the F statistic against the classical restriction test.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

We test the augmented Solow model of Mankiw, Romer and Weil on 98 non-oil countries. The logarithm of 1985 income per working-age person rises with the saving rate (0.70, s.e. 0.15) and schooling (0.65, s.e. 0.07) and falls with \(n + g + \delta\) (-1.75, s.e. 0.34); the R-squared is 0.79. The equal-and-opposite restriction of the Solow model is rejected at 5% (F = 5.21, p = 0.025) in this sample but not in the intermediate sample (p = 0.085) or with oil producers included (p = 0.82). The restricted model implies a capital share of 0.34. There is no unconditional convergence (coefficient on 1960 income +0.094, t = 2.2), but there is conditional convergence (-0.288, t = -5.3), at a speed of 1.4% a year with a half-life of 51 years. These are cross-sectional associations: saving, population growth and schooling are themselves outcomes, so the coefficients are not causal effects.
</div>

**References.** Solow (1956), *Quarterly Journal of Economics*; Mankiw, Romer and Weil (1992), *Quarterly Journal of Economics*; Barro and Sala-i-Martin (1992), *Journal of Political Economy*; Durlauf and Johnson (1995), *Journal of Applied Econometrics*; Durlauf, Johnson and Temple (2005), *Handbook of Economic Growth*.
