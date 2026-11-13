title: Panel VAR: Six Estimators for Heterogeneous Countries
short: Panel VAR
when: A VAR for many countries, regions or firms with short time series
data: Garrett (1998) OECD panel: GDP growth and unemployment in 14 countries, 1966–1990
tools: Python · OES engine · statsmodels
oes: pvar
lead: Use this guide when the same VAR should hold for many units but each unit has only 25 or 30 observations, too few to estimate a VAR on its own. A panel VAR shares information across units. The guide helps you choose among six estimators (mean group, pooled, Zellner-Hong, hierarchical, static and dynamic factors), and covers the pre-tests, estimation, spillovers between countries, convergence, robustness and the write-up.
---

## The question and the model

**How do output growth and unemployment move together across rich countries, and are the dynamics the same everywhere?** The data are 14 OECD countries observed for 25 years, 336 usable observations. Each country has the VAR

\[
y_{it} = c_i + A_{i1} y_{i,t-1} + \dots + A_{ip} y_{i,t-p} + u_{it}, \qquad y_{it} = (\text{GDP growth}_{it}, \text{unemployment}_{it})'.
\]

With one lag each country has 6 coefficients and 25 observations per equation: estimable, but noisy. Pool all countries and the noise averages out, at the price of assuming that every country has the same dynamics. A panel VAR is a choice on the line between these two extremes:

| Estimator | Idea | Use it when |
|---|---|---|
| **OLS mean group** (Pesaran-Smith) | Estimate each country separately and average | \(T\) is not too small, units differ, and you want the average effect with honest standard errors |
| **Bayesian pooled** | One set of coefficients for all units, Minnesota prior | Units are similar and you want the most precise estimate |
| **Zellner-Hong** random effects | Unit coefficients are drawn around a common mean, with a fixed amount of shrinkage \(\lambda_1\) | You want partial pooling and can set the shrinkage yourself |
| **Hierarchical** (Jarociński 2010) | Same, but the amount of shrinkage is estimated | You want the data to decide how alike the units are |
| **Static factor** (Canova-Ciccarelli) | Coefficients are sums of common, unit, variable and lag factors; all units' lags enter every equation | Countries influence each other and you want spillovers |
| **Dynamic factor** | Same, with factors that change over time | The relationships are unstable |

<div class="read" markdown="1">
**How to choose**

Start with mean group and pooled: if they give similar answers, units are similar and the choice does not matter. If they differ, use the hierarchical model to let the data decide how much to pool (its shrinkage parameter is a measure of heterogeneity). Use the factor estimators when cross-country spillovers are the question, or the pre-tests show strong dependence between countries (Step 2). Report at least two estimators.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We estimate a panel VAR in GDP growth and unemployment for 14 OECD countries, 1966–1990, with one lag. To allow for heterogeneity across countries we use the mean-group estimator of Pesaran and Smith (1995), which averages country-specific VARs and computes standard errors from the dispersion of the coefficients across countries, and compare it with a Bayesian pooled estimator with a Minnesota prior, the Zellner-Hong (1989) random-effects estimator and the hierarchical prior of Jarociński (2010). Spillovers between countries are examined with the static and dynamic factor estimators of Canova and Ciccarelli (2009).
</div>

## Load the data and pre-test

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

data = open("garrett.csv").read()
df = pd.read_csv("garrett.csv")
print(df.groupby("country").size().describe().round(1)[["count", "min", "max"]])
print(df[["gdp_growth", "unem"]].describe().round(2).loc[["mean", "std", "min", "max"]])

spec = dict(vars=["gdp_growth", "unem"], unit="country", time="year", lags=1, estimator="mg")
r = oes.run("pvar", data, spec)
oes.show(r, "Panel unit-root")
```

<div class="read" markdown="1">
**How to read it**

Two tests, one per variable. The **Fisher ADF** test combines the unit-root tests of the 14 countries: GDP growth rejects a unit root (p < 0.001), but **unemployment does not** (p = 0.76): it is persistent, and a VAR in levels is an approximation. The **Pesaran CD** test rejects cross-sectional independence for both variables (mean absolute correlation 0.45 for growth and 0.72 for unemployment): countries share common shocks, such as the oil crises. This matters in two ways: standard errors that assume independent countries are too small, and the factor estimators of Step 7 are designed for this.

What to do about persistent unemployment: the VAR in levels with a persistent variable is common in the literature and consistent, but you should say so and check robustness with the variable in first differences, using a time-series transformation of your own data before loading it.
</div>

## Choose the number of lags

```python
oes.show(r, "Lag order")
```

<div class="read" markdown="1">
**How to read it**

BIC of the pooled VAR is lowest at two lags (0.739), against 0.849 for one lag and 0.750 for three. With 25 observations per country, one lag already costs six parameters per unit, so the guide uses one lag for the main results and two as a robustness check in Step 10. Report the BIC table, whichever lag length you choose.
</div>

## Mean-group estimates

```python
oes.summary(r)
oes.show(r, "Mean-group equation")
```

<div class="read" markdown="1">
**How to read it**

Each row is the average over the 14 country VARs, with the Pesaran-Smith standard error (the standard deviation across countries divided by \(\sqrt{14}\)); the last two columns show the smallest and largest country coefficient. **Unemployment equation:** unemployment is persistent (0.81, every country between 0.69 and 0.94) and falls when growth is high: a one-point increase in growth lowers unemployment next year by 0.19 percentage point (z = −8.3), negative in every one of the 14 countries (range −0.36 to −0.04). **Growth equation:** growth has modest persistence (0.25) and is not affected by lagged unemployment (−0.01, z = −0.1), but the country values range from −0.62 to +0.73, so the average hides large differences between countries.
</div>

## Compare the estimators

```python
def key_coefficients(res):
    """L1.unem and L1.gdp_growth in the unemployment equation; L1.unem in the growth equation."""
    out = {}
    for t in res["tables"]:
        title = t["title"]
        if title.endswith(": unem") and "equation" in title.lower():
            f = oes.frame(t); out["unem <- own lag"] = f.loc["L1.unem"].iloc[0]; out["unem <- growth"] = f.loc["L1.gdp_growth"].iloc[0]
        if title.endswith(": gdp_growth") and "equation" in title.lower():
            out["growth <- unem"] = oes.frame(t).loc["L1.unem"].iloc[0]
    return out

rows = {}
for est in ["mg", "pooled", "zh", "hierarchical"]:
    rows[est] = key_coefficients(oes.run("pvar", data, dict(spec, estimator=est, draws=2000, burn=1000)))
print(pd.DataFrame(rows).T.round(3))
```

<div class="read" markdown="1">
**How to read it**

Four estimators, three coefficients. All agree on the sign and the main picture: unemployment is persistent (0.81 to 0.89) and falls when growth rises (\(-0.13\) to \(-0.19\)), while lagged unemployment has little or a slightly negative effect on growth. They differ in size. The mean-group effect of growth on unemployment (\(-0.19\)) is about 50% larger than the pooled one (\(-0.125\)), and pooled persistence (0.88) is higher than mean-group (0.81). This is the pattern expected when slopes differ across units and a pooled estimator imposes a common value: dynamic coefficients are biased towards more persistence (Pesaran and Smith 1995). The hierarchical estimate sits between the two (\(-0.15\) and 0.84), which is what partial pooling should do. For growth on lagged unemployment, mean group finds nothing (\(-0.01\), standard error 0.10), while the pooled and hierarchical estimates are small and negative (\(-0.07\) and \(-0.09\)): this relationship is weak and not robust to the estimator, and should not be a headline result.

In a paper: report the mean-group estimate as the main result if the aim is the average effect and the units differ, show the others as robustness, and describe how they differ.
</div>

## Impulse responses and variance decomposition

```python
oes.plot(r, "Response of unem to a shock in gdp_growth", ncols=1, size=(6, 3))
oes.plot(r, "Response of gdp_growth to a shock in unem", ncols=1, size=(6, 3))
oes.show(r, "Forecast-error variance")
```

<div class="read" markdown="1">
**How to read it**

The response is to a one-standard-deviation growth shock (2.3 percentage points), with growth ordered first: unemployment may respond within the year, growth responds to unemployment only with a lag. The median country sees unemployment fall by 0.5 point on impact and by 0.9 point after one year (range across countries \(-1.1\) to \(-0.4\)), 0.5 after four years and 0.3 after eight; it has not quite returned to baseline after twelve (\(-0.17\)). Growth itself fades in two years (2.3, 0.5, 0.2). The response of growth to an unemployment shock is zero on average (\(-0.008\)), with country values from \(-0.09\) to +0.17.

The bands in this model are the 16th to 84th percentiles **across countries**, not estimation uncertainty: they measure heterogeneity. Say so under the figure.

Variance decomposition: growth shocks explain 43% of the forecast-error variance of unemployment at one year, 76% at four and 79% at twelve, while unemployment shocks explain under 1% of the variance of growth: in this panel, growth drives unemployment and not the reverse.
</div>

## Spillovers between countries: the factor estimators

```python
rs = oes.run("pvar", data, dict(spec, estimator="dynamic", draws=3000, burn=2000))
oes.show(rs, "MCMC")
oes.plot(rs, "Response of unem to a gdp_growth shock", ncols=1, size=(6, 3))
```

<div class="read" markdown="1">
**How to read it**

The factor estimators let every country's equation include the lags of all countries and report the effect of a growth shock in one country on **the same country** and on **the other countries** (average per country). Convergence first: with 3000 draws after 2000 burn-in the effective sample size of the own-lag factor is 964 and all Geweke statistics are below 2 in absolute value (with the default 1000 draws and 500 burn-in it was 181 and the statistic was \(-4.6\): do not use the defaults for the dynamic model).

The responses: a shock to growth in one country lowers its own unemployment by 0.13 to 0.17 over the next two years, and moves other countries' growth by \(-0.04\) (3% of the domestic growth response of 1.2 after a year) and their unemployment by +0.04 (about a third of the domestic fall of 0.13, with the opposite sign). Spillovers are small and their sign is not stable across estimators: the static version gives \(-0.01\) to \(-0.02\) for the unemployment spillover. Two cautions. First, these estimators assume uncorrelated shocks, so there is no contemporaneous effect by construction, although the CD test in Step 2 shows strong correlation between countries. Second, the own-country response of unemployment is much smaller than in the mean-group model because the factor prior shrinks coefficients strongly in a short panel and there is no impact effect. Use these estimators to compare the size of domestic and foreign effects, not for the size of the domestic effect.
</div>

## Hierarchical estimation: how alike are the countries?

```python
rh = oes.run("pvar", data, dict(spec, estimator="hierarchical", draws=3000, burn=1500))
oes.show(rh, "Overall shrinkage")
oes.show(rh, "MCMC")
oes.show(rh, "Own first-lag")
```

<div class="read" markdown="1">
**How to read it**

The shrinkage parameter \(\lambda_1\) measures how different the countries are: it is 0.016 (95% interval 0.006 to 0.034), small, so the hierarchical model pools the countries strongly. The country-specific own-lag coefficients are accordingly close together: unemployment persistence from 0.73 (Japan) to 0.93 (United Kingdom), growth persistence from 0.07 (United Kingdom, Belgium) to 0.34 (Japan), a much narrower range than the unit-by-unit OLS estimates behind the mean-group table (growth: \(-0.14\) to 0.56). The reading is that most of the apparent difference between countries in Step 4 is noise from 25 observations, not real heterogeneity. That supports reporting the mean-group and pooled estimates together.

Convergence: with 3000 draws after 1500 burn-in the effective sample sizes are 232 or more and every Geweke statistic is below 2 (the largest is 1.88, for \(\lambda_1\)). With the default 1000 draws they were 85 to 124 for most parameters, too few: use at least 3000 for this estimator.
</div>

## Diagnostics

```python
oes.show(r, "Residual diagnostics")
```

<div class="read" markdown="1">
**How to read it**

Growth residuals have remaining autocorrelation (Ljung-Box p = 0.003) while unemployment residuals have none (p = 0.42), and the joint portmanteau test rejects (p = 0.003; with 336 observations and 48 degrees of freedom it is only approximate). Normality is rejected for both (Jarque-Bera 46 and 101): the residuals have heavy tails, as expected with recessions in the sample. There is no sign of ARCH at 5% (p = 0.54 and 0.07). Serial correlation in the growth equation suggests a second lag; Step 10 shows that the main result does not depend on it. Dependence between countries is not in this table: it is tested in Step 2 and motivates the factor estimators.
</div>

## Robustness: lags, estimators and dropping one country at a time

```python
for p in (1, 2, 3):
    k = key_coefficients(oes.run("pvar", data, dict(spec, lags=p)))
    print(f"mean group, {p} lag(s): unem <- growth =", round(float(k["unem <- growth"]), 3))

est = []
for c in sorted(df["country"].unique()):
    sub = df[df["country"] != c].to_csv(index=False)
    est.append((c, key_coefficients(oes.run("pvar", sub, spec))["unem <- growth"]))
est = pd.DataFrame(est, columns=["dropped country", "unem <- growth"]).set_index("dropped country")
print(est.round(3).T)
print("range:", est.min().round(3).iloc[0], "to", est.max().round(3).iloc[0])
```

<div class="read" markdown="1">
**How to read it**

Three checks of the effect of growth on unemployment (mean group, \(-0.19\)). **Lags:** with one, two and three lags the coefficient is \(-0.19\), \(-0.18\) and \(-0.20\). **Leave one country out:** re-estimating without each country in turn gives values between \(-0.20\) (dropping Japan) and \(-0.18\) (dropping the United Kingdom): no single country drives the result. **Estimators:** Step 5 shows \(-0.13\) to \(-0.19\). The conclusion that growth lowers unemployment next year is robust; the size depends on the estimator, by about a third.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

Table 1 reports the mean-group estimates. Unemployment is persistent (0.81) and responds negatively to lagged growth: a one-percentage-point increase in growth lowers unemployment by 0.19 percentage point in the following year (Pesaran-Smith standard error 0.023), and the coefficient is negative in all 14 countries. Lagged unemployment has no effect on growth on average (\(-0.01\), standard error 0.10) but the country estimates range widely. A one-standard-deviation growth shock lowers unemployment by 0.9 percentage point after one year (the 16th to 84th percentile across countries is \(-1.1\) to \(-0.4\)); the effect halves after about four years. Growth shocks explain 76% of the forecast-error variance of unemployment at a four-year horizon. The pooled, Zellner-Hong and hierarchical estimators give the same signs and a smaller effect (\(-0.13\) to \(-0.15\)), consistent with the upward bias in the persistence of pooled dynamic estimators when slopes differ. The hierarchical model estimates a small degree of heterogeneity (\(\lambda_1 = 0.016\)). Spillovers estimated with the Canova-Ciccarelli factor model are small: a growth shock in one country moves other countries' growth by 3% of its domestic effect and their unemployment by about a third of the domestic effect, with the opposite sign, and the sign is not stable across the static and dynamic versions. Results are unchanged with two or three lags and when any one country is dropped (\(-0.20\) to \(-0.18\)). Unemployment fails the panel unit-root test, so we also note that the estimates relate to a persistent variable in levels.
</div>

| Report | What | Where |
|---|---|---|
| Data | Countries, years, balance, unit-root and cross-sectional dependence tests | Step 2 |
| Estimators | Which, and why; prior settings for the Bayesian ones | Step 1 |
| Lags | BIC and the choice | Step 3 |
| Results | Average coefficients with standard errors, heterogeneity (range across units), impulse responses | Steps 4 to 6 |
| Spillovers | Factor-model responses, convergence statistics | Step 7 |
| Diagnostics | Residual tests, effective sample sizes for samplers | Steps 7 to 9 |
| Robustness | Other estimators, lags, leave-one-country-out | Steps 5 and 10 |

**References.** Pesaran and Smith (1995), *Journal of Econometrics*; Canova and Ciccarelli (2009), *International Economic Review*; Jarociński (2010), *Journal of Applied Econometrics*; Zellner and Hong (1989), *Journal of Econometrics*; Pesaran (2004), CESifo Working Paper 1229; Garrett (1998), *Partisan Politics in the Global Economy*; Canova and Ciccarelli (2013), ECB Working Paper 1507.
