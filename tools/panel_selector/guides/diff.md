title: Panel Model in First Differences
short: Differences
when: All variables I(1), not cointegrated
data: 18 OECD countries, annual, 1960–1978 (Baltagi and Griffin 1983)
lead: Use this guide when every variable in a long panel has a unit root and the panel cointegration tests find no long-run relationship. A regression in levels is then spurious. The guide shows the spurious regression, runs the cointegration tests, and estimates the short-run relationship in first differences with estimators that allow for common shocks and heterogeneous slopes.
---

## The question and the model

The question: **how does gasoline demand respond to income and to the price of gasoline?** The dataset is the one Baltagi and Griffin used to compare panel estimators of gasoline demand across OECD countries, before and during the oil shocks of the 1970s.

\[
\Delta \ln G_{it} = \beta_1 \Delta \ln Y_{it} + \beta_2 \Delta \ln P_{it} + \beta_3 \Delta \ln C_{it} + \varepsilon_{it}
\]

| Symbol | Variable in the file | Definition | Expected sign |
|---|---|---|---|
| \(\ln G_{it}\) | `lgaspcar` | Log gasoline consumption per car | dependent variable |
| \(\ln Y_{it}\) | `lincomep` | Log real income per capita | positive: income elasticity |
| \(\ln P_{it}\) | `lrpmg` | Log real price of gasoline | negative: price elasticity |
| \(\ln C_{it}\) | `lcarpcap` | Log cars per capita | negative: more cars, each driven less |

In first differences, the coefficients are **short-run elasticities**: how much gasoline use per car changes in the year that income or the price changes.

<div class="read" markdown="1">
**Why differences, and what you give up**

If the variables are I(1) and cointegrated, a long-run relationship in levels exists and an error-correction model (the PMG or CS-ARDL guides) estimates it. If they are I(1) and **not** cointegrated, there is no long-run relationship to estimate: a regression in levels would be spurious, and the valid model is in differences. You give up the long-run elasticities, which is why the cointegration tests deserve care.
</div>

<div class="write" markdown="1">
**How to write it: model section**

We relate the growth of gasoline consumption per car to the growth of real income per capita, the real price of gasoline and the stock of cars per capita, \(\Delta \ln G_{it} = \beta_1 \Delta \ln Y_{it} + \beta_2 \Delta \ln P_{it} + \beta_3 \Delta \ln C_{it} + \varepsilon_{it}\). The coefficients are short-run elasticities.
</div>

## Load the panel and run the pre-tests

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from panel_tests import cd_test, cips_test, slope_test, kao_test, fisher_eg_test
from panel_estimators import mean_group, cce_mg

df = pd.read_csv("gasoline.csv").sort_values(["country", "year"])
y, xs = "lgaspcar", ["lincomep", "lrpmg", "lcarpcap"]
print(f"{df['country'].nunique()} countries, {df['year'].min()}-{df['year'].max()}, {len(df)} observations\n")

print(f"{'':10s}{'CD':>7s}{'CIPS level':>12s}{'p':>8s}{'CIPS diff':>11s}{'p':>8s}")
for v in [y] + xs:
    W = df.pivot(index="year", columns="country", values=v)
    cd = cd_test(W)[0]; c0, c1 = cips_test(W), cips_test(W.diff())
    print(f"{v:10s}{cd:7.1f}{c0['stat']:12.2f}{c0['p']:8.3f}{c1['stat']:11.2f}{c1['p']:8.3f}")

s = slope_test(df, "country", y, xs)
print(f"\nPesaran-Yamagata adjusted delta = {s['delta_adj']:.2f}   p-value = {s['delta_adj_p']:.4f}")
```

<div class="read" markdown="1">
**How to read it**

- **Cross-sectional dependence** is strong for every variable (the oil shocks hit every country), so CIPS is the unit-root test to use.
- **Unit roots.** CIPS does not reject a unit root in levels for any variable, and rejects it in first differences for all of them: **all four are I(1)**.
- **Slopes** are heterogeneous: the Δ test rejects. This will matter for the choice of cointegration test and of estimator.
</div>

## The spurious regression in levels

```python
x = df.set_index(["country", "year"])
lev = PanelOLS(x[y], sm.add_constant(x[xs]), entity_effects=True).fit(cov_type="clustered", cluster_entity=True)
print(lev.params[xs].round(3).to_frame("coef").join(lev.tstats[xs].round(1).rename("t")))
print(f"R-squared (within): {lev.rsquared_within:.3f}")
```

<div class="read" markdown="1">
**How to read it**

In levels, fixed effects gives large, highly significant elasticities and a high within R-squared. **This is exactly what a spurious regression looks like.** With I(1) variables, the t-statistics do not follow the t distribution, and trending series appear related whether or not they are. These numbers are only meaningful if the variables are cointegrated, which is the next test.
</div>

## Test for cointegration

```python
k = kao_test(df, "country", y, xs)
f = fisher_eg_test(df, "country", y, xs)
print(f"Kao ADF test (common slopes):              t* = {k['stat']:6.2f}   p-value = {k['p']:.3f}")
print(f"Fisher-Engle-Granger (country slopes):     Z  = {f['Z']:6.2f}   p-value = {f['Z_p']:.3f}")
print(f"                                           P  = {f['P']:6.2f}   p-value = {f['P_p']:.3f}   (chi2 with {2 * f['n']} df)")
```

<div class="read" markdown="1">
**How to read it**

The null hypothesis of both tests is **no cointegration**.

- **Kao** pools the fixed-effects residuals and assumes the same slopes in every country. It rejects narrowly, at the 5% level.
- **Fisher–Engle–Granger** runs an Engle–Granger test for each country, with its own slopes, and combines the p-values. Its Z version does not reject (p = 0.25); its P version comes close only at the 10% level.

The tests disagree, and the pre-tests tell us which to believe: **the slopes are heterogeneous**, which violates Kao's assumption. A Kao test applied to heterogeneous panels can reject spuriously, because the pooled residuals mix different long-run relationships. The model selection tool therefore uses the heterogeneous test here, and concludes that there is no cointegration.

This is a close call, and the paper must say so: report both tests, and add the Westerlund (2007) error-correction tests with bootstrap critical values, which allow for both heterogeneity and cross-sectional dependence (`xtwest` in Stata). With T = 19, every panel cointegration test has limited power.
</div>

## Estimate the model in first differences

```python
d = df.copy()
for v in [y] + xs:
    d["d_" + v] = d.groupby("country")[v].diff()
d = d.dropna()
dy, dxs = "d_" + y, ["d_" + v for v in xs]
xd = d.set_index(["country", "year"])

fd_fe = PanelOLS(xd[dy], sm.add_constant(xd[dxs])).fit(cov_type="kernel", kernel="bartlett", bandwidth=2)
fd_tw = PanelOLS(xd[dy], sm.add_constant(xd[dxs]), time_effects=True).fit(cov_type="kernel", kernel="bartlett", bandwidth=2)
mg = mean_group(d, "country", "year", dy, dxs)
cce = cce_mg(d, "country", "year", dy, dxs)

res = {"Pooled, DK s.e.": (fd_fe.params[dxs], fd_fe.std_errors[dxs]), "Pooled + year FE": (fd_tw.params[dxs], fd_tw.std_errors[dxs]),
       "Mean group": (mg["table"]["coef"], mg["table"]["std err"]), "CCE mean group": (cce["table"]["coef"], cce["table"]["std err"])}
print(f"{'':20s}" + "".join(f"{v:>18s}" for v in dxs))
for name, (b, se) in res.items():
    print(f"{name:20s}" + "".join(f"{b[v]:9.3f} ({se[v]:.3f})" for v in dxs))
```

<div class="read" markdown="1">
**How to read it**

Differencing removes the country fixed effects (permanent differences in levels), so the pooled regression on differences needs no country dummies. Compare the elasticities with the spurious levels regression: **the income and price responses are much smaller in differences**, as one would expect when drivers can adjust only gradually. (The cars coefficient is similar, because the stock of cars and fuel use per car are linked almost mechanically in every year.)

- **Price.** A 10% rise in the real price of gasoline reduces consumption per car by roughly 1–3% within the year, depending on the estimator.
- **Income.** The short-run income elasticity is positive.
- **Cars per capita.** More cars means each car is driven less, so consumption per car falls.

With heterogeneous slopes and cross-sectional dependence, **CCE-MG is the preferred estimator**: it averages the country-specific elasticities and removes the common factors (the oil shocks) through the cross-sectional averages. Its standard errors are larger, because they reflect the cross-country dispersion of the elasticities.
</div>

## Diagnostic test: residual cross-sectional dependence

```python
fe_res = fd_fe.resids.unstack("country")
print(f"Pooled differences, residual CD:  {cd_test(fe_res)[0]:6.2f}   p-value = {cd_test(fe_res)[1]:.4f}")

bars = d.groupby("year")[[dy] + dxs].transform("mean").add_suffix("_bar")
dd = d.join(bars)
R = {}
for c, g in dd.groupby("country"):
    X = np.column_stack([np.ones(len(g)), g[dxs + list(bars.columns)]])
    b, *_ = np.linalg.lstsq(X, g[dy].to_numpy(), rcond=None)
    R[c] = pd.Series(g[dy].to_numpy() - X @ b, index=g["year"])
cd, p, r = cd_test(pd.DataFrame(R))
print(f"CCE-MG residual CD:               {cd:6.2f}   p-value = {p:.4f}")
```

<div class="read" markdown="1">
**How to read it**

The pooled regression in differences leaves strong common shocks in the residuals. In the CCE-MG residuals the CD statistic is close to zero and no longer significant: the cross-sectional averages absorb the common oil-price and business-cycle shocks.
</div>

## Build the results table for the paper

```python
from scipy import stats
def star(b, se):
    p = 2 * stats.norm.sf(abs(b / se))
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

print(f"{'':12s}" + "".join(f"{c:>18s}" for c in ["(1) Levels FE", "(2) Pooled Δ", "(3) MG Δ", "(4) CCE-MG Δ"]))
cols = [(lev.params[xs].values, lev.std_errors[xs].values)] + [(res[k][0].values, res[k][1].values) for k in ("Pooled, DK s.e.", "Mean group", "CCE mean group")]
for i, v in enumerate(xs):
    print(f"{v:12s}" + "".join(f"{b[i]:15.3f}{star(b[i], se[i]):3s}" for b, se in cols))
    print(f"{'':12s}" + "".join(f"{'(' + format(se[i], '.3f') + ')':>15s}   " for b, se in cols))
print("Column (1) is shown for comparison only: the variables are I(1) and not cointegrated, so it is spurious.")
print(f"Kao t* = {k['stat']:.2f} (p = {k['p']:.3f}); Fisher-EG Z = {f['Z']:.2f} (p = {f['Z_p']:.3f}); Pesaran-Yamagata adj. delta = {s['delta_adj']:.2f} (p < 0.001)")
```

<div class="read" markdown="1">
**How to read it**

Including the levels regression in column (1) shows the reader the spurious result the paper avoids. Column (4) is the preferred estimate.
</div>

## Write the paper

<div class="write" markdown="1">
**How to write it: pre-tests**

CIPS tests, which allow for the strong cross-sectional dependence indicated by the CD test, do not reject a unit root in levels for any variable but reject it in first differences. Because the Pesaran–Yamagata test rejects slope homogeneity, we test for cointegration with the Fisher-type combination of country-level Engle–Granger tests, which allows the long-run coefficients to differ across countries; it does not reject the null of no cointegration. The Kao test, which assumes homogeneous slopes, rejects at the 5 percent level. Given the heterogeneity of slopes, we treat the evidence for cointegration as weak and estimate the model in first differences.
</div>

<div class="write" markdown="1">
**How to write it: results**

Table 4 reports short-run elasticities. In the common correlated effects mean group model (column 4), gasoline consumption per car rises with income and falls with the real price and with the number of cars per capita. The short-run income and price elasticities are much smaller than the coefficients of the levels regression in column 1, which, given the absence of cointegration, should not be interpreted.
</div>

<div class="warn" markdown="1">
**Common mistakes**

- Running a levels regression on I(1) variables without testing for cointegration, and interpreting the t-statistics.
- Choosing the cointegration test that gives the desired answer. Choose it from the pre-tests (homogeneous or heterogeneous slopes, dependence or not), and report the others.
- Calling short-run elasticities from a model in differences “long-run” effects.
- Ignoring cross-sectional dependence in the differenced model. Common shocks remain after differencing.
</div>
