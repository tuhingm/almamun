title: Fixed Effects with Driscoll–Kraay Standard Errors
short: FE-DK
when: Long panel, stationary variables, homogeneous slopes
data: 14 OECD economies, annual, 1967–1990 (Garrett 1998)
lead: Use this guide when the panel has a long time dimension, every variable is stationary, and the slope-homogeneity test does not reject. A pooled fixed-effects regression is then valid, but countries hit by the same shocks are not independent, so the guide shows how Driscoll–Kraay standard errors fix the inference, and how to check whether the common shocks also bias the coefficients.
---

## The question and the model

The question: **in the OECD economies of 1967–1990, how was output growth related to unemployment and to changes in trade openness?** Macro panels like this one, a few dozen countries over a few decades, are typical of empirical papers in international and environmental economics. Their defining feature is that the countries share the same shocks: the oil crises of 1973 and 1979 and the recessions that followed hit all of them at once.

\[
g_{it} = \alpha_i + \beta_1\, u_{it} + \beta_2\, \Delta open_{it} + \varepsilon_{it}
\]

| Symbol | Variable in the file | Definition | Expected sign |
|---|---|---|---|
| \(g_{it}\) | `gdp_growth` | Real GDP growth, % | dependent variable |
| \(u_{it}\) | `unem` | Unemployment rate, % | negative: slack and low growth go together (Okun's law) |
| \(\Delta open_{it}\) | `d_trade` | Change in trade (exports + imports) as % of GDP | positive if openness raises growth |
| \(\alpha_i\) | — | Country fixed effect | permanent differences in growth |

<div class="read" markdown="1">
**Before you estimate anything**

In a macro panel, three properties decide the estimator, and each needs a test: **cross-sectional dependence** (do the countries share shocks?), **stationarity** (do the variables have unit roots?) and **slope homogeneity** (is \(\beta\) the same in every country?). The next three steps run them. Fixed effects with Driscoll–Kraay errors is the answer when the variables are stationary and the slopes are homogeneous.
</div>

<div class="write" markdown="1">
**How to write it: model section**

We estimate \(g_{it} = \alpha_i + \beta_1 u_{it} + \beta_2 \Delta open_{it} + \varepsilon_{it}\) for 14 OECD countries over 1967–1990, where \(g_{it}\) is real GDP growth, \(u_{it}\) the unemployment rate and \(\Delta open_{it}\) the change in trade openness, and \(\alpha_i\) are country fixed effects.
</div>

## Load the panel and test for cross-sectional dependence

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from panel_tests import cd_test, cips_test, fisher_adf_test, slope_test   # examples/panel_tests.py

raw = pd.read_csv("garrett.csv").sort_values(["country", "year"])
raw["d_trade"] = raw.groupby("country")["trade"].diff()
long = raw.dropna(subset=["d_trade"])
V = ["gdp_growth", "unem", "d_trade"]
print(f"{long['country'].nunique()} countries, {long['year'].min()}-{long['year'].max()}, {len(long)} observations")

print(f"\n{'':12s}{'CD':>8s}{'p-value':>10s}{'mean |rho|':>12s}")
for v in V:
    cd, p, r = cd_test(long.pivot(index="year", columns="country", values=v))
    print(f"{v:12s}{cd:8.2f}{p:10.4f}{r:12.3f}")
```

<div class="read" markdown="1">
**How to read it**

A balanced panel of 14 countries over 24 years. The Pesaran CD test rejects cross-sectional independence for every variable, decisively. The average absolute correlation between two countries' growth rates is about 0.45: **the OECD economies move together**. Two consequences follow. Unit-root tests must allow for this (CIPS rather than Fisher-ADF), and standard errors that treat countries as independent will be too small.
</div>

## Unit-root tests and slope homogeneity

```python
print(f"{'':12s}{'CIPS':>8s}{'p-value':>10s}{'Fisher Z':>10s}{'p-value':>10s}")
for v in V:
    W = long.pivot(index="year", columns="country", values=v)
    c, f = cips_test(W), fisher_adf_test(W)
    print(f"{v:12s}{c['stat']:8.2f}{c['p']:10.4f}{f['Z']:10.2f}{f['Z_p']:10.4f}")

s = slope_test(long, "country", "gdp_growth", ["unem", "d_trade"])
print(f"\nPesaran-Yamagata: delta = {s['delta']:.2f} (p = {s['delta_p']:.3f})   adjusted delta = {s['delta_adj']:.2f} (p = {s['delta_adj_p']:.3f})")
```

<div class="read" markdown="1">
**How to read it**

- **Stationarity.** Because the variables are cross-sectionally dependent, CIPS is the test to use. It rejects a unit root in levels for all three: **every variable is I(0)**. Notice that for unemployment the first-generation Fisher test does *not* reject. It ignores the common shocks, and the two tests disagree; CIPS is the valid one here.
- **Slope homogeneity.** The adjusted Δ test does not reject: there is no evidence that the effects of unemployment and openness differ across countries. **Pooling the slopes is defensible.**

These are the conditions under which the model selection tool recommends fixed effects with Driscoll–Kraay standard errors.
</div>

## Estimate fixed effects with three kinds of standard errors

```python
x = long.set_index(["country", "year"])
xs = ["unem", "d_trade"]
y, X = x["gdp_growth"], sm.add_constant(x[xs])

fits = {"Conventional": PanelOLS(y, X, entity_effects=True).fit(),
        "Clustered by country": PanelOLS(y, X, entity_effects=True).fit(cov_type="clustered", cluster_entity=True),
        "Driscoll-Kraay": PanelOLS(y, X, entity_effects=True).fit(cov_type="kernel", kernel="bartlett", bandwidth=3)}
dk = fits["Driscoll-Kraay"]
print(f"{'':10s}{'coef':>8s}" + "".join(f"{k:>22s}" for k in fits))
for v in xs:
    print(f"{v:10s}{dk.params[v]:8.3f}" + "".join(f"{m.std_errors[v]:13.3f} (p {m.pvalues[v]:.3f})" for m in fits.values()))
```

<div class="read" markdown="1">
**How to read it**

The coefficients are the same in every column; only the standard errors differ.

- **Unemployment.** A one-point higher unemployment rate goes with growth about a third of a point lower. Significant under every choice of standard errors.
- **Trade openness.** With conventional or clustered standard errors, the negative coefficient looks significant. **With Driscoll–Kraay standard errors it is not.** Clustering by country allows for correlation over time within a country, but still treats countries as independent of each other. Driscoll–Kraay standard errors also allow for correlation **across countries in the same year**, which the CD test showed is strong. Openness moves together across countries (world trade booms and slumps), so its standard error is the one most affected.

The Driscoll–Kraay estimator needs a reasonably long T, since it averages over years rather than countries; with 24 years it is appropriate. The bandwidth (lags of the Newey–West kernel) is set to 3, roughly \(T^{1/3}\).
</div>

## Diagnostic tests on the residuals

```python
e = dk.resids.unstack("country")
cd, p, r = cd_test(e)
print(f"CD test on the FE residuals: CD = {cd:.2f}   p-value = {p:.4f}   mean |rho| = {r:.3f}")

r1 = (e * e.shift(1)).sum().sum() / (e ** 2).sum().sum()
print(f"First-order autocorrelation of the residuals (pooled): {r1:.3f}")
```

<div class="read" markdown="1">
**How to read it**

- **Residual dependence.** Even after the two regressors, the residuals of different countries remain correlated: the common shocks are not explained by the model. This is exactly the case Driscoll–Kraay standard errors are built for.
- **Serial correlation.** The residuals are positively autocorrelated, as is usual for annual macro data. The Bartlett kernel in the Driscoll–Kraay estimator corrects for it.

But there is a deeper question. Driscoll–Kraay fixes the **standard errors**. If the common shocks are also correlated with the regressors, the **coefficients** themselves are biased, and no standard-error correction helps. The robustness checks test this.
</div>

## Robustness: are the coefficients driven by common shocks?

```python
twfe = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(cov_type="kernel", kernel="bartlett", bandwidth=3)

# CCE pooled (Pesaran 2006): add cross-sectional averages, with country-specific coefficients
bars = x.groupby(level="year")[["gdp_growth"] + xs].transform("mean").add_suffix("_bar")
cty = pd.get_dummies(x.index.get_level_values("country"), dtype=float).set_index(x.index)
inter = pd.concat({f"{b}_{c}": bars[b] * cty[c] for b in bars for c in cty}, axis=1)
ccep = PanelOLS(y, sm.add_constant(x[xs]).join(inter), entity_effects=True, drop_absorbed=True).fit(cov_type="kernel", kernel="bartlett", bandwidth=3)

bw5 = PanelOLS(y, X, entity_effects=True).fit(cov_type="kernel", kernel="bartlett", bandwidth=5)
print(f"{'':30s}{'unem':>18s}{'d_trade':>18s}")
for name, m in (("FE, Driscoll-Kraay (baseline)", dk), ("FE, DK, bandwidth 5", bw5), ("Two-way FE (year effects), DK", twfe), ("CCE pooled, DK", ccep)):
    print(f"{name:30s}" + "".join(f"{m.params[v]:9.3f} ({m.std_errors[v]:.3f})" for v in xs))
```

<div class="read" markdown="1">
**How to read it**

This is the most important table in a macro-panel paper.

- **Bandwidth.** A longer kernel changes the standard errors only a little.
- **Year effects.** Adding a dummy for each year removes everything common to all countries in that year. The unemployment coefficient shrinks towards zero and the openness coefficient **turns positive and significant**.
- **CCE pooled** removes common factors through cross-sectional averages, allowing each country to respond to them differently. It tells the same story as the year effects.

The interpretation: in the baseline, both coefficients mostly reflected the **common OECD cycle**. In the 1970s oil shocks, growth fell everywhere while unemployment rose and the value of trade (inflated by oil prices) rose. Once the common shocks are removed, countries whose openness rose *faster than the OECD average* grew faster. **The baseline coefficient was biased by the common shocks, not just its standard error.**

So for this question the robustness check changes the conclusion: report the model with year effects or CCE as the main specification, and the plain FE-DK as the comparison. The model selection tool flagged the residual cross-sectional dependence; this step is how you act on it.
</div>

## Build the results table for the paper

```python
def star(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

cols = {"(1) FE": dk, "(2) Two-way FE": twfe, "(3) CCE pooled": ccep}
print(f"{'':12s}" + "".join(f"{c:>18s}" for c in cols))
for v in xs:
    print(f"{v:12s}" + "".join(f"{m.params[v]:15.3f}{star(m.pvalues[v]):3s}" for m in cols.values()))
    print(f"{'':12s}" + "".join(f"{'(' + format(m.std_errors[v], '.3f') + ')':>15s}   " for m in cols.values()))
print(f"{'Country FE':12s}" + f"{'yes':>15s}   " * 3)
print(f"{'Year FE':12s}{'no':>15s}   {'yes':>15s}   {'no':>15s}")
print(f"{'CS averages':12s}{'no':>15s}   {'no':>15s}   {'yes':>15s}")
print(f"{'Obs.':12s}" + "".join(f"{m.nobs:>15d}   " for m in cols.values()))
print("Driscoll-Kraay standard errors (Bartlett kernel, 3 lags). *** p<0.01, ** p<0.05, * p<0.10.")
```

<div class="read" markdown="1">
**How to read it**

Report the pre-tests (CD, CIPS, slope homogeneity) in a separate table before this one. In this table, the progression from column (1) to columns (2) and (3) shows the reader how much of the raw correlation is due to common shocks.
</div>

## Write the paper

<div class="write" markdown="1">
**How to write it: pre-tests and estimator**

Pesaran's CD test rejects cross-sectional independence for all variables, so we test for unit roots with the CIPS test, which rejects a unit root in levels for each series. The Pesaran–Yamagata test does not reject slope homogeneity (adjusted Δ, p = 0.16). We therefore estimate a fixed-effects model with Driscoll–Kraay standard errors, which are robust to heteroskedasticity, serial correlation and cross-sectional dependence.
</div>

<div class="write" markdown="1">
**How to write it: results**

In the fixed-effects model, higher unemployment is associated with lower growth, and the change in trade openness has a negative but insignificant coefficient once standard errors allow for cross-sectional dependence (Table 2, column 1). Because the residuals remain cross-sectionally dependent, we also control for common shocks with year effects and with the common correlated effects estimator. In both specifications, increases in trade openness relative to the OECD average are associated with significantly higher growth (columns 2 and 3), indicating that the baseline estimate was confounded by the common business cycle.
</div>

<div class="warn" markdown="1">
**Common mistakes**

- Clustering by country in a macro panel and calling the standard errors robust. They are not robust to common shocks.
- Using Driscoll–Kraay with a short T. The estimator averages over time periods, so it needs many of them.
- Thinking Driscoll–Kraay solves cross-sectional dependence. It corrects the inference only; if the common factor is correlated with the regressors, the coefficients are biased. Always compare with year effects or CCE.
- Using first-generation unit-root tests (LLC, IPS, Fisher) when the CD test rejects.
</div>
