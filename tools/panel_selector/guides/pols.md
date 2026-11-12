title: Pooled OLS
short: Pooled OLS
when: Short panel, no significant unit effects
data: 140 UK manufacturing firms, annual growth rates, 1977–1984 (Arellano–Bond data)
lead: Use this guide when the panel is short and neither the F test nor the Breusch–Pagan LM test finds unit-specific effects. Pooling is then defensible, but the observations on the same unit are still not independent, so the guide shows how to get the inference right and how to show a referee that pooling is safe.
---

## The question and the model

Pooled OLS is rarely the right model for variables in levels: firms, workers and countries almost always differ permanently. It often becomes the right model once the variables are **growth rates**, because differencing removes the permanent differences in levels. The question here: **how does a firm's employment growth respond to the growth of its wages, capital and industry demand?**

\[
\Delta \ln n_{it} = \beta_1 \Delta \ln w_{it} + \beta_2 \Delta \ln k_{it} + \beta_3 \Delta \ln ys_{it} + \lambda_t + \varepsilon_{it}
\]

| Symbol | Variable | Definition | Expected sign |
|---|---|---|---|
| \(\Delta \ln n_{it}\) | `g_emp` | Employment growth, % | dependent variable |
| \(\Delta \ln w_{it}\) | `g_wage` | Real wage growth, % | negative: labour demand slopes down |
| \(\Delta \ln k_{it}\) | `g_capital` | Capital stock growth, % | positive: capital and labour are complements |
| \(\Delta \ln ys_{it}\) | `g_output` | Industry output growth, % | positive: demand for the firm's goods |
| \(\lambda_t\) | — | Year effects: the 1980–81 recession | common to all firms |

<div class="read" markdown="1">
**Before you estimate anything**

Pooling assumes that all units share the same intercept and the same slopes. With growth rates the first assumption is plausible: a firm with permanently high employment does not need to have permanently high employment *growth*. But it is still an assumption, and the F and LM tests below check it. If they reject, use the fixed- or random-effects guide instead.
</div>

<div class="write" markdown="1">
**How to write it: model section**

We relate employment growth to the growth of the real wage, the capital stock and industry output, \(\Delta \ln n_{it} = \beta_1 \Delta \ln w_{it} + \beta_2 \Delta \ln k_{it} + \beta_3 \Delta \ln ys_{it} + \lambda_t + \varepsilon_{it}\), where \(\lambda_t\) are year effects. Specifying the model in growth rates removes permanent firm-specific differences in employment levels.
</div>

## Load the data and build the growth rates

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects

raw = pd.read_csv("employment.csv").sort_values(["firm", "year"])
for v in ["log_emp", "log_wage", "log_capital", "log_output"]:
    raw["g_" + v[4:]] = 100 * raw.groupby("firm")[v].diff()      # growth rate in percent
df = raw.dropna().set_index(["firm", "year"])

xs = ["g_wage", "g_capital", "g_output"]
T_i = df.groupby(level=0).size()
print(f"{len(T_i)} firms, {len(df)} firm-years, years per firm from {T_i.min()} to {T_i.max()} (unbalanced)")
print(df[["g_emp"] + xs].describe().T[["mean", "std", "min", "max"]].round(2))
```

<div class="read" markdown="1">
**How to read it**

The panel is short and unbalanced: 140 firms with 6 to 8 growth observations each. Taking differences loses the first year of every firm. Employment fell on average over the period, the years of the early-1980s recession in British manufacturing.

The model selection tool, run on this file, finds no significant firm effects and recommends pooled OLS. The next steps show why.
</div>

## Test for firm effects: F and Breusch–Pagan LM

```python
years = pd.get_dummies(df.index.get_level_values(1), prefix="y", drop_first=True, dtype=float).set_index(df.index)
y = df["g_emp"]
X = sm.add_constant(df[xs].join(years))

fe = PanelOLS(y, X, entity_effects=True).fit()
f = fe.f_pooled
print(f"F test that all firm effects are zero:  F({f.df}, {f.df_denom}) = {f.stat:.2f}   p-value = {f.pval:.3f}")

pooled = PooledOLS(y, X).fit(cov_type="clustered", cluster_entity=True)
e = pooled.resids
n, Ti = len(e), e.groupby(level=0).size()
lm = n ** 2 / (2 * (Ti * (Ti - 1)).sum()) * ((e.groupby(level=0).sum() ** 2).sum() / (e ** 2).sum() - 1) ** 2
print(f"Breusch-Pagan LM test:                  chi2(1) = {lm:.2f}   p-value = {stats.chi2.sf(lm, 1):.3f}")
```

<div class="read" markdown="1">
**How to read it**

- **F test.** Null: all 140 firms have the same intercept. It is not rejected, so firm-specific intercepts do not improve the fit.
- **Breusch–Pagan LM test.** Null: the variance of a random firm effect is zero. It is not rejected either.

**With neither test rejecting, pooling the firms is defensible.** Fixed effects would estimate 139 extra intercepts for no gain in fit, and lose precision.
</div>

## Estimate the pooled model with clustered standard errors

```python
print(pooled.summary.tables[1])
naive = PooledOLS(y, X).fit()
print("\nStandard errors:      clustered   conventional")
for v in xs:
    print(f"{v:18s}{pooled.std_errors[v]:12.4f}{naive.std_errors[v]:14.4f}")
```

<div class="read" markdown="1">
**How to read it**

- **Wages.** A 1 percentage point faster growth of the real wage lowers employment growth by about 0.4 points in the same year.
- **Capital.** Faster capital growth is associated with faster employment growth: firms that invest also hire.
- **Industry output.** Faster growth of demand in the firm's industry raises employment growth: about half a point for each point of output growth.

**Clustered standard errors are not optional.** Even without firm effects, the errors of the same firm can be correlated across years and their variance can differ across firms. The second table compares them: the clustered standard error of the wage coefficient is more than three times the conventional one. With conventional standard errors you would overstate the precision of every estimate. Report the clustered ones.
</div>

## Diagnostic tests: serial correlation and heteroskedasticity

```python
r = pooled.resids.rename("e").to_frame()
r["e_lag"] = r.groupby(level=0)["e"].shift(1)
r = r.dropna()
ar = PooledOLS(r["e"], sm.add_constant(r[["e_lag"]])).fit(cov_type="clustered", cluster_entity=True)
print(f"AR(1) coefficient of the residuals: {ar.params['e_lag']:.3f}  (s.e. {ar.std_errors['e_lag']:.3f}, p-value {ar.pvalues['e_lag']:.3f})")

from statsmodels.stats.diagnostic import het_breuschpagan
bp = het_breuschpagan(pooled.resids, X)
print(f"Breusch-Pagan heteroskedasticity test: LM = {bp[0]:.2f}   p-value = {bp[1]:.4f}")
```

<div class="read" markdown="1">
**How to read it**

- **Serial correlation.** Regressing each residual on the same firm's residual a year earlier gives a small positive coefficient (0.09), significant only at the 10% level. The serial correlation that remains is weak.
- **Heteroskedasticity.** The Breusch–Pagan test rejects clearly: the error variance differs across observations, as is usual with firm data (small firms have more volatile growth).

Both problems leave the pooled OLS coefficients unbiased; they only affect the standard errors. **Clustering by firm corrects for both**, which is why it was used from the start.
</div>

## Robustness checks

```python
fe_c = PanelOLS(y, X, entity_effects=True).fit(cov_type="clustered", cluster_entity=True)
re_c = RandomEffects(y, X).fit(cov_type="clustered", cluster_entity=True)
no_years = PooledOLS(y, sm.add_constant(df[xs])).fit(cov_type="clustered", cluster_entity=True)
trimmed = df[(df["g_emp"].abs() < df["g_emp"].abs().quantile(0.99))]
tr = PooledOLS(trimmed["g_emp"], sm.add_constant(trimmed[xs].join(years))).fit(cov_type="clustered", cluster_entity=True)

print(f"{'':28s}" + "".join(f"{v:>16s}" for v in xs))
for name, m in (("Pooled OLS (baseline)", pooled), ("Fixed effects", fe_c), ("Random effects", re_c),
                ("Without year effects", no_years), ("Drop top 1% |g_emp|", tr)):
    print(f"{name:28s}" + "".join(f"{m.params[v]:8.3f} ({m.std_errors[v]:.3f})" for v in xs))
```

<div class="read" markdown="1">
**How to read it**

- **Fixed effects** gives coefficients close to pooled OLS. This is what the F test predicted, and it is the strongest evidence a referee will accept that pooling is harmless.
- **Random effects** is identical to pooled OLS. The estimated variance of the firm effect is zero, so the random-effects transformation subtracts nothing: this is the LM test result seen from the other side.
- **Without year effects** the output coefficient changes the most: industry output and the recession move together, so without year effects output growth picks up the common cycle. Keep the year effects.
- **Dropping the most extreme 1% of employment growth** (mergers, plant closures) leaves the capital and output effects similar, but **halves the wage effect**, to −0.21. It remains significant, but its size depends on a few firm-years with very large swings. Report this check: a reader needs to know that the magnitude, though not the sign, is fragile.
</div>

## Build the results table for the paper

```python
def star(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

cols = {"(1) Pooled OLS": pooled, "(2) FE": fe_c, "(3) RE": re_c}
print(f"{'':14s}" + "".join(f"{c:>16s}" for c in cols))
for v in xs:
    print(f"{v:14s}" + "".join(f"{m.params[v]:13.4f}{star(m.pvalues[v]):3s}" for m in cols.values()))
    print(f"{'':14s}" + "".join(f"{'(' + format(m.std_errors[v], '.4f') + ')':>13s}   " for m in cols.values()))
print(f"{'Observations':14s}" + "".join(f"{m.nobs:>13d}   " for m in cols.values()))
print("Year effects in all columns. Standard errors clustered by firm. *** p<0.01, ** p<0.05, * p<0.10.")
print(f"F test for firm effects: F({f.df}, {f.df_denom}) = {f.stat:.2f} (p = {f.pval:.3f}); Breusch-Pagan LM = {lm:.2f} (p = {stats.chi2.sf(lm, 1):.3f})")
```

<div class="read" markdown="1">
**How to read it**

Column (1) is the preferred estimate; columns (2) and (3) show that it does not depend on the treatment of firm effects. The F and LM statistics in the notes are the justification for pooling.
</div>

## Write the paper

<div class="write" markdown="1">
**How to write it: specification**

Because the model is specified in growth rates, permanent differences in employment levels across firms are differenced out. Consistent with this, an F test does not reject the hypothesis that firm-specific intercepts are jointly zero, and a Breusch–Pagan LM test does not reject the absence of random firm effects. We therefore estimate the model by pooled OLS with year effects, and cluster standard errors by firm to allow for heteroskedasticity and serial correlation within firms.
</div>

<div class="write" markdown="1">
**How to write it: results**

Real wage growth has a negative and significant effect on employment growth: a one percentage point faster increase in the real wage reduces employment growth by about 0.4 percentage points (Table 1, column 1). Capital growth and industry output growth are positively associated with employment growth. Fixed- and random-effects estimates are very close to the pooled estimates (columns 2 and 3), confirming that unobserved firm heterogeneity does not drive the results. The magnitude of the wage effect is, however, sensitive to a small number of extreme observations: excluding the top 1 percent of absolute employment changes reduces it to about −0.2, still significant.
</div>

<div class="warn" markdown="1">
**Common mistakes**

- Pooling variables in levels without testing for unit effects. In levels the F test almost always rejects.
- Reporting conventional standard errors. Pooled observations on the same unit are not independent; cluster by unit.
- Omitting year effects. Common shocks then load onto whichever regressor moves with the cycle.
- Using pooled OLS for a dynamic model with a lagged dependent variable. Pooled OLS is then biased upwards; see the GMM guide.
</div>
