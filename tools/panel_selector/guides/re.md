title: Random Effects
short: RE
when: Unit effects present, uncorrelated with the regressors
data: 532 U.S. men, annual hours and wages, 1979–1988 (PSID, Ziliak 1997)
lead: Use this guide when the panel is short, the units differ systematically, but the Hausman and Mundlak tests do not reject that those differences are uncorrelated with the regressors. Random effects then uses both the within and the between variation, keeps time-invariant regressors, and gives more precise estimates than fixed effects.
---

## The question and the model

The question is a central one in labour economics and in macroeconomic models of the labour market: **how much do men change their working hours when their wage changes?** The answer, the labour-supply elasticity, governs how taxes affect work and how much hours move over the business cycle.

\[
\ln h_{it} = \beta_1 \ln w_{it} + \beta_2\, kids_{it} + \beta_3\, age_{it} + \beta_4\, disab_{it} + \lambda_t + u_i + \varepsilon_{it}
\]

| Symbol | Variable in the file | Definition | Expected sign |
|---|---|---|---|
| \(\ln h_{it}\) | `lnhr` | Log annual hours worked | dependent variable |
| \(\ln w_{it}\) | `lnwg` | Log hourly wage | positive and small: the labour-supply elasticity |
| \(kids_{it}\) | `kids` | Number of children | positive: more needs to provide for |
| \(age_{it}\) | `age` | Age in years | negative over the life cycle |
| \(disab_{it}\) | `disab` | 1 if a health limitation is reported | negative |
| \(u_i\) | — | Worker effect: tastes for work, occupation | treated as random |

In the random-effects model the worker effect \(u_i\) is part of the error, with variance \(\sigma_u^2\). The key assumption is that \(u_i\) is **uncorrelated with every regressor in every year**. The tests below check it.

<div class="read" markdown="1">
**Before you estimate anything**

Random effects is not a weaker version of fixed effects. If its assumption holds, it is consistent and more efficient, and it can estimate the effect of variables that do not change over time. If the assumption fails, it is biased. So the choice is made by a test, and the paper must report it.
</div>

<div class="write" markdown="1">
**How to write it: model section**

We model log annual hours as \(\ln h_{it} = \beta_1 \ln w_{it} + \beta_2 kids_{it} + \beta_3 age_{it} + \beta_4 disab_{it} + \lambda_t + u_i + \varepsilon_{it}\), where \(\lambda_t\) are year effects and \(u_i\) is a worker-specific effect capturing persistent differences in tastes for work. The coefficient \(\beta_1\) is the elasticity of hours with respect to the wage.
</div>

## Load the panel and look at the variation

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects, compare

df = pd.read_csv("laborsupply.csv").set_index(["id", "year"])
print(f"{df.index.get_level_values(0).nunique()} men x {df.index.get_level_values(1).nunique()} years = {len(df)} rows")

xs = ["lnwg", "kids", "age", "disab"]
for v in ["lnhr"] + xs:
    x = df[v]
    within = x - x.groupby(level=0).transform("mean")
    print(f"{v:6s} mean = {x.mean():7.3f}   between sd = {x.groupby(level=0).mean().std():6.3f}   within sd = {within.std():6.3f}")
```

<div class="read" markdown="1">
**How to read it**

A balanced, short panel: 532 men, each observed for 10 years. The model selection tool ran the specification tests on this file and recommended random effects; this guide repeats them in full.

Look at the wage. Its between standard deviation (0.39) is more than twice its within standard deviation (0.17): **most of the wage variation is across men, not over time for the same man.** Fixed effects throws the between variation away. Random effects uses it, which is why it will be much more precise here, provided its assumption holds.
</div>

## Estimate the random-effects model

```python
years = pd.get_dummies(df.index.get_level_values(1), prefix="y", drop_first=True, dtype=float).set_index(df.index)
y = df["lnhr"]
X = sm.add_constant(df[xs].join(years))

re = RandomEffects(y, X).fit(cov_type="clustered", cluster_entity=True)
print(re.params[xs].round(4).to_frame("coef").join(re.std_errors[xs].round(4).rename("std err")).join(re.pvalues[xs].round(4).rename("p-value")))

vd = re.variance_decomposition
print(f"\nsigma_u^2 (worker effect) = {vd['Effects']:.4f}   sigma_e^2 (idiosyncratic) = {vd['Residual']:.4f}   rho = {vd['Percent due to Effects']:.3f}")
print(f"theta (share of the worker mean removed): {re.theta.iloc[0, 0]:.3f}")
```

<div class="read" markdown="1">
**How to read it**

- **Wage elasticity.** A 10% higher wage is associated with about 1.2% more hours, significant at the 5% level. This is small, as the literature on prime-age men consistently finds.
- **Children** raise hours slightly; **a health limitation** lowers them by about 7%.
- **Variance components.** About a third of the unexplained variation in hours is a permanent worker effect (\(\rho\)); the rest changes from year to year.
- **\(\theta\).** Random effects subtracts a fraction \(\theta\) of each man's average from his data. With \(\theta = 0\) it would be pooled OLS, with \(\theta = 1\) fixed effects. Here \(\theta\) is about 0.6, so RE sits between the two: it is a weighted average of the within and between estimators.
</div>

## Test 1: are there worker effects at all? Breusch–Pagan LM

```python
pooled = PooledOLS(y, X).fit(cov_type="clustered", cluster_entity=True)
e = pooled.resids
n, T_i = len(e), e.groupby(level=0).size()
lm = n ** 2 / (2 * (T_i * (T_i - 1)).sum()) * ((e.groupby(level=0).sum() ** 2).sum() / (e ** 2).sum() - 1) ** 2
print(f"Breusch-Pagan LM test: chi2(1) = {lm:.1f}   p-value = {stats.chi2.sf(lm, 1):.4f}")
```

<div class="read" markdown="1">
**How to read it**

The null hypothesis is \(\sigma_u^2 = 0\): no worker effect, in which case pooled OLS would do. The statistic is very large, so **the worker effects are real**. Pooled OLS with clustered standard errors would still be consistent here, but random effects is more efficient, and the clustered pooled standard errors would hide this structure.
</div>

## Test 2: is the random-effects assumption credible? Hausman and Mundlak

```python
fe = PanelOLS(y, sm.add_constant(df[xs].join(years)), entity_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
common = [v for v in xs if v in fe.params.index]
print("Coefficients compared:", common)

fe_p = PanelOLS(y, sm.add_constant(df[xs].join(years)), entity_effects=True, drop_absorbed=True).fit()
re_p = RandomEffects(y, X).fit()
d = fe_p.params[common] - re_p.params[common]
H = float(d @ np.linalg.inv(fe_p.cov.loc[common, common] - re_p.cov.loc[common, common]) @ d)
print(f"Hausman test:  chi2({len(common)}) = {H:.2f}   p-value = {stats.chi2.sf(H, len(common)):.4f}")

means = df[xs].groupby(level=0).transform("mean").add_suffix("_mean")
mundlak = RandomEffects(y, X.join(means)).fit(cov_type="clustered", cluster_entity=True)
w = mundlak.wald_test(formula=" = ".join(means.columns) + " = 0")
print(f"Mundlak test:  chi2({w.df}) = {w.stat:.2f}   p-value = {w.pval:.4f}")
print(mundlak.params[means.columns].round(4).to_string())
```

<div class="read" markdown="1">
**How to read it**

The two tests disagree, and the reason matters for every panel paper.

- **Conventional Hausman test.** It compares the FE and RE coefficients using the textbook variance formula, which assumes that RE is fully efficient: homoskedastic errors and no correlation over time other than the worker effect. It rejects (p = 0.001).
- **Mundlak test.** It adds each man's average wage, number of children, age and disability to the RE model. If tastes for work were correlated with these averages, their coefficients would be significant. Because it uses clustered standard errors, it stays valid when the errors are heteroskedastic or serially correlated. It does **not** reject (p = 0.16).

Hours data are strongly heteroskedastic: a few men have very low hours in some years (the minimum of log hours is 2.8, about 16 hours a year). Under heteroskedasticity the conventional variance formula is wrong, so the conventional test cannot be trusted. **The robust test is the one to base the decision on, and it supports random effects**, which, as the more efficient estimator, is the one to report.

Two cautions. Failing to reject is not proof that the assumption holds. And because the two tests disagree, report both, and report fixed effects alongside as the robustness check.
</div>

## Compare with fixed effects

```python
for name, m in (("Random effects", re), ("Fixed effects", fe)):
    print(f"{name:16s} lnwg = {m.params['lnwg']:.4f} ({m.std_errors['lnwg']:.4f})   kids = {m.params['kids']:.4f} ({m.std_errors['kids']:.4f})   disab = {m.params['disab']:.4f} ({m.std_errors['disab']:.4f})")
print(f"\nStandard error of the wage elasticity is {fe.std_errors['lnwg'] / re.std_errors['lnwg']:.1f} times larger with fixed effects")
```

<div class="read" markdown="1">
**How to read it**

The point estimates are close: the FE wage elasticity is somewhat larger, but the gap is well inside the confidence intervals. The main difference is **precision**: fixed effects uses only the small within variation in wages, so its standard error is larger. This is the efficiency gain that random effects buys when its assumption holds.
</div>

## Diagnostic test: serial correlation

```python
from linearmodels.panel import FirstDifferenceOLS
fd = FirstDifferenceOLS(y, df[["lnwg", "kids", "disab"]]).fit(cov_type="clustered", cluster_entity=True)
r = fd.resids.rename("e").to_frame()
r["e_lag"] = r.groupby(level=0)["e"].shift(1)
r = r.dropna()
aux = PooledOLS(r["e"], r[["e_lag"]]).fit(cov_type="clustered", cluster_entity=True)
F = ((aux.params["e_lag"] + 0.5) / aux.std_errors["e_lag"]) ** 2
N = df.index.get_level_values(0).nunique()
print(f"Wooldridge test: coefficient = {aux.params['e_lag']:.3f} (-0.5 under H0)   F(1, {N - 1}) = {F:.2f}   p-value = {stats.f.sf(F, 1, N - 1):.4f}")
```

<div class="read" markdown="1">
**How to read it**

The random-effects model assumes that, apart from the permanent worker effect, the errors are uncorrelated over time. The Wooldridge test checks this: if the errors in levels are serially uncorrelated, the first-differenced errors have a correlation of exactly −0.5. Here the estimate is −0.41 and the test does not reject at 5% (p = 0.11), so **there is no strong evidence of serial correlation beyond the worker effect**, which supports the RE error structure. Had it rejected, the RE point estimates would remain consistent but the textbook standard errors would be wrong. **The clustered standard errors used throughout this guide are valid either way**, which is why they are the ones to report.
</div>

## Build the results table for the paper

```python
def star(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

cols = {"(1) Pooled OLS": pooled, "(2) RE": re, "(3) FE": fe, "(4) Mundlak": mundlak}
print(f"{'':12s}" + "".join(f"{c:>16s}" for c in cols))
for v in xs:
    print(f"{v:12s}" + "".join(f"{m.params[v]:13.4f}{star(m.pvalues[v]):3s}" if v in m.params else f"{'–':>13s}   " for m in cols.values()))
    print(f"{'':12s}" + "".join(f"{'(' + format(m.std_errors[v], '.4f') + ')':>13s}   " if v in m.params else " " * 16 for m in cols.values()))
print(f"{'Observations':12s}" + "".join(f"{m.nobs:>13d}   " for m in cols.values()))
print("All columns include year effects. Standard errors clustered by worker. *** p<0.01, ** p<0.05, * p<0.10.")
print(f"Breusch-Pagan LM = {lm:.1f} (p < 0.001); Hausman chi2({len(common)}) = {H:.2f} (p = {stats.chi2.sf(H, len(common)):.3f}); Mundlak chi2({w.df}) = {w.stat:.2f} (p = {w.pval:.3f})")
```

<div class="read" markdown="1">
**How to read it**

Column (2) is the preferred estimate. Columns (1) and (3) show that the conclusion does not depend on how the worker effects are treated, and column (4) shows the Mundlak model, whose coefficients on the worker averages are the basis of the test. The three test statistics belong in the table notes.
</div>

## Write the paper

<div class="write" markdown="1">
**How to write it: specification tests**

A Breusch–Pagan LM test strongly rejects the absence of worker-specific effects, so pooled OLS is inefficient. The conventional Hausman test rejects random effects, but it assumes homoskedastic errors, which is implausible for hours data; its cluster-robust Mundlak version does not reject the hypothesis that the worker effects are uncorrelated with the regressors (p = 0.16). We therefore estimate the model by random effects, with year effects and standard errors clustered by worker, and report fixed-effects estimates as a robustness check.
</div>

<div class="write" markdown="1">
**How to write it: results**

The estimated elasticity of hours with respect to the wage is about 0.12 and significant at the 5 percent level (Table 1, column 2). A 10 percent rise in the hourly wage is thus associated with a 1.2 percent increase in annual hours, in line with the small elasticities typically found for prime-age men. A reported health limitation reduces hours by about 7 percent. The fixed-effects estimate of the wage elasticity is somewhat larger, 0.17, but much less precise (column 3).
</div>

<div class="warn" markdown="1">
**Common mistakes**

- Choosing random effects because fixed effects “loses too many variables”. That is a reason to use the Mundlak model, not to assume the RE condition holds.
- Reporting only the conventional Hausman test. With heteroskedastic or serially correlated errors it is invalid, as this example shows, and it can even be negative. Report the Mundlak test too.
- Reading “do not reject” as “random effects is proven”. Show that the FE estimates tell the same story.
- Interpreting the wage elasticity causally. Wages may respond to hours (overtime pay, measurement error in hours, which also enters the wage as earnings divided by hours). The serious literature instruments the wage.
</div>
