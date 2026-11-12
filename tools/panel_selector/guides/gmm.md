title: Difference GMM for Dynamic Panels
short: GMM
when: Dynamic panel, large N, short T
data: 140 UK manufacturing firms, annual, 1976–1984 (Arellano and Bond 1991)
lead: Use this guide when the model has a lagged dependent variable and the panel has many units but few periods. Pooled OLS and fixed effects are then both biased, in opposite directions. The guide estimates the Arellano–Bond difference GMM estimator on the data it was developed for, runs the three specification tests every referee asks for, and checks the sensitivity to the instrument set.
---

## The question and why the usual estimators fail

The question: **how persistent is firm employment, and how does it respond to wages and capital?** Employment adjusts slowly because hiring and firing are costly, so this year's employment depends on last year's. That makes the model dynamic:

\[
n_{it} = \alpha_1 n_{i,t-1} + \alpha_2 n_{i,t-2} + \beta_0 w_{it} + \beta_1 w_{i,t-1} + \gamma' k_{it} + \delta' ys_{it} + \lambda_t + \eta_i + \varepsilon_{it}
\]

with all variables in logs: employment \(n\), the real wage \(w\), capital \(k\) (current and two lags) and industry output \(ys\) (current and two lags). \(\eta_i\) is a firm fixed effect.

| Estimator | Problem with a lagged dependent variable | Bias in \(\alpha_1\) |
|---|---|---|
| Pooled OLS | \(n_{i,t-1}\) contains \(\eta_i\), which is in the error | **upwards** |
| Fixed effects | demeaning makes \(n_{i,t-1}\) correlated with the demeaned error (Nickell bias, of order 1/T) | **downwards** |
| Difference GMM | first-difference to remove \(\eta_i\), then instrument \(\Delta n_{i,t-1}\) with \(n_{i,t-2}, n_{i,t-3}, \dots\) | consistent as N grows, if the instruments are valid |

<div class="read" markdown="1">
**When to use this guide, and when not**

GMM for dynamic panels is designed for **many units and few periods**. Its consistency comes from N growing, not T. With 20 countries over 40 years the Nickell bias is small, the instruments multiply out of control, and fixed effects (or the macro-panel guides) are the better choice. The model selection tool sends a panel here only when T is short (below 15), N is large (at least 30 units) and you say the model is dynamic. With fewer units it recommends bias-corrected fixed effects (LSDVC) instead.
</div>

<div class="write" markdown="1">
**How to write it: model section**

We estimate a dynamic labour-demand equation in which log employment depends on its own two lags, the current and lagged log real wage, and current and lagged log capital and industry output, together with year effects \(\lambda_t\) and firm effects \(\eta_i\). Because the lagged dependent variable is correlated with the firm effect, pooled OLS is biased upwards, and because T is short, the within estimator is subject to the Nickell bias. We therefore use the difference GMM estimator of Arellano and Bond (1991).
</div>

## Load the panel and build the lags

```python
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from diff_gmm import diff_gmm          # helper file: panel-data-guide/examples/diff_gmm.py

panel = pd.read_csv("employment.csv").rename(columns={"log_emp": "n", "log_wage": "w", "log_capital": "k", "log_output": "ys"})
panel = panel.sort_values(["firm", "year"])
by_firm = panel.groupby("firm")
for var, lag in (("n", 1), ("n", 2), ("w", 1), ("k", 1), ("k", 2), ("ys", 1), ("ys", 2)):
    panel[f"{var}_l{lag}"] = by_firm[var].shift(lag)

print(f"{panel['firm'].nunique()} firms, {panel['year'].min()}-{panel['year'].max()}, {len(panel)} observations")
print("Years per firm:", by_firm["year"].count().value_counts().sort_index().to_dict())
```

<div class="read" markdown="1">
**How to read it**

This is the shape GMM needs: **140 firms and 7 to 9 years each**. The model selection tool ran the static tests on this file (fixed effects is preferred to random effects) and then asked whether the model is dynamic. Because employment adjusts gradually, the answer is yes, which leads here. [Download the helper file](examples/diff_gmm.py) and the [data](examples/employment.csv).
</div>

## Step 1: pooled OLS and fixed effects give the bounds

```python
controls = ["w", "w_l1", "k", "k_l1", "k_l2", "ys", "ys_l1", "ys_l2"]
formula = "n ~ n_l1 + n_l2 + " + " + ".join(controls) + " + C(year)"
sample = panel.dropna()
cluster = {"cov_type": "cluster", "cov_kwds": {"groups": sample["firm"]}}

pooled = smf.ols(formula, sample).fit(**cluster)
fixed = smf.ols(formula + " + C(firm)", sample).fit(**cluster)
print(f"{'':16s}{'L1.n':>8s}{'L2.n':>8s}{'w':>8s}{'k':>8s}")
for name, m in (("Pooled OLS", pooled), ("Fixed effects", fixed)):
    print(f"{name:16s}{m.params['n_l1']:8.3f}{m.params['n_l2']:8.3f}{m.params['w']:8.3f}{m.params['k']:8.3f}")
```

<div class="read" markdown="1">
**How to read it**

The coefficient on lagged employment is above 1 by pooled OLS and about 0.73 by fixed effects. Because the first is biased upwards and the second downwards, **a credible GMM estimate should lie between them, or close to the fixed-effects bound**. This check (Bond 2002) costs two regressions and is the first thing to run in any dynamic panel study.
</div>

## Step 2: difference GMM

```python
gmm = diff_gmm(panel, unit="firm", time="year", y="n", lags_y=2, x=controls, gmm_lags=(2, None))
print(gmm["table"].loc[["L1.n", "L2.n"] + controls].round(4))
print(f"\nObservations: {gmm['nobs']}   firms: {gmm['units']}   instruments: {gmm['instruments']}")

t = gmm["table"]["coef"]
persistence = 1 - t["L1.n"] - t["L2.n"]
print(f"Long-run wage elasticity:    {(t['w'] + t['w_l1']) / persistence:.2f}")
print(f"Long-run capital elasticity: {(t['k'] + t['k_l1'] + t['k_l2']) / persistence:.2f}")
```

<div class="read" markdown="1">
**How to read it**

These are one-step estimates with standard errors robust to heteroskedasticity and to correlation within each firm. They reproduce the published Arellano–Bond results for this dataset (0.686 on lagged employment), which is how the helper function was checked.

- **Persistence.** About 0.69 on the first lag. Employment adjusts gradually.
- **Wages.** A 1% rise in the real wage lowers employment by about 0.6% in the same year; the long-run elasticity, which allows for the dynamics, is about −0.5.
- **Capital.** A 1% rise in capital raises employment by about 0.36% at once, and by about 0.7% in the long run.
- **The bounds.** The GMM persistence lies slightly **below** the fixed-effects estimate. That is a warning sign of weak instruments in the differenced equation: when employment is highly persistent, its lagged levels predict its differences poorly. Step 4 returns to this.
</div>

## Step 3: the three specification tests

```python
z1, p1 = gmm["ar1"]; z2, p2 = gmm["ar2"]; j, dof, pj = gmm["hansen"]
print(f"Arellano-Bond AR(1) test:  z = {z1:6.2f}   p-value = {p1:.3f}")
print(f"Arellano-Bond AR(2) test:  z = {z2:6.2f}   p-value = {p2:.3f}")
print(f"Hansen J test:             chi2({dof}) = {j:.2f}   p-value = {pj:.3f}")
print(f"Instruments: {gmm['instruments']}    firms: {gmm['units']}")
```

<div class="read" markdown="1">
**How to read it**

A dynamic panel GMM table is not complete without these.

- **AR(1)** in the differenced residuals is expected: differencing creates it. It says nothing about validity.
- **AR(2) is the test that matters.** If the errors in levels are serially uncorrelated, the differenced errors have no second-order correlation. Here it is not rejected, so levels dated \(t-2\) are valid instruments. If it were rejected, the instruments would have to start at \(t-3\).
- **Hansen J.** The null is that all instruments are valid. It is not rejected. Be suspicious of p-values very close to 1: that is a symptom of too many instruments, not of very good ones.
- **Instrument count.** The count must stay well below the number of firms. Too many instruments overfit the endogenous variable, pull GMM towards the biased OLS estimate and weaken the Hansen test.
</div>

## Step 4: robustness to the instrument set

```python
specs = {"All lags (baseline)": {}, "Lags 2 to 4": {"gmm_lags": (2, 4)}, "Lags 2 to 3": {"gmm_lags": (2, 3)},
         "Lags 3 and deeper": {"gmm_lags": (3, None)}, "Collapsed instruments": {"collapse": True}}
print(f"{'':24s}{'L1.n':>7s}{'(s.e.)':>9s}{'w':>8s}{'k':>8s}{'instr.':>8s}{'AR(2) p':>9s}{'Hansen p':>10s}")
results = {}
for name, options in specs.items():
    r = results[name] = diff_gmm(panel, unit="firm", time="year", y="n", lags_y=2, x=controls, **options)
    c, se = r["table"]["coef"], r["table"]["std err"]
    print(f"{name:24s}{c['L1.n']:7.3f}{'(' + format(se['L1.n'], '.3f') + ')':>9s}{c['w']:8.3f}{c['k']:8.3f}{r['instruments']:8d}{r['ar2'][1]:9.3f}{r['hansen'][2]:10.3f}")
print(f"\nBounds for L1.n:  fixed effects {fixed.params['n_l1']:.3f}   pooled OLS {pooled.params['n_l1']:.3f}")
```

<div class="read" markdown="1">
**How to read it**

Referees always ask whether the results depend on the instruments. Report a table like this one.

- **The wage and capital effects are stable** across all instrument sets: they are the robust findings.
- **The persistence coefficient is not.** It moves with the lag depth, and with collapsed instruments its standard error grows. Persistence is the parameter for which difference GMM is known to be weak when the series are highly persistent.
- **What to do.** Report the baseline, show this table, and estimate **system GMM** (Blundell and Bond 1998), which adds the equations in levels instrumented by lagged differences and is much stronger when persistence is high. In Stata: `xtabond2 n L(1/2).n w L.w k L(1/2).k ys L(1/2).ys i.year, gmm(L.n, collapse) iv(w L.w k L(1/2).k ys L(1/2).ys i.year) twostep robust` (see the Stata workflow on the main page).
</div>

## Build the results table for the paper

```python
def star(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

base, col = results["All lags (baseline)"], results["Collapsed instruments"]
names = ["L1.n", "L2.n", "w", "w_l1", "k"]
fmap = {"L1.n": "n_l1", "L2.n": "n_l2"}
print(f"{'':10s}{'(1) Pooled OLS':>16s}{'(2) FE':>16s}{'(3) Diff. GMM':>16s}{'(4) GMM collapsed':>19s}")
for v in names:
    o = fmap.get(v, v)
    cells = [(pooled.params[o], pooled.bse[o], pooled.pvalues[o]), (fixed.params[o], fixed.bse[o], fixed.pvalues[o]),
             tuple(base["table"].loc[v, ["coef", "std err", "p-value"]]), tuple(col["table"].loc[v, ["coef", "std err", "p-value"]])]
    print(f"{v:10s}" + "".join(f"{b:13.3f}{star(p):3s}" for b, s, p in cells))
    print(f"{'':10s}" + "".join(f"{'(' + format(s, '.3f') + ')':>13s}   " for b, s, p in cells))
print(f"{'AR(2) p':10s}{'':32s}{base['ar2'][1]:13.3f}   {col['ar2'][1]:13.3f}")
print(f"{'Hansen p':10s}{'':32s}{base['hansen'][2]:13.3f}   {col['hansen'][2]:13.3f}")
print(f"{'Instruments':10s}{'':31s}{base['instruments']:13d}   {col['instruments']:13d}")
print("Year effects in all columns. Standard errors robust to within-firm correlation. *** p<0.01, ** p<0.05, * p<0.10.")
```

<div class="read" markdown="1">
**How to read it**

Columns (1) and (2) are the bounds, columns (3) and (4) the GMM estimates. Always report the AR(2) and Hansen p-values and the instrument count in the table itself, not only in the text.
</div>

## Write the paper

<div class="write" markdown="1">
**How to write it: estimation strategy**

Because the model includes lagged employment and firm fixed effects, and T is short, both pooled OLS and the within estimator are inconsistent, with biases of opposite sign. We use the Arellano–Bond difference GMM estimator, instrumenting the differenced lags of employment with levels dated \(t-2\) and earlier and treating wages, capital and output as strictly exogenous. We report one-step estimates with standard errors robust to heteroskedasticity and within-firm correlation.
</div>

<div class="write" markdown="1">
**How to write it: results**

The Arellano–Bond test does not detect second-order serial correlation in the differenced residuals, and the Hansen test does not reject the validity of the instruments (Table 1, column 3). A 1 percent increase in the real wage reduces employment by about 0.6 percent on impact and by about 0.5 percent in the long run. The wage and capital elasticities are robust to the choice of instrument lags and to collapsing the instrument matrix, whereas the estimated persistence of employment is sensitive to the instrument set, consistent with weak instruments for highly persistent series.
</div>

<div class="warn" markdown="1">
**Common mistakes**

- Using GMM with few units (say, 20 countries). The asymptotics need large N, and the instrument count quickly exceeds the number of units.
- Not reporting the instrument count, or using all available lags without collapsing in longer panels.
- Treating a Hansen p-value of 0.99 as reassuring. It usually means too many instruments.
- Ignoring the bounds check. A GMM persistence estimate above pooled OLS or well below fixed effects signals a problem.
- Treating endogenous regressors as exogenous. If wages respond to employment shocks, they must be instrumented with their own lags, just like the lagged dependent variable.
</div>
