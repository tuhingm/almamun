# Lecture 14 — Quantitative Macro Policy Analysis, Validation, and Replication

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Design a transparent quantitative macroeconomic application; distinguish calibration from estimation; evaluate mechanisms, welfare, and robustness.

## 1. Research design
Begin with a clearly stated economic question, a model mechanism, observable implications, and a counterfactual. Map each empirical object to a model variable. State the unit, frequency, transformation, sample, and source of each series.

## 2. Calibration and estimation
Calibration selects parameters from external evidence or long-run targets. Estimation uses a statistical criterion such as likelihood or moments. A hybrid strategy must identify which parameters are calibrated and which are estimated, and explain why.

## 3. Model validation
Evaluate steady-state restrictions, equilibrium residuals, impulse-response signs and timing, unconditional moments, forecast errors, and posterior predictive distributions. Validation is not merely reproducing the calibration targets.

## 4. Counterfactual policy
A policy counterfactual changes a rule, shock process, or structural parameter and resolves the model. Keep the comparison coherent: same initial conditions, shock normalization, and welfare metric. Separate transition effects from long-run effects.

## 5. Welfare
For a representative household, compare expected lifetime utility across allocations. A consumption-equivalent variation \(\lambda\) solves
\[
E_0\sum_t\beta^tu((1+\lambda)c_t^{base})=
E_0\sum_t\beta^tu(c_t^{policy}),
\]
with timing and risk conventions stated. In heterogeneous-agent settings, report distributional welfare and the social aggregation rule rather than only an average.

## 6. Reproducibility protocol
Archive source data or access instructions, transformations, parameter files, software versions, random seeds, estimation settings, and scripts. A replication package should regenerate tables and figures from raw or documented input data.

## 7. Computational lab
Replicate a published RBC or NK result, then change one structural assumption. Produce a model description, parameter table, steady-state diagnostics, IRFs, robustness table, and concise policy interpretation.

## Problem set
1. Distinguish calibration and estimation.
2. Design three validation tests for a DSGE model.
3. Define a consumption-equivalent welfare comparison.
4. Specify a reproducibility checklist for a quantitative macro paper.

## Coding exercises

These exercises combine **real U.S. macroeconomic data** (the quarterly `macrodata` series in `statsmodels`, 1959–2009,
and annual series from the `wooldridge` package) with **numerical solutions of the models** in this lecture. Install once
with `pip install numpy scipy pandas statsmodels wooldridge`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 14.1 — Local projections of a monetary shock measured as a Taylor-rule residual

*Data:* `macrodata` — U.S. quarterly data, 1960–2007.

1. Estimate a Taylor rule with smoothing on 1960–2007 (as in Lecture 7) including two lags of inflation and the output gap, and treat its residual as a monetary policy shock (a Romer–Romer-style shortcut).
2. Estimate local projections of the unemployment rate and inflation on the shock for \(h=0,\dots,12\) quarters, controlling for four lags of unemployment, inflation and the T-bill rate. Use HAC standard errors.
3. Report the responses to a 100 bp shock. Do you find a price puzzle? What identification assumption is doing the work?

```python
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
# TODO: rule regression -> shock = residual; for h: y_{t+h} - y_{t-1} on shock_t + controls
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"year": m["year"], "i": m["tbilrate"], "pi": m["infl"].rolling(4).mean(), "u": m["unemp"],
                  "x": 100 * hpfilter(np.log(m["realgdp"]), 1600)[0]})
for v in ("i", "pi", "x", "u"):
    for k in (1, 2, 3, 4): d[f"{v}{k}"] = d[v].shift(k)
s = d[(d.year >= 1960) & (d.year <= 2007)].dropna().copy()
rule = smf.ols("i ~ pi + x + pi1 + x1 + i1 + i2", data=s).fit()
s["shock"] = rule.resid
print(f"policy-rule R2 {rule.rsquared:.3f}; shock sd {s['shock'].std():.3f} pp")
ctrl = " + ".join(f"{v}{k}" for v in ("u", "pi", "i") for k in (1, 2, 3, 4))
rows = []
for h in range(0, 13):
    for v in ("u", "pi"):
        s[f"lead"] = s[v].shift(-h) - s[f"{v}1"]
        r = smf.ols(f"lead ~ shock + {ctrl}", data=s.dropna(subset=["lead"])).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        rows.append((h, v, r.params["shock"], r.bse["shock"]))
out = pd.DataFrame(rows, columns=["h", "var", "coef", "se"]).pivot(index="h", columns="var", values=["coef", "se"])
print(out.loc[[0, 2, 4, 6, 8, 10, 12]].round(3).to_string())
```

Output:

```text
policy-rule R2 0.921; shock sd 0.764 pp
      coef            se       
var     pi      u     pi      u
h                              
0   -0.037 -0.079  0.104  0.024
2    0.091 -0.142  0.113  0.058
4    0.026 -0.099  0.127  0.080
6   -0.152  0.028  0.166  0.096
8   -0.387  0.119  0.181  0.105
10  -0.537  0.116  0.182  0.075
12  -0.606  0.042  0.211  0.072
```

The Taylor-rule residual is a crude shock: the rule explains 92% of the T-bill rate, and the residual still contains the Fed's
reaction to information not in the rule. The responses show it. Unemployment **falls** for the first year after a "tightening"
and rises only after about two years, peaking near 0.12 points; inflation is flat at first and then falls by about 0.6 points after
three years. The early unemployment response has the wrong sign, a symptom of residual endogeneity — the same problem as the
price puzzle. Narrative (Romer and Romer, 2004) or high-frequency shocks (Gertler and Karadi, 2015) give cleaner identification.
The identification assumption here is that the residual is uncorrelated with all other shocks affecting the outcomes.

</details>

### Exercise 14.2 — Okun's law: estimation and parameter stability

*Data:* `okun` — U.S. annual real GDP growth `pcrgdp` and the change in the unemployment rate `cunem`, 1959–2005.

1. Estimate Okun's law \(\Delta u_t=a+b\,g_t+e_t\). Compute the implied "normal" growth rate \(-a/b\) at which unemployment is stable, with a delta-method standard error.
2. Test whether \(b=-0.5\) (the textbook rule of thumb).
3. Test for a break in 1983 with a Chow test, and estimate the two subsamples. Has the relationship changed?

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
o = wd.data("okun")
r = smf.ols("cunem ~ pcrgdp", data=o).fit()
# TODO: -a/b with delta method; t_test("pcrgdp = -0.5"); Chow with post-1983 interactions
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
o = wd.data("okun")
r = smf.ols("cunem ~ pcrgdp", data=o).fit(cov_type="HC1")
a, b = r.params["Intercept"], r.params["pcrgdp"]; V = r.cov_params().to_numpy()
g = np.array([-1 / b, a / b**2]); se = np.sqrt(g @ V @ g)
print(f"Okun: d(unemployment) = {a:.3f} {b:+.3f} * growth;  normal growth -a/b = {-a/b:.2f}% (se {se:.2f})")
t = r.t_test("pcrgdp = -0.5")
print(f"test b = -0.5: t = {float(np.squeeze(t.tvalue)):.2f}, p = {float(t.pvalue):.3f}")
o["post"] = (o["year"] >= 1983).astype(int)
c = smf.ols("cunem ~ pcrgdp * post", data=o).fit(cov_type="HC1")
w = c.wald_test("post = 0, pcrgdp:post = 0", scalar=True)
print(f"Chow test for a break in 1983: chi2(2) = {w.statistic:.2f}, p = {w.pvalue:.3f}")
for lab, sub in (("1959-1982", o[o.post == 0]), ("1983-2005", o[o.post == 1])):
    rs = smf.ols("cunem ~ pcrgdp", data=sub).fit()
    print(f"  {lab}: b = {rs.params['pcrgdp']:.3f}, normal growth {-rs.params['Intercept']/rs.params['pcrgdp']:.2f}%")
```

Output:

```text
Okun: d(unemployment) = 1.254 -0.376 * growth;  normal growth -a/b = 3.34% (se 0.19)
test b = -0.5: t = 2.98, p = 0.003
Chow test for a break in 1983: chi2(2) = 10.73, p = 0.005
  1959-1982: b = -0.368, normal growth 3.87%
  1983-2005: b = -0.398, normal growth 2.85%
```

Each extra point of GDP growth lowers unemployment by about 0.38 points, and unemployment is stable when growth is about 3.3%.
The textbook coefficient of −0.5 is rejected. The Chow test detects a break in 1983: the slope is similar across subperiods, but
the growth rate needed to keep unemployment stable fell from about 3.9% to 2.9%, reflecting slower labor-force and productivity
growth. The break is in the **intercept**, which is why rule-of-thumb versions of Okun's law need periodic updating.

</details>

## Further reading
Smets and Wouters (2007); Fernández-Villaverde (2010); Del Negro and Schorfheide, *DSGE Models in Macroeconomics*.
