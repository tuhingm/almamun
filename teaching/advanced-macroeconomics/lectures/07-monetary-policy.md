# Lecture 07 — Monetary Policy: Rules, Commitment, and the Effective Lower Bound

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Evaluate policy rules and optimal policy; distinguish commitment from discretion; analyze the effective lower bound and forward guidance.

## 1. Policy rules
A Taylor-type rule links the nominal policy rate to inflation and economic activity:
\[
i_t=\rho_i i_{t-1}+(1-\rho_i)(\bar i+\phi_\pi\pi_t+\phi_xx_t)+\varepsilon_t^i.
\]
The intercept and inflation reference must be defined relative to the steady state. Interest-rate smoothing introduces policy inertia.

## 2. Commitment and discretion
Under commitment, policymakers choose a state-contingent plan and internalize how expectations respond to future policy. Under discretion, policymakers reoptimize each period taking private expectations as given. In forward-looking models, the two regimes can produce different inflation and output dynamics.

## 3. Loss function
A common quadratic approximation is
\[
L=\frac12E_0\sum_{t=0}^{\infty}\beta^t(\pi_t^2+\lambda_xx_t^2).
\]
This is a model-based welfare approximation under specified assumptions, not a universal social objective. The weight \(\lambda_x\) depends on structural parameters and the welfare derivation.

## 4. Effective lower bound
When the nominal rate cannot fall sufficiently, the real rate may remain too high relative to the natural rate. Forward guidance affects current demand through expected future nominal rates and inflation. Asset purchases can operate through term premia and portfolio-balance channels absent from the simplest representative-agent model.

## 5. Worked example
If expected inflation is 2% and the nominal rate is 1%, the ex ante real rate is approximately -1% using the Fisher approximation. If expected inflation falls while the nominal rate is constrained, the real rate rises, tightening policy endogenously.

## 6. Computational lab
Compare a Taylor rule, optimal commitment, and discretionary policy in a linear NK model. Plot inflation, output gap, and interest rate after a cost-push shock. State the welfare criterion and solution method.

## Problem set
1. Explain policy inertia.
2. Distinguish commitment from discretion.
3. Derive the approximate real interest rate.
4. Explain why forward guidance can affect current output.

## Coding exercises

These exercises combine **real U.S. macroeconomic data** (the quarterly `macrodata` series in `statsmodels`, 1959–2009,
and annual series from the `wooldridge` package) with **numerical solutions of the models** in this lecture. Install once
with `pip install numpy scipy pandas statsmodels wooldridge`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 7.1 — Taylor rules before and after Volcker

*Data:* `macrodata` — U.S. quarterly 3-month T-bill rate, CPI inflation and real GDP.

Clarida, Galí and Gertler (2000) argue that U.S. policy was "passive" (\(\phi_\pi<1\)) before 1979 and "active" (\(\phi_\pi>1\)) afterwards.

1. Construct four-quarter inflation and the HP output gap.
2. Estimate a Taylor rule with interest-rate smoothing, \(i_t=(1-\rho)(c+\phi_\pi\pi_t+\phi_xx_t)+\rho i_{t-1}+e_t\), by nonlinear least squares for 1960–1979 and 1987–2007.
3. Report the long-run responses \(\phi_\pi,\phi_x\) with delta-method standard errors, and relate them to the determinacy condition of Lecture 6.

```python
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
# TODO: pi4 = rolling 4-quarter mean of infl; gap = 100*HP cycle of log GDP; OLS of i on pi4, gap, i_lag -> recover long-run coefficients
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"year": m["year"], "i": m["tbilrate"], "pi": m["infl"].rolling(4).mean(),
                  "x": 100 * hpfilter(np.log(m["realgdp"]), 1600)[0]})
d["i_lag"] = d["i"].shift(1)
for lo, hi in ((1960, 1979), (1987, 2007)):
    s = d[(d.year >= lo) & (d.year <= hi)].dropna()
    r = smf.ols("i ~ pi + x + i_lag", data=s).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    rho = r.params["i_lag"]; V = r.cov_params()
    out = {}
    for v in ("pi", "x"):
        est = r.params[v] / (1 - rho)
        g = np.zeros(len(r.params)); g[list(r.params.index).index(v)] = 1 / (1 - rho); g[list(r.params.index).index("i_lag")] = r.params[v] / (1 - rho)**2
        out[v] = (est, np.sqrt(g @ V.to_numpy() @ g))
    print(f"{lo}-{hi}: smoothing rho = {rho:.3f};  long-run phi_pi = {out['pi'][0]:.2f} (se {out['pi'][1]:.2f}), phi_x = {out['x'][0]:.2f} (se {out['x'][1]:.2f})")
```

Output:

```text
1960-1979: smoothing rho = 0.688;  long-run phi_pi = 0.56 (se 0.09), phi_x = 0.70 (se 0.20)
1987-2007: smoothing rho = 0.917;  long-run phi_pi = 0.55 (se 0.73), phi_x = 2.03 (se 0.73)
```

For 1960–79, the long-run inflation response is about 0.56, well below one: policy accommodated inflation, consistent with
Clarida, Galí and Gertler (2000). For 1987–2007, this simple specification does **not** find \(\phi_\pi>1\): the point estimate
is similar but has a huge standard error (0.73), because inflation varied little after the mid-1980s and smoothing is high, so
the inflation response is poorly identified. CGG use expected future inflation (GMM), the GDP deflator, the federal funds
rate and a different gap measure, and find about 2. Specification choices matter a lot for this famous result — an honest
lesson.

</details>

### Exercise 7.2 — The effective lower bound in 2009: what did a Taylor rule prescribe?

*Data:* `macrodata` — U.S. data through 2009Q3.

1. Compute the rate prescribed by Taylor's (1993) original rule, \(i_t=1+1.5\pi_t+0.5x_t\), for each quarter of 2008–2009, using an output gap whose trend is estimated only through 2007 (to avoid the end-point problem of the HP filter).
2. Compare it with the actual T-bill rate and with the rule estimated on 1987–2007 in the previous exercise. In which quarter did the prescribed rate first fall below zero, and how negative did it get?
3. What tools did the Fed use when the rule called for negative rates (forward guidance, asset purchases), and how does a "shadow rate" summarize them?

```python
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
# TODO: estimate on 1987-2007 as before; prescribed = c + phi_pi * pi + phi_x * x for 2008-2009
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"year": m["year"], "q": m["quarter"], "i": m["tbilrate"], "pi": m["infl"].rolling(4).mean(),
                  "x": 100 * hpfilter(np.log(m["realgdp"].loc[:195]), 1600)[0].reindex(m.index)})
# HP trend estimated through 2007Q4 only and extended linearly, to avoid the end-point problem of the filter
tr = np.log(m["realgdp"]) - d["x"] / 100
slope = tr.loc[190:195].diff().mean()
d.loc[196:, "x"] = 100 * (np.log(m["realgdp"].loc[196:]) - (tr.loc[195] + slope * np.arange(1, len(m) - 195)))
d["i_lag"] = d["i"].shift(1)
s = d[(d.year >= 1987) & (d.year <= 2007)].dropna()
r = smf.ols("i ~ pi + x + i_lag", data=s).fit()
rho = r.params["i_lag"]; c, fpi, fx = (r.params[v] / (1 - rho) for v in ("Intercept", "pi", "x"))
print(f"estimated 1987-2007 rule: i = {c:.2f} + {fpi:.2f} pi + {fx:.2f} x;   Taylor (1993): i = 1 + 1.5 pi + 0.5 x")
for _, row in d[d.year >= 2008].iterrows():
    print(f"  {int(row.year)}Q{int(row.q)}: inflation {row.pi:5.2f}, gap {row.x:6.2f} -> Taylor 1993 {1 + 1.5*row.pi + 0.5*row.x:6.2f}%, estimated rule {c + fpi*row.pi + fx*row.x:7.2f}%, actual T-bill {row.i:.2f}%")
```

Output:

```text
estimated 1987-2007 rule: i = 2.95 + 0.38 pi + 2.71 x;   Taylor (1993): i = 1 + 1.5 pi + 0.5 x
  2008Q1: inflation  3.85, gap  -0.98 -> Taylor 1993   6.28%, estimated rule    1.75%, actual T-bill 1.56%
  2008Q2: inflation  5.29, gap  -1.27 -> Taylor 1993   8.31%, estimated rule    1.52%, actual T-bill 1.74%
  2008Q3: inflation  3.64, gap  -2.60 -> Taylor 1993   5.16%, estimated rule   -2.72%, actual T-bill 1.17%
  2008Q4: inflation -0.15, gap  -4.64 -> Taylor 1993  -1.54%, estimated rule   -9.69%, actual T-bill 0.12%
  2009Q1: inflation -0.62, gap  -6.95 -> Taylor 1993  -3.40%, estimated rule  -16.14%, actual T-bill 0.22%
  2009Q2: inflation -1.91, gap  -7.79 -> Taylor 1993  -5.76%, estimated rule  -18.91%, actual T-bill 0.18%
  2009Q3: inflation -0.23, gap  -7.75 -> Taylor 1993  -3.22%, estimated rule  -18.17%, actual T-bill 0.12%
```

Taylor's 1993 rule would have called for a negative policy rate from 2008Q4, reaching about −5.8% in 2009Q2 — close to
estimates in the literature (Rudebusch, 2009). The rule estimated on 1987–2007 gives absurd values (below −18%), because its
large output-gap coefficient was estimated in a period of small gaps — an extrapolation warning. The T-bill rate was stuck
near zero. With conventional policy constrained, the Fed turned to forward guidance and large-scale asset purchases;
"shadow rate" estimates (Wu and Xia, 2016) translate their effect into an equivalent negative policy rate of a few percent.

</details>

## Further reading
Woodford, *Interest and Prices*; Clarida, Galí and Gertler (1999); Galí, *Monetary Policy, Inflation, and the Business Cycle*.
