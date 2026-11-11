# Lecture 08 — Fiscal Policy, Public Debt, and Monetary-Fiscal Interactions

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Derive debt dynamics; distinguish primary deficits from total deficits; analyze fiscal multipliers, sustainability, and Ricardian equivalence.

## 1. Government budget constraint
In nominal terms,
\[
B_t=(1+i_{t-1})B_{t-1}+P_tG_t-P_tT_t,
\]
where \(B_t\) is end-of-period nominal debt, \(G_t\) government purchases, and \(T_t\) net taxes. Dividing by nominal GDP and using a consistent timing convention gives an approximate debt ratio law
\[
b_t\approx\frac{1+r_t}{1+g_t}b_{t-1}+d_t,
\]
where \(r_t\) is the real interest rate, \(g_t\) real GDP growth, and \(d_t\) the primary deficit-to-GDP ratio. For small rates, \(\Delta b_t\approx(r_t-g_t)b_{t-1}+d_t\).

## 2. Intertemporal constraint
Under a no-Ponzi condition, current debt equals the present value of future primary surpluses, subject to valuation and discounting conventions. Fiscal sustainability is therefore about the feasible path of future primary balances, not a single debt threshold.

## 3. Fiscal multipliers
The output effect of government spending depends on monetary accommodation, price rigidity, household liquidity constraints, openness, financing, and the horizon. A multiplier is a ratio of an output response to a specified fiscal impulse; report its definition and horizon.

## 4. Ricardian equivalence
Under restrictive assumptions—lump-sum taxes, complete markets, forward-looking unconstrained households, and appropriate bequest motives—debt-financed tax cuts may not change consumption because households anticipate future taxes. Violations of these assumptions generate non-Ricardian effects.

## 5. Worked example
If \(b_{t-1}=0.8\), \(r=0.04\), \(g=0.02\), and the primary deficit is 0.01, the exact approximate ratio update is \(b_t=(1.04/1.02)0.8+0.01\). Calculate the new debt ratio and isolate the interest-growth contribution.

## 6. Computational lab
Simulate debt paths under alternative primary-balance rules and interest-growth differentials. Add a spending shock to an NK model and report impact and cumulative multipliers.

## Problem set
1. Derive the debt-ratio transition.
2. Explain the role of \(r-g\).
3. State the assumptions behind Ricardian equivalence.
4. Compare spending and tax multipliers in a specified model.

## Coding exercises

These exercises combine **real U.S. macroeconomic data** (the quarterly `macrodata` series in `statsmodels`, 1959–2009,
and annual series from the `wooldridge` package) with **numerical solutions of the models** in this lecture. Install once
with `pip install numpy scipy pandas statsmodels wooldridge`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 8.1 — Debt dynamics and the debt-stabilizing primary surplus

*Data:* `intdef` — U.S. federal receipts and outlays (% of GDP), 3-month T-bill rate and inflation, 1948–2003; `macrodata` for real GDP growth.

Government debt evolves as \(b_{t+1}=\frac{1+r}{1+g}b_t-s_t\), where \(b\) is debt/GDP, \(r\) the real interest rate, \(g\) real growth and \(s\) the primary surplus/GDP.

1. Compute the average real T-bill rate \(r\) and average real GDP growth \(g\). What is \(r-g\), and why does its sign matter?
2. Compute the debt-stabilizing primary surplus \(s^*=\frac{r-g}{1+g}b\) at debt ratios of 60% and 100%.
3. Simulate debt for 30 years from \(b_0=60\%\) under (a) \(s=0\), (b) \(s=-2\%\) (a primary deficit), for \(r-g\) equal to its historical value and to \(+2\) percentage points.

```python
import numpy as np, wooldridge as wd, statsmodels.api as sm
d = wd.data("intdef")
r = (d["i3"] - d["inf"]).mean() / 100
m = sm.datasets.macrodata.load_pandas().data; g = 4 * np.mean(np.diff(np.log(m["realgdp"])))
# TODO: s* = (r-g)/(1+g) b; simulate b' = (1+r)/(1+g) b - s
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd, statsmodels.api as sm
d = wd.data("intdef")
r = (d["i3"] - d["inf"]).mean() / 100
m = sm.datasets.macrodata.load_pandas().data; g = 4 * np.mean(np.diff(np.log(m["realgdp"])))
print(f"average real T-bill rate r = {100*r:.2f}%, average real growth g = {100*g:.2f}%, r - g = {100*(r-g):.2f} pp")
print(f"average federal deficit (outlays - receipts, including interest) = {(d['out'] - d['rec']).mean():.2f}% of GDP")
for b in (0.6, 1.0):
    print(f"debt-stabilizing primary surplus at b = {b:.0%}: {100*(r-g)/(1+g)*b:+.2f}% of GDP")
for rg_label, rr in (("historical r-g", r), ("r-g = +2 pp", g + 0.02)):
    for s in (0.0, -0.02):
        b = 0.6
        for _ in range(30): b = (1 + rr) / (1 + g) * b - s
        print(f"  {rg_label:15s}, primary surplus {100*s:+.0f}%: debt after 30 years = {100*b:.1f}% of GDP")
```

Output:

```text
average real T-bill rate r = 1.02%, average real growth g = 3.10%, r - g = -2.08 pp
average federal deficit (outlays - receipts, including interest) = 1.60% of GDP
debt-stabilizing primary surplus at b = 60%: -1.21% of GDP
debt-stabilizing primary surplus at b = 100%: -2.02% of GDP
  historical r-g , primary surplus +0%: debt after 30 years = 32.6% of GDP
  historical r-g , primary surplus -2%: debt after 30 years = 77.9% of GDP
  r-g = +2 pp    , primary surplus +0%: debt after 30 years = 106.8% of GDP
  r-g = +2 pp    , primary surplus -2%: debt after 30 years = 187.1% of GDP
```

Over this sample, the average real T-bill rate (about 1%) was well below real growth (about 3%), so \(r-g\approx-2\) points.
With \(r<g\), debt stabilizes even with a primary **deficit**: about 1.2% of GDP at a 60% debt ratio. With a balanced primary
budget, debt falls from 60% to about 33% in 30 years. If \(r-g\) rises to +2 points, the same policies make debt explode — a
2% primary deficit takes it to almost 190%. The sign of \(r-g\) is central to the debate on fiscal space (Blanchard, 2019).
(The T-bill rate understates the average cost of federal debt, which has longer maturities.)

</details>

### Exercise 8.2 — A government-spending multiplier from a recursive VAR

*Data:* `macrodata` — U.S. quarterly real government spending, real GDP and real consumption, per capita, 1959–2009.

Following Blanchard and Perotti (2002), assume government spending does not respond to output within the quarter, so
spending is ordered first in a Cholesky VAR.

1. Estimate a VAR(4) in log real per-capita government spending, GDP and consumption, with a linear trend.
2. Compute the impulse responses to a spending shock, and convert them into **dollar multipliers** using the sample average ratios \(Y/G\) and \(C/G\).
3. Report the impact and cumulative 8-quarter multipliers \(\sum\Delta Y/\sum\Delta G\). Does consumption rise or fall?

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.api import VAR
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"g": np.log(m["realgovt"] / m["pop"]), "y": np.log(m["realgdp"] / m["pop"]), "c": np.log(m["realcons"] / m["pop"])})
# TODO: VAR(d).fit(4, trend="ct"); irf(12).orth_irfs; multipliers = response_y / response_g * (Y/G)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.api import VAR
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"g": np.log(m["realgovt"] / m["pop"]), "y": np.log(m["realgdp"] / m["pop"]), "c": np.log(m["realcons"] / m["pop"])})
res = VAR(d).fit(4, trend="ct"); ir = res.irf(12).orth_irfs[:, :, 0]          # responses to the spending shock
YG, CG = (m["realgdp"] / m["realgovt"]).mean(), (m["realcons"] / m["realgovt"]).mean()
dG, dY, dC = ir[:, 0], ir[:, 1] * YG, ir[:, 2] * CG                             # in units of G
print(f"average Y/G = {YG:.2f}, C/G = {CG:.2f}")
for h in (0, 1, 2, 4, 8):
    print(f"  h = {h}: dG {dG[h]:.4f}, dY/dG(0) = {dY[h]/dG[0]:.2f}, dC/dG(0) = {dC[h]/dG[0]:.2f}")
print(f"impact multiplier dY/dG = {dY[0]/dG[0]:.2f};  cumulative 8-quarter multiplier = {dY[:9].sum()/dG[:9].sum():.2f};  consumption: {dC[:9].sum()/dG[:9].sum():.2f}")
```

Output:

```text
average Y/G = 10.49, C/G = 6.97
  h = 0: dG 0.0193, dY/dG(0) = 0.49, dC/dG(0) = -0.17
  h = 1: dG 0.0199, dY/dG(0) = 0.02, dC/dG(0) = -0.25
  h = 2: dG 0.0210, dY/dG(0) = -0.13, dC/dG(0) = -0.40
  h = 4: dG 0.0253, dY/dG(0) = -0.57, dC/dG(0) = -0.50
  h = 8: dG 0.0254, dY/dG(0) = -0.76, dC/dG(0) = -0.58
impact multiplier dY/dG = 0.49;  cumulative 8-quarter multiplier = -0.31;  consumption: -0.37
```

On impact, output rises by about \$0.50 per dollar of spending, but the effect turns negative within a few quarters, and the
cumulative 8-quarter multiplier is slightly negative, with consumption falling. That is lower than Blanchard and Perotti's
(2002) estimates (around 1), which use tax data, quarterly fiscal series and a different trend specification. Ramey (2011)
argues that recursive VARs mismeasure the timing of spending shocks, because spending is anticipated (military buildups are
announced in advance), and that narrative shocks give multipliers between 0.6 and 1.2. Multiplier estimates are fragile.

</details>

## Further reading
Leeper (1991); Woodford, *Interest and Prices*; Blanchard, *Macroeconomics*.
