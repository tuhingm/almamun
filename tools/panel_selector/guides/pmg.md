title: Panel ARDL: Pooled Mean Group
short: PMG
when: Long-run relationship, cross-sectionally independent units
data: 46 U.S. states, annual cigarette sales, 1963–1992 (Baltagi and Levin)
lead: Use this guide when a long panel has a long-run relationship among variables that are a mix of I(0) and I(1), or I(1) and cointegrated, and the units are not cross-sectionally dependent. The pooled mean group estimator imposes the same long-run elasticities on all units while letting short-run dynamics and the speed of adjustment differ. The guide estimates PMG, MG and dynamic fixed effects and uses the Hausman test to choose among them.
---

## The question and the model

The question: **what are the long-run price and income elasticities of cigarette demand?** Smoking is addictive, so consumption adjusts slowly to price changes, and the long-run response is what matters for tax policy. The data also contain the minimum price in neighbouring states, which captures cross-border buying (“bootlegging”).

The panel ARDL(1,1) model in error-correction form is

\[
\Delta c_{it} = \phi_i \left( c_{i,t-1} - \theta_1 p_{i,t-1} - \theta_2 y_{i,t-1} - \theta_3 pn_{i,t-1} \right) + \gamma_{i1} \Delta p_{it} + \gamma_{i2} \Delta y_{it} + \gamma_{i3} \Delta pn_{it} + \mu_i + \varepsilon_{it}
\]

| Symbol | Variable | Definition | Expected sign of \(\theta\) |
|---|---|---|---|
| \(c_{it}\) | `log_sales` | Log packs per capita | dependent variable |
| \(p_{it}\) | `log_price` | Log real price per pack | negative: the long-run price elasticity |
| \(y_{it}\) | `log_income` | Log real disposable income per capita | positive if cigarettes are a normal good |
| \(pn_{it}\) | `log_pimin` | Log real minimum price in neighbouring states | positive: cheaper neighbours draw buyers away |
| \(\phi_i\) | — | Speed of adjustment to the long run | negative, between −2 and 0 |

**PMG** restricts the long-run coefficients \(\theta\) to be the same in every state, and lets \(\phi_i\), the short-run coefficients \(\gamma_i\) and the intercepts differ. **MG** lets everything differ and averages. **Dynamic fixed effects (DFE)** restricts everything except the intercepts.

<div class="read" markdown="1">
**Why PMG suits this question**

Long-run elasticities are often similar across units, because they reflect preferences and technology, whereas short-run adjustment depends on local institutions (here, state tax changes and their timing). PMG imposes the restriction where it is plausible and estimates it efficiently. Whether the restriction holds is tested by comparing PMG with MG.
</div>

<div class="write" markdown="1">
**How to write it: model section**

We estimate an ARDL(1,1) model of cigarette demand in error-correction form, in which the long-run elasticities of per-capita sales with respect to the real price, real income and the real minimum price in neighbouring states are common across states, while the short-run coefficients and the speed of adjustment are state-specific. The model is estimated by the pooled mean group (PMG) estimator of Pesaran, Shin and Smith (1999).
</div>

## Load the data and remove the common shocks

```python
import pandas as pd
import numpy as np
from panel_tests import cd_test, cips_test, fisher_adf_test, slope_test
from panel_estimators import ardl_ec, hausman      # examples/panel_estimators.py

raw = pd.read_csv("cigar_raw.csv").sort_values(["state", "year"])
df = pd.DataFrame({"state": raw["state"], "year": raw["year"], "log_sales": np.log(raw["sales"]),
                   "log_price": np.log(raw["price"] / raw["cpi"]), "log_income": np.log(raw["ndi"] / raw["cpi"]),
                   "log_pimin": np.log(raw["pimin"] / raw["cpi"])})
V = ["log_sales", "log_price", "log_income", "log_pimin"]
dm = df.copy()
for v in V:
    dm[v] = df[v] - df.groupby("year")[v].transform("mean")     # subtract the cross-state mean of each year

print(f"{df['state'].nunique()} states, {df['year'].min()}-{df['year'].max()}, {len(df)} observations\n")
print(f"{'':12s}{'CD raw':>9s}{'CD demeaned':>13s}{'p':>8s}")
for v in V:
    c0 = cd_test(df.pivot(index="year", columns="state", values=v))
    c1 = cd_test(dm.pivot(index="year", columns="state", values=v))
    print(f"{v:12s}{c0[0]:9.1f}{c1[0]:13.2f}{c1[1]:8.3f}")
```

<div class="read" markdown="1">
**How to read it**

In the raw data every variable is strongly cross-sectionally dependent: national trends (the 1964 Surgeon General's report, federal taxes, inflation) move all states together. **Cross-sectional demeaning**, subtracting each year's average across states, removes common shocks that affect all states equally. It is the standard first step before PMG, which assumes independent units.

Demeaning removes almost all of the dependence: the CD statistics fall from above 100 to about −3. What is left is small and **negative**, partly a mechanical by-product of demeaning (subtracting the cross-state average makes deviations slightly negatively correlated). Demeaning cannot remove common shocks to which states respond differently; for those, use CS-ARDL (see that guide, which uses the same data without demeaning). The test that matters is the CD test on the PMG residuals, below.
</div>

## Unit roots and slope heterogeneity

```python
print(f"{'':12s}{'CIPS level':>11s}{'p':>8s}{'Fisher level':>14s}{'p':>8s}{'CIPS diff':>11s}{'p':>8s}")
for v in V:
    W = dm.pivot(index="year", columns="state", values=v)
    a, f, b = cips_test(W), fisher_adf_test(W), cips_test(W.diff())
    print(f"{v:12s}{a['stat']:11.2f}{a['p']:8.3f}{f['Z']:14.2f}{f['Z_p']:8.3f}{b['stat']:11.2f}{b['p']:8.3f}")
s = slope_test(dm, "state", "log_sales", V[1:])
print(f"\nPesaran-Yamagata adjusted delta = {s['delta_adj']:.2f}   p-value = {s['delta_adj_p']:.4f}")
```

<div class="read" markdown="1">
**How to read it**

- **Unit roots.** The variables are a mix: some do not reject a unit root in levels, others do. Every variable is stationary in first differences, and none is I(2). **This mix of I(0) and I(1) is the case the panel ARDL is designed for**: the bounds-testing logic of ARDL does not require all variables to share the same order of integration.
- **Slope heterogeneity.** The static Δ test rejects. That does not rule PMG out: PMG lets the short-run coefficients differ, and only restricts the long run. Whether the long-run restriction is acceptable is tested directly in the Hausman step.
</div>

## Estimate PMG, MG and dynamic fixed effects

```python
x = V[1:]
pmg = ardl_ec(dm, "state", "year", "log_sales", x, method="pmg")
mg = ardl_ec(dm, "state", "year", "log_sales", x, method="mg")
dfe = ardl_ec(dm, "state", "year", "log_sales", x, method="dfe")
print(f"PMG converged in {pmg['iterations']} iterations\n")
print(f"{'Long run':18s}" + "".join(f"{v:>20s}" for v in x))
for name, r in (("PMG", pmg), ("MG", mg), ("DFE", dfe)):
    t = r["long_run"]
    print(f"{name:18s}" + "".join(f"{t.loc[v, 'coef']:11.3f} ({t.loc[v, 'std err']:.3f})" for v in x))
print()
for name, r in (("PMG", pmg), ("MG", mg), ("DFE", dfe)):
    e = r["phi"].loc["ec"]
    print(f"{name:4s} error-correction coefficient phi = {e['coef']:.3f} ({e['std err']:.3f})   half-life = {np.log(.5) / np.log(1 + e['coef']):.1f} years")
```

<div class="read" markdown="1">
**How to read it**

- **PMG long run.** A 10% rise in the real price lowers cigarette sales by about 7.7% in the long run. Income has a positive long-run effect, and a higher price in neighbouring states raises sales at home: when the neighbours are cheap, smokers cross the border. All three are precisely estimated, because PMG pools the long-run information from all 46 states.
- **MG** estimates each state separately. Its price elasticity is larger in absolute value and its income and neighbour-price effects are imprecise: with 29 years per state, individual long-run coefficients are noisy, and averaging noisy ratios inflates the standard errors.
- **DFE** pools everything, including the adjustment speed, which biases the long run when dynamics differ across states.
- **Speed of adjustment.** \(\phi\) is negative and significant, which supports the long-run relationship: deviations from it are corrected. With PMG, about a fifth of the gap closes each year, a half-life of about three years. Smoking adjusts slowly, as addiction models predict.
</div>

## Test 1: is the common long run acceptable? Hausman PMG vs MG

```python
H, dfree, p = hausman(pmg["theta"], pmg["V"], mg["theta"], mg["V"])
print(f"Hausman test, PMG vs MG:  chi2({dfree}) = {H:.2f}   p-value = {p:.3f}")
```

<div class="read" markdown="1">
**How to read it**

Under the null that the long-run coefficients are the same in every state, both PMG and MG are consistent and PMG is efficient. Under the alternative, only MG is consistent. The test does not reject, so **the common long-run restriction is accepted, and PMG is the preferred estimator.** This is the test referees expect in every PMG paper.
</div>

## Test 2: residual cross-sectional dependence and the adjustment speeds

```python
U = {}
for st, g in dm.sort_values(["state", "year"]).groupby("state"):
    ec = g["log_sales"].shift(1) - g[x].shift(1) @ pmg["theta"]
    Z = np.column_stack([np.ones(len(g)), ec, g[x].diff()])[1:]
    dy = g["log_sales"].diff().to_numpy()[1:]
    b, *_ = np.linalg.lstsq(Z, dy, rcond=None)
    U[st] = pd.Series(dy - Z @ b, index=g["year"].to_numpy()[1:])
cd_r, p_r, r_r = cd_test(pd.DataFrame(U))
print(f"CD test on PMG residuals: CD = {cd_r:.2f}   p-value = {p_r:.3f}   mean |rho| = {r_r:.3f}")
print(f"Correlation induced by demeaning alone: -1/(N-1) = {-1 / (len(U) - 1):.3f}")

phis = pmg["phi_units"]
print(f"State adjustment speeds: min {phis.min():.2f}, median {np.median(phis):.2f}, max {phis.max():.2f}; "
      f"{(phis < 0).sum()} of {len(phis)} negative, {(phis < -2).sum()} below -2")
```

<div class="read" markdown="1">
**How to read it**

- **Residual dependence.** PMG assumes the states' errors are independent. The CD test on the residuals rejects narrowly (p = 0.03), but the remaining dependence is weak and negative, of the kind demeaning itself induces, not the strong positive dependence of common shocks in the raw data (CD above 100). Report this honestly, and use CS-ARDL as the robustness check: it handles any remaining common factors.
- **Adjustment speeds.** Every state should have \(-2 < \phi_i < 0\) for a stable long-run relationship. None is below −2, but 7 of the 46 states have a small positive \(\phi_i\): for them, sales show no tendency to return to the long-run relationship within the sample. Report the range, and check that the results survive without those states.
</div>

## Build the results table for the paper

```python
from scipy import stats
def star(b, se):
    p = 2 * stats.norm.sf(abs(b / se))
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

cols = {"(1) PMG": pmg, "(2) MG": mg, "(3) DFE": dfe}
print(f"{'Long run':14s}" + "".join(f"{c:>16s}" for c in cols))
for v in x:
    print(f"{v:14s}" + "".join(f"{r['long_run'].loc[v, 'coef']:13.3f}{star(*r['long_run'].loc[v, ['coef', 'std err']]):3s}" for r in cols.values()))
    print(f"{'':14s}" + "".join(f"{'(' + format(r['long_run'].loc[v, 'std err'], '.3f') + ')':>13s}   " for r in cols.values()))
print(f"{'Adjustment':14s}" + "".join(f"{r['phi'].loc['ec', 'coef']:13.3f}{star(*r['phi'].loc['ec', ['coef', 'std err']]):3s}" for r in cols.values()))
print(f"{'':14s}" + "".join(f"{'(' + format(r['phi'].loc['ec', 'std err'], '.3f') + ')':>13s}   " for r in cols.values()))
print(f"Hausman PMG vs MG: chi2({dfree}) = {H:.2f} (p = {p:.3f}). States: {pmg['n']}. Data cross-sectionally demeaned. *** p<0.01, ** p<0.05, * p<0.10.")
```

<div class="read" markdown="1">
**How to read it**

The standard PMG table: long-run coefficients and the error-correction coefficient for PMG, with MG and DFE alongside, and the Hausman statistic in the notes. Report the short-run coefficients in an appendix table.
</div>

## Write the paper

<div class="write" markdown="1">
**How to write it: pre-tests and estimator**

To remove common national shocks, all variables are expressed as deviations from their cross-state means in each year. CIPS tests indicate a mixture of I(0) and I(1) variables, none of them I(2), which motivates a panel ARDL specification. We estimate it by the pooled mean group estimator, which constrains the long-run elasticities to be equal across states while allowing state-specific short-run dynamics. A Hausman test does not reject the long-run homogeneity restriction against the mean group estimator, and the remaining cross-sectional dependence in the PMG residuals is weak (CD = −2.1).
</div>

<div class="write" markdown="1">
**How to write it: results**

The long-run price elasticity of cigarette demand is about −0.77 (Table 5, column 1): a 10 percent increase in the real price reduces per-capita sales by 7.7 percent in the long run. A higher minimum price in neighbouring states raises home-state sales, consistent with cross-border purchases. The error-correction coefficient is negative and significant, implying that about one fifth of a deviation from long-run equilibrium is corrected each year.
</div>

<div class="warn" markdown="1">
**Common mistakes**

- Running PMG on cross-sectionally dependent data. Demean first, test the residuals, and switch to CS-ARDL if dependence remains.
- Skipping the Hausman test against MG. The common long run is a restriction and must be tested.
- Using PMG with an I(2) variable, or with short T. Each state's short-run dynamics are estimated separately and need enough years.
- Reporting only the long run. The error-correction coefficient is the evidence that a long-run relationship exists.
</div>
