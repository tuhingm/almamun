title: Cross-Sectionally Augmented ARDL
short: CS-ARDL
when: Long-run relationship with common shocks
data: 46 U.S. states, annual cigarette sales, 1963–1992 (Baltagi and Levin)
lead: Use this guide when a long panel has a long-run relationship among I(0) and I(1) variables and the units share common shocks. Panel ARDL estimators that assume independent units are then biased. CS-ARDL (Chudik, Mohaddes, Pesaran and Raissi 2016) adds cross-sectional averages and their lags to each unit's ARDL regression, so the common factors are absorbed, and averages the unit-specific long-run effects.
---

## The question and the model

The question is the same as in the PMG guide: **what are the long-run price and income elasticities of cigarette demand?** This time the data are used as they are. Cigarette consumption in every state was driven by national forces: the 1964 Surgeon General's report, advertising bans, changing social norms, federal excise taxes. These common factors affect states to different degrees, and they are correlated with prices and incomes. A model that ignores them confuses the national anti-smoking trend with the effect of price.

For each state \(i\), CS-ARDL(1,1) estimates

\[
c_{it} = a_i + \lambda_i c_{i,t-1} + \beta_{0i}' x_{it} + \beta_{1i}' x_{i,t-1} + \sum_{l=0}^{p_T} \delta_{il}' \bar z_{t-l} + e_{it}
\]

where \(x = (p, y)\) are the log real price and log real income, and \(\bar z_t\) are the **cross-sectional averages** of \(c\) and \(x\) in year \(t\), with \(p_T\) lags. Each state's long-run effect is \(\theta_i = (\beta_{0i} + \beta_{1i}) / (1 - \lambda_i)\), and the estimate reported is the mean group average of the \(\theta_i\).

The neighbours' price used in the PMG guide is left out here. It moves almost one-for-one with the state's own real price, and once the cross-sectional averages absorb the national component, the two can no longer be separated in 29 annual observations per state: with both included, the state-level long-run estimates become erratic. With CS-ARDL, keep the model small.

<div class="read" markdown="1">
**Why the cross-sectional averages work**

If the errors contain unobserved common factors \(f_t\) with state-specific loadings, the cross-sectional averages of the observed variables are (as N grows) a linear combination of those factors. Adding them, and their lags, to each state's regression with state-specific coefficients controls for the factors without having to name or measure them. Because every state gets its own coefficients on the averages, states can respond to the common shocks to different degrees, which year dummies cannot do.
</div>

<div class="write" markdown="1">
**How to write it: model section**

We estimate state-specific ARDL(1,1) models of cigarette demand augmented with the contemporaneous and lagged cross-sectional averages of all variables, following Chudik et al. (2016). The averages proxy unobserved common factors, such as national anti-smoking campaigns and federal tax changes, to which states may respond heterogeneously. Long-run elasticities are computed for each state and averaged across states (mean group).
</div>

## Load the data and run the pre-tests

```python
import pandas as pd
import numpy as np
from panel_tests import cd_test, cips_test, slope_test
from panel_estimators import cs_ardl, ardl_ec

raw = pd.read_csv("cigar_raw.csv").sort_values(["state", "year"])
df = pd.DataFrame({"state": raw["state"], "year": raw["year"], "log_sales": np.log(raw["sales"]),
                   "log_price": np.log(raw["price"] / raw["cpi"]), "log_income": np.log(raw["ndi"] / raw["cpi"]),
                   "log_pimin": np.log(raw["pimin"] / raw["cpi"])})
V = ["log_sales", "log_price", "log_income"]
x = V[1:]
print("Correlation of own and neighbours' real price:", round(df["log_price"].corr(df["log_pimin"]), 3))
print(f"{df['state'].nunique()} states x {df['year'].nunique()} years\n")
print(f"{'':12s}{'CD':>8s}{'mean |rho|':>12s}{'CIPS level':>12s}{'p':>8s}{'CIPS diff':>11s}{'p':>8s}")
for v in V:
    W = df.pivot(index="year", columns="state", values=v)
    cd, p, r = cd_test(W); a, b = cips_test(W), cips_test(W.diff())
    print(f"{v:12s}{cd:8.1f}{r:12.2f}{a['stat']:12.2f}{a['p']:8.3f}{b['stat']:11.2f}{b['p']:8.3f}")
s = slope_test(df, "state", "log_sales", x)
print(f"\nPesaran-Yamagata adjusted delta = {s['delta_adj']:.2f}   p-value = {s['delta_adj_p']:.4f}")
```

<div class="read" markdown="1">
**How to read it**

- **Cross-sectional dependence** is overwhelming: CD statistics above 100 and average absolute correlations between states of 0.7 to 0.9. The states are driven by national forces.
- **Unit roots (CIPS).** Sales and income are I(1); the real price rejects a unit root in levels and is I(0). A mix of I(0) and I(1), none I(2): the ARDL family applies.
- **Slopes** are heterogeneous.

Mixed integration orders and strong cross-sectional dependence: this is the combination for which the model selection tool recommends CS-ARDL. It is the result the tool gives for this example dataset.
</div>

## What goes wrong if the common factors are ignored

```python
pmg_raw = ardl_ec(df, "state", "year", "log_sales", x, method="pmg")
mg_raw = ardl_ec(df, "state", "year", "log_sales", x, method="mg")
U = {}
for st, g in df.groupby("state"):
    ec = g["log_sales"].shift(1) - g[x].shift(1) @ pmg_raw["theta"]
    Z = np.column_stack([np.ones(len(g)), ec, g[x].diff()])[1:]
    dy = g["log_sales"].diff().to_numpy()[1:]
    bb, *_ = np.linalg.lstsq(Z, dy, rcond=None)
    U[st] = pd.Series(dy - Z @ bb, index=g["year"].to_numpy()[1:])
cd_pmg = cd_test(pd.DataFrame(U))
print("Long run, ignoring common factors:")
for name, r in (("PMG", pmg_raw), ("MG", mg_raw)):
    print(f"  {name:4s}" + "".join(f"{v} = {r['long_run'].loc[v, 'coef']:6.3f} ({r['long_run'].loc[v, 'std err']:.3f})   " for v in x))
print(f"CD test on the PMG residuals: CD = {cd_pmg[0]:.1f}   p-value = {cd_pmg[1]:.4f}")
```

<div class="read" markdown="1">
**How to read it**

PMG and MG on the raw data assume independent states. The CD test on the PMG residuals shows that the assumption fails badly: the residuals are strongly correlated across states. With common factors in the errors that are also correlated with the regressors, these long-run estimates are biased. The clearest symptom is income: real incomes rose everywhere while smoking fell everywhere for reasons unrelated to income (health information, social norms), so the national trend cancels the income effect in PMG and turns it negative in MG. The PMG guide deals with this by demeaning; CS-ARDL deals with it inside the model.
</div>

## Estimate CS-ARDL

```python
cs = cs_ardl(df, "state", "year", "log_sales", x, pt=1)
print(f"CS-ARDL(1,1) with {cs['pt']} lag(s) of the cross-sectional averages, {cs['n']} states")
print("\nLong-run coefficients (mean group):")
print(cs["long_run"].round(3))
e = cs["phi"].loc["ec"]
print(f"\nError-correction coefficient (mean of lambda_i - 1): {e['coef']:.3f} ({e['std err']:.3f})")
print(f"Half-life of a deviation: {np.log(.5) / np.log(1 + e['coef']):.1f} years")
```

<div class="read" markdown="1">
**How to read it**

- **Price.** Once the common factors are absorbed, the long-run price elasticity is about −0.9: a 10% rise in the real price lowers sales by about 9% in the long run. The estimate is close to the PMG estimate on demeaned data (−0.77), which used a different way of dealing with the same common shocks. **Two methods agreeing is strong evidence.**
- **Income.** Positive and significant, about 0.6: cigarettes are a normal good once the national anti-smoking trend is no longer confounded with rising incomes.
- **Adjustment.** The error-correction coefficient is negative and significant: each state's consumption returns towards its long-run relationship. Adjustment is faster than in PMG on demeaned data, because the averages also absorb slow-moving national trends.
- **Standard errors** are mean-group standard errors, from the dispersion of the 46 state estimates. They are valid whatever the heterogeneity across states.
</div>

## Test 1: are the common factors removed?

```python
zb = df.groupby("year")[V].mean()
lagged = pd.concat({f"{c}_l{l}": zb[c].shift(l) for l in range(2) for c in V}, axis=1)
d = df.join(lagged, on="year")
R = {}
for st, g in d.groupby("state"):
    Y, X = g["log_sales"].to_numpy(), g[x].to_numpy()
    Z = np.column_stack([np.ones(len(Y) - 1), Y[:-1], X[1:], X[:-1], g[lagged.columns].to_numpy()[1:]])
    ok = np.isfinite(Z).all(1)
    bb, *_ = np.linalg.lstsq(Z[ok], Y[1:][ok], rcond=None)
    R[st] = pd.Series(Y[1:][ok] - Z[ok] @ bb, index=g["year"].to_numpy()[1:][ok])
cd, p, r = cd_test(pd.DataFrame(R))
print(f"CD test on the CS-ARDL residuals: CD = {cd:.2f}   p-value = {p:.3f}   mean |rho| = {r:.3f}")
print(f"(PMG on the raw data: CD = {cd_pmg[0]:.1f})")
```

<div class="read" markdown="1">
**How to read it**

The CD statistic falls from the very large value of the raw PMG residuals to a small one: **the cross-sectional averages have absorbed the common factors.** Report this test; it is the evidence that the augmentation did its job. A remaining small, insignificant or weakly significant CD statistic is normal, since the averages approximate the factors only up to an error that shrinks with N.
</div>

## Robustness: the number of lags of the averages

```python
print(f"{'lags p_T':>9s}" + "".join(f"{v:>20s}" for v in x) + f"{'phi':>10s}")
for pt in (0, 1, 2):
    r = cs_ardl(df, "state", "year", "log_sales", x, pt=pt)
    t = r["long_run"]
    print(f"{pt:9d}" + "".join(f"{t.loc[v, 'coef']:11.3f} ({t.loc[v, 'std err']:.3f})" for v in x) + f"{r['phi'].loc['ec', 'coef']:10.3f}")
k = 2 + 2 * len(x)
print(f"\nRegressors per state with p_T = 2: {k + len(V) * 3}, observations per state: {df['year'].nunique() - 1}")
```

<div class="read" markdown="1">
**How to read it**

Chudik et al. recommend about \(T^{1/3}\) lags of the averages, which is 3 here. But each state regression has only 29 observations, and every lag adds four regressors, so with this many variables the number of lags must stay small. Show that the results do not depend on the choice: here the long-run price elasticity stays between −0.73 and −0.88 and the income elasticity between 0.51 and 0.61. The results are robust to the lag length.
</div>

## Build the results table for the paper

```python
from scipy import stats
def star(b, se):
    p = 2 * stats.norm.sf(abs(b / se))
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

cols = {"(1) PMG raw": (pmg_raw["long_run"], pmg_raw["phi"]), "(2) MG raw": (mg_raw["long_run"], mg_raw["phi"]), "(3) CS-ARDL": (cs["long_run"], cs["phi"])}
print(f"{'Long run':12s}" + "".join(f"{c:>16s}" for c in cols))
for v in x + ["ec"]:
    get = lambda lr, ph: (ph if v == "ec" else lr).loc[v, ["coef", "std err"]]
    print(f"{v:12s}" + "".join(f"{get(*c)['coef']:13.3f}{star(*get(*c)):3s}" for c in cols.values()))
    print(f"{'':12s}" + "".join(f"{'(' + format(get(*c)['std err'], '.3f') + ')':>13s}   " for c in cols.values()))
print(f"{'Residual CD':12s}{cd_pmg[0]:13.1f}   {'':16s}{cd:13.2f}")
print("Columns (1)-(2) ignore common factors; column (3) includes cross-sectional averages (1 lag). *** p<0.01, ** p<0.05, * p<0.10.")
```

<div class="read" markdown="1">
**How to read it**

Put the estimators that ignore the common factors next to CS-ARDL, with the residual CD test in the table, so the reader sees both the bias and its correction. Column (3) is the preferred estimate. Cite the PMG-on-demeaned-data estimate as a second robustness check.
</div>

## Write the paper

<div class="write" markdown="1">
**How to write it: pre-tests and estimator**

All variables exhibit strong cross-sectional dependence (Pesaran CD tests), reflecting national anti-smoking trends and federal policy. CIPS tests indicate a mixture of I(0) and I(1) variables, and the Pesaran–Yamagata test rejects slope homogeneity. Because estimators that assume cross-sectional independence would be biased, and because states may respond differently to common shocks, we estimate the long-run elasticities with the cross-sectionally augmented ARDL (CS-ARDL) approach of Chudik et al. (2016).
</div>

<div class="write" markdown="1">
**How to write it: results**

The mean group long-run price elasticity is about −0.9 and the income elasticity about 0.6 (Table 6, column 3). The price elasticity is close to the estimate from a pooled mean group model on cross-sectionally demeaned data, and both are robust to the number of lags of the cross-sectional averages. The error-correction coefficient is negative and significant, confirming adjustment towards a long-run relationship. The CD test on the CS-ARDL residuals indicates that the cross-sectional averages successfully absorb the common factors, in contrast with PMG and MG estimates on the raw data, whose residuals remain strongly dependent.
</div>

<div class="warn" markdown="1">
**Common mistakes**

- Adding too many lags of the averages relative to T. Each lag costs one regressor per variable in every unit regression.
- Using CS-ARDL with small N. The averages approximate the common factors only as N grows; with fewer than about 15 units they are poor proxies.
- Reporting the long run without the residual CD test, which is the evidence that the method worked.
- Forgetting that the mean group long run is an average of unit-specific ratios: a few units with \(\lambda_i\) close to 1 produce extreme \(\theta_i\). Check the distribution, and report the median as well.
</div>
