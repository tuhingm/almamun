title: 2SLS and GMM
short: 2SLS / GMM
when: Endogenous regressors that need instruments
data: U.S. policy rule, quarterly, 1979–2007; UK firm employment panel, 1976–1984
lead: Use this guide when a regressor is correlated with the error term, so that OLS is biased. It covers choosing instruments, two-stage least squares, instrument strength, overidentification and endogeneity tests, efficient GMM and robustness for a time-series model, then difference GMM for a dynamic panel with its specification tests, with sample text for the paper.
---

## The question and why OLS fails

The question is: **how strongly does the Federal Reserve raise its interest rate when inflation rises?** The standard description of policy is a Taylor rule with interest-rate smoothing:

\[
i_t = c + \beta\,\pi_t + \gamma\,x_t + \rho\, i_{t-1} + \varepsilon_t
\]

where \(i_t\) is the federal funds rate, \(\pi_t\) is inflation over the past four quarters and \(x_t\) is the output gap. The long-run response to inflation is \(\beta/(1-\rho)\). The **Taylor principle** says it must exceed 1, so that the real interest rate rises when inflation rises.

OLS cannot estimate this equation consistently, for two reasons.

- **Simultaneity.** The interest rate affects inflation and output within the same quarter, so the regressors respond to the dependent variable.
- **Policy shocks.** The error term \(\varepsilon_t\) is the unexpected part of policy. Because a policy surprise moves inflation and output, the regressors are correlated with the error.

A regressor that is correlated with the error term is **endogenous**. OLS is then biased, and more data does not fix it. The solution is instrumental variables.

<div class="read" markdown="1">
**What an instrument must do**

An instrument \(z\) is a variable that satisfies two conditions.

1. **Relevance:** \(z\) is correlated with the endogenous regressor. This can be tested (Step 5).
2. **Exogeneity:** \(z\) is uncorrelated with the error term. This cannot be tested directly; it must be argued. With more instruments than endogenous regressors, it can be partly tested (Step 6).

Here the instruments are **lagged values** of inflation, the output gap and the interest rate. They are relevant because these series are persistent. They are plausibly exogenous because a policy surprise in quarter \(t\) cannot have influenced what happened in earlier quarters. This is the approach of Clarida, Galí and Gertler (2000).
</div>

<div class="write" markdown="1">
**How to write it: identification**

Estimating the policy rule by OLS would be inconsistent because inflation and the output gap respond contemporaneously to the interest rate and are therefore correlated with the policy shock \(\varepsilon_t\). Following Clarida, Galí and Gertler (2000), we treat current inflation and the output gap as endogenous and instrument them with their own first four lags and the second lag of the federal funds rate. These variables are predetermined with respect to the period-\(t\) policy shock and, given the persistence of inflation and output, are strongly correlated with the endogenous regressors.
</div>

## Load the data and build the instruments

This guide uses the `linearmodels` package for instrumental-variable estimation. Install it once with `pip install linearmodels`.

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS, IVGMM

data = pd.read_csv("taylor.csv").set_index("date")
for lag in range(1, 5):
    for col in ("fedfunds", "inflation", "output_gap"):
        data[f"{col}_l{lag}"] = data[col].shift(lag)
data["const"] = 1.0

df = data.loc["1979Q3":"2007Q4"].copy()        # Volcker and Greenspan years, up to the financial crisis
print(len(df), "quarterly observations")
print(df[["fedfunds", "inflation", "output_gap"]].describe().T[["mean", "std", "min", "max"]].round(2))

endog = ["inflation", "output_gap"]                                   # endogenous regressors
exog = ["const", "fedfunds_l1"]                                       # included exogenous regressors
instruments = [f"{c}_l{k}" for c in endog for k in (1, 2, 3, 4)] + ["fedfunds_l2"]
print(f"\n{len(endog)} endogenous regressors, {len(instruments)} instruments  ->  {len(instruments) - len(endog)} overidentifying restrictions")
```

<div class="read" markdown="1">
**How to read it**

There are 114 quarters. The model has 2 endogenous regressors and 9 instruments. It is **overidentified**: there are 7 more instruments than needed. Overidentification is useful, because the extra instruments make it possible to test instrument validity in Step 6. With exactly as many instruments as endogenous regressors the model is *just identified* and that test is not available.

The lagged interest rate is treated as exogenous because it was set before the current shock. That requires the errors to be serially uncorrelated, which is one reason the second lag is also added as an instrument.
</div>

## The OLS benchmark

Always estimate OLS first. It is the benchmark that shows how much the instruments change the answer.

```python
ols = sm.OLS(df["fedfunds"], df[exog + endog]).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
print(ols.summary().tables[1])
rho = ols.params["fedfunds_l1"]
print(f"\nLong-run response to inflation: {ols.params['inflation'] / (1 - rho):.2f}    to the output gap: {ols.params['output_gap'] / (1 - rho):.2f}")
```

## Two-stage least squares

2SLS works in two steps. **First stage:** regress each endogenous regressor on all the instruments and exogenous variables, and keep the fitted values. **Second stage:** regress the dependent variable on those fitted values. The fitted values contain only the variation that the instruments explain, which is unrelated to the error.

Never run the two stages by hand: the standard errors from a manual second stage are wrong. Use a routine that does both.

```python
iv = IV2SLS(df["fedfunds"], df[exog], df[endog], df[instruments]).fit(cov_type="kernel", kernel="bartlett", bandwidth=4)
print(iv.summary.tables[1])
rho = iv.params["fedfunds_l1"]
print(f"\nLong-run response to inflation: {iv.params['inflation'] / (1 - rho):.2f}    to the output gap: {iv.params['output_gap'] / (1 - rho):.2f}")
```

<div class="read" markdown="1">
**How to read it**

Compare with OLS. The coefficient on inflation is similar (0.28 against 0.32), but the coefficient on the output gap falls from 0.26 to 0.16. OLS overstated the response to the output gap, as expected if a strong economy and a tight policy stance move together within the quarter for reasons the rule does not capture.

The smoothing coefficient of 0.83 means the Fed moves the rate gradually: only about 17% of the desired adjustment happens in a quarter. This is why the long-run responses are much larger than the impact coefficients.

`cov_type="kernel"` requests standard errors that are robust to heteroskedasticity and serial correlation, the IV equivalent of Newey–West.
</div>

## Test 1: are the instruments strong?

Weak instruments, only loosely related to the endogenous regressors, are worse than no instruments: 2SLS becomes biased towards OLS and its standard errors are unreliable.

```python
first = iv.first_stage.diagnostics
print(first[["rsquared", "partial.rsquared", "f.stat", "f.pval"]].round(3))

for var in endog:                                   # the classical first-stage F statistic, from the first-stage regression itself
    full_ = sm.OLS(df[var], df[exog + instruments]).fit()
    restricted = sm.OLS(df[var], df[exog]).fit()
    f_stat = ((restricted.ssr - full_.ssr) / len(instruments)) / (full_.ssr / full_.df_resid)
    print(f"First-stage F for {var:11s} = {f_stat:7.1f}")
```

<div class="read" markdown="1">
**How to read it**

- **Partial R-squared** is the share of each endogenous regressor explained by the instruments after removing the exogenous variables. Here it is 0.96 for inflation and 0.83 for the output gap.
- **First-stage F** tests whether the instruments are jointly zero in the first stage. The rule of thumb of Staiger and Stock (1997) is that an F below 10 signals weak instruments. Both statistics are far above 10.

These instruments are very strong, which is no surprise: four-quarter inflation and the output gap are highly persistent, so their own lags predict them well. In cross-section work you will rarely see values like these; an F of 15 to 30 is typical of a good instrument there. With several endogenous regressors, also report a joint statistic such as Cragg–Donald or Kleibergen–Paap.
</div>

## Test 2: are the instruments valid? The overidentification test

```python
print(iv.sargan)
```

<div class="read" markdown="1">
**How to read it**

The Sargan test (called the Hansen J test in GMM) has the null hypothesis that **all instruments are valid**, meaning uncorrelated with the error. A **large** p-value is what you want. Here p = 0.35, so the null is not rejected: the data give no evidence against the instruments.

Three cautions. A rejection tells you that at least one instrument is invalid but not which one. A non-rejection is not proof of validity: the test assumes that at least enough instruments are valid to identify the model, and it has low power when all instruments share the same flaw. And with a very large number of instruments the test almost never rejects. Keep the instrument set small and justify each instrument in words.
</div>

## Test 3: is the regressor really endogenous?

```python
print(iv.wu_hausman())
print()
print(iv.durbin())
```

<div class="read" markdown="1">
**How to read it**

The Wu–Hausman and Durbin tests have the null that the regressors are **exogenous**, in which case OLS and 2SLS estimate the same thing and OLS, being more precise, should be preferred. Both reject strongly (p below 0.001). Inflation and the output gap are endogenous, and the IV estimates are needed.

If this test does not reject, report OLS as the main result and 2SLS as a robustness check. Instrumenting a regressor that is not endogenous only makes the estimates noisier.
</div>

## Efficient GMM

The generalised method of moments (GMM) uses the same instruments but weights them optimally. With heteroskedastic or serially correlated errors, 2SLS is no longer the most efficient estimator; two-step GMM is.

```python
gmm = IVGMM(df["fedfunds"], df[exog], df[endog], df[instruments], weight_type="kernel", kernel="bartlett", bandwidth=4).fit(iter_limit=2)
print(gmm.summary.tables[1])
print(gmm.j_stat)

b, V = gmm.params, gmm.cov
rho = b["fedfunds_l1"]
print(f"\n{'':20s}{'long-run response':>18s}{'std. error':>12s}")
for var in endog:
    lr = b[var] / (1 - rho)
    grad = pd.Series(0.0, index=b.index); grad[var] = 1 / (1 - rho); grad["fedfunds_l1"] = b[var] / (1 - rho) ** 2
    se = float(np.sqrt(grad @ V @ grad))
    print(f"{var:20s}{lr:18.2f}{se:12.2f}")
    if var == "inflation":
        from scipy import stats
        lr_pi, se_pi = lr, se
print(f"\nTaylor principle, H0: long-run inflation response <= 1:  t = {(lr_pi - 1) / se_pi:.2f}   one-sided p-value = {1 - stats.norm.cdf((lr_pi - 1) / se_pi):.3f}")
```

<div class="read" markdown="1">
**How to read it**

- **Coefficients.** GMM and 2SLS give similar estimates, as they should when the instruments are valid; large differences between them would be a warning. The GMM standard error on inflation is smaller (0.079 against 0.109), which is the efficiency gain.
- **Hansen J test.** p = 0.34, so the overidentifying restrictions are not rejected.
- **Long-run responses.** The long-run response to inflation is 1.79: in the long run the Fed raised the funds rate by about 1.8 points for each point of inflation. Its standard error, 0.49, comes from the delta method, because the long-run response is a nonlinear function of two coefficients.
- **The Taylor principle.** The estimate is above 1, but the one-sided test of "at most 1" has a p-value of 0.054. The point estimate supports the Taylor principle; the statistical evidence is borderline at the 5% level.

Report the test even when it is borderline. Writing that the Taylor principle "is satisfied" on the strength of a point estimate alone would overstate the result.
</div>

## Robustness checks

```python
def estimate(sample, lags_used, label):
    d_ = data.loc[sample[0]:sample[1]]
    inst = [f"{c}_l{k}" for c in endog for k in lags_used] + ["fedfunds_l2"]
    m = IVGMM(d_["fedfunds"], d_[exog], d_[endog], d_[inst], weight_type="kernel", kernel="bartlett", bandwidth=4).fit(iter_limit=2)
    r = m.params["fedfunds_l1"]
    print(f"{label:34s}{m.params['inflation'] / (1 - r):9.2f}{m.params['output_gap'] / (1 - r):9.2f}{r:9.2f}{m.j_stat.pval:10.3f}{int(m.nobs):6d}")

print(f"{'':34s}{'LR infl.':>9s}{'LR gap':>9s}{'rho':>9s}{'J p-val':>10s}{'N':>6s}")
estimate(("1979Q3", "2007Q4"), (1, 2, 3, 4), "Baseline: 1979-2007, 4 lags")
estimate(("1979Q3", "2007Q4"), (1, 2), "Fewer instruments: 2 lags")
estimate(("1979Q3", "2007Q4"), (2, 3, 4), "Instruments from lag 2 only")
estimate(("1983Q1", "2007Q4"), (1, 2, 3, 4), "After the disinflation: 1983-2007")
estimate(("1987Q3", "2007Q4"), (1, 2, 3, 4), "Greenspan: 1987-2007")
estimate(("1961Q1", "1979Q2"), (1, 2, 3, 4), "Before Volcker: 1961-1979")
```

<div class="read" markdown="1">
**How to read it**

- **Instrument set.** With two lags, or with instruments dated from lag 2 only, the long-run inflation response stays between 1.6 and 1.8 and the J test never rejects. The result does not depend on the particular instruments.
- **Before Volcker.** For 1961–1979 the long-run response is 0.52, well below 1. Policy did not satisfy the Taylor principle in the 1970s. This contrast with the later period is the central finding of Clarida, Galí and Gertler (2000), and the data reproduce it.
- **Within the later period the result is fragile.** Dropping the 1979–82 disinflation lowers the response to 0.99, and for the Greenspan years alone it is 0.35. The strong response in the baseline comes largely from the Volcker disinflation.

Why would the estimate collapse after 1983? Inflation was low and stable, so there is little variation to identify the response from, and the smoothing coefficient above 0.9 makes the long-run ratio very sensitive. The measurement of the variables matters too: studies that use real-time data and forecasts, rather than the revised data and HP-filtered gap used here, find stronger responses in this period. A careful paper reports this sensitivity and does not claim more than the 1979–2007 result supports.
</div>

## Part 2. Dynamic panels: why OLS and fixed effects both fail

The second half of this guide applies the same idea to **panel data with a lagged dependent variable**, the setting of growth regressions across countries and of investment, leverage or employment equations across firms:

\[
y_{it} = \alpha\, y_{i,t-1} + \beta' x_{it} + \eta_i + \lambda_t + \varepsilon_{it}
\]

Here \(\eta_i\) is a fixed effect for each unit and \(\lambda_t\) a year effect. The lagged dependent variable is endogenous by construction, and the two usual estimators are biased in opposite directions.

| Estimator | Problem | Direction of bias in \(\alpha\) |
|---|---|---|
| Pooled OLS | \(y_{i,t-1}\) contains \(\eta_i\), which is in the error | **upwards** |
| Fixed effects | demeaning makes \(y_{i,t-1}\) correlated with the demeaned error when T is small (Nickell bias) | **downwards** |
| Difference GMM (Arellano–Bond) | first-difference to remove \(\eta_i\), then instrument \(\Delta y_{i,t-1}\) with levels dated \(t-2\) and earlier | consistent if the instruments are valid and strong |

The example is the one the method was developed on: employment in 140 UK manufacturing firms, 1976–1984, from Arellano and Bond (1991). The question is **how persistent is firm employment, and how does it respond to wages and capital?**

```python
from diff_gmm import diff_gmm          # helper file: research-guide/examples/diff_gmm.py
import statsmodels.formula.api as smf

raw = pd.read_csv("employment.csv")
panel = pd.DataFrame({"firm": raw["firm"], "year": raw["year"], "n": np.log(raw["emp"]), "w": np.log(raw["wage"]),
                      "k": np.log(raw["capital"]), "ys": np.log(raw["output"])})
by_firm = panel.groupby("firm")
for var, lag in (("n", 1), ("n", 2), ("w", 1), ("k", 1), ("k", 2), ("ys", 1), ("ys", 2)):
    panel[f"{var}_l{lag}"] = by_firm[var].shift(lag)

print(f"{panel['firm'].nunique()} firms, {panel['year'].min()}-{panel['year'].max()}, {len(panel)} observations")
print("Years per firm:", by_firm["year"].count().value_counts().sort_index().to_dict())
```

<div class="read" markdown="1">
**How to read it**

This is the typical shape for dynamic panel GMM: **many units (140) and few periods (7 to 9)**. The method relies on the number of units being large. With a long panel of few units, such as 20 countries over 40 years, fixed effects is the better choice and the Nickell bias is small.

The variables are in logs: employment `n`, the real wage `w`, capital `k` and industry output `ys`. The model has two lags of employment, current and lagged wages, and current and two lags of capital and output, as in the original paper. [Download the helper file](examples/diff_gmm.py) and the [data](examples/employment.csv) into your working folder.
</div>

## Panel step 1: pooled OLS and fixed effects give the bounds

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

The coefficient on lagged employment is 1.05 by pooled OLS and 0.73 by fixed effects. Because the first is biased upwards and the second downwards, **a credible estimate should lie between them**. This bounds check, suggested by Bond (2002), costs two regressions and is the first thing to run in any dynamic panel study.
</div>

## Panel step 2: difference GMM

```python
gmm_panel = diff_gmm(panel, unit="firm", time="year", y="n", lags_y=2, x=controls, gmm_lags=(2, None))
print(gmm_panel["table"].loc[["L1.n", "L2.n"] + controls].round(4))
print(f"\nObservations: {gmm_panel['nobs']}   firms: {gmm_panel['units']}   instruments: {gmm_panel['instruments']}")

t = gmm_panel["table"]["coef"]
persistence = 1 - t["L1.n"] - t["L2.n"]
print(f"Long-run wage elasticity:    {(t['w'] + t['w_l1']) / persistence:.2f}")
print(f"Long-run capital elasticity: {(t['k'] + t['k_l1'] + t['k_l2']) / persistence:.2f}")
```

<div class="read" markdown="1">
**How to read it**

These are one-step estimates with standard errors robust to heteroskedasticity and to correlation within each firm. They reproduce the published results for this dataset exactly (a coefficient of 0.686 on lagged employment), which is how the helper function was checked.

- **Wages.** A 1% rise in the real wage lowers employment by 0.61% in the same year. Allowing for the dynamics, the long-run elasticity is about −0.54.
- **Capital.** A 1% rise in the capital stock raises employment by 0.36% at once, and by about 0.70% in the long run.
- **Persistence.** The coefficient on lagged employment is 0.69. Compare it with the bounds from the previous step: it lies **below** the fixed-effects estimate of 0.73, not between the two. Keep this in mind for the next two steps.

`gmm_lags=(2, None)` uses all available lags from \(t-2\) back as instruments, which gives 41 instruments for 140 firms.
</div>

## Panel step 3: the three specification tests

```python
z1, p1 = gmm_panel["ar1"]; z2, p2 = gmm_panel["ar2"]; j, dof, pj = gmm_panel["hansen"]
print(f"Arellano-Bond AR(1) test:  z = {z1:6.2f}   p-value = {p1:.3f}")
print(f"Arellano-Bond AR(2) test:  z = {z2:6.2f}   p-value = {p2:.3f}")
print(f"Hansen J test:             chi2({dof}) = {j:.2f}   p-value = {pj:.3f}")
print(f"Instruments: {gmm_panel['instruments']}    firms: {gmm_panel['units']}")
```

<div class="read" markdown="1">
**How to read it**

A dynamic panel GMM table is not complete without these.

- **AR(1)** is significant, which is expected: first-differencing creates first-order correlation by construction. It tells you nothing about validity.
- **AR(2) is the test that matters.** Its null is no second-order serial correlation in the differenced residuals. If it is rejected, the errors in levels are serially correlated and levels dated \(t-2\) are not valid instruments. Here p = 0.61, so the instruments pass. If it had rejected, you would start the instruments at \(t-3\).
- **Hansen J** has the null that the instruments are valid as a group. Here p = 0.18, which does not reject. Be suspicious of p-values very close to 1: that is a sign of too many instruments, not of very good ones.
- **Instrument count.** 41 instruments for 140 firms. The count must stay well below the number of units, because too many instruments overfit the endogenous variable and weaken the Hansen test.
</div>

## Panel step 4: robustness to the instrument set

```python
specs = {"All lags (baseline)": {}, "Lags 2 to 4": {"gmm_lags": (2, 4)}, "Lags 2 to 3": {"gmm_lags": (2, 3)},
         "Lags 3 and deeper": {"gmm_lags": (3, None)}, "Collapsed instruments": {"collapse": True}}
print(f"{'':24s}{'L1.n':>7s}{'(s.e.)':>9s}{'w':>8s}{'k':>8s}{'instr.':>8s}{'AR(2) p':>9s}{'Hansen p':>10s}")
for name, options in specs.items():
    r = diff_gmm(panel, unit="firm", time="year", y="n", lags_y=2, x=controls, **options)
    c, se = r["table"]["coef"], r["table"]["std err"]
    print(f"{name:24s}{c['L1.n']:7.3f}{'(' + format(se['L1.n'], '.3f') + ')':>9s}{c['w']:8.3f}{c['k']:8.3f}{r['instruments']:8d}{r['ar2'][1]:9.3f}{r['hansen'][2]:10.3f}")
print(f"\nBounds for L1.n:  fixed effects {fixed.params['n_l1']:.3f}   pooled OLS {pooled.params['n_l1']:.3f}")
```

<div class="read" markdown="1">
**How to read it**

The table separates what this dataset can tell you from what it cannot.

- **The wage and capital effects are robust.** Across all five instrument sets the wage coefficient stays between −0.60 and −0.71 and the capital coefficient between 0.31 and 0.37. Neither the AR(2) test nor the Hansen test rejects at the 5% level in any row, although two rows come close (p = 0.07 and 0.08).
- **The persistence coefficient is not.** It ranges from 0.39 to 1.36 depending on the instruments, with large standard errors, and in four of the five rows it falls outside the bounds of 0.73 and 1.05.

This is a known weakness of difference GMM, not a mistake in the code. When a series is highly persistent, its past **levels** are weak instruments for its current **changes**, and the estimate of \(\alpha\) becomes imprecise and biased towards the fixed-effects value. Blundell and Bond (1998) proposed **system GMM** for exactly this case: it adds the equation in levels, instrumented with lagged differences, and is now the standard estimator for persistent series such as output, capital or employment. In Stata it is `xtabond2` without the `noleveleq` option; in R, `plm::pgmm` with `transformation = "ld"`.

So the paper can report the wage and capital elasticities with confidence, and should say plainly that the degree of persistence is not well identified by difference GMM in this sample.
</div>

## Write the paper

| Section | What goes in it | From which step |
|---|---|---|
| Identification | Why the regressor is endogenous; what the instruments are; why each is relevant and exogenous | Step 1 |
| Data | Sources, sample, construction of variables | Step 2 |
| Results | OLS and IV estimates side by side | Steps 3, 4 and 8 |
| Instrument diagnostics | First-stage F and partial R², overidentification test, endogeneity test, all in the main table | Steps 5 to 7 |
| Robustness | Alternative instrument sets and samples | Step 9 |
| Dynamic panel | Pooled OLS and fixed-effects bounds, GMM estimates, AR(1), AR(2), Hansen test, number of instruments and of units | Steps 10 to 14 |

<div class="write" markdown="1">
**How to write it: results**

Table 2 reports OLS, 2SLS and two-step GMM estimates of the policy rule for 1979:Q3–2007:Q4. The instruments are strong, with first-stage F-statistics of 301 for inflation and 58 for the output gap, and the Hansen J test does not reject the overidentifying restrictions (p = 0.34). A Wu–Hausman test rejects the exogeneity of inflation and the output gap (p < 0.01), confirming the need for instrumental-variable estimation. The GMM estimates imply a smoothing coefficient of 0.87 and long-run responses of 1.79 to inflation (s.e. 0.49) and 1.50 to the output gap (s.e. 0.86). The point estimate of the inflation response exceeds one, consistent with the Taylor principle, although the hypothesis that it is at most one is rejected only at the 10% level (p = 0.054). For 1961–1979 the corresponding estimate is 0.52. The post-1979 estimate is sensitive to the sample: it falls to about one when the 1979–82 disinflation is excluded.
</div>

<div class="write" markdown="1">
**How to write it: dynamic panel results**

Table 4 reports one-step difference GMM estimates of the employment equation for 140 firms over 1976–1984, with standard errors robust to heteroskedasticity and within-firm correlation. Lagged employment is instrumented with its levels dated \(t-2\) and earlier, giving 41 instruments. The Arellano–Bond test does not reject the absence of second-order serial correlation in the differenced residuals (z = −0.52, p = 0.61), and the Hansen test does not reject the overidentifying restrictions (\(\chi^2(25)\) = 31.4, p = 0.18). The short-run wage elasticity of employment is −0.61 (s.e. 0.18) and the short-run capital elasticity is 0.36 (s.e. 0.06); both are stable across alternative instrument sets. The coefficient on lagged employment is 0.69, slightly below the fixed-effects estimate of 0.73, and it is sensitive to the choice of instruments. This pattern is characteristic of weak instruments in difference GMM for persistent series, so the degree of employment persistence should be interpreted with caution.
</div>

<div class="warn" markdown="1">
**What a referee will ask**

- **Why are your instruments exogenous?** This is the central question, and no test answers it. Give an economic argument for each instrument.
- **Are they strong?** Report the first-stage F for every endogenous regressor.
- **Do the results depend on the instrument set?** Show at least one alternative set.
- **How many instruments?** Report the count. In panels it must be below the number of units.
- **Does the estimate lie between fixed effects and pooled OLS?** If not, as in the panel example here, suspect weak instruments and consider system GMM.
- **Did AR(2) pass?** If it is rejected, lag-2 instruments are invalid.
- **Is IV needed at all?** Report the endogeneity test and the OLS estimates.
- **How was the output gap measured?** The gap here uses a two-sided HP filter, which uses future data; a real-time measure would be a useful check, and the paper should say so.
</div>
