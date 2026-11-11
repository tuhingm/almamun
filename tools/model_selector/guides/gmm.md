title: 2SLS and GMM
short: 2SLS / GMM
when: Endogenous regressors that need instruments
data: U.S. federal funds rate, inflation and the output gap, quarterly, 1979–2007
lead: Use this guide when a regressor is correlated with the error term, so that OLS is biased. It covers choosing instruments, two-stage least squares, instrument strength, overidentification and endogeneity tests, efficient GMM, and robustness, with sample text for the paper and a note on dynamic panels.
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

## If your data is a panel: dynamic panel GMM

The same logic extends to panel data with a lagged dependent variable, such as growth regressions across countries or leverage regressions across firms:

\[
y_{it} = \alpha\, y_{i,t-1} + \beta' x_{it} + \eta_i + \varepsilon_{it}
\]

The lagged dependent variable is correlated with the fixed effect \(\eta_i\), so pooled OLS overstates \(\alpha\); the fixed-effects estimator understates it when the number of periods is small (the Nickell bias). The remedies are GMM estimators.

| Estimator | Idea | Use when |
|---|---|---|
| Difference GMM (Arellano–Bond) | First-difference to remove \(\eta_i\); instrument \(\Delta y_{i,t-1}\) with levels dated \(t-2\) and earlier | many units, few periods |
| System GMM (Blundell–Bond) | Adds the level equation, instrumented with lagged differences | the series are very persistent, where difference GMM has weak instruments |

A dynamic panel GMM table is not complete without these four items.

1. **AR(1) and AR(2) tests** on the differenced residuals. AR(1) should be significant by construction. **AR(2) must not be**: if it is, the lag-2 instruments are invalid and you must start from deeper lags.
2. **Hansen J test** of the overidentifying restrictions, with a p-value that is neither below 0.05 nor suspiciously close to 1.
3. **The number of instruments**, which must be smaller than the number of units. Use collapsed instruments and limit the lag depth; too many instruments overfit and make the Hansen test useless.
4. **A bounds check:** the GMM estimate of \(\alpha\) should lie between the fixed-effects estimate (too low) and the pooled OLS estimate (too high).

<div class="warn" markdown="1">
**Why this guide has no worked panel example**

A worked example was prepared on a panel of annual inflation in 115 countries, but it did not pass the tests above: the Hansen test rejected the instruments in most specifications. A guide should not teach from an example that fails its own specification tests, so it is left out. The method and its code are covered in the [Dynamic panels lecture](../teaching/econometrics/lectures/08-dynamic-panels.html) of the Econometrics course. In Stata the standard tool is `xtabond2`; in R, `plm::pgmm`.
</div>

## Write the paper

| Section | What goes in it | From which step |
|---|---|---|
| Identification | Why the regressor is endogenous; what the instruments are; why each is relevant and exogenous | Step 1 |
| Data | Sources, sample, construction of variables | Step 2 |
| Results | OLS and IV estimates side by side | Steps 3, 4 and 8 |
| Instrument diagnostics | First-stage F and partial R², overidentification test, endogeneity test, all in the main table | Steps 5 to 7 |
| Robustness | Alternative instrument sets and samples | Step 9 |

<div class="write" markdown="1">
**How to write it: results**

Table 2 reports OLS, 2SLS and two-step GMM estimates of the policy rule for 1979:Q3–2007:Q4. The instruments are strong, with first-stage F-statistics of 301 for inflation and 58 for the output gap, and the Hansen J test does not reject the overidentifying restrictions (p = 0.34). A Wu–Hausman test rejects the exogeneity of inflation and the output gap (p < 0.01), confirming the need for instrumental-variable estimation. The GMM estimates imply a smoothing coefficient of 0.87 and long-run responses of 1.79 to inflation (s.e. 0.49) and 1.50 to the output gap (s.e. 0.86). The point estimate of the inflation response exceeds one, consistent with the Taylor principle, although the hypothesis that it is at most one is rejected only at the 10% level (p = 0.054). For 1961–1979 the corresponding estimate is 0.52. The post-1979 estimate is sensitive to the sample: it falls to about one when the 1979–82 disinflation is excluded.
</div>

<div class="warn" markdown="1">
**What a referee will ask**

- **Why are your instruments exogenous?** This is the central question, and no test answers it. Give an economic argument for each instrument.
- **Are they strong?** Report the first-stage F for every endogenous regressor.
- **Do the results depend on the instrument set?** Show at least one alternative set.
- **How many instruments?** Report the count. In panels it must be below the number of units.
- **Is IV needed at all?** Report the endogeneity test and the OLS estimates.
- **How was the output gap measured?** The gap here uses a two-sided HP filter, which uses future data; a real-time measure would be a useful check, and the paper should say so.
</div>
