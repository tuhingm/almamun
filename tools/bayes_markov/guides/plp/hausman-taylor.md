title: Hausman-Taylor Estimator
short: Hausman-Taylor
when: You want the return to a time-invariant variable such as schooling, but you suspect it is correlated with the unit effect, so random effects is biased and fixed effects cannot estimate it
data: Cornwell and Rupert (1988) PSID wage panel: 595 workers observed for seven years, log wage, experience, weeks worked, union status, schooling, sex and race
tools: Python · OES engine · NumPy
oes: hausman_taylor
lead: Use this guide when a variable you care about never changes within a unit, so fixed effects wipes it out, and you cannot assume it is uncorrelated with the unit effect, so random effects is suspect. The Hausman-Taylor (1981) estimator is the middle route: you declare which regressors are exogenous and which are not, and it uses the exogenous time-varying ones as instruments for the endogenous time-invariant ones. The guide covers the classification, the estimates, the comparison with fixed and random effects, the Hausman test, how sensitive the answer is to the classification, and the text for the paper.
---

## The question and the model

**What is the return to schooling when ability is unobserved?** In a wage panel the model is

\[
y_{it} = X_{1it}'\beta_1 + X_{2it}'\beta_2 + Z_{1i}'\gamma_1 + Z_{2i}'\gamma_2 + \mu_i + e_{it},
\]

where \(X_1\) are time-varying regressors uncorrelated with the unit effect \(\mu_i\), \(X_2\) are time-varying regressors that may be correlated with it, \(Z_1\) are time-invariant regressors uncorrelated with \(\mu_i\) and \(Z_2\) are time-invariant regressors that may be correlated with it. Schooling is the standard example of \(Z_2\): unobserved ability raises both schooling and wages. Fixed effects cannot estimate the schooling coefficient because schooling does not change, and random effects is biased if ability is in \(\mu_i\). Hausman and Taylor use the within deviations of all time-varying variables and the unit means of \(X_1\) as instruments. The unit means of the exogenous time-varying variables identify \(\gamma_2\), which needs at least as many variables in \(X_1\) as in \(Z_2\).

<div class="read" markdown="1">
**Three checks before you trust the estimator**

1. **Is the classification defensible?** Every result depends on which variables you call exogenous. State the reason for each choice, not only the choice.
2. **Is the model identified?** The number of variables in \(X_1\) must be at least the number in \(Z_2\). The engine stops with a message if it is not.
3. **Do the unit means of \(X_1\) predict \(Z_2\)?** If the exogenous time-varying variables are unrelated to the endogenous time-invariant ones, the instruments are weak and the standard errors are large.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We estimate a log wage equation on a panel of 595 workers observed for seven years with the Hausman-Taylor (1981) estimator. Experience, experience squared, weeks worked, marital status and union membership are treated as correlated with the worker effect; occupation, region, urban residence and industry are treated as uncorrelated with it. Sex and race are treated as uncorrelated with the worker effect, and years of schooling as correlated with it. The model is identified because four time-varying exogenous variables instrument one endogenous time-invariant variable. We follow the classification of Baltagi and Khanti-Akom (1990).
</div>

## Load the data and check which variables vary within workers

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

data = open("psid_wages.csv").read()
df = pd.read_csv("psid_wages.csv")
print(f"{df.id.nunique()} workers, years coded {df.year.min()} to {df.year.max()}, observations per worker: {sorted(int(n) for n in df.groupby('id').size().unique())}")
vars_ = ["lwage", "exp", "wks", "union", "ms", "occ", "ed", "fem", "blk"]
share = {v: float(round(df.groupby("id")[v].transform("mean").var() / df[v].var(), 3)) for v in vars_}
print("share of variance that is between workers:", share)
print("mean log wage:", round(df.lwage.mean(), 3), "  share female:", round(df.fem.mean(), 3), "  share black:", round(df.blk.mean(), 3))
```

<div class="read" markdown="1">
**How to read it**

The panel is balanced: 595 workers for seven years (coded 1 to 7), 4,165 observations. Schooling, sex and race do not change within a worker (between-worker share 1.0), so fixed effects cannot estimate them. Most of the variance of log wages is between workers (0.73), which is the variation that the unit effect absorbs. Experience is almost entirely between workers too (0.97), because every worker ages one year per year, so its within variation is a time trend. Only 11% of the workers are women and 7% are black, so the coefficients on sex and race rest on small groups.
</div>

## Estimate the model

```python
spec = dict(y="lwage", x1=["occ", "south", "smsa", "ind"], x2=["exp", "exp2", "wks", "ms", "union"],
            z1=["fem", "blk"], z2=["ed"], unit="id", time="year")
r = oes.run("hausman_taylor", data, spec)
oes.summary(r)
oes.show(r, "Hausman-Taylor estimates")
oes.show(r, "Variance components")
```

<div class="read" markdown="1">
**How to read it**

The estimated return to schooling is 0.138 (standard error 0.021): one more year of schooling is associated with about 14% higher wages, after allowing schooling to be correlated with the worker effect. Experience raises wages by 0.113 per year at the start of a career, with a negative squared term. Union membership is associated with a 3.3% wage premium (p = 0.028), and living in a metropolitan area with a 4.2% discount (p = 0.027). The coefficient on race is -0.286 (p = 0.066) and on sex -0.131 (p = 0.30); both are imprecise because they are identified from a few workers and from the unit-level variation only.

The unit effect dominates the error: its standard deviation is 0.94 against 0.15 for the idiosyncratic part, so \(\rho\) = 0.975. With so much weight on the unit effect, \(\theta\) is 0.94 and the estimator is close to the within estimator for the time-varying variables.
</div>

## Compare with fixed effects and random effects

```python
oes.show(r, "Time-varying coefficients")
X = spec["x1"] + spec["x2"] + spec["z1"] + spec["z2"]
re = oes.run("re", data, dict(y="lwage", x=X, unit="id", time="year"))
rows = {x[0]: x[1] for x in re["tables"][0]["rows"]}
ht = {x[0]: x[2] for x in r["tables"][0]["rows"]}
print(pd.DataFrame({"random effects": {k: rows[k] for k in ["ed", "fem", "blk", "exp", "union"]},
                    "Hausman-Taylor": {k: ht[k] for k in ["ed", "fem", "blk", "exp", "union"]}}).round(3))
oes.show(r, "Specification test")
```

<div class="read" markdown="1">
**How to read it**

For the time-varying variables the Hausman-Taylor and within estimates are nearly identical (experience 0.1131 against 0.1132, union 0.0328 in both), which is expected when most variables are treated as endogenous. They differ only for the variables treated as exogenous: the region coefficient moves from -0.002 to 0.008 and industry from 0.019 to 0.014, small changes within their standard errors.

The big difference is in the time-invariant variables. Random effects gives a schooling return of 0.100 (standard error 0.008), Hausman-Taylor 0.138 (0.021). Random effects assumes schooling is uncorrelated with the worker effect. If ability sat in the worker effect and raised schooling, random effects would be biased upward, so the Hausman-Taylor estimate would be lower; here it is higher, so the data do not show that pattern. Possible reasons, which this guide does not test, are measurement error in schooling or a negative correlation between schooling and the worker effect. The instrumented estimate is also much less precise. The other contrasts are large too: the union premium falls from 0.063 to 0.033, the sex coefficient from -0.339 to -0.131 (no longer significant) and the race coefficient moves from -0.210 to -0.286.

The Hausman test compares the within and Hausman-Taylor estimates of the time-varying coefficients: the statistic is 5.33 with 3 degrees of freedom (p = 0.149), so the extra instruments, the unit means of the four exogenous variables, are not rejected.
</div>

## How sensitive is the answer to the classification?

```python
alt = {
    "baseline": {},
    "union exogenous": dict(x1=spec["x1"] + ["union"], x2=["exp", "exp2", "wks", "ms"]),
    "experience exogenous": dict(x1=spec["x1"] + ["exp", "exp2"], x2=["wks", "ms", "union"]),
    "weeks, marriage, union exogenous": dict(x1=spec["x1"] + ["wks", "ms", "union"], x2=["exp", "exp2"]),
    "schooling exogenous": dict(z1=["fem", "blk", "ed"], z2=[]),
}
out = {}
for name, v in alt.items():
    rr = oes.run("hausman_taylor", data, dict(spec, **v))
    c = {x[0]: x[2] for x in rr["tables"][0]["rows"]}
    t = rr["tables"][2]["rows"][0]
    out[name] = [c["ed"], c["union"], c["exp"], t[1], t[2]]
print(pd.DataFrame(out, index=["schooling", "union", "experience", "Hausman stat", "p-value"]).T.round(4))
```

<div class="read" markdown="1">
**How to read it**

The schooling coefficient is between 0.132 and 0.144 in four of the five classifications, and the union premium between 0.033 and 0.038 in all of them. The exception is the classification that treats experience as exogenous: schooling jumps to 0.237. The Hausman test rejects that classification decisively (statistic 372, p < 0.001), which is what it is for: experience is a function of age and past career choices and is not plausibly uncorrelated with the worker effect. Treating weeks worked, marriage and union as exogenous also rejects (p = 0.0045). Treating schooling as exogenous is not rejected (p = 0.238) and gives a coefficient of 0.144, so the test does not distinguish between random effects and Hausman-Taylor for schooling. A test that does not reject is not proof of the classification.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

The Hausman-Taylor estimate of the return to schooling is 0.138 (standard error 0.021), compared with 0.100 (0.008) from random effects. The estimate allows schooling, experience, weeks worked, marital status and union membership to be correlated with the worker effect. A Hausman test comparing the within and Hausman-Taylor estimates of the time-varying coefficients does not reject the classification (chi-squared with 3 degrees of freedom = 5.33, p = 0.149). The union premium is 3.3% (p = 0.028). The estimates are robust to moving union membership or schooling to the exogenous group (schooling 0.132 to 0.144), whereas classifying experience as exogenous is rejected (p < 0.001) and would raise the schooling coefficient to 0.237. Standard errors assume the random-effects error structure; the worker effect accounts for 97.5% of the variance of the error.
</div>

| Report | What | Where |
|---|---|---|
| Model | Four groups of regressors and the reason for each | Step 1 |
| Data | Which variables vary within workers, between-worker share | Step 2 |
| Estimates | Coefficients, variance components | Step 3 |
| Comparison | Within and random-effects estimates, Hausman test | Step 4 |
| Robustness | Alternative classifications, Hausman test for each | Step 5 |

**References.** Hausman, J. and Taylor, W. (1981), "Panel Data and Unobservable Individual Effects," *Econometrica* 49. Amemiya, T. and MaCurdy, T. (1986), "Instrumental-Variable Estimation of an Error-Components Model," *Econometrica* 54. Baltagi, B. and Khanti-Akom, S. (1990), "On Efficient Estimation with Panel Data: An Empirical Comparison of Instrumental Variables Estimators," *Journal of Applied Econometrics* 5. Cornwell, C. and Rupert, P. (1988), "Efficient Estimation with Panel Data: An Empirical Comparison of Instrumental Variables Estimators," *Journal of Applied Econometrics* 3.
