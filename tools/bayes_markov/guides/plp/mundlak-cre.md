title: Correlated Random Effects (Mundlak) and the Robust Hausman Test
short: Mundlak / CRE
when: You want random-effects estimates of time-invariant variables but need to guard against a unit effect that is correlated with the regressors, and you need a Hausman test that survives clustering
data: NLS young men, 545 workers, 1980-87 (Vella and Verbeek 1998): log wage, union membership, marriage, experience squared, schooling, race
tools: Python · OES engine · NumPy
oes: cre
lead: Use this guide when you must choose between fixed and random effects and the Hausman test is not enough: the classical test fails with heteroskedastic or serially correlated errors, and fixed effects discards the time-invariant variables you care about. The Mundlak (1978) and Chamberlain approach adds the unit means of the time-varying regressors to the model. It reproduces the fixed-effects estimates for the time-varying variables, keeps the time-invariant variables, and the test that the means are zero is a Hausman test that is robust to clustering. The guide covers the setup, the estimates, the comparison with fixed and random effects, the test, and the text for the paper.
---

## The question and the model

**Is the unit effect correlated with the regressors?** In the model \(y_{it}=x_{it}'\beta+z_i'\gamma+c_i+e_{it}\), random effects needs \(c_i\) to be uncorrelated with \(x_{it}\) and fixed effects does not, but fixed effects cannot estimate \(\gamma\). Mundlak models the correlation directly,

\[
c_i = \bar x_i'\,\pi + a_i,\qquad y_{it} = x_{it}'\beta + \bar x_i'\pi + z_i'\gamma + a_i + e_{it},
\]

where \(\bar x_i\) is the unit mean of \(x_{it}\). The coefficient \(\beta\) is then the within (fixed-effects) estimate, \(\pi\) is the difference between the between and the within effects, and \(\gamma\) is estimated from the variation between units. The null \(\pi = 0\) is the hypothesis that random effects and fixed effects agree, so a Wald test of \(\pi=0\) with standard errors clustered by unit is a Hausman test that remains valid under heteroskedasticity and serial correlation.

<div class="read" markdown="1">
**Three checks before you trust the estimates**

1. **Do the time-varying regressors vary within units?** The within estimate for a regressor that barely changes is imprecise, and the engine rejects one that never changes.
2. **Is the unit effect really correlated with the regressors?** The Mundlak test answers this. If it does not reject, random effects is more efficient.
3. **Are the time-invariant variables credible?** Their coefficients are identified only from differences between units, so they carry whatever omitted unit characteristics are correlated with them.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We estimate a log wage equation for 545 men observed for eight years (1980-87) with the correlated random-effects approach of Mundlak (1978) and Chamberlain (1982). The model includes union membership, marital status and the square of experience as time-varying regressors, schooling and race as time-invariant regressors, period dummies, and the worker means of the three time-varying regressors. Standard errors are clustered by worker. The coefficients on the time-varying regressors are numerically the fixed-effects estimates, and a cluster-robust Wald test that the coefficients on the worker means are jointly zero is a robust Hausman test of random effects against fixed effects.
</div>

## Load the data and see where the variation is

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

data = open("wagepan_full.csv").read()
df = pd.read_csv("wagepan_full.csv")
print(f"{df.nr.nunique()} workers, {df.year.min()}-{df.year.max()}, observations per worker: {sorted(int(n) for n in df.groupby('nr').size().unique())}")
for v in ["lwage", "union", "married", "expersq", "educ"]:
    between = df.groupby("nr")[v].transform("mean").var() / df[v].var()
    print(f"{v:8s} share of variance between workers: {between:.2f}")
print("union share:", round(df.union.mean(), 3), "  married share:", round(df.married.mean(), 3), "  workers who ever change union status:",
      int((df.groupby("nr").union.nunique() > 1).sum()))
```

<div class="read" markdown="1">
**How to read it**

The panel is balanced: 545 workers observed for eight years, 4,360 observations. Between 42% and 59% of the variance of log wages, union status, marital status and squared experience lies between workers, so a large part of the variation is lost when the unit effect is removed; schooling, which never changes, is entirely between workers (1.00). Twenty-four percent of the worker-years are union years and 44% are married years. The within variation in union status comes from 246 workers (45%) who change status at least once, which is the group that identifies the fixed-effects union coefficient.
</div>

## Estimate the correlated random-effects model

```python
spec = dict(y="lwage", x=["union", "married", "expersq"], z=["educ", "black", "hisp"], unit="nr", time="year")
r = oes.run("cre", data, spec)
oes.summary(r)
tab = pd.DataFrame(r["tables"][0]["rows"], columns=r["tables"][0]["columns"])
print(tab[~tab.Variable.str.startswith("period")].to_string(index=False))
```

<div class="read" markdown="1">
**How to read it**

The coefficients on the time-varying variables are the within estimates: union membership is associated with 8.0% higher wages (standard error 0.023), marriage with 4.7% (0.021) and the square of experience has a coefficient of -0.0052 (0.0008). The coefficients on the worker means are significant for all three variables: 0.183 for union, 0.094 for marriage and 0.0072 for squared experience. Each is the between effect minus the within effect, so the between-worker association of union membership with wages is 0.263 (0.080 + 0.183), more than three times the within effect, and that of marriage is 0.141. The time-invariant coefficients are 0.094 for a year of schooling (standard error 0.011) and -0.142 for black workers (0.051); the Hispanic coefficient is 0.008 (0.038) and not significant.
</div>

## Compare fixed effects, random effects and CRE, and test

```python
oes.show(r, "Time-varying coefficients")
oes.show(r, "Specification test")
oes.show(r, "Variance components")
```

<div class="read" markdown="1">
**How to read it**

Random effects gives a union coefficient of 0.110 and a marriage coefficient of 0.077, against 0.080 and 0.047 from fixed effects (the CRE column reproduces the fixed-effects estimates exactly). The coefficient on squared experience is -0.0015 under random effects and -0.0052 under fixed effects. Random effects therefore overstates the union and marriage premiums by about 0.03 each, which is what the positive coefficients on the worker means predict: workers with higher average union and marriage rates are also workers with higher unobserved wage components. The cluster-robust Wald test that the three means are jointly zero is 68.2 with 3 degrees of freedom (p < 0.001), so random effects is rejected. The variance of the worker effect (standard deviation 0.327) is about as large as that of the idiosyncratic error (0.351), which gives rho = 0.47.
</div>

## Which variables drive the rejection?

```python
out = {}
for v in ["union", "married", "expersq"]:
    rr = oes.run("cre", data, dict(spec, x=[v]))
    row = [x for x in rr["tables"][0]["rows"] if x[0] == "mean of " + v][0]
    t = rr["tables"][2]["rows"][0]
    out[v] = [row[1], row[2], row[4], t[2]]
print(pd.DataFrame(out, index=["coef. on mean", "s.e.", "p-value", "Mundlak p (alone)"]).T.round(4))
```

<div class="read" markdown="1">
**How to read it**

Each regressor rejects on its own: the coefficient on the worker mean is 0.193 for union (p = 0.0001), 0.122 for marriage (p = 0.0085) and 0.0079 for squared experience (p < 0.001). The conclusion that random effects is inconsistent does not rest on one variable. The squared-experience means are the strongest signal, partly because average squared experience is closely related to how long a worker has been in the labour market, so it picks up worker heterogeneity mechanically. The union and marriage results are the ones with an economic interpretation.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

A correlated random-effects specification in the sense of Mundlak (1978) rejects the random-effects assumption that the worker effect is uncorrelated with union membership, marital status and experience (cluster-robust Wald chi-squared = 68.2, 3 degrees of freedom, p < 0.001); each regressor rejects on its own. The within estimates, which the model reproduces for the time-varying variables, show a union premium of 8.0% (standard error 0.023) and a marriage premium of 4.7% (0.021), compared with 11.0% and 7.7% under random effects. Schooling raises wages by 9.4% per year (0.011) and the coefficient for black workers is -0.142 (0.051); these time-invariant coefficients are identified from differences between workers and may reflect unobserved worker characteristics. Standard errors are clustered by worker.
</div>

| Report | What | Where |
|---|---|---|
| Model | Time-varying and time-invariant regressors, means, clustering | Step 1 |
| Data | Within and between variation | Step 2 |
| Estimates | CRE coefficients | Step 3 |
| Comparison | FE, RE, CRE and the Mundlak test | Step 4 |
| Diagnosis | Which regressor drives the test | Step 5 |

**References.** Mundlak, Y. (1978), "On the Pooling of Time Series and Cross Section Data," *Econometrica* 46. Chamberlain, G. (1982), "Multivariate Regression Models for Panel Data," *Journal of Econometrics* 18. Wooldridge, J. (2010), *Econometric Analysis of Cross Section and Panel Data*, 2nd ed., MIT Press, section 10.7.2. Hausman, J. (1978), "Specification Tests in Econometrics," *Econometrica* 46. Vella, F. and Verbeek, M. (1998), "Whose Wages Do Unions Raise? A Dynamic Model of Unionism and Wage Rate Determination for Young Men," *Journal of Applied Econometrics* 13.
