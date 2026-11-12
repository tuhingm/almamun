title: Fixed Effects
short: FE
when: Unit effects correlated with the regressors
data: 545 U.S. men, annual wages, 1980–1987 (Wooldridge wagepan)
lead: Use this guide when the panel is short, the units differ systematically, and the Hausman or Mundlak test shows that those differences are correlated with the regressors. It takes one real panel from the model equation to the finished results table, with the specification tests, diagnostics, robustness checks and sample text for each part of the paper.
---

## The question and the model

The question is a classic of labour economics: **do union members earn more than comparable non-members?** A cross-section cannot answer it, because workers who join unions may differ in ability, motivation or occupation in ways the data do not record. A panel that follows the same workers over time can: it compares each worker with himself, in years with and without union membership.

\[
\ln w_{it} = \beta_1\,union_{it} + \beta_2\,married_{it} + \beta_3\,exper^2_{it} + \alpha_i + \lambda_t + \varepsilon_{it}
\]

| Symbol | Variable in the file | Definition | Expected sign |
|---|---|---|---|
| \(\ln w_{it}\) | `lwage` | Log hourly wage | dependent variable |
| \(union_{it}\) | `union` | 1 if wage set by a union contract | positive: the union wage premium |
| \(married_{it}\) | `married` | 1 if married | positive: the marriage premium |
| \(exper^2_{it}\) | `expersq` | Years of experience, squared | negative: wage growth slows with experience |
| \(\alpha_i\) | — | Worker fixed effect: ability, motivation, occupation | unobserved |
| \(\lambda_t\) | — | Year effect: inflation, the business cycle | common to all workers |

Experience itself is not in the model. Every worker gains exactly one year of experience per year, so with worker and year effects the level of experience is perfectly collinear and cannot be estimated. Its square still varies within workers.

<div class="read" markdown="1">
**Before you estimate anything**

Write down what the fixed effect \(\alpha_i\) stands for in your application and why it may be correlated with the regressor of interest. Here: more able workers may be less likely to join a union and also earn more. If that is true, pooled OLS mixes the union effect with the ability effect, and only the within-worker comparison isolates it. That argument is the reason for using fixed effects, and the paper must make it.
</div>

<div class="write" markdown="1">
**How to write it: model section**

We estimate \(\ln w_{it} = \beta_1 union_{it} + \beta_2 married_{it} + \beta_3 exper^2_{it} + \alpha_i + \lambda_t + \varepsilon_{it}\), where \(\alpha_i\) is a worker fixed effect and \(\lambda_t\) a year effect. The worker effect absorbs all time-invariant characteristics, including unobserved ability, which may be correlated with union status. The union premium \(\beta_1\) is therefore identified from workers who change union status during the sample.
</div>

## Load the panel and confirm the starting point

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects, FirstDifferenceOLS, compare

raw = pd.read_csv("wagepan_full.csv")
df = raw.set_index(["nr", "year"])
print(f"{df.index.get_level_values(0).nunique()} workers x {df.index.get_level_values(1).nunique()} years = {len(df)} rows")

xs = ["union", "married", "expersq"]
for v in ["lwage"] + xs + ["educ"]:
    x = df[v]
    within = x - x.groupby(level=0).transform("mean")
    print(f"{v:8s} overall sd = {x.std():6.3f}   between sd = {x.groupby(level=0).mean().std():6.3f}   within sd = {within.std():6.3f}")

switchers = df["union"].groupby(level=0).agg(lambda s: s.nunique() > 1)
print(f"Workers who change union status at least once: {switchers.sum()} of {len(switchers)}")
```

<div class="read" markdown="1">
**How to read it**

The panel is **balanced and short**: 545 workers, each observed in all 8 years. With so few periods, unit-root tests are not the issue; the question is how to treat the worker effects. The model selection tool ran that sequence on this file and recommended fixed effects (the F, LM and Mundlak tests are repeated below).

The within standard deviations matter most. Union status has a within standard deviation of about 0.28: plenty of workers move in and out of union jobs, and the line below shows how many. **Those switchers identify the fixed-effects estimate.** Education has a within standard deviation of zero: it never changes for these men, so fixed effects cannot estimate its return.
</div>

## The benchmark: pooled OLS

```python
years = pd.get_dummies(df.index.get_level_values(1), prefix="y", drop_first=True, dtype=float).set_index(df.index)
y = df["lwage"]
X = sm.add_constant(df[xs].join(years))

pooled = PooledOLS(y, X).fit(cov_type="clustered", cluster_entity=True)
print(pooled.params[xs].round(4).to_string())
print("Clustered standard errors:")
print(pooled.std_errors[xs].round(4).to_string())
```

<div class="read" markdown="1">
**How to read it**

Pooled OLS treats the 4,360 rows as if they were 4,360 different people. It puts the union premium at about 18%. This estimate uses both the comparison across workers (union members against non-members) and the comparison over time, so if union members differ in unobserved ways, the premium is biased.

Even in pooled OLS, **the standard errors must be clustered by worker**: the same man appears eight times, and his errors are correlated across years. Unclustered standard errors would be far too small.
</div>

## Estimate the fixed-effects model

```python
fe = PanelOLS(y, sm.add_constant(df[xs]), entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
print(fe.summary.tables[1])
print(f"Observations: {fe.nobs}   workers: {fe.entity_info['total']:.0f}")
```

<div class="read" markdown="1">
**How to read it**

- **Union.** Comparing the same worker in union and non-union years, union membership raises the wage by about 8%. That is less than half the pooled OLS premium: much of the cross-sectional premium reflects who joins unions, not what unions pay.
- **Married.** The marriage premium falls from about 15% in pooled OLS to about 5%, significant only at the 5% level. Much of the cross-sectional marriage premium is a selection effect.
- **Experience squared.** Negative and highly significant: wage growth slows with experience.
- **Standard errors.** Clustered by worker, which allows for any pattern of heteroskedasticity and serial correlation within each worker. This is the standard choice for a short panel with many units.

The constant reported by `PanelOLS` is the average of the fixed effects and has no economic meaning.
</div>

## Test 1: do the workers differ? The F test

```python
f = fe.f_pooled
print(f"F test that all worker effects are zero: F({f.df}, {f.df_denom}) = {f.stat:.2f}   p-value = {f.pval:.4f}")
```

<div class="read" markdown="1">
**How to read it**

The null hypothesis is that every worker has the same intercept, in which case pooled OLS would be enough. The F statistic is large and the p-value is essentially zero: **the worker effects are jointly significant**, so ignoring them is not an option. The remaining question is whether to treat them as fixed or random.
</div>

## Test 2: fixed or random? Hausman and Mundlak

```python
re = RandomEffects(y, X).fit(cov_type="clustered", cluster_entity=True)
re_plain = RandomEffects(y, X).fit()
fe_plain = PanelOLS(y, sm.add_constant(df[xs].join(years)), entity_effects=True).fit()

# conventional Hausman test (non-robust variances, as in the textbook formula)
d = fe_plain.params[xs] - re_plain.params[xs]
V = fe_plain.cov.loc[xs, xs] - re_plain.cov.loc[xs, xs]
H = float(d @ np.linalg.inv(V) @ d)
from scipy import stats
print(f"Hausman test:  chi2({len(xs)}) = {H:.2f}   p-value = {stats.chi2.sf(H, len(xs)):.4f}")

# Mundlak test: add each worker's average of the regressors, test them jointly with clustered errors
means = df[xs].groupby(level=0).transform("mean").add_suffix("_mean")
mundlak = RandomEffects(y, X.join(means)).fit(cov_type="clustered", cluster_entity=True)
w = mundlak.wald_test(formula=" = ".join(means.columns) + " = 0")
print(f"Mundlak test:  chi2({w.df}) = {w.stat:.2f}   p-value = {w.pval:.4f}")
print(f"Union coefficient in the Mundlak model: {mundlak.params['union']:.4f}  (FE: {fe.params['union']:.4f})")
```

<div class="read" markdown="1">
**How to read it**

Random effects is more efficient than fixed effects, but only consistent if the worker effects are **uncorrelated** with union status, marriage and experience.

- **Hausman test.** It compares the FE and RE coefficients. Under the null both are consistent and differ only by sampling error. Here the gap is large and the test rejects strongly.
- **Mundlak test.** It adds each worker's average union status, marriage rate and squared experience to the RE model. If the worker effect is correlated with the regressors, those averages pick it up. The test uses clustered standard errors, so unlike the conventional Hausman test it remains valid under heteroskedasticity and serial correlation. It also rejects.
- **A useful check.** In the Mundlak model, the coefficient on union is identical to the fixed-effects estimate. This is an exact algebraic result, and it is why the Mundlak model is also called correlated random effects.

**Conclusion: random effects is rejected, and fixed effects is the estimator to report.**
</div>

## Diagnostic tests: serial correlation and cross-sectional dependence

```python
# Wooldridge (2002) test for serial correlation: regress the first-differenced residuals on their own lag.
# Under no serial correlation in the levels errors, the coefficient is -0.5.
fd = FirstDifferenceOLS(y, df[xs]).fit(cov_type="clustered", cluster_entity=True)
e = fd.resids.rename("e").to_frame()
e["e_lag"] = e.groupby(level=0)["e"].shift(1)
e = e.dropna()
aux = PooledOLS(e["e"], e[["e_lag"]]).fit(cov_type="clustered", cluster_entity=True)
F = ((aux.params["e_lag"] + 0.5) / aux.std_errors["e_lag"]) ** 2
print(f"Wooldridge test: coefficient on lagged residual = {aux.params['e_lag']:.3f}   F(1, {fe.entity_info['total'] - 1:.0f}) = {F:.2f}   p-value = {stats.f.sf(F, 1, fe.entity_info['total'] - 1):.4f}")

# residual correlation between workers, averaged over all pairs (Pesaran's CD is not designed for T = 8, so describe only)
r = fe.resids.unstack(0)
c = r.corr().to_numpy()[np.triu_indices(r.shape[1], 1)]
print(f"Average correlation of residuals between workers: {np.nanmean(c):.3f}  (mean absolute: {np.nanmean(np.abs(c)):.3f})")
```

<div class="read" markdown="1">
**How to read it**

- **Serial correlation.** The Wooldridge test rejects: the errors of the same worker are correlated over time, as they almost always are in wage data. This does not bias the FE coefficients, but it invalidates conventional standard errors. **Clustering by worker already corrects for it**, which is why the clustered standard errors are the ones to report.
- **Cross-sectional dependence.** With T = 8, the residual correlation between any two workers rests on eight observations and is very noisy, so a mean absolute correlation of this size is what pure noise would produce. The average signed correlation is close to zero: once year effects are included, there is no common shock left. Year effects are the standard way to deal with common shocks in short micro panels.
</div>

## Robustness checks

```python
models = {"Pooled OLS": pooled, "Random effects": re, "Fixed effects": fe}

fd_years = FirstDifferenceOLS(y, df[xs].join(years)).fit(cov_type="clustered", cluster_entity=True)
models["First differences"] = fd_years

cre_x = sm.add_constant(df[xs + ["educ", "black", "hisp"]].join(years).join(means))
cre = RandomEffects(y, cre_x).fit(cov_type="clustered", cluster_entity=True)
models["CRE (Mundlak)"] = cre

print(f"{'':20s}{'union':>16s}{'married':>16s}")
for name, m in models.items():
    print(f"{name:20s}" + "".join(f"{m.params[v]:8.3f} ({m.std_errors[v]:.3f})" for v in ["union", "married"]))
print(f"\nReturn to a year of education in the CRE model: {cre.params['educ']:.3f} ({cre.std_errors['educ']:.3f})")
```

<div class="read" markdown="1">
**How to read it**

- **First differences** is the other estimator that removes the worker effect. It compares adjacent years only, so it is more sensitive to measurement error in union status: a misreported switch in one year looks like two genuine switches. The FD premium is smaller, which is the usual sign of measurement error in a binary regressor. Report it, and explain the difference.
- **Correlated random effects** adds the worker averages of the time-varying regressors. Its union and marriage coefficients equal the fixed-effects ones, but it can also estimate the effect of time-invariant variables such as education and race. This is the answer when a referee asks for the return to education in a fixed-effects paper.
- **The ranking pooled > RE > FE** for the union premium is exactly what positive selection on unobserved ability predicts.
</div>

## Build the results table for the paper

```python
def star(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

cols = {"(1) Pooled OLS": pooled, "(2) RE": re, "(3) FE": fe, "(4) FD": fd_years}
print(f"{'':16s}" + "".join(f"{c:>16s}" for c in cols))
for v in xs:
    print(f"{v:16s}" + "".join(f"{m.params[v]:13.4f}{star(m.pvalues[v]):3s}" for m in cols.values()))
    print(f"{'':16s}" + "".join(f"{'(' + format(m.std_errors[v], '.4f') + ')':>13s}   " for m in cols.values()))
print(f"{'Worker effects':16s}{'no':>13s}   {'random':>13s}   {'fixed':>13s}   {'differenced':>13s}")
print(f"{'Year effects':16s}" + f"{'yes':>13s}   " * 4)
print(f"{'Observations':16s}" + "".join(f"{m.nobs:>13d}   " for m in cols.values()))
print("Standard errors clustered by worker in parentheses. *** p<0.01, ** p<0.05, * p<0.10.")
```

<div class="read" markdown="1">
**How to read it**

A panel results table shows the main estimator next to the benchmarks, so the reader can see how much the unit effects matter. Report under the table: the number of workers and years, that standard errors are clustered by worker, that all columns include year effects (except where they are differenced away), and the F, Hausman and Mundlak statistics. Column (3) is the preferred estimate.
</div>

## Write the paper

<div class="write" markdown="1">
**How to write it: specification tests**

An F test rejects the hypothesis that the worker effects are jointly zero, so pooled OLS is inappropriate. A Hausman test rejects the consistency of random effects, and a cluster-robust Mundlak test, which remains valid under heteroskedasticity and serial correlation, leads to the same conclusion. We therefore estimate the model by fixed effects with worker and year effects, and cluster standard errors by worker; a Wooldridge test confirms serial correlation within workers.
</div>

<div class="write" markdown="1">
**How to write it: results**

Table 1 reports the estimates. Pooled OLS puts the union wage premium at about 18 percent (column 1). Once time-invariant worker characteristics are controlled for, the premium falls to about 8 percent (column 3), and it remains significant at the 1 percent level. The decline is consistent with workers of higher unobserved ability being less likely to hold union jobs. The first-difference estimate is smaller (column 4), as expected if union status is measured with error.
</div>

<div class="warn" markdown="1">
**Common mistakes**

- Reporting unclustered standard errors. In a panel they are almost always too small.
- Including a regressor that moves one-for-one with time (age, experience) together with year effects. It is collinear and either drops out or gives a meaningless coefficient.
- Claiming the FE coefficient on a variable that never changes. It is not identified; use correlated random effects instead.
- Treating fixed effects as a cure for all endogeneity. It removes time-invariant confounders only. If union status responds to wage shocks in the same year, FE is still biased, and you need an instrument or a dynamic model.
- Using only the conventional Hausman test. Report the robust Mundlak version as well.
</div>
