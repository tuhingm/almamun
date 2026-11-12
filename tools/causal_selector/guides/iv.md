title: Instrumental Variables
short: IV
when: An instrument shifts the treatment
data: 3,010 U.S. men in 1976 (National Longitudinal Survey of Young Men, Card 1995)
lead: Use this guide when the treatment is endogenous but a variable is available that shifts it and affects the outcome only through it. It estimates the return to schooling with Card's college-proximity instrument: OLS, the first stage and its strength, the reduced form, 2SLS, weak-instrument-robust inference, the endogeneity test, and what the estimate means.
---

## The question and why OLS fails

**What is the return to a year of schooling?** A regression of log wages on years of education gives an answer, but not a causal one: people who study longer may also be more able or more motivated, and those traits raise wages on their own. Ability is not in the data, so OLS is biased (most likely upwards).

Card (1995) proposed an instrument: whether a man **grew up near a four-year college**. Living near a college lowers the cost of attending, so it raises schooling; the argument is that, given region and urban residence, it affects wages only through schooling.

\[
\text{First stage: } educ_i = \pi\, nearc4_i + X_i'\gamma + v_i \qquad \text{Second stage: } \ln w_i = \beta\, \widehat{educ}_i + X_i'\delta + u_i
\]

| Variable | Role | Definition |
|---|---|---|
| `lwage` | outcome | Log hourly wage, 1976 |
| `educ` | endogenous treatment | Years of schooling |
| `nearc4` | instrument | 1 if a four-year college was in the county at age 14 |
| `black`, `smsa`, `south`, `married`, `exper` | controls | Race, urban residence, region, marital status, experience |

<div class="read" markdown="1">
**The three conditions**

- **Relevance:** the instrument moves schooling. Testable: the first stage.
- **Exclusion:** proximity affects wages only through schooling. Not testable. Threat: families who live near colleges may differ in ways that also raise wages; that is why urban and regional controls matter.
- **Independence:** proximity is as good as random given the controls. Not testable. It must be argued.

A paper's IV section stands or falls on the argument for exclusion and independence, not on the regression output.
</div>

<div class="write" markdown="1">
**How to write it: identification**

Because schooling is likely correlated with unobserved ability, we instrument years of education with an indicator for growing up near a four-year college, following Card (1995). Proximity lowers the cost of college attendance and so raises schooling; conditional on race, region, urban residence, marital status and experience, we assume it affects wages only through its effect on schooling.
</div>

## Load the data and estimate OLS

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

df = pd.read_csv("card.csv").dropna(subset=["lwage", "educ", "nearc4", "black", "smsa", "south", "married", "exper"])
controls = ["black", "smsa", "south", "married", "exper"]
W = sm.add_constant(df[controls])
print(f"{len(df):,} men; {df['nearc4'].mean():.0%} grew up near a four-year college")

ols = sm.OLS(df["lwage"], W.join(df["educ"])).fit(cov_type="HC1")
print(f"OLS return to schooling: {ols.params['educ']:.4f} (s.e. {ols.bse['educ']:.4f})")
```

<div class="read" markdown="1">
**How to read it**

OLS says each additional year of schooling is associated with wages about 7% higher, very precisely estimated. Precision is not the problem; bias is. The IV estimate will tell us how much of this association is causal.
</div>

## The first stage: is the instrument relevant?

```python
fs = sm.OLS(df["educ"], W.join(df["nearc4"])).fit(cov_type="HC1")
F = fs.tvalues["nearc4"] ** 2
print(f"Effect of growing up near a college on schooling: {fs.params['nearc4']:.3f} years (s.e. {fs.bse['nearc4']:.3f})")
print(f"First-stage F statistic (robust): {F:.1f}")

partial = sm.OLS(df["educ"], W).fit()
print(f"Partial R-squared of the instrument: {1 - fs.ssr / partial.ssr:.4f}")
```

<div class="read" markdown="1">
**How to read it**

Men who grew up near a four-year college completed about a third of a year more schooling, holding the controls fixed. The robust F statistic is about 16.5, above the Staiger–Stock rule of thumb of 10, so the instrument is not weak in the traditional sense. It is not strong either: it explains only half a percent of the variation in schooling left after the controls. Lee, McCrary, Moreira and Porter (2022) show that conventional 2SLS t-tests have correct size only with F above about 105, which is why the Anderson–Rubin test below matters.
</div>

## The reduced form and 2SLS

```python
rf = sm.OLS(df["lwage"], W.join(df["nearc4"])).fit(cov_type="HC1")
print(f"Reduced form: effect of proximity on log wages = {rf.params['nearc4']:.4f} (s.e. {rf.bse['nearc4']:.4f})")
print(f"Ratio reduced form / first stage = {rf.params['nearc4'] / fs.params['nearc4']:.4f}")

iv = IV2SLS(df["lwage"], W, df["educ"], df["nearc4"]).fit(cov_type="robust")
print(f"\n2SLS return to schooling: {iv.params['educ']:.4f} (s.e. {iv.std_errors['educ']:.4f}), 95% CI [{iv.conf_int().loc['educ', 'lower']:.3f}, {iv.conf_int().loc['educ', 'upper']:.3f}]")
```

<div class="read" markdown="1">
**How to read it**

- **Reduced form.** Men who grew up near a college earn about 4% more. Show this regression: it is the causal effect of the instrument itself, and if it is zero there is nothing to scale up.
- **2SLS.** With one instrument, the 2SLS estimate is exactly the reduced form divided by the first stage: about 0.12, a return of 12% per year of schooling, **larger** than OLS.
- **Precision.** The 2SLS standard error is more than ten times the OLS one. The confidence interval is wide.

Why would the IV estimate exceed OLS when ability bias pushes OLS up? Because the IV estimate is a **local average treatment effect** (LATE): the return for men whose schooling was changed by college proximity, typically from families for whom cost was a real barrier. Their return to schooling may be higher than average. Measurement error in schooling, which biases OLS towards zero, is another explanation.
</div>

## Weak-instrument-robust inference

```python
from scipy import stats
def ar_test(b0):
    r = sm.OLS(df["lwage"] - b0 * df["educ"], W.join(df["nearc4"])).fit(cov_type="HC1")
    return r.tvalues["nearc4"] ** 2

print(f"Anderson-Rubin test of no effect: F = {ar_test(0):.2f}, p-value = {stats.f.sf(ar_test(0), 1, len(df) - W.shape[1] - 1):.4f}")
grid = np.linspace(-0.1, 0.4, 1001)
crit = stats.f.ppf(0.95, 1, len(df) - W.shape[1] - 1)
inside = grid[[ar_test(b) <= crit for b in grid]]
print(f"95% Anderson-Rubin confidence set: [{inside.min():.3f}, {inside.max():.3f}]")
```

<div class="read" markdown="1">
**How to read it**

The Anderson–Rubin test has the correct size **however weak the instrument**. It tests each candidate value of the return \(b_0\) by asking whether the instrument still predicts \(\ln w - b_0 \cdot educ\); the confidence set collects the values that are not rejected. Here it rejects a zero return, and the 95% set runs from about 0.03 to 0.25: wider than the conventional 2SLS interval, but excluding zero. This is the interval to report when the first stage is not overwhelmingly strong.
</div>

## Is schooling really endogenous?

```python
df["v_hat"] = fs.resid
cf = sm.OLS(df["lwage"], W.join(df[["educ", "v_hat"]])).fit(cov_type="HC1")
print(f"Control-function (Durbin-Wu-Hausman) test: t = {cf.tvalues['v_hat']:.2f}, p-value = {cf.pvalues['v_hat']:.3f}")
print(f"(OLS {ols.params['educ']:.3f} against 2SLS {iv.params['educ']:.3f})")
```

<div class="read" markdown="1">
**How to read it**

The test adds the first-stage residual to the OLS regression: if schooling were exogenous, the residual would add nothing. It does not reject. That does not prove schooling is exogenous: with an imprecise 2SLS estimate the test has little power, and the gap between 0.07 and 0.12 is economically large. With one instrument, the exclusion restriction cannot be tested at all (no overidentification test); Card's full dataset adds proximity to a two-year college, which allows a Hansen J test.
</div>

## Write the paper

<div class="write" markdown="1">
**How to write it: results**

OLS estimates imply a return to schooling of about 7 percent per year. Instrumenting schooling with proximity to a four-year college, the 2SLS estimate is about 12 percent (robust standard error 0.05). The first stage is significant, with a robust F statistic of 16.5; because conventional 2SLS inference can be unreliable at this instrument strength, we also report the Anderson–Rubin 95 percent confidence set, which excludes zero. The 2SLS estimate is a local average treatment effect for men whose schooling was affected by college proximity, and its exceeding OLS is consistent with higher returns for individuals facing cost barriers to college, or with measurement error in schooling.
</div>

<div class="warn" markdown="1">
**Common mistakes**

- Treating the first-stage F above 10 as the end of the weak-instrument question. Report Anderson–Rubin intervals.
- Not reporting the first stage and the reduced form.
- Arguing exclusion with a statistical test. With one instrument there is none; even with several, the Hansen test only checks that instruments agree.
- Interpreting 2SLS as an average effect for everyone. It is local to the compliers.
- Computing the second stage by hand with OLS on fitted values: the standard errors are wrong. Use an IV routine.
</div>
