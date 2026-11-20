title: Panel Logit and Probit (Pooled, Random and Fixed Effects)
short: Panel logit and probit
when: A binary outcome observed for the same units over several periods (union membership, employment, default, adoption) and unobserved unit differences that may matter
data: 545 men observed for 8 years (1980-1987), union membership (Vella and Verbeek 1998; Wooldridge)
tools: Python · statsmodels · scipy · OES engine
oes: paneldiscrete
lead: Use this guide when a yes/no outcome is observed repeatedly for the same people. A pooled logit ignores that observations of one person are related; a random-effects model adds a person-specific intercept that is independent of the regressors; a fixed-effects (conditional) logit removes the person effect without assuming anything about it, at the cost of using only people whose outcome changes. This guide takes the union-membership panel through the pooled, random-effects, correlated-random-effects and fixed-effects estimators, tests the random-effects assumption with the Mundlak test, and shows what each estimator can and cannot tell you, with sample text for each part of the paper.
---

## The question and the models

**Does getting married change whether a man belongs to a union, once we allow for stable personal traits?** Let \(y_{it}\) be 1 if man \(i\) is in a union in year \(t\). The latent-variable model with a person effect \(c_i\) is

\[
y_{it} = 1\{x_{it}'\beta + c_i + \varepsilon_{it} > 0\}, \qquad \varepsilon_{it} \sim \text{logistic}.
\]

*Pooled* logit sets \(c_i = 0\) (or treats it as part of the error) and corrects the standard errors for clustering. *Random effects* assumes \(c_i \sim N(0,\sigma_c^2)\) independent of \(x_{it}\). *Fixed effects* leaves \(c_i\) free and removes it by conditioning on the number of years the person is a union member (Chamberlain 1980). The *Mundlak* version of random effects adds each person's average of the time-varying regressors, which allows \(c_i\) to be correlated with them.

| Variable | Definition |
|---|---|
| `union` | 1 if the man is a union member in the year |
| `married` | 1 if married (changes over time) |
| `educ`, `black`, `hisp` | years of schooling, race and ethnicity (time-invariant) |
| `exper`, `expersq` | labor-market experience and its square (change every year) |

<div class="read" markdown="1">
**Before you estimate anything**

1. **How much does the outcome change within people?** Fixed-effects logit uses only people whose outcome changes at least once. If few do, the estimator is imprecise.
2. **Which regressors vary within people?** Time-invariant variables (education, race) are absorbed by a fixed effect and cannot be estimated.
3. **Is the person effect plausibly correlated with the regressors?** If so, pooled and random-effects estimates are inconsistent for the within-person effect.
4. **What do you want to report?** Fixed-effects logit gives odds ratios but no probabilities or marginal effects; random effects and pooled give average partial effects.
</div>

<div class="write" markdown="1">
**How to write it: model section**

We model the probability of union membership for 545 men observed in eight consecutive years. We estimate pooled logit with standard errors clustered by individual, a random-effects logit with a normally distributed individual effect, a correlated random-effects (Mundlak) logit that adds the individual averages of the time-varying regressors, and Chamberlain's conditional fixed-effects logit. The Mundlak test and a comparison of the within-person coefficients across estimators indicate how much unobserved heterogeneity matters.
</div>

## Load the panel and check the within variation

```python
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats, optimize
from statsmodels.tools.numdiff import approx_hess

d = pd.read_csv("wagepan_union.csv")
g = d.groupby("nr")
print(f"{len(d)} observations on {d.nr.nunique()} men, {len(d) // d.nr.nunique()} years each; union share {d.union.mean():.1%}")
m = g.union.mean()
print(f"Men never in a union: {(m == 0).mean():.1%}; always in a union: {(m == 1).mean():.1%}; switching at least once: {((m > 0) & (m < 1)).mean():.1%}")
print("\nShare of variation that is between men (variance of person means / total variance):")
for c in ["union", "married", "educ", "exper"]:
    print(f"  {c:8s} {g[c].mean().var() / d[c].var():.2f}")
print("Number of union transitions per man (0, 1, 2, ...):", g.union.apply(lambda s: int((s.diff().abs() == 1).sum())).value_counts().sort_index().to_dict())
```

<div class="read" markdown="1">
**How to read it**

Only 45% of men change union status at some point; the other 55% are always or never members and carry no information for the fixed-effects logit. Union status is a persistent trait: 59% of its variance is between men, and marital status is similar (58%). Experience changes every year (34% between men), while education never changes (it is entirely between-person).
</div>

## Step 1: Pooled logit with clustered standard errors

```python
xs = ["married", "educ", "black", "hisp", "exper", "expersq"]
X = sm.add_constant(d[xs])
pooled = sm.Logit(d.union, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.nr})
naive = sm.Logit(d.union, X).fit(disp=0)
print(pd.DataFrame({"coef": pooled.params, "cluster s.e.": pooled.bse, "naive s.e.": naive.bse, "z": pooled.tvalues}).round(4).to_string())
print("\nAverage marginal effects (pooled):")
print(naive.get_margeff().summary_frame()[["dy/dx", "Std. Err."]].round(4).to_string())
```

<div class="read" markdown="1">
**How to read it**

The pooled estimates treat the 4,360 observations as independent. The clustered standard errors are up to twice the naive ones (for black, 0.22 against 0.10), since the eight observations of a man are strongly related. Married men are 5 percentage points more likely to be in a union, and Black men 15 points more likely. This is a population-average relation: it compares different men and makes no allowance for stable unobserved differences.
</div>

## Step 2: Random-effects logit

```python
GX = np.linspace(-6, 6, 61); GW = stats.norm.pdf(GX) * (GX[1] - GX[0])      # nodes and weights for the standard-normal person effect
gi = pd.factorize(d.nr)[0]; ng = gi.max() + 1; Xn = X.to_numpy(float); y = d.union.to_numpy(float); q = 2 * y - 1
def re_nll(t, Xm=Xn):
    from scipy.special import logsumexp
    b, s = t[:-1], np.exp(t[-1])
    z = q[:, None] * ((Xm @ b)[:, None] + s * GX[None, :])
    lg = np.zeros((ng, len(GX))); np.add.at(lg, gi, -np.logaddexp(0, -z))
    return -logsumexp(lg + np.log(GW)[None, :], axis=1).sum()

def fit_re(Xm):
    t0 = np.r_[sm.Logit(y, Xm).fit(disp=0).params, 0.0]
    o = optimize.minimize(re_nll, t0, args=(Xm,), method="BFGS", options={"gtol": 1e-6})
    V = np.linalg.inv(approx_hess(o.x, lambda t: re_nll(t, Xm)))
    return o.x, np.sqrt(np.diag(V)), -o.fun, V

th, se_re, ll_re, V_re = fit_re(Xn)
sig = np.exp(th[-1]); rho = sig ** 2 / (sig ** 2 + np.pi ** 2 / 3)
tab = pd.DataFrame({"pooled": pooled.params.values, "random effects": th[:-1], "RE s.e.": se_re[:-1]}, index=pooled.params.index)
print(tab.round(3).to_string())
print(f"\nsigma_u = {sig:.3f}, intraclass correlation of the latent variable = {rho:.3f}")
lr = 2 * (ll_re - naive.llf)
print(f"LR test of sigma_u = 0: chi2 = {lr:.0f}, p = {0.5 * stats.chi2.sf(lr, 1):.1e}")
```

<div class="read" markdown="1">
**How to read it**

The unit effect is huge: sigma_u is 3.0, which means that 73% of the latent variance is a stable personal trait, and the likelihood-ratio test rejects its absence with overwhelming force. The random-effects coefficients are on a different scale from the pooled ones (they are conditional on the person effect, roughly 2 times larger for variables with a strong signal, such as race). They are not comparable with the pooled coefficients unless converted to average partial effects. The random-effects model assumes the person effect is unrelated to marriage and experience, which the next steps test.
</div>

## Step 3: Allow the person effect to correlate with the regressors (Mundlak)

```python
tv = ["married", "exper", "expersq"]                                      # variables that change within men
Xm = np.column_stack([Xn] + [g[c].transform("mean").to_numpy() for c in tv])
th_c, se_c, ll_c, V_c = fit_re(Xm)
nm = len(tv); bm = th_c[Xn.shape[1]:-1]; Vm = V_c[Xn.shape[1]:-1, Xn.shape[1]:-1]
W = bm @ np.linalg.solve(Vm, bm)
cols = ["const"] + xs + ["mean " + c for c in tv]
print(pd.DataFrame({"coef": th_c[:-1], "s.e.": se_c[:-1], "z": th_c[:-1] / se_c[:-1]}, index=cols).round(3).to_string())
print(f"\nMundlak test (group means jointly zero): chi2({nm}) = {W:.1f}, p = {stats.chi2.sf(W, nm):.4f}")
print(f"log-likelihood: RE {ll_re:.1f}, correlated RE {ll_c:.1f}")
```

<div class="read" markdown="1">
**How to read it**

The averages of experience and its square are strongly significant, and the Mundlak test rejects the random-effects assumption (chi-squared 13.9 on 3 degrees of freedom, p = 0.003). Men with more average experience differ in unobserved ways that are associated with union membership, so the person effect is correlated with the regressors. In the correlated-random-effects model the coefficient on marriage is 0.27, close to the pooled estimate and to what the fixed-effects model will give, but the model still assumes a particular distribution for the person effect.
</div>

## Step 4: Fixed-effects (conditional) logit

```python
from statsmodels.discrete.conditional_models import ConditionalLogit
fe = ConditionalLogit(d.union, d[tv], groups=d.nr).fit(disp=0)
print(f"Men contributing (those who switch): {int(((m > 0) & (m < 1)).sum())} of {d.nr.nunique()}")
print(pd.DataFrame({"coef": fe.params, "s.e.": fe.bse, "z": fe.tvalues, "odds ratio": np.exp(fe.params)}).round(4).to_string())
print("\nWithin-person coefficients across estimators")
print(pd.DataFrame({"pooled": pooled.params[tv], "random effects": pd.Series(th[:-1], index=pooled.params.index)[tv],
                    "correlated RE": pd.Series(th_c[:-1][:len(cols)], index=cols)[tv], "fixed effects": fe.params}).round(3).to_string())
```

<div class="read" markdown="1">
**How to read it**

Only 246 men switch union status and contribute to the conditional likelihood. For them, getting married multiplies the odds of union membership by 1.31 (95% interval 0.94 to 1.83), not significantly different from 1 (z = 1.57). Experience has no significant within-man effect. The pooled coefficient on experience (0.17) is much larger than the fixed-effects one (0.02): the pooled estimate reflects the comparison between men, where those with more experience belong more often, and the fixed-effects estimate removes this. Marriage is similar in all estimators, so the "marriage premium" is not driven by stable traits.
</div>

<div class="write" markdown="1">
**How to write it: main result**

Pooled logit suggests that married men are 5 percentage points more likely to be in a union (clustered standard error 1.4), but this partly compares different men. The random-effects assumption is rejected by a Mundlak test (chi-squared 13.9, p = 0.003), and the fixed-effects logit, which uses the 246 men who change union status, gives an odds ratio of 1.31 for marriage (95% confidence interval 0.94 to 1.83). The experience effect found in the pooled model disappears within men.
</div>

## Step 5: Average partial effects from the random-effects model

```python
# Average partial effect: integrate the unit effect out, then average over the sample. This gives probabilities, which the fixed-effects model cannot.
def ape(t, Xm, col):
    b, s = t[:-1], np.exp(t[-1]); A1, A0 = Xm.copy(), Xm.copy(); A1[:, col], A0[:, col] = 1, 0
    p = lambda A: ((1 / (1 + np.exp(-((A @ b)[:, None] + s * GX[None, :])))) * GW[None, :]).sum(1)
    return (p(A1) - p(A0)).mean()
print(f"Average partial effect of marriage, correlated RE: {ape(th_c, Xm, 1):+.4f}")
print(f"Average partial effect of marriage, RE:            {ape(th, Xn, 1):+.4f}")
print(f"Pooled average marginal effect:                    {naive.get_margeff().summary_frame().loc['married', 'dy/dx']:+.4f}")
```

<div class="read" markdown="1">
**How to read it**

The random-effects estimate of the average partial effect of marriage is 2.9 percentage points and the correlated random-effects estimate 2.4 points, much smaller than the pooled 5.0, which is a population-average comparison that includes the stable differences. Only the models with a distributional assumption for the person effect give probabilities; the fixed-effects model gives the odds ratio alone.
</div>

## What to report in the paper

Report the number of units and periods, the share of units whose outcome changes, the four estimators for the same set of time-varying variables, the Mundlak test, sigma_u or the intraclass correlation, and what each estimate answers: the pooled and RE estimates a population-average effect, the fixed-effects odds ratio the within-person effect.

<div class="write" markdown="1">
**How to write it: results and robustness**

Table 3 reports the four estimators. The unit effect is large (sigma = 3.0, intraclass correlation 0.73) and correlated with experience (Mundlak p = 0.003), so we base inference on the conditional fixed-effects logit, which relies on the 246 men whose union status changes. The effect of marriage is positive in all estimators (odds ratio 1.31 in the fixed-effects model) but imprecisely estimated in the within-man models. Average partial effects of marriage from the correlated random-effects and random-effects specifications are 2.4 and 2.9 percentage points, against 5.0 for the pooled model.
</div>

## Common mistakes

<div class="warn" markdown="1">
**Mistakes to avoid**

1. **Using plain pooled standard errors.** Observations of the same unit are dependent; cluster by unit.
2. **Comparing random-effects and pooled coefficients directly.** They are on different scales; compare average partial effects.
3. **Assuming random effects without a test.** Run the Mundlak test or a Hausman test.
4. **Estimating a fixed-effects probit with dummy variables.** With few periods this is biased (the incidental-parameters problem); use conditional logit.
5. **Wondering why time-invariant variables vanish.** A fixed effect absorbs them; use correlated random effects if you need them.
6. **Ignoring the units that drop out.** Fixed-effects logit uses only units whose outcome changes, so it describes switchers.
7. **Reporting marginal effects from a fixed-effects logit.** They are not identified without assumptions about the effect.
8. **Too few quadrature points.** The random-effects likelihood is an integral; check that the result is stable.
</div>
