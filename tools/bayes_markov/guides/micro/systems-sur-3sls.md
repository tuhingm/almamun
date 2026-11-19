title: Systems of Equations: SUR and Three-Stage Least Squares
short: SUR and 3SLS
when: Several equations whose errors are correlated (time use, expenditure shares, labor supply and wages), or equations that determine each other jointly
data: Time use of 706 people (Biddle and Hamermesh 1990) for SUR; labor supply and wage of 428 working women (Mroz 1987) for 3SLS
tools: Python · numpy · scipy · OES engine
oes: systems
lead: Use this guide when you model several outcomes at once. If the equations look unrelated but the errors are correlated (a person who sleeps less works more), seemingly unrelated regressions (SUR) estimates them jointly and can be more efficient than equation-by-equation OLS. If the outcomes also determine each other (hours worked depends on the wage, the wage on hours), OLS is inconsistent and you need instruments; three-stage least squares (3SLS) combines instrumental variables with the SUR weighting. This guide builds both estimators from scratch, tests whether the equations are really independent, shows when SUR gains nothing, and tests the instruments, with sample text for each part of the paper.
---

## The question and the model

**Two outcomes, one person.** Sleep and paid work compete for the same 168 hours. We write one equation for each:

\[
y_{1i} = x_{1i}'\beta_1 + \varepsilon_{1i}, \qquad y_{2i} = x_{2i}'\beta_2 + \varepsilon_{2i}, \qquad \operatorname{Cov}(\varepsilon_{1i},\varepsilon_{2i}) = \sigma_{12}.
\]

If \(\sigma_{12} \neq 0\) and the regressors differ between the equations, generalized least squares weighted by the estimated error covariance (SUR, Zellner 1962) is more efficient than OLS. If the right-hand side of one equation contains the outcome of another, the regressor is correlated with the error, and 3SLS uses instruments for it.

| Variable | Definition |
|---|---|
| `sleep`, `totwrk` | minutes of sleep and of paid work per week |
| `educ`, `age`, `male` | years of schooling, age, 1 for men |
| `yngkid`, `gdhlth` | 1 if a child under 3 is in the home; 1 if in good health |
| `union`, `south` | 1 if union member; 1 if living in the South |

<div class="read" markdown="1">
**Before you estimate anything**

1. **Are the errors plausibly correlated?** Outcomes that share unmodelled determinants (time, budgets, a common shock) are. A Breusch-Pagan test says whether the correlation is large enough to matter.
2. **Do the equations have different regressors?** With identical regressors in every equation, SUR equals equation-by-equation OLS. The gain comes from the different sets.
3. **Is any regressor an outcome of another equation?** Then OLS and SUR are inconsistent. Use 3SLS with at least one excluded instrument for every endogenous regressor.
</div>

<div class="write" markdown="1">
**How to write it: model section**

Weekly minutes of sleep and of paid work are determined by overlapping factors, so we estimate the two equations as a seemingly unrelated regression system with different regressors in each equation and test for correlated errors with a Breusch-Pagan Lagrange-multiplier test. For the labor-supply system, in which hours depend on the wage and the wage on hours, we estimate the two equations by three-stage least squares with experience, age, children and other household income as instruments, and test the overidentifying restrictions.
</div>

## Load the data and estimate by OLS

```python
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats

d = pd.read_csv("sleep_work.csv"); n = len(d)
x1 = ["educ", "age", "male", "yngkid", "gdhlth"]      # sleep equation
x2 = ["educ", "age", "male", "union", "south"]        # work equation
X1, X2 = sm.add_constant(d[x1]).to_numpy(float), sm.add_constant(d[x2]).to_numpy(float)
y1, y2 = d.sleep.to_numpy(float), d.totwrk.to_numpy(float)
b1 = np.linalg.lstsq(X1, y1, rcond=None)[0]; b2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
e = np.column_stack([y1 - X1 @ b1, y2 - X2 @ b2])
S = e.T @ e / n
print(f"{n} people; mean sleep {y1.mean():.0f} minutes, mean work {y2.mean():.0f} minutes per week")
print("Residual covariance matrix:\n", np.round(S, 0))
print(f"\nCorrelation of the OLS residuals: {np.corrcoef(e.T)[0, 1]:.3f}")
lm = n * np.corrcoef(e.T)[0, 1] ** 2
print(f"Breusch-Pagan LM test of independent equations: chi2(1) = {lm:.1f}, p = {stats.chi2.sf(lm, 1):.2e}")
```

<div class="read" markdown="1">
**How to read it**

The residuals of the two OLS equations are correlated at −0.33, and the Breusch-Pagan test rejects independence decisively (chi-squared 76 on 1 degree of freedom). A person who works unusually much, given the observed characteristics, also sleeps unusually little. This is the situation in which SUR can help, since the two equations contain different regressors (children and health in one, union membership and region in the other).
</div>

## Step 1: SUR by feasible GLS

```python
def sur(Xs, ys, S):
    Si = np.linalg.inv(S); ks = [x.shape[1] for x in Xs]; o = np.r_[0, np.cumsum(ks)]; M = len(Xs)
    A = np.zeros((o[-1], o[-1])); r = np.zeros(o[-1])
    for a in range(M):
        for c in range(M):
            A[o[a]:o[a + 1], o[c]:o[c + 1]] = Si[a, c] * Xs[a].T @ Xs[c]
        r[o[a]:o[a + 1]] = sum(Si[a, c] * Xs[a].T @ ys[c] for c in range(M))
    b = np.linalg.solve(A, r); V = np.linalg.inv(A)
    return b, np.sqrt(np.diag(V)), V

b, se, V = sur([X1, X2], [y1, y2], S)
names = [f"sleep: {c}" for c in ["const"] + x1] + [f"work: {c}" for c in ["const"] + x2]
se_ols = np.r_[np.sqrt(np.diag(np.linalg.inv(X1.T @ X1)) * S[0, 0]), np.sqrt(np.diag(np.linalg.inv(X2.T @ X2)) * S[1, 1])]
out = pd.DataFrame({"OLS": np.r_[b1, b2], "OLS s.e.": se_ols, "SUR": b, "SUR s.e.": se}, index=names)
out["s.e. ratio"] = out["SUR s.e."] / out["OLS s.e."]
print(out.round(3).to_string())
print(f"\nStandard errors: SUR / OLS ratio ranges from {out['s.e. ratio'].min():.3f} to {out['s.e. ratio'].max():.3f}")
```

<div class="read" markdown="1">
**How to read it**

The SUR coefficients are close to the OLS ones, and the standard errors are smaller by 0 to 6%. The largest changes are for regressors that appear in only one equation: the coefficient on a young child in the sleep equation falls from 30 minutes to 1, and the coefficient on living in the South in the work equation rises from 152 to 209 minutes. This is how SUR works: it uses the information in the other equation's residual to adjust each equation's estimates. The gain in precision is modest here, as it usually is. Large gains need strongly correlated errors and very different regressors.
</div>

<div class="write" markdown="1">
**How to write it: main result**

Table 1 reports the SUR estimates. The equation errors are significantly correlated (−0.33; Breusch-Pagan chi-squared 76, p < 0.001), and SUR standard errors are up to 6% smaller than the OLS ones. Men work 729 more minutes per week than women, and people living in the South work 209 more minutes. The estimates for sleep are imprecise: no single regressor has a significant effect at the 5% level.
</div>

## Step 2: When SUR gains nothing

```python
Xc = sm.add_constant(d[["educ", "age", "male"]]).to_numpy(float)         # same regressors in both equations
bs, ses, _ = sur([Xc, Xc], [y1, y2], S)
bo = np.r_[np.linalg.lstsq(Xc, y1, rcond=None)[0], np.linalg.lstsq(Xc, y2, rcond=None)[0]]
print("Largest difference between SUR and OLS coefficients with identical regressors:", np.abs(bs - bo).max().round(8))
```

<div class="read" markdown="1">
**How to read it**

With the same regressors in both equations, SUR and OLS are the same estimator, to numerical precision. A system whose equations have identical right-hand sides is not a case for SUR. The gains come only from the exclusion of different variables from different equations, or from cross-equation restrictions.
</div>

## Step 3: Simultaneous equations and 3SLS

```python
w = pd.read_csv("mroz_workers.csv"); m = len(w)
# hours = a0 + a1*lwage + educ age kidslt6 kidsge6 nwifeinc ;  lwage = b0 + b1*hours + educ exper expersq ;  both lwage and hours are endogenous
XH = sm.add_constant(w[["lwage", "educ", "age", "kidslt6", "kidsge6", "nwifeinc"]]).to_numpy(float)
XW = sm.add_constant(w[["hours", "educ", "exper", "expersq"]]).to_numpy(float)
yH, yW = w.hours.to_numpy(float), w.lwage.to_numpy(float)
Z = sm.add_constant(w[["educ", "age", "kidslt6", "kidsge6", "nwifeinc", "exper", "expersq"]]).to_numpy(float)   # all exogenous variables
PZ = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
XHh, XWh = PZ @ XH, PZ @ XW

ols_h = np.linalg.lstsq(XH, yH, rcond=None)[0]; ols_w = np.linalg.lstsq(XW, yW, rcond=None)[0]
tsls_h = np.linalg.solve(XHh.T @ XH, XHh.T @ yH); tsls_w = np.linalg.solve(XWh.T @ XW, XWh.T @ yW)
e2 = np.column_stack([yH - XH @ tsls_h, yW - XW @ tsls_w]); S2 = e2.T @ e2 / m
b3, se3, V3 = sur([XHh, XWh], [yH, yW], S2)
se_h = np.sqrt(np.diag(np.linalg.inv(XHh.T @ XHh)) * S2[0, 0])
print("Hours equation: coefficient on the log wage")
print(f"  OLS {ols_h[1]:.1f}   2SLS {tsls_h[1]:.1f} (s.e. {se_h[1]:.1f})   3SLS {b3[1]:.1f} (s.e. {se3[1]:.1f})")
print("Wage equation: coefficient on hours (x 1000)")
print(f"  OLS {1000 * ols_w[1]:.3f}   2SLS {1000 * tsls_w[1]:.3f}   3SLS {1000 * b3[XH.shape[1] + 1]:.3f}")
print("Correlation of the 2SLS residuals of the two equations:", np.round(np.corrcoef(e2.T)[0, 1], 3))
```

<div class="read" markdown="1">
**How to read it**

OLS says that the wage has no effect on hours (−17 minutes per unit of log wage), because the wage and hours determine each other and OLS mixes the two. With instruments the labor-supply elasticity becomes large and significant: 2SLS gives 1,545 hours per unit of the log wage (standard error 477) and 3SLS 1,677 (431). The 3SLS standard error is smaller, the gain from using the correlation between the equations. In the wage equation the effect of hours turns from slightly negative with OLS to positive with instruments (0.16 and 0.20 per thousand hours), but it is small and imprecise. The residuals of the two equations are strongly negatively correlated (−0.91), which is a sign that this simultaneous structure is mechanically related by the hours-wage identity and that 3SLS should be compared with 2SLS rather than trusted alone.
</div>

## Step 4: Are the instruments valid?

```python
# Sargan test, equation by equation: regress the 2SLS residuals on all instruments; n * R2 ~ chi2(number of overidentifying restrictions)
def sargan(res, k_endog, k_exog_in_eq):
    u = res - res.mean(); r2 = 1 - ((u - PZ @ u) ** 2).sum() / (u ** 2).sum()
    df = Z.shape[1] - 1 - k_exog_in_eq - k_endog
    return m * r2, df
for nm, res, ke, kx in [("hours equation", e2[:, 0], 1, 5), ("wage equation", e2[:, 1], 1, 3)]:
    st, df = sargan(res, ke, kx)
    print(f"Sargan test, {nm}: chi2({df}) = {st:.2f}, p = {stats.chi2.sf(st, df):.3f}")
# strength of the instruments for the log wage: F test of exper and expersq in the first stage
fs_u = sm.OLS(w.lwage, sm.add_constant(w[["educ", "age", "kidslt6", "kidsge6", "nwifeinc", "exper", "expersq"]])).fit()
fs_r = sm.OLS(w.lwage, sm.add_constant(w[["educ", "age", "kidslt6", "kidsge6", "nwifeinc"]])).fit()
F = ((fs_r.ssr - fs_u.ssr) / 2) / (fs_u.ssr / fs_u.df_resid)
print(f"\nFirst stage for the log wage: F test of the excluded instruments (exper, expersq) = {F:.1f}")
```

<div class="read" markdown="1">
**How to read it**

Neither Sargan test rejects (p = 0.35 for the hours equation and 0.40 for the wage equation), so the extra instruments are not contradicted by the data. The excluded instruments are only moderately strong in the first stage for the wage: the F statistic is 8.3, below the rule-of-thumb value of 10, so the 2SLS and 3SLS estimates may be biased toward OLS and their standard errors understated. Experience is a valid instrument for the hours equation only if it affects hours through the wage and not directly, which is an economic assumption that no test confirms. The Sargan test checks only the consistency of the instruments among themselves.
</div>

## What to report in the paper

For SUR: the equations and their regressors; the residual correlation and its test; the coefficients with standard errors; and the comparison with OLS. For 3SLS: the endogenous variables; the instruments and the argument for exclusion; first-stage strength; the Sargan test; and the 2SLS estimates next to 3SLS.

<div class="write" markdown="1">
**How to write it: results and robustness**

Table 2 reports the labor-supply system. OLS suggests that the wage has no effect on hours, but with experience and family variables as instruments the labor-supply coefficient is 1,677 hours per unit of the log wage by 3SLS (standard error 431) and 1,545 by 2SLS (477). Sargan tests do not reject the overidentifying restrictions (p = 0.35 and 0.40), but the first-stage F statistic for the excluded instruments is 8.3, so we treat the magnitude with caution. The 3SLS and 2SLS estimates have the same sign and similar size, so the conclusion does not depend on the cross-equation weighting.
</div>

## Common mistakes

<div class="warn" markdown="1">
**Mistakes to avoid**

1. **Using SUR with identical regressors.** It reproduces OLS exactly; the gain is zero.
2. **Expecting large gains from SUR.** The gain needs strongly correlated errors and different regressors. It is usually modest.
3. **Using SUR when a regressor is endogenous.** SUR does not solve simultaneity. Use instruments.
4. **Using 3SLS when an equation is not identified.** Each equation needs at least as many excluded instruments as endogenous regressors.
5. **Trusting 3SLS over 2SLS without comparing.** A specification error in one equation contaminates all equations in 3SLS.
6. **Treating the Sargan test as proof of validity.** It tests the consistency of the instruments with one another, not their exogeneity.
7. **Ignoring weak instruments.** Check the first-stage F statistic.
8. **Reading R-squared in 3SLS as a measure of fit.** It can be negative.
</div>
