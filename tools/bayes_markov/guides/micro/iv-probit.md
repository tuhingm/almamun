title: IV Probit and the Control Function (Endogenous Regressors in a Binary Model)
short: IV probit
when: A binary outcome (employment, participation, adoption) with a regressor that is correlated with the error, and a valid instrument for it
data: Mroz (1987): labor-force participation of 753 married women, with education instrumented by the education of the parents
tools: Python · statsmodels · scipy · OES engine
oes: ivprobit
lead: Use this guide when you want the effect of a regressor on a yes/no outcome and the regressor is endogenous: chosen by people who differ in unobserved ways that also affect the outcome, or measured with error. Two-stage least squares does not carry over to a probit, because the nonlinear index cannot simply be replaced by its fitted value. The control-function approach (Rivers and Vuong 1988) adds the first-stage residual to the probit; the same model can be estimated by maximum likelihood. This guide runs both on the Mroz data, tests exogeneity, shows how to compute average effects that are comparable with the ordinary probit, and warns about weak instruments, with sample text for each part of the paper.
---

## The question and the model

**Does schooling raise the probability that a married woman is in the labor force?** Years of education `educ` is plausibly correlated with unobserved ability and motivation that also drive employment. With an instrument \(z\) (the parents' education) that moves education but has no direct effect on participation, the model is

\[
y_i = 1\{\beta_0 + x_i'\beta + \gamma\,e_i + u_i > 0\}, \qquad e_i = z_i'\pi + v_i, \qquad (u_i, v_i) \text{ jointly normal},\; \operatorname{Corr}(u_i, v_i) = \rho .
\]

If \(\rho \neq 0\), ordinary probit is inconsistent. The control function writes \(u_i = \rho v_i/\sigma_v + \text{noise}\), so adding the first-stage residual \(\hat v_i\) to the probit "controls" for the endogeneity, and the coefficient on \(\hat v_i\) tests exogeneity (\(H_0: \rho = 0\)).

| Variable | Definition |
|---|---|
| `inlf` | 1 if the woman was in the labor force in 1975 |
| `educ` | years of education (the endogenous regressor) |
| `motheduc`, `fatheduc` | years of education of the mother and father (excluded instruments) |
| `exper`, `expersq`, `age` | labor-market experience, its square, age |
| `kidslt6`, `kidsge6`, `nwifeinc` | children under 6 and aged 6 to 18, other household income |

<div class="read" markdown="1">
**Before you estimate anything**

1. **Is there an instrument that is relevant and excluded?** Relevant means it predicts education in the first stage (check the F statistic); excluded means it has no direct effect on participation. Parents' education satisfies the first condition easily; the second is a debatable assumption (educated parents may pass on other advantages).
2. **How many instruments?** At least one per endogenous regressor. With more, you can test overidentification only in a linear model; for the probit, the check is informal.
3. **What do you report?** Coefficients depend on the scale of the error, so report average effects on the probability.
</div>

<div class="write" markdown="1">
**How to write it: model section**

We estimate the effect of education on the probability of labor-force participation of 753 married women. Because education may be correlated with unobserved ability, we instrument it with the education of the woman's mother and father and estimate a probit with a control function (Rivers and Vuong, 1988), with standard errors that account for the estimated first-stage residual. We test exogeneity of education with the coefficient on the first-stage residual, check the strength of the instruments in the first stage, and compare average effects with those of an ordinary probit.
</div>

## Step 1: The first stage

```python
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats, optimize
from statsmodels.tools.numdiff import approx_hess

d = pd.read_csv("mroz_iv.csv"); n = len(d)
xs = ["exper", "expersq", "age", "kidslt6", "kidsge6", "nwifeinc"]; inst = ["motheduc", "fatheduc"]
Z = sm.add_constant(d[xs + inst]); X0 = sm.add_constant(d[xs])
fs = sm.OLS(d.educ, Z).fit(); fs_r = sm.OLS(d.educ, X0).fit()
F = ((fs_r.ssr - fs.ssr) / 2) / (fs.ssr / fs.df_resid)
print(pd.DataFrame({"coef": fs.params, "s.e.": fs.bse}).loc[inst].round(4).to_string())
print(f"\nFirst stage: F test of the two instruments = {F:.1f} (p = {stats.f.sf(F, 2, fs.df_resid):.1e}); R-squared {fs.rsquared:.3f} (without instruments {fs_r.rsquared:.3f})")
d["vhat"] = fs.resid
```

<div class="read" markdown="1">
**How to read it**

Each year of the mother's or father's schooling raises the woman's schooling by 0.17 and 0.16 years, and the F statistic of the two instruments is 96, far above the rule-of-thumb value of 10: the instruments are strong. The first-stage residual `vhat` is the part of education that the instruments and the controls do not explain, and it is the "control function".
</div>

## Step 2: Ordinary probit, then the control-function probit

```python
y = d.inlf
plain = sm.Probit(y, sm.add_constant(d[xs + ["educ"]])).fit(disp=0)
cf = sm.Probit(y, sm.add_constant(d[xs + ["educ", "vhat"]])).fit(disp=0)
print(pd.DataFrame({"probit": plain.params, "control function": cf.params, "naive s.e. (CF)": cf.bse}).round(4).to_string())
print(f"\nExogeneity test, coefficient on vhat: z = {cf.tvalues['vhat']:.2f}, p = {cf.pvalues['vhat']:.3f}")
```

<div class="read" markdown="1">
**How to read it**

The coefficient on education rises from 0.131 in the ordinary probit to 0.150 once the first-stage residual is included, and the residual itself is small and not significant (z = −0.40, p = 0.69): there is no sign that education is endogenous for participation here, so the ordinary probit is consistent and the instrumented estimate is just less precise. The test of the residual is a valid exogeneity test with the naive standard error; the other standard errors in the second stage need a correction, because \(\hat v_i\) is estimated.
</div>

## Step 3: Standard errors that account for the first stage (bootstrap)

```python
def cf_fit(df):
    Zb = sm.add_constant(df[xs + inst]); v = sm.OLS(df.educ, Zb).fit().resid
    W = sm.add_constant(df[xs + ["educ"]].assign(vhat=v)); m = sm.Probit(df.inlf, W).fit(disp=0)
    return m.params

rng = np.random.default_rng(0); draws = []
for _ in range(300):
    draws.append(cf_fit(d.iloc[rng.integers(0, n, n)]))
draws = pd.DataFrame(draws)
tab = pd.DataFrame({"estimate": cf.params, "naive s.e.": cf.bse, "bootstrap s.e.": draws.std()})
print(tab.round(4).to_string())
```

<div class="read" markdown="1">
**How to read it**

The bootstrap re-estimates both stages in every sample, so it reflects the uncertainty of the first stage. For education the bootstrap standard error (about 0.05) is of the same order as the naive 0.055; for the other coefficients the two are close. The OES engine computes the correction analytically from the stacked first-stage and probit moment conditions, and it agrees with the bootstrap to within sampling error. Use one of these, not the naive second-stage standard errors, for the exogenous variables and the endogenous one.
</div>

## Step 4: Average effects on the probability

```python
# Average structural function: for a change in x, average the change in P(y = 1) over the sample, holding the first-stage residual at its own value
b = cf.params.to_numpy(); full = sm.add_constant(d[xs + ["educ"]].assign(vhat=d.vhat)).to_numpy()
cols = list(cf.params.index)
def ame_cf(var, h=1e-4):
    i = cols.index(var); A1, A0 = full.copy(), full.copy(); A1[:, i] += h; A0[:, i] -= h
    return ((stats.norm.cdf(A1 @ b) - stats.norm.cdf(A0 @ b)) / (2 * h)).mean()
ame_plain = plain.get_margeff().summary_frame()["dy/dx"]
out = pd.DataFrame({"ordinary probit": ame_plain, "control function": pd.Series({v: ame_cf(v) for v in xs + ["educ"]})})
print(out.round(4).to_string())
print(f"\nOne more year of schooling raises the probability of working by {ame_cf('educ') * 100:.1f} points (control function) against {ame_plain['educ'] * 100:.1f} (ordinary probit)")
```

<div class="read" markdown="1">
**How to read it**

One more year of education raises the probability of working by 4.5 percentage points with the instrument and by 3.9 points in the ordinary probit. The effects of the other variables are almost the same in the two models, as the weak evidence of endogeneity suggests (young children lower the probability by 26 points, other household income by 0.4 points per thousand dollars). The OES engine reports a standard error of 0.016 for the instrumented effect against 0.007 for the ordinary probit: more than twice as large, which is the price of the instrument.
</div>

<div class="write" markdown="1">
**How to write it: main result**

An additional year of education raises the probability that a married woman works by 4.5 percentage points in the instrumented probit (standard error 1.6 points), against 3.9 points (0.7) in the ordinary probit. The coefficient on the first-stage residual is not significant (p = 0.7), so we cannot reject exogeneity of education, and the two estimates are statistically indistinguishable.
</div>

## Step 5: Maximum likelihood for the same model

```python
e = d.educ.to_numpy(float); Xn = sm.add_constant(d[xs]).to_numpy(float); Zn = Z.to_numpy(float); q = 2 * y.to_numpy() - 1
kx, kz = Xn.shape[1], Zn.shape[1]
def nll(t):
    b_, pi = t[:kx + 1], t[kx + 1:kx + 1 + kz]; sv, rho = np.exp(t[-2]), np.tanh(t[-1])
    v = (e - Zn @ pi) / sv; idx = (np.column_stack([Xn, e]) @ b_ + rho * v) / np.sqrt(1 - rho ** 2)
    return -(stats.norm.logpdf(v) - np.log(sv) + stats.norm.logcdf(q * idx)).sum()
t0 = np.r_[plain.params, fs.params, np.log(fs.resid.std()), 0.0]
o = optimize.minimize(nll, t0, method="BFGS", options={"gtol": 1e-8}); o = optimize.minimize(nll, o.x, method="BFGS", options={"gtol": 1e-9})
V = np.linalg.inv(approx_hess(o.x, nll)); se = np.sqrt(np.diag(V)); rho = np.tanh(o.x[-1]); se_rho = (1 - rho ** 2) * se[-1]
print(f"MLE: coefficient on educ {o.x[kx]:.4f} (s.e. {se[kx]:.4f});  control function: {cf.params['educ']:.4f}")
print(f"rho = {rho:.3f} (s.e. {se_rho:.3f}); Wald test of rho = 0: chi2 = {(rho / se_rho) ** 2:.2f}, p = {stats.chi2.sf((rho / se_rho) ** 2, 1):.3f}")
```

<div class="read" markdown="1">
**How to read it**

Maximum likelihood gives almost the same coefficient on education (0.150) as the control function and a correlation rho between the two errors of −0.05 (standard error 0.11), also not significant. The two estimators agree because the model is the same; maximum likelihood is more efficient if normality holds, and the control function is more robust to non-normality of the outcome error. Both rely on the normality of the first-stage error.
</div>

## What to report in the paper

Report the first-stage coefficients of the instruments and the F statistic, the exogeneity test, the second-stage estimates with standard errors that account for the first stage, and average effects with the ordinary probit alongside. State the exclusion argument for each instrument.

<div class="write" markdown="1">
**How to write it: results and robustness**

Table 4 reports the first stage (F = 96) and the instrumented probit. The coefficient on the first-stage residual is small and insignificant (z = −0.4), and a maximum-likelihood estimate of the same model gives a correlation between the errors of −0.05 (standard error 0.11). We therefore cannot reject that education is exogenous. The exclusion restriction (parents' schooling affects participation only through the woman's schooling) is an assumption: it fails if educated parents also transmit labor-market attachment, in which case both instrumented and ordinary estimates would be biased in ways the tests above cannot reveal.
</div>

## Common mistakes

<div class="warn" markdown="1">
**Mistakes to avoid**

1. **Plugging fitted values into a probit ("forbidden regression").** This is inconsistent in nonlinear models. Use the control function or maximum likelihood.
2. **Using the naive second-stage standard errors.** The first-stage residual is estimated. Bootstrap or use the stacked-moment correction.
3. **Comparing coefficients across models.** The scale of the error differs; compare average effects.
4. **A weak instrument.** Check the first-stage F; with a weak instrument the estimates are unstable and biased toward the ordinary probit.
5. **A non-excluded instrument.** No test confirms exclusion when the model is exactly identified; argue it on economic grounds.
6. **Using a binary endogenous regressor in a continuous first stage without caution.** The control-function approach assumes a linear first stage with normal errors; with a binary endogenous variable use a bivariate probit or a different approach.
7. **Reading an insignificant exogeneity test as proof of exogeneity.** It may only reflect low power.
</div>
