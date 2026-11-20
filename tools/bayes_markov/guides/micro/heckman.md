title: Heckman Sample-Selection Model
short: Heckman selection
when: An outcome you observe only for a self-selected part of the sample, such as wages for women who work, and you want the effect for everyone
data: Mroz (1987), 753 married women: wages observed for the 428 who worked in 1975
tools: Python · scipy · statsmodels · OES engine
oes: heckman
lead: Use this guide when the outcome is observed only for people who chose, or were chosen, to be in the sample, and the same unobserved factors may drive both the choice and the outcome. Wages exist only for women who work; if those who work have unusually high unobserved earning ability, OLS on workers is biased. The Heckman model estimates the selection and the outcome equations together, tests whether selection matters at all and shows when the correction is credible. This guide takes the Mroz data from OLS to the full Heckman model, by the two-step method and by maximum likelihood, with sample text for each part of the paper.
---

## The question and the model

**What is the return to a year of schooling for women, including those who do not work?** Wages are observed only for women in the labor force. The Heckman model has two equations:

\[
\text{work}_i = 1\{z_i'\gamma + u_i > 0\}, \qquad \ln w_i = x_i'\beta + \varepsilon_i \ \text{ observed only if work}_i = 1,
\]

with \((u_i,\varepsilon_i)\) jointly normal and correlation \(\rho\). If \(\rho \neq 0\), the workers are a selected sample and OLS on them is biased. The correction adds a term, the inverse Mills ratio, that measures how strongly each worker was selected.

| Variable in the file | Definition |
|---|---|
| `inlf` | 1 if the woman worked in 1975 (selection indicator) |
| `lwage` | log wage (blank for women who did not work) |
| `educ`, `exper`, `expersq` | years of schooling, labor-market experience and its square |
| `age` | age in years |
| `kidslt6`, `kidsge6` | children under 6 and aged 6 to 18 |
| `nwifeinc` | household income other than the woman's own earnings, in thousands |

<div class="read" markdown="1">
**Before you estimate anything**

1. **Do you have an exclusion restriction?** At least one variable should affect selection but not the outcome. Here `kidslt6`, `kidsge6`, `age` and `nwifeinc` affect whether a woman works but are assumed not to affect her wage given education and experience. Without such a variable the model is identified only by the shape of the normal distribution, which is fragile.
2. **Is the exclusion restriction believable?** This is a judgment about the economics, and no test can settle it. State the argument in the paper.
3. **Is the outcome truly missing for the non-selected?** The model is not for outcomes that are zero (use Tobit or a hurdle model).
</div>

<div class="write" markdown="1">
**How to write it: model section**

Wages are observed only for the 428 of 753 women who worked, so OLS on workers may be biased by selection. We estimate a Heckman sample-selection model in which a probit for working is estimated jointly with the log-wage equation. Age, the number of young and older children and other household income enter the selection equation only. We report the two-step estimator with corrected standard errors and the full-information maximum likelihood estimator, and we test whether the errors of the two equations are correlated.
</div>

## Load the data and look at the selected sample

```python
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats, optimize

d = pd.read_csv("mroz.csv")
s = d.inlf == 1
print(f"{len(d)} women; {int(s.sum())} ({s.mean():.0%}) worked, so log wages are observed for {int(d.lwage.notna().sum())}")
print("\nMean of the variables, workers and non-workers")
print(d.groupby("inlf")[["educ", "exper", "age", "kidslt6", "kidsge6", "nwifeinc"]].mean().round(2).rename(index={0: "did not work", 1: "worked"}).T)
```

<div class="read" markdown="1">
**How to read it**

Women who worked are more educated, more experienced, younger, and have far fewer young children (0.14 against 0.37) than those who did not. They are not a random sample of women, and the variables that differ most are experience (13.0 against 7.5 years) and young children. Young children, age and other household income are the candidates for the exclusion restriction because they predict selection.
</div>

## Step 1: OLS on the selected sample

```python
xs = ["educ", "exper", "expersq"]
zs = ["educ", "exper", "expersq", "age", "kidslt6", "kidsge6", "nwifeinc"]
W = d[s]
ols = sm.OLS(W.lwage, sm.add_constant(W[xs])).fit(cov_type="HC1")
print(pd.DataFrame({"coef": ols.params, "std err": ols.bse}).round(4).to_string())
print(f"\nObservations: {int(ols.nobs)}.  Return to a year of schooling: {100 * ols.params['educ']:.1f}%")
```

<div class="read" markdown="1">
**How to read it**

On working women, a year of schooling raises wages by about 10.7% and experience has the usual concave profile. This is the benchmark. The question is whether it describes all women or only those who chose to work.
</div>

## Step 2: The selection equation and the inverse Mills ratio

```python
Z = sm.add_constant(d[zs])
pr = sm.Probit(d.inlf, Z).fit(disp=0)
print(pd.DataFrame({"coef": pr.params, "std err": pr.bse, "z": pr.tvalues}).round(4).to_string())
zg = Z.to_numpy() @ pr.params.to_numpy()
d["imr"] = stats.norm.pdf(zg) / stats.norm.cdf(zg)
print(f"\nInverse Mills ratio among workers: mean {d.imr[s].mean():.3f}, range {d.imr[s].min():.2f} to {d.imr[s].max():.2f}")
print("Correlation of the ratio with education among workers:", round(d.imr[s].corr(d.educ[s]), 3))
```

<div class="read" markdown="1">
**How to read it**

Young children lower the probability of working sharply (−0.87), as do age (−0.053 per year) and other household income (−0.012 per thousand), while education and experience raise it. The inverse Mills ratio is large for women who were unlikely to work but did, and small for women who very likely work. Because the ratio is correlated with education among workers (−0.33), omitting it would bias the education coefficient if the ratio mattered for wages.
</div>

## Step 3: Two-step estimator

```python
# Heckman (1979): add the inverse Mills ratio to the wage equation. OLS standard errors are wrong because the ratio is estimated;
# the correction (Greene) uses the variance of the probit estimates.
X2 = sm.add_constant(W[xs].assign(lam=d.imr[s]))
fit2 = sm.OLS(W.lwage, X2).fit()
e = fit2.resid.to_numpy(); n1 = len(e); bl = fit2.params["lam"]
lam, zgs = d.imr[s].to_numpy(), zg[s]
delta = lam * (lam + zgs)
sig2 = e @ e / n1 + bl ** 2 * delta.mean(); rho2 = bl ** 2 / sig2
X2n = X2.to_numpy(); A = np.linalg.inv(X2n.T @ X2n); D = np.diag(delta)
Q = rho2 * (X2n.T @ D @ Z.to_numpy()[s.values]) @ pr.cov_params().to_numpy() @ (Z.to_numpy()[s.values].T @ D @ X2n)
V2 = sig2 * A @ (X2n.T @ (np.eye(n1) - rho2 * D) @ X2n + Q) @ A
se2 = np.sqrt(np.diag(V2))
t2 = pd.DataFrame({"two-step": fit2.params.values, "corrected s.e.": se2, "naive OLS s.e.": fit2.bse.values, "OLS, no correction": np.r_[ols.params.values, np.nan]}, index=fit2.params.index)
print(t2.round(4).to_string())
print(f"\nt test on the inverse Mills ratio: {bl / se2[-1]:.2f}  (p = {2 * stats.norm.sf(abs(bl / se2[-1])):.3f})")
print(f"sigma = {np.sqrt(sig2):.3f}, rho = {bl / np.sqrt(sig2):.3f}")
```

<div class="read" markdown="1">
**How to read it**

The coefficient on the inverse Mills ratio is 0.032 with a corrected standard error of 0.134 (t = 0.24): there is no evidence that selection biases the wage equation. The education coefficient moves only from 0.107 to 0.109, and the implied correlation between the two errors is close to zero. The corrected standard errors are almost identical to the naive OLS ones (marginally smaller here), because the correction term matters little when rho is small. In this sample, then, OLS on workers is a good guide to the wage equation, a legitimate and useful finding, not a failure of the method.
</div>

## Step 4: Full-information maximum likelihood

```python
Xn = sm.add_constant(d[xs].fillna(W[xs].mean())).to_numpy(); Zn = Z.to_numpy(); y = d.lwage.fillna(0).to_numpy(); sm_ = s.to_numpy()
kz, kx = Zn.shape[1], Xn.shape[1]
def nll(t):
    g, b, sg, rh = t[:kz], t[kz:kz + kx], np.exp(t[-2]), np.tanh(t[-1]); zg_ = Zn @ g; u = (y[sm_] - Xn[sm_] @ b) / sg
    return -(stats.norm.logcdf(-zg_[~sm_]).sum() + (stats.norm.logpdf(u) - np.log(sg) + stats.norm.logcdf((zg_[sm_] + rh * u) / np.sqrt(1 - rh ** 2))).sum())
start = np.r_[pr.params, fit2.params[:-1], np.log(np.sqrt(sig2)), 0.0]
opt = optimize.minimize(nll, start, method="BFGS", options={"gtol": 1e-8})
from statsmodels.tools.numdiff import approx_hess
V = np.linalg.inv(approx_hess(opt.x, nll)); se = np.sqrt(np.diag(V))
b, sg, rh = opt.x[kz:kz + kx], np.exp(opt.x[-2]), np.tanh(opt.x[-1])
ml = pd.DataFrame({"ML coef": b, "ML s.e.": se[kz:kz + kx]}, index=["const"] + xs)
print(ml.round(4).to_string())
se_rho = (1 - rh ** 2) * se[-1]
print(f"\nsigma = {sg:.3f}   rho = {rh:.3f} (s.e. {se_rho:.3f})   log-likelihood = {-opt.fun:.3f}")
# likelihood-ratio test of rho = 0 against independent probit and OLS
ll0 = pr.llf + np.sum(stats.norm.logpdf(ols.resid / np.sqrt(ols.ssr / n1)) - np.log(np.sqrt(ols.ssr / n1)))
lr = 2 * (-opt.fun - ll0)
print(f"LR test of rho = 0: chi2(1) = {lr:.3f}, p = {stats.chi2.sf(lr, 1):.3f}")
```

<div class="read" markdown="1">
**How to read it**

Maximum likelihood estimates both equations at once and is more efficient than the two-step estimator if the normality assumption holds. It gives almost the same wage equation (education 0.108, experience 0.043) with smaller standard errors, and an estimated correlation rho of 0.03 (standard error 0.15). The likelihood-ratio test of rho = 0 gives chi-squared of 0.03 (p = 0.86), so independent equations are not rejected. Two-step and ML agree, which is reassuring.
</div>

## Step 5: How fragile is the correction?

```python
# Without an exclusion restriction the ratio is nearly collinear with the outcome regressors, and the estimates become imprecise.
Z_noexcl = sm.add_constant(d[xs])
pr_n = sm.Probit(d.inlf, Z_noexcl).fit(disp=0)
imr_n = stats.norm.pdf(Z_noexcl.to_numpy() @ pr_n.params) / stats.norm.cdf(Z_noexcl.to_numpy() @ pr_n.params)
fn = sm.OLS(W.lwage, sm.add_constant(W[xs].assign(lam=imr_n[s.values]))).fit()
print("Two-step without an exclusion restriction (selection uses only educ, exper, expersq)")
print(pd.DataFrame({"coef": fn.params, "naive s.e.": fn.bse}).round(3).to_string())
print(f"\nCorrelation of the ratio with the regressors (R-squared of the ratio on them): {sm.OLS(imr_n[s.values], sm.add_constant(W[xs])).fit().rsquared:.3f}")
print(f"With the exclusion restriction:                                               {sm.OLS(d.imr[s], sm.add_constant(W[xs])).fit().rsquared:.3f}")
```

<div class="read" markdown="1">
**How to read it**

When the selection equation uses only the outcome regressors, the inverse Mills ratio is almost a linear function of them, so the coefficients become unstable: the standard error of education is 0.051 against 0.016 with the restriction, and the ratio's own coefficient has a standard error of 0.95. The R-squared of the ratio on the outcome regressors is 0.986 without the restriction and 0.539 with it. The exclusion restriction is what separates selection from the outcome equation. The R-squared of the ratio on the outcome regressors tells you how much room the correction has; report it.
</div>

## What to report in the paper

Report the sample sizes (selected and total); the selection equation; the outcome equation by two-step and maximum likelihood; rho and sigma; the test of rho = 0; the argument for each exclusion restriction; and the OLS estimate for comparison.

<div class="write" markdown="1">
**How to write it: results and robustness**

Table 1 reports the Heckman model for 753 women, 428 of whom worked. Young children, age and other household income strongly affect participation. The wage equation implies a return to schooling of 10.9% (two-step) and 10.8% (maximum likelihood), close to the OLS estimate of 10.7% on workers. The estimated correlation between the errors is 0.03 and a likelihood-ratio test does not reject independence (p = 0.86), so we find no evidence of selection bias in this sample. The result depends on the exclusion restrictions, which we motivate by the fact that family circumstances change the value of time at home but not market productivity.
</div>

## Common mistakes

<div class="warn" markdown="1">
**Mistakes to avoid**

1. **No exclusion restriction.** Identification then rests on normality alone. Always include a variable that shifts selection but not the outcome, and defend it.
2. **A weak or invalid exclusion restriction.** If the excluded variable affects the outcome directly (children and wages through hours), the correction is wrong. No test can prove validity.
3. **Using the naive OLS standard errors in the two-step.** They ignore that the ratio is estimated. Use the corrected ones or maximum likelihood.
4. **Treating a rejected rho = 0 as a causal result.** It says selection matters, and the corrected estimates are only as good as the normality assumption.
5. **Applying Heckman to a corner solution.** Zero hours are not missing data; use Tobit or a hurdle model.
6. **Reading the coefficients of the selection equation as effects on wages.** They are effects on the probability of being in the sample.
7. **Dropping non-selected observations before estimation.** The probit needs them.
</div>
