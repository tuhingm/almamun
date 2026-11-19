title: Feasible GLS for Panels (Parks-Kmenta)
short: FGLS
when: A long, balanced panel where you want efficiency gains from modelling heteroskedasticity, correlation across units and AR(1) errors
data: Gasoline demand in 18 OECD countries, 1960–1978 (Baltagi)
tools: Python · statsmodels · numpy
lead: Use this guide when T is large relative to N and you want to model the error structure instead of only correcting standard errors. It builds the Parks-Kmenta feasible GLS estimator step by step on an OECD gasoline panel (heteroskedastic, then cross-correlated, each with AR(1) errors), compares it with OLS and panel-corrected standard errors, and uses a simulation to show why Beck and Katz warn that its standard errors are overconfident when T is not much larger than N. Sample text is given for each part of the paper.
---

## The question and the model

The model is the same gasoline demand equation as in the PCSE guide,

\[
\ln g_{it}=\beta_0+\beta_1\ln y_{it}+\beta_2\ln p_{it}+\beta_3\ln c_{it}+\varepsilon_{it}.
\]

Feasible GLS (FGLS) assumes that the errors have a known structure up to a few parameters: a variance \(\sigma_i^2\) for each unit, a covariance \(\sigma_{ij}\) between each pair of units in the same period, and an AR(1) coefficient \(\rho\) in time. It estimates these from OLS residuals and then weights the data by the inverse of the estimated error covariance. If the structure is right and well estimated, FGLS is more efficient than OLS. The problem is the word *well*: the N × N cross-covariance matrix has N(N+1)/2 free elements, all estimated from T observations.

<div class="read" markdown="1">
**Before you estimate anything**

FGLS with cross-correlation needs a balanced panel and T of at least N. Beck and Katz (1995) show that Parks FGLS gives standard errors that are far too small unless T is at least about three times N. Check this rule first, then decide whether to model only the unit variances (always feasible) or the full covariance structure.
</div>

<div class="write" markdown="1">
**How to write it: model section**

We estimate the gasoline demand equation by feasible generalized least squares, allowing for panel-specific heteroskedasticity, contemporaneous correlation across countries and a common AR(1) process in the errors (Parks 1967; Kmenta 1986). Because the efficiency gains of FGLS depend on a good estimate of the error covariance, we compare it with OLS with panel-corrected standard errors.
</div>

## Load the data and check the sample size rule

```python
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats

d = pd.read_csv("gasoline.csv").sort_values(["country", "year"]).reset_index(drop=True)
N, T = d.country.nunique(), d.year.nunique()
y = d.lgaspcar.values
X = sm.add_constant(d[["lincomep", "lrpmg", "lcarpcap"]])
names = list(X.columns); Xm = X.values
print(f"N = {N}, T = {T}, balanced: {len(d) == N * T}")
print(f"Full covariance needs T >= N: {T >= N};  Beck-Katz rule T >= 3N: {T >= 3 * N} (3N = {3 * N})")
print(f"Free covariance parameters to estimate: {N * (N + 1) // 2} from {N * T} residuals")
```

<div class="read" markdown="1">
**How to read it**

The panel is balanced with N = 18 and T = 19, so the full covariance estimator can be computed (T is at least N). But it has 171 free covariance parameters estimated from 342 residuals, and the sample is far from the Beck-Katz threshold of T of at least 3N = 54. This is a case where FGLS runs without complaint and its output should still not be believed.
</div>

## Step 1: OLS, rho and Prais-Winsten transformation

```python
def pw(Y, Xm, N, T, rho):
    Yz, Xz = Y.reshape(N, T), Xm.reshape(N, T, -1)
    Ys = np.concatenate([np.sqrt(1 - rho ** 2) * Yz[:, :1], Yz[:, 1:] - rho * Yz[:, :-1]], axis=1)
    Xs = np.concatenate([np.sqrt(1 - rho ** 2) * Xz[:, :1, :], Xz[:, 1:, :] - rho * Xz[:, :-1, :]], axis=1)
    return Ys, Xs                                              # shapes (N, T) and (N, T, k)

ols = sm.OLS(y, Xm).fit()
E0 = ols.resid.reshape(N, T)
rho = (E0[:, 1:] * E0[:, :-1]).sum() / (E0[:, :-1] ** 2).sum()
Ys, Xs = pw(y, Xm, N, T, rho)
b_pw = np.linalg.lstsq(Xs.reshape(N * T, -1), Ys.ravel(), rcond=None)[0]
Ew = (Ys - (Xs @ b_pw))                                        # (N, T) transformed residuals
print(f"OLS residual AR(1) coefficient rho = {rho:.3f}")
print(pd.DataFrame({"OLS": ols.params, "Prais-Winsten OLS": b_pw}, index=names).round(3).to_string())
```

<div class="read" markdown="1">
**How to read it**

The first stage reproduces the PCSE guide: the OLS residuals have an AR(1) coefficient of 0.923, and removing it moves the income elasticity from 0.89 to 0.41 and the price elasticity from −0.89 to −0.39.
</div>

## Step 2: the error covariance and GLS

```python
def gls(Ys, Xs, Sigma):
    """Parks GLS: weight by Sigma^-1 within each period. Ys: (N,T), Xs: (N,T,k)."""
    Si = np.linalg.inv(Sigma)
    k = Xs.shape[2]
    A = np.zeros((k, k)); c = np.zeros(k)
    for t in range(Xs.shape[1]):
        Xt = Xs[:, t, :]
        A += Xt.T @ Si @ Xt; c += Xt.T @ Si @ Ys[:, t]
    V = np.linalg.inv(A)
    return V @ c, V

S_hom = np.eye(N) * (Ew ** 2).mean()
S_het = np.diag((Ew ** 2).mean(axis=1))                       # panel-specific variances
S_cor = Ew @ Ew.T / T                                         # full N x N cross-covariance
res = {}
for lab, S in [("iid", S_hom), ("heteroskedastic", S_het), ("correlated (Parks)", S_cor)]:
    bb, V = gls(Ys, Xs, S)
    res[lab] = (bb, np.sqrt(np.diag(V)))
tab = pd.DataFrame({f"{k} {w}": v[i] for k, v in res.items() for i, w in [(0, "coef"), (1, "se")]}, index=names)
print(tab.round(3).T.to_string())
print(f"\nrank of the estimated cross-covariance matrix: {np.linalg.matrix_rank(S_cor)} (N = {N}); condition number {np.linalg.cond(S_cor):.0f}")
```

<div class="read" markdown="1">
**How to read it**

Modelling more of the error structure changes the standard errors dramatically. The price standard error is 0.033 with identical weights, 0.029 with country-specific variances, and 0.004 once the full cross-covariance is used. The coefficients barely move (price −0.39, −0.34, −0.39), so nothing in the data has become ten times more informative; the estimated covariance matrix has simply been treated as known. Its condition number is about 8,000, a sign that it is nearly singular.
</div>

## Compare with PCSE

```python
def pcse(Xs, Ee, N, T):
    S = Ee @ Ee.T / T
    meat = sum(Xs[:, t, :].T @ S @ Xs[:, t, :] for t in range(T))
    A = sum(Xs[:, t, :].T @ Xs[:, t, :] for t in range(T))
    Ai = np.linalg.inv(A)
    return Ai @ meat @ Ai

se_pcse = np.sqrt(np.diag(pcse(Xs, Ew, N, T)))
comp = pd.DataFrame({"PW coef": b_pw, "PCSE": se_pcse, "Parks FGLS coef": res["correlated (Parks)"][0],
                     "Parks SE": res["correlated (Parks)"][1]}, index=names)
comp["Parks SE / PCSE"] = comp["Parks SE"] / comp["PCSE"]
print(comp.round(3).to_string())
```

<div class="read" markdown="1">
**How to read it**

The Parks coefficients are almost identical to Prais-Winsten OLS, but the Parks standard errors are only 10 to 25 per cent of the PCSE (0.095 for the price). Both cannot be right. Beck and Katz show that it is the Parks errors that are too small, and the simulation below confirms it.
</div>

## Why the Parks standard errors cannot be trusted here: a simulation

```python
rng = np.random.default_rng(7)
def simulate(N, T, reps=300, rho=0.5):
    cover_p, cover_c = [], []
    beta = 1.0
    for _ in range(reps):
        f = rng.normal(size=(T, 1))                           # common shock in the errors creates cross-correlation
        g = rng.normal(size=(T, 1))                           # a separate common factor in x, so x is not correlated with the error
        x = rng.normal(size=(N, T)) + g.T
        e = rng.normal(size=(N, T)) + 0.8 * f.T
        for t in range(1, T):
            e[:, t] = rho * e[:, t - 1] + e[:, t]
        Yv = beta * x + e
        Xk = np.stack([np.ones((N, T)), x], axis=2)
        o = np.linalg.lstsq(Xk.reshape(N * T, -1), Yv.ravel(), rcond=None)[0]
        r0 = (Yv - Xk @ o)
        rh = (r0[:, 1:] * r0[:, :-1]).sum() / (r0[:, :-1] ** 2).sum()
        Yp, Xp = pw(Yv.ravel(), Xk.reshape(N * T, -1), N, T, rh)
        bp = np.linalg.lstsq(Xp.reshape(N * T, -1), Yp.ravel(), rcond=None)[0]
        Ep = Yp - Xp @ bp
        S = Ep @ Ep.T / T
        bg, Vg = gls(Yp, Xp, S)
        Vc = pcse(Xp, Ep, N, T)
        cover_p.append(abs(bg[1] - beta) < 1.96 * np.sqrt(Vg[1, 1]))
        cover_c.append(abs(bp[1] - beta) < 1.96 * np.sqrt(Vc[1, 1]))
    return np.mean(cover_p), np.mean(cover_c)

rows = []
for n_, t_ in [(10, 10), (10, 20), (10, 40), (10, 100)]:
    a, c = simulate(n_, t_)
    rows.append({"N": n_, "T": t_, "T / N": t_ / n_, "coverage of 95% interval, Parks FGLS": a, "coverage, PW with PCSE": c})
print(pd.DataFrame(rows).round(3).to_string(index=False))
```

<div class="read" markdown="1">
**How to read it**

The simulated data have N = 10 countries, a common shock in the errors, AR(1) errors with coefficient 0.5 and a true slope of 1. A 95 per cent interval should contain the truth 95 per cent of the time. With Parks FGLS it does so only 18 per cent of the time when T = 10, 71 per cent at T = 20, 88 per cent at T = 40 and 94 per cent at T = 100. Prais-Winsten with PCSE is close to 95 per cent from T = 10 on (93, 94, 92 and 97 per cent). Here T = 19 and N = 18, so the gasoline data sit in the bad region of this table.
</div>

<div class="warn" markdown="1">
**Do not report the Parks errors for this panel**

A narrow confidence interval is not evidence that FGLS is efficient. If the sample does not satisfy T of at least about 3N, report OLS or Prais-Winsten with PCSE, and at most show FGLS with only panel-specific variances as a robustness check.
</div>

## What to report in the paper

| Item | What to give |
|---|---|
| Sample | N, T, balanced, and the ratio T / N |
| Error structure | Which of heteroskedasticity, cross-correlation and AR(1) was modelled, and the estimated rho |
| Estimates | FGLS next to pooled OLS and Prais-Winsten with PCSE |
| Justification | Why T is large enough for the covariance matrix, or why only the diagonal was used |
| Robustness | The same table with PCSE or clustered errors |

<div class="write" markdown="1">
**How to write it: results section**

With N = 18 and T = 19 the sample is far below the T of at least 3N that Beck and Katz (1995) recommend for the Parks estimator. In a simulation with N = 10 and T = 20 its 95 per cent intervals cover the true value in 71 per cent of draws, against 94 per cent for Prais-Winsten with panel-corrected standard errors. We therefore base inference on the latter (price elasticity −0.39, PCSE 0.044) and report FGLS only to show that the coefficients are similar.
</div>

## Common mistakes

<div class="warn" markdown="1">
- **Trusting FGLS because its standard errors are small.** With T not much larger than N they are too small.
- **Running full FGLS on an unbalanced panel.** The cross-covariance is not defined.
- **Forgetting serial correlation.** Without the AR(1) step the GLS weights are wrong for long panels.
- **Assuming FGLS corrects bias.** If regressors are correlated with the errors, FGLS is as inconsistent as OLS.
- **Reporting only one covariance structure.** Show how much the standard errors move across the three structures.
</div>
