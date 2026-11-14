title: Spatial Panel Models (SAR, SEM, SDM)
short: Spatial panel
when: Outcomes of neighbouring units are related (spillovers, common regional shocks), and you can define who is a neighbour
data: Output and public capital in 48 U.S. states, 1970–1986 (Munnell)
tools: Python · numpy · scipy
lead: Use this guide when units are not independent because they share a border, a market or a supply chain. It takes one real panel of U.S. states from a map-based weights matrix to a test for spatial dependence in the residuals (Moran's I and the Lagrange-multiplier tests), the three standard models (spatial lag, spatial error, spatial Durbin) estimated by maximum likelihood with state and year effects, the choice between them, and the direct, indirect (spillover) and total effects with simulated standard errors. The code is written out in full. Sample text is given for each part of the paper.
---

## The question and the model

A state's output may depend on the output of its neighbours, through trade, labour mobility or shared shocks, and public infrastructure in one state may help its neighbours. Our question is: **is there spatial dependence in state output after accounting for inputs and fixed effects, and does public capital in one state spill over to its neighbours?** We use the Cobb–Douglas production function with public capital from Munnell (1990).

The three models differ in where the dependence enters. With \(W\) the row-standardised weights matrix and \(\alpha_i,\lambda_t\) state and year effects:

\[
\text{SAR:}\quad y_{it}=\rho\sum_j w_{ij}y_{jt}+x_{it}'\beta+\alpha_i+\lambda_t+\varepsilon_{it},
\qquad
\text{SEM:}\quad y_{it}=x_{it}'\beta+\alpha_i+\lambda_t+u_{it},\ \ u_{it}=\lambda\sum_j w_{ij}u_{jt}+\varepsilon_{it},
\]
\[
\text{SDM:}\quad y_{it}=\rho\sum_j w_{ij}y_{jt}+x_{it}'\beta+\sum_j w_{ij}x_{jt}'\theta+\alpha_i+\lambda_t+\varepsilon_{it}.
\]

In the SAR and SDM a change in one state's regressor affects its own output and, through \(\rho\), its neighbours' output, so the coefficient \(\beta\) is not the marginal effect. Direct, indirect and total effects come from \((I-\rho W)^{-1}(\beta_kI+\theta_kW)\).

| Symbol | Variable in the file | Definition |
|---|---|---|
| \(y_{it}\) | `log_gsp` | log gross state product |
| public capital | `log_pcap` | log public capital stock |
| private capital | `log_pc` | log private capital stock |
| employment | `log_emp` | log employment |
| unemployment | `unemp` | unemployment rate, percent |
| \(W\) | | rook contiguity of the 48 contiguous states, rows sum to one |

<div class="read" markdown="1">
**Before you estimate anything**

The weights matrix is a modelling choice and part of your hypothesis: you decide who the neighbours are before you see the results, and you report how the findings change with another choice. Row-standardised contiguity says that a state's neighbours matter equally and that distance beyond the border does not. Also say whether you expect dependence in the outcome (a lag), in the unobserved shocks (an error) or in the regressors (a Durbin term).
</div>

<div class="write" markdown="1">
**How to write it: model section**

We estimate a production function for 48 U.S. states over 1970–1986 with state and year fixed effects. Spatial dependence is defined by a row-standardised matrix of shared borders. We test for it with Moran's I and the Lagrange-multiplier tests of Anselin et al. (1996) on the residuals of the non-spatial model, and then estimate spatial lag, spatial error and spatial Durbin models by maximum likelihood following Elhorst (2014). We choose among them with likelihood-ratio tests and report direct and indirect effects, with standard errors from 1,000 simulated draws of the parameters.
</div>

## Load the data and build the weights matrix

```python
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar
from statsmodels.tools.numdiff import approx_hess
from us_contiguity import contiguity_matrix

d = pd.read_csv("produc.csv").sort_values(["year", "state"]).reset_index(drop=True)
names = sorted(d.state.unique()); N = len(names); T = d.year.nunique()
xs = ["log_pcap", "log_pc", "log_emp", "unemp"]
W = contiguity_matrix(names)                                    # row-standardised, N x N
A = contiguity_matrix(names, standardise=False)
om = np.linalg.eigvals(W).real
print(f"{N} states x {T} years ({d.year.min()}-{d.year.max()}) = {len(d)} observations")
print(f"Neighbour pairs: {int(A.sum() / 2)}; neighbours per state: mean {A.sum(1).mean():.1f}, min {int(A.sum(1).min())}, max {int(A.sum(1).max())}")
print(f"Eigenvalues of W run from {om.min():.2f} to {om.max():.2f}, so the spatial coefficient must lie between {1 / om.min():.2f} and 1")
print("Most connected:", [names[i] for i in np.argsort(-A.sum(1))[:3]], "| least connected:", [names[i] for i in np.argsort(A.sum(1))[:3]])

Y = d.log_gsp.values.reshape(T, N); X = np.stack([d[v].values.reshape(T, N) for v in xs], -1)     # time x state
twoway = lambda B: B - B.mean(1, keepdims=True) - B.mean(0, keepdims=True) + B.mean((0, 1))
Ys = twoway(Y); Xs = np.stack([twoway(X[:, :, j]) for j in range(4)], -1)
WY = Ys @ W.T; WX = np.stack([Xs[:, :, j] @ W.T for j in range(4)], -1)
```

<div class="read" markdown="1">
**How to read it**

The panel has 48 states and 17 years (816 observations). Alaska, Hawaii and the District of Columbia are not in the data. The weights matrix has 105 neighbour pairs, with 4.4 neighbours per state on average, between 1 (Maine) and 8 (Missouri and Tennessee). Its eigenvalues run from −0.72 to 1.00, so the spatial coefficient of a lag or error model has to lie between −1.39 and 1. Row-standardising means that each state's neighbours get weights that sum to one, so that the lag is a weighted average of neighbours' output: each of Missouri's eight neighbours gets weight 1/8, while Maine's one neighbour gets weight 1.
</div>

## Test for spatial dependence in the non-spatial model

```python
Z = Xs.reshape(-1, 4); y = Ys.ravel(); n_obs = N * T
b_ols = np.linalg.lstsq(Z, y, rcond=None)[0]; e = y - Z @ b_ols; s2 = e @ e / n_obs
E = e.reshape(T, N)
print("Two-way fixed-effects (non-spatial) coefficients:", dict(zip(xs, b_ols.round(3))))

# Moran's I of the residuals, year by year and pooled
I_t = np.array([E[t] @ W @ E[t] / (E[t] @ E[t]) for t in range(T)])
I_pool = (E * (E @ W.T)).sum() / (E ** 2).sum()
EI = -1 / (N - 1)
print(f"\nMoran's I of the residuals: pooled {I_pool:.3f} (expected {EI:.3f} under independence), by year from {I_t.min():.2f} to {I_t.max():.2f}; positive in {int((I_t > 0).sum())} of {T} years")

# Lagrange-multiplier tests (Anselin et al. 1996), panel version
tr = np.trace(W.T @ W + W @ W)
We = (E @ W.T).ravel(); Wy = WY.ravel()
WXb = (WX.reshape(-1, 4)) @ b_ols
M = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
D = (WXb @ M(WXb)) / s2 + T * tr
g_err, g_lag = e @ We / s2, e @ Wy / s2
lm_err = g_err ** 2 / (T * tr); lm_lag = g_lag ** 2 / D
rlm_err = (g_err - T * tr / D * g_lag) ** 2 / (T * tr - (T * tr) ** 2 / D); rlm_lag = (g_lag - g_err) ** 2 / (D - T * tr)
print(f"\n{'test':22s} {'statistic':>10s} {'p-value':>9s}")
for nm, v in [("LM error", lm_err), ("LM lag", lm_lag), ("robust LM error", rlm_err), ("robust LM lag", rlm_lag)]:
    print(f"{nm:22s} {v:10.2f} {stats.chi2.sf(v, 1):9.4f}")
```

<div class="read" markdown="1">
**How to read it**

The non-spatial fixed-effects model gives −0.03 for public capital, 0.17 for private capital, 0.77 for employment and −0.004 for unemployment. Its residuals are spatially correlated: Moran's I is 0.244 pooled, against −0.021 expected under independence, and is positive in all 17 years (from 0.09 to 0.60). All four Lagrange-multiplier tests reject: LM error 96.3 and LM lag 68.8 (both p < 0.001), and the robust versions 37.7 for the error form (p < 0.001) and 10.3 for the lag form (p = 0.001). Both forms survive when we control for the other, with the error form stronger. A pure lag model and a pure error model each leave something out, so we estimate all three models and let the likelihood choose.
</div>

## Estimate the spatial lag, error and Durbin models

```python
lo_b, hi_b = 1 / om.min() + 1e-3, 1 - 1e-3
const = -(n_obs / 2) * (np.log(2 * np.pi) + 1)

def sar_fit(Zm, W_=W, om_=om):
    y_, wy = Ys.ravel(), (Ys @ W_.T).ravel()
    b0 = np.linalg.lstsq(Zm, y_, rcond=None)[0]; b1 = np.linalg.lstsq(Zm, wy, rcond=None)[0]; e0, e1 = y_ - Zm @ b0, wy - Zm @ b1
    nll = lambda r: (n_obs / 2) * np.log(((e0 - r * e1) @ (e0 - r * e1)) / n_obs) - T * np.sum(np.log(1 - r * om_))
    r = minimize_scalar(nll, bounds=(1 / om_.min() + 1e-3, 1 - 1e-3), method="bounded").x
    return b0 - r * b1, r, -nll(r) + const

def sem_fit(Zm, WZm, W_=W, om_=om):
    y_, wy = Ys.ravel(), (Ys @ W_.T).ravel()
    def parts(l):
        yt, Zt = y_ - l * wy, Zm - l * WZm; b = np.linalg.lstsq(Zt, yt, rcond=None)[0]; ee = yt - Zt @ b; return b, ee
    nll = lambda l: (n_obs / 2) * np.log((lambda ee: ee @ ee)(parts(l)[1]) / n_obs) - T * np.sum(np.log(1 - l * om_))
    l = minimize_scalar(nll, bounds=(1 / om_.min() + 1e-3, 1 - 1e-3), method="bounded").x
    return parts(l)[0], l, -nll(l) + const

# full log-likelihood of the SAR/SDM, used for standard errors (parameters: beta, rho, log sigma2)
def sar_loglik(p, Zm):
    k_ = Zm.shape[1]; b, r, ls2 = p[:k_], p[k_], p[k_ + 1]
    ee = Ys.ravel() - r * WY.ravel() - Zm @ b
    return -(n_obs / 2) * (np.log(2 * np.pi) + ls2) + T * np.sum(np.log(1 - r * om)) - ee @ ee / (2 * np.exp(ls2))
def sem_loglik(p, Zm, WZm):
    k_ = Zm.shape[1]; b, l, ls2 = p[:k_], p[k_], p[k_ + 1]
    ee = (Ys.ravel() - l * WY.ravel()) - (Zm - l * WZm) @ b
    return -(n_obs / 2) * (np.log(2 * np.pi) + ls2) + T * np.sum(np.log(1 - l * om)) - ee @ ee / (2 * np.exp(ls2))
def se_from(f, p0, *a):
    H = approx_hess(p0, lambda p: f(p, *a)); V = np.linalg.inv(-H); return V

Zs = Xs.reshape(-1, 4); Zd = np.concatenate([Xs, WX], -1).reshape(-1, 8); WZs = WX.reshape(-1, 4)
b_sar, r_sar, ll_sar = sar_fit(Zs); b_sdm, r_sdm, ll_sdm = sar_fit(Zd); b_sem, l_sem, ll_sem = sem_fit(Zs, WZs)
ll_ols = -(n_obs / 2) * (np.log(2 * np.pi) + 1 + np.log(s2))
res_sar = Ys.ravel() - r_sar * WY.ravel() - Zs @ b_sar; res_sdm = Ys.ravel() - r_sdm * WY.ravel() - Zd @ b_sdm
res_sem = None
p_sar = np.r_[b_sar, r_sar, np.log(res_sar @ res_sar / n_obs)]; V_sar = se_from(sar_loglik, p_sar, Zs)
p_sdm = np.r_[b_sdm, r_sdm, np.log(res_sdm @ res_sdm / n_obs)]; V_sdm = se_from(sar_loglik, p_sdm, Zd)
yt = Ys.ravel() - l_sem * WY.ravel(); Zt = Zs - l_sem * WZs; ee = yt - Zt @ b_sem
p_sem = np.r_[b_sem, l_sem, np.log(ee @ ee / n_obs)]; V_sem = se_from(sem_loglik, p_sem, Zs, WZs)
se = lambda V, k_: np.sqrt(np.diag(V))[:k_]

lab = xs + ["W x " + v for v in xs]
tab = pd.DataFrame(index=lab + ["spatial coefficient (rho or lambda)", "log-likelihood", "AIC"])
def col(b, sp, V, k_, ll, npar):
    c = [f"{b[i]:.3f} ({se(V, k_ + 1)[i]:.3f})" for i in range(len(b))] + [""] * (8 - len(b))
    return c + [f"{sp:.3f} ({se(V, k_ + 1)[k_]:.3f})", f"{ll:.1f}", f"{2 * npar - 2 * ll:.1f}"]
tab["non-spatial"] = [f"{b_ols[i]:.3f}" for i in range(4)] + [""] * 4 + ["", f"{ll_ols:.1f}", f"{2 * 5 - 2 * ll_ols:.1f}"]
tab["SAR"] = col(b_sar, r_sar, V_sar, 4, ll_sar, 6)
tab["SEM"] = col(b_sem, l_sem, V_sem, 4, ll_sem, 6)
tab["SDM"] = col(b_sdm, r_sdm, V_sdm, 8, ll_sdm, 10)
print("Coefficients with standard errors in parentheses (state and year effects removed by demeaning)")
print(tab.to_string())

# likelihood-ratio tests
lr = lambda a, b, df: (2 * (a - b), stats.chi2.sf(2 * (a - b), df))
print("\nLikelihood-ratio tests")
for nm, (s_, p_) in [("SAR vs non-spatial (rho = 0)", lr(ll_sar, ll_ols, 1)), ("SEM vs non-spatial (lambda = 0)", lr(ll_sem, ll_ols, 1)),
                     ("SDM vs SAR (theta = 0)", lr(ll_sdm, ll_sar, 4)), ("SDM vs SEM (common-factor restriction)", lr(ll_sdm, ll_sem, 4))]:
    print(f"  {nm:40s} LR = {s_:6.2f}, p = {p_:.4f}")
```

<div class="read" markdown="1">
**How to read it**

The spatial coefficients are 0.196 (s.e. 0.025) in the lag model, 0.384 (0.039) in the error model and 0.361 (0.039) in the Durbin model, all clearly positive. The coefficient on public capital stays small and insignificant in every model (−0.034, −0.015 and −0.014 with standard errors of 0.025). Private capital (0.15 to 0.17) and employment (0.69 to 0.77) are significant throughout, and the employment coefficient is lower in the lag model (0.69) than elsewhere. In the Durbin model, the neighbours' employment has a large negative coefficient (−0.283, s.e. 0.053), and the other neighbour terms are small. The likelihood-ratio tests all reject: the Durbin model against the lag model (LR = 41.7, p < 0.001) and against the error model (LR = 18.4, p = 0.001), where the second test is the test of the common-factor restriction. The Durbin model also has the lowest AIC (−3,340.8 against −3,330.4 for the error and −3,307.1 for the lag model), so we use it.
</div>

## Direct, indirect and total effects

```python
rng = np.random.default_rng(4)
I_N = np.eye(N)

def effects(beta, theta, rho):
    S = np.linalg.inv(I_N - rho * W)
    out = []
    for j in range(len(beta)):
        Sk = S @ (beta[j] * I_N + theta[j] * W)
        dir_ = np.trace(Sk) / N; tot = Sk.sum() / N; out.append((dir_, tot - dir_, tot))
    return np.array(out)

def simulated(b, rho, V, k_, R=1000):
    mean = np.r_[b, rho]; cov = V[:k_ + 1, :k_ + 1]
    draws = rng.multivariate_normal(mean, cov, R); res = []
    for dr in draws:
        if not (lo_b < dr[-1] < hi_b): continue
        be = dr[:4]; th = dr[4:8] if k_ == 8 else np.zeros(4); res.append(effects(be, th, dr[-1]))
    return np.array(res)

rows = []
for nm, b, th, r_, V, k_ in [("SAR", b_sar, np.zeros(4), r_sar, V_sar, 4), ("SDM", b_sdm[:4], b_sdm[4:], r_sdm, V_sdm, 8)]:
    est_e = effects(b, th, r_); sim = simulated(b if k_ == 4 else b_sdm, r_, V, k_)
    se_e = sim.std(0)
    for j, v in enumerate(xs):
        for m_, kind in enumerate(["direct", "indirect", "total"]):
            rows.append((nm, v, kind, est_e[j, m_], se_e[j, m_], est_e[j, m_] / se_e[j, m_]))
out = pd.DataFrame(rows, columns=["model", "variable", "effect", "estimate", "s.e.", "t"])
print(out.pivot_table(index=["model", "variable"], columns="effect", values=["estimate", "t"], sort=False).round(3).to_string())
print("\nNon-spatial fixed-effects coefficients for comparison:", dict(zip(xs, b_ols.round(3))))
```

<div class="read" markdown="1">
**How to read it**

In the lag model every effect has the same split by construction: the indirect effect is 23% of the direct effect. For employment it gives a direct effect of 0.697 and an indirect (spillover) effect of 0.161 (t = 7.0), for a total of 0.858. In the Durbin model, which does not impose this split, the direct effect of employment is 0.752 and the indirect effect is −0.016 (t = −0.29), so the spillover of the lag model is an artefact of the restriction. In the Durbin model, private capital has a direct effect of 0.164 and a positive spillover of 0.130 (t = 2.0), for a total of 0.294. Public capital has a direct effect of −0.020, an indirect effect of −0.081 and a total of −0.101, none significantly different from zero (t between −0.8 and −1.3). An increase in a state's unemployment rate has a negative spillover on its neighbours of −0.006 (t = −2.6). We find no evidence that public capital raises output in the state or its neighbours.
</div>

<div class="write" markdown="1">
**How to write it: results**

Residuals of the non-spatial two-way fixed-effects model are spatially correlated (Moran's I = 0.244, positive in all 17 years; LM error 96.3, LM lag 68.8, robust LM error 37.7, robust LM lag 10.3, all p ≤ 0.001). Likelihood-ratio tests prefer the spatial Durbin model to the lag model (LR = 41.7) and to the error model (LR = 18.4, p = 0.001), and the Durbin model has the lowest AIC. In the Durbin model the spatial coefficient is 0.361 (s.e. 0.039). The output elasticity of employment is 0.75 with no significant spillover (indirect effect −0.016), and that of private capital is 0.16 with a positive spillover of 0.13 (t = 2.0). Public capital has direct, indirect and total effects of −0.020, −0.081 and −0.101, none significant, so we find no evidence that public capital raises output in the state or in its neighbours.
</div>

## Robustness: the weights matrix and the fixed effects

```python
A2 = ((A @ A) > 0).astype(float); np.fill_diagonal(A2, 0); A2 = np.maximum(A2 - A, 0)          # neighbours of neighbours that are not neighbours
Wb = A / A.sum(1, keepdims=True)
W2 = A2 / A2.sum(1, keepdims=True)
Wc = (A + A2) / (A + A2).sum(1, keepdims=True)
rows = []
for lab_, Wx in [("rook contiguity (baseline)", Wb), ("second-order neighbours only", W2), ("first and second order", Wc)]:
    omx = np.linalg.eigvals(Wx).real; WYx = Ys @ Wx.T
    y_, wy = Ys.ravel(), WYx.ravel()
    b0 = np.linalg.lstsq(Zs, y_, rcond=None)[0]; b1 = np.linalg.lstsq(Zs, wy, rcond=None)[0]; e0, e1 = y_ - Zs @ b0, wy - Zs @ b1
    nll = lambda r: (n_obs / 2) * np.log(((e0 - r * e1) @ (e0 - r * e1)) / n_obs) - T * np.sum(np.log(1 - r * omx))
    r = minimize_scalar(nll, bounds=(1 / omx.min() + 1e-3, 1 - 1e-3), method="bounded").x
    rows.append((lab_, r, -nll(r) + const, (b0 - r * b1)[0], (b0 - r * b1)[2]))
print("Spatial lag model with other weights matrices")
print(pd.DataFrame(rows, columns=["W", "rho", "log-likelihood", "coef. public capital", "coef. employment"]).round(3).to_string(index=False))

# state effects only (no year effects)
dm = lambda B: B - B.mean(0, keepdims=True)
Ys0 = dm(Y); Xs0 = np.stack([dm(X[:, :, j]) for j in range(4)], -1); Z0 = Xs0.reshape(-1, 4)
y0, wy0 = Ys0.ravel(), (Ys0 @ W.T).ravel()
b0 = np.linalg.lstsq(Z0, y0, rcond=None)[0]; b1 = np.linalg.lstsq(Z0, wy0, rcond=None)[0]; e0, e1 = y0 - Z0 @ b0, wy0 - Z0 @ b1
nll0 = lambda r: (n_obs / 2) * np.log(((e0 - r * e1) @ (e0 - r * e1)) / n_obs) - T * np.sum(np.log(1 - r * om))
r0 = minimize_scalar(nll0, bounds=(lo_b, hi_b), method="bounded").x
print(f"\nState effects only: rho = {r0:.3f}, coefficient on public capital {(b0 - r0 * b1)[0]:.3f}, on employment {(b0 - r0 * b1)[2]:.3f}")
```

<div class="read" markdown="1">
**How to read it**

The weights matrix matters. A lag model with only second-order neighbours (the neighbours of neighbours) has a spatial coefficient of 0.066 and a log-likelihood of 1,631.8, against 0.196 and 1,659.6 with the baseline rook matrix, so dependence is concentrated among direct neighbours. With first- and second-order neighbours together the coefficient is 0.200 and the log-likelihood 1,645.2, which is lower than the baseline. Without year effects (state effects only) the spatial coefficient rises to 0.272 and the employment coefficient falls to 0.628, because part of what looks like spatial dependence is a common national shock. The public-capital coefficient is small and negative in every version (−0.027 to −0.046), as in the baseline. The main result is not sensitive to these choices; the size of ρ is.
</div>

## What to report in the paper

| Item | What to show |
|---|---|
| Weights matrix | how neighbours are defined, row-standardisation, and results for at least one alternative |
| Diagnostics | Moran's I and the LM and robust LM tests on the non-spatial residuals |
| Model choice | the LR tests of the Durbin model against the lag and the error model |
| Estimation | the method (ML with fixed effects), the effects included, and the standard errors |
| Effects | direct, indirect and total effects, not the raw coefficients, with simulated standard errors |
| Robustness | other weights, with and without year effects |

<div class="warn" markdown="1">
**Common mistakes**

1. **Reading the coefficient of a spatial lag or Durbin model as the marginal effect.** It is not, because of the feedback through \(\rho\). Report direct and indirect effects.
2. **Choosing the weights matrix to get the result.** Fix it on theory and show others.
3. **Ignoring common shocks.** Without year effects, spatial dependence partly reflects national shocks.
4. **Testing only one kind of dependence.** Use the LM and robust LM tests for both the lag and the error form.
5. **Using OLS standard errors in a spatial lag model.** The lag is endogenous, and OLS is inconsistent. Use ML or IV.
6. **Interpreting a significant spillover as a causal channel.** Spatial dependence can come from omitted common factors.
</div>
