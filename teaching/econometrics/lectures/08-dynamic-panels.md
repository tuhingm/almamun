# Lecture 08 — Dynamic Panels, Difference GMM, and System GMM

## Learning objectives
By the end of this lecture you should be able to:

1. derive the Nickell bias of the within estimator in a dynamic panel with short \(T\);
2. state the generalized method of moments (GMM) framework, the efficient weight matrix, and the \(J\) test;
3. derive the Anderson–Hsiao and Arellano–Bond (difference GMM) moment conditions;
4. derive the additional Blundell–Bond (system GMM) moment conditions and the initial-conditions assumption they require;
5. diagnose dynamic-panel GMM with AR(1)/AR(2) tests, Hansen tests, and instrument-count checks, and recognize instrument proliferation.

## Motivation
Many economic relationships are dynamic: employment adjusts slowly, consumption is habit-forming, and growth depends on initial income. A lagged dependent variable in a panel with unit effects creates a problem that the tools of Lecture 7 cannot handle. The lag is correlated with the unit effect by construction, so pooled OLS is biased, and it violates strict exogeneity, so fixed effects are biased too. The solution — using deeper lags as instruments in a GMM framework — is among the most widely used and most widely misused tools in empirical economics.

## 1. The dynamic model and Nickell bias
\[
y_{it}=\rho y_{i,t-1}+x_{it}'\beta+c_i+u_{it},\qquad |\rho|<1,\quad E[u_{it}\mid y_{i,t-1},\dots,y_{i0},x_i,c_i]=0 .
\]
**Pooled OLS** is biased upward for \(\rho\), because \(y_{i,t-1}\) contains \(c_i\).

**Fixed effects** is biased downward. After demeaning, \(\ddot y_{i,t-1}=y_{i,t-1}-\bar y_{i,-1}\) is correlated with \(\ddot u_{it}=u_{it}-\bar u_i\): the time average \(\bar u_i\) contains \(u_{i,t-1}\), which is part of \(y_{i,t-1}\).

<div class="theorem" markdown="1">
**Theorem 8.1 (Nickell, 1981).** In the pure AR(1) panel with i.i.d. \(u_{it}\) and stationary initial conditions, as \(N\to\infty\) with \(T\) fixed,
\[
\operatorname{plim}(\widehat\rho_{FE}-\rho)\approx-\frac{1+\rho}{T-1}
\]
for moderately large \(T\). The bias is \(O(1/T)\) and does not disappear as \(N\) grows.
</div>

With \(T=5\) and \(\rho=0.5\), the bias is about \(-0.375\), so FE would report a persistence of about 0.12. Bias in \(\rho\) also contaminates \(\widehat\beta\) when \(x_{it}\) is correlated with \(y_{i,t-1}\). A practical consequence: with persistent data, **pooled OLS and FE bracket the true \(\rho\)** (upward and downward bias respectively), a useful sanity check on any GMM estimate.

## 2. GMM in one page
Let \(g_i(\theta)\) be an \(\ell\times1\) vector of moment functions with \(E[g_i(\theta_0)]=0\), and \(\ell\ge k=\dim\theta\). With \(\bar g_n(\theta)=n^{-1}\sum_ig_i(\theta)\) and a positive definite weight matrix \(W_n\), the GMM estimator is
\[
\widehat\theta=\arg\min_\theta\;\bar g_n(\theta)'W_n\bar g_n(\theta).
\]

<div class="theorem" markdown="1">
**Theorem 8.2 (GMM asymptotics).** Under identification, regularity conditions and \(W_n\to_pW\), with \(G=E[\partial g_i(\theta_0)/\partial\theta']\) and \(S=E[g_ig_i']\),
\[
\sqrt n(\widehat\theta-\theta_0)\to_dN\big(0,\,(G'WG)^{-1}G'WSWG(G'WG)^{-1}\big).
\]
The variance is minimized by \(W=S^{-1}\), giving \((G'S^{-1}G)^{-1}\) (Hansen, 1982).
</div>

**Two-step GMM** first estimates \(\theta\) with a convenient \(W\), forms \(\widehat S\) from first-step residuals, and re-estimates with \(W=\widehat S^{-1}\). Linear IV is the special case \(g_i=Z_i(y_i-X_i'\theta)\). With \(W=(Z'Z/n)^{-1}\), GMM is 2SLS (Lecture 5).

**The \(J\) test.** \(J=n\,\bar g_n(\widehat\theta)'\widehat S^{-1}\bar g_n(\widehat\theta)\to_d\chi^2_{\ell-k}\) under correct specification.

**Finite-sample caveats.** Two-step standard errors are severely downward-biased in small samples, because they ignore the estimation of \(\widehat S\). The Windmeijer (2005) correction fixes this and should always be used. With many moments, two-step GMM is biased toward OLS (as in Lecture 5), and \(\widehat S\) becomes poorly estimated.

## 3. Difference GMM
**Step 1: difference out \(c_i\).**
\[
\Delta y_{it}=\rho\Delta y_{i,t-1}+\Delta x_{it}'\beta+\Delta u_{it}.
\]
\(\Delta y_{i,t-1}=y_{i,t-1}-y_{i,t-2}\) is correlated with \(\Delta u_{it}=u_{it}-u_{i,t-1}\) through \(u_{i,t-1}\).

**Step 2: find instruments.** Under sequential exogeneity and serially uncorrelated \(u_{it}\), levels dated \(t-2\) and earlier are uncorrelated with \(\Delta u_{it}\):
\[
E[y_{i,t-s}\,\Delta u_{it}]=0,\qquad s\ge2,\; t=2,\dots,T .
\]
**Anderson and Hsiao (1981)** use \(y_{i,t-2}\) (or \(\Delta y_{i,t-2}\)) as a single instrument in 2SLS. **Arellano and Bond (1991)** use *all* available lags in each period, which gives a block-diagonal instrument matrix
\[
Z_i=\begin{pmatrix}y_{i0}&0&0&0&0&0&\cdots\\0&y_{i0}&y_{i1}&0&0&0&\cdots\\0&0&0&y_{i0}&y_{i1}&y_{i2}&\cdots\\\vdots&&&&&&\ddots\end{pmatrix},
\]
yielding \(T(T-1)/2\) moment conditions for the lagged dependent variable alone. Regressors are instrumented according to their exogeneity status:

| Status of \(x_{it}\) | Valid instruments for the differenced equation |
|---|---|
| strictly exogenous | all of \(x_{i1},\dots,x_{iT}\) (or just \(\Delta x_{it}\)) |
| predetermined: \(E[x_{is}u_{it}]=0\) for \(s\le t\) | \(x_{i,t-1},x_{i,t-2},\dots\) |
| endogenous: \(E[x_{is}u_{it}]=0\) for \(s<t\) | \(x_{i,t-2},x_{i,t-3},\dots\) |

**Weakness when \(\rho\) is near 1.** If \(y_{it}\) is highly persistent, lagged levels are nearly uncorrelated with subsequent differences. In the AR(1) with \(\rho\to1\), the first-stage coefficient of \(\Delta y_{i,t-1}\) on \(y_{i,t-2}\) goes to zero. Difference GMM then suffers the weak-instrument bias toward OLS-on-differences, which in this case means downward bias (Blundell and Bond, 1998).

## 4. System GMM
**Blundell and Bond (1998)** add the levels equation, instrumented with lagged **differences**:
\[
E\big[\Delta y_{i,t-1}(c_i+u_{it})\big]=0 .
\]
This moment holds if deviations from the long-run mean are uncorrelated with \(c_i\). In the AR(1), the **mean-stationarity** condition is
\[
E\Big[\Big(y_{i0}-\frac{c_i}{1-\rho}\Big)c_i\Big]=0 ,
\]
which says the initial observation is drawn from the steady-state distribution relative to each unit's own long-run mean. System GMM stacks the differenced and levels equations. With persistent series, it can be much more precise than difference GMM.

**The assumption is substantive.** Mean stationarity fails, for example, when units start far from their steady state: young firms in a firm panel, or countries early in a convergence process. It is directly relevant to growth regressions (Bond, Hoeffler and Temple, 2001). A difference-in-Hansen test compares the full system with difference GMM and checks the additional moments, but its power is limited when instruments are numerous.

## 5. Diagnostics and the instrument-proliferation problem
**Serial correlation tests (Arellano–Bond).** If \(u_{it}\) is serially uncorrelated, then \(\Delta u_{it}\) has first-order autocorrelation (because \(\operatorname{Cov}(\Delta u_{it},\Delta u_{i,t-1})=-\sigma^2_u\)) but **no second-order autocorrelation**. Expect to reject in AR(1) and not to reject in AR(2). Rejection in AR(2) invalidates \(t-2\) instruments; use \(t-3\) and deeper.

**Hansen \(J\).** The test of over-identifying restrictions (use the robust Hansen version, not Sargan). **A very high p-value (close to 1) is a warning sign, not reassurance.** With many instruments, \(\widehat S\) is poorly estimated, and \(J\) loses power and "passes" almost always.

**Instrument proliferation (Roodman, 2009).** The instrument count grows quadratically in \(T\). Too many instruments overfit the endogenous variables, bias estimates toward OLS/FE, and weaken the \(J\) test. Remedies:

- **report the instrument count** and keep it well below \(N\);
- **collapse** the instrument matrix (one column per lag distance instead of per period × lag);
- **limit lag depth** (for example, lags 2–4 only);
- show that results are robust across these choices, and that \(\widehat\rho\) lies between the FE and pooled-OLS estimates.

## Worked example
**Labor demand (Arellano and Bond, 1991).** Firm employment \(n_{it}\) depends on its own lag, wages and capital, with \(N=140\) UK firms and \(T=7\)–9. The published estimates follow a well-known pattern (approximate magnitudes; see Blundell and Bond, 1998, and the replication in Roodman, 2009, for exact figures and specifications):

| Estimator | \(\widehat\rho\) on \(n_{i,t-1}\) |
|---|---|
| Pooled OLS | ≈ 1.05 (upward biased) |
| Within (FE) | ≈ 0.73 (Nickell bias, downward) |
| Difference GMM | ≈ 0.6–0.7 (below FE) |
| System GMM | ≈ 0.8–0.9 (inside the bracket) |

The difference-GMM estimate lies **below** FE, outside the OLS–FE bracket. This is the symptom of weak instruments with persistent series. System GMM moves back inside the bracket, which is consistent with Blundell and Bond's diagnosis, *provided* mean stationarity is believable for these firms. A careful paper would report the instrument count, AR(2) and difference-in-Hansen tests, and estimates with collapsed and lag-limited instrument sets.

## Problem set
1. **Nickell bias with \(T=2\).** In \(y_{it}=\rho y_{i,t-1}+c_i+u_{it}\) with observations \(t=0,1,2\), the FE estimator uses \(t=1,2\) and equals the FD estimator. Show that \(\operatorname{plim}\widehat\rho_{FD}=\rho-(1+\rho)/2\) under stationarity, so the bias is \(-(1+\rho)/2\).

    <details class="solution" markdown="1"><summary>Solution</summary>

    \(\widehat\rho_{FD}\to\operatorname{Cov}(\Delta y_2,\Delta y_1)/\operatorname{Var}(\Delta y_1)\). Write \(\Delta y_2=\rho\Delta y_1+\Delta u_2\). Then \(\operatorname{Cov}(\Delta u_2,\Delta y_1)=-\operatorname{Cov}(u_1,y_1-y_0)=-\sigma_u^2\). Under stationarity, \(\operatorname{Var}(\Delta y_1)=2\sigma_u^2/(1+\rho)\), so the bias is \(-\sigma_u^2(1+\rho)/(2\sigma_u^2)=-(1+\rho)/2\).
    </details>

2. **Counting moments.** With \(T=10\) (periods \(0,\dots,9\)), one lagged dependent variable and one endogenous regressor, count the Arellano–Bond moment conditions using all lags, and with collapsed instruments.

    <details class="solution" markdown="1"><summary>Solution</summary>

    For \(y\), differenced equations run for \(t=2,\dots,9\), with \(t-1\) instruments each: \(\sum_{t=2}^{9}(t-1)=36\). For an endogenous \(x\), the same count, 36. Total 72 (plus time dummies). Collapsed: one column per lag distance, at most 8 each, so at most 16.
    </details>

3. **AR(2) logic.** Show that \(\operatorname{Cov}(\Delta u_{it},\Delta u_{i,t-2})=0\) if \(u_{it}\) is serially uncorrelated, but not if \(u_{it}\) follows an MA(1).

    <details class="solution" markdown="1"><summary>Solution</summary>

    \(\operatorname{Cov}(u_t-u_{t-1},u_{t-2}-u_{t-3})\) involves only covariances at lags 1, 2 and 3, which are all zero for white noise. If \(u_t=e_t+\theta e_{t-1}\), then \(\operatorname{Cov}(u_{t-1},u_{t-2})=\theta\sigma^2_e\neq0\), so the covariance is \(-\theta\sigma^2_e\neq0\).
    </details>

4. **System GMM assumption.** Show that in the AR(1), \(E[\Delta y_{i,t-1}c_i]=0\) for all \(t\) if \(y_{i0}=c_i/(1-\rho)+e_{i0}\) with \(e_{i0}\) uncorrelated with \(c_i\).

    <details class="solution" markdown="1"><summary>Solution</summary>

    Write \(y_{it}=c_i/(1-\rho)+e_{it}\) with \(e_{it}=\rho e_{i,t-1}+u_{it}\). Then \(\Delta y_{it}=\Delta e_{it}\) does not involve \(c_i\). Since every \(e_{it}\) is built from \(e_{i0}\) and \(u\)'s, all uncorrelated with \(c_i\), the covariance is zero.
    </details>

5. **Reading output.** A paper reports system GMM with 312 instruments, \(N=180\), AR(2) p = 0.41, Hansen p = 1.000. Write a two-sentence referee comment.

    <details class="solution" markdown="1"><summary>Solution</summary>

    The instrument count exceeds the number of units, so the weight matrix is singular or poorly estimated. A Hansen p-value of 1.000 is the signature of instrument proliferation, not evidence of validity. Please report results with collapsed and lag-limited instruments, Windmeijer-corrected standard errors, difference-in-Hansen tests for the levels moments, and the pooled OLS and FE estimates as bounds.
    </details>

## Computational lab
**Goal:** verify Nickell bias and compare Anderson–Hsiao IV with FE.

```python
import numpy as np
rng = np.random.default_rng(8)
N, T, rho = 2_000, 6, 0.6
c = rng.normal(size=N)
y = np.empty((N, T + 50))
y[:, 0] = c / (1 - rho) + rng.normal(size=N) / np.sqrt(1 - rho**2)   # stationary start
for t in range(1, T + 50):
    y[:, t] = rho * y[:, t - 1] + c + rng.normal(size=N)
y = y[:, -(T + 1):]                                                  # keep T+1 periods: 0..T

Y, L = y[:, 1:], y[:, :-1]
Yd, Ld = Y - Y.mean(1, keepdims=True), L - L.mean(1, keepdims=True)
rho_fe = (Yd * Ld).sum() / (Ld * Ld).sum()
rho_ols = np.polyfit(L.ravel(), Y.ravel(), 1)[0]

dY, dL = np.diff(y, axis=1)[:, 1:], np.diff(y, axis=1)[:, :-1]       # Δy_t and Δy_{t-1}, t = 2..T
Z = y[:, :-2]                                                        # y_{t-2}
rho_ah = (Z * dY).sum() / (Z * dL).sum()
print(f"OLS {rho_ols:.3f}  FE {rho_fe:.3f} (Nickell approx {rho - (1 + rho) / (T - 1):.3f})  AH-IV {rho_ah:.3f}")
```

Tasks: (i) increase \(T\) and confirm that the FE bias shrinks at rate \(1/T\); (ii) set \(\rho=0.95\) and observe the instability of Anderson–Hsiao (weak instruments); (iii) estimate difference and system GMM with `pydynpd` (Python), `plm::pgmm` (R) or `xtabond2` (Stata) with `collapse` and lag limits, and report the instrument count, AR(2) and Hansen tests.

## Coding exercises

These exercises use **real data** and run in Python. Install the packages once with
`pip install wooldridge statsmodels linearmodels scipy`. The `wooldridge` package contains the datasets
from Wooldridge's *Introductory Econometrics*; `statsmodels` ships U.S. and Danish macroeconomic series.
Try the starter code first, then open the solution. Every printed result below was produced by running the solution code exactly as shown.

### Exercise 8.1 — Wage persistence: the OLS–FE bracket and Anderson–Hsiao IV

*Data:* `wagepan` — 545 men, 1980–1987.

Model \(lwage_{it}=\rho\,lwage_{i,t-1}+\beta\,union_{it}+\lambda_t+c_i+u_{it}\).

1. Estimate \(\rho\) by pooled OLS and by fixed effects. Which way should each be biased?
2. Estimate the first-differenced equation by 2SLS, instrumenting \(\Delta lwage_{i,t-1}\) with the level \(lwage_{i,t-2}\) (Anderson–Hsiao). Report the first-stage \(F\).
3. Does the IV estimate fall inside the OLS–FE bracket?
4. Test for second-order serial correlation in the differenced residuals.

```python
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wagepan").sort_values(["nr", "year"])
g = df.groupby("nr")
df["l1"] = g["lwage"].shift(1); df["l2"] = g["lwage"].shift(2)
df["dy"] = df["lwage"] - df["l1"]; df["dl1"] = df["l1"] - df["l2"]
# TODO: pooled OLS and FE of lwage on l1; 2SLS of dy on dl1 (instrument: l2 = lwage two years back)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS
df = wd.data("wagepan").sort_values(["nr", "year"])
g = df.groupby("nr")
df["l1"], df["l2"], df["l3"] = g["lwage"].shift(1), g["lwage"].shift(2), g["lwage"].shift(3)
df["dy"], df["dl1"] = df["lwage"] - df["l1"], df["l1"] - df["l2"]
df["dunion"] = df["union"] - g["union"].shift(1)

po = smf.ols("lwage ~ l1 + union + C(year)", data=df.dropna(subset=["l1"])).fit(cov_type="cluster", cov_kwds={"groups": df.dropna(subset=["l1"])["nr"]})
fe = PanelOLS.from_formula("lwage ~ l1 + union + TimeEffects + EntityEffects", data=df.dropna(subset=["l1"]).set_index(["nr", "year"])).fit(cov_type="clustered", cluster_entity=True)
d = df.dropna(subset=["dy", "dl1", "l2", "dunion"]).copy()
ah = IV2SLS.from_formula("dy ~ 1 + dunion + C(year) + [dl1 ~ l2]", data=d).fit(cov_type="clustered", clusters=d["nr"])
fs = smf.ols("dl1 ~ l2 + dunion + C(year)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["nr"]})
print(f"pooled OLS rho = {po.params['l1']:.3f}   FE rho = {fe.params['l1']:.3f}   Anderson-Hsiao rho = {ah.params['dl1']:.3f} (se {ah.std_errors['dl1']:.3f})")
print(f"AH first stage: coefficient on l2 = {fs.params['l2']:.3f}, clustered F = {fs.tvalues['l2']**2:.1f}")
print(f"union effect: pooled {po.params['union']:.3f}, FE {fe.params['union']:.3f}, AH {ah.params['dunion']:.3f}")

d["e"] = ah.resids
d["e_lag2"] = d.groupby("nr")["e"].shift(2)
dd = d.dropna(subset=["e_lag2"])
ar2 = smf.ols("e ~ e_lag2", data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd["nr"]})
print(f"AR(2) check: corr of differenced residuals at lag 2 = {dd['e'].corr(dd['e_lag2']):.3f}, cluster t = {ar2.tvalues['e_lag2']:.2f}")
```

Output:

```text
pooled OLS rho = 0.602   FE rho = 0.064   Anderson-Hsiao rho = 0.120 (se 0.034)
AH first stage: coefficient on l2 = -0.392, clustered F = 178.9
union effect: pooled 0.064, FE 0.069, AH 0.024
AR(2) check: corr of differenced residuals at lag 2 = 0.039, cluster t = 1.71
```

Pooled OLS is biased **up** (the lag contains the worker effect) and FE is biased **down** (Nickell bias with only
\(T=8\)). They bracket the true \(\rho\) widely. The Anderson–Hsiao estimate lies inside the bracket, which is a reassuring
sanity check. Its first stage is strong here because wages are only moderately persistent once worker effects are
removed. The AR(2) check is what licenses the \(t-2\) instrument: a significant lag-2 correlation of the differenced
residuals would mean serial correlation in \(u_{it}\), and deeper lags would be needed. Arellano–Bond GMM uses all
available lags; try `pydynpd` and compare, reporting the instrument count.

</details>

### Exercise 8.2 — Grunfeld investment: how instrument count changes GMM

*Data:* `grunfeld` (statsmodels) — investment, market value and capital of 11 large U.S. firms, 1935–1954.

With only \(N=11\) firms and \(T=20\), the Arellano–Bond instrument count explodes relative to \(N\).

1. Estimate \(invest_{it}=\rho\,invest_{i,t-1}+\beta_1 value_{it}+\beta_2 capital_{it}+c_i+u_{it}\) by pooled OLS and FE.
2. Count how many Arellano–Bond moment conditions "all lags" would use for the lagged dependent variable alone.
3. Estimate the differenced equation by just-identified IV (lag-2 level) and by an over-identified IV with lags 2–4 (collapsed). Compare.
4. Explain why "panel GMM" is a poor tool for this dataset.

```python
import statsmodels.api as sm
df = sm.datasets.grunfeld.load_pandas().data.sort_values(["firm", "year"])
# TODO: create lags by firm; pooled OLS, FE; IV on first differences with y_{t-2}, y_{t-3}, y_{t-4}
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import statsmodels.api as sm, statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
df = sm.datasets.grunfeld.load_pandas().data.sort_values(["firm", "year"]).reset_index(drop=True)
g = df.groupby("firm")
for k in (1, 2, 3, 4):
    df[f"i{k}"] = g["invest"].shift(k)
df["dy"], df["dl1"] = df["invest"] - df["i1"], df["i1"] - df["i2"]
df["dv"], df["dk"] = g["value"].diff(), g["capital"].diff()

d1 = df.dropna(subset=["i1"])
po = smf.ols("invest ~ i1 + value + capital", data=d1).fit()
fe = smf.ols("invest ~ i1 + value + capital + C(firm)", data=d1).fit()
print(f"pooled OLS rho = {po.params['i1']:.3f}    FE rho = {fe.params['i1']:.3f}")
T = df["year"].nunique()
print(f"Arellano-Bond 'all lags' moments for the lagged dependent variable: {sum(t - 1 for t in range(2, T))} with N = {df['firm'].nunique()} firms")

d2 = df.dropna(subset=["dy", "dl1", "i2", "i3", "i4", "dv", "dk"])
cl = dict(cov_type="clustered", clusters=d2["firm"].astype("category").cat.codes)
ji = IV2SLS.from_formula("dy ~ 0 + dv + dk + [dl1 ~ i2]", data=d2).fit(**cl)
oi = IV2SLS.from_formula("dy ~ 0 + dv + dk + [dl1 ~ i2 + i3 + i4]", data=d2).fit(**cl)
print(f"IV, lag 2 only:   rho = {ji.params['dl1']:.3f} (se {ji.std_errors['dl1']:.3f})")
print(f"IV, lags 2-4:     rho = {oi.params['dl1']:.3f} (se {oi.std_errors['dl1']:.3f})")
```

Output:

```text
pooled OLS rho = 0.876    FE rho = 0.684
Arellano-Bond 'all lags' moments for the lagged dependent variable: 171 with N = 11 firms
IV, lag 2 only:   rho = 0.010 (se 0.215)
IV, lags 2-4:     rho = 0.804 (se 0.213)
```

"All lags" would use 171 moment conditions for the lagged dependent variable alone — with 11 firms, the GMM weight matrix
cannot even be estimated. The two IV estimates differ wildly (about 0.01 with one instrument, 0.80 with three), and both have
standard errors above 0.2: the instruments carry little reliable information. With \(T=20\), the Nickell bias of FE is modest (order \(1/T\)), so FE is already a reasonable
estimator here; difference-GMM methods are designed for **large \(N\), small \(T\)**, the opposite of this dataset.
The clustered standard errors rest on only 11 clusters, so even they should not be trusted (Lecture 4). Choosing the estimator to fit the dimensions of the data is part of the econometrics.

</details>

## Reading
**Required:** Arellano, *Panel Data Econometrics* (2003), chapters 6–8; Roodman (2009), "How to do xtabond2: an introduction to difference and system GMM in Stata", *Stata Journal* 9(1).
**Further:** Nickell (1981), "Biases in dynamic models with fixed effects", *Econometrica* 49(6); Hansen (1982), "Large sample properties of generalized method of moments estimators", *Econometrica* 50(4); Anderson and Hsiao (1981), *JASA* 76(375); Arellano and Bond (1991), *Review of Economic Studies* 58(2); Blundell and Bond (1998), *Journal of Econometrics* 87(1); Windmeijer (2005), *Journal of Econometrics* 126(1); Roodman (2009), "A note on the theme of too many instruments", *Oxford Bulletin* 71(1); Bond, Hoeffler and Temple (2001), "GMM estimation of empirical growth models", CEPR DP 3048.
