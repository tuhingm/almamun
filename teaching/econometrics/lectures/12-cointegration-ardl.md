# Lecture 12 — Cointegration, VECM, and ARDL Models

## Learning objectives
By the end of this lecture you should be able to:

1. define cointegration and explain why a regression in levels of \(I(1)\) variables can be meaningful;
2. state the Granger representation theorem and write a vector error-correction model (VECM);
3. implement the Engle–Granger and Johansen procedures, with the correct critical values and deterministic specification;
4. conduct valid inference on cointegrating vectors (DOLS, FMOLS, Johansen MLE) and test weak exogeneity;
5. use the ARDL bounds test, and explain when it is and is not the right tool.

## Motivation
Lecture 10 showed that regressing one random walk on another usually produces nonsense. But economic theory often predicts that non-stationary variables move together in the long run: consumption and income, money and nominal output, prices at home and abroad, short and long interest rates. If a linear combination of \(I(1)\) variables is stationary, the variables share a common stochastic trend. The levels regression then estimates a genuine equilibrium relationship, and deviations from it predict future adjustment. That is cointegration, and it reconciles the statistical finding of unit roots with economic models of long-run equilibrium.

## 1. Definition
<div class="definition" markdown="1">
**Cointegration (Engle and Granger, 1987).** The components of an \(n\times1\) vector \(y_t\) are cointegrated of order \((1,1)\) if each is \(I(1)\) and there exists a nonzero vector \(\beta\) such that \(\beta'y_t\) is \(I(0)\). The vector \(\beta\) is a **cointegrating vector**. There can be up to \(r\le n-1\) linearly independent cointegrating vectors; \(r\) is the **cointegrating rank**.
</div>

With \(r\) cointegrating relations, the system is driven by \(n-r\) common stochastic trends (Stock and Watson, 1988). Example: if short and long interest rates are both \(I(1)\) and the term spread is \(I(0)\), then \(\beta=(1,-1)'\), and there is one common trend: the level of interest rates.

## 2. Error correction and the Granger representation theorem
Start from a VAR(\(p\)) in levels, \(y_t=c+A_1y_{t-1}+\dots+A_py_{t-p}+u_t\). It can always be rewritten as
\[
\Delta y_t=c+\Pi y_{t-1}+\sum_{j=1}^{p-1}\Gamma_j\Delta y_{t-j}+u_t,\qquad \Pi=-\Big(I-\sum_{j=1}^pA_j\Big),\quad \Gamma_j=-\sum_{i=j+1}^pA_i .
\]
- If \(\operatorname{rank}(\Pi)=0\), there is no cointegration: a VAR in differences is appropriate.
- If \(\operatorname{rank}(\Pi)=n\), then \(y_t\) is stationary: a VAR in levels is appropriate.
- If \(0<\operatorname{rank}(\Pi)=r<n\), then \(\Pi=\alpha\beta'\), with \(\alpha\) and \(\beta\) both \(n\times r\).

<div class="theorem" markdown="1">
**Theorem 12.1 (Granger representation; Engle and Granger, 1987; Johansen, 1991).** Under regularity conditions, \(y_t\) is \(I(1)\) with cointegrating rank \(r\) if and only if it has the **VECM** representation
\[
\Delta y_t=c+\alpha\beta'y_{t-1}+\sum_{j=1}^{p-1}\Gamma_j\Delta y_{t-j}+u_t ,
\]
where \(\beta'y_{t-1}\) is \(I(0)\). Cointegrated systems have error-correction representations, and vice versa.
</div>

**Interpretation.** \(\beta'y_{t-1}\) measures disequilibrium, and the **loading** (adjustment) coefficients \(\alpha\) say how fast each variable moves to restore equilibrium. If a row of \(\alpha\) is zero, that variable is **weakly exogenous** for \(\beta\): it does not respond to disequilibrium, and the other variables do the adjusting. Weak exogeneity of a variable means it drives the common trend.

**Consequences for VARs in levels.** A VAR in levels remains consistent when variables are cointegrated, and many coefficient tests are standard (Sims, Stock and Watson, 1990). Some tests, including Granger causality, can have non-standard distributions. The Toda–Yamamoto (1995) lag-augmentation approach restores standard Wald inference. Differencing a cointegrated system throws away the error-correction term: a misspecification.

## 3. Engle–Granger two-step procedure
**Step 1.** Estimate the levels regression \(y_{1t}=\mu+\gamma'y_{2t}+z_t\) by OLS and save \(\widehat z_t\).

**Step 2.** Test \(\widehat z_t\) for a unit root with an ADF regression without deterministics. Under the null of **no** cointegration, the test statistic has a distribution that depends on the number of regressors and the deterministics, because \(\widehat z_t\) is constructed to look as stationary as possible. Critical values come from Engle and Yoo (1987) or MacKinnon (2010); for two variables with a constant, the 5% value is about \(-3.34\), against \(-2.86\) for an ordinary ADF test.

**Super-consistency and inference.** If cointegrated, \(\widehat\gamma\) converges at rate \(T\). But its limit distribution is generally non-normal and depends on nuisance parameters (endogeneity of \(\Delta y_{2t}\) and serial correlation), so OLS \(t\)-statistics in the levels regression are invalid. Use:

- **dynamic OLS (DOLS; Stock and Watson, 1993):** add leads and lags of \(\Delta y_{2t}\), \(y_{1t}=\mu+\gamma'y_{2t}+\sum_{j=-q}^{q}\delta_j'\Delta y_{2,t-j}+e_t\), with HAC standard errors. Inference on \(\gamma\) is then asymptotically normal;
- **fully modified OLS (FMOLS; Phillips and Hansen, 1990):** non-parametric corrections for the same nuisance parameters.

**Limitations.** Results can depend on which variable is on the left. Only one cointegrating vector is estimated. In small samples, the static regression can have substantial bias.

## 4. Johansen's maximum-likelihood approach
Johansen (1988, 1991) estimates the VECM by Gaussian (reduced-rank) maximum likelihood. Concentrating out the short-run terms, regress \(\Delta y_t\) and \(y_{t-1}\) on the lagged differences and deterministics, and keep the residuals \(R_{0t}\) and \(R_{1t}\). With moment matrices \(S_{ij}=T^{-1}\sum_tR_{it}R_{jt}'\), solve the eigenvalue problem
\[
\big|\lambda S_{11}-S_{10}S_{00}^{-1}S_{01}\big|=0,
\]
giving ordered eigenvalues \(\widehat\lambda_1>\dots>\widehat\lambda_n\), the squared canonical correlations between \(\Delta y_t\) and \(y_{t-1}\). The MLE of \(\beta\) is given by the corresponding eigenvectors.

**Rank tests.**
\[
\text{Trace: }\;LR(r)=-T\sum_{i=r+1}^n\ln(1-\widehat\lambda_i);\qquad \text{Max-eigenvalue: }\;LR(r,r+1)=-T\ln(1-\widehat\lambda_{r+1}).
\]
Test sequentially: \(r=0\), then \(r\le1\), and so on, and stop at the first non-rejection. The distributions are non-standard and depend on the **deterministic specification** (no constant; restricted constant in the cointegrating space; unrestricted constant, giving linear trends in levels; restricted trend; unrestricted trend). The usual choice for trending macro data is an unrestricted constant or a restricted trend. The Johansen trace test is known to over-reject in small samples; Bartlett corrections or bootstrap tests help (Cavaliere, Rahbek and Taylor, 2012).

**Identifying \(\beta\).** Only the cointegrating *space* is identified: \(\alpha\beta'=(\alpha H^{-1\prime})(\beta H)'\) for any non-singular \(r\times r\) matrix \(H\). With \(r=1\), normalize one coefficient to 1. With \(r>1\), economically motivated restrictions (Johansen and Juselius, 1992) are needed, tested by LR tests that are \(\chi^2\)-distributed, because inference on \(\beta\) given the rank is asymptotically mixed normal.

## 5. ARDL models and the bounds test
The autoregressive distributed lag model
\[
\Delta y_t=c+\phi y_{t-1}+\theta'x_{t-1}+\sum_{j=1}^{p-1}\gamma_j\Delta y_{t-j}+\sum_{j=0}^{q-1}\delta_j'\Delta x_{t-j}+e_t
\]
is a single-equation conditional ECM. The long-run coefficients are \(-\theta/\phi\), and \(\phi\in(-2,0)\) is the speed of adjustment.

**Bounds test (Pesaran, Shin and Smith, 2001).** Test \(H_0:\phi=\theta=0\) (no level relationship) with an \(F\) statistic, and the \(t\)-statistic on \(\phi\). The critical values depend on whether the regressors are \(I(0)\) or \(I(1)\). PSS tabulate **lower bounds** (all regressors \(I(0)\)) and **upper bounds** (all \(I(1)\)). An \(F\) above the upper bound indicates a level relationship regardless of the regressors' integration order; below the lower bound, none; in between, the test is inconclusive. Use finite-sample critical values (Narayan, 2005; Kripfganz and Schneider, 2020) when \(T\) is below about 100.

**Assumptions that are often overlooked.** The bounds test (i) requires that \(y_t\) is \(I(1)\) or \(I(0)\) **and that the regressors are not \(I(2)\)**; (ii) assumes **at most one** level relationship, with \(x_t\) weakly exogenous (\(x\) does not error-correct toward \(y\)); and (iii) is a single-equation method. If there may be several cointegrating relations, or feedback from \(y\) to \(x\), use a Johansen VECM instead. "We use ARDL because our variables are a mix of I(0) and I(1)" is a valid reason only when these assumptions hold.

## Worked example
**Long-run money demand.** Let \(m_t\) be log real M2, \(y_t\) log real GDP, and \(i_t\) an opportunity-cost interest rate, with quarterly data. Suppose all three are \(I(1)\) by ADF and KPSS. A Johansen trace test with an unrestricted constant and 4 lags rejects \(r=0\) (trace 41.2 against a 5% critical value of 29.8) but not \(r\le1\) (12.1 against 15.5): one cointegrating vector. Normalizing on money gives
\[
m_t=1.02\,y_t-0.045\,i_t+\text{const}.
\]
The income elasticity is close to 1, and an LR test of \(\beta_y=1\) does not reject. The interest semi-elasticity of \(-0.045\) means a one-point rise in \(i\) lowers money demand by about 4.5%. The adjustment coefficient on the money equation is \(-0.12\) (12% of the gap closes per quarter). The output equation's loading is insignificant, so output is weakly exogenous. (Illustrative numbers. Many studies find that the stability of such relations breaks down around financial innovations, so test for parameter stability in \(\beta\) as in Lecture 10.)

## Problem set
1. **Common trends.** With \(n=3\) variables and \(r=1\), how many common stochastic trends are there? Give an economic example.

    <details class="solution" markdown="1"><summary>Solution</summary>

    \(n-r=2\). Example: money, output and prices, with one long-run money-demand relation. Output (a technology trend) and nominal money (a policy trend) are the two common trends.
    </details>

2. **VECM algebra.** Rewrite the VAR(2) \(y_t=A_1y_{t-1}+A_2y_{t-2}+u_t\) as a VECM and identify \(\Pi\) and \(\Gamma_1\).

    <details class="solution" markdown="1"><summary>Solution</summary>

    Subtract \(y_{t-1}\) from both sides and add and subtract \(A_2y_{t-1}\): \(\Delta y_t=(A_1+A_2-I)y_{t-1}-A_2\Delta y_{t-1}+u_t\). So \(\Pi=-(I-A_1-A_2)\) and \(\Gamma_1=-A_2\).
    </details>

3. **Why special critical values?** Explain why the ADF test on Engle–Granger residuals uses critical values more negative than the ordinary ADF test.

    <details class="solution" markdown="1"><summary>Solution</summary>

    OLS chooses \(\widehat\gamma\) to minimize the residual variance, which makes \(\widehat z_t\) look as stationary as possible even when no cointegration exists. Under the null, the test statistic is therefore shifted left, and the critical values must be more negative. The shift grows with the number of regressors.
    </details>

4. **Weak exogeneity.** In a bivariate VECM with \(\alpha=(-0.2,\;0)'\), which variable adjusts? What does this imply for forecasting and for the common trend?

    <details class="solution" markdown="1"><summary>Solution</summary>

    Only \(y_1\) adjusts, closing 20% of the disequilibrium per period. \(y_2\) is weakly exogenous, so its cumulated shocks are the common trend. Forecasts of \(y_1\) improve by using the lagged gap; forecasts of \(y_2\) do not.
    </details>

5. **ARDL misuse.** A study applies the ARDL bounds test to GDP, exports and imports, finds a level relationship, and interprets it as "exports cause growth". List three problems.

    <details class="solution" markdown="1"><summary>Solution</summary>

    (i) Exports and GDP almost certainly feed back on each other, violating the weak-exogeneity assumption of the conditional ECM. (ii) With three variables there may be more than one cointegrating relation (for example, GDP–imports and GDP–exports), which the single-equation bounds test cannot handle. (iii) A level relationship is a statement about co-movement, not causation; causal claims need an identification strategy.
    </details>

## Computational lab
**Goal:** simulate a cointegrated system, then apply Engle–Granger, Johansen and DOLS.

```python
import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

rng = np.random.default_rng(14)
T = 300
trend = np.cumsum(rng.normal(size=T))              # one common stochastic trend
x = trend + rng.normal(0, 0.5, T)
y = 2.0 + 1.5 * trend + rng.normal(0, 0.5, T)       # beta = (1, -1.5) up to noise in x

t_eg, p_eg, crit = coint(y, x)
print(f"Engle-Granger t = {t_eg:.2f}, p = {p_eg:.3f}, 5% cv = {crit[1]:.2f}")

j = coint_johansen(np.column_stack([y, x]), det_order=0, k_ar_diff=1)
print("trace stats:", j.lr1.round(1), " 5% cvs:", j.cvt[:, 1])
print("normalized beta:", np.real(j.evec[:, 0] / j.evec[0, 0]).round(3))

m = VECM(np.column_stack([y, x]), k_ar_diff=1, coint_rank=1, deterministic="ci").fit()
print("alpha (loadings):", m.alpha.ravel().round(3))
```

Tasks: (i) explain why the estimated \(\beta\) differs from \(-1.5\) (measurement noise in \(x\) relative to the trend); (ii) estimate DOLS with 2 leads and lags and HAC errors, and compare with static OLS; (iii) re-run with other seeds and note how often the \(r\le1\) hypothesis is (wrongly) rejected: the trace test over-rejects in small samples and is sensitive to the deterministic specification; (iv) simulate two *independent* random walks and confirm that EG and Johansen rarely find cointegration; (v) implement the ARDL bounds test with `statsmodels.tsa.ardl.UECM` and its `bounds_test` method. In R, use `urca::ca.jo`, `cointReg` (DOLS/FMOLS) and `ARDL::bounds_f_test`.

## Coding exercises

These exercises use **real data** and run in Python. Install the packages once with
`pip install wooldridge statsmodels linearmodels scipy`. The `wooldridge` package contains the datasets
from Wooldridge's *Introductory Econometrics*; `statsmodels` ships U.S. and Danish macroeconomic series.
Try the starter code first, then open the solution. Every printed result below was produced by running the solution code exactly as shown.

### Exercise 12.1 — Danish money demand: Johansen's classic dataset

*Data:* `danish_data` (statsmodels) — Danish quarterly data, 1974Q1–1987Q3 (Johansen and Juselius, 1990): log real money `lrm`, log real income `lry`, bond rate `ibo`, deposit rate `ide`.

1. Run the Johansen trace test on `(lrm, lry, ibo, ide)` with a constant and one lagged difference.
2. Assuming rank 1, estimate the VECM and normalize the cointegrating vector on money.
3. Interpret the income elasticity and the interest-rate coefficients, and the adjustment coefficients.

```python
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
d = sm.datasets.danish_data.load_pandas().data[["lrm", "lry", "ibo", "ide"]]
j = coint_johansen(d, det_order=0, k_ar_diff=1)
# TODO: print j.lr1 against j.cvt[:,1]; VECM(d, k_ar_diff=1, coint_rank=1, deterministic="ci").fit()
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
d = sm.datasets.danish_data.load_pandas().data[["lrm", "lry", "ibo", "ide"]]
j = coint_johansen(d, det_order=0, k_ar_diff=1)
print(pd.DataFrame({"H0: r <=": range(4), "trace": j.lr1.round(2), "5% cv": j.cvt[:, 1]}).to_string(index=False))
v = VECM(d, k_ar_diff=1, coint_rank=1, deterministic="ci").fit()
beta = v.beta[:, 0] / v.beta[0, 0]
print("\ncointegrating vector (normalized on lrm):", dict(zip(d.columns, beta.round(3))))
print("long run: lrm =", " + ".join(f"{-b:.3f}*{c}" for b, c in zip(beta[1:], d.columns[1:])), "+ const")
print("adjustment coefficients alpha:", dict(zip(d.columns, v.alpha[:, 0].round(3))))
```

Output:

```text
H0: r <=  trace   5% cv
        0  48.80 47.8545
        1  17.29 29.7961
        2   7.14 15.4943
        3   0.56  3.8415

cointegrating vector (normalized on lrm): {'lrm': np.float64(1.0), 'lry': np.float64(-0.969), 'ibo': np.float64(5.403), 'ide': np.float64(-4.14)}
long run: lrm = 0.969*lry + -5.403*ibo + 4.140*ide + const
adjustment coefficients alpha: {'lrm': np.float64(-0.3), 'lry': np.float64(0.027), 'ibo': np.float64(0.004), 'ide': np.float64(0.02)}
```

The trace test rejects \(r=0\) only narrowly (48.8 against a critical value of 47.9) and does not reject \(r\le1\) in this
short sample of 55 quarters, so one cointegrating relation — as Johansen and Juselius (1990) found with seasonal dummies
and a restricted constant. Under
rank one, the long-run income elasticity of money demand is close to one, and the coefficients on the two interest rates
have opposite signs: the bond rate (the opportunity cost of holding money) lowers money demand, and the deposit rate (the
own rate on money) raises it — with similar magnitudes, so the spread matters. Money is the variable that adjusts to
disequilibrium, which supports reading the relation as **money demand**. The sensitivity of the rank test to the
deterministic terms is part of the lesson.

</details>

### Exercise 12.2 — The consumption–income relation: Engle–Granger, DOLS and the error-correction model

*Data:* `macrodata` — U.S. quarterly real consumption `realcons` and real disposable income `realdpi`, 1959–2009.

1. Test both logs for a unit root (ADF with trend).
2. Run the Engle–Granger cointegration test of `log(realcons)` on `log(realdpi)`.
3. Estimate the cointegrating slope by static OLS and by DOLS with 2 leads and lags (HAC standard errors).
4. Estimate the error-correction model \(\Delta c_t=\alpha+\gamma\,\widehat z_{t-1}+\delta\,\Delta y_t+\varepsilon_t\). How fast does consumption close the gap?

```python
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.stattools import coint, adfuller
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"c": np.log(m["realcons"]), "y": np.log(m["realdpi"])})
# TODO: coint(d.c, d.y); DOLS: add dy leads/lags; ECM with lagged residual
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.stattools import coint, adfuller
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"c": np.log(m["realcons"]), "y": np.log(m["realdpi"])})
for v in ("c", "y"):
    a = adfuller(d[v], regression="ct", autolag="AIC"); print(f"ADF(ct) {v}: t = {a[0]:.2f}, p = {a[1]:.3f}")
t, p, cv = coint(d["c"], d["y"])
print(f"Engle-Granger: t = {t:.2f}, p = {p:.4f}, 5% cv = {cv[1]:.2f}")

static = smf.ols("c ~ y", data=d).fit()
d["dy"] = d["y"].diff()
for k in (-2, -1, 1, 2):
    d[f"dy_{k}"] = d["dy"].shift(k)
d = d.rename(columns={"dy_-1": "dy_m1", "dy_-2": "dy_m2"})         # leads of dy (k = -1, -2)
dols = smf.ols("c ~ y + dy + dy_1 + dy_2 + dy_m1 + dy_m2", data=d.dropna()).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
print(f"static OLS slope {static.params['y']:.4f};  DOLS slope {dols.params['y']:.4f} (HAC se {dols.bse['y']:.4f})")
w = dols.t_test("y = 1")
print(f"DOLS test of slope = 1: t = {float(np.squeeze(w.tvalue)):.2f}")

d["z"] = static.resid; d["dc"] = d["c"].diff(); d["z1"] = d["z"].shift(1)
ecm = smf.ols("dc ~ z1 + dy", data=d.dropna(subset=["dc", "z1", "dy"])).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
g = ecm.params["z1"]
print(f"ECM: adjustment {g:.4f} (t = {ecm.tvalues['z1']:.2f}), short-run income effect {ecm.params['dy']:.3f}")
print(f"half-life of a deviation from equilibrium: {np.log(0.5)/np.log(1+g):.1f} quarters")
```

Output:

```text
ADF(ct) c: t = -2.54, p = 0.309
ADF(ct) y: t = -1.62, p = 0.785
Engle-Granger: t = -2.54, p = 0.2620, 5% cv = -3.37
static OLS slope 1.0320;  DOLS slope 1.0359 (HAC se 0.0077)
DOLS test of slope = 1: t = 4.65
ECM: adjustment -0.0473 (t = -1.74), short-run income effect 0.360
half-life of a deviation from equilibrium: 14.3 quarters
```

Both series are I(1). The Engle–Granger test, with its special critical values, does **not** reject the null of no
cointegration (p ≈ 0.26). That is a common finding for consumption and income in levels: the consumption–income ratio
drifted upward from the 1980s as the saving rate fell, so the gap looks persistent. The remaining steps are therefore
conditional on cointegration — exactly the situation where a researcher must say so explicitly. The long-run elasticity is close to but significantly above one in the DOLS regression, which uses
valid HAC inference unlike the static regression. The error-correction coefficient is negative, as theory requires, but
small: consumption closes only a few percent of the gap each quarter. Short-run consumption responds much less than
one-for-one to income growth — the consumption smoothing that permanent-income theory predicts.

</details>

## Reading
**Required:** Hamilton, *Time Series Analysis*, chapters 19–20; Juselius, *The Cointegrated VAR Model* (2006), chapters 1–7.
**Further:** Engle and Granger (1987), "Co-integration and error correction: representation, estimation, and testing", *Econometrica* 55(2); Johansen (1991), *Econometrica* 59(6); Stock and Watson (1993), "A simple estimator of cointegrating vectors in higher order integrated systems", *Econometrica* 61(4); Phillips and Hansen (1990), *Review of Economic Studies* 57(1); Pesaran, Shin and Smith (2001), "Bounds testing approaches to the analysis of level relationships", *Journal of Applied Econometrics* 16(3); Sims, Stock and Watson (1990), *Econometrica* 58(1); Toda and Yamamoto (1995), *Journal of Econometrics* 66(1–2).
