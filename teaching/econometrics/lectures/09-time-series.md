# Lecture 09 — Time-Series Foundations: Stationarity, ARMA, and Forecasting

## Learning objectives
By the end of this lecture you should be able to:

1. define strict and weak stationarity, ergodicity and martingale differences, and explain why they replace i.i.d. sampling;
2. derive the moments, autocorrelation function and stationarity conditions of AR, MA and ARMA processes;
3. state the Wold decomposition and use ACF/PACF patterns and information criteria to choose a model;
4. estimate autoregressions by OLS and know when the usual asymptotics apply, including the inconsistency caused by a lagged dependent variable with serially correlated errors;
5. construct optimal forecasts, forecast intervals, and tests of relative forecast accuracy.

## Motivation
In a cross-section, observations are independent draws. In a time series we observe a single realization of a stochastic process, one draw per period, and today's value depends on yesterday's. Inference is still possible, because a stationary, weakly dependent process "forgets" its past fast enough for time averages to behave like population averages. This lecture sets up that framework. Lectures 10–12 relax stationarity, and Lecture 11 extends it to systems.

## 1. Stationarity, ergodicity and dependence
A process \(\{y_t\}\) is **strictly stationary** if the joint distribution of \((y_{t},\dots,y_{t+h})\) does not depend on \(t\), for every \(h\). It is **weakly (covariance) stationary** if \(E[y_t]=\mu\) and \(\operatorname{Cov}(y_t,y_{t-j})=\gamma_j\) do not depend on \(t\). The **autocorrelation function** (ACF) is \(\rho_j=\gamma_j/\gamma_0\).

Stationarity alone does not justify time averages: \(y_t=Z\) for all \(t\), with \(Z\) random, is stationary, but \(\bar y_T=Z\) never learns \(E[Z]\). **Ergodicity** rules this out. Loosely, it requires that observations far apart are nearly independent.

<div class="theorem" markdown="1">
**Theorem 9.1 (Ergodic theorem).** If \(\{y_t\}\) is strictly stationary and ergodic with \(E|y_t|<\infty\), then \(\bar y_T\to_{a.s.}E[y_t]\).
</div>

<div class="theorem" markdown="1">
**Theorem 9.2 (CLT for martingale differences).** If \(\{g_t\}\) is a stationary, ergodic martingale difference sequence (\(E[g_t\mid g_{t-1},g_{t-2},\dots]=0\)) with \(E[g_tg_t']=S\) finite, then \(T^{-1/2}\sum_tg_t\to_dN(0,S)\).
</div>

These two results replace the WLLN and CLT of Lecture 2. The MDS condition is exactly what correct specification of a dynamic conditional mean delivers: if \(E[u_t\mid\text{past}]=0\), then \(x_tu_t\) is an MDS whenever \(x_t\) contains only past information. If the scores are serially correlated, the long-run variance and HAC standard errors of Lecture 4 are needed.

## 2. Moving-average and autoregressive processes
Let \(\varepsilon_t\) be white noise: mean zero, variance \(\sigma^2\), uncorrelated over time.

**MA(q).** \(y_t=\mu+\varepsilon_t+\theta_1\varepsilon_{t-1}+\dots+\theta_q\varepsilon_{t-q}\). It is always stationary, and \(\gamma_j=0\) for \(j>q\): the ACF **cuts off** after lag \(q\). It is **invertible** (expressible as an AR(\(\infty\)) in current and past \(y\)) if the roots of \(\theta(z)=1+\theta_1z+\dots+\theta_qz^q\) lie outside the unit circle.

**AR(1).** \(y_t=c+\phi y_{t-1}+\varepsilon_t\). By repeated substitution, if \(|\phi|<1\),
\[
y_t=\frac{c}{1-\phi}+\sum_{j=0}^{\infty}\phi^j\varepsilon_{t-j},\qquad \gamma_0=\frac{\sigma^2}{1-\phi^2},\qquad \rho_j=\phi^j .
\]
The ACF decays geometrically. The effect of a shock on \(y_{t+h}\) is \(\phi^h\), and the **half-life** of a shock is \(\ln0.5/\ln|\phi|\). With \(\phi=0.9\) per quarter, it is about 6.6 quarters.

**AR(p).** \(\phi(L)y_t=c+\varepsilon_t\), with \(\phi(L)=1-\phi_1L-\dots-\phi_pL^p\) and \(L\) the lag operator. It is stationary if all roots of \(\phi(z)=0\) lie outside the unit circle. Equivalently, all eigenvalues of the companion matrix
\[
F=\begin{pmatrix}\phi_1&\phi_2&\cdots&\phi_{p-1}&\phi_p\\1&0&\cdots&0&0\\\vdots&&\ddots&&\vdots\\0&0&\cdots&1&0\end{pmatrix}
\]
lie inside the unit circle. The autocovariances satisfy the **Yule–Walker equations** \(\gamma_j=\phi_1\gamma_{j-1}+\dots+\phi_p\gamma_{j-p}\) for \(j\ge1\). Complex roots generate damped cycles in the ACF and in impulse responses.

**ARMA(p,q).** \(\phi(L)y_t=c+\theta(L)\varepsilon_t\): a parsimonious representation of long autocorrelation patterns.

<div class="theorem" markdown="1">
**Theorem 9.3 (Wold decomposition).** Every weakly stationary, purely non-deterministic process can be written as \(y_t=\mu+\sum_{j=0}^\infty\psi_j\varepsilon_{t-j}\), with \(\psi_0=1\), \(\sum\psi_j^2<\infty\), and \(\varepsilon_t\) the one-step linear forecast error (white noise).
</div>

The Wold theorem justifies linear time-series models as approximations: any stationary process has an MA(\(\infty\)) representation, and finite ARMA models are parsimonious approximations to it. But the \(\varepsilon_t\) are *forecast errors*, not structural shocks. That distinction drives Lecture 11.

## 3. Model identification
| Process | ACF | PACF |
|---|---|---|
| AR(p) | decays (geometric or damped cycle) | cuts off after lag \(p\) |
| MA(q) | cuts off after lag \(q\) | decays |
| ARMA(p,q) | decays after lag \(q\) | decays after lag \(p\) |

The partial autocorrelation at lag \(j\) is the coefficient on \(y_{t-j}\) in a regression of \(y_t\) on \(y_{t-1},\dots,y_{t-j}\). Under white noise, sample autocorrelations are approximately \(N(0,1/T)\), giving the familiar \(\pm1.96/\sqrt T\) bands. The **Ljung–Box** statistic \(Q=T(T+2)\sum_{j=1}^m\widehat\rho_j^2/(T-j)\sim\chi^2_{m-p-q}\) tests jointly for remaining residual autocorrelation.

**Information criteria.** Choose the order minimizing
\[
AIC=\ln\widehat\sigma^2+\frac{2k}{T},\qquad BIC=\ln\widehat\sigma^2+\frac{k\ln T}{T}.
\]
BIC is consistent: it selects the true finite order with probability approaching one. AIC tends to over-fit but is asymptotically efficient for forecasting when the true model is infinite-order. Compare criteria on a **common estimation sample**.

## 4. Estimation and inference
**Autoregressions by OLS.** For a stationary AR(p) with MDS errors, OLS of \(y_t\) on \((1,y_{t-1},\dots,y_{t-p})\) is consistent and asymptotically normal, by Theorems 9.1–9.2. It is **biased** in finite samples, because \(E[\varepsilon_t\mid y_{t+1}]\neq0\): strict exogeneity fails although contemporaneous exogeneity holds. In the AR(1), the Kendall bias is approximately \(-(1+3\phi)/T\) (with an intercept). This is non-trivial for persistent series in short samples.

**MA terms** require nonlinear least squares or maximum likelihood, using the Kalman filter or conditional sums of squares.

**A lagged dependent variable with serially correlated errors.** Suppose \(y_t=\beta y_{t-1}+u_t\) with \(u_t=\rho u_{t-1}+e_t\). Then \(y_{t-1}\) depends on \(u_{t-1}\), which is correlated with \(u_t\), so **OLS is inconsistent**. HAC standard errors do not fix this: the problem is in the conditional mean. The remedy is to model the dynamics fully: here \(y_t=(\beta+\rho)y_{t-1}-\beta\rho y_{t-2}+e_t\) is an AR(2) with white-noise errors. Test for remaining serial correlation with the Breusch–Godfrey LM test, which, unlike Durbin–Watson, is valid with lagged dependent variables.

## 5. Forecasting
Under quadratic loss, the optimal forecast is the conditional mean \(\widehat y_{T+h\mid T}=E[y_{T+h}\mid\mathcal F_T]\). For the AR(1),
\[
\widehat y_{T+h\mid T}=\mu+\phi^h(y_T-\mu),\qquad \operatorname{Var}(y_{T+h}-\widehat y_{T+h\mid T})=\sigma^2\sum_{j=0}^{h-1}\phi^{2j}\to\gamma_0 .
\]
Forecasts revert to the mean, and forecast uncertainty grows toward the unconditional variance. Iterating a one-step model ("iterated") and regressing \(y_{t+h}\) on information at \(t\) ("direct") are both common. Iterated forecasts are more efficient if the model is correct; direct forecasts are more robust to misspecification. Reported intervals usually ignore parameter uncertainty, which matters in small samples.

**Evaluation.** Use **pseudo out-of-sample** forecasts: estimate on data up to \(t\), forecast \(t+h\), roll forward, and compute the root mean squared forecast error (RMSFE). Compare two forecasts with the **Diebold–Mariano** test: with loss differential \(d_t=e_{1t}^2-e_{2t}^2\), the statistic \(\bar d/\widehat{se}(\bar d)\to_dN(0,1)\), where the standard error is HAC because \(h\)-step errors are MA(\(h-1\)). For nested models, the DM test is non-standard. Use Clark and West (2007).

## Worked example
**U.S. inflation persistence.** Quarterly CPI inflation is often summarized by an AR(p) chosen by BIC, typically with \(p\) between 1 and 4. Suppose the fitted AR(1) has \(\widehat\phi=0.85\) and \(\widehat\mu=2.8\%\), and current inflation is 5%. Then:

- one-quarter-ahead forecast: \(2.8+0.85(2.2)=4.67\%\);
- four quarters ahead: \(2.8+0.85^4(2.2)=2.8+0.522\times2.2=3.95\%\);
- half-life of a shock: \(\ln0.5/\ln0.85=4.3\) quarters.

Caveats: inflation persistence has changed over time (high in the 1970s, low after the mid-1990s; Stock and Watson, 2007), so a constant-parameter AR(1) estimated on the full sample can mislead. Lecture 10 discusses breaks. And with \(T=160\) and \(\widehat\phi=0.85\), the Kendall bias of about \(-(1+2.55)/160\approx-0.022\) means true persistence is likely a little higher than estimated.

## Problem set
1. **AR(2) stationarity.** For \(y_t=1.2y_{t-1}-0.35y_{t-2}+\varepsilon_t\), find the roots of \(\phi(z)\) and determine whether the process is stationary. Is the ACF oscillating?

    <details class="solution" markdown="1"><summary>Solution</summary>

    \(\phi(z)=1-1.2z+0.35z^2=0\) has roots \(z=(1.2\pm\sqrt{1.44-1.4})/0.7=(1.2\pm0.2)/0.7\), that is, \(2\) and \(1.43\). Both lie outside the unit circle, so the process is stationary. The roots are real, so the ACF decays without oscillation (companion eigenvalues 0.5 and 0.7).
    </details>

2. **MA(1) ACF.** Derive \(\rho_1\) for \(y_t=\varepsilon_t+\theta\varepsilon_{t-1}\) and show \(|\rho_1|\le1/2\). Why do \(\theta\) and \(1/\theta\) give the same ACF, and which do we choose?

    <details class="solution" markdown="1"><summary>Solution</summary>

    \(\gamma_0=\sigma^2(1+\theta^2)\) and \(\gamma_1=\theta\sigma^2\), so \(\rho_1=\theta/(1+\theta^2)\), maximized in absolute value at \(|\theta|=1\), giving \(1/2\). Replacing \(\theta\) by \(1/\theta\) leaves \(\rho_1\) unchanged. Choose the invertible \(|\theta|<1\), for which \(\varepsilon_t\) is the forecast error from past \(y\).
    </details>

3. **LDV and AR errors.** Show that \(\operatorname{plim}\widehat\beta_{OLS}=\beta+\dfrac{\rho(1-\beta^2)}{1+\beta\rho}\) in \(y_t=\beta y_{t-1}+u_t\), \(u_t=\rho u_{t-1}+e_t\).

    <details class="solution" markdown="1"><summary>Solution</summary>

    \(y_t\) is an AR(2) with roots \(\beta,\rho\), so its first autocorrelation is \(\rho_1=(\beta+\rho)/(1+\beta\rho)\), and \(\operatorname{plim}\widehat\beta=\rho_1\). Then \(\rho_1-\beta=(\beta+\rho-\beta-\beta^2\rho)/(1+\beta\rho)=\rho(1-\beta^2)/(1+\beta\rho)\).
    </details>

4. **Forecast variance.** Derive the \(h\)-step forecast error variance for an AR(1) and show it converges to \(\gamma_0\).

    <details class="solution" markdown="1"><summary>Solution</summary>

    \(y_{T+h}-\widehat y_{T+h\mid T}=\sum_{j=0}^{h-1}\phi^j\varepsilon_{T+h-j}\), with variance \(\sigma^2(1-\phi^{2h})/(1-\phi^2)\to\sigma^2/(1-\phi^2)=\gamma_0\).
    </details>

5. **Spurious fit.** A colleague selects a lag order by AIC on the full sample, then reports an in-sample \(R^2\) of 0.92 as evidence of forecasting ability. Explain the problems and propose a better evaluation.

    <details class="solution" markdown="1"><summary>Solution</summary>

    In-sample fit of a persistent series mostly reflects persistence, and the same data were used for selection and evaluation. Use a rolling or expanding pseudo out-of-sample exercise with re-selection at each origin, report RMSFE relative to a random-walk or AR(1) benchmark, and test with Clark–West.
    </details>

## Computational lab
**Goal:** simulate ARMA processes, identify them, and evaluate forecasts.

```python
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

rng = np.random.default_rng(9)
proc = ArmaProcess(ar=[1, -1.2, 0.35], ma=[1])          # AR(2) from problem 1
y = proc.generate_sample(400, distrvs=rng.standard_normal)
print("ACF :", acf(y, nlags=5).round(2))
print("PACF:", pacf(y, nlags=5).round(2))

bic = {p: ARIMA(y[4:], order=(p, 0, 0)).fit().bic for p in range(5)}
p_hat = min(bic, key=bic.get)
print("BIC chooses p =", p_hat)

errs = []
for t in range(300, 399):                               # expanding-window 1-step forecasts
    fc = ARIMA(y[:t], order=(p_hat, 0, 0)).fit().forecast(1)[0]
    errs.append(y[t] - fc)
print("pseudo out-of-sample RMSFE:", np.sqrt(np.mean(np.square(errs))).round(3))
```

Tasks: (i) relate the sample ACF/PACF to the theoretical patterns; (ii) simulate \(y_t=0.5y_{t-1}+u_t\), \(u_t=0.5u_{t-1}+e_t\), and confirm the OLS limit from problem 3; (iii) compare the AR forecast with a random-walk forecast using a Diebold–Mariano test. In R, use `arima()`, `forecast::auto.arima()` and `forecast::dm.test()`.

## Coding exercises

These exercises use **real data** and run in Python. Install the packages once with
`pip install wooldridge statsmodels linearmodels scipy`. The `wooldridge` package contains the datasets
from Wooldridge's *Introductory Econometrics*; `statsmodels` ships U.S. and Danish macroeconomic series.
Try the starter code first, then open the solution. Every printed result below was produced by running the solution code exactly as shown.

### Exercise 9.1 — Forecasting U.S. inflation: AR(p) by BIC against a random walk

*Data:* `macrodata` (statsmodels) — U.S. quarterly macro series, 1959Q1–2009Q3; `infl` = annualized CPI inflation.

1. Plot the ACF and PACF of inflation and choose an AR order by BIC on the 1960–1984 sample.
2. Produce pseudo out-of-sample one-quarter-ahead forecasts for 1985Q1–2009Q3 from (a) an AR(\(p\)) re-estimated each quarter on an expanding window and (b) a random walk (the forecast is the last observed value).
3. Report the RMSFE of each and a Diebold–Mariano test with a HAC standard error.

```python
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
y = sm.datasets.macrodata.load_pandas().data["infl"].to_numpy()
# observations 4..103 are 1960Q1..1984Q4; 104..202 are 1985Q1..2009Q3
# TODO: ar_select_order(y[4:104], maxlag=8, ic="bic"); loop over t, fit AutoReg(y[:t], p), forecast y[t]
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.stattools import acf, pacf
y = sm.datasets.macrodata.load_pandas().data["infl"].to_numpy()
print("ACF  lags 1-6:", acf(y[4:104], nlags=6)[1:].round(2))
print("PACF lags 1-6:", pacf(y[4:104], nlags=6)[1:].round(2))
p = len(ar_select_order(y[4:104], maxlag=8, ic="bic").ar_lags)
print("BIC-selected AR order on 1960-1984:", p)

e_ar, e_rw = [], []
for t in range(104, len(y)):
    fit = AutoReg(y[:t], lags=p, trend="c").fit()
    e_ar.append(y[t] - fit.predict(start=t, end=t)[0])
    e_rw.append(y[t] - y[t - 1])
e_ar, e_rw = np.array(e_ar), np.array(e_rw)
print(f"RMSFE 1985Q1-2009Q3:  AR({p}) {np.sqrt(np.mean(e_ar**2)):.3f}   random walk {np.sqrt(np.mean(e_rw**2)):.3f}")
d = e_ar**2 - e_rw**2
dm = smf.ols("d ~ 1", data={"d": d}).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
print(f"Diebold-Mariano: mean loss differential {d.mean():.3f}, HAC t = {dm.tvalues.iloc[0]:.2f}, p = {dm.pvalues.iloc[0]:.3f}")
```

Output:

```text
ACF  lags 1-6: [0.75 0.72 0.72 0.62 0.52 0.48]
PACF lags 1-6: [ 0.76  0.36  0.31 -0.06 -0.2  -0.1 ]
BIC-selected AR order on 1960-1984: 3
RMSFE 1985Q1-2009Q3:  AR(3) 2.457   random walk 2.913
Diebold-Mariano: mean loss differential -2.450, HAC t = -2.01, p = 0.044
```

Inflation in 1960–84 is highly persistent: the ACF decays slowly and the PACF cuts off after a few lags, pointing to a
low-order AR. Out of sample, after 1985, the AR model **beats** the random walk, because quarterly CPI inflation became
noisy and much less persistent during the Great Moderation; the random walk then carries every transitory spike forward.
That is the opposite of the 1970s pattern, and a reminder that forecasting rankings depend on the sample (Stock and
Watson, 2007). The DM test uses HAC standard errors; the sharp 2008Q4 deflation spike dominates the loss differential —
check how sensitive the result is to excluding it.

</details>

### Exercise 9.2 — The persistence of unemployment: AR(1), half-life and the Kendall bias

*Data:* `phillips` — U.S. annual unemployment and inflation, 1948–2003.

1. Estimate an AR(1) for the unemployment rate `unem`. Compute the half-life of a shock.
2. Correct \(\widehat\rho\) for the Kendall bias \(-(1+3\rho)/T\) and recompute the half-life.
3. Check the residuals for remaining autocorrelation with a Ljung–Box test.
4. Forecast unemployment for 2004–2008 with 95% intervals (ignoring parameter uncertainty).

```python
import numpy as np, wooldridge as wd
from statsmodels.tsa.ar_model import AutoReg
u = wd.data("phillips")["unem"].to_numpy()
fit = AutoReg(u, lags=1, trend="c").fit()
# TODO: half-life = ln(0.5)/ln(rho); bias-corrected rho_bc = rho + (1+3 rho)/T
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
ph = wd.data("phillips")
u = ph["unem"].to_numpy(); T = len(u) - 1
fit = AutoReg(u, lags=1, trend="c").fit()
c, rho = fit.params
hl = lambda r: np.log(0.5) / np.log(r)
rho_bc = rho + (1 + 3 * rho) / T
print(f"rho = {rho:.3f} (se {fit.bse[1]:.3f}), mean = {c/(1-rho):.2f}%, half-life = {hl(rho):.2f} years")
print(f"bias-corrected rho = {rho_bc:.3f}, half-life = {hl(rho_bc):.2f} years")
lb = acorr_ljungbox(fit.resid, lags=[4], return_df=True)
print(f"Ljung-Box(4) on residuals: Q = {lb['lb_stat'].iloc[0]:.2f}, p = {lb['lb_pvalue'].iloc[0]:.3f}")
s2 = fit.sigma2; last = u[-1]; mu = c / (1 - rho)
for h in range(1, 6):
    f = mu + rho**h * (last - mu); v = s2 * sum(rho**(2*j) for j in range(h))
    print(f"  {int(ph['year'].iloc[-1]) + h}: {f:.2f}%  [{f - 1.96*np.sqrt(v):.2f}, {f + 1.96*np.sqrt(v):.2f}]")
```

Output:

```text
rho = 0.742 (se 0.088), mean = 5.78%, half-life = 2.33 years
bias-corrected rho = 0.801, half-life = 3.12 years
Ljung-Box(4) on residuals: Q = 3.06, p = 0.548
  2004: 5.94%  [4.02, 7.87]
  2005: 5.90%  [3.51, 8.30]
  2006: 5.87%  [3.25, 8.49]
  2007: 5.85%  [3.12, 8.58]
  2008: 5.83%  [3.04, 8.63]
```

Annual unemployment is persistent: \(\widehat\rho\approx0.74\), a half-life of about 2.3 years. Because OLS is biased
downward in a sample of 56 years, the bias-corrected estimate (about 0.80) implies a half-life of about 3 years — a large
difference for a policy question such as how long a recession's unemployment lingers. The
Ljung–Box test does not detect remaining autocorrelation, but with annual data it has little power, and a business-cycle
AR(2) is worth trying. The intervals widen quickly toward the unconditional variance. Actual U.S. unemployment in 2004–2008
was 5.5, 5.1, 4.6, 4.6 and 5.8% — all inside the intervals. But 2009's 9.3% would have been far above them: an AR(1) cannot
anticipate a financial crisis.

</details>

## Reading
**Required:** Hamilton, *Time Series Analysis*, chapters 3–4; Stock and Watson, *Introduction to Econometrics*, chapter 15.
**Further:** Box, Jenkins, Reinsel and Ljung, *Time Series Analysis: Forecasting and Control*; Diebold and Mariano (1995), "Comparing predictive accuracy", *JBES* 13(3); Clark and West (2007), *Journal of Econometrics* 138(1); Stock and Watson (2007), "Why has U.S. inflation become harder to forecast?", *Journal of Money, Credit and Banking* 39(s1).
