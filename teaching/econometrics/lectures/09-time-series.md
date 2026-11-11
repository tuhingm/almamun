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

## Reading
**Required:** Hamilton, *Time Series Analysis*, chapters 3–4; Stock and Watson, *Introduction to Econometrics*, chapter 15.
**Further:** Box, Jenkins, Reinsel and Ljung, *Time Series Analysis: Forecasting and Control*; Diebold and Mariano (1995), "Comparing predictive accuracy", *JBES* 13(3); Clark and West (2007), *Journal of Econometrics* 138(1); Stock and Watson (2007), "Why has U.S. inflation become harder to forecast?", *Journal of Money, Credit and Banking* 39(s1).
