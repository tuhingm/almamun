# Lecture 10 — Unit Roots, Structural Breaks, and Persistence

## Learning objectives
By the end of this lecture you should be able to:

1. distinguish trend-stationary from difference-stationary (integrated) processes, and explain why the distinction matters for inference and policy;
2. derive the Dickey–Fuller regression, state its non-standard limiting distribution, and choose the correct deterministic terms and critical values;
3. use ADF, DF-GLS and KPSS tests together, and understand their size and power trade-offs;
4. test for structural breaks with known and unknown break dates (Chow, Andrews sup-\(F\), Bai–Perron);
5. explain why ignored breaks bias unit-root tests toward non-rejection, and how break-robust tests respond.

## Motivation
Many macroeconomic and financial series — GDP, prices, exchange rates, interest rates — wander without an obvious tendency to revert to a fixed mean. Whether shocks to such series are *permanent* (a unit root) or *transitory* (stationary around a trend) changes forecasts, the interpretation of business cycles, and the validity of every regression in which these series appear. Yet the question is statistically hard: over a few decades of data, a unit root and a root of 0.97 are nearly indistinguishable. This lecture gives the tools and an honest sense of their limits.

## 1. Integrated processes
A process is **integrated of order \(d\)**, \(I(d)\), if it must be differenced \(d\) times to become stationary. The prototype \(I(1)\) process is the random walk with drift:
\[
y_t=\mu+y_{t-1}+\varepsilon_t=y_0+\mu t+\sum_{s=1}^t\varepsilon_s .
\]
Compare the **trend-stationary** model \(y_t=\alpha+\mu t+u_t\), with \(u_t\) stationary.

| | Trend-stationary | Difference-stationary (unit root) |
|---|---|---|
| Effect of a shock on \(y_{t+h}\) | dies out as \(h\to\infty\) | permanent |
| \(\operatorname{Var}(y_t)\) | bounded | grows linearly in \(t\) |
| Forecast interval as \(h\to\infty\) | bounded | widens without limit |
| Appropriate transformation | detrend | difference |

**Beveridge–Nelson decomposition.** Any \(I(1)\) process with \(\Delta y_t=\mu+\psi(L)\varepsilon_t\) can be written as a random-walk **permanent component** \(\tau_t=\tau_{t-1}+\mu+\psi(1)\varepsilon_t\) plus a stationary **transitory component**. The long-run impact of a shock is \(\psi(1)=\sum_j\psi_j\). A process that is "nearly" trend-stationary has \(\psi(1)\) close to zero.

## 2. Why unit roots break standard inference
In \(y_t=\phi y_{t-1}+\varepsilon_t\) with \(\phi=1\), the OLS estimator is **super-consistent**: \(T(\widehat\phi-1)\) converges to a non-degenerate distribution, so \(\widehat\phi\) converges at rate \(T\), not \(\sqrt T\). The limit is non-normal and skewed to the left. It is expressed through a standard Brownian motion \(W(r)\) on \([0,1]\):
\[
T(\widehat\phi-1)\to_d\frac{\tfrac12\big(W(1)^2-1\big)}{\int_0^1W(r)^2\,dr},\qquad
t_{\widehat\phi=1}\to_d\frac{\tfrac12\big(W(1)^2-1\big)}{\big(\int_0^1W(r)^2\,dr\big)^{1/2}} .
\]
These results follow from the **functional CLT** (\(T^{-1/2}\sum_{s\le rT}\varepsilon_s/\sigma\Rightarrow W(r)\)) and the continuous mapping theorem. The second limit is the **Dickey–Fuller distribution**. Its 5% critical value is about \(-1.95\) with no deterministics, \(-2.86\) with a constant, and \(-3.41\) with a constant and trend, far from the normal \(-1.645\).

**Spurious regression.** Regressing one independent random walk on another yields "significant" \(t\) statistics with probability approaching one, and \(R^2\) does not go to zero (Granger and Newbold, 1974; Phillips, 1986). The \(t\) statistic diverges at rate \(\sqrt T\). A telltale sign is a high \(R^2\) together with a Durbin–Watson statistic near zero. Lecture 12 explains when a regression in levels of \(I(1)\) variables *is* meaningful (cointegration).

## 3. Dickey–Fuller tests
Rewrite the AR(1) as \(\Delta y_t=\pi y_{t-1}+\varepsilon_t\), with \(\pi=\phi-1\), and test \(H_0:\pi=0\) (unit root) against \(H_1:\pi<0\) (stationarity). Three versions differ in their deterministic terms:
\[
\Delta y_t=\pi y_{t-1}+\varepsilon_t;\qquad \Delta y_t=\alpha+\pi y_{t-1}+\varepsilon_t;\qquad \Delta y_t=\alpha+\delta t+\pi y_{t-1}+\varepsilon_t .
\]
**Choosing deterministics.** The alternative must be plausible: for a trending series (log GDP), include a trend, so the alternative is trend-stationarity. For a non-trending series (interest rates, unemployment), include only a constant. Including unnecessary trends costs power; omitting necessary ones can make the test inconsistent.

**Augmented DF (ADF).** Serial correlation in \(\Delta y_t\) is absorbed by lagged differences:
\[
\Delta y_t=\alpha+\delta t+\pi y_{t-1}+\sum_{j=1}^{p}\gamma_j\Delta y_{t-j}+\varepsilon_t .
\]
The \(t\) statistic on \(\pi\) has the same DF limit distribution. Choose \(p\) by the modified AIC of Ng and Perron (2001), or by a general-to-specific sequence starting from \(p_{\max}=\lfloor12(T/100)^{1/4}\rfloor\). Too few lags distort size; too many lose power.

**Phillips–Perron** corrects the DF statistic non-parametrically with a long-run variance estimate instead of adding lags. It has poor size when there is a large negative MA root, as in many inflation series.

**Low power and DF-GLS.** Against near-unit-root alternatives such as \(\phi=0.95\), ADF tests have very low power in samples of typical macro length. **DF-GLS** (Elliott, Rothenberg and Stock, 1996) first detrends by GLS under a local alternative, then runs the ADF regression without deterministics. It is nearly efficient and should be the default.

**KPSS (stationarity as the null).** Kwiatkowski, Phillips, Schmidt and Shin (1992) test \(H_0\): stationary (around a level or trend), using the scaled sum of squared partial sums of residuals with a long-run variance estimate. Using ADF/DF-GLS and KPSS together gives four possible outcomes:

| ADF rejects unit root? | KPSS rejects stationarity? | Reading |
|---|---|---|
| yes | no | evidence of stationarity |
| no | yes | evidence of a unit root |
| no | no | data uninformative (low power) |
| yes | yes | possible misspecification: breaks, nonlinearity, long memory |

## 4. Structural breaks
**Known break date (Chow test).** For a break at \(T_b\), interact all regressors with a post-break dummy and test the interactions jointly with a robust Wald test. The break date must be chosen *without* looking at the data. Choosing the date where the series visibly changes invalidates the critical values.

**Unknown break date (Andrews, 1993; Quandt).** Compute the Chow statistic \(F(\tau)\) for every candidate break fraction \(\tau\in[\tau_0,1-\tau_0]\), typically with 15% trimming, and use
\[
\sup F=\max_{\tau}F(\tau),
\]
which has a non-standard limit (a supremum of a squared Brownian-bridge process) with tabulated critical values. Andrews and Ploberger (1994) propose average and exponential versions with optimal power properties. The break date estimator \(\widehat T_b=\arg\min SSR(T_b)\) is consistent at rate \(T\).

**Multiple breaks (Bai and Perron, 1998, 2003).** These methods sequentially test \(\ell\) versus \(\ell+1\) breaks, estimate break dates by global minimization of the SSR using dynamic programming, and provide confidence intervals for the break dates.

**Parameter instability without discrete breaks.** CUSUM tests of recursive residuals (Brown, Durbin and Evans, 1975) and the Nyblom (1989) test detect gradual drift. Time-varying-parameter models estimate it directly.

## 5. Breaks and unit roots
**Perron (1989)** showed that a stationary series with a one-time break in level or trend looks highly persistent to an ADF test that ignores the break. The test is then **biased toward non-rejection** of the unit root. Allowing for the 1929 crash and the 1973 oil-price slowdown, he rejected unit roots in many of the Nelson–Plosser (1982) series. When the break date is unknown, treating it as known biases the test the other way. **Zivot and Andrews (1992)** and Lumsdaine–Papell search over break dates under the unit-root null, with appropriately larger critical values.

**Practical conclusion.** Unit-root and break tests are complements. Persistence, breaks, nonlinearity (for example, threshold adjustment) and long memory (fractional integration) can all produce similar-looking data. Choose the model that the economics supports, and show robustness to the alternatives.

## Worked example
**Is U.S. real GDP trend-stationary?** Take quarterly log real GDP, 1960Q1–2019Q4 (\(T=240\)), with an ADF regression including a constant, a trend and 2 lags. Typical results:

- ADF \(t\)-statistic around \(-2.5\), against a 5% critical value of \(-3.41\): do not reject the unit root;
- KPSS with trend: reject trend-stationarity at 5%;
- the sup-\(F\) test for a break in the trend slope identifies a break near 1973 (the productivity slowdown) and rejects stability.

Reading: the evidence is consistent with a unit root, but also with trend-stationarity around a broken trend (Perron, 1989). A Zivot–Andrews test with an endogenous break typically comes close to rejecting, depending on the sample and lag choice. For forecasting at short horizons, the distinction matters little. For long-horizon forecasts and for business-cycle measurement — the size of the "cycle" depends on the trend model — it matters a lot. (The figures above are illustrative magnitudes; reproduce them in the lab with current FRED vintages.)

## Problem set
1. **Variance of a random walk.** For \(y_t=y_{t-1}+\varepsilon_t\) with \(y_0=0\), show that \(\operatorname{Var}(y_t)=t\sigma^2\) and \(\operatorname{Corr}(y_t,y_{t-k})=\sqrt{(t-k)/t}\). What does the second result imply for the sample ACF?

    <details class="solution" markdown="1"><summary>Solution</summary>

    \(y_t=\sum_{s=1}^t\varepsilon_s\), so \(\operatorname{Var}(y_t)=t\sigma^2\) and \(\operatorname{Cov}(y_t,y_{t-k})=(t-k)\sigma^2\). The correlation is \((t-k)/\sqrt{t(t-k)}=\sqrt{(t-k)/t}\to1\) for fixed \(k\). The sample ACF of a random walk decays very slowly, roughly linearly, which is the classic visual signature.
    </details>

2. **Deterministic terms.** Explain why running a DF test without a trend on a series that is trend-stationary with a strong trend can fail to reject the unit root even in very large samples.

    <details class="solution" markdown="1"><summary>Solution</summary>

    Without a trend, the regression approximates the linear trend through the lagged level \(y_{t-1}\), pushing \(\widehat\pi\) toward 0. The estimated persistence converges to 1, so the test is inconsistent against the trend-stationary alternative.
    </details>

3. **Spurious regression.** Simulate two independent random walks with \(T=100\), regress one on the other 1,000 times, and record how often \(|t|>1.96\). Repeat with \(T=1{,}000\). Explain the pattern.

    <details class="solution" markdown="1"><summary>Solution</summary>

    The rejection rate is about 75% at \(T=100\) and higher at \(T=1{,}000\), because the \(t\) statistic diverges at rate \(\sqrt T\) (Phillips, 1986). Regressing first differences, or including lags of both variables, restores valid inference.
    </details>

4. **Chow with a data-chosen date.** Why does choosing \(T_b\) to maximize the Chow statistic, and then using \(F\) critical values, over-reject?

    <details class="solution" markdown="1"><summary>Solution</summary>

    The maximum of many correlated \(F\) statistics is stochastically larger than a single \(F\). Its null distribution is the sup-\(F\) distribution, whose 5% critical value is far above the single-break \(F\) critical value. For one coefficient with 15% trimming, the asymptotic 5% critical value for the Wald form is about 8.7 (Andrews, 1993) versus 3.84 for \(\chi^2_1\).
    </details>

5. **Perron's insight.** Simulate \(y_t=0.5y_{t-1}+\varepsilon_t+3\cdot1\{t>100\}\) for \(T=200\), and run ADF tests with and without a level-shift dummy. Explain the result.

    <details class="solution" markdown="1"><summary>Solution</summary>

    Ignoring the shift, ADF often fails to reject: the level shift looks like a permanent stochastic shock, and \(\widehat\phi\) is biased toward 1. Including the shift dummy (with its impulse dummy at the break date, as in Perron's specification), the test rejects easily, because the remaining dynamics have \(\phi=0.5\).
    </details>

## Computational lab
**Goal:** see the DF distribution, the low power of ADF, and the effect of a break.

```python
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

rng = np.random.default_rng(10)
T, R = 200, 2000
# 1. DF distribution under the null (constant only)
tstats = [adfuller(np.cumsum(rng.normal(size=T)), maxlag=0, regression="c", autolag=None)[0] for _ in range(R)]
print("simulated 5% critical value:", np.quantile(tstats, 0.05).round(2), "(tabulated -2.88 for T=200)")

# 2. power against phi = 0.95
def ar1(phi):
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = phi * y[t - 1] + rng.normal()
    return y
power = np.mean([adfuller(ar1(0.95), regression="c", autolag="AIC")[1] < 0.05 for _ in range(500)])
print("ADF power vs phi=0.95, T=200:", power)

# 3. stationary series with a level break
y = ar1(0.5) + 3 * (np.arange(T) > 100)
print("ADF p-value ignoring break:", round(adfuller(y, regression="c")[1], 3),
      "| KPSS p-value:", kpss(y, regression="c", nlags="auto")[1])
```

Tasks: (i) compare the simulated critical value with the tabulated one; (ii) plot power as a function of \(\phi\in[0.8,1]\) for \(T=100,200,500\); (iii) apply `statsmodels.tsa.stattools.zivot_andrews` to the break series; (iv) download log real GDP (FRED: GDPC1) and reproduce the worked example. In R, use `urca::ur.df`, `ur.ers` (DF-GLS), `ur.kpss`, and `strucchange::Fstats` / `breakpoints`.

## Reading
**Required:** Hamilton, *Time Series Analysis*, chapters 15–17; Stock and Watson, *Introduction to Econometrics*, chapter 15.6–15.7.
**Further:** Dickey and Fuller (1979), *JASA* 74(366); Nelson and Plosser (1982), "Trends and random walks in macroeconomic time series", *Journal of Monetary Economics* 10(2); Perron (1989), "The great crash, the oil price shock, and the unit root hypothesis", *Econometrica* 57(6); Elliott, Rothenberg and Stock (1996), *Econometrica* 64(4); Kwiatkowski, Phillips, Schmidt and Shin (1992), *Journal of Econometrics* 54(1–3); Andrews (1993), "Tests for parameter instability and structural change with unknown change point", *Econometrica* 61(4); Bai and Perron (2003), *Journal of Applied Econometrics* 18(1); Zivot and Andrews (1992), *JBES* 10(3).
