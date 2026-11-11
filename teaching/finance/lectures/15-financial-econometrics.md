# Lecture 15 — Empirical Finance and Financial Econometrics

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Estimate return and factor models; model conditional volatility; design event studies; evaluate predictability with careful data controls.

## 1. Return regressions
A factor regression is
\[
R_{it}-R_{ft}=\alpha_i+\beta_i'f_t+\varepsilon_{it}.
\]
Inference should consider heteroskedasticity, serial correlation, cross-sectional dependence, and generated regressors. Rolling estimates reveal parameter instability but introduce window-choice sensitivity.

## 2. Volatility models
ARCH models specify conditional variance as a function of past squared innovations. GARCH(1,1) is
\[
h_t=\omega+\alpha\varepsilon_{t-1}^2+\beta h_{t-1},
\]
with positivity restrictions and covariance-stationarity condition \(\alpha+\beta<1\) in the standard model. Volatility clustering is distinct from predictability of returns.

## 3. Stochastic and realized volatility
Stochastic-volatility models treat latent variance as a stochastic process. Realized variance aggregates high-frequency intraday returns, with market microstructure noise requiring attention at very high sampling frequencies.

## 4. Predictability and cointegration
Predictive regressions relate future returns to instruments but face persistence and small-sample bias. Price levels may be nonstationary; cointegration tests are relevant for linked price or yield series, not a generic substitute for return modeling.

## 5. Research design and data integrity
Avoid look-ahead bias, survivorship bias, delisting omissions, data snooping, stale prices, and non-synchronous trading. Define universe membership as of each date and preserve point-in-time information.

## 6. Forecast evaluation
Use genuine out-of-sample splits, benchmark models, loss functions, and forecast-encompassing tests where appropriate. Report economic performance net of transaction costs, turnover, and implementation constraints.

## 7. Computational lab
Estimate CAPM and multifactor regressions, fit GARCH, produce volatility forecasts, and evaluate out of sample. Include code, data dictionary, software versions, and robustness checks.

## Problem set
1. State GARCH positivity and persistence conditions.
2. Distinguish conditional volatility from expected return.
3. Explain look-ahead and survivorship bias.
4. Design an out-of-sample return-prediction evaluation.

## Coding exercises

These exercises use **real market data** in Python: monthly Fama–French factor returns (1926–2018), daily S&P 500 and
NASDAQ prices (1999–2018), the VIX, Moody's AAA and BAA corporate bond yields (1919–2018), all bundled with the `arch`
package, plus U.S. Treasury yields and household data from the `wooldridge` package. Install once with
`pip install arch wooldridge statsmodels scipy numpy pandas`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 15.1 — Do dividend yields predict stock returns? Long horizons and Stambaugh bias

*Data:* `volat` — monthly S&P 500 index and dividend yield `divyld`, 1947–1993.

1. Compute log returns over the next 1, 12, 24 and 48 months (price returns plus the dividend yield accrued) and regress them on the current dividend yield, with Newey–West standard errors (lags = horizon).
2. Report slopes, \(t\)-statistics and \(R^2\). How do they change with the horizon?
3. Estimate the persistence of the dividend yield (AR(1)) and the correlation between its innovations and return innovations. Use Stambaugh's formula to compute the small-sample bias of the one-month slope: \(E[\widehat b-b]\approx\frac{\sigma_{uv}}{\sigma_v^2}E[\widehat\rho-\rho]\) with \(E[\widehat\rho-\rho]\approx-(1+3\rho)/T\).

```python
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
v = wd.data("volat")
lp = np.log(v["sp500"]); dy = v["divyld"] / 100
# TODO: future h-month return = lp.shift(-h) - lp + sum of dy/12 over the next h months; regress on dy
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
v = wd.data("volat")
lp = np.log(v["sp500"]); dy = v["divyld"] / 100
d = pd.DataFrame({"dy": 100 * dy})
for h in (1, 12, 24, 48):
    d[f"r{h}"] = 100 * (lp.shift(-h) - lp + (dy / 12)[::-1].rolling(h).sum()[::-1].shift(-1))
    s = d.dropna(subset=[f"r{h}"])
    r = smf.ols(f"r{h} ~ dy", data=s).fit(cov_type="HAC", cov_kwds={"maxlags": h})
    print(f"horizon {h:2d} months: slope {r.params['dy']:6.2f} (NW t = {r.tvalues['dy']:5.2f}), R2 = {r.rsquared:.3f}")
s = d.dropna(subset=["r1"]); T = len(s)
ar = smf.ols("dy ~ dy_lag", data=pd.DataFrame({"dy": s["dy"].iloc[1:].to_numpy(), "dy_lag": s["dy"].iloc[:-1].to_numpy()})).fit()
rho = ar.params["dy_lag"]; v_ = ar.resid.to_numpy()
u = smf.ols("r1 ~ dy", data=s).fit().resid.to_numpy()[:-1]
suv, sv2 = np.cov(u, v_)[0, 1], v_.var(ddof=1)
bias = suv / sv2 * (-(1 + 3 * rho) / T)
print(f"dividend-yield AR(1) rho = {rho:.4f}; corr(return, yield innovations) = {np.corrcoef(u, v_)[0,1]:.3f}")
print(f"Stambaugh bias in the 1-month slope ~ {bias:.3f} (vs estimate {smf.ols('r1 ~ dy', data=s).fit().params['dy']:.3f})")
```

Output:

```text
horizon  1 months: slope   0.36 (NW t =  2.69), R2 = 0.014
horizon 12 months: slope   5.59 (NW t =  4.64), R2 = 0.187
horizon 24 months: slope  10.03 (NW t =  4.82), R2 = 0.356
horizon 48 months: slope  16.29 (NW t =  7.12), R2 = 0.603
dividend-yield AR(1) rho = 0.9901; corr(return, yield innovations) = -0.890
Stambaugh bias in the 1-month slope ~ 0.129 (vs estimate 0.364)
```

The one-month slope is significant but tiny in \(R^2\) (1.4%). The slopes and \(R^2\) grow almost proportionally with the
horizon, reaching 60% at four years. This is the famous long-horizon predictability. But the dividend yield is almost a unit
root (\(\rho\approx0.99\)), and its innovations are strongly negatively correlated with returns (\(-0.89\)), so the
one-month slope is biased upward: Stambaugh's formula attributes about a third of it to small-sample bias. Overlapping
long-horizon regressions add no independent information — they mostly cumulate the short-horizon slope — and their
\(t\)-statistics are badly oversized even with Newey–West errors (Boudoukh, Richardson and Whitelaw, 2008).

</details>

### Exercise 15.2 — Asymmetric volatility: GJR-GARCH and the leverage effect

*Data:* Daily S&P 500 log returns 1999–2018, and the VIX 2014–2018.

1. Estimate GARCH(1,1) and GJR-GARCH(1,1,1) models with Student-\(t\) errors. Compare log-likelihoods and BIC. Is the asymmetry term \(\gamma\) significant?
2. Compute the "news impact" — next-day variance after a −2% and a +2% return — implied by the GJR model.
3. Compare the GJR one-day-ahead volatility forecast (annualized) with the VIX over 2014–2018: correlation and average gap. Why is the VIX systematically higher?

```python
import numpy as np, pandas as pd
from arch import arch_model
from arch.data import sp500, vix
r = 100 * np.log(sp500.load()["Adj Close"]).diff().dropna()
g = arch_model(r, vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
# TODO: compare with o=0; news impact; conditional_volatility * sqrt(252) vs vix
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd
from arch import arch_model
from arch.data import sp500, vix
r = 100 * np.log(sp500.load()["Adj Close"]).diff().dropna()
g0 = arch_model(r, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
g1 = arch_model(r, vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
print(f"GARCH: logL {g0.loglikelihood:.1f}, BIC {g0.bic:.1f};  GJR: logL {g1.loglikelihood:.1f}, BIC {g1.bic:.1f}")
P = g1.params
print(f"GJR: alpha {P['alpha[1]']:.4f}, gamma {P['gamma[1]']:.4f} (t = {g1.tvalues['gamma[1]']:.1f}), beta {P['beta[1]']:.4f}")
s2 = g1.conditional_volatility.iloc[-1]**2
for shock in (-2.0, 2.0):
    e = shock - P["mu"]
    nxt = P["omega"] + (P["alpha[1]"] + P["gamma[1]"] * (e < 0)) * e**2 + P["beta[1]"] * s2
    print(f"next-day volatility after a {shock:+.0f}% day: {np.sqrt(nxt):.2f}% (from {np.sqrt(s2):.2f}%)")
vol = g1.conditional_volatility * np.sqrt(252)
df = pd.concat([vol.rename("garch"), vix.load()["vix"].rename("vix")], axis=1).dropna().loc["2014":"2018"]
print(f"2014-2018: corr(GJR vol, VIX) = {df.corr().iloc[0,1]:.3f}; mean GJR vol {df.garch.mean():.1f}%, mean VIX {df.vix.mean():.1f}%")
```

Output:

```text
GARCH: logL -6834.5, BIC 13711.6;  GJR: logL -6748.3, BIC 13547.7
GJR: alpha 0.0000, gamma 0.1815 (t = 8.1), beta 0.8987
next-day volatility after a -2% day: 2.00% (from 1.90%)
next-day volatility after a +2% day: 1.80% (from 1.90%)
2014-2018: corr(GJR vol, VIX) = 0.844; mean GJR vol 13.0%, mean VIX 14.9%
```

The GJR model improves the fit substantially (BIC falls by over 160). The asymmetry term is large and highly significant,
while the symmetric \(\alpha\) is essentially zero: only negative returns raise future volatility. After a −2% day, volatility
rises; after a +2% day, it falls. This is the leverage (or volatility-feedback) effect. The GJR volatility forecast tracks the
VIX closely (correlation 0.84) but is about 2 points lower on average: the VIX is a risk-neutral expectation and includes the
variance risk premium measured in Lecture 8.

</details>

## Further reading
Campbell, Lo and MacKinlay, *The Econometrics of Financial Markets*; Engle (1982); Bollerslev (1986); Tsay, *Analysis of Financial Time Series*.
