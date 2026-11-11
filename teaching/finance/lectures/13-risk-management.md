# Lecture 13 — Financial Risk Management: VaR, Expected Shortfall, and Stress Testing

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Quantify market and credit risk; estimate VaR and expected shortfall; understand tail dependence, model risk, and stress testing.

## 1. Value at Risk
For loss \(L\), confidence level \(\alpha\), VaR is the loss quantile
\[
VaR_\alpha(L)=\inf\{\ell:P(L\leq\ell)\geq\alpha\}.
\]
It does not describe the severity of losses beyond the quantile.

## 2. Expected shortfall
For continuous distributions,
\[
ES_\alpha(L)=E[L\mid L\geq VaR_\alpha(L)].
\]
For distributions with atoms, use a quantile-integral definition to handle probability mass at VaR. ES summarizes tail severity beyond the threshold.

## 3. Estimation methods
Historical simulation revalues positions under observed changes. Variance-covariance methods impose distributional assumptions, often multivariate normality. Monte Carlo simulation generates risk factors from a specified stochastic model and reprices the portfolio.

## 4. Credit and counterparty risk
Credit portfolio risk depends on default probabilities, loss severity, exposure, and dependence. Counterparty exposure can vary with market values and netting agreements. Wrong-way risk occurs when exposure and counterparty default risk increase together.

## 5. Stress testing and model risk
Stress tests apply coherent adverse scenarios; reverse stress testing identifies conditions that cause failure. Backtesting compares realized losses with risk forecasts. Model risk includes misspecification, parameter uncertainty, implementation error, and unstable correlations.

## 6. Worked example
Given ordered portfolio losses, calculate empirical 95% VaR and average losses in the worst 5%, stating the finite-sample quantile convention.

## 7. Computational lab
Estimate historical and parametric VaR/ES. Backtest exceedances, compare normal and heavy-tailed innovations, and run a correlated market shock scenario.

## Problem set
1. Define VaR precisely.
2. Explain why ES captures tail severity.
3. Compare historical and parametric simulation.
4. Design a stress test for a leveraged portfolio.

## Coding exercises

These exercises use **real market data** in Python: monthly Fama–French factor returns (1926–2018), daily S&P 500 and
NASDAQ prices (1999–2018), the VIX, Moody's AAA and BAA corporate bond yields (1919–2018), all bundled with the `arch`
package, plus U.S. Treasury yields and household data from the `wooldridge` package. Install once with
`pip install arch wooldridge statsmodels scipy numpy pandas`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 13.1 — Value-at-Risk and Expected Shortfall for the S&P 500, with backtests

*Data:* Daily S&P 500 log returns, 1999–2018.

1. Using a rolling 500-day window, compute one-day 99% VaR each day by (a) historical simulation, (b) a normal distribution with the window's mean and SD, and (c) RiskMetrics EWMA volatility (\(\lambda=0.94\)) with normal quantiles.
2. Count VaR violations over 2001–2018 and run Kupiec's unconditional-coverage likelihood-ratio test for each method.
3. Report the violations in 2008 alone. Why does historical simulation over a long window react slowly to a crisis?
4. Compute the full-sample 99% Expected Shortfall by historical simulation and compare it with VaR.

```python
import numpy as np, pandas as pd
from scipy.stats import norm, chi2
from arch.data import sp500
r = 100 * np.log(sp500.load()["Adj Close"]).diff().dropna()
# TODO: rolling quantile, rolling std, EWMA variance; violations; Kupiec LR
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd
from scipy.stats import norm, chi2
from arch.data import sp500
r = 100 * np.log(sp500.load()["Adj Close"]).diff().dropna()
hs = -r.rolling(500).quantile(0.01).shift(1)
nv = -(r.rolling(500).mean() - norm.ppf(0.99) * r.rolling(500).std()).shift(1)
ew = np.zeros(len(r)); ew[0] = r.iloc[:50].var()
for t in range(1, len(r)): ew[t] = 0.94 * ew[t - 1] + 0.06 * r.iloc[t - 1]**2
em = pd.Series(norm.ppf(0.99) * np.sqrt(ew), index=r.index)
def kupiec(v, p=0.01):
    n, x = len(v), v.sum(); ph = x / n
    return -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - ph) - x * np.log(ph))
for lab, var in (("historical", hs), ("normal", nv), ("EWMA", em)):
    s = pd.concat([r, var], axis=1, keys=["r", "var"]).loc["2001":].dropna(); v = (s.r < -s["var"])
    lr = kupiec(v)
    print(f"{lab:10s}: {v.sum():3d} violations in {len(v)} days ({v.mean():.2%}, expected 1%), Kupiec LR = {lr:.2f} (p = {chi2.sf(lr, 1):.3f}); in 2008: {v.loc['2008'].sum()}")
q = r.quantile(0.01)
print(f"full sample: 99% VaR {-q:.2f}%, 99% Expected Shortfall {-r[r <= q].mean():.2f}%")
```

Output:

```text
historical:  73 violations in 4527 days (1.61%, expected 1%), Kupiec LR = 14.47 (p = 0.000); in 2008: 21
normal    : 113 violations in 4527 days (2.50%, expected 1%), Kupiec LR = 72.30 (p = 0.000); in 2008: 28
EWMA      :  96 violations in 4527 days (2.12%, expected 1%), Kupiec LR = 43.44 (p = 0.000); in 2008: 9
full sample: 99% VaR 3.36%, 99% Expected Shortfall 4.81%
```

All three methods produce too many violations over 2001–2018, and Kupiec's test rejects correct coverage for each. The
normal model is worst (2.5% violations), because daily returns have fat tails. Historical simulation does better on average
but clusters its violations: 21 in 2008 alone, because a 500-day window adapts slowly to a sudden jump in volatility. EWMA
reacts quickly (only 9 violations in 2008) but uses normal quantiles, so it fails in the tails overall. Expected Shortfall
(4.8%) is much larger than VaR (3.4%): the average loss beyond the VaR threshold is large, which is why Basel III moved to ES.

</details>

### Exercise 13.2 — GARCH volatility and filtered historical simulation

*Data:* Daily S&P 500 log returns, 1999–2018.

1. Estimate a GARCH(1,1) with Student-\(t\) errors using the `arch` package. Report the persistence \(\alpha+\beta\), the half-life of volatility shocks and the degrees of freedom.
2. Compare the conditional volatility at its 2008 peak and at the end of 2018 with the unconditional volatility.
3. Compute a one-day 99% VaR for 2 January 2019 by **filtered historical simulation**: scale the empirical quantile of standardized residuals by the forecast volatility. Compare with the GARCH-\(t\) quantile.

```python
import numpy as np
from arch import arch_model
from arch.data import sp500
r = 100 * np.log(sp500.load()["Adj Close"]).diff().dropna()
res = arch_model(r, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
# TODO: persistence, half-life, conditional_volatility, forecast(horizon=1), std_resid quantile
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np
from scipy import stats
from arch import arch_model
from arch.data import sp500
r = 100 * np.log(sp500.load()["Adj Close"]).diff().dropna()
res = arch_model(r, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
a, b, nu = res.params["alpha[1]"], res.params["beta[1]"], res.params["nu"]
print(res.params.round(4).to_dict())
print(f"persistence alpha + beta = {a + b:.4f}, half-life {np.log(0.5)/np.log(a + b):.0f} days, t degrees of freedom {nu:.1f}")
cv = res.conditional_volatility; unc = np.sqrt(res.params["omega"] / (1 - a - b))
print(f"conditional vol: 2008 peak {cv.loc['2008'].max():.2f}% ({cv.loc['2008'].idxmax().date()}), end-2018 {cv.iloc[-1]:.2f}%, unconditional {unc:.2f}% (sample sd {r.std():.2f}%)")
f = np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0])
z = res.std_resid.dropna()
var_fhs = -(res.params["mu"] + f * np.quantile(z, 0.01))
var_t = -(res.params["mu"] + f * stats.t.ppf(0.01, nu) * np.sqrt((nu - 2) / nu))
print(f"forecast vol for the next day {f:.2f}%: 99% VaR by filtered historical simulation {var_fhs:.2f}%, GARCH-t {var_t:.2f}%")
```

Output:

```text
{'mu': 0.0646, 'omega': 0.0086, 'alpha[1]': 0.0995, 'beta[1]': 0.9002, 'nu': 6.5094}
persistence alpha + beta = 0.9996, half-life 1980 days, t degrees of freedom 6.5
conditional vol: 2008 peak 5.61% (2008-10-16), end-2018 2.03%, unconditional 4.97% (sample sd 1.20%)
forecast vol for the next day 1.94%: 99% VaR by filtered historical simulation 5.22%, GARCH-t 4.88%
```

Volatility is extremely persistent: \(\alpha+\beta\approx0.9996\), close to integrated GARCH, so the implied "unconditional"
volatility (5%) is unreliable — far above the sample SD of 1.2%. In practice, shocks to volatility decay over months. The
Student-\(t\) degrees of freedom (about 6.5) confirm fat tails even after conditioning on volatility. Conditional volatility
peaked at about 5.6% a day in October 2008. Filtered historical simulation gives a slightly higher VaR than the GARCH-\(t\)
quantile, because the empirical standardized residuals have a heavier left tail than a symmetric \(t\) distribution.

</details>

## Further reading
Jorion, *Value at Risk*; McNeil, Frey and Embrechts, *Quantitative Risk Management*.
