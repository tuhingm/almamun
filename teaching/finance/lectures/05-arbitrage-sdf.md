# Lecture 05 — Arbitrage Pricing, Stochastic Discount Factors, and No-Arbitrage

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Use no-arbitrage restrictions to price payoffs; derive SDF relations; understand factor pricing and risk-premium covariance.

## 1. Law of one price
Two portfolios with identical state-contingent payoffs must have the same price. If not, a long-short strategy can generate an arbitrage under the idealized frictionless assumptions.

## 2. SDF pricing
For payoff \(X_{t+1}\),
\[
P_t=E_t[M_{t+1}X_{t+1}].
\]
For a gross return \(R_{t+1}\), \(1=E_t[M_{t+1}R_{t+1}]\). Decomposing the expectation gives
\[
E_t(R_{t+1})-R_{f,t}
=-\frac{\operatorname{Cov}_t(M_{t+1},R_{t+1})}{E_t(M_{t+1})},
\]
when the risk-free gross return is \(R_{f,t}=1/E_tM_{t+1}\).

## 3. Linear factor pricing
If the SDF is affine in factors, expected excess returns are related to factor covariances. APT derives approximate linear pricing restrictions under diversified idiosyncratic risk and no-arbitrage; it does not by itself identify the economically correct factors.

## 4. Traded and non-traded factors
A traded factor has a return that can be held as a portfolio. A non-traded factor may represent macroeconomic risk and requires a factor-mimicking portfolio or additional restrictions for empirical pricing.

## 5. Worked example
Suppose an SDF is high in bad states. An asset paying more in those states covaries positively with the SDF and can command a lower expected return. Explain the insurance interpretation.

## 6. Computational lab
Estimate candidate factor models and compare pricing errors, cross-sectional fit, and stability. Construct factor-mimicking portfolios and document the normalization.

## Problem set
1. Derive the SDF pricing equation.
2. Derive the covariance expression for the risk premium.
3. State the law of one price.
4. Distinguish APT from CAPM.

## Coding exercises

These exercises use **real market data** in Python: monthly Fama–French factor returns (1926–2018), daily S&P 500 and
NASDAQ prices (1999–2018), the VIX, Moody's AAA and BAA corporate bond yields (1919–2018), all bundled with the `arch`
package, plus U.S. Treasury yields and household data from the `wooldridge` package. Install once with
`pip install arch wooldridge statsmodels scipy numpy pandas`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 5.1 — A linear stochastic discount factor that prices the factors

*Data:* Fama–French monthly excess returns `Mkt-RF`, `SMB`, `HML`, 1963–2018.

Look for an SDF of the form \(m_t=1-b'(f_t-E f)\) such that \(E[m_tR^e_t]=0\) for the three factor returns.

1. Show that the moment conditions give \(b=\Sigma_f^{-1}\mu\), and compute \(b\).
2. Verify that the pricing errors \(E[mR^e]\) are zero in sample.
3. Compute \(\sigma(m)\) and compare it with the maximum Sharpe ratio of the three factors (the Hansen–Jagannathan bound). Why are they equal?
4. Use \(b\) estimated on 1963–1990 to price the 1991–2018 returns. Report the out-of-sample pricing errors (annualized).

```python
import numpy as np, pandas as pd
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

F = ff.loc["1963-07":, ["Mkt-RF", "SMB", "HML"]].to_numpy()
# TODO: b = solve(cov(F), mean(F)); m = 1 - (F - mean) @ b; check mean(m[:,None] * F)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

X = ff.loc["1963-07":, ["Mkt-RF", "SMB", "HML"]]
def sdf(Y):
    mu, S = Y.mean(0), np.cov(Y, rowvar=False, ddof=0); return np.linalg.solve(S, mu), mu
F = X.to_numpy(); b, mu = sdf(F); m = 1 - (F - mu) @ b
print("b =", b.round(3))
print("in-sample pricing errors E[m Re] (per month):", (m[:, None] * F).mean(0).round(8))
maxSR = np.sqrt(mu @ np.linalg.solve(np.cov(F, rowvar=False, ddof=0), mu))
print(f"sd(m) = {m.std():.4f} per month;  max Sharpe ratio = {maxSR:.4f} per month ({np.sqrt(12)*maxSR:.3f} annualized)")
ins, oos = X.loc[:"1990-12"].to_numpy(), X.loc["1991-01":].to_numpy()
b1, mu1 = sdf(ins); m_oos = 1 - (oos - mu1) @ b1
print("out-of-sample pricing errors, 1991-2018 (% per year):", (1200 * (m_oos[:, None] * oos).mean(0)).round(2))
```

Output:

```text
b = [3.369 1.901 5.955]
in-sample pricing errors E[m Re] (per month): [ 0. -0. -0.]
sd(m) = 0.2031 per month;  max Sharpe ratio = 0.2031 per month (0.704 annualized)
out-of-sample pricing errors, 1991-2018 (% per year): [ 4.01  1.67 -5.38]
```

The SDF coefficients are the (unnormalized) mean–variance weights, \(b=\Sigma^{-1}\mu\), and they price all three factors
exactly in sample. The standard deviation of \(m\) equals the maximum Sharpe ratio of the factors (0.20 per month, 0.70 a
year): the SDF built from the tangency portfolio is the minimum-variance SDF, so it attains the Hansen–Jagannathan bound. Out
of sample, the SDF estimated on 1963–90 misprices the 1991–2018 returns by several percent a year — it overprices value (HML
earned less than the SDF required). The same overfitting problem as in Exercise 3.1, in SDF language.

</details>

### Exercise 5.2 — Arbitrage bounds in the term structure of T-bills

*Data:* `intqrt` — U.S. quarterly yields on 3-, 6- and 12-month Treasury bills, 1960s–1990s.

With no arbitrage, the price of a 6-month bill must equal the price of rolling two 3-month bills **in expectation, under the risk-neutral measure**, and forward rates are pinned down by spot rates.

1. Compute the implied 3-month forward rate three months ahead from the 3- and 6-month yields, \(f_t=2y^{(6)}_t-y^{(3)}_t\) (annualized, simple approximation).
2. Compare the forward rate with the 3-month yield actually realized three months later. Compute the average forecast error (the term premium) and its HAC standard error.
3. Regress the realized change in the 3-month rate on the forward–spot spread (Fama, 1984). Under the pure expectations hypothesis, the slope is 1. What do you find?

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("intqrt")
d["fwd"] = 2 * d["r6"] - d["r3"]; d["r3_next"] = d["r3"].shift(-1)
# TODO: error = r3_next - fwd; regression (r3_next - r3) ~ (fwd - r3)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("intqrt")
d["fwd"] = 2 * d["r6"] - d["r3"]; d["r3_next"] = d["r3"].shift(-1)
d["err"] = d["r3_next"] - d["fwd"]; d["dr"] = d["r3_next"] - d["r3"]; d["spread"] = d["fwd"] - d["r3"]
s = d.dropna(subset=["err"])
e = smf.ols("err ~ 1", data=s).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
print(f"average forward-rate forecast error {e.params.iloc[0]:.3f} pp (HAC se {e.bse.iloc[0]:.3f}) -> term premium ~ {-e.params.iloc[0]:.3f} pp")
r = smf.ols("dr ~ spread", data=s).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
print(f"Fama regression: change in 3m rate = {r.params['Intercept']:.3f} + {r.params['spread']:.3f} x (forward - spot)  (HAC se {r.bse['spread']:.3f})")
t = r.t_test("spread = 1")
print(f"test slope = 1: t = {float(np.squeeze(t.tvalue)):.2f}, p = {float(t.pvalue):.4f};  R2 = {r.rsquared:.3f};  n = {int(r.nobs)}")
```

Output:

```text
average forward-rate forecast error -0.569 pp (HAC se 0.128) -> term premium ~ 0.569 pp
Fama regression: change in 3m rate = 0.120 + -0.129 x (forward - spot)  (HAC se 0.216)
test slope = 1: t = -5.23, p = 0.0000;  R2 = 0.003;  n = 123
```

Forward rates systematically **overpredict** future short rates by about 0.6 percentage points: the gap is a term premium.
The Fama regression slope is about \(-0.1\), far below the value of 1 implied by the pure expectations hypothesis, and the
\(R^2\) is close to zero: forward–spot spreads contain almost no information about future rate changes at this horizon.
Arbitrage pins forward rates down by spot rates, but it does not make them unbiased forecasts — that requires an assumption
about risk premia.

</details>

## Further reading
Ross (1976); Cochrane, *Asset Pricing*.
