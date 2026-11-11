# Lecture 01 — Returns, Risk, and the Foundations of Asset Pricing

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Measure simple and log returns; distinguish arithmetic and geometric performance; quantify risk and explain the role of stochastic discounting.

## 1. Return measurement
For price \(P_t\) and cash distribution \(D_t\), the one-period simple total return is
\[
R_{t+1}=\frac{P_{t+1}+D_{t+1}-P_t}{P_t}.
\]
The gross return is \(1+R_{t+1}\), while the log return is \(r_{t+1}=\log(1+R_{t+1})\). Simple returns compound multiplicatively; log returns add over time. The approximation \(r\approx R\) is accurate only for small returns.

## 2. Arithmetic and geometric averages
The arithmetic mean estimates average one-period return. The geometric mean
\[
R_G=\left[\prod_{t=1}^T(1+R_t)\right]^{1/T}-1
\]
is the constant per-period return producing the same cumulative wealth. Volatility generally makes geometric growth lower than arithmetic average return.

## 3. Risk and covariance
For return vector \(R\), expected return is \(\mu=E(R)\), covariance matrix is \(\Sigma=E[(R-\mu)(R-\mu)']\), and portfolio variance is \(w'\Sigma w\). Correlation is scale-free dependence; covariance retains units.

## 4. Stochastic discount factor
A fundamental pricing relation is
\[
P_t=E_t[M_{t+1}X_{t+1}],
\]
where \(M_{t+1}\) is the stochastic discount factor (SDF) and \(X_{t+1}\) the payoff. For a gross return, \(1=E_t[M_{t+1}R_{t+1}]\). Risk premia depend on covariance with the SDF, not simply on standalone volatility.

## 5. Worked example
An asset rises from 100 to 108 and pays a distribution of 2. Its total simple return is 10%. Calculate its log return and compare the two measures.

## 6. Computational lab
Load adjusted price data, compute simple and log returns, annualize means and volatility under an explicitly stated frequency assumption, and compare arithmetic and geometric annualized performance. Check missing dates and corporate-action treatment.

## Problem set
1. Derive the total return formula.
2. Show that multi-period log returns add.
3. Explain why geometric mean is below arithmetic mean in volatile samples.
4. Interpret the SDF pricing equation.

## Coding exercises

These exercises use **real market data** in Python: monthly Fama–French factor returns (1926–2018), daily S&P 500 and
NASDAQ prices (1999–2018), the VIX, Moody's AAA and BAA corporate bond yields (1919–2018), all bundled with the `arch`
package, plus U.S. Treasury yields and household data from the `wooldridge` package. Install once with
`pip install arch wooldridge statsmodels scipy numpy pandas`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 1.1 — Ninety years of U.S. stock returns: arithmetic, geometric and compounded

*Data:* Fama–French monthly market excess return `Mkt-RF` and T-bill rate `RF`, July 1926 – November 2018.

1. Build monthly total market returns \(R_m=\)`Mkt-RF`+`RF`. Report annualized arithmetic and geometric mean returns for stocks and bills, and the annualized volatility of stocks.
2. Check the approximation \(\text{geometric}\approx\text{arithmetic}-\sigma^2/2\).
3. Compute the value in 2018 of \$1 invested in July 1926 in stocks and in bills.
4. Report the annualized Sharpe ratio with its standard error \(\sqrt{(1+SR^2/2)/T}\) (Lo, 2002, i.i.d. case).

```python
import numpy as np, pandas as pd
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

rm, rf = ff["Mkt-RF"] + ff["RF"], ff["RF"]
# TODO: 12 * mean, (prod(1+r))^(12/T) - 1, sqrt(12) * std, cumulative wealth
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

rm, rf, ex = ff["Mkt-RF"] + ff["RF"], ff["RF"], ff["Mkt-RF"]; T = len(rm)
for lab, r in (("stocks", rm), ("T-bills", rf)):
    arith, geo, vol = 12 * r.mean(), np.prod(1 + r)**(12 / T) - 1, np.sqrt(12) * r.std()
    print(f"{lab:7s}: arithmetic {100*arith:5.2f}%, geometric {100*geo:5.2f}%, volatility {100*vol:5.2f}%, arith - vol^2/2 = {100*(arith - vol**2/2):5.2f}%, $1 -> ${np.prod(1 + r):,.0f}")
sr_m = ex.mean() / ex.std(); sr = np.sqrt(12) * sr_m; se = np.sqrt(12) * np.sqrt((1 + sr_m**2 / 2) / T)
print(f"annualized equity premium {100*12*ex.mean():.2f}%, Sharpe ratio {sr:.3f} (se {se:.3f})")
print(f"sample: {ff.index[0]} to {ff.index[-1]}, {T} months")
```

Output:

```text
stocks : arithmetic 11.21%, geometric  9.94%, volatility 18.42%, arith - vol^2/2 =  9.51%, $1 -> $6,381
T-bills: arithmetic  3.29%, geometric  3.34%, volatility  0.88%, arith - vol^2/2 =  3.29%, $1 -> $21
annualized equity premium 7.92%, Sharpe ratio 0.429 (se 0.104)
sample: 1926-07 to 2018-11, 1109 months
```

Over 92 years, U.S. stocks returned about 11.2% a year on average (arithmetic) and 9.9% compounded, against 3.3% for
T-bills. The geometric mean is close to arithmetic minus \(\sigma^2/2\): volatility drag. Compounding turns the gap into
an enormous difference in wealth — \$1 grew to about \$6,400 in stocks versus \$21 in bills. The Sharpe ratio of about 0.43
has a standard error of 0.10, even with 1,109 months: expected returns are hard to estimate precisely, which is a recurring
theme of this course. (Bills look almost riskless in nominal terms here; in real terms they are not.)

</details>

### Exercise 1.2 — Daily S&P 500 returns: fat tails and volatility clustering

*Data:* Daily S&P 500 adjusted close, 1999–2018 (`arch.data.sp500`).

1. Compute daily log returns. Report mean, standard deviation, skewness and excess kurtosis, and run a Jarque–Bera test.
2. Count the days with moves larger than 4 standard deviations, and compare with the number a normal distribution predicts.
3. Compare the autocorrelation of returns with the autocorrelation of **squared** returns (lags 1–5). What does the contrast say about predictability of returns versus predictability of risk?

```python
import numpy as np, pandas as pd
from arch.data import sp500
from scipy import stats
p = sp500.load()["Adj Close"]; r = 100 * np.log(p).diff().dropna()
# TODO: stats.skew, stats.kurtosis, stats.jarque_bera; |r - mean| > 4 sd; acf of r and r^2
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd
from arch.data import sp500
from scipy import stats
from statsmodels.tsa.stattools import acf
p = sp500.load()["Adj Close"]; r = 100 * np.log(p).diff().dropna()
jb = stats.jarque_bera(r)
print(f"daily log returns (%): mean {r.mean():.4f}, sd {r.std():.3f}, skewness {stats.skew(r):.3f}, excess kurtosis {stats.kurtosis(r):.2f}")
print(f"Jarque-Bera = {jb.statistic:,.0f}, p = {jb.pvalue:.1e}")
big = (np.abs(r - r.mean()) > 4 * r.std()).sum()
print(f"days beyond 4 sd: {big} observed vs {len(r) * 2 * stats.norm.sf(4):.2f} expected under normality ({len(r)} days)")
print("autocorrelation of returns,         lags 1-5:", acf(r, nlags=5)[1:].round(3))
print("autocorrelation of squared returns, lags 1-5:", acf(r**2, nlags=5)[1:].round(3))
print("largest daily losses:", r.nsmallest(3).round(2).to_dict())
```

Output:

```text
daily log returns (%): mean 0.0142, sd 1.204, skewness -0.205, excess kurtosis 8.17
Jarque-Bera = 14,022, p = 0.0e+00
days beyond 4 sd: 32 observed vs 0.32 expected under normality (5030 days)
autocorrelation of returns,         lags 1-5: [-0.07  -0.047  0.014 -0.013 -0.046]
autocorrelation of squared returns, lags 1-5: [0.208 0.379 0.201 0.297 0.322]
largest daily losses: {Timestamp('2008-10-15 00:00:00'): -9.47, Timestamp('2008-12-01 00:00:00'): -9.35, Timestamp('2008-09-29 00:00:00'): -9.22}
```

Daily returns are far from normal: excess kurtosis is about 8, and there were 32 days with moves beyond four standard
deviations, where a normal distribution predicts 0.3. The three worst days all fell in autumn 2008. Returns themselves have
small (slightly negative) autocorrelations, but **squared** returns are strongly autocorrelated: large moves cluster in time.
Returns are close to unpredictable; risk is highly predictable. This is the motivation for GARCH models (Lectures 13 and 15)
and for fat-tailed risk measures.

</details>

## Further reading
Cochrane, *Asset Pricing*; Bodie, Kane and Marcus, *Investments*.
