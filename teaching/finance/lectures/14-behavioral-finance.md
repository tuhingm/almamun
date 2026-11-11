# Lecture 14 — Behavioral Finance: Beliefs, Preferences, and Market Outcomes

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Understand behavioral departures from benchmark rational choice; connect psychological mechanisms to asset prices and trading behavior.

## 1. Prospect theory
Prospect theory evaluates outcomes relative to a reference point. The value function is typically concave for gains, convex for losses, and steeper for losses, capturing loss aversion. Probability weighting can overweight small probabilities and underweight moderate or high probabilities.

## 2. Behavioral mechanisms
Overconfidence can increase trading volume; limited attention can delay price responses; representativeness can generate extrapolative beliefs; disposition effects can lead investors to realize gains too early and hold losses too long.

## 3. Market implications
Sentiment and biased beliefs can affect demand, volatility, and cross-sectional returns. Limits to arbitrage determine whether mispricing persists. A behavioral explanation should specify a mechanism and a testable prediction.

## 4. Empirical identification
Use account-level trading data, experiments, surveys, or quasi-experimental variation where appropriate. Address selection, measurement error, multiple testing, and alternative rational explanations.

## 5. Worked example
A decision maker chooses between a sure gain of 50 and a 50% chance of 110. Compare expected value and a prospect-theory interpretation, emphasizing reference dependence rather than assuming a universal choice.

## 6. Computational lab
Analyze investor-level turnover and disposition patterns or simulate heterogeneous belief agents. Define the reference portfolio, holding period, and transaction-cost treatment.

## Problem set
1. Explain reference dependence and loss aversion.
2. Define probability weighting.
3. Link overconfidence to trading volume.
4. Propose a testable disposition-effect design.

## Coding exercises

These exercises use **real market data** in Python: monthly Fama–French factor returns (1926–2018), daily S&P 500 and
NASDAQ prices (1999–2018), the VIX, Moody's AAA and BAA corporate bond yields (1919–2018), all bundled with the `arch`
package, plus U.S. Treasury yields and household data from the `wooldridge` package. Install once with
`pip install arch wooldridge statsmodels scipy numpy pandas`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 14.1 — How do people invest their retirement money?

*Data:* `pension` — 194 U.S. workers with self-directed pension accounts (1989): share invested in stocks `pctstck` (0, 50 or 100), whether they chose their allocation `choice`, age, education, gender, family income and wealth.

1. Tabulate the distribution of `pctstck`. What does the clustering at 0, 50 and 100 suggest about how people make allocation decisions?
2. Regress `pctstck` on `choice`, `age`, `educ`, `female`, `black`, `married`, income dummies and `prftshr`. Interpret the coefficients on age and gender.
3. Relate the results to the "1/n" heuristic (Benartzi and Thaler, 2001) and to life-cycle portfolio advice.

```python
import wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("pension")
# TODO: value_counts; OLS with robust se
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("pension")
print("distribution of % in stocks:", d["pctstck"].value_counts().sort_index().to_dict())
r = smf.ols("pctstck ~ choice + age + educ + female + black + married + finc25 + finc35 + finc50 + finc75 + finc100 + finc101 + prftshr",
            data=d).fit(cov_type="HC1")
for v in ("choice", "age", "educ", "female", "prftshr"):
    print(f"  {v:8s}: {r.params[v]:7.2f} (robust se {r.bse[v]:.2f})")
print(f"R2 = {r.rsquared:.3f}, n = {int(r.nobs)}")
print("mean % in stocks by choice:", d.groupby("choice")["pctstck"].mean().round(1).to_dict())
```

Output:

```text
distribution of % in stocks: {0: 64, 50: 72, 100: 58}
  choice  :   12.10 (robust se 5.99)
  age     :   -1.66 (robust se 0.79)
  educ    :    0.74 (robust se 1.17)
  female  :    1.11 (robust se 7.00)
  prftshr :   15.87 (robust se 8.07)
R2 = 0.100, n = 194
mean % in stocks by choice: {0: 42.7, 1: 52.1}
```

Allocations cluster at 0, 50 and 100% in stocks, and 50% is the most common choice: many people seem to split evenly rather
than optimize — consistent with the "1/n" heuristic of Benartzi and Thaler (2001). Workers who chose their own allocation
hold about 12 points more in stocks. Each year of age lowers the stock share by about 1.7 points, in line with life-cycle
advice. The gender coefficient is small and insignificant here. With 194 observations and \(R^2=0.10\), most of the
variation in allocations is unexplained by these characteristics — another sign of heuristics and inertia.

</details>

### Exercise 14.2 — Was the NASDAQ a bubble? A test for explosive prices

*Data:* Weekly NASDAQ composite closing prices, 1999–2002 (`arch.data.nasdaq`).

Phillips, Wu and Yu (2011) detect bubbles with right-tailed unit-root tests: an explosive root (\(\rho>1\)) in log prices indicates prices rising faster than any random walk.

1. Compute the forward-recursive sup-ADF statistic: ADF \(t\)-statistics on samples starting at the first observation and ending at each later week (minimum window 20% of the sample), and take the maximum.
2. Simulate the 95% critical value of the sup-ADF statistic under a driftless random walk (500 replications).
3. Date the episode: simulate the 95% critical value of the recursive ADF statistic **for each end point**, and find the weeks in which the statistic exceeds it. Compare with the March 2000 peak.

```python
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller
from arch.data import nasdaq
p = np.log(nasdaq.load()["Adj Close"].loc["1999":"2002"].resample("W").last())
# TODO: recursive adfuller(p[:k], maxlag=1, regression="c", autolag=None)[0] for k >= 0.2 n
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller
from arch.data import nasdaq
p = np.log(nasdaq.load()["Adj Close"].loc["1999":"2002"].resample("W").last()).dropna()
n = len(p); r0 = int(0.2 * n)
def recursive(x):
    return np.array([adfuller(x[:k], maxlag=1, regression="c", autolag=None)[0] for k in range(r0, len(x) + 1)])
stat = recursive(p.to_numpy()); sadf = stat.max()
rng = np.random.default_rng(1)
sims = np.array([recursive(np.cumsum(rng.standard_normal(n))) for _ in range(500)])
cv = np.quantile(sims.max(1), 0.95); cv_t = np.quantile(sims, 0.95, axis=0)        # sup-ADF and pointwise critical values
print(f"{n} weeks; sup-ADF = {sadf:.2f}, simulated 95% critical value {cv:.2f}  -> explosive episode detected: {sadf > cv}")
ends = p.index[r0 - 1:]
flag = pd.Series(stat > cv_t, index=ends)
runs = (flag != flag.shift()).cumsum()[flag]
for _, w in flag[flag].groupby(runs):
    print(f"episode: {w.index[0].date()} to {w.index[-1].date()} ({len(w)} week{'s' if len(w) > 1 else ''} above the pointwise 95% critical value)")
print(f"NASDAQ peak week: {p.idxmax().date()} (level {np.exp(p.max()):,.0f}); end-2002 level {np.exp(p.iloc[-1]):,.0f} ({np.exp(p.iloc[-1] - p.max()) - 1:.0%})")
```

Output:

```text
209 weeks; sup-ADF = 1.78, simulated 95% critical value 1.30  -> explosive episode detected: True
episode: 1999-11-21 to 2000-04-02 (20 weeks above the pointwise 95% critical value)
episode: 2001-09-23 to 2001-09-23 (1 week above the pointwise 95% critical value)
episode: 2002-07-28 to 2002-08-11 (3 weeks above the pointwise 95% critical value)
episode: 2002-09-22 to 2002-10-13 (4 weeks above the pointwise 95% critical value)
NASDAQ peak week: 2000-03-12 (level 5,049); end-2002 level 1,336 (-74%)
```

The sup-ADF statistic exceeds its simulated 95% critical value, so the test detects explosive behavior. Date-stamping finds
the main episode from November 1999 to April 2000, bracketing the March 2000 peak, after which the index fell 74%. A few
short episodes during the collapse (a single week after 9/11 and several weeks in mid-2002) reflect the persistent slide,
which a right-tailed test with a fixed starting point can also flag; isolated one-week crossings are best treated as noise. The test detects prices rising faster than a random walk; calling that a "bubble" requires the
additional argument that fundamentals did not grow explosively. The full Phillips–Shi–Yu (2015) GSADF procedure also varies
the starting point.

</details>

## Further reading
Kahneman and Tversky (1979); Barberis and Thaler (2003); Shleifer, *Inefficient Markets*.
