# Lecture 11 — Market Efficiency, Information, Governance, and Event Studies

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Distinguish forms of market efficiency; design event studies; understand information asymmetry, governance, and limits to arbitrage.

## 1. Efficient-market hypotheses
Weak-form efficiency concerns information in past prices; semi-strong form concerns public information; strong form includes private information. Efficiency does not mean prices are always correct or that returns are perfectly forecastable; it concerns abnormal profit conditional on an information set and model.

## 2. Event-study design
Define an event date, estimation window, event window, and expected-return model. Abnormal return is
\[
AR_{it}=R_{it}-\widehat E(R_{it}\mid\mathcal I).
\]
Cumulative abnormal return sums abnormal returns across the event window. Inference must address cross-sectional dependence, event clustering, and confounding news.

## 3. Information asymmetry
Adverse selection occurs before contracting when one party has private information; moral hazard occurs after contracting when actions are hidden or imperfectly monitored. These frictions affect financing, liquidity, and corporate governance.

## 4. Governance and agency
Ownership concentration, board monitoring, executive incentives, creditor rights, and disclosure rules shape agency costs. Empirical designs must separate governance selection from causal effects.

## 5. Limits to arbitrage
Short-sale constraints, funding risk, transaction costs, noise-trader risk, and model uncertainty can prevent correction of mispricing.

## 6. Computational lab
Conduct an event study using a market model. Report estimation and event windows, abnormal returns, CAR, standard errors, robustness to alternative benchmarks, and exclusion rules.

## Problem set
1. Define weak, semi-strong, and strong efficiency.
2. Construct an event-study timeline.
3. Distinguish adverse selection and moral hazard.
4. Explain limits to arbitrage.

## Coding exercises

These exercises use **real market data** in Python: monthly Fama–French factor returns (1926–2018), daily S&P 500 and
NASDAQ prices (1999–2018), the VIX, Moody's AAA and BAA corporate bond yields (1919–2018), all bundled with the `arch`
package, plus U.S. Treasury yields and household data from the `wooldridge` package. Install once with
`pip install arch wooldridge statsmodels scipy numpy pandas`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 11.1 — Weak-form efficiency: autocorrelations and variance ratios

*Data:* Weekly NYSE index returns (`nyse`, Wooldridge), and daily S&P 500 log returns 1999–2018.

1. Estimate the AR(1) of weekly NYSE returns (Wooldridge, 2020, Example 11.4). Is last week's return a useful predictor?
2. Compute Lo–MacKinlay variance ratios \(VR(q)=\operatorname{Var}(r_t+\dots+r_{t-q+1})/(q\operatorname{Var}(r_t))\) for the daily S&P 500 at \(q=2,5,10,20\), with heteroskedasticity-robust \(z\)-statistics.
3. Interpret. Do the results contradict market efficiency? What would trading costs do to any apparent predictability?

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
from arch.data import sp500
n = wd.data("nyse")
r = smf.ols("Q('return') ~ return_1", data=n).fit()
# TODO: variance ratios from overlapping q-day sums; robust z (Lo-MacKinlay 1988)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
from arch.data import sp500
n = wd.data("nyse")
r = smf.ols("Q('return') ~ return_1", data=n).fit(cov_type="HC1")
print(f"weekly NYSE: return_t = {r.params.iloc[0]:.3f} + {r.params['return_1']:.4f} return_(t-1)  (robust t = {r.tvalues['return_1']:.2f}, R2 = {r.rsquared:.4f})")
x = np.log(sp500.load()["Adj Close"]).diff().dropna().to_numpy(); T = len(x); mu = x.mean(); s2 = np.mean((x - mu)**2)
for q in (2, 5, 10, 20):
    sq = np.convolve(x, np.ones(q), "valid")
    vr = np.mean((sq - q * mu)**2) / (q * s2)
    e2 = (x - mu)**2
    theta = sum((2 * (q - j) / q)**2 * np.sum(e2[j:] * e2[:-j]) / np.sum(e2)**2 * T for j in range(1, q))
    print(f"q = {q:2d}: VR = {vr:.3f}, robust z = {(vr - 1) / np.sqrt(theta / T):.2f}")
```

Output:

```text
weekly NYSE: return_t = 0.180 + 0.0589 return_(t-1)  (robust t = 0.85, R2 = 0.0035)
q =  2: VR = 0.930, robust z = -2.81
q =  5: VR = 0.836, robust z = -2.78
q = 10: VR = 0.747, robust z = -2.73
q = 20: VR = 0.715, robust z = -2.08
```

Weekly NYSE returns show a small, insignificant positive autocorrelation (\(R^2\) below 0.4%), as in Wooldridge's example.
For the daily S&P 500 in 1999–2018, variance ratios are significantly **below one** at all horizons: daily returns show mild
negative serial correlation (short-term reversals), concentrated in volatile periods. The predictability is small, and it
would be very hard to exploit after trading costs and the bid–ask bounce. Statistical rejection of a random walk is not the
same as a profitable trading rule, which is what market efficiency rules out.

</details>

### Exercise 11.2 — Event studies of market-wide shocks

*Data:* Daily S&P 500 log returns, 1999–2018.

Measure the market's reaction to three events, using a constant-mean model estimated over the 250 trading days ending 30 days before each event:
the reopening of markets after 9/11 (17 September 2001), Lehman Brothers' bankruptcy filing (15 September 2008) and the Brexit referendum result (24 June 2016).

1. Compute abnormal returns on the event day and cumulative abnormal returns (CAR) over days 0 to +4.
2. Compute \(t\)-statistics using the estimation-window standard deviation.
3. Did the market fully react on day 0, or did prices drift? What does a drift imply for semi-strong efficiency, and why are market-wide events harder to study than firm-specific ones?

```python
import numpy as np, pandas as pd
from arch.data import sp500
r = 100 * np.log(sp500.load()["Adj Close"]).diff().dropna()
events = {"9/11 reopening": "2001-09-17", "Lehman": "2008-09-15", "Brexit": "2016-06-24"}
# TODO: estimation window [-280, -31]; AR = r - mean; CAR(0, 4); t = CAR / (sd * sqrt(5))
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd
from arch.data import sp500
r = 100 * np.log(sp500.load()["Adj Close"]).diff().dropna()
events = {"9/11 reopening": "2001-09-17", "Lehman": "2008-09-15", "Brexit": "2016-06-24"}
for name, day in events.items():
    i = r.index.get_loc(pd.Timestamp(day))
    est = r.iloc[i - 280:i - 30]; mu, sd = est.mean(), est.std()
    ar = r.iloc[i:i + 5] - mu
    car = ar.cumsum()
    print(f"{name:15s} ({day}): AR day 0 = {ar.iloc[0]:6.2f}% (t = {ar.iloc[0]/sd:6.2f});  CAR(0,1) = {car.iloc[1]:6.2f}%;  CAR(0,4) = {car.iloc[4]:6.2f}% (t = {car.iloc[4]/(sd*np.sqrt(5)):6.2f});  estimation sd {sd:.2f}%")
```

Output:

```text
9/11 reopening  (2001-09-17): AR day 0 =  -4.98% (t =  -3.68);  CAR(0,1) =  -5.49%;  CAR(0,4) = -11.99% (t =  -3.97);  estimation sd 1.35%
Lehman          (2008-09-15): AR day 0 =  -4.78% (t =  -3.65);  CAR(0,1) =  -2.99%;  CAR(0,4) =   0.52% (t =   0.18);  estimation sd 1.31%
Brexit          (2016-06-24): AR day 0 =  -3.65% (t =  -3.48);  CAR(0,1) =  -5.46%;  CAR(0,4) =  -0.63% (t =  -0.27);  estimation sd 1.05%
```

All three events produced day-0 abnormal returns of about −3.7% to −5% (\(t\) around −3.5). After the 9/11 reopening, prices
kept falling all week (5-day CAR about −12%). After Lehman and Brexit, the 5-day CARs were close to zero: the market partly
reversed the initial drop. Persistent drift would suggest under-reaction, and reversal over-reaction, but with market-wide
events there is no control group: the "abnormal" return is just the market return minus its usual mean, and other news
arrives on the same days. Firm-level event studies, with the market as a benchmark, are much more powerful.

</details>

## Further reading
Fama (1970); MacKinlay (1997); Shleifer and Vishny (1997).
