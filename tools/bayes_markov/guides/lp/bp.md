title: Structural Breaks at Unknown Dates (Bai–Perron)
short: Bai–Perron breaks
when: A mean, a trend or a regression coefficient may have shifted, and you want to find how many times and when
data: U.S. quarterly data, 1960–2019: GDP-deflator inflation and real GDP growth
tools: Python · OES engine · statsmodels
oes: bp
lead: Use this guide when a series or a relationship may have changed at dates you do not know: inflation after the 1970s, the coefficient of a policy rule after a regime change, a growth slowdown. The Bai–Perron (1998, 2003) procedure finds the best partition into regimes, tests how many breaks there are, gives confidence intervals for the dates and estimates each regime separately. The guide covers the setup, the tests, the choice of the number of breaks, the break-date intervals, robustness to serial correlation and the text for the paper.
---

## The question and the model

**How many times did the parameters shift, and when?** With \(m\) breaks at dates \(T_1 < \dots < T_m\) the model is

\[
y_t = x_t' \beta_j + u_t, \qquad t = T_{j-1}+1, \dots, T_j, \qquad j = 1, \dots, m+1,
\]

where \(x_t\) always contains a constant, so the simplest case is a model of shifting means. For each \(m\) the break dates are the least-squares minimisers over all partitions in which every regime has at least a fraction \(\varepsilon\) of the sample (the trimming). A dynamic-programming algorithm finds them exactly. The number of breaks is chosen by the \(\sup F(\ell)\) tests of no break against \(\ell\) breaks, the UDmax test against an unknown number, the sequential tests of \(\ell\) against \(\ell+1\) breaks, or an information criterion.

<div class="read" markdown="1">
**Three checks before you trust break dates**

1. **Are the errors close to uncorrelated?** The tests assume serially uncorrelated errors. For a persistent series, shifts in the mean can be mistaken for persistence, and the reverse. Re-estimate with a lagged dependent variable among the regressors.
2. **Does the number of breaks depend on the rule?** BIC, LWZ and the sequential test can disagree. Report all of them, and show that the dates you discuss are stable.
3. **Are the break dates precise?** A small shift gives a wide confidence interval for the date. Say so, and do not present a date as exact when its interval spans years.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We test for structural change in the mean of U.S. inflation with the procedure of Bai and Perron (1998, 2003). The model allows the mean to shift at up to five unknown dates, with each regime containing at least 15% of the 240 quarterly observations (36 quarters). Break dates are the global least-squares minimisers, and the number of breaks is chosen by the BIC and checked against the sequential supF(\(\ell+1\,|\,\ell\)) test at the 5% level. Critical values are asymptotic and were simulated for this engine. Confidence intervals for the dates follow Bai (1997).
</div>

## Load the data and look at the series

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf

data = open("yield.csv").read()
df = pd.read_csv("yield.csv")
print(df.date.iloc[0], "to", df.date.iloc[-1], "|", len(df), "quarters")
print(df[["inflation", "gdp_growth"]].describe().round(2).loc[["mean", "std", "min", "max"]])
for c in ["inflation", "gdp_growth"]:
    print(f"{c:11s} ADF p = {adfuller(df[c], autolag='AIC')[1]:.3f}   KPSS p = {kpss(df[c], nlags='auto')[1]:.3f}   first autocorrelation = {acf(df[c], nlags=1)[1]:.2f}")
```

<div class="read" markdown="1">
**How to read it**

The sample runs from 1960Q1 to 2019Q4 (240 quarters). Inflation averages 3.19% with a standard deviation of 2.31; GDP growth averages 3.01% with a standard deviation of 3.25. Inflation is highly persistent (first autocorrelation 0.90, ADF p = 0.19) and the KPSS test rejects stationarity (p = 0.01), the pattern a series with shifting means produces. GDP growth is stationary with a first autocorrelation of 0.30. Persistence matters because the break tests assume uncorrelated errors: for inflation, a shift in the mean and a high autoregressive coefficient are hard to tell apart, which the AR(1) check below makes visible.
</div>

## Test for breaks and choose their number

```python
spec = dict(y="inflation", time="date", trim="0.15", max_breaks=5, select="bic")
r = oes.run("bp", data, spec)
oes.summary(r)
oes.show(r, "Tests of no break")
oes.show(r, "Sequential tests")
oes.show(r, "Information criteria")
```

<div class="read" markdown="1">
**How to read it**

Every test of no break against \(\ell\) breaks rejects at 5%: the supF(1) statistic is 106.9 against a critical value of 8.6, and the UDmax statistic is 226.8 against 8.9. The sequential tests reject zero, one and two breaks against one more (statistics 106.9, 174.8 and 29.7 against critical values of 8.6, 10.3 and 11.3) and stop at the fourth: the statistic for three against four breaks is 6.7, below the 5% critical value of 11.9. BIC and LWZ also choose three breaks, and the sum of squared residuals falls from 1,274 with no break to 437 with two and 388 with three, then barely at all. All four rules therefore agree on three breaks.
</div>

## Break dates, regimes and the fitted path

```python
oes.show(r, "Estimated break dates")
oes.show(r, "Regimes of the model")
oes.plot(r, "inflation and the fitted", ncols=1, size=(6.5, 3))
```

<div class="read" markdown="1">
**How to read it**

The three breaks fall in 1973Q2, 1982Q4 and 1991Q4. The mean of inflation is 2.95% before 1973 (standard error 0.42), 7.44% from 1973Q2 to 1982Q3 (0.48), 3.21% from 1982Q4 to 1991Q3 (0.21) and 1.87% from 1991Q4 onwards (0.13). These are the great inflation, the Volcker disinflation and the lower-inflation period that followed. The 95% intervals for the first two dates are tight (1971Q4 to 1974Q4 and 1981Q4 to 1983Q4), because the shifts are large; the third date is much less precise (1988Q4 to 1994Q4), because the drop from 3.21% to 1.87% is small relative to the noise.
</div>

## Does the answer depend on the selection rule and on the trimming?

```python
out = {}
for name, v in {"BIC": dict(select="bic"), "LWZ": dict(select="lwz"), "sequential": dict(select="sequential"), "UDmax then sequential": dict(select="udmax"),
                "trim 10%": dict(trim="0.10", max_breaks=8), "trim 20%": dict(trim="0.20", max_breaks=3), "trim 25%": dict(trim="0.25", max_breaks=2)}.items():
    rr = oes.run("bp", data, dict(spec, **v))
    s = {x["label"]: x["value"] for x in rr["summary"]}
    dates = [x[1] for x in [t for t in rr["tables"] if t["title"] == "Estimated break dates"][0]["rows"]] if s["Breaks selected"] else []
    out[name] = (s["Breaks selected"], ", ".join(dates) or "none")
print(pd.DataFrame(out, index=["breaks", "first quarter of the new regime"]).T)
```

<div class="read" markdown="1">
**How to read it**

The three breaks (1973Q2, 1982Q4, 1991Q4) are the same under BIC, LWZ, the sequential test and UDmax followed by the sequential test. The trimming fraction matters more. With 10% trimming the procedure adds an earlier break (1967Q3) and finds four; with 20% trimming it finds three breaks at 1972Q1, 1984Q1 and 1996Q1, because a break must leave at least 48 quarters in each regime, which rules out the 1973 and 1982 dates; with 25% it finds two. A larger trimming fraction restricts where the breaks can fall, so it moves the dates, and 15% is the usual choice because it balances precision and flexibility.
</div>

## Serial correlation: add a lag of the dependent variable

```python
d2 = df.copy()
d2["inflation_l1"] = d2.inflation.shift(1)
d2 = d2.dropna(subset=["inflation_l1"]).to_csv(index=False)
ar = oes.run("bp", d2, dict(spec, x=["inflation_l1"]))
oes.show(ar, "Sequential tests")
oes.show(ar, "Estimated break dates")
oes.show(ar, "Regimes of the model")
```

<div class="read" markdown="1">
**How to read it**

Adding a lagged dependent variable changes the story: one break is found, in 2006Q4 (95% interval 2005Q2 to 2008Q2), and the sequential tests stop after the first. Before 2006Q4 inflation is very persistent (autoregressive coefficient 0.91, standard error 0.04); after 2006Q4 persistence drops to 0.27 and the intercept rises. The earlier breaks in the mean are absorbed because a high autoregressive coefficient already explains the long swings. The two models answer different questions: the mean model asks when the average level shifted, the AR(1) model asks when the dynamics changed. Report both, and be explicit about which one your argument needs.
</div>

## A second series: real GDP growth

```python
g = oes.run("bp", data, dict(spec, y="gdp_growth"))
oes.show(g, "Sequential tests")
oes.show(g, "Estimated break dates")
oes.show(g, "Regimes of the model")
```

<div class="read" markdown="1">
**How to read it**

For real GDP growth the tests find one break, in 1969Q2: the mean falls from 4.64% (standard error 0.62) to 2.72% (0.30). The sequential test does not reject one break against two (6.2 against a critical value of 10.3). The 95% interval for the date is very wide (1960Q1 to 1982Q4) because the shift is small relative to the volatility of growth, so the break is detected but not dated. Do not present 1969Q2 as an exact date.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

The mean of U.S. inflation shifted three times between 1960 and 2019. The supF, UDmax and sequential tests all reject the null of no break at 5% (supF(1) = 106.9, critical value 8.6), and the BIC, LWZ and sequential procedures all select three breaks, dated 1973Q2 (95% interval 1971Q4 to 1974Q4), 1982Q4 (1981Q4 to 1983Q4) and 1991Q4 (1988Q4 to 1994Q4). The mean inflation rate is 2.95% before 1973, 7.44% from 1973Q2 to 1982Q3, 3.21% from 1982Q4 to 1991Q3 and 1.87% from 1991Q4. The number of breaks is robust to the selection rule but the dates depend on the trimming fraction: with 20% trimming the breaks move to 1972Q1, 1984Q1 and 1996Q1. When a lagged dependent variable is added the tests find a single break in 2006Q4 in which persistence falls from 0.91 to 0.27, which shows that the mean shifts and the persistence of inflation cannot be separated by these tests alone. The critical values are asymptotic and simulated for the pure structural change model.
</div>

| Report | What | Where |
|---|---|---|
| Model | Regressors, trimming, maximum breaks, selection rule | Step 1 |
| Data | Persistence and stationarity | Step 2 |
| Tests | supF, UDmax, sequential tests, information criteria | Step 3 |
| Dates | Break dates with confidence intervals, regime means | Step 4 |
| Robustness | Selection rule and trimming | Step 5 |
| Serial correlation | Re-estimate with a lagged dependent variable | Step 6 |
| Second series | The same procedure on GDP growth | Step 7 |

**References.** Bai, J. and Perron, P. (1998), "Estimating and Testing Linear Models with Multiple Structural Changes," *Econometrica* 66. Bai, J. and Perron, P. (2003), "Computation and Analysis of Multiple Structural Change Models," *Journal of Applied Econometrics* 18. Bai, J. (1997), "Estimating Multiple Breaks One at a Time," *Econometric Theory* 13. Andrews, D. (1993), "Tests for Parameter Instability and Structural Change with Unknown Change Point," *Econometrica* 61. Liu, J., Wu, S. and Zidek, J. (1997), "On Segmented Multivariate Regression," *Statistica Sinica* 7. Stock, J. and Watson, M. (2003), "Has the Business Cycle Changed and Why?" *NBER Macroeconomics Annual* 17.
