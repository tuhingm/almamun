title: Threshold Regression and SETAR: Regimes Defined by a Variable
short: Threshold (TAR / SETAR)
when: The relationship or the dynamics change abruptly when a variable (often the series' own past) crosses a level, and you want to find the level
data: U.S. quarterly data, 1960–2019: GDP-deflator inflation
tools: Python · OES engine · statsmodels
oes: tar
lead: Use this guide when behaviour differs above and below a level that you do not know: inflation that is persistent only when it is high, a policy rule that reacts only past a limit, growth that slows after a downturn. A threshold regression (Tong's TAR, and SETAR when the threshold variable is the series' own lag) estimates the level by least squares, tests the null of no threshold with a bootstrap, gives a confidence interval for the level, and estimates each regime separately. The guide covers the model, the test, the regimes, robustness to lags, delay and trimming, and the text for the paper.
---

## The question and the model

**Does the process switch between regimes when a variable crosses a level, and where is that level?** In a threshold regression the coefficients depend on whether a threshold variable \(s_t\) lies below or above a threshold \(\gamma\):

\[
y_t = x_t'\beta_1 \,\mathbf 1(s_t \le \gamma) + x_t'\beta_2 \,\mathbf 1(s_t > \gamma) + e_t .
\]

When \(s_t = y_{t-d}\) and \(x_t\) holds lags of \(y_t\), the model is the self-exciting threshold autoregression (SETAR); \(d\) is the delay. The regime is determined by the data, not by the calendar, which is the difference from a structural break (see the [Bai–Perron guide](bp.html)): the same regime can occur in 1965 and in 1985. The switch is abrupt. If you expect a gradual move between regimes, use the [smooth-transition guide](str.html).

The threshold \(\gamma\) is estimated as the value that minimises the sum of squared residuals, searched over observed values of \(s_t\) that leave at least a fraction \(\varepsilon\) of the sample in each regime. Because \(\gamma\) does not exist under the null of linearity, the usual F test is not valid. Hansen (1996) shows how to bootstrap its distribution, and the confidence interval for \(\gamma\) is the set of values whose likelihood-ratio statistic stays below 7.35 (Hansen 2000).

<div class="read" markdown="1">
**Four checks before you trust a threshold**

1. **Does the bootstrap reject linearity?** Without rejection the threshold estimate is not reliable, whatever it looks like. Report the p-value.
2. **Is the threshold stable?** Re-estimate with other lags, delays and trimming. A threshold that moves everywhere is not identified.
3. **Do the regimes contain enough observations?** A regime with a handful of observations gives imprecise coefficients and can be driven by a few episodes.
4. **Is the switch really abrupt?** If the residuals look better under a gradual change, use the smooth-transition model.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We test for threshold nonlinearity in U.S. inflation with a self-exciting threshold autoregression (Tong 1978; Hansen 1996, 1997). Quarterly inflation, the GDP-deflator rate, follows an AR(2) whose intercept and coefficients may differ between two regimes defined by whether inflation \(d\) quarters earlier was above a threshold \(\gamma\). The delay \(d\) is chosen from one to four quarters and the threshold is searched over values that leave at least 15% of the 236 observations in each regime. We test linearity with the sup-F statistic and a fixed-regressor bootstrap with 999 draws, which accounts for the search over \(\gamma\) and over \(d\), and we report a confidence interval for \(\gamma\) from the likelihood-ratio statistic.
</div>

## Load the data and look at the series

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine
from statsmodels.tsa.stattools import adfuller, acf

data = open("yield.csv").read()
df = pd.read_csv("yield.csv")
print(df.date.iloc[0], "to", df.date.iloc[-1], "|", len(df), "quarters")
print(df.inflation.describe().round(2).loc[["mean", "std", "min", "25%", "50%", "75%", "max"]])
print(f"ADF p = {adfuller(df.inflation, autolag='AIC')[1]:.3f}   first autocorrelation = {acf(df.inflation, nlags=1)[1]:.2f}")
```

<div class="read" markdown="1">
**How to read it**

The sample runs from 1960Q1 to 2019Q4 (240 quarters; 236 remain after allowing for two lags and a delay of up to four quarters). Inflation averages 3.19% with a standard deviation of 2.31, ranges from −0.53% to 12.14% and is skewed: the median is 2.42%, well below the mean, because of the 1970s. The series is highly persistent (first autocorrelation 0.90, ADF p = 0.19). That matters for a threshold model: a persistent series that spends a long spell at a high level can look like a threshold process, so the bootstrap test and the robustness checks below carry the weight.
</div>

## Test for a threshold

```python
spec = dict(y="inflation", time="date", ar_lags=2, search_delay="yes", max_delay=4,
            trim="0.15", max_thresholds="2", boot=999, seed=12345)
r = oes.run("tar", data, spec)
oes.summary(r)
oes.show(r, "Tests for a threshold")
```

<div class="read" markdown="1">
**How to read it**

The sup-F statistic is 25.9. The bootstrap p-value is 0.081 with 999 draws, with the delay searched from one to four quarters in the data and in every draw, and the 5% bootstrap critical value is 28.6. Linearity is therefore rejected at 10% but not at 5%: the evidence for a threshold is suggestive, not conclusive. The search chooses a delay of three quarters. The test of one threshold against two does not reject (statistic 10.0, p = 0.13), so the model has one threshold and two regimes. Note that the bootstrap already accounts for the search over \(\gamma\) and \(d\); testing at a delay picked after looking at the data would give a p-value that is too small.
</div>

## The threshold, its interval and the regimes

```python
oes.show(r, "Estimated thresholds")
oes.show(r, "Coefficients by regime")
oes.show(r, "The regimes")
oes.show(r, "Fit against the linear model")
```

<div class="read" markdown="1">
**How to read it**

The threshold is 4.23% inflation, three quarters earlier, with a 95% interval from 4.01% to 5.51%. The lower regime holds 180 quarters and the upper regime 56 (24%). Below the threshold the AR coefficients are 0.51 (standard error 0.076) and 0.30 (0.062), they sum to 0.81, and the implied mean is 2.21%. Above it they are 0.88 (0.12) and −0.12 (0.14), sum 0.76, intercept 1.55 (0.46), and the implied mean is 6.56%. Persistence is similar in the two regimes; what differs is the level the series gravitates to, which is the intercept. The threshold model lowers the residual standard deviation from 0.990 to 0.945 and the sum of squared residuals from 228.3 to 205.2. AIC falls from −1.9 to −19.0 even after counting the extra coefficients and the threshold as parameters.
</div>

## Look at the likelihood and at the regimes over time

```python
oes.plot(r, "Likelihood-ratio statistic for the threshold", ncols=1, size=(6.5, 3))
oes.plot(r, "inflation(-3) and the estimated", ncols=1, size=(6.5, 3))

s = {x["label"]: x["value"] for x in r["summary"]}
high = df.inflation.shift(s["Delay"]) > s["Threshold 1"]
decade = (df.date.str[:4].astype(int) // 10) * 10
print("Quarters in the high-inflation regime, by decade")
print(high.groupby(decade).sum().astype(int).to_string())
```

<div class="read" markdown="1">
**How to read it**

The likelihood-ratio statistic is far above the 5% critical value of 7.35 (between about 12 and 21) for almost every candidate threshold, and it falls below it only for values between 4.0% and 5.5%. That is the confidence interval: one well-defined region, with a sharp edge, as an abrupt switch implies. The second chart shows where the regimes sit. The upper regime contains 4 quarters in the 1960s, 34 in the 1970s, 16 in the 1980s, 3 in the 1990s and none since 2000. Fifty of the 56 high-regime quarters therefore belong to the Great Inflation. This is the main caution: the threshold may be picking up that episode, which the [Bai–Perron guide](bp.html) dates with mean shifts in 1973Q2 and 1982Q4 and a high-inflation mean of 7.4% between them. The two readings agree on what happened. They differ on why: a threshold model says high inflation sustains itself, while a break model says the regime was a period.
</div>

## Does the answer depend on the lags, the delay and the trimming?

```python
out = {}
variants = {"baseline (AR(2), delay 1 to 4)": {}, "AR(1)": dict(ar_lags=1), "AR(3)": dict(ar_lags=3), "AR(4)": dict(ar_lags=4),
            "delay fixed at 1": dict(search_delay="no", delay=1), "delay fixed at 3": dict(search_delay="no", delay=3),
            "trim 10%": dict(trim="0.10"), "trim 20%": dict(trim="0.20"), "trim 25%": dict(trim="0.25")}
for name, v in variants.items():
    rr = oes.run("tar", data, dict(spec, max_thresholds="1", boot=499, **v))
    s = {x["label"]: x["value"] for x in rr["summary"]}
    out[name] = (s["Delay"], s["Threshold 1"], s["Sup-F against linearity"], s["Bootstrap p (linearity)"])
print(pd.DataFrame(out, index=["delay", "threshold", "sup-F", "bootstrap p"]).T.round(3))
```

<div class="read" markdown="1">
**How to read it**

Six of the nine specifications put the threshold between 3.98% and 4.45%: the baseline (4.23%), AR(1) (3.98%), AR(4) (4.45%), a delay fixed at three (4.23%) and 20% and 25% trimming (4.23% and 4.01%). The evidence for it, however, depends on the lag order. The bootstrap p-value is 0.03 with one lag, 0.08 with two, 0.16 with three and 0.19 with four, because more lags absorb part of the nonlinearity. A delay fixed at one gives no threshold (p = 0.32, threshold 1.40%), so the delay matters. The p-values for a fixed delay (0.020 for a delay of three) are smaller than the baseline because they ignore the search over delays, so report the baseline. With 10% trimming the search picks a delay of four and a threshold of 5.75% (p = 0.076). Taken together: a threshold near 4% is the most stable finding, and it is moderately supported.
</div>

## Is a second threshold needed, and what if the threshold variable is another series?

```python
two = oes.run("tar", data, dict(spec, trim="0.10"))
oes.show(two, "Tests for a threshold")
ext = oes.run("tar", data, dict(y="inflation", x=["unemployment"], threshold="unemployment", time="date",
                                                  ar_lags=1, delay=1, boot=499, max_thresholds="1"))
oes.show(ext, "Tests for a threshold")
oes.show(ext, "Estimated thresholds")
oes.show(ext, "Coefficients by regime")
```

<div class="read" markdown="1">
**How to read it**

A second threshold is not supported. With 10% trimming, so that smaller regimes are allowed, the sup-F test of one threshold against two has p = 0.79. The second experiment replaces the threshold variable with the unemployment rate (lagged one quarter), in a regression of inflation on its lag and unemployment. Linearity is not rejected (sup-F 5.8, p = 0.86). The estimated threshold of 5.2% (interval 4.3% to 7.6%) is then not meaningful, and the coefficients by regime should not be interpreted. This is the output you should expect when there is no threshold: a threshold estimate always exists, and only the bootstrap test says whether it is real.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

We find moderate evidence of threshold nonlinearity in U.S. inflation. The bootstrap sup-F test of linearity gives a statistic of 25.9 and a p-value of 0.081 (999 draws, with the delay searched over one to four quarters), so linearity is rejected at 10% but not at 5%. The estimated threshold is 4.23% inflation, three quarters earlier (95% interval 4.01% to 5.51%). When inflation was at or below this level (180 quarters), the AR(2) coefficients sum to 0.81 and the implied mean is 2.2%; above it (56 quarters) they sum to 0.76 and the implied mean is 6.6%. Fifty of the 56 high-regime quarters lie in the 1970s and 1980s, so the threshold may reflect the Great Inflation, which the Bai–Perron procedure dates with mean shifts in 1973Q2 and 1982Q4. The threshold near 4% is stable across lag orders, delays and trimming, but its statistical support weakens with three or four lags (p = 0.16 and 0.19) and we treat it as suggestive. We find no evidence of a second threshold (p = 0.79), nor of a threshold in the unemployment rate (p = 0.86).
</div>

| Report | What | Where |
|---|---|---|
| Model | Regressors, threshold variable, delay, trimming | Step 1 |
| Data | Persistence of the series | Step 2 |
| Test | Sup-F statistic, bootstrap p-value, number of draws | Step 3 |
| Threshold | Estimate, confidence interval, regimes and their coefficients | Step 4 |
| Diagnostics | Likelihood-ratio plot, time in each regime | Step 5 |
| Robustness | Lags, delay and trimming | Step 6 |
| Extensions | Second threshold, another threshold variable | Step 7 |

**References.** Tong, H. (1978), "On a Threshold Model," in *Pattern Recognition and Signal Processing*. Hansen, B. E. (1996), "Inference When a Nuisance Parameter Is Not Identified Under the Null Hypothesis," *Econometrica* 64. Hansen, B. E. (1997), "Inference in TAR Models," *Studies in Nonlinear Dynamics and Econometrics* 2. Hansen, B. E. (2000), "Sample Splitting and Threshold Estimation," *Econometrica* 68. Bai, J. (1997), "Estimating Multiple Breaks One at a Time," *Econometric Theory* 13.
