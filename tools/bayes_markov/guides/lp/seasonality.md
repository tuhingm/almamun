title: Seasonality: STL Decomposition and Seasonal Adjustment
short: Seasonality
when: Your series repeats a pattern every quarter or month and you must decide whether to adjust, model or ignore it
data: UK quarterly gas consumption, 1960–1986
tools: Python · statsmodels · scipy
lead: Use this guide when your data are not seasonally adjusted. Seasonality can swamp the economic signal, create false cycles and bias a regression, so you must measure it, test it and decide what to do. The guide decomposes a seasonal series with STL, measures how strong the seasonal pattern is, tests for it, checks whether the pattern is stable or changes over time, and shows when to adjust the series and when to put seasonal terms in the model instead.
---

## The question and the model

**How much of the movement in UK gas consumption is a regular seasonal pattern, and has that pattern changed?** The series is decomposed into a trend, a seasonal component and a remainder. Gas consumption has an amplitude that grows with the level, so we work with logs, which make the decomposition additive:

\[
\ln y_t = T_t + S_t + R_t ,
\]

and the multiplicative seasonal factor is \(\exp(S_t)\). STL (seasonal-trend decomposition using LOESS, Cleveland et al., 1990) estimates the components by repeated local regressions. Unlike fixed seasonal dummies, it lets the seasonal pattern change slowly, and with the robust option it is not distorted by outliers. Two summary numbers describe the series:

\[
F_S = \max\!\Big(0,\,1-\frac{\mathrm{Var}(R_t)}{\mathrm{Var}(S_t+R_t)}\Big),\qquad F_T = \max\!\Big(0,\,1-\frac{\mathrm{Var}(R_t)}{\mathrm{Var}(T_t+R_t)}\Big).
\]

A value near one means that the component dominates the variation after the remainder is removed.

<div class="read" markdown="1">
**Before you estimate anything**

1. **Plot the series by season.** A plot of each year's pattern shows if the seasonal effect is stable and if its size grows with the level (use logs then).
2. **Decide what you need.** For a descriptive or forecasting task, a seasonal component may be what you want; for a question about the business cycle, you want the adjusted series.
3. **Know what STL is not.** It is not the X-13 ARIMA-SEATS method used by statistical agencies, and it does not remove trading-day or holiday effects.
</div>

<div class="write" markdown="1">
**How to write it: methods sentence**

The series is not seasonally adjusted, so we decompose its logarithm into trend, seasonal and remainder components with STL (Cleveland et al., 1990), with a robust fit and a seasonal window of 7, which lets the seasonal pattern change slowly. We test for seasonality with an F test on seasonal dummies in the detrended series.
</div>

## Load the data and plot it

```python
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from scipy import stats

d = pd.read_csv("uk_gas.csv"); y = d["gas"].to_numpy(float); n = len(y)
d["year"] = d["date"].str[:4].astype(int); d["q"] = d["date"].str[-1].astype(int)
print(f"{n} quarters, {d.date.iloc[0]} to {d.date.iloc[-1]}")
print(d.pivot(index="year", columns="q", values="gas").iloc[[0, 5, 10, 15, 20, 26]].round(0).to_string())
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(y); ax[0].set_title("Gas consumption")
for q in (1, 2, 3, 4): ax[1].plot(d[d.q == q].year, d[d.q == q].gas, label=f"Q{q}")
ax[1].legend(frameon=False, ncol=4); ax[1].set_title("By quarter"); plt.tight_layout(); plt.show()
```

<div class="read" markdown="1">
**How to read it**

Consumption is highest in the first quarter (winter heating) and lowest in the third, and the gap between them widens as consumption grows: in 1960 the first quarter is 160 and the third 85, and in 1986 they are 1,164 and 347. The seasonal swing grows with the level, which is why we use logs.
</div>

## Step 1: Decompose the series with STL

```python
z = np.log(y)
res = STL(z, period=4, robust=True, seasonal=7).fit()
S, T, R = np.asarray(res.seasonal), np.asarray(res.trend), np.asarray(res.resid)
Fs = max(0, 1 - R.var() / (S + R).var()); Ft = max(0, 1 - R.var() / (T + R).var())
print(f"seasonal strength {Fs:.3f}, trend strength {Ft:.3f}")
print("share of the variance of the log series: trend %.1f%%, seasonal %.1f%%, remainder %.2f%%" % tuple(100 * np.var(c) / np.var(z) for c in (T, S, R)))
fig, ax = plt.subplots(3, 1, figsize=(8, 5.2), sharex=True)
ax[0].plot(T); ax[0].set_title("Trend (log)"); ax[1].plot(S); ax[1].set_title("Seasonal"); ax[2].plot(R); ax[2].set_title("Remainder")
plt.tight_layout(); plt.show()
```

<div class="read" markdown="1">
**How to read it**

Both components are strong: the seasonal strength is 0.97 and the trend strength 0.99, so, once the remainder is removed, almost all of the variation is explained by the trend and the season. In the variance of the log series, the trend accounts for 67.9%, the seasonal component for 31.0% and the remainder for only 0.9%. The remainder is small and is the part that contains the news in the series.
</div>

## Step 2: Test for seasonality

```python
per = np.arange(n) % 4
det = z - T                                                  # detrended series
D = np.column_stack([(per == j).astype(float) for j in range(4)])
fit = sm.OLS(det, D).fit()
F = ((det - det.mean()) @ (det - det.mean()) - fit.ssr) / 3 / (fit.ssr / (n - 4))
kw = stats.kruskal(*[det[per == j] for j in range(4)])
print(f"F test of equal quarterly means: F = {F:.1f}, p = {stats.f.sf(F, 3, n - 4):.2g}")
print(f"Kruskal-Wallis test: H = {kw[0]:.1f}, p = {kw[1]:.2g}")
fac = pd.DataFrame({"mean factor": [np.exp(S[per == j]).mean() for j in range(4)], "min": [np.exp(S[per == j]).min() for j in range(4)], "max": [np.exp(S[per == j]).max() for j in range(4)]},
                   index=["Q1", "Q2", "Q3", "Q4"])
print(fac.round(3).to_string())
```

<div class="read" markdown="1">
**How to read it**

Both tests reject equal means across the quarters (F = 159, Kruskal-Wallis 86, p below 0.001). The factors show consumption in Q1 about 55% above the trend and in Q3 about 42% below it (a factor of 0.58). The range from minimum to maximum of each factor is wide (for example 1.33 to 1.86 in Q1), which means that the seasonal pattern is not fixed, which the next step examines.
</div>

## Step 3: Is the seasonal pattern stable?

```python
thirds = np.array_split(np.arange(n), 3)
amp = [np.ptp([np.mean([S[k] for k in th if per[k] == j]) for j in range(4)]) for th in thirds]
print(pd.Series(amp, index=["1960-68", "1969-77", "1978-86"], name="seasonal amplitude (log units)").round(3).to_string())
fixed = sm.OLS(z - T, D).fit().fittedvalues                    # a fixed-dummy seasonal pattern
print("\nvariance of the remainder: STL %.4f, fixed dummies %.4f" % (R.var(), (z - T - fixed).var()))
```

<div class="read" markdown="1">
**How to read it**

The amplitude of the seasonal pattern in logs doubles over the sample: 0.67 in 1960–68, 0.94 in 1969–77 and 1.35 in 1978–86. This is a change in the relative seasonal swing, and it means that fixed seasonal dummies are not a good model: the remainder of the fixed-dummy model has a larger variance than that of STL (see the output). When the pattern evolves, use a method that allows it, and do not assume constant dummies.
</div>

## Step 4: The seasonally adjusted series

```python
adj = np.exp(z - S)
fig, ax = plt.subplots(figsize=(8, 3.2)); ax.plot(y, alpha=.5, label="original"); ax.plot(adj, label="seasonally adjusted"); ax.legend(frameon=False)
ax.set_title("Gas consumption, original and seasonally adjusted"); plt.tight_layout(); plt.show()
print("quarterly growth, standard deviation (%):  original", round(100 * np.diff(np.log(y)).std(), 1), "  adjusted", round(100 * np.diff(np.log(adj)).std(), 1))
```

<div class="read" markdown="1">
**How to read it**

Adjustment removes the saw-tooth from the series and leaves the underlying growth; the standard deviation of quarterly growth falls from about 56% to 10%, which shows how much of the quarter-to-quarter movement in the raw data is seasonal. Adjusting is appropriate when the question is about the trend or the cycle. When seasonality is part of the question, such as forecasting demand, keep it in the model.
</div>

## What to report in the paper

Say whether the series is seasonally adjusted and by what method; report the strength of the seasonal pattern, the test, and whether the pattern changes; and state how you treat it in the model (adjusted data, seasonal dummies or a seasonal model).

<div class="write" markdown="1">
**How to write it: data section**

The gas series is not seasonally adjusted and has a strong seasonal pattern (seasonal strength 0.97; F test of equal quarterly means, p < 0.001) whose amplitude in logs doubles between 1960–68 and 1978–86. Because fixed seasonal dummies cannot capture this change, we use the STL-adjusted series in the analysis of the trend and report the results with the unadjusted data in the appendix.
</div>

## Common mistakes

<div class="warn" markdown="1">
**Mistakes to avoid**

1. **Using unadjusted data without saying so.** Seasonal peaks produce false cycles and spurious correlations.
2. **Adjusting twice.** Do not adjust a series that the agency has already adjusted.
3. **Fixed dummies for a pattern that evolves.** The residual keeps the seasonal.
4. **Additive decomposition when the swing grows with the level.** Use logs.
5. **Adjusting only one of the variables in a regression.** Treat all the series the same way.
6. **Adjusting a series and then using it in a unit-root or break test.** Adjustment can distort the dynamics.
7. **Reading the end of the adjusted series as final.** The adjustment of recent points is revised as data are added.
8. **Treating STL as an official adjustment.** It does not model calendar effects.
</div>
