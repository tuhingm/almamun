title: Forecast Comparison and the Diebold-Mariano Test
short: Forecast comparison
when: You claim that a model forecasts better than another, or better than a simple benchmark
data: U.S. quarterly inflation, 1959–2009, forecasts one and four quarters ahead
tools: Python · statsmodels · scipy
lead: Use this guide when your paper says that a model forecasts well. A good in-sample fit says little about forecasting, so the claim must be tested out of sample against a simple benchmark, with a test that tells you whether the gain is larger than chance. The guide builds rolling forecasts without look-ahead, compares four models, applies the Diebold-Mariano test with the small-sample correction, and shows how the answer changes with the horizon.
---

## The question and the model

**Does a model forecast inflation better than a random walk, and is the difference more than luck?** Forecasts are produced at each date \(t\) using only the data up to \(t\) (an expanding window), \(h\) periods ahead. If \(e^{A}_{t+h}\) and \(e^{B}_{t+h}\) are the errors of two models and \(L\) is the loss (squared or absolute error), the loss differential is \(d_t = L(e^{A}_t) - L(e^{B}_t)\). The Diebold-Mariano test checks \(H_0: E[d_t]=0\):

\[
DM = \frac{\bar d}{\sqrt{\hat\lambda / T}},\qquad \hat\lambda = \hat\gamma_0 + 2\sum_{j=1}^{h-1}\hat\gamma_j ,
\]

where the long-run variance uses \(h-1\) autocovariances, because forecast errors \(h\) steps ahead are correlated up to lag \(h-1\). We use the small-sample correction of Harvey, Leybourne and Newbold and the \(t\) distribution.

| Model | Forecast |
|---|---|
| Random walk | the last observed value |
| Historical mean | the average of all data so far |
| AR (BIC) | an autoregression whose lag length is chosen by BIC at each date |
| Damped exponential smoothing | a damped-trend exponential smoothing model |

<div class="read" markdown="1">
**Before you test anything**

1. **Use a benchmark that is hard to beat.** For a persistent series a random walk is the usual benchmark; for a noisy series the historical mean may be better.
2. **Never use future data.** Models must be estimated, and their lags selected, with data up to the forecast date only.
3. **Keep a long evaluation sample.** With few forecasts the test has little power; the sample used here has about 100.
</div>

<div class="write" markdown="1">
**How to write it: methods sentence**

We evaluate forecasts out of sample with an expanding window that starts with 100 quarters, and compare squared forecast errors with the Diebold-Mariano test, using the Harvey-Leybourne-Newbold correction. The benchmark is a random walk.
</div>

## Load the data

```python
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing

d = pd.read_csv("us_macro.csv")
x = d["inflation"].to_numpy(float); n = len(x); w = 100
print(f"{n} quarters; the first forecast uses {w} quarters; mean {x.mean():.2f}, sd {x.std():.2f}")
print("first-order autocorrelation:", round(np.corrcoef(x[1:], x[:-1])[0, 1], 3))
```

<div class="read" markdown="1">
**How to read it**

Inflation is persistent but noisy: the first-order autocorrelation is about 0.65. That is why a random walk is a poor forecaster for a noisy series (it passes all the noise into the forecast) and why a model that averages over the past can do better.
</div>

## Step 1: Rolling forecasts without look-ahead

```python
def ar_forecast(z, h, maxp=4):
    best = None
    for p in range(1, maxp + 1):
        Y = z[p:]; X = np.column_stack([np.ones(len(Y))] + [z[p - j:len(z) - j] for j in range(1, p + 1)])
        b = np.linalg.lstsq(X, Y, rcond=None)[0]; e = Y - X @ b
        bic = np.log(e @ e / len(Y)) + (p + 1) * np.log(len(Y)) / len(Y)
        if best is None or bic < best[0]: best = (bic, p, b)
    _, p, b = best; hist = list(z[-p:][::-1])
    for _ in range(h):
        hist.insert(0, b[0] + sum(b[j + 1] * hist[j] for j in range(p)))
    return hist[0]

def rolling(h):
    fc = {k: [] for k in ["random walk", "historical mean", "AR (BIC)", "exp. smoothing"]}; act = []
    for t in range(w, n - h + 1):
        z = x[:t]; act.append(x[t + h - 1])
        fc["random walk"].append(z[-1]); fc["historical mean"].append(z.mean()); fc["AR (BIC)"].append(ar_forecast(z, h))
        fc["exp. smoothing"].append(ExponentialSmoothing(z, trend="add", damped_trend=True).fit().forecast(h)[-1])
    return np.array(act), {k: np.array(v) for k, v in fc.items()}

act1, fc1 = rolling(1)
print(len(act1), "one-step-ahead forecasts")
fig, ax = plt.subplots(figsize=(8, 3.3)); ax.plot(act1, color="k", lw=1.2, label="actual")
for k in ("random walk", "AR (BIC)"): ax.plot(fc1[k], lw=1, label=k)
ax.legend(frameon=False); ax.set_title("Inflation: actual and one-step forecasts"); plt.tight_layout(); plt.show()
```

<div class="read" markdown="1">
**How to read it**

The random-walk forecast is the actual series delayed by one quarter, so it is as noisy as the data; the AR forecast is smoother and tracks the level without chasing every spike. The forecasts for the 102 quarters after 100 quarters of estimation are never based on data from the future.
</div>

## Step 2: Accuracy and the Diebold-Mariano test

```python
def dm(dl, h):
    T = len(dl); m = dl.mean(); lam = ((dl - m) @ (dl - m)) / T
    for j in range(1, h): lam += 2 * ((dl[j:] - m) @ (dl[:-j] - m)) / T
    s = m / np.sqrt(lam / T) * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return s, 2 * stats.t.sf(abs(s), T - 1)

def table(act, fc, h, bench="random walk"):
    err = {k: act - v for k, v in fc.items()}; rows = []
    for k in fc:
        rmse = np.sqrt((err[k] ** 2).mean()); rel = rmse / np.sqrt((err[bench] ** 2).mean())
        s, p = (np.nan, np.nan) if k == bench else dm(err[bench] ** 2 - err[k] ** 2, h)
        rows.append([k, round(rmse, 3), round(np.abs(err[k]).mean(), 3), round(rel, 3), round(s, 3), round(p, 4)])
    return pd.DataFrame(rows, columns=["model", "RMSE", "MAE", "RMSE / benchmark", "DM stat", "p-value"])
print("One quarter ahead (benchmark: random walk)"); print(table(act1, fc1, 1).to_string(index=False))
```

<div class="read" markdown="1">
**How to read it**

All three models have a lower root mean squared error than the random walk (2.88): 2.70 for the historical mean, 2.43 for the AR and 2.41 for the smoothing model, which is a gain of about 16% for the last two. But none of the differences is significant at 5% (p-values 0.63, 0.21 and 0.11). With about 100 forecasts and noisy errors, a gain of this size cannot be told from chance. The finding to report is that the data are consistent with an improvement, and not that the AR model is better.
</div>

## Step 3: The answer depends on the horizon

```python
act4, fc4 = rolling(4)
print("Four quarters ahead (benchmark: random walk)"); print(table(act4, fc4, 4).to_string(index=False))
```

<div class="read" markdown="1">
**How to read it**

Four quarters ahead the random walk does worse (RMSE 3.16) and the AR and smoothing models are significantly better (relative RMSE 0.80, p = 0.015 and 0.014). The gain from averaging over the past grows with the horizon, since the noise in the last observation matters more. The historical mean does not beat the random walk significantly (p = 0.21). A paper that reports one horizon only can give a misleading picture, so report several.
</div>

## Step 4: Cumulative errors over time

```python
err = {k: act1 - v for k, v in fc1.items()}
fig, ax = plt.subplots(figsize=(8, 3.3))
for k in fc1: ax.plot(np.cumsum(err[k] ** 2), label=k)
ax.legend(frameon=False); ax.set_title("Cumulative squared one-step forecast error"); plt.tight_layout(); plt.show()
print("Share of the random walk's squared error accumulated in the final third:", round(float((err['random walk'][-34:] ** 2).sum() / (err['random walk'] ** 2).sum()), 2))
```

<div class="read" markdown="1">
**How to read it**

A model that is good on average can be good only in some periods. If the lines separate in a few quarters (for example during the 2008 crisis) and run parallel for the rest, the gain comes from those episodes. Here, 66% of the random walk's total squared error falls in the last third of the forecasts, which includes the 2008 crisis, so the large errors are concentrated in a short period. Plot the cumulative error, and check that the gain does not come from a few observations.
</div>

## What to report in the paper

Report the benchmark, the estimation scheme (expanding or rolling window and its starting size), the number of forecasts, the loss function, the RMSE of each model relative to the benchmark, the test, and the horizons. Say if the data are revised values or real-time vintages.

<div class="write" markdown="1">
**How to write it: results**

Table 4 reports out-of-sample forecasts of inflation. One quarter ahead, an autoregression and a damped exponential-smoothing model reduce the root mean squared error by about 16% relative to a random walk, but the differences are not significant (Diebold-Mariano p = 0.21 and 0.11). Four quarters ahead, the reduction is 20% and is significant at the 5% level (p = 0.015 and 0.014).
</div>

## Common mistakes

<div class="warn" markdown="1">
**Mistakes to avoid**

1. **Using in-sample fit as evidence of forecasting.** A model with more parameters always fits better in sample.
2. **Look-ahead.** Estimating the model or choosing the lags on the full sample before forecasting.
3. **No benchmark.** An RMSE without a benchmark has no meaning.
4. **Not correcting for the horizon.** For \(h>1\) the errors are autocorrelated; use \(h-1\) autocovariances in the long-run variance.
5. **Using Diebold-Mariano for nested models.** The distribution is not standard; use the Clark-West test.
6. **Too few forecasts.** The test has little power below about 50 forecasts.
7. **Reporting one horizon.** The ranking of the models changes with the horizon.
8. **Ignoring data revisions.** Forecast comparisons with revised data can differ from real-time ones.
</div>
