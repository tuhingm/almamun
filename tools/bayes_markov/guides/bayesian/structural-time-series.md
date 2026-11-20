title: Bayesian Structural Time Series
short: Bayesian structural time series
when: A series with a drifting level, a trend or a seasonal pattern that you want to decompose, interpolate over gaps or forecast with honest uncertainty
data: Nile river flow at Aswan, 1871 to 1970 (100 years); monthly international airline passengers, 1949 to 1960 (144 months), in logs
tools: Python · OES engine · statsmodels
oes: bsts
lead: Use this guide when a series has a level that drifts, perhaps a trend and a seasonal pattern, and you want to split it into those components, fill in missing observations and forecast with an interval that includes every source of uncertainty. The Bayesian version gives a posterior for the level at every date, for the variance of each component and for every forecast, and it can do without the observations that are missing. The guide covers the model, a local level for the Nile, the sampler check, a gap in the data, a trend with a seasonal pattern and a two-year forecast for airline passengers, a hold-out test of the forecast, a coverage simulation and the write-up.
---

## The question and the model

**What is the flow of the Nile now, behind the year-to-year noise, and how sure are we?** Annual flow at Aswan jumps around from year to year because of rainfall, but there is also a slowly moving level (the drop around 1898 that the change-point guide found). A **structural time-series model** separates the two. The simplest, the **local level model**, is

\[
y_t = \mu_t + \varepsilon_t, \qquad \mu_{t+1} = \mu_t + \eta_t, \qquad \varepsilon_t \sim N(0, \sigma_\varepsilon^2), \quad \eta_t \sim N(0, \sigma_\eta^2).
\]

The level \(\mu_t\) is a random walk that is seen through noise. The ratio \(\sigma_\eta / \sigma_\varepsilon\) (the signal-to-noise ratio) decides how fast the level may move: small means that the level is nearly constant and each year's flow is mostly noise. Two components can be added:

- a **slope**: \(\mu_{t+1} = \mu_t + \delta_t + \eta_t\), \(\delta_{t+1} = \delta_t + \zeta_t\), the **local linear trend**, for a series that grows;
- a **seasonal** pattern of period \(s\) (12 for monthly data): \(\gamma_t + \gamma_{t-1} + \dots + \gamma_{t-s+1} = \omega_t\), so that the effects sum to about zero over a cycle and may evolve.

Each component has its own variance. In the Bayesian version all the states and all the variances have a joint posterior. The engine draws **all states at once** from their exact Gaussian conditional distribution (the precision matrix is banded, so a banded Cholesky factorisation does it in milliseconds; Chan and Jeliazkov 2009), then draws each variance from an inverse gamma, so that there is no Kalman-filter loop and no tuning. Observations that are missing are simply left out of the likelihood and the states over the gap are drawn from the model, which interpolates. Future periods are treated the same way as missing ones, which gives forecasts.

| Output | What it answers |
|---|---|
| **Component standard deviations** | How much the level, slope, seasonal and noise move, next to the classical (Kalman filter) estimate |
| **States** | Level and slope at the last date, and the seasonal effects, with intervals |
| **Forecast** | Mean and 95% interval for each future period |
| **Missing values** | The level over a gap, with a band that widens where there is no data |

<div class="read" markdown="1">
**Why a prior on the variances matters**

The variance of a component that is small but not zero is weakly identified: a random-walk slope with a standard deviation of 0.003 and one of 0.000 describe the past about equally well, but they give very different forecasts two years ahead. The Bayesian answer is to put a prior on it. The engine's default guesses that each component moves by 1% of the standard deviation of the series per period (the choice of Scott and Varian 2014), with a weak inverse-gamma prior of shape 1 so that the data can overrule it. Step 7 shows how much the forecasts depend on the guess and how to test it.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We model annual Nile flow as a local level plus noise, estimated by Bayesian methods. The level follows a random walk with variance \(\sigma_\eta^2\); observations equal the level plus independent normal noise with variance \(\sigma_\varepsilon^2\). The prior on each variance is inverse gamma, with a prior guess of 1% of the standard deviation of the series for the level (Scott and Varian 2014) and a weak prior for the noise. We draw the whole state vector jointly from its Gaussian conditional distribution and each variance from its inverse-gamma conditional (Chan and Jeliazkov 2009), in two chains of 3,000 draws after 1,000 burn-in draws, and compare the variances with the maximum-likelihood estimates from the Kalman filter.
</div>

## Load the data

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

nile = open("nile.csv").read()
d = pd.read_csv("nile.csv")
print(d.shape, d.year.min(), "to", d.year.max())
print(d.flow.describe().round(0).loc[["mean", "std", "min", "max"]])
print("first differences: s.d. =", d.flow.diff().std().round(0), " first-order autocorrelation =", d.flow.diff().autocorr().round(2))
```

<div class="read" markdown="1">
**How to read it**

The 100 annual flows have a mean of 919 and a standard deviation of 169 (in hundreds of millions of cubic metres). The first differences have a standard deviation of 168, no smaller than the series itself, and a first-order autocorrelation of \(-0.4\). That is the signature of a local level: a noisy series around a slowly moving level has differences with a negative first autocorrelation. For pure noise around a constant level it would be \(-0.5\); for a random walk without noise it would be 0.
</div>

## Estimate

```python
spec = dict(y="flow", time="year", draws=3000, burn=1000)
r = oes.run("bsts", nile, spec)
oes.summary(r)
oes.show(r, "Posterior of the standard deviations")
oes.show(r, "States")
oes.plot(r)
```

<div class="read" markdown="1">
**How to read it**

The noise standard deviation is 127 (95% interval 105 to 150), close to the Kalman-filter maximum-likelihood value of 123. The level standard deviation is 29 (13 to 60) against the maximum-likelihood 38, which lies well inside the interval. The posterior is wide for this component because the level moves a little each year and the data can hardly tell a small drift from none; the maximum-likelihood point estimate hides that. The signal-to-noise ratio is 0.23: the level moves by about a fifth of the noise each year, so the level is much smoother than the series. The level at the last observation (1970) is 822 with a standard deviation of 61 and a 95% interval of 695 to 931, while the last observation itself is 740, a value that the model treats as mostly noise and pulls back towards the level of the recent years.

The first chart is the series with the posterior level and its 95% band: the level falls from about 1,100 before 1900 to about 850, the same break that the change-point guide dated to 1898, here seen as a fast but smooth descent.
</div>

## Diagnostics: the sampler and the residuals

```python
oes.show(r, "Residual check")
oes.show(r, "MCMC convergence")
```

<div class="read" markdown="1">
**How to read it**

The noise variance has an R-hat of 1.014 and an effective sample size of 455 out of 6,000 draws. The level variance mixes more slowly: R-hat 1.059 and an effective sample size of 125. That is typical, since a small variance is weakly identified; the states themselves mix well because they are drawn in one block. If you report the interval for the level variance, use more draws (10,000 or more) until the R-hat is below 1.05 and the effective size is above 400; the level and forecasts are far less sensitive.

The residual check finds nothing wrong: 5% of the standardised residuals are beyond \(\pm 2\), as expected; the first autocorrelation is 0.09 and the Ljung-Box p-value is 0.15. These residuals come from the smoothed states, so they are a little too well behaved by construction; the hold-out test in Step 7 is the stronger check.
</div>

## Filling in missing observations

Remove the flow for 1920 to 1929 and let the model fill the gap.

```python
m = d.copy()
m.loc[m.year.between(1920, 1929), "flow"] = np.nan          # ten years with no observation
rm = oes.run("bsts", m.to_csv(index=False), spec)
oes.summary(rm)
width = lambda res: np.array(res["charts"][0]["band"]["hi"]) - np.array(res["charts"][0]["band"]["lo"])
lev = lambda res: np.array(res["charts"][0]["series"][1]["y"])
yrs = np.array(rm["charts"][0]["x"]).astype(int)
for y0 in (1915, 1920, 1925, 1929, 1935):
    i = list(yrs).index(y0)
    print(y0, "level with gap:", lev(rm)[i].round(0), " width", width(rm)[i].round(0), "| full data:", lev(r)[i].round(0), " width", width(r)[i].round(0))
```

<div class="read" markdown="1">
**How to read it**

The summary shows that 10 values are filled in and 90 are used. Where the data are missing the level is no longer pinned down by an observation, so the band widens: 228 at 1920 against 165 with all the data, and 268 in 1925 against 173, the middle of the gap. After the gap the band returns to its normal width (182 in 1935 against 177). The level over the gap moves by at most about 20 units from the one estimated with all the data (848 against 827 in 1925), well within the band. A gap is thus filled with an estimate whose uncertainty is reported, not with a point that looks like data.
</div>

## Trend, seasonality and a forecast: airline passengers

The Box and Jenkins airline series, 144 months of international passengers, in logs so that the multiplicative seasonal pattern becomes additive. The model has a local linear trend, a seasonal of period 12 and a 24-month forecast.

```python
air = open("airline.csv").read()
a = oes.run("bsts", air, dict(y="lpass", time="month", trend="trend", seasonal=12, horizon=24, draws=2000, burn=800))
oes.summary(a)
oes.show(a, "Posterior of the standard deviations")
oes.show(a, "States")
oes.show(a, "Seasonal effects")
oes.show(a, "Forecast")
oes.plot(a)
```

<div class="read" markdown="1">
**How to read it**

*Components.* The noise standard deviation is 0.013 (maximum likelihood 0.011), the level 0.026 (0.026), the seasonal 0.0067 (0.0080) and the slope 0.0028 against a maximum-likelihood estimate of essentially zero. That is the weakly identified one: the posterior says the slope varies a little (0.0017 to 0.0049) where the point estimate says it does not vary at all, and it passes this uncertainty to the forecast. The slope at the last observation is 0.0057 a month (about 7% a year) with a posterior probability of 0.75 that it is positive; its interval, \(-0.013\) to 0.023, shows that the recent growth is not well pinned down.

*Seasonality.* July and August stand 0.23 and 0.22 log points above the level (about 26% and 24% in passengers), the lowest month is November, 0.22 log points below (19% fewer passengers), then February, 0.12 below (11% fewer). The intervals are about \(\pm 0.03\) wide, so the pattern is well determined.

*Forecast.* The mean forecast for July 1961 is 6.45 in logs, about 632 passengers (95% interval 6.24 to 6.65, or 510 to 776), and for December 1962 it is 6.20, or about 495 (5.55 to 6.83, or 258 to 923). The interval widens with the horizon as the level and the slope uncertainties accumulate: a standard deviation of 0.04 one month ahead, 0.16 at a year and 0.32 at two years. The smoothed residuals have a negative autocorrelation (\(-0.30\)), which is a property of residuals computed from smoothed states, not a sign of misfit.

The charts show the series with the level and its band, the forecast with the predictive band and the seasonal pattern of the last year.
</div>

## Test the forecast, and the prior, on held-out data

The only honest test of a forecast is to hide the end of the series, forecast it and compare. Fit to December 1958 and forecast the 24 months of 1959 and 1960, with three different guesses for how much each component moves.

```python
ad = pd.read_csv("airline.csv")
train, test = ad.iloc[:120], ad.iloc[120:]                   # fit to December 1958, forecast the 24 months of 1959-60
rows = []
for c in (0.001, 0.01, 0.05):
    h = oes.run("bsts", train.to_csv(index=False), dict(y="lpass", time="month", trend="trend", seasonal=12, horizon=24, prior_scale=c, draws=1500, burn=500))
    fc = [t for t in h["tables"] if t["title"].startswith("Forecast")][0]["rows"]
    sd = {q[0]: q[1] for q in h["tables"][0]["rows"]}
    err = np.array([f[1] - v for f, v in zip(fc, test.lpass)])
    inside = np.mean([f[3] <= v <= f[4] for f, v in zip(fc, test.lpass)])
    rows.append([c, sd["Slope"], np.abs(err).mean().round(3), np.sqrt((err ** 2).mean()).round(3), inside.round(2), round(fc[-1][4] - fc[-1][3], 2)])

print(pd.DataFrame(rows, columns=["prior guess", "slope s.d.", "mean abs. error", "RMSE", "inside 95% band", "band width at +24"]).to_string(index=False))
```

<div class="read" markdown="1">
**How to read it**

All 24 actual values lie inside the 95% band at every prior, so none of the bands is wrong; they differ in width and accuracy. With the tightest guess (0.1% of the standard deviation) the slope standard deviation is 0.0005, the mean absolute error of the forecast is 0.071 in logs (about 7%) and the band at two years is 0.67 wide. With the default (1%) the slope standard deviation is 0.0028, the error is 0.149 and the band 1.28. With a loose guess (5%) the slope is allowed to wander (0.0097), the error is 0.308 and the band 3.14, almost five times as wide as the first. For this series, which has a steady trend, a smaller prior is better, in agreement with the maximum-likelihood estimate of zero for the slope variance; the default is calibrated but conservative.

The lesson is general. The forecast depends on the prior for the slope variance, so report the prior, show a hold-out test and, if forecasts matter, choose the prior by hold-out error (an out-of-sample comparison of a few values, as here), not by the width of the band alone.
</div>

## Diagnostics: do the intervals cover?

Simulate 40 series of 150 observations from a local level with known parameters (noise s.d. 1, level s.d. 0.3), run the default model and count how often the 95% interval for the last level contains the true level.

```python
rng = np.random.default_rng(1)
hit = []
for s in range(40):
    mu = 10 + np.cumsum(rng.normal(scale=0.3, size=150))
    sim = pd.DataFrame({"t": np.arange(150), "y": mu + rng.normal(scale=1.0, size=150)})
    rs = oes.run("bsts", sim.to_csv(index=False), dict(y="y", time="t", draws=500, burn=250, seed=s))
    lo, hi = [rs["tables"][1]["rows"][0][i] for i in (3, 4)]
    hit.append(lo <= mu[-1] <= hi)
print("share of 95% intervals for the last level that contain the truth:", np.mean(hit).round(2), "out of", len(hit))
```

<div class="read" markdown="1">
**How to read it**

The interval contains the true level in 95% of the 40 series (38 of 40), as a 95% interval should; with 40 replications the simulation standard error is about 3.5 points. The test suite repeats this with other designs, compares the Nile variances with the Kalman-filter maximum likelihood (the published values of Durbin and Koopman: 15,099 and 1,469 for the variances) and checks that a seasonal pattern and a regression coefficient are recovered and that forecasts are calibrated.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

The annual flow of the Nile is a smooth level observed with large noise: the posterior mean of the noise standard deviation is 127 (95% interval 105 to 150) and that of the level innovations is 29 (13 to 60), so the signal-to-noise ratio is 0.23. The level at the end of the sample is 822 (95% interval 695 to 931). The variance estimates agree with the maximum-likelihood Kalman-filter values (123 and 38). The R-hat of the variances is below 1.06 and the effective sample sizes range from 125 to 455. For monthly airline passengers, a local linear trend with a seasonal of period 12 gives a seasonal peak in July 0.23 log points (about 26%) above the level and a trough in November 0.22 log points (19%) below, and a forecast of about 632 passengers in July 1961 (95% interval 510 to 776). In a hold-out test, a fit to 1958 forecast 1959 and 1960 with a mean absolute error of 0.07 in logs when the prior guess for the component variances was 0.1% of the standard deviation of the series and 0.15 at 1%, with all 24 observations inside the 95% band.
</div>

| Report | What | Where |
|---|---|---|
| Model | Components, priors, sampler | Step 1 |
| Variances | Posterior of each component's standard deviation, maximum-likelihood comparison | Steps 3 and 6 |
| Level, trend, seasonal | Posterior at the last date, seasonal effects | Steps 3 and 6 |
| Convergence | R-hat, effective sample size, residual check | Step 4 |
| Missing data | Level and band over a gap | Step 5 |
| Forecast | Predictive mean and interval, hold-out test, prior sensitivity | Steps 6 and 7 |
| Coverage | Simulation | Step 8 |

**References.** Harvey (1989), *Forecasting, Structural Time Series Models and the Kalman Filter*, Cambridge University Press; Durbin and Koopman (2012), *Time Series Analysis by State Space Methods*, Oxford University Press; Chan and Jeliazkov (2009), *International Journal of Mathematical Modelling and Numerical Optimisation*; Scott and Varian (2014), *International Journal of Mathematical Modelling and Numerical Optimisation*; Frühwirth-Schnatter (1994), *Journal of Time Series Analysis*; Carter and Kohn (1994), *Biometrika*; Cobb (1978), *Biometrika*; Box and Jenkins (1970), *Time Series Analysis: Forecasting and Control*.
