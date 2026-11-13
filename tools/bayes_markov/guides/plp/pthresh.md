title: Panel Threshold Regression (Hansen)
short: Panel threshold
when: The effect of a regressor may differ above and below a threshold in another variable, and you want to estimate where the threshold is
data: North Carolina counties, 1981–1987 (Cornwell and Trumbull 1994): crime rate, probability of arrest, probability of conviction, average sentence and police per capita
tools: Python · OES engine · statsmodels
oes: pthresh
lead: Use this guide when theory suggests a regime change in a panel: deterrence that works only where enforcement is strong, debt that hurts growth only above a ratio, credit that matters only for small firms. Hansen's (1999) panel threshold regression estimates the threshold, tests whether it exists, builds a confidence interval for it and reports separate slopes in each regime. The guide covers the setup, the bootstrap test, the interval, the second threshold, robustness and the text for the paper.
---

## The question and the model

**Does the slope change when a threshold variable crosses a value?** With unit effects \(\mu_i\), a regressor \(x_{it}\), controls \(z_{it}\) and a threshold variable \(q_{it}\), the one-threshold model is

\[
y_{it} = \mu_i + \beta_1' x_{it}\,\mathbf{1}(q_{it} \le \gamma) + \beta_2' x_{it}\,\mathbf{1}(q_{it} > \gamma) + \delta' z_{it} + e_{it}.
\]

The unit effects are removed by the within transformation. For each candidate \(\gamma\) the model is linear in the slopes, so \(\gamma\) is estimated by least squares: the value that gives the smallest sum of squared residuals over a grid. Inference has two nonstandard parts. Under the null of no threshold, \(\gamma\) is not identified, so the usual \(F\) test fails and Hansen's bootstrap supplies the \(p\)-value. The confidence interval for \(\gamma\) is the set of values at which the likelihood-ratio statistic stays below a critical value, not a symmetric interval around the estimate.

<div class="read" markdown="1">
**Three checks before you trust a threshold model**

1. **Is there a threshold at all?** Look at the bootstrap \(p\)-value of the test against a linear model first. If it does not reject, the regime estimates are descriptive.
2. **Is the threshold precisely located?** A wide likelihood-ratio interval or a flat likelihood means the data barely locate the threshold, and the regime slopes depend on where you put it.
3. **Is each regime well populated?** A regime with few observations or few units gives noisy slopes. The trimming fraction keeps every regime above a minimum share of the sample.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We estimate a panel threshold regression (Hansen 1999) in which the effect of the probability of arrest on the crime rate may differ across counties with low and high police presence. The crime rate and the probability of arrest are in logs, the model includes county fixed effects, and the probability of conviction and the average sentence enter with a common slope. The threshold variable is log police per capita. The threshold is searched over 100 grid points between the 5th and 95th percentiles of the threshold variable, the test against a linear model uses a bootstrap with 300 replications, and standard errors of the slopes are clustered by county.
</div>

## Load the data and look at the threshold variable

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

data = open("crime4.csv").read()
df = pd.read_csv("crime4.csv")
print(f"{df.county.nunique()} counties, {df.year.min()}-{df.year.max()}, observations per county: {sorted(df.groupby('county').size().unique())}")
print(df[["lcrmrte", "lprbarr", "lprbconv", "lavgsen", "lpolpc"]].describe().round(2).loc[["mean", "std", "min", "max"]])
print(df.lpolpc.quantile([.05, .25, .5, .75, .95]).round(2).to_dict())
print("share of the variance of log police that is between counties:",
      round(df.groupby("county").lpolpc.transform("mean").var() / df.lpolpc.var(), 2))
```

<div class="read" markdown="1">
**How to read it**

The panel is balanced: 90 counties observed for seven years (1981-87), 630 observations. The threshold variable, log police per capita, has a median of -6.54 and a 5th to 95th percentile range of -7.17 to -5.74. Seventy-eight percent of its variance is between counties, so the regimes are mostly defined by which counties are well policed, while the slope is identified from changes within counties over time. A regime that is almost a fixed county characteristic is worth knowing about before interpreting the result as an effect of changing police levels.
</div>

## Test for a threshold and estimate it

```python
spec = dict(y="lcrmrte", x=["lprbarr"], controls=["lprbconv", "lavgsen"], threshold="lpolpc", unit="county", time="year",
            trim=0.05, grid=100, boot=300)
r = oes.run("pthresh", data, spec)
oes.summary(r)
oes.show(r, "Tests for the number")
oes.show(r, "Estimated thresholds")
oes.plot(r, "Likelihood-ratio", ncols=1, size=(6.5, 3))
```

<div class="read" markdown="1">
**How to read it**

The bootstrap test rejects a linear model: the F statistic is 31.5 and the bootstrap p-value is 0.027, with 5% and 1% critical values of 24.8 and 33.4. The threshold is -6.38 in log police per capita, which puts 69% of the observations (432 of 630) in the low-police regime. The 95% confidence interval runs from -6.49 to -5.76; its upper end is the upper edge of the grid (the 95th percentile of the threshold variable), so the likelihood-ratio statistic stays flat above the estimate and the data cannot rule out a higher threshold. The likelihood-ratio chart shows the same: a sharp dip at the estimate and a long flat stretch to the right.
</div>

## Slopes by regime

```python
oes.show(r, "Coefficients by regime")
oes.show(r, "Observations by regime")
lin = oes.run("fe", data, dict(y="lcrmrte", x=["lprbarr", "lprbconv", "lavgsen"], unit="county", time="year", effects="unit"))
print("Linear fixed-effects slope on lprbarr:", lin["tables"][0]["rows"][0][1:3])
```

<div class="read" markdown="1">
**How to read it**

The slope of the crime rate on the probability of arrest is -0.153 (standard error 0.122) in counties with police per capita at or below the threshold and -0.289 (0.105) above it. Only the high-police slope is significantly different from zero (p = 0.006); the 95% intervals of the two slopes overlap (-0.39 to 0.09 against -0.49 to -0.08). The linear fixed-effects slope is -0.194 (0.112), roughly the average of the two regimes. The pattern fits the view that a higher arrest probability deters crime more where enforcement is stronger, but the data support it only moderately: the threshold is detected, while the difference between the two slopes is imprecisely estimated.
</div>

## One threshold or two?

```python
oes.show(r, "Tests for the number")
two = oes.run("pthresh", data, dict(spec, max_thresholds=2))
print([s["value"] for s in two["summary"] if s["label"] in ("Thresholds found", "Threshold estimate")])
```

<div class="read" markdown="1">
**How to read it**

The test of one threshold against two does not reject (F = 12.7, bootstrap p = 0.19, 5% critical value 18.9), so one threshold is enough and the engine reports the single-threshold model. The second threshold is searched with the first fixed at its estimate, which is valid because the first estimate converges faster.
</div>

## Robustness: grid, trimming, bootstrap seed and controls

```python
variants = {"baseline": {}, "trim 10%": dict(trim=0.10), "trim 15%": dict(trim=0.15), "grid 50": dict(grid=50), "grid 200": dict(grid=200),
            "other seed": dict(seed=2024), "no controls": dict(controls=[]), "add conviction slope": dict(x=["lprbarr", "lprbconv"], controls=["lavgsen"])}
out = {}
for name, v in variants.items():
    rr = oes.run("pthresh", data, dict(spec, **v))
    t = {s["label"]: s["value"] for s in rr["summary"]}
    c = [x for x in rr["tables"] if x["title"].startswith("Coefficients")][0]["rows"]
    arrest = [x[1] for x in c if x[0].startswith("lprbarr")]
    out[name] = [t["Threshold estimate"], t["Bootstrap p (F1)"], arrest[0], arrest[1]]
print(pd.DataFrame(out, index=["threshold", "p (no threshold)", "arrest slope, low", "arrest slope, high"]).T.round(3))
```

<div class="read" markdown="1">
**How to read it**

The threshold estimate is stable: -6.37 to -6.41 under every variant. The test against a linear model rejects at 5% in every case except without controls (p = 0.073), where both arrest slopes are smaller in size (-0.044 and -0.160) but the high-police slope is still about three times the low-police slope. When the slope of the probability of conviction is also allowed to change with the regime, the arrest slopes become -0.164 and -0.312, so the difference is, if anything, larger. A grid of 50, 100 or 200 points, trimming of 5% to 15% and a different bootstrap seed change the threshold by at most 0.01 and the bootstrap p-value stays between 0.003 and 0.027.
</div>

```python
drop = []
for y in sorted(df.year.unique()):
    rr = oes.run("pthresh", df[df.year != y].to_csv(index=False), dict(spec, boot=100))
    t = {s["label"]: s["value"] for s in rr["summary"]}
    drop.append((int(y), round(t["Threshold estimate"], 2), round(t["Bootstrap p (F1)"], 3)))
print(drop)
```

<div class="read" markdown="1">
**How to read it**

Dropping one year at a time moves the threshold estimate between -6.67 and -5.72 (it is -6.37 to -6.45 in four of seven cases) and the bootstrap p-value between 0.00 and 0.06, with only 1981 pushing it above 5% (p = 0.06). The result does not depend on a single year, though with seven years a leave-one-year-out check has little power to distinguish. Report the range.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

A test against a linear model rejects at the 5% level (Hansen bootstrap, F = 31.5, p = 0.027) and locates a threshold at -6.38 in log police per capita, with 69% of the county-years below it (95% confidence interval -6.49 to -5.76, bounded above by the edge of the search grid). Below the threshold the elasticity of the crime rate with respect to the probability of arrest is -0.15 (clustered standard error 0.12); above it, -0.29 (0.10). A second threshold is not supported (p = 0.19). The threshold is stable across grid size, trimming, the bootstrap seed and the choice of controls, and the high-police slope is larger in size in every variant, but the confidence intervals of the two slopes overlap, so we read the evidence as consistent with stronger deterrence where policing is heavier rather than as a precise estimate of the difference. Because 78% of the variation in police per capita is between counties, the regimes largely separate counties rather than periods.
</div>

| Report | What | Where |
|---|---|---|
| Model | Regressors, threshold variable, controls, effects, grid, bootstrap | Step 1 |
| Data | Distribution of the threshold variable | Step 2 |
| Test and estimate | Bootstrap test, threshold, likelihood-ratio interval | Step 3 |
| Slopes | Coefficients by regime, comparison with the linear model | Step 4 |
| Number of thresholds | One against two | Step 5 |
| Robustness | Grid, trimming, seed, controls, dropping years | Step 6 |

**References.** Hansen, B. (1999), "Threshold Effects in Non-Dynamic Panels: Estimation, Testing, and Inference," *Journal of Econometrics* 93. Hansen, B. (2000), "Sample Splitting and Threshold Estimation," *Econometrica* 68. Bai, J. (1997), "Estimating Multiple Breaks One at a Time," *Econometric Theory* 13. Cornwell, C. and Trumbull, W. (1994), "Estimating the Economic Model of Crime with Panel Data," *Review of Economics and Statistics* 76. Wang, Q. (2015), "Fixed-Effect Panel Threshold Model Using Stata," *Stata Journal* 15.
