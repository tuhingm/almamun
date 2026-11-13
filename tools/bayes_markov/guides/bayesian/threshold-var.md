title: Bayesian Threshold VAR
short: Threshold VAR
when: Dynamics that differ when an observed variable, such as inflation, is above or below a threshold
data: U.S. quarterly GDP growth, inflation and the T-bill rate, 1959–2009
tools: Python · OES engine · statsmodels
oes: tvar
lead: Use this guide when the economy behaves differently in two states that you can observe, such as high and low inflation, and you want to know where the threshold is and whether the two-regime model beats a single linear VAR. The Bayesian threshold VAR finds the threshold and the delay with an exact posterior, and the Bayes factor tells you whether the nonlinearity is needed. The guide covers choosing the threshold variable, estimation, regime dates, responses by regime, robustness and the write-up.
---

## The question and the model

**Does monetary policy work differently when inflation is high?** A threshold VAR (Tsay 1998; Koop and Potter 2006) splits the sample into two regimes according to whether a *threshold variable* \(z_{t-d}\) lies above a threshold \(r\):

\[
y_t = \begin{cases} c_1 + A_{11} y_{t-1} + \dots + A_{1p} y_{t-p} + u_t, \; u_t \sim N(0, \Sigma_1), & z_{t-d} \le r, \\[3pt] c_2 + A_{21} y_{t-1} + \dots + A_{2p} y_{t-p} + u_t, \; u_t \sim N(0, \Sigma_2), & z_{t-d} > r. \end{cases}
\]

The coefficients and the covariance matrix differ between regimes. The threshold \(r\) and the delay \(d\) are unknown. In the Bayesian version used here, each regime has a Normal-Wishart prior (Minnesota-type moments), so its marginal likelihood is available in closed form and the **posterior of \((r, d)\) is exact**, computed over a grid of candidate thresholds. There is nothing to simulate for the threshold, so no convergence to check; coefficients are then drawn within each regime.

<div class="read" markdown="1">
**Two checks before you trust a threshold model**

1. **Does the data prefer two regimes?** The log Bayes factor of the threshold model against the linear VAR answers this directly. A threshold VAR can always fit better than a linear VAR; the Bayes factor penalises the extra parameters. If it is negative, use the linear VAR.
2. **Is the threshold variable the right one?** Different candidates give very different Bayes factors (Step 3). Choose by the Bayes factor, and say that you did: the choice is part of the model.

Also check that each regime has enough observations: with two lags of three variables, a regime has 21 coefficients per equation, so a regime of 30 quarters is thin. The prior helps but results for the small regime are imprecise.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We estimate a Bayesian threshold VAR (Tsay 1998; Koop and Potter 2006) in GDP growth, inflation and the three-month Treasury bill rate, with two lags. The regime is determined by whether inflation, \(d\) quarters earlier, exceeds a threshold \(r\). Each regime has its own coefficients and covariance matrix, with a Normal-Wishart prior with Minnesota moments (overall tightness 0.2). Because the marginal likelihood is available analytically in each regime, the joint posterior of the threshold (over a 40-point grid between the 15th and 85th percentiles of the threshold variable) and the delay (one or two quarters) is computed exactly. We compare the threshold model with a linear VAR using the log Bayes factor, and report results for the posterior mode of the threshold.
</div>

## Load the data and test for unit roots

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

data = open("us_macro.csv").read()
df = pd.read_csv("us_macro.csv")
vars_ = ["gdp_growth", "inflation", "tbilrate"]
print(df[vars_].describe().round(2).loc[["mean", "std", "min", "max"]])
spec = dict(vars=vars_, threshold_var="inflation", time="date", lags=2, max_delay=2, trim=0.15, grid=40, lambda1=0.2, horizon=20, draws=1000)
r = oes.run("tvar", data, spec)
oes.show(r, "Unit-root")
```

<div class="read" markdown="1">
**How to read it**

Inflation and GDP growth reject a unit root; the T-bill rate does not (p = 0.28). The regimes are defined by a *stationary* variable, which is what a threshold model needs: a trending threshold variable would spend the whole sample in one regime. Quarterly annualised inflation is volatile (it ranges widely), so the high regime also catches quarters of temporary price spikes, not only the high-inflation years.
</div>

## Do you need a threshold, and on which variable?

Run the model with each candidate threshold variable and compare the log Bayes factor against the linear VAR.

```python
def threshold_summary(res):
    t = oes.frame(next(t for t in res["tables"] if t["title"].startswith("Threshold and regimes")))["Value"]
    return t

rows = []
for z in ["gdp_growth", "inflation", "tbilrate", "unemp", "cons_growth", "inv_growth", "m1_growth"]:
    t = threshold_summary(oes.run("tvar", data, dict(spec, threshold_var=z)))
    rows.append([z, round(t["Threshold r"], 2), int(t["Observations: low regime"]), int(t["Observations: high regime"]),
                 round(t["Log Bayes factor, threshold vs linear VAR"], 1)])
print(pd.DataFrame(rows, columns=["threshold variable", "threshold r", "n low", "n high", "log Bayes factor"]).set_index("threshold variable"))
```

<div class="read" markdown="1">
**How to read it**

A positive log Bayes factor favours the threshold model; above 4.6 is "decisive" on Jeffreys' scale. **Inflation** (26.4) and the **T-bill rate** (16.8) are strongly supported; GDP growth (\(-4.0\)), consumption growth (\(-3.7\)), unemployment (\(-8.4\)) and investment growth (\(-2.9\)) are not (the linear VAR is preferred), and money growth is at zero (1.0). So the economy's dynamics depend on the level of inflation or interest rates, not on the stage of the business cycle measured by growth. Inflation is the natural choice for the question, and its Bayes factor is the largest; we use it below. This is a model choice made on the data: report the table.

Caution: a Bayes factor in favour of a regime split can also reflect different *volatility* in the two regimes (the 1970s were more volatile than the 1990s). Compare with the SV-VAR guide, which allows changing volatility without splitting the coefficients.
</div>

## Estimate

```python
oes.summary(r)
oes.show(r, "Posterior of the delay")
oes.show(r, "Threshold and regimes")
```

<div class="read" markdown="1">
**How to read it**

The data choose a **threshold of 6.65%** annualised quarterly inflation, with a delay of two quarters (the posterior probability of a delay of one quarter is below 0.0001): the economy is in the high regime when inflation two quarters earlier exceeded 6.65%. That regime holds 30 of the 200 quarters (15%) and the low regime 170. The log Bayes factor against a single linear VAR is **26.4**, overwhelming support for two regimes (on Jeffreys' scale, above 4.6 is decisive). The posterior mean of the threshold (6.646) equals the mode (6.647): with a finer grid of 100 points, 95% of the posterior mass sits on 6.65 and 3% on 6.58. Do not over-read this precision: it arises because moving a single quarter between regimes changes the marginal likelihood a great deal when the high regime has only 30 quarters. Step 9 shows how the threshold moves when the minimum regime size changes.
</div>

## Regime dates

```python
oes.show(r, "Regime dating")
oes.plot(r, "inflation(t-2) and the threshold", ncols=1, size=(6.5, 3))
```

<div class="read" markdown="1">
**How to read it**

Because the regime is a function of an observed variable, regime membership is known once the threshold is fixed, so every probability is 1. The 30 high-inflation quarters fall into two real episodes and eight isolated quarters. The episodes are 1974Q1–1975Q2 (the first oil shock) and **1978Q2–1982Q1** (the second oil shock and the Volcker disinflation, 16 quarters). The single quarters are 1969Q3, 1973Q3, 1975Q4, 1977Q3, 1982Q4, 1991Q1, 2006Q1 and 2008Q4: spikes in the quarterly CPI, some from energy prices, not episodes of sustained inflation. In a paper, describe the high regime as largely the oil-shock years, and note that it is close to a period dummy for the 1970s, which limits what can be inferred about "high inflation" in general.
</div>

## What differs between the regimes?

```python
oes.show(r, "Low regime, equation: tbilrate")
oes.show(r, "High regime, equation: tbilrate")
```

<div class="read" markdown="1">
**How to read it**

The policy-rate equation shows what changes. In the low regime the rate follows the economy: it responds to lagged GDP growth (0.046, 95% interval 0.016 to 0.072; probability of a positive sign 1.00) and is very persistent (own lag 0.91). In the high regime the response to growth is small and imprecise (0.015, interval \(-0.10\) to 0.14), the response to inflation is 0.059 (\(-0.11\) to 0.22), and the own lag is lower (0.71, interval 0.38 to 1.02). With only 30 quarters, the intervals are wide: the data cannot say that the rule changed, only that the low-regime rule is estimated precisely and the high-regime rule is not. Report this limitation rather than reading the point estimates as a regime-dependent policy rule.
</div>

## Impulse responses by regime

```python
oes.plot(r, "Response of gdp_growth to a shock in tbilrate", ncols=1, size=(6, 3))
oes.plot(r, "Response of inflation to a shock in tbilrate", ncols=1, size=(6, 3))
```

<div class="read" markdown="1">
**How to read it**

Responses to a one-standard-deviation rise in the T-bill rate, within each regime (the chart shows posterior medians without credible bands). **Output:** in the low-inflation regime GDP growth does not respond (+0.01 after a year); in the high-inflation regime it falls by 0.40 percentage point after a year, 0.18 after two and 0.07 after three. **Inflation:** in the low regime it rises (0.15 after a year, 0.11 after two): the price puzzle of the BVAR guide; in the high regime there is no price puzzle (\(-0.006\) after a year and \(-0.06\) after two). So the textbook sign of monetary transmission appears in the high regime, which is the regime of the oil-shock years, and the price puzzle in the low regime. Because the high regime has 30 observations, do not report these differences as significant without computing credible bands from the draws; the Bayes factor in Step 3 says that the regimes differ, not that each difference is large.
</div>

## Diagnostics

```python
oes.show(r, "Residual diagnostics, low")
oes.show(r, "Residual diagnostics, high")
```

<div class="read" markdown="1">
**How to read it**

In the low regime (170 quarters) the joint portmanteau test does not reject (p = 0.10), but Ljung-Box rejects for inflation (p = 0.0001) and the interest rate (p = 0.034), normality is rejected for both, and the ARCH test rejects for GDP growth (p = 0.035). In the high regime (30 quarters) Ljung-Box rejects for inflation (p = 0.018) and nothing else does, but these tests have little power with 30 observations, so the absence of rejections says little. Remaining autocorrelation in inflation suggests more lags; Step 9 shows that the threshold and the Bayes factor survive three and four lags. Changing volatility between the regimes is not modelled beyond the regime-specific covariance matrix; if it matters to your question, see the SV-VAR guide.
</div>

## Robustness: lags, trimming, grid and prior

```python
def row(**kw):
    t = threshold_summary(oes.run("tvar", data, dict(spec, **kw)))
    return [round(t["Threshold r"], 2), int(t["Observations: high regime"]), round(t["Log Bayes factor, threshold vs linear VAR"], 1)]

variants = {"baseline": {}, "1 lag": dict(lags=1), "3 lags": dict(lags=3), "4 lags": dict(lags=4),
            "min regime share 10%": dict(trim=0.10), "min regime share 25%": dict(trim=0.25),
            "grid of 100": dict(grid=100), "lambda1 = 0.1": dict(lambda1=0.1), "lambda1 = 0.5": dict(lambda1=0.5), "delay up to 4": dict(max_delay=4)}
print(pd.DataFrame({k: row(**v) for k, v in variants.items()}, index=["threshold r", "n high", "log Bayes factor"]).T)
```

<div class="read" markdown="1">
**How to read it**

Each row changes one thing. **Lags:** with one to four lags the threshold is 6.65 and the Bayes factor is 31.8, 26.4, 22.9 and 23.9. **Minimum regime share:** allowing 10% in each regime moves the threshold to 7.05 (29 high quarters); requiring 25% moves it to 5.0 (50 high quarters), with a Bayes factor of 16.7: the threshold is sensitive to how small the high regime may be, so report the trimming. **Grid and delay:** a grid of 100 points and a maximum delay of four quarters change nothing (the delay stays at two). **Prior:** the Bayes factor ranges from 5.4 (\(\lambda_1 = 0.1\)) to 45.7 (\(\lambda_1 = 0.5\)). Marginal likelihoods depend on the prior, so the *size* of the Bayes factor is not robust, but it is positive and large in every variant, so the conclusion that two regimes beat one is. Report a range, not a single number.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

We compare a Bayesian threshold VAR with a linear VAR using the log Bayes factor, for seven candidate threshold variables. The data support inflation (log Bayes factor 26.4) and the Treasury bill rate (16.8) as threshold variables; GDP growth, consumption growth, unemployment and investment growth are not supported (log Bayes factors between \(-8.4\) and \(-2.9\)). With inflation as the threshold variable the posterior concentrates on a delay of two quarters and a threshold of 6.65%, which defines a high-inflation regime of 30 quarters, 22 of them in the oil-shock episodes of 1974–75 and 1978–82. In the low-inflation regime, a rise in the policy rate does not reduce output and raises inflation (the price puzzle); in the high-inflation regime output falls by 0.4 percentage point after a year and inflation does not rise. These regime differences are estimated with 30 observations and are imprecise. The threshold, the delay and the sign of the Bayes factor do not change with one to four lags, the grid, or the prior tightness; the threshold moves between 5.0 and 7.05 depending on the minimum regime size (10% to 25%), and the Bayes factor ranges from 5.4 to 45.7 with the prior.
</div>

| Report | What | Where |
|---|---|---|
| Model | Threshold variable, delay and threshold grid, prior, minimum regime share | Step 1 |
| Choice | Log Bayes factors for the candidate threshold variables; linear VAR comparison | Step 3 |
| Threshold | Posterior mode and mean, regime sizes, regime dates | Steps 4 and 5 |
| Differences | Regime-specific coefficients; responses by regime | Steps 6 and 7 |
| Diagnostics | Residual tests by regime | Step 8 |
| Robustness | Lags, trimming, grid, prior, delay | Step 9 |

**References.** Tsay (1998), *Journal of the American Statistical Association*; Koop and Potter (2006), *Econometrics Journal*; Balke (2000), *Review of Economics and Statistics*; Hubrich and Teräsvirta (2013), *Journal of Economic Surveys*; Kass and Raftery (1995), *Journal of the American Statistical Association*; Chen and Lee (1995), *Journal of Business & Economic Statistics*.
