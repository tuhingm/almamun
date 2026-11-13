title: Markov-Switching VAR
short: MS-VAR
when: Several macro variables whose means, volatility or dynamics shift between hidden regimes, such as calm and turbulent periods
data: U.S. quarterly GDP growth and inflation, 1959–2009
tools: Python · OES engine
oes: msvar
lead: Use this guide when you want to know whether the joint behaviour of several variables changes between unobserved regimes, how long each regime lasts, and whether shocks have different effects in each. The guide covers the unit-root check, choosing the number of regimes and what switches (MSI, MSIH, MSIAH), estimation, regime dates, regime-specific impulse responses, diagnostics, robustness and the write-up.
---

## The question and the model

**Do GDP growth and inflation behave differently in calm and turbulent regimes, and do shocks have different effects?** A Markov-switching VAR (Krolzig 1997) lets the parameters of a VAR depend on an unobserved state \(s_t\) that follows a Markov chain:

\[
y_t = \nu_{s_t} + A_{1,s_t} y_{t-1} + \dots + A_{p,s_t} y_{t-p} + u_t, \qquad u_t \sim N(0, \Sigma_{s_t}).
\]

You choose what switches. **MSI**: only the intercepts. **MSIH**: intercepts and covariance matrix; the autoregressive coefficients are common to all regimes. **MSIAH**: everything switches. The more switches, the more parameters and the more regimes can be fitted to outliers. The model is estimated by maximum likelihood with the EM algorithm (the Hamilton filter and the Kim smoother) from several starting values; standard errors come from the numerical Hessian.

<div class="read" markdown="1">
**Three checks before you trust a Markov-switching VAR**

1. **Are the variables stationary?** Regimes should describe shifts in means and volatility of stationary variables.
2. **How many regimes, and what switches?** A model that switches too much can create a regime out of a few outliers. Compare the information criteria and look at the *size* of every regime (Step 7).
3. **Is it better than a linear VAR?** The usual \(\chi^2\) test is not valid; use the information criteria and the Davies bound.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We estimate a Markov-switching VAR (Krolzig 1997) with two lags in GDP growth and inflation. The regime, which is unobserved, follows a first-order Markov chain with two states. The intercepts and the covariance matrix of the innovations are regime-specific (MSIH); the autoregressive coefficients are common to both regimes. We estimate the model by maximum likelihood with the EM algorithm from several starting values, using the Hamilton filter and the Kim smoother for the regime probabilities, and take standard errors from the numerical Hessian. We choose the number of regimes with AIC and BIC and compare with a linear VAR using the likelihood-ratio statistic and its Davies bound (Davies 1987), because the standard asymptotic distribution does not hold when the transition probabilities are unidentified under the null.
</div>

## Load the data and test for unit roots

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

data = open("us_macro.csv").read()
df = pd.read_csv("us_macro.csv")
print(df[["gdp_growth", "inflation"]].describe().round(2).loc[["count", "mean", "std", "min", "max"]])
spec = dict(vars=["gdp_growth", "inflation"], time="date", lags=2, regimes=2, switching="MSIH", horizon=20)
r = oes.run("msvar", data, spec)
oes.show(r, "Unit-root")
```

<div class="read" markdown="1">
**How to read it**

GDP growth rejects a unit root at any usual level; inflation rejects it at 5% with a constant (p = 0.025), is borderline with a trend (p = 0.07), and the KPSS test does not reject stationarity at 5% (p = 0.077). Both are treated as stationary, so the VAR can be estimated in levels of these growth rates.
</div>

## How many regimes? Better than a linear VAR?

```python
oes.show(r, "Number of regimes")
oes.show(r, "Model comparison")
```

<div class="read" markdown="1">
**How to read it**

The two-regime model fits much better than a linear VAR: the log-likelihood is \(-910.8\) against \(-968.8\) (LR statistic 115.9, Davies p < 0.001), and AIC (1861.7 against 1963.6) and BIC (1927.6 against 2006.5) both prefer two regimes. Two against three regimes: AIC prefers three (1853.6), BIC prefers two (1927.6 against 1949.3). We use two, because the BIC prefers it and the third regime is hard to interpret (Step 8).
</div>

## Estimate

```python
oes.show(r, "Regime-dependent intercepts")
oes.show(r, "Implied regime means")
oes.show(r, "Transition")
oes.show(r, "Regimes")
oes.show(r, "Autoregressive")
oes.show(r, "Residual covariance, regime 0")
oes.show(r, "Residual covariance, regime 1")
oes.show(r, "Regime classification")
```

<div class="read" markdown="1">
**How to read it**

Regime 0 is the turbulent regime: 39.5% of quarters, 10.4 quarters on average (\(p_{00} = 0.90\), standard error 0.04); implied mean GDP growth 1.9% and inflation 5.2%. Regime 1 is the calm regime: 60.5% of quarters, 16.1 quarters on average (\(p_{11} = 0.94\), standard error 0.03); mean GDP growth 3.8% and inflation 3.3%. The covariance matrices differ most: the standard deviation of the GDP-growth innovation is 4.2 in regime 0 and 2.2 in regime 1, and that of inflation is 3.5 against 1.0. The correlation of the two innovations is 0.18 in regime 0 and about zero in regime 1. All four intercepts are significant. The autoregressive coefficients are common: inflation is persistent (0.38 and 0.37 on its two lags), GDP growth depends on its own lags (0.20 and 0.17) and, weakly, inflation affects growth negatively (\(-0.17\) at lag 2 but not significant). The RCM is 19.5, a fairly sharp classification.
</div>

## Regime dates

```python
oes.show(r, "Regime dating")
oes.plot(r, "Smoothed regime probabilities", ncols=1, size=(6.5, 3))
oes.plot(r, "inflation and the regime-weighted mean", ncols=1, size=(6.5, 3))
```

<div class="read" markdown="1">
**How to read it**

The turbulent regime is dated 1972Q4–1983Q2 (43 quarters, average probability 0.96) and 2005Q3–2009Q3 (17 quarters, 0.94), with short episodes in 1959–61, 1966, 1970–71, 1985–86 and 1990–91. The calm regime covers 1991Q2–2005Q2 (57 quarters) and most of the 1960s. The short episodes of two to six quarters have lower and more variable probabilities (for 1966 the average is 0.65): do not read them as separate events. Present the probability figure with the table.
</div>

## Impulse responses by regime

```python
oes.plot(r, "Response of inflation to a shock in gdp_growth", ncols=1, size=(6, 3))
oes.plot(r, "Response of gdp_growth to a shock in inflation", ncols=1, size=(6, 3))
```

<div class="read" markdown="1">
**How to read it**

The shock is one standard deviation of the regime's Cholesky factor, with GDP growth ordered first (so inflation does not affect GDP growth on impact: this is an assumption, state it). In the MSIH-VAR the autoregressive coefficients are common, so the **shape** of the responses is the same in both regimes and the difference is the **size of the shocks**: a one-standard-deviation inflation shock is 3.4 percentage points in the turbulent regime and 1.0 in the calm one. A GDP-growth shock raises inflation by 0.63 on impact and 0.58 a quarter later in regime 0, against 0.03 and 0.19 in regime 1. An inflation shock lowers GDP growth by 0.52 after four quarters in regime 0 and by 0.16 in regime 1. If you want to claim that the *propagation* of shocks differs between regimes, you need the MSIAH model (Step 8). These responses have no confidence bands; do not claim significance for the differences.
</div>

## Diagnostics

```python
oes.show(r, "Residual diagnostics")
```

<div class="read" markdown="1">
**How to read it**

Residuals are acceptable for GDP growth (Ljung-Box p = 0.65, ARCH p = 0.21) and not for inflation: autocorrelation (p = 0.02) and ARCH (p < 0.001). Both reject normality (p < 0.001), which reflects outliers. The multivariate portmanteau test rejects at 5% (p = 0.026). The standard errors are therefore approximate. Residual ARCH in inflation suggests that two variance regimes do not capture all the volatility change; the SV-VAR in the Bayesian section handles this.
</div>

## Robustness: what switches, lags, regimes and starting values

```python
def run_variant(**kw):
    res = oes.run("msvar", data, dict(spec, **kw))
    s = {x["label"]: x["value"] for x in res["summary"]}
    ic = oes.frame(next(t for t in res["tables"] if t["title"].startswith("Number of regimes")))
    k = int(s["Regimes"])
    reg = oes.frame(next(t for t in res["tables"] if t["title"] == "Regimes"))
    return [s["Log-likelihood"], ic.loc[k, "BIC"], " / ".join(f"{v:.0%}" for v in reg["Share of periods (smoothed prob. > 0.5)"]),
            " / ".join(f"{v:.1f}" for v in reg["Expected duration"])]

variants = {"baseline (MSIH, 2 lags)": {}, "MSI: intercepts only": dict(switching="MSI"), "MSIAH: all switch": dict(switching="MSIAH"),
            "1 lag": dict(lags=1), "3 lags": dict(lags=3), "4 lags": dict(lags=4), "3 regimes": dict(regimes=3), "20 starting values": dict(starts=20)}
tab = {k: run_variant(**v) for k, v in variants.items()}
print(pd.DataFrame(tab, index=["log-lik", "BIC", "share of periods", "duration (quarters)"]).T)
```

<div class="read" markdown="1">
**How to read it**

**What switches.** With intercepts only (MSI), the second regime is a 1.4-quarter episode holding 6% of the periods: a collapse onto outliers, and the BIC is 1981.3. Letting all coefficients switch (MSIAH) raises the log-likelihood by only 3.4 and the BIC to 1963.2, against 1927.6 for MSIH, so there is no evidence that the dynamics switch. **Lags.** BIC is lowest with three lags (1912.9; four lags 1916.8, two lags 1927.6, one lag 1959.7; the sample shrinks by one quarter per extra lag, so these are close but not exact comparisons). With three lags the classification agrees with the baseline in 91% of the quarters, but the regimes no longer differ in the level of inflation (implied means 4.12 and 4.08), only in growth (1.9% and 3.8%) and volatility: **the label "high-inflation regime" depends on the lag order**, so describe the regimes by volatility and growth unless the result holds across lags. **Regimes.** Three regimes hold 27%, 28% and 44% of the periods, with expected durations of 7.1, 51.8 and 12.5 quarters. **Starting values.** Twenty starting values give exactly the same log-likelihood as the default, so the solution is not a local maximum.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

We estimate a Markov-switching VAR (Krolzig 1997) with two regimes and two lags in quarterly GDP growth and inflation, 1959Q4–2009Q3, with regime-specific intercepts and covariance matrices (MSIH). It fits much better than a linear VAR (BIC 1927.6 against 2006.5; likelihood-ratio statistic 115.9, Davies p < 0.001). The turbulent regime holds 39.5% of the quarters, lasts 10.4 quarters on average, and has mean GDP growth of 1.9%, mean inflation of 5.2% and innovation standard deviations of 4.2 and 3.5; the calm regime holds 60.5% of the quarters, lasts 16.1 quarters, and has mean growth of 3.8%, mean inflation of 3.3% and innovation standard deviations of 2.2 and 1.0. The turbulent regime is dated 1972Q4–1983Q2 and 2005Q3–2009Q3. Because the autoregressive coefficients are common, impulse responses have the same shape in both regimes and differ in size, since shocks are about three times larger in the turbulent regime; allowing the coefficients to switch does not improve the BIC (1963.2). The result is robust to the starting values (20 starts give the same maximum). It is not robust to the lag order in one respect: with three lags the regimes no longer differ in mean inflation, only in growth and volatility. A model with intercepts only collapses onto a 1.4-quarter regime. Residuals for inflation show autocorrelation and ARCH, so standard errors are approximate.
</div>

| Report | What | Where |
|---|---|---|
| Model | Variables, lags, what switches, number of regimes, estimation and starting values | Step 1 |
| Data | Unit-root tests | Step 2 |
| Regimes | AIC, BIC for one to three regimes; comparison with the linear VAR | Step 3 |
| Estimates | Regime intercepts and means, covariances, transition probabilities, durations | Step 4 |
| Dates | Regime dates, probabilities, figures | Step 5 |
| Responses | Regime-specific impulse responses (Cholesky ordering stated) | Step 6 |
| Diagnostics | Residual tests | Step 7 |
| Robustness | What switches, lags, regimes, starting values, regime sizes | Step 8 |

**References.** Krolzig (1997), *Markov-Switching Vector Autoregressions*, Springer; Hamilton (1989), *Econometrica*; Kim (1994), *Journal of Econometrics*; Sims and Zha (2006), *American Economic Review*; Davies (1987), *Biometrika*; Ang and Bekaert (2002), *Review of Financial Studies*.
