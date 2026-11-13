title: Markov-Switching Autoregression (Hamilton)
short: MS autoregression
when: A single series whose mean shifts between expansion-like and recession-like states that you do not observe
data: U.S. quarterly GNP growth, 1951Q2–1984Q4 (Hamilton 1989)
tools: Python · OES engine · statsmodels
oes: ms_ar
lead: Use this guide when you have one time series, such as GDP growth, that seems to move between a fast-growth state and a slow or negative-growth state, and you want the data to date the states and tell you how long they last. It is Hamilton's (1989) model. The guide covers the unit-root check, choosing the number of regimes, the linearity test, estimation, regime dates and a comparison with NBER recession dates, diagnostics, robustness and the write-up.
---

## The question and the model

**Are there recessions in the data, and can the data find them without being told?** In a Markov-switching autoregression (Hamilton 1989) the mean of the series depends on an unobserved state \(s_t\):

\[
y_t - \mu_{s_t} = \phi_1 (y_{t-1} - \mu_{s_{t-1}}) + \dots + \phi_p (y_{t-p} - \mu_{s_{t-p}}) + \varepsilon_t, \qquad \varepsilon_t \sim N(0, \sigma^2),
\]

and the state follows a first-order Markov chain with transition probabilities \(p_{ij} = \Pr(s_t = j \mid s_{t-1} = i)\). The model is estimated by maximum likelihood with the Hamilton filter. For each period it gives a *filtered* probability (using data up to \(t\)) and a *smoothed* probability (using the whole sample) of being in each state.

<div class="read" markdown="1">
**Three checks before you trust a regime model**

1. **Is the series stationary?** The regimes should describe shifts in the mean of a stationary series. A trending series produces "regimes" that are just the trend.
2. **Does the data need two regimes?** Compare one, two and three regimes with information criteria, and run the linearity test. A regime model fits better than a linear one by construction; the question is whether it fits *enough* better.
3. **Are the regimes sharply identified?** A good model puts each period in one regime with high probability. The regime classification measure (RCM, Ang and Bekaert 2002) summarises this.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We estimate Hamilton's (1989) Markov-switching autoregression for quarterly GNP growth, with a state-dependent mean, a common variance and four autoregressive lags. The state follows a two-state first-order Markov chain. We estimate the model by maximum likelihood using the Hamilton filter and report smoothed state probabilities (Kim 1994). To compare the number of regimes we use AIC and BIC, and we test linearity with the likelihood-ratio statistic and the critical values of Garcia (1998) and the bound of Davies (1987), because the standard asymptotic distribution does not apply when parameters are unidentified under the null.
</div>

## Load the data and test for unit roots

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

data = open("hamilton_gnp.csv").read()
df = pd.read_csv("hamilton_gnp.csv")
print(df.describe().round(2).loc[["count", "mean", "std", "min", "max"]])
spec = dict(y="gnp_growth", time="date", order=4, regimes=2, reference="nber_recession")
r = oes.run("ms_ar", data, spec)
oes.show(r, "Unit-root")
```

<div class="read" markdown="1">
**How to read it**

Quarterly GNP growth is stationary: the ADF tests reject a unit root and the KPSS test does not reject stationarity. That is what a regime-in-the-mean model needs. If your series is in levels and trending, difference it (growth rates) first.
</div>

## How many regimes? Is the series linear?

```python
oes.show(r, "Number of regimes")
oes.show(r, "Linearity")
```

<div class="read" markdown="1">
**How to read it**

One regime has the lowest AIC (379.3) and BIC (396.6); two regimes give 380.5 and 406.4, three give 386.4 and 426.6. The information criteria do **not** prefer a switching model for this sample. The linearity test agrees: the statistic is 4.8, far below the Garcia 5% critical value of about 14, and the Davies bound gives p = 0.95. So the data **do not reject** a linear AR model. This is an honest and well-known result for Hamilton's sample: with 131 quarters, the evidence for regimes is weak when judged by the likelihood alone. The regimes are still useful as a description, because they line up with the NBER dates (Step 5), but you should not claim that the test proves two regimes. Report this table.
</div>

## Estimate

```python
oes.show(r, "Parameter estimates")
oes.show(r, "Transition")
oes.show(r, "Regimes")
oes.show(r, "Regime classification")
```

<div class="read" markdown="1">
**How to read it**

Regime 1 is the high-growth state: the mean growth is 1.15% per quarter (about 4.6% a year) and it lasts 10.4 quarters on average. Regime 0 is the low-growth state: the mean is \(-0.42\)% per quarter, it lasts 4.1 quarters on average, and the state occupies 27.5% of the sample. The transition probabilities are \(p_{00} = 0.75\) and \(p_{11} = 0.90\); the expected duration of a regime is \(1/(1-p_{ii})\). The low-growth mean is imprecise (\(z = -1.4\), p = 0.18) while the high-growth mean is very precisely estimated (\(z = 15.6\)): the recession state has few observations. The estimate \(p_{10} = 0.096\) is significant (p = 0.011), so a move from expansion to the low state is likely about once in ten quarters. Only the third lag of growth is significant at 5% (\(-0.25\)). The RCM is 21.7, below the usual threshold of 30, so the periods are assigned to regimes fairly sharply.
</div>

## Regime dates and the NBER recession dates

```python
oes.show(r, "Regime dating")
oes.show(r, "Comparison")
oes.plot(r, "Smoothed regime probabilities", ncols=1, size=(6.5, 3))
oes.plot(r, "gnp_growth and its regime-dependent fitted mean", ncols=1, size=(6.5, 3))
```

<div class="read" markdown="1">
**How to read it**

The low-growth regime is dated 1953Q3–1954Q2, 1957Q1–1958Q1, 1960Q2–1960Q4, 1969Q3–1970Q4, 1974Q1–1975Q1, 1979Q2–1980Q3 and 1981Q2–1982Q4, mostly with an average probability above 0.9. These are the post-war US recessions. Against the NBER recession indicator, 91.6% of quarters are classified alike, 96.3% of NBER recession quarters are detected, and 9.6% of non-recession quarters are flagged as low growth (false signals). The model finds the recessions without being told where they are. The dated episodes also include 1969Q3–1970Q4 and 1979Q2–1980Q3, which overlap the NBER recessions of 1969–70 and 1980. When you write this up, say which dates you compare with and how you define "detected" and "false signal".
</div>

## Diagnostics

```python
oes.show(r, "Residual diagnostics")
```

<div class="read" markdown="1">
**How to read it**

The residuals pass: no autocorrelation (Ljung-Box p = 0.40; Hosking portmanteau p = 0.42), no sign of non-normality (Jarque-Bera p = 0.19) and no ARCH effects (p = 0.85). The model that allows only the mean to switch is adequate for the series' dynamics. If the ARCH test rejected, you would let the variance switch too (Step 7).
</div>

## Robustness: lags, switching variance, switching dynamics and regimes

```python
def summary(res):
    s = {x["label"]: x["value"] for x in res["summary"]}
    t = [t for t in res["tables"] if t["title"].startswith("Comparison")]
    c = oes.frame(t[0])["Value"].tolist() if t else [np.nan] * 3
    return [s["Log-likelihood"], s["BIC"]] + c

variants = {"baseline": {}, "AR(2)": dict(order=2), "AR(1)": dict(order=1),
            "switching variance": dict(switching_variance="yes"),
            "switching AR coefficients": dict(switching_ar="yes"), "3 regimes": dict(regimes=3)}
tab = {k: summary(oes.run("ms_ar", data, dict(spec, **v))) for k, v in variants.items()}
print(pd.DataFrame(tab, index=["log-lik", "BIC", "concordance", "recessions detected", "false signals"]).T.round(3))
```

<div class="read" markdown="1">
**How to read it**

**The lag order matters a great deal for the dating.** With four lags (Hamilton's choice) the model detects 96% of NBER recession quarters; with two lags it detects 56% and with one lag 52%, though concordance stays above 90% because most quarters are expansions. The shorter models put the low state on fewer quarters, so a claim like "the model finds the recessions" holds for the four-lag specification only. Allowing the variance to switch keeps the detection rate (96%) but raises false signals to 17% and worsens the BIC (410.1 against 406.4). Allowing the autoregressive coefficients to switch raises the log-likelihood to \(-176.3\) but the BIC is 415.9 and only 67% of recession quarters are detected. A third regime improves the log-likelihood to \(-177.0\) but the BIC is 422.3 and one of the three regimes is a single quarter. The baseline is the best on the information criteria among the switching models.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

The two-regime model separates a high-growth state (mean 1.15% per quarter, expected duration 10.4 quarters) from a low-growth state (mean \(-0.42\)% per quarter, expected duration 4.1 quarters). The transition probabilities are 0.75 for staying in the low state and 0.90 for staying in the high state. The smoothed probabilities date seven low-growth episodes between 1953 and 1982 that correspond to the post-war US recessions: 91.6% of quarters are classified in agreement with the NBER indicator, 96.3% of NBER recession quarters are detected and 9.6% of other quarters are flagged. The regime classification measure is 21.7, which indicates a sharp classification. However, the likelihood evidence for two regimes is weak: one regime has the lowest AIC and BIC, and the linearity statistic (4.8) is below the Garcia 5% critical value (about 14; Davies p = 0.95). The regime estimates are therefore best read as a description of the cycle that agrees with the NBER dates, not as a rejection of a linear model. The dating depends on the lag order: with two or fewer lags only about half of the NBER recession quarters are detected.
</div>

| Report | What | Where |
|---|---|---|
| Model | Series, lags, regimes, what switches, estimation method | Step 1 |
| Data | Unit-root tests | Step 2 |
| Regimes | AIC, BIC, linearity statistic and its critical value | Step 3 |
| Estimates | Regime means, transition probabilities, durations, RCM | Step 4 |
| Dates | Regime dates, comparison with NBER dates, figures | Step 5 |
| Diagnostics | Residual tests | Step 6 |
| Robustness | Lags, switching variance and dynamics, number of regimes | Step 7 |

**References.** Hamilton (1989), *Econometrica*; Kim (1994), *Journal of Econometrics*; Garcia (1998), *International Economic Review*; Davies (1987), *Biometrika*; Hansen (1992), *Journal of Applied Econometrics*; Ang and Bekaert (2002), *Review of Financial Studies*; Hamilton (2016), *Handbook of Macroeconomics*.
