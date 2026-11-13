title: Smooth-Transition Regression (LSTAR and ESTAR)
short: Smooth transition (STR)
when: The coefficients depend on the state of the economy, but they change gradually, not abruptly, as a variable moves through a range
data: U.S. quarterly data, 1960–2019: the federal funds rate, inflation and the output gap
tools: Python · OES engine · statsmodels · SciPy
oes: str
lead: Use this guide when behaviour depends on a state variable and you expect a gradual move between regimes: a central bank that reacts to inflation more strongly in a boom, a consumer who responds differently to income in good and bad times, an exchange rate that reverts only when far from parity. A smooth-transition regression (Teräsvirta's LSTAR and ESTAR) lets the coefficients move with a transition function of that variable, estimates the location and speed of the move, tests linearity, and chooses between the two forms. The guide covers the model, the linearity tests, the choice of the transition variable and delay, estimation, the regimes, robustness, and the text for the paper.
---

## The question and the model

**Do the coefficients depend on the level of a state variable, and how fast do they move?** In a smooth-transition regression

\[
y_t = x_t'\phi + (x_t'\theta)\, G(s_t;\gamma,c) + e_t ,
\]

the transition function \(G\) rises from 0 to 1 as the transition variable \(s_t\) moves through a range. In the logistic form (LSTAR), \(G = 1/\{1+\exp[-\gamma (s_t-c)/\sigma_s]\}\): the regimes at low and high \(s_t\) differ, and \(c\) is where they are halfway between. In the exponential form (ESTAR), \(G = 1-\exp[-\gamma (s_t-c)^2/\sigma_s^2]\): the two outer regimes are alike and differ from the middle. The slope parameter \(\gamma\) (scaled by the standard deviation of \(s_t\)) sets the speed. As \(\gamma\to\infty\) the logistic form becomes the threshold model of the [threshold regression guide](tar.html).

A linear model is nested in two ways, \(\theta=0\) or \(\gamma=0\), so the usual tests do not apply. Teräsvirta (1994) replaces \(G\) by a third-order Taylor approximation, which turns the linearity test into an F test, and the pattern of rejections chooses between LSTAR and ESTAR. Given \(\gamma\) and \(c\), the model is linear in the other coefficients, so estimation concentrates the sum of squares on \((\gamma, c)\), searches a grid and polishes with a simplex search.

<div class="read" markdown="1">
**Four checks before you trust a smooth-transition model**

1. **Is linearity rejected for the transition variable you use?** If not, the nonlinear estimates are not determined. Try the candidate variables and delays and report the search.
2. **Is \(\gamma\) estimated away from its limits?** A very large \(\gamma\) means an abrupt switch, and its standard error is unreliable because the likelihood is almost flat. A very small \(\gamma\) makes the model a polynomial in \(s_t\), with no meaningful location.
3. **Does the location \(c\) lie inside the data and away from the edge of its range?** It is searched between the 10th and 90th percentiles of \(s_t\), so that both regimes have observations.
4. **Are the residuals clean?** Remaining serial correlation means the dynamics are too short. Add lags before interpreting the regimes.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We estimate a nonlinear monetary policy rule with a smooth-transition regression (Teräsvirta 1994). The federal funds rate depends on its own lag, inflation and the output gap, and all coefficients, including the intercept, move with a logistic function of the output gap lagged three quarters. The transition function has a speed \(\gamma\) and a location \(c\). We test linearity with Teräsvirta's third-order Taylor approximation for every candidate transition variable and delay, choose the logistic or the exponential form by the sequence of tests, estimate \(\gamma\) and \(c\) by nonlinear least squares on 237 quarterly observations from 1960 to 2019, and report Newey–West standard errors.
</div>

## Load the data and look at the series

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine
from statsmodels.tsa.stattools import adfuller

data = open("taylor.csv").read()
df = pd.read_csv("taylor.csv")
print(df.date.iloc[0], "to", df.date.iloc[-1], "|", len(df), "quarters")
print(df[["fedfunds", "inflation", "output_gap"]].describe().round(2).loc[["mean", "std", "min", "max"]])
print(df[["fedfunds", "inflation", "output_gap"]].corr().round(2))
print(f"ADF p, federal funds rate = {adfuller(df.fedfunds, autolag='AIC')[1]:.3f}")
```

<div class="read" markdown="1">
**How to read it**

The sample runs from 1960Q1 to 2019Q4 (240 quarters). The federal funds rate averages 4.99% with a standard deviation of 3.65 and ranges from 0.07% to 17.78%. Inflation averages 3.20% (standard deviation 2.19). The output gap averages about zero (standard deviation 1.43) and runs from −4.80 to 3.72 points, so large positive gaps are rare. The funds rate is correlated with inflation (0.76) and weakly with the output gap (0.20), and it is persistent (ADF p = 0.33), so a lag of the funds rate is included to capture interest-rate smoothing.
</div>

## Test linearity and choose the transition variable and the delay

```python
spec = dict(y="fedfunds", x=["inflation", "output_gap"], time="date", ar_lags=1, form="auto", trim="0.10")
rows = []
for sv in ["output_gap", "inflation"]:
    for d in [1, 2, 3, 4]:
        rr = oes.run("str", data, dict(spec, transition=sv, delay=d))
        lin = [t for t in rr["tables"] if t["title"].startswith("Linearity")][0]["rows"][0]
        s = {x["label"]: x["value"] for x in rr["summary"]}
        rows.append((sv, d, lin[1], lin[2], s["Transition function"], s["gamma"], s["c"], s["AIC gain over linear"]))
print(pd.DataFrame(rows, columns=["transition", "delay", "F", "p", "form", "gamma", "c", "AIC gain"]).to_string(index=False))
```

<div class="read" markdown="1">
**How to read it**

Linearity is rejected at 5% for all eight combinations of transition variable and delay, and at 1% for seven. Two things guide the choice. First, the quality of the fit: with the output gap lagged three quarters the AIC gain over the linear model is 57.4, against 11 to 18 for the other output-gap delays and 6 to 16 for inflation. Second, whether the transition is identified. With inflation as the transition variable the slope \(\gamma\) sits at its lower limit of 0.1 for a delay of one quarter (so the model is close to a polynomial in inflation, with a location of 6.5% that carries no meaning) and at its upper limit of 500 for a delay of four. With the output gap at a delay of three quarters, \(\gamma\) is 10.0 and the location is 1.47 points, inside the range of the data. We use the output gap lagged three quarters. The search is exploratory, and the p-values of the linearity tests do not account for it; the size of the AIC gain is the more informative number.
</div>

## Estimate the model

```python
spec = dict(y="fedfunds", x=["inflation", "output_gap"], transition="output_gap", time="date", ar_lags=1, delay=3, form="auto", trim="0.10")
r = oes.run("str", data, spec)
oes.summary(r)
oes.show(r, "Linearity tests")
oes.show(r, "Transition function")
```

<div class="read" markdown="1">
**How to read it**

The linearity test rejects strongly: F = 4.73 for the 12 restrictions, p < 0.0001. In the sequence that chooses the form, the first-order test (H02) has the smallest p-value (below 0.0001, against 0.011 for H03 and 0.018 for H04), so the rule chooses the logistic form. The speed \(\gamma\) is 10.0 (standard error 7.5, so it is not precisely estimated, as expected when the transition is quick) and the location is \(c = 1.47\) output-gap points (standard error 0.16, t = 9.4). A \(\gamma\) of 10 means that the transition function rises from about 0.12 to 0.88 as the lagged output gap moves through about half a point around 1.47, which is roughly one standard deviation above the mean gap. So the switch is quick but not a step.
</div>

## The coefficients in the two regimes

```python
oes.show(r, "Coefficients")
imp = [t for t in r["tables"] if t["title"].startswith("Coefficients in the two")][0]["rows"]
val = {row[0]: (float(row[1].split()[0]), float(row[2].split()[0])) for row in imp}
for k, name in enumerate(["low output gap (G = 0)", "high output gap (G = 1)"]):
    print(f"Long-run response of the funds rate to inflation, {name}: {val['inflation'][k] / (1 - val['fedfunds(-1)'][k]):.2f}")
```

<div class="read" markdown="1">
**How to read it**

The linear part is the regime when the lagged output gap is low (G = 0) and the nonlinear part is what high gaps add. When the gap was low, the funds rate adjusts slowly (coefficient on its lag 0.963, standard error 0.021), does not respond to inflation (0.025, 0.041) and responds to the output gap with 0.268 (0.031). When the gap was high (G = 1), the lag coefficient falls to 0.381 (0.261), the response to inflation rises to 0.928 (0.376) and the response to the gap is 0.303 (0.137). The changes in the lag and inflation coefficients are significant at 5% (p = 0.030 and 0.017). The changes in the intercept and the output-gap coefficient are not (p = 0.52 and 0.81). The long-run response of the funds rate to inflation, the inflation coefficient divided by one minus the lag coefficient, is 0.68 in the low regime and 1.50 in the high regime. The second is above 1, the Taylor principle, so the rule is stabilising only after a boom. The first should be read with care, because it divides a small coefficient by a number near zero. The high regime rests on few observations, so its standard errors are wide.
</div>

## Fit, residuals and the transition over time

```python
oes.show(r, "Fit against the linear model")
oes.show(r, "Residual diagnostics")
oes.plot(r, "Transition function against", ncols=1, size=(6.5, 3))
oes.plot(r, "Transition function over time", ncols=1, size=(6.5, 3))
```

<div class="read" markdown="1">
**How to read it**

The smooth-transition model lowers the sum of squared residuals from 144.3 to 107.7 and the residual standard deviation from 0.787 to 0.689. AIC falls from −109.6 to −167.0 and BIC from −95.7 to −132.4, so both criteria prefer it even after counting \(\gamma\) and \(c\). The Ljung–Box test rejects no serial correlation (statistic 15.7, p = 0.003), so the dynamics are not fully captured with one lag; the AR(2) variant below gives a similar \(\gamma\) and location, so the main result does not depend on it, but the standard errors should be treated with care. The first chart shows the transition function: it is zero for output gaps below about 0.9, rises between about 1.1 and 2.5 and is one above that. The second chart shows it over time. It is at or near zero most of the time and rises only in booms: the mid-1960s, the early 1970s, around 1979–1980, around 1990, in 2000 and in 2006–2007. It is zero throughout the 2010s.
</div>

## Does the answer depend on the form, the lags, the delay and the range?

```python
out = {}
variants = {"baseline (LSTAR, AR(1), delay 3)": {}, "ESTAR": dict(form="exponential"), "AR(2)": dict(ar_lags=2),
            "delay 2": dict(delay=2), "delay 4": dict(delay=4), "range for c: 5th to 95th": dict(trim="0.05"),
            "range for c: 15th to 85th": dict(trim="0.15"), "range for c: 20th to 80th": dict(trim="0.20")}
for name, v in variants.items():
    rr = oes.run("str", data, dict(spec, **v))
    s = {x["label"]: x["value"] for x in rr["summary"]}
    out[name] = (s["Transition function"], s["gamma"], s["c"], s["c at the edge of its range"], s["R-squared"], s["AIC gain over linear"])
print(pd.DataFrame(out, index=["form", "gamma", "c", "c at edge", "R-squared", "AIC gain"]).T.to_string())
```

<div class="read" markdown="1">
**How to read it**

The baseline results hold across the variants that are well behaved. With two lags of the funds rate the speed is 10.5, the location is 1.50 and the AIC gain is 57.1. Widening the range for \(c\) to the 5th to 95th percentiles changes nothing, because the baseline location (1.47) is inside the range. The variants that change the answer are flagged in the table. A narrower range for \(c\) (15th to 85th and 20th to 80th) puts the location on the edge of the range (1.45 and 1.14), which shows the range is cutting off the best location. The exponential form, imposed by hand, also ends on the edge of its range (1.65) and its AIC gain is 32.4, well below the logistic 57.4. A delay of four puts \(c\) on the edge with \(\gamma\) above 100, and a delay of two gives a slower transition (\(\gamma = 3.0\)) but an AIC gain of only 17.0. The delay of three quarters, the logistic form and a location of about 1.5 points are what the data support.
</div>

## Smooth or abrupt? Compare with the threshold regression

```python
th = oes.run("tar", data, dict(y="fedfunds", x=["inflation", "output_gap"], threshold="output_gap", time="date",
                               ar_lags=1, delay=3, boot=499, max_thresholds="1"))
oes.show(th, "Tests for a threshold")
oes.show(th, "Estimated thresholds")
oes.show(th, "Fit against the linear model")
```

<div class="read" markdown="1">
**How to read it**

The threshold regression with the same specification and the same delay rejects linearity (sup-F = 72.9, bootstrap p = 0.018 with 499 draws, 5% critical value 59.0). It places the threshold at 1.38 (95% interval 1.22 to 1.43), with 38 of 237 observations above it. Its fit is almost the same as the smooth model: a sum of squared residuals of 109.4 against 107.7, and an AIC of −165.1 against −167.0. The data therefore cannot separate an abrupt switch from a quick smooth one, which agrees with the wide standard error of \(\gamma\). Both place the switch at an output gap of about 1.4 to 1.5 points. Choose the threshold model if you want a simple statement (a rule that changes at a level), and the smooth-transition model if you want to show how the response changes through the range.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

We find that the response of U.S. monetary policy to inflation depends on the state of the cycle. The federal funds rate is modelled with a logistic smooth-transition regression on its own lag, inflation and the output gap, with all coefficients moving with the output gap lagged three quarters. Teräsvirta's linearity test rejects strongly (F = 4.73, p < 0.0001) and the sequence of tests selects the logistic form. The transition is centred at an output gap of 1.47 points (standard error 0.16) with a speed of 10.0 (standard error 7.5). When the lagged gap is low, the rate adjusts slowly (coefficient on its lag 0.96) and does not respond to inflation (0.03). When the gap is high, adjustment is faster (0.38) and the coefficient on inflation is 0.93 (standard error 0.38), a long-run response of 1.5, above the Taylor-principle threshold of 1. The model lowers AIC by 57 points and BIC by 37 points relative to the linear rule. Residual serial correlation remains (Ljung–Box p = 0.003), and the high regime contains few observations, so these estimates are imprecise. The results are similar with two lags and for a wider range of the location, and a threshold regression with the same specification gives a similar fit and a threshold of 1.38.
</div>

| Report | What | Where |
|---|---|---|
| Model | Regressors, transition variable, delay, function, range for \(c\) | Step 1 |
| Data | Scale and persistence | Step 2 |
| Search | Linearity tests for each candidate variable and delay | Step 3 |
| Estimates | Linearity tests, \(\gamma\) and \(c\) with standard errors | Step 4 |
| Regimes | Coefficients at G = 0 and G = 1, long-run response | Step 5 |
| Fit | AIC, BIC, residual test, the transition function | Step 6 |
| Robustness | Form, lags, delay, range | Step 7 |
| Comparison | Threshold regression | Step 8 |

**References.** Teräsvirta, T. (1994), "Specification, Estimation, and Evaluation of Smooth Transition Autoregressive Models," *Journal of the American Statistical Association* 89. Granger, C. W. J. and Teräsvirta, T. (1993), *Modelling Nonlinear Economic Relationships*, Oxford University Press. van Dijk, D., Teräsvirta, T. and Franses, P. H. (2002), "Smooth Transition Autoregressive Models: A Survey of Recent Developments," *Econometric Reviews* 21. Hansen, B. E. (2000), "Sample Splitting and Threshold Estimation," *Econometrica* 68.
