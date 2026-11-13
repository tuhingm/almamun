title: Factor-Augmented VAR (FAVAR)
short: FAVAR
when: A VAR that should use information from many macro series without adding them all as variables
data: U.S. quarterly macro data, 1959–2009: inflation, unemployment and the T-bill rate, with six real-activity series as the information set
tools: Python · OES engine · statsmodels
oes: favar
lead: Use this guide when a small VAR omits information that the policymaker or the economy uses, and you have a wide panel of related series. A FAVAR summarises the panel in a few factors, adds them to the VAR, and traces the effect of a shock on every series in the panel. The guide covers the factor choice, estimation, impulse responses of the whole panel, diagnostics, robustness and the write-up, and is honest about what a small information set can and cannot do.
---

## The question and the model

**Does a small monetary VAR omit information that matters?** A VAR in inflation, unemployment and the interest rate has the price puzzle: inflation rises after a tightening. One explanation is that the central bank sees far more than three series, and the VAR leaves that information in the residual. A **factor-augmented VAR** (Bernanke, Boivin and Eliasz 2005) adds a few unobserved factors \(F_t\) that summarise a large panel \(X_t\) of \(N\) informational series:

\[
X_t = \Lambda^F F_t + \Lambda^Y Y_t + e_t, \qquad
\begin{pmatrix} F_t \\ Y_t \end{pmatrix} = c + A_1 \begin{pmatrix} F_{t-1} \\ Y_{t-1} \end{pmatrix} + \dots + u_t,
\]

where \(Y_t\) are the observed variables (here inflation, unemployment and the policy rate, ordered last). Two things follow from the first equation. The factors enter the VAR, so the policy shock is identified with more information in the system. And every series in \(X_t\) has an impulse response, \(\Lambda' \times\) the VAR response, so you can report the effect of the shock on dozens of series without estimating a dozen VARs.

OES implements the **two-step** estimator: the factors are the principal components of the standardised panel, and the VAR on \((F_t, Y_t)\) is a Bayesian VAR with the Minnesota-type priors of the BVAR guide, with prior mean zero on the factors.

<div class="read" markdown="1">
**What this example is, and is not**

The example panel has only **six** series, all real-activity growth rates from the same national accounts. A real FAVAR uses 50 to 150 series (the McCracken-Ng FRED-QD data are the standard source) with prices, interest rates, employment, financial and monetary variables. With six series the factors can say little that the three observed variables do not already say, and the guide shows this honestly in Step 8. Use this guide for the mechanics and diagnostics, and bring a large panel to your own application. The OES page accepts any number of series.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We estimate a factor-augmented VAR (Bernanke, Boivin and Eliasz 2005). The informational panel, \(N\) quarterly series standardised to mean zero and variance one, is summarised by \(K\) principal components. The factors, together with inflation, unemployment and the Treasury bill rate, enter a VAR(\(p\)) estimated with a Normal-Wishart prior with Minnesota moments (Litterman 1986), with prior mean zero for the factors and for stationary variables and one for the variables with a unit root. The policy shock is the innovation to the Treasury bill rate, identified recursively with the rate ordered last. Responses of the panel series are obtained from their estimated loadings on the factors and the observed variables.
</div>

## Load the data and test for unit roots

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

data = open("us_macro.csv").read()
y = ["inflation", "unemp", "tbilrate"]                      # observed variables, policy rate last
x = ["gdp_growth", "cons_growth", "inv_growth", "gov_growth", "income_growth", "m1_growth"]   # information set
spec = dict(y=y, x=x, time="date", factors=2, lags=4, delta="0, 1, 1")
r = oes.run("favar", data, spec)
oes.summary(r)
oes.show(r, "Unit-root")
```

<div class="read" markdown="1">
**How to read it**

The principal components do not need stationary data, but they behave well when the panel is stationary, which is why informational series are transformed to growth rates or differences. All six series reject a unit root. Of the observed variables, inflation rejects (ADF p = 0.03), but unemployment and the T-bill rate do not, so they get a random-walk prior (`delta = "0, 1, 1"`). The factors are standardised and stationary, so they get zero.
</div>

## Choose the number of factors

```python
oes.show(r, "Number of factors")
```

<div class="read" markdown="1">
**How to read it**

Two tools: the cumulative share of variance, and the **Bai and Ng (2002)** information criterion, which finds the number of factors in a large panel. It needs a large \(N\) and \(T\); with \(N = 6\) it is not informative (it keeps falling up to the maximum of five factors, which is nearly the whole panel). In a real application, \(N \ge 50\) and you would choose with the criterion and a scree plot, then check that the results do not depend on the choice (Step 8). Here two factors explain 61% of the variance of the six series, which is what the model uses.
</div>

## The factors and the VAR

```python
oes.show(r, "Principal-component")
oes.show(r, "Equation: tbilrate")
```

<div class="read" markdown="1">
**How to read it**

The first factor explains 42% of the panel: the common real-activity cycle. The second adds 19%. The interest-rate equation is a policy rule: the rate is persistent and responds to inflation, unemployment and the real-activity factors; the question for a FAVAR is whether the *factors* add to the rule beyond the three observed variables.
</div>

## The policy shock and the whole panel

```python
oes.plot(r, "Response of inflation to a shock in tbilrate")
oes.plot(r, "Response of unemp to a shock in tbilrate")
oes.show(r, "Responses of the informational")
oes.plot(r, "Response of gdp_growth to a shock in tbilrate")
oes.plot(r, "Response of inv_growth to a shock in tbilrate")
```

<div class="read" markdown="1">
**How to read it**

Observed variables first. A one-standard-deviation rise in the T-bill rate (about 0.6 percentage point) raises unemployment gradually, by 0.10 percentage point after two years and 0.14 after three, with the 68% band above zero from the second year. This is the expected sign and size. **Inflation, however, rises** on impact, by 0.28 percentage point, with a band that excludes zero for more than a year: the price puzzle is still there. This is a result, not an error: six real-activity series carry no information about prices that the VAR does not already have. Step 8 shows that the response is the same with and without the factors.

The advantage of a FAVAR shows in the table and the last two charts: the same shock traced through all six panel series, with no extra VARs. Output growth falls by about 0.15 percentage point after one year (68% band from \(-0.23\) to \(-0.09\)) and returns to zero after about three years; consumption and income growth fall by 0.12 and 0.09; investment growth falls by 0.58, four times more than output, with a wide band, as in most monetary VARs. Government spending growth hardly moves. The \(R^2\) column tells you how well the factors and observed variables explain each series (0.91 for GDP growth, 0.44 for income growth): the response of a series with a low \(R^2\) is mostly about the common component, so treat it with care.
</div>

## Variance decomposition

```python
oes.show(r, "Forecast-error variance")
```

<div class="read" markdown="1">
**How to read it**

Rows are the variables, columns the shocks (Cholesky). The policy shock explains 9% of the forecast-error variance of inflation after five years, 14% of unemployment and 2% of the real-activity factor; the interest rate's own shock accounts for 53% of its variance. The first factor, the real cycle, is the main driver of unemployment (47% to 61%). The policy shock is a modest source of fluctuations, the standard result; policy matters through the rule, not through surprises.
</div>

## Diagnostics

```python
oes.show(r, "Residual diagnostics")
```

<div class="read" markdown="1">
**How to read it**

The factors are fairly well behaved (Ljung-Box does not reject at 5% for either factor), but the observed variables have serial correlation in the residuals (Ljung-Box p below 0.002), non-normal residuals and ARCH. The interest rate is the extreme case (Jarque-Bera 1875): the largest quarterly changes in the T-bill rate, up to 5.9 percentage points, are all in 1980 to 1982, the Volcker disinflation. Adding more lags helps only a little (the portmanteau rejects with two, four and six lags, p = 0.0002 with six), and the joint test has 300 degrees of freedom with 198 observations, so it over-rejects in samples this short; rely on the individual tests. The practical answer is the stochastic-volatility VAR (see the SV-VAR guide), which allows the shock variance to change; in the paper, state the limitation and show that the main results do not depend on the lag length (Step 8).
</div>

## Robustness: factors, lags and the plain VAR

```python
def irf(res, title, h=(1, 4, 8, 12)):
    c = next(c for c in res["charts"] if c["title"] == title)["series"][0]["y"]
    return [round(c[i], 3) for i in h]

title = "Response of inflation to a shock in tbilrate"
rows = {}
for k in (1, 2, 3, 4):
    rows[f"{k} factor(s), 4 lags"] = irf(oes.run("favar", data, dict(spec, factors=k)), title)
for p in (2, 6):
    rows[f"2 factors, {p} lags"] = irf(oes.run("favar", data, dict(spec, lags=p)), title)
rb = oes.run("bvar", data, dict(vars=y, time="date", lags=4, prior="nw", lambda1=0.1, delta="0, 1, 1"))
rows["no factors (BVAR)"] = irf(rb, title)
print(pd.DataFrame(rows, index=["h=1", "h=4", "h=8", "h=12"]).T)
```

<div class="read" markdown="1">
**How to read it**

The response of inflation to a policy shock, at 1, 4, 8 and 12 quarters, under different choices. It is stable across the number of factors (0.26 to 0.28 at one quarter, 0.16 to 0.17 at three years) and lags, and the same as the BVAR without factors (0.28, 0.29, 0.24, 0.18). Shorter or longer lag lengths move the response at one year by about 0.04, never the sign. Two conclusions for a paper: (1) the FAVAR results are robust to \(K\) and \(p\); (2) with this *small* information set the factors do not change the answer. When the factors do not matter, say so; in a real application with 100 series, this comparison is the main test that the FAVAR is worth the extra complexity.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

Two factors, which explain 61% of the variance of the six informational series, are added to a VAR(4) in inflation, unemployment and the Treasury bill rate. A one-standard-deviation shock to the policy rate (about 0.6 percentage point) raises unemployment gradually, by 0.10 percentage point after two years, with a 68% credible band above zero from the second year. Output growth falls by 0.15 percentage point after one year and returns to its baseline after about three years; investment growth falls by 0.58, with a wide band. Inflation rises by 0.28 percentage point on impact and stays above zero for about three years, a price puzzle that is unchanged when the number of factors varies between one and four or the lags between two and six, and that is the same as in the VAR without factors. We conclude that the six real-activity series add no information about prices; a larger panel with price and commodity series is needed to test whether omitted information explains the puzzle. Policy shocks explain 9% of the forecast-error variance of inflation and 14% of that of unemployment after five years.
</div>

| Report | What | Where |
|---|---|---|
| Data | Panel (source, \(N\), transformations), observed variables, sample | Step 2 |
| Factors | Number and how chosen (Bai-Ng, variance share), \(R^2\) of the loadings | Steps 3 and 5 |
| Prior | Family, \(\lambda_1\), prior means of the own first lags | Steps 1 and 2 |
| Identification | Ordering of the observed variables, the policy rate last | Step 5 |
| Results | Responses of the observed variables and of selected panel series with 68% bands, variance decomposition | Steps 5 and 6 |
| Diagnostics | Residual tests | Step 7 |
| Robustness | Number of factors, lags, comparison with the VAR without factors | Step 8 |

**References.** Bernanke, Boivin and Eliasz (2005), *Quarterly Journal of Economics*; Stock and Watson (2002), *Journal of the American Statistical Association*; Bai and Ng (2002), *Econometrica*; Sims (1992), *European Economic Review*; Litterman (1986), *Journal of Business and Economic Statistics*; McCracken and Ng (2020), *FRED-QD: A Quarterly Database for Macroeconomic Research*.
