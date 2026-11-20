title: Value at Risk and Expected Shortfall
short: Value at risk
when: You need to say how much a portfolio can lose on a bad day, compare risk models, and show a regulator or a reader that the model was right
data: Daily returns of the CRSP value-weighted market index, 1989-1998 (Ecdat); 2,528 trading days
tools: Python · OES engine · NumPy
oes: var_es
lead: Use this guide to forecast the daily loss that is exceeded only 1% of the time (value at risk), the average loss when it is exceeded (expected shortfall), and to test whether the forecasts were right. Five models are compared out of sample: historical simulation, a rolling normal, the RiskMetrics EWMA, a GARCH(1,1) with t errors and a GARCH filtered historical simulation. The guide covers the definitions, the forecasts, the backtests of Kupiec and Christoffersen, the expected shortfall check, a different tail level and window, a simulation and the write-up.
---

## The question and the model

**How bad can a bad day be?** The value at risk at level \(p\) is the return quantile \(\text{VaR}_t\) with \(\Pr(r_t < \text{VaR}_t \mid \mathcal F_{t-1}) = p\), and the expected shortfall is the mean return below it:

\[
\text{ES}_t = E\left[r_t \mid r_t < \text{VaR}_t\right].
\]

Each model forecasts the conditional distribution of tomorrow's return:

| Model | Forecast of the distribution |
|---|---|
| **Historical simulation** | The empirical distribution of the last \(w\) returns |
| **Normal, rolling window** | Normal with the mean and standard deviation of the last \(w\) returns |
| **EWMA** | Normal with variance \(\sigma_t^2 = \lambda\sigma_{t-1}^2 + (1 - \lambda)r_{t-1}^2\) (RiskMetrics, \(\lambda = 0.94\)) |
| **GARCH(1,1), t errors** | Variance from a GARCH(1,1) fitted on the first window, and a t distribution for the standardised residuals |
| **GARCH filtered historical simulation** | The same variance, and the empirical quantile of the standardised residuals |

The forecasts are **out of sample**: no forecast uses data from its own day or later, and the GARCH parameters are estimated once on the first window. A forecast is **violated** when the return falls below it. Backtests ask three questions:

| Test | Null hypothesis |
|---|---|
| **Kupiec** (unconditional coverage) | The violation rate equals \(p\) |
| **Christoffersen independence** | A violation today does not make one tomorrow more likely |
| **Conditional coverage** | Both together (chi-square with 2 degrees of freedom) |

<div class="write" markdown="1">
**How to write it: the model**

We forecast the one-day 99% value at risk and expected shortfall of the market index with five models: historical simulation on a rolling window of 500 days, a normal distribution with the rolling mean and standard deviation, the RiskMetrics EWMA with decay 0.94, a GARCH(1,1) with t errors estimated on the first 500 days and filtered forward, and a filtered historical simulation that uses the empirical quantile of the GARCH residuals. All forecasts are out of sample. We test the violation rate with Kupiec's likelihood-ratio test, the independence of the violations with Christoffersen's test, and their combination, and compare the realised losses on violation days with the expected shortfall forecast.
</div>

## Load the data

```python
@@LOAD@@
```

@@READ_LOAD@@

## Forecast and backtest

```python
@@EST@@
```

@@READ_EST@@

## The backtests

```python
@@BT@@
```

@@READ_BT@@

## Expected shortfall and the chart

```python
@@PLOT@@
```

@@READ_PLOT@@

## Robustness: the tail level and the window

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: do the tests work?

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Kupiec (1995), *Journal of Derivatives*; Christoffersen (1998), *International Economic Review*; J.P. Morgan (1996), *RiskMetrics Technical Document*; Barone-Adesi, Giannopoulos and Vosper (1999), *Journal of Futures Markets*; McNeil and Frey (2000), *Journal of Empirical Finance*; Bollerslev (1986), *Journal of Econometrics*; Acerbi and Tasche (2002), *Journal of Banking and Finance*.
