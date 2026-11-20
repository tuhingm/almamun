title: Volatility Forecast Evaluation
short: Volatility forecasts
when: You have several volatility models (GARCH, EWMA, rolling windows) and want to know which forecasts best, and whether the differences are statistically meaningful
data: Daily returns of the CRSP value-weighted market index, 1989-1998 (Ecdat); 2,528 trading days
tools: Python · OES engine · NumPy
oes: vol_eval
lead: Use this guide to compare volatility forecasts out of sample. Variance is never observed, so every comparison uses a proxy, and the choice of loss function matters: some losses rank forecasts correctly even with a noisy proxy and some do not. The guide covers the proxy, the QLIKE and squared-error losses, the Mincer-Zarnowitz regression for bias, the Diebold-Mariano test for the difference between two forecasts, robustness to the window, a simulation and the write-up.
---

## The question and the model

**Which forecast of tomorrow's variance is best?** Five forecasts of the one-day-ahead variance \(h_t\) are compared on the days after the first estimation window:

| Forecast | Definition |
|---|---|
| **GARCH(1,1)** | \(h_t = \omega + \alpha\varepsilon_{t-1}^2 + \beta h_{t-1}\), parameters from the first window |
| **EWMA** | \(h_t = \lambda h_{t-1} + (1 - \lambda)\varepsilon_{t-1}^2\), \(\lambda = 0.94\) |
| **Rolling variance (window \(w\) and 25 days)** | The sample variance of the previous \(w\) (or 25) returns |
| **Constant** | The variance of the first window |

The realised variance is not observed. The squared return \(\varepsilon_t^2\) is an unbiased but very noisy proxy. Patton (2011) showed that with a noisy proxy only some losses rank forecasts correctly, among them the **QLIKE** loss, \(\log h_t + \varepsilon_t^2/h_t\), and the **squared error**, \((\varepsilon_t^2 - h_t)^2\); the squared error is correct but dominated by a few extreme days. Two tests complete the comparison:

| Test | What it checks |
|---|---|
| **Mincer-Zarnowitz regression** | \(\varepsilon_t^2 = a + b\,h_t + u_t\): an unbiased forecast has \(a = 0\), \(b = 1\) |
| **Diebold-Mariano** (Diebold and Mariano 1995) | Whether the average loss of two forecasts differs, with a Newey-West variance for the serial correlation of the loss differences |

<div class="write" markdown="1">
**How to write it: the model**

We compare out-of-sample one-day variance forecasts from a GARCH(1,1) estimated on the first 500 days, the RiskMetrics EWMA with decay 0.94, rolling 500-day and 25-day sample variances and a constant. The squared demeaned return is the variance proxy. Forecasts are ranked by the QLIKE loss and the squared error, bias is tested with Mincer-Zarnowitz regressions with Newey-West standard errors, and differences from the EWMA benchmark are tested with Diebold-Mariano statistics with 10 lags.
</div>

## Load the data

```python
@@LOAD@@
```

@@READ_LOAD@@

## Forecast and compare the losses

```python
@@EST@@
```

@@READ_EST@@

## Are the forecasts unbiased?

```python
@@MZ@@
```

@@READ_MZ@@

## Are the differences significant?

```python
@@DM@@
```

@@READ_DM@@

## Robustness: the window and the benchmark

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: does QLIKE pick the right model?

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Patton (2011), *Journal of Econometrics*; Diebold and Mariano (1995), *Journal of Business and Economic Statistics*; Mincer and Zarnowitz (1969); Andersen and Bollerslev (1998), *International Economic Review*; Hansen and Lunde (2005), *Journal of Applied Econometrics*; Bollerslev (1986), *Journal of Econometrics*; J.P. Morgan (1996), *RiskMetrics Technical Document*.
