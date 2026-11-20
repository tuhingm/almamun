title: Dynamic Conditional Correlation (DCC-GARCH)
short: DCC-GARCH
when: You want to know whether the correlation between assets is constant, how it moves over time, and whether it rises when markets are turbulent
data: Daily returns of IBM, General Electric and Mobil, 1989-1998 (Ecdat); 2,528 trading days
tools: Python · OES engine · NumPy
oes: dcc
lead: Use this guide when the correlation between two or more assets matters for hedging, diversification or risk, and you doubt that it is constant. The DCC-GARCH model of Engle (2002) estimates a GARCH(1,1) for each series, then lets the correlation between the standardised residuals follow its own dynamics. The guide covers the model, the two estimation steps, the correlation paths, the test of constant correlation, how the correlation behaves in volatile periods, robustness, a simulation and the write-up.
---

## The question and the model

**Is the correlation between two assets a constant?** A constant correlation makes portfolio risk easy to compute, but it is unlikely to be true: diversification tends to fail when it is needed most. The DCC model of Engle (2002) has two steps. First each return follows a univariate GARCH(1,1):

\[
r_{it} = \mu_i + \sigma_{it} z_{it}, \qquad \sigma_{it}^2 = \omega_i + \alpha_i \varepsilon_{i,t-1}^2 + \beta_i \sigma_{i,t-1}^2 .
\]

Second, the standardised residuals \(z_t\) have the conditional correlation matrix \(R_t\) obtained from

\[
Q_t = (1 - a - b)\,\bar S + a\, z_{t-1} z_{t-1}' + b\, Q_{t-1}, \qquad R_t = \operatorname{diag}(Q_t)^{-1/2}\, Q_t \,\operatorname{diag}(Q_t)^{-1/2},
\]

where \(\bar S\) is the unconditional correlation of the standardised residuals. The parameter \(a\) says how much the correlation reacts to yesterday's joint shock, \(b\) how persistent it is. If \(a = b = 0\) the correlation is constant.

<div class="write" markdown="1">
**How to write it: the model**

We estimate a DCC-GARCH model (Engle 2002) for the daily returns of three stocks. In the first step each return has a GARCH(1,1) model estimated by Gaussian quasi-maximum likelihood. In the second step the correlation of the standardised residuals follows the DCC recursion, estimated by maximum likelihood with the first-step estimates as given. We test constant correlation (a = b = 0) with a likelihood-ratio test, and check that the standardised residuals have no remaining ARCH.
</div>

## Load the data

```python
@@LOAD@@
```

@@READ_LOAD@@

## Estimate

```python
@@EST@@
```

@@READ_EST@@

## The correlations over time

```python
@@CORR@@
```

@@READ_CORR@@

## Do correlations rise when the market is volatile?

```python
@@CRISIS@@
```

@@READ_CRISIS@@

## Robustness: the pair and the sub-sample

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: does the estimator recover the dynamics?

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Engle (2002), *Journal of Business and Economic Statistics*; Bollerslev (1990), *Review of Economics and Statistics*; Engle and Sheppard (2001), NBER Working Paper 8554; Aielli (2013), *Journal of Business and Economic Statistics*; Longin and Solnik (2001), *Journal of Finance*; Bollerslev (1986), *Journal of Econometrics*.
