title: Portfolio Optimisation and Out-of-Sample Comparison
short: Portfolio optimisation
when: You must decide how to split money across assets, and want to know whether an optimised portfolio beats equal weights once estimation error is taken into account
data: Daily log returns of four European stock indexes (DAX, SMI, CAC, FTSE), 1991-1998; 1,859 trading days
tools: Python · OES engine · SciPy · scikit-learn
oes: portfolio
lead: Use this guide to build portfolios from estimated means and covariances and to test them honestly. It covers minimum-variance, maximum-Sharpe-ratio, risk-parity and equal-weight portfolios, the effect of shrinking the covariance matrix and of forbidding short positions, the sampling uncertainty of the weights by bootstrap, and a rolling out-of-sample comparison with a test of the Sharpe-ratio difference.
---

## The question and the model

**Does optimising beat not optimising?** Given mean returns \(\mu\) and a covariance matrix \(\Sigma\), Markowitz (1952) chooses weights \(w\) with \(\sum_i w_i = 1\):

| Strategy | Weights |
|---|---|
| **Equal weights** | \(w_i = 1/N\) |
| **Minimum variance** | \(\min_w w'\Sigma w\); unconstrained solution \(w \propto \Sigma^{-1}\mathbf 1\) |
| **Maximum Sharpe ratio** | \(\max_w (w'\mu - r_f)/\sqrt{w'\Sigma w}\); unconstrained solution \(w \propto \Sigma^{-1}(\mu - r_f)\) |
| **Risk parity** | each asset contributes the same share \(w_i(\Sigma w)_i/w'\Sigma w = 1/N\) |

The problem is that \(\mu\) and \(\Sigma\) are estimated, and the optimiser puts large weights where the estimates are most favourable (and most wrong). Three remedies are standard: **shrink the covariance matrix** towards a structured target (Ledoit and Wolf 2004), **forbid short positions**, and **use strategies that need no means** (minimum variance, risk parity, equal weights). The test of any of them is the **out-of-sample** performance: estimate on a rolling window, hold the weights for a month, and compare the realised returns.

<div class="write" markdown="1">
**How to write it: the model**

We compare four strategies on the daily returns of four European stock indexes: equal weights, minimum variance, maximum Sharpe ratio and risk parity, all long-only and estimated with a Ledoit-Wolf shrunk covariance matrix. The sampling uncertainty of the weights is measured by a bootstrap that resamples the periods with replacement. The out-of-sample comparison estimates the weights on the previous 250 days, holds them for 21 days and rolls forward; the Sharpe-ratio differences from equal weights are tested with the Jobson-Korkie test with Memmel's correction.
</div>

## Load the data

```python
@@LOAD@@
```

@@READ_LOAD@@

## Estimate the portfolios

```python
@@EST@@
```

@@READ_EST@@

## How certain are the weights?

```python
@@BOOT@@
```

@@READ_BOOT@@

## The honest test: out of sample

```python
@@OOS@@
```

@@READ_OOS@@

## Robustness: shorting, shrinkage and the window

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: the estimation-error problem in a simulation

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Markowitz (1952), *Journal of Finance*; Ledoit and Wolf (2004), *Journal of Multivariate Analysis*; DeMiguel, Garlappi and Uppal (2009), *Review of Financial Studies*; Jobson and Korkie (1981), *Journal of Finance*; Memmel (2003), *Applied Economics Letters*; Maillard, Roncalli and Teïletche (2010), *Journal of Portfolio Management*; Michaud (1989), *Financial Analysts Journal*.
