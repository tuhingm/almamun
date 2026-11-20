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
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

d = pd.read_csv("eustock.csv")
assets = ["r_dax", "r_smi", "r_cac", "r_ftse"]
data = d[["t"] + assets].to_csv(index=False)
print(len(d), "daily log returns; prices run from", d.DAX.iloc[0].round(0), "(DAX) to", d.DAX.iloc[-1].round(0))
print((100 * d[assets].agg(["mean", "std"])).round(3))
print("annualised Sharpe ratios:", (d[assets].mean() / d[assets].std() * np.sqrt(252)).round(2).to_dict())
print(d[assets].corr().round(2))
```

<div class="read" markdown="1">
**How to read it**

The sample has 1,859 daily log returns for the German DAX, Swiss SMI, French CAC and British FTSE from 1991 to 1998, a period in which all four rose strongly (the DAX from 1,614 to 5,474). The SMI has the best annualised Sharpe ratio (1.4), the CAC the worst (0.63) and the FTSE the lowest volatility (0.80% a day). The markets are highly correlated (0.58 to 0.73), so diversification gains are limited.
</div>

## Estimate the portfolios

```python
spec = dict(assets=assets, time="t", boot=300, seed=1)
r = oes.run("portfolio", data, spec)
oes.summary(r)
oes.show(r, "Portfolio weights")
```

<div class="read" markdown="1">
**How to read it**

In sample, the maximum-Sharpe portfolio puts 93% in the SMI (weights: DAX 1%, SMI 93%, CAC 0%, FTSE 5%), earns 20.1% a year with volatility 14.3% and a Sharpe ratio of 1.40; its effective number of assets is only 1.1. The minimum-variance portfolio holds the two least volatile markets (SMI 33%, FTSE 67%), with volatility 12.0% and Sharpe 1.18. Equal weights and risk parity are spread across all four assets (effective number about 4) with Sharpe ratios of 1.12 and 1.14. The maximum-Sharpe portfolio looks best because it is the one fitted to this sample.
</div>

## How certain are the weights?

```python
oes.show(r, "Sampling uncertainty")
oes.plot(r)
```

<div class="read" markdown="1">
**How to read it**

The bootstrap shows how little the maximum-Sharpe weights are determined by the data. The SMI weight has a 95% interval from 0.24 to 1.00; the DAX and FTSE weights range from 0 to 0.60 and 0.56. The CAC weight is below 1% in 92% of the resamples and the DAX weight in 59%: the optimiser puts the money in whichever market looked best in each resample. This is the estimation-error problem: means are estimated imprecisely (a daily mean has a standard error of about 0.02 percentage points, a third of its size), and the optimiser acts on the noise. The chart plots the weights of the four strategies.
</div>

## The honest test: out of sample

```python
oes.show(r, "Out-of-sample")
```

<div class="read" markdown="1">
**How to read it**

Out of sample, the picture is different. All four strategies earn very similar Sharpe ratios: 1.18 for equal weights, 1.20 for minimum variance, 1.19 for maximum Sharpe and 1.21 for risk parity. The in-sample Sharpe ratio of the maximum-Sharpe portfolio, 1.40, falls to 1.19 out of sample. None of the differences from equal weights is statistically significant (z between 0.05 and 1.09; p between 0.28 and 0.96). The optimised portfolios differ in how they get there: the maximum-Sharpe portfolio trades 40% of the portfolio at each monthly rebalancing (turnover 0.40), minimum variance 9% and risk parity only 1.7%, which matters if there are trading costs. Minimum variance and risk parity reduce volatility (12.0% and 13.0% against 13.3%) at a small cost in return.
</div>

## Robustness: shorting, shrinkage and the window

```python
for label, extra in (("short positions allowed, sample covariance", dict(long_only="no", shrink="none")), ("long only, sample covariance", dict(shrink="none")),
                     ("long only, Ledoit-Wolf, window 500", dict(window=500)), ("long only, Ledoit-Wolf, window 120", dict(window=120))):
    rr = oes.run("portfolio", data, dict(spec, boot=0, **extra))
    print("---", label)
    for row in rr["tables"][-1]["rows"]:
        print(f"  {row[0]:22s} return {row[1]:6.3f}  volatility {row[2]:6.3f}  Sharpe {row[3]:5.2f}  turnover {row[4]:5.2f}")
```

<div class="read" markdown="1">
**How to read it**

*Short positions.* Allowing short positions with the sample covariance matrix wrecks the maximum-Sharpe portfolio: volatility of 140% a year, a Sharpe ratio of 0.04 and a turnover of 10 times the portfolio at each rebalancing, because the unconstrained solution takes huge leveraged positions on noisy means. Minimum variance, which needs no means, is unaffected (Sharpe 1.22).

*Shrinkage.* With long-only positions and the sample covariance, the results are very close to those with shrinkage; with four assets and 250 days the covariance matrix is well estimated, and shrinkage matters more when the number of assets is large.

*Window.* With a 500-day window the Sharpe ratios are 1.33 (equal weights), 1.31 (minimum variance), 1.13 (maximum Sharpe) and 1.35 (risk parity); with a 120-day window they are 1.24, 1.27, 1.00 and 1.27. The strategy that uses means is the worst in both and gets worse as the window shortens, which is what estimation error predicts. The ranking of the other three changes with the window, a sign that their differences are noise.
</div>

## Diagnostics: the estimation-error problem in a simulation

```python
# 20 assets, 250 daily observations to estimate the weights, true Sharpe ratios drawn at random
rng = np.random.default_rng(0)
N, T, reps = 20, 250, 200
A = rng.normal(size=(N, N)); C = (A @ A.T / N + np.eye(N)) * 1e-4
sd = np.sqrt(np.diag(C)); mu = 0.5 * sd * rng.normal(0.05, 0.03, N) * 1.0
out = {"equal": [], "minvar": [], "maxsharpe": []}
from numpy.linalg import solve
def true_sharpe(w): return (w @ mu) / np.sqrt(w @ C @ w)
for k in range(reps):
    R = rng.multivariate_normal(mu, C, T)
    m, S = R.mean(0), np.cov(R.T)
    for name, w in (("equal", np.ones(N) / N), ("minvar", solve(S, np.ones(N))), ("maxsharpe", solve(S, m))):
        w = w / w.sum()
        out[name].append(true_sharpe(w) * np.sqrt(252))
print("true annualised Sharpe ratio of the portfolios estimated from 250 days (mean over 200 samples):")
for name, v in out.items():
    print(f"  {name:10s} {np.mean(v):6.2f}")
w_opt = solve(C, mu); w_opt /= w_opt.sum()
print("  best possible (true weights)", round(true_sharpe(w_opt) * np.sqrt(252), 2))
```

<div class="read" markdown="1">
**How to read it**

In a simulation with 20 assets (best possible Sharpe ratio 2.32), the true Sharpe ratio of a portfolio estimated from 250 days of data is 1.74 for equal weights, 1.83 for minimum variance and 1.05 for the maximum-Sharpe weights, averaged over 200 samples. With many assets and a short sample, the portfolio that uses the estimated means is by far the worst, even though it is the optimal one for the sample: a result in the spirit of DeMiguel, Garlappi and Uppal (2009). The test suite checks the unconstrained weights against the closed forms \(\Sigma^{-1}\mathbf 1\) and \(\Sigma^{-1}\mu\), the equalisation of risk contributions in risk parity and the out-of-sample equal-weight returns against a manual calculation.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

We compare four long-only strategies on the daily returns of the DAX, SMI, CAC and FTSE, 1991 to 1998 (1,859 days), with a Ledoit-Wolf covariance matrix. In sample, the maximum-Sharpe portfolio has an annualised Sharpe ratio of 1.40, but its weights are imprecise (the SMI weight is 0.93 with a bootstrap 95% interval of 0.24 to 1.00). Estimating on a rolling 250-day window and holding for 21 days, the out-of-sample Sharpe ratios are 1.18 (equal weights), 1.20 (minimum variance), 1.19 (maximum Sharpe) and 1.21 (risk parity); none differs significantly from equal weights (Jobson-Korkie p between 0.28 and 0.96), and the maximum-Sharpe portfolio trades 40% of the portfolio a month against 2% for risk parity. Allowing short positions makes the maximum-Sharpe portfolio unusable (volatility 140%, Sharpe 0.04). In a simulation with 20 assets and 250 days of data the maximum-Sharpe weights deliver a true Sharpe ratio of 1.05 against 1.74 for equal weights.
</div>

**References.** Markowitz (1952), *Journal of Finance*; Ledoit and Wolf (2004), *Journal of Multivariate Analysis*; DeMiguel, Garlappi and Uppal (2009), *Review of Financial Studies*; Jobson and Korkie (1981), *Journal of Finance*; Memmel (2003), *Applied Economics Letters*; Maillard, Roncalli and Teïletche (2010), *Journal of Portfolio Management*; Michaud (1989), *Financial Analysts Journal*.
