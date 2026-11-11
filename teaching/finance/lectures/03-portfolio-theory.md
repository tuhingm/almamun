# Lecture 03 — Mean-Variance Portfolio Theory and the Efficient Frontier

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Derive minimum-variance portfolios and efficient frontiers; understand diversification, tangency portfolios, and estimation sensitivity.

## 1. Portfolio moments
For risky-asset weights \(w\), expected return and variance are
\[
E(R_p)=w'\mu,\qquad \sigma_p^2=w'\Sigma w.
\]
A fully invested portfolio satisfies \(\mathbf1'w=1\).

## 2. Global minimum-variance portfolio
Minimize \(w'\Sigma w\) subject to \(\mathbf1'w=1\). The Lagrangian FOC gives \(2\Sigma w-\lambda\mathbf1=0\), so
\[
w_{GMV}=\frac{\Sigma^{-1}\mathbf1}{\mathbf1'\Sigma^{-1}\mathbf1},
\quad
\sigma_{GMV}^2=\frac{1}{\mathbf1'\Sigma^{-1}\mathbf1}.
\]
This expression assumes invertible covariance matrix and unrestricted short positions.

## 3. Efficient frontier
For a target mean \(m\), solve minimum variance subject to \(\mathbf1'w=1\) and \(\mu'w=m\). The resulting frontier is a quadratic locus in mean-standard-deviation space. Only its upper branch is mean-variance efficient.

## 4. Risk-free asset and tangency portfolio
With risk-free rate \(r_f\), the tangency risky-asset weights are proportional to \(\Sigma^{-1}(\mu-r_f\mathbf1)\), normalized to sum to one when the normalization is well-defined. Investors combine the risk-free asset and the tangency portfolio according to risk tolerance.

## 5. Constraints and estimation error
Long-only, leverage, turnover, and position constraints alter the optimization problem. Estimated means are especially noisy; unconstrained optimizers can produce extreme weights. Report estimation window, frequency, shrinkage, and constraint set.

## 6. Worked example
For two assets, derive the GMV weight in asset 1:
\[
w_1^* = \frac{\sigma_2^2-\sigma_{12}}{\sigma_1^2+\sigma_2^2-2\sigma_{12}}.
\]
Evaluate for supplied variances and covariance, then verify the weights sum to one.

## 7. Computational lab
Estimate \(\mu\) and \(\Sigma\), calculate GMV and tangency portfolios, and compare unconstrained and long-only allocations out of sample. Plot the frontier and report turnover and realized risk.

## Problem set
1. Derive the GMV solution using a Lagrangian.
2. Derive the two-asset GMV weight.
3. Explain the role of covariance in diversification.
4. Discuss why estimated mean returns can destabilize portfolio weights.

## Coding exercises

These exercises use **real market data** in Python: monthly Fama–French factor returns (1926–2018), daily S&P 500 and
NASDAQ prices (1999–2018), the VIX, Moody's AAA and BAA corporate bond yields (1919–2018), all bundled with the `arch`
package, plus U.S. Treasury yields and household data from the `wooldridge` package. Install once with
`pip install arch wooldridge statsmodels scipy numpy pandas`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 3.1 — The tangency portfolio of the market, size and value factors

*Data:* Fama–French monthly excess returns `Mkt-RF`, `SMB`, `HML`, 1963–2018.

Treat the three factor portfolios as investable excess returns.

1. Estimate the mean vector and covariance matrix on 1963–1990. Compute the tangency (maximum-Sharpe) weights \(w\propto\Sigma^{-1}\mu\), normalized to sum to one, and the in-sample Sharpe ratio.
2. Hold these weights fixed and evaluate the portfolio **out of sample** on 1991–2018. Compare its Sharpe ratio with the market's.
3. Why do out-of-sample Sharpe ratios of estimated optimal portfolios usually fall short of in-sample ones?

```python
import numpy as np, pandas as pd
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

X = ff[["Mkt-RF", "SMB", "HML"]]
ins, oos = X.loc["1963-07":"1990-12"], X.loc["1991-01":]
# TODO: w = inv(Sigma) mu / sum(...); Sharpe = sqrt(12) * mean / sd
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

X = ff[["Mkt-RF", "SMB", "HML"]]
ins, oos = X.loc["1963-07":"1990-12"], X.loc["1991-01":]
mu, S = ins.mean().to_numpy(), ins.cov().to_numpy()
w = np.linalg.solve(S, mu); w = w / w.sum()
SR = lambda r: np.sqrt(12) * r.mean() / r.std()
print("tangency weights (Mkt, SMB, HML):", w.round(3))
print(f"in-sample 1963-1990:  tangency Sharpe {SR(ins @ w):.3f}, market Sharpe {SR(ins['Mkt-RF']):.3f}")
print(f"out-of-sample 1991-2018: tangency Sharpe {SR(oos @ w):.3f}, market Sharpe {SR(oos['Mkt-RF']):.3f}")
print("factor Sharpe ratios in each period:")
print(pd.DataFrame({"1963-1990": ins.apply(SR), "1991-2018": oos.apply(SR)}).round(3).to_string())
```

Output:

```text
tangency weights (Mkt, SMB, HML): [0.221 0.135 0.645]
in-sample 1963-1990:  tangency Sharpe 0.794, market Sharpe 0.260
out-of-sample 1991-2018: tangency Sharpe 0.564, market Sharpe 0.590
factor Sharpe ratios in each period:
        1963-1990  1991-2018
Mkt-RF      0.260      0.590
SMB         0.285      0.197
HML         0.588      0.250
```

In 1963–90, the optimal combination puts most weight on HML and raises the Sharpe ratio from 0.26 (market) to 0.79. But held
fixed for 1991–2018, it earns a Sharpe ratio of 0.56, **below** the market's 0.59: the value premium weakened and the market
did unusually well. This is the classic out-of-sample problem. Estimated optimal portfolios load heavily on the assets with
the best past performance, and past means are poor predictors of future means, so in-sample Sharpe ratios are biased upward
(DeMiguel, Garlappi and Uppal, 2009, find that naive 1/N often wins).

</details>

### Exercise 3.2 — Estimation risk in optimal weights: a bootstrap

*Data:* Fama–French monthly excess returns, 1963–2018.

1. Compute the full-sample tangency weights for `(Mkt-RF, SMB, HML)` and for the global minimum-variance portfolio \(w\propto\Sigma^{-1}\mathbf 1\).
2. Bootstrap the monthly returns (1,000 draws, resampling months) and report the 5th–95th percentile range of each weight for both portfolios.
3. Which portfolio's weights are more stable, and why? (Hint: which inputs does each depend on?)

```python
import numpy as np, pandas as pd
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

X = ff.loc["1963-07":, ["Mkt-RF", "SMB", "HML"]].to_numpy()
# TODO: tangency and GMV weights; bootstrap rows of X
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

X = ff.loc["1963-07":, ["Mkt-RF", "SMB", "HML"]].to_numpy(); T = len(X)
def weights(Y):
    S = np.cov(Y, rowvar=False); one = np.ones(3)
    t = np.linalg.solve(S, Y.mean(0)); g = np.linalg.solve(S, one)
    return t / t.sum(), g / g.sum()
wt, wg = weights(X)
rng = np.random.default_rng(0); B = [weights(X[rng.integers(0, T, T)]) for _ in range(1000)]
bt, bg = np.array([b[0] for b in B]), np.array([b[1] for b in B])
for lab, w, b in (("tangency", wt, bt), ("min-variance", wg, bg)):
    lo, hi = np.percentile(b, 5, axis=0), np.percentile(b, 95, axis=0)
    print(f"{lab:12s}: weights {w.round(3)};  90% bootstrap ranges: " + ", ".join(f"[{l:.2f}, {h:.2f}]" for l, h in zip(lo, hi)))
```

Output:

```text
tangency    : weights [0.3   0.169 0.531];  90% bootstrap ranges: [0.17, 0.45], [-0.02, 0.32], [0.40, 0.68]
min-variance: weights [0.168 0.327 0.505];  90% bootstrap ranges: [0.14, 0.20], [0.29, 0.37], [0.48, 0.53]
```

The tangency weights are uncertain: the 90% range for the market weight is about 0.17 to 0.45, and the SMB weight could be
zero. The minimum-variance weights are much more stable (ranges about four times narrower), because they depend only on the
covariance matrix, which is estimated precisely from monthly data, while tangency weights also depend on mean returns,
which are not. That is why many practitioners use minimum-variance or risk-parity portfolios, or shrink estimated means
toward a model (Black–Litterman).

</details>

## Further reading
Markowitz (1952); Merton (1972); Bodie, Kane and Marcus, *Investments*.
