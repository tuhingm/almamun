# Lecture 12 — Heterogeneous-Agent Macroeconomics and Incomplete Markets

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Formulate household dynamic programming under idiosyncratic risk; derive policy functions and stationary distributions; define aggregate equilibrium.

## 1. Household problem
Households face idiosyncratic productivity \(z\), asset holdings \(a\), and borrowing limit \(\underline a\). The Bellman equation is
\[
V(a,z)=\max_{c,a'}\left\{u(c)+\beta\sum_{z'}P(z'\mid z)V(a',z')\right\}
\]
subject to \(c+a'=wz+(1+r)a\) and \(a'\geq\underline a\).

## 2. Optimality
For an interior choice,
\[
u'(c)=\beta(1+r)\sum_{z'}P(z'\mid z)V_a(a',z').
\]
The envelope condition is \(V_a(a,z)=u'(c)(1+r)\). Combining conditions yields the stochastic Euler equation. At the borrowing limit, the complementary-slackness condition replaces equality with an inequality.

## 3. Distribution and aggregation
Let \(\mu_t(a,z)\) denote the cross-sectional distribution. Household policy functions and the Markov transition for \(z\) induce a law of motion \(\mu_{t+1}=\mathcal T(\mu_t)\). Aggregate assets are \(A_t=\int a\,d\mu_t\); labor and consumption are aggregated similarly.

## 4. Stationary equilibrium
A stationary recursive competitive equilibrium consists of prices, household value and policy functions, and an invariant distribution consistent with household optimization, exogenous transitions, firm pricing, and market clearing. In a closed economy without government debt, aggregate capital supply equals firms' capital demand.

## 5. Worked algorithm
Discretize assets and productivity. Guess prices, solve the household problem by value-function iteration or endogenous-grid methods, compute the invariant distribution, aggregate assets, update prices, and iterate until market clearing holds.

## 6. Computational lab
Compare incomplete-markets consumption-saving behavior across borrowing limits and idiosyncratic risk levels. Report policy functions, invariant distributions, aggregate capital, and market-clearing errors. Check grid and Markov-chain accuracy.

## Problem set
1. Derive the household Euler and envelope equations.
2. Explain why individual states differ from aggregate states.
3. Define the invariant distribution.
4. Describe the equilibrium fixed-point algorithm.

## Coding exercises

These exercises combine **real U.S. macroeconomic data** (the quarterly `macrodata` series in `statsmodels`, 1959–2009,
and annual series from the `wooldridge` package) with **numerical solutions of the models** in this lecture. Install once
with `pip install numpy scipy pandas statsmodels wooldridge`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 12.1 — An Aiyagari household solved by the endogenous grid method

*Data:* Calibration (annual): \(\beta=0.96\), CRRA \(\sigma=2\), \(r=0.03\), borrowing limit 0; log labor income AR(1) with \(\rho=0.9\), \(\sigma=0.2\) (discretized by Rouwenhorst with 7 states).

1. Discretize income with the Rouwenhorst method.
2. Solve the household's consumption policy by the **endogenous grid method** (Carroll, 2006): invert the Euler equation on a grid of next-period assets, then handle the borrowing constraint.
3. Compute the stationary wealth distribution (Young's histogram method). Report the share of households at the borrowing constraint, the wealth Gini, and the average marginal propensity to consume (MPC) out of a small windfall.

```python
import numpy as np
b, s, r, rho, sig, n = 0.96, 2.0, 0.03, 0.9, 0.2, 7
# TODO: Rouwenhorst; asset grid a' in [0, 50] (dense near 0); EGM loop; Young's method for the distribution
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np
b, s, r, rho, sig, n = 0.96, 2.0, 0.03, 0.9, 0.2, 7
p = (1 + rho) / 2; P = np.array([[p, 1 - p], [1 - p, p]])
for m_ in range(3, n + 1):
    Pn = np.zeros((m_, m_))
    Pn[:-1, :-1] += p * P; Pn[:-1, 1:] += (1 - p) * P; Pn[1:, :-1] += (1 - p) * P; Pn[1:, 1:] += p * P
    Pn[1:-1] /= 2; P = Pn
psi = sig / np.sqrt(1 - rho**2) * np.sqrt(n - 1)
z = np.exp(np.linspace(-psi, psi, n)); pi = np.ones(n) / n
for _ in range(2000): pi = pi @ P
z = z / (pi @ z)                                          # normalize mean income to 1
a = 50 * np.linspace(0, 1, 500)**2                         # asset grid, dense near the constraint
up = lambda c: c**(-s); upinv = lambda u: u**(-1 / s)
C = (1 + r) * a[:, None] + z[None, :] - a[0]               # initial guess: consume income plus a little
for it in range(5000):
    Emu = (up(C) @ P.T) * b * (1 + r)                      # E[u'(c')] for each (a', z)
    c_endo = upinv(Emu); a_endo = (c_endo + a[:, None] - z[None, :]) / (1 + r)
    Cn = np.empty_like(C)
    for j in range(n):
        Cn[:, j] = np.interp(a, a_endo[:, j], c_endo[:, j])
        bind = a < a_endo[0, j]; Cn[bind, j] = (1 + r) * a[bind] + z[j] - a[0]     # constraint binds
    if np.max(np.abs(Cn - C)) < 1e-9: break
    C = Cn
ap = (1 + r) * a[:, None] + z[None, :] - C
D = np.ones((len(a), n)) / (len(a) * n)
idx = np.clip(np.searchsorted(a, ap) - 1, 0, len(a) - 2); w = np.clip((a[idx + 1] - ap) / (a[idx + 1] - a[idx]), 0, 1)
for _ in range(5000):
    Dn = np.zeros_like(D)
    for j in range(n):
        np.add.at(Dn[:, j], idx[:, j], w[:, j] * D[:, j]); np.add.at(Dn[:, j], idx[:, j] + 1, (1 - w[:, j]) * D[:, j])
    Dn = Dn @ P
    if np.max(np.abs(Dn - D)) < 1e-12: break
    D = Dn
wealth = D.sum(1); order = np.argsort(a); cw = np.cumsum(wealth[order]); cwa = np.cumsum(wealth[order] * a[order])
gini = 1 - np.sum((cwa[1:] + cwa[:-1]) * np.diff(cw)) / cwa[-1] - cwa[0] * cw[0] / cwa[-1]
mpc = (C[1:] - C[:-1]) / ((1 + r) * (a[1:] - a[:-1]))[:, None]
avg_mpc = np.sum(D[:-1] * mpc) / D[:-1].sum()
print(f"EGM converged in {it} iterations; mean assets {np.sum(D.sum(1) * a):.2f} (mean income 1)")
print(f"share at the borrowing constraint: {D[0].sum():.3f};  wealth Gini: {gini:.3f};  average annual MPC: {avg_mpc:.3f}")
np.save("/tmp/aiyagari_D.npy", D); np.save("/tmp/aiyagari_mpc.npy", mpc); np.save("/tmp/aiyagari_a.npy", a)
```

Output:

```text
EGM converged in 375 iterations; mean assets 7.65 (mean income 1)
share at the borrowing constraint: 0.016;  wealth Gini: 0.487;  average annual MPC: 0.077
```

EGM converges in a few hundred iterations without any root-finding: inverting the Euler equation on a grid of next-period
assets gives consumption directly. The stationary distribution has a wealth Gini of about 0.49 — far below the U.S. value of
about 0.8 — and only 1.6% of households at the borrowing constraint. The average annual MPC is 0.08, close to the permanent-income
benchmark. One-asset models with these parameters generate too few constrained households and too little wealth inequality;
HANK models fix this with two assets (Kaplan, Moll and Violante, 2018) or with a lower discount factor.

</details>

### Exercise 12.2 — Hand-to-mouth households and wealth inequality in U.S. survey data

*Data:* `401ksubs` — 9,275 U.S. households, 1991 SIPP: family income `inc` and net total financial assets `nettfa` (thousands of dollars).

Kaplan, Violante and Weidner (2014) classify households as hand-to-mouth (HtM) if their liquid wealth is below about half a month of income.

1. Compute the share of households with non-positive financial assets, and the share with financial assets below half a month of income.
2. Compute the Gini coefficient of financial wealth (allowing negative values) and the share of financial wealth held by the top 10%.
3. Break the HtM share down by income quintile. Why does the presence of HtM households at **all** income levels matter for fiscal policy?

```python
import numpy as np, pandas as pd, wooldridge as wd
d = wd.data("401ksubs")
htm = d["nettfa"] < d["inc"] / 24
# TODO: shares; Gini via mean absolute difference; by pd.qcut(d.inc, 5)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, wooldridge as wd
d = wd.data("401ksubs")
d["htm"] = d["nettfa"] < d["inc"] / 24
x = np.sort(d["nettfa"].to_numpy()); n = len(x); i = np.arange(1, n + 1)
gini = np.sum((2 * i - n - 1) * x) / (n * x.sum())
print(f"share with nettfa <= 0: {np.mean(d['nettfa'] <= 0):.3f};  hand-to-mouth (nettfa < half a month of income): {d['htm'].mean():.3f}")
print(f"Gini of financial wealth: {gini:.3f};  top 10% share: {x[int(0.9*n):].sum()/x.sum():.3f};  median ${np.median(x):.1f}k vs mean ${x.mean():.1f}k")
d["quintile"] = pd.qcut(d["inc"], 5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
print(d.groupby("quintile", observed=True).agg(income=("inc", "mean"), htm_share=("htm", "mean"), median_nettfa=("nettfa", "median")).round(3).to_string())
```

Output:

```text
share with nettfa <= 0: 0.350;  hand-to-mouth (nettfa < half a month of income): 0.464
Gini of financial wealth: 0.982;  top 10% share: 0.722;  median $2.0k vs mean $19.1k
              income  htm_share  median_nettfa
quintile                                      
Q1 (lowest)   15.060      0.688          0.000
Q2            23.942      0.590          0.299
Q3            33.410      0.476          1.849
Q4            46.261      0.353          6.375
Q5 (highest)  77.607      0.212         26.299
```

In these 1991 data, 35% of households have non-positive financial assets, and 46% hold less than half a month of income in
financial assets — somewhat above Kaplan, Violante and Weidner's (2014) estimate of about one third, which uses a broader
liquid-wealth measure and later survey years. Financial
wealth is extremely concentrated (Gini 0.98, top 10% share 72%; the mean is ten times the median). The hand-to-mouth share
falls with income but remains above 20% even in the top income quintile. Those are the "wealthy hand-to-mouth" — households
with illiquid wealth (housing, pensions) but little cash — which is why transfers raise spending even among middle-income
households.

</details>

## Further reading
Aiyagari (1994); Huggett (1993); Ljungqvist and Sargent, *Recursive Macroeconomic Theory*.
