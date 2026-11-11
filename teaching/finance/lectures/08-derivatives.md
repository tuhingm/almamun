# Lecture 08 — Forwards, Futures, Options, and Swaps

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Value derivative payoffs; derive put-call parity and binomial pricing; distinguish physical and risk-neutral probabilities; interpret Greeks and hedging.

## 1. Forward and futures contracts
A forward commits counterparties to trade an asset at maturity for delivery price \(K\). For a non-income asset under simple assumptions, the no-arbitrage forward price is \(F_{0,T}=S_0(1+r)^T\). Income, storage costs, convenience yield, and funding conventions modify this relation. Futures are marked to market daily.

## 2. Options and parity
A European call payoff is \((S_T-K)^+\); a put payoff is \((K-S_T)^+\). For a non-dividend-paying asset,
\[
C_0-P_0=S_0-Ke^{-rT}.
\]
The parity follows by comparing portfolios with identical terminal payoffs.

## 3. Binomial risk-neutral valuation
In a one-step binomial model, stock moves to \(uS_0\) or \(dS_0\). The risk-neutral probability is
\[
q=\frac{(1+r)-d}{u-d},
\]
for a one-period gross risk-free return \(1+r\), with no-arbitrage requiring \(d<1+r<u\). The derivative value is the discounted risk-neutral expected payoff.

## 4. Black-Scholes framework
For a European option under geometric Brownian motion, constant volatility, frictionless trading, and continuous hedging, the Black-Scholes formula gives a benchmark price. Its assumptions and market conventions must be stated.

## 5. Greeks and hedging
Delta measures first-order sensitivity to the underlying; gamma measures curvature; vega measures volatility sensitivity; theta measures time decay; rho measures interest-rate sensitivity. Delta hedging leaves residual gamma, volatility, jump, and basis risks.

## 6. Worked example
Use put-call parity to solve for a put given call price, spot price, strike, rate, and maturity. Verify by constructing the two equivalent portfolios.

## 7. Computational lab
Build a binomial option pricer and compare convergence to Black-Scholes. Calculate Greeks by analytic formulas and finite differences; test hedge P&L under discrete rebalancing.

## Problem set
1. Derive forward price by replication.
2. Prove put-call parity.
3. Derive risk-neutral probability.
4. Explain why risk-neutral probabilities are not forecasts of physical frequencies.

## Coding exercises

These exercises use **real market data** in Python: monthly Fama–French factor returns (1926–2018), daily S&P 500 and
NASDAQ prices (1999–2018), the VIX, Moody's AAA and BAA corporate bond yields (1919–2018), all bundled with the `arch`
package, plus U.S. Treasury yields and household data from the `wooldridge` package. Install once with
`pip install arch wooldridge statsmodels scipy numpy pandas`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 8.1 — Black–Scholes with the VIX, and the variance risk premium

*Data:* Daily S&P 500 (1999–2018) and VIX (2014–2018), from `arch.data`.

1. Price a one-month at-the-money call and put on the S&P 500 on the last trading day of 2018, using the VIX as the implied volatility and a 2.4% risk-free rate. Verify put–call parity.
2. Compute delta, gamma and vega analytically and check delta by finite differences.
3. For 2014–2018, compare the VIX with the **realized** volatility of the S&P 500 over the following 21 trading days. What is the average gap, and what does it measure?

```python
import numpy as np, pandas as pd
from scipy.stats import norm
from arch.data import sp500, vix
S = sp500.load()["Adj Close"]; V = vix.load()["vix"]
def bs(S, K, T, r, sig, call=True):
    d1 = (np.log(S / K) + (r + sig**2 / 2) * T) / (sig * np.sqrt(T)); d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) if call else K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
# TODO: price, parity, greeks; realized vol = sqrt(252) * rolling std of future 21 daily log returns
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd
from scipy.stats import norm
from arch.data import sp500, vix
S = sp500.load()["Adj Close"]; V = vix.load()["vix"] / 100
def bs(S, K, T, r, sig, call=True):
    d1 = (np.log(S / K) + (r + sig**2 / 2) * T) / (sig * np.sqrt(T)); d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) if call else K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
s0, sig, T, r = S.iloc[-1], V.loc[:"2018-12-31"].iloc[-1], 21 / 252, 0.024; K = s0
c, p = bs(s0, K, T, r, sig), bs(s0, K, T, r, sig, call=False)
print(f"S = {s0:.2f}, VIX = {100*sig:.1f}%: call {c:.2f}, put {p:.2f};  C - P = {c - p:.4f} vs S - K e^(-rT) = {s0 - K*np.exp(-r*T):.4f}")
d1 = (np.log(s0 / K) + (r + sig**2 / 2) * T) / (sig * np.sqrt(T))
delta, gamma, vega = norm.cdf(d1), norm.pdf(d1) / (s0 * sig * np.sqrt(T)), s0 * norm.pdf(d1) * np.sqrt(T)
h = 0.01; fd = (bs(s0 + h, K, T, r, sig) - bs(s0 - h, K, T, r, sig)) / (2 * h)
print(f"delta {delta:.4f} (finite difference {fd:.4f}), gamma {gamma:.5f}, vega {vega:.2f} per unit vol")
lr = np.log(S).diff()
rv = lr[::-1].rolling(21).std()[::-1].shift(-1) * np.sqrt(252)        # realized vol over the next 21 days
df = pd.concat([V.rename("vix"), rv.rename("rv")], axis=1).dropna().loc["2014":"2018"]
print(f"2014-2018: mean VIX {100*df.vix.mean():.1f}%, mean subsequent realized vol {100*df.rv.mean():.1f}%, VIX above realized on {np.mean(df.vix > df.rv):.0%} of days")
print(f"average variance risk premium (VIX^2 - RV^2): {1e4*np.mean(df.vix**2 - df.rv**2):.1f} (squared percentage points)")
```

Output:

```text
S = 2506.85, VIX = 25.4%: call 75.83, put 70.82;  C - P = 5.0087 vs S - K e^(-rT) = 5.0087
delta 0.5255 (finite difference 0.5255), gamma 0.00216, vega 288.11 per unit vol
2014-2018: mean VIX 14.7%, mean subsequent realized vol 11.8%, VIX above realized on 80% of days
average variance risk premium (VIX^2 - RV^2): 61.5 (squared percentage points)
```

At the end of 2018, with the VIX at 25.4%, the one-month at-the-money call is worth about 76 index points and the put about
71; put–call parity holds to four decimals, because it is a no-arbitrage relation independent of the model. The analytic
delta matches the finite difference. Over 2014–2018, the VIX exceeded the volatility realized over the following month on
80% of days, by about 3 points on average: option buyers pay a **variance risk premium** to insure against volatility
spikes. Option sellers earn it most of the time — and lose heavily in crises.

</details>

### Exercise 8.2 — Binomial trees: convergence to Black–Scholes and the value of early exercise

*Data:* Inputs from the previous exercise: the S&P 500 level and VIX on 31 December 2018, one-month and one-year maturities.

1. Implement a Cox–Ross–Rubinstein binomial tree. Price the European call with 10, 50, 200 and 1,000 steps and show convergence to Black–Scholes.
2. Price a one-year **American put** at the money and 10% in the money, and compute the early-exercise premium over the European put.
3. Why is early exercise of an American call on a non-dividend-paying stock never optimal, while it can be optimal for a put?

```python
import numpy as np
from scipy.stats import norm
def crr(S, K, T, r, sig, n, call=True, american=False):
    dt = T / n; u = np.exp(sig * np.sqrt(dt)); d = 1 / u; q = (np.exp(r * dt) - d) / (u - d)
    # TODO: terminal payoffs, backward induction with optional early exercise
    ...
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np
from scipy.stats import norm
from arch.data import sp500, vix
s0 = float(sp500.load()["Adj Close"].iloc[-1]); sig = float(vix.load()["vix"].loc[:"2018-12-31"].iloc[-1]) / 100; r = 0.024
def crr(S, K, T, r, sig, n, call=True, american=False):
    dt = T / n; u = np.exp(sig * np.sqrt(dt)); d = 1 / u; q = (np.exp(r * dt) - d) / (u - d); disc = np.exp(-r * dt)
    ST = S * u**np.arange(n, -1, -1) * d**np.arange(0, n + 1)
    V = np.maximum(ST - K, 0) if call else np.maximum(K - ST, 0)
    for k in range(n - 1, -1, -1):
        ST = S * u**np.arange(k, -1, -1) * d**np.arange(0, k + 1)
        V = disc * (q * V[:-1] + (1 - q) * V[1:])
        if american: V = np.maximum(V, (ST - K) if call else (K - ST))
    return V[0]
def bs(S, K, T, r, sig, call=True):
    d1 = (np.log(S / K) + (r + sig**2 / 2) * T) / (sig * np.sqrt(T)); d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) if call else K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
T = 21 / 252
print(f"Black-Scholes one-month ATM call: {bs(s0, s0, T, r, sig):.4f}")
for n in (10, 50, 200, 1000):
    print(f"  CRR with {n:4d} steps: {crr(s0, s0, T, r, sig, n):.4f}")
for lab, K in (("at the money", s0), ("10% in the money", 1.1 * s0)):
    eu, am = crr(s0, K, 1.0, r, sig, 1000, call=False), crr(s0, K, 1.0, r, sig, 1000, call=False, american=True)
    print(f"one-year put {lab}: European {eu:.2f}, American {am:.2f}, early-exercise premium {am - eu:.2f} ({(am-eu)/eu:.2%})")
print(f"one-year ATM call: European {crr(s0, s0, 1, r, sig, 1000):.2f}, American {crr(s0, s0, 1, r, sig, 1000, american=True):.2f}")
```

Output:

```text
Black-Scholes one-month ATM call: 75.8295
  CRR with   10 steps: 74.0220
  CRR with   50 steps: 75.4638
  CRR with  200 steps: 75.7379
  CRR with 1000 steps: 75.8112
one-year put at the money: European 221.86, American 227.28, early-exercise premium 5.42 (2.44%)
one-year put 10% in the money: European 365.74, American 376.54, early-exercise premium 10.80 (2.95%)
one-year ATM call: European 281.30, American 281.30
```

The CRR price converges to Black–Scholes as the number of steps grows, with the oscillating, first-order convergence
typical of binomial trees. Early exercise is worth about 2.4% of the European price for an at-the-money one-year put and
about 3% when the put is 10% in the money: when the put is deep in the money, receiving the strike now and earning
interest beats waiting. For a call on a non-dividend-paying asset, exercising early forfeits the time value and the interest
on the strike, so the American and European calls have the same price — as the output confirms.

</details>

## Further reading
Hull, *Options, Futures, and Other Derivatives*; Shreve, *Stochastic Calculus for Finance I*.
