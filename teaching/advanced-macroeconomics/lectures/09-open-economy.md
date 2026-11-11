# Lecture 09 — Open-Economy Macroeconomics and Exchange-Rate Dynamics

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Interpret balance-of-payments accounts; derive uncovered interest parity; analyze exchange-rate regimes and external shocks.

## 1. External accounts
The current account records net exports plus net primary income and transfers. National accounting gives, under consistent definitions, current account = national saving − domestic investment. Net foreign assets evolve with the current account and valuation changes.

## 2. Exchange rates
Define the nominal exchange rate \(S_t\) as domestic currency per unit of foreign currency. An increase is a domestic-currency depreciation. The real exchange rate is
\[
Q_t=S_tP_t^*/P_t,
\]
so an increase in \(Q_t\) is a real depreciation under this convention.

## 3. Uncovered interest parity
Under risk neutrality and no currency risk premium,
\[
i_t-i_t^*=E_t\Delta s_{t+1},
\]
where \(s_t=\log S_t\). With a risk premium \(\psi_t\), write \(i_t-i_t^*=E_t\Delta s_{t+1}+\psi_t\), defining the sign convention explicitly.

## 4. Small open economy
A small open economy takes the world interest rate as given in a benchmark model. With incomplete international asset markets, net foreign assets may follow a stationarity-inducing portfolio adjustment or debt-elastic interest premium. The choice affects long-run external dynamics.

## 5. Worked example
If the domestic interest rate is 5% and the foreign rate is 3%, UIP predicts expected domestic-currency depreciation of approximately 2%, under the stated exchange-rate convention and zero risk premium.

## 6. Computational lab
Simulate a small-open-economy model following a terms-of-trade shock. Compare a fixed exchange rate with inflation targeting. Report current account, output, inflation, policy rate, and real exchange rate responses.

## Problem set
1. Derive the saving-investment-current-account identity.
2. Explain the sign of a real depreciation under the given convention.
3. State the assumptions behind UIP.
4. Compare monetary autonomy under fixed and flexible exchange rates.

## Coding exercises

These exercises combine **real U.S. macroeconomic data** (the quarterly `macrodata` series in `statsmodels`, 1959–2009,
and annual series from the `wooldridge` package) with **numerical solutions of the models** in this lecture. Install once
with `pip install numpy scipy pandas statsmodels wooldridge`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 9.1 — Current-account responses to temporary and persistent income shocks

*Data:* `macrodata` — U.S. quarterly real GDP; the persistence of its cyclical component calibrates the income process.

In a small open economy with the permanent-income hypothesis, consumption responds to the annuity value of an income
shock. If detrended income follows an AR(1) with persistence \(\rho\), a unit shock raises consumption by
\(\frac{r}{1+r-\rho}\), so the current account changes by \(CA=1-\frac{r}{1+r-\rho}\) on impact.

1. Estimate \(\rho\) for the quarterly HP cycle of U.S. log real GDP and for its linearly detrended log.
2. With \(r=1\%\) per quarter, compute the impact responses of consumption and the current account for both persistence estimates and for \(\rho\in\{0,0.5,0.9,0.99,1\}\).
3. Interpret: why do countries run current-account **surpluses** after temporary income gains, but not after permanent ones?

```python
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data; ly = np.log(m["realgdp"].to_numpy())
# TODO: rho from AR(1) of the HP cycle and of linear-detrended log GDP; responses r/(1+r-rho)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data; ly = np.log(m["realgdp"].to_numpy())
ar1 = lambda x: np.corrcoef(x[1:], x[:-1])[0, 1]
hp = hpfilter(ly, 1600)[0]; t = np.arange(len(ly)); lin = ly - np.polyval(np.polyfit(t, ly, 1), t)
r = 0.01
cases = {"HP cycle": ar1(hp), "linear detrend": ar1(lin), "rho = 0": 0.0, "rho = 0.5": 0.5, "rho = 0.9": 0.9, "rho = 0.99": 0.99, "random walk": 1.0}
for lab, rho in cases.items():
    dc = r / (1 + r - rho)
    print(f"{lab:15s}: rho = {rho:.3f} -> consumption +{dc:.3f}, current account {1 - dc:+.3f} per unit income shock")
```

Output:

```text
HP cycle       : rho = 0.861 -> consumption +0.067, current account +0.933 per unit income shock
linear detrend : rho = 0.970 -> consumption +0.252, current account +0.748 per unit income shock
rho = 0        : rho = 0.000 -> consumption +0.010, current account +0.990 per unit income shock
rho = 0.5      : rho = 0.500 -> consumption +0.020, current account +0.980 per unit income shock
rho = 0.9      : rho = 0.900 -> consumption +0.091, current account +0.909 per unit income shock
rho = 0.99     : rho = 0.990 -> consumption +0.500, current account +0.500 per unit income shock
random walk    : rho = 1.000 -> consumption +1.000, current account +0.000 per unit income shock
```

With the HP cycle's persistence (0.86), a unit income shock raises consumption by only 0.07 and the current account by
0.93: households save almost all of a temporary gain abroad. With the linearly detrended series (0.97), consumption rises by
a quarter. As persistence approaches one, the consumption response approaches one and the current account response
vanishes. That is why the current account should absorb temporary shocks (a harvest failure, a temporary oil price boom) but
not permanent ones. The estimated persistence depends heavily on the detrending method — the same issue as in the RBC
calibration.

</details>

### Exercise 9.2 — Dornbusch overshooting, solved and simulated

*Data:* Stylized calibration of the Dornbusch (1976) model: money-demand semi-elasticity to the quarterly interest rate \(\lambda=4\); price-level convergence speed \(\nu\) per quarter varied from 0.05 to 1; long-run PPP.

With sticky prices, uncovered interest parity and money demand \(m-p=\phi y-\lambda i\), a permanent money increase raises
the long-run price level and exchange rate one-for-one. On the saddle path, the exchange rate satisfies
\(e_t-\bar e=-\frac{1}{\lambda\nu}(p_t-\bar p)\), where \(\nu\) is the speed at which prices converge,
\(p_{t+1}-p_t=\nu(\bar p-p_t)\).

1. For a permanent 10% money increase, compute the impact depreciation and the overshooting ratio \(1+\frac{1}{\lambda\nu}\) for several values of \(\nu\).
2. Simulate prices, the exchange rate and the interest differential (from money-market equilibrium, \(i-i^*=(p-m)/\lambda\)) for 20 quarters with \(\nu=0.25\). Check that the differential and the expected appreciation satisfy UIP.
3. How does overshooting depend on price stickiness?

```python
import numpy as np
lam, dm = 4.0, 10.0
# impact: e0 = dm * (1 + 1/(lam*nu)); then p' = p + nu (dm - p), e = dm + (dm - p)/(lam*nu)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np
lam, dm = 4.0, 10.0
for nu in (0.05, 0.10, 0.25, 0.50, 1.00):
    jump = dm * (1 + 1 / (lam * nu))
    print(f"nu = {nu:.2f} (half-life of price gap {np.log(0.5)/np.log(1-nu) if nu < 1 else 0:.1f} q): impact depreciation {jump:.1f}% (overshooting ratio {jump/dm:.2f})")
nu, p = 0.25, 0.0; e_prev = None
print("quarter | price level | exchange rate | i - i* (pp/quarter) | expected depreciation next quarter")
for t in range(0, 21):
    e = dm + (dm - p) / (lam * nu); idiff = (p - dm) / lam
    p_next = p + nu * (dm - p); e_next = dm + (dm - p_next) / (lam * nu)
    if t % 4 == 0:
        print(f"  {t:2d}    |   {p:6.2f}    |    {e:6.2f}     |       {idiff:6.3f}        |        {e_next - e:6.3f}")
    p = p_next
```

Output:

```text
nu = 0.05 (half-life of price gap 13.5 q): impact depreciation 60.0% (overshooting ratio 6.00)
nu = 0.10 (half-life of price gap 6.6 q): impact depreciation 35.0% (overshooting ratio 3.50)
nu = 0.25 (half-life of price gap 2.4 q): impact depreciation 20.0% (overshooting ratio 2.00)
nu = 0.50 (half-life of price gap 1.0 q): impact depreciation 15.0% (overshooting ratio 1.50)
nu = 1.00 (half-life of price gap 0.0 q): impact depreciation 12.5% (overshooting ratio 1.25)
quarter | price level | exchange rate | i - i* (pp/quarter) | expected depreciation next quarter
   0    |     0.00    |     20.00     |       -2.500        |        -2.500
   4    |     6.84    |     13.16     |       -0.791        |        -0.791
   8    |     9.00    |     11.00     |       -0.250        |        -0.250
  12    |     9.68    |     10.32     |       -0.079        |        -0.079
  16    |     9.90    |     10.10     |       -0.025        |        -0.025
  20    |     9.97    |     10.03     |       -0.008        |        -0.008
```

On impact, prices are fixed and the exchange rate jumps by more than the 10% long-run depreciation: by 20% when the price gap
closes 25% a quarter, and by 60% when prices are very sticky. Higher real balances lower the domestic interest rate, and
uncovered interest parity then requires an expected **appreciation** of exactly the same size — the two columns of the
simulation coincide. The more sluggish prices are, the larger the overshooting. This explains why exchange rates are so much
more volatile than the fundamentals that drive them.

</details>

## Further reading
Obstfeld and Rogoff, *Foundations of International Macroeconomics*; Galí and Monacelli (2005).
