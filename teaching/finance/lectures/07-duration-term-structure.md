# Lecture 07 — Duration, Convexity, Immunization, and Term Structure

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Measure bond price sensitivity; derive duration and convexity approximations; understand immunization and term-structure models.

## 1. Duration
Macaulay duration is the present-value-weighted average time to cash flow:
\[
D_M=\frac{\sum_t t\,PV(CF_t)}{P}.
\]
For annual compounding, modified duration is \(D_{mod}=D_M/(1+y)\). For a small yield change,
\[
\frac{\Delta P}{P}\approx-D_{mod}\Delta y.
\]

## 2. Convexity
Convexity captures curvature:
\[
C=\frac{1}{P}\frac{\partial^2P}{\partial y^2}.
\]
The second-order approximation is
\[
\frac{\Delta P}{P}\approx-D_{mod}\Delta y+\frac12C(\Delta y)^2.
\]
Duration-convexity approximations are local and depend on the assumed yield-curve shift.

## 3. Key-rate duration
A parallel shift is restrictive. Key-rate durations measure sensitivity to changes at selected curve maturities, helping distinguish level, slope, and curvature exposures.

## 4. Immunization
A liability-driven portfolio can be immunized by matching present value and duration, with convexity and rebalancing conditions considered. Immunization is not a one-time guarantee under arbitrary nonparallel curve movements.

## 5. Term-structure theories
The expectations hypothesis links long yields to expected future short rates plus possible term premia. Affine term-structure models represent yields as functions of latent state variables and risk adjustments.

## 6. Worked example
For modified duration 6 and convexity 45, approximate the price change for a 50-basis-point yield increase. Compare duration-only and duration-convexity estimates.

## 7. Computational lab
Calculate duration, convexity, and key-rate exposures for a bond portfolio. Apply parallel and nonparallel curve shocks and compare linear and second-order estimates with full repricing.

## Problem set
1. Derive duration from the bond pricing function.
2. Explain convexity correction.
3. Compare parallel and key-rate risk.
4. State conditions for duration immunization.

## Coding exercises

These exercises use **real market data** in Python: monthly Fama–French factor returns (1926–2018), daily S&P 500 and
NASDAQ prices (1999–2018), the VIX, Moody's AAA and BAA corporate bond yields (1919–2018), all bundled with the `arch`
package, plus U.S. Treasury yields and household data from the `wooldridge` package. Install once with
`pip install arch wooldridge statsmodels scipy numpy pandas`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 7.1 — Duration and convexity against historical yield changes

*Data:* Moody's AAA corporate bond yield, monthly, 1919–2018.

Take a 20-year bond with a 6% annual coupon, priced at the AAA yield.

1. Compute its price, modified duration and convexity at the December 2018 AAA yield.
2. For every historical **monthly** change in the AAA yield, compute the exact price change and the duration and duration-plus-convexity approximations.
3. Report the mean absolute approximation error of each, and the error for the largest yield increase and decrease in the sample.

```python
import numpy as np, pandas as pd
from arch.data import default
y = default.load()["AAA"] / 100; dy = y.diff().dropna()
t = np.arange(1, 21); cf = np.r_[np.full(19, 6.0), 106.0]
P = lambda r: np.sum(cf / (1 + r)**t)
# TODO: D = -P'(y)/P, C = P''(y)/P; dP/P ~ -D dy + 0.5 C dy^2
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd
from arch.data import default
y = default.load()["AAA"] / 100; dy = y.diff().dropna()
t = np.arange(1, 21); cf = np.r_[np.full(19, 6.0), 106.0]
P = lambda r: np.sum(cf / (1 + r)**t)
y0 = y.iloc[-1]; p0 = P(y0)
D = np.sum(t * cf / (1 + y0)**(t + 1)) / p0; C = np.sum(t * (t + 1) * cf / (1 + y0)**(t + 2)) / p0
print(f"AAA yield Dec 2018 {100*y0:.2f}%: price {p0:.2f}, modified duration {D:.2f}, convexity {C:.1f}")
exact = np.array([P(y0 + h) / p0 - 1 for h in dy]); d1 = -D * dy.to_numpy(); d2 = d1 + 0.5 * C * dy.to_numpy()**2
print(f"mean |error|: duration {100*np.mean(np.abs(exact - d1)):.4f} pp, duration + convexity {100*np.mean(np.abs(exact - d2)):.5f} pp")
for lab, k in (("largest increase", dy.idxmax()), ("largest decrease", dy.idxmin())):
    i = dy.index.get_loc(k)
    print(f"{lab} ({k.strftime('%Y-%m')}, {100*dy.iloc[i]:+.2f} pp): exact {100*exact[i]:+.3f}%, duration {100*d1[i]:+.3f}%, + convexity {100*d2[i]:+.3f}%")
```

Output:

```text
AAA yield Dec 2018 4.02%: price 126.86, modified duration 12.57, convexity 213.9
mean |error|: duration 0.0330 pp, duration + convexity 0.00114 pp
largest increase (1980-02, +1.29 pp): exact -14.570%, duration -16.212%, + convexity -14.432%
largest decrease (1981-11, -1.18 pp): exact +16.439%, duration +14.829%, + convexity +16.318%
```

At a 4% yield, the 20-year 6% bond has a modified duration of about 12.6 and convexity of about 214. Over all monthly yield
changes since 1919, the duration approximation misses by 0.03 percentage points on average, and adding convexity cuts the
error by a factor of about 30. For the largest moves (around ±1.2–1.3 points in 1980–81), duration alone is off by about
1.6 points of price, always **overstating** losses and **understating** gains; with convexity, the error falls to about 0.1
points. Convexity is valuable to bondholders for exactly this reason.

</details>

### Exercise 7.2 — Immunizing a liability with two bonds

*Data:* Moody's AAA yields (for the level of rates and historical yield shocks).

A pension fund owes \$1 million in 10 years. It can buy a 2-year zero-coupon bond and a 30-year zero-coupon bond, priced at a flat yield equal to the December 2018 AAA yield.

1. Choose holdings so that the portfolio's present value and duration match the liability's (two equations, two unknowns).
2. Shock the yield curve with every historical 12-month change in the AAA yield (parallel shifts). Report the distribution of the surplus (assets minus liability, in present value) for the immunized portfolio and for an unhedged portfolio holding only the 2-year bond.
3. Why does the immunized surplus stay **positive** for both large rises and large falls? (Hint: compare convexities.)

```python
import numpy as np
from arch.data import default
y = default.load()["AAA"] / 100; y0 = y.iloc[-1]; shocks = (y - y.shift(12)).dropna().to_numpy()
pv = lambda T, r: (1 + r)**-T
# TODO: solve [pv2, pv30; 2 pv2, 30 pv30] [x2, x30] = [L pv10, 10 L pv10]
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np
from arch.data import default
y = default.load()["AAA"] / 100; y0 = y.iloc[-1]; shocks = (y - y.shift(12)).dropna().to_numpy()
pv = lambda T, r: (1 + r)**-T; L = 1e6
A = np.array([[pv(2, y0), pv(30, y0)], [2 * pv(2, y0), 30 * pv(30, y0)]])
x2, x30 = np.linalg.solve(A, [L * pv(10, y0), 10 * L * pv(10, y0)])
print(f"yield {100*y0:.2f}%: buy face ${x2:,.0f} of the 2-year zero and ${x30:,.0f} of the 30-year zero (liability PV ${L*pv(10, y0):,.0f})")
x2_only = L * pv(10, y0) / pv(2, y0)
for lab, f in (("immunized", lambda r: x2 * pv(2, r) + x30 * pv(30, r)), ("2-year only", lambda r: x2_only * pv(2, r))):
    surplus = np.array([f(y0 + s) - L * pv(10, y0 + s) for s in shocks])
    print(f"{lab:11s}: surplus over {len(shocks)} historical 12-month shocks: min ${surplus.min():,.0f}, median ${np.median(surplus):,.0f}, max ${surplus.max():,.0f}")
```

Output:

```text
yield 4.02%: buy face $521,119 of the 2-year zero and $628,447 of the 30-year zero (liability PV $674,266)
immunized  : surplus over 1188 historical 12-month shocks: min $-0, median $469, max $108,828
2-year only: surplus over 1188 historical 12-month shocks: min $-232,242, median $-2,601, max $149,771
```

Matching present value and duration with a 2-year and a 30-year zero ("barbell") immunizes the liability: across all 1,188
historical 12-month parallel shocks, the surplus never falls below zero, and it rises for large moves in either direction.
The barbell has more convexity than the 10-year liability, so it gains more than the liability when rates fall and loses
less when they rise. The unhedged 2-year position can lose over \$230,000. Real immunization must also handle
non-parallel shifts (slope and curvature), which a barbell is exposed to.

</details>

## Further reading
Tuckman and Serrat, *Fixed Income Securities*; Diebold and Li (2006).
