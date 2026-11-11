# Lecture 02 — Sequences, Convergence, and Infinite Series

## Learning objectives
Define convergence rigorously; use monotone convergence and Cauchy criteria; test infinite series; distinguish absolute and conditional convergence; analyze power series.

## 1. Sequences
A sequence \((a_n)\) converges to \(L\) if for every \(\varepsilon>0\), there exists \(N\) such that \(n\ge N\) implies \(|a_n-L|<\varepsilon\). Limits are unique and convergent sequences are bounded. A sequence is Cauchy if its terms become arbitrarily close to each other; completeness of \(\mathbb R\) ensures every Cauchy sequence converges.

## 2. Monotone convergence
Every monotone increasing sequence bounded above converges to its supremum. Every monotone decreasing sequence bounded below converges to its infimum. This result is useful for recursive sequences and iterative economic processes.

## 3. Infinite series
The series \(\sum_{n=1}^\infty a_n\) converges when its partial sums \(S_N=\sum_{n=1}^N a_n\) converge. A necessary condition is \(a_n\to0\), but this is not sufficient. The geometric series has sum \(a/(1-r)\) for \(|r|<1\).

## 4. Convergence tests
The p-series \(\sum n^{-p}\) converges iff \(p>1\). Use comparison and limit comparison for nonnegative terms. The ratio test uses \(L=\lim|a_{n+1}/a_n|\): convergence if \(L<1\), divergence if \(L>1\), and no conclusion if \(L=1\). The root test uses \(\limsup |a_n|^{1/n}\). Alternating-series convergence follows when positive terms decrease to zero.

## 5. Absolute and conditional convergence
Absolute convergence means \(\sum|a_n|\) converges and implies convergence of \(\sum a_n\). Conditional convergence occurs when the original series converges but its absolute-value series diverges. Rearrangement can change the sum of a conditionally convergent real series.

## 6. Power series
A power series \(\sum a_n(x-c)^n\) has radius of convergence \(R\). It converges absolutely for \(|x-c|<R\), diverges for \(|x-c|>R\), and endpoints must be checked separately. Termwise differentiation and integration are valid inside the interval of convergence.

## Worked example
The integral test compares \(\sum n^{-p}\) with \(\int_1^\infty x^{-p}dx\), finite exactly when \(p>1\). Thus the p-series converges iff \(p>1\).

## Problem set
1. Prove uniqueness of a sequence limit.
2. Apply monotone convergence to a bounded recursive sequence.
3. Test \(\sum 1/[n(n+1)]\) and find its sum.
4. Determine convergence of \(\sum(-1)^{n+1}/n\).
5. Find the radius and endpoint behavior of a power series.

## Computational lab
Plot partial sums for geometric, harmonic, p-series, and alternating harmonic series. Compare numerical behavior with analytic tests.

## Coding exercises

These exercises apply the calculus of this lecture to **real economic data** in Python, mixing symbolic (`sympy`),
numerical (`scipy`) and statistical (`statsmodels`) tools. Install once with
`pip install sympy scipy numpy pandas statsmodels wooldridge`. Try the starter code first, then open the solution.
Every printed result below comes from running the solution code exactly as shown.

### Exercise 2.1 — Log differences versus percent changes: a Taylor series for growth rates

*Data:* `macrodata` (statsmodels) — U.S. quarterly real GDP and CPI, 1959–2009.

Economists often measure growth as \(\Delta\log y\) instead of \(\Delta y/y\). Since \(\log(1+g)=g-g^2/2+g^3/3-\dots\),
the two differ at second order.

1. Compute quarterly percent changes and log differences of real GDP and of the CPI. Report the largest gap between them.
2. Show that adding the second-order term, \(g-g^2/2\), removes most of the gap.
3. Annualize quarterly growth in two ways — \(4\times\) the quarterly rate and \((1+g)^4-1\) — and compare for the most extreme quarter.

```python
import numpy as np, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
y = m["realgdp"].to_numpy()
g = y[1:] / y[:-1] - 1; lg = np.diff(np.log(y))
# TODO: max |g - lg|; max |g - g**2/2 - lg|; annualization
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
for name in ("realgdp", "cpi"):
    y = m[name].to_numpy(); g = y[1:] / y[:-1] - 1; lg = np.diff(np.log(y))
    k = np.argmax(np.abs(g))
    print(f"{name}: largest |quarterly change| {100*g[k]:.3f}% (log diff {100*lg[k]:.3f}%)")
    print(f"   max |g - log| = {np.max(np.abs(g - lg))*100:.4f} pp;  max |g - g^2/2 - log| = {np.max(np.abs(g - g**2/2 - lg))*100:.5f} pp;  max |g - g^2/2 + g^3/3 - log| = {np.max(np.abs(g - g**2/2 + g**3/3 - lg))*100:.6f} pp")
    print(f"   annualized at that quarter: 4g = {400*g[k]:.2f}%,  (1+g)^4 - 1 = {100*((1+g[k])**4-1):.2f}%,  4*log diff = {400*lg[k]:.2f}%")
```

Output:

```text
realgdp: largest |quarterly change| 3.934% (log diff 3.859%)
   max |g - log| = 0.0754 pp;  max |g - g^2/2 - log| = 0.00197 pp;  max |g - g^2/2 + g^3/3 - log| = 0.000058 pp
   annualized at that quarter: 4g = 15.74%,  (1+g)^4 - 1 = 16.69%,  4*log diff = 15.43%
cpi: largest |quarterly change| 3.723% (log diff 3.656%)
   max |g - log| = 0.0676 pp;  max |g - g^2/2 - log| = 0.00167 pp;  max |g - g^2/2 + g^3/3 - log| = 0.000047 pp
   annualized at that quarter: 4g = 14.89%,  (1+g)^4 - 1 = 15.75%,  4*log diff = 14.62%
```

Quarterly changes are small, so log differences and percent changes differ by at most 0.08 percentage points. Adding the
\(-g^2/2\) term cuts the gap by a factor of about 40, and the cubic term by another 30 — the Taylor series converges fast
for small \(g\). Annualization matters more: for GDP's strongest quarter, simple multiplication by four gives 15.7%, true
compounding 16.7%, and four times the log difference 15.4%. Always state which convention a reported growth rate uses.

</details>

### Exercise 2.2 — Present values as geometric series: pricing a perpetuity and a bond

*Data:* `intqrt` — U.S. Treasury yields; we use the sample's average 12-month yield `r12` as a discount rate.

1. A perpetuity pays \$100 per year forever. Write its value as a geometric series and sum it with `sympy`.
2. Compute the partial sums for 10, 30 and 100 years. How many years of payments make up 90% of the perpetuity's value?
3. Price a 10-year bond with a 5% annual coupon and face value \$1,000 at the average yield, using the closed form for a finite geometric series. Check it with a direct sum.

```python
import numpy as np, sympy as sp, wooldridge as wd
r = wd.data("intqrt")["r12"].mean() / 100
t, R = sp.symbols("t R", positive=True)
# TODO: sp.summation(100 / (1 + R)**t, (t, 1, sp.oo)); partial sums; bond price
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, sympy as sp, wooldridge as wd
r = wd.data("intqrt")["r12"].mean() / 100
t, R = sp.symbols("t R", positive=True)
pv = sp.simplify(sp.summation(100 / (1 + R)**t, (t, 1, sp.oo)))
print(f"discount rate r = {100*r:.3f}%;  symbolic perpetuity value: {pv}  ->  ${float(pv.subs(R, r)):,.2f}")
total = 100 / r
for T in (10, 30, 100):
    s = sum(100 / (1 + r)**k for k in range(1, T + 1))
    print(f"  first {T:3d} years: ${s:,.2f} ({s/total:.1%} of the perpetuity)")
T90 = np.log(10) / np.log(1 + r)
print(f"years needed for 90% of the value: {T90:.1f}")
c, F, n = 50, 1000, 10
closed = c * (1 - (1 + r)**-n) / r + F / (1 + r)**n
direct = sum(c / (1 + r)**k for k in range(1, n + 1)) + F / (1 + r)**n
print(f"10-year 5% coupon bond: closed form ${closed:,.2f}, direct sum ${direct:,.2f}")
```

Output:

```text
discount rate r = 6.927%;  symbolic perpetuity value: 100/R  ->  $1,443.64
  first  10 years: $704.74 (48.8% of the perpetuity)
  first  30 years: $1,250.07 (86.6% of the perpetuity)
  first 100 years: $1,441.86 (99.9% of the perpetuity)
years needed for 90% of the value: 34.4
10-year 5% coupon bond: closed form $864.20, direct sum $864.20
```

`sympy` sums the geometric series to \(100/R\), about \$1,444 at a 6.9% discount rate. The first 10 years contribute less
than half of that value, and it takes about 34 years of payments to reach 90%: distant payments matter a lot when rates are
moderate, which is why long-duration assets are so sensitive to interest rates. The closed form for the finite geometric
series prices the 5% coupon bond at \$864, below par because the coupon is below the yield.

</details>

## Reading
Apostol, *Calculus*, Vol. 1; Spivak, *Calculus*; Rudin, *Principles of Mathematical Analysis*.