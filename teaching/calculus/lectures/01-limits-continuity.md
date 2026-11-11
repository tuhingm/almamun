# Lecture 01 — Limits, Continuity, and the Logic of Analysis

**Level:** Undergraduate with graduate mathematical extensions

## Learning objectives
Use epsilon-delta definitions; evaluate limits; distinguish pointwise values from limiting behavior; establish continuity; apply the intermediate and extreme value theorems.

## 1. Functions and domains
A function \(f:D\to\mathbb R\) assigns one output to each input in its domain. Domain restrictions arise from denominators, even roots, logarithms, and feasibility constraints. Composition and inverse functions require compatible domains; an inverse requires injectivity on the selected domain.

## 2. Limits
We say \(\lim_{x\to a}f(x)=L\) if for every \(\varepsilon>0\), there exists \(\delta>0\) such that
\[
0<|x-a|<\delta\implies |f(x)-L|<\varepsilon.
\]
The puncture excludes the value at \(a\): the limit concerns nearby behavior. Left and right limits must agree for a two-sided limit to exist. Limit laws follow from the epsilon-delta definition.

## 3. Techniques
Use algebraic simplification, factorization, rationalization, squeeze arguments, and standard trigonometric or exponential limits. For rational functions at infinity, compare leading powers. L'Hôpital's rule applies only to indeterminate forms such as \(0/0\) or \(\infty/\infty\), under its differentiability and denominator-derivative conditions.

## 4. Continuity
A function is continuous at \(a\) if it is defined there and \(\lim_{x\to a}f(x)=f(a)\). Removable, jump, and infinite discontinuities differ in limiting behavior. Sums, products, and compositions of continuous functions are continuous wherever defined.

## 5. Fundamental theorems
The intermediate value theorem states that a continuous function on \([a,b]\) attains every value between \(f(a)\) and \(f(b)\). The extreme value theorem guarantees attainment of a maximum and minimum on a compact interval. These results establish existence, not uniqueness.

## Worked example
For \(f(x)=(x^2-1)/(x-1)\), \(x\ne1\), factor to obtain \(f(x)=x+1\). Thus \(\lim_{x\to1}f(x)=2\), although the original expression is undefined at 1. Defining \(f(1)=2\) removes the discontinuity.

## Problem set
1. Prove directly that \(\lim_{x\to2}(3x+1)=7\).
2. Evaluate \(\lim_{x\to0}(\sqrt{1+x}-1)/x\).
3. Give examples of removable, jump, and infinite discontinuities.
4. State the assumptions of the intermediate value theorem.
5. Explain why continuity on an open interval does not guarantee an attained maximum.

## Computational lab
Plot functions with removable and jump discontinuities. Compare numerical limits at decreasing step sizes with analytic limits; explain why numerical evidence is not a proof.

## Coding exercises

These exercises apply the calculus of this lecture to **real economic data** in Python, mixing symbolic (`sympy`),
numerical (`scipy`) and statistical (`statsmodels`) tools. Install once with
`pip install sympy scipy numpy pandas statsmodels wooldridge`. Try the starter code first, then open the solution.
Every printed result below comes from running the solution code exactly as shown.

### Exercise 1.1 — From discrete to continuous compounding with real interest rates

*Data:* `intqrt` — U.S. 3-month Treasury bill yields (`r3`, % per year), quarterly.

1. Take the average T-bill yield \(r\) in the sample. Compute the growth factor of \$1 after one year, \((1+r/n)^n\), for \(n=1,2,4,12,365,10^6\).
2. Show numerically that the sequence converges to \(e^r\), and that the error behaves like \(r^2e^r/(2n)\).
3. Use `sympy` to prove the limit symbolically.
4. Why does the choice between annual and continuous compounding matter more when rates are high (as in 1981)?

```python
import numpy as np, sympy as sp, wooldridge as wd
r = wd.data("intqrt")["r3"].mean() / 100
for n in (1, 2, 4, 12, 365, 10**6):
    pass  # TODO: (1 + r/n)**n and the gap to np.exp(r)
x, n = sp.symbols("x n", positive=True)
# TODO: sp.limit((1 + x/n)**n, n, sp.oo)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, sympy as sp, wooldridge as wd
d = wd.data("intqrt"); r = d["r3"].mean() / 100
print(f"average 3-month T-bill yield: {100*r:.3f}% per year")
for n in (1, 2, 4, 12, 365, 10**6):
    g = (1 + r / n)**n
    print(f"n = {n:>7}: (1+r/n)^n = {g:.8f}   gap to e^r = {np.exp(r) - g:.2e}   approx r^2 e^r/(2n) = {r**2*np.exp(r)/(2*n):.2e}")
x, n = sp.symbols("x n", positive=True)
print("sympy limit:", sp.limit((1 + x / n)**n, n, sp.oo))
rmax = d["r3"].max() / 100
print(f"at the sample's peak yield {100*rmax:.2f}%: annual {1+rmax:.5f} vs continuous {np.exp(rmax):.5f} (gap {100*(np.exp(rmax)-1-rmax):.2f} pp)")
```

Output:

```text
average 3-month T-bill yield: 6.450% per year
n =       1: (1+r/n)^n = 1.06449839   gap to e^r = 2.13e-03   approx r^2 e^r/(2n) = 2.22e-03
n =       2: (1+r/n)^n = 1.06553840   gap to e^r = 1.09e-03   approx r^2 e^r/(2n) = 1.11e-03
n =       4: (1+r/n)^n = 1.06607524   gap to e^r = 5.49e-04   approx r^2 e^r/(2n) = 5.55e-04
n =      12: (1+r/n)^n = 1.06643965   gap to e^r = 1.84e-04   approx r^2 e^r/(2n) = 1.85e-04
n =     365: (1+r/n)^n = 1.06661778   gap to e^r = 6.08e-06   approx r^2 e^r/(2n) = 6.08e-06
n = 1000000: (1+r/n)^n = 1.06662386   gap to e^r = 2.28e-09   approx r^2 e^r/(2n) = 2.22e-09
sympy limit: exp(x)
at the sample's peak yield 16.46%: annual 1.16460 vs continuous 1.17892 (gap 1.43 pp)
```

At the sample's average yield of 6.45%, the growth factor rises from 1.0645 (annual compounding) toward \(e^{0.0645}=1.0666\).
The gap shrinks in proportion to \(1/n\), exactly as the second-order expansion \(r^2e^r/(2n)\) predicts: doubling \(n\)
halves the error. `sympy` confirms \(\lim_{n\to\infty}(1+x/n)^n=e^x\). The difference between annual and continuous
compounding is of order \(r^2/2\), so it grows quadratically with the rate: at the 16.5% peak of 1981 it was 1.4 percentage
points a year — enough to matter when quoting or comparing yields.

</details>

### Exercise 1.2 — Arc elasticity converges to point elasticity: demand for eco-labeled apples

*Data:* `apple` — 660 households in a 1990s survey that randomly assigned prices of eco-labeled (`ecoprc`) and regular (`regprc`) apples and recorded pounds demanded (`ecolbs`).

1. Estimate an exponential demand curve \(E[q\mid p]=\exp(\beta_0+\beta_1p_{eco}+\beta_2p_{reg})\) by Poisson regression (which handles the many zero purchases).
2. At the mean prices, compute the **arc elasticity** \(\frac{\Delta q/q}{\Delta p/p}\) for price changes of 20, 10, 1 and 0.1 cents. Show that it converges to the point elasticity \(\beta_1p\) as \(\Delta p\to0\), and that the error is roughly proportional to \(\Delta p\).
3. Why would the same exercise with a **linear** demand curve show no convergence at all?

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
a = wd.data("apple")
r = smf.poisson("ecolbs ~ ecoprc + regprc", data=a).fit()
q = lambda p: np.exp(r.params["Intercept"] + r.params["ecoprc"] * p + r.params["regprc"] * a["regprc"].mean())
# TODO: arc elasticities at p0 = a.ecoprc.mean() for dp in (0.2, 0.1, 0.01, 0.001); point elasticity b1 * p0
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
a = wd.data("apple")
r = smf.poisson("ecolbs ~ ecoprc + regprc", data=a).fit(disp=0, cov_type="HC0")
b0, b1, b2 = r.params; p0, preg = a["ecoprc"].mean(), a["regprc"].mean()
q = lambda p: np.exp(b0 + b1 * p + b2 * preg)
print(f"E[q|p] = exp({b0:.3f} {b1:+.3f} p_eco {b2:+.3f} p_reg)   (robust se on p_eco {r.bse['ecoprc']:.3f})")
point = b1 * p0
for dp in (0.20, 0.10, 0.01, 0.001):
    arc = ((q(p0 + dp) - q(p0)) / q(p0)) / (dp / p0)
    print(f"  dp = {dp:5.3f}: arc elasticity {arc:.5f}   error {arc - point:+.5f}   error/dp {(arc - point)/dp:+.3f}")
print(f"point elasticity b1 * p = {point:.5f}")
lin = smf.ols("ecolbs ~ ecoprc + regprc", data=a).fit()
ql = lambda p: lin.params["Intercept"] + lin.params["ecoprc"] * p + lin.params["regprc"] * preg
print("linear demand, arc elasticity for dp = 0.2 and 0.001:", round(((ql(p0+.2)-ql(p0))/ql(p0))/(.2/p0), 5), round(((ql(p0+.001)-ql(p0))/ql(p0))/(.001/p0), 5))
```

Output:

```text
E[q|p] = exp(0.683 -2.045 p_eco +2.108 p_reg)   (robust se on p_eco 0.355)
  dp = 0.200: arc elasticity -1.81493   error +0.39629   error/dp +1.981
  dp = 0.100: arc elasticity -1.99983   error +0.21140   error/dp +2.114
  dp = 0.010: arc elasticity -2.18877   error +0.02245   error/dp +2.245
  dp = 0.001: arc elasticity -2.20897   error +0.00226   error/dp +2.259
point elasticity b1 * p = -2.21123
linear demand, arc elasticity for dp = 0.2 and 0.001: -2.14725 -2.14725
```

At mean prices, the point elasticity of demand for eco-labeled apples is about \(-2.2\): demand is elastic. The arc
elasticity approaches it as the price step shrinks, and the error divided by \(\Delta p\) settles near a constant: the error
is first order in \(\Delta p\), as a Taylor expansion of \(e^{\beta_1\Delta p}\) predicts. For a linear demand curve, the
one-sided arc elasticity \((\Delta q/q)/(\Delta p/p)=\beta_1p/q\) is exact for every step size, so there is nothing to
converge — the last line shows the same number for 20 cents and 0.1 cent. Prices were randomized in this survey, so the
slope can be read as a causal demand response for stated purchases.

</details>

## Reading
Apostol, *Calculus*, Vol. 1; Spivak, *Calculus*; Rudin, *Principles of Mathematical Analysis*.