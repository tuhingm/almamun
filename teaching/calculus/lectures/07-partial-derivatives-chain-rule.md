# Lecture 07 — Partial Derivatives, Total Differentials, and the Chain Rule

## Learning objectives
Compute partial derivatives; interpret gradients and directional derivatives; derive total differentials; apply multivariate chain rules and Jacobians.

## 1. Partial derivatives
For \(f(x,y)\), the partial derivative \(f_x\) differentiates with respect to \(x\) holding \(y\) fixed. Partial derivatives describe coordinate-wise local responses. Existence of partial derivatives alone does not guarantee differentiability.

## 2. Gradient and directional derivative
The gradient is \(\nabla f=(f_{x_1},\ldots,f_{x_n})'\). For a unit direction \(u\), the directional derivative is \(D_uf=\nabla f'u\). Under the Euclidean norm, the gradient points in the direction of steepest local increase.

## 3. Differentiability and total differential
A function is differentiable at \(x\) if it admits a linear approximation with remainder small relative to the distance:
\[
f(x+h)=f(x)+\nabla f(x)'h+o(\|h\|).
\]
The total differential is \(df=\sum_i f_{x_i}dx_i\). It approximates changes locally; it is not generally an exact finite-change identity.

## 4. Chain rule
If \(z=f(x,y)\), with \(x=x(t)\), \(y=y(t)\), then
\[
\frac{dz}{dt}=f_x\frac{dx}{dt}+f_y\frac{dy}{dt}.
\]
For vector maps, Jacobian matrices multiply in composition order: \(D(f\circ g)=Df(g(x))Dg(x)\).

## 5. Jacobian and transformations
For a vector-valued map \(y=g(x)\), the Jacobian is \(J_g(x)=\partial g/\partial x'\). A nonsingular Jacobian indicates local invertibility under the inverse function theorem. In probability transformations, its determinant adjusts volume.

## Worked example
For \(Q(K,L)=AK^\alpha L^\beta\), \(Q_K=\alpha AK^{\alpha-1}L^\beta\) and \(Q_L=\beta AK^\alpha L^{\beta-1}\). The total differential is \(dQ=Q_KdK+Q_LdL\), giving a first-order approximation to output changes.

## Problem set
1. Compute gradient and Hessian of a quadratic function.
2. Derive directional derivative in a specified direction.
3. Apply the chain rule to a composed production function.
4. Distinguish partial differentiability from total differentiability.
5. Calculate a Jacobian determinant for a coordinate transformation.

## Computational lab
Use SymPy to derive gradients and Jacobians; verify with finite differences and plot gradient fields over contour maps.

## Coding exercises

These exercises apply the calculus of this lecture to **real economic data** in Python, mixing symbolic (`sympy`),
numerical (`scipy`) and statistical (`statsmodels`) tools. Install once with
`pip install sympy scipy numpy pandas statsmodels wooldridge`. Try the starter code first, then open the solution.
Every printed result below comes from running the solution code exactly as shown.

### Exercise 7.1 — Growth accounting with the chain rule

*Data:* `macrodata` — U.S. quarterly real GDP, population and CPI.

1. Use the chain rule on \(\log Y=\log(Y/N)+\log N\) to decompose real GDP growth into per-capita growth and population growth, decade by decade.
2. Similarly decompose **nominal** GDP growth into real growth and inflation (CPI-based), using \(\log(PY)=\log P+\log Y\).
3. Verify that the decompositions add up exactly in logs, and show the small error when using percent changes instead.

```python
import numpy as np, pandas as pd, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
# TODO: annual averages; log differences of Y, Y/N, N, P; group by decade
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
a = m.groupby("year")[["realgdp", "pop", "cpi"]].mean().loc[1960:2008]
lg = np.log(a).diff().dropna() * 100
lg["percap"] = lg["realgdp"] - lg["pop"]; lg["nominal"] = lg["realgdp"] + lg["cpi"]
lg["decade"] = (lg.index // 10) * 10
t = lg.groupby("decade")[["realgdp", "percap", "pop", "cpi", "nominal"]].mean().round(2)
print("average annual log growth (%), by decade:"); print(t.to_string())
pc = a.pct_change().dropna() * 100
err = (pc["realgdp"] - ((1 + pc["realgdp"]/100) / (1 + pc["pop"]/100) - 1) * 100 - pc["pop"]).abs().max()
print(f"\nmax error if percent changes are simply added (per-capita + population = total): {err:.3f} pp")
```

Output:

```text
average annual log growth (%), by decade:
        realgdp  percap   pop   cpi  nominal
decade                                      
1960.0     4.55    3.28  1.27  2.48     7.02
1970.0     3.18    2.13  1.05  6.94    10.11
1980.0     2.98    2.04  0.94  5.23     8.20
1990.0     3.13    1.91  1.22  2.92     6.04
2000.0     2.34    1.39  0.96  2.80     5.14

max error if percent changes are simply added (per-capita + population = total): 0.068 pp
```

In logs, the decomposition is exact: real GDP growth equals per-capita growth plus population growth, and nominal growth
equals real growth plus inflation. The table tells a familiar story: the 1960s had the fastest real growth; the 1970s had
nominal growth above 10% driven by inflation of almost 7%; and per-capita growth slowed steadily to about 1.4% in the
2000s. Adding percent changes instead of log changes introduces a small error (up to 0.07 percentage points a year), from
the cross term in \((1+a)(1+b)=1+a+b+ab\).

</details>

### Exercise 7.2 — Partial elasticities and a total derivative in a hedonic price function

*Data:* `hprice1` — 88 house sales.

1. Fit \(\log(price)=\beta_0+\beta_1\log(sqrft)+\beta_2 bdrms+\beta_3\log(lotsize)\).
2. Compute the partial effect of one more bedroom, holding square footage fixed. Why can it be small or even negative?
3. In the data, bigger houses have more bedrooms: estimate the auxiliary relation \(bdrms=\gamma_0+\gamma_1\log(sqrft)\). Use the chain rule to compute the **total** derivative \(\frac{d\log price}{d\log sqrft}=\beta_1+\beta_2\gamma_1\) and compare it with the slope from regressing \(\log price\) on \(\log sqrft\) and \(\log lotsize\) only.

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
h = wd.data("hprice1")
full = smf.ols("np.log(price) ~ np.log(sqrft) + bdrms + np.log(lotsize)", data=h).fit()
# TODO: aux = bdrms ~ log(sqrft) + log(lotsize); total = b1 + b2 * g1
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
h = wd.data("hprice1")
full = smf.ols("np.log(price) ~ np.log(sqrft) + bdrms + np.log(lotsize)", data=h).fit()
aux = smf.ols("bdrms ~ np.log(sqrft) + np.log(lotsize)", data=h).fit()
short = smf.ols("np.log(price) ~ np.log(sqrft) + np.log(lotsize)", data=h).fit()
b1, b2, g1 = full.params["np.log(sqrft)"], full.params["bdrms"], aux.params["np.log(sqrft)"]
print(f"partial effects: d log price / d log sqrft = {b1:.4f};  one more bedroom (same size) = {b2:.4f} ({100*(np.exp(b2)-1):.1f}%)")
print(f"auxiliary: d bdrms / d log sqrft = {g1:.4f}")
print(f"total derivative b1 + b2*g1 = {b1 + b2*g1:.4f};  short-regression slope = {short.params['np.log(sqrft)']:.4f}")
```

Output:

```text
partial effects: d log price / d log sqrft = 0.7002;  one more bedroom (same size) = 0.0370 (3.8%)
auxiliary: d bdrms / d log sqrft = 1.6813
total derivative b1 + b2*g1 = 0.7624;  short-regression slope = 0.7624
```

Holding size fixed, one more bedroom raises price by only about 4% (and insignificantly): at a fixed floor area, an extra
bedroom means smaller rooms. But bigger houses have more bedrooms, so the **total** derivative of log price with respect to
log size, \(\beta_1+\beta_2\gamma_1\approx0.76\), exceeds the partial effect of 0.70. It equals the short-regression slope
exactly — the chain rule here is the omitted-variable bias formula of Econometrics Lecture 3, read the other way round.

</details>

## Reading
Apostol, *Calculus*, Vol. 2; Simon and Blume, *Mathematics for Economists*; Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*.