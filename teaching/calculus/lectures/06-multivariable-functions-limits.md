# Lecture 06 — Multivariable Functions, Limits, and Continuity

## Learning objectives
Analyze domains and level sets; define multivariate limits and continuity; distinguish pathwise behavior from genuine limits; interpret surfaces and economic production functions.

## 1. Functions of several variables
A scalar function \(f:D\subseteq\mathbb R^n\to\mathbb R\) maps a vector input to a scalar output. Its domain, range, level sets, and contour surfaces describe its geometry. In economics, examples include utility, production, cost, and welfare functions.

## 2. Multivariate limits
The limit \(\lim_{x\to a}f(x)=L\) requires that every sequence approaching \(a\) within the domain produces function values approaching \(L\). A limit must be independent of the path. Showing two paths with different limiting values disproves existence; checking a finite number of paths cannot generally prove existence.

## 3. Continuity
A function is continuous at \(a\) if its multivariate limit equals \(f(a)\). Polynomials are continuous everywhere; rational functions are continuous wherever denominators are nonzero. Compositions preserve continuity under compatible domains.

## 4. Open and closed sets
A ball around a point defines a neighborhood. Open sets contain a neighborhood around every point; closed sets contain their limit points. Compact subsets of Euclidean space are closed and bounded. Continuity on a compact set implies boundedness and attainment of maximum and minimum.

## 5. Level sets and economic interpretation
A level set \(\{x:f(x)=c\}\) defines an indifference curve or isoquant in common applications. The gradient, developed in the next lecture, is normal to a regular level set. Curvature and feasibility depend on the domain and constraints.

## Worked example
Consider \(f(x,y)=xy/(x^2+y^2)\) away from the origin. Along \(y=x\), the value is \(1/2\); along \(y=-x\), it is \(-1/2\). Therefore the limit at the origin does not exist.

## Problem set
1. Determine the domain of \(\log(x-y)\).
2. Test a multivariate limit along two paths.
3. Explain why checking coordinate axes alone is insufficient.
4. Describe level sets of a Cobb–Douglas production function.
5. State the extreme value theorem on compact subsets of Euclidean space.

## Computational lab
Plot surfaces and contour maps for quadratic, Cobb–Douglas, and non-convergent examples. Numerically explore path dependence near a candidate limit.

## Coding exercises

These exercises apply the calculus of this lecture to **real economic data** in Python, mixing symbolic (`sympy`),
numerical (`scipy`) and statistical (`statsmodels`) tools. Install once with
`pip install sympy scipy numpy pandas statsmodels wooldridge`. Try the starter code first, then open the solution.
Every printed result below comes from running the solution code exactly as shown.

### Exercise 6.1 — The CES production function and its Cobb–Douglas limit

*Data:* We calibrate to the U.S. labor share of about 2/3 (capital share \(\alpha=1/3\)) and evaluate at the capital–output ratio implied by `macrodata` investment data.

The CES function is \(Y=\big(\alpha K^{\rho}+(1-\alpha)L^{\rho}\big)^{1/\rho}\), with elasticity of substitution \(\sigma=1/(1-\rho)\).

1. Show with `sympy` that \(\lim_{\rho\to0}Y=K^{\alpha}L^{1-\alpha}\) (take logs and use L'Hôpital's rule).
2. Evaluate the CES function at \(K=3\), \(L=1\) for \(\rho\) approaching 0 from both sides, and show convergence to the Cobb–Douglas value.
3. Compute the capital share \(\frac{\partial Y}{\partial K}\frac{K}{Y}\) as \(\rho\) varies. Why does the Cobb–Douglas case imply a constant factor share, and what does the U.S. evidence on a falling labor share suggest about \(\sigma\)?

```python
import sympy as sp
K, L, a, r = sp.symbols("K L alpha rho", positive=True)
logY = sp.log(a * K**r + (1 - a) * L**r) / r
# TODO: sp.limit(logY, r, 0)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, sympy as sp
K, L, a = sp.symbols("K L alpha", positive=True); r = sp.symbols("rho", real=True)
logY = sp.log(a * K**r + (1 - a) * L**r) / r
lim = sp.simplify(sp.limit(logY, r, 0))
print("lim_{rho->0} log Y =", lim, " -> Y = K^alpha L^(1-alpha)")
ces = lambda rho, k=3.0, l=1.0, al=1/3: (al * k**rho + (1 - al) * l**rho)**(1 / rho)
print(f"Cobb-Douglas value at K=3, L=1: {3**(1/3):.8f}")
for rho in (-0.5, -0.1, -0.01, -0.001, 0.001, 0.01, 0.1, 0.5):
    Y = ces(rho); share = (1/3) * 3**rho / ((1/3) * 3**rho + 2/3)
    print(f"  rho = {rho:+.3f} (sigma = {1/(1-rho):.3f}): Y = {Y:.8f}, capital share = {share:.4f}")
```

Output:

```text
lim_{rho->0} log Y = log(K**alpha*L**(1 - alpha))  -> Y = K^alpha L^(1-alpha)
Cobb-Douglas value at K=3, L=1: 1.44224957
  rho = -0.500 (sigma = 0.667): Y = 1.35486374, capital share = 0.2240
  rho = -0.100 (sigma = 0.909): Y = 1.42327646, capital share = 0.3094
  rho = -0.010 (sigma = 0.990): Y = 1.44031910, capital share = 0.3309
  rho = -0.001 (sigma = 0.999): Y = 1.44205619, capital share = 0.3331
  rho = +0.001 (sigma = 1.001): Y = 1.44244302, capital share = 0.3336
  rho = +0.010 (sigma = 1.010): Y = 1.44418736, capital share = 0.3358
  rho = +0.100 (sigma = 1.111): Y = 1.46195368, capital share = 0.3582
  rho = +0.500 (sigma = 2.000): Y = 1.54757814, capital share = 0.4641
```

`sympy` evaluates the 0/0 limit of \(\log Y\) and returns the Cobb–Douglas form, and the numerical values converge to
\(3^{1/3}\) from both sides. The capital share is constant at \(\alpha=1/3\) only in the Cobb–Douglas case: with
\(\sigma<1\) (\(\rho<0\)), a high capital–labor ratio lowers capital's share, and with \(\sigma>1\) it raises it. A falling
U.S. labor share alongside capital deepening is therefore consistent with \(\sigma>1\) — one explanation debated in the
literature (Karabarbounis and Neiman, 2014); others stress markups and automation.

</details>

### Exercise 6.2 — Level sets of an estimated two-input investment function

*Data:* `grunfeld` (statsmodels) — investment, firm market value and capital stock of 11 large U.S. firms, 1935–1954.

1. Estimate the log-linear investment function \(\log I=\beta_0+\beta_1\log V+\beta_2\log K\) (pooled OLS).
2. The fitted function \(I(V,K)=e^{\beta_0}V^{\beta_1}K^{\beta_2}\) is a function of two variables. Find the level curve on which predicted investment equals the sample median, solving for \(K\) as a function of \(V\).
3. Compute the slope \(dK/dV\) of that level curve at the median firm-year (the marginal rate of substitution between market value and capital in generating investment).

```python
import numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
g = sm.datasets.grunfeld.load_pandas().data.query("invest > 0 and capital > 0")
r = smf.ols("np.log(invest) ~ np.log(value) + np.log(capital)", data=g).fit()
# TODO: K(V) on the level set: K = (I0 / (e^b0 V^b1))^(1/b2); slope -(b1 K)/(b2 V)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
g = sm.datasets.grunfeld.load_pandas().data.query("invest > 0 and capital > 0")
r = smf.ols("np.log(invest) ~ np.log(value) + np.log(capital)", data=g).fit()
b0, b1, b2 = r.params
print(f"log I = {b0:.3f} + {b1:.3f} log V + {b2:.3f} log K   (R2 {r.rsquared:.3f})")
I0, V0, K0 = g["invest"].median(), g["value"].median(), g["capital"].median()
Kcurve = lambda V: (I0 / (np.exp(b0) * V**b1))**(1 / b2)
for V in (0.5 * V0, V0, 2 * V0):
    print(f"  on the level set I = {I0:.1f}: V = {V:8.1f} -> K = {Kcurve(V):8.1f}")
print(f"slope dK/dV of the level curve at V = {V0:.1f}: {-(b1 * Kcurve(V0)) / (b2 * V0):.4f}")
```

Output:

```text
log I = -2.777 + 0.803 log V + 0.375 log K   (R2 0.880)
  on the level set I = 52.4: V =    202.3 -> K =    720.7
  on the level set I = 52.4: V =    404.6 -> K =    163.6
  on the level set I = 52.4: V =    809.3 -> K =     37.1
slope dK/dV of the level curve at V = 404.6: -0.8650
```

The fitted function explains 88% of the variation in log investment. On the level curve through the median investment,
doubling market value lets predicted investment stay constant with much less capital: the curve is steep and convex, as for
any Cobb–Douglas level set. The slope at the median value says that \$1 more market value substitutes for about \$0.87
less capital in generating the same predicted investment. This is a descriptive relation across firms, not a production
technology.

</details>

## Reading
Apostol, *Calculus*, Vol. 2; Simon and Blume, *Mathematics for Economists*; Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*.