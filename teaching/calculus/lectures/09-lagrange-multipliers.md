# Lecture 09 — Equality Constraints and Lagrange Multipliers

## Learning objectives
Derive Lagrange conditions; interpret multipliers as shadow values; solve constrained optimization problems; understand regularity and second-order conditions.

## 1. Equality-constrained problem
Consider maximizing \(f(x)\) subject to \(g(x)=c\). Define
\[
\mathcal L(x,\lambda)=f(x)+\lambda[c-g(x)].
\]
At a regular interior constrained optimum, stationarity and feasibility require \(\nabla f(x^*)-\lambda\nabla g(x^*)=0\) and \(g(x^*)=c\). The constraint qualification requires a nonzero constraint gradient in the single-constraint case.

## 2. Geometric interpretation
At an interior constrained optimum, the objective gradient is parallel to the constraint gradient. The objective's level surface is tangent to the constraint surface. The multiplier scales the constraint normal and measures the marginal value of relaxing the constraint under the stated sign convention.

## 3. Comparative statics and envelope interpretation
Let \(V(c)=\max_{g(x)=c}f(x)\). Under regularity and differentiability, the envelope result gives \(V'(c)=\lambda^*\) for the Lagrangian convention above. The multiplier is a local shadow value, not necessarily a globally constant effect.

## 4. Second-order conditions
The Hessian of the Lagrangian is evaluated along feasible directions tangent to the constraint. For a maximum, the quadratic form must be nonpositive on the tangent space; negative definiteness on nonzero feasible directions gives a strict local maximum. The unrestricted Hessian of the objective alone is not the correct constrained test.

## 5. Cobb–Douglas utility
Maximize \(x^\alpha y^{1-\alpha}\) subject to \(p_xx+p_yy=m\), with positive prices and income. Using log utility gives FOCs \(\alpha/x=\lambda p_x\) and \((1-\alpha)/y=\lambda p_y\), plus the budget equation. Solving yields \(x^*=\alpha m/p_x\), \(y^*=(1-\alpha)m/p_y\).

## Problem set
1. Derive FOCs for a two-variable equality-constrained problem.
2. Interpret the multiplier under a clearly stated Lagrangian sign convention.
3. Solve Cobb–Douglas utility maximization.
4. Derive the expenditure-minimization dual problem.
5. Explain why constrained SOCs use feasible directions.

## Computational lab
Solve constrained quadratic and Cobb–Douglas problems symbolically and numerically. Compare the multiplier with a finite-difference approximation to the optimized value function.

## Coding exercises

These exercises apply the calculus of this lecture to **real economic data** in Python, mixing symbolic (`sympy`),
numerical (`scipy`) and statistical (`statsmodels`) tools. Install once with
`pip install sympy scipy numpy pandas statsmodels wooldridge`. Try the starter code first, then open the solution.
Every printed result below comes from running the solution code exactly as shown.

### Exercise 9.1 — Cobb–Douglas demand and the multiplier as marginal utility of income

*Data:* `engel` (statsmodels) — 235 Belgian working-class households (Engel's 19th-century data): annual income and food expenditure.

A household maximizes \(u(f,o)=\alpha\log f+(1-\alpha)\log o\) subject to \(f+o=m\) (prices normalized to 1).

1. Solve the Lagrangian with `sympy` for the demands and the multiplier \(\lambda\).
2. Calibrate \(\alpha\) to the average food budget share in Engel's data and compute the demands for the median household.
3. Verify the envelope theorem numerically: \(dV/dm=\lambda\), where \(V(m)\) is maximized utility.
4. Cobb–Douglas implies a constant budget share. Check this against the data by regressing the food share on log income. What does Engel's law say?

```python
import numpy as np, sympy as sp, statsmodels.api as sm
e = sm.datasets.engel.load_pandas().data
f, o, lam, a, m = sp.symbols("f o lambda alpha m", positive=True)
Lag = a * sp.log(f) + (1 - a) * sp.log(o) + lam * (m - f - o)
# TODO: sp.solve([diff(Lag, f), diff(Lag, o), diff(Lag, lam)], [f, o, lam])
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, sympy as sp, statsmodels.api as sm, statsmodels.formula.api as smf
e = sm.datasets.engel.load_pandas().data
f, o, lam, a, m = sp.symbols("f o lambda alpha m", positive=True)
Lag = a * sp.log(f) + (1 - a) * sp.log(o) + lam * (m - f - o)
sol = sp.solve([sp.diff(Lag, v) for v in (f, o, lam)], [f, o, lam], dict=True)[0]
print("demands and multiplier:", sol)
alpha = (e["foodexp"] / e["income"]).mean(); mmed = e["income"].median()
vals = {a: alpha, m: mmed}
print(f"alpha = mean food share = {alpha:.4f};  median income {mmed:.1f}: food {float(sol[f].subs(vals)):.1f}, other {float(sol[o].subs(vals)):.1f}, lambda = {float(sol[lam].subs(vals)):.6f}")
V = sp.lambdify(m, (a * sp.log(sol[f]) + (1 - a) * sp.log(sol[o])).subs(a, alpha))
h = 1e-3
print(f"envelope check: dV/dm numerically = {(V(mmed + h) - V(mmed - h)) / (2*h):.6f}  vs lambda = {1/mmed:.6f}")
r = smf.ols("I(foodexp/income) ~ np.log(income)", data=e).fit(cov_type="HC1")
print(f"food share on log income: slope {r.params.iloc[1]:.4f} (robust se {r.bse.iloc[1]:.4f})")
```

Output:

```text
demands and multiplier: {f: alpha*m, lambda: 1/m, o: m*(1 - alpha)}
alpha = mean food share = 0.6558;  median income 884.0: food 579.7, other 304.3, lambda = 0.001131
envelope check: dV/dm numerically = 0.001131  vs lambda = 0.001131
food share on log income: slope -0.0863 (robust se 0.0144)
```

The Lagrangian gives the familiar Cobb–Douglas demands \(f=\alpha m\) and \(o=(1-\alpha)m\), with \(\lambda=1/m\). The
envelope theorem holds to six decimals: the multiplier is the marginal utility of income. Calibrated to Engel's data, the
average household spends 66% of income on food. But Cobb–Douglas implies a **constant** budget share, and the data reject
it: the food share falls by about 8.6 percentage points for each log point of income. That is Engel's law — food is a
necessity — and it calls for non-homothetic preferences such as Stone–Geary.

</details>

### Exercise 9.2 — Restricted least squares via a Lagrangian: constant returns in investment

*Data:* `grunfeld` (statsmodels) — 11 U.S. firms, 1935–1954.

In \(\log I=\beta_0+\beta_1\log V+\beta_2\log K+u\), impose \(\beta_1+\beta_2=1\).

1. Write the Lagrangian for minimizing SSR subject to \(R\beta=r\) and derive \(\tilde\beta=\widehat\beta-(X'X)^{-1}R'[R(X'X)^{-1}R']^{-1}(R\widehat\beta-r)\).
2. Compute \(\tilde\beta\) and the multiplier, and check the result by substituting the constraint into the regression.
3. Compute the \(F\) statistic from the restricted and unrestricted SSRs. Interpret the size of the multiplier.

```python
import numpy as np, statsmodels.api as sm
g = sm.datasets.grunfeld.load_pandas().data.query("invest > 0 and capital > 0")
y = np.log(g["invest"].to_numpy()); X = np.column_stack([np.ones(len(g)), np.log(g["value"]), np.log(g["capital"])])
R = np.array([[0, 1, 1]]); r = np.array([1.0])
# TODO: restricted estimator, multiplier = 2 [R (X'X)^-1 R']^-1 (R b - r)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, statsmodels.api as sm
from scipy import stats
g = sm.datasets.grunfeld.load_pandas().data.query("invest > 0 and capital > 0")
y = np.log(g["invest"].to_numpy()); X = np.column_stack([np.ones(len(g)), np.log(g["value"]), np.log(g["capital"])])
R = np.array([[0.0, 1.0, 1.0]]); r = np.array([1.0]); n, k = X.shape
XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y
mult = 2 * np.linalg.solve(R @ XtXi @ R.T, R @ b - r)
bt = b - XtXi @ R.T @ np.linalg.solve(R @ XtXi @ R.T, R @ b - r)
print(f"unrestricted: {b.round(4)}  (b1 + b2 = {b[1] + b[2]:.4f})")
print(f"restricted:   {bt.round(4)}  (b1 + b2 = {bt[1] + bt[2]:.4f});  Lagrange multiplier = {mult[0]:.3f}")
ys = y - X[:, 2]; Xs = np.column_stack([np.ones(n), X[:, 1] - X[:, 2]])          # substitute b2 = 1 - b1
c = np.linalg.lstsq(Xs, ys, rcond=None)[0]
print(f"substitution method: b0 = {c[0]:.4f}, b1 = {c[1]:.4f}, b2 = {1 - c[1]:.4f}")
ssr_u, ssr_r = np.sum((y - X @ b)**2), np.sum((y - X @ bt)**2)
F = (ssr_r - ssr_u) / (ssr_u / (n - k))
print(f"F(1, {n-k}) = {F:.2f}, p = {stats.f.sf(F, 1, n - k):.2e}")
```

Output:

```text
unrestricted: [-2.7772  0.8032  0.3755]  (b1 + b2 = 1.1786)
restricted:   [-1.7826  0.6967  0.3033]  (b1 + b2 = 1.0000);  Lagrange multiplier = 115.227
substitution method: b0 = -1.7826, b1 = 0.6967, b2 = 0.3033
F(1, 217) = 36.03, p = 8.05e-09
```

The unrestricted elasticities sum to 1.18. Imposing \(\beta_1+\beta_2=1\) through the Lagrangian reproduces the
substitution method exactly. The large multiplier says the constraint is costly in terms of fit, and the \(F\) test rejects
it decisively (\(F\approx36\)): investment rises more than proportionally with the scale of the firm. The Lagrange
multiplier and the \(F\) statistic carry the same information, which is the logic of the LM test.

</details>

## Reading
Sundaram, *A First Course in Optimization Theory*; Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*; Simon and Blume, *Mathematics for Economists*.