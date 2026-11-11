# Lecture 10 — Inequality Constraints and Karush–Kuhn–Tucker Conditions

## Learning objectives
State KKT conditions; identify active constraints; interpret complementary slackness; distinguish necessary and sufficient conditions; solve inequality-constrained economic problems.

## 1. Standard form
Consider minimizing \(f(x)\) subject to \(g_j(x)\le0\), \(j=1,\ldots,m\), and equality constraints as needed. Define
\[
\mathcal L(x,\mu)=f(x)+\sum_{j=1}^m\mu_jg_j(x).
\]
For a differentiable problem, KKT conditions include stationarity, primal feasibility, dual feasibility \(\mu_j\ge0\), and complementary slackness \(\mu_jg_j(x)=0\).

## 2. Interpretation
An inactive constraint has \(g_j(x^*)<0\), implying \(\mu_j=0\). An active constraint binds and may have a positive multiplier. Complementary slackness links scarcity values to binding restrictions. Multipliers are local shadow prices under regularity.

## 3. Constraint qualifications
KKT conditions are not automatically necessary for every nonlinear program. Conditions such as LICQ or an appropriate weaker qualification ensure the existence of multipliers at a local optimum. State the qualification used rather than treating KKT as universally necessary.

## 4. Sufficiency and convexity
For a convex minimization problem—convex objective and convex feasible set—any feasible KKT point is globally optimal. Strict convexity can give uniqueness. In nonconvex problems, KKT points may be local minima, maxima, or saddle points.

## 5. Economic applications
Capacity, non-negativity, borrowing, labor-supply, and resource constraints generate inequality restrictions. A binding resource constraint can have a positive shadow value; a slack resource constraint has zero multiplier under complementary slackness.

## Worked example
Minimize \((x-3)^2\) subject to \(x\le2\). Write \(g(x)=x-2\le0\). The constrained optimum is \(x^*=2\), with multiplier \(\mu=2\), satisfying stationarity \(2(x-3)+\mu=0\), feasibility, dual feasibility, and complementary slackness.

## Problem set
1. State all KKT conditions for a problem with two inequalities.
2. Identify active constraints in a numerical solution.
3. Solve a one-dimensional constrained quadratic.
4. Explain the role of LICQ.
5. State conditions under which KKT conditions are sufficient for global optimality.

## Computational lab
Solve quadratic programs with binding and slack constraints. Compare analytical KKT solutions with numerical optimizers and verify complementary slackness.

## Coding exercises

These exercises apply the calculus of this lecture to **real economic data** in Python, mixing symbolic (`sympy`),
numerical (`scipy`) and statistical (`statsmodels`) tools. Install once with
`pip install sympy scipy numpy pandas statsmodels wooldridge`. Try the starter code first, then open the solution.
Every printed result below comes from running the solution code exactly as shown.

### Exercise 10.1 — Stocks or bills? A portfolio choice with no borrowing and no short sales

*Data:* `volat` — monthly S&P 500 returns and 3-month T-bill yields, 1947–1993.

A mean–variance investor chooses the share \(w\) in stocks (the rest in T-bills) to maximize \(w\mu_e-\tfrac{\gamma}{2}w^2\sigma^2\), where \(\mu_e\) and \(\sigma^2\) are the mean and variance of the monthly **excess** return, subject to \(0\le w\le1\).

1. Write the KKT conditions with multipliers for both bounds.
2. Estimate \(\mu_e\) and \(\sigma\) from the data. Solve for \(w^*\) and the multipliers for \(\gamma=1,2,5,10\).
3. For which \(\gamma\) is the borrowing constraint \(w\le1\) binding, and what is its shadow value? Check complementary slackness.

```python
import numpy as np, wooldridge as wd
v = wd.data("volat").dropna(subset=["rsp500"])
ex = (v["rsp500"] - v["i3"]).to_numpy() / 1200          # monthly excess return
mu, s2 = ex.mean(), ex.var()
# TODO: interior solution w = mu/(gamma s2); clip to [0,1]; multipliers from the FOC
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
v = wd.data("volat").dropna(subset=["rsp500"])
ex = (v["rsp500"] - v["i3"]).to_numpy() / 1200
mu, s2 = ex.mean(), ex.var()
print(f"monthly excess return: mean {100*mu:.3f}%, sd {100*np.sqrt(s2):.3f}%  (annualized Sharpe ratio {mu/np.sqrt(s2)*np.sqrt(12):.2f})")
for g in (1, 2, 5, 10):
    w_int = mu / (g * s2); w = min(max(w_int, 0.0), 1.0)
    foc = mu - g * w * s2                       # = lam_upper - lam_lower at the optimum
    lam_up, lam_lo = max(foc, 0.0), max(-foc, 0.0)
    cs = lam_up * (1 - w) + lam_lo * w
    print(f"gamma = {g:2d}: unconstrained w = {w_int:5.2f}, optimal w = {w:.3f}, multiplier on w<=1 = {lam_up:.6f}, on w>=0 = {lam_lo:.6f}, complementary slackness {cs:.1e}")
```

Output:

```text
monthly excess return: mean 0.594%, sd 3.394%  (annualized Sharpe ratio 0.61)
gamma =  1: unconstrained w =  5.16, optimal w = 1.000, multiplier on w<=1 = 0.004791, on w>=0 = 0.000000, complementary slackness 0.0e+00
gamma =  2: unconstrained w =  2.58, optimal w = 1.000, multiplier on w<=1 = 0.003639, on w>=0 = 0.000000, complementary slackness 0.0e+00
gamma =  5: unconstrained w =  1.03, optimal w = 1.000, multiplier on w<=1 = 0.000183, on w>=0 = 0.000000, complementary slackness 0.0e+00
gamma = 10: unconstrained w =  0.52, optimal w = 0.516, multiplier on w<=1 = 0.000000, on w>=0 = 0.000000, complementary slackness 4.5e-19
```

Over 1947–1993, stocks earned about 0.6% a month over T-bills, with a Sharpe ratio of about 0.6 a year. That premium is
so large relative to the risk that a mean–variance investor with \(\gamma\le5\) would like to **borrow** to hold more than
100% in stocks; with borrowing ruled out, the constraint \(w\le1\) binds. Its multiplier is the marginal gain from relaxing
it and falls as risk aversion rises. At \(\gamma=10\), the interior solution of about 52% is feasible, and both multipliers
are zero. Complementary slackness holds in every case. (The large equity premium relative to risk is the "equity premium
puzzle" of Mehra and Prescott, 1985.)

</details>

### Exercise 10.2 — Synthetic weights as a KKT problem

*Data:* `statecrime` (statsmodels) — U.S. states: % high-school graduates, % in poverty, % single-parent households, % urban.

A synthetic-control-style problem: find non-negative weights summing to one so that a weighted average of other states matches California's characteristics as closely as possible:
\(\min_w\|x_{CA}-X_0w\|^2\) s.t. \(w\ge0\), \(\sum w=1\).

1. Solve it with `scipy.optimize.minimize` (SLSQP), using standardized characteristics.
2. Report the states with positive weight, and the fit.
3. Check the KKT conditions: for states with \(w_j=0\), the gradient component must be at least as large as for states with \(w_j>0\) (whose gradient components are all equal, to the multiplier on the adding-up constraint).

```python
import numpy as np, statsmodels.api as sm
from scipy import optimize
d = sm.datasets.statecrime.load_pandas().data
cols = ["hs_grad", "poverty", "single", "urban"]
Z = (d[cols] - d[cols].mean()) / d[cols].std()
x1 = Z.loc["California"].to_numpy(); X0 = Z.drop("California").to_numpy().T
# TODO: minimize ||x1 - X0 w||^2 with bounds (0,1) and constraint sum(w) = 1
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, statsmodels.api as sm
from scipy import optimize
d = sm.datasets.statecrime.load_pandas().data
cols = ["hs_grad", "poverty", "single", "urban"]
Z = (d[cols] - d[cols].mean()) / d[cols].std()
x1 = Z.loc["California"].to_numpy(); donors = Z.drop("California"); X0 = donors.to_numpy().T; J = X0.shape[1]
obj = lambda w: np.sum((x1 - X0 @ w)**2); grad = lambda w: -2 * X0.T @ (x1 - X0 @ w)
res = optimize.minimize(obj, np.full(J, 1 / J), jac=grad, method="SLSQP", bounds=[(0, 1)] * J,
                        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"ftol": 1e-14, "maxiter": 500})
w = res.x; pos = w > 1e-6
print("positive weights:", {s: round(x, 3) for s, x in zip(donors.index[pos], w[pos])})
print("California (std units):", x1.round(3), "  synthetic:", (X0 @ w).round(3), f"  loss {obj(w):.4f}")
gr = grad(w)
print(f"gradient on positive-weight states: {gr[pos].round(4)}  (equal -> common multiplier {gr[pos].mean():.4f})")
print(f"min gradient on zero-weight states: {gr[~pos].min():.4f}  (KKT requires >= {gr[pos].mean():.4f})")
```

Output:

```text
positive weights: {'Rhode Island': np.float64(0.433), 'Texas': np.float64(0.567)}
California (std units): [-1.859  0.111  0.546  1.397]   synthetic: [-1.45   0.281  0.477  1.021]   loss 0.3425
gradient on positive-weight states: [-1.9238 -1.9238]  (equal -> common multiplier -1.9238)
min gradient on zero-weight states: -1.9106  (KKT requires >= -1.9238)
```

Only two states get positive weight — Texas (57%) and Rhode Island (43%). Corner solutions with most weights at zero are
typical of synthetic control. The synthetic unit matches California's poverty and single-parent shares well but not its
very high urban share or low high-school graduation, because no convex combination of other states reaches those
extremes. The KKT conditions hold: the gradient components of the two active states are equal (the common multiplier on
the adding-up constraint), and every inactive state's gradient component is at least as large, so moving weight to it
would not lower the loss.

</details>

## Reading
Sundaram, *A First Course in Optimization Theory*; Nocedal and Wright, *Numerical Optimization*; Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*.