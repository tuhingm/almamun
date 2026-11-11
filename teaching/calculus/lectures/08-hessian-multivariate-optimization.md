# Lecture 08 — Hessians and Unconstrained Multivariable Optimization

## Learning objectives
Use Hessian matrices to classify curvature; derive multivariate first- and second-order conditions; apply optimization to utility, production, and likelihood functions.

## 1. Hessian matrix
For twice continuously differentiable \(f:\mathbb R^n\to\mathbb R\), the Hessian is
\[
H_f(x)=\left[\frac{\partial^2f}{\partial x_i\partial x_j}\right]_{i,j}.
\]
Under continuity of second partial derivatives, the Hessian is symmetric. It describes local curvature of the function.

## 2. First-order necessary condition
At an interior local optimum \(x^*\), \(\nabla f(x^*)=0\). This condition is not sufficient and does not apply directly at boundary optima. A stationary point may be a maximum, minimum, or saddle point.

## 3. Second-order conditions
If the Hessian is negative definite at a stationary point, it is a strict local maximum; if positive definite, a strict local minimum. Indefiniteness implies a saddle point. Semidefinite Hessians require higher-order or direct analysis. For a symmetric matrix, definiteness can be assessed by eigenvalues or principal minors.

## 4. Concavity and global optima
If \(f\) is concave on a convex domain, every stationary point is a global maximum. Strict concavity implies at most one maximizer. Convexity gives the corresponding result for minimization. These global results are stronger than local SOCs.

## 5. Economic applications
For utility maximization, concavity of utility supports global optimality. For cost minimization, convex cost functions support unique solutions under suitable feasibility and strict convexity. In maximum likelihood, the Hessian of the log-likelihood describes local curvature around an estimate.

## Worked example
Let \(f(x,y)=-(x^2+2y^2)+4x+8y\). The gradient is \((-2x+4,-4y+8)'\), giving \((2,2)\). The Hessian is diagonal with entries -2 and -4, so it is negative definite and the point is a strict global maximum.

## Problem set
1. Compute the gradient and Hessian of a quadratic form.
2. Classify a stationary point using Hessian eigenvalues.
3. Explain why a semidefinite Hessian is inconclusive.
4. State a sufficient condition for global optimality.
5. Apply the SOC to a Cobb–Douglas utility function after taking logarithms.

## Computational lab
Use symbolic algebra and numerical eigenvalues to classify stationary points. Plot contours and identify maxima, minima, and saddle points.

## Coding exercises

These exercises apply the calculus of this lecture to **real economic data** in Python, mixing symbolic (`sympy`),
numerical (`scipy`) and statistical (`statsmodels`) tools. Install once with
`pip install sympy scipy numpy pandas statsmodels wooldridge`. Try the starter code first, then open the solution.
Every printed result below comes from running the solution code exactly as shown.

### Exercise 8.1 — Least squares as unconstrained optimization: gradient and Hessian checks

*Data:* `wage1` — 526 workers.

1. Write the sum of squared residuals \(SSR(b)=(y-Xb)'(y-Xb)\) for \(\log wage\) on \((1, educ, exper, tenure)\).
2. Minimize it numerically with BFGS from \(b=0\), supplying the analytic gradient \(-2X'(y-Xb)\).
3. At the optimum, check that the gradient is (numerically) zero and that the Hessian \(2X'X\) is positive definite (eigenvalues). Report the condition number and explain what it means for numerical accuracy.
4. Compare with the closed-form solution.

```python
import numpy as np, wooldridge as wd
from scipy import optimize
d = wd.data("wage1")
y = d["lwage"].to_numpy(); X = np.column_stack([np.ones(len(d)), d[["educ", "exper", "tenure"]]])
ssr = lambda b: np.sum((y - X @ b)**2); grad = lambda b: -2 * X.T @ (y - X @ b)
# TODO: optimize.minimize(ssr, np.zeros(4), jac=grad, method="BFGS")
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
from scipy import optimize
d = wd.data("wage1")
y = d["lwage"].to_numpy(); X = np.column_stack([np.ones(len(d)), d[["educ", "exper", "tenure"]]])
ssr = lambda b: np.sum((y - X @ b)**2); grad = lambda b: -2 * X.T @ (y - X @ b)
res = optimize.minimize(ssr, np.zeros(4), jac=grad, method="BFGS", options={"gtol": 1e-10})
closed = np.linalg.solve(X.T @ X, X.T @ y)
print("BFGS:       ", res.x.round(6), f"({res.nit} iterations)")
print("closed form:", closed.round(6))
H = 2 * X.T @ X; ev = np.linalg.eigvalsh(H)
print(f"max |gradient| at optimum: {np.abs(grad(res.x)).max():.2e}")
print(f"Hessian eigenvalues: {ev.round(2)}  -> positive definite: {bool(np.all(ev > 0))};  condition number {ev.max()/ev.min():.0f}")
Xs = (X[:, 1:] - X[:, 1:].mean(0)) / X[:, 1:].std(0); Hs = 2 * np.column_stack([np.ones(len(d)), Xs]).T @ np.column_stack([np.ones(len(d)), Xs])
print(f"condition number after standardizing the regressors: {np.linalg.cond(Hs):.1f}")
```

Output:

```text
BFGS:        [0.2844 0.092  0.0041 0.0221] (5 iterations)
closed form: [0.2844 0.092  0.0041 0.0221]
max |gradient| at optimum: 1.15e-12
Hessian eigenvalues: [    35.64  37907.16  69136.44 648320.76]  -> positive definite: True;  condition number 18189
condition number after standardizing the regressors: 3.6
```

BFGS converges to the closed-form OLS solution in five iterations, and the gradient is zero to 12 decimals. The Hessian
\(2X'X\) is positive definite, so the SSR is strictly convex and the minimum is unique. Its condition number of about
18,000 comes from regressors on very different scales (a constant, years of education, years of tenure). Standardizing
the regressors reduces it to about 4, which is why numerical optimizers and ridge-type penalties work better on
standardized data. Here the problem is still well conditioned enough for double-precision arithmetic.

</details>

### Exercise 8.2 — Newton–Raphson for a logit likelihood, by hand

*Data:* `mroz` — 753 married women; `inlf` = labor force participation.

1. Write the logit log-likelihood \(\ell(\beta)=\sum_i[y_i x_i'\beta-\log(1+e^{x_i'\beta})]\), its gradient \(X'(y-p)\) and Hessian \(-X'WX\) with \(W=\text{diag}(p_i(1-p_i))\).
2. Run Newton–Raphson from \(\beta=0\) with regressors \((1, educ, exper, age, kidslt6, nwifeinc)\). Print the log-likelihood and the gradient norm at each iteration.
3. Show that the Hessian is negative definite everywhere (so \(\ell\) is globally concave and the maximum is unique), and compare with `statsmodels`.

```python
import numpy as np, wooldridge as wd
d = wd.data("mroz")
y = d["inlf"].to_numpy(); X = np.column_stack([np.ones(len(d)), d[["educ", "exper", "age", "kidslt6", "nwifeinc"]]])
b = np.zeros(X.shape[1])
# TODO: loop: p = 1/(1+exp(-Xb)); g = X'(y-p); H = -X' diag(p(1-p)) X; b = b - solve(H, g)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd, statsmodels.api as sm
d = wd.data("mroz")
y = d["inlf"].to_numpy(); X = np.column_stack([np.ones(len(d)), d[["educ", "exper", "age", "kidslt6", "nwifeinc"]]])
b = np.zeros(X.shape[1])
for k in range(8):
    p = 1 / (1 + np.exp(-X @ b))
    ll = np.sum(y * (X @ b) - np.log1p(np.exp(X @ b)))
    g = X.T @ (y - p); H = -(X * (p * (1 - p))[:, None]).T @ X
    print(f"iter {k}: logL = {ll:.8f}   |gradient| = {np.linalg.norm(g):.2e}   max eigenvalue of H = {np.linalg.eigvalsh(H).max():.3f}")
    b = b - np.linalg.solve(H, g)
print("Newton estimates:     ", b.round(5))
print("statsmodels estimates:", sm.Logit(y, X).fit(disp=0).params.round(5))
```

Output:

```text
iter 0: logL = -521.93982696   |gradient| = 2.68e+03   max eigenvalue of H = -2.515
iter 1: logL = -412.01828780   |gradient| = 4.15e+02   max eigenvalue of H = -2.026
iter 2: logL = -406.55793779   |gradient| = 5.43e+01   max eigenvalue of H = -1.831
iter 3: logL = -406.45831881   |gradient| = 1.30e+00   max eigenvalue of H = -1.802
iter 4: logL = -406.45827107   |gradient| = 7.12e-04   max eigenvalue of H = -1.801
iter 5: logL = -406.45827107   |gradient| = 1.91e-10   max eigenvalue of H = -1.801
iter 6: logL = -406.45827107   |gradient| = 5.30e-12   max eigenvalue of H = -1.801
iter 7: logL = -406.45827107   |gradient| = 5.66e-12   max eigenvalue of H = -1.801
Newton estimates:      [ 1.1532  0.2234  0.1179 -0.0951 -1.4636 -0.0199]
statsmodels estimates: [ 1.1532  0.2234  0.1179 -0.0951 -1.4636 -0.0199]
```

Starting from zero, Newton–Raphson reaches the maximum in about five iterations: the gradient norm falls from 2,700 to
\(10^{-10}\), with the quadratic convergence typical near the optimum. The largest eigenvalue of the Hessian stays
negative at every iterate: the logit log-likelihood is globally concave, because \(-X'WX\) is negative definite whenever
\(X\) has full rank. The maximum is therefore unique, and the estimates match `statsmodels` (which uses the same
algorithm) to every printed digit.

</details>

## Reading
Simon and Blume, *Mathematics for Economists*; Sundaram, *A First Course in Optimization Theory*; Apostol, *Calculus*, Vol. 2.