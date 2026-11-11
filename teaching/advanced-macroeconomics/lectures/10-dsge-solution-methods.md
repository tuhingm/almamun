# Lecture 10 — DSGE Solution Methods: Linearization, Perturbation, and QZ

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Compute deterministic steady states; linearize equilibrium conditions; understand rational-expectations solution and Blanchard–Kahn conditions.

## 1. General equilibrium system
Write a DSGE model as
\[
E_tF(y_{t+1},y_t,x_{t+1},x_t,z_t;\theta)=0,
\]
where \(y_t\) are jump variables, \(x_t\) predetermined states, \(z_t\) exogenous shocks, and \(\theta\) structural parameters.

## 2. Steady state
Set shocks to their steady values and impose time invariance. Solve \(F(\bar y,\bar y,\bar x,\bar x,\bar z;\theta)=0\). Check positivity, feasibility, market clearing, and residuals. A solver's convergence flag alone is not sufficient.

## 3. First-order approximation
Let deviations from steady state be denoted by hats. A first-order Taylor expansion produces a linear rational-expectations system. In state-space form:
\[
A E_t s_{t+1}=B s_t+C\varepsilon_t.
\]
The generalized Schur decomposition (QZ) separates stable and unstable generalized eigenvalues.

## 4. Blanchard–Kahn conditions
A unique stable rational-expectations solution requires the number of unstable roots to match the number of non-predetermined variables, under the chosen system representation and regularity conditions. Report eigenvalue conventions and variable timing; do not count roots without documenting the matrix pencil.

## 5. Higher-order perturbation
Second- and third-order approximations capture curvature and risk effects omitted at first order. Pruning may be needed to control explosive simulation artifacts. Approximation accuracy should be evaluated over the shock region relevant to the application.

## 6. Worked diagnostic
For a solved model, tabulate generalized eigenvalues, classify stable roots by modulus under the software's convention, count jump variables, and compare counts. If counts fail, inspect timing, redundant equations, and parameter restrictions.

## 7. Computational lab
Implement a small RBC model in Dynare. Save steady-state residuals, eigenvalues, decision rules, and IRFs. Re-solve after changing one parameter and document whether determinacy changes.

## Problem set
1. Define states and jump variables.
2. Explain the role of QZ decomposition.
3. State the Blanchard–Kahn root-count condition.
4. Explain why a steady-state residual check is necessary.

## Coding exercises

These exercises combine **real U.S. macroeconomic data** (the quarterly `macrodata` series in `statsmodels`, 1959–2009,
and annual series from the `wooldridge` package) with **numerical solutions of the models** in this lecture. Install once
with `pip install numpy scipy pandas statsmodels wooldridge`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 10.1 — Linearize the Ramsey model and solve it with the QZ decomposition

*Data:* Calibration as in Lecture 3: \(\alpha=1/3\), \(\beta=0.96\), \(\delta=0.08\), \(\sigma=2\).

1. Write the two equilibrium conditions \(f(k_{t+1},c_{t+1},k_t,c_t)=0\) and linearize them **numerically** (finite differences) around the steady state into \(A\,E_ty_{t+1}=B\,y_t\), with \(y=(\hat k,\hat c)\).
2. Use `scipy.linalg.ordqz` to compute the generalized eigenvalues. Check the Blanchard–Kahn condition: one unstable root for one jump variable (\(c\)).
3. Recover the policy function \(\hat c_t=h\,\hat k_t\) and the law of motion \(\hat k_{t+1}=g\,\hat k_t\). Compare the linear transition with the nonlinear shooting solution of Lecture 3 after 10 years.

```python
import numpy as np
from scipy.linalg import ordqz
a, b, d, s = 1/3, 0.96, 0.08, 2.0
kss = (a / (1/b - 1 + d))**(1 / (1 - a)); css = kss**a - d * kss
def f(k1, c1, k0, c0):
    return np.array([k1 - (k0**a + (1 - d) * k0 - c0), c0**(-s) - b * c1**(-s) * (a * k1**(a - 1) + 1 - d)])
# TODO: Jacobians by finite differences; A = df/d(k1,c1) * ss, B = -df/d(k0,c0) * ss (log-deviations)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np
from scipy.linalg import ordqz
a, b, d, s = 1/3, 0.96, 0.08, 2.0
kss = (a / (1/b - 1 + d))**(1 / (1 - a)); css = kss**a - d * kss
def f(x):                                     # x = (k1, c1, k0, c0) in levels
    k1, c1, k0, c0 = x
    return np.array([k1 - (k0**a + (1 - d) * k0 - c0), c0**(-s) - b * c1**(-s) * (a * k1**(a - 1) + 1 - d)])
x0 = np.array([kss, css, kss, css]); h = 1e-7; J = np.zeros((2, 4))
for j in range(4):
    e = np.zeros(4); e[j] = h * x0[j]; J[:, j] = (f(x0 + e) - f(x0 - e)) / (2 * h)      # derivative w.r.t. log deviations
A, B = J[:, :2], -J[:, 2:]                     # A E[y_{t+1}] = B y_t
AA, BB, alpha, beta_, Q, Z = ordqz(B, A, sort="iuc", output="real")    # stable generalized eigenvalues first
lam = alpha / beta_
print("generalized eigenvalues:", np.round(lam, 4), " -> unstable roots:", int(np.sum(np.abs(lam) > 1)), "(Blanchard-Kahn needs 1)")
Z11, Z21 = Z[0, 0], Z[1, 0]
hc = Z21 / Z11                                 # c_hat = h k_hat on the stable subspace
g = float(np.real(lam[np.abs(lam) < 1][0]))
print(f"policy: c_hat = {hc:.4f} k_hat;  law of motion: k_hat' = {g:.4f} k_hat  (half-life {np.log(0.5)/np.log(g):.1f} years)")
k0 = np.log(0.5)
print(f"linear solution from k0 = 0.5 k*: k/k* after 10 years = {np.exp(g**10 * k0):.3f}, initial c/c* = {np.exp(hc * k0):.3f}")
```

Output:

```text
generalized eigenvalues: [0.9178+0.j 1.135 +0.j]  -> unstable roots: 1 (Blanchard-Kahn needs 1)
policy: c_hat = 0.4347 k_hat;  law of motion: k_hat' = 0.9178 k_hat  (half-life 8.1 years)
linear solution from k0 = 0.5 k*: k/k* after 10 years = 0.745, initial c/c* = 0.740
```

The linearized system has one stable and one unstable generalized eigenvalue, matching one predetermined variable (capital)
and one jump variable (consumption): the Blanchard–Kahn condition holds and the saddle path is unique. The stable root, 0.918,
implies a half-life of about 8 years, and the policy function sets consumption 0.43% above steady state for each 1% of extra
capital. Started at half the steady-state capital, the linear solution gives \(k/k^*=0.745\) after 10 years against 0.775 in the
nonlinear shooting solution: linearization is accurate near the steady state but less so far from it.

</details>

### Exercise 10.2 — Value function iteration versus the exact solution

*Data:* Brock–Mirman model: log utility, full depreciation, \(\alpha=1/3\), \(\beta=0.96\) — the exact policy \(k'=\alpha\beta k^{\alpha}\) is known.

1. Solve the deterministic model \(V(k)=\max_{k'}\{\log(k^{\alpha}-k')+\beta V(k')\}\) by value function iteration on grids of 100, 300 and 1,000 points (vectorized).
2. Report the number of iterations, computing time, and the maximum error of the policy function relative to the exact \(k'=\alpha\beta k^{\alpha}\).
3. Why does the error fall only slowly with the number of grid points, and what methods (interpolation, endogenous grids — Lecture 12) do better?

```python
import numpy as np, time
a, b = 1/3, 0.96
kss = (a * b)**(1 / (1 - a))
# TODO: grid around kss; utility matrix U[i, j] = log(k_i^a - k_j); iterate V = max_j (U + b V_j)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, time
a, b = 1/3, 0.96
kss = (a * b)**(1 / (1 - a))
for n in (100, 300, 1000):
    k = np.linspace(0.2 * kss, 2.0 * kss, n)
    c = k[:, None]**a - k[None, :]
    U = np.where(c > 0, np.log(np.maximum(c, 1e-300)), -1e10)
    V = np.zeros(n); t0 = time.time()
    for it in range(2000):
        W = U + b * V[None, :]; V1 = W.max(1)
        if np.max(np.abs(V1 - V)) < 1e-8: break
        V = V1
    pol = k[W.argmax(1)]
    inner = (k > 0.3 * kss) & (k < 1.8 * kss)
    err = np.max(np.abs(pol - a * b * k**a)[inner])
    print(f"grid {n:5d}: {it:4d} iterations, {time.time() - t0:6.2f} s, max policy error {err:.2e} (grid spacing {k[1]-k[0]:.2e})")
```

Output:

```text
grid   100:  451 iterations,   0.01 s, max policy error 1.87e-03 (grid spacing 3.29e-03)
grid   300:  451 iterations,   0.06 s, max policy error 6.70e-04 (grid spacing 1.09e-03)
grid  1000:  451 iterations,   0.69 s, max policy error 1.96e-04 (grid spacing 3.26e-04)
```

VFI converges in about 450 iterations regardless of the grid, because the contraction rate is \(\beta\). The policy error falls
roughly in proportion to the grid spacing (first order), because the maximization is restricted to grid points: a 10 times
finer grid gives about 10 times smaller errors, at about 100 times the computing time, since the vectorized search is
\(O(n^2)\). Interpolating the value function, using first-order conditions (policy-function iteration), or the endogenous grid
method (Lecture 12) reach much higher accuracy at a fraction of the cost.

</details>

## Further reading
Klein (2000); Sims (2002); Heer and Maussner, *Dynamic General Equilibrium Modeling*.
