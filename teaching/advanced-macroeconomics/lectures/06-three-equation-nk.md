# Lecture 06 — The Three-Equation New Keynesian Model

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Combine the dynamic IS relation, Phillips curve, and monetary-policy rule; analyze equilibrium responses to demand, supply, and policy shocks.

## 1. Core system
The canonical linearized model is
\[
x_t=E_tx_{t+1}-\frac{1}{\sigma}(i_t-E_t\pi_{t+1}-r_t^n),
\]
\[
\pi_t=\beta E_t\pi_{t+1}+\kappa x_t+u_t,
\]
\[
i_t=\rho_i i_{t-1}+(1-\rho_i)(\phi_\pi\pi_t+\phi_xx_t)+\varepsilon_t^i.
\]
Here \(x_t\) is the output gap, \(r_t^n\) the natural real rate, \(u_t\) a cost-push shock, and \(\varepsilon_t^i\) a monetary-policy shock.

## 2. Taylor principle and determinacy
In the simplest model without policy inertia or output feedback, the Taylor principle requires the nominal rate to rise more than one-for-one with inflation, \(\phi_\pi>1\). This is a useful benchmark, not a universal determinacy test for every extended model. With inertia, additional state variables, and fiscal interactions, apply the full rational-expectations eigenvalue conditions.

## 3. Shock transmission
A positive natural-rate shock raises demand at a given real rate. A positive cost-push shock raises inflation while reducing the feasible stabilization trade-off. A contractionary policy shock raises the real rate and lowers the output gap and inflation over time.

## 4. Worked analysis
Set \(\rho_i=0\), \(\phi_x=0\), and suppress shocks. Substitute the policy rule into the IS equation and use the Phillips curve to obtain a two-equation system in \((x_t,\pi_t)\). Discuss how a higher \(\phi_\pi\) changes the real-rate response to inflation.

## 5. Computational lab
Implement the three-equation model in Dynare. Generate IRFs for a demand shock, cost-push shock, and policy shock. Compare \(\phi_\pi=0.8,1.2,1.8\), checking determinacy rather than assuming it.

## Problem set
1. Derive the IS relation from household optimality.
2. Explain the Taylor principle.
3. Compare demand and cost-push shocks.
4. Explain why the output gap and natural output must be defined consistently.

## Coding exercises

These exercises combine **real U.S. macroeconomic data** (the quarterly `macrodata` series in `statsmodels`, 1959–2009,
and annual series from the `wooldridge` package) with **numerical solutions of the models** in this lecture. Install once
with `pip install numpy scipy pandas statsmodels wooldridge`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 6.1 — Solving the three-equation NK model by undetermined coefficients

*Data:* Calibration following Galí (2015, ch. 3): \(\beta=0.99\), \(\sigma=1\), \(\kappa=0.17\), \(\phi_\pi=1.5\), \(\phi_y=0.125\), monetary shock persistence \(\rho_v=0.5\).

The model: \(x_t=E_tx_{t+1}-\frac1\sigma(i_t-E_t\pi_{t+1})\); \(\pi_t=\beta E_t\pi_{t+1}+\kappa x_t\); \(i_t=\phi_\pi\pi_t+\phi_yx_t+v_t\), \(v_t=\rho_vv_{t-1}+\varepsilon_t\).

1. Guess \(x_t=\psi_xv_t\), \(\pi_t=\psi_\pi v_t\), and solve for the coefficients with `sympy`.
2. Compute impulse responses of the output gap, inflation, the nominal rate and the real rate to a 25 basis-point shock (annualized: 1 percentage point).
3. Repeat for shock persistence \(\rho_v=0.8\) and \(0.9\). Why can the nominal rate **fall** after a contractionary monetary shock, while the real rate rises?

```python
import sympy as sp
px, pp, b, s, k, fp, fy, r = sp.symbols("psi_x psi_pi beta sigma kappa phi_pi phi_y rho")
# x = px v, pi = pp v, E v' = r v
eqs = [sp.Eq(px, r * px - (1 / s) * (fp * pp + fy * px + 1 - r * pp)), sp.Eq(pp, b * r * pp + k * px)]
# TODO: sp.solve(eqs, [px, pp])
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, sympy as sp
px, pp, b, s, k, fp, fy, r = sp.symbols("psi_x psi_pi beta sigma kappa phi_pi phi_y rho")
eqs = [sp.Eq(px, r * px - (1 / s) * (fp * pp + fy * px + 1 - r * pp)), sp.Eq(pp, b * r * pp + k * px)]
sol = sp.solve(eqs, [px, pp], dict=True)[0]
print("psi_x  =", sp.simplify(sol[px])); print("psi_pi =", sp.simplify(sol[pp]))
cal = {b: 0.99, s: 1.0, k: 0.17, fp: 1.5, fy: 0.125, r: 0.5}
cx, cp = float(sol[px].subs(cal)), float(sol[pp].subs(cal))
v = 0.25 * 0.5**np.arange(7)                                     # 25 bp shock, quarterly rates
x, pi = cx * v, cp * v; i = 1.5 * pi + 0.125 * x + v; rr = i - np.r_[pi[1:], 0.5 * pi[-1]]
print("quarter | output gap % | inflation (ann. %) | nominal rate (ann. %) | real rate (ann. %)")
for t in range(7):
    print(f"   {t}    |   {x[t]:7.3f}    |      {4*pi[t]:7.3f}       |       {4*i[t]:7.3f}         |     {4*rr[t]:7.3f}")
for rv in (0.5, 0.8, 0.9):
    c2 = {**cal, r: rv}; x0, p0 = 0.25 * float(sol[px].subs(c2)), 0.25 * float(sol[pp].subs(c2))
    i0 = 1.5 * p0 + 0.125 * x0 + 0.25
    print(f"rho_v = {rv}: impact nominal rate {4*i0:+.3f} pp, real rate {4*(i0 - rv*p0):+.3f} pp, output gap {x0:+.3f}%")
```

Output:

```text
psi_x  = (-beta*rho + 1)/(beta*phi_y*rho - beta*rho**2*sigma + beta*rho*sigma - kappa*phi_pi + kappa*rho - phi_y + rho*sigma - sigma)
psi_pi = kappa/(beta*phi_y*rho - beta*rho**2*sigma + beta*rho*sigma - kappa*phi_pi + kappa*rho - phi_y + rho*sigma - sigma)
quarter | output gap % | inflation (ann. %) | nominal rate (ann. %) | real rate (ann. %)
   0    |    -0.260    |       -0.350       |         0.345         |       0.520
   1    |    -0.130    |       -0.175       |         0.172         |       0.260
   2    |    -0.065    |       -0.088       |         0.086         |       0.130
   3    |    -0.032    |       -0.044       |         0.043         |       0.065
   4    |    -0.016    |       -0.022       |         0.022         |       0.032
   5    |    -0.008    |       -0.011       |         0.011         |       0.016
   6    |    -0.004    |       -0.005       |         0.005         |       0.008
rho_v = 0.5: impact nominal rate +0.345 pp, real rate +0.520 pp, output gap -0.260%
rho_v = 0.8: impact nominal rate -0.506 pp, real rate +0.223 pp, output gap -0.279%
rho_v = 0.9: impact nominal rate -1.123 pp, real rate +0.086 pp, output gap -0.215%
```

A 25 basis-point (1 point annualized) contractionary shock lowers the output gap by about 0.26% and annualized inflation by
about 0.35 points on impact; both responses decay at the shock's rate \(\rho_v\). The real rate rises by more than the nominal
rate because expected inflation falls. With more persistent shocks, the fall in inflation and output is so large that the
Taylor rule's endogenous response **lowers** the nominal rate on impact even though policy is tighter: with
\(\rho_v=0.8\), the nominal rate falls by about 0.5 points while the real rate still rises. The nominal interest rate is a
poor indicator of the policy stance.

</details>

### Exercise 6.2 — The Taylor principle and determinacy

*Data:* Same calibration (\(\beta=0.99\), \(\sigma=1\), \(\kappa=0.17\)).

Write the model without shocks as \(E_tz_{t+1}=Mz_t\) with \(z_t=(x_t,\pi_t)'\). With two forward-looking variables, the equilibrium is determinate when both eigenvalues of \(M\) lie outside the unit circle.

1. Derive \(M\) and compute its eigenvalues on a grid of \(\phi_\pi\in[0,3]\) for \(\phi_y=0\) and \(\phi_y=0.5\).
2. Find the critical \(\phi_\pi\) numerically and compare with the analytical condition \(\kappa(\phi_\pi-1)+(1-\beta)\phi_y>0\) (Bullard and Mitra, 2002).
3. Interpret: why does the central bank have to raise the nominal rate more than one-for-one with inflation?

```python
import numpy as np
b, s, k = 0.99, 1.0, 0.17
def M(fp, fy):
    return np.array([[1 + fy / s + k / (s * b), (fp - 1 / b) / s], [-k / b, 1 / b]])
# TODO: eigenvalues over a grid; find where min |eig| crosses 1
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np
from scipy import optimize
b, s, k = 0.99, 1.0, 0.17
def M(fp, fy):
    return np.array([[1 + fy / s + k / (s * b), (fp - 1 / b) / s], [-k / b, 1 / b]])
determinate = lambda fp, fy: np.min(np.abs(np.linalg.eigvals(M(fp, fy)))) > 1
for fy in (0.0, 0.5):
    grid = np.linspace(0, 3, 3001); det = np.array([determinate(f, fy) for f in grid])
    crit = grid[np.argmax(det)]
    analytic = 1 - (1 - b) * fy / k
    print(f"phi_y = {fy}: determinate for phi_pi > {crit:.3f} (grid);  Bullard-Mitra threshold {analytic:.3f}")
for fp in (0.8, 1.0, 1.2, 1.5):
    print(f"  phi_pi = {fp}, phi_y = 0: |eigenvalues| = {np.abs(np.linalg.eigvals(M(fp, 0))).round(4)}, determinate: {determinate(fp, 0)}")
```

Output:

```text
phi_y = 0.0: determinate for phi_pi > 1.001 (grid);  Bullard-Mitra threshold 1.000
phi_y = 0.5: determinate for phi_pi > 0.971 (grid);  Bullard-Mitra threshold 0.971
  phi_pi = 0.8, phi_y = 0: |eigenvalues| = [1.2973 0.8845], determinate: False
  phi_pi = 1.0, phi_y = 0: |eigenvalues| = [1.1818 1.    ], determinate: False
  phi_pi = 1.2, phi_y = 0: |eigenvalues| = [1.1028 1.1028], determinate: True
  phi_pi = 1.5, phi_y = 0: |eigenvalues| = [1.1259 1.1259], determinate: True
```

With \(\phi_y=0\), determinacy requires \(\phi_\pi>1\) — the Taylor principle — and the numerical boundary matches the
Bullard–Mitra condition. Responding to the output gap relaxes it slightly: with \(\phi_y=0.5\), \(\phi_\pi>0.97\) suffices,
because in the long run a permanent rise in inflation also raises the output gap by \((1-\beta)/\kappa\). Below the threshold,
one eigenvalue lies inside the unit circle and sunspot equilibria exist: self-fulfilling inflation expectations are not ruled
out, because the real rate falls when inflation rises.

</details>

## Further reading
Galí, *Monetary Policy, Inflation, and the Business Cycle*; Clarida, Galí and Gertler (1999).
