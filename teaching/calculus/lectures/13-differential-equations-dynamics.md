# Lecture 13 — Differential Equations and Continuous-Time Dynamics

## Learning objectives
Solve first-order ODEs; analyze second-order linear equations; study systems using eigenvalues; interpret steady states and stability in economic dynamics.

## 1. First-order differential equations
A differential equation relates an unknown function to its derivatives. A separable equation \(\dot x=a(x)b(t)\) can be rearranged and integrated when division is valid. A linear equation \(\dot x+p(t)x=q(t)\) uses integrating factor \(M(t)=e^{\int p(t)dt}\).

## 2. Linear constant-coefficient equations
For \(\ddot x+a\dot x+bx=0\), try \(x(t)=e^{rt}\). The characteristic equation \(r^2+ar+b=0\) determines exponential, repeated-root, or oscillatory solutions. Initial conditions determine constants.

## 3. Systems of differential equations
For \(\dot x=Ax\), solutions are governed by eigenvalues and eigenvectors of \(A\). If all eigenvalues have negative real parts, the origin is asymptotically stable; if any eigenvalue has positive real part, it is unstable. Zero real parts require further analysis.

## 4. Nonlinear steady states
A steady state \(x^*\) solves \(f(x^*)=0\). Local dynamics around it are approximated by \(\dot u=J_f(x^*)u\), where \(u=x-x^*\). Eigenvalues of the Jacobian determine local stability for hyperbolic steady states.

## 5. Economic applications
Capital accumulation may be written \(\dot k=s f(k)-(n+g+\delta)k\). A steady state equates investment per effective worker with break-even investment. Local stability follows from the derivative of the dynamic law at the steady state.

## Worked example
For \(\dot k=sk^\alpha-\delta k\), with \(0<\alpha<1\), the positive steady state satisfies \(k^*=(s/\delta)^{1/(1-\alpha)}\). The derivative is \(s\alpha k^{\alpha-1}-\delta\), which is negative at the positive steady state, giving local asymptotic stability.

## Problem set
1. Solve a separable first-order ODE with an initial condition.
2. Solve a linear ODE using an integrating factor.
3. Classify roots of a second-order characteristic equation.
4. Determine stability from a two-dimensional Jacobian.
5. Derive and assess a capital-accumulation steady state.

## Computational lab
Simulate ODE paths using numerical solvers. Compare analytical and numerical solutions; plot phase diagrams and trajectories near stable and unstable steady states.

## Coding exercises

These exercises apply the calculus of this lecture to **real economic data** in Python, mixing symbolic (`sympy`),
numerical (`scipy`) and statistical (`statsmodels`) tools. Install once with
`pip install sympy scipy numpy pandas statsmodels wooldridge`. Try the starter code first, then open the solution.
Every printed result below comes from running the solution code exactly as shown.

### Exercise 13.1 — The continuous-time Solow model calibrated to the U.S.

*Data:* `macrodata` — U.S. investment share and population growth (calibration as in Lecture 11).

Capital per worker follows \(\dot k=s\,k^{\alpha}-(n+\delta)k\).

1. Solve the ODE numerically with `solve_ivp` starting from half the steady-state capital.
2. Linearize around \(k^*\): \(\dot k\approx-\lambda(k-k^*)\) with \(\lambda=(1-\alpha)(n+\delta)\). Compute the implied half-life \(\ln2/\lambda\).
3. Measure the half-life in the numerical solution. Why is it close but not identical to the linear prediction?

```python
import numpy as np, statsmodels.api as sm
from scipy.integrate import solve_ivp
m = sm.datasets.macrodata.load_pandas().data
s = (m["realinv"] / m["realgdp"]).mean(); n = np.log(m["pop"]).diff().mean() * 4; a, dl = 1/3, 0.05
kss = (s / (n + dl))**(1 / (1 - a))
# TODO: solve_ivp(lambda t, k: s*k**a - (n+dl)*k, (0, 150), [kss/2], dense_output=True)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, statsmodels.api as sm
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
m = sm.datasets.macrodata.load_pandas().data
s = (m["realinv"] / m["realgdp"]).mean(); n = np.log(m["pop"]).diff().mean() * 4; a, dl = 1/3, 0.05
kss = (s / (n + dl))**(1 / (1 - a))
sol = solve_ivp(lambda t, k: s * k**a - (n + dl) * k, (0, 150), [kss / 2], dense_output=True, rtol=1e-10, atol=1e-12)
lam = (1 - a) * (n + dl)
print(f"k* = {kss:.4f}; convergence rate lambda = {lam:.4f} per year; linear half-life = {np.log(2)/lam:.2f} years")
gap = lambda t: (kss - sol.sol(t)[0]) - 0.5 * (kss - kss / 2)
print(f"numerical half-life (gap halves): {brentq(gap, 0.1, 100):.2f} years")
for t in (0, 10, 20, 40, 80):
    print(f"  t = {t:3d}: k = {sol.sol(t)[0]:.4f} ({sol.sol(t)[0]/kss:.1%} of k*)")
```

Output:

```text
k* = 3.2142; convergence rate lambda = 0.0406 per year; linear half-life = 17.06 years
numerical half-life (gap halves): 18.50 years
  t =   0: k = 1.6071 (50.0% of k*)
  t =  10: k = 2.1024 (65.4% of k*)
  t =  20: k = 2.4561 (76.4% of k*)
  t =  40: k = 2.8695 (89.3% of k*)
  t =  80: k = 3.1453 (97.9% of k*)
```

Starting at half the steady state, capital per worker closes 90% of the gap after about 40 years. Linearization predicts a
half-life of 17.1 years; the numerical solution takes about 18.5. The linear approximation is accurate only near \(k^*\):
starting far below, the true dynamics are slower at the beginning in terms of the **proportional** gap, because
\(\dot k/k=sk^{\alpha-1}-(n+\delta)\) is not linear in \(k\). The calibrated convergence rate of about 4% a year is in
the range of empirical estimates (2–3%) only roughly; a higher capital share brings it down.

</details>

### Exercise 13.2 — Exponential growth: fitting \(\dot y=gy\) to U.S. real GDP

*Data:* `macrodata` — U.S. quarterly real GDP, 1959–2009.

1. The solution of \(\dot y=gy\) is \(y(t)=y_0e^{gt}\), so \(\log y\) is linear in \(t\). Estimate \(g\) by regressing \(\log GDP\) on time (in years).
2. Compute the implied doubling time \(\ln2/g\) and compare with the actual doubling time in the data.
3. Plot (or tabulate) the deviation of log GDP from the fitted exponential path at business-cycle troughs (1975Q1, 1982Q4, 1991Q1, 2009Q2). Is the growth rate constant over 1959–2009? Compare \(g\) for 1959–1983 and 1984–2009.

```python
import numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
m = sm.datasets.macrodata.load_pandas().data
m["t"] = m["year"] + (m["quarter"] - 1) / 4 - 1959
# TODO: smf.ols("np.log(realgdp) ~ t", m).fit()
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
m = sm.datasets.macrodata.load_pandas().data
m["t"] = m["year"] + (m["quarter"] - 1) / 4 - 1959
r = smf.ols("np.log(realgdp) ~ t", data=m).fit()
g = r.params["t"]
print(f"g = {100*g:.3f}% per year;  implied doubling time {np.log(2)/g:.1f} years")
y0 = m["realgdp"].iloc[0]; k = np.argmax(m["realgdp"].to_numpy() >= 2 * y0)
print(f"actual first doubling: {m['t'].iloc[k]:.2f} years (reached in {int(m['year'].iloc[k])}Q{int(m['quarter'].iloc[k])})")
for yq in ((1975, 1), (1982, 4), (1991, 1), (2009, 2)):
    row = m[(m.year == yq[0]) & (m.quarter == yq[1])]
    if len(row): print(f"  {yq[0]}Q{yq[1]}: log deviation from exponential trend {100*r.resid[row.index[0]]:+.2f}%")
for lo, hi in ((1959, 1983), (1984, 2009)):
    s = m[(m.year >= lo) & (m.year <= hi)]
    print(f"g over {lo}-{hi}: {100*smf.ols('np.log(realgdp) ~ t', data=s).fit().params['t']:.3f}% per year")
```

Output:

```text
g = 3.161% per year;  implied doubling time 21.9 years
actual first doubling: 18.50 years (reached in 1977Q3)
  1975Q1: log deviation from exponential trend -1.32%
  1982Q4: log deviation from exponential trend -5.58%
  1991Q1: log deviation from exponential trend -1.34%
  2009Q2: log deviation from exponential trend -10.60%
g over 1959-1983: 3.433% per year
g over 1984-2009: 2.985% per year
```

Real GDP grew at about 3.2% a year on average, implying a doubling time of about 22 years. It actually doubled faster, in
18.5 years (by 1977), because growth was higher early in the sample: 3.4% a year in 1959–83 against 3.0% in 1984–2009. The
constant-growth model is a first approximation. The deviations from trend at recession troughs grow over time, ending
10.6% below trend in 2009 — partly because of the deep recession and partly because the trend slowed, which a single
exponential cannot capture.

</details>

## Reading
Boyce, DiPrima, and Meade, *Elementary Differential Equations*; Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*; Strogatz, *Nonlinear Dynamics and Chaos*.