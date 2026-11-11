# Lecture 14 — Difference Equations and Discrete-Time Dynamics

## Learning objectives
Solve first-order difference equations; analyze fixed points and local stability; study linear systems; apply discrete dynamics to economic adjustment and growth.

## 1. First-order difference equations
A discrete dynamic process has form \(x_{t+1}=f(x_t)\). A fixed point satisfies \(x^*=f(x^*)\). The path depends on the initial condition and the global properties of \(f\).

## 2. Linear difference equations
For \(x_{t+1}=a+bx_t\), if \(b\ne1\), the fixed point is \(x^*=a/(1-b)\). The solution is
\[
x_t-x^*=b^t(x_0-x^*).
\]
Thus convergence occurs iff \(|b|<1\); positive \(b\) gives monotone convergence, while negative \(b\) gives oscillatory convergence.

## 3. Local stability
For nonlinear \(x_{t+1}=f(x_t)\), linearize around a fixed point:
\[
x_{t+1}-x^*\approx f'(x^*)(x_t-x^*).
\]
If \(|f'(x^*)|<1\), the fixed point is locally asymptotically stable; if \(|f'(x^*)|>1\), it is unstable. The boundary cases require higher-order analysis.

## 4. Multidimensional systems
For \(x_{t+1}=F(x_t)\), the Jacobian at a fixed point determines local dynamics. If all eigenvalues have modulus below one, the fixed point is locally asymptotically stable; if any eigenvalue has modulus above one, it is unstable. Unit-modulus eigenvalues require additional analysis.

## 5. Economic applications
Applications include cobweb price adjustment, debt dynamics, adaptive expectations, inventory adjustment, and discrete capital accumulation. Stability depends on adjustment parameters and the slope of the map, not merely on the existence of a steady state.

## Worked example
For \(x_{t+1}=4+0.5x_t\), the fixed point is 8 and \(x_t-8=0.5^t(x_0-8)\). Every initial condition converges geometrically to 8.

## Problem set
1. Solve a first-order linear difference equation.
2. Classify convergence for positive and negative slopes.
3. Apply the derivative stability criterion to a nonlinear map.
4. Determine stability of a two-dimensional linear system.
5. Analyze a cobweb model under alternative supply and demand slopes.

## Computational lab
Simulate linear and nonlinear difference equations over alternative parameters and initial values. Plot cobweb diagrams and compare simulated paths with analytical stability conditions.

## Coding exercises

These exercises apply the calculus of this lecture to **real economic data** in Python, mixing symbolic (`sympy`),
numerical (`scipy`) and statistical (`statsmodels`) tools. Install once with
`pip install sympy scipy numpy pandas statsmodels wooldridge`. Try the starter code first, then open the solution.
Every printed result below comes from running the solution code exactly as shown.

### Exercise 14.1 — A first-order difference equation for unemployment

*Data:* `phillips` — U.S. annual unemployment, 1948–2003.

1. Estimate \(u_t=a+bu_{t-1}+e_t\) by OLS. Find the fixed point \(u^*=a/(1-b)\) and check stability (\(|b|<1\)).
2. Write the general solution \(u_t=u^*+b^t(u_0-u^*)\) and simulate the deterministic path from the 2003 level for 10 years.
3. Plot the phase diagram: \(u_t\) against \(u_{t-1}\) with the 45° line. How many years until the gap halves?

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
p = wd.data("phillips")
p["u1"] = p["unem"].shift(1)
r = smf.ols("unem ~ u1", data=p).fit()
# TODO: fixed point, path u_t = u* + b^t (u_0 - u*)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
p = wd.data("phillips"); p["u1"] = p["unem"].shift(1)
r = smf.ols("unem ~ u1", data=p).fit()
a, b = r.params["Intercept"], r.params["u1"]; us = a / (1 - b)
print(f"u_t = {a:.3f} + {b:.3f} u_(t-1);  fixed point u* = {us:.3f}%;  stable: {abs(b) < 1}")
u0 = p["unem"].iloc[-1]
print(f"start: {int(p['year'].iloc[-1])} unemployment {u0:.1f}%")
for t in range(0, 11, 2):
    print(f"  {int(p['year'].iloc[-1]) + t}: {us + b**t * (u0 - us):.3f}%")
print(f"half-life of the gap: {np.log(0.5)/np.log(b):.2f} years")
```

Output:

```text
u_t = 1.490 + 0.742 u_(t-1);  fixed point u* = 5.783%;  stable: True
start: 2003 unemployment 6.0%
  2003: 6.000%
  2005: 5.902%
  2007: 5.849%
  2009: 5.819%
  2011: 5.803%
  2013: 5.794%
half-life of the gap: 2.33 years
```

The fitted equation has a fixed point at about 5.8% and a root of 0.74, so it is stable, and any gap closes by 26% a
year — a half-life of 2.3 years. Starting from 6.0% in 2003, the deterministic path converges slowly to 5.8%. The
deterministic solution describes the expected path only; the actual path is hit by new shocks every year (Econometrics
Lecture 9).

</details>

### Exercise 14.2 — Business cycles from a second-order difference equation

*Data:* `macrodata` — U.S. quarterly real GDP growth, 1959–2009.

Samuelson's multiplier–accelerator model produces a second-order difference equation, \(y_t=a_1y_{t-1}+a_2y_{t-2}\).

1. Estimate an AR(2) for quarterly GDP growth (in deviations from the mean).
2. Find the roots of the characteristic equation \(\lambda^2-a_1\lambda-a_2=0\). Are they real or complex? Is the system stable?
3. If complex, compute the implied cycle period \(2\pi/\theta\) where \(\lambda=re^{\pm i\theta}\), and the damping factor \(r\). Compute the impulse response for 16 quarters.
4. Repeat for an AR(2) in the **level** of the unemployment rate (`unemp`). Which series shows a clearer cycle?

```python
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
m = sm.datasets.macrodata.load_pandas().data
g = 400 * np.diff(np.log(m["realgdp"].to_numpy()))
fit = AutoReg(g, lags=2, trend="c").fit()
# TODO: roots of lambda^2 - a1 lambda - a2; period 2 pi / angle
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
m = sm.datasets.macrodata.load_pandas().data
series = {"GDP growth": 400 * np.diff(np.log(m["realgdp"].to_numpy())), "unemployment rate": m["unemp"].to_numpy()}
for name, x in series.items():
    fit = AutoReg(x, lags=2, trend="c").fit(); a1, a2 = fit.params[1], fit.params[2]
    roots = np.roots([1, -a1, -a2]); mod = np.abs(roots)
    print(f"{name}: a1 = {a1:.3f}, a2 = {a2:.3f}; roots {np.round(roots, 3)}; |roots| {mod.round(3)}; stable {bool(np.all(mod < 1))}")
    if np.iscomplexobj(roots) and abs(roots[0].imag) > 1e-12:
        th = abs(np.angle(roots[0])); print(f"   complex roots: period {2*np.pi/th:.1f} quarters, damping r = {mod[0]:.3f}")
    else:
        print("   real roots: no oscillation, monotone adjustment")
    irf = [1.0, a1]
    for _ in range(14): irf.append(a1 * irf[-1] + a2 * irf[-2])
    print("   impulse response:", np.round(irf[:16:3], 3))
```

Output:

```text
GDP growth: a1 = 0.269, a2 = 0.159; roots [ 0.556 -0.287]; |roots| [0.556 0.287]; stable True
   real roots: no oscillation, monotone adjustment
   impulse response: [1.    0.105 0.02  0.003 0.001 0.   ]
unemployment rate: a1 = 1.639, a2 = -0.680; roots [0.82+0.09j 0.82-0.09j]; |roots| [0.825 0.825]; stable True
   complex roots: period 57.7 quarters, damping r = 0.825
   impulse response: [1.    2.177 1.997 1.437 0.898 0.502]
```

For GDP growth, both roots are real and small: shocks die out within two or three quarters with no oscillation. Growth
rates have little persistence. The **unemployment rate** behaves differently: its roots are complex with modulus 0.83, so a
shock produces a hump-shaped response that peaks after a few quarters and then decays with a cycle. The implied period of
about 58 quarters is long, and the damping is strong, so the "cycle" is barely visible. Samuelson's oscillations exist,
but the data favor damped, mostly monotone dynamics.

</details>

## Reading
Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*; Simon and Blume, *Mathematics for Economists*; Strogatz, *Nonlinear Dynamics and Chaos*.