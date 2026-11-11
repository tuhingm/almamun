# Lecture 11 — Implicit and Inverse Function Theorems

## Learning objectives
State local implicit and inverse function results; derive comparative statics; analyze equilibrium systems using Jacobians; distinguish local from global uniqueness.

## 1. Implicit function theorem
Let \(F(x,y)=0\), where \(F\) is continuously differentiable and \(F_y\ne0\) at a solution \((x_0,y_0)\). Locally, the equation defines \(y=h(x)\), with
\[
h'(x)=-\frac{F_x}{F_y}.
\]
The theorem is local: it does not establish a globally defined function or global uniqueness.

## 2. Comparative statics
If an equilibrium condition is \(F(y,\theta)=0\), then
\[
\frac{dy^*}{d\theta}=-\frac{F_\theta}{F_y}
\]
at a regular equilibrium. The sign depends on both partial derivatives. For a system \(F(x,\theta)=0\), the derivative is \(dx^*/d\theta=-F_x^{-1}F_\theta\) when the Jacobian is nonsingular.

## 3. Inverse function theorem
If \(f:\mathbb R^n\to\mathbb R^n\) is continuously differentiable and its Jacobian is nonsingular at \(x_0\), then it has a local differentiable inverse near that point. The derivative of the inverse is the inverse Jacobian.

## 4. Equilibrium systems
For multiple markets or endogenous variables, write equilibrium conditions as a vector system. The Jacobian describes local responses and regularity. Singular Jacobians signal failure of this local theorem; they may correspond to bifurcation, redundant equations, or other structural features requiring separate analysis.

## Worked example
Let equilibrium satisfy \(F(y,\theta)=y^2+\theta y-1=0\). Then \(dy/d\theta=-y/(2y+\theta)\) wherever \(2y+\theta\ne0\). Evaluate the derivative at a specified equilibrium branch.

## Problem set
1. Derive the scalar implicit derivative.
2. Compute comparative statics for a supply-demand equilibrium.
3. State the nonsingularity condition for a vector system.
4. Explain why a local inverse need not be globally unique.
5. Interpret a singular equilibrium Jacobian.

## Computational lab
Solve nonlinear equilibrium systems numerically. Compare finite-difference comparative statics with derivatives from the implicit function theorem.

## Coding exercises

These exercises apply the calculus of this lecture to **real economic data** in Python, mixing symbolic (`sympy`),
numerical (`scipy`) and statistical (`statsmodels`) tools. Install once with
`pip install sympy scipy numpy pandas statsmodels wooldridge`. Try the starter code first, then open the solution.
Every printed result below comes from running the solution code exactly as shown.

### Exercise 11.1 — Comparative statics of the Solow steady state with the implicit function theorem

*Data:* `macrodata` — U.S. investment share of GDP and population growth, used to calibrate the model.

In the Solow model, steady-state capital per worker \(k^*\) solves \(F(k,s)=s\,k^{\alpha}-(n+\delta)k=0\).

1. Calibrate \(s\) as the average ratio of real investment to real GDP and \(n\) as average annual population growth; set \(\alpha=1/3\), \(\delta=0.05\).
2. Solve for \(k^*\) numerically (`brentq`), without using the closed form.
3. Use the implicit function theorem, \(dk^*/ds=-F_s/F_k\), to compute the effect of a higher saving rate, and the elasticity \(\frac{dk^*}{ds}\frac{s}{k^*}\). Compare with a finite difference and with the closed-form elasticity \(1/(1-\alpha)\).

```python
import numpy as np, statsmodels.api as sm
from scipy import optimize
m = sm.datasets.macrodata.load_pandas().data
s = (m["realinv"] / m["realgdp"]).mean(); n = np.log(m["pop"]).diff().mean() * 4
a, dl = 1/3, 0.05
F = lambda k, s: s * k**a - (n + dl) * k
# TODO: brentq for k*, IFT derivative: -F_s/F_k with F_s = k^a, F_k = s a k^(a-1) - (n+dl)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, statsmodels.api as sm
from scipy import optimize
m = sm.datasets.macrodata.load_pandas().data
s = (m["realinv"] / m["realgdp"]).mean(); n = np.log(m["pop"]).diff().mean() * 4
a, dl = 1/3, 0.05
F = lambda k, s: s * k**a - (n + dl) * k
kstar = lambda s: optimize.brentq(F, 1e-6, 1e6, args=(s,), xtol=1e-14)
k = kstar(s)
print(f"calibration: s = {s:.4f}, n = {n:.4f}, alpha = {a:.3f}, delta = {dl}")
print(f"steady state k* = {k:.4f}  (closed form {(s/(n+dl))**(1/(1-a)):.4f})")
Fs, Fk = k**a, s * a * k**(a - 1) - (n + dl)
dk = -Fs / Fk
h = 1e-6; fd = (kstar(s + h) - kstar(s - h)) / (2 * h)
print(f"IFT: dk*/ds = {dk:.4f}  vs finite difference {fd:.4f};  F_k = {Fk:.4f} < 0 (stable)")
print(f"elasticity dk*/ds * s/k* = {dk * s / k:.4f}  vs closed form 1/(1-alpha) = {1/(1-a):.4f}")
```

Output:

```text
calibration: s = 0.1328, n = 0.0110, alpha = 0.333, delta = 0.05
steady state k* = 3.2142  (closed form 3.2142)
IFT: dk*/ds = 36.3174  vs finite difference 36.3174;  F_k = -0.0406 < 0 (stable)
elasticity dk*/ds * s/k* = 1.5000  vs closed form 1/(1-alpha) = 1.5000
```

Calibrated to the U.S. (investment share 13%, population growth 1.1%), the numerical root matches the closed-form steady
state. The implicit function theorem gives \(dk^*/ds=-F_s/F_k\) without solving for \(k^*\) explicitly, and it agrees with
the finite difference. \(F_k<0\) at the steady state, which is both the condition for the IFT to apply and the condition
for stability. The elasticity of steady-state capital with respect to the saving rate is exactly \(1/(1-\alpha)=1.5\).

</details>

### Exercise 11.2 — Yield to maturity as an inverse function, and duration

*Data:* `intqrt` — U.S. 12-month Treasury yield (average, used as a reference yield).

The price of a bond is a decreasing function of its yield, \(P(y)=\sum_t c_t/(1+y)^t\). Its inverse \(y(P)\) is the yield to maturity.

1. For a 10-year bond with a 6% annual coupon and face value 100, compute \(P\) at the average 12-month yield.
2. Invert numerically: find \(y\) from \(P\), and check that the round trip recovers the yield.
3. By the inverse function theorem, \(dy/dP=1/P'(y)\). Compute it, and the modified duration \(-P'(y)/P\). Use duration to approximate the price change from a 1-point yield rise and compare with the exact change.

```python
import numpy as np, wooldridge as wd
from scipy import optimize
y0 = wd.data("intqrt")["r12"].mean() / 100
cf = np.r_[np.full(9, 6.0), 106.0]; t = np.arange(1, 11)
P = lambda y: np.sum(cf / (1 + y)**t); dP = lambda y: -np.sum(t * cf / (1 + y)**(t + 1))
# TODO: price, inverse via brentq, dy/dP = 1/dP, modified duration
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
from scipy import optimize
y0 = wd.data("intqrt")["r12"].mean() / 100
cf = np.r_[np.full(9, 6.0), 106.0]; t = np.arange(1, 11)
P = lambda y: np.sum(cf / (1 + y)**t); dP = lambda y: -np.sum(t * cf / (1 + y)**(t + 1))
p0 = P(y0)
yinv = optimize.brentq(lambda y: P(y) - p0, -0.5, 1.0, xtol=1e-15)
print(f"yield {100*y0:.4f}% -> price {p0:.4f};  inverted yield {100*yinv:.10f}%")
print(f"dy/dP = 1/P'(y) = {1/dP(y0):.6f}  (yield change per $1 of price)")
D = -dP(y0) / p0
print(f"modified duration {D:.4f} years")
print(f"1-point yield rise: duration approximation {-D*0.01*100:.3f}% of price, exact {100*(P(y0+0.01)/p0 - 1):.3f}%")
```

Output:

```text
yield 6.9269% -> price 93.4675;  inverted yield 6.9269354824%
dy/dP = 1/P'(y) = -0.001483  (yield change per $1 of price)
modified duration 7.2163 years
1-point yield rise: duration approximation -7.216% of price, exact -6.890%
```

Because the price is strictly decreasing in the yield, the inverse function exists, and the numerical inversion recovers
the yield to ten decimals. The inverse function theorem gives \(dy/dP=1/P'(y)\): a \$1 price fall raises the yield by about
15 basis points. Modified duration (7.2 years) predicts a 7.2% price fall for a 1-point yield rise, against an exact fall
of 6.9%. The gap is convexity: the price–yield curve is convex, so duration overstates losses and understates gains (Finance,
Lecture 7).

</details>

## Reading
Simon and Blume, *Mathematics for Economists*; Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*; Apostol, *Calculus*, Vol. 2.