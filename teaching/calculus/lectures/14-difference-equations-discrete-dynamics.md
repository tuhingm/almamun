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

## Reading
Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*; Simon and Blume, *Mathematics for Economists*; Strogatz, *Nonlinear Dynamics and Chaos*.