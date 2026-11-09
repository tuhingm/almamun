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

## Reading
Boyce, DiPrima, and Meade, *Elementary Differential Equations*; Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*; Strogatz, *Nonlinear Dynamics and Chaos*.