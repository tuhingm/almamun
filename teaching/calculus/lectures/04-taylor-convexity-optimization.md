# Lecture 04 — Taylor Approximation, Convexity, and Single-Variable Optimization

## Learning objectives
Derive Taylor approximations; classify stationary points; use convexity; solve economic optimization problems and interpret first- and second-order conditions.

## 1. Taylor's theorem
\[
f(a+h)=f(a)+f'(a)h+\frac12f''(a)h^2+o(h^2).
\]
The first-order approximation is local and linear; the second-order term captures curvature. Taylor remainders determine approximation error.

## 2. Necessary conditions
For an interior local optimum \(x^*\) of a differentiable function, \(f'(x^*)=0\). This is necessary, not sufficient. Boundary points and points where the derivative does not exist must also be considered.

## 3. Second-order conditions
If \(f'(x^*)=0\) and \(f''(x^*)<0\), then \(x^*\) is a strict local maximum; if \(f''(x^*)>0\), it is a strict local minimum. If \(f''(x^*)=0\), the test is inconclusive. Global optimality follows under suitable concavity or convexity on the full feasible interval.

## 4. Convexity
A differentiable function is convex if its derivative is nondecreasing. Twice differentiability and \(f''\ge0\) provide a sufficient characterization. Strict convexity implies at most one minimizer. Jensen's inequality links convexity to expectations.

## 5. Economic optimization
For profit \(\pi(q)=pq-C(q)\), an interior optimum satisfies \(p=C'(q)\); if \(C''(q)>0\), profit is locally concave. Interpret the FOC as marginal benefit equaling marginal cost.

## Worked example
Let \(\pi(q)=20q-q^2-16\), with \(q\ge0\). The FOC gives \(q^*=10\); the SOC is \(-2<0\), and this is the global maximum on the nonnegative domain.

## Problem set
1. Derive the second-order Taylor approximation for \(e^x\) around zero.
2. Classify stationary points using the second derivative.
3. Explain why a zero second derivative is inconclusive.
4. Prove a differentiable convex function lies above each tangent.
5. Solve profit maximization with a capacity constraint.

## Computational lab
Plot objectives and tangent/quadratic approximations. Numerically optimize convex and non-convex functions; compare local optimization with grid search.

## Reading
Apostol, *Calculus*, Vol. 1; Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*; Simon and Blume, *Mathematics for Economists*.