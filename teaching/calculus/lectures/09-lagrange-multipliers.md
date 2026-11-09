# Lecture 09 — Equality Constraints and Lagrange Multipliers

## Learning objectives
Derive Lagrange conditions; interpret multipliers as shadow values; solve constrained optimization problems; understand regularity and second-order conditions.

## 1. Equality-constrained problem
Consider maximizing \(f(x)\) subject to \(g(x)=c\). Define
\[
\mathcal L(x,\lambda)=f(x)+\lambda[c-g(x)].
\]
At a regular interior constrained optimum, stationarity and feasibility require \(\nabla f(x^*)-\lambda\nabla g(x^*)=0\) and \(g(x^*)=c\). The constraint qualification requires a nonzero constraint gradient in the single-constraint case.

## 2. Geometric interpretation
At an interior constrained optimum, the objective gradient is parallel to the constraint gradient. The objective's level surface is tangent to the constraint surface. The multiplier scales the constraint normal and measures the marginal value of relaxing the constraint under the stated sign convention.

## 3. Comparative statics and envelope interpretation
Let \(V(c)=\max_{g(x)=c}f(x)\). Under regularity and differentiability, the envelope result gives \(V'(c)=\lambda^*\) for the Lagrangian convention above. The multiplier is a local shadow value, not necessarily a globally constant effect.

## 4. Second-order conditions
The Hessian of the Lagrangian is evaluated along feasible directions tangent to the constraint. For a maximum, the quadratic form must be nonpositive on the tangent space; negative definiteness on nonzero feasible directions gives a strict local maximum. The unrestricted Hessian of the objective alone is not the correct constrained test.

## 5. Cobb–Douglas utility
Maximize \(x^\alpha y^{1-\alpha}\) subject to \(p_xx+p_yy=m\), with positive prices and income. Using log utility gives FOCs \(\alpha/x=\lambda p_x\) and \((1-\alpha)/y=\lambda p_y\), plus the budget equation. Solving yields \(x^*=\alpha m/p_x\), \(y^*=(1-\alpha)m/p_y\).

## Problem set
1. Derive FOCs for a two-variable equality-constrained problem.
2. Interpret the multiplier under a clearly stated Lagrangian sign convention.
3. Solve Cobb–Douglas utility maximization.
4. Derive the expenditure-minimization dual problem.
5. Explain why constrained SOCs use feasible directions.

## Computational lab
Solve constrained quadratic and Cobb–Douglas problems symbolically and numerically. Compare the multiplier with a finite-difference approximation to the optimized value function.

## Reading
Sundaram, *A First Course in Optimization Theory*; Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*; Simon and Blume, *Mathematics for Economists*.