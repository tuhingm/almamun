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

## Reading
Simon and Blume, *Mathematics for Economists*; Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*; Apostol, *Calculus*, Vol. 2.