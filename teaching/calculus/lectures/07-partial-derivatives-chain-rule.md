# Lecture 07 — Partial Derivatives, Total Differentials, and the Chain Rule

## Learning objectives
Compute partial derivatives; interpret gradients and directional derivatives; derive total differentials; apply multivariate chain rules and Jacobians.

## 1. Partial derivatives
For \(f(x,y)\), the partial derivative \(f_x\) differentiates with respect to \(x\) holding \(y\) fixed. Partial derivatives describe coordinate-wise local responses. Existence of partial derivatives alone does not guarantee differentiability.

## 2. Gradient and directional derivative
The gradient is \(\nabla f=(f_{x_1},\ldots,f_{x_n})'\). For a unit direction \(u\), the directional derivative is \(D_uf=\nabla f'u\). Under the Euclidean norm, the gradient points in the direction of steepest local increase.

## 3. Differentiability and total differential
A function is differentiable at \(x\) if it admits a linear approximation with remainder small relative to the distance:
\[
f(x+h)=f(x)+\nabla f(x)'h+o(\|h\|).
\]
The total differential is \(df=\sum_i f_{x_i}dx_i\). It approximates changes locally; it is not generally an exact finite-change identity.

## 4. Chain rule
If \(z=f(x,y)\), with \(x=x(t)\), \(y=y(t)\), then
\[
\frac{dz}{dt}=f_x\frac{dx}{dt}+f_y\frac{dy}{dt}.
\]
For vector maps, Jacobian matrices multiply in composition order: \(D(f\circ g)=Df(g(x))Dg(x)\).

## 5. Jacobian and transformations
For a vector-valued map \(y=g(x)\), the Jacobian is \(J_g(x)=\partial g/\partial x'\). A nonsingular Jacobian indicates local invertibility under the inverse function theorem. In probability transformations, its determinant adjusts volume.

## Worked example
For \(Q(K,L)=AK^\alpha L^\beta\), \(Q_K=\alpha AK^{\alpha-1}L^\beta\) and \(Q_L=\beta AK^\alpha L^{\beta-1}\). The total differential is \(dQ=Q_KdK+Q_LdL\), giving a first-order approximation to output changes.

## Problem set
1. Compute gradient and Hessian of a quadratic function.
2. Derive directional derivative in a specified direction.
3. Apply the chain rule to a composed production function.
4. Distinguish partial differentiability from total differentiability.
5. Calculate a Jacobian determinant for a coordinate transformation.

## Computational lab
Use SymPy to derive gradients and Jacobians; verify with finite differences and plot gradient fields over contour maps.

## Reading
Apostol, *Calculus*, Vol. 2; Simon and Blume, *Mathematics for Economists*; Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*.