# Lecture 12 — Multiple Integrals, Change of Variables, and Probability

## Learning objectives
Evaluate double and triple integrals; use Fubini's theorem; change coordinates with Jacobians; derive marginal densities and expectations.

## 1. Double integrals
For an integrable function over a region \(D\subset\mathbb R^2\), the double integral \(\iint_D f(x,y)dA\) aggregates values over area. Iterated integration is valid under Fubini's conditions, including integrability; Tonelli's theorem applies to nonnegative measurable functions.

## 2. Integration over general regions
Describe regions using bounds such as \(a\le x\le b\), \(g_1(x)\le y\le g_2(x)\). Changing the order of integration requires rewriting the same region with the other variable outermost. Sketch the region before changing limits.

## 3. Change of variables
For a differentiable one-to-one transformation \((u,v)\mapsto(x,y)\),
\[
\iint_D f(x,y)dxdy
=\iint_{D^*}f(x(u,v),y(u,v))\left|\frac{\partial(x,y)}{\partial(u,v)}\right|dudv.
\]
The absolute Jacobian determinant adjusts local area. Polar coordinates use \(x=r\cos\theta\), \(y=r\sin\theta\), with Jacobian \(r\).

## 4. Probability applications
For joint density \(f_{X,Y}\), marginals integrate out other variables. Expectations are integrals of functions against the joint density:
\[
E[g(X,Y)]=\iint g(x,y)f_{X,Y}(x,y)dxdy.
\]
Covariance is \(E[XY]-E[X]E[Y]\) when the moments exist. Transformations of random vectors use the same Jacobian principle.

## 5. Applications
Multiple integrals calculate volumes, expected utilities, aggregate welfare, and probability over regions. In economics, they can represent expected payoffs under joint uncertainty and aggregation over heterogeneous agents.

## Worked example
For a uniform point on the unit disk, the density is \(1/\pi\). Polar coordinates give \(P(R\le r)=\pi r^2/\pi=r^2\), for \(0\le r\le1\), and hence radial density \(f_R(r)=2r\).

## Problem set
1. Evaluate a double integral over a rectangular region.
2. Reverse the order of integration over a triangular region.
3. Derive the polar-coordinate Jacobian.
4. Obtain a marginal density from a joint density.
5. Compute an expectation under a bivariate distribution.

## Computational lab
Evaluate integrals symbolically and numerically. Compare Cartesian and polar integration for a circular region and verify numerical marginal densities.

## Reading
Apostol, *Calculus*, Vol. 2; Casella and Berger, *Statistical Inference*; Simon and Blume, *Mathematics for Economists*.