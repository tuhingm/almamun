# Lecture 06 — Multivariable Functions, Limits, and Continuity

## Learning objectives
Analyze domains and level sets; define multivariate limits and continuity; distinguish pathwise behavior from genuine limits; interpret surfaces and economic production functions.

## 1. Functions of several variables
A scalar function \(f:D\subseteq\mathbb R^n\to\mathbb R\) maps a vector input to a scalar output. Its domain, range, level sets, and contour surfaces describe its geometry. In economics, examples include utility, production, cost, and welfare functions.

## 2. Multivariate limits
The limit \(\lim_{x\to a}f(x)=L\) requires that every sequence approaching \(a\) within the domain produces function values approaching \(L\). A limit must be independent of the path. Showing two paths with different limiting values disproves existence; checking a finite number of paths cannot generally prove existence.

## 3. Continuity
A function is continuous at \(a\) if its multivariate limit equals \(f(a)\). Polynomials are continuous everywhere; rational functions are continuous wherever denominators are nonzero. Compositions preserve continuity under compatible domains.

## 4. Open and closed sets
A ball around a point defines a neighborhood. Open sets contain a neighborhood around every point; closed sets contain their limit points. Compact subsets of Euclidean space are closed and bounded. Continuity on a compact set implies boundedness and attainment of maximum and minimum.

## 5. Level sets and economic interpretation
A level set \(\{x:f(x)=c\}\) defines an indifference curve or isoquant in common applications. The gradient, developed in the next lecture, is normal to a regular level set. Curvature and feasibility depend on the domain and constraints.

## Worked example
Consider \(f(x,y)=xy/(x^2+y^2)\) away from the origin. Along \(y=x\), the value is \(1/2\); along \(y=-x\), it is \(-1/2\). Therefore the limit at the origin does not exist.

## Problem set
1. Determine the domain of \(\log(x-y)\).
2. Test a multivariate limit along two paths.
3. Explain why checking coordinate axes alone is insufficient.
4. Describe level sets of a Cobb–Douglas production function.
5. State the extreme value theorem on compact subsets of Euclidean space.

## Computational lab
Plot surfaces and contour maps for quadratic, Cobb–Douglas, and non-convergent examples. Numerically explore path dependence near a candidate limit.

## Reading
Apostol, *Calculus*, Vol. 2; Simon and Blume, *Mathematics for Economists*; Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*.