# Lecture 08 — Hessians and Unconstrained Multivariable Optimization

## Learning objectives
Use Hessian matrices to classify curvature; derive multivariate first- and second-order conditions; apply optimization to utility, production, and likelihood functions.

## 1. Hessian matrix
For twice continuously differentiable \(f:\mathbb R^n\to\mathbb R\), the Hessian is
\[
H_f(x)=\left[\frac{\partial^2f}{\partial x_i\partial x_j}\right]_{i,j}.
\]
Under continuity of second partial derivatives, the Hessian is symmetric. It describes local curvature of the function.

## 2. First-order necessary condition
At an interior local optimum \(x^*\), \(\nabla f(x^*)=0\). This condition is not sufficient and does not apply directly at boundary optima. A stationary point may be a maximum, minimum, or saddle point.

## 3. Second-order conditions
If the Hessian is negative definite at a stationary point, it is a strict local maximum; if positive definite, a strict local minimum. Indefiniteness implies a saddle point. Semidefinite Hessians require higher-order or direct analysis. For a symmetric matrix, definiteness can be assessed by eigenvalues or principal minors.

## 4. Concavity and global optima
If \(f\) is concave on a convex domain, every stationary point is a global maximum. Strict concavity implies at most one maximizer. Convexity gives the corresponding result for minimization. These global results are stronger than local SOCs.

## 5. Economic applications
For utility maximization, concavity of utility supports global optimality. For cost minimization, convex cost functions support unique solutions under suitable feasibility and strict convexity. In maximum likelihood, the Hessian of the log-likelihood describes local curvature around an estimate.

## Worked example
Let \(f(x,y)=-(x^2+2y^2)+4x+8y\). The gradient is \((-2x+4,-4y+8)'\), giving \((2,2)\). The Hessian is diagonal with entries -2 and -4, so it is negative definite and the point is a strict global maximum.

## Problem set
1. Compute the gradient and Hessian of a quadratic form.
2. Classify a stationary point using Hessian eigenvalues.
3. Explain why a semidefinite Hessian is inconclusive.
4. State a sufficient condition for global optimality.
5. Apply the SOC to a Cobb–Douglas utility function after taking logarithms.

## Computational lab
Use symbolic algebra and numerical eigenvalues to classify stationary points. Plot contours and identify maxima, minima, and saddle points.

## Reading
Simon and Blume, *Mathematics for Economists*; Sundaram, *A First Course in Optimization Theory*; Apostol, *Calculus*, Vol. 2.