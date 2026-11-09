# Calculus: Mathematical Methods for Economics and Statistics

**Level:** Undergraduate with graduate mathematical extensions. **Coverage:** Limits, differentiation, integration, sequences and series, multivariable calculus, optimization, and dynamic systems.

## 1. Functions, limits and continuity
Domains, ranges, composition, inverse functions, exponential and logarithmic functions. Define limit using ε-δ language. Cover one-sided limits, squeeze theorem, continuity, intermediate value theorem, and limits at infinity.

## 2. Differentiation
Derivative f'(x)=lim[h→0](f(x+h)-f(x))/h. Derive power, product, quotient, chain, exponential and logarithmic rules; implicit differentiation; higher derivatives; Taylor approximation; convexity and concavity. Taylor expansion: f(x+h)=f(x)+f'(x)h+(1/2)f''(x)h²+o(h²).

## 3. Applications
Monotonicity, stationary points, first- and second-order conditions, inflection points, elasticity, comparative statics, marginal cost and revenue. For profit π(q)=pq-c(q), the interior FOC is p=c'(q); convex cost gives the relevant SOC for a local maximum of profit.

## 4. Integration
Riemann sums, antiderivatives, definite and indefinite integrals, substitution, integration by parts, partial fractions, improper integrals, and the fundamental theorem. Applications include areas, surplus, accumulated flows, expected values, and present value.

## 5. Sequences and series
Convergence, monotone bounded sequences, Cauchy criterion, geometric series, p-series, comparison, limit comparison, ratio, root and alternating-series tests. Absolute versus conditional convergence, power series and radius of convergence. Geometric sum Σ ar^n=a/(1-r) for |r|<1.

## 6. Multivariable calculus
Partial derivatives, gradient, directional derivatives, total differential, Jacobian, Hessian, chain rule, and differentiability. The Hessian classifies local curvature; the gradient gives steepest local increase under the Euclidean norm.

## 7. Unconstrained optimization
Interior necessary condition ∇f(x*)=0. Negative definite Hessian gives a strict local maximum; positive definite gives a strict local minimum. Semidefinite cases require further analysis. Applications: utility, cost, likelihood, and least squares.

## 8. Lagrange multipliers
For maximize f(x) subject to g(x)=c, define L=f(x)+λ(c-g(x)). Derive stationarity and feasibility. Interpret λ as the marginal value of relaxing the constraint under the selected sign convention. Cover multiple constraints, constraint qualifications, and bordered Hessians.

## 9. KKT conditions
For inequalities g_j(x)≤0, state stationarity, primal feasibility, dual feasibility μ_j≥0, and complementary slackness μ_jg_j(x)=0. Explain active constraints, convexity, and conditions under which KKT conditions are sufficient for global optimality.

## 10. Implicit and inverse function theorems
If F(x,y)=0 and F_y≠0, then dy/dx=-F_x/F_y locally. Apply to comparative statics and equilibrium systems. Discuss nonsingular Jacobians and local invertibility.

## 11. Multiple integration and probability
Double and triple integrals, Fubini, change of variables and Jacobians. Use integrals to obtain marginal densities, expectations, covariance and transformations of random vectors.

## 12. Differential and difference equations
First-order separable and linear ODEs, integrating factors, second-order constant-coefficient equations, systems xdot=Ax, eigenvalue stability, steady states, and phase diagrams. Discrete dynamics x_(t+1)=f(x_t) are locally stable when |f'(x*)|<1. Applications: capital accumulation, debt, consumption, and adjustment dynamics.

## Worked exercises
1. Differentiate x² exp(3x) using product and chain rules.
2. Maximize Cobb-Douglas utility x^a y^(1-a) subject to p_x x+p_y y=m; derive Marshallian demands.
3. Evaluate integral from 0 to 1 of x exp(x²) dx by substitution.
4. Test convergence of Σ 1/n^p using the p-series test.
5. For F(K,L)=A K^α L^(1-α), derive marginal products and returns to scale.
6. For kdot=s k^α-(n+g+δ)k, derive the positive steady state and discuss local stability.

## Computing
Use SymPy for symbolic derivations, NumPy for numerical approximation, and plots for functions, gradients, contours, and dynamic paths. Verify symbolic outputs numerically and state assumptions explicitly.

## Assessment and references
Problem sets 30%; computational assignments 15%; midterm 20%; final 25%; applied modeling project 10%. References: Apostol, Calculus; Spivak, Calculus; Chiang and Wainwright, Fundamental Methods of Mathematical Economics; Simon and Blume, Mathematics for Economists.

## Lecture series

Each lecture develops the mathematical argument from definitions through proofs, worked applications, problem sessions, and selected readings.

1. [Lecture 01 — Limits, Continuity, and the Logic of Analysis](lectures/01-limits-continuity.md)
