# Calculus — Problem Sets and Selected Solution Guidance

The problems are organized by lecture. Students should show derivations, state assumptions, and interpret results. Selected guidance is included to clarify the intended mathematical method; it is not a substitute for complete proofs.

## Set 1 — Limits and continuity
1. Prove \(\lim_{x\to2}(3x+1)=7\) using epsilon-delta.
2. Evaluate \(\lim_{x\to0}(\sqrt{1+x}-1)/x\).
3. Construct examples of removable, jump, and infinite discontinuities.

**Guidance:** For Question 1, choose \(\delta=\varepsilon/3\), since \(|3x+1-7|=3|x-2|\). For Question 2, rationalize the numerator before taking the limit.

## Set 2 — Sequences and series
1. Determine convergence of \(a_n=(2n+1)/(n+3)\).
2. Test \(\sum_{n=1}^\infty1/[n(n+1)]\) and find its sum.
3. Determine absolute or conditional convergence of \(\sum(-1)^{n+1}/n\).

**Guidance:** Use the sequence limit for Question 1. For Question 2, use partial fractions and telescoping. For Question 3, apply the alternating-series test and separately test absolute convergence.

## Set 3 — Differentiation
1. Differentiate \(x^2e^{3x}\).
2. Differentiate \(\log(1+x^2)\).
3. Use the mean value theorem to prove a monotonicity result.

**Guidance:** Apply product and chain rules; state the interval and regularity assumptions for the theorem.

## Set 4 — Taylor and optimization
1. Obtain the second-order Taylor polynomial of \(e^x\) at zero.
2. Find and classify stationary points of \(f(x)=x^3-3x\).
3. Maximize \(20q-q^2-16\) over \(q\ge0\).

**Guidance:** Solve the FOC, apply the SOC, and compare with boundary values when the domain is constrained.

## Set 5 — Integration
1. Evaluate \(\int_0^1xe^{x^2}dx\).
2. Evaluate \(\int xe^x dx\).
3. Determine when \(\int_1^\infty x^{-p}dx\) converges.

**Guidance:** Use substitution, integration by parts, and an improper-integral limit, respectively.

## Set 6 — Multivariable limits
1. Test the limit of \(xy/(x^2+y^2)\) at the origin.
2. Describe level curves of \(x^2+4y^2=c\).
3. Explain the role of compactness in the extreme value theorem.

**Guidance:** Two paths with distinct limiting values disprove a multivariate limit.

## Set 7 — Partial derivatives and chain rule
1. Compute the gradient and Hessian of \(f(x,y)=x^2+xy+2y^2\).
2. Derive marginal products for Cobb–Douglas production.
3. Apply the multivariate chain rule to a composite function.

## Set 8 — Unconstrained optimization
1. Classify stationary points using Hessian eigenvalues.
2. Maximize a strictly concave quadratic.
3. Explain why semidefinite curvature is inconclusive.

## Set 9 — Lagrange multipliers
1. Maximize \(x^\alpha y^{1-\alpha}\) subject to \(p_xx+p_yy=m\).
2. Interpret the multiplier as the marginal value of income.
3. Derive the expenditure-minimization dual problem.

**Guidance:** Log-transform the Cobb–Douglas objective, derive both FOCs, and substitute into the budget constraint.

## Set 10 — KKT conditions
1. Minimize \((x-3)^2\) subject to \(x\le2\).
2. Identify active and inactive constraints in a quadratic program.
3. State a sufficient convexity condition for global optimality.

## Set 11 — Implicit functions
1. Derive \(dy/dx\) from \(F(x,y)=0\).
2. Compute equilibrium comparative statics in a supply-demand system.
3. Explain why a nonsingular Jacobian gives local, not necessarily global, invertibility.

## Set 12 — Multiple integrals
1. Reverse the order of integration over a triangular region.
2. Derive the polar-coordinate Jacobian.
3. Obtain marginal densities and expectations from a joint density.

## Set 13 — Differential equations
1. Solve a separable first-order ODE.
2. Solve a linear ODE using an integrating factor.
3. Determine local stability from a Jacobian matrix.

## Set 14 — Difference equations
1. Solve \(x_{t+1}=a+bx_t\).
2. Classify stability for positive and negative slopes.
3. Analyze a cobweb adjustment process.

## Computational assignment
Use Python or R to:
- approximate derivatives and integrals;
- plot functions, contours, gradients, and Hessian eigenvalues;
- solve constrained optimization problems;
- simulate ODE and difference-equation paths;
- compare numerical output with analytical results.

Submit source code, a short methods note, and reproducible output.