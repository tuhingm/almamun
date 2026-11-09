# Lecture 10 — Inequality Constraints and Karush–Kuhn–Tucker Conditions

## Learning objectives
State KKT conditions; identify active constraints; interpret complementary slackness; distinguish necessary and sufficient conditions; solve inequality-constrained economic problems.

## 1. Standard form
Consider minimizing \(f(x)\) subject to \(g_j(x)\le0\), \(j=1,\ldots,m\), and equality constraints as needed. Define
\[
\mathcal L(x,\mu)=f(x)+\sum_{j=1}^m\mu_jg_j(x).
\]
For a differentiable problem, KKT conditions include stationarity, primal feasibility, dual feasibility \(\mu_j\ge0\), and complementary slackness \(\mu_jg_j(x)=0\).

## 2. Interpretation
An inactive constraint has \(g_j(x^*)<0\), implying \(\mu_j=0\). An active constraint binds and may have a positive multiplier. Complementary slackness links scarcity values to binding restrictions. Multipliers are local shadow prices under regularity.

## 3. Constraint qualifications
KKT conditions are not automatically necessary for every nonlinear program. Conditions such as LICQ or an appropriate weaker qualification ensure the existence of multipliers at a local optimum. State the qualification used rather than treating KKT as universally necessary.

## 4. Sufficiency and convexity
For a convex minimization problem—convex objective and convex feasible set—any feasible KKT point is globally optimal. Strict convexity can give uniqueness. In nonconvex problems, KKT points may be local minima, maxima, or saddle points.

## 5. Economic applications
Capacity, non-negativity, borrowing, labor-supply, and resource constraints generate inequality restrictions. A binding resource constraint can have a positive shadow value; a slack resource constraint has zero multiplier under complementary slackness.

## Worked example
Minimize \((x-3)^2\) subject to \(x\le2\). Write \(g(x)=x-2\le0\). The constrained optimum is \(x^*=2\), with multiplier \(\mu=2\), satisfying stationarity \(2(x-3)+\mu=0\), feasibility, dual feasibility, and complementary slackness.

## Problem set
1. State all KKT conditions for a problem with two inequalities.
2. Identify active constraints in a numerical solution.
3. Solve a one-dimensional constrained quadratic.
4. Explain the role of LICQ.
5. State conditions under which KKT conditions are sufficient for global optimality.

## Computational lab
Solve quadratic programs with binding and slack constraints. Compare analytical KKT solutions with numerical optimizers and verify complementary slackness.

## Reading
Sundaram, *A First Course in Optimization Theory*; Nocedal and Wright, *Numerical Optimization*; Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*.