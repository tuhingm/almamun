# Lecture 10 — DSGE Solution Methods: Linearization, Perturbation, and QZ

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Compute deterministic steady states; linearize equilibrium conditions; understand rational-expectations solution and Blanchard–Kahn conditions.

## 1. General equilibrium system
Write a DSGE model as
\[
E_tF(y_{t+1},y_t,x_{t+1},x_t,z_t;\theta)=0,
\]
where \(y_t\) are jump variables, \(x_t\) predetermined states, \(z_t\) exogenous shocks, and \(\theta\) structural parameters.

## 2. Steady state
Set shocks to their steady values and impose time invariance. Solve \(F(\bar y,\bar y,\bar x,\bar x,\bar z;\theta)=0\). Check positivity, feasibility, market clearing, and residuals. A solver's convergence flag alone is not sufficient.

## 3. First-order approximation
Let deviations from steady state be denoted by hats. A first-order Taylor expansion produces a linear rational-expectations system. In state-space form:
\[
A E_t s_{t+1}=B s_t+C\varepsilon_t.
\]
The generalized Schur decomposition (QZ) separates stable and unstable generalized eigenvalues.

## 4. Blanchard–Kahn conditions
A unique stable rational-expectations solution requires the number of unstable roots to match the number of non-predetermined variables, under the chosen system representation and regularity conditions. Report eigenvalue conventions and variable timing; do not count roots without documenting the matrix pencil.

## 5. Higher-order perturbation
Second- and third-order approximations capture curvature and risk effects omitted at first order. Pruning may be needed to control explosive simulation artifacts. Approximation accuracy should be evaluated over the shock region relevant to the application.

## 6. Worked diagnostic
For a solved model, tabulate generalized eigenvalues, classify stable roots by modulus under the software's convention, count jump variables, and compare counts. If counts fail, inspect timing, redundant equations, and parameter restrictions.

## 7. Computational lab
Implement a small RBC model in Dynare. Save steady-state residuals, eigenvalues, decision rules, and IRFs. Re-solve after changing one parameter and document whether determinacy changes.

## Problem set
1. Define states and jump variables.
2. Explain the role of QZ decomposition.
3. State the Blanchard–Kahn root-count condition.
4. Explain why a steady-state residual check is necessary.

## Further reading
Klein (2000); Sims (2002); Heer and Maussner, *Dynamic General Equilibrium Modeling*.
