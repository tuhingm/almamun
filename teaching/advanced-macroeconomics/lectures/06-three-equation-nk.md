# Lecture 06 — The Three-Equation New Keynesian Model

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Combine the dynamic IS relation, Phillips curve, and monetary-policy rule; analyze equilibrium responses to demand, supply, and policy shocks.

## 1. Core system
The canonical linearized model is
\[
x_t=E_tx_{t+1}-\frac{1}{\sigma}(i_t-E_t\pi_{t+1}-r_t^n),
\]
\[
\pi_t=\beta E_t\pi_{t+1}+\kappa x_t+u_t,
\]
\[
i_t=\rho_i i_{t-1}+(1-\rho_i)(\phi_\pi\pi_t+\phi_xx_t)+\varepsilon_t^i.
\]
Here \(x_t\) is the output gap, \(r_t^n\) the natural real rate, \(u_t\) a cost-push shock, and \(\varepsilon_t^i\) a monetary-policy shock.

## 2. Taylor principle and determinacy
In the simplest model without policy inertia or output feedback, the Taylor principle requires the nominal rate to rise more than one-for-one with inflation, \(\phi_\pi>1\). This is a useful benchmark, not a universal determinacy test for every extended model. With inertia, additional state variables, and fiscal interactions, apply the full rational-expectations eigenvalue conditions.

## 3. Shock transmission
A positive natural-rate shock raises demand at a given real rate. A positive cost-push shock raises inflation while reducing the feasible stabilization trade-off. A contractionary policy shock raises the real rate and lowers the output gap and inflation over time.

## 4. Worked analysis
Set \(\rho_i=0\), \(\phi_x=0\), and suppress shocks. Substitute the policy rule into the IS equation and use the Phillips curve to obtain a two-equation system in \((x_t,\pi_t)\). Discuss how a higher \(\phi_\pi\) changes the real-rate response to inflation.

## 5. Computational lab
Implement the three-equation model in Dynare. Generate IRFs for a demand shock, cost-push shock, and policy shock. Compare \(\phi_\pi=0.8,1.2,1.8\), checking determinacy rather than assuming it.

## Problem set
1. Derive the IS relation from household optimality.
2. Explain the Taylor principle.
3. Compare demand and cost-push shocks.
4. Explain why the output gap and natural output must be defined consistently.

## Further reading
Galí, *Monetary Policy, Inflation, and the Business Cycle*; Clarida, Galí and Gertler (1999).
