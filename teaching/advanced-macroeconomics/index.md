# Advanced Macroeconomics
## Graduate course in dynamic macroeconomic theory and policy

**Prerequisites:** Intermediate macroeconomics, calculus, linear algebra, probability, and introductory econometrics.

## Learning outcomes
Derive intertemporal household and firm conditions; characterize competitive equilibrium; solve dynamic systems; analyze monetary and fiscal policy; calibrate and estimate DSGE models; interpret impulse responses and counterfactuals.

## 1. National accounts and dynamic constraints
Goods-market identity Y_t=C_t+I_t+G_t+NX_t. Distinguish stocks and flows, nominal and real quantities, national saving, current accounts, and sectoral balances. Derive government debt accumulation and the intertemporal government budget constraint.

## 2. Intertemporal consumption and saving
Household problem: max E_0 Σ β^t u(C_t), subject to A_(t+1)=(1+r_t)A_t+Y_t-C_t. Derive u'(C_t)=βE_t[(1+r_(t+1))u'(C_(t+1))]. For CRRA preferences, derive the deterministic consumption-growth condition. Discuss transversality, borrowing constraints, precautionary saving, and uncertainty.

## 3. Production, capital and investment
Production Y_t=A_tK_t^αL_t^(1-α), capital law K_(t+1)=(1-δ)K_t+I_t. Derive factor prices, firm optimization, investment adjustment costs, Tobin's q, and balanced-growth normalization.

## 4. Solow and Ramsey growth
Capital per effective worker, steady states, golden-rule capital, comparative statics, saddle-path stability, and transitional dynamics. Derive the Ramsey Euler equation and distinguish exogenous saving from intertemporal optimization.

## 5. Real business cycle models
Technology process log A_t=ρ_a log A_(t-1)+ε_t. Set up household labor-leisure choice, firms, market clearing, and competitive equilibrium. Study calibration, impulse responses, variance decompositions, and propagation mechanisms.

## 6. Nominal rigidities and New Keynesian pricing
Derive Calvo reset-price behavior and the log-linear Phillips curve π_t=βE_tπ_(t+1)+κx_t+u_t. Explain price index aggregation, real marginal cost, Rotemberg adjustment costs, and indexation.

## 7. The three-equation New Keynesian model
IS relation: x_t=E_tx_(t+1)-(1/σ)(i_t-E_tπ_(t+1)-r_t^n). Phillips curve: π_t=βE_tπ_(t+1)+κx_t+u_t. Policy rule: i_t=ρ_i i_(t-1)+(1-ρ_i)(φ_ππ_t+φ_xx_t)+ε_t^i. Analyze determinacy, the Taylor principle, cost-push shocks, and stabilization trade-offs.

## 8. DSGE solution methods
Steady-state computation, log-linearization, first-order perturbation, generalized Schur/QZ decomposition, Blanchard-Kahn conditions, state-space representation, Kalman filtering, and solution diagnostics. Validate equilibrium residuals and timing conventions.

## 9. Monetary policy
Taylor rules, interest-rate smoothing, inflation targeting, commitment versus discretion, time inconsistency, zero lower bound, forward guidance, unconventional policy, and policy transmission.

## 10. Fiscal policy and sovereign debt
Government budget constraints, primary balances, debt-to-GDP dynamics, fiscal multipliers, automatic stabilizers, tax distortions, Ricardian equivalence, fiscal limits, and monetary-fiscal interactions.

## 11. Open-economy macroeconomics
Balance of payments, real and nominal exchange rates, uncovered interest parity, small-open-economy models, terms-of-trade shocks, capital flows, and monetary policy under exchange-rate regimes.

## 12. Heterogeneous-agent macroeconomics
Incomplete markets, idiosyncratic labor-income risk, borrowing constraints, precautionary saving, household asset distributions, and aggregate equilibrium. Bellman equation: V(a,z)=max_{c,a'}{u(c)+βΣ_z'P(z'|z)V(a',z')} subject to the household budget and borrowing limit.

## 13. HANK and distribution dynamics
Continuous-time HJB equation, Kolmogorov forward equation, stationary distributions, market clearing, heterogeneous marginal propensities to consume, redistribution, and monetary/fiscal transmission. Introduce sequence-space Jacobians and numerical distribution iteration.

## 14. Estimation and empirical validation
Calibration, simulated method of moments, likelihood, Bayesian estimation, prior specification, MCMC diagnostics, posterior predictive checks, identification, model comparison, and sensitivity analysis.

## Problem sets with solution guidance
1. Derive the household Euler equation from the Lagrangian and state the transversality condition.
2. Solve for the Solow steady state and compare the effects of saving and population growth.
3. Derive the firm's labor demand and capital Euler equation.
4. Explain the role of the Taylor principle in the canonical New Keynesian model.
5. Construct a debt-ratio transition equation and identify the interest-growth differential.
6. Describe the household problem and equilibrium fixed point in an incomplete-markets model.

## Computational labs
- Python/Julia: dynamic programming, steady-state solving, numerical integration, and heterogeneous-agent distribution iteration.
- MATLAB/Dynare: RBC and New Keynesian DSGE models, perturbation, impulse responses, Bayesian estimation.
- Every lab includes parameter tables, equilibrium residuals, steady-state checks, code, and replication instructions.

## Assessment
Theory problem sets 25%; DSGE replication 25%; policy analysis 15%; midterm 15%; final research paper 20%.

## Core references
Ljungqvist and Sargent, Recursive Macroeconomic Theory; Galí, Monetary Policy, Inflation, and the Business Cycle; Woodford, Interest and Prices; Heer and Maussner, Dynamic General Equilibrium Modeling; Acemoglu, Introduction to Modern Economic Growth.
