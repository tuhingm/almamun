# Advanced Macroeconomics — Problem Sets and Selected Solution Guidance

These problems are designed to test derivation, economic interpretation, and quantitative implementation. Students should show intermediate steps and state timing conventions.

## Problem set 1 — Household optimization
1. Derive the Euler equation for a household with utility \(E_0\sum_{t=0}^\infty\beta^tu(c_t)\) and budget \(c_t+a_{t+1}=y_t+(1+r_t)a_t\).
2. Under CRRA utility, derive consumption growth for a deterministic return.
3. Add a borrowing limit and derive the complementary-slackness condition.
4. Explain the role of the transversality condition.

**Solution guidance:** Form the Lagrangian with one multiplier per period. Differentiate with respect to consumption and next-period assets, then eliminate the multiplier. Under CRRA, substitute \(u'(c)=c^{-\sigma}\). For a binding borrowing constraint, include a nonnegative multiplier and impose feasibility and complementary slackness.

## Problem set 2 — Production and growth
1. Derive factor prices for \(y=Ak^\alpha n^{1-\alpha}\).
2. Derive the Solow steady state with effective labor growth.
3. Derive the golden-rule condition.
4. Explain saddle-path stability in the Ramsey model.

**Solution guidance:** Equate marginal products to competitive factor prices. Set capital accumulation to zero for the Solow steady state. Differentiate steady-state consumption with respect to capital for the golden rule. In Ramsey, classify capital as predetermined and consumption as a jump variable.

## Problem set 3 — RBC equilibrium
1. Write the household, firm, resource, and capital accumulation conditions.
2. Derive the steady-state capital-output ratio.
3. Explain how shock persistence changes impulse-response duration.
4. Design a calibration table with parameter, value, source, and target.

**Solution guidance:** Combine the Euler equation with the steady-state condition. Keep productivity innovation variance distinct from persistence. Calibration must identify the empirical moment or external estimate supporting each parameter.

## Problem set 4 — New Keynesian model
1. Interpret the IS equation, Phillips curve, and policy rule.
2. Explain the Taylor principle in the simplest benchmark.
3. Compare the response to a natural-rate shock and a cost-push shock.
4. Discuss how price stickiness affects the Phillips-curve slope.

**Solution guidance:** The IS relation links the output gap to the expected future gap and ex-ante real rate. The Phillips curve links inflation to expected inflation and real marginal cost/output gap. Do not infer determinacy from the Taylor coefficient alone in an extended model; check the full eigenvalue conditions.

## Problem set 5 — Fiscal debt
1. Starting from the nominal government budget constraint, derive the approximate debt-ratio transition.
2. Calculate the contribution of the interest-growth differential.
3. State the assumptions required for Ricardian equivalence.
4. Define an impact and cumulative fiscal multiplier.

**Solution guidance:** Divide the nominal constraint by nominal GDP, express debt in real terms, and distinguish primary balances from interest payments. For small rates, the debt change contains approximately \((r-g)b_{t-1}\) plus the primary deficit.

## Problem set 6 — Open economy
1. Derive the saving-investment-current-account identity.
2. State uncovered interest parity using domestic currency per unit of foreign currency.
3. Explain the role of a risk premium.
4. Compare fixed and flexible exchange-rate regimes.

**Solution guidance:** Use national income accounting and define the exchange-rate quote before interpreting appreciation or depreciation. Add a risk-premium term to UIP and maintain a consistent sign convention.

## Problem set 7 — DSGE solution and estimation
1. Define states and jump variables in a linear rational-expectations model.
2. State the Blanchard–Kahn root-count condition.
3. Write the state-space measurement system.
4. Derive the Gaussian innovation log likelihood.
5. Propose MCMC and posterior-predictive diagnostics.

**Solution guidance:** Report the generalized eigenvalue convention and timing. The Kalman filter supplies forecast errors and their covariance matrices; these determine the Gaussian likelihood. Posterior assessment requires more than reporting a point estimate.

## Problem set 8 — Heterogeneous agents and HANK
1. Derive the household Bellman equation and Euler condition.
2. Explain how household policy functions induce a distribution transition.
3. Define stationary recursive competitive equilibrium.
4. Interpret the HJB and Kolmogorov forward equations.
5. Explain why equal aggregate transfers can have different demand effects across household distributions.

**Solution guidance:** Separate individual state variables from aggregate distributions. Market clearing requires aggregation under the invariant distribution. In HANK, MPC heterogeneity explains why transfer incidence affects aggregate consumption.

## Research extension
Select one published quantitative macroeconomic model. Reproduce its steady state and one central figure, document all data and parameter choices, then change one structural mechanism. Submit a short report distinguishing replication results, modifications, and new conclusions.
