# Finance — Problem Sets and Selected Solution Guidance

Show derivations, state assumptions, and identify compounding, units, and timing conventions.

## Problem set 1 — Returns and risk
1. A share moves from 100 to 108 and pays a cash distribution of 2. Calculate simple total return and log return.
2. Calculate arithmetic and geometric average returns for a supplied sequence.
3. Explain the SDF pricing relation.

**Guidance:** Include distributions in total return. Compute log return as \(\log(1+R)\). Geometric return is the constant rate that reproduces compounded wealth.

## Problem set 2 — Expected utility
1. Define absolute and relative risk aversion.
2. Derive the certainty equivalent.
3. Derive a second-order approximation to the risk premium.
4. Explain why variance may not capture downside-tail risk.

**Guidance:** Use a Taylor expansion of utility around expected wealth. Concavity produces a positive risk premium for a small mean-preserving risk.

## Problem set 3 — Portfolio theory
1. Derive the global minimum-variance portfolio.
2. Derive the two-asset GMV weight.
3. Construct the efficient frontier from supplied \(\mu\) and \(\Sigma\).
4. Compare unconstrained and long-only allocations.

**Guidance:** Use the Lagrangian with budget constraint. Verify \(\mathbf1'w=1\), and check that the covariance matrix is positive definite for the standard inverse formula.

## Problem set 4 — CAPM and factor models
1. Derive beta and the CAPM expected-return equation.
2. Estimate Jensen's alpha from a supplied regression.
3. Compare CAPM with a multifactor model.
4. Explain generated-regressor and beta-estimation errors.

**Guidance:** Beta is covariance with the market divided by market variance. Interpret alpha relative to the selected benchmark and sample.

## Problem set 5 — No-arbitrage and SDF
1. Derive the SDF price equation.
2. Derive the covariance expression for the risk premium.
3. Explain the law of one price.
4. Compare APT and CAPM.

**Guidance:** Use \(1=E(MR)\) and \(R_f=1/E(M)\) to derive the risk-premium covariance identity.

## Problem set 6 — Fixed income
1. Price a coupon bond using maturity-specific spot rates.
2. Derive a one-period forward rate from two spot rates.
3. Explain why YTM differs from realized return.
4. Distinguish clean and dirty prices.

**Guidance:** Discount each cash flow at its maturity spot rate. Keep coupon timing and compounding basis consistent.

## Problem set 7 — Duration and convexity
1. Calculate Macaulay and modified duration.
2. Derive the first- and second-order price approximations.
3. Construct a key-rate duration interpretation.
4. Explain duration immunization and its limits.

**Guidance:** Use the first derivative of price with respect to yield for duration and the second derivative for convexity. Compare approximations with full repricing.

## Problem set 8 — Derivatives
1. Derive a forward price from replication.
2. Prove put-call parity.
3. Derive risk-neutral probability in a one-step binomial model.
4. Explain delta and gamma hedging.

**Guidance:** Match terminal payoffs across replicating portfolios. Risk-neutral probabilities are pricing weights under an equivalent martingale measure, not physical forecasts.

## Problem set 9 — Capital budgeting
1. Calculate NPV from a project cash-flow stream.
2. Derive incremental free cash flow after taxes.
3. Explain an IRR ranking conflict.
4. State when WACC is an appropriate discount rate.

**Guidance:** Exclude sunk costs, include opportunity costs, and match project risk to the discount rate.

## Problem set 10 — Capital structure
1. State Modigliani-Miller propositions.
2. Derive the simplified value of a perpetual interest tax shield.
3. Compare trade-off and pecking-order theories.
4. Explain payout irrelevance assumptions.

**Guidance:** Clearly state the benchmark assumptions before applying the tax-shield formula.

## Problem set 11 — Market efficiency
1. Define the three forms of market efficiency.
2. Design an event-study timeline.
3. Calculate abnormal and cumulative abnormal returns.
4. Explain adverse selection, moral hazard, and limits to arbitrage.

**Guidance:** Specify estimation window, event window, expected-return model, and inference method. Address clustered events and confounding news.

## Problem set 12 — Banking
1. Analyze a stylized bank balance sheet.
2. Distinguish liquidity and solvency.
3. Calculate expected credit loss from PD, LGD, and EAD.
4. Explain fire-sale externalities.

**Guidance:** Report capital and liquidity positions separately; a solvent bank can still face a liquidity crisis.

## Problem set 13 — Risk management
1. Calculate empirical VaR and expected shortfall.
2. Compare historical, parametric, and Monte Carlo methods.
3. Design a stress test.
4. Explain tail dependence and model risk.

**Guidance:** State the loss sign convention, confidence level, finite-sample quantile method, and treatment of observations beyond VaR.

## Problem set 14 — Behavioral finance
1. Explain reference dependence and loss aversion.
2. Describe probability weighting.
3. Link overconfidence to trading volume.
4. Propose an empirical design for the disposition effect.

**Guidance:** State the behavioral mechanism and a falsifiable prediction. Address selection and alternative rational explanations.

## Problem set 15 — Financial econometrics
1. State GARCH(1,1) positivity and stationarity conditions.
2. Distinguish volatility forecasting from return prediction.
3. Explain look-ahead and survivorship bias.
4. Design an out-of-sample forecasting evaluation.

**Guidance:** For the standard GARCH(1,1), require \(\omega>0\), \(\alpha,\beta\geq0\), and \(\alpha+\beta<1\) for covariance stationarity. Use a genuine time-ordered holdout and include transaction costs when evaluating trading strategies.

## Applied project
Select an asset class, portfolio, financial institution, or corporate investment decision. Define the research question, data universe, sample, return conventions, model, benchmark, and evaluation criterion. Deliver reproducible analysis, diagnostics, robustness checks, and an interpretation of economic significance.
