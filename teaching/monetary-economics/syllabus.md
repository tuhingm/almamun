# Monetary Economics — Graduate Syllabus

## Course description
This course studies the theory and evidence of monetary policy. It begins with money demand, money in general equilibrium and the classical monetary model, then develops the New Keynesian framework in depth: price and wage setting, determinacy, optimal policy and policy rules. Extensions cover small open economies, financial frictions, the zero lower bound and unconventional policy, monetary–fiscal interactions, and heterogeneous-agent (HANK) transmission. The course closes with Bayesian estimation of DSGE models, the identification of monetary policy shocks, and central banking in practice. Models are solved and estimated in Dynare; data work uses MATLAB.

## Prerequisites
Graduate macroeconomics (or Advanced Macroeconomics, Course 04); calculus and linear algebra; probability and introductory econometrics. Basic MATLAB programming is assumed; Dynare is introduced in Lecture 3.

## Software
MATLAB (R2021a or later) and Dynare 6 only. No toolboxes are required: the course helper functions provide QZ solution, Kalman filtering, Newey–West standard errors and quantile regression. Every exercise runs from the course data files.

## Learning outcomes
1. Derive the New Keynesian Phillips curve, the dynamic IS equation and welfare-based loss functions from first principles.
2. Solve and simulate monetary DSGE models in Dynare and verify them against analytical solutions.
3. Analyze determinacy, the Taylor principle and the fiscal theory of the price level.
4. Compare discretion, commitment and simple rules, including inflation, price-level and average-inflation targeting.
5. Explain how open-economy features, financial frictions, the zero lower bound and heterogeneity change transmission.
6. Estimate a small DSGE model by Bayesian methods and identify monetary shocks with VARs, local projections and external instruments.

## 15-week schedule
| Week | Topic | Required preparation / output |
|---|---|---|
| 1 | Money, inflation and transmission: the facts | Quantity theory and money demand on U.S. data |
| 2 | Money in general equilibrium | Money-in-utility, cash-in-advance and the Friedman rule |
| 3 | The classical monetary model | First Dynare model; neutrality and price-level determinacy |
| 4 | Price and wage setting | Calvo pricing and the New Keynesian Phillips curve |
| 5 | The basic New Keynesian model | Impulse responses and the determinacy region |
| 6 | Optimal monetary policy | Discretion versus commitment; welfare loss |
| 7 | Policy rules and frameworks | Taylor rules; inflation versus price-level targeting |
| 8 | Small open economies | Galí–Monacelli model; exchange-rate regimes |
| 9 | Financial frictions and the credit channel | Financial accelerator; credit spreads |
| 10 | The zero lower bound and unconventional policy | Liquidity trap; forward guidance puzzle |
| 11 | Monetary–fiscal interactions | Active/passive regimes; fiscal theory of the price level |
| 12 | HANK and monetary transmission | TANK multiplier; marginal propensities to consume |
| 13 | Bayesian estimation of monetary DSGE models | Posterior mode and Metropolis–Hastings |
| 14 | Identifying monetary policy shocks | Recursive VAR; local projections with narrative shocks |
| 15 | Central banking in practice | Inflation targeting evidence; inflation-at-risk |

## Teaching and learning
Each week combines a lecture, a derivation session and a computational lab. Labs build each model in a Dynare `.mod` file and check it against the analytical solution derived in the lecture. Students must report the Dynare version, parameter values, steady state and determinacy diagnostics for every model they submit.

## Assessment
- Problem sets: 20%
- Dynare model labs: 20%
- Midterm examination (Weeks 1–7): 20%
- Replication project: 20%
- Final examination: 20%

## Assessment standards
Derivations must state timing, log-linearization points and sign conventions. Dynare submissions must run from a clean folder and report determinacy checks and impulse responses with units. Empirical work must state the identifying assumption behind every estimated policy effect and distinguish it from model-based inference.

## Core texts
- Galí, *Monetary Policy, Inflation, and the Business Cycle*, 2nd ed.
- Walsh, *Monetary Theory and Policy*, 4th ed.
- Woodford, *Interest and Prices*.
- Cochrane, *The Fiscal Theory of the Price Level*.
- Herbst and Schorfheide, *Bayesian Estimation of DSGE Models*.
- Ramey, "Macroeconomic Shocks and Their Propagation," *Handbook of Macroeconomics* (2016).
