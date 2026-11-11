# Monetary Economics

<p class="lead">This course studies how money, interest rates and central-bank policy affect inflation and economic activity. Most of the course builds and solves New Keynesian DSGE models in Dynare — from Calvo pricing and optimal policy to open economies, financial frictions, the zero lower bound, monetary–fiscal interactions, HANK and Bayesian estimation. The remaining lectures confront the models with evidence: the long-run facts about money and inflation, the identification of monetary policy shocks, and how central banks operate in practice. All computational work uses MATLAB and Dynare.</p>

## What you will learn

1. Derive the New Keynesian Phillips curve, the dynamic IS equation and welfare-based loss functions from first principles.
2. Solve and simulate monetary DSGE models in Dynare and check the results against analytical solutions.
3. Analyze determinacy, the Taylor principle and the fiscal theory of the price level.
4. Compare discretion, commitment and simple policy rules, including inflation, price-level and average-inflation targeting.
5. Explain how open-economy features, financial frictions, the zero lower bound and household heterogeneity change monetary transmission.
6. Estimate a small DSGE model by Bayesian methods and identify monetary shocks with VARs, local projections and narrative or high-frequency series.

## How the course is organized

Each lecture has full written notes with learning objectives, derivations, a problem set, two MATLAB/Dynare coding exercises built on real data, and readings. About two thirds of the lectures are DSGE-based; Lectures 1, 14 and 15 are empirical and institutional. Problem sets with solution guidance and examination specifications are under Assignments and Exams.

<!-- units -->

## Data and code

The coding exercises run in MATLAB (R2021a or later) with Dynare 6. Save these files in your working folder.

**Data (all real series)**

- [us_quarterly.csv](data/us_quarterly.csv) — U.S. quarterly macro data from FRED-QD, 1959–2025: GDP and components, prices, interest rates, money, spreads, debt.
- [us_monthly.csv](data/us_monthly.csv) — U.S. monthly data from FRED-MD, 1959–2025: federal funds rate, Treasury yields, CPI, industrial production, unemployment, M2, S&P 500, Baa yield.
- [mp_shocks_monthly.csv](data/mp_shocks_monthly.csv) — published monetary-policy shock series: Romer–Romer, Bu–Rogers–Wu, Nakamura–Steinsson and Jarociński–Karadi.
- [cpi_inflation_wdi.csv](data/cpi_inflation_wdi.csv) — World Bank CPI inflation for 190+ countries, with inflation-targeting adoption years.
- [krw_usd_monthly.csv](data/krw_usd_monthly.csv) — the KRW/USD exchange rate, monthly.

**Helper functions (no toolboxes required)**

- [solve_lre.m](code/solve_lre.m) — solves linear rational-expectations models with the QZ decomposition and reports determinacy.
- [kalman_loglik.m](code/kalman_loglik.m) — Kalman-filter log-likelihood of a linear state-space model.
- [ols_nw.m](code/ols_nw.m) — OLS with Newey–West standard errors.
- [qreg.m](code/qreg.m) — linear quantile regression.
- [hp_trend.m](code/hp_trend.m) — Hodrick–Prescott trend.

## Assessment

Problem sets 20%; Dynare model labs 20%; midterm 20%; replication project 20%; final examination 20%.

## Core references

- Galí, *Monetary Policy, Inflation, and the Business Cycle*, 2nd ed.
- Walsh, *Monetary Theory and Policy*, 4th ed.
- Woodford, *Interest and Prices*.
- Cochrane, *The Fiscal Theory of the Price Level*.
- Herbst and Schorfheide, *Bayesian Estimation of DSGE Models*.
- Ramey, "Macroeconomic Shocks and Their Propagation," *Handbook of Macroeconomics* (2016).
