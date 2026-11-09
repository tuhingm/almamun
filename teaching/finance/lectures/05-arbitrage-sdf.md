# Lecture 05 — Arbitrage Pricing, Stochastic Discount Factors, and No-Arbitrage

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Use no-arbitrage restrictions to price payoffs; derive SDF relations; understand factor pricing and risk-premium covariance.

## 1. Law of one price
Two portfolios with identical state-contingent payoffs must have the same price. If not, a long-short strategy can generate an arbitrage under the idealized frictionless assumptions.

## 2. SDF pricing
For payoff \(X_{t+1}\),
\[
P_t=E_t[M_{t+1}X_{t+1}].
\]
For a gross return \(R_{t+1}\), \(1=E_t[M_{t+1}R_{t+1}]\). Decomposing the expectation gives
\[
E_t(R_{t+1})-R_{f,t}
=-\frac{\operatorname{Cov}_t(M_{t+1},R_{t+1})}{E_t(M_{t+1})},
\]
when the risk-free gross return is \(R_{f,t}=1/E_tM_{t+1}\).

## 3. Linear factor pricing
If the SDF is affine in factors, expected excess returns are related to factor covariances. APT derives approximate linear pricing restrictions under diversified idiosyncratic risk and no-arbitrage; it does not by itself identify the economically correct factors.

## 4. Traded and non-traded factors
A traded factor has a return that can be held as a portfolio. A non-traded factor may represent macroeconomic risk and requires a factor-mimicking portfolio or additional restrictions for empirical pricing.

## 5. Worked example
Suppose an SDF is high in bad states. An asset paying more in those states covaries positively with the SDF and can command a lower expected return. Explain the insurance interpretation.

## 6. Computational lab
Estimate candidate factor models and compare pricing errors, cross-sectional fit, and stability. Construct factor-mimicking portfolios and document the normalization.

## Problem set
1. Derive the SDF pricing equation.
2. Derive the covariance expression for the risk premium.
3. State the law of one price.
4. Distinguish APT from CAPM.

## Further reading
Ross (1976); Cochrane, *Asset Pricing*.
