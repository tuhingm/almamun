# Lecture 08 — Forwards, Futures, Options, and Swaps

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Value derivative payoffs; derive put-call parity and binomial pricing; distinguish physical and risk-neutral probabilities; interpret Greeks and hedging.

## 1. Forward and futures contracts
A forward commits counterparties to trade an asset at maturity for delivery price \(K\). For a non-income asset under simple assumptions, the no-arbitrage forward price is \(F_{0,T}=S_0(1+r)^T\). Income, storage costs, convenience yield, and funding conventions modify this relation. Futures are marked to market daily.

## 2. Options and parity
A European call payoff is \((S_T-K)^+\); a put payoff is \((K-S_T)^+\). For a non-dividend-paying asset,
\[
C_0-P_0=S_0-Ke^{-rT}.
\]
The parity follows by comparing portfolios with identical terminal payoffs.

## 3. Binomial risk-neutral valuation
In a one-step binomial model, stock moves to \(uS_0\) or \(dS_0\). The risk-neutral probability is
\[
q=\frac{(1+r)-d}{u-d},
\]
for a one-period gross risk-free return \(1+r\), with no-arbitrage requiring \(d<1+r<u\). The derivative value is the discounted risk-neutral expected payoff.

## 4. Black-Scholes framework
For a European option under geometric Brownian motion, constant volatility, frictionless trading, and continuous hedging, the Black-Scholes formula gives a benchmark price. Its assumptions and market conventions must be stated.

## 5. Greeks and hedging
Delta measures first-order sensitivity to the underlying; gamma measures curvature; vega measures volatility sensitivity; theta measures time decay; rho measures interest-rate sensitivity. Delta hedging leaves residual gamma, volatility, jump, and basis risks.

## 6. Worked example
Use put-call parity to solve for a put given call price, spot price, strike, rate, and maturity. Verify by constructing the two equivalent portfolios.

## 7. Computational lab
Build a binomial option pricer and compare convergence to Black-Scholes. Calculate Greeks by analytic formulas and finite differences; test hedge P&L under discrete rebalancing.

## Problem set
1. Derive forward price by replication.
2. Prove put-call parity.
3. Derive risk-neutral probability.
4. Explain why risk-neutral probabilities are not forecasts of physical frequencies.

## Further reading
Hull, *Options, Futures, and Other Derivatives*; Shreve, *Stochastic Calculus for Finance I*.
