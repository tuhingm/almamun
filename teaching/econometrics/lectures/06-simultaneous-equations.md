# Lecture 06 — Simultaneous Equations and Structural Identification

## Objectives
Distinguish structural and reduced forms; explain simultaneity bias; state order and rank conditions; derive identification logic for a structural equation.

## 1. Structural system
A linear simultaneous system can be written
\[
By_t+\Gamma x_t=\varepsilon_t,
\]
where endogenous variables appear jointly in \(y_t\), exogenous variables in \(x_t\), and structural shocks in \(\varepsilon_t\). Structural coefficients describe behavioral or technological relations; reduced-form coefficients describe projections of endogenous variables on exogenous variables.

## 2. Reduced form
If \(B\) is nonsingular,
\[
y_t=-B^{-1}\Gamma x_t+B^{-1}\varepsilon_t
=\Pi x_t+v_t.
\]
The reduced form is estimable by OLS under appropriate exogeneity, but structural recovery requires restrictions on \(B\) and \(\Gamma\).

## 3. Simultaneity bias
In a market equilibrium, price and quantity are jointly determined by supply and demand. Equilibrium price is correlated with demand and supply shocks, so regressing quantity on price generally does not recover either structural curve.

## 4. Identification
The order condition is a necessary counting condition: an equation must exclude at least as many exogenous variables as the number of endogenous right-hand-side variables it contains. The rank condition is the exact condition: excluded exogenous variables must generate sufficient independent variation in the endogenous regressors. Order alone is not sufficient.

## 5. Estimation
2SLS estimates a structural equation using excluded exogenous variables as instruments. 3SLS combines equation-level IV estimation with cross-equation covariance restrictions; efficiency gains depend on correct system specification and covariance assumptions.

## Problem set
1. Derive the reduced form from the structural system.
2. Explain why OLS on an equilibrium demand equation is generally inconsistent.
3. State order and rank conditions and distinguish their logical status.
4. Explain when 3SLS may be more efficient than equation-by-equation 2SLS and why misspecification matters.

## Lab
Simulate a two-equation supply-demand system with correlated equilibrium price and structural shocks. Compare OLS, 2SLS, and known-parameter structural coefficients.

## Reading
Kmenta, *Elements of Econometrics*; Greene, *Econometric Analysis*; Hayashi, *Econometrics*.