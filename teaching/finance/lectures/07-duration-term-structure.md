# Lecture 07 — Duration, Convexity, Immunization, and Term Structure

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Measure bond price sensitivity; derive duration and convexity approximations; understand immunization and term-structure models.

## 1. Duration
Macaulay duration is the present-value-weighted average time to cash flow:
\[
D_M=\frac{\sum_t t\,PV(CF_t)}{P}.
\]
For annual compounding, modified duration is \(D_{mod}=D_M/(1+y)\). For a small yield change,
\[
\frac{\Delta P}{P}\approx-D_{mod}\Delta y.
\]

## 2. Convexity
Convexity captures curvature:
\[
C=\frac{1}{P}\frac{\partial^2P}{\partial y^2}.
\]
The second-order approximation is
\[
\frac{\Delta P}{P}\approx-D_{mod}\Delta y+\frac12C(\Delta y)^2.
\]
Duration-convexity approximations are local and depend on the assumed yield-curve shift.

## 3. Key-rate duration
A parallel shift is restrictive. Key-rate durations measure sensitivity to changes at selected curve maturities, helping distinguish level, slope, and curvature exposures.

## 4. Immunization
A liability-driven portfolio can be immunized by matching present value and duration, with convexity and rebalancing conditions considered. Immunization is not a one-time guarantee under arbitrary nonparallel curve movements.

## 5. Term-structure theories
The expectations hypothesis links long yields to expected future short rates plus possible term premia. Affine term-structure models represent yields as functions of latent state variables and risk adjustments.

## 6. Worked example
For modified duration 6 and convexity 45, approximate the price change for a 50-basis-point yield increase. Compare duration-only and duration-convexity estimates.

## 7. Computational lab
Calculate duration, convexity, and key-rate exposures for a bond portfolio. Apply parallel and nonparallel curve shocks and compare linear and second-order estimates with full repricing.

## Problem set
1. Derive duration from the bond pricing function.
2. Explain convexity correction.
3. Compare parallel and key-rate risk.
4. State conditions for duration immunization.

## Further reading
Tuckman and Serrat, *Fixed Income Securities*; Diebold and Li (2006).
