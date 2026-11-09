# Lecture 13 — Causal Inference: Potential Outcomes, DiD, and Regression Discontinuity

## Objectives
Define causal estimands; state identification assumptions; derive difference-in-differences; understand event studies, RD, and synthetic control.

## 1. Potential outcomes
For treatment \(D_i\in\{0,1\}\), define potential outcomes \(Y_i(1)\) and \(Y_i(0)\). The individual effect is \(Y_i(1)-Y_i(0)\), but only one potential outcome is observed. The average treatment effect is
\[
ATE=E[Y_i(1)-Y_i(0)].
\]
Identification requires assumptions that connect observed outcomes to missing counterfactuals.

## 2. Selection and ignorability
Under conditional independence \((Y(1),Y(0))\perp D\mid X\) and overlap, treatment effects can be identified using adjustment, weighting, or matching. These assumptions are substantive and cannot be verified solely from observed balance.

## 3. Difference-in-differences
For treated and control groups observed before and after treatment,
\[
DiD=(\bar Y_{T,post}-\bar Y_{T,pre})
-(\bar Y_{C,post}-\bar Y_{C,pre}).
\]
The central identifying restriction is parallel trends in untreated potential outcomes. With multiple periods, assess pre-treatment dynamics and use estimators appropriate to treatment timing and effect heterogeneity.

## 4. Event studies
Event-time coefficients describe outcome dynamics relative to treatment timing. Pre-treatment coefficients can be informative diagnostics but failure to reject pre-trends is not proof of parallel trends. Inference must reflect the assignment and dependence structure.

## 5. Regression discontinuity
When treatment changes at cutoff \(c\), compare units just above and below the threshold. A sharp RD identifies a local treatment effect if potential outcomes and covariates evolve smoothly through the cutoff and agents cannot precisely manipulate assignment. Fuzzy RD uses discontinuity in treatment probability as an instrument.

## 6. Synthetic control
Construct a weighted combination of untreated units to approximate the treated unit's pre-treatment trajectory. Credibility depends on pre-treatment fit, donor pool, post-treatment contamination, and inference appropriate to the number of treated units.

## Problem set
1. Define ATE and explain the fundamental problem of causal inference.
2. Derive the two-by-two DiD estimator.
3. State parallel trends and explain why it concerns untreated potential outcomes.
4. State continuity and no-manipulation conditions in RD.
5. Explain the donor-pool and pre-treatment-fit requirements of synthetic control.

## Lab
Estimate a two-way DiD and event study; graph pre- and post-treatment coefficients; conduct placebo and sensitivity analyses; report clustering consistent with treatment assignment.

## Reading
Imbens and Rubin, *Causal Inference*; Angrist and Pischke, *Mostly Harmless Econometrics*; Cunningham, *Causal Inference: The Mixtape*; Callaway and Sant’Anna (2021).