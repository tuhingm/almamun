# Lecture 14 — Quantitative Macro Policy Analysis, Validation, and Replication

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Design a transparent quantitative macroeconomic application; distinguish calibration from estimation; evaluate mechanisms, welfare, and robustness.

## 1. Research design
Begin with a clearly stated economic question, a model mechanism, observable implications, and a counterfactual. Map each empirical object to a model variable. State the unit, frequency, transformation, sample, and source of each series.

## 2. Calibration and estimation
Calibration selects parameters from external evidence or long-run targets. Estimation uses a statistical criterion such as likelihood or moments. A hybrid strategy must identify which parameters are calibrated and which are estimated, and explain why.

## 3. Model validation
Evaluate steady-state restrictions, equilibrium residuals, impulse-response signs and timing, unconditional moments, forecast errors, and posterior predictive distributions. Validation is not merely reproducing the calibration targets.

## 4. Counterfactual policy
A policy counterfactual changes a rule, shock process, or structural parameter and resolves the model. Keep the comparison coherent: same initial conditions, shock normalization, and welfare metric. Separate transition effects from long-run effects.

## 5. Welfare
For a representative household, compare expected lifetime utility across allocations. A consumption-equivalent variation \(\lambda\) solves
\[
E_0\sum_t\beta^tu((1+\lambda)c_t^{base})=
E_0\sum_t\beta^tu(c_t^{policy}),
\]
with timing and risk conventions stated. In heterogeneous-agent settings, report distributional welfare and the social aggregation rule rather than only an average.

## 6. Reproducibility protocol
Archive source data or access instructions, transformations, parameter files, software versions, random seeds, estimation settings, and scripts. A replication package should regenerate tables and figures from raw or documented input data.

## 7. Computational lab
Replicate a published RBC or NK result, then change one structural assumption. Produce a model description, parameter table, steady-state diagnostics, IRFs, robustness table, and concise policy interpretation.

## Problem set
1. Distinguish calibration and estimation.
2. Design three validation tests for a DSGE model.
3. Define a consumption-equivalent welfare comparison.
4. Specify a reproducibility checklist for a quantitative macro paper.

## Further reading
Smets and Wouters (2007); Fernández-Villaverde (2010); Del Negro and Schorfheide, *DSGE Models in Macroeconomics*.
