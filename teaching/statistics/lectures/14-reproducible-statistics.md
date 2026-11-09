# Lecture 14 — Statistical Modeling Workflow, Simulation, and Reproducibility

## Learning objectives
Specify a statistical model; evaluate estimator performance; design simulation studies; report uncertainty; create reproducible analysis archives.

## 1. Model specification
State observational unit, outcome, predictors, sampling process, distributional assumptions, estimand, and inferential target. Distinguish descriptive, predictive, and causal objectives.

## 2. Simulation study
Specify a data-generating process, parameter values, sample sizes, dependence, estimator, and performance criteria. Across \(R\) replications:
\[
Bias=R^{-1}\sum_r(\widehat\theta_r-\theta_0),\quad
RMSE=\sqrt{R^{-1}\sum_r(\widehat\theta_r-\theta_0)^2}.
\]
Also evaluate coverage, rejection rates, convergence, and Monte Carlo uncertainty.

## 3. Reproducibility
Preserve raw data; document transformations; set random seeds; record software and package versions; separate code, data, output, and figures; provide a README and data dictionary. A reproducible project should run from a clean environment.

## 4. Reporting
Report estimates, uncertainty intervals, assumptions, diagnostics, effect sizes, missing-data handling, multiplicity, and limitations. Do not present exploratory findings as pre-specified confirmatory tests.

## Capstone
Prepare a complete statistical analysis plan and a reproducible report containing model rationale, data dictionary, analysis code, diagnostics, sensitivity analysis, and interpretation.

## Problem session
1. Distinguish estimand, estimator, and estimate.
2. Design a simulation comparing unbiased and biased estimators.
3. Calculate bias and RMSE from simulated estimates.
4. Explain Monte Carlo standard error.
5. Draft a reproducibility checklist.

## Reading
Gentle, *Elements of Computational Statistics*; Wickham and Grolemund, *R for Data Science*; National Academies, *Reproducibility and Replicability in Science*.