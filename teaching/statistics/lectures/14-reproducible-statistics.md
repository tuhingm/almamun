# Lecture 14 — Statistical Modeling Workflow, Simulation, and Reproducibility

## Learning objectives
Specify estimands and statistical models; design simulation studies; evaluate estimator performance; document computational decisions; report uncertainty and create reproducible archives.

## 1. Analysis plan
State the observational unit, target population, outcome, predictors, sampling process, estimand, model, identifying assumptions, missing-data strategy, and inferential target. Distinguish descriptive, predictive, and causal objectives. Select models for the target question rather than the largest in-sample fit statistic.

## 2. Data provenance
Preserve immutable raw data; document transformations, units, coding, exclusions, merges, duplicates, and missingness. Separate data cleaning from analysis. Record software versions, dependencies, random seeds, and computational environment.

## 3. Simulation design
Specify a data-generating process, parameter values, sample sizes, dependence, estimator, and performance measures. Across \(R\) replications,
\[
Bias=R^{-1}\sum_r(\widehat\theta_r-\theta_0),\quad
RMSE=\sqrt{R^{-1}\sum_r(\widehat\theta_r-\theta_0)^2}.
\]
Also evaluate empirical standard deviation, average estimated standard error, interval coverage, rejection rates, convergence, and failure rates.

## 4. Monte Carlo uncertainty
For estimated rejection frequency \(\widehat p\),
\[
MCSE(\widehat p)=\sqrt{\widehat p(1-\widehat p)/R}.
\]
Simulation estimates are themselves random. Choose \(R\) so Monte Carlo error is small relative to the differences under study.

## 5. Reproducible workflow
Use scripts or notebooks that execute from raw inputs to final tables and figures. Use version control, relative paths, structured project folders, automated checks, and execution logs. A random seed does not replace documentation of the DGP, software, and transformations.

## 6. Reporting and robustness
Report estimand, estimator, sample definition, assumptions, standard errors, confidence intervals, and diagnostics. Separate pre-specified from exploratory analysis. Robustness checks should vary assumptions or design choices that materially affect the target estimate, not search mechanically for significance.

## Worked example
Compare OLS and a robust estimator under heteroskedastic errors. Specify coefficients, sample sizes, and the error variance function; repeat estimation; calculate bias, RMSE, coverage, and rejection rates. A robust covariance estimator may change inference without changing the OLS point estimate.

## Problem set
1. Define a DGP for a heteroskedastic linear model.
2. Derive the Monte Carlo standard error of a rejection rate.
3. Distinguish estimator bias from Monte Carlo error.
4. Design a simulation comparing estimators under dependence.
5. List files and metadata required for independent replication.

## Computational lab
Build a configurable simulation script. Save replication-level estimates, summarize bias/RMSE/coverage/size, and create an archive with README and dependency record.

## Reading
Gentle, *Elements of Computational Statistics*; Boos and Stefanski, *Essential Statistical Inference*; Peng, “Reproducible Research in Computational Science”; National Academies, *Reproducibility and Replicability in Science*.