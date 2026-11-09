# Lecture 14 — Research Design, Monte Carlo, Replication, and Reporting

## Objectives
Translate a research question into an estimand; plan identification and estimation; design Monte Carlo experiments; build reproducible workflows; communicate uncertainty and limitations precisely.

## 1. From question to estimand
State the population, unit, outcome, treatment/exposure, horizon, and target parameter before choosing an estimator. A model should be selected because its assumptions and estimand match the question, not because it is familiar.

## 2. Identification before estimation
Write the structural or causal assumptions explicitly. Distinguish assumptions testable from observed data from restrictions requiring institutional or theoretical justification. Explain what variation identifies the parameter and which threats remain.

## 3. Monte Carlo design
A simulation study specifies a data-generating process (DGP), parameter values, sample dimensions, dependence structure, estimators, and performance metrics. For estimator \(\widehat\theta_r\) across \(R\) replications:
\[
Bias=R^{-1}\sum_r(\widehat\theta_r-\theta_0),\quad
RMSE=\sqrt{R^{-1}\sum_r(\widehat\theta_r-\theta_0)^2}.
\]
Also assess empirical standard-error accuracy, confidence-interval coverage, rejection rates, and convergence failures. State random seeds and Monte Carlo standard errors.

## 4. Replication workflow
Preserve raw data; document transformations; separate data construction, estimation, tables, and figures; record software versions and seeds; write a README; and ensure a clean run from start to finish. Never silently overwrite source data.

## 5. Reporting standards
Report estimand, sample, estimator, identification assumptions, covariance estimator, diagnostics, uncertainty, effect magnitude, and robustness. Distinguish confirmatory from exploratory analyses. Make code and data availability statements accurate.

## Capstone
Prepare a reproducible empirical paper with:
1. research question and estimand;
2. conceptual framework and literature;
3. data dictionary and sample construction;
4. identification argument;
5. estimator and inference;
6. diagnostics and robustness;
7. interpretation and limitations;
8. replication archive.

## Problem set
1. Distinguish an estimand, estimator, and estimate.
2. Design a Monte Carlo experiment comparing OLS and IV under varying instrument strength.
3. Define bias, variance, RMSE, and coverage.
4. Identify minimum elements of a reproducibility archive.
5. Write a one-paragraph identification argument for a proposed empirical question.

## Reading
Gentle, *Elements of Computational Statistics*; Andrews and Gentzkow research design discussions; National Academies, *Reproducibility and Replicability in Science*; relevant estimator-specific methodological papers.