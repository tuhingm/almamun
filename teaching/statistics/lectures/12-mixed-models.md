# Lecture 12 — Linear Mixed Models and Hierarchical Data

## Learning objectives
Specify random intercept and slope models; distinguish within- and between-group variation; interpret variance components; understand partial pooling and inference.

## 1. Hierarchical structure
Observations may be nested within schools, hospitals, firms, regions, or repeated subjects. Independence at the row level is generally implausible when observations share group-level influences.

## 2. Random intercept model
\[
y_{ij}=x_{ij}'\beta+b_j+\varepsilon_{ij},
\quad b_j\sim N(0,\tau^2),\quad\varepsilon_{ij}\sim N(0,\sigma^2).
\]
The intraclass correlation is \(\rho=\tau^2/(\tau^2+\sigma^2)\) in the simple random-intercept model.

## 3. Random slopes
Allow a coefficient to vary across groups:
\[
y_{ij}=\beta_0+b_{0j}+ (\beta_1+b_{1j})x_{ij}+\varepsilon_{ij}.
\]
The random-effects covariance matrix determines intercept-slope variation and correlation. Interpretation requires enough independent groups and a defensible covariance structure.

## 4. Partial pooling
Complete pooling assumes common group parameters; no pooling estimates every group separately; partial pooling combines group-specific information through a shared distribution. Shrinkage is stronger for groups with less information and depends on the estimated variance components.

## 5. Estimation and diagnostics
Likelihood or restricted maximum likelihood can estimate variance components. Assess convergence, singular fits, residual patterns, influential groups, and sensitivity to random-effects structure. Cluster-robust inference and mixed models answer different modeling questions.

## Problem session
1. Derive the ICC for a random-intercept model.
2. Interpret partial pooling.
3. Distinguish fixed group effects from random intercepts.
4. Explain a singular fit.
5. Identify the number of independent groups relevant for generalization.

## Reading
Pinheiro and Bates, *Mixed-Effects Models in S and S-PLUS*; Gelman and Hill, *Data Analysis Using Regression and Multilevel/Hierarchical Models*; McCulloch, Searle and Neuhaus, *Generalized, Linear, and Mixed Models*.