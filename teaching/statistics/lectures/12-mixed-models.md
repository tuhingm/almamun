# Lecture 12 — Linear Mixed Models and Hierarchical Data

## Learning objectives
Specify random-intercept and random-slope models; distinguish within- and between-group effects; derive covariance structures and ICC; understand partial pooling and likelihood-based estimation.

## 1. Hierarchical data
Observations may be nested within schools, hospitals, firms, regions, or repeated subjects. Shared group-level influences induce dependence. Row-level independence is generally implausible when observations share group-level determinants.

## 2. Random-intercept model
For unit \(i\) in group \(j\),
\[
y_{ij}=x_{ij}'\beta+b_j+\varepsilon_{ij},\quad
b_j\sim N(0,\tau^2),\quad\varepsilon_{ij}\sim N(0,\sigma^2).
\]
Assuming independence of random effects and residuals, marginal variance is \(\tau^2+\sigma^2\), and
\[
ICC=\frac{\tau^2}{\tau^2+\sigma^2}.
\]
Two observations in the same group have covariance \(\tau^2\) under this model.

## 3. Random slopes
Allow coefficients to vary:
\[
y_{ij}=\beta_0+b_{0j}+(\beta_1+b_{1j})x_{ij}+\varepsilon_{ij}.
\]
The random-effect vector has covariance matrix \(G\), which can allow intercept-slope correlation. The implied covariance depends on covariate values.

## 4. Within- and between-group effects
Decompose \(x_{ij}=(x_{ij}-\bar x_j)+\bar x_j\). Including both components separates within-group and between-group associations. A pooled coefficient can conflate the two.

## 5. Estimation and partial pooling
Mixed models are commonly estimated by maximum likelihood or REML. Empirical Bayes group predictions shrink estimates toward the population mean, with greater shrinkage for groups with less information. This is partial pooling, distinct from complete pooling and separate estimation.

## 6. Diagnostics
Inspect residuals, random-effect distributions, variance estimates, convergence, singular fits, and covariance-structure sensitivity. Variance-component tests can involve boundary conditions, so ordinary chi-square reference distributions may not apply directly. Report the number of groups as well as observations.

## Worked example
If between-school variance is 16 and within-school variance is 64, then \(ICC=16/(16+64)=0.20\). Under the random-intercept model, two students from the same school have correlation 0.20 conditional on fixed effects.

## Problem set
1. Derive marginal variance and within-group covariance.
2. Interpret the ICC.
3. Distinguish random intercepts from fixed group indicators.
4. Explain partial pooling.
5. Explain boundary issues in variance-component testing.

## Computational lab
Fit pooled OLS, group fixed-effects, and random-intercept models to nested data. Compare estimates, variance components, ICC, and group predictions; then fit a random-slope model.

## Reading
Pinheiro and Bates, *Mixed-Effects Models in S and S-PLUS*; Gelman and Hill, *Data Analysis Using Regression and Multilevel/Hierarchical Models*; McCulloch et al., *Generalized, Linear, and Mixed Models*.