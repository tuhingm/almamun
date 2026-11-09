# Lecture 13 — Multivariate Analysis: PCA, Covariance, and Dimension Reduction

## Learning objectives
Understand covariance structure; derive principal components; interpret eigenvalues and loadings; distinguish PCA from factor analysis and causal models.

## 1. Covariance matrix
For centered data matrix \(X\), sample covariance is \(S=X'X/(n-1)\). Eigenvectors identify orthogonal directions of sample variation; eigenvalues measure variance along those directions.

## 2. Principal components
The first principal component solves
\[
\max_{a'a=1}a'Sa.
\]
The solution is the eigenvector associated with the largest eigenvalue. Subsequent components maximize variance subject to orthogonality to previous components.

## 3. Scaling
PCA on covariance matrix is sensitive to units; PCA on the correlation matrix standardizes variables. The choice should reflect whether scale differences are meaningful or merely measurement units.

## 4. Interpretation
Loadings describe variable contributions to components; scores locate observations in component space. PCA is a descriptive linear dimension-reduction method, not a causal model and not equivalent to common-factor analysis.

## 5. Diagnostics
Inspect explained variance, scree plots, stability of loadings, sensitivity to scaling, outliers, and sample composition. Interpret components using substantive context rather than assigning labels solely from large loadings.

## Problem session
1. Derive the first principal component optimization.
2. Explain the eigenvalue decomposition of a covariance matrix.
3. Compare covariance-based and correlation-based PCA.
4. Distinguish PCA and factor analysis.
5. Explain why PCA does not establish causal latent variables.

## Reading
Jolliffe and Cadima (2016), principal component analysis; Anderson, *An Introduction to Multivariate Statistical Analysis*; Johnson and Wichern, *Applied Multivariate Statistical Analysis*.