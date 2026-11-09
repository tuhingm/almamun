# Lecture 13 — Multivariate Analysis: PCA, Covariance, and Dimension Reduction

## Learning objectives
Analyze covariance matrices; derive principal components; interpret eigenvalues and loadings; distinguish PCA from factor analysis; understand scaling and validation.

## 1. Covariance geometry
For centered data matrix \(X\in\mathbb R^{n\times p}\), sample covariance is \(S=X'X/(n-1)\). It is symmetric positive semidefinite. Eigenvectors identify orthogonal directions; eigenvalues measure variance along those directions. Variables on different scales can dominate covariance-based analysis.

## 2. Principal components
The first component solves
\[
\max_{a'a=1}a'Sa.
\]
A Lagrange multiplier yields \(Sa=\lambda a\); the maximizing direction is the eigenvector associated with the largest eigenvalue. Subsequent components maximize variance subject to orthogonality. The proportion of variance explained by component \(j\) is \(\lambda_j/\sum_k\lambda_k\).

## 3. Scores and loadings
For eigenvector matrix \(A\), component scores are \(Z=XA\). Loadings describe contributions of original variables, but scaling conventions differ. Eigenvector signs are arbitrary: multiplying an eigenvector by -1 leaves the component direction unchanged. Component labels require substantive interpretation.

## 4. Scaling
PCA on the covariance matrix preserves original units. PCA on the correlation matrix is equivalent to standardizing variables first. Neither is universally preferable; choose based on whether scale differences are substantively meaningful.

## 5. PCA and factor analysis
PCA decomposes total observed variance into orthogonal linear combinations. Common factor analysis models shared covariance through latent factors and variable-specific errors. PCA is a descriptive dimension-reduction transformation, not automatically a measurement or causal model.

## 6. Dimension selection and validation
Use scree plots, parallel analysis, cumulative explained variance, interpretability, and out-of-sample reconstruction or prediction. When PCA is used before prediction, estimate scaling and components within each training fold to prevent information leakage.

## Worked example
For \(S=\begin{pmatrix}4&2\\2&3\end{pmatrix}\), solve \(|S-\lambda I|=0\), then normalize the eigenvectors. The leading eigenvector gives the direction of maximum sample variance.

## Problem set
1. Prove covariance eigenvalues are nonnegative.
2. Derive the first-component solution using a Lagrange multiplier.
3. Calculate explained-variance proportions.
4. Explain standardization before PCA.
5. Distinguish PCA and common factor analysis.

## Computational lab
Compare covariance- and correlation-based PCA. Report eigenvalues, loadings, scores, scree plot, and reconstruction error; evaluate component stability by bootstrap.

## Reading
Jolliffe and Cadima, “Principal Component Analysis: A Review and Recent Developments”; Johnson and Wichern, *Applied Multivariate Statistical Analysis*; Anderson, *An Introduction to Multivariate Statistical Analysis*.