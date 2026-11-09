# Lecture 02 — Probability Distributions and Multivariate Models

## Learning objectives
Derive moments of standard distributions; use transformations and Jacobians; characterize joint and conditional distributions; understand covariance matrices and multivariate normality.

## 1. Discrete distributions
For Bernoulli \(X\sim Bernoulli(p)\), \(E(X)=p\), \(Var(X)=p(1-p)\). If \(X=\sum_{i=1}^nX_i\) for independent Bernoulli indicators, then \(X\sim Binomial(n,p)\), with mean \(np\) and variance \(np(1-p)\). The Poisson distribution with rate \(\lambda\) has mean and variance \(\lambda\).

## 2. Continuous distributions
Uniform, exponential, gamma, beta, normal, lognormal, chi-square, t, and F distributions arise in probability modeling and inference. For each distribution, specify support, density, parameters, and moments. For example, if \(Z\sim N(0,1)\), then \(Z^2\sim\chi_1^2\); sums of independent squared standard normals produce chi-square distributions.

## 3. Joint distributions
A joint density \(f_{X,Y}(x,y)\) integrates to one. Marginal densities are obtained by integration, and conditional densities satisfy \(f_{Y|X}(y|x)=f_{X,Y}(x,y)/f_X(x)\) when the denominator is positive.

## 4. Transformations
For a differentiable one-to-one transformation \(Y=g(X)\), the density includes the absolute Jacobian determinant. In multiple dimensions, if \(Y=g(X)\) is invertible, \(f_Y(y)=f_X(g^{-1}(y))|J_{g^{-1}}(y)|\). The Jacobian is essential; omitting it generally produces a non-normalized density.

## 5. Covariance matrices
For random vector \(X\in\mathbb R^p\), \(\Sigma=E[(X-\mu)(X-\mu)']\). It is positive semidefinite because \(a'\Sigma a=Var(a'X)\ge0\). Its eigenvalues are nonnegative. Correlation matrices rescale covariance but do not eliminate dependence beyond linear association.

## 6. Multivariate normal
If \(X\sim N_p(\mu,\Sigma)\), linear transformations remain normal. Partitioned multivariate normal distributions yield linear conditional expectations and conditional covariance matrices. Under joint normality, zero cross-covariance implies independence.

## Problem session
1. Derive the binomial mean and variance using indicator variables.
2. Find the density of \(Y=\log X\) for positive continuous \(X\).
3. Explain why a covariance matrix is positive semidefinite.
4. Derive a conditional normal distribution from a partitioned covariance matrix.
5. Distinguish zero correlation from independence.

## Reading
Casella and Berger; Johnson, Kotz and Balakrishnan, *Continuous Univariate Distributions*; Anderson, *An Introduction to Multivariate Statistical Analysis*.