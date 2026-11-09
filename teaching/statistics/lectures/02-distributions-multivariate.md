# Lecture 02 — Probability Distributions and Multivariate Models

## Learning objectives
Derive moments of standard distributions; transform random variables using Jacobians; obtain joint, marginal, and conditional distributions; interpret covariance matrices and multivariate normality.

## 1. Discrete distributions
For \(X\sim Bernoulli(p)\), \(E(X)=p\) and \(Var(X)=p(1-p)\). A sum of independent Bernoulli indicators is \(Binomial(n,p)\), with mean \(np\) and variance \(np(1-p)\). The Poisson mass function is \(P(X=x)=e^{-\lambda}\lambda^x/x!\), with mean and variance \(\lambda\). State the support and parameterization whenever using a distribution.

## 2. Continuous distributions
The uniform, exponential, gamma, beta, normal, lognormal, chi-square, Student-t, and F distributions appear in probability modeling and inference. For example, the exponential density is \(f(x)=\lambda e^{-\lambda x}\), \(x\ge0\), with mean \(1/\lambda\) and variance \(1/\lambda^2\). The normal family is closed under linear transformations; chi-square, t, and F distributions arise from independent normal and chi-square quantities.

## 3. Transformations
For a monotone differentiable scalar transformation \(Y=g(X)\),
\[
f_Y(y)=f_X(g^{-1}(y))\left|\frac{d}{dy}g^{-1}(y)\right|.
\]
For a one-to-one vector transformation, multiply the original density by the absolute determinant of the inverse Jacobian. For non-monotone scalar transformations, sum over all inverse branches. The support must be transformed as well as the density.

## 4. Joint and conditional distributions
Marginal densities integrate out other coordinates. The conditional density is \(f_{X|Y}(x|y)=f_{X,Y}(x,y)/f_Y(y)\) where the denominator is positive. Independence is equivalent to factorization of the joint distribution into its marginals. Conditional independence is a separate property.

## 5. Covariance matrices and multivariate normality
For random vector \(X\), \(\Sigma=E[(X-\mu)(X-\mu)']\). It is symmetric positive semidefinite because \(a'\Sigma a=Var(a'X)\ge0\). If \(X\sim N_p(\mu,\Sigma)\), every linear combination is normal. Within the jointly normal family, zero covariance implies independence; this implication is not generally valid outside that family.

## Worked example
If \(X\sim N(\mu,\sigma^2)\), then \(Z=(X-\mu)/\sigma\sim N(0,1)\). For independent standard normals, the sum of \(k\) squared values follows \(\chi_k^2\), which underlies exact variance inference.

## Problem set
1. Derive the binomial mean and variance using indicator variables.
2. Transform an exponential variable using \(Y=\log X\).
3. Derive a marginal density from a bivariate joint density.
4. Prove covariance matrices are positive semidefinite.
5. Explain why zero covariance implies independence for jointly normal variables but not generally.

## Computational lab
Simulate binomial, Poisson, gamma, and multivariate normal data. Compare empirical and theoretical moments, and visualize how covariance changes the geometry of bivariate normal samples.

## Reading
Casella and Berger, *Statistical Inference*; DeGroot and Schervish, *Probability and Statistics*; Anderson, *An Introduction to Multivariate Statistical Analysis*.