# Lecture 02 — Probability Distributions and Multivariate Models

## Core question
How do distributional assumptions determine moments, dependence, transformations, and the sampling models used in statistical inference?

## Learning objectives
Work with standard discrete and continuous distributions; derive moments; transform random variables; obtain joint and conditional distributions; interpret covariance matrices and the multivariate normal model.

## 1. Discrete probability models
For \(X\sim Bernoulli(p)\), \(P(X=1)=p\), \(E(X)=p\), and \(\operatorname{Var}(X)=p(1-p)\). A sum of independent Bernoulli indicators has a binomial distribution:
\[
X=\sum_{i=1}^nX_i\sim Binomial(n,p),\quad E(X)=np,\quad Var(X)=np(1-p).
\]
The Poisson model has mass function \(P(X=x)=e^{-\lambda}\lambda^x/x!\), for non-negative integers, with mean and variance \(\lambda\). It is often used for event counts over a fixed exposure period.

## 2. Continuous distributions
The uniform distribution models constant density over a bounded interval. The exponential distribution has density \(\lambda e^{-\lambda x}\) for \(x\ge0\), mean \(1/\lambda\), and the memoryless property. The gamma family generalizes waiting-time models. The beta distribution is supported on \([0,1]\) and is useful for probabilities. The normal distribution is central to Gaussian models; chi-square, Student-t, and F distributions arise in exact inference under normal sampling. For every distribution, identify support, parameters, density or mass function, and moments.

## 3. Transformations and Jacobians
For a monotone differentiable transformation \(Y=g(X)\),
\[
f_Y(y)=f_X(g^{-1}(y))\left|\frac{d}{dy}g^{-1}(y)\right|.
\]
For a one-to-one vector transformation \(Y=g(X)\), the density is
\[
f_Y(y)=f_X(g^{-1}(y))\left|\det J_{g^{-1}}(y)\right|.
\]
The absolute determinant accounts for local changes in volume. For non-monotone scalar transformations, sum contributions over all inverse branches.

## 4. Joint, marginal, and conditional distributions
A joint density integrates to one over its support. Marginal densities integrate out unwanted coordinates; conditional densities satisfy
\[
f_{X\mid Y}(x\mid y)=\frac{f_{X,Y}(x,y)}{f_Y(y)}
\]
where \(f_Y(y)>0\). Independence is equivalent to factorization of the joint density into marginal densities. Conditional independence is a distinct property and is central to graphical models and hierarchical probability structures.

## 5. Covariance matrices
For a random vector \(X\in\mathbb R^p\), define \(\mu=E(X)\) and
\[
\Sigma=E[(X-\mu)(X-\mu)'].
\]
The matrix is symmetric and positive semidefinite. For any vector \(a\), \(Var(a'X)=a'\Sigma a\ge0\). Off-diagonal elements encode pairwise covariance; the correlation matrix rescales these entries to unit marginal variances.

## 6. Multivariate normal distribution
If \(X\sim N_p(\mu,\Sigma)\), every linear combination \(a'X\) is normally distributed. When \(\Sigma\) is positive definite, the density is
\[
f(x)=(2\pi)^{-p/2}|\Sigma|^{-1/2}
\exp\{-\tfrac12(x-\mu)'\Sigma^{-1}(x-\mu)\}.
\]
For a jointly normal vector, zero covariance implies independence. This equivalence is special to the joint normal family and does not hold generally.

## Worked example
If \(X\sim N(\mu,\sigma^2)\) and \(Z=(X-\mu)/\sigma\), then the transformation formula gives \(Z\sim N(0,1)\). For independent standard normals \(Z_1,\ldots,Z_k\), the sum of squares follows \(\chi_k^2\), which provides the basis for variance inference.

## Problem set
1. Derive the mean and variance of a binomial variable using indicator sums.
2. Transform an exponential variable using \(Y=\log X\) and derive its density.
3. Obtain the marginal density from a specified bivariate joint density.
4. Prove that a covariance matrix is positive semidefinite.
5. Explain why uncorrelated jointly normal variables are independent.

## Computational lab
Generate binomial, Poisson, gamma, and multivariate normal samples. Compare theoretical and empirical moments; visualize bivariate normal samples under different covariance matrices; verify the Jacobian formula numerically for a monotone transformation.

## Reading
Casella and Berger, *Statistical Inference*; Johnson, Kotz, and Balakrishnan, *Continuous Univariate Distributions*; Anderson, *An Introduction to Multivariate Statistical Analysis*.