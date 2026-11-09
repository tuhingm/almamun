# Lecture 01 — Probability Models, Random Variables, and Expectation

**Course:** Statistics: Probability, Mathematical Inference and Experimental Design  
**Level:** Graduate foundation

## Learning objectives
Define probability spaces and random variables; derive expectations and moments; use conditional expectation; distinguish independence from uncorrelatedness; apply total expectation and total variance.

## 1. Probability space
A probability model is \((\Omega,\mathcal F,P)\), where \(\Omega\) is the outcome space, \(\mathcal F\) is a sigma-algebra, and \(P\) is a probability measure. For disjoint events \(A_j\), countable additivity gives \(P(\cup_jA_j)=\sum_jP(A_j)\). The model formalizes uncertainty before observations are collected.

## 2. Random variables and distribution functions
A random variable \(X\) is a measurable mapping from outcomes to the real line. Its CDF is \(F_X(x)=P(X\le x)\). For a discrete variable use a probability mass function; for an absolutely continuous variable use a density \(f_X\) with \(F_X(x)=\int_{-\infty}^xf_X(t)dt\). A density is not a point probability.

## 3. Expectation and moments
\[
E[g(X)]=\sum_xg(x)p_X(x)
\quad\text{or}\quad
E[g(X)]=\int g(x)f_X(x)dx.
\]
Linearity of expectation does not require independence. Variance is \(Var(X)=E[X^2]-E[X]^2\); covariance is \(Cov(X,Y)=E[(X-EX)(Y-EY)]\).

## 4. Conditional expectation
\(E[Y|X]\) is the conditional mean given information in \(X\). It is a random variable, not merely a number. The tower property is \(E[E(Y|X)]=E(Y)\). In regression, the conditional expectation function is the population prediction target.

## 5. Total variance
\[
Var(Y)=E[Var(Y|X)]+Var(E[Y|X]).
\]
The first component is average within-condition variation; the second is variation in conditional means. This identity underlies ANOVA and hierarchical variance decomposition.

## 6. Independence
Independence implies zero covariance when moments exist, but zero covariance does not generally imply independence. Joint Gaussian variables are a special case in which zero covariance implies independence.

## Worked example
Let a two-state economy have \(P(H)=0.4\), \(P(L)=0.6\), conditional means \(E(R|H)=0.12\), \(E(R|L)=0.04\), and conditional variances 0.09 and 0.01. Then \(E(R)=0.072\). Total variance is \(0.4(0.09)+0.6(0.01)+0.4(0.6)(0.08)^2=0.045536\).

## Problem session
1. Derive the mean and variance of a Bernoulli variable.
2. Prove total expectation.
3. Derive total variance by expanding squared deviations.
4. Give dependent variables with zero covariance.
5. Explain why a density value is not a probability.

## Reading
Casella and Berger, *Statistical Inference*; Rice, *Mathematical Statistics and Data Analysis*; Wasserman, *All of Statistics*.