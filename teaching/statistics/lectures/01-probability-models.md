# Lecture 01 — Probability Models, Random Variables, and Expectation

**Course:** Statistics: Probability, Mathematical Inference and Experimental Design  
**Level:** Graduate foundation

## Core question
How does a probability model translate uncertainty about outcomes into quantities that can be estimated and tested?

## Learning objectives
Define probability spaces and random variables; derive expectations and moments; work with conditional distributions; distinguish independence from zero correlation; apply iterated expectations and the law of total variance.

## 1. Probability spaces and events
A probability space is ((\Omega,\mathcal F,P)). The sample space \(\Omega\) contains possible outcomes, \(\mathcal F\) is a sigma-algebra of measurable events, and \(P\) is a probability measure satisfying non-negativity, normalization, and countable additivity. For pairwise disjoint events \(A_j\),
\[
P\left(\bigcup_{j=1}^{\infty}A_j\right)=\sum_{j=1}^{\infty}P(A_j).
\]
The axioms imply monotonicity, complements, and the addition rule. Conditional probability is defined by \(P(A\mid B)=P(A\cap B)/P(B)\) when \(P(B)>0\). Bayes' rule follows by expressing the joint probability in two ways.

## 2. Random variables and distributions
A random variable is a measurable function \(X:\Omega\to\mathbb R\). Its cumulative distribution function is \(F_X(x)=P(X\le x)\). For a discrete variable, probabilities are represented by a probability mass function; for an absolutely continuous variable, by a density \(f_X\), with
\[
F_X(x)=\int_{-\infty}^{x}f_X(t)dt,\qquad P(a<X\le b)=\int_a^b f_X(t)dt.
\]
A density is not itself a probability at a point. The support and normalization condition must always be stated.

## 3. Expectation and moments
For discrete \(X\), \(E[X]=\sum_x xf_X(x)\); for continuous \(X\), \(E[X]=\int xf_X(x)dx\), when the expectation exists. For a measurable transformation \(g\), the law of the unconscious statistician gives \(E[g(X)]=\int g(x)f_X(x)dx\). The variance is \(\operatorname{Var}(X)=E[(X-\mu)^2]=E[X^2]-\mu^2\). Higher central moments describe skewness and kurtosis.

Linearity, \(E[aX+bY]=aE[X]+bE[Y]\), does not require independence. In contrast, \(E[XY]=E[X]E[Y]\) generally requires independence (or a suitable weaker condition).

## 4. Conditional expectation
For a discrete conditioning variable,
\[
E[Y\mid X=x]=\sum_y yP(Y=y\mid X=x).
\]
The conditional expectation \(E[Y\mid X]\) is a random variable and the best mean-square predictor of \(Y\) among functions measurable with respect to \(X\). It satisfies the tower property:
\[
E\{E(Y\mid X)\}=E(Y).
\]
The law of total variance is
\[
\operatorname{Var}(Y)=E[\operatorname{Var}(Y\mid X)]+\operatorname{Var}(E[Y\mid X]).
\]
The first term is average within-condition uncertainty; the second is variation explained by differences in conditional means.

## 5. Independence, covariance, and correlation
Independence means the joint distribution factors into marginal distributions. Zero covariance is weaker: it rules out linear association but not nonlinear dependence. For example, if \(X\) is symmetric around zero and \(Y=X^2\), covariance may be zero although \(Y\) is completely determined by \(X\). Correlation is covariance standardized by marginal standard deviations and is defined only when both variances are finite and positive.

## Worked example
Let \(X\sim Bernoulli(p)\), and let \(Y=a+bX+\varepsilon\), where \(E(\varepsilon\mid X)=0\). Then \(E(Y\mid X)=a+bX\), \(E(Y)=a+bp\), and
\[
\operatorname{Var}(Y)=b^2p(1-p)+E[\operatorname{Var}(\varepsilon\mid X)].
\]
This separates explained variation due to the binary regressor from average conditional residual variation.

## Problem set
1. Prove the complement and addition rules from the probability axioms.
2. Show that \(E[aX+b]=aE[X]+b\) and derive the variance transformation.
3. Construct dependent variables with zero covariance.
4. Derive the law of total variance using the tower property.
5. Explain why \(E[Y\mid X]\) is generally not equal to \(E[Y]\).

## Computational lab
Simulate Bernoulli and normal variables; compare empirical means, variances, and conditional means with their population values. Simulate \(Y=X^2\) for symmetric \(X\) and verify that sample correlation can be near zero despite dependence.

## Reading
Casella and Berger, *Statistical Inference*; DeGroot and Schervish, *Probability and Statistics*; Wasserman, *All of Statistics*.