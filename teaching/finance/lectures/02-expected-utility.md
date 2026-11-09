# Lecture 02 — Risk, Probability, and Expected Utility

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Represent uncertainty with probability distributions; derive expected utility and certainty equivalents; understand risk aversion and downside risk.

## 1. Random returns
A risky payoff is a random variable with distribution \(F(R)\). Expected return and variance are
\[
E(R)=\int R\,dF(R),\qquad \operatorname{Var}(R)=E[(R-E(R))^2].
\]
Higher moments capture skewness and tail thickness, which variance alone cannot describe.

## 2. Expected utility
An investor chooses a portfolio to maximize \(E[u(W_1)]\). Absolute risk aversion is \(A(W)=-u''(W)/u'(W)\); relative risk aversion is \(R(W)=-Wu''(W)/u'(W)\). Concavity, \(u''<0\), implies risk aversion.

## 3. Certainty equivalent and risk premium
The certainty equivalent \(CE\) solves \(u(CE)=E[u(W)]\). The risk premium is \(E(W)-CE\). For small risks, a second-order approximation gives a risk premium near \(\frac12 A(W)\operatorname{Var}(W)\), evaluated at a suitable wealth level.

## 4. Mean-variance preferences
A quadratic objective can be written
\[
U(w)=w'\mu-\frac{\gamma}{2}w'\Sigma w.
\]
This is exact for quadratic utility and is also used as a portfolio criterion. It does not imply that all investors care only about variance under arbitrary return distributions.

## 5. Worked example
An investment pays 120 or 80 with equal probability. Compute expected payoff and variance. For logarithmic utility, compare expected utility with utility at expected wealth to establish the certainty-equivalent relation.

## 6. Computational lab
Simulate normal and negatively skewed return distributions with equal means and variances. Compare expected shortfall and expected utility to demonstrate why equal variance need not imply equal downside risk.

## Problem set
1. Define absolute and relative risk aversion.
2. Derive the second-order risk-premium approximation.
3. Explain the certainty equivalent.
4. Give a case where variance fails to distinguish economically relevant risk.

## Further reading
Pratt (1964); Arrow, *Essays in the Theory of Risk-Bearing*; Cochrane, *Asset Pricing*.
