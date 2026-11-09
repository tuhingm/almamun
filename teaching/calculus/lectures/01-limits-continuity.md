# Lecture 01 — Limits, Continuity, and the Logic of Analysis

**Level:** Undergraduate with graduate mathematical extensions

## Learning objectives
Use epsilon-delta definitions; evaluate limits; distinguish pointwise values from limiting behavior; establish continuity; apply the intermediate and extreme value theorems.

## 1. Functions and domains
A function \(f:D\to\mathbb R\) assigns one output to each input in its domain. Domain restrictions arise from denominators, even roots, logarithms, and feasibility constraints. Composition and inverse functions require compatible domains; an inverse requires injectivity on the selected domain.

## 2. Limits
We say \(\lim_{x\to a}f(x)=L\) if for every \(\varepsilon>0\), there exists \(\delta>0\) such that
\[
0<|x-a|<\delta\implies |f(x)-L|<\varepsilon.
\]
The puncture excludes the value at \(a\): the limit concerns nearby behavior. Left and right limits must agree for a two-sided limit to exist. Limit laws follow from the epsilon-delta definition.

## 3. Techniques
Use algebraic simplification, factorization, rationalization, squeeze arguments, and standard trigonometric or exponential limits. For rational functions at infinity, compare leading powers. L'Hôpital's rule applies only to indeterminate forms such as \(0/0\) or \(\infty/\infty\), under its differentiability and denominator-derivative conditions.

## 4. Continuity
A function is continuous at \(a\) if it is defined there and \(\lim_{x\to a}f(x)=f(a)\). Removable, jump, and infinite discontinuities differ in limiting behavior. Sums, products, and compositions of continuous functions are continuous wherever defined.

## 5. Fundamental theorems
The intermediate value theorem states that a continuous function on \([a,b]\) attains every value between \(f(a)\) and \(f(b)\). The extreme value theorem guarantees attainment of a maximum and minimum on a compact interval. These results establish existence, not uniqueness.

## Worked example
For \(f(x)=(x^2-1)/(x-1)\), \(x\ne1\), factor to obtain \(f(x)=x+1\). Thus \(\lim_{x\to1}f(x)=2\), although the original expression is undefined at 1. Defining \(f(1)=2\) removes the discontinuity.

## Problem set
1. Prove directly that \(\lim_{x\to2}(3x+1)=7\).
2. Evaluate \(\lim_{x\to0}(\sqrt{1+x}-1)/x\).
3. Give examples of removable, jump, and infinite discontinuities.
4. State the assumptions of the intermediate value theorem.
5. Explain why continuity on an open interval does not guarantee an attained maximum.

## Computational lab
Plot functions with removable and jump discontinuities. Compare numerical limits at decreasing step sizes with analytic limits; explain why numerical evidence is not a proof.

## Reading
Apostol, *Calculus*, Vol. 1; Spivak, *Calculus*; Rudin, *Principles of Mathematical Analysis*.