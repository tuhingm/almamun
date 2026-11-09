# Lecture 05 — Integration, Riemann Sums, and the Fundamental Theorem

## Learning objectives
Define the Riemann integral; connect antiderivatives and accumulated area; apply substitution and integration by parts; evaluate improper integrals and interpret economic accumulation.

## 1. Riemann sums
Partition \([a,b]\) into subintervals and choose sample points \(x_i^*\). The Riemann sum is \(\sum_i f(x_i^*)\Delta x_i\). If the limit exists as the maximum partition width tends to zero, it defines the Riemann integral \(\int_a^b f(x)dx\). Continuous functions on compact intervals are Riemann integrable.

## 2. Antiderivatives
An antiderivative \(F\) satisfies \(F'=f\). Indefinite integration describes a family \(F(x)+C\). Definite integration is a signed accumulation: regions below the axis contribute negatively.

## 3. Fundamental theorem of calculus
If \(f\) is continuous on \([a,b]\), then
\[
\frac{d}{dx}\int_a^x f(t)dt=f(x),\qquad
\int_a^b f(x)dx=F(b)-F(a).
\]
The first statement connects accumulation to marginal change; the second evaluates a definite integral through an antiderivative.

## 4. Integration techniques
Substitution reverses the chain rule. Integration by parts follows from the product rule:
\[
\int u\,dv=uv-\int v\,du.
\]
Partial fractions decompose rational functions when the denominator factorization permits. Select a method by inspecting the integrand structure.

## 5. Improper integrals
An infinite-interval integral is defined by a limit, e.g. \(\int_a^\infty f(x)dx=\lim_{b\to\infty}\int_a^b f(x)dx\). Integrals with singularities are similarly defined by one-sided limits. Convergence must be established before assigning a finite value.

## 6. Economic applications
Integrals measure accumulated flows, consumer and producer surplus, present value under continuous discounting, and expected values. If a flow is \(r(t)\), accumulated quantity over \([0,T]\) is \(\int_0^T r(t)dt\).

## Worked example
\[
\int_0^1 xe^{x^2}dx=\frac12\int_0^1e^u du=\frac{e-1}{2},
\]
using \(u=x^2\).

## Problem set
1. Interpret a definite integral as a limit of sums.
2. Evaluate \(\int (3x^2+2x)dx\).
3. Evaluate \(\int_0^1 xe^{x^2}dx\).
4. Use integration by parts to evaluate \(\int x e^x dx\).
5. Determine convergence of \(\int_1^\infty x^{-p}dx\).

## Computational lab
Approximate definite integrals with left, right, and midpoint sums; compare errors with exact antiderivatives. Explore numerical integration near an integrable singularity.

## Reading
Apostol, *Calculus*, Vol. 1; Spivak, *Calculus*; Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*.