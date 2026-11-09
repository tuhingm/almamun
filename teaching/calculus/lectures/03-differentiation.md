# Lecture 03 — Differentiation, Mean Value Theorems, and Comparative Rates

## Learning objectives
Derive derivatives from first principles; apply differentiation rules; use mean value theorems; interpret marginal effects, elasticities, and comparative rates.

## 1. Derivative definition
\[
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h},
\]
when the limit exists. It is the local linear approximation slope. Differentiability implies continuity, but continuity does not imply differentiability.

## 2. Differentiation rules
Derive sum, product, quotient, and chain rules from the limit definition. For compositions, \((f\circ g)'(x)=f'(g(x))g'(x)\). Exponential and logarithmic derivatives follow from their defining properties. Implicit differentiation applies the chain rule to equations defining variables jointly.

## 3. Higher derivatives and curvature
The second derivative measures the change in slope. If \(f''>0\) on an interval, the function is strictly convex there; if \(f''<0\), it is strictly concave. An inflection point requires a change in concavity, not merely a zero second derivative.

## 4. Mean value theorems
If \(f\) is continuous on \([a,b]\) and differentiable on \((a,b)\), there exists \(c\in(a,b)\) such that
\[
f'(c)=\frac{f(b)-f(a)}{b-a}.
\]
Rolle's theorem is the special case \(f(a)=f(b)\). These results establish monotonicity tests and uniqueness arguments.

## 5. Economic derivatives
Marginal cost is \(C'(q)\), marginal revenue is \(R'(q)\), and elasticity is \(\varepsilon_{y,x}=(dy/dx)(x/y)\). For positive variables, \(d\log y/d\log x\) equals elasticity. Interpret derivatives only over the relevant domain.

## Worked example
For \(C(q)=F+cq+dq^2\), marginal cost is \(MC=c+2dq\), increasing when \(d>0\). Average cost is \(F/q+c+dq\), which can have an interior minimum.

## Problem set
1. Derive the derivative of \(x^2\) from first principles.
2. Differentiate \(x^2e^{3x}\) and \(\log(1+x^2)\).
3. Prove that a differentiable function with zero derivative on an interval is constant.
4. Apply the mean value theorem to establish monotonicity.
5. Derive elasticity for a power function.

## Computational lab
Use symbolic differentiation and finite differences. Plot approximation error as step size changes and discuss round-off versus truncation error.

## Reading
Apostol, *Calculus*, Vol. 1; Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*; Simon and Blume, *Mathematics for Economists*.