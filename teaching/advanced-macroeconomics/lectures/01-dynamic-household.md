# Lecture 01 — The Dynamic Household: Intertemporal Choice and Euler Equations

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Derive household optimality conditions in discrete time; distinguish budget feasibility from optimality; interpret the Euler equation, borrowing constraints, and transversality.

## 1. Environment
A representative household chooses consumption \(c_t\) and next-period assets \(a_{t+1}\) to maximize
\[
\max_{\{c_t,a_{t+1}\}} E_0\sum_{t=0}^{\infty}\beta^t u(c_t),\quad 0<\beta<1,
\]
subject to
\[
c_t+a_{t+1}=w_t n_t+(1+r_t)a_t+T_t.
\]
Here \(\beta\) is the discount factor, \(r_t\) the real interest rate, \(w_t\) the real wage, and \(T_t\) net transfers.

## 2. Euler equation
Let \(\lambda_t\) be the multiplier on the period budget. The Lagrangian yields \(u'(c_t)=\lambda_t\) and \(\lambda_t=\beta E_t[(1+r_{t+1})\lambda_{t+1}]\). Hence
\[
u'(c_t)=\beta E_t[(1+r_{t+1})u'(c_{t+1})].
\]
With CRRA utility \(u(c)=c^{1-\sigma}/(1-\sigma)\), deterministic consumption growth satisfies
\[
\frac{c_{t+1}}{c_t}=\left[\beta(1+r_{t+1})\right]^{1/\sigma}.
\]
Under uncertainty, the expectation applies to marginal utility times the gross return; it cannot generally be separated into products of expectations.

## 3. Lifetime budget and transversality
Iterating the flow budget forward gives the present-value lifetime resource constraint, subject to a terminal condition. A standard transversality condition is
\[
\lim_{T\to\infty}E_0[\beta^T u'(c_T)a_{T+1}]=0.
\]
It rules out plans inconsistent with household optimality and the lifetime resource constraint.

## 4. Borrowing constraints
If \(a_{t+1}\geq\underline a\), the Euler condition becomes an inequality at a binding lower bound. With multiplier \(\mu_t\geq0\),
\[
u'(c_t)=\beta E_t[(1+r_{t+1})u'(c_{t+1})]+\mu_t,\quad
\mu_t(a_{t+1}-\underline a)=0.
\]
A constrained household may consume less today than the unconstrained Euler equation prescribes.

## 5. Worked example
For log utility, deterministic constant return \(r\), and no labor income after date zero, the Euler equation gives \(c_{t+1}/c_t=\beta(1+r)\). The lifetime budget and terminal condition determine the consumption level.

## 6. Computational lab
Implement a finite-horizon household problem. Discretize assets, solve by backward induction, and compare policy functions with and without a borrowing limit. Report grid convergence, Euler-equation errors, and the share of states at the constraint.

## Problem set
1. Derive the Euler equation from the Lagrangian.
2. Derive the CRRA consumption-growth equation.
3. State the transversality condition and explain its economic role.
4. Show how a binding borrowing constraint modifies the Euler condition.

## Further reading
Ljungqvist and Sargent, *Recursive Macroeconomic Theory*; Deaton, *Understanding Consumption*.
