# Lecture 02 — Firms, Capital Accumulation, and Investment

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Derive factor demands, the capital Euler equation, and Tobin's \(q\); understand depreciation, adjustment costs, and investment dynamics.

## 1. Technology and static optimization
A competitive firm produces
\[
y_t=A_t k_t^\alpha n_t^{1-\alpha},\quad 0<\alpha<1.
\]
Taking factor prices as given, profit maximization implies
\[
w_t=(1-\alpha)A_t(k_t/n_t)^\alpha,\qquad
r_t^k=\alpha A_t(k_t/n_t)^{\alpha-1},
\]
where \(r_t^k\) is the rental rate of capital before depreciation.

## 2. Capital accumulation
Capital evolves according to
\[
k_{t+1}=(1-\delta)k_t+i_t,
\]
with depreciation rate \(\delta\in(0,1)\). Investment is a flow; installed capital is a stock. Confusing these objects leads to incorrect timing in dynamic models.

## 3. Investment with adjustment costs
Let investment adjustment costs be \(\frac{\phi}{2}(i_t/k_t-\delta)^2k_t\). The firm chooses investment to maximize discounted dividends. Define marginal value of installed capital as \(q_t\). The investment FOC is
\[
q_t=1+\phi(i_t/k_t-\delta).
\]
The capital valuation equation equates the value of installed capital to its marginal product plus the discounted continuation value:
\[
q_t=E_t\left[M_{t,t+1}\{r_{t+1}^k+(1-\delta)q_{t+1}\}\right],
\]
where \(M_{t,t+1}\) is the stochastic discount factor.

## 4. Balanced growth and normalization
When technology grows, detrend real quantities by the appropriate technology or labor-augmenting trend. A stationary model requires consistent scaling of output, consumption, investment, and capital. Verify that the normalized capital law preserves the original resource constraint.

## 5. Worked example
With Cobb–Douglas production, \(A=1\), \(k/n=4\), and \(\alpha=0.36\), the marginal product of capital is \(0.36(4)^{-0.64}\). The wage is \(0.64(4)^{0.36}\). These prices satisfy the competitive firm's factor-share conditions.

## 6. Computational lab
Compute factor prices and a steady-state capital stock. Add quadratic adjustment costs and simulate the response of investment and Tobin's \(q\) to a temporary productivity shock. Check the capital law period by period.

## Problem set
1. Derive both factor-demand equations.
2. Explain why investment can respond more slowly when \(\phi\) is large.
3. Derive the investment FOC from the adjustment-cost function.
4. Interpret the capital valuation equation.

## Further reading
Hayashi (1982), “Tobin's Marginal q and Average q”; Ljungqvist and Sargent, *Recursive Macroeconomic Theory*.
