# Lecture 12 — Heterogeneous-Agent Macroeconomics and Incomplete Markets

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Formulate household dynamic programming under idiosyncratic risk; derive policy functions and stationary distributions; define aggregate equilibrium.

## 1. Household problem
Households face idiosyncratic productivity \(z\), asset holdings \(a\), and borrowing limit \(\underline a\). The Bellman equation is
\[
V(a,z)=\max_{c,a'}\left\{u(c)+\beta\sum_{z'}P(z'\mid z)V(a',z')\right\}
\]
subject to \(c+a'=wz+(1+r)a\) and \(a'\geq\underline a\).

## 2. Optimality
For an interior choice,
\[
u'(c)=\beta(1+r)\sum_{z'}P(z'\mid z)V_a(a',z').
\]
The envelope condition is \(V_a(a,z)=u'(c)(1+r)\). Combining conditions yields the stochastic Euler equation. At the borrowing limit, the complementary-slackness condition replaces equality with an inequality.

## 3. Distribution and aggregation
Let \(\mu_t(a,z)\) denote the cross-sectional distribution. Household policy functions and the Markov transition for \(z\) induce a law of motion \(\mu_{t+1}=\mathcal T(\mu_t)\). Aggregate assets are \(A_t=\int a\,d\mu_t\); labor and consumption are aggregated similarly.

## 4. Stationary equilibrium
A stationary recursive competitive equilibrium consists of prices, household value and policy functions, and an invariant distribution consistent with household optimization, exogenous transitions, firm pricing, and market clearing. In a closed economy without government debt, aggregate capital supply equals firms' capital demand.

## 5. Worked algorithm
Discretize assets and productivity. Guess prices, solve the household problem by value-function iteration or endogenous-grid methods, compute the invariant distribution, aggregate assets, update prices, and iterate until market clearing holds.

## 6. Computational lab
Compare incomplete-markets consumption-saving behavior across borrowing limits and idiosyncratic risk levels. Report policy functions, invariant distributions, aggregate capital, and market-clearing errors. Check grid and Markov-chain accuracy.

## Problem set
1. Derive the household Euler and envelope equations.
2. Explain why individual states differ from aggregate states.
3. Define the invariant distribution.
4. Describe the equilibrium fixed-point algorithm.

## Further reading
Aiyagari (1994); Huggett (1993); Ljungqvist and Sargent, *Recursive Macroeconomic Theory*.
