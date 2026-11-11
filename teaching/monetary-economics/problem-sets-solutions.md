# Monetary Economics — Problem Sets and Selected Solution Guidance

Each problem set combines derivations with a short MATLAB/Dynare task. Show intermediate steps, state timing conventions, and submit `.mod` and `.m` files that run from a clean folder.

## Problem set 1 — Money demand and the cost of inflation (Lectures 1–2)
1. From the quantity equation \(M_tV_t=P_tY_t\), derive the relation between inflation, money growth, velocity growth and output growth.
2. Under semi-log money demand \(\log(M/PY)=a-\xi i\), derive the Bailey welfare cost of a nominal rate \(i\).
3. In a cash-in-advance economy where only consumption requires cash, show that money growth acts as a tax on market work.
4. *MATLAB:* estimate \(\xi\) for M1 on `us_quarterly.csv` and compute the welfare cost of 2% versus 10% inflation.

**Solution guidance:** The Bailey cost is the area under the inverse money-demand curve, \(w(i)=\int_0^i m(x)\,dx-i\,m(i)\), which for the semi-log form is \(\frac{e^{a}}{\xi}\left[1-(1+\xi i)e^{-\xi i}\right]\). In the cash-in-advance model the first-order conditions give \(u_l/u_c=w/R\): the gross nominal rate \(R\) drives a wedge between the marginal rate of substitution and the real wage, which vanishes only at the Friedman rule.

## Problem set 2 — The classical monetary model (Lecture 3)
1. Show that output, employment and the real interest rate do not depend on monetary policy.
2. Under an exogenous AR(1) money-growth process, solve for inflation and the nominal rate by the method of undetermined coefficients.
3. Show that an interest-rate peg leaves the price level indeterminate and that \(\phi_\pi>1\) restores determinacy.
4. *Dynare:* write a `.mod` file for the model and confirm the impact response of the nominal rate.

**Solution guidance:** The real block (labour supply, production, Euler equation) is self-contained. Combining money demand with the Fisher equation gives a forward-looking difference equation in the price level; with \(\phi_\pi>1\) the only bounded solution is the forward solution. Check Blanchard–Kahn: one forward-looking variable requires one unstable root.

## Problem set 3 — Price setting and the New Keynesian Phillips curve (Lecture 4)
1. Derive the optimal reset price under Calvo pricing and log-linearize it.
2. Derive \(\pi_t=\beta E_t\pi_{t+1}+\kappa\tilde y_t\) and express \(\kappa\) in terms of \(\theta,\beta,\sigma,\varphi,\alpha,\epsilon\).
3. Show how real rigidities (decreasing returns, strategic complementarity) flatten the Phillips curve.
4. Add Calvo wage setting and derive the wage-inflation equation.

**Solution guidance:** Start from the reset-price first-order condition, divide by the price level and linearize around the zero-inflation steady state. Combine with the law of motion of the aggregate price index, \(\pi_t=(1-\theta)(p^*_t-p_{t-1})\). The slope \(\lambda=\frac{(1-\theta)(1-\beta\theta)}{\theta}\Theta\) falls steeply as \(\theta\to1\).

## Problem set 4 — The New Keynesian model and optimal policy (Lectures 5–6)
1. Solve the three-equation model under a Taylor rule and an AR(1) monetary shock; derive the impact output gap and inflation.
2. Derive the determinacy condition \(\kappa(\phi_\pi-1)+(1-\beta)\phi_y>0\).
3. Derive the second-order welfare loss and the weight \(\vartheta=\kappa/\epsilon\) on the output gap.
4. Derive optimal policy under discretion and under commitment after a cost-push shock.

**Solution guidance:** Guess \(\tilde y_t=\psi_y v_t\), \(\pi_t=\psi_\pi v_t\) and match coefficients. Under discretion the targeting rule is \(\kappa\pi_t=-\vartheta\tilde y_t\); under commitment it is \(\kappa\pi_t=-\vartheta(\tilde y_t-\tilde y_{t-1})\), equivalently \(\kappa\hat p_t=-\vartheta\tilde y_t\), which makes the price level stationary.

## Problem set 5 — Policy rules and open economies (Lectures 7–8)
1. Estimate a Taylor rule with interest-rate smoothing before and after 1979; compute the long-run inflation response.
2. Show how a price-level target reproduces the history dependence of commitment.
3. In the Galí–Monacelli model, derive the open-economy IS curve and Phillips curve for domestic inflation.
4. *Dynare:* compare a domestic-inflation rule, a CPI-inflation rule and an exchange-rate peg.

**Solution guidance:** The long-run response is \(\phi_\pi/(1-\rho)\) when the rule is written as \(i_t=\rho i_{t-1}+\phi_\pi\pi_t+\dots\); use lagged variables as instruments to deal with simultaneity. In the small open economy, openness changes the slope coefficients through \(\sigma_\alpha\) but the model keeps the closed-economy form in terms of domestic inflation and the output gap.

## Problem set 6 — Financial frictions and the zero lower bound (Lectures 9–10)
1. Explain how the external finance premium depends on borrower net worth in the BGG model.
2. Solve a two-state liquidity trap with absorbing exit and derive output and inflation in the trap.
3. Show that the effect of a promised rate cut \(T\) periods ahead on today's output does not decline with \(T\) under the standard Euler equation.
4. Show how a discounted Euler equation resolves the forward-guidance puzzle.

**Solution guidance:** In the trap state the system is \(x_L=\mu x_L+\sigma^{-1}(\mu\pi_L+r^n_L)\), \(\pi_L=\kappa x_L+\beta\mu\pi_L\); solve the two linear equations and note that the denominator \((1-\mu)(1-\beta\mu)-\sigma^{-1}\mu\kappa\) can approach zero. Under the standard Euler equation, iterate forward: \(x_t=-\sigma^{-1}\sum_{j\ge0}E_t(i_{t+j}-\pi_{t+j+1}-r^n_{t+j})\).

## Problem set 7 — Monetary–fiscal interactions and HANK (Lectures 11–12)
1. Classify policy as active or passive and derive the determinacy conditions with a debt-feedback tax rule.
2. Use the government valuation equation to explain the fiscal theory of the price level.
3. Derive the aggregate IS curve of a two-agent model and the amplification factor \(\delta\).
4. Explain why high marginal propensities to consume make indirect effects of monetary policy dominate.

**Solution guidance:** A unique equilibrium needs exactly one active authority. In the TANK model the aggregate IS curve has the slope \(\sigma^{-1}\delta\) with \(\delta=\frac{1-\lambda}{1-\lambda\chi}\); amplification occurs when hand-to-mouth income is more cyclical than average (\(\chi>1\)).

## Problem set 8 — Estimation and identification (Lectures 13–15)
1. Write the state-space form of the three-equation model and the Kalman-filter log-likelihood.
2. Set priors for \(\kappa,\phi_\pi,\phi_y\) and explain how to detect weak identification.
3. Explain the price puzzle in a recursive VAR and how narrative or high-frequency shocks address it.
4. Compare VAR and local-projection impulse responses; explain the bias–variance trade-off.

**Solution guidance:** Report the acceptance rate, run several chains and compare prior and posterior densities. A posterior that barely moves from its prior signals weak identification. A recursive ordering assumes the policy rate reacts to, but does not affect within the period, the variables ordered before it; external instruments require relevance and exogeneity.
