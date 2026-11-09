# Lecture 04 — Real Business Cycle Models and Competitive Equilibrium

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Construct a real business cycle (RBC) model; derive equilibrium conditions; understand calibration, shock propagation, and impulse responses.

## 1. Household and firm
The household maximizes \(E_0\sum_t\beta^t[\log c_t-\chi n_t^{1+\varphi}/(1+\varphi)]\), subject to its budget and capital accumulation. The firm has
\[
y_t=A_tk_t^\alpha n_t^{1-\alpha}.
\]
Optimality gives the consumption Euler equation, labor-supply condition
\[
\chi n_t^\varphi=\frac{w_t}{c_t},
\]
and factor prices equal marginal products.

## 2. Productivity process
Let \(\log A_t=\rho_a\log A_{t-1}+\varepsilon_t^a\), with \(|\rho_a|<1\) and innovation variance \(\sigma_a^2\).

## 3. Competitive equilibrium
Given initial capital and shock processes, an equilibrium is a set of allocations and prices satisfying household and firm optimality, market clearing, and the capital law:
\[
y_t=c_t+i_t,\qquad k_{t+1}=(1-\delta)k_t+i_t.
\]
Government and external sectors are absent in the baseline model.

## 4. Quantitative analysis
Calibration assigns parameters using long-run averages or external estimates. A solution method maps shocks and states into decision rules. Impulse responses describe conditional dynamic responses; variance decompositions quantify contributions of shocks under the model; moments compare simulated data with empirical targets.

## 5. Worked steady state
At a deterministic steady state, \(1=\beta(\alpha y/k+1-\delta)\), so
\[
k/y=\frac{\alpha}{\beta^{-1}-1+\delta}.
\]
Use the resource constraint to obtain consumption and investment shares.

## 6. Computational lab
Build a Dynare or Python RBC model. Compute the steady state, verify residuals, solve at first order, simulate productivity shocks, and report IRFs for output, consumption, investment, hours, and wages. Change \(\rho_a\) and \(\sigma_a\) separately.

## Problem set
1. List all equilibrium conditions and endogenous variables.
2. Derive the steady-state capital-output ratio.
3. Explain the distinction between shock size and persistence.
4. Discuss which model moments are targeted in calibration.

## Further reading
King, Plosser and Rebelo (1988); Heer and Maussner, *Dynamic General Equilibrium Modeling*.
