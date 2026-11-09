# Lecture 07 — Monetary Policy: Rules, Commitment, and the Effective Lower Bound

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Evaluate policy rules and optimal policy; distinguish commitment from discretion; analyze the effective lower bound and forward guidance.

## 1. Policy rules
A Taylor-type rule links the nominal policy rate to inflation and economic activity:
\[
i_t=\rho_i i_{t-1}+(1-\rho_i)(\bar i+\phi_\pi\pi_t+\phi_xx_t)+\varepsilon_t^i.
\]
The intercept and inflation reference must be defined relative to the steady state. Interest-rate smoothing introduces policy inertia.

## 2. Commitment and discretion
Under commitment, policymakers choose a state-contingent plan and internalize how expectations respond to future policy. Under discretion, policymakers reoptimize each period taking private expectations as given. In forward-looking models, the two regimes can produce different inflation and output dynamics.

## 3. Loss function
A common quadratic approximation is
\[
L=\frac12E_0\sum_{t=0}^{\infty}\beta^t(\pi_t^2+\lambda_xx_t^2).
\]
This is a model-based welfare approximation under specified assumptions, not a universal social objective. The weight \(\lambda_x\) depends on structural parameters and the welfare derivation.

## 4. Effective lower bound
When the nominal rate cannot fall sufficiently, the real rate may remain too high relative to the natural rate. Forward guidance affects current demand through expected future nominal rates and inflation. Asset purchases can operate through term premia and portfolio-balance channels absent from the simplest representative-agent model.

## 5. Worked example
If expected inflation is 2% and the nominal rate is 1%, the ex ante real rate is approximately -1% using the Fisher approximation. If expected inflation falls while the nominal rate is constrained, the real rate rises, tightening policy endogenously.

## 6. Computational lab
Compare a Taylor rule, optimal commitment, and discretionary policy in a linear NK model. Plot inflation, output gap, and interest rate after a cost-push shock. State the welfare criterion and solution method.

## Problem set
1. Explain policy inertia.
2. Distinguish commitment from discretion.
3. Derive the approximate real interest rate.
4. Explain why forward guidance can affect current output.

## Further reading
Woodford, *Interest and Prices*; Clarida, Galí and Gertler (1999); Galí, *Monetary Policy, Inflation, and the Business Cycle*.
