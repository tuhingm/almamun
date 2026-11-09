# Lecture 08 — Fiscal Policy, Public Debt, and Monetary-Fiscal Interactions

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Derive debt dynamics; distinguish primary deficits from total deficits; analyze fiscal multipliers, sustainability, and Ricardian equivalence.

## 1. Government budget constraint
In nominal terms,
\[
B_t=(1+i_{t-1})B_{t-1}+P_tG_t-P_tT_t,
\]
where \(B_t\) is end-of-period nominal debt, \(G_t\) government purchases, and \(T_t\) net taxes. Dividing by nominal GDP and using a consistent timing convention gives an approximate debt ratio law
\[
b_t\approx\frac{1+r_t}{1+g_t}b_{t-1}+d_t,
\]
where \(r_t\) is the real interest rate, \(g_t\) real GDP growth, and \(d_t\) the primary deficit-to-GDP ratio. For small rates, \(\Delta b_t\approx(r_t-g_t)b_{t-1}+d_t\).

## 2. Intertemporal constraint
Under a no-Ponzi condition, current debt equals the present value of future primary surpluses, subject to valuation and discounting conventions. Fiscal sustainability is therefore about the feasible path of future primary balances, not a single debt threshold.

## 3. Fiscal multipliers
The output effect of government spending depends on monetary accommodation, price rigidity, household liquidity constraints, openness, financing, and the horizon. A multiplier is a ratio of an output response to a specified fiscal impulse; report its definition and horizon.

## 4. Ricardian equivalence
Under restrictive assumptions—lump-sum taxes, complete markets, forward-looking unconstrained households, and appropriate bequest motives—debt-financed tax cuts may not change consumption because households anticipate future taxes. Violations of these assumptions generate non-Ricardian effects.

## 5. Worked example
If \(b_{t-1}=0.8\), \(r=0.04\), \(g=0.02\), and the primary deficit is 0.01, the exact approximate ratio update is \(b_t=(1.04/1.02)0.8+0.01\). Calculate the new debt ratio and isolate the interest-growth contribution.

## 6. Computational lab
Simulate debt paths under alternative primary-balance rules and interest-growth differentials. Add a spending shock to an NK model and report impact and cumulative multipliers.

## Problem set
1. Derive the debt-ratio transition.
2. Explain the role of \(r-g\).
3. State the assumptions behind Ricardian equivalence.
4. Compare spending and tax multipliers in a specified model.

## Further reading
Leeper (1991); Woodford, *Interest and Prices*; Blanchard, *Macroeconomics*.
