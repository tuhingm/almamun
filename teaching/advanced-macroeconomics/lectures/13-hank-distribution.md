# Lecture 13 — HANK Models: Distribution Dynamics and Aggregate Transmission

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Understand how household heterogeneity changes aggregate policy transmission; formulate continuous-time HJB and Kolmogorov forward equations; interpret sequence-space methods.

## 1. Heterogeneous-agent New Keynesian structure
HANK models combine nominal rigidities and monetary policy with incomplete-market households. Aggregate demand depends on the distribution of wealth and income because households have different marginal propensities to consume (MPCs). Transfers and interest-rate changes therefore have distribution-dependent effects.

## 2. Continuous-time household problem
For liquid assets \(a\), idiosyncratic state \(z\), and value function \(V(a,z)\), a stationary HJB equation takes the form
\[
\rho V(a,z)=\max_c\left\{u(c)+V_a(a,z)[wz+ra-c]\right\}+\mathcal L_zV,
\]
subject to the borrowing constraint. \(\mathcal L_z\) is the generator of the idiosyncratic income process.

## 3. Distribution dynamics
The Kolmogorov forward equation evolves the density \(g_t(a,z)\):
\[
\partial_t g_t=-\partial_a[\dot a(a,z)g_t]+\mathcal L_z^*g_t.
\]
The adjoint operator \(\mathcal L_z^*\) governs probability flows across idiosyncratic states. Stationarity requires zero net change in the distribution.

## 4. General equilibrium and policy
The household block determines consumption and asset supply; firms and nominal rigidities determine production and inflation; monetary and fiscal rules close the model. A transfer shock can have a large demand effect when directed toward high-MPC households, even if aggregate transfer expenditure is unchanged.

## 5. Sequence-space Jacobians
Linear response can be represented by Jacobians mapping aggregate paths (interest rates, wages, transfers) into paths of consumption, assets, and prices. Combining household and firm Jacobians with market-clearing conditions yields aggregate impulse responses. Explain the discretization and horizon underlying every computed Jacobian.

## 6. Computational lab
Solve a stationary heterogeneous-household block, compute MPCs by wealth percentile, and simulate a temporary transfer. Compare equal-per-capita and targeted transfers with the same aggregate budget. Report distributional and aggregate outcomes.

## Problem set
1. Interpret the HJB equation.
2. Explain the forward equation and its adjoint operator.
3. Why does the wealth distribution matter for aggregate demand?
4. Describe the sequence-space Jacobian workflow.

## Further reading
Kaplan, Moll and Violante (2018); Auclert et al. (2021); Achdou et al. (2022).
