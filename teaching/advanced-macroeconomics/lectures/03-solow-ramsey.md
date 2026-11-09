# Lecture 03 — Solow and Ramsey Growth: Steady States and Transitional Dynamics

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Derive the Solow steady state, golden-rule capital, and Ramsey saddle path; distinguish exogenous saving from optimal saving.

## 1. Solow model
In per-worker terms, output is \(y=f(k)=k^\alpha\). Capital evolves as
\[
\dot k=s f(k)-(n+g+\delta)k,
\]
where \(s\) is the saving rate, \(n\) population growth, \(g\) labor-augmenting technology growth, and \(\delta\) depreciation.

The steady state satisfies \(sf(k^*)=(n+g+\delta)k^*\), giving
\[
k^*=\left(\frac{s}{n+g+\delta}\right)^{1/(1-\alpha)}.
\]
A higher saving rate raises steady-state capital and output per effective worker, but does not permanently raise the exogenous long-run growth rate.

## 2. Golden-rule capital
Steady-state consumption is \(c=f(k)-(n+g+\delta)k\). The golden-rule condition is
\[
f'(k_{GR})=n+g+\delta.
\]
It maximizes steady-state consumption, not necessarily welfare along the transition.

## 3. Ramsey household
A planner chooses consumption to maximize
\[
\int_0^\infty e^{-\rho t}u(c_t)dt
\]
subject to \(\dot k=f(k)-c-(n+g+\delta)k\). For CRRA utility, the Euler equation is
\[
\frac{\dot c}{c}=\frac{1}{\sigma}[f'(k)-\delta-\rho-\sigma g],
\]
under the standard effective-worker normalization and corresponding parameter convention. State the normalization explicitly when applying this expression.

## 4. Phase diagram and stability
The \(\dot k=0\) locus is \(c=f(k)-(n+g+\delta)k\). The \(\dot c=0\) locus is determined by the modified golden-rule condition. The steady state is typically saddle-path stable: one stable eigenvalue and one unstable eigenvalue. The initial capital stock is predetermined; consumption jumps onto the stable arm.

## 5. Worked example
For \(f(k)=k^{0.3}\), \(s=0.2\), and \(n+g+\delta=0.05\), the Solow steady state is \(k^*=(0.2/0.05)^{1/0.7}\). Evaluate numerically and calculate steady-state output and consumption.

## 6. Computational lab
Plot Solow transition paths for alternative saving rates. For Ramsey, linearize the two-dimensional system around steady state, compute eigenvalues, and identify the stable eigenvector.

## Problem set
1. Derive the Solow steady state.
2. Derive the golden-rule condition.
3. Explain why the Ramsey model has a jump variable.
4. Compare a saving-rate increase in Solow and Ramsey.

## Further reading
Acemoglu, *Introduction to Modern Economic Growth*; Barro and Sala-i-Martin, *Economic Growth*.
