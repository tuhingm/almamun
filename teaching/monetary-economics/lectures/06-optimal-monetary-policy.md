# Lecture 06 — Optimal Monetary Policy

**Level:** Graduate | **Field:** Monetary Economics

## Learning objectives
Derive the welfare-based loss function of the New Keynesian model, characterise optimal policy under discretion and under commitment, and explain the divine coincidence, inflation bias and stabilization bias. Compare optimal policy with simple rules.

## 1. The welfare loss
A second-order approximation of household utility around an efficient steady state (Rotemberg and Woodford, 1997; Woodford, 2003) gives the expected loss
\[
\mathbb W=\frac12E_0\sum_{t=0}^\infty\beta^t\left[\left(\sigma+\frac{\varphi+\alpha}{1-\alpha}\right)\tilde y_t^2+\frac{\epsilon}{\lambda}\pi_t^2\right]
\;\propto\;E_0\sum_{t=0}^\infty\beta^t\left(\pi_t^2+\vartheta\tilde y_t^2\right),\qquad \vartheta=\frac{\kappa}{\epsilon}.
\]
Inflation is costly because, with staggered prices, it creates **price dispersion**: identical goods sell at different prices, so resources are misallocated. The weight on inflation is large: \(\epsilon/\lambda\) is high when goods are close substitutes and prices change rarely. The small weight \(\vartheta\) on the output gap (about 0.02 in a standard quarterly calibration) is why welfare-based analysis favours strict inflation stability.

## 2. The divine coincidence
With only technology and demand shocks, setting \(\pi_t=0\) for all \(t\) implies \(\tilde y_t=0\) through the Phillips curve. Both terms of the loss are zero. The optimal policy tracks the natural rate, \(i_t=r^n_t\), and can be implemented by \(i_t=r^n_t+\phi_\pi\pi_t\) with \(\phi_\pi>1\) to rule out indeterminacy. Blanchard and Galí (2007) call this the **divine coincidence**: there is no trade-off between stabilising inflation and stabilising the welfare-relevant output gap.

## 3. Cost-push shocks and the trade-off
The coincidence fails if the Phillips curve contains a **cost-push shock** \(u_t\) (time-varying markups or wage markups):
\[
\pi_t=\beta E_t\pi_{t+1}+\kappa\tilde y_t+u_t .
\]
Now the central bank must trade off inflation against the output gap.

**Discretion.** A central bank that re-optimises every period takes expectations as given. The first-order condition is \(\tilde y_t=-\frac{\kappa}{\vartheta}\pi_t\): lean against inflation by contracting output. With AR(1) cost-push shocks,
\[
\pi_t=\frac{\vartheta}{\kappa^2+\vartheta(1-\beta\rho_u)}u_t,\qquad \tilde y_t=-\frac{\kappa}{\kappa^2+\vartheta(1-\beta\rho_u)}u_t .
\]
Bygones are bygones: once the shock is over, the price level stays permanently higher.

**Commitment.** A central bank that can commit chooses state-contingent paths for \(\{\pi_t,\tilde y_t\}\) at time 0. The Lagrangian conditions give \(\pi_t=-\frac{\vartheta}{\kappa}(\tilde y_t-\tilde y_{t-1})\) for \(t\ge1\); in the *timeless perspective*, the same condition applies at \(t=0\). Summing over time, \(\hat p_t=-\frac{\vartheta}{\kappa}\tilde y_t\): the optimal commitment is a **price-level target** in disguise. The solution is \(\tilde y_t=\delta\tilde y_{t-1}-\frac{\kappa\delta}{\vartheta(1-\delta\beta\rho_u)}u_t\), with \(\delta=\frac{1-\sqrt{1-4\beta a^2}}{2a\beta}\) and \(a=\frac{\vartheta}{\vartheta(1+\beta)+\kappa^2}\). By promising a persistent output contraction, the central bank lowers expected future inflation, which improves today's trade-off: inflation rises by less on impact for a given output cost. This **stabilization bias** of discretion exists even when the central bank has no incentive to push output above its natural level.

## 4. Inflation bias
If the steady state is inefficient (because of monopoly markups not offset by subsidies), the loss contains a target \(\tilde y^*>0\) for the output gap. Under discretion the central bank is tempted to create surprise inflation; rational agents anticipate it, and the equilibrium has average inflation \(\pi=\frac{\vartheta\kappa}{\kappa^2+\vartheta(1-\beta)}\tilde y^*>0\) with an output gap of only \((1-\beta)\pi/\kappa\), close to zero: inflation is higher with almost no output gain (Kydland and Prescott, 1977; Barro and Gordon, 1983). Solutions include reputation, delegation to a conservative central banker (Rogoff, 1985), inflation contracts (Walsh, 1995) and central-bank independence with a clear mandate (Lecture 15).

## 5. Optimal policy versus simple rules
Optimal commitment policy is history dependent and depends on all shocks. In practice central banks follow, or are described by, simple rules. One can choose the coefficients of a simple rule to minimise the welfare loss — the approach of Exercise 6.2 — or compute the **Ramsey policy** directly. Dynare supports both: `ramsey_model(planner_discount=0.99)` with a `planner_objective` computes optimal commitment, `discretionary_policy` the discretionary solution, and `osr` optimises the coefficients of a simple rule. Schmitt-Grohé and Uribe (2007) find that optimised simple rules responding strongly to inflation, with little response to output, come close to Ramsey welfare in medium-scale models.

## Problem set
1. Explain why price dispersion makes inflation costly in the Calvo model, and why the loss weight on inflation rises with \(\epsilon\) and with \(\theta\).
2. Derive the discretionary solution with AR(1) cost-push shocks.
3. Derive the first-order conditions under commitment and show that the timeless-perspective policy implies a stationary price level.
4. Show how a positive output target produces inflation bias under discretion but not under commitment.
5. Explain why an optimised simple rule may assign little weight to the output gap even though the central bank cares about it.

## Coding exercises

These exercises use **MATLAB and Dynare only**. Download the course [data files and helper functions](../index.html#data-and-code)
into your working folder. Every data series is real: U.S. quarterly and monthly macro data (FRED-QD / FRED-MD,
1959–2025), published monetary-policy shock series (Romer–Romer, Jarociński–Karadi, Bu–Rogers–Wu,
Nakamura–Steinsson), World Bank CPI inflation for 190+ countries, and the KRW/USD exchange rate. Dynare models are
`.mod` files run with `dynare file noclearall` (Dynare 6). Try the starter files first, then open the solution. Every
printed result was produced by running the solution files as shown: the `.mod` files were processed by the Dynare
preprocessor and solved at first order, and the MATLAB code was run in GNU Octave.

### Exercise 6.1 — Discretion versus commitment after a cost-push shock

*Data:* Model calibration (\(\beta=0.99,\ \kappa=0.1717,\ \epsilon=9\)).

The central bank minimises \(E_0\sum\beta^t(\pi_t^2+\vartheta\tilde y_t^2)\) with \(\vartheta=\kappa/\epsilon\),
subject to \(\pi_t=\beta E_t\pi_{t+1}+\kappa\tilde y_t+u_t\). Consider an i.i.d. cost-push shock of 1 at \(t=0\).

1. Under **discretion**, \(\tilde y_t=-(\kappa/\vartheta)\pi_t\), giving
   \(\pi_t=\frac{\vartheta}{\kappa^2+\vartheta}u_t\). Compute the paths of inflation, the output gap and the price level.
2. Under **commitment** (timeless perspective), \(\tilde y_t=\tilde y_{t-1}-(\kappa/\vartheta)\pi_t\), giving
   \(\tilde y_t=\delta\tilde y_{t-1}-\frac{\kappa\delta}{\vartheta}u_t\) with
   \(\delta=\frac{1-\sqrt{1-4\beta a^2}}{2a\beta}\), \(a=\frac{\vartheta}{\vartheta(1+\beta)+\kappa^2}\).
   Compute the same paths.
3. Compare the discounted losses. Why is commitment better, and what happens to the price level?

*File `ex6_1.m`*

```matlab
beta = 0.99; kappa = 0.1717; epsilon = 9; vartheta = kappa / epsilon; H = 40;
u = [1 zeros(1, H - 1)];
% TODO: discretion paths; commitment paths using delta; price level = cumsum(pi); losses
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `ex6_1.m`*

```matlab
beta = 0.99; kappa = 0.1717; epsilon = 9; vartheta = kappa / epsilon; H = 200;
u = [1 zeros(1, H - 1)];
% discretion
piD = vartheta / (kappa^2 + vartheta) * u;  xD = -(kappa / vartheta) * piD;
% commitment (timeless perspective)
a = vartheta / (vartheta * (1 + beta) + kappa^2);
delta = (1 - sqrt(1 - 4 * beta * a^2)) / (2 * a * beta);
xC = zeros(1, H); xlag = 0;
for t = 1:H, xC(t) = delta * xlag - kappa * delta / vartheta * u(t); xlag = xC(t); end
piC = -(vartheta / kappa) * [xC(1) diff(xC)];
disc = beta .^ (0:H-1);
LD = sum(disc .* (piD.^2 + vartheta * xD.^2)); LC = sum(disc .* (piC.^2 + vartheta * xC.^2));
fprintf('vartheta = %.4f, delta = %.4f\n\n', vartheta, delta);
fprintf('  h    discretion: pi      x      p      commitment: pi      x      p\n');
pD = cumsum(piD); pC = cumsum(piC);
for h = [0 1 2 4 8 16]
  fprintf('%3d            %7.3f %7.3f %6.3f              %7.3f %7.3f %6.3f\n', h, piD(h+1), xD(h+1), pD(h+1), ...
          piC(h+1), xC(h+1), pC(h+1));
end
fprintf('\ndiscounted loss: discretion %.4f, commitment %.4f (%.0f%% lower)\n', LD, LC, 100 * (1 - LC / LD));
```

Output:

```text
vartheta = 0.0191, delta = 0.3097

  h    discretion: pi      x      p      commitment: pi      x      p
  0              0.393  -3.536  0.393                0.310  -2.788  0.310
  1              0.000  -0.000  0.393               -0.214  -0.863  0.096
  2              0.000  -0.000  0.393               -0.066  -0.267  0.030
  4              0.000  -0.000  0.393               -0.006  -0.026  0.003
  8              0.000  -0.000  0.393               -0.000  -0.000  0.000
 16              0.000  -0.000  0.393               -0.000  -0.000  0.000

discounted loss: discretion 0.3929, commitment 0.3097 (21% lower)
```

Under discretion the central bank lets inflation rise on impact and closes the gap as soon as the shock has
gone: the price level stays permanently higher. Under commitment it promises to keep the output gap negative for
several quarters after the shock. That promise lowers expected future inflation, which improves the current trade-off
in the Phillips curve, so inflation rises less on impact (0.31 instead of 0.39) and then turns negative. The price level
returns towards its original path (it is stationary under commitment). The discounted loss is about a fifth lower. The gain
comes from managing expectations: commitment is history dependent, and that history dependence is exactly what
price-level and average-inflation targeting try to mimic (Lecture 7).

</details>

### Exercise 6.2 — The best simple Taylor rule (Dynare)

*Data:* Model calibration; shocks to the natural real rate (s.d. 0.5) and cost-push shocks (s.d. 0.3).

Use the three-equation model with a natural-rate shock and a cost-push shock and the rule
\(i_t=\phi_\pi\pi_t+\phi_y\tilde y_t\).

1. For each \((\phi_\pi,\phi_y)\) on a grid, compute the theoretical variances of inflation and the output gap from
   `oo_.var` and the loss \(L=\mathrm{var}(\pi)+\vartheta\,\mathrm{var}(\tilde y)\) with \(\vartheta=\kappa/9\).
2. Which rule minimises the loss? Repeat with the cost-push shock switched off (set its variance to zero in
   `M_.Sigma_e`).
3. Explain the result using the "divine coincidence".

*File `nkloss.mod`*

```matlab
// Three-equation NK model with demand (natural-rate) and cost-push shocks, simple Taylor rule
var y_gap pi i r_nat u;
varexo eps_r eps_u;
parameters beta sigma kappa phi_pi phi_y rho_r rho_u;
beta = 0.99; sigma = 1; kappa = 0.1717; phi_pi = 1.5; phi_y = 0.125; rho_r = 0.8; rho_u = 0.5;
model(linear);
  pi = beta * pi(+1) + kappa * y_gap + u;
  y_gap = y_gap(+1) - (1 / sigma) * (i - pi(+1) - r_nat);
  i = phi_pi * pi + phi_y * y_gap;
  r_nat = rho_r * r_nat(-1) + eps_r;
  u = rho_u * u(-1) + eps_u;
end;
shocks; var eps_r; stderr 0.5; var eps_u; stderr 0.3; end;
stoch_simul(order=1, irf=12, noprint, nograph) pi y_gap;
```

*File `ex6_2.m`*

```matlab
dynare nkloss noclearall
vartheta = kappa / 9;
% TODO: loop over phi_pi in [1.1 1.5 2 3 5 10] and phi_y in [0 0.125 0.5 1]
%       [info, oo_] = stoch_simul(M_, options_, oo_, {'pi'; 'y_gap'}); L = oo_.var(1,1) + vartheta*oo_.var(2,2)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `nkloss.mod`*

```matlab
// Three-equation NK model with demand (natural-rate) and cost-push shocks, simple Taylor rule
var y_gap pi i r_nat u;
varexo eps_r eps_u;
parameters beta sigma kappa phi_pi phi_y rho_r rho_u;
beta = 0.99; sigma = 1; kappa = 0.1717; phi_pi = 1.5; phi_y = 0.125; rho_r = 0.8; rho_u = 0.5;
model(linear);
  pi = beta * pi(+1) + kappa * y_gap + u;
  y_gap = y_gap(+1) - (1 / sigma) * (i - pi(+1) - r_nat);
  i = phi_pi * pi + phi_y * y_gap;
  r_nat = rho_r * r_nat(-1) + eps_r;
  u = rho_u * u(-1) + eps_u;
end;
shocks; var eps_r; stderr 0.5; var eps_u; stderr 0.3; end;
stoch_simul(order=1, irf=12, noprint, nograph) pi y_gap;
```

*File `ex6_2.m`*

```matlab
dynare nkloss noclearall
vartheta = kappa / 9;
pps = [1.1 1.5 2 3 5 10]; pys = [0 0.125 0.5 1];
for case_ = 1:2
  if case_ == 2, M_.Sigma_e(2, 2) = 0; end
  L = zeros(numel(pys), numel(pps));
  for a = 1:numel(pys)
    for b = 1:numel(pps)
      set_param_value('phi_pi', pps(b)); set_param_value('phi_y', pys(a));
      [info, oo_] = stoch_simul(M_, options_, oo_, {'pi'; 'y_gap'});
      L(a, b) = oo_.var(1, 1) + vartheta * oo_.var(2, 2);
    end
  end
  if case_ == 1, fprintf('Both shocks: loss x 100\n'); else, fprintf('\nDemand shocks only: loss x 100\n'); end
  fprintf('phi_y \\ phi_pi %s\n', sprintf('%8.1f', pps));
  for a = 1:numel(pys), fprintf('   %5.3f     %s\n', pys(a), sprintf('%8.4f', 100 * L(a, :))); end
  [m, k] = min(L(:)); [a, b] = ind2sub(size(L), k);
  fprintf('best rule on the grid: phi_pi = %.1f, phi_y = %.3f\n', pps(b), pys(a));
end
```

Output:

```text
Both shocks: loss x 100
phi_y \ phi_pi      1.1     1.5     2.0     3.0     5.0    10.0
   0.000     267.1474 98.3459 47.8303 21.5016 10.8834  7.4693
   0.125     175.5609 80.3815 43.9428 21.6390 11.2552  7.4641
   0.500      86.8519 56.4967 38.5433 23.0721 12.9926  7.8647
   1.000      59.7858 46.9272 36.8426 25.5346 15.7452  8.9823
best rule on the grid: phi_pi = 10.0, phi_y = 0.125

Demand shocks only: loss x 100
phi_y \ phi_pi      1.1     1.5     2.0     3.0     5.0    10.0
   0.000     242.7601 80.4020 34.3185 11.9684  3.6176  0.8007
   0.125     148.3452 59.6795 28.1068 10.6117  3.3830  0.7756
   0.500      54.1692 29.7915 17.0205  7.6843  2.8015  0.7071
   1.000      23.2124 15.3907 10.1374  5.3477  2.2334  0.6290
best rule on the grid: phi_pi = 10.0, phi_y = 1.000
```

With only natural-rate (demand) shocks there is no trade-off: the loss keeps falling as the rule becomes more
aggressive, whether it responds to inflation or to the output gap, and it would reach zero in the limit. That is the
"divine coincidence" (Blanchard and Galí, 2007): stabilising inflation also closes the output gap, and an aggressive
rule approximates the optimal policy of tracking the natural rate. Cost-push shocks break this. Once the rule
responds strongly to inflation, adding an output-gap response raises the loss, because stabilising the gap after a
cost-push shock means letting inflation move more, and inflation carries the larger welfare weight
(\(\vartheta\approx0.02\)). The best simple rule responds very strongly to inflation and only a little to the
output gap. Responding to the gap helps only when the inflation response is weak.

</details>

## Further reading
- Galí, J. (2015), *Monetary Policy, Inflation, and the Business Cycle*, ch. 4–5.
- Clarida, R., J. Galí and M. Gertler (1999), "The Science of Monetary Policy," *Journal of Economic Literature*.
- Woodford, M. (2003), *Interest and Prices*, ch. 6–7.
- Blanchard, O. and J. Galí (2007), "Real Wage Rigidities and the New Keynesian Model," *Journal of Money, Credit and Banking*.
- Schmitt-Grohé, S. and M. Uribe (2007), "Optimal Simple and Implementable Monetary and Fiscal Rules," *Journal of Monetary Economics*.
