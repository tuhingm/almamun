# Lecture 05 — The Basic New Keynesian Model

**Level:** Graduate | **Field:** Monetary Economics

## Learning objectives
Assemble the canonical three-equation New Keynesian model, define the natural rate and the output gap, derive the closed-form responses to monetary and technology shocks, and characterise determinacy under interest-rate rules. Solve and simulate the model in Dynare.

## 1. The model
Combining the household Euler equation, goods-market clearing and the Calvo Phillips curve of Lecture 4 gives (Galí, 2015, ch. 3):
\[
\begin{aligned}
\pi_t&=\beta E_t\pi_{t+1}+\kappa\tilde y_t &&\text{(NKPC)}\\
\tilde y_t&=E_t\tilde y_{t+1}-\tfrac1\sigma\left(i_t-E_t\pi_{t+1}-r^n_t\right) &&\text{(dynamic IS)}\\
i_t&=\rho+\phi_\pi\pi_t+\phi_y\tilde y_t+v_t &&\text{(Taylor rule)}
\end{aligned}
\]
The **output gap** \(\tilde y_t=y_t-y^n_t\) is output relative to the **natural level** that would prevail with flexible prices, \(y^n_t=\psi_{ya}a_t\) (Lecture 3). The **natural real rate** \(r^n_t=\rho+\sigma E_t\Delta y^n_{t+1}\) is the real rate consistent with a zero gap; with AR(1) technology, \(r^n_t=\rho-\sigma(1-\rho_a)\psi_{ya}a_t\). The model is purely forward-looking: nominal rigidities create no endogenous state variable, so all persistence comes from the shocks and from the policy rule.

## 2. Equilibrium under an AR(1) monetary shock
Let \(v_t=\rho_vv_{t-1}+\varepsilon^v_t\) and set technology to its mean. Guess \(\tilde y_t=\psi_yv_t\), \(\pi_t=\psi_\pi v_t\). Substituting into the NKPC and IS equation and matching coefficients (the method of undetermined coefficients) gives
\[
\tilde y_t=-(1-\beta\rho_v)\Lambda_vv_t,\qquad \pi_t=-\kappa\Lambda_vv_t,\qquad
\Lambda_v=\frac{1}{(1-\beta\rho_v)[\sigma(1-\rho_v)+\phi_y]+\kappa(\phi_\pi-\rho_v)} .
\]
A positive shock lowers output and inflation. The real rate rises, \(r_t=\sigma(1-\rho_v)(1-\beta\rho_v)\Lambda_vv_t>0\), but the nominal rate may rise or fall: \(i_t=\rho+[(1-\beta\rho_v)(1-\rho_v)\sigma-\rho_v\kappa]\Lambda_vv_t\). When the shock is persistent the fall in expected inflation dominates and the nominal rate *falls* — an observed cut need not mean an easing. Money supply adjusts passively to deliver the chosen rate.

## 3. Technology shocks
After a positive technology shock natural output rises. Under a Taylor rule the central bank does not cut the nominal rate enough to track the falling natural rate, so actual output rises less than natural output: the output gap and inflation turn negative and **employment falls**. Galí (1999) argues that this explains the negative short-run response of hours to identified technology shocks. Under the optimal policy (Lecture 6), which sets \(i_t=r^n_t+\phi_\pi\pi_t\), the gap is closed and employment rises with the natural level.

## 4. Determinacy
Write the system as \(E_tx_{t+1}=\mathbf A x_t+\dots\) with \(x_t=(\tilde y_t,\pi_t)'\). Both variables are non-predetermined, so a unique stable equilibrium requires both eigenvalues of \(\mathbf A\) outside the unit circle. Bullard and Mitra (2002) show this holds if and only if
\[
\kappa(\phi_\pi-1)+(1-\beta)\phi_y>0 .
\]
This is the general form of the **Taylor principle**: in the long run the nominal rate must rise more than one-for-one with a permanent increase in inflation, taking into account that a permanent rise in inflation of \(d\pi\) raises the long-run output gap by \((1-\beta)d\pi/\kappa\). Under indeterminacy, sunspot shocks and multiple responses to fundamentals are possible; Clarida, Galí and Gertler (2000) argue that pre-1979 U.S. policy was in this region (Lecture 7).

## 5. Solving the model in Dynare
The model is written directly in Dynare. Model-local variables (`#`) keep derived parameters consistent when a deep parameter changes:

```matlab
var y_gap pi i r_real r_nat y y_nat n a v;
varexo eps_a eps_v;
parameters beta sigma varphi alpha epsilon theta phi_pi phi_y rho_a rho_v;
...
model(linear);
  # kappa  = (1-theta)*(1-beta*theta)/theta*(1-alpha)/(1-alpha+alpha*epsilon)*(sigma+(varphi+alpha)/(1-alpha));
  # psi_ya = (1+varphi)/(sigma*(1-alpha)+varphi+alpha);
  pi    = beta*pi(+1) + kappa*y_gap;
  y_gap = y_gap(+1) - (1/sigma)*(i - pi(+1) - r_nat);
  i     = phi_pi*pi + phi_y*y_gap + v;
  ...
end;
stoch_simul(order=1, irf=12);
```

Dynare log-linearises (not needed here), computes the generalized Schur (QZ) decomposition, checks the Blanchard–Kahn conditions and returns the decision rules \(y_t=\bar y+g_x(y_{t-1}-\bar y)+g_u u_t\), impulse responses (`oo_.irfs`) and theoretical moments (`oo_.var`). Inside a MATLAB loop, `set_param_value` and `stoch_simul(M_, options_, oo_, var_list)` re-solve the model for new parameters without re-running the preprocessor.

## 6. Assessment
The three-equation model captures the logic of modern monetary policy — expectations, real rates and the output gap — but it cannot by itself match the hump-shaped, delayed responses in the data (Lecture 1). Medium-scale models add habit formation, investment adjustment costs, indexation and sticky wages (Christiano, Eichenbaum and Evans, 2005; Smets and Wouters, 2007) to generate inertia. Lectures 9–12 add financial frictions, the lower bound, fiscal interactions and household heterogeneity.

## Problem set
1. Derive \(\Lambda_v\) by the method of undetermined coefficients, and the implied response of the nominal rate.
2. Under what condition does the nominal rate fall after a contractionary policy shock? Interpret it.
3. Derive the natural real rate for an AR(1) technology process. Why does it fall after a positive technology shock?
4. Prove the Bullard–Mitra determinacy condition for the case \(\phi_y=0\).
5. Implement the model in Dynare and verify the closed-form IRFs numerically (Exercise 5.1).

## Coding exercises

These exercises use **MATLAB and Dynare only**. Download the course [data files and helper functions](../index.html#data-and-code)
into your working folder. Every data series is real: U.S. quarterly and monthly macro data (FRED-QD / FRED-MD,
1959–2025), published monetary-policy shock series (Romer–Romer, Jarociński–Karadi, Bu–Rogers–Wu,
Nakamura–Steinsson), World Bank CPI inflation for 190+ countries, and the KRW/USD exchange rate. Dynare models are
`.mod` files run with `dynare file noclearall` (Dynare 6). Try the starter files first, then open the solution. Every
printed result was produced by running the solution files as shown: the `.mod` files were processed by the Dynare
preprocessor and solved at first order, and the MATLAB code was run in GNU Octave.

### Exercise 5.1 — The basic New Keynesian model: Dynare versus the analytical solution

*Data:* Model calibration (Galí, 2015, ch. 3).

1. Implement Galí's baseline model in Dynare with a technology shock and an AR(1) monetary-policy shock
   (\(\rho_v=0.5\), 25 bp).
2. With an AR(1) policy shock the model has the closed-form solution \(\tilde y_t=-(1-\beta\rho_v)\Lambda_v v_t\),
   \(\pi_t=-\kappa\Lambda_v v_t\), where \(\Lambda_v=1/[(1-\beta\rho_v)(\sigma(1-\rho_v)+\phi_y)+\kappa(\phi_\pi-\rho_v)]\).
   Check Dynare's impact responses against it.
3. Report the response of the nominal rate and of employment to the policy shock and to a technology shock. Why does
   employment fall after a positive technology shock under this rule?

*File `nk.mod`*

```matlab
var y_gap pi i r_real r_nat y y_nat n a v;
varexo eps_a eps_v;
parameters beta sigma varphi alpha epsilon theta phi_pi phi_y rho_a rho_v;
beta = 0.99; sigma = 1; varphi = 5; alpha = 1/4; epsilon = 9; theta = 3/4;
phi_pi = 1.5; phi_y = 0.5/4; rho_a = 0.9; rho_v = 0.5;
model(linear);
  // TODO: kappa and psi_ya as model-local variables; NKPC, IS, Taylor rule, natural rate, production
end;
shocks; var eps_a; stderr 1; var eps_v; stderr 0.25; end;
stoch_simul(order=1, irf=12, noprint, nograph);
```

*File `ex5_1.m`*

```matlab
dynare nk noclearall
% TODO: compute kappa and Lambda_v in MATLAB and compare with oo_.irfs.y_gap_eps_v(1), oo_.irfs.pi_eps_v(1)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `nk.mod`*

```matlab
// Basic New Keynesian model (Gali, 2015, ch. 3), log-linear
var y_gap pi i r_real r_nat y y_nat n a v;
varexo eps_a eps_v;
parameters beta sigma varphi alpha epsilon theta phi_pi phi_y rho_a rho_v;
beta = 0.99; sigma = 1; varphi = 5; alpha = 1/4; epsilon = 9; theta = 3/4;
phi_pi = 1.5; phi_y = 0.5/4; rho_a = 0.9; rho_v = 0.5;
model(linear);
  # Theta  = (1 - alpha) / (1 - alpha + alpha * epsilon);
  # lambda = (1 - theta) * (1 - beta * theta) / theta * Theta;
  # kappa  = lambda * (sigma + (varphi + alpha) / (1 - alpha));
  # psi_ya = (1 + varphi) / (sigma * (1 - alpha) + varphi + alpha);
  pi = beta * pi(+1) + kappa * y_gap;                       // New Keynesian Phillips curve
  y_gap = y_gap(+1) - (1 / sigma) * (i - pi(+1) - r_nat);   // dynamic IS equation
  i = phi_pi * pi + phi_y * y_gap + v;                      // Taylor rule
  r_real = i - pi(+1);
  r_nat = -sigma * (1 - rho_a) * psi_ya * a;               // natural real rate (deviation)
  y_nat = psi_ya * a;
  y = y_nat + y_gap;
  y = a + (1 - alpha) * n;
  a = rho_a * a(-1) + eps_a;
  v = rho_v * v(-1) + eps_v;
end;
shocks; var eps_a; stderr 1; var eps_v; stderr 0.25; end;
stoch_simul(order=1, irf=12, noprint, nograph);
```

*File `ex5_1.m`*

```matlab
dynare nk noclearall
Theta = (1 - alpha) / (1 - alpha + alpha * epsilon);
kappa = (1 - theta) * (1 - beta * theta) / theta * Theta * (sigma + (varphi + alpha) / (1 - alpha));
Lv = 1 / ((1 - beta * rho_v) * (sigma * (1 - rho_v) + phi_y) + kappa * (phi_pi - rho_v));
fprintf('kappa = %.4f, Lambda_v = %.4f\n\n', kappa, Lv);
fprintf('Monetary shock (25 bp):      Dynare     closed form\n');
fprintf('  output gap, impact       %8.4f    %8.4f\n', oo_.irfs.y_gap_eps_v(1), -(1 - beta * rho_v) * Lv * 0.25);
fprintf('  inflation (ann.), impact %8.4f    %8.4f\n', 4 * oo_.irfs.pi_eps_v(1), -4 * kappa * Lv * 0.25);
fprintf('  nominal rate (ann.)      %8.4f\n', 4 * oo_.irfs.i_eps_v(1));
fprintf('  real rate (ann.)         %8.4f\n', 4 * oo_.irfs.r_real_eps_v(1));
fprintf('  employment               %8.4f\n\n', oo_.irfs.n_eps_v(1));
fprintf('Technology shock (1%%):      impact\n');
for v = {'y', 'y_nat', 'y_gap', 'n', 'pi', 'i'}
  f = [v{1} '_eps_a']; s = 1 + 3 * any(strcmp(v{1}, {'pi', 'i'}));
  fprintf('  %-8s               %8.4f\n', v{1}, s * oo_.irfs.(f)(1));
end
```

Output:

```text
kappa = 0.1717, Lambda_v = 2.0522

Monetary shock (25 bp):      Dynare     closed form
  output gap, impact        -0.2591     -0.2591
  inflation (ann.), impact  -0.3523     -0.3523
  nominal rate (ann.)        0.3420
  real rate (ann.)           0.5182
  employment                -0.3454

Technology shock (1%):      impact
  y                        0.9145
  y_nat                    1.0000
  y_gap                   -0.0855
  n                       -0.1140
  pi                      -0.5385
  i                       -0.8504
```

Dynare's impact responses match the closed-form solution to all reported digits. A 25 bp policy shock lowers the
output gap by about 0.26% and annualised inflation by about 0.35 percentage points. The nominal
rate rises by less than the shock (and the real rate by more), because the endogenous fall in inflation and output
pulls the rule back down; with a persistent shock the nominal rate can even fall. After a technology improvement,
natural output rises more than actual output because prices are sticky and the rule does not cut rates enough: the
output gap and inflation turn negative and employment falls. That is Galí's (1999) explanation for the negative
response of hours to technology shocks in the data.

</details>

### Exercise 5.2 — Mapping the determinacy region

*Data:* Model calibration.

For the model of Exercise 5.1, Bullard and Mitra (2002) show that the equilibrium is determinate if and only if
\(\kappa(\phi_\pi-1)+(1-\beta)\phi_y>0\).

1. Loop over \(\phi_\pi\in\{0.5,0.8,0.95,1.05,1.5\}\) and \(\phi_y\in\{0,0.5,1\}\) using `set_param_value`,
   record Dynare's Blanchard–Kahn status, and compare it with the analytical condition.
2. Why can a strong enough response to the output gap make the equilibrium determinate even when \(\phi_\pi<1\)?

*File `ex5_2.m`*

```matlab
dynare nk noclearall
for py = [0 0.5 1]
  for pp = [0.5 0.8 0.95 1.05 1.5]
    set_param_value('phi_pi', pp); set_param_value('phi_y', py);
    % TODO: stoch_simul, record info(1); compare with kappa*(pp-1) + (1-beta)*py > 0
  end
end
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `nk.mod`*

```matlab
// Basic New Keynesian model (Gali, 2015, ch. 3), log-linear
var y_gap pi i r_real r_nat y y_nat n a v;
varexo eps_a eps_v;
parameters beta sigma varphi alpha epsilon theta phi_pi phi_y rho_a rho_v;
beta = 0.99; sigma = 1; varphi = 5; alpha = 1/4; epsilon = 9; theta = 3/4;
phi_pi = 1.5; phi_y = 0.5/4; rho_a = 0.9; rho_v = 0.5;
model(linear);
  # Theta  = (1 - alpha) / (1 - alpha + alpha * epsilon);
  # lambda = (1 - theta) * (1 - beta * theta) / theta * Theta;
  # kappa  = lambda * (sigma + (varphi + alpha) / (1 - alpha));
  # psi_ya = (1 + varphi) / (sigma * (1 - alpha) + varphi + alpha);
  pi = beta * pi(+1) + kappa * y_gap;                       // New Keynesian Phillips curve
  y_gap = y_gap(+1) - (1 / sigma) * (i - pi(+1) - r_nat);   // dynamic IS equation
  i = phi_pi * pi + phi_y * y_gap + v;                      // Taylor rule
  r_real = i - pi(+1);
  r_nat = -sigma * (1 - rho_a) * psi_ya * a;               // natural real rate (deviation)
  y_nat = psi_ya * a;
  y = y_nat + y_gap;
  y = a + (1 - alpha) * n;
  a = rho_a * a(-1) + eps_a;
  v = rho_v * v(-1) + eps_v;
end;
shocks; var eps_a; stderr 1; var eps_v; stderr 0.25; end;
stoch_simul(order=1, irf=12, noprint, nograph);
```

*File `ex5_2.m`*

```matlab
dynare nk noclearall
Theta = (1 - alpha) / (1 - alpha + alpha * epsilon);
kappa = (1 - theta) * (1 - beta * theta) / theta * Theta * (sigma + (varphi + alpha) / (1 - alpha));
pps = [0.5 0.8 0.95 1.05 1.5];
fprintf('phi_y \\ phi_pi %s\n', sprintf('%8.2f', pps));
agree = 0;
for py = [0 0.5 1]
  row = '';
  for pp = pps
    set_param_value('phi_pi', pp); set_param_value('phi_y', py);
    [info, oo_] = stoch_simul(M_, options_, oo_, {});
    det = info(1) == 0; theory = kappa * (pp - 1) + (1 - beta) * py > 0;
    agree = agree + (det == theory);
    if det, row = [row '     det']; else, row = [row '   indet']; end
  end
  fprintf('   %4.2f      %s\n', py, row);
end
fprintf('Dynare agrees with kappa(phi_pi-1)+(1-beta)phi_y>0 in %d of 15 cases (kappa = %.4f)\n', agree, kappa);
```

Output:

```text
phi_y \ phi_pi     0.50    0.80    0.95    1.05    1.50
   0.00         indet   indet   indet     det     det
   0.50         indet   indet   indet     det     det
   1.00         indet   indet     det     det     det
Dynare agrees with kappa(phi_pi-1)+(1-beta)phi_y>0 in 15 of 15 cases (kappa = 0.1717)
```

Dynare's classification matches the analytical condition everywhere on the grid. With \(\phi_y=0\) the
Taylor principle \(\phi_\pi>1\) is necessary and sufficient. A response to the output gap helps because a sustained
rise in inflation is accompanied by a permanently higher output gap through the long-run Phillips curve
(\(\tilde y=\pi(1-\beta)/\kappa\)); an output response therefore also raises the long-run nominal rate. The general
principle is that the nominal rate must eventually rise by more than one-for-one with any permanent increase in
inflation. Because \(1-\beta\) is small, a very large output response is needed to compensate for \(\phi_\pi<1\).

</details>

## Further reading
- Galí, J. (2015), *Monetary Policy, Inflation, and the Business Cycle*, ch. 3.
- Clarida, R., J. Galí and M. Gertler (1999), "The Science of Monetary Policy: A New Keynesian Perspective," *Journal of Economic Literature*.
- Bullard, J. and K. Mitra (2002), "Learning about Monetary Policy Rules," *Journal of Monetary Economics*.
- Woodford, M. (2003), *Interest and Prices*, ch. 4.
- Adjemian, S. et al., *Dynare Reference Manual* (stoch_simul and the Blanchard–Kahn conditions).
