# Lecture 12 — HANK and Monetary Transmission

**Level:** Graduate | **Field:** Monetary Economics

## Learning objectives
Understand how household heterogeneity changes the transmission of monetary policy. Decompose the consumption response into direct and indirect effects, derive the two-agent (TANK) model analytically, and see why high marginal propensities to consume (MPCs) and the distribution of liquid wealth are central to Heterogeneous-Agent New Keynesian (HANK) models.

## 1. Direct and indirect effects
In the representative-agent model (RANK), consumption responds to interest rates almost entirely through intertemporal substitution: \(c_t=E_tc_{t+1}-\frac1\sigma r_t\). Kaplan, Moll and Violante (2018) decompose the aggregate consumption response to a rate cut into a **direct effect** — the response to the change in interest rates holding income fixed — and **indirect effects** that work through the general-equilibrium rise in labour income, profits and transfers. In RANK the direct effect accounts for about 95% of the response. In their two-asset HANK model it accounts for less than a third; the rest comes from households with high MPCs spending the extra income.

## 2. The two-agent New Keynesian model
Bilbiie (2008, 2020) gives an analytical benchmark. A share \(\lambda\) of households are hand-to-mouth (H) and consume their income each period; the rest are savers (S) who satisfy the Euler equation. If hand-to-mouth income moves with aggregate income with elasticity \(\chi\), \(c^H_t=\chi y_t\), then market clearing \(y_t=\lambda c^H_t+(1-\lambda)c^S_t\) gives \(y_t=\frac{1-\lambda}{1-\lambda\chi}c^S_t\), and the aggregate IS curve becomes
\[
\tilde y_t=E_t\tilde y_{t+1}-\frac{\delta}{\sigma}\left(i_t-E_t\pi_{t+1}-r^n_t\right),\qquad \delta=\frac{1-\lambda}{1-\lambda\chi}.
\]
- If \(\chi>1\) (hand-to-mouth income is more cyclical than average, inequality is countercyclical), \(\delta>1\): heterogeneity **amplifies** monetary policy.
- If \(\chi<1\) (for example, profits accrue to savers or transfers cushion the poor), heterogeneity **dampens** it.
- If \(\lambda\chi\to1\) the multiplier explodes; for \(\lambda\chi>1\) the IS logic is inverted and determinacy requires different policy rules.

With idiosyncratic risk and transitions between types, the Euler equation also acquires a discount factor \(M\neq1\) on expected future output, giving the "discounted" Euler equation of Lecture 10 when \(M<1\) and resolving the forward-guidance puzzle.

## 3. The household block of a HANK model
The core of a HANK model is an income-fluctuation problem:
\[
V(a,z)=\max_{c,a'}\ u(c)+\beta E\left[V(a',z')\mid z\right]\quad\text{s.t.}\quad c+a'=(1+r)a+wz,\ a'\ge\underline a ,
\]
with idiosyncratic productivity \(z\). It is solved efficiently by the **endogenous grid method** (Carroll, 2006): invert the Euler equation \(u'(c)=\beta(1+r)E[u'(c')]\) on a grid of next-period assets. The stationary distribution follows from iterating the transition of the histogram (Young, 2010). The key outputs for monetary policy are the distribution of MPCs and the sensitivity of aggregate saving to \(r\).

**The MPC puzzle.** Empirical quarterly MPCs out of small windfalls are around 0.15–0.25 (Johnson, Parker and Souleles, 2006; Parker et al., 2013; Fagereng, Holm and Natvik, 2021). One-asset models match this only if households are poor on average, at odds with total wealth in the data. **Two-asset models** distinguish a low-return liquid asset from a high-return illiquid asset with adjustment costs. Many households are "wealthy hand-to-mouth" (Kaplan, Violante and Weidner, 2014): they own housing or retirement wealth but little liquid wealth, and they have high MPCs.

## 4. Solving HANK models
Modern methods linearise the aggregate dynamics around the stationary distribution. The **sequence-space Jacobian** method (Auclert, Bardóczy, Rognlie and Straub, 2021) computes how aggregate household outcomes respond to sequences of prices and solves for equilibrium paths with a few linear-algebra operations. Dynare 6 includes support for heterogeneous-agent blocks, and the method underlies most current HANK work. Intertemporal MPCs — how spending responds over time to an income change — summarise the household block for fiscal and monetary analysis (Auclert, Rognlie and Straub, 2024).

## 5. Distributional channels of monetary policy
Monetary policy redistributes through several channels (Auclert, 2019):
- the **earnings heterogeneity channel** — the incomes of some groups are more cyclical;
- the **income composition channel** — labour versus capital income;
- the **Fisher channel** — surprise inflation transfers wealth from nominal creditors to debtors;
- the **interest-rate exposure channel** — households with short-duration assets and long-duration liabilities gain when rates rise, and vice versa.
Because MPCs are correlated with these exposures, redistribution has aggregate effects. Patterson (2023) shows that workers whose earnings are most cyclical also have high MPCs, which amplifies shocks.

## 6. Korea
Korean households hold most wealth in housing and have high levels of debt, much of it at variable rates. This raises the cash-flow effects of monetary policy on indebted households and makes the interest-rate exposure channel especially important; two-asset HANK models calibrated to Korean balance-sheet data are a natural tool for studying it.

## Problem set
1. Derive the TANK aggregate IS curve and the expression for \(\delta\).
2. Show that \(\delta>1\) if and only if \(\chi>1\) (for \(0<\lambda<1\)). Interpret.
3. Explain why a high average MPC raises the share of indirect effects in monetary transmission.
4. Write the endogenous grid method step by step for the income-fluctuation problem.
5. Explain the four redistribution channels of Auclert (2019) and the sign of each for a typical Korean household.

## Coding exercises

These exercises use **MATLAB and Dynare only**. Download the course [data files and helper functions](../index.html#data-and-code)
into your working folder. Every data series is real: U.S. quarterly and monthly macro data (FRED-QD / FRED-MD,
1959–2025), published monetary-policy shock series (Romer–Romer, Jarociński–Karadi, Bu–Rogers–Wu,
Nakamura–Steinsson), World Bank CPI inflation for 190+ countries, and the KRW/USD exchange rate. Dynare models are
`.mod` files run with `dynare file noclearall` (Dynare 6). Try the starter files first, then open the solution. Every
printed result was produced by running the solution files as shown: the `.mod` files were processed by the Dynare
preprocessor and solved at first order, and the MATLAB code was run in GNU Octave.

### Exercise 12.1 — Monetary policy in a two-agent New Keynesian model (Dynare)

*Data:* Model calibration; 25 bp monetary-policy shock.

In Bilbiie's TANK model a share \(\lambda\) of households are hand-to-mouth and their income moves \(\chi\)-for-one
with aggregate income. The aggregate IS curve becomes \(\tilde y_t=E_t\tilde y_{t+1}-\frac{\delta}{\sigma}(i_t-E_t\pi_{t+1})\)
with \(\delta=\frac{1-\lambda}{1-\lambda\chi}\).

1. Compare the impact responses of output and inflation for \(\lambda\in\{0,0.2,0.4\}\) when \(\chi=1.5\)
   (inequality is countercyclical) and when \(\chi=0.5\) (procyclical).
2. When does heterogeneity amplify monetary policy, and when does it dampen it?
3. What happens as \(\lambda\chi\) approaches one?

*File `tank.mod`*

```matlab
// Two-agent New Keynesian model (Bilbiie, 2008, 2020): share lambda of hand-to-mouth households whose
// income moves chi-for-one with aggregate income
var y_gap pi i v;
varexo eps_v;
parameters beta sigma kappa phi_pi phi_y lambda chi rho_v;
beta = 0.99; sigma = 1; kappa = 0.1717; phi_pi = 1.5; phi_y = 0.125; lambda = 0.2; chi = 1.5; rho_v = 0.5;
model(linear);
  # delta = (1 - lambda) / (1 - lambda * chi);           // aggregate demand elasticity relative to RANK
  pi = beta * pi(+1) + kappa * y_gap;
  y_gap = y_gap(+1) - (delta / sigma) * (i - pi(+1));
  i = phi_pi * pi + phi_y * y_gap + v;
  v = rho_v * v(-1) + eps_v;
end;
shocks; var eps_v; stderr 0.25; end;
stoch_simul(order=1, irf=12, noprint, nograph);
```

*File `ex12_1.m`*

```matlab
dynare tank noclearall
% TODO: loops over chi and lambda; print delta and oo_.irfs.y_gap_eps_v(1), oo_.irfs.pi_eps_v(1)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `tank.mod`*

```matlab
// Two-agent New Keynesian model (Bilbiie, 2008, 2020): share lambda of hand-to-mouth households whose
// income moves chi-for-one with aggregate income
var y_gap pi i v;
varexo eps_v;
parameters beta sigma kappa phi_pi phi_y lambda chi rho_v;
beta = 0.99; sigma = 1; kappa = 0.1717; phi_pi = 1.5; phi_y = 0.125; lambda = 0.2; chi = 1.5; rho_v = 0.5;
model(linear);
  # delta = (1 - lambda) / (1 - lambda * chi);           // aggregate demand elasticity relative to RANK
  pi = beta * pi(+1) + kappa * y_gap;
  y_gap = y_gap(+1) - (delta / sigma) * (i - pi(+1));
  i = phi_pi * pi + phi_y * y_gap + v;
  v = rho_v * v(-1) + eps_v;
end;
shocks; var eps_v; stderr 0.25; end;
stoch_simul(order=1, irf=12, noprint, nograph);
```

*File `ex12_1.m`*

```matlab
dynare tank noclearall
fprintf(' chi   lambda   delta    output gap (impact)   inflation (ann., impact)\n');
for c = [1.5 0.5]
  for l = [0 0.2 0.4]
    set_param_value('chi', c); set_param_value('lambda', l);
    [info, oo_] = stoch_simul(M_, options_, oo_, {});
    fprintf('%4.1f    %4.1f    %5.3f        %7.3f               %7.3f\n', c, l, (1 - l) / (1 - l * c), ...
            oo_.irfs.y_gap_eps_v(1), 4 * oo_.irfs.pi_eps_v(1));
  end
end
```

Output:

```text
chi   lambda   delta    output gap (impact)   inflation (ann., impact)
1.5     0.0    1.000         -0.259                -0.352
1.5     0.2    1.143         -0.277                -0.377
1.5     0.4    1.500         -0.313                -0.426
0.5     0.0    1.000         -0.259                -0.352
0.5     0.2    0.889         -0.243                -0.331
0.5     0.4    0.750         -0.221                -0.300
```

When hand-to-mouth income is more cyclical than average (\(\chi>1\)), \(\delta>1\) and monetary policy has
larger effects than in the representative-agent model: a tightening lowers aggregate income, which falls
disproportionately on households who spend everything they earn, so the indirect (income) effects multiply the
direct effect. With \(\chi<1\), for example because profits that accrue to savers are more cyclical than wages or
because transfers cushion the poor, heterogeneity dampens transmission. As \(\lambda\chi\to1\) the multiplier
explodes, and with \(\lambda\chi>1\) the IS curve slopes the wrong way ("inverted aggregate demand logic") and the
determinacy conditions change. Whether HANK amplifies or dampens monetary policy is therefore an empirical question
about the incidence of income fluctuations (Bilbiie, 2020; Patterson, 2023).

</details>

### Exercise 12.2 — Marginal propensities to consume in a heterogeneous-agent household block

*Data:* Model calibration: quarterly, CRRA \(\sigma=2\), log income AR(1) with \(\rho=0.966\), \(\sigma_e=0.13\) (annualised persistence 0.87), borrowing limit 0, real rate 1% a year.

Solve the income-fluctuation problem \(\max E\sum\beta^tc_t^{1-\sigma}/(1-\sigma)\) subject to
\(a_{t+1}=(1+r)a_t+y_t-c_t\), \(a_{t+1}\ge0\), by the endogenous grid method (Carroll, 2006), with income discretised by
the Rouwenhorst method (7 states).

1. Compute the stationary wealth distribution for \(\beta=0.985\) (high wealth) and \(\beta=0.96\) (low wealth).
2. For each, report mean assets relative to mean quarterly income, the share of households at the borrowing limit and
   the average quarterly MPC out of a one-off income gain of 1% of mean annual income.
3. Why do HANK models need many households with little liquid wealth, and how do two-asset models (Kaplan, Moll and
   Violante, 2018) square this with large average wealth?

*File `ex12_2.m`*

```matlab
sig = 2; r = 0.01/4; rho = 0.966; se = 0.13; N = 7;
% Rouwenhorst: p = (1+rho)/2; build the N x N transition matrix recursively; grid of log income +-sqrt(N-1)*sd
agrid = linspace(0, 1, 400)'.^2 * 200;     % asset grid (multiples of mean quarterly income)
% TODO: EGM iteration on the Euler equation; then the stationary distribution by iterating on the histogram
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `ex12_2.m`*

```matlab
sig = 2; r = 0.01/4; rho = 0.966; se = 0.13; N = 7;
% Rouwenhorst discretisation of log income
p = (1 + rho) / 2; Pi = [p 1-p; 1-p p];
for n = 3:N
  Z = zeros(n); Pn = p * [Pi zeros(n-1,1); Z(n,1:n-1) 0] + (1-p) * [zeros(n-1,1) Pi; 0 Z(n,1:n-1)] ...
    + (1-p) * [Z(1,1:n-1) 0; Pi zeros(n-1,1)] + p * [0 Z(1,1:n-1); zeros(n-1,1) Pi];
  Pn(2:end-1, :) = Pn(2:end-1, :) / 2; Pi = Pn;
end
sdy = se / sqrt(1 - rho^2); ly = linspace(-sqrt(N-1) * sdy, sqrt(N-1) * sdy, N);
pistat = ones(1, N) / N; for it = 1:2000, pistat = pistat * Pi; end
y = exp(ly); y = y / (pistat * y');                       % mean quarterly income = 1
agrid = linspace(0, 1, 400)'.^2 * 200; na = numel(agrid);
fprintf(' beta    mean assets / qtr income   share at limit   average MPC (qtr)\n');
for beta = [0.985 0.96]
  c = (r * agrid + y) + 0.1;                              % initial guess, na x N
  for it = 1:5000                                         % endogenous grid method
    Emu = (c.^(-sig)) * Pi';                              % E[u'(c')] at a' on the grid, na x N
    cend = (beta * (1 + r) * Emu).^(-1/sig);
    aend = (cend + agrid - y) / (1 + r);                  % current assets consistent with a' = agrid
    cn = zeros(na, N);
    for j = 1:N
      cn(:, j) = interp1(aend(:, j), cend(:, j), agrid, 'linear', 'extrap');
      lim = agrid < aend(1, j); cn(lim, j) = (1 + r) * agrid(lim) + y(j);   % borrowing constraint binds
    end
    if max(abs(cn(:) - c(:))) < 1e-9, c = cn; break; end
    c = cn;
  end
  ap = (1 + r) * agrid + y - c;                           % savings policy
  % stationary distribution (Young's lottery method)
  D = ones(na, N) / (na * N);
  for it = 1:20000
    Dn = zeros(na, N);
    for j = 1:N
      a1 = min(max(ap(:, j), agrid(1)), agrid(end));
      k = min(max(sum(agrid <= a1', 1)', 1), na - 1); w = (agrid(k+1) - a1) ./ (agrid(k+1) - agrid(k));
      for jj = 1:N
        Dn(:, jj) = Dn(:, jj) + accumarray(k, w .* D(:, j) * Pi(j, jj), [na 1]) ...
                              + accumarray(k + 1, (1 - w) .* D(:, j) * Pi(j, jj), [na 1]);
      end
    end
    if max(abs(Dn(:) - D(:))) < 1e-11, D = Dn; break; end
    D = Dn;
  end
  da = 0.04;                                               % 1% of annual income = 0.04 quarterly incomes
  mpc = zeros(na, N);
  for j = 1:N, mpc(:, j) = (interp1(agrid, c(:, j), agrid + da, 'linear', 'extrap') - c(:, j)) / da; end
  atlim = sum(D(ap <= 1e-8));
  fprintf('%5.3f          %6.2f                  %5.1f%%           %5.3f\n', beta, sum(sum(D .* agrid)), ...
          100 * atlim, sum(sum(D .* mpc)));
end
```

Output:

```text
beta    mean assets / qtr income   share at limit   average MPC (qtr)
0.985            6.99                   10.1%           0.076
0.960            0.95                   45.1%           0.273
```

With a patient household (\(\beta=0.985\)) people build a large buffer, about one in ten is at the borrowing
limit, and the average quarterly MPC is small — close to the representative-agent value. With \(\beta=0.96\) wealth is low, a large
share of households is at or near the constraint, and the average MPC is several times larger. Empirical quarterly
MPCs out of small windfalls are around 0.15–0.25, far above the representative-agent benchmark, which requires many
households with little *liquid* wealth. One-asset models can match this only by making households poor on average,
contradicting the large wealth in the data. Two-asset models resolve the tension: households hold most wealth in
illiquid assets (housing, retirement accounts) with high returns and transaction costs, so many are "wealthy hand-to-
mouth" (Kaplan, Violante and Weidner, 2014). High MPCs make the indirect, income-driven effects of monetary policy
dominate the direct intertemporal-substitution effect (Kaplan, Moll and Violante, 2018).

</details>

## Further reading
- Kaplan, G., B. Moll and G. Violante (2018), "Monetary Policy According to HANK," *American Economic Review*.
- Bilbiie, F. (2020), "The New Keynesian Cross," *Journal of Monetary Economics*.
- Auclert, A. (2019), "Monetary Policy and the Redistribution Channel," *American Economic Review*.
- Auclert, A., B. Bardóczy, M. Rognlie and L. Straub (2021), "Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models," *Econometrica*.
- Kaplan, G., G. Violante and J. Weidner (2014), "The Wealthy Hand-to-Mouth," *Brookings Papers on Economic Activity*.
