title: Medium-Scale New Keynesian Model
short: Medium-scale
when: Estimate with data: a richer model, 5 observables
data: U.S. output, consumption and investment growth, inflation and the federal funds rate, 1966Q1–2007Q4
lead: Use this guide when the question involves investment, consumption dynamics or several shocks at once, so a three-equation model is too small. It estimates a medium-scale model in the spirit of Smets and Wouters (2007), with capital, habit formation, investment adjustment costs and price indexation, on five U.S. observables, and reads the results the way a referee will.
---

## The question and the model

**Which frictions and which shocks are needed to fit the joint behaviour of output, consumption, investment, inflation and interest rates?** Christiano, Eichenbaum and Evans (2005) and Smets and Wouters (2007) showed that a New Keynesian model with a handful of real and nominal frictions can fit U.S. data about as well as a VAR. Their model is the basis of most policy models in central banks.

This guide estimates a compact version: capital accumulation with **investment adjustment costs**, consumption **habits**, Calvo prices with **indexation**, a Taylor rule, and six shocks (technology, preference, investment-specific, government spending, price mark-up and monetary policy). Wages are flexible, which keeps the model small but, as the results show, pushes the price-rigidity parameter to absorb what wage rigidity would explain.

| Equation | Log-linear form (deviations from the steady state) |
|---|---|
| Consumption Euler, with habits \(h\) | \(c_t = \frac{h}{1+h} c_{t-1} + \frac{1}{1+h} E_t c_{t+1} - \frac{1-h}{1+h}(R_t - E_t\pi_{t+1}) + b_t\) |
| Investment, adjustment cost \(\varphi_i\) | \(i_t = \frac{1}{1+\beta} i_{t-1} + \frac{\beta}{1+\beta} E_t i_{t+1} + \frac{1}{(1+\beta)\varphi_i} q_t + \varepsilon^i_t\) |
| Value of capital | \(q_t = \beta(1-\delta) E_t q_{t+1} + (1-\beta(1-\delta)) E_t r^k_{t+1} - (R_t - E_t \pi_{t+1})\) |
| Phillips curve, indexation \(\iota_p\) | \(\pi_t = \frac{\iota_p}{1+\beta\iota_p}\pi_{t-1} + \frac{\beta}{1+\beta\iota_p}E_t\pi_{t+1} + \kappa\, mc_t + \varepsilon^p_t\) |
| Labour supply (flexible wages) | \(w_t = \sigma_l l_t + \frac{1}{1-h}(c_t - h c_{t-1})\) |

<div class="read" markdown="1">
**Five observables, six shocks**

The model has more shocks than observables, so it is not stochastically singular. Each observable has a natural shock: preference shocks for consumption, investment-specific shocks for investment, government spending for output, mark-up shocks for inflation and monetary shocks for the interest rate. With fewer shocks, the model would be forced to explain everything with too few sources of variation.
</div>

## Write and estimate the model

```dynare med.mod from="ESTIMATION RESULTS" max=34
// Medium-scale New Keynesian model (Smets-Wouters type, flexible wages), estimated on demeaned U.S. data.
var y c i k q rk l w mc pi R a b ei g ep YGR CGR IGR INFL INT;
varexo e_a e_b e_i e_g e_p e_R;
parameters beta delta alpha cy iy gy h phi_i sigma_l iota_p xi_p psi_pi psi_y rho_R rho_a rho_b rho_i rho_g rho_p;

beta = 0.995;  delta = 0.025;  alpha = 0.3;  gy = 0.18;
iy = delta * alpha / (1/beta - 1 + delta);  cy = 1 - iy - gy;      // steady-state expenditure shares
h = 0.7;  phi_i = 5;  sigma_l = 2;  iota_p = 0.3;  xi_p = 0.75;
psi_pi = 1.5;  psi_y = 0.1;  rho_R = 0.8;
rho_a = 0.95;  rho_b = 0.7;  rho_i = 0.7;  rho_g = 0.9;  rho_p = 0.5;

model(linear);
  # kappa = (1 - xi_p) * (1 - beta*xi_p) / xi_p / (1 + beta*iota_p);
  c  = h/(1+h)*c(-1) + 1/(1+h)*c(+1) - (1-h)/(1+h)*(R - pi(+1)) + b;      // consumption Euler with habits
  i  = 1/(1+beta)*i(-1) + beta/(1+beta)*i(+1) + 1/((1+beta)*phi_i)*q + ei; // investment
  q  = beta*(1-delta)*q(+1) + (1 - beta*(1-delta))*rk(+1) - (R - pi(+1)); // Tobin's q
  k  = (1-delta)*k(-1) + delta*i;                                        // capital
  y  = a + alpha*k(-1) + (1-alpha)*l;                                    // production
  rk = w + l - k(-1);                                                    // cost minimisation
  mc = alpha*rk + (1-alpha)*w - a;                                       // real marginal cost
  w  = sigma_l*l + (c - h*c(-1))/(1-h);                                  // labour supply
  pi = iota_p/(1+beta*iota_p)*pi(-1) + beta/(1+beta*iota_p)*pi(+1) + kappa*mc + ep;  // Phillips curve
  y  = cy*c + iy*i + gy*g;                                               // resource constraint
  R  = rho_R*R(-1) + (1-rho_R)*(psi_pi*pi + psi_y*y) + e_R;               // Taylor rule
  a  = rho_a*a(-1) + e_a;   b = rho_b*b(-1) + e_b;   ei = rho_i*ei(-1) + e_i;
  g  = rho_g*g(-1) + e_g;   ep = rho_p*ep(-1) + e_p;
  YGR = y - y(-1);  CGR = c - c(-1);  IGR = i - i(-1);  INFL = 4*pi;  INT = 4*R;   // measurement
end;

shocks;
  var e_a; stderr 0.5;  var e_b; stderr 0.3;  var e_i; stderr 0.5;
  var e_g; stderr 1;    var e_p; stderr 0.2;  var e_R; stderr 0.2;
end;

steady;
check;

estimated_params;
  h,      beta_pdf,  0.70, 0.10;   phi_i,  gamma_pdf, 4.00, 1.50;   sigma_l, gamma_pdf, 2.00, 0.75;
  iota_p, beta_pdf,  0.50, 0.15;   xi_p,   beta_pdf,  0.66, 0.10;
  psi_pi, gamma_pdf, 1.50, 0.25;   psi_y,  gamma_pdf, 0.125, 0.05;  rho_R,  beta_pdf,  0.75, 0.10;
  rho_a,  beta_pdf,  0.50, 0.20;   rho_b,  beta_pdf,  0.50, 0.20;   rho_i,  beta_pdf,  0.50, 0.20;
  rho_g,  beta_pdf,  0.50, 0.20;   rho_p,  beta_pdf,  0.50, 0.20;
  stderr e_a, inv_gamma_pdf, 0.5, 2;  stderr e_b, inv_gamma_pdf, 0.3, 2;  stderr e_i, inv_gamma_pdf, 0.5, 2;
  stderr e_g, inv_gamma_pdf, 0.5, 2;  stderr e_p, inv_gamma_pdf, 0.2, 2;  stderr e_R, inv_gamma_pdf, 0.2, 2;
end;

varobs YGR CGR IGR INFL INT;

estimation(datafile = 'us_medium_demeaned.csv', mode_compute = 5, mh_replic = 4000, mh_nblocks = 2,
           mh_jscale = 0.25, mh_drop = 0.5, moments_varendo, nograph) y c i pi R;
```

<div class="read" markdown="1">
**How to read it**

Two short chains of 4,000 draws are run so that the page can be rebuilt quickly; for a paper, run at least 2 × 100,000 and check convergence as in the estimation guide.

- **Frictions.** Habit persistence is very high (around 0.9): consumption responds sluggishly, as in the data. The investment adjustment cost is clearly positive, which delivers the hump-shaped response of investment.
- **Price rigidity.** The Calvo parameter is estimated near 0.96, implying that prices stay fixed for about 25 quarters (six years) on average, far longer than micro evidence suggests. This is the classic symptom of a missing friction: without sticky wages, marginal cost moves too much, and the only way to fit smooth inflation is to make prices extremely sticky. Smets and Wouters add wage rigidity for exactly this reason. Report the number and say what it means.
- **Policy.** The inflation response is about 1.3 with strong smoothing (about 0.85), close to the small model.
- **Persistence.** Technology and government spending shocks are very persistent; the investment and preference shocks much less so.
</div>

## Which shocks drive which variables?

```octave
sh = cellstr(M_.exo_names)';
vn = {'y', 'c', 'i', 'pi', 'R'};
VD = oo_.PosteriorTheoreticalMoments.dsge.VarianceDecomposition.Mean;   % posterior mean over the draws
vd = zeros(5, numel(sh));
for j = 1:5, for k = 1:numel(sh), vd(j, k) = VD.(vn{j}).(sh{k}); end, end
if max(vd(:)) <= 1, vd = 100 * vd; end                   % report in percent
fprintf('%-6s', ''); fprintf('%8s', sh{:}); fprintf('\n');
for j = 1:5, fprintf('%-6s', vn{j}); fprintf('%8.1f', vd(j, :)); fprintf('\n'); end
```

<div class="read" markdown="1">
**How to read it**

The table gives, for each variable, the percentage of its unconditional variance due to each shock, averaged over the posterior draws (computed by the `moments_varendo` option). Investment is dominated by the investment-specific shock (about 70%), and through investment so is output (more than half); consumption leans on the government-spending and preference shocks; inflation is driven mainly by technology shocks, with mark-up shocks second (about 30%). Each shock has a clear role, but the investment shock explaining most of output is a familiar warning sign. Whether that is economics or a sign that the shocks are absorbing misspecification is the central debate about medium-scale models (Chari, Kehoe and McGrattan 2009). A paper should report this table and discuss it.
</div>

```octave hide
pn = cellstr(bayestopt_.name)';
pm = zeros(1, numel(pn));
for j = 1:numel(pn)
  n = pn{j};
  if isfield(oo_.posterior_mean.parameters, n), pm(j) = oo_.posterior_mean.parameters.(n);
  else, pm(j) = oo_.posterior_mean.shocks_std.(regexprep(n, '^SE_', '')); end
end
ex = struct('names', {pn}, 'prior_mean', bayestopt_.p1', 'post_mean', pm, 'vd', vd, 'shocks', {sh});
fid = fopen(fullfile(export_dir, 'medium.json'), 'w'); fputs(fid, jsonencode(ex)); fclose(fid);
```

## Write the paper

<div class="write" markdown="1">
**How to write it: results**

We estimate a medium-scale New Keynesian model with habit formation, investment adjustment costs and partially indexed Calvo prices on five U.S. observables. The posterior points to strong habit persistence and substantial investment adjustment costs, in line with Smets and Wouters (2007). The estimated degree of price stickiness is implausibly high relative to micro evidence, which we attribute to the absence of nominal wage rigidity in our specification. Investment-specific shocks account for most of the variance of investment and output, and technology shocks for most of the variance of inflation, with mark-up shocks second.
</div>

<div class="warn" markdown="1">
**Common mistakes**

- Adding shocks until the model fits, then interpreting each shock as structural.
- Ignoring implausible estimates (a Calvo parameter of 0.96): they are diagnostics of misspecification.
- Short chains in a model with twenty parameters: check convergence for every parameter.
- Comparing models by marginal data density without the same data and the same treatment of the means.
</div>
