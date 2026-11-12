title: Three-Equation New Keynesian Model
short: NK
when: Explain a mechanism with nominal rigidities
data: Calibrated quarterly model (Galí 2015 calibration)
lead: Use this guide when the question is about monetary policy, inflation or demand shocks, so sticky prices matter. It writes the three-equation New Keynesian model in Dynare, checks determinacy (and shows what an indeterminate model looks like), computes impulse responses and variance decompositions, and compares policy rules.
---

## The question and the model

**How does a monetary policy shock move output and inflation, and how should the central bank respond to shocks?** The three-equation New Keynesian model is the workhorse for this question. It adds monopolistic competition and Calvo price setting to the household and firm problems of the RBC model, and replaces money with an interest-rate rule. Log-linearised around a zero-inflation steady state:

\[
\begin{aligned}
x_t &= E_t x_{t+1} - \tfrac{1}{\sigma}\left(i_t - E_t \pi_{t+1} - r^n_t\right) && \text{dynamic IS curve}\\
\pi_t &= \beta E_t \pi_{t+1} + \kappa x_t + u_t && \text{New Keynesian Phillips curve}\\
i_t &= \rho_i i_{t-1} + (1-\rho_i)(\phi_\pi \pi_t + \phi_x x_t) + v_t && \text{Taylor rule}
\end{aligned}
\]

with \(\kappa = \frac{(1-\theta)(1-\beta\theta)}{\theta}(\sigma + \varphi)\). Here \(x_t\) is the output gap, \(\pi_t\) inflation, \(i_t\) the nominal rate, \(r^n_t\) the natural rate of interest (a demand shock), \(u_t\) a cost-push shock and \(v_t\) a monetary policy shock.

| Parameter | Value | Meaning |
|---|---|---|
| \(\beta\) | 0.99 | Discount factor |
| \(\sigma\) | 1 | Inverse intertemporal elasticity of substitution |
| \(\varphi\) | 1 | Inverse Frisch elasticity |
| \(\theta\) | 0.75 | Calvo probability of not resetting prices (average price duration of 4 quarters) |
| \(\phi_\pi\), \(\phi_x\) | 1.5, 0.125 | Taylor-rule responses to inflation and the output gap |
| \(\rho_i\) | 0.8 | Interest-rate smoothing |

<div class="read" markdown="1">
**Why the model is written in gaps and deviations**

Every variable is a deviation from a zero-inflation steady state, so the steady state is zero and no `steady_state_model` is needed. The economics lies in the coefficients: \(\kappa\) summarises how strongly inflation reacts to the output gap, and it falls quickly as prices become stickier.
</div>

## Write the model in Dynare

```dynare nk.mod to="COEFFICIENTS OF AUTOCORRELATION"
// Three-equation New Keynesian model. Inflation and interest rates are quarterly rates in percent;
// the reports below annualise them.
var x pi i rn u v;
varexo e_rn e_u e_v;
parameters beta sigma varphi theta kappa phi_pi phi_x rho_i rho_rn rho_u;

beta = 0.99;  sigma = 1;  varphi = 1;  theta = 0.75;
kappa = (1 - theta) * (1 - beta*theta) / theta * (sigma + varphi);
phi_pi = 1.5;  phi_x = 0.125;  rho_i = 0.8;
rho_rn = 0.8;  rho_u = 0.5;

model(linear);
  x  = x(+1) - (1/sigma) * (i - pi(+1) - rn);                       // dynamic IS curve
  pi = beta * pi(+1) + kappa * x + u;                                // Phillips curve
  i  = rho_i * i(-1) + (1 - rho_i) * (phi_pi*pi + phi_x*x) + v;      // Taylor rule
  rn = rho_rn * rn(-1) + e_rn;                                       // natural-rate (demand) shock
  u  = rho_u * u(-1) + e_u;                                          // cost-push shock
  v  = e_v;                                                          // monetary policy shock
end;

shocks;
  var e_rn; stderr 0.5;
  var e_u;  stderr 0.2;
  var e_v;  stderr 0.25;
end;

steady;
check;
stoch_simul(order=1, irf=20) x pi i;
```

<div class="read" markdown="1">
**How to read it**

- **`model(linear)`** tells Dynare the equations are already linear, so it skips the linearisation.
- **The check.** The model has two forward-looking variables (the output gap and inflation) and two eigenvalues outside the unit circle: the Blanchard–Kahn conditions hold, and the equilibrium is unique.
- **Variance decomposition.** In this calibration demand shocks dominate both the output gap (75%) and inflation (69%); cost-push shocks explain about 14% of inflation and monetary shocks under a fifth of each. With different shock variances these shares change: they are a property of the calibration, not a finding.
</div>

## Impulse responses

```octave
shocks = {'e_v', 'e_rn', 'e_u'};
titles = {'Monetary policy shock (+25 bp)', 'Demand shock', 'Cost-push shock'};
f = figure('position', [0 0 900 290]);
for s = 1:3
  subplot(1, 3, s);
  plot(0:19, oo_.irfs.(['x_' shocks{s}])); hold on;
  plot(0:19, 4 * oo_.irfs.(['pi_' shocks{s}]), '--');
  plot(0:19, 4 * oo_.irfs.(['i_' shocks{s}]), ':');
  plot([0 19], [0 0], 'k-', 'linewidth', 0.5); title(titles{s}); grid on; xlim([0 19]);
  if s == 1, legend('Output gap', 'Inflation (annual)', 'Interest rate (annual)', 'location', 'southeast'); end
end
print(f, '-dpng', '-r110', 'irf.png');
fprintf('Monetary shock, impact: output gap %.2f%%, annual inflation %.2f pp, annual rate %.2f pp\n', ...
        oo_.irfs.x_e_v(1), 4*oo_.irfs.pi_e_v(1), 4*oo_.irfs.i_e_v(1));
```

<div class="read" markdown="1">
**How to read it**

- **Monetary policy shock.** A surprise increase in the policy rate lowers the output gap and inflation on impact. Because inflation and the gap fall, the Taylor rule partly offsets the shock, so the interest rate rises by less than the shock itself.
- **Demand shock.** A rise in the natural rate raises the output gap and inflation together; the central bank raises the rate. Demand shocks move output and inflation in the **same** direction.
- **Cost-push shock.** Inflation rises while the output gap falls: the central bank faces a **trade-off**, the reason inflation targeting is hard in practice.
</div>

## Comparing policy rules

```octave
par = @(n) find(strcmp(cellstr(M_.param_names), n));
rules = [1.5 0.125; 3.0 0.125; 1.5 0.5];
fprintf('%-26s %10s %10s %10s\n', 'Rule (phi_pi, phi_x)', 'sd(x)', 'sd(pi) ann', 'sd(i) ann');
options_.noprint = 1; options_.nograph = 1;
for k = 1:size(rules, 1)
  M_.params(par('phi_pi')) = rules(k, 1); M_.params(par('phi_x')) = rules(k, 2);
  [info, oo_, options_] = stoch_simul(M_, options_, oo_, {'x', 'pi', 'i'});
  sd = sqrt(diag(oo_.var));
  fprintf('%-26s %10.3f %10.3f %10.3f\n', sprintf('(%.2f, %.3f)', rules(k, :)), sd(1), 4*sd(2), 4*sd(3));
end
M_.params(par('phi_pi')) = 1.5; M_.params(par('phi_x')) = 0.125;
```

<div class="read" markdown="1">
**How to read it**

Doubling the response to inflation lowers the volatility of **both** inflation and the output gap. That is because demand shocks dominate in this calibration, and for demand shocks there is no trade-off: stabilising inflation also stabilises the gap (the “divine coincidence”). A stronger response to the output gap also lowers both, by less for inflation. The policy rate becomes more volatile in both cases. A trade-off appears only through cost-push shocks; with larger cost-push shocks, the ranking of rules depends on how society weighs the two goals, which is the subject of the optimal-policy guide.
</div>

```octave hide
ex = struct();
ex.n_explosive = sum(abs(oo_.dr.eigval) > 1 + 1e-6); ex.n_forward = M_.nfwrd + M_.nboth;
[info, oo_, options_] = stoch_simul(M_, options_, oo_, {'x', 'pi', 'i'});
ex.irf_v = struct('x', oo_.irfs.x_e_v, 'pi', 4*oo_.irfs.pi_e_v, 'i', 4*oo_.irfs.i_e_v);
ex.irf_u = struct('x', oo_.irfs.x_e_u, 'pi', 4*oo_.irfs.pi_e_u, 'i', 4*oo_.irfs.i_e_u);
ex.vardec = oo_.variance_decomposition; ex.kappa = M_.params(par('kappa'));
fid = fopen(fullfile(export_dir, 'nk.json'), 'w'); fputs(fid, jsonencode(ex)); fclose(fid);
```

## Determinacy: the Taylor principle

```dynare nk_passive.mod norun
var x pi i rn u v;
varexo e_rn e_u e_v;
parameters beta sigma varphi theta kappa phi_pi phi_x rho_i rho_rn rho_u;
beta = 0.99;  sigma = 1;  varphi = 1;  theta = 0.75;
kappa = (1 - theta) * (1 - beta*theta) / theta * (sigma + varphi);
phi_pi = 0.9;  phi_x = 0.125;  rho_i = 0.8;  rho_rn = 0.8;  rho_u = 0.5;     // passive policy: phi_pi < 1
model(linear);
  x  = x(+1) - (1/sigma) * (i - pi(+1) - rn);
  pi = beta * pi(+1) + kappa * x + u;
  i  = rho_i * i(-1) + (1 - rho_i) * (phi_pi*pi + phi_x*x) + v;
  rn = rho_rn * rn(-1) + e_rn;  u = rho_u * u(-1) + e_u;  v = e_v;
end;
steady;
check;
```

```octave
try
  dynare nk_passive noclearall nograph nolog
catch err
  disp('Dynare stopped with the message:'); disp(err.message);
end
```

<div class="read" markdown="1">
**How to read it**

Dynare reports one eigenvalue outside the unit circle for two forward-looking variables: “The rank condition ISN'T verified!”. With \(\phi_\pi = 0.9\), the central bank raises the nominal rate by less than one-for-one with inflation, so the **real** rate falls when inflation rises. Self-fulfilling expectations of inflation are then possible, and the model has many equilibria (it is **indeterminate**). This is the **Taylor principle**: the response to inflation must exceed one (with a small output-gap response, roughly \(\phi_\pi > 1\)). When you see this Blanchard–Kahn error in your own model, check the policy rule before anything else.
</div>

## Write the paper

<div class="write" markdown="1">
**How to write it: results**

Figure 1 shows the responses to monetary, demand and cost-push shocks in the calibrated model. A 25-basis-point monetary tightening lowers the output gap and annualised inflation on impact, and its effects dissipate within a few quarters. Demand shocks move output and inflation in the same direction, while cost-push shocks create a trade-off between stabilising inflation and stabilising the output gap. The model is determinate under the baseline rule; with a response to inflation below one, the equilibrium is indeterminate, as implied by the Taylor principle.
</div>

<div class="warn" markdown="1">
**Common mistakes**

- Mixing quarterly and annual rates. Write the model in quarterly rates and annualise only when reporting.
- Reporting impulse responses from an indeterminate model. Dynare refuses to solve it; do not “fix” this by changing timing until it runs.
- Treating the variance decomposition of a calibrated model as evidence. It reflects the assumed shock variances.
- Forgetting interest-rate smoothing, which makes the response of the policy rate gradual, as in the data.
</div>
