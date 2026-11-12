title: Real Business Cycle Model
short: RBC
when: Explain a mechanism, no nominal rigidities
data: Calibrated to the U.S. economy; moments compared with U.S. data, 1966–2007
lead: Use this guide when the question is about real mechanisms (productivity, capital accumulation, labour supply) and money and price rigidities play no role. It builds the canonical real business cycle model in Dynare from equations to steady state, checks the solution, computes impulse responses and compares the model's business-cycle moments with U.S. data.
---

## The question and the model

**Can productivity shocks alone generate business cycles that look like the data?** That was the question of Kydland and Prescott (1982), and the real business cycle (RBC) model is still the starting point of every DSGE model: the New Keynesian models in the later guides add frictions to exactly this core.

A representative household chooses consumption \(C_t\), hours \(H_t\) and investment to maximise

\[
E_0 \sum_{t=0}^{\infty} \beta^t \left[ \ln C_t - \psi \frac{H_t^{1+\varphi}}{1+\varphi} \right]
\]

subject to \(C_t + I_t = Y_t\), \(K_{t+1} = (1-\delta) K_t + I_t\) and production \(Y_t = A_t K_t^{\alpha} H_t^{1-\alpha}\), with productivity \(\ln A_t = \rho \ln A_{t-1} + \varepsilon_t\). The equilibrium conditions are:

| Condition | Equation |
|---|---|
| Euler equation | \(\frac{1}{C_t} = \beta E_t\left[\frac{1}{C_{t+1}}\left(\alpha \frac{Y_{t+1}}{K_{t+1}} + 1 - \delta\right)\right]\) |
| Labour supply | \(\psi H_t^{\varphi} C_t = W_t = (1-\alpha)\frac{Y_t}{H_t}\) |
| Production | \(Y_t = A_t K_t^{\alpha} H_t^{1-\alpha}\) |
| Capital accumulation | \(K_{t+1} = (1-\delta)K_t + I_t\) |
| Resource constraint | \(Y_t = C_t + I_t\) |

<div class="read" markdown="1">
**Before writing any code**

Write the model down completely: every agent's problem, every constraint, the first-order conditions and the market-clearing conditions. Count equations and endogenous variables: they must be equal. Most Dynare errors are mistakes in this step, not in the code.
</div>

## Calibration

| Parameter | Value | Source or target |
|---|---|---|
| \(\beta\) | 0.99 | Annual real interest rate of about 4% |
| \(\alpha\) | 0.33 | Capital share of income |
| \(\delta\) | 0.025 | Annual depreciation of about 10% |
| \(\varphi\) | 1 | Frisch elasticity of labour supply of 1 |
| \(\psi\) | set in the steady state | Hours equal to 1/3 of the time endowment |
| \(\rho\) | 0.95 | Persistence of measured productivity (Solow residual) |
| \(\sigma_\varepsilon\) | 0.007 | Volatility of the Solow residual |

<div class="write" markdown="1">
**How to write it: calibration**

The model is calibrated at a quarterly frequency to the U.S. economy. The discount factor of 0.99 implies an annual real interest rate of about 4 percent, the capital share is 0.33, and the depreciation rate of 0.025 corresponds to 10 percent per year. The labour-disutility parameter is chosen so that households work one third of their time in the steady state. The productivity process has persistence 0.95 and innovation standard deviation 0.007, consistent with estimates for the Solow residual.
</div>

## Write the model in Dynare

```dynare rbc.mod to="COEFFICIENTS OF AUTOCORRELATION"
// Real business cycle model. Variables are in logs, so impulse responses are percentage deviations.
var y c i k h w r a;
varexo e;
parameters beta alpha delta phi psi rho sigma_e h_ss;

beta = 0.99;  alpha = 0.33;  delta = 0.025;  phi = 1;
rho = 0.95;   sigma_e = 0.007;  h_ss = 1/3;

model;
  exp(-c) = beta * exp(-c(+1)) * (alpha * exp(y(+1) - k) + 1 - delta);    // Euler equation (k is K_{t+1})
  psi * exp(phi*h) * exp(c) = exp(w);                                     // labour supply
  exp(w) = (1 - alpha) * exp(y - h);                                      // wage = marginal product of labour
  exp(r) = alpha * exp(y - k(-1));                                        // rental rate = marginal product of capital
  exp(y) = exp(a) * exp(alpha*k(-1)) * exp((1 - alpha)*h);                // production
  exp(k) = (1 - delta) * exp(k(-1)) + exp(i);                             // capital accumulation
  exp(y) = exp(c) + exp(i);                                               // resource constraint
  a = rho * a(-1) + e;                                                    // productivity
end;

steady_state_model;
  // analytical steady state: hours are fixed at h_ss and psi is backed out
  r = log(1/beta - 1 + delta);
  KY = alpha / exp(r);
  k_h = KY^(1/(1 - alpha));
  h = log(h_ss);
  k = log(k_h * h_ss);
  y = log(k_h^alpha * h_ss);
  i = log(delta * exp(k));
  c = log(exp(y) - exp(i));
  w = log((1 - alpha) * exp(y) / h_ss);
  psi = exp(w) / (h_ss^phi * exp(c));
  a = 0;
end;

shocks;
  var e; stderr sigma_e;
end;

steady;
check;
stoch_simul(order=1, irf=40, hp_filter=1600, periods=0) y c i h w;
```

<div class="read" markdown="1">
**How to read it**

- **Blocks.** `var` declares the endogenous variables, `varexo` the shocks, `parameters` the parameters. The `model` block contains one equation per endogenous variable. Variables dated `(+1)` are expectations, `(-1)` lags. Capital `k` is the end-of-period stock, so production uses `k(-1)`.
- **Logs.** Writing `exp(y)` wherever \(Y\) appears makes `y` the log of output, so all results are percentage deviations from the steady state.
- **`steady_state_model`.** The steady state is computed analytically, which is faster and more reliable than a numerical solver. The parameter `psi` is set here so that hours equal 1/3: a standard way to calibrate a parameter to a target.
- **`steady` and `check`** compute the steady state and verify the Blanchard–Kahn conditions. **`stoch_simul`** solves the model by a first-order perturbation and reports moments of HP-filtered variables (`hp_filter=1600`), comparable with HP-filtered data.
</div>

## Check the steady state and the solution

```octave
names = cellstr(M_.endo_names);
fprintf('%-6s %10s %10s\n', 'var', 'log', 'level');
for j = 1:numel(names)
  fprintf('%-6s %10.4f %10.4f\n', names{j}, oo_.steady_state(j), exp(oo_.steady_state(j)));
end
fprintf('\nK/Y (quarterly) = %.2f   I/Y = %.3f   C/Y = %.3f   psi = %.3f\n', ...
  exp(oo_.steady_state(4) - oo_.steady_state(1)), exp(oo_.steady_state(3) - oo_.steady_state(1)), ...
  exp(oo_.steady_state(2) - oo_.steady_state(1)), M_.params(strcmp(cellstr(M_.param_names), 'psi')));
ev = abs(oo_.dr.eigval);
fprintf('Eigenvalues larger than 1 in modulus: %d   forward-looking variables: %d\n', sum(ev > 1 + 1e-6), M_.nfwrd + M_.nboth);
```

<div class="read" markdown="1">
**How to read it**

- **Steady state.** A capital–output ratio of about 9.4 (quarterly, so about 2.4 annually), an investment share of about 24% and a consumption share of about 76% are close to the long-run ratios of the U.S. economy. Check these ratios in every model before going further: a wrong steady state invalidates everything after it.
- **Blanchard–Kahn condition.** A unique stable solution exists when the number of eigenvalues outside the unit circle equals the number of forward-looking variables. Here they match, so the model is determinate. `check` stops with an error when they do not; the New Keynesian guide shows such a failure.
</div>

## Impulse responses to a productivity shock

```octave
vars = {'y', 'c', 'i', 'h', 'w'};
labels = {'Output', 'Consumption', 'Investment', 'Hours', 'Real wage'};
f = figure('position', [0 0 860 330]);
for j = 1:5
  subplot(2, 3, j);
  plot(0:39, 100 * oo_.irfs.([vars{j} '_e'])); hold on; plot([0 39], [0 0], 'k-', 'linewidth', 0.5);
  title(labels{j}); xlim([0 39]); grid on;
end
par = @(n) M_.params(strcmp(cellstr(M_.param_names), n));
subplot(2, 3, 6); plot(0:39, 100 * par('sigma_e') * par('rho').^(0:39)); title('Productivity'); xlim([0 39]); grid on;
print(f, '-dpng', '-r110', 'irf.png');
fprintf('Impact responses (percent): output %.2f, consumption %.2f, investment %.2f, hours %.2f\n', ...
  100*oo_.irfs.y_e(1), 100*oo_.irfs.c_e(1), 100*oo_.irfs.i_e(1), 100*oo_.irfs.h_e(1));
```

<div class="read" markdown="1">
**How to read it**

A one-standard-deviation productivity shock (0.7%) raises output on impact by more than the shock itself, because hours and capital respond. **Investment moves far more than consumption**: households smooth consumption and channel the temporary windfall into capital. Hours rise because the wage is temporarily high (intertemporal substitution). These are the qualitative predictions to compare with the data.
</div>

## Compare the model with the data

```octave
data = csvread('us_levels.csv', 1, 1);           % log output, consumption, investment, 1966Q1-2007Q4
[~, cyc] = sample_hp_filter(data, 1600);         % Dynare's HP filter
sd_data = 100 * std(cyc);
cm = corrcoef(cyc); corr_data = cm(1, :);
sd_model = 100 * sqrt(diag(oo_.var(1:3, 1:3)))';
corr_model = oo_.var(1, 1:3) ./ sqrt(oo_.var(1, 1) * diag(oo_.var(1:3, 1:3))');
names = {'Output', 'Consumption', 'Investment'};
fprintf('%-12s %10s %10s %12s %12s\n', '', 'sd data', 'sd model', 'corr(y) data', 'corr(y) model');
for j = 1:3
  fprintf('%-12s %10.2f %10.2f %12.2f %12.2f\n', names{j}, sd_data(j), sd_model(j), corr_data(j), corr_model(j));
end
fprintf('\nRelative volatility (sd / sd of output): consumption data %.2f model %.2f; investment data %.2f model %.2f\n', ...
  sd_data(2)/sd_data(1), sd_model(2)/sd_model(1), sd_data(3)/sd_data(1), sd_model(3)/sd_model(1));
```

<div class="read" markdown="1">
**How to read it**

This is the classic RBC table. The model is judged on second moments it was **not** calibrated to match.

- **Output volatility.** The model produces about 80% of the observed volatility of output (1.20% against 1.49%) with productivity shocks alone: the original RBC finding.
- **Relative volatility.** As in the data, investment is several times more volatile than output (3.3 times in the model, 4.3 in the data) and consumption less volatile. The model gets the ranking right.
- **Comovement.** Consumption and investment are strongly procyclical in both.
- **Failures.** The model's consumption is far too smooth (a relative volatility of 0.34 against 0.82 in the data) and its correlations too close to one: with a single shock, everything moves together. These failures motivate the extra shocks and frictions of the medium-scale model.
</div>

```octave hide
ex = struct();
ex.ss_names = cellstr(M_.endo_names)'; ex.ss = oo_.steady_state';
ex.n_explosive = sum(abs(oo_.dr.eigval) > 1 + 1e-6); ex.n_forward = M_.nfwrd + M_.nboth;
ex.irf = struct('y', 100*oo_.irfs.y_e, 'c', 100*oo_.irfs.c_e, 'i', 100*oo_.irfs.i_e, 'h', 100*oo_.irfs.h_e);
ex.sd_data = sd_data; ex.sd_model = sd_model; ex.corr_data = corr_data; ex.corr_model = corr_model;
fid = fopen(fullfile(export_dir, 'rbc.json'), 'w'); fputs(fid, jsonencode(ex)); fclose(fid);
```

## Write the paper

<div class="write" markdown="1">
**How to write it: results**

Figure 1 shows the responses to a one-standard-deviation productivity shock. Output rises on impact by more than productivity itself, as hours and, gradually, the capital stock increase; investment responds several times more strongly than consumption, reflecting households' desire to smooth consumption. Table 2 compares the model's business-cycle moments with HP-filtered U.S. data for 1966–2007. Productivity shocks alone account for most of the observed volatility of output and reproduce the ranking of volatilities: investment is far more volatile than output, and consumption less. The model, however, generates too little volatility in consumption and too much comovement among the aggregates.
</div>

<div class="warn" markdown="1">
**Common mistakes**

- Timing of capital: Dynare uses end-of-period stocks, so production uses `k(-1)`. Getting this wrong produces Blanchard–Kahn errors.
- Comparing model moments with unfiltered data, or filtering one and not the other.
- Calibrating a parameter to a moment and then claiming the model “explains” that moment.
- Solving for the steady state numerically when it has a closed form: a wrong initial value can give a wrong steady state without an error message.
</div>
