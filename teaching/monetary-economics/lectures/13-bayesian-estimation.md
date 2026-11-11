# Lecture 13 — Bayesian Estimation of Monetary DSGE Models

**Level:** Graduate | **Field:** Monetary Economics

## Learning objectives
Write a linear DSGE model in state-space form, evaluate its likelihood with the Kalman filter, combine it with priors, find the posterior mode and sample from the posterior with Metropolis–Hastings. Understand identification, model comparison and the estimated medium-scale models used by central banks. Implement the procedure by hand in MATLAB and with Dynare's `estimation` command.

## 1. State-space form
The first-order solution of a DSGE model is a linear transition equation for the state vector, and the observables are linear combinations of states plus measurement error:
\[
s_t=P(\theta)s_{t-1}+Q(\theta)\varepsilon_t,\quad \varepsilon_t\sim N(0,\Sigma_\varepsilon(\theta));\qquad
Y_t=H s_t+u_t,\quad u_t\sim N(0,R).
\]
The mapping from structural parameters \(\theta\) to \(P\) and \(Q\) is computed by the QZ-based solver (Lecture 5); if the Blanchard–Kahn conditions fail, \(\theta\) is assigned zero posterior probability.

## 2. The Kalman filter likelihood
With Gaussian shocks the likelihood is computed recursively. Starting from the unconditional mean and variance of \(s_0\):
\[
\begin{aligned}
&\text{predict:} && s_{t|t-1}=Ps_{t-1|t-1},\quad V_{t|t-1}=PV_{t-1|t-1}P'+Q\Sigma_\varepsilon Q',\\
&\text{innovation:} && v_t=Y_t-Hs_{t|t-1},\quad F_t=HV_{t|t-1}H'+R,\\
&\text{update:} && K_t=V_{t|t-1}H'F_t^{-1},\quad s_{t|t}=s_{t|t-1}+K_tv_t,\quad V_{t|t}=V_{t|t-1}-K_tHV_{t|t-1},
\end{aligned}
\]
and \(\log L(\theta)=-\frac12\sum_t\left[k\log2\pi+\log|F_t|+v_t'F_t^{-1}v_t\right]\). The number of shocks plus measurement errors must be at least the number of observables to avoid stochastic singularity.

## 3. Priors and the posterior
Bayes' rule gives \(p(\theta\mid Y)\propto L(Y\mid\theta)p(\theta)\). Priors encode information from micro data and other studies and regularise weakly identified parameters. Standard choices: Beta distributions for parameters in \((0,1)\) such as autocorrelations and Calvo probabilities; Gamma for positive parameters such as the Phillips-curve slope; Normal for policy coefficients; inverse Gamma for shock standard deviations. Priors should be reported and their influence checked by comparing prior and posterior densities.

## 4. Computing the posterior
1. **Mode.** Maximise \(\log p(\theta\mid Y)\) numerically after transforming parameters to an unbounded space (logs for positive parameters, logits for probabilities). Multiple starting points guard against local maxima.
2. **Metropolis–Hastings.** Draw \(\theta'=\theta^{(j)}+c\,\Sigma^{1/2}\eta\), \(\eta\sim N(0,I)\), with \(\Sigma\) the inverse Hessian at the mode, and accept with probability \(\min\{1,p(\theta'\mid Y)/p(\theta^{(j)}\mid Y)\}\). When sampling in transformed parameters, add the log Jacobian of the transformation. Tune \(c\) for an acceptance rate of 20–35%.
3. **Diagnostics.** Run several chains, check trace plots, the Brooks–Gelman (1998) diagnostic and effective sample sizes.
4. **Model comparison.** The marginal likelihood \(p(Y)=\int L(Y\mid\theta)p(\theta)d\theta\) (computed with the modified harmonic mean, Geweke 1999) gives posterior odds between models.

## 5. Identification
A parameter is weakly identified if the likelihood is flat in its direction. Common examples in monetary models: the Taylor-rule inflation coefficient when policy has kept inflation stable (Cochrane, 2011), the Phillips-curve slope when inflation barely moves, and the separate roles of Calvo and indexation parameters. Iskrev (2010) proposes checking the rank of the Jacobian of the model's moments with respect to \(\theta\); Dynare's `identification` command implements such tests.

## 6. Estimation in Dynare
The hand-written procedure of the exercises corresponds to a few lines of Dynare:

```matlab
varobs y_gap pi i;
estimated_params;
  kappa,  gamma_pdf,  0.1,   0.05;
  phi_pi, normal_pdf, 1.5,   0.25;
  phi_y,  normal_pdf, 0.125, 0.1;
  rho_i,  beta_pdf,   0.5,   0.2;
  stderr eps_v, inv_gamma_pdf, 0.25, inf;
  ...
end;
estimation(datafile=usdata, first_obs=1, mode_compute=4, mh_replic=20000, mh_nblocks=2, mh_jscale=0.3);
```

Dynare computes the mode, the Hessian, runs the Metropolis–Hastings chains, reports diagnostics and the marginal likelihood, and produces smoothed shocks and variance decompositions.

## 7. Estimated models in practice
Smets and Wouters (2007) estimate a medium-scale model with seven shocks and seven U.S. observables; it forecasts as well as Bayesian VARs and has become a template for central-bank models (the ECB's NAWM, the Fed's EDO and FRB/US for comparison, the Bank of Korea's BOKDSGE). Estimated models attribute output fluctuations largely to demand and investment-specific shocks at business-cycle frequencies and to monetary shocks only modestly, and they imply a flat Phillips curve with large estimated price and wage rigidity. Their limitations — linearity, representative agents, a stylised financial sector — motivated the extensions in Lectures 9–12.

## Problem set
1. Derive the Kalman filter recursions from the properties of conditional Gaussian distributions.
2. Why does a model with two shocks and three observables have a singular likelihood? Give two remedies.
3. Show how a change of variables affects the target density in Metropolis–Hastings and derive the log Jacobian for a logit transformation.
4. In Exercise 13.2, which parameters are weakly identified? How would you check?
5. Explain how the marginal likelihood penalises model complexity.

## Coding exercises

These exercises use **MATLAB and Dynare only**. Download the course [data files and helper functions](../index.html#data-and-code)
into your working folder. Every data series is real: U.S. quarterly and monthly macro data (FRED-QD / FRED-MD,
1959–2025), published monetary-policy shock series (Romer–Romer, Jarociński–Karadi, Bu–Rogers–Wu,
Nakamura–Steinsson), World Bank CPI inflation for 190+ countries, and the KRW/USD exchange rate. Dynare models are
`.mod` files run with `dynare file noclearall` (Dynare 6). Try the starter files first, then open the solution. Every
printed result was produced by running the solution files as shown: the `.mod` files were processed by the Dynare
preprocessor and solved at first order, and the MATLAB code was run in GNU Octave.

### Exercise 13.1 — The posterior mode of a small New Keynesian model

*Data:* `us_quarterly.csv` — HP output gap, GDP-deflator inflation and the federal funds rate, 1984Q1–2007Q4 (96 quarters).

Estimate the three-equation New Keynesian model with interest-rate smoothing,
\(i_t=\rho_i i_{t-1}+(1-\rho_i)(\phi_\pi\pi_t+\phi_y\tilde y_t)+\varepsilon^v_t\), AR(1) natural-rate and cost-push
shocks, and \(\beta=0.99\), \(\sigma=1\) fixed.

1. Write the log posterior as a function (`nk_post.m`): solve the model with `solve_lre`, evaluate the likelihood with
   `kalman_loglik`, and add the log prior (Gamma for \(\kappa\), normal for the policy coefficients, Beta for the
   autocorrelations, uniform for the shock standard deviations).
2. Find the posterior mode with `fminsearch`, reparameterising so that every parameter is unbounded.
3. Compare the mode with the prior means. Which parameters do the data move most?

*File `nk_post.m`*

```matlab
function [lp, ll] = nk_post(th, Y)
% th = [kappa phi_pi phi_y rho_i rho_r rho_u sd_r sd_u sd_v]
% TODO: log prior; matrices A, B, C, D for y = [x pi i r u]; solve_lre; kalman_loglik with H = [eye(3) zeros(3,2)]
end
```

*File `ex13_1.m`*

```matlab
T = readtable('us_quarterly.csv');
% TODO: build Y (demeaned, 1984Q1-2007Q4); transform parameters; fminsearch on -nk_post
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `nk_post.m`*

```matlab
function [lp, ll] = nk_post(th, Y)
% NK_POST  Log posterior (and log likelihood) of the three-equation NK model with interest-rate smoothing.
%   th = [kappa phi_pi phi_y rho_i rho_r rho_u sd_r sd_u sd_v];  Y = T x 3 data [output gap, inflation, rate].
beta = 0.99; sigma = 1;
kappa = th(1); phi_pi = th(2); phi_y = th(3); rho_i = th(4); rho_r = th(5); rho_u = th(6); sd = th(7:9);
lp = -Inf; ll = -Inf;
if kappa <= 0 || any(th(4:6) <= 0) || any(th(4:6) >= 1) || any(sd <= 0) || any(sd > 5), return; end
% log prior: kappa ~ Gamma(mean 0.1, sd 0.05); phi_pi ~ N(1.5, 0.25); phi_y ~ N(0.125, 0.1);
%            rho ~ Beta(mean 0.5, sd 0.2); shock s.d. ~ Uniform(0, 5)
k = 4; s = 0.025; a = 2.625;
lprior = (k-1)*log(kappa) - kappa/s - gammaln(k) - k*log(s) ...
       - 0.5*((phi_pi - 1.5)/0.25)^2 - 0.5*((phi_y - 0.125)/0.1)^2 ...
       + sum((a-1)*log(th(4:6)) + (a-1)*log(1 - th(4:6))) - 3*(2*gammaln(a) - gammaln(2*a));
% model: y = [x pi i r u];  A y(-1) + B y + C E y(+1) + D e = 0,  e = [e_r e_u e_v]
A = zeros(5); B = zeros(5); C = zeros(5); D = zeros(5, 3);
B(1, [1 2 5]) = [kappa -1 1];               C(1, 2) = beta;                         % Phillips curve
B(2, [1 3 4]) = [-1 -1/sigma 1/sigma];      C(2, [1 2]) = [1 1/sigma];              % IS curve
B(3, [1 2 3]) = [(1-rho_i)*phi_y (1-rho_i)*phi_pi -1]; A(3, 3) = rho_i; D(3, 3) = 1; % policy rule
B(4, 4) = -1; A(4, 4) = rho_r; D(4, 1) = 1;                                         % natural rate
B(5, 5) = -1; A(5, 5) = rho_u; D(5, 2) = 1;                                         % cost-push shock
[P, Q, info] = solve_lre(A, B, C, D);
if info, return; end
H = [eye(3) zeros(3, 2)];
ll = kalman_loglik(Y, P, Q, diag(sd.^2), H, 1e-4 * eye(3));
lp = ll + lprior;
end
```

*File `ex13_1.m`*

```matlab
T = readtable('us_quarterly.csv');
x  = 100 * (log(T.rgdp) - hp_trend(log(T.rgdp), 1600));          % output gap (%)
pq = [NaN; 100 * diff(log(T.gdpdef))];                           % quarterly inflation (%)
iq = T.ffr / 4;                                                  % quarterly policy rate (%)
k = T.t >= 1984 & T.t < 2008;
Y = [x(k) pq(k) iq(k)]; Y = Y - mean(Y);                          % demeaned, 1984Q1-2007Q4
% parameters in unbounded form z: kappa = e^z1, phi_pi = 1 + e^z2, phi_y = z3, rho = logistic, sd = e^z
lg = @(z) 1 ./ (1 + exp(-z));
th = @(z) [exp(z(1)) 1+exp(z(2)) z(3) lg(z(4)) lg(z(5)) lg(z(6)) exp(z(7)) exp(z(8)) exp(z(9))];
z0 = [log(0.1) log(0.5) 0.125 0 0 0 log(0.5) log(0.2) log(0.2)];
opt = optimset('MaxFunEvals', 20000, 'MaxIter', 20000, 'TolX', 1e-7, 'TolFun', 1e-7, 'Display', 'off');
zhat = z0;
for rep = 1:3, zhat = fminsearch(@(z) -nk_post(th(z), Y), zhat, opt); end

names = {'kappa', 'phi_pi', 'phi_y', 'rho_i', 'rho_r', 'rho_u', 'sd_r', 'sd_u', 'sd_v'};
pmean = [0.1 1.5 0.125 0.5 0.5 0.5 2.5 2.5 2.5];
[lp, ll] = nk_post(th(zhat), Y); m = th(zhat);
fprintf('observations: %d; log posterior at mode %.2f, log likelihood %.2f\n\n', size(Y, 1), lp, ll);
fprintf('parameter   prior mean   posterior mode\n');
for j = 1:9, fprintf('%-8s     %7.3f       %8.4f\n', names{j}, pmean(j), m(j)); end
```

Output:

```text
observations: 96; log posterior at mode 18.91, log likelihood 25.62

parameter   prior mean   posterior mode
kappa          0.100         0.0154
phi_pi         1.500         1.5492
phi_y          0.125         0.4493
rho_i          0.500         0.8140
rho_r          0.500         0.9235
rho_u          0.500         0.5999
sd_r           2.500         0.1688
sd_u           2.500         0.0671
sd_v           2.500         0.1236
```

The data move several parameters far from their priors. Interest-rate smoothing is strong (\(\rho_i\approx0.8\)),
the response to the output gap is much larger than the prior mean, and the natural-rate shock is very persistent
(\(\rho_r\approx0.9\)), because the model needs a slow-moving force to explain the persistent output gap. The
Phillips-curve slope is pushed far below its prior mean, to about 0.015: inflation barely responded to the output gap
over 1984–2007, the flat Phillips curve of Lecture 4. The long-run inflation response stays close to its prior mean of
1.5, a sign that the data say little about it once the Taylor principle is imposed. In Dynare the same estimation needs
`varobs`, an `estimated_params` block and one command, `estimation(datafile=..., mode_compute=4, mh_replic=0)`;
writing it by hand shows what that command does.

</details>

### Exercise 13.2 — Random-walk Metropolis–Hastings

*Data:* As in Exercise 13.1.

1. Starting at the posterior mode, compute the Hessian of the log posterior (in the unbounded parameters) by finite
   differences and use \(c^2(-H)^{-1}\) with \(c=2.38/\sqrt{9}\) as the proposal covariance.
2. Run 20,000 random-walk Metropolis–Hastings draws, discard the first 5,000 and report the acceptance rate, posterior
   means and 90% credible intervals. Remember the Jacobian of the reparameterisation.
3. For which parameters is the posterior much tighter than the prior? What would you check before trusting the
   results (convergence, identification)?

*File `ex13_2.m`*

```matlab
T = readtable('us_quarterly.csv');
x  = 100 * (log(T.rgdp) - hp_trend(log(T.rgdp), 1600));          % output gap (%)
pq = [NaN; 100 * diff(log(T.gdpdef))];                           % quarterly inflation (%)
iq = T.ffr / 4;                                                  % quarterly policy rate (%)
k = T.t >= 1984 & T.t < 2008;
Y = [x(k) pq(k) iq(k)]; Y = Y - mean(Y);                          % demeaned, 1984Q1-2007Q4
% parameters in unbounded form z: kappa = e^z1, phi_pi = 1 + e^z2, phi_y = z3, rho = logistic, sd = e^z
lg = @(z) 1 ./ (1 + exp(-z));
th = @(z) [exp(z(1)) 1+exp(z(2)) z(3) lg(z(4)) lg(z(5)) lg(z(6)) exp(z(7)) exp(z(8)) exp(z(9))];
z0 = [log(0.1) log(0.5) 0.125 0 0 0 log(0.5) log(0.2) log(0.2)];
opt = optimset('MaxFunEvals', 20000, 'MaxIter', 20000, 'TolX', 1e-7, 'TolFun', 1e-7, 'Display', 'off');
zhat = z0;
for rep = 1:3, zhat = fminsearch(@(z) -nk_post(th(z), Y), zhat, opt); end

% TODO: numerical Hessian of f(z) = nk_post(th(z), Y) + log|J(z)| at zhat
% TODO: RWMH: z' = z + chol(Sigma)' * randn(9, 1); accept with probability exp(f(z') - f(z))
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `nk_post.m`*

```matlab
function [lp, ll] = nk_post(th, Y)
% NK_POST  Log posterior (and log likelihood) of the three-equation NK model with interest-rate smoothing.
%   th = [kappa phi_pi phi_y rho_i rho_r rho_u sd_r sd_u sd_v];  Y = T x 3 data [output gap, inflation, rate].
beta = 0.99; sigma = 1;
kappa = th(1); phi_pi = th(2); phi_y = th(3); rho_i = th(4); rho_r = th(5); rho_u = th(6); sd = th(7:9);
lp = -Inf; ll = -Inf;
if kappa <= 0 || any(th(4:6) <= 0) || any(th(4:6) >= 1) || any(sd <= 0) || any(sd > 5), return; end
% log prior: kappa ~ Gamma(mean 0.1, sd 0.05); phi_pi ~ N(1.5, 0.25); phi_y ~ N(0.125, 0.1);
%            rho ~ Beta(mean 0.5, sd 0.2); shock s.d. ~ Uniform(0, 5)
k = 4; s = 0.025; a = 2.625;
lprior = (k-1)*log(kappa) - kappa/s - gammaln(k) - k*log(s) ...
       - 0.5*((phi_pi - 1.5)/0.25)^2 - 0.5*((phi_y - 0.125)/0.1)^2 ...
       + sum((a-1)*log(th(4:6)) + (a-1)*log(1 - th(4:6))) - 3*(2*gammaln(a) - gammaln(2*a));
% model: y = [x pi i r u];  A y(-1) + B y + C E y(+1) + D e = 0,  e = [e_r e_u e_v]
A = zeros(5); B = zeros(5); C = zeros(5); D = zeros(5, 3);
B(1, [1 2 5]) = [kappa -1 1];               C(1, 2) = beta;                         % Phillips curve
B(2, [1 3 4]) = [-1 -1/sigma 1/sigma];      C(2, [1 2]) = [1 1/sigma];              % IS curve
B(3, [1 2 3]) = [(1-rho_i)*phi_y (1-rho_i)*phi_pi -1]; A(3, 3) = rho_i; D(3, 3) = 1; % policy rule
B(4, 4) = -1; A(4, 4) = rho_r; D(4, 1) = 1;                                         % natural rate
B(5, 5) = -1; A(5, 5) = rho_u; D(5, 2) = 1;                                         % cost-push shock
[P, Q, info] = solve_lre(A, B, C, D);
if info, return; end
H = [eye(3) zeros(3, 2)];
ll = kalman_loglik(Y, P, Q, diag(sd.^2), H, 1e-4 * eye(3));
lp = ll + lprior;
end
```

*File `ex13_2.m`*

```matlab
T = readtable('us_quarterly.csv');
x  = 100 * (log(T.rgdp) - hp_trend(log(T.rgdp), 1600));          % output gap (%)
pq = [NaN; 100 * diff(log(T.gdpdef))];                           % quarterly inflation (%)
iq = T.ffr / 4;                                                  % quarterly policy rate (%)
k = T.t >= 1984 & T.t < 2008;
Y = [x(k) pq(k) iq(k)]; Y = Y - mean(Y);                          % demeaned, 1984Q1-2007Q4
% parameters in unbounded form z: kappa = e^z1, phi_pi = 1 + e^z2, phi_y = z3, rho = logistic, sd = e^z
lg = @(z) 1 ./ (1 + exp(-z));
th = @(z) [exp(z(1)) 1+exp(z(2)) z(3) lg(z(4)) lg(z(5)) lg(z(6)) exp(z(7)) exp(z(8)) exp(z(9))];
z0 = [log(0.1) log(0.5) 0.125 0 0 0 log(0.5) log(0.2) log(0.2)];
opt = optimset('MaxFunEvals', 20000, 'MaxIter', 20000, 'TolX', 1e-7, 'TolFun', 1e-7, 'Display', 'off');
zhat = z0;
for rep = 1:3, zhat = fminsearch(@(z) -nk_post(th(z), Y), zhat, opt); end

logJ = @(z) z(1) + z(2) + sum(log(lg(z(4:6)) .* (1 - lg(z(4:6))))) + sum(z(7:9));   % log |d theta / d z|
f = @(z) nk_post(th(z), Y) + logJ(z);
n = 9; h = 1e-3; Hs = zeros(n); E = eye(n) * h; f0 = f(zhat);
for a = 1:n
  for b = a:n
    Hs(a, b) = (f(zhat + E(a,:) + E(b,:)) - f(zhat + E(a,:) - E(b,:)) - f(zhat - E(a,:) + E(b,:)) ...
              + f(zhat - E(a,:) - E(b,:))) / (4 * h^2);
    Hs(b, a) = Hs(a, b);
  end
end
Sig = (2.38 / sqrt(n))^2 * inv(-Hs); Sig = (Sig + Sig') / 2; L = chol(Sig)';
rand('seed', 1); randn('seed', 1);
N = 20000; burn = 5000; Z = zeros(N, n); z = zhat; fz = f(z); acc = 0;
for d = 1:N
  zp = z + (L * randn(n, 1))'; fp = f(zp);
  if log(rand) < fp - fz, z = zp; fz = fp; acc = acc + 1; end
  Z(d, :) = z;
end
TH = zeros(N - burn, n); for d = 1:N - burn, TH(d, :) = th(Z(burn + d, :)); end
names = {'kappa', 'phi_pi', 'phi_y', 'rho_i', 'rho_r', 'rho_u', 'sd_r', 'sd_u', 'sd_v'};
fprintf('acceptance rate: %.1f%%\n\n', 100 * acc / N);
fprintf('parameter   posterior mean    5%%      95%%\n');
for j = 1:n
  q = sort(TH(:, j)); fprintf('%-8s       %7.3f     %7.3f  %7.3f\n', names{j}, mean(q), q(round(0.05 * end)), q(round(0.95 * end)));
end
```

Output:

```text
acceptance rate: 24.6%

parameter   posterior mean    5%      95%
kappa            0.021       0.008    0.038
phi_pi           1.606       1.270    1.945
phi_y            0.452       0.361    0.544
rho_i            0.813       0.781    0.844
rho_r            0.911       0.872    0.946
rho_u            0.599       0.474    0.718
sd_r             0.182       0.148    0.220
sd_u             0.071       0.051    0.094
sd_v             0.126       0.110    0.144
```

An acceptance rate of about 25% indicates a well-tuned proposal. Most posteriors are much tighter than their priors:
the smoothing parameter, the output-gap response, the shock persistences and standard deviations, and the Phillips
slope, whose 90% interval (roughly 0.01–0.04) excludes the prior mean of 0.1. The exception is the inflation response,
whose posterior interval is about as wide as the prior: \(\phi_\pi\) is weakly identified in this model, a
well-known problem when policy has kept inflation stable (Cochrane, 2011). Before trusting results you would run
several chains from dispersed starting points and compare them (Gelman–Rubin \(\hat R\)), check trace plots and
effective sample sizes, compare prior and posterior densities for each parameter to detect weak identification
(Iskrev, 2010), and test robustness to the output-gap measure and the sample. Your exact numbers will differ slightly
because random-number generators differ across MATLAB and Octave versions.

</details>

## Further reading
- An, S. and F. Schorfheide (2007), "Bayesian Analysis of DSGE Models," *Econometric Reviews*.
- Smets, F. and R. Wouters (2007), "Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach," *American Economic Review*.
- Herbst, E. and F. Schorfheide (2016), *Bayesian Estimation of DSGE Models*, Princeton University Press.
- Iskrev, N. (2010), "Local Identification in DSGE Models," *Journal of Monetary Economics*.
- Fernández-Villaverde, J., J. Rubio-Ramírez and F. Schorfheide (2016), "Solution and Estimation Methods for DSGE Models," *Handbook of Macroeconomics*.
