# Lecture 04 — Price and Wage Setting

**Level:** Graduate | **Field:** Monetary Economics

## Learning objectives
Derive the New Keynesian Phillips curve from Calvo price setting step by step, relate its slope to the frequency of price changes and to real rigidities, and extend the framework to sticky wages. Confront the model with micro evidence on price setting and macro evidence on the Phillips curve.

## 1. Monopolistic competition
A continuum of firms \(j\in[0,1]\) produce differentiated goods aggregated with a CES aggregator with elasticity \(\epsilon\). Demand for good \(j\) is \(Y_t(j)=(P_t(j)/P_t)^{-\epsilon}Y_t\), with price index \(P_t=\left(\int P_t(j)^{1-\epsilon}dj\right)^{1/(1-\epsilon)}\). With flexible prices each firm sets a constant markup \(\mathcal M=\epsilon/(\epsilon-1)\) over nominal marginal cost.

## 2. Calvo pricing
Each period a firm may reset its price with probability \(1-\theta\), independently of its history (Calvo, 1983). The expected duration of a price is \(1/(1-\theta)\). Because resetting firms are a random sample, the price index evolves as
\[
\pi_t=(1-\theta)(p^*_t-p_{t-1}),
\]
where \(p^*_t\) is the (common) log reset price. A resetting firm maximises the expected discounted profits over the period its price is in force:
\[
\max_{P^*_t}\sum_{k=0}^{\infty}\theta^kE_t\left[Q_{t,t+k}\left(P^*_tY_{t+k|t}-\Psi_{t+k}(Y_{t+k|t})\right)\right].
\]
The log-linear first-order condition is
\[
p^*_t=\mu+(1-\beta\theta)\sum_{k=0}^{\infty}(\beta\theta)^kE_t\left[\psi_{t+k|t}\right],
\]
the desired markup over a weighted average of expected future nominal marginal costs \(\psi_{t+k|t}\). Firms that reset rarely look further into the future.

## 3. The New Keynesian Phillips curve
With decreasing returns (\(\alpha>0\)), a firm's marginal cost depends on its own output, so \(\psi_{t+k|t}=\psi_{t+k}-\frac{\alpha\epsilon}{1-\alpha}(p^*_t-p_{t+k})\). Combining with the price index gives
\[
\pi_t=\beta E_t\pi_{t+1}+\lambda\,\widehat{mc}_t,\qquad \lambda=\frac{(1-\theta)(1-\beta\theta)}{\theta}\,\Theta,\quad \Theta=\frac{1-\alpha}{1-\alpha+\alpha\epsilon},
\]
where \(\widehat{mc}_t\) is real marginal cost in deviation from steady state. Using the labour-market equilibrium, marginal cost is proportional to the output gap, \(\widehat{mc}_t=\left(\sigma+\frac{\varphi+\alpha}{1-\alpha}\right)\tilde y_t\), so
\[
\boxed{\pi_t=\beta E_t\pi_{t+1}+\kappa\tilde y_t,\qquad \kappa=\lambda\left(\sigma+\frac{\varphi+\alpha}{1-\alpha}\right).}
\]
Solving forward, \(\pi_t=\kappa\sum_k\beta^kE_t\tilde y_{t+k}\): inflation reflects expected future output gaps. The factor \(\Theta<1\) is a **real rigidity** (strategic complementarity): with decreasing returns a firm that raises its price loses sales, lowers its marginal cost and so raises its price by less. Real rigidities let modest nominal rigidity generate large monetary non-neutrality (Ball and Romer, 1990).

## 4. Micro evidence and alternatives
Bils and Klenow (2004) and Nakamura and Steinsson (2008) find that U.S. consumer prices change every 4–11 months on average, depending on how temporary sales are treated, with large heterogeneity across goods. Calvo is a convenient but not literal description. **Menu-cost models** (Golosov and Lucas, 2007) imply a strong *selection effect*: the firms that change prices are those furthest from their optimum, which makes the aggregate price level much more flexible. Midrigan (2011) and Alvarez, Le Bihan and Lippi (2016) show that accounting for multi-product firms and small price changes restores sizable non-neutrality. **Rotemberg** quadratic adjustment costs give the same linear Phillips curve as Calvo to first order. **Sticky information** (Mankiw and Reis, 2002) produces delayed inflation responses.

## 5. Sticky wages
Erceg, Henderson and Levin (2000) add monopolistically competitive households who reset nominal wages with probability \(1-\theta_w\). The wage Phillips curve is
\[
\pi^w_t=\beta E_t\pi^w_{t+1}-\lambda_w\left(\omega_t-\omega^{mrs}_t\right),\qquad \lambda_w=\frac{(1-\theta_w)(1-\beta\theta_w)}{\theta_w(1+\epsilon_w\varphi)},
\]
where \(\omega_t\) is the real wage. In gap form (Galí, 2015, ch. 6) the price and wage Phillips curves are
\[
\pi^p_t=\beta E_t\pi^p_{t+1}+\kappa_p\tilde y_t+\lambda_p\tilde\omega_t,\qquad
\pi^w_t=\beta E_t\pi^w_{t+1}+\kappa_w\tilde y_t-\lambda_w\tilde\omega_t,
\]
with \(\tilde\omega_t=\tilde\omega_{t-1}+\pi^w_t-\pi^p_t-\Delta\omega^n_t\). With both prices and wages sticky, stabilising price inflation alone is no longer optimal: the central bank faces a trade-off between price inflation, wage inflation and the output gap.

## 6. The Phillips curve in the data
Reduced-form Phillips curves estimated on U.S. data have flattened since the 1980s. Two explanations compete: anchored expectations and changes in the structure of the economy (globalisation, lower trend inflation, more rigid prices at low inflation), and an identification problem. If the central bank offsets demand shocks, the observed correlation between inflation and slack weakens even when the structural slope is unchanged (McLeay and Tenreyro, 2020). Regional data, which net out common monetary policy, suggest the structural slope is small but not zero (Hazell, Herreño, Nakamura and Steinsson, 2022). The 2021–23 inflation surge revived interest in nonlinear Phillips curves that steepen when labour markets are very tight.

## Problem set
1. Derive the law of motion of the price index under Calvo pricing and the optimal reset price condition.
2. Derive the NKPC slope \(\kappa\) for \(\alpha=0\) and for \(\alpha>0\). Explain the role of \(\Theta\).
3. Show that the Rotemberg model with adjustment cost \(\frac{\phi}{2}(P_t(j)/P_{t-1}(j)-1)^2\) gives a Phillips curve of the same form, and map \(\phi\) to \(\theta\).
4. Explain the selection effect in menu-cost models and why it weakens monetary non-neutrality.
5. Why can a flat reduced-form Phillips curve coexist with a steeper structural one?

## Coding exercises

These exercises use **MATLAB and Dynare only**. Download the course [data files and helper functions](../index.html#data-and-code)
into your working folder. Every data series is real: U.S. quarterly and monthly macro data (FRED-QD / FRED-MD,
1959–2025), published monetary-policy shock series (Romer–Romer, Jarociński–Karadi, Bu–Rogers–Wu,
Nakamura–Steinsson), World Bank CPI inflation for 190+ countries, and the KRW/USD exchange rate. Dynare models are
`.mod` files run with `dynare file noclearall` (Dynare 6). Try the starter files first, then open the solution. Every
printed result was produced by running the solution files as shown: the `.mod` files were processed by the Dynare
preprocessor and solved at first order, and the MATLAB code was run in GNU Octave.

### Exercise 4.1 — From Calvo parameters to the Phillips-curve slope — and what the data say

*Data:* Model calibration; `us_quarterly.csv` — core CPI and the unemployment rate.

In the New Keynesian Phillips curve \(\pi_t=\beta E_t\pi_{t+1}+\kappa\tilde y_t\), the slope is
\(\kappa=\lambda\left(\sigma+\frac{\varphi+\alpha}{1-\alpha}\right)\) with
\(\lambda=\frac{(1-\theta)(1-\beta\theta)}{\theta}\frac{1-\alpha}{1-\alpha+\alpha\epsilon}\).

1. Compute \(\kappa\) and the average price duration \(1/(1-\theta)\) for \(\theta\in\{0.5,0.67,0.75,0.85\}\)
   (\(\beta=0.99,\sigma=1,\varphi=5,\alpha=1/4,\epsilon=6\)).
2. Estimate a reduced-form Phillips curve on U.S. data:
   \(\pi_t=c+\gamma\bar\pi_{t-1}+\delta(u_t-u^*_t)\), with \(\pi_t\) annualised quarterly core CPI inflation,
   \(\bar\pi_{t-1}\) the average of the previous four quarters, and \(u^*\) the HP trend of unemployment. Compare
   1960–1984 with 1985–2019.
3. Has the Phillips curve flattened? Give two explanations.

*File `ex4_1.m`*

```matlab
beta = 0.99; sigma = 1; varphi = 5; alpha = 1/4; epsilon = 6;
% TODO: kappa(theta) and durations
T = readtable('us_quarterly.csv');
% TODO: pi = 400*diff(log(T.corecpi)); ugap = T.unrate - hp_trend(T.unrate, 1600); ols_nw by subsample
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `ex4_1.m`*

```matlab
beta = 0.99; sigma = 1; varphi = 5; alpha = 1/4; epsilon = 6;
Theta = (1 - alpha) / (1 - alpha + alpha * epsilon);
fprintf('theta   duration (qtrs)   lambda    kappa\n');
for theta = [0.5 0.67 0.75 0.85]
  lam = (1 - theta) * (1 - beta * theta) / theta * Theta;
  kap = lam * (sigma + (varphi + alpha) / (1 - alpha));
  fprintf('%5.2f      %5.1f          %6.4f   %6.4f\n', theta, 1 / (1 - theta), lam, kap);
end
T = readtable('us_quarterly.csv');
t = T.t(2:end); p = 400 * diff(log(T.corecpi)); u = T.unrate(2:end);
ug = u - hp_trend(u, 1600);
pbar = filter(ones(4, 1) / 4, 1, [NaN; p(1:end-1)]);        % average of the previous four quarters
fprintf('\nsample      gamma (lagged infl)   delta (unemployment gap)\n');
for s = {[1960 1984.75], [1985 2019.75]}
  r = s{1}; k = t >= r(1) & t <= r(2) & ~isnan(pbar);
  [b, se] = ols_nw(p(k), [ones(sum(k), 1) pbar(k) ug(k)], 4);
  fprintf('%d-%d      %5.2f (%.2f)          %6.2f (%.2f)\n', floor(r(1)), floor(r(2)), b(2), se(2), b(3), se(3));
end
```

Output:

```text
theta   duration (qtrs)   lambda    kappa
 0.50        2.0          0.1683   1.3467
 0.67        3.0          0.0553   0.4422
 0.75        4.0          0.0286   0.2289
 0.85        6.7          0.0093   0.0746

sample      gamma (lagged infl)   delta (unemployment gap)
1960-1984       0.89 (0.06)           -1.39 (0.22)
1985-2019       0.87 (0.06)           -0.19 (0.07)
```

Longer price spells flatten the Phillips curve sharply: moving from two-quarter to roughly seven-quarter price
durations divides \(\kappa\) by more than ten, because firms that reset prices rarely put little weight on current
marginal cost. In the data, the coefficient on the unemployment gap is large and negative before 1985 and much smaller
afterwards (about a seventh as large), while the coefficient on lagged inflation stays close to 0.9 in both samples.
This backward-looking specification cannot capture a change in how expectations are formed, so two leading
explanations for the flattening lie outside it: better-anchored inflation expectations after the Volcker disinflation,
and successful monetary policy that offsets demand shocks, which makes the reduced-form slope hard to
identify even if the structural \(\kappa\) is unchanged (McLeay and Tenreyro, 2020). The 2021–23 surge suggests the
curve can steepen when slack is very low.

</details>

### Exercise 4.2 — Sticky prices and sticky wages (Dynare)

*Data:* Model calibration (Galí, 2015, ch. 6).

Extend the New Keynesian model with Calvo wage setting (Erceg, Henderson and Levin, 2000). The price and wage Phillips
curves are
\(\pi^p_t=\beta E_t\pi^p_{t+1}+\kappa_p\tilde y_t+\lambda_p\tilde\omega_t\) and
\(\pi^w_t=\beta E_t\pi^w_{t+1}+\kappa_w\tilde y_t-\lambda_w\tilde\omega_t\), where \(\tilde\omega_t\) is the
real-wage gap, which evolves as \(\tilde\omega_t=\tilde\omega_{t-1}+\pi^w_t-\pi^p_t\).

1. Write the model in Dynare using model-local variables (`#`) for \(\lambda_p,\lambda_w,\kappa_p,\kappa_w\) so they
   update when \(\theta_w\) changes.
2. Compare the response to a 25 bp monetary-policy shock for \(\theta_w=0.75\) and \(\theta_w=0.25\).
3. How do sticky wages change the response of output, price inflation and the real wage?

*File `ehl.mod`*

```matlab
var y_gap pi_p pi_w w_gap i v;
varexo eps_v;
parameters beta sigma varphi alpha eps_p eps_w theta_p theta_w phi_pi phi_y rho_v;
beta = 0.99; sigma = 1; varphi = 5; alpha = 1/4; eps_p = 6; eps_w = 4.5;
theta_p = 0.75; theta_w = 0.75; phi_pi = 1.5; phi_y = 0.125; rho_v = 0.5;
model(linear);
  # lambda_p = (1 - theta_p) * (1 - beta * theta_p) / theta_p * (1 - alpha) / (1 - alpha + alpha * eps_p);
  // TODO: lambda_w, kappa_p = alpha*lambda_p/(1-alpha), kappa_w = lambda_w*(sigma + varphi/(1-alpha))
  // TODO: two Phillips curves, wage-gap identity, IS curve, Taylor rule, AR(1) shock
end;
shocks; var eps_v; stderr 0.25; end;
stoch_simul(order=1, irf=12, noprint, nograph);
```

*File `ex4_2.m`*

```matlab
dynare ehl noclearall
% TODO: loop over theta_w with set_param_value and stoch_simul; print impact and peak responses
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `ehl.mod`*

```matlab
// Sticky prices and wages (Erceg-Henderson-Levin; Gali, 2015, ch. 6), monetary shock only
var y_gap pi_p pi_w w_gap i v;
varexo eps_v;
parameters beta sigma varphi alpha eps_p eps_w theta_p theta_w phi_pi phi_y rho_v;
beta = 0.99; sigma = 1; varphi = 5; alpha = 1/4; eps_p = 6; eps_w = 4.5;
theta_p = 0.75; theta_w = 0.75; phi_pi = 1.5; phi_y = 0.125; rho_v = 0.5;
model(linear);
  # lambda_p = (1 - theta_p) * (1 - beta * theta_p) / theta_p * (1 - alpha) / (1 - alpha + alpha * eps_p);
  # lambda_w = (1 - theta_w) * (1 - beta * theta_w) / (theta_w * (1 + eps_w * varphi));
  # kappa_p = alpha * lambda_p / (1 - alpha);
  # kappa_w = lambda_w * (sigma + varphi / (1 - alpha));
  pi_p = beta * pi_p(+1) + kappa_p * y_gap + lambda_p * w_gap;     // price Phillips curve
  pi_w = beta * pi_w(+1) + kappa_w * y_gap - lambda_w * w_gap;     // wage Phillips curve
  w_gap = w_gap(-1) + pi_w - pi_p;                                 // real-wage gap
  y_gap = y_gap(+1) - (1 / sigma) * (i - pi_p(+1));                // dynamic IS (r^n constant)
  i = phi_pi * pi_p + phi_y * y_gap + v;                           // Taylor rule
  v = rho_v * v(-1) + eps_v;
end;
shocks; var eps_v; stderr 0.25; end;
stoch_simul(order=1, irf=12, noprint, nograph);
```

*File `ex4_2.m`*

```matlab
dynare ehl noclearall
fprintf('theta_w   output gap: impact  cum(0-7)   price infl (ann.): impact   real wage gap: impact\n');
for tw = [0.75 0.25]
  set_param_value('theta_w', tw);
  [info, oo_] = stoch_simul(M_, options_, oo_, {});
  ir = oo_.irfs;
  fprintf('  %.2f            %7.3f   %7.3f              %7.3f               %7.3f\n', tw, ir.y_gap_eps_v(1), ...
          sum(ir.y_gap_eps_v(1:8)), 4 * ir.pi_p_eps_v(1), ir.w_gap_eps_v(1));
end
```

Output:

```text
theta_w   output gap: impact  cum(0-7)   price infl (ann.): impact   real wage gap: impact
  0.75             -0.382    -0.740               -0.040                -0.010
  0.25             -0.322    -0.546               -0.163                -0.235
```

With sticky wages (\(\theta_w=0.75\)) the contraction in output is larger and more persistent, while price
inflation falls by less. Firms' marginal cost depends on the real wage; when nominal wages adjust slowly, the real wage
barely moves after the shock, so firms have less reason to cut prices and more of the adjustment falls on quantities.
With more flexible wages (\(\theta_w=0.25\)) wages and marginal cost fall quickly, price inflation drops more and the
output loss is smaller. Sticky wages are one reason estimated DSGE models (Lecture 13) find large real effects of
monetary policy with only moderate price stickiness.

</details>

## Further reading
- Galí, J. (2015), *Monetary Policy, Inflation, and the Business Cycle*, ch. 3 and 6.
- Calvo, G. (1983), "Staggered Prices in a Utility-Maximizing Framework," *Journal of Monetary Economics*.
- Erceg, C., D. Henderson and A. Levin (2000), "Optimal Monetary Policy with Staggered Wage and Price Contracts," *Journal of Monetary Economics*.
- Nakamura, E. and J. Steinsson (2008), "Five Facts about Prices," *Quarterly Journal of Economics*.
- Hazell, J., J. Herreño, E. Nakamura and J. Steinsson (2022), "The Slope of the Phillips Curve: Evidence from U.S. States," *Quarterly Journal of Economics*.
