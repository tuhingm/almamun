title: Small Open-Economy New Keynesian Model
short: Open economy
when: Explain a mechanism in an open economy
data: Calibrated quarterly model (Galí and Monacelli 2005, Canada calibration)
lead: Use this guide when exchange rates, terms of trade or imported inflation matter. It writes the Galí–Monacelli small open-economy model in Dynare, uses the macro language to switch between monetary regimes, and compares domestic-inflation targeting, CPI-inflation targeting and an exchange-rate peg by volatility and welfare.
---

## The question and the model

**Should a small open economy target domestic inflation, CPI inflation, or the exchange rate?** Many emerging-market central banks face exactly this choice. Galí and Monacelli (2005) answered it in a New Keynesian model in which home and foreign goods are imperfect substitutes and the home economy is too small to affect the rest of the world.

With the parameter values of Cole and Obstfeld (unit elasticities, \(\sigma = \eta = \gamma = 1\)), the log-linear model reduces to:

\[
\begin{aligned}
\tilde y_t &= E_t \tilde y_{t+1} - \tfrac{1}{\sigma_\alpha}\left(i_t - E_t \pi_{H,t+1} - \bar r^n_t\right) && \text{IS curve, in terms of domestic inflation}\\
\pi_{H,t} &= \beta E_t \pi_{H,t+1} + \kappa_\alpha \tilde y_t && \text{domestic Phillips curve}\\
s_t &= \sigma_\alpha\, y_t && \text{terms of trade (world output fixed)}\\
\pi_t &= \pi_{H,t} + \alpha\, \Delta s_t && \text{CPI inflation}\\
\Delta e_t &= \Delta s_t + \pi_{H,t} && \text{nominal exchange rate (world prices fixed)}
\end{aligned}
\]

where \(\tilde y_t = y_t - \bar y^n_t\) is the output gap, \(\bar y^n_t = \Gamma_a a_t\) natural output driven by domestic productivity \(a_t\), \(\bar r^n_t = -\sigma_\alpha \Gamma_a (1-\rho_a) a_t\) the natural rate, \(\alpha\) the degree of openness, and \(\kappa_\alpha = \lambda(\sigma_\alpha + \varphi)\).

Four regimes are compared:

| Regime | Rule |
|---|---|
| DIT | Strict domestic-inflation targeting: \(\pi_{H,t} = 0\), the optimal policy in this model |
| DITR | Domestic-inflation Taylor rule: \(i_t = \phi_\pi \pi_{H,t}\) |
| CITR | CPI-inflation Taylor rule: \(i_t = \phi_\pi \pi_t\) |
| PEG | Fixed exchange rate: \(e_t = 0\) |

<div class="read" markdown="1">
**Why one model file with a switch**

The regimes differ in one equation. Writing four files invites inconsistencies. Dynare's macro language (`@#if`, `@#define`) keeps one file and selects the rule at run time: `dynare soe -DRULE=2`.
</div>

## Write the model in Dynare

```dynare soe.mod to="POLICY AND TRANSITION"
// Galí-Monacelli (2005) small open economy, Cole-Obstfeld case. Quarterly rates in percent.
@#ifndef RULE
  @#define RULE = 1
@#endif
var x y yn pih pi s e i a rrn;
varexo eps_a;
parameters beta alpha theta varphi phi_pi rho_a sigma_a lambda kappa_a Gamma_a;

beta = 0.99;  alpha = 0.4;  theta = 0.75;  varphi = 3;  phi_pi = 1.5;  rho_a = 0.66;
sigma_a = 1;                                            // sigma/((1-alpha)+alpha*omega) = 1 in the Cole-Obstfeld case
lambda  = (1 - theta) * (1 - beta*theta) / theta;
kappa_a = lambda * (sigma_a + varphi);
Gamma_a = (1 + varphi) / (sigma_a + varphi);

model(linear);
  x   = x(+1) - (1/sigma_a) * (i - pih(+1) - rrn);       // IS curve
  pih = beta * pih(+1) + kappa_a * x;                     // domestic Phillips curve
  x   = y - yn;                                           // output gap
  yn  = Gamma_a * a;                                      // natural output
  rrn = -sigma_a * Gamma_a * (1 - rho_a) * a;             // natural real rate
  s   = sigma_a * y;                                      // terms of trade
  pi  = pih + alpha * (s - s(-1));                        // CPI inflation
  e - e(-1) = s - s(-1) + pih;                            // nominal exchange rate
  a   = rho_a * a(-1) + eps_a;                            // domestic productivity
@#if RULE == 0
  pih = 0;                                                // strict domestic-inflation targeting (the optimum)
@#elseif RULE == 1
  i = phi_pi * pih;                                       // DITR
@#elseif RULE == 2
  i = phi_pi * pi;                                        // CITR
@#else
  e = 0;                                                  // PEG
@#endif
end;

shocks;
  var eps_a; stderr 0.712;                                // percent, as in Galí-Monacelli
end;

steady;
check;
stoch_simul(order=1, irf=16, nograph) y pih pi s e i x;
```

<div class="read" markdown="1">
**How to read it**

The default run uses the domestic-inflation rule (`RULE = 1`). The model is determinate. Note how the exchange rate enters: the level `e` appears with a lag, so it is a state variable, and under the peg it is pinned at zero.
</div>

## Impulse responses under the three regimes

```octave
names = {'DIT', 'DITR', 'CITR', 'PEG'};
vars = {'y', 'pih', 'pi', 's', 'e', 'i'};
titles = {'Output', 'Domestic inflation', 'CPI inflation', 'Terms of trade', 'Nominal exchange rate', 'Interest rate'};
R = struct();
for r = 1:4
  clear functions;                               % Octave caches the compiled model; clear it before re-running
  evalc(sprintf('dynare soe noclearall nograph nolog -DRULE=%d', r - 1));   % run quietly
  for v = 1:numel(vars), R(r).(vars{v}) = oo_.irfs.([vars{v} '_eps_a']); end
  R(r).sd = sqrt(diag(oo_.var))'; R(r).x = oo_.irfs.x_eps_a;
end
f = figure('position', [0 0 900 420]);
for v = 1:numel(vars)
  subplot(2, 3, v); hold on;
  for r = 1:4, plot(0:15, R(r).(vars{v})); end
  plot([0 15], [0 0], 'k-', 'linewidth', 0.5); title(titles{v}); grid on; xlim([0 15]);
  if v == 1, legend(names, 'location', 'northeast'); end
end
print(f, '-dpng', '-r110', 'irf.png');
```

<div class="read" markdown="1">
**How to read it**

A domestic productivity improvement lowers home production costs and raises natural output.

- **DIT (the optimum).** Domestic inflation stays at zero and output rises one-for-one with its natural level. The interest rate falls with the natural rate and the currency depreciates: the terms of trade worsen so that foreigners buy the extra home output. The exchange rate does all the adjusting, as it would with flexible prices.
- **DITR.** The simple rule cuts the rate too little, so output rises by less than its natural level, the output gap is negative and domestic inflation falls.
- **CITR.** CPI inflation includes the rise in import prices from the depreciation, so it falls less than domestic inflation and the central bank cuts the rate even less at first. Output rises less, but the gap closes faster.
- **PEG.** The nominal exchange rate cannot move, so the relative price must adjust through sticky domestic prices. Domestic inflation falls and the adjustment is slow.
</div>

## Volatility and welfare

```octave
eps = 6;                                        % elasticity of substitution between goods
lam = (1 - 0.75) * (1 - 0.99*0.75) / 0.75;
fprintf('%-6s %8s %8s %8s %8s %8s %8s %10s\n', '', 'sd(y)', 'sd(piH)', 'sd(pi)', 'sd(s)', 'sd(e)', 'sd(i)', 'loss');
for r = 1:4
  sd = R(r).sd;                                 % order: y pih pi s e i x
  loss = (1 - 0.4)/2 * ((eps/lam) * sd(2)^2 + (1 + 3) * sd(7)^2);
  fprintf('%-6s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %10.4f\n', names{r}, sd(1:6), loss);
end
```

<div class="read" markdown="1">
**How to read it**

The table follows Table 1 of Galí and Monacelli (2005). The welfare loss is their second-order approximation to household welfare, which penalises domestic-inflation variance (price dispersion) and output-gap variance:

\[
\mathbb{L} = \frac{1-\alpha}{2}\left[\frac{\epsilon}{\lambda}\operatorname{var}(\pi_H) + (1+\varphi)\operatorname{var}(\tilde y)\right]
\]

- **The optimum.** Strict domestic-inflation targeting has zero loss: it reproduces the flexible-price allocation. Every simple rule falls short of it.
- **Among the simple rules**, the CPI rule does best in this calibration and the domestic-inflation Taylor rule worst, with the peg close to it. The ranking of simple rules is not universal: it depends on the response coefficient, the shock process and the degree of openness. Re-run the comparison under your own calibration rather than quoting a ranking.
- **The exchange rate** has a unit root under the floating regimes (its level drifts after every shock), so its unconditional standard deviation is infinite and Dynare reports `NaN`. Under the peg it is zero by construction.
- **Caveat.** These results rest on the Cole–Obstfeld assumptions, complete risk sharing and the absence of foreign-currency debt. Emerging-market models with currency mismatch or imported-input costs can change them substantially.
</div>

```octave hide
ex = struct('names', {names});
for r = 1:4, ex.(names{r}) = struct('sd', R(r).sd, 'y', R(r).y, 'pih', R(r).pih, 'e', R(r).e, 'i', R(r).i); end
ex.loss = zeros(1,4);
for r = 1:4, ex.loss(r) = (1 - 0.4)/2 * ((eps/lam) * R(r).sd(2)^2 + 4 * R(r).sd(7)^2); end
fid = fopen(fullfile(export_dir, 'soe.json'), 'w'); fputs(fid, jsonencode(ex)); fclose(fid);
```

## Write the paper

<div class="write" markdown="1">
**How to write it: results**

Figure 1 compares the responses to a domestic productivity shock under three monetary regimes. Under domestic-inflation targeting, the nominal exchange rate depreciates and the terms of trade adjust, leaving domestic inflation and the output gap nearly unchanged. Under an exchange-rate peg, the adjustment of relative prices must occur through sticky domestic prices, generating larger fluctuations in domestic inflation and the output gap. Strict domestic-inflation targeting replicates the flexible-price allocation and is optimal. Among simple rules, a CPI-inflation Taylor rule yields a smaller welfare loss than a domestic-inflation Taylor rule or a peg in our calibration.
</div>

<div class="warn" markdown="1">
**Common mistakes**

- Writing the Phillips curve in CPI inflation. Firms set domestic prices; CPI inflation follows from the definition.
- Forgetting that the exchange-rate level is a state variable once a peg is possible.
- Ranking regimes by the volatility of the variables they target. Rank them by a welfare criterion.
- Using the Cole–Obstfeld special case and then generalising the ranking to economies with incomplete markets or dollar debt.
</div>
