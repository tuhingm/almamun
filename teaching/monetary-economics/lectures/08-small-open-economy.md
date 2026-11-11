# Lecture 08 — Monetary Policy in Small Open Economies

**Level:** Graduate | **Field:** Monetary Economics

## Learning objectives
Derive the Galí–Monacelli small-open-economy New Keynesian model, understand the roles of the terms of trade, the exchange rate and CPI versus domestic inflation, and compare domestic inflation targeting, CPI inflation targeting and an exchange-rate peg. Discuss pass-through, dollar pricing and the policy dilemmas of emerging markets such as Korea and Bangladesh.

## 1. Environment
A small open economy trades with a large world economy (Galí and Monacelli, 2005). Households consume domestic and imported goods with home bias \(1-\alpha\), where \(\alpha\) measures openness. Key definitions (in logs):
\[
s_t=p_{F,t}-p_{H,t}\ \text{(terms of trade)},\qquad p_t=(1-\alpha)p_{H,t}+\alpha p_{F,t}\ \Rightarrow\ \pi_t=\pi_{H,t}+\alpha\Delta s_t ,
\]
so CPI inflation equals domestic inflation plus the contribution of import prices. With the law of one price for imports, \(p_{F,t}=e_t+p^*_t\), and so \(s_t=e_t+p^*_t-p_{H,t}\).

International risk sharing under complete markets ties consumption to world consumption and the real exchange rate, and goods-market clearing then gives a simple link between output and the terms of trade: with \(\sigma=\eta=\gamma=1\),
\[
s_t=\sigma_\alpha(y_t-y^*_t),\qquad \sigma_\alpha=\frac{\sigma}{(1-\alpha)+\alpha\omega},\quad \omega=\sigma\gamma+(1-\alpha)(\sigma\eta-1).
\]
A domestic boom worsens the terms of trade (domestic goods become cheaper relative to foreign goods).

## 2. The open-economy New Keynesian model
Calvo pricing of domestic goods gives a Phillips curve in **domestic** inflation, and the Euler equation becomes an IS curve in the domestic real rate:
\[
\pi_{H,t}=\beta E_t\pi_{H,t+1}+\kappa_\alpha x_t,\qquad
x_t=E_tx_{t+1}-\frac{1}{\sigma_\alpha}\left(i_t-E_t\pi_{H,t+1}-r^n_t\right),
\]
with \(\kappa_\alpha=\lambda(\sigma_\alpha+\varphi)\) and a natural rate that depends on domestic technology and world output. Formally the model is identical to the closed economy with modified coefficients. Openness changes the slope of the IS curve (the real exchange rate adds an expenditure-switching channel) and the Phillips curve (the terms of trade affect real wages and marginal cost).

## 3. Optimal policy and simple regimes
In the special case \(\sigma=\eta=\gamma=1\), with an optimal subsidy, welfare depends only on domestic inflation and the output gap:
\[
\mathbb W=-\frac{1-\alpha}{2}\sum\beta^t\left[\frac{\epsilon}{\lambda}\pi_{H,t}^2+(1+\varphi)x_t^2\right].
\]
The optimal policy is **strict domestic inflation targeting** (\(\pi_H=x=0\)), and the exchange rate floats freely to absorb shocks. Galí and Monacelli compare three simple regimes:

- **DITR**: \(i_t=\phi_\pi\pi_{H,t}\) (Taylor rule on domestic inflation);
- **CITR**: \(i_t=\phi_\pi\pi_t\) (Taylor rule on CPI inflation), which leans against depreciation because import prices enter CPI;
- **PEG**: \(e_t=0\), which removes exchange-rate volatility but forces the terms of trade to adjust through domestic prices.

The peg generates the largest output-gap volatility; the CPI rule trades exchange-rate stability for output-gap volatility. How the simple rules rank in welfare depends on the shocks and on how aggressively each responds to inflation (Exercise 8.1).

## 4. Exchange-rate pass-through and dollar pricing
The model assumes producer-currency pricing with full and immediate pass-through to import prices. In the data, pass-through to consumer prices is incomplete and slow, and it fell in many countries after they adopted inflation targeting (Taylor, 2000; Gagnon and Ihrig, 2004). With **local-currency pricing**, deviations from the law of one price make the exchange rate less effective at expenditure switching. Under **dominant-currency pricing** (Gopinath et al., 2020) most trade is invoiced in U.S. dollars, so a country's trade volumes respond mainly to its dollar exchange rate and the U.S. monetary cycle has global spillovers.

## 5. Emerging-market dilemmas
Emerging-market central banks face features the baseline omits:

- **Fear of floating** (Calvo and Reinhart, 2002): foreign-currency debt means depreciation damages balance sheets, so central banks resist exchange-rate movements.
- **The global financial cycle** (Rey, 2013): capital flows and risk premia move with U.S. monetary policy and global risk appetite, weakening the insulation a float provides.
- **Integrated policy frameworks** (IMF, 2020): foreign-exchange intervention, capital-flow management and macroprudential tools complement the policy rate.

**Korea** moved to a free float with inflation targeting after the 1997–98 crisis, when the won lost almost 40% of its value against the dollar. Since then pass-through to CPI inflation has been low (Exercise 8.2). **Bangladesh** long operated a tightly managed exchange rate with monetary targeting. Amid inflation near 10% and falling reserves, Bangladesh Bank introduced an interest-rate corridor in July 2023, a crawling peg in May 2024 and a market-based exchange rate in May 2025, as part of its IMF programme.

## 6. Implementation in Dynare
Dynare's macro processor makes it easy to switch regimes in one file:

```matlab
@#if policy == 1
  i = phi_pi * pi_H;            // domestic inflation rule
@#elseif policy == 2
  i = phi_pi * pi;              // CPI inflation rule
@#else
  e = 0;                        // exchange-rate peg
@#endif
```

and `dynare soe -Dpolicy=3` runs the peg.

## Problem set
1. Derive \(\pi_t=\pi_{H,t}+\alpha\Delta s_t\) and the link between the terms of trade and the nominal exchange rate.
2. Show that the open-economy model nests the closed economy when \(\alpha=0\).
3. Explain why the peg makes the terms of trade adjust slowly and why this raises output-gap volatility.
4. Under dominant-currency pricing, how does a depreciation of the won affect Korean exports and imports in the short run?
5. Why might an emerging-market central bank rationally deviate from strict domestic inflation targeting?

## Coding exercises

These exercises use **MATLAB and Dynare only**. Download the course [data files and helper functions](../index.html#data-and-code)
into your working folder. Every data series is real: U.S. quarterly and monthly macro data (FRED-QD / FRED-MD,
1959–2025), published monetary-policy shock series (Romer–Romer, Jarociński–Karadi, Bu–Rogers–Wu,
Nakamura–Steinsson), World Bank CPI inflation for 190+ countries, and the KRW/USD exchange rate. Dynare models are
`.mod` files run with `dynare file noclearall` (Dynare 6). Try the starter files first, then open the solution. Every
printed result was produced by running the solution files as shown: the `.mod` files were processed by the Dynare
preprocessor and solved at first order, and the MATLAB code was run in GNU Octave.

### Exercise 8.1 — Policy regimes in a small open economy (Dynare)

*Data:* Calibration of Galí and Monacelli (2005): \(\alpha=0.4\), \(\varphi=3\), \(\rho_a=0.66\), \(\sigma_a=0.71\%\).

Implement the Galí–Monacelli small open economy with a domestic technology shock. Use Dynare's macro processor
(`@#if`) to switch between three regimes: a Taylor rule on domestic inflation (DITR), a Taylor rule on CPI inflation
(CITR) and an exchange-rate peg (PEG).

1. Run the model three times with `dynare soe -Dpolicy=1` (2, 3) and collect the standard deviations of the output
   gap, domestic and CPI inflation, nominal depreciation and the terms of trade.
2. In the special case \(\sigma=\eta=\gamma=1\), the welfare loss is
   \(\frac{1-\alpha}{2}\left[\frac{\epsilon}{\lambda}\mathrm{var}(\pi_H)+(1+\varphi)\mathrm{var}(x)\right]\).
   Compute it for each regime and rank them.
3. Repeat the domestic-inflation rule with \(\phi_\pi=5\). What does this tell you about the optimal policy?
4. Which regime stabilises the exchange rate best, and at what cost?

*File `soe.mod`*

```matlab
@#ifndef policy
  @#define policy = 1
@#endif
var x pi_H pi s de e p_H i r_nat y a;
varexo eps_a;
parameters beta sigma varphi alpha epsilon theta phi_pi rho_a;
beta = 0.99; sigma = 1; varphi = 3; alpha = 0.4; epsilon = 6; theta = 0.75; phi_pi = 1.5; rho_a = 0.66;
model(linear);
  // TODO: Phillips curve, IS curve, natural rate, output, terms of trade, CPI inflation, exchange rate
@#if policy == 1
  i = phi_pi * pi_H;
@#elseif policy == 2
  i = phi_pi * pi;
@#else
  e = 0;
@#endif
  a = rho_a * a(-1) + eps_a;
end;
shocks; var eps_a; stderr 0.71; end;
stoch_simul(order=1, irf=12, noprint, nograph) x pi_H pi de s y;
```

*File `ex8_1.m`*

```matlab
for pol = 1:3
  eval(sprintf('dynare soe -Dpolicy=%d noclearall', pol));
  % TODO: sqrt(diag(oo_.var)); inflation and depreciation annualised (x4)
end
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `soe.mod`*

```matlab
// Small open economy (Gali and Monacelli, 2005; Gali, 2015, ch. 8), technology shock
// policy: 1 = domestic inflation Taylor rule, 2 = CPI inflation Taylor rule, 3 = exchange-rate peg
@#ifndef policy
  @#define policy = 1
@#endif
var x pi_H pi s de e p_H i r_nat y a;
varexo eps_a;
parameters beta sigma varphi alpha epsilon theta phi_pi rho_a;
beta = 0.99; sigma = 1; varphi = 3; alpha = 0.4; epsilon = 6; theta = 0.75; phi_pi = 1.5; rho_a = 0.66;
model(linear);
  # lambda  = (1 - theta) * (1 - beta * theta) / theta;
  # sigma_a = sigma;                                   // with sigma = eta = gamma = 1, omega = 1
  # kappa_a = lambda * (sigma_a + varphi);
  # Gamma_a = (1 + varphi) / (sigma_a + varphi);
  pi_H = beta * pi_H(+1) + kappa_a * x;                         // domestic-inflation Phillips curve
  x = x(+1) - (1 / sigma_a) * (i - pi_H(+1) - r_nat);           // open-economy IS curve
  r_nat = -sigma_a * Gamma_a * (1 - rho_a) * a;                 // natural real rate
  y = x + Gamma_a * a;                                          // output (world output constant)
  s = sigma_a * y;                                              // terms of trade
  pi = pi_H + alpha * (s - s(-1));                              // CPI inflation
  p_H = p_H(-1) + pi_H;                                         // domestic price level
  e = s + p_H;                                                  // nominal exchange rate (foreign prices fixed)
  de = e - e(-1);
@#if policy == 1
  i = phi_pi * pi_H;
@#elseif policy == 2
  i = phi_pi * pi;
@#else
  e = 0;
@#endif
  a = rho_a * a(-1) + eps_a;
end;
shocks; var eps_a; stderr 0.71; end;
stoch_simul(order=1, irf=12, noprint, nograph) x pi_H pi de s y;
```

*File `ex8_1.m`*

```matlab
names = {'DITR', 'CITR', 'PEG '};
fprintf('regime   sd(x)   sd(pi_H)  sd(pi)   sd(de)   sd(s)   welfare loss\n');
for pol = 1:3
  eval(sprintf('dynare soe -Dpolicy=%d noclearall', pol));
  sd = sqrt(diag(oo_.var))';           % order: x pi_H pi de s y
  lam = (1 - theta) * (1 - beta * theta) / theta;
  W = (1 - alpha) / 2 * ((epsilon / lam) * oo_.var(2, 2) + (1 + varphi) * oo_.var(1, 1));   % Gali-Monacelli loss
  fprintf('%s   %6.3f  %7.3f  %7.3f  %7.3f  %6.3f   %8.4f\n', names{pol}, sd(1), 4 * sd(2), 4 * sd(3), 4 * sd(4), sd(5), W);
end
dynare soe -Dpolicy=1 noclearall                % domestic-inflation rule again, now responding more strongly
set_param_value('phi_pi', 5); [info, oo_] = stoch_simul(M_, options_, oo_, var_list_);
sd = sqrt(diag(oo_.var))'; W = (1 - alpha) / 2 * ((epsilon / lam) * oo_.var(2, 2) + (1 + varphi) * oo_.var(1, 1));
fprintf('DITR5  %6.3f  %7.3f  %7.3f  %7.3f  %6.3f   %8.4f   (phi_pi = 5)\n', sd(1), 4 * sd(2), 4 * sd(3), 4 * sd(4), sd(5), W);
fprintf('(standard deviations in %%; inflation and depreciation annualised; loss per period, in %% squared)\n');
```

Output:

```text
regime   sd(x)   sd(pi_H)  sd(pi)   sd(de)   sd(s)   welfare loss
DITR    0.274    1.086    1.082    2.024   0.671     1.6367
CITR    0.382    0.829    0.494    0.979   0.614     1.0747
PEG     0.539    0.962    0.577    0.000   0.517     1.5621
DITR5   0.069    0.274    1.072    2.787   0.876     0.1045   (phi_pi = 5)
(standard deviations in %; inflation and depreciation annualised; loss per period, in % squared)
```

The domestic-inflation rule gives the most stable output gap but the most volatile exchange rate: the nominal
exchange rate and the terms of trade absorb the shock. The CPI rule leans against depreciation because import prices
enter CPI inflation, so it halves the volatility of depreciation and CPI inflation at the cost of a more volatile
output gap. The peg eliminates nominal exchange-rate volatility, but the terms of trade can then adjust only through
domestic prices, so the output gap is twice as volatile as under the domestic-inflation rule. With
\(\phi_\pi=1.5\) none of the simple rules tracks the fall in the natural rate after a technology improvement, and in
this calibration the CPI rule happens to give the smallest loss. The point of Galí and Monacelli (2005) is that the
optimal policy is strict domestic-inflation targeting (\(\pi_H=x=0\)): a domestic-inflation rule that responds
strongly enough comes close to it, and its loss is far below all three regimes, even though the exchange rate is then
very volatile. The model ignores balance-sheet and pass-through effects that make emerging-market central banks wary
of large depreciations (Exercise 8.2).

</details>

### Exercise 8.2 — Exchange-rate pass-through in Korea

*Data:* `cpi_inflation_wdi.csv` (Korea CPI inflation) and `krw_usd_monthly.csv` (KRW per USD, 1981–2026).

1. Build annual Korean CPI inflation and the annual percentage change in the average KRW/USD rate.
2. Compare the mean and volatility of inflation and depreciation before the 1997–98 crisis (1982–1996) and under
   inflation targeting with a floating rate (1999–2024).
3. For 1999–2024 estimate \(\pi_t=a+b\,\Delta e_t+c\,\pi_{t-1}+\varepsilon_t\) and the long-run pass-through
   \(b/(1-c)\). Compare it with the pass-through implied by the 1998 episode, \((\pi_{1998}-\pi_{1997})/\Delta e_{1998}\).
4. Why might pass-through be low under a credible inflation target?

*File `ex8_2.m`*

```matlab
W = readtable('cpi_inflation_wdi.csv'); X = readtable('krw_usd_monthly.csv');
k = strcmp(W.code, 'KOR'); yr = W.year(k); infl = W.cpi_inflation(k);
% TODO: annual average exchange rate with accumarray; depreciation; align years; summary statistics; ols_nw
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `ex8_2.m`*

```matlab
W = readtable('cpi_inflation_wdi.csv'); X = readtable('krw_usd_monthly.csv');
k = strcmp(W.code, 'KOR'); yr = W.year(k); infl = W.cpi_inflation(k);
ys = unique(X.year); E = accumarray(X.year - ys(1) + 1, X.krw_usd, [], @mean);
dep = [NaN; 100 * diff(log(E))];                        % annual % change in KRW per USD (+ = depreciation)
years = (1982:2024)';
pi_t = arrayfun(@(y) infl(yr == y), years); pi_l = arrayfun(@(y) infl(yr == y - 1), years);
de = arrayfun(@(y) dep(ys == y), years);
fprintf('period       mean infl   sd infl   mean dep   sd dep\n');
for s = {[1982 1996], [1999 2024]}
  r = s{1}; m = years >= r(1) & years <= r(2);
  fprintf('%d-%d     %6.1f     %6.1f     %6.1f    %6.1f\n', r(1), r(2), mean(pi_t(m)), std(pi_t(m)), mean(de(m)), std(de(m)));
end
m = years >= 1999;
[b, se] = ols_nw(pi_t(m), [ones(sum(m), 1) de(m) pi_l(m)], 2);
fprintf('\n1999-2024: infl = %.2f + %.3f (%.3f) dep + %.2f (%.2f) infl(-1); long-run pass-through %.3f\n', ...
        b(1), b(2), se(2), b(3), se(3), b(2) / (1 - b(3)));
k98 = years == 1998; k97 = years == 1997;
fprintf('1998: depreciation %.1f%%, inflation %.1f%% (1997: %.1f%%), implied pass-through %.3f\n', ...
        de(k98), pi_t(k98), pi_t(k97), (pi_t(k98) - pi_t(k97)) / de(k98));
```

Output:

```text
period       mean infl   sd infl   mean dep   sd dep
1982-1996        5.2        2.2        1.1       6.1
1999-2024        2.4        1.3       -0.1       8.1

1999-2024: infl = 1.80 + 0.059 (0.029) dep + 0.24 (0.17) infl(-1); long-run pass-through 0.078
1998: depreciation 38.6%, inflation 7.5% (1997: 4.4%), implied pass-through 0.080
```

Before the crisis the won was tightly managed — depreciation was small and stable — while inflation averaged
about 5%. Under inflation targeting since 1999, inflation is lower and more stable even though the exchange rate moves
far more. Pass-through is small: a 10% depreciation raises inflation by roughly half a percentage point in the year it
occurs. Even the 1998 collapse, a depreciation of almost 40%, raised inflation by only about 3 points. Taylor (2000)
argued that pass-through falls when inflation is low and stable, because firms expect exchange-rate changes to be
temporary and offset by policy, so they adjust prices less. With about 26 annual observations the estimate is
imprecise. Bangladesh, with a managed exchange rate and higher inflation, is a useful comparison if you can obtain its
exchange-rate series.

</details>

## Further reading
- Galí, J. and T. Monacelli (2005), "Monetary Policy and Exchange Rate Volatility in a Small Open Economy," *Review of Economic Studies*.
- Galí, J. (2015), *Monetary Policy, Inflation, and the Business Cycle*, ch. 8.
- Calvo, G. and C. Reinhart (2002), "Fear of Floating," *Quarterly Journal of Economics*.
- Gopinath, G. et al. (2020), "Dominant Currency Paradigm," *American Economic Review*.
- Rey, H. (2013), "Dilemma not Trilemma: The Global Financial Cycle and Monetary Policy Independence," Jackson Hole Symposium.
