# Lecture 07 — Policy Rules and Frameworks

**Level:** Graduate | **Field:** Monetary Economics

## Learning objectives
Interpret Taylor rules as descriptions of and prescriptions for policy, estimate them, and understand their pitfalls. Compare strict and flexible inflation targeting, price-level targeting, average-inflation targeting and nominal-GDP targeting, and analyse them in a New Keynesian model.

## 1. The Taylor rule
Taylor (1993) showed that U.S. policy in 1987–1992 was well described by
\[
i_t=r^*+\pi_t+0.5(\pi_t-\pi^*)+0.5\,x_t ,
\]
with \(r^*=\pi^*=2\%\) and \(x_t\) the output gap. The coefficient on inflation is 1.5, so the rule satisfies the Taylor principle of Lecture 5. Empirical versions add **interest-rate smoothing** and forward-looking targets:
\[
i_t=\rho_i i_{t-1}+(1-\rho_i)\left[r^*+\pi^*+\phi_\pi(E_t\pi_{t+k}-\pi^*)+\phi_xE_tx_{t+q}\right]+\varepsilon_t .
\]
Clarida, Galí and Gertler (2000) estimate this by GMM with lagged variables as instruments and find \(\phi_\pi<1\) before 1979 and \(\phi_\pi\approx2\) under Volcker and Greenspan. In the New Keynesian model, the pre-1979 rule allows self-fulfilling inflation, offering an explanation of the Great Inflation.

## 2. Pitfalls in estimating and using rules
- **Real-time data.** Orphanides (2001) shows that output-gap estimates available to the Fed in the 1970s were far too pessimistic. In real-time data the 1970s Fed responded strongly to its (mismeasured) inflation and gap forecasts; the problem was bad measurement, not a passive rule.
- **Identification.** In the New Keynesian model with determinacy, the coefficient \(\phi_\pi\) is not identified from equilibrium data on inflation and interest rates without further restrictions (Cochrane, 2011), because off-equilibrium threats never materialise.
- **The neutral rate.** Rules need an estimate of \(r^*\). Laubach–Williams estimates fell from about 2% before 2008 to below 1% in the 2010s, which shifts the whole rule down.
- **The lower bound.** A rule may prescribe negative rates (Lecture 10).

## 3. Inflation targeting
Since New Zealand in 1990, about 45 central banks have adopted explicit inflation targets. **Strict** inflation targeting stabilises inflation only; **flexible** inflation targeting (Svensson, 1997) minimises a loss in inflation and the output gap, and in practice means bringing inflation back to target over a medium-term horizon. Key features are a numerical target, a medium-term horizon, transparency (published forecasts, inflation reports) and accountability. Korea adopted inflation targeting in 1998 after the Asian financial crisis; the Bank of Korea's target is 2% CPI inflation since 2016. The Federal Reserve announced a 2% PCE target in 2012.

## 4. History dependence: price-level and average-inflation targeting
Under inflation targeting past misses are not made up: a temporary inflation shock permanently raises the price level. Under **price-level targeting** (PLT) the central bank brings the price level back to a deterministic path:
\[
i_t=\phi\,(p_t-p^*_t) .
\]
In forward-looking models, PLT is attractive because it mimics the commitment solution of Lecture 6: expected future deflation after a cost-push shock dampens current inflation, and expected future inflation after a deflationary shock lowers real rates at the lower bound. Its costs are the need for credibility and a deeper output response when supply shocks are persistent. In 2020 the Federal Reserve adopted **flexible average-inflation targeting** (FAIT): after periods of inflation persistently below 2%, policy would aim for inflation moderately above 2% for some time. The 2025 framework review removed this make-up element and returned to flexible inflation targeting, citing the experience of 2021–23.

## 5. Nominal-GDP targeting and other rules
Nominal-GDP level targeting (Hall and Mankiw, 1994; Woodford, 2012) combines a price-level target with an output stabilisation motive and responds automatically to supply shocks by splitting them between prices and output. It requires reliable real-time nominal-GDP data. Money-growth rules (Friedman's k-percent rule) and exchange-rate targets are discussed in Lectures 1 and 8.

## 6. Analysing rules in Dynare
A useful practice is to write one model with a switch parameter for the rule, as in Exercise 7.2:

```matlab
p = p(-1) + pi;                              // price level relative to its target path
i = phi * ((1 - plt) * pi + plt * p);        // plt = 0: inflation targeting; plt = 1: price-level targeting
```

and compare impulse responses and welfare losses with `set_param_value('plt', 1)` inside a loop. For rules that need different variables, use Dynare's macro processor (`@#if`, `-D` flags), as in Exercise 8.1.

## Problem set
1. Show that the original Taylor rule satisfies the Taylor principle. What long-run inflation rate does it imply if \(r^*\) is overestimated by one point?
2. Explain how interest-rate smoothing changes the determinacy condition and the interpretation of \(\phi_\pi\).
3. Why does price-level targeting mimic the commitment policy after a cost-push shock?
4. Discuss the costs of price-level targeting if the central bank lacks credibility.
5. Compare inflation targeting and nominal-GDP targeting after a negative supply shock.

## Coding exercises

These exercises use **MATLAB and Dynare only**. Download the course [data files and helper functions](../index.html#data-and-code)
into your working folder. Every data series is real: U.S. quarterly and monthly macro data (FRED-QD / FRED-MD,
1959–2025), published monetary-policy shock series (Romer–Romer, Jarociński–Karadi, Bu–Rogers–Wu,
Nakamura–Steinsson), World Bank CPI inflation for 190+ countries, and the KRW/USD exchange rate. Dynare models are
`.mod` files run with `dynare file noclearall` (Dynare 6). Try the starter files first, then open the solution. Every
printed result was produced by running the solution files as shown: the `.mod` files were processed by the Dynare
preprocessor and solved at first order, and the MATLAB code was run in GNU Octave.

### Exercise 7.1 — Estimating Taylor rules before and after Volcker

*Data:* `us_quarterly.csv` — federal funds rate, GDP deflator and real GDP, 1960–2007.

Estimate the partial-adjustment rule \(i_t=c+\rho\,i_{t-1}+(1-\rho)(\phi_\pi\pi_t+\phi_x x_t)+\varepsilon_t\), where
\(\pi_t\) is four-quarter GDP-deflator inflation and \(x_t\) is the HP-filtered output gap.

1. Estimate the regression \(i_t=a+\rho i_{t-1}+b\pi_t+dx_t\) by OLS and by 2SLS, instrumenting \(\pi_t\) and \(x_t\)
   with lags 1–4 of inflation and 1–2 of the gap. Recover \(\phi_\pi=b/(1-\rho)\) and \(\phi_x=d/(1-\rho)\).
2. Compare 1960Q1–1979Q2 with 1979Q3–2007Q4 (Clarida, Galí and Gertler, 2000).
3. Relate your estimates to the determinacy condition from Lecture 5.

*File `ex7_1.m`*

```matlab
T = readtable('us_quarterly.csv');
t = T.t; i = T.ffr; pi = 100 * (log(T.gdpdef) - [NaN(4,1); log(T.gdpdef(1:end-4))]);
x = 100 * (log(T.rgdp) - hp_trend(log(T.rgdp), 1600));
% TODO: build lags, choose sample, OLS and 2SLS (Xhat = Z*(Z\X); b = Xhat\y)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `ex7_1.m`*

```matlab
T = readtable('us_quarterly.csv');
t = T.t; i = T.ffr; lp = log(T.gdpdef);
pi = 100 * (lp - [NaN(4, 1); lp(1:end-4)]);
x = 100 * (log(T.rgdp) - hp_trend(log(T.rgdp), 1600));
L = @(v, k) [NaN(k, 1); v(1:end-k)];
y = i; X = [ones(size(i)) L(i, 1) pi x];
Z = [ones(size(i)) L(i, 1) L(pi, 1) L(pi, 2) L(pi, 3) L(pi, 4) L(x, 1) L(x, 2)];
fprintf('sample            method   rho     phi_pi   phi_x\n');
for s = {[1960 1979.25], [1979.5 2007.75]}
  r = s{1}; k = t >= r(1) & t <= r(2) & all(~isnan([X Z]), 2);
  bo = X(k, :) \ y(k);
  Xh = Z(k, :) * (Z(k, :) \ X(k, :)); bi = Xh \ y(k);
  for m = 1:2
    if m == 1, b = bo; nm = 'OLS '; else, b = bi; nm = '2SLS'; end
    fprintf('%7.2f-%7.2f   %s   %5.2f   %6.2f   %6.2f\n', r(1), r(2), nm, b(2), b(3) / (1 - b(2)), b(4) / (1 - b(2)));
  end
end
```

Output:

```text
sample            method   rho     phi_pi   phi_x
1960.00-1979.25   OLS     0.79     0.57     1.71
1960.00-1979.25   2SLS    0.77     0.58     1.78
1979.50-2007.75   OLS     0.82     1.75     1.39
1979.50-2007.75   2SLS    0.83     1.70     1.00
```

Both samples show strong interest-rate smoothing (\(\rho\) around 0.8). Before 1979 the long-run response to
inflation is below one: the Fed let real rates fall when inflation rose, which violates the Taylor principle and, in
the New Keynesian model, leaves the door open to self-fulfilling inflation (Clarida, Galí and Gertler, 2000). After
1979 the response is well above one. Caveats: the output gap is measured with hindsight, not in real time (Orphanides,
2001, finds that real-time gaps were badly overestimated in the 1970s), and contemporaneous inflation is used instead
of the forecasts the Fed actually reacted to. Estimates are therefore sensitive to the gap measure and instruments,
but the shift in the inflation response is a robust finding.

</details>

### Exercise 7.2 — Inflation targeting versus price-level targeting (Dynare)

*Data:* Model calibration; an i.i.d. cost-push shock of 1.

Under inflation targeting (IT) the rule is \(i_t=\phi\pi_t\); under price-level targeting (PLT) it is
\(i_t=\phi p_t\), where \(p_t\) is the log price level relative to its target path.

1. Use one Dynare model with a switch parameter `plt` and compare the responses of inflation, the output gap and the
   price level to a cost-push shock.
2. Compute the discounted loss \(\sum\beta^t(\pi_t^2+\vartheta\tilde y_t^2)\) from the impulse responses for both
   regimes (\(\vartheta=\kappa/9\)).
3. Relate the PLT outcome to the commitment solution in Exercise 6.1.

*File `plt.mod`*

```matlab
// Inflation targeting versus price-level targeting after a cost-push shock
var y_gap pi p i u;
varexo eps_u;
parameters beta sigma kappa phi plt rho_u;
beta = 0.99; sigma = 1; kappa = 0.1717; phi = 1.5; plt = 0; rho_u = 0;
model(linear);
  pi = beta * pi(+1) + kappa * y_gap + u;
  y_gap = y_gap(+1) - (1 / sigma) * (i - pi(+1));
  p = p(-1) + pi;                                    // price level (log deviation from target path)
  i = phi * ((1 - plt) * pi + plt * p);              // IT (plt = 0) or PLT (plt = 1)
  u = rho_u * u(-1) + eps_u;
end;
shocks; var eps_u; stderr 1; end;
stoch_simul(order=1, irf=40, noprint, nograph);
```

*File `ex7_2.m`*

```matlab
dynare plt noclearall
% TODO: for plt = [0 1]: set_param_value, stoch_simul, read oo_.irfs.pi_eps_u, y_gap_eps_u, p_eps_u
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `plt.mod`*

```matlab
// Inflation targeting versus price-level targeting after a cost-push shock
var y_gap pi p i u;
varexo eps_u;
parameters beta sigma kappa phi plt rho_u;
beta = 0.99; sigma = 1; kappa = 0.1717; phi = 1.5; plt = 0; rho_u = 0;
model(linear);
  pi = beta * pi(+1) + kappa * y_gap + u;
  y_gap = y_gap(+1) - (1 / sigma) * (i - pi(+1));
  p = p(-1) + pi;                                    // price level (log deviation from target path)
  i = phi * ((1 - plt) * pi + plt * p);              // IT (plt = 0) or PLT (plt = 1)
  u = rho_u * u(-1) + eps_u;
end;
shocks; var eps_u; stderr 1; end;
stoch_simul(order=1, irf=40, noprint, nograph);
```

*File `ex7_2.m`*

```matlab
dynare plt noclearall
vartheta = kappa / 9; H = options_.irf; disc = beta .^ (0:H-1);
names = {'IT ', 'PLT'};
for reg = [0 1]
  set_param_value('plt', reg);
  [info, oo_] = stoch_simul(M_, options_, oo_, {});
  ir = oo_.irfs; Lr = sum(disc .* (ir.pi_eps_u.^2 + vartheta * ir.y_gap_eps_u.^2));
  fprintf('%s  pi: %s\n', names{reg+1}, sprintf('%7.3f', ir.pi_eps_u([1 2 3 5 9])));
  fprintf('     x : %s\n', sprintf('%7.3f', ir.y_gap_eps_u([1 2 3 5 9])));
  fprintf('     p : %s   (h = 0, 1, 2, 4, 8)\n', sprintf('%7.3f', ir.p_eps_u([1 2 3 5 9])));
  fprintf('     price level after 40 quarters %.3f, discounted loss %.4f\n\n', ir.p_eps_u(end), Lr);
end
```

Output:

```text
IT   pi:   0.795  0.000  0.000  0.000  0.000
     x :  -1.193  0.000  0.000  0.000  0.000
     p :   0.795  0.795  0.795  0.795  0.795   (h = 0, 1, 2, 4, 8)
     price level after 40 quarters 0.795, discounted loss 0.6595

PLT  pi:   0.458 -0.248 -0.114 -0.024 -0.001
     x :  -1.725 -0.790 -0.362 -0.076 -0.003
     p :   0.458  0.210  0.096  0.020  0.001   (h = 0, 1, 2, 4, 8)
     price level after 40 quarters 0.000, discounted loss 0.3584
```

Under IT the price level drifts up permanently after the shock ("bygones are bygones"). Under PLT the central bank
must bring the price level back, so after the initial rise inflation turns negative for a while. Because firms expect
this, they raise prices by less on impact: inflation rises by 0.46 instead of 0.80. The cost is a deeper and longer
fall in output, but with the welfare weights implied by the model the discounted loss is almost halved, even though
neither rule is optimal. PLT builds in the history dependence of
the commitment solution in Exercise 6.1, where the price level is also stationary. This is the logic behind the
Fed's 2020 move to flexible average-inflation targeting, and it depends on the private sector believing the promise
to make up past misses.

</details>

## Further reading
- Taylor, J. (1993), "Discretion versus Policy Rules in Practice," *Carnegie-Rochester Conference Series on Public Policy*.
- Clarida, R., J. Galí and M. Gertler (2000), "Monetary Policy Rules and Macroeconomic Stability," *Quarterly Journal of Economics*.
- Orphanides, A. (2001), "Monetary Policy Rules Based on Real-Time Data," *American Economic Review*.
- Svensson, L. (1997), "Inflation Forecast Targeting," *European Economic Review*.
- Vestin, D. (2006), "Price-Level versus Inflation Targeting," *Journal of Monetary Economics*.
