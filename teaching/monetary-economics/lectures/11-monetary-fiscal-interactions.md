# Lecture 11 — Monetary–Fiscal Interactions

**Level:** Graduate | **Field:** Monetary Economics

## Learning objectives
Use the government budget constraint to link monetary and fiscal policy, classify policy regimes as active or passive (Leeper, 1991), and understand the fiscal theory of the price level. Analyse the inflation tax on nominal debt and apply the framework to the 2021–23 inflation.

## 1. The government's intertemporal budget constraint
Let \(B_{t-1}\) be nominal one-period debt maturing at \(t\), \(s_t\) the real primary surplus and \(P_t\) the price level. In equilibrium the real value of outstanding debt equals the present value of surpluses:
\[
\frac{B_{t-1}}{P_t}=E_t\sum_{j=0}^{\infty}\Lambda_{t,t+j}\,s_{t+j},
\]
where \(\Lambda\) is the real stochastic discount factor. Seigniorage can be included in \(s_t\). This **valuation equation** must hold in every equilibrium; the question is which variable adjusts to make it hold.

## 2. Active and passive policies
Leeper (1991) calls a policy **active** if it is not constrained by current budget conditions and pursues its own objective, and **passive** if it responds to current conditions to satisfy the other authority's constraints. With a monetary rule \(i_t=\phi_\pi\pi_t\) and a surplus rule \(s_t=\psi b_{t-1}+z_t\) (with \(b\) real debt):

| | Passive fiscal (\(\psi>1/\beta-1\)) | Active fiscal (\(\psi<1/\beta-1\)) |
|---|---|---|
| **Active monetary** (\(\phi_\pi>1\)) | Unique equilibrium: monetary policy determines inflation; surpluses adjust (Regime M) | No stable equilibrium |
| **Passive monetary** (\(\phi_\pi<1\)) | Indeterminacy | Unique equilibrium: fiscal policy determines the price level (Regime F) |

In **Regime M** a debt-financed transfer leads to higher future taxes; Ricardian equivalence holds and inflation does not move. In **Regime F** surpluses do not respond to debt, so the price level must jump to make the real value of debt equal to the present value of surpluses: unfunded transfers are inflationary.

## 3. The fiscal theory of the price level
The fiscal theory (Leeper, 1991; Sims, 1994; Woodford, 1995; Cochrane, 2023) reads the valuation equation as an equilibrium condition that determines the price level, like a stock price determined by the present value of dividends. Key implications:

- A fiscal expansion without expected future surpluses raises inflation even with an unchanged policy rate.
- With long-term debt, the adjustment can come through a gradual rise in inflation and a fall in bond prices rather than a price-level jump, so a fiscal shock produces persistent inflation (Cochrane, 2001).
- An interest-rate rise with passive fiscal policy can raise inflation in the short run in some versions (the "Neo-Fisherian" result), because higher interest costs are not backed by surpluses.

Critics question whether governments can credibly commit to "non-Ricardian" behaviour and whether the theory's equilibrium selection is plausible (Buiter, 2002; Niepelt, 2004).

## 4. Unpleasant monetarist arithmetic
Sargent and Wallace (1981) show that if fiscal policy is dominant and the public will only hold a limited amount of debt, tighter money today means more debt and therefore more seigniorage, and higher inflation, later. Monetary control of inflation requires fiscal backing. Many hyperinflations and the high inflation of several emerging markets fit this pattern.

## 5. The inflation tax on nominal debt
Surprise inflation lowers the real value of nominal debt with fixed coupons. In the debt-ratio decomposition
\[
\Delta b_t=\frac{D_t-D_{t-1}}{Y_t}-b_{t-1}\frac{g_t}{1+g_t},
\]
the nominal-growth term splits into real growth and inflation. In 2021–2023 U.S. inflation lowered the federal debt ratio by roughly 17 points of GDP in gross terms (Exercise 11.2). Hall and Sargent (2011) show that inflation and growth were the main forces that reduced U.S. debt after World War II.

## 6. The 2021–23 inflation through a fiscal lens
Large pandemic transfers in 2020–21, with no announced future surpluses, coincided with the inflation surge. Fiscal-theory accounts (Cochrane, 2023; Bianchi, Faccini and Melosi, 2023) attribute part of the inflation to unfunded fiscal expansion; Bianchi, Faccini and Melosi distinguish funded and unfunded spending in an estimated model with regime switching. Other accounts emphasise supply-chain disruptions, energy prices and labour-market tightness with a steep Phillips curve (Bernanke and Blanchard, 2023). The episodes need not be mutually exclusive.

## 7. Implementation in Dynare
A linear model with debt needs one more state variable:

```matlab
b  = (1/beta) * b(-1) + (bbar/beta) * (i(-1) - pi) - sp;   // real debt, share of quarterly GDP
sp = psi * b(-1) - z;                                       // surplus rule; z = transfers
```

Looping over \(\phi_\pi\) and \(\psi\) and checking `info(1)` from `stoch_simul` reproduces Leeper's regime map (Exercise 11.1).

## Problem set
1. Derive the government's intertemporal budget constraint from the period budget constraint and a transversality condition.
2. Using the linear model of Section 7, show that with \(\phi_\pi>1\) and \(\psi>1/\beta-1\) a transfer shock does not affect inflation.
3. In Regime F, derive the impact response of inflation to a transfer shock when debt is one-period.
4. Explain Sargent and Wallace's "unpleasant monetarist arithmetic."
5. Evaluate the evidence for a fiscal explanation of the 2021–23 inflation.

## Coding exercises

These exercises use **MATLAB and Dynare only**. Download the course [data files and helper functions](../index.html#data-and-code)
into your working folder. Every data series is real: U.S. quarterly and monthly macro data (FRED-QD / FRED-MD,
1959–2025), published monetary-policy shock series (Romer–Romer, Jarociński–Karadi, Bu–Rogers–Wu,
Nakamura–Steinsson), World Bank CPI inflation for 190+ countries, and the KRW/USD exchange rate. Dynare models are
`.mod` files run with `dynare file noclearall` (Dynare 6). Try the starter files first, then open the solution. Every
printed result was produced by running the solution files as shown: the `.mod` files were processed by the Dynare
preprocessor and solved at first order, and the MATLAB code was run in GNU Octave.

### Exercise 11.1 — Active and passive policy regimes (Dynare)

*Data:* Model calibration; government debt of 60% of annual GDP (\(\bar b=2.4\) quarters of GDP).

Add one-period nominal government debt to the New Keynesian model. Real debt evolves as
\(b_t=\beta^{-1}b_{t-1}+\beta^{-1}\bar b(i_{t-1}-\pi_t)-s_t\), and the primary surplus follows
\(s_t=\psi b_{t-1}-z_t\), where \(z_t\) is a transfer shock.

1. Loop over \(\phi_\pi\in\{0.5,1.5\}\) and \(\psi\in\{0,0.05\}\) and record the Blanchard–Kahn status. Label
   each combination using Leeper's (1991) active/passive terminology.
2. Compare the responses of inflation, the output gap and debt to a transfer shock (1% of quarterly GDP) in the two
   regimes with a unique equilibrium.
3. Why does a debt-financed transfer raise inflation in one regime but not the other?

*File `leeper.mod`*

```matlab
var y_gap pi i b sp z;
varexo eps_z;
parameters beta sigma kappa phi_pi psi bbar rho_z;
beta = 0.99; sigma = 1; kappa = 0.1717; phi_pi = 1.5; psi = 0.05; bbar = 2.4; rho_z = 0.5;
model(linear);
  // TODO: NKPC, IS curve, policy rule, government budget constraint, surplus rule, transfer shock
end;
shocks; var eps_z; stderr 1; end;
stoch_simul(order=1, irf=20, noprint, nograph);
```

*File `ex11_1.m`*

```matlab
dynare leeper noclearall
% TODO: double loop over phi_pi and psi; [info, oo_] = stoch_simul(M_, options_, oo_, {});
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `leeper.mod`*

```matlab
// New Keynesian model with nominal government debt (Leeper, 1991; Woodford, 2001)
var y_gap pi i b sp z;
varexo eps_z;
parameters beta sigma kappa phi_pi psi bbar rho_z;
beta = 0.99; sigma = 1; kappa = 0.1717; phi_pi = 1.5; psi = 0.05; bbar = 2.4; rho_z = 0.5;
model(linear);
  pi = beta * pi(+1) + kappa * y_gap;
  y_gap = y_gap(+1) - (1 / sigma) * (i - pi(+1));
  i = phi_pi * pi;                                              // monetary policy
  b = (1 / beta) * b(-1) + (bbar / beta) * (i(-1) - pi) - sp;   // real debt (share of quarterly GDP)
  sp = psi * b(-1) - z;                                         // primary surplus rule; z = transfers
  z = rho_z * z(-1) + eps_z;
end;
shocks; var eps_z; stderr 1; end;
stoch_simul(order=1, irf=20, noprint, nograph);
```

*File `ex11_1.m`*

```matlab
dynare leeper noclearall
status = {'unique', '', 'no stable solution', 'indeterminate'};
lab = @(pp, ps) [char('passive M' * (pp < 1) + 'active M ' * (pp >= 1)) ' / ' ...
                 char('active F ' * (ps < 1/beta - 1) + 'passive F' * (ps >= 1/beta - 1))];
for pp = [1.5 0.5]
  for ps = [0.05 0]
    set_param_value('phi_pi', pp); set_param_value('psi', ps);
    [info, oo_] = stoch_simul(M_, options_, oo_, {});
    fprintf('phi_pi = %.1f, psi = %.2f  (%s): %s\n', pp, ps, lab(pp, ps), status{info(1) + (info(1) == 0)});
    if info(1) == 0
      ir = oo_.irfs; f = @(v) ir.([v '_eps_z']);
      if isfield(ir, 'pi_eps_z')
        fprintf('    pi (ann.) h=0..3: %s   price level after 20q: %.3f%%\n', sprintf('%7.3f', 4 * f('pi')(1:4)), sum(f('pi')));
        fprintf('    output gap h=0..3: %s\n', sprintf('%7.3f', f('y_gap')(1:4)));
      else
        fprintf('    inflation and output do not respond (Ricardian)\n');
      end
      fprintf('    debt h=0,4,8,19:   %s\n', sprintf('%7.3f', f('b')([1 5 9 20])));
    end
  end
end
```

Output:

```text
phi_pi = 1.5, psi = 0.05  (active M  / passive F): unique
    inflation and output do not respond (Ricardian)
    debt h=0,4,8,19:     1.000  1.705  1.502  0.963
phi_pi = 1.5, psi = 0.00  (active M  / active F ): no stable solution
phi_pi = 0.5, psi = 0.05  (passive M / passive F): indeterminate
phi_pi = 0.5, psi = 0.00  (passive M / active F ): unique
    pi (ann.) h=0..3:   1.453  1.138  0.891  0.698   price level after 20q: 1.663%
    output gap h=0..3:   0.475  0.372  0.291  0.228
    debt h=0,4,8,19:     0.120  0.353  0.152  0.011
```

Only two combinations deliver a unique equilibrium: active monetary with passive fiscal policy (\(\phi_\pi>1\),
\(\psi>1/\beta-1\)), and passive monetary with active fiscal policy (\(\phi_\pi<1\), \(\psi=0\)). With both
policies active there is no stable solution — each tries to pin down the price level — and with both passive the
equilibrium is indeterminate. In the first regime a transfer is financed by higher future surpluses: debt rises and is
then paid down, and inflation and output do not move at all (Ricardian equivalence holds in this model). In the second
regime no future surpluses are promised, so the real value of debt must fall: inflation rises, the price level jumps
and output rises temporarily. That is the fiscal theory of the price level (Leeper, 1991; Sims, 1994; Cochrane, 2023),
one candidate explanation of the 2021–23 inflation after large unfunded transfers.

</details>

### Exercise 11.2 — Inflation and the U.S. debt ratio, 2019–2024

*Data:* `us_quarterly.csv` — gross federal debt (`fed_debt`, billions of dollars), real GDP and the GDP deflator.

The change in the debt-to-GDP ratio \(b_t=D_t/Y_t\) can be written as
\[ \Delta b_t=\underbrace{\frac{D_t-D_{t-1}}{Y_t}}_{\text{new borrowing}}-\underbrace{b_{t-1}\frac{g_t}{1+g_t}}_{\text{nominal growth}}, \]
where \(g_t\) is nominal GDP growth. Split the nominal-growth term into a part due to inflation and a part due to
real growth in proportion to their shares of \(g_t\).

1. Compute the decomposition for each year 2015–2024 using fourth-quarter values (GDP at an annual rate).
2. How much did inflation lower the debt ratio in 2021–2023? Compare it with the new borrowing.
3. Why does surprise inflation reduce the real value of debt, and why would this effect be smaller if debt were
   short-term or inflation-indexed?

*File `ex11_2.m`*

```matlab
T = readtable('us_quarterly.csv');
q4 = mod(T.t, 1) == 0.75; yr = floor(T.t(q4));
D = T.fed_debt(q4);                              % billions
Y = T.rgdp(q4) .* T.gdpdef(q4) / 100;            % nominal GDP, billions at an annual rate
% TODO: b = D./Y; decomposition year by year
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `ex11_2.m`*

```matlab
T = readtable('us_quarterly.csv');
q4 = mod(T.t, 1) == 0.75; yr = floor(T.t(q4));
D = T.fed_debt(q4);                              % billions
Y = T.rgdp(q4) .* T.gdpdef(q4) / 100;            % nominal GDP, billions at an annual rate
P = T.gdpdef(q4); b = 100 * D ./ Y;
fprintf('                          contribution to the change (points of GDP)\nyear   debt/GDP   change   new borrowing   inflation   real growth\n');
for t = find(yr >= 2015 & yr <= 2024)'
  g = Y(t) / Y(t-1) - 1; infl = P(t) / P(t-1) - 1; real = g - infl;
  nb = 100 * (D(t) - D(t-1)) / Y(t); ng = b(t-1) * g / (1 + g);
  fprintf('%d   %6.1f    %+5.1f       %+5.1f        %+5.1f        %+5.1f\n', yr(t), b(t), b(t) - b(t-1), nb, ...
          -ng * infl / g, -ng * real / g);
end
k = yr >= 2021 & yr <= 2023; kk = find(k);
fprintf('2021-23: inflation lowered the ratio by %.1f points in total\n', ...
        sum(arrayfun(@(t) b(t-1) * (Y(t)/Y(t-1) - 1) / (Y(t)/Y(t-1)) * (P(t)/P(t-1) - 1) / (Y(t)/Y(t-1) - 1), kk)));
```

Output:

```text
contribution to the change (points of GDP)
year   debt/GDP   change   new borrowing   inflation   real growth
2015    102.7     +1.9        +4.7         -0.7         -2.1
2016    104.6     +1.8        +5.4         -1.4         -2.2
2017    102.4     -2.2        +2.7         -1.9         -3.0
2018    105.4     +3.1        +7.4         -2.2         -2.1
2019    106.2     +0.8        +5.7         -1.5         -3.4
2020    126.8    +20.6       +21.3         -1.8         +1.1
2021    121.0     -5.8        +8.0         -7.0         -6.9
2022    119.5     -1.5        +7.4         -7.3         -1.6
2023    121.9     +2.4        +9.0         -2.9         -3.7
2024    123.6     +1.7        +7.5         -2.8         -3.0
2021-23: inflation lowered the ratio by 17.1 points in total
```

In 2020 new borrowing raised gross federal debt by about 21% of GDP and the ratio jumped to about 127%. Over
2021–2023 new borrowing added roughly another 24 points, yet the ratio ended 2023 below its 2020 peak: inflation
lowered it by about 17 points in total, and real growth by about 12 more. Surprise inflation devalues
nominal debt whose interest rate was fixed before the inflation arrived. With long maturities the loss is borne by
existing bondholders and the effect on the ratio is large; with short-term or inflation-indexed debt, interest costs
would rise with inflation and most of the erosion would disappear. Gross debt includes debt held by government trust
funds; debt held by the public gives smaller numbers but the same pattern. (The series is reconstructed from FRED-QD's
deflated debt series and the GDP deflator, so values differ slightly from Treasury figures.) Whether this was a
fiscal-theory adjustment or a by-product of supply shocks is still debated.

</details>

## Further reading
- Leeper, E. (1991), "Equilibria under 'Active' and 'Passive' Monetary and Fiscal Policies," *Journal of Monetary Economics*.
- Sargent, T. and N. Wallace (1981), "Some Unpleasant Monetarist Arithmetic," Federal Reserve Bank of Minneapolis *Quarterly Review*.
- Woodford, M. (2001), "Fiscal Requirements for Price Stability," *Journal of Money, Credit and Banking*.
- Cochrane, J. (2023), *The Fiscal Theory of the Price Level*, Princeton University Press.
- Bianchi, F., R. Faccini and L. Melosi (2023), "A Fiscal Theory of Persistent Inflation," *Quarterly Journal of Economics*.
