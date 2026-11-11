# Lecture 09 — Financial Frictions and the Credit Channel

**Level:** Graduate | **Field:** Monetary Economics

## Learning objectives
Explain why the external finance premium depends on borrower net worth, derive the financial accelerator, and understand bank-capital and risk-taking channels of monetary policy. Build a reduced-form financial accelerator in Dynare and confront it with evidence on credit spreads.

## 1. Why financial frictions matter for monetary policy
In the New Keynesian model all agents borrow and lend at the policy rate. In reality firms and households borrow at rates that include a spread, and that spread moves with the business cycle and with policy. Bernanke and Gertler (1995) distinguish two credit channels: the **balance-sheet channel** (tighter policy lowers borrowers' net worth and collateral values, raising the premium they pay) and the **bank-lending channel** (tighter policy reduces the supply of loans by banks that cannot easily replace deposits). Both amplify and propagate the effects of interest-rate changes.

## 2. Costly state verification and the external finance premium
Entrepreneurs with net worth \(N\) borrow \(QK-N\) to buy capital. The return on capital is subject to idiosyncratic risk observable to lenders only at a monitoring cost (Townsend, 1979). The optimal contract is standard debt, and the equilibrium return on capital satisfies
\[
E_t R^k_{t+1}=s\!\left(\frac{N_t}{Q_tK_t}\right)R_t,\qquad s'(\cdot)<0 ,
\]
where \(s(\cdot)\) is the **external finance premium**. The premium falls when borrowers have more skin in the game (Bernanke, Gertler and Gilchrist, 1999, "BGG").

## 3. The financial accelerator
Net worth evolves with the realised return on capital:
\[
N_{t+1}=\gamma\left[R^k_{t+1}Q_tK_t-E_tR^k_{t+1}(Q_tK_t-N_t)\right]+\text{transfers}.
\]
A monetary tightening lowers asset prices \(Q_t\) and returns, which lowers net worth, which raises the premium, which lowers investment and asset prices further. Because net worth is rebuilt only slowly from retained earnings, the effects are persistent. In BGG's calibration the accelerator raises the initial output response to a policy shock by about half and makes it markedly more persistent. A reduced form captures the mechanism in the three-equation model: let spending depend on the borrowing rate \(i_t+s_t\) and let the spread fall with net worth,
\[
\tilde y_t=E_t\tilde y_{t+1}-\tfrac1\sigma(i_t+s_t-E_t\pi_{t+1}-r^n_t),\qquad s_t=-\chi\,nw_t,\qquad nw_t=\rho_nnw_{t-1}+\tilde y_t .
\]
Exercise 9.1 shows that the impact response barely changes but the cumulative output loss rises by a third to two-thirds.

## 4. Banks and the risk-taking channel
Gertler and Karadi (2011) and Gertler and Kiyotaki (2010) model banks whose ability to raise deposits is limited by an agency problem: depositors lend only if the bank's franchise value exceeds what it could divert, \(V_t\ge\lambda Q_tS_t\). Bank leverage is therefore tied to bank capital, and losses on bank assets force deleveraging and fire sales. This framework explains how the 2008 crisis originated in the financial sector, and it rationalises central-bank credit policy: when private intermediaries are constrained, central-bank purchases of private assets can lower spreads (Lecture 10). The **risk-taking channel** (Borio and Zhu, 2012) adds that low rates encourage banks and investors to take more risk, raising asset prices and leverage — a potential cost of keeping rates low for long.

## 5. Evidence
- Credit spreads predict economic activity. Gilchrist and Zakrajšek (2012) decompose corporate bond spreads into expected default and an **excess bond premium (EBP)**; the EBP has strong predictive power for output and rises in financial crises.
- Gertler and Karadi (2015) use high-frequency monetary surprises as external instruments in a VAR and find that a tightening raises the EBP and mortgage spreads, beyond what the expected path of short rates implies.
- Bank-level data show stronger lending responses for banks with weaker balance sheets (Kashyap and Stein, 2000).
- Raw index spreads such as Baa–Treasury are noisier measures; Exercise 9.2 shows the difficulty of finding a precise response in them.

## 6. Policy implications
With financial frictions, the welfare-relevant gap includes credit conditions, and optimal policy may respond to spreads (Cúrdia and Woodford, 2010). Whether monetary policy should "lean against the wind" of credit booms is debated: Svensson (2017) argues the costs of leaning exceed the benefits in most cases and that macroprudential tools should be the first line of defence.

## Problem set
1. Derive the external finance premium in a two-state costly-state-verification model and show it falls with net worth.
2. Explain the financial-accelerator loop step by step after a monetary tightening.
3. In the reduced form of Section 3, why does \(\chi\) affect the cumulative response more than the impact response?
4. Explain the incentive constraint in the Gertler–Karadi model and why it links bank leverage to bank capital.
5. Why is the excess bond premium a better measure of the credit channel than the raw Baa spread?

## Coding exercises

These exercises use **MATLAB and Dynare only**. Download the course [data files and helper functions](../index.html#data-and-code)
into your working folder. Every data series is real: U.S. quarterly and monthly macro data (FRED-QD / FRED-MD,
1959–2025), published monetary-policy shock series (Romer–Romer, Jarociński–Karadi, Bu–Rogers–Wu,
Nakamura–Steinsson), World Bank CPI inflation for 190+ countries, and the KRW/USD exchange rate. Dynare models are
`.mod` files run with `dynare file noclearall` (Dynare 6). Try the starter files first, then open the solution. Every
printed result was produced by running the solution files as shown: the `.mod` files were processed by the Dynare
preprocessor and solved at first order, and the MATLAB code was run in GNU Octave.

### Exercise 9.1 — A financial accelerator in the New Keynesian model (Dynare)

*Data:* Model calibration; 25 bp monetary-policy shock.

Add a credit spread to the IS curve: spending depends on the borrowing rate \(i_t+s_t\), and the spread falls with
borrower net worth, \(s_t=-\chi\,nw_t\), where \(nw_t=\rho_n nw_{t-1}+\tilde y_t\).

1. Compare the responses to a 25 bp monetary tightening for \(\chi\in\{0,0.05,0.1\}\): impact and cumulative
   output gap, inflation and the spread.
2. Explain the amplification and the persistence it adds. Which features of Bernanke, Gertler and Gilchrist (1999) does
   this reduced form capture, and which does it miss?

*File `fa.mod`*

```matlab
var y_gap pi i s nw v;
varexo eps_v;
parameters beta sigma kappa phi_pi phi_y chi rho_n rho_v;
beta = 0.99; sigma = 1; kappa = 0.1717; phi_pi = 1.5; phi_y = 0.125; chi = 0.05; rho_n = 0.9; rho_v = 0.5;
model(linear);
  // TODO: NKPC, IS curve with the borrowing rate i + s, spread, net worth, Taylor rule, shock
end;
shocks; var eps_v; stderr 0.25; end;
stoch_simul(order=1, irf=20, noprint, nograph);
```

*File `ex9_1.m`*

```matlab
dynare fa noclearall
% TODO: loop over chi with set_param_value; report impact and cumulative responses
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `fa.mod`*

```matlab
// New Keynesian model with a reduced-form financial accelerator:
// households and firms borrow at i + s, the spread s falls when borrower net worth nw rises
var y_gap pi i s nw v;
varexo eps_v;
parameters beta sigma kappa phi_pi phi_y chi rho_n rho_v;
beta = 0.99; sigma = 1; kappa = 0.1717; phi_pi = 1.5; phi_y = 0.125; chi = 0.05; rho_n = 0.9; rho_v = 0.5;
model(linear);
  pi = beta * pi(+1) + kappa * y_gap;
  y_gap = y_gap(+1) - (1 / sigma) * (i + s - pi(+1));     // spending depends on the borrowing rate
  s = -chi * nw;                                         // external finance premium
  nw = rho_n * nw(-1) + y_gap;                           // net worth accumulates profits
  i = phi_pi * pi + phi_y * y_gap + v;
  v = rho_v * v(-1) + eps_v;
end;
shocks; var eps_v; stderr 0.25; end;
stoch_simul(order=1, irf=20, noprint, nograph);
```

*File `ex9_1.m`*

```matlab
dynare fa noclearall
fprintf('chi    output gap: impact   cum(0-19)   trough qtr   infl (ann.) impact   spread (ann.) peak\n');
for c = [0 0.05 0.1]
  set_param_value('chi', c);
  [info, oo_] = stoch_simul(M_, options_, oo_, {});
  ir = oo_.irfs; [mn, q] = min(ir.y_gap_eps_v);
  sp = 0; if isfield(ir, 's_eps_v'), sp = 4 * max(ir.s_eps_v); end
  fprintf('%4.2f        %7.3f      %7.3f        %2d            %7.3f             %6.3f\n', c, ...
          ir.y_gap_eps_v(1), sum(ir.y_gap_eps_v), q - 1, 4 * ir.pi_eps_v(1), sp);
end
```

Output:

```text
chi    output gap: impact   cum(0-19)   trough qtr   infl (ann.) impact   spread (ann.) peak
0.00         -0.259       -0.518         0             -0.352              0.000
0.05         -0.263       -0.703         0             -0.504              0.084
0.10         -0.259       -0.852         0             -0.673              0.174
```

With \(\chi=0\) the model is the basic New Keynesian model and the output gap recovers as fast as the policy
shock decays. With a financial accelerator, the tightening lowers output and profits, borrower net worth falls, and
the external finance premium rises. The higher borrowing cost depresses spending further, so the cumulative output
loss is 35–65% larger and the recovery slower, because net worth is rebuilt only gradually. Inflation falls more too.
The impact response barely changes, because net worth is a stock that erodes over time. This captures the
core BGG mechanism: credit conditions amplify and propagate shocks through balance sheets. It misses the explicit
contract and default problem, the role of asset prices and leverage, and the possibility that shocks start in the
financial sector itself (Gertler–Karadi, 2011) — as in 2008.

</details>

### Exercise 9.2 — How do credit spreads respond to monetary policy surprises?

*Data:* `mp_shocks_monthly.csv` (Jarociński–Karadi monetary-policy shocks, 1990–2024) and `us_monthly.csv` (Baa and 10-year Treasury yields, 1-year Treasury yield).

Use local projections (Jordà, 2005) to estimate the response of the Baa–Treasury spread to monetary-policy surprises:
\[ y_{t+h}-y_{t-1}=a_h+\beta_h\,\text{shock}_t+\sum_{j=1}^{3}\gamma_{h,j}\Delta y_{t-j}+u_{t+h}. \]

1. Scale the Jarociński–Karadi "pure monetary policy" shock so that it raises the 1-year Treasury yield by 25 bp on
   impact (estimate the \(h=0\) response of the 1-year yield first).
2. Estimate \(\beta_h\) for \(h=0,\dots,24\) months for the 1-year yield and the Baa spread, 1990–2019, with
   Newey–West standard errors.
3. Does a monetary tightening widen credit spreads, as the financial-accelerator mechanism predicts?

*File `ex9_2.m`*

```matlab
D = readtable('us_monthly.csv'); S = readtable('mp_shocks_monthly.csv');
% TODO: align by year and month; spread = baa - gs10; loop h = 0:24; ols_nw with L = h + 1
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `ex9_2.m`*

```matlab
D = readtable('us_monthly.csv'); S = readtable('mp_shocks_monthly.csv');
key = @(y, m) 12 * y + m;
[~, ia, ib] = intersect(key(D.year, D.month), key(S.year, S.month));
yr = D.year(ia); shock = S.jk_mp(ib);
series = {D.gs1(ia), D.baa(ia) - D.gs10(ia)}; names = {'1-year yield', 'Baa spread  '};
H = 24; T = numel(yr); B = zeros(2, H+1); SE = B;
for v = 1:2
  y = series{v}; dy = [NaN; diff(y)];
  for h = 0:H
    lhs = NaN(T, 1); lhs(1:T-h) = y(1+h:T) - [NaN; y(1:T-h-1)];
    X = [ones(T, 1) shock [NaN; dy(1:end-1)] [NaN(2,1); dy(1:end-2)] [NaN(3,1); dy(1:end-3)]];
    k = yr >= 1990 & yr <= 2019 & all(~isnan([lhs X]), 2);
    [b, se] = ols_nw(lhs(k), X(k, :), h + 1); B(v, h+1) = b(2); SE(v, h+1) = se(2);
  end
end
scale = 0.25 / B(1, 1);                       % shock size that raises the 1-year yield by 25 bp on impact
fprintf('scale: a shock of %.3f units raises the 1-year yield by 25 bp\n', scale);
fprintf('h (months)   %s\n', sprintf('%7d', [0 3 6 12 18 24]));
for v = 1:2
  fprintf('%s %s\n', names{v}, sprintf('%7.3f', scale * B(v, [0 3 6 12 18 24] + 1)));
  fprintf('   (s.e.)    %s\n', sprintf('%7.3f', abs(scale) * SE(v, [0 3 6 12 18 24] + 1)));
end
```

Output:

```text
scale: a shock of 0.373 units raises the 1-year yield by 25 bp
h (months)         0      3      6     12     18     24
1-year yield   0.250  0.354  0.662  0.948  1.226  1.248
   (s.e.)      0.083  0.223  0.232  0.470  0.577  0.602
Baa spread    -0.010  0.192  0.114  0.124 -0.105  0.253
   (s.e.)      0.053  0.151  0.208  0.210  0.233  0.230
```

The 1-year yield rises by 25 bp by construction and keeps rising for about a year, although the later
responses are imprecise. The raw Baa–Treasury spread shows no clear response: the point estimates are mostly positive
but within one or two standard errors of zero. This is a useful negative result. Gertler and Karadi (2015) find a
significant rise in the excess bond premium, a cleaner measure of credit risk premia built from bond-level data that
removes expected default; the Baa index spread also moves with liquidity, duration and default expectations, which
adds noise. The Jarociński–Karadi shocks separate pure policy surprises from central-bank information effects using
the stock-market response, which removes one bias but not the noise. Detecting the financial-accelerator channel
needs good measures of balance-sheet strength and risk premia, not just more precise shocks.

</details>

## Further reading
- Bernanke, B., M. Gertler and S. Gilchrist (1999), "The Financial Accelerator in a Quantitative Business Cycle Framework," *Handbook of Macroeconomics*.
- Gertler, M. and P. Karadi (2011), "A Model of Unconventional Monetary Policy," *Journal of Monetary Economics*.
- Gertler, M. and P. Karadi (2015), "Monetary Policy Surprises, Credit Costs, and Economic Activity," *American Economic Journal: Macroeconomics*.
- Gilchrist, S. and E. Zakrajšek (2012), "Credit Spreads and Business Cycle Fluctuations," *American Economic Review*.
- Cúrdia, V. and M. Woodford (2010), "Credit Spreads and Monetary Policy," *Journal of Money, Credit and Banking*.
