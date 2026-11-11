# Lecture 03 — The Classical Monetary Model

**Level:** Graduate | **Field:** Monetary Economics

## Learning objectives
Solve a monetary model with flexible prices, show monetary neutrality, and study how the price level is determined under money-supply rules and interest-rate rules. Understand price-level indeterminacy under an interest-rate peg and the role of the Taylor principle. Implement the model in Dynare.

## 1. Households and firms
Following Galí (2015, ch. 2), a representative household maximises \(E_0\sum\beta^t\left(\frac{C_t^{1-\sigma}}{1-\sigma}-\frac{N_t^{1+\varphi}}{1+\varphi}\right)\) subject to \(P_tC_t+Q_tB_t\le B_{t-1}+W_tN_t+D_t\), where \(Q_t=1/(1+i_t)\) is the price of a one-period nominal bond. The optimality conditions are
\[
\frac{W_t}{P_t}=C_t^{\sigma}N_t^{\varphi},\qquad Q_t=\beta E_t\left[\left(\frac{C_{t+1}}{C_t}\right)^{-\sigma}\frac{P_t}{P_{t+1}}\right].
\]
A representative competitive firm produces \(Y_t=A_tN_t^{1-\alpha}\) and hires labour until \(W_t/P_t=(1-\alpha)A_tN_t^{-\alpha}\). Goods-market clearing requires \(Y_t=C_t\).

## 2. Log-linear equilibrium and neutrality
Log-linearising around the steady state (lower-case letters denote log deviations) gives
\[
\begin{aligned}
w_t-p_t&=\sigma y_t+\varphi n_t, & w_t-p_t&=a_t-\alpha n_t,\\
y_t&=a_t+(1-\alpha)n_t, & y_t&=E_ty_{t+1}-\tfrac1\sigma(i_t-E_t\pi_{t+1}-\rho).
\end{aligned}
\]
The first three equations determine employment, output and the real wage as functions of technology alone:
\[
n_t=\psi_{na}a_t,\qquad y_t=\psi_{ya}a_t,\qquad \psi_{ya}=\frac{1+\varphi}{\sigma(1-\alpha)+\varphi+\alpha}.
\]
The Euler equation then determines the real interest rate, \(r_t\equiv i_t-E_t\pi_{t+1}=\rho+\sigma E_t\Delta y_{t+1}\). Real variables are independent of monetary policy: this is **monetary neutrality**. Monetary policy determines only nominal variables — the price level, inflation and the nominal rate — through the Fisher equation \(i_t=r_t+E_t\pi_{t+1}\).

## 3. Price-level determination under a money rule
Add a money-demand equation \(m_t-p_t=y_t-\eta i_t\) and an exogenous process for money growth \(\Delta m_t=\rho_m\Delta m_{t-1}+\varepsilon^m_t\). Combining money demand with the Fisher equation gives a forward-looking difference equation for the price level,
\[
p_t=\frac{\eta}{1+\eta}E_tp_{t+1}+\frac{1}{1+\eta}m_t+u_t,
\]
where \(u_t\) collects real factors. Because \(\eta/(1+\eta)<1\), solving forward gives a unique stable solution: the price level is the discounted value of expected future money supplies. With AR(1) money growth the nominal rate is
\[
i_t=\rho+\frac{\rho_m}{1+\eta(1-\rho_m)}\Delta m_t .
\]
A monetary expansion with \(\rho_m>0\) *raises* the nominal rate because it raises expected inflation. There is no liquidity effect. Inflation rises by more than money growth on impact: higher expected inflation lowers money demand, so prices must jump further to clear the money market.

## 4. Interest-rate rules and determinacy
Suppose instead the central bank sets \(i_t=\rho+\phi_\pi\pi_t+v_t\). Combining with the Fisher equation,
\[
E_t\pi_{t+1}=\phi_\pi\pi_t+v_t-\hat r_t .
\]
If \(\phi_\pi>1\) the only stationary solution is \(\pi_t=\sum_{k\ge0}\phi_\pi^{-(k+1)}E_t(\hat r_{t+k}-v_{t+k})\): inflation is uniquely determined. If \(\phi_\pi<1\), including an interest-rate peg \(\phi_\pi=0\), any process \(\pi_{t+1}=\phi_\pi\pi_t+v_t-\hat r_t+\xi_{t+1}\) with an arbitrary martingale-difference "sunspot" \(\xi_{t+1}\) is an equilibrium. The price level is **indeterminate** (Sargent and Wallace, 1975). The condition \(\phi_\pi>1\) is the simplest form of the **Taylor principle**. McCallum (1981) pointed out that pegs can deliver determinacy if the rule responds to some nominal anchor, and the fiscal theory of the price level (Lecture 11) offers another way out.

## 5. Implementation in Dynare
The model is written in `model(linear)` form with real balances \(l_t=m_t-p_t\) as a stationary variable:

```matlab
model(linear);
  y = a + (1 - alpha) * n;          // production
  w = sigma * y + phi * n;          // labour supply
  w = a - alpha * n;                // labour demand
  r = sigma * (y(+1) - y);          // real rate from the Euler equation
  i = r + pi(+1);                   // Fisher equation
  l = y - eta * i;                  // money demand
  l = l(-1) + dm - pi;              // real balances
  dm = rho_m * dm(-1) + eps_m;
  a = rho_a * a(-1) + eps_a;
end;
```

Dynare checks the Blanchard–Kahn conditions: the number of eigenvalues outside the unit circle must equal the number of forward-looking variables. With an interest-rate rule and \(\phi_\pi<1\), Dynare reports indeterminacy.

## Problem set
1. Derive \(\psi_{ya}\) and the natural real rate \(r^n_t=\rho-\sigma(1-\rho_a)\psi_{ya}a_t\) for an AR(1) technology process.
2. Solve the price-level equation forward under a money rule and derive the formula for \(i_t\) in Section 3.
3. Show that with \(\phi_\pi<1\) the inflation process admits sunspot solutions. Write one explicitly.
4. Explain McCallum's (1981) critique of the Sargent–Wallace result.
5. In the classical model, what is the optimal monetary policy? Relate your answer to the Friedman rule of Lecture 2.

## Coding exercises

These exercises use **MATLAB and Dynare only**. Download the course [data files and helper functions](../index.html#data-and-code)
into your working folder. Every data series is real: U.S. quarterly and monthly macro data (FRED-QD / FRED-MD,
1959–2025), published monetary-policy shock series (Romer–Romer, Jarociński–Karadi, Bu–Rogers–Wu,
Nakamura–Steinsson), World Bank CPI inflation for 190+ countries, and the KRW/USD exchange rate. Dynare models are
`.mod` files run with `dynare file noclearall` (Dynare 6). Try the starter files first, then open the solution. Every
printed result was produced by running the solution files as shown: the `.mod` files were processed by the Dynare
preprocessor and solved at first order, and the MATLAB code was run in GNU Octave.

### Exercise 3.1 — Monetary neutrality in the classical model (Dynare)

*Data:* Model calibration (Galí, 2015, ch. 2).

Implement the log-linear classical monetary model with exogenous money growth \(\Delta m_t=\rho_m\Delta m_{t-1}+
\varepsilon^m_t\), written in terms of real balances \(l_t=m_t-p_t\).

1. Run the model and plot (or print) the responses of output, the real rate, inflation and the nominal rate to a 1%
   money-growth shock.
2. Verify neutrality: which variables do not move at all?
3. Galí shows that \(i_t=\rho+\frac{\rho_m}{1+\eta(1-\rho_m)}\Delta m_t\). Check the impact response of \(i\)
   against this formula. Why does the nominal rate rise, not fall, after a monetary expansion when \(\rho_m>0\)?

*File `classical.mod`*

```matlab
var y n w r i pi l dm a;
varexo eps_a eps_m;
parameters sigma phi alpha eta rho_a rho_m;
sigma = 1; phi = 1; alpha = 1/3; eta = 4; rho_a = 0.9; rho_m = 0.5;
model(linear);
  // TODO: production, labour supply and demand, Euler, Fisher, money demand, real balances, shocks
end;
shocks; var eps_a; stderr 1; var eps_m; stderr 1; end;
stoch_simul(order=1, irf=8, noprint, nograph);
```

*File `ex3_1.m`*

```matlab
dynare classical noclearall
% responses are stored in oo_.irfs.<variable>_<shock>, e.g. oo_.irfs.pi_eps_m
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `classical.mod`*

```matlab
// Classical monetary model (Gali, 2015, ch. 2), log-linear, exogenous money growth
var y n w r i pi l dm a;           // l = m - p, real balances
varexo eps_a eps_m;
parameters sigma phi alpha eta rho_a rho_m;
sigma = 1; phi = 1; alpha = 1/3; eta = 4; rho_a = 0.9; rho_m = 0.5;
model(linear);
  y = a + (1 - alpha) * n;                   // production
  w = sigma * y + phi * n;                   // labour supply (real wage)
  w = a - alpha * n;                         // labour demand
  r = sigma * (y(+1) - y);                   // Euler equation: real rate
  i = r + pi(+1);                            // Fisher equation
  l = y - eta * i;                           // money demand
  l = l(-1) + dm - pi;                       // law of motion of real balances
  dm = rho_m * dm(-1) + eps_m;               // money growth
  a = rho_a * a(-1) + eps_a;                 // technology
end;
shocks; var eps_a; stderr 1; var eps_m; stderr 1; end;
stoch_simul(order=1, irf=8, noprint, nograph);
```

*File `ex3_1.m`*

```matlab
dynare classical noclearall
names = {'y', 'n', 'w', 'r', 'i', 'pi', 'l'};
fprintf('Responses to a 1%% money-growth shock (rho_m = %.1f)\n', rho_m);
fprintf('%-4s %s\n', 'var', sprintf('   h=%d  ', 0:4));
for k = 1:numel(names)
  f = [names{k} '_eps_m'];
  if isfield(oo_.irfs, f), fprintf('%-4s %s\n', names{k}, sprintf('%8.4f ', oo_.irfs.(f)(1:5)));
  else, fprintf('%-4s    does not respond (neutral)\n', names{k}); end
end
fprintf('\nimpact on i: Dynare %.4f, formula rho_m/(1+eta(1-rho_m)) = %.4f\n', ...
        oo_.irfs.i_eps_m(1), rho_m / (1 + eta * (1 - rho_m)));
fprintf('price level after 5 quarters: %.3f%% (money stock: %.3f%%)\n', ...
        sum(oo_.irfs.pi_eps_m(1:5)), sum(rho_m .^ (0:4)));
```

Output:

```text
Responses to a 1% money-growth shock (rho_m = 0.5)
var     h=0     h=1     h=2     h=3     h=4
y       does not respond (neutral)
n       does not respond (neutral)
w       does not respond (neutral)
r       does not respond (neutral)
i      0.1667   0.0833   0.0417   0.0208   0.0104
pi     1.6667   0.1667   0.0833   0.0417   0.0208
l     -0.6667  -0.3333  -0.1667  -0.0833  -0.0417

impact on i: Dynare 0.1667, formula rho_m/(1+eta(1-rho_m)) = 0.1667
price level after 5 quarters: 1.979% (money stock: 1.938%)
```

Output, hours, the real wage and the real interest rate do not respond at all: in the classical model money is
neutral, and real variables are pinned down by technology and preferences alone. Inflation jumps by more than money
growth on impact because higher expected money growth raises expected inflation and the nominal rate, which lowers
desired real balances, so prices must rise even faster. The nominal rate rises by exactly
\(\rho_m/(1+\eta(1-\rho_m))\): with persistent money growth the expected-inflation (Fisher) effect dominates and there
is no liquidity effect. Generating a fall in interest rates after a monetary expansion requires nominal rigidities
(Lectures 4–5) or limited participation.

</details>

### Exercise 3.2 — Interest-rate pegs and price-level determinacy (Dynare)

*Data:* Model calibration.

Replace the money rule with an interest-rate rule \(i_t=\phi_\pi\pi_t+v_t\). With the Fisher equation
\(i_t=r_t+E_t\pi_{t+1}\) and an exogenous real rate, inflation satisfies
\(E_t\pi_{t+1}=\phi_\pi\pi_t+v_t-r_t\).

1. Write the model in Dynare and loop over \(\phi_\pi\in\{0,0.5,0.95,1.05,1.5,3\}\) with `set_param_value` and
   `stoch_simul`, recording the Blanchard–Kahn status.
2. When the equilibrium is unique, compare the impact response of inflation to a policy shock with
   \(-1/(\phi_\pi-\rho_v)\).
3. Explain Sargent and Wallace's (1975) indeterminacy result for an interest-rate peg.

*File `peg.mod`*

```matlab
var pi i r v;
varexo eps_r eps_v;
parameters phi_pi rho_r rho_v;
phi_pi = 1.5; rho_r = 0.8; rho_v = 0.5;
model(linear);
  // TODO: Fisher equation, policy rule, AR(1) real rate and policy shock
end;
shocks; var eps_r; stderr 1; var eps_v; stderr 0.25; end;
stoch_simul(order=1, irf=8, noprint, nograph);
```

*File `ex3_2.m`*

```matlab
dynare peg noclearall
for phi = [0 0.5 0.95 1.05 1.5 3]
  set_param_value('phi_pi', phi);
  [info, oo_] = stoch_simul(M_, options_, oo_, {});
  % TODO: info(1) == 0 unique, 3 no stable solution, 4 indeterminacy
end
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `peg.mod`*

```matlab
var pi i r v;
varexo eps_r eps_v;
parameters phi_pi rho_r rho_v;
phi_pi = 1.5; rho_r = 0.8; rho_v = 0.5;
model(linear);
  i = r + pi(+1);                 // Fisher equation
  i = phi_pi * pi + v;            // interest-rate rule
  r = rho_r * r(-1) + eps_r;      // real rate (from the real side of the economy)
  v = rho_v * v(-1) + eps_v;      // policy shock
end;
shocks; var eps_r; stderr 1; var eps_v; stderr 0.25; end;
stoch_simul(order=1, irf=8, noprint, nograph);
```

*File `ex3_2.m`*

```matlab
dynare peg noclearall
status = {'unique', '', 'no stable solution', 'indeterminate'};
fprintf('phi_pi   equilibrium          pi impact (eps_v)   -sd/(phi-rho_v)\n');
for phi = [0 0.5 0.95 1.05 1.5 3]
  set_param_value('phi_pi', phi);
  [info, oo_] = stoch_simul(M_, options_, oo_, {});
  if info(1) == 0
    fprintf('%5.2f    %-18s   %8.4f            %8.4f\n', phi, status{1}, oo_.irfs.pi_eps_v(1), -0.25 / (phi - rho_v));
  else
    fprintf('%5.2f    %-18s\n', phi, status{info(1)});
  end
end
```

Output:

```text
phi_pi   equilibrium          pi impact (eps_v)   -sd/(phi-rho_v)
 0.00    indeterminate
 0.50    indeterminate
 0.95    indeterminate
 1.05    unique                -0.4545             -0.4545
 1.50    unique                -0.2500             -0.2500
 3.00    unique                -0.1000             -0.1000
```

For \(\phi_\pi<1\), including a pure peg (\(\phi_\pi=0\)), Dynare reports indeterminacy: the stable root
condition fails and any inflation path satisfying \(E_t\pi_{t+1}=\phi_\pi\pi_t+\dots\) is an equilibrium, so self-fulfilling
expectations can move prices. This is the Sargent–Wallace result: an interest-rate peg pins down expected inflation
but not the price level. For \(\phi_\pi>1\) the solution is unique and inflation responds to a policy shock by
exactly \(-1/(\phi_\pi-\rho_v)\) times the shock. A stronger response to inflation means smaller equilibrium
movements in inflation. This "Taylor principle" carries over to the New Keynesian model (Lecture 5).

</details>

## Further reading
- Galí, J. (2015), *Monetary Policy, Inflation, and the Business Cycle*, 2nd ed., ch. 2.
- Sargent, T. and N. Wallace (1975), "'Rational' Expectations, the Optimal Monetary Instrument, and the Optimal Money Supply Rule," *Journal of Political Economy*.
- McCallum, B. (1981), "Price Level Determinacy with an Interest Rate Policy Rule and Rational Expectations," *Journal of Monetary Economics*.
- Woodford, M. (2003), *Interest and Prices*, ch. 2.
