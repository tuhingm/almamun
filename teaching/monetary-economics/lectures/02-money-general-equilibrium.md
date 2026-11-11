# Lecture 02 — Money in General Equilibrium

**Level:** Graduate | **Field:** Monetary Economics

## Learning objectives
Explain why intrinsically useless money is valued, and build the two workhorse ways of putting money into a dynamic general-equilibrium model: money in the utility function and cash in advance. Derive superneutrality, the Friedman rule and the welfare cost of inflation.

## 1. Why model money explicitly?
In an Arrow–Debreu economy there is no role for money. Money is valued because it overcomes frictions: the absence of a double coincidence of wants, limited commitment and imperfect record keeping (Kiyotaki and Wright, 1989; Kocherlakota, 1998). Search-theoretic models microfound these frictions, but for macroeconomic questions two shortcuts are standard. They deliver a demand for real balances that depends on the nominal interest rate, which is all most policy analysis needs.

## 2. Money in the utility function (Sidrauski, 1967)
A household maximises \(\sum_{t}\beta^tU(c_t,m_t)\), where \(m_t=M_t/P_t\) are real balances, subject to
\[
c_t+k_{t}+m_t+b_t = f(k_{t-1})+(1-\delta)k_{t-1}+\frac{m_{t-1}}{\Pi_t}+\frac{(1+i_{t-1})b_{t-1}}{\Pi_t}+\tau_t ,
\]
with \(\Pi_t=P_t/P_{t-1}\) and lump-sum transfers \(\tau_t\) of newly printed money. The first-order conditions give the Euler equation and a money-demand condition:
\[
U_c(t)=\beta(1+r_t)U_c(t+1),\qquad \frac{U_m(t)}{U_c(t)}=\frac{i_t}{1+i_t}.
\]
The marginal rate of substitution between money and consumption equals the opportunity cost of holding money. With \(U=\frac{c^{1-\sigma}}{1-\sigma}+\chi\frac{m^{1-\nu}}{1-\nu}\) the money demand is \(\log m_t=\text{const}+\frac{\sigma}{\nu}\log c_t-\frac1\nu\log\frac{i_t}{1+i_t}\).

**Superneutrality.** In the steady state with money growth \(\mu\), \(\Pi=1+\mu\), the Euler equation gives \(f'(k)=1/\beta-1+\delta\): the capital stock and consumption do not depend on \(\mu\). Money growth affects only real balances. This result relies on separability between money and consumption and on inelastic labour supply; with non-separable utility or a leisure choice, inflation changes real allocations.

**The Friedman rule.** Real balances are free to produce, so welfare is maximised when their marginal utility is zero: \(i=0\), which requires deflation at the rate of time preference, \(\Pi=\beta\) (Friedman, 1969).

## 3. Cash in advance (Clower, 1967; Lucas and Stokey, 1987)
Alternatively, require purchases of (some) goods to be paid with money carried into the period: \(P_tc_t\le M_{t-1}+\text{transfers}\). In the steady state the household's marginal rate of substitution between leisure and consumption becomes
\[
\frac{U_l}{U_c}=\frac{w}{1+i},
\]
so the nominal interest rate drives a wedge between the real wage and the marginal rate of substitution. Inflation is a tax on activities that use cash (here, consumption earned by working) and households substitute towards leisure. Hours and output fall with inflation — money is not superneutral — and the Friedman rule again eliminates the distortion. With cash and credit goods (Lucas and Stokey), inflation distorts the mix of goods as well.

## 4. The welfare cost of inflation
Bailey (1956) measured the cost of a nominal interest rate \(i\) as the area under the inverse money-demand curve between 0 and \(i\), net of seigniorage transferred back to households:
\[
w(i)=\int_0^{i}m(x)\,dx-i\,m(i).
\]
For semi-log demand \(m(i)=Ae^{-\xi i}\), \(w(i)=\frac{A}{\xi}[1-(1+\xi i)e^{-\xi i}]\); for log-log demand \(m(i)=Ai^{-\eta}\), \(w(i)=A\frac{\eta}{1-\eta}i^{1-\eta}\). Lucas (2000) shows the general-equilibrium welfare cost in a money-in-utility model is well approximated by these formulas. With U.S. M1 demand the cost of 10% inflation relative to zero is around 1% of GDP with the log-log form and smaller with the semi-log form. These "shoe-leather" costs are modest; most arguments for low inflation in modern models come from relative-price distortions under sticky prices (Lectures 4–6), from tax-system non-neutralities and from inflation uncertainty.

## 5. Seigniorage
The government's real revenue from money creation is \(s_t=\frac{M_t-M_{t-1}}{P_t}=\mu_t\frac{m_t}{1+\mu_t}\) in a steady state. Because real balances fall with inflation, seigniorage is a Laffer curve in the inflation rate: revenue first rises and then falls. Cagan's (1956) study of hyperinflations shows that governments often printed money at rates beyond the revenue-maximising rate — a sign that hyperinflations are fiscal and political, not just monetary, phenomena (Lecture 11).

## 6. Computational lab
The coding exercises estimate U.S. money demand, compute Bailey–Lucas welfare triangles, and solve a cash-in-advance model with endogenous labour to show how inflation taxes market work. The model in Exercise 2.2 has a closed-form steady state, \(n=1/(1+\psi R)\), which makes the mechanism transparent.

## Problem set
1. Derive the money-demand condition \(U_m/U_c=i/(1+i)\) from the MIU problem, stating the timing of money holdings.
2. Show that steady-state capital is independent of money growth in the Sidrauski model. Which assumption would you relax to break superneutrality?
3. In the cash-in-advance model with labour, derive the steady-state labour–consumption condition and show that \(n\) falls with \(\mu\).
4. Derive the Bailey welfare cost for semi-log money demand. At what inflation rate is seigniorage maximised?
5. Explain why the Friedman rule is rarely followed in practice. Give at least two reasons from later lectures.

## Coding exercises

These exercises use **MATLAB and Dynare only**. Download the course [data files and helper functions](../index.html#data-and-code)
into your working folder. Every data series is real: U.S. quarterly and monthly macro data (FRED-QD / FRED-MD,
1959–2025), published monetary-policy shock series (Romer–Romer, Jarociński–Karadi, Bu–Rogers–Wu,
Nakamura–Steinsson), World Bank CPI inflation for 190+ countries, and the KRW/USD exchange rate. Dynare models are
`.mod` files run with `dynare file noclearall` (Dynare 6). Try the starter files first, then open the solution. Every
printed result was produced by running the solution files as shown: the `.mod` files were processed by the Dynare
preprocessor and solved at first order, and the MATLAB code was run in GNU Octave.

### Exercise 2.1 — The welfare cost of inflation (Bailey and Lucas)

*Data:* `us_quarterly.csv` — M1/GDP and the T-bill rate, 1959–2007.

With money in the utility function, the welfare cost of a nominal interest rate \(i\) is the area under the inverse
money-demand curve (Bailey, 1956; Lucas, 2000). For semi-log demand \(m(i)=Ae^{-\xi i}\), measured as a share of
income,
\[ w(i)=\frac{A}{\xi}\left[1-(1+\xi i)e^{-\xi i}\right]. \]

1. Estimate \(A\) and \(\xi\) from annual M1/GDP and the T-bill rate over 1959–2007 (rates in decimals).
   (M1 was redefined in May 2020 to include savings deposits, so later data are not comparable.)
2. Compute \(w(i)\) for inflation of 0, 2, 5, 10 and 20% with a real rate of 2%, so \(i=r+\pi\). The Friedman rule
   \(i=0\) gives \(w=0\).
3. Repeat the estimation with M2. Why does the broader aggregate give a very different answer?

*File `ex2_1.m`*

```matlab
T = readtable('us_quarterly.csv');
% TODO: annual averages 1959-2007 of M1/GDP and tbill/100; regress log(m) on [1, i]
% A = exp(b(1)); xi = -b(2);
w = @(i, A, xi) (A / xi) * (1 - (1 + xi * i) .* exp(-xi * i));
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `ex2_1.m`*

```matlab
T = readtable('us_quarterly.csv');
yr = floor(T.t); k = yr <= 2007; yr = yr(k);
py = T.rgdp(k) .* T.gdpdef(k) / 100; i = T.tb3(k) / 100;
years = unique(yr); idx = yr - years(1) + 1;
I = accumarray(idx, i, [], @mean);
for agg = {'m1', 'm2'}
  M = accumarray(idx, T.(agg{1})(k) ./ py, [], @mean);
  b = [ones(numel(M), 1) I] \ log(M); A.(agg{1}) = exp(b(1)); xi.(agg{1}) = -b(2);
  fprintf('%s/GDP = %.3f * exp(%+.2f i)\n', upper(agg{1}), A.(agg{1}), -xi.(agg{1}));
end
w = @(i) (A.m1 / xi.m1) * (1 - (1 + xi.m1 * i) .* exp(-xi.m1 * i));
r = 0.02;
fprintf('\nM1 demand: inflation   nominal rate   welfare cost (%% of GDP)\n');
for p = [0 0.02 0.05 0.10 0.20]
  fprintf('             %4.0f%%        %4.1f%%          %6.3f\n', 100 * p, 100 * (r + p), 100 * w(r + p));
end
fprintf('cost of 10%% vs 2%% inflation: %.2f%% of GDP per year\n', 100 * (w(0.12) - w(0.04)));
```

Output:

```text
M1/GDP = 0.169 * exp(-1.60 i)
M2/GDP = 0.533 * exp(+0.25 i)

M1 demand: inflation   nominal rate   welfare cost (% of GDP)
                0%         2.0%           0.005
                2%         4.0%           0.021
                5%         7.0%           0.061
               10%        12.0%           0.171
               20%        22.0%           0.518
cost of 10% vs 2% inflation: 0.15% of GDP per year
```

With M1 the semi-elasticity is about 1.6, and moving from 2% to 10% inflation costs roughly 0.15% of GDP per
year; 20% inflation costs about half a percent. These are small numbers. Lucas (2000) finds somewhat larger costs with
a log-log specification, which makes money demand very elastic near zero interest rates and so raises the cost of
even moderate inflation. M2 gives a positive coefficient over 1959–2007: most of M2 pays interest that moves with
market rates, so the T-bill rate is not its opportunity cost, and its velocity was driven by financial innovation
rather than by interest rates. The choice of aggregate matters. The general lesson holds anyway: shoe-leather costs of moderate inflation are small, and the case for low inflation rests
more on the relative-price distortions that sticky prices create (Lectures 4–6).

</details>

### Exercise 2.2 — Cash in advance: inflation as a tax on market work

*Data:* Model calibration (no data).

Households maximise \(\sum\beta^t[\log c_t+\psi\log(1-n_t)]\), output is \(y_t=n_t\), and consumption must be paid
with money held from the previous period (cash in advance). In the steady state with money growth \(\mu\) the gross
nominal rate is \(R=(1+\mu)/\beta\), and the labour–consumption condition becomes
\[ \frac{\psi c}{1-n}=\frac{1}{R},\qquad c=n \;\Rightarrow\; n=\frac{1}{1+\psi R}. \]

1. Calibrate \(\psi\) so that \(n=1/3\) at 2% money growth with \(\beta=0.99^4\) (annual).
2. Compute hours, output and the welfare cost relative to the Friedman rule (\(R=1\)) in consumption-equivalent terms
   for \(\mu\in\{\beta-1,0,2\%,5\%,10\%,50\%\}\).
3. Is money superneutral in this model? Explain the mechanism.

*File `ex2_2.m`*

```matlab
beta = 0.99^4;
R = @(mu) (1 + mu) / beta;
% TODO: psi from n = 1/3 at mu = 0.02; n(mu) = 1/(1 + psi*R(mu))
% TODO: consumption-equivalent loss lambda solving log((1-lam) c_FR) + psi log(1-n_FR) = log(c) + psi log(1-n)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `ex2_2.m`*

```matlab
beta = 0.99^4;
R = @(mu) (1 + mu) / beta;
psi = (3 - 1) / R(0.02);                         % n = 1/(1+psi R) = 1/3
n = @(mu) 1 ./ (1 + psi * R(mu));
U = @(mu) log(n(mu)) + psi * log(1 - n(mu));     % c = n
nFR = n(beta - 1);
fprintf('psi = %.3f\n', psi);
fprintf('money growth   R      hours   output vs FR   welfare cost (%% of consumption)\n');
for mu = [beta - 1, 0, 0.02, 0.05, 0.10, 0.50]
  lam = 1 - exp(U(mu) - U(beta - 1));          % log((1-lam) c_FR) + psi log(1-n_FR) = U(mu)
  fprintf('  %6.2f%%    %5.3f   %.4f    %6.2f%%        %6.3f\n', 100 * mu, R(mu), n(mu), ...
          100 * (n(mu) / nFR - 1), 100 * lam);
end
```

Output:

```text
psi = 1.884
money growth   R      hours   output vs FR   welfare cost (% of consumption)
   -3.94%    1.000   0.3468      0.00%         0.000
    0.00%    1.041   0.3377     -2.61%         0.053
    2.00%    1.062   0.3333     -3.88%         0.117
    5.00%    1.093   0.3269     -5.73%         0.256
   10.00%    1.145   0.3168     -8.66%         0.589
   50.00%    1.562   0.2537    -26.84%         5.972
```

Money is not superneutral here. Consumption needs cash, and holding cash costs the nominal interest rate, while
leisure does not need cash. Higher money growth raises \(R\) and so acts as a tax on market activity, and households
substitute towards leisure: hours and output fall steadily with inflation. The Friedman rule (\(\mu=\beta-1\), so
\(R=1\)) removes the wedge and gives the highest welfare. Moving from the Friedman rule to 2% money growth costs
about 0.1% of consumption, 10% money growth about 0.6%, and 50% money growth about 6%. The same wedge appears in money-in-utility models
whenever money and consumption are not separable in utility. With separable preferences and inelastic labour,
steady-state real variables would not depend on \(\mu\) (Sidrauski superneutrality).

</details>

## Further reading
- Sidrauski, M. (1967), "Rational Choice and Patterns of Growth in a Monetary Economy," *American Economic Review*.
- Lucas, R. E. and N. Stokey (1987), "Money and Interest in a Cash-in-Advance Economy," *Econometrica*.
- Lucas, R. E. (2000), "Inflation and Welfare," *Econometrica*.
- Walsh, C., *Monetary Theory and Policy*, ch. 2–3.
- Galí, J. (2015), *Monetary Policy, Inflation, and the Business Cycle*, ch. 2 (appendix on money in the utility function).
