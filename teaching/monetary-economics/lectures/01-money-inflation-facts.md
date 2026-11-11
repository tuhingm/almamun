# Lecture 01 — Money, Inflation and Transmission: The Facts

**Level:** Graduate | **Field:** Monetary Economics

## Learning objectives
Know the long-run relation between money and inflation and the short-run facts about monetary transmission that any model must explain. Understand the quantity equation, money demand and the opportunity cost of money, and why central banks moved from monetary aggregates to interest rates.

## 1. The quantity equation
For any economy, nominal spending can be written as money times velocity:
\[
M_tV_t=P_tY_t \quad\Longrightarrow\quad \pi_t=\Delta m_t+\Delta v_t-\Delta y_t ,
\]
where lower-case letters are logs. The equation is an identity; it becomes a theory, the **quantity theory of money**, when velocity is stable or predictable and real output is determined by real factors in the long run. Then sustained money growth in excess of output growth shows up one-for-one in inflation. Across countries and over long periods this is one of the best-established facts in macroeconomics: McCandless and Weber (1995) report correlations between money growth and inflation close to one across 110 countries over 30 years. Hyperinflations are always episodes of extreme money growth, usually to finance government deficits.

In low-inflation economies the link is much weaker over horizons of a few years, because velocity shocks are large. Deregulation of deposit rates, money-market funds and sweep accounts changed the demand for M1 and M2 in the 1980s and 1990s; after 2008 and again in 2020, near-zero interest rates made households and firms willing to hold very large money balances.

## 2. Money demand and the opportunity cost of money
Money pays no interest (or less than bonds), so holding it has an opportunity cost equal to the nominal interest rate \(i_t\) (or the spread between \(i_t\) and the own rate on money). Standard specifications are
\[
\log\frac{M_t}{P_tY_t}=a-\xi\, i_t \quad\text{(semi-log, Cagan 1956)},\qquad \log\frac{M_t}{P_tY_t}=a-\eta\log i_t\quad\text{(log-log)} .
\]
The semi-elasticity \(\xi\) measures by how much real balances relative to income fall when the interest rate rises by one point. Lucas (2000) shows that the log-log form fits long U.S. samples of M1 well and implies very elastic money demand near zero rates. Two consequences follow. First, a central bank that targets money must forecast velocity shocks; when \(\xi\) shifts, a money target delivers volatile interest rates and inflation. Second, near the zero lower bound money and short-term bonds become close substitutes, which is the starting point for the analysis of liquidity traps (Lecture 10).

## 3. Short-run facts about monetary transmission
Identified monetary-policy shocks (Lecture 14) deliver a consistent set of facts for the United States and other advanced economies (Christiano, Eichenbaum and Evans, 1999; Ramey, 2016):

1. **Output and employment fall** after a tightening, gradually, with a trough after one to two years.
2. **Prices respond sluggishly**; the price level starts to fall only after a year or more. Early VARs even showed a temporary rise (the "price puzzle").
3. **Short-term nominal and real rates rise together**: because prices are sticky, a higher nominal rate is also a higher real rate — the *liquidity effect*.
4. **Real wages and profits** fall slightly; productivity is procyclical.
5. **Long-term rates and asset prices** respond on impact; credit spreads widen and the exchange rate appreciates.

The first three facts are the key evidence against a frictionless world. In the classical model of Lecture 3, money is neutral and a monetary expansion *raises* nominal rates through expected inflation. Explaining the facts requires nominal rigidities (Lectures 4–5) and, for the size and persistence of the effects, real rigidities and financial frictions (Lecture 9).

## 4. From money targets to interest-rate rules
In the 1970s and early 1980s many central banks announced targets for monetary aggregates. Unstable money demand made these targets unreliable guides, and most central banks moved to steering a short-term interest rate, with a numerical inflation target as the nominal anchor (Lecture 7). Money has not disappeared from monetary economics: the price level is still the price of money in terms of goods; money demand determines seigniorage; and central-bank balance sheets became policy instruments again after 2008. But modern models typically treat the interest rate as the policy instrument and let money be supplied on demand.

## 5. Measuring inflation
Policy targets use consumer price indexes (CPI, HICP) or the personal consumption expenditures (PCE) deflator; analysts also track core measures that exclude food and energy, trimmed means and medians. The GDP deflator measures the price of domestic production rather than consumption. The measures differ in coverage, weights and formulas; the CPI's fixed basket overstates the cost of living when consumers substitute towards goods whose relative price falls (the Boskin Commission estimated the bias at about one percentage point a year in the mid-1990s, since reduced by methodological changes).

## Problem set
1. Using the quantity equation, explain why money growth of 10% a year could coexist with 2% inflation for five years. Which assumption of the quantity theory fails?
2. Derive the semi-elasticity of real balances with respect to the interest rate implied by the log-log specification. Why is it large near \(i=0\)?
3. List the short-run facts of Section 3 that the classical model of Lecture 3 cannot explain, and say which friction each requires.
4. Explain why a central bank might prefer an interest-rate instrument when money-demand shocks are large (Poole, 1970).
5. Compare CPI and PCE inflation. Which would you target, and why?

## Coding exercises

These exercises use **MATLAB and Dynare only**. Download the course [data files and helper functions](../index.html#data-and-code)
into your working folder. Every data series is real: U.S. quarterly and monthly macro data (FRED-QD / FRED-MD,
1959–2025), published monetary-policy shock series (Romer–Romer, Jarociński–Karadi, Bu–Rogers–Wu,
Nakamura–Steinsson), World Bank CPI inflation for 190+ countries, and the KRW/USD exchange rate. Dynare models are
`.mod` files run with `dynare file noclearall` (Dynare 6). Try the starter files first, then open the solution. Every
printed result was produced by running the solution files as shown: the `.mod` files were processed by the Dynare
preprocessor and solved at first order, and the MATLAB code was run in GNU Octave.

### Exercise 1.1 — The quantity theory in the long run

*Data:* `us_quarterly.csv` — M2, GDP deflator and real GDP, 1959–2025.

The quantity equation \(M_tV_t=P_tY_t\) implies \(\pi\approx\Delta m+\Delta v-\Delta y\). Over long horizons the
quantity theory predicts that money growth and inflation move one-for-one.

1. Compute average annual growth rates of M2, the GDP deflator, real GDP and nominal GDP over non-overlapping five-year
   windows (1960–64, …, 2020–24).
2. Regress five-year inflation on five-year money growth. Is the slope close to one?
3. Repeat with nominal GDP growth as the dependent variable, and report the average growth of velocity in each window.
   What breaks the quarterly link between money and prices but leaves the long-run link partly intact?

*File `ex1_1.m`*

```matlab
T = readtable('us_quarterly.csv');
starts = 1960:5:2020;
g = @(x, a, b) 100 * (log(x(T.t == b + 0.75)) - log(x(T.t == a - 0.25))) / (b - a + 1);  % avg growth, Q4 to Q4
% TODO: loop over windows, compute money growth, inflation, real growth and nominal GDP growth
% TODO: regress inflation on [1, money growth]
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `ex1_1.m`*

```matlab
T = readtable('us_quarterly.csv');
starts = 1960:5:2020; K = numel(starts);
g = @(x, a, b) 100 * (log(x(T.t == b + 0.75)) - log(x(T.t == a - 0.25))) / (b - a + 1);  % avg growth, Q4 to Q4
ngdp = T.rgdp .* T.gdpdef / 100;
G = zeros(K, 5);
for k = 1:K
  a = starts(k); b = a + 4;
  G(k, :) = [g(T.m2, a, b), g(T.gdpdef, a, b), g(T.rgdp, a, b), g(ngdp, a, b), 0];
  G(k, 5) = G(k, 4) - G(k, 1);                       % velocity growth = nominal GDP growth - money growth
end
fprintf('window      dM2   infl  dGDP  dNGDP  dV\n');
for k = 1:K, fprintf('%d-%02d  %5.1f %5.1f %5.1f %6.1f %5.1f\n', starts(k), mod(starts(k)+4, 100), G(k, :)); end
X = [ones(K, 1) G(:, 1)];
b1 = X \ G(:, 2); b2 = X \ G(:, 4);
fprintf('\ninflation  = %.2f + %.2f * money growth   (corr %.2f)\n', b1, corr(G(:, 1), G(:, 2)));
fprintf('nominal GDP = %.2f + %.2f * money growth   (corr %.2f)\n', b2, corr(G(:, 1), G(:, 4)));
```

Output:

```text
window      dM2   infl  dGDP  dNGDP  dV
1960-64    7.0   1.3   4.3    5.5  -1.5
1965-69    6.6   3.5   4.4    8.0   1.4
1970-74    8.6   6.1   2.6    8.7   0.1
1975-79    9.8   6.7   3.9   10.6   0.8
1980-84    8.8   5.9   2.6    8.4  -0.4
1985-89    6.3   3.0   3.6    6.5   0.2
1990-94    2.1   2.7   2.5    5.2   3.1
1995-99    5.6   1.6   4.1    5.7   0.1
2000-04    6.6   2.2   2.5    4.7  -1.8
2005-09    5.7   2.1   1.0    3.1  -2.5
2010-14    6.3   1.7   2.3    4.0  -2.3
2015-19    5.4   1.5   2.5    4.1  -1.4
2020-24    6.8   3.8   2.3    6.1  -0.7

inflation  = -1.36 + 0.70 * money growth   (corr 0.70)
nominal GDP = 0.86 + 0.81 * money growth   (corr 0.71)
```

Across five-year windows money growth, inflation and nominal GDP growth move together (correlations of about
0.7), but the inflation slope is about 0.7 rather than one and the intercept is negative. Two things break the simple
link. First, velocity is far from
constant: it rose in the 1980s–90s as deposit deregulation and money-market funds pulled funds out of M2, then fell
sharply after 2008 and again in 2020–21, when M2 grew by about 40% without a matching rise in spending. Second, real
output growth varies across windows. The quantity theory works well across countries with very different money growth
(see Lecture 15), but in a low-inflation economy velocity shocks dominate, which is one reason central banks moved from
money targets to interest-rate rules (Lecture 7).

</details>

### Exercise 1.2 — Money demand and the opportunity cost of money

*Data:* `us_quarterly.csv` — M2, nominal GDP and the 3-month Treasury bill rate.

A semi-log money-demand function is \(\log(M/PY)=a-\xi\,i\): real balances relative to income fall when the nominal
interest rate rises.

1. Build annual averages of \(\log(M2/PY)\) and the 3-month T-bill rate.
2. Estimate \(\xi\) by OLS on 1959–1989 and on 1990–2024, with Newey–West standard errors (`ols_nw`).
3. Interpret \(\xi\): by how much do real balances change when the interest rate rises by one percentage point? Why
   might the relationship look different after 1990?

*File `ex1_2.m`*

```matlab
T = readtable('us_quarterly.csv');
yr = floor(T.t); years = (1959:2024)';
lm = log(T.m2 ./ (T.rgdp .* T.gdpdef / 100));
% TODO: annual averages of lm and T.tb3 with accumarray, then ols_nw on two subsamples
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `ex1_2.m`*

```matlab
T = readtable('us_quarterly.csv');
yr = floor(T.t); keep = yr <= 2024; yr = yr(keep);
lm = log(T.m2(keep) ./ (T.rgdp(keep) .* T.gdpdef(keep) / 100)); tb = T.tb3(keep);
years = unique(yr); idx = yr - years(1) + 1;
LM = accumarray(idx, lm, [], @mean); TB = accumarray(idx, tb, [], @mean);
for s = {[1959 1989], [1990 2024]}
  r = s{1}; k = years >= r(1) & years <= r(2);
  [b, se] = ols_nw(LM(k), [ones(sum(k), 1) TB(k)], 3);
  fprintf('%d-%d: log(M2/PY) = %.3f %+.4f * tbill   (NW se %.4f)  -> xi = %.1f%% per pp\n', ...
          r(1), r(2), b(1), b(2), se(2), -100 * b(2));
end
fprintf('M2/GDP: 1980 %.2f, 2007 %.2f, 2021 %.2f, 2024 %.2f\n', exp(LM(years == 1980)), exp(LM(years == 2007)), ...
        exp(LM(years == 2021)), exp(LM(years == 2024)));
```

Output:

```text
1959-1989: log(M2/PY) = -0.505 -0.0095 * tbill   (NW se 0.0016)  -> xi = 1.0% per pp
1990-2024: log(M2/PY) = -0.434 -0.0443 * tbill   (NW se 0.0151)  -> xi = 4.4% per pp
M2/GDP: 1980 0.54, 2007 0.50, 2021 0.87, 2024 0.72
```

Before 1990 the semi-elasticity is small for M2, because much of M2 pays interest that moves with market
rates, so the opportunity cost of holding it is the spread between the T-bill rate and the M2 own rate, not the
T-bill rate itself. After 1990 the estimated effect of interest rates is much larger and money relative to GDP swings
widely: the ratio fell into the 2000s, jumped when rates hit zero in 2009–15 and again in 2020–21, then fell as rates
rose in 2022–24. Near the zero lower bound the opportunity cost of money vanishes and households hold large balances.
That is the satiation region that underlies the Friedman rule (Lecture 2) and the liquidity trap (Lecture 10).

</details>

## Further reading
- Lucas, R. E. (2000), "Inflation and Welfare," *Econometrica* 68(2).
- McCandless, G. and W. Weber (1995), "Some Monetary Facts," Federal Reserve Bank of Minneapolis *Quarterly Review*.
- Christiano, L., M. Eichenbaum and C. Evans (1999), "Monetary Policy Shocks: What Have We Learned and to What End?" *Handbook of Macroeconomics*.
- Ramey, V. (2016), "Macroeconomic Shocks and Their Propagation," *Handbook of Macroeconomics*.
- Walsh, C., *Monetary Theory and Policy*, ch. 1.
