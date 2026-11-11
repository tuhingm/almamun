# Lecture 15 — Central Banking in Practice

**Level:** Graduate | **Field:** Monetary Economics

## Learning objectives
Connect the theory of the course to how central banks operate: independence and mandates, communication, inflation expectations, operating frameworks and balance sheets, and the measurement of inflation risk. Evaluate the evidence on inflation targeting and the lessons of the 2021–23 inflation.

## 1. Independence and mandates
The time-inconsistency problem (Lecture 6) motivates delegating monetary policy to an independent central bank with a clear mandate (Rogoff, 1985). Cross-country evidence shows a negative correlation between legal independence and inflation in advanced economies (Alesina and Summers, 1993), weaker evidence for emerging markets, where de facto independence (for example, governor turnover) matters more (Cukierman, Webb and Neyapti, 1992). Garriga (2016) and Romelli (2022) provide updated indices. Mandates differ: the Fed has a dual mandate of maximum employment and stable prices; the ECB a primary price-stability mandate; the Bank of Korea a price-stability mandate with attention to financial stability; Bangladesh Bank broader development objectives.

## 2. Inflation targeting: does it work?
Inflation targeters have had lower and more stable inflation since the 1990s. But so did non-targeters. Ball and Sheridan (2005) show that once one controls for initial inflation (regression to the mean), the advantage of inflation targeting in OECD countries largely disappears. Studies of emerging markets find larger effects (Gonçalves and Salles, 2008), but adoption is endogenous: countries tend to adopt inflation targeting at the end of a reform process — after granting independence, reducing fiscal dominance and abandoning pegs — so pre-adoption trends must be modelled. Modern staggered difference-in-differences and event-study estimators (Callaway and Sant'Anna, 2021; Sun and Abraham, 2021) address the bias of two-way fixed effects with heterogeneous adoption dates. Exercise 15.1 reproduces the Ball–Sheridan logic with World Bank data.

## 3. Inflation expectations and communication
Because the New Keynesian model is forward-looking, expectations are the main transmission channel, and communication is a policy instrument. Central banks publish forecasts, policy-rate projections (the Fed's dot plot, the Riksbank's and Norges Bank's rate paths), minutes and press conferences. Long-term expectations in surveys and break-even inflation stayed relatively anchored in 2021–23, which helped bring inflation down without a deep recession. Households' and firms' expectations, however, are dispersed, respond to salient prices such as food and gasoline, and are often far from professional forecasts (Coibion, Gorodnichenko and Kumar, 2018). Simple communication reaches the public better than technical statements (Haldane and McMahon, 2018).

## 4. Inflation at risk
Policymakers care about the risk of very high or very low inflation, not only the mean. **Inflation-at-risk** estimates the conditional quantiles of future inflation, \(Q_\tau(\pi_{t+h}\mid X_t)\), by quantile regression, analogous to growth-at-risk (Adrian, Boyarchenko and Giannone, 2019; López-Salido and Loria, 2024). Inflation persistence is typically stronger in the upper tail: when inflation is high, the risk of further increases rises disproportionately. Inflation targeting and central-bank independence compress the predictive distribution, but their additional predictive power is limited once current inflation is known, and they offered limited protection against the common 2021–23 shock. Exercise 15.2 estimates a pooled cross-country version.

## 5. Operating frameworks and the balance sheet
Central banks implement the policy rate either in a **corridor system** with scarce reserves, steering the overnight rate between a lending and a deposit facility through open-market operations, or in a **floor system** with abundant reserves, where the rate paid on reserves sets the floor (the Fed since 2008, ECB since 2015). Large balance sheets after QE have fiscal implications: when policy rates rise, interest paid on reserves exceeds income on long-term assets, and central banks such as the Fed and the Bank of England have reported operating losses since 2022. Quantitative tightening reduces the balance sheet gradually.

## 6. Lessons from the 2021–23 inflation
The inflation surge followed pandemic demand, supply-chain disruptions, energy price shocks after Russia's invasion of Ukraine, and large fiscal transfers. Central banks initially judged it transitory, then raised rates at the fastest pace in four decades. Lessons that recur in reviews: inflation can rise quickly and nonlinearly when labour markets are tight and shocks are large (Benigno and Eggertsson, 2023); forward guidance that commits to low rates can delay the response; make-up strategies designed for the lower bound were ill-suited to an inflation surge (the Fed removed its average-inflation-targeting language in 2025); and anchored expectations and credibility made disinflation less costly. Bernanke and Blanchard (2023) decompose U.S. inflation into contributions from energy and food prices, shortages and labour-market tightness.

## 7. Monetary policy in emerging and developing economies
Emerging-market central banks face fiscal dominance, dollarisation, shallow financial markets and volatile capital flows. Many combine interest-rate policy with foreign-exchange intervention and macroprudential tools (Lecture 8). Bangladesh's shift from monetary targeting to an interest-rate corridor (2023), a crawling peg (2024) and a market-based exchange rate (2025) illustrates the transition towards modern frameworks under high inflation and reserve pressures. Korea's experience since 1998 shows how inflation targeting with a flexible exchange rate can lower inflation and pass-through.

## 8. Course synthesis
The course moved from the quantity theory to the New Keynesian model, optimal policy, open economies, financial frictions, the lower bound, fiscal interactions, heterogeneity, estimation and identification. A good policy analysis combines a model for mechanisms and counterfactuals (Lectures 3–12), estimation that disciplines the model with data (Lecture 13), credible empirical evidence on transmission (Lecture 14), and institutional knowledge of how policy is made (this lecture).

## Problem set
1. Explain why central-bank independence can lower inflation without raising output volatility in the Rogoff (1985) model.
2. Using Exercise 15.1, explain the regression-to-the-mean critique of simple comparisons between inflation targeters and non-targeters.
3. Why might household inflation expectations differ systematically from professional forecasts? What does this imply for communication?
4. Explain the difference between corridor and floor systems for implementing the policy rate.
5. Write a two-page memo assessing the monetary-policy framework of Korea or Bangladesh using the tools of this course.

## Coding exercises

These exercises use **MATLAB and Dynare only**. Download the course [data files and helper functions](../index.html#data-and-code)
into your working folder. Every data series is real: U.S. quarterly and monthly macro data (FRED-QD / FRED-MD,
1959–2025), published monetary-policy shock series (Romer–Romer, Jarociński–Karadi, Bu–Rogers–Wu,
Nakamura–Steinsson), World Bank CPI inflation for 190+ countries, and the KRW/USD exchange rate. Dynare models are
`.mod` files run with `dynare file noclearall` (Dynare 6). Try the starter files first, then open the solution. Every
printed result was produced by running the solution files as shown: the `.mod` files were processed by the Dynare
preprocessor and solved at first order, and the MATLAB code was run in GNU Octave.

### Exercise 15.1 — Did inflation targeting lower inflation? A Ball–Sheridan comparison

*Data:* `cpi_inflation_wdi.csv` — World Bank CPI inflation for about 190 countries, 1960–2024, with inflation-targeting adoption years (`it_year`).

1. For each inflation-targeting country that adopted between 1990 and 2016, compute average inflation in the five
   years before and the five years after adoption. For non-targeters, use the same windows around 1998.
   Drop countries whose pre-period inflation exceeds 40%.
2. Compare the average change in inflation for adopters and non-adopters.
3. Following Ball and Sheridan (2005), regress the change on an IT dummy and pre-period inflation. What happens to the
   IT coefficient once you control for initial inflation, and why?

*File `ex15_1.m`*

```matlab
W = readtable('cpi_inflation_wdi.csv');
codes = unique(W.code);
% TODO: for each country: adoption year (or 1998 for non-IT), pre and post averages; regression with ols_nw(..., 0)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `ex15_1.m`*

```matlab
W = readtable('cpi_inflation_wdi.csv');
codes = unique(W.code); R = [];
for c = 1:numel(codes)
  k = strcmp(W.code, codes{c}); yr = W.year(k); p = W.cpi_inflation(k); ity = W.it_year(find(k, 1));
  isit = ~isnan(ity) && ity >= 1990 && ity <= 2016;
  if ~isnan(ity) && ~isit, continue; end            % late or very recent adopters: drop
  t0 = 1998; if isit, t0 = ity; end
  pre = p(yr >= t0 - 5 & yr <= t0 - 1); post = p(yr >= t0 + 1 & yr <= t0 + 5);
  if numel(pre) < 5 || numel(post) < 5 || any(isnan([pre; post])) || mean(pre) > 40, continue; end
  R = [R; isit mean(pre) mean(post)];
end
d = R(:, 3) - R(:, 2);
for g = [1 0]
  m = R(:, 1) == g;
  if g, nm = 'IT adopters  '; else, nm = 'non-adopters '; end
  fprintf('%s n = %3d   before %5.1f%%   after %5.1f%%   change %+5.1f\n', nm, sum(m), mean(R(m, 2)), mean(R(m, 3)), mean(d(m)));
end
[b1, s1] = ols_nw(d, [ones(size(d)) R(:, 1)], 0);
[b2, s2] = ols_nw(d, [ones(size(d)) R(:, 1) R(:, 2)], 0);
fprintf('\nchange = %.2f %+.2f IT                       (IT s.e. %.2f)\n', b1(1), b1(2), s1(2));
fprintf('change = %.2f %+.2f IT %+.2f pre-inflation    (IT s.e. %.2f)\n', b2(1), b2(2), b2(3), s2(2));
```

Output:

```text
IT adopters   n =  38   before  10.2%   after   4.7%   change  -5.5
non-adopters  n =  94   before   9.2%   after   4.6%   change  -4.6

change = -4.64 -0.85 IT                       (IT s.e. 1.32)
change = 1.25 -0.24 IT -0.64 pre-inflation    (IT s.e. 0.78)
```

Inflation fell by about 5.5 points in adopters and 4.6 points in non-adopters over the same kind of window: adopters
started from slightly higher inflation, and inflation fell almost everywhere in the 1990s. The raw difference of under
one point is small and statistically insignificant. Once the change is regressed on initial inflation, the strong
negative coefficient on pre-period inflation captures regression to the mean and the global disinflation, and the IT
coefficient shrinks further towards zero — Ball and Sheridan's (2005) finding. This does not show that inflation
targeting is useless: adoption is endogenous and often comes at the end of a reform process (central-bank
independence, ending exchange-rate pegs), which is why staggered difference-in-differences and event-study designs
that model the timing of adoption give more credible answers.

</details>

### Exercise 15.2 — Inflation at risk: do monetary frameworks thin the upper tail?

*Data:* `cpi_inflation_wdi.csv`, 1985–2024; observations with inflation above 50% are excluded.

Estimate pooled quantile regressions of next year's inflation on current inflation, an inflation-targeting indicator
and their interaction,
\[ Q_\tau(\pi_{i,t+1}\mid\cdot)=a_\tau+b_\tau\pi_{it}+c_\tau IT_{it}+d_\tau IT_{it}\pi_{it}, \]
for \(\tau\in\{0.1,0.5,0.9\}\), using `qreg`.

1. Report the coefficients.
2. Predict the 10th, 50th and 90th percentiles of next year's inflation when current inflation is 3% and when it is 8%,
   with and without inflation targeting.
3. Is the effect of the framework larger in the upper tail? What are the limits of this exercise?

*File `ex15_2.m`*

```matlab
W = readtable('cpi_inflation_wdi.csv');
% TODO: build pi_t and pi_{t+1} pairs within countries; IT = year >= it_year; qreg at tau = 0.1, 0.5, 0.9
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `ex15_2.m`*

```matlab
W = readtable('cpi_inflation_wdi.csv');
codes = unique(W.code); Y = []; X = [];
for c = 1:numel(codes)
  k = find(strcmp(W.code, codes{c})); yr = W.year(k); p = W.cpi_inflation(k); ity = W.it_year(k(1));
  for t = 1:numel(yr) - 1
    if yr(t) < 1985 || yr(t+1) ~= yr(t) + 1 || isnan(p(t)) || isnan(p(t+1)) || abs(p(t)) > 50 || abs(p(t+1)) > 50, continue; end
    it = ~isnan(ity) && yr(t) >= ity;
    Y = [Y; p(t+1)]; X = [X; 1 p(t) it it * p(t)];
  end
end
fprintf('observations: %d (country-years), IT share %.0f%%\n\n', numel(Y), 100 * mean(X(:, 3)));
taus = [0.1 0.5 0.9]; Bq = zeros(4, 3);
fprintf('quantile    const    pi_t     IT     IT*pi_t\n');
for j = 1:3
  Bq(:, j) = qreg(Y, X, taus(j));
  fprintf('  %.1f     %6.2f   %6.3f  %6.2f   %6.3f\n', taus(j), Bq(:, j));
end
fprintf('\npredicted quantiles of next year''s inflation (%%)\n');
fprintf('current infl   framework    q10     q50     q90\n');
for p0 = [3 8]
  for it = [0 1]
    q = [1 p0 it it * p0] * Bq;
    if it, nm = 'IT     '; else, nm = 'non-IT '; end
    fprintf('     %d%%        %s   %6.2f  %6.2f  %6.2f\n', p0, nm, q);
  end
end
```

Output:

```text
observations: 6098 (country-years), IT share 13%

quantile    const    pi_t     IT     IT*pi_t
  0.1      -0.40    0.309    0.35    0.081
  0.5       1.02    0.709    0.04   -0.060
  0.9       4.36    1.060   -1.67   -0.055

predicted quantiles of next year's inflation (%)
current infl   framework    q10     q50     q90
     3%        non-IT      0.53    3.14    7.54
     3%        IT          1.13    3.00    5.70
     8%        non-IT      2.08    6.69   12.83
     8%        IT          3.08    6.24   10.72
```

Persistence rises across the distribution: the coefficient on current inflation is about 0.3 at the 10th percentile,
0.7 at the median and about 1 at the 90th, so high inflation raises the risk of even higher inflation more than it
raises the median. Under inflation targeting the median is almost unchanged, but the distribution is compressed from
both sides: the 90th percentile is about 2 points lower and the 10th percentile is higher, which reduces both
inflation and deflation risk. The exercise is descriptive: IT countries differ in many ways (income, institutions,
exchange-rate regimes), adoption is endogenous, pooling assumes common coefficients across very different economies,
and the 2021–23 global surge showed that frameworks offer limited protection against large common shocks. Adding
income and central-bank independence as controls, or country fixed effects, is a natural extension.

</details>

## Further reading
- Ball, L. and N. Sheridan (2005), "Does Inflation Targeting Matter?" in Bernanke and Woodford (eds.), *The Inflation-Targeting Debate*.
- Bernanke, B. and O. Blanchard (2023), "What Caused the U.S. Pandemic-Era Inflation?" Brookings Hutchins Center Working Paper.
- Coibion, O., Y. Gorodnichenko and S. Kumar (2018), "How Do Firms Form Their Expectations? New Survey Evidence," *American Economic Review*.
- López-Salido, D. and F. Loria (2024), "Inflation at Risk," *Journal of Monetary Economics*.
- Blinder, A., M. Ehrmann, J. de Haan and D.-J. Jansen (2024), "Central Bank Communication with the General Public: Promise or False Hope?" *Journal of Economic Literature*.
