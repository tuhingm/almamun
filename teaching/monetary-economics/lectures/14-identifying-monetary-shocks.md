# Lecture 14 — Identifying Monetary Policy Shocks

**Level:** Graduate | **Field:** Monetary Economics

## Learning objectives
Understand the identification problem in measuring the effects of monetary policy, and compare recursive VARs, narrative shocks, high-frequency identification and local projections. Interpret the price puzzle, information effects and the empirical evidence that DSGE impulse responses should match.

## 1. The identification problem
Policy rates respond to the state of the economy, so the correlation between interest rates and output says little about the causal effect of policy. Write policy as a rule plus a shock, \(i_t=f(\Omega_t)+\varepsilon^m_t\), where \(\Omega_t\) is the central bank's information. The effects of policy are the responses to \(\varepsilon^m_t\), the part of policy not explained by the systematic response. Every identification strategy is a way of isolating \(\varepsilon^m_t\) (Ramey, 2016).

## 2. Recursive VARs
Christiano, Eichenbaum and Evans (1999) estimate a VAR \(Y_t=c+\sum_{l=1}^pA_lY_{t-l}+u_t\) and assume that output and prices do not respond to the policy shock within the period, while policy may respond to them. With the policy rate ordered last, the shock is the last element of \(S^{-1}u_t\), where \(S\) is the Cholesky factor of \(\Sigma_u\). The approach produces the facts of Lecture 1 — a gradual fall in output with a trough after 1–2 years — but also the **price puzzle**: prices often rise after a tightening (Sims, 1992). The usual explanation is that the Fed reacts to information about future inflation not in the VAR. Adding commodity prices or Fed forecasts reduces the puzzle.

## 3. Narrative shocks
Romer and Romer (2004) construct the Fed's intended funds-rate changes at each FOMC meeting from narrative records and regress them on the Greenbook forecasts the Fed was looking at; the residual is a shock purged of the Fed's information. With these shocks, output falls substantially more than in recursive VARs, and prices fall after a delay. Updates extend the series to 2019 (Exercise 14.2 uses Acosta's update). Narrative methods depend on the quality of records and forecasts.

## 4. High-frequency identification
Monetary-policy surprises can be measured as changes in interest-rate futures in a narrow window (e.g., 30 minutes) around FOMC announcements (Kuttner, 2001; Gürkaynak, Sack and Swanson, 2005). Within the window, nothing else systematically moves rates, so the surprise is plausibly exogenous. Surprises are used as **external instruments** (proxies) in VARs (Stock and Watson, 2012; Mertens and Ravn, 2013; Gertler and Karadi, 2015). Two problems arise:
- **Information effects.** A surprise tightening can signal that the Fed sees a stronger economy, so stock prices may rise rather than fall (Nakamura and Steinsson, 2018). Jarociński and Karadi (2020) separate pure policy shocks from information shocks using the sign of the co-movement between interest rates and stock prices.
- **Predictability.** Surprises are predictable from public data released before the announcement, reflecting the Fed's response to news (Bauer and Swanson, 2023); orthogonalising them yields cleaner shocks.

## 5. Local projections
Jordà (2005) estimates the impulse response at horizon \(h\) directly:
\[
y_{t+h}-y_{t-1}=\alpha_h+\beta_h\,\text{shock}_t+\gamma_h'X_{t}+u_{t+h},\qquad h=0,1,\dots,H .
\]
Local projections do not impose the dynamic restrictions of a VAR, so they are robust to misspecification at long horizons, but they have larger variance. Plagborg-Møller and Wolf (2021) show that LPs and VARs estimate the same population impulse responses with enough lags; the difference is a bias–variance trade-off (Li, Plagborg-Møller and Wolf, 2024). Residuals are serially correlated by construction, so inference uses heteroskedasticity- and autocorrelation-robust standard errors or lag augmentation (Montiel Olea and Plagborg-Møller, 2021).

## 6. What we know
Across methods, a 100 bp contractionary shock lowers industrial production by roughly 1–5% at the peak after one to two years and eventually lowers the price level, with substantial uncertainty about magnitudes and timing (Ramey, 2016). Estimates based on high-frequency instruments and on narrative shocks tend to find larger output effects than recursive VARs. Monetary shocks explain a modest share of output fluctuations — systematic policy matters more than shocks. **Matching** these responses is a common way to estimate DSGE models (Christiano, Eichenbaum and Evans, 2005): choose structural parameters to minimise the distance between model and VAR impulse responses.

## 7. Evidence outside the United States
For small open economies such as Korea, identification must account for the exchange rate and foreign shocks (Kim and Roubini, 2000). Block-exogeneity assumptions for foreign variables and high-frequency surprises from government bond futures around Bank of Korea announcements have been used. For economies with monetary targeting and administered rates, such as Bangladesh before 2023, the policy instrument itself must be chosen carefully.

## Problem set
1. Explain the recursive identification assumptions in a VAR with output, prices and the policy rate. Which are plausible at monthly frequency?
2. Why might the price puzzle indicate misspecification rather than a true effect of policy?
3. What is the central-bank information effect, and how do Jarociński and Karadi separate it from pure policy shocks?
4. Compare VARs and local projections in terms of bias and variance.
5. Design an identification strategy for monetary policy shocks in Korea.

## Coding exercises

These exercises use **MATLAB and Dynare only**. Download the course [data files and helper functions](../index.html#data-and-code)
into your working folder. Every data series is real: U.S. quarterly and monthly macro data (FRED-QD / FRED-MD,
1959–2025), published monetary-policy shock series (Romer–Romer, Jarociński–Karadi, Bu–Rogers–Wu,
Nakamura–Steinsson), World Bank CPI inflation for 190+ countries, and the KRW/USD exchange rate. Dynare models are
`.mod` files run with `dynare file noclearall` (Dynare 6). Try the starter files first, then open the solution. Every
printed result was produced by running the solution files as shown: the `.mod` files were processed by the Dynare
preprocessor and solved at first order, and the MATLAB code was run in GNU Octave.

### Exercise 14.1 — A recursive VAR and the price puzzle

*Data:* `us_monthly.csv` — industrial production, CPI and the federal funds rate, 1965–2007.

Estimate a VAR(12) in \(100\log IP_t\), \(100\log CPI_t\) and the federal funds rate, with a constant, on
1965M1–2007M12. Identify the monetary-policy shock recursively (Cholesky, with the funds rate ordered last).

1. Compute the impulse responses to a 25 bp funds-rate shock for 48 months.
2. Report the responses of output and prices at 6, 12, 24, 36 and 48 months.
3. Do prices initially rise after a tightening? What is the usual explanation, and how does adding commodity prices
   change the result?

*File `ex14_1.m`*

```matlab
D = readtable('us_monthly.csv');
k = D.year >= 1965 & D.year <= 2007;
Y = [100 * log(D.indpro(k)) 100 * log(D.cpi(k)) D.ffr(k)];
p = 12;
% TODO: X = [1, Y(t-1), ..., Y(t-p)]; B = X \ Y(p+1:end,:); Sigma; chol; companion-form IRFs
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `ex14_1.m`*

```matlab
D = readtable('us_monthly.csv');
k = D.year >= 1965 & D.year <= 2007;
Y = [100 * log(D.indpro(k)) 100 * log(D.cpi(k)) D.ffr(k)];
p = 12; [T, n] = size(Y);
X = ones(T - p, 1); for l = 1:p, X = [X Y(p+1-l:T-l, :)]; end
B = X \ Y(p+1:end, :); U = Y(p+1:end, :) - X * B; Sigma = U' * U / (T - p - size(X, 2));
S = chol(Sigma)';                                   % lower triangular: IP, CPI, FFR
shock = S(:, 3) * 0.25 / S(3, 3);                   % 25 bp funds-rate shock
H = 49; IR = zeros(n, H); Acomp = [B(2:end, :)'; eye(n * (p - 1)) zeros(n * (p - 1), n)];
s = [shock; zeros(n * (p - 1), 1)];
for h = 1:H, IR(:, h) = s(1:n); s = Acomp * s; end
fprintf('months after shock   %s\n', sprintf('%7d', [0 6 12 24 36 48]));
lab = {'industrial prod.', 'CPI             ', 'funds rate      '};
for v = 1:3, fprintf('%s     %s\n', lab{v}, sprintf('%7.3f', IR(v, [0 6 12 24 36 48] + 1))); end
```

Output:

```text
months after shock         0      6     12     24     36     48
industrial prod.       0.000 -0.072 -0.188 -0.252 -0.247 -0.214
CPI                    0.000  0.066  0.078  0.085  0.068  0.040
funds rate             0.250  0.172  0.100  0.071  0.039  0.021
```

After a 25 bp tightening the funds rate stays elevated for over a year and industrial production falls gradually,
reaching its trough after one to two years. The CPI, however, rises for a long time instead of falling: the "price
puzzle" (Sims, 1992). The usual interpretation is misidentification. The Fed raises rates when it expects inflation,
using information the VAR does not contain, so the "shock" is partly a response to anticipated inflation. Adding
commodity prices or Greenbook forecasts reduces the puzzle, and narrative or high-frequency identification
(Exercise 14.2) removes most of it. Note also the sensitivity to the sample: the 1970s, with oil shocks and
accommodative policy, drive much of the puzzle.

</details>

### Exercise 14.2 — Local projections with Romer–Romer shocks

*Data:* `mp_shocks_monthly.csv` (Romer–Romer narrative shocks, updated by Acosta, 1969–2019) and `us_monthly.csv`.

Romer and Romer (2004) purge the Fed's intended rate changes of the Greenbook forecasts the Fed reacted to. Estimate
\[ y_{t+h}-y_{t-1}=a_h+\beta_h\,RR_t+\sum_{j=1}^{12}\gamma_{h,j}\Delta y_{t-j}+\sum_{j=1}^{12}\delta_{h,j}RR_{t-j}+u_{t+h} \]
for \(y\) = 100 log IP and 100 log CPI, \(h=0,\dots,48\), on 1970–2007, with Newey–West standard errors.

1. Report \(\beta_h\) per 1 percentage point shock at 6, 12, 24, 36 and 48 months.
2. Compare the shapes with the recursive VAR of Exercise 14.1. Is there a price puzzle?
3. Why are local projections less efficient but more robust than a VAR?

*File `ex14_2.m`*

```matlab
D = readtable('us_monthly.csv'); S = readtable('mp_shocks_monthly.csv');
% TODO: align on year/month; build lags of dy and RR; loop over h with ols_nw(lhs, X, h+1)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `ex14_2.m`*

```matlab
D = readtable('us_monthly.csv'); S = readtable('mp_shocks_monthly.csv');
key = @(y, m) 12 * y + m;
[~, ia, ib] = intersect(key(D.year, D.month), key(S.year, S.month));
yr = D.year(ia); rr = S.rr(ib); T = numel(yr);
L = @(v, j) [NaN(j, 1); v(1:end-j)];
hs = [0 6 12 24 36 48]; lab = {'industrial prod.', 'CPI             '};
fprintf('months after shock   %s\n', sprintf('%7d', hs));
for v = 1:2
  if v == 1, y = 100 * log(D.indpro(ia)); else, y = 100 * log(D.cpi(ia)); end
  dy = [NaN; diff(y)];
  ctrl = []; for j = 1:12, ctrl = [ctrl L(dy, j) L(rr, j)]; end
  b = zeros(size(hs)); se = b;
  for m = 1:numel(hs)
    h = hs(m); lhs = NaN(T, 1); lhs(1:T-h) = y(1+h:T) - L(y, 1)(1:T-h);
    X = [ones(T, 1) rr ctrl];
    k = yr >= 1970 & yr <= 2007 & all(~isnan([lhs X]), 2);
    [bb, ss] = ols_nw(lhs(k), X(k, :), h + 1); b(m) = bb(2); se(m) = ss(2);
  end
  fprintf('%s     %s\n', lab{v}, sprintf('%7.2f', b));
  fprintf('   (s.e.)            %s\n', sprintf('%7.2f', se));
end
```

Output:

```text
months after shock         0      6     12     24     36     48
industrial prod.        0.34   0.29  -0.89  -2.10  -1.65  -1.49
   (s.e.)               0.14   0.39   0.75   1.20   1.27   1.59
CPI                     0.06   0.47   0.58   0.49  -0.85  -2.51
   (s.e.)               0.03   0.11   0.22   0.42   0.62   0.87
```

With narrative shocks, a 1 pp tightening is followed by a large fall in industrial production of about 2% after two
years, roughly twice the VAR response once both are scaled to a 1 pp rate change. Prices still rise for one to two years —
a short-lived price puzzle that is statistically significant here — before falling steadily, by about 2.5% after four
years. Romer and Romer (2004) also find that prices respond only after a long delay. Local projections estimate each
horizon by a separate regression, so they do not impose the VAR's dynamic restrictions: they are robust to
misspecification at long horizons but have larger standard errors, as the widening intervals show. Plagborg-Møller and
Wolf (2021) show that VARs and LPs estimate the same impulse responses in population; the choice is a bias–variance
trade-off (Li, Plagborg-Møller and Wolf, 2024).

</details>

## Further reading
- Christiano, L., M. Eichenbaum and C. Evans (1999), "Monetary Policy Shocks: What Have We Learned and to What End?" *Handbook of Macroeconomics*.
- Romer, C. and D. Romer (2004), "A New Measure of Monetary Shocks: Derivation and Implications," *American Economic Review*.
- Gertler, M. and P. Karadi (2015), "Monetary Policy Surprises, Credit Costs, and Economic Activity," *American Economic Journal: Macroeconomics*.
- Jarociński, M. and P. Karadi (2020), "Deconstructing Monetary Policy Surprises — The Role of Information Shocks," *American Economic Journal: Macroeconomics*.
- Plagborg-Møller, M. and C. Wolf (2021), "Local Projections and VARs Estimate the Same Impulse Responses," *Econometrica*.
