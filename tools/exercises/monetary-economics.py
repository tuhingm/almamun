INTRO = """
These exercises use **MATLAB and Dynare only**. Download the course [data files and helper functions](../index.html#data-and-code)
into your working folder. Every data series is real: U.S. quarterly and monthly macro data (FRED-QD / FRED-MD,
1959–2025), published monetary-policy shock series (Romer–Romer, Jarociński–Karadi, Bu–Rogers–Wu,
Nakamura–Steinsson), World Bank CPI inflation for 190+ countries, and the KRW/USD exchange rate. Dynare models are
`.mod` files run with `dynare file noclearall` (Dynare 6). Try the starter files first, then open the solution. Every
printed result was produced by running the solution files as shown: the `.mod` files were processed by the Dynare
preprocessor and solved at first order, and the MATLAB code was run in GNU Octave.
"""

EXERCISES = {}

# --------------------------------------------------------------------------------------------- Lecture 1
EXERCISES["01-money-inflation-facts.md"] = [
dict(title="The quantity theory in the long run",
data="`us_quarterly.csv` — M2, GDP deflator and real GDP, 1959–2025.",
task="""
The quantity equation \\(M_tV_t=P_tY_t\\) implies \\(\\pi\\approx\\Delta m+\\Delta v-\\Delta y\\). Over long horizons the
quantity theory predicts that money growth and inflation move one-for-one.

1. Compute average annual growth rates of M2, the GDP deflator, real GDP and nominal GDP over non-overlapping five-year
   windows (1960–64, …, 2020–24).
2. Regress five-year inflation on five-year money growth. Is the slope close to one?
3. Repeat with nominal GDP growth as the dependent variable, and report the average growth of velocity in each window.
   What breaks the quarterly link between money and prices but leaves the long-run link partly intact?
""",
starter=[("ex1_1.m", """
T = readtable('us_quarterly.csv');
starts = 1960:5:2020;
g = @(x, a, b) 100 * (log(x(T.t == b + 0.75)) - log(x(T.t == a - 0.25))) / (b - a + 1);  % avg growth, Q4 to Q4
% TODO: loop over windows, compute money growth, inflation, real growth and nominal GDP growth
% TODO: regress inflation on [1, money growth]
""")],
solution=[("ex1_1.m", """
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
fprintf('window      dM2   infl  dGDP  dNGDP  dV\\n');
for k = 1:K, fprintf('%d-%02d  %5.1f %5.1f %5.1f %6.1f %5.1f\\n', starts(k), mod(starts(k)+4, 100), G(k, :)); end
X = [ones(K, 1) G(:, 1)];
b1 = X \\ G(:, 2); b2 = X \\ G(:, 4);
fprintf('\\ninflation  = %.2f + %.2f * money growth   (corr %.2f)\\n', b1, corr(G(:, 1), G(:, 2)));
fprintf('nominal GDP = %.2f + %.2f * money growth   (corr %.2f)\\n', b2, corr(G(:, 1), G(:, 4)));
""")],
notes="""Across five-year windows money growth, inflation and nominal GDP growth move together (correlations of about
0.7), but the inflation slope is about 0.7 rather than one and the intercept is negative. Two things break the simple
link. First, velocity is far from
constant: it rose in the 1980s–90s as deposit deregulation and money-market funds pulled funds out of M2, then fell
sharply after 2008 and again in 2020–21, when M2 grew by about 40% without a matching rise in spending. Second, real
output growth varies across windows. The quantity theory works well across countries with very different money growth
(see Lecture 15), but in a low-inflation economy velocity shocks dominate, which is one reason central banks moved from
money targets to interest-rate rules (Lecture 7).
"""),

dict(title="Money demand and the opportunity cost of money",
data="`us_quarterly.csv` — M2, nominal GDP and the 3-month Treasury bill rate.",
task="""
A semi-log money-demand function is \\(\\log(M/PY)=a-\\xi\\,i\\): real balances relative to income fall when the nominal
interest rate rises.

1. Build annual averages of \\(\\log(M2/PY)\\) and the 3-month T-bill rate.
2. Estimate \\(\\xi\\) by OLS on 1959–1989 and on 1990–2024, with Newey–West standard errors (`ols_nw`).
3. Interpret \\(\\xi\\): by how much do real balances change when the interest rate rises by one percentage point? Why
   might the relationship look different after 1990?
""",
starter=[("ex1_2.m", """
T = readtable('us_quarterly.csv');
yr = floor(T.t); years = (1959:2024)';
lm = log(T.m2 ./ (T.rgdp .* T.gdpdef / 100));
% TODO: annual averages of lm and T.tb3 with accumarray, then ols_nw on two subsamples
""")],
solution=[("ex1_2.m", """
T = readtable('us_quarterly.csv');
yr = floor(T.t); keep = yr <= 2024; yr = yr(keep);
lm = log(T.m2(keep) ./ (T.rgdp(keep) .* T.gdpdef(keep) / 100)); tb = T.tb3(keep);
years = unique(yr); idx = yr - years(1) + 1;
LM = accumarray(idx, lm, [], @mean); TB = accumarray(idx, tb, [], @mean);
for s = {[1959 1989], [1990 2024]}
  r = s{1}; k = years >= r(1) & years <= r(2);
  [b, se] = ols_nw(LM(k), [ones(sum(k), 1) TB(k)], 3);
  fprintf('%d-%d: log(M2/PY) = %.3f %+.4f * tbill   (NW se %.4f)  -> xi = %.1f%% per pp\\n', ...
          r(1), r(2), b(1), b(2), se(2), -100 * b(2));
end
fprintf('M2/GDP: 1980 %.2f, 2007 %.2f, 2021 %.2f, 2024 %.2f\\n', exp(LM(years == 1980)), exp(LM(years == 2007)), ...
        exp(LM(years == 2021)), exp(LM(years == 2024)));
""")],
notes="""Before 1990 the semi-elasticity is small for M2, because much of M2 pays interest that moves with market
rates, so the opportunity cost of holding it is the spread between the T-bill rate and the M2 own rate, not the
T-bill rate itself. After 1990 the estimated effect of interest rates is much larger and money relative to GDP swings
widely: the ratio fell into the 2000s, jumped when rates hit zero in 2009–15 and again in 2020–21, then fell as rates
rose in 2022–24. Near the zero lower bound the opportunity cost of money vanishes and households hold large balances.
That is the satiation region that underlies the Friedman rule (Lecture 2) and the liquidity trap (Lecture 10).
"""),
]

# --------------------------------------------------------------------------------------------- Lecture 2
EXERCISES["02-money-general-equilibrium.md"] = [
dict(title="The welfare cost of inflation (Bailey and Lucas)",
data="`us_quarterly.csv` — M1/GDP and the T-bill rate, 1959–2007.",
task="""
With money in the utility function, the welfare cost of a nominal interest rate \\(i\\) is the area under the inverse
money-demand curve (Bailey, 1956; Lucas, 2000). For semi-log demand \\(m(i)=Ae^{-\\xi i}\\), measured as a share of
income,
\\[ w(i)=\\frac{A}{\\xi}\\left[1-(1+\\xi i)e^{-\\xi i}\\right]. \\]

1. Estimate \\(A\\) and \\(\\xi\\) from annual M1/GDP and the T-bill rate over 1959–2007 (rates in decimals).
   (M1 was redefined in May 2020 to include savings deposits, so later data are not comparable.)
2. Compute \\(w(i)\\) for inflation of 0, 2, 5, 10 and 20% with a real rate of 2%, so \\(i=r+\\pi\\). The Friedman rule
   \\(i=0\\) gives \\(w=0\\).
3. Repeat the estimation with M2. Why does the broader aggregate give a very different answer?
""",
starter=[("ex2_1.m", """
T = readtable('us_quarterly.csv');
% TODO: annual averages 1959-2007 of M1/GDP and tbill/100; regress log(m) on [1, i]
% A = exp(b(1)); xi = -b(2);
w = @(i, A, xi) (A / xi) * (1 - (1 + xi * i) .* exp(-xi * i));
""")],
solution=[("ex2_1.m", """
T = readtable('us_quarterly.csv');
yr = floor(T.t); k = yr <= 2007; yr = yr(k);
py = T.rgdp(k) .* T.gdpdef(k) / 100; i = T.tb3(k) / 100;
years = unique(yr); idx = yr - years(1) + 1;
I = accumarray(idx, i, [], @mean);
for agg = {'m1', 'm2'}
  M = accumarray(idx, T.(agg{1})(k) ./ py, [], @mean);
  b = [ones(numel(M), 1) I] \\ log(M); A.(agg{1}) = exp(b(1)); xi.(agg{1}) = -b(2);
  fprintf('%s/GDP = %.3f * exp(%+.2f i)\\n', upper(agg{1}), A.(agg{1}), -xi.(agg{1}));
end
w = @(i) (A.m1 / xi.m1) * (1 - (1 + xi.m1 * i) .* exp(-xi.m1 * i));
r = 0.02;
fprintf('\\nM1 demand: inflation   nominal rate   welfare cost (%% of GDP)\\n');
for p = [0 0.02 0.05 0.10 0.20]
  fprintf('             %4.0f%%        %4.1f%%          %6.3f\\n', 100 * p, 100 * (r + p), 100 * w(r + p));
end
fprintf('cost of 10%% vs 2%% inflation: %.2f%% of GDP per year\\n', 100 * (w(0.12) - w(0.04)));
""")],
notes="""With M1 the semi-elasticity is about 1.6, and moving from 2% to 10% inflation costs roughly 0.15% of GDP per
year; 20% inflation costs about half a percent. These are small numbers. Lucas (2000) finds somewhat larger costs with
a log-log specification, which makes money demand very elastic near zero interest rates and so raises the cost of
even moderate inflation. M2 gives a positive coefficient over 1959–2007: most of M2 pays interest that moves with
market rates, so the T-bill rate is not its opportunity cost, and its velocity was driven by financial innovation
rather than by interest rates. The choice of aggregate matters. The general lesson holds anyway: shoe-leather costs of moderate inflation are small, and the case for low inflation rests
more on the relative-price distortions that sticky prices create (Lectures 4–6).
"""),

dict(title="Cash in advance: inflation as a tax on market work",
data="Model calibration (no data).",
task="""
Households maximise \\(\\sum\\beta^t[\\log c_t+\\psi\\log(1-n_t)]\\), output is \\(y_t=n_t\\), and consumption must be paid
with money held from the previous period (cash in advance). In the steady state with money growth \\(\\mu\\) the gross
nominal rate is \\(R=(1+\\mu)/\\beta\\), and the labour–consumption condition becomes
\\[ \\frac{\\psi c}{1-n}=\\frac{1}{R},\\qquad c=n \\;\\Rightarrow\\; n=\\frac{1}{1+\\psi R}. \\]

1. Calibrate \\(\\psi\\) so that \\(n=1/3\\) at 2% money growth with \\(\\beta=0.99^4\\) (annual).
2. Compute hours, output and the welfare cost relative to the Friedman rule (\\(R=1\\)) in consumption-equivalent terms
   for \\(\\mu\\in\\{\\beta-1,0,2\\%,5\\%,10\\%,50\\%\\}\\).
3. Is money superneutral in this model? Explain the mechanism.
""",
starter=[("ex2_2.m", """
beta = 0.99^4;
R = @(mu) (1 + mu) / beta;
% TODO: psi from n = 1/3 at mu = 0.02; n(mu) = 1/(1 + psi*R(mu))
% TODO: consumption-equivalent loss lambda solving log((1-lam) c_FR) + psi log(1-n_FR) = log(c) + psi log(1-n)
""")],
solution=[("ex2_2.m", """
beta = 0.99^4;
R = @(mu) (1 + mu) / beta;
psi = (3 - 1) / R(0.02);                         % n = 1/(1+psi R) = 1/3
n = @(mu) 1 ./ (1 + psi * R(mu));
U = @(mu) log(n(mu)) + psi * log(1 - n(mu));     % c = n
nFR = n(beta - 1);
fprintf('psi = %.3f\\n', psi);
fprintf('money growth   R      hours   output vs FR   welfare cost (%% of consumption)\\n');
for mu = [beta - 1, 0, 0.02, 0.05, 0.10, 0.50]
  lam = 1 - exp(U(mu) - U(beta - 1));          % log((1-lam) c_FR) + psi log(1-n_FR) = U(mu)
  fprintf('  %6.2f%%    %5.3f   %.4f    %6.2f%%        %6.3f\\n', 100 * mu, R(mu), n(mu), ...
          100 * (n(mu) / nFR - 1), 100 * lam);
end
""")],
notes="""Money is not superneutral here. Consumption needs cash, and holding cash costs the nominal interest rate, while
leisure does not need cash. Higher money growth raises \\(R\\) and so acts as a tax on market activity, and households
substitute towards leisure: hours and output fall steadily with inflation. The Friedman rule (\\(\\mu=\\beta-1\\), so
\\(R=1\\)) removes the wedge and gives the highest welfare. Moving from the Friedman rule to 2% money growth costs
about 0.1% of consumption, 10% money growth about 0.6%, and 50% money growth about 6%. The same wedge appears in money-in-utility models
whenever money and consumption are not separable in utility. With separable preferences and inelastic labour,
steady-state real variables would not depend on \\(\\mu\\) (Sidrauski superneutrality).
"""),
]

# --------------------------------------------------------------------------------------------- Lecture 3
CLASSICAL_MOD = """
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
"""

EXERCISES["03-classical-monetary-model.md"] = [
dict(title="Monetary neutrality in the classical model (Dynare)",
data="Model calibration (Galí, 2015, ch. 2).",
task="""
Implement the log-linear classical monetary model with exogenous money growth \\(\\Delta m_t=\\rho_m\\Delta m_{t-1}+
\\varepsilon^m_t\\), written in terms of real balances \\(l_t=m_t-p_t\\).

1. Run the model and plot (or print) the responses of output, the real rate, inflation and the nominal rate to a 1%
   money-growth shock.
2. Verify neutrality: which variables do not move at all?
3. Galí shows that \\(i_t=\\rho+\\frac{\\rho_m}{1+\\eta(1-\\rho_m)}\\Delta m_t\\). Check the impact response of \\(i\\)
   against this formula. Why does the nominal rate rise, not fall, after a monetary expansion when \\(\\rho_m>0\\)?
""",
starter=[("classical.mod", """
var y n w r i pi l dm a;
varexo eps_a eps_m;
parameters sigma phi alpha eta rho_a rho_m;
sigma = 1; phi = 1; alpha = 1/3; eta = 4; rho_a = 0.9; rho_m = 0.5;
model(linear);
  // TODO: production, labour supply and demand, Euler, Fisher, money demand, real balances, shocks
end;
shocks; var eps_a; stderr 1; var eps_m; stderr 1; end;
stoch_simul(order=1, irf=8, noprint, nograph);
"""), ("ex3_1.m", """
dynare classical noclearall
% responses are stored in oo_.irfs.<variable>_<shock>, e.g. oo_.irfs.pi_eps_m
""")],
solution=[("classical.mod", CLASSICAL_MOD), ("ex3_1.m", """
dynare classical noclearall
names = {'y', 'n', 'w', 'r', 'i', 'pi', 'l'};
fprintf('Responses to a 1%% money-growth shock (rho_m = %.1f)\\n', rho_m);
fprintf('%-4s %s\\n', 'var', sprintf('   h=%d  ', 0:4));
for k = 1:numel(names)
  f = [names{k} '_eps_m'];
  if isfield(oo_.irfs, f), fprintf('%-4s %s\\n', names{k}, sprintf('%8.4f ', oo_.irfs.(f)(1:5)));
  else, fprintf('%-4s    does not respond (neutral)\\n', names{k}); end
end
fprintf('\\nimpact on i: Dynare %.4f, formula rho_m/(1+eta(1-rho_m)) = %.4f\\n', ...
        oo_.irfs.i_eps_m(1), rho_m / (1 + eta * (1 - rho_m)));
fprintf('price level after 5 quarters: %.3f%% (money stock: %.3f%%)\\n', ...
        sum(oo_.irfs.pi_eps_m(1:5)), sum(rho_m .^ (0:4)));
""")],
notes="""Output, hours, the real wage and the real interest rate do not respond at all: in the classical model money is
neutral, and real variables are pinned down by technology and preferences alone. Inflation jumps by more than money
growth on impact because higher expected money growth raises expected inflation and the nominal rate, which lowers
desired real balances, so prices must rise even faster. The nominal rate rises by exactly
\\(\\rho_m/(1+\\eta(1-\\rho_m))\\): with persistent money growth the expected-inflation (Fisher) effect dominates and there
is no liquidity effect. Generating a fall in interest rates after a monetary expansion requires nominal rigidities
(Lectures 4–5) or limited participation.
"""),

dict(title="Interest-rate pegs and price-level determinacy (Dynare)",
data="Model calibration.",
task="""
Replace the money rule with an interest-rate rule \\(i_t=\\phi_\\pi\\pi_t+v_t\\). With the Fisher equation
\\(i_t=r_t+E_t\\pi_{t+1}\\) and an exogenous real rate, inflation satisfies
\\(E_t\\pi_{t+1}=\\phi_\\pi\\pi_t+v_t-r_t\\).

1. Write the model in Dynare and loop over \\(\\phi_\\pi\\in\\{0,0.5,0.95,1.05,1.5,3\\}\\) with `set_param_value` and
   `stoch_simul`, recording the Blanchard–Kahn status.
2. When the equilibrium is unique, compare the impact response of inflation to a policy shock with
   \\(-1/(\\phi_\\pi-\\rho_v)\\).
3. Explain Sargent and Wallace's (1975) indeterminacy result for an interest-rate peg.
""",
starter=[("peg.mod", """
var pi i r v;
varexo eps_r eps_v;
parameters phi_pi rho_r rho_v;
phi_pi = 1.5; rho_r = 0.8; rho_v = 0.5;
model(linear);
  // TODO: Fisher equation, policy rule, AR(1) real rate and policy shock
end;
shocks; var eps_r; stderr 1; var eps_v; stderr 0.25; end;
stoch_simul(order=1, irf=8, noprint, nograph);
"""), ("ex3_2.m", """
dynare peg noclearall
for phi = [0 0.5 0.95 1.05 1.5 3]
  set_param_value('phi_pi', phi);
  [info, oo_] = stoch_simul(M_, options_, oo_, {});
  % TODO: info(1) == 0 unique, 3 no stable solution, 4 indeterminacy
end
""")],
solution=[("peg.mod", """
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
"""), ("ex3_2.m", """
dynare peg noclearall
status = {'unique', '', 'no stable solution', 'indeterminate'};
fprintf('phi_pi   equilibrium          pi impact (eps_v)   -sd/(phi-rho_v)\\n');
for phi = [0 0.5 0.95 1.05 1.5 3]
  set_param_value('phi_pi', phi);
  [info, oo_] = stoch_simul(M_, options_, oo_, {});
  if info(1) == 0
    fprintf('%5.2f    %-18s   %8.4f            %8.4f\\n', phi, status{1}, oo_.irfs.pi_eps_v(1), -0.25 / (phi - rho_v));
  else
    fprintf('%5.2f    %-18s\\n', phi, status{info(1)});
  end
end
""")],
notes="""For \\(\\phi_\\pi<1\\), including a pure peg (\\(\\phi_\\pi=0\\)), Dynare reports indeterminacy: the stable root
condition fails and any inflation path satisfying \\(E_t\\pi_{t+1}=\\phi_\\pi\\pi_t+\\dots\\) is an equilibrium, so self-fulfilling
expectations can move prices. This is the Sargent–Wallace result: an interest-rate peg pins down expected inflation
but not the price level. For \\(\\phi_\\pi>1\\) the solution is unique and inflation responds to a policy shock by
exactly \\(-1/(\\phi_\\pi-\\rho_v)\\) times the shock. A stronger response to inflation means smaller equilibrium
movements in inflation. This "Taylor principle" carries over to the New Keynesian model (Lecture 5).
"""),
]

# --------------------------------------------------------------------------------------------- Lecture 4
EXERCISES["04-price-wage-setting.md"] = [
dict(title="From Calvo parameters to the Phillips-curve slope — and what the data say",
data="Model calibration; `us_quarterly.csv` — core CPI and the unemployment rate.",
task="""
In the New Keynesian Phillips curve \\(\\pi_t=\\beta E_t\\pi_{t+1}+\\kappa\\tilde y_t\\), the slope is
\\(\\kappa=\\lambda\\left(\\sigma+\\frac{\\varphi+\\alpha}{1-\\alpha}\\right)\\) with
\\(\\lambda=\\frac{(1-\\theta)(1-\\beta\\theta)}{\\theta}\\frac{1-\\alpha}{1-\\alpha+\\alpha\\epsilon}\\).

1. Compute \\(\\kappa\\) and the average price duration \\(1/(1-\\theta)\\) for \\(\\theta\\in\\{0.5,0.67,0.75,0.85\\}\\)
   (\\(\\beta=0.99,\\sigma=1,\\varphi=5,\\alpha=1/4,\\epsilon=6\\)).
2. Estimate a reduced-form Phillips curve on U.S. data:
   \\(\\pi_t=c+\\gamma\\bar\\pi_{t-1}+\\delta(u_t-u^*_t)\\), with \\(\\pi_t\\) annualised quarterly core CPI inflation,
   \\(\\bar\\pi_{t-1}\\) the average of the previous four quarters, and \\(u^*\\) the HP trend of unemployment. Compare
   1960–1984 with 1985–2019.
3. Has the Phillips curve flattened? Give two explanations.
""",
starter=[("ex4_1.m", """
beta = 0.99; sigma = 1; varphi = 5; alpha = 1/4; epsilon = 6;
% TODO: kappa(theta) and durations
T = readtable('us_quarterly.csv');
% TODO: pi = 400*diff(log(T.corecpi)); ugap = T.unrate - hp_trend(T.unrate, 1600); ols_nw by subsample
""")],
solution=[("ex4_1.m", """
beta = 0.99; sigma = 1; varphi = 5; alpha = 1/4; epsilon = 6;
Theta = (1 - alpha) / (1 - alpha + alpha * epsilon);
fprintf('theta   duration (qtrs)   lambda    kappa\\n');
for theta = [0.5 0.67 0.75 0.85]
  lam = (1 - theta) * (1 - beta * theta) / theta * Theta;
  kap = lam * (sigma + (varphi + alpha) / (1 - alpha));
  fprintf('%5.2f      %5.1f          %6.4f   %6.4f\\n', theta, 1 / (1 - theta), lam, kap);
end
T = readtable('us_quarterly.csv');
t = T.t(2:end); p = 400 * diff(log(T.corecpi)); u = T.unrate(2:end);
ug = u - hp_trend(u, 1600);
pbar = filter(ones(4, 1) / 4, 1, [NaN; p(1:end-1)]);        % average of the previous four quarters
fprintf('\\nsample      gamma (lagged infl)   delta (unemployment gap)\\n');
for s = {[1960 1984.75], [1985 2019.75]}
  r = s{1}; k = t >= r(1) & t <= r(2) & ~isnan(pbar);
  [b, se] = ols_nw(p(k), [ones(sum(k), 1) pbar(k) ug(k)], 4);
  fprintf('%d-%d      %5.2f (%.2f)          %6.2f (%.2f)\\n', floor(r(1)), floor(r(2)), b(2), se(2), b(3), se(3));
end
""")],
notes="""Longer price spells flatten the Phillips curve sharply: moving from two-quarter to roughly seven-quarter price
durations divides \\(\\kappa\\) by more than ten, because firms that reset prices rarely put little weight on current
marginal cost. In the data, the coefficient on the unemployment gap is large and negative before 1985 and much smaller
afterwards (about a seventh as large), while the coefficient on lagged inflation stays close to 0.9 in both samples.
This backward-looking specification cannot capture a change in how expectations are formed, so two leading
explanations for the flattening lie outside it: better-anchored inflation expectations after the Volcker disinflation,
and successful monetary policy that offsets demand shocks, which makes the reduced-form slope hard to
identify even if the structural \\(\\kappa\\) is unchanged (McLeay and Tenreyro, 2020). The 2021–23 surge suggests the
curve can steepen when slack is very low.
"""),

dict(title="Sticky prices and sticky wages (Dynare)",
data="Model calibration (Galí, 2015, ch. 6).",
task="""
Extend the New Keynesian model with Calvo wage setting (Erceg, Henderson and Levin, 2000). The price and wage Phillips
curves are
\\(\\pi^p_t=\\beta E_t\\pi^p_{t+1}+\\kappa_p\\tilde y_t+\\lambda_p\\tilde\\omega_t\\) and
\\(\\pi^w_t=\\beta E_t\\pi^w_{t+1}+\\kappa_w\\tilde y_t-\\lambda_w\\tilde\\omega_t\\), where \\(\\tilde\\omega_t\\) is the
real-wage gap, which evolves as \\(\\tilde\\omega_t=\\tilde\\omega_{t-1}+\\pi^w_t-\\pi^p_t\\).

1. Write the model in Dynare using model-local variables (`#`) for \\(\\lambda_p,\\lambda_w,\\kappa_p,\\kappa_w\\) so they
   update when \\(\\theta_w\\) changes.
2. Compare the response to a 25 bp monetary-policy shock for \\(\\theta_w=0.75\\) and \\(\\theta_w=0.25\\).
3. How do sticky wages change the response of output, price inflation and the real wage?
""",
starter=[("ehl.mod", """
var y_gap pi_p pi_w w_gap i v;
varexo eps_v;
parameters beta sigma varphi alpha eps_p eps_w theta_p theta_w phi_pi phi_y rho_v;
beta = 0.99; sigma = 1; varphi = 5; alpha = 1/4; eps_p = 6; eps_w = 4.5;
theta_p = 0.75; theta_w = 0.75; phi_pi = 1.5; phi_y = 0.125; rho_v = 0.5;
model(linear);
  # lambda_p = (1 - theta_p) * (1 - beta * theta_p) / theta_p * (1 - alpha) / (1 - alpha + alpha * eps_p);
  // TODO: lambda_w, kappa_p = alpha*lambda_p/(1-alpha), kappa_w = lambda_w*(sigma + varphi/(1-alpha))
  // TODO: two Phillips curves, wage-gap identity, IS curve, Taylor rule, AR(1) shock
end;
shocks; var eps_v; stderr 0.25; end;
stoch_simul(order=1, irf=12, noprint, nograph);
"""), ("ex4_2.m", """
dynare ehl noclearall
% TODO: loop over theta_w with set_param_value and stoch_simul; print impact and peak responses
""")],
solution=[("ehl.mod", """
// Sticky prices and wages (Erceg-Henderson-Levin; Gali, 2015, ch. 6), monetary shock only
var y_gap pi_p pi_w w_gap i v;
varexo eps_v;
parameters beta sigma varphi alpha eps_p eps_w theta_p theta_w phi_pi phi_y rho_v;
beta = 0.99; sigma = 1; varphi = 5; alpha = 1/4; eps_p = 6; eps_w = 4.5;
theta_p = 0.75; theta_w = 0.75; phi_pi = 1.5; phi_y = 0.125; rho_v = 0.5;
model(linear);
  # lambda_p = (1 - theta_p) * (1 - beta * theta_p) / theta_p * (1 - alpha) / (1 - alpha + alpha * eps_p);
  # lambda_w = (1 - theta_w) * (1 - beta * theta_w) / (theta_w * (1 + eps_w * varphi));
  # kappa_p = alpha * lambda_p / (1 - alpha);
  # kappa_w = lambda_w * (sigma + varphi / (1 - alpha));
  pi_p = beta * pi_p(+1) + kappa_p * y_gap + lambda_p * w_gap;     // price Phillips curve
  pi_w = beta * pi_w(+1) + kappa_w * y_gap - lambda_w * w_gap;     // wage Phillips curve
  w_gap = w_gap(-1) + pi_w - pi_p;                                 // real-wage gap
  y_gap = y_gap(+1) - (1 / sigma) * (i - pi_p(+1));                // dynamic IS (r^n constant)
  i = phi_pi * pi_p + phi_y * y_gap + v;                           // Taylor rule
  v = rho_v * v(-1) + eps_v;
end;
shocks; var eps_v; stderr 0.25; end;
stoch_simul(order=1, irf=12, noprint, nograph);
"""), ("ex4_2.m", """
dynare ehl noclearall
fprintf('theta_w   output gap: impact  cum(0-7)   price infl (ann.): impact   real wage gap: impact\\n');
for tw = [0.75 0.25]
  set_param_value('theta_w', tw);
  [info, oo_] = stoch_simul(M_, options_, oo_, {});
  ir = oo_.irfs;
  fprintf('  %.2f            %7.3f   %7.3f              %7.3f               %7.3f\\n', tw, ir.y_gap_eps_v(1), ...
          sum(ir.y_gap_eps_v(1:8)), 4 * ir.pi_p_eps_v(1), ir.w_gap_eps_v(1));
end
""")],
notes="""With sticky wages (\\(\\theta_w=0.75\\)) the contraction in output is larger and more persistent, while price
inflation falls by less. Firms' marginal cost depends on the real wage; when nominal wages adjust slowly, the real wage
barely moves after the shock, so firms have less reason to cut prices and more of the adjustment falls on quantities.
With more flexible wages (\\(\\theta_w=0.25\\)) wages and marginal cost fall quickly, price inflation drops more and the
output loss is smaller. Sticky wages are one reason estimated DSGE models (Lecture 13) find large real effects of
monetary policy with only moderate price stickiness.
"""),
]

# --------------------------------------------------------------------------------------------- Lecture 5
NK_MOD = """
// Basic New Keynesian model (Gali, 2015, ch. 3), log-linear
var y_gap pi i r_real r_nat y y_nat n a v;
varexo eps_a eps_v;
parameters beta sigma varphi alpha epsilon theta phi_pi phi_y rho_a rho_v;
beta = 0.99; sigma = 1; varphi = 5; alpha = 1/4; epsilon = 9; theta = 3/4;
phi_pi = 1.5; phi_y = 0.5/4; rho_a = 0.9; rho_v = 0.5;
model(linear);
  # Theta  = (1 - alpha) / (1 - alpha + alpha * epsilon);
  # lambda = (1 - theta) * (1 - beta * theta) / theta * Theta;
  # kappa  = lambda * (sigma + (varphi + alpha) / (1 - alpha));
  # psi_ya = (1 + varphi) / (sigma * (1 - alpha) + varphi + alpha);
  pi = beta * pi(+1) + kappa * y_gap;                       // New Keynesian Phillips curve
  y_gap = y_gap(+1) - (1 / sigma) * (i - pi(+1) - r_nat);   // dynamic IS equation
  i = phi_pi * pi + phi_y * y_gap + v;                      // Taylor rule
  r_real = i - pi(+1);
  r_nat = -sigma * (1 - rho_a) * psi_ya * a;               // natural real rate (deviation)
  y_nat = psi_ya * a;
  y = y_nat + y_gap;
  y = a + (1 - alpha) * n;
  a = rho_a * a(-1) + eps_a;
  v = rho_v * v(-1) + eps_v;
end;
shocks; var eps_a; stderr 1; var eps_v; stderr 0.25; end;
stoch_simul(order=1, irf=12, noprint, nograph);
"""

EXERCISES["05-basic-new-keynesian.md"] = [
dict(title="The basic New Keynesian model: Dynare versus the analytical solution",
data="Model calibration (Galí, 2015, ch. 3).",
task="""
1. Implement Galí's baseline model in Dynare with a technology shock and an AR(1) monetary-policy shock
   (\\(\\rho_v=0.5\\), 25 bp).
2. With an AR(1) policy shock the model has the closed-form solution \\(\\tilde y_t=-(1-\\beta\\rho_v)\\Lambda_v v_t\\),
   \\(\\pi_t=-\\kappa\\Lambda_v v_t\\), where \\(\\Lambda_v=1/[(1-\\beta\\rho_v)(\\sigma(1-\\rho_v)+\\phi_y)+\\kappa(\\phi_\\pi-\\rho_v)]\\).
   Check Dynare's impact responses against it.
3. Report the response of the nominal rate and of employment to the policy shock and to a technology shock. Why does
   employment fall after a positive technology shock under this rule?
""",
starter=[("nk.mod", """
var y_gap pi i r_real r_nat y y_nat n a v;
varexo eps_a eps_v;
parameters beta sigma varphi alpha epsilon theta phi_pi phi_y rho_a rho_v;
beta = 0.99; sigma = 1; varphi = 5; alpha = 1/4; epsilon = 9; theta = 3/4;
phi_pi = 1.5; phi_y = 0.5/4; rho_a = 0.9; rho_v = 0.5;
model(linear);
  // TODO: kappa and psi_ya as model-local variables; NKPC, IS, Taylor rule, natural rate, production
end;
shocks; var eps_a; stderr 1; var eps_v; stderr 0.25; end;
stoch_simul(order=1, irf=12, noprint, nograph);
"""), ("ex5_1.m", """
dynare nk noclearall
% TODO: compute kappa and Lambda_v in MATLAB and compare with oo_.irfs.y_gap_eps_v(1), oo_.irfs.pi_eps_v(1)
""")],
solution=[("nk.mod", NK_MOD), ("ex5_1.m", """
dynare nk noclearall
Theta = (1 - alpha) / (1 - alpha + alpha * epsilon);
kappa = (1 - theta) * (1 - beta * theta) / theta * Theta * (sigma + (varphi + alpha) / (1 - alpha));
Lv = 1 / ((1 - beta * rho_v) * (sigma * (1 - rho_v) + phi_y) + kappa * (phi_pi - rho_v));
fprintf('kappa = %.4f, Lambda_v = %.4f\\n\\n', kappa, Lv);
fprintf('Monetary shock (25 bp):      Dynare     closed form\\n');
fprintf('  output gap, impact       %8.4f    %8.4f\\n', oo_.irfs.y_gap_eps_v(1), -(1 - beta * rho_v) * Lv * 0.25);
fprintf('  inflation (ann.), impact %8.4f    %8.4f\\n', 4 * oo_.irfs.pi_eps_v(1), -4 * kappa * Lv * 0.25);
fprintf('  nominal rate (ann.)      %8.4f\\n', 4 * oo_.irfs.i_eps_v(1));
fprintf('  real rate (ann.)         %8.4f\\n', 4 * oo_.irfs.r_real_eps_v(1));
fprintf('  employment               %8.4f\\n\\n', oo_.irfs.n_eps_v(1));
fprintf('Technology shock (1%%):      impact\\n');
for v = {'y', 'y_nat', 'y_gap', 'n', 'pi', 'i'}
  f = [v{1} '_eps_a']; s = 1 + 3 * any(strcmp(v{1}, {'pi', 'i'}));
  fprintf('  %-8s               %8.4f\\n', v{1}, s * oo_.irfs.(f)(1));
end
""")],
notes="""Dynare's impact responses match the closed-form solution to all reported digits. A 25 bp policy shock lowers the
output gap by about 0.26% and annualised inflation by about 0.35 percentage points. The nominal
rate rises by less than the shock (and the real rate by more), because the endogenous fall in inflation and output
pulls the rule back down; with a persistent shock the nominal rate can even fall. After a technology improvement,
natural output rises more than actual output because prices are sticky and the rule does not cut rates enough: the
output gap and inflation turn negative and employment falls. That is Galí's (1999) explanation for the negative
response of hours to technology shocks in the data.
"""),

dict(title="Mapping the determinacy region",
data="Model calibration.",
task="""
For the model of Exercise 5.1, Bullard and Mitra (2002) show that the equilibrium is determinate if and only if
\\(\\kappa(\\phi_\\pi-1)+(1-\\beta)\\phi_y>0\\).

1. Loop over \\(\\phi_\\pi\\in\\{0.5,0.8,0.95,1.05,1.5\\}\\) and \\(\\phi_y\\in\\{0,0.5,1\\}\\) using `set_param_value`,
   record Dynare's Blanchard–Kahn status, and compare it with the analytical condition.
2. Why can a strong enough response to the output gap make the equilibrium determinate even when \\(\\phi_\\pi<1\\)?
""",
starter=[("ex5_2.m", """
dynare nk noclearall
for py = [0 0.5 1]
  for pp = [0.5 0.8 0.95 1.05 1.5]
    set_param_value('phi_pi', pp); set_param_value('phi_y', py);
    % TODO: stoch_simul, record info(1); compare with kappa*(pp-1) + (1-beta)*py > 0
  end
end
""")],
solution=[("nk.mod", NK_MOD), ("ex5_2.m", """
dynare nk noclearall
Theta = (1 - alpha) / (1 - alpha + alpha * epsilon);
kappa = (1 - theta) * (1 - beta * theta) / theta * Theta * (sigma + (varphi + alpha) / (1 - alpha));
pps = [0.5 0.8 0.95 1.05 1.5];
fprintf('phi_y \\\\ phi_pi %s\\n', sprintf('%8.2f', pps));
agree = 0;
for py = [0 0.5 1]
  row = '';
  for pp = pps
    set_param_value('phi_pi', pp); set_param_value('phi_y', py);
    [info, oo_] = stoch_simul(M_, options_, oo_, {});
    det = info(1) == 0; theory = kappa * (pp - 1) + (1 - beta) * py > 0;
    agree = agree + (det == theory);
    if det, row = [row '     det']; else, row = [row '   indet']; end
  end
  fprintf('   %4.2f      %s\\n', py, row);
end
fprintf('Dynare agrees with kappa(phi_pi-1)+(1-beta)phi_y>0 in %d of 15 cases (kappa = %.4f)\\n', agree, kappa);
""")],
notes="""Dynare's classification matches the analytical condition everywhere on the grid. With \\(\\phi_y=0\\) the
Taylor principle \\(\\phi_\\pi>1\\) is necessary and sufficient. A response to the output gap helps because a sustained
rise in inflation is accompanied by a permanently higher output gap through the long-run Phillips curve
(\\(\\tilde y=\\pi(1-\\beta)/\\kappa\\)); an output response therefore also raises the long-run nominal rate. The general
principle is that the nominal rate must eventually rise by more than one-for-one with any permanent increase in
inflation. Because \\(1-\\beta\\) is small, a very large output response is needed to compensate for \\(\\phi_\\pi<1\\).
"""),
]

# --------------------------------------------------------------------------------------------- Lecture 6
NKLOSS_MOD = """
// Three-equation NK model with demand (natural-rate) and cost-push shocks, simple Taylor rule
var y_gap pi i r_nat u;
varexo eps_r eps_u;
parameters beta sigma kappa phi_pi phi_y rho_r rho_u;
beta = 0.99; sigma = 1; kappa = 0.1717; phi_pi = 1.5; phi_y = 0.125; rho_r = 0.8; rho_u = 0.5;
model(linear);
  pi = beta * pi(+1) + kappa * y_gap + u;
  y_gap = y_gap(+1) - (1 / sigma) * (i - pi(+1) - r_nat);
  i = phi_pi * pi + phi_y * y_gap;
  r_nat = rho_r * r_nat(-1) + eps_r;
  u = rho_u * u(-1) + eps_u;
end;
shocks; var eps_r; stderr 0.5; var eps_u; stderr 0.3; end;
stoch_simul(order=1, irf=12, noprint, nograph) pi y_gap;
"""

EXERCISES["06-optimal-monetary-policy.md"] = [
dict(title="Discretion versus commitment after a cost-push shock",
data="Model calibration (\\(\\beta=0.99,\\ \\kappa=0.1717,\\ \\epsilon=9\\)).",
task="""
The central bank minimises \\(E_0\\sum\\beta^t(\\pi_t^2+\\vartheta\\tilde y_t^2)\\) with \\(\\vartheta=\\kappa/\\epsilon\\),
subject to \\(\\pi_t=\\beta E_t\\pi_{t+1}+\\kappa\\tilde y_t+u_t\\). Consider an i.i.d. cost-push shock of 1 at \\(t=0\\).

1. Under **discretion**, \\(\\tilde y_t=-(\\kappa/\\vartheta)\\pi_t\\), giving
   \\(\\pi_t=\\frac{\\vartheta}{\\kappa^2+\\vartheta}u_t\\). Compute the paths of inflation, the output gap and the price level.
2. Under **commitment** (timeless perspective), \\(\\tilde y_t=\\tilde y_{t-1}-(\\kappa/\\vartheta)\\pi_t\\), giving
   \\(\\tilde y_t=\\delta\\tilde y_{t-1}-\\frac{\\kappa\\delta}{\\vartheta}u_t\\) with
   \\(\\delta=\\frac{1-\\sqrt{1-4\\beta a^2}}{2a\\beta}\\), \\(a=\\frac{\\vartheta}{\\vartheta(1+\\beta)+\\kappa^2}\\).
   Compute the same paths.
3. Compare the discounted losses. Why is commitment better, and what happens to the price level?
""",
starter=[("ex6_1.m", """
beta = 0.99; kappa = 0.1717; epsilon = 9; vartheta = kappa / epsilon; H = 40;
u = [1 zeros(1, H - 1)];
% TODO: discretion paths; commitment paths using delta; price level = cumsum(pi); losses
""")],
solution=[("ex6_1.m", """
beta = 0.99; kappa = 0.1717; epsilon = 9; vartheta = kappa / epsilon; H = 200;
u = [1 zeros(1, H - 1)];
% discretion
piD = vartheta / (kappa^2 + vartheta) * u;  xD = -(kappa / vartheta) * piD;
% commitment (timeless perspective)
a = vartheta / (vartheta * (1 + beta) + kappa^2);
delta = (1 - sqrt(1 - 4 * beta * a^2)) / (2 * a * beta);
xC = zeros(1, H); xlag = 0;
for t = 1:H, xC(t) = delta * xlag - kappa * delta / vartheta * u(t); xlag = xC(t); end
piC = -(vartheta / kappa) * [xC(1) diff(xC)];
disc = beta .^ (0:H-1);
LD = sum(disc .* (piD.^2 + vartheta * xD.^2)); LC = sum(disc .* (piC.^2 + vartheta * xC.^2));
fprintf('vartheta = %.4f, delta = %.4f\\n\\n', vartheta, delta);
fprintf('  h    discretion: pi      x      p      commitment: pi      x      p\\n');
pD = cumsum(piD); pC = cumsum(piC);
for h = [0 1 2 4 8 16]
  fprintf('%3d            %7.3f %7.3f %6.3f              %7.3f %7.3f %6.3f\\n', h, piD(h+1), xD(h+1), pD(h+1), ...
          piC(h+1), xC(h+1), pC(h+1));
end
fprintf('\\ndiscounted loss: discretion %.4f, commitment %.4f (%.0f%% lower)\\n', LD, LC, 100 * (1 - LC / LD));
""")],
notes="""Under discretion the central bank lets inflation rise on impact and closes the gap as soon as the shock has
gone: the price level stays permanently higher. Under commitment it promises to keep the output gap negative for
several quarters after the shock. That promise lowers expected future inflation, which improves the current trade-off
in the Phillips curve, so inflation rises less on impact (0.31 instead of 0.39) and then turns negative. The price level
returns towards its original path (it is stationary under commitment). The discounted loss is about a fifth lower. The gain
comes from managing expectations: commitment is history dependent, and that history dependence is exactly what
price-level and average-inflation targeting try to mimic (Lecture 7).
"""),

dict(title="The best simple Taylor rule (Dynare)",
data="Model calibration; shocks to the natural real rate (s.d. 0.5) and cost-push shocks (s.d. 0.3).",
task="""
Use the three-equation model with a natural-rate shock and a cost-push shock and the rule
\\(i_t=\\phi_\\pi\\pi_t+\\phi_y\\tilde y_t\\).

1. For each \\((\\phi_\\pi,\\phi_y)\\) on a grid, compute the theoretical variances of inflation and the output gap from
   `oo_.var` and the loss \\(L=\\mathrm{var}(\\pi)+\\vartheta\\,\\mathrm{var}(\\tilde y)\\) with \\(\\vartheta=\\kappa/9\\).
2. Which rule minimises the loss? Repeat with the cost-push shock switched off (set its variance to zero in
   `M_.Sigma_e`).
3. Explain the result using the "divine coincidence".
""",
starter=[("nkloss.mod", NKLOSS_MOD), ("ex6_2.m", """
dynare nkloss noclearall
vartheta = kappa / 9;
% TODO: loop over phi_pi in [1.1 1.5 2 3 5 10] and phi_y in [0 0.125 0.5 1]
%       [info, oo_] = stoch_simul(M_, options_, oo_, {'pi'; 'y_gap'}); L = oo_.var(1,1) + vartheta*oo_.var(2,2)
""")],
solution=[("nkloss.mod", NKLOSS_MOD), ("ex6_2.m", """
dynare nkloss noclearall
vartheta = kappa / 9;
pps = [1.1 1.5 2 3 5 10]; pys = [0 0.125 0.5 1];
for case_ = 1:2
  if case_ == 2, M_.Sigma_e(2, 2) = 0; end
  L = zeros(numel(pys), numel(pps));
  for a = 1:numel(pys)
    for b = 1:numel(pps)
      set_param_value('phi_pi', pps(b)); set_param_value('phi_y', pys(a));
      [info, oo_] = stoch_simul(M_, options_, oo_, {'pi'; 'y_gap'});
      L(a, b) = oo_.var(1, 1) + vartheta * oo_.var(2, 2);
    end
  end
  if case_ == 1, fprintf('Both shocks: loss x 100\\n'); else, fprintf('\\nDemand shocks only: loss x 100\\n'); end
  fprintf('phi_y \\\\ phi_pi %s\\n', sprintf('%8.1f', pps));
  for a = 1:numel(pys), fprintf('   %5.3f     %s\\n', pys(a), sprintf('%8.4f', 100 * L(a, :))); end
  [m, k] = min(L(:)); [a, b] = ind2sub(size(L), k);
  fprintf('best rule on the grid: phi_pi = %.1f, phi_y = %.3f\\n', pps(b), pys(a));
end
""")],
notes="""With only natural-rate (demand) shocks there is no trade-off: the loss keeps falling as the rule becomes more
aggressive, whether it responds to inflation or to the output gap, and it would reach zero in the limit. That is the
"divine coincidence" (Blanchard and Galí, 2007): stabilising inflation also closes the output gap, and an aggressive
rule approximates the optimal policy of tracking the natural rate. Cost-push shocks break this. Once the rule
responds strongly to inflation, adding an output-gap response raises the loss, because stabilising the gap after a
cost-push shock means letting inflation move more, and inflation carries the larger welfare weight
(\\(\\vartheta\\approx0.02\\)). The best simple rule responds very strongly to inflation and only a little to the
output gap. Responding to the gap helps only when the inflation response is weak.
"""),
]

# --------------------------------------------------------------------------------------------- Lecture 7
PLT_MOD = """
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
"""

EXERCISES["07-policy-rules-frameworks.md"] = [
dict(title="Estimating Taylor rules before and after Volcker",
data="`us_quarterly.csv` — federal funds rate, GDP deflator and real GDP, 1960–2007.",
task="""
Estimate the partial-adjustment rule \\(i_t=c+\\rho\\,i_{t-1}+(1-\\rho)(\\phi_\\pi\\pi_t+\\phi_x x_t)+\\varepsilon_t\\), where
\\(\\pi_t\\) is four-quarter GDP-deflator inflation and \\(x_t\\) is the HP-filtered output gap.

1. Estimate the regression \\(i_t=a+\\rho i_{t-1}+b\\pi_t+dx_t\\) by OLS and by 2SLS, instrumenting \\(\\pi_t\\) and \\(x_t\\)
   with lags 1–4 of inflation and 1–2 of the gap. Recover \\(\\phi_\\pi=b/(1-\\rho)\\) and \\(\\phi_x=d/(1-\\rho)\\).
2. Compare 1960Q1–1979Q2 with 1979Q3–2007Q4 (Clarida, Galí and Gertler, 2000).
3. Relate your estimates to the determinacy condition from Lecture 5.
""",
starter=[("ex7_1.m", """
T = readtable('us_quarterly.csv');
t = T.t; i = T.ffr; pi = 100 * (log(T.gdpdef) - [NaN(4,1); log(T.gdpdef(1:end-4))]);
x = 100 * (log(T.rgdp) - hp_trend(log(T.rgdp), 1600));
% TODO: build lags, choose sample, OLS and 2SLS (Xhat = Z*(Z\\X); b = Xhat\\y)
""")],
solution=[("ex7_1.m", """
T = readtable('us_quarterly.csv');
t = T.t; i = T.ffr; lp = log(T.gdpdef);
pi = 100 * (lp - [NaN(4, 1); lp(1:end-4)]);
x = 100 * (log(T.rgdp) - hp_trend(log(T.rgdp), 1600));
L = @(v, k) [NaN(k, 1); v(1:end-k)];
y = i; X = [ones(size(i)) L(i, 1) pi x];
Z = [ones(size(i)) L(i, 1) L(pi, 1) L(pi, 2) L(pi, 3) L(pi, 4) L(x, 1) L(x, 2)];
fprintf('sample            method   rho     phi_pi   phi_x\\n');
for s = {[1960 1979.25], [1979.5 2007.75]}
  r = s{1}; k = t >= r(1) & t <= r(2) & all(~isnan([X Z]), 2);
  bo = X(k, :) \\ y(k);
  Xh = Z(k, :) * (Z(k, :) \\ X(k, :)); bi = Xh \\ y(k);
  for m = 1:2
    if m == 1, b = bo; nm = 'OLS '; else, b = bi; nm = '2SLS'; end
    fprintf('%7.2f-%7.2f   %s   %5.2f   %6.2f   %6.2f\\n', r(1), r(2), nm, b(2), b(3) / (1 - b(2)), b(4) / (1 - b(2)));
  end
end
""")],
notes="""Both samples show strong interest-rate smoothing (\\(\\rho\\) around 0.8). Before 1979 the long-run response to
inflation is below one: the Fed let real rates fall when inflation rose, which violates the Taylor principle and, in
the New Keynesian model, leaves the door open to self-fulfilling inflation (Clarida, Galí and Gertler, 2000). After
1979 the response is well above one. Caveats: the output gap is measured with hindsight, not in real time (Orphanides,
2001, finds that real-time gaps were badly overestimated in the 1970s), and contemporaneous inflation is used instead
of the forecasts the Fed actually reacted to. Estimates are therefore sensitive to the gap measure and instruments,
but the shift in the inflation response is a robust finding.
"""),

dict(title="Inflation targeting versus price-level targeting (Dynare)",
data="Model calibration; an i.i.d. cost-push shock of 1.",
task="""
Under inflation targeting (IT) the rule is \\(i_t=\\phi\\pi_t\\); under price-level targeting (PLT) it is
\\(i_t=\\phi p_t\\), where \\(p_t\\) is the log price level relative to its target path.

1. Use one Dynare model with a switch parameter `plt` and compare the responses of inflation, the output gap and the
   price level to a cost-push shock.
2. Compute the discounted loss \\(\\sum\\beta^t(\\pi_t^2+\\vartheta\\tilde y_t^2)\\) from the impulse responses for both
   regimes (\\(\\vartheta=\\kappa/9\\)).
3. Relate the PLT outcome to the commitment solution in Exercise 6.1.
""",
starter=[("plt.mod", PLT_MOD), ("ex7_2.m", """
dynare plt noclearall
% TODO: for plt = [0 1]: set_param_value, stoch_simul, read oo_.irfs.pi_eps_u, y_gap_eps_u, p_eps_u
""")],
solution=[("plt.mod", PLT_MOD), ("ex7_2.m", """
dynare plt noclearall
vartheta = kappa / 9; H = options_.irf; disc = beta .^ (0:H-1);
names = {'IT ', 'PLT'};
for reg = [0 1]
  set_param_value('plt', reg);
  [info, oo_] = stoch_simul(M_, options_, oo_, {});
  ir = oo_.irfs; Lr = sum(disc .* (ir.pi_eps_u.^2 + vartheta * ir.y_gap_eps_u.^2));
  fprintf('%s  pi: %s\\n', names{reg+1}, sprintf('%7.3f', ir.pi_eps_u([1 2 3 5 9])));
  fprintf('     x : %s\\n', sprintf('%7.3f', ir.y_gap_eps_u([1 2 3 5 9])));
  fprintf('     p : %s   (h = 0, 1, 2, 4, 8)\\n', sprintf('%7.3f', ir.p_eps_u([1 2 3 5 9])));
  fprintf('     price level after 40 quarters %.3f, discounted loss %.4f\\n\\n', ir.p_eps_u(end), Lr);
end
""")],
notes="""Under IT the price level drifts up permanently after the shock ("bygones are bygones"). Under PLT the central bank
must bring the price level back, so after the initial rise inflation turns negative for a while. Because firms expect
this, they raise prices by less on impact: inflation rises by 0.46 instead of 0.80. The cost is a deeper and longer
fall in output, but with the welfare weights implied by the model the discounted loss is almost halved, even though
neither rule is optimal. PLT builds in the history dependence of
the commitment solution in Exercise 6.1, where the price level is also stationary. This is the logic behind the
Fed's 2020 move to flexible average-inflation targeting, and it depends on the private sector believing the promise
to make up past misses.
"""),
]

# --------------------------------------------------------------------------------------------- Lecture 8
SOE_MOD = """
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
"""

EXERCISES["08-small-open-economy.md"] = [
dict(title="Policy regimes in a small open economy (Dynare)",
data="Calibration of Galí and Monacelli (2005): \\(\\alpha=0.4\\), \\(\\varphi=3\\), \\(\\rho_a=0.66\\), \\(\\sigma_a=0.71\\%\\).",
task="""
Implement the Galí–Monacelli small open economy with a domestic technology shock. Use Dynare's macro processor
(`@#if`) to switch between three regimes: a Taylor rule on domestic inflation (DITR), a Taylor rule on CPI inflation
(CITR) and an exchange-rate peg (PEG).

1. Run the model three times with `dynare soe -Dpolicy=1` (2, 3) and collect the standard deviations of the output
   gap, domestic and CPI inflation, nominal depreciation and the terms of trade.
2. In the special case \\(\\sigma=\\eta=\\gamma=1\\), the welfare loss is
   \\(\\frac{1-\\alpha}{2}\\left[\\frac{\\epsilon}{\\lambda}\\mathrm{var}(\\pi_H)+(1+\\varphi)\\mathrm{var}(x)\\right]\\).
   Compute it for each regime and rank them.
3. Repeat the domestic-inflation rule with \\(\\phi_\\pi=5\\). What does this tell you about the optimal policy?
4. Which regime stabilises the exchange rate best, and at what cost?
""",
starter=[("soe.mod", """
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
"""), ("ex8_1.m", """
for pol = 1:3
  eval(sprintf('dynare soe -Dpolicy=%d noclearall', pol));
  % TODO: sqrt(diag(oo_.var)); inflation and depreciation annualised (x4)
end
""")],
solution=[("soe.mod", SOE_MOD), ("ex8_1.m", """
names = {'DITR', 'CITR', 'PEG '};
fprintf('regime   sd(x)   sd(pi_H)  sd(pi)   sd(de)   sd(s)   welfare loss\\n');
for pol = 1:3
  eval(sprintf('dynare soe -Dpolicy=%d noclearall', pol));
  sd = sqrt(diag(oo_.var))';           % order: x pi_H pi de s y
  lam = (1 - theta) * (1 - beta * theta) / theta;
  W = (1 - alpha) / 2 * ((epsilon / lam) * oo_.var(2, 2) + (1 + varphi) * oo_.var(1, 1));   % Gali-Monacelli loss
  fprintf('%s   %6.3f  %7.3f  %7.3f  %7.3f  %6.3f   %8.4f\\n', names{pol}, sd(1), 4 * sd(2), 4 * sd(3), 4 * sd(4), sd(5), W);
end
dynare soe -Dpolicy=1 noclearall                % domestic-inflation rule again, now responding more strongly
set_param_value('phi_pi', 5); [info, oo_] = stoch_simul(M_, options_, oo_, var_list_);
sd = sqrt(diag(oo_.var))'; W = (1 - alpha) / 2 * ((epsilon / lam) * oo_.var(2, 2) + (1 + varphi) * oo_.var(1, 1));
fprintf('DITR5  %6.3f  %7.3f  %7.3f  %7.3f  %6.3f   %8.4f   (phi_pi = 5)\\n', sd(1), 4 * sd(2), 4 * sd(3), 4 * sd(4), sd(5), W);
fprintf('(standard deviations in %%; inflation and depreciation annualised; loss per period, in %% squared)\\n');
""")],
notes="""The domestic-inflation rule gives the most stable output gap but the most volatile exchange rate: the nominal
exchange rate and the terms of trade absorb the shock. The CPI rule leans against depreciation because import prices
enter CPI inflation, so it halves the volatility of depreciation and CPI inflation at the cost of a more volatile
output gap. The peg eliminates nominal exchange-rate volatility, but the terms of trade can then adjust only through
domestic prices, so the output gap is twice as volatile as under the domestic-inflation rule. With
\\(\\phi_\\pi=1.5\\) none of the simple rules tracks the fall in the natural rate after a technology improvement, and in
this calibration the CPI rule happens to give the smallest loss. The point of Galí and Monacelli (2005) is that the
optimal policy is strict domestic-inflation targeting (\\(\\pi_H=x=0\\)): a domestic-inflation rule that responds
strongly enough comes close to it, and its loss is far below all three regimes, even though the exchange rate is then
very volatile. The model ignores balance-sheet and pass-through effects that make emerging-market central banks wary
of large depreciations (Exercise 8.2).
"""),

dict(title="Exchange-rate pass-through in Korea",
data="`cpi_inflation_wdi.csv` (Korea CPI inflation) and `krw_usd_monthly.csv` (KRW per USD, 1981–2026).",
task="""
1. Build annual Korean CPI inflation and the annual percentage change in the average KRW/USD rate.
2. Compare the mean and volatility of inflation and depreciation before the 1997–98 crisis (1982–1996) and under
   inflation targeting with a floating rate (1999–2024).
3. For 1999–2024 estimate \\(\\pi_t=a+b\\,\\Delta e_t+c\\,\\pi_{t-1}+\\varepsilon_t\\) and the long-run pass-through
   \\(b/(1-c)\\). Compare it with the pass-through implied by the 1998 episode, \\((\\pi_{1998}-\\pi_{1997})/\\Delta e_{1998}\\).
4. Why might pass-through be low under a credible inflation target?
""",
starter=[("ex8_2.m", """
W = readtable('cpi_inflation_wdi.csv'); X = readtable('krw_usd_monthly.csv');
k = strcmp(W.code, 'KOR'); yr = W.year(k); infl = W.cpi_inflation(k);
% TODO: annual average exchange rate with accumarray; depreciation; align years; summary statistics; ols_nw
""")],
solution=[("ex8_2.m", """
W = readtable('cpi_inflation_wdi.csv'); X = readtable('krw_usd_monthly.csv');
k = strcmp(W.code, 'KOR'); yr = W.year(k); infl = W.cpi_inflation(k);
ys = unique(X.year); E = accumarray(X.year - ys(1) + 1, X.krw_usd, [], @mean);
dep = [NaN; 100 * diff(log(E))];                        % annual % change in KRW per USD (+ = depreciation)
years = (1982:2024)';
pi_t = arrayfun(@(y) infl(yr == y), years); pi_l = arrayfun(@(y) infl(yr == y - 1), years);
de = arrayfun(@(y) dep(ys == y), years);
fprintf('period       mean infl   sd infl   mean dep   sd dep\\n');
for s = {[1982 1996], [1999 2024]}
  r = s{1}; m = years >= r(1) & years <= r(2);
  fprintf('%d-%d     %6.1f     %6.1f     %6.1f    %6.1f\\n', r(1), r(2), mean(pi_t(m)), std(pi_t(m)), mean(de(m)), std(de(m)));
end
m = years >= 1999;
[b, se] = ols_nw(pi_t(m), [ones(sum(m), 1) de(m) pi_l(m)], 2);
fprintf('\\n1999-2024: infl = %.2f + %.3f (%.3f) dep + %.2f (%.2f) infl(-1); long-run pass-through %.3f\\n', ...
        b(1), b(2), se(2), b(3), se(3), b(2) / (1 - b(3)));
k98 = years == 1998; k97 = years == 1997;
fprintf('1998: depreciation %.1f%%, inflation %.1f%% (1997: %.1f%%), implied pass-through %.3f\\n', ...
        de(k98), pi_t(k98), pi_t(k97), (pi_t(k98) - pi_t(k97)) / de(k98));
""")],
notes="""Before the crisis the won was tightly managed — depreciation was small and stable — while inflation averaged
about 5%. Under inflation targeting since 1999, inflation is lower and more stable even though the exchange rate moves
far more. Pass-through is small: a 10% depreciation raises inflation by roughly half a percentage point in the year it
occurs. Even the 1998 collapse, a depreciation of almost 40%, raised inflation by only about 3 points. Taylor (2000)
argued that pass-through falls when inflation is low and stable, because firms expect exchange-rate changes to be
temporary and offset by policy, so they adjust prices less. With about 26 annual observations the estimate is
imprecise. Bangladesh, with a managed exchange rate and higher inflation, is a useful comparison if you can obtain its
exchange-rate series.
"""),
]

# --------------------------------------------------------------------------------------------- Lecture 9
FA_MOD = """
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
"""

EXERCISES["09-financial-frictions.md"] = [
dict(title="A financial accelerator in the New Keynesian model (Dynare)",
data="Model calibration; 25 bp monetary-policy shock.",
task="""
Add a credit spread to the IS curve: spending depends on the borrowing rate \\(i_t+s_t\\), and the spread falls with
borrower net worth, \\(s_t=-\\chi\\,nw_t\\), where \\(nw_t=\\rho_n nw_{t-1}+\\tilde y_t\\).

1. Compare the responses to a 25 bp monetary tightening for \\(\\chi\\in\\{0,0.05,0.1\\}\\): impact and cumulative
   output gap, inflation and the spread.
2. Explain the amplification and the persistence it adds. Which features of Bernanke, Gertler and Gilchrist (1999) does
   this reduced form capture, and which does it miss?
""",
starter=[("fa.mod", """
var y_gap pi i s nw v;
varexo eps_v;
parameters beta sigma kappa phi_pi phi_y chi rho_n rho_v;
beta = 0.99; sigma = 1; kappa = 0.1717; phi_pi = 1.5; phi_y = 0.125; chi = 0.05; rho_n = 0.9; rho_v = 0.5;
model(linear);
  // TODO: NKPC, IS curve with the borrowing rate i + s, spread, net worth, Taylor rule, shock
end;
shocks; var eps_v; stderr 0.25; end;
stoch_simul(order=1, irf=20, noprint, nograph);
"""), ("ex9_1.m", """
dynare fa noclearall
% TODO: loop over chi with set_param_value; report impact and cumulative responses
""")],
solution=[("fa.mod", FA_MOD), ("ex9_1.m", """
dynare fa noclearall
fprintf('chi    output gap: impact   cum(0-19)   trough qtr   infl (ann.) impact   spread (ann.) peak\\n');
for c = [0 0.05 0.1]
  set_param_value('chi', c);
  [info, oo_] = stoch_simul(M_, options_, oo_, {});
  ir = oo_.irfs; [mn, q] = min(ir.y_gap_eps_v);
  sp = 0; if isfield(ir, 's_eps_v'), sp = 4 * max(ir.s_eps_v); end
  fprintf('%4.2f        %7.3f      %7.3f        %2d            %7.3f             %6.3f\\n', c, ...
          ir.y_gap_eps_v(1), sum(ir.y_gap_eps_v), q - 1, 4 * ir.pi_eps_v(1), sp);
end
""")],
notes="""With \\(\\chi=0\\) the model is the basic New Keynesian model and the output gap recovers as fast as the policy
shock decays. With a financial accelerator, the tightening lowers output and profits, borrower net worth falls, and
the external finance premium rises. The higher borrowing cost depresses spending further, so the cumulative output
loss is 35–65% larger and the recovery slower, because net worth is rebuilt only gradually. Inflation falls more too.
The impact response barely changes, because net worth is a stock that erodes over time. This captures the
core BGG mechanism: credit conditions amplify and propagate shocks through balance sheets. It misses the explicit
contract and default problem, the role of asset prices and leverage, and the possibility that shocks start in the
financial sector itself (Gertler–Karadi, 2011) — as in 2008.
"""),

dict(title="How do credit spreads respond to monetary policy surprises?",
data="`mp_shocks_monthly.csv` (Jarociński–Karadi monetary-policy shocks, 1990–2024) and `us_monthly.csv` (Baa and "
     "10-year Treasury yields, 1-year Treasury yield).",
task="""
Use local projections (Jordà, 2005) to estimate the response of the Baa–Treasury spread to monetary-policy surprises:
\\[ y_{t+h}-y_{t-1}=a_h+\\beta_h\\,\\text{shock}_t+\\sum_{j=1}^{3}\\gamma_{h,j}\\Delta y_{t-j}+u_{t+h}. \\]

1. Scale the Jarociński–Karadi "pure monetary policy" shock so that it raises the 1-year Treasury yield by 25 bp on
   impact (estimate the \\(h=0\\) response of the 1-year yield first).
2. Estimate \\(\\beta_h\\) for \\(h=0,\\dots,24\\) months for the 1-year yield and the Baa spread, 1990–2019, with
   Newey–West standard errors.
3. Does a monetary tightening widen credit spreads, as the financial-accelerator mechanism predicts?
""",
starter=[("ex9_2.m", """
D = readtable('us_monthly.csv'); S = readtable('mp_shocks_monthly.csv');
% TODO: align by year and month; spread = baa - gs10; loop h = 0:24; ols_nw with L = h + 1
""")],
solution=[("ex9_2.m", """
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
fprintf('scale: a shock of %.3f units raises the 1-year yield by 25 bp\\n', scale);
fprintf('h (months)   %s\\n', sprintf('%7d', [0 3 6 12 18 24]));
for v = 1:2
  fprintf('%s %s\\n', names{v}, sprintf('%7.3f', scale * B(v, [0 3 6 12 18 24] + 1)));
  fprintf('   (s.e.)    %s\\n', sprintf('%7.3f', abs(scale) * SE(v, [0 3 6 12 18 24] + 1)));
end
""")],
notes="""The 1-year yield rises by 25 bp by construction and keeps rising for about a year, although the later
responses are imprecise. The raw Baa–Treasury spread shows no clear response: the point estimates are mostly positive
but within one or two standard errors of zero. This is a useful negative result. Gertler and Karadi (2015) find a
significant rise in the excess bond premium, a cleaner measure of credit risk premia built from bond-level data that
removes expected default; the Baa index spread also moves with liquidity, duration and default expectations, which
adds noise. The Jarociński–Karadi shocks separate pure policy surprises from central-bank information effects using
the stock-market response, which removes one bias but not the noise. Detecting the financial-accelerator channel
needs good measures of balance-sheet strength and risk premia, not just more precise shocks.
"""),
]

# --------------------------------------------------------------------------------------------- Lecture 10
EXERCISES["10-zlb-unconventional-policy.md"] = [
dict(title="A liquidity trap and the value of forward guidance",
data="Model calibration: \\(\\beta=0.99,\\ \\sigma=1,\\ \\kappa=0.02\\); the natural rate falls to −1% a quarter and "
     "stays there with probability \\(\\mu=0.8\\) each quarter.",
task="""
Follow Eggertsson and Woodford (2003). The natural real rate is \\(r^n_L<0\\) while the trap lasts and returns to
\\(r^*=1/\\beta-1\\) with probability \\(1-\\mu\\) each quarter, after which policy achieves \\(\\pi=\\tilde y=0\\). While the
trap lasts \\(i=0\\).

1. Show that in the trap \\(\\tilde y_L=\\frac{r^n_L/\\sigma}{(1-\\mu)-\\mu\\kappa/[\\sigma(1-\\beta\\mu)]}\\) and
   \\(\\pi_L=\\kappa\\tilde y_L/(1-\\beta\\mu)\\), and compute them.
2. **Forward guidance:** the central bank promises to keep \\(i=0\\) for one extra quarter after the natural rate
   recovers. In that exit quarter \\(\\tilde y_E=r^*/\\sigma\\) and \\(\\pi_E=\\kappa\\tilde y_E\\). Recompute the trap
   outcomes.
3. How much of the recession does one quarter of forward guidance prevent? Why is the effect so large?
""",
starter=[("ex10_1.m", """
beta = 0.99; sigma = 1; kappa = 0.02; mu = 0.8; rL = -0.01; rstar = 1 / beta - 1;
% TODO: solve the two linear equations for (x_L, pi_L), first with x_E = pi_E = 0, then with forward guidance
""")],
solution=[("ex10_1.m", """
beta = 0.99; sigma = 1; kappa = 0.02; mu = 0.8; rL = -0.01; rstar = 1 / beta - 1;
% in the trap (i = 0):  x_L = mu x_L + (1-mu) x_E + (1/sigma)(mu pi_L + (1-mu) pi_E + rL)
%                       pi_L = kappa x_L + beta (mu pi_L + (1-mu) pi_E)
A = [1 - mu, -mu / sigma; -kappa, 1 - beta * mu];
fprintf('                         output gap (%%)   inflation (%%, annualised)\\n');
for fg = [0 1]
  xE = fg * rstar / sigma; piE = kappa * xE;                % exit quarter with i = 0 instead of r*
  rhs = [(1 - mu) * xE + ((1 - mu) * piE + rL) / sigma; beta * (1 - mu) * piE];
  sol = A \\ rhs;
  if fg == 0, nm = 'no forward guidance  '; base = sol; else, nm = '1 qtr of guidance    '; end
  fprintf('%s      %7.2f            %7.2f\\n', nm, 100 * sol(1), 400 * sol(2));
end
fprintf('expected trap length: %.1f quarters; guidance removes %.0f%% of the output loss\\n', 1 / (1 - mu), ...
        100 * (1 - sol(1) / base(1)));
""")],
notes="""With a natural rate of −4% a year and an expected trap of five quarters, the zero bound turns a moderate shock into
a deep recession with deflation. Falling inflation raises the real rate when it should fall, which lowers output,
which lowers inflation further. This feedback is why the denominator can approach zero (a "deflationary spiral"). A
promise to hold the rate at zero for a single additional quarter, once the shock is over, raises expected output and
inflation in every trap state; higher expected inflation lowers today's real rate, and the improvement compounds
through the same loop. In this calibration that single quarter removes about a fifth of the output loss. The catch is
credibility: when the trap ends, the central bank is tempted to renege (Lecture 6).
"""),

dict(title="The forward-guidance puzzle",
data="Model calibration: \\(\\sigma=1\\), \\(\\kappa=0.1717\\), \\(\\beta=0.99\\).",
task="""
Under perfect foresight, suppose the central bank announces at \\(t=0\\) that the real rate will be 25 bp lower in
quarter \\(T\\) only. With the standard Euler equation \\(\\tilde y_t=\\tilde y_{t+1}-\\sigma^{-1}r_t\\) and Phillips curve
\\(\\pi_t=\\beta\\pi_{t+1}+\\kappa\\tilde y_t\\), solve backwards from \\(\\tilde y=\\pi=0\\) after \\(T\\).

1. Compute \\(\\tilde y_0\\) and \\(\\pi_0\\) for \\(T\\in\\{0,4,8,20,40\\}\\).
2. Repeat with the "discounted" Euler equation of McKay, Nakamura and Steinsson (2016),
   \\(\\tilde y_t=M\\tilde y_{t+1}-\\sigma^{-1}r_t\\) with \\(M=0.97\\).
3. Why does the standard model imply that distant promises are so powerful? What resolves the puzzle?
""",
starter=[("ex10_2.m", """
sigma = 1; kappa = 0.1717; beta = 0.99; dr = -0.25;
% TODO: for each T and M in {1, 0.97}: backward recursion over t = T, T-1, ..., 0
""")],
solution=[("ex10_2.m", """
sigma = 1; kappa = 0.1717; beta = 0.99; dr = -0.25;
fprintf('   T    standard Euler: x0    pi0 (ann.)   |   discounted Euler (M = 0.97): x0   pi0 (ann.)\\n');
for T = [0 4 8 20 40]
  out = zeros(1, 4);
  for j = 1:2
    M = 1 - 0.03 * (j == 2); x = 0; p = 0;
    for t = T:-1:0
      r = dr * (t == T);
      x = M * x - r / sigma; p = beta * p + kappa * x;
    end
    out(2*j-1:2*j) = [x 4 * p];
  end
  fprintf('%4d        %6.3f      %7.3f      |                    %6.3f      %7.3f\\n', T, out);
end
""")],
notes="""With the standard Euler equation the output gap today rises by the same 0.25% whatever the horizon of the
promise, because today's output depends on the sum of all future real rates. Inflation today rises more the more
distant the promise, because the higher output path accumulates in the forward-looking Phillips curve: a promise
ten years out has a far larger effect on today's inflation than an immediate cut. That is the forward-guidance
puzzle (Del Negro, Giannoni and Patterson, 2012). With the discounted Euler equation, which arises with incomplete
markets and borrowing constraints (McKay, Nakamura and Steinsson, 2016) or bounded rationality, the effect of distant
promises dies out. Heterogeneous-agent models (Lecture 12) give a microfoundation.
"""),
]

# --------------------------------------------------------------------------------------------- Lecture 11
LEEPER_MOD = """
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
"""

EXERCISES["11-monetary-fiscal-interactions.md"] = [
dict(title="Active and passive policy regimes (Dynare)",
data="Model calibration; government debt of 60% of annual GDP (\\(\\bar b=2.4\\) quarters of GDP).",
task="""
Add one-period nominal government debt to the New Keynesian model. Real debt evolves as
\\(b_t=\\beta^{-1}b_{t-1}+\\beta^{-1}\\bar b(i_{t-1}-\\pi_t)-s_t\\), and the primary surplus follows
\\(s_t=\\psi b_{t-1}-z_t\\), where \\(z_t\\) is a transfer shock.

1. Loop over \\(\\phi_\\pi\\in\\{0.5,1.5\\}\\) and \\(\\psi\\in\\{0,0.05\\}\\) and record the Blanchard–Kahn status. Label
   each combination using Leeper's (1991) active/passive terminology.
2. Compare the responses of inflation, the output gap and debt to a transfer shock (1% of quarterly GDP) in the two
   regimes with a unique equilibrium.
3. Why does a debt-financed transfer raise inflation in one regime but not the other?
""",
starter=[("leeper.mod", """
var y_gap pi i b sp z;
varexo eps_z;
parameters beta sigma kappa phi_pi psi bbar rho_z;
beta = 0.99; sigma = 1; kappa = 0.1717; phi_pi = 1.5; psi = 0.05; bbar = 2.4; rho_z = 0.5;
model(linear);
  // TODO: NKPC, IS curve, policy rule, government budget constraint, surplus rule, transfer shock
end;
shocks; var eps_z; stderr 1; end;
stoch_simul(order=1, irf=20, noprint, nograph);
"""), ("ex11_1.m", """
dynare leeper noclearall
% TODO: double loop over phi_pi and psi; [info, oo_] = stoch_simul(M_, options_, oo_, {});
""")],
solution=[("leeper.mod", LEEPER_MOD), ("ex11_1.m", """
dynare leeper noclearall
status = {'unique', '', 'no stable solution', 'indeterminate'};
lab = @(pp, ps) [char('passive M' * (pp < 1) + 'active M ' * (pp >= 1)) ' / ' ...
                 char('active F ' * (ps < 1/beta - 1) + 'passive F' * (ps >= 1/beta - 1))];
for pp = [1.5 0.5]
  for ps = [0.05 0]
    set_param_value('phi_pi', pp); set_param_value('psi', ps);
    [info, oo_] = stoch_simul(M_, options_, oo_, {});
    fprintf('phi_pi = %.1f, psi = %.2f  (%s): %s\\n', pp, ps, lab(pp, ps), status{info(1) + (info(1) == 0)});
    if info(1) == 0
      ir = oo_.irfs; f = @(v) ir.([v '_eps_z']);
      if isfield(ir, 'pi_eps_z')
        fprintf('    pi (ann.) h=0..3: %s   price level after 20q: %.3f%%\\n', sprintf('%7.3f', 4 * f('pi')(1:4)), sum(f('pi')));
        fprintf('    output gap h=0..3: %s\\n', sprintf('%7.3f', f('y_gap')(1:4)));
      else
        fprintf('    inflation and output do not respond (Ricardian)\\n');
      end
      fprintf('    debt h=0,4,8,19:   %s\\n', sprintf('%7.3f', f('b')([1 5 9 20])));
    end
  end
end
""")],
notes="""Only two combinations deliver a unique equilibrium: active monetary with passive fiscal policy (\\(\\phi_\\pi>1\\),
\\(\\psi>1/\\beta-1\\)), and passive monetary with active fiscal policy (\\(\\phi_\\pi<1\\), \\(\\psi=0\\)). With both
policies active there is no stable solution — each tries to pin down the price level — and with both passive the
equilibrium is indeterminate. In the first regime a transfer is financed by higher future surpluses: debt rises and is
then paid down, and inflation and output do not move at all (Ricardian equivalence holds in this model). In the second
regime no future surpluses are promised, so the real value of debt must fall: inflation rises, the price level jumps
and output rises temporarily. That is the fiscal theory of the price level (Leeper, 1991; Sims, 1994; Cochrane, 2023),
one candidate explanation of the 2021–23 inflation after large unfunded transfers.
"""),

dict(title="Inflation and the U.S. debt ratio, 2019–2024",
data="`us_quarterly.csv` — gross federal debt (`fed_debt`, billions of dollars), real GDP and the GDP deflator.",
task="""
The change in the debt-to-GDP ratio \\(b_t=D_t/Y_t\\) can be written as
\\[ \\Delta b_t=\\underbrace{\\frac{D_t-D_{t-1}}{Y_t}}_{\\text{new borrowing}}-\\underbrace{b_{t-1}\\frac{g_t}{1+g_t}}_{\\text{nominal growth}}, \\]
where \\(g_t\\) is nominal GDP growth. Split the nominal-growth term into a part due to inflation and a part due to
real growth in proportion to their shares of \\(g_t\\).

1. Compute the decomposition for each year 2015–2024 using fourth-quarter values (GDP at an annual rate).
2. How much did inflation lower the debt ratio in 2021–2023? Compare it with the new borrowing.
3. Why does surprise inflation reduce the real value of debt, and why would this effect be smaller if debt were
   short-term or inflation-indexed?
""",
starter=[("ex11_2.m", """
T = readtable('us_quarterly.csv');
q4 = mod(T.t, 1) == 0.75; yr = floor(T.t(q4));
D = T.fed_debt(q4);                              % billions
Y = T.rgdp(q4) .* T.gdpdef(q4) / 100;            % nominal GDP, billions at an annual rate
% TODO: b = D./Y; decomposition year by year
""")],
solution=[("ex11_2.m", """
T = readtable('us_quarterly.csv');
q4 = mod(T.t, 1) == 0.75; yr = floor(T.t(q4));
D = T.fed_debt(q4);                              % billions
Y = T.rgdp(q4) .* T.gdpdef(q4) / 100;            % nominal GDP, billions at an annual rate
P = T.gdpdef(q4); b = 100 * D ./ Y;
fprintf('                          contribution to the change (points of GDP)\\nyear   debt/GDP   change   new borrowing   inflation   real growth\\n');
for t = find(yr >= 2015 & yr <= 2024)'
  g = Y(t) / Y(t-1) - 1; infl = P(t) / P(t-1) - 1; real = g - infl;
  nb = 100 * (D(t) - D(t-1)) / Y(t); ng = b(t-1) * g / (1 + g);
  fprintf('%d   %6.1f    %+5.1f       %+5.1f        %+5.1f        %+5.1f\\n', yr(t), b(t), b(t) - b(t-1), nb, ...
          -ng * infl / g, -ng * real / g);
end
k = yr >= 2021 & yr <= 2023; kk = find(k);
fprintf('2021-23: inflation lowered the ratio by %.1f points in total\\n', ...
        sum(arrayfun(@(t) b(t-1) * (Y(t)/Y(t-1) - 1) / (Y(t)/Y(t-1)) * (P(t)/P(t-1) - 1) / (Y(t)/Y(t-1) - 1), kk)));
""")],
notes="""In 2020 new borrowing raised gross federal debt by about 21% of GDP and the ratio jumped to about 127%. Over
2021–2023 new borrowing added roughly another 24 points, yet the ratio ended 2023 below its 2020 peak: inflation
lowered it by about 17 points in total, and real growth by about 12 more. Surprise inflation devalues
nominal debt whose interest rate was fixed before the inflation arrived. With long maturities the loss is borne by
existing bondholders and the effect on the ratio is large; with short-term or inflation-indexed debt, interest costs
would rise with inflation and most of the erosion would disappear. Gross debt includes debt held by government trust
funds; debt held by the public gives smaller numbers but the same pattern. (The series is reconstructed from FRED-QD's
deflated debt series and the GDP deflator, so values differ slightly from Treasury figures.) Whether this was a
fiscal-theory adjustment or a by-product of supply shocks is still debated.
"""),
]

# --------------------------------------------------------------------------------------------- Lecture 12
TANK_MOD = """
// Two-agent New Keynesian model (Bilbiie, 2008, 2020): share lambda of hand-to-mouth households whose
// income moves chi-for-one with aggregate income
var y_gap pi i v;
varexo eps_v;
parameters beta sigma kappa phi_pi phi_y lambda chi rho_v;
beta = 0.99; sigma = 1; kappa = 0.1717; phi_pi = 1.5; phi_y = 0.125; lambda = 0.2; chi = 1.5; rho_v = 0.5;
model(linear);
  # delta = (1 - lambda) / (1 - lambda * chi);           // aggregate demand elasticity relative to RANK
  pi = beta * pi(+1) + kappa * y_gap;
  y_gap = y_gap(+1) - (delta / sigma) * (i - pi(+1));
  i = phi_pi * pi + phi_y * y_gap + v;
  v = rho_v * v(-1) + eps_v;
end;
shocks; var eps_v; stderr 0.25; end;
stoch_simul(order=1, irf=12, noprint, nograph);
"""

EXERCISES["12-hank-monetary-transmission.md"] = [
dict(title="Monetary policy in a two-agent New Keynesian model (Dynare)",
data="Model calibration; 25 bp monetary-policy shock.",
task="""
In Bilbiie's TANK model a share \\(\\lambda\\) of households are hand-to-mouth and their income moves \\(\\chi\\)-for-one
with aggregate income. The aggregate IS curve becomes \\(\\tilde y_t=E_t\\tilde y_{t+1}-\\frac{\\delta}{\\sigma}(i_t-E_t\\pi_{t+1})\\)
with \\(\\delta=\\frac{1-\\lambda}{1-\\lambda\\chi}\\).

1. Compare the impact responses of output and inflation for \\(\\lambda\\in\\{0,0.2,0.4\\}\\) when \\(\\chi=1.5\\)
   (inequality is countercyclical) and when \\(\\chi=0.5\\) (procyclical).
2. When does heterogeneity amplify monetary policy, and when does it dampen it?
3. What happens as \\(\\lambda\\chi\\) approaches one?
""",
starter=[("tank.mod", TANK_MOD), ("ex12_1.m", """
dynare tank noclearall
% TODO: loops over chi and lambda; print delta and oo_.irfs.y_gap_eps_v(1), oo_.irfs.pi_eps_v(1)
""")],
solution=[("tank.mod", TANK_MOD), ("ex12_1.m", """
dynare tank noclearall
fprintf(' chi   lambda   delta    output gap (impact)   inflation (ann., impact)\\n');
for c = [1.5 0.5]
  for l = [0 0.2 0.4]
    set_param_value('chi', c); set_param_value('lambda', l);
    [info, oo_] = stoch_simul(M_, options_, oo_, {});
    fprintf('%4.1f    %4.1f    %5.3f        %7.3f               %7.3f\\n', c, l, (1 - l) / (1 - l * c), ...
            oo_.irfs.y_gap_eps_v(1), 4 * oo_.irfs.pi_eps_v(1));
  end
end
""")],
notes="""When hand-to-mouth income is more cyclical than average (\\(\\chi>1\\)), \\(\\delta>1\\) and monetary policy has
larger effects than in the representative-agent model: a tightening lowers aggregate income, which falls
disproportionately on households who spend everything they earn, so the indirect (income) effects multiply the
direct effect. With \\(\\chi<1\\), for example because profits that accrue to savers are more cyclical than wages or
because transfers cushion the poor, heterogeneity dampens transmission. As \\(\\lambda\\chi\\to1\\) the multiplier
explodes, and with \\(\\lambda\\chi>1\\) the IS curve slopes the wrong way ("inverted aggregate demand logic") and the
determinacy conditions change. Whether HANK amplifies or dampens monetary policy is therefore an empirical question
about the incidence of income fluctuations (Bilbiie, 2020; Patterson, 2023).
"""),

dict(title="Marginal propensities to consume in a heterogeneous-agent household block",
data="Model calibration: quarterly, CRRA \\(\\sigma=2\\), log income AR(1) with \\(\\rho=0.966\\), \\(\\sigma_e=0.13\\) "
     "(annualised persistence 0.87), borrowing limit 0, real rate 1% a year.",
task="""
Solve the income-fluctuation problem \\(\\max E\\sum\\beta^tc_t^{1-\\sigma}/(1-\\sigma)\\) subject to
\\(a_{t+1}=(1+r)a_t+y_t-c_t\\), \\(a_{t+1}\\ge0\\), by the endogenous grid method (Carroll, 2006), with income discretised by
the Rouwenhorst method (7 states).

1. Compute the stationary wealth distribution for \\(\\beta=0.985\\) (high wealth) and \\(\\beta=0.96\\) (low wealth).
2. For each, report mean assets relative to mean quarterly income, the share of households at the borrowing limit and
   the average quarterly MPC out of a one-off income gain of 1% of mean annual income.
3. Why do HANK models need many households with little liquid wealth, and how do two-asset models (Kaplan, Moll and
   Violante, 2018) square this with large average wealth?
""",
starter=[("ex12_2.m", """
sig = 2; r = 0.01/4; rho = 0.966; se = 0.13; N = 7;
% Rouwenhorst: p = (1+rho)/2; build the N x N transition matrix recursively; grid of log income +-sqrt(N-1)*sd
agrid = linspace(0, 1, 400)'.^2 * 200;     % asset grid (multiples of mean quarterly income)
% TODO: EGM iteration on the Euler equation; then the stationary distribution by iterating on the histogram
""")],
solution=[("ex12_2.m", """
sig = 2; r = 0.01/4; rho = 0.966; se = 0.13; N = 7;
% Rouwenhorst discretisation of log income
p = (1 + rho) / 2; Pi = [p 1-p; 1-p p];
for n = 3:N
  Z = zeros(n); Pn = p * [Pi zeros(n-1,1); Z(n,1:n-1) 0] + (1-p) * [zeros(n-1,1) Pi; 0 Z(n,1:n-1)] ...
    + (1-p) * [Z(1,1:n-1) 0; Pi zeros(n-1,1)] + p * [0 Z(1,1:n-1); zeros(n-1,1) Pi];
  Pn(2:end-1, :) = Pn(2:end-1, :) / 2; Pi = Pn;
end
sdy = se / sqrt(1 - rho^2); ly = linspace(-sqrt(N-1) * sdy, sqrt(N-1) * sdy, N);
pistat = ones(1, N) / N; for it = 1:2000, pistat = pistat * Pi; end
y = exp(ly); y = y / (pistat * y');                       % mean quarterly income = 1
agrid = linspace(0, 1, 400)'.^2 * 200; na = numel(agrid);
fprintf(' beta    mean assets / qtr income   share at limit   average MPC (qtr)\\n');
for beta = [0.985 0.96]
  c = (r * agrid + y) + 0.1;                              % initial guess, na x N
  for it = 1:5000                                         % endogenous grid method
    Emu = (c.^(-sig)) * Pi';                              % E[u'(c')] at a' on the grid, na x N
    cend = (beta * (1 + r) * Emu).^(-1/sig);
    aend = (cend + agrid - y) / (1 + r);                  % current assets consistent with a' = agrid
    cn = zeros(na, N);
    for j = 1:N
      cn(:, j) = interp1(aend(:, j), cend(:, j), agrid, 'linear', 'extrap');
      lim = agrid < aend(1, j); cn(lim, j) = (1 + r) * agrid(lim) + y(j);   % borrowing constraint binds
    end
    if max(abs(cn(:) - c(:))) < 1e-9, c = cn; break; end
    c = cn;
  end
  ap = (1 + r) * agrid + y - c;                           % savings policy
  % stationary distribution (Young's lottery method)
  D = ones(na, N) / (na * N);
  for it = 1:20000
    Dn = zeros(na, N);
    for j = 1:N
      a1 = min(max(ap(:, j), agrid(1)), agrid(end));
      k = min(max(sum(agrid <= a1', 1)', 1), na - 1); w = (agrid(k+1) - a1) ./ (agrid(k+1) - agrid(k));
      for jj = 1:N
        Dn(:, jj) = Dn(:, jj) + accumarray(k, w .* D(:, j) * Pi(j, jj), [na 1]) ...
                              + accumarray(k + 1, (1 - w) .* D(:, j) * Pi(j, jj), [na 1]);
      end
    end
    if max(abs(Dn(:) - D(:))) < 1e-11, D = Dn; break; end
    D = Dn;
  end
  da = 0.04;                                               % 1% of annual income = 0.04 quarterly incomes
  mpc = zeros(na, N);
  for j = 1:N, mpc(:, j) = (interp1(agrid, c(:, j), agrid + da, 'linear', 'extrap') - c(:, j)) / da; end
  atlim = sum(D(ap <= 1e-8));
  fprintf('%5.3f          %6.2f                  %5.1f%%           %5.3f\\n', beta, sum(sum(D .* agrid)), ...
          100 * atlim, sum(sum(D .* mpc)));
end
""")],
notes="""With a patient household (\\(\\beta=0.985\\)) people build a large buffer, about one in ten is at the borrowing
limit, and the average quarterly MPC is small — close to the representative-agent value. With \\(\\beta=0.96\\) wealth is low, a large
share of households is at or near the constraint, and the average MPC is several times larger. Empirical quarterly
MPCs out of small windfalls are around 0.15–0.25, far above the representative-agent benchmark, which requires many
households with little *liquid* wealth. One-asset models can match this only by making households poor on average,
contradicting the large wealth in the data. Two-asset models resolve the tension: households hold most wealth in
illiquid assets (housing, retirement accounts) with high returns and transaction costs, so many are "wealthy hand-to-
mouth" (Kaplan, Violante and Weidner, 2014). High MPCs make the indirect, income-driven effects of monetary policy
dominate the direct intertemporal-substitution effect (Kaplan, Moll and Violante, 2018).
"""),
]

# --------------------------------------------------------------------------------------------- Lecture 13
NK_POST = """
function [lp, ll] = nk_post(th, Y)
% NK_POST  Log posterior (and log likelihood) of the three-equation NK model with interest-rate smoothing.
%   th = [kappa phi_pi phi_y rho_i rho_r rho_u sd_r sd_u sd_v];  Y = T x 3 data [output gap, inflation, rate].
beta = 0.99; sigma = 1;
kappa = th(1); phi_pi = th(2); phi_y = th(3); rho_i = th(4); rho_r = th(5); rho_u = th(6); sd = th(7:9);
lp = -Inf; ll = -Inf;
if kappa <= 0 || any(th(4:6) <= 0) || any(th(4:6) >= 1) || any(sd <= 0) || any(sd > 5), return; end
% log prior: kappa ~ Gamma(mean 0.1, sd 0.05); phi_pi ~ N(1.5, 0.25); phi_y ~ N(0.125, 0.1);
%            rho ~ Beta(mean 0.5, sd 0.2); shock s.d. ~ Uniform(0, 5)
k = 4; s = 0.025; a = 2.625;
lprior = (k-1)*log(kappa) - kappa/s - gammaln(k) - k*log(s) ...
       - 0.5*((phi_pi - 1.5)/0.25)^2 - 0.5*((phi_y - 0.125)/0.1)^2 ...
       + sum((a-1)*log(th(4:6)) + (a-1)*log(1 - th(4:6))) - 3*(2*gammaln(a) - gammaln(2*a));
% model: y = [x pi i r u];  A y(-1) + B y + C E y(+1) + D e = 0,  e = [e_r e_u e_v]
A = zeros(5); B = zeros(5); C = zeros(5); D = zeros(5, 3);
B(1, [1 2 5]) = [kappa -1 1];               C(1, 2) = beta;                         % Phillips curve
B(2, [1 3 4]) = [-1 -1/sigma 1/sigma];      C(2, [1 2]) = [1 1/sigma];              % IS curve
B(3, [1 2 3]) = [(1-rho_i)*phi_y (1-rho_i)*phi_pi -1]; A(3, 3) = rho_i; D(3, 3) = 1; % policy rule
B(4, 4) = -1; A(4, 4) = rho_r; D(4, 1) = 1;                                         % natural rate
B(5, 5) = -1; A(5, 5) = rho_u; D(5, 2) = 1;                                         % cost-push shock
[P, Q, info] = solve_lre(A, B, C, D);
if info, return; end
H = [eye(3) zeros(3, 2)];
ll = kalman_loglik(Y, P, Q, diag(sd.^2), H, 1e-4 * eye(3));
lp = ll + lprior;
end
"""

NK_DATA = """
T = readtable('us_quarterly.csv');
x  = 100 * (log(T.rgdp) - hp_trend(log(T.rgdp), 1600));          % output gap (%)
pq = [NaN; 100 * diff(log(T.gdpdef))];                           % quarterly inflation (%)
iq = T.ffr / 4;                                                  % quarterly policy rate (%)
k = T.t >= 1984 & T.t < 2008;
Y = [x(k) pq(k) iq(k)]; Y = Y - mean(Y);                          % demeaned, 1984Q1-2007Q4
% parameters in unbounded form z: kappa = e^z1, phi_pi = 1 + e^z2, phi_y = z3, rho = logistic, sd = e^z
lg = @(z) 1 ./ (1 + exp(-z));
th = @(z) [exp(z(1)) 1+exp(z(2)) z(3) lg(z(4)) lg(z(5)) lg(z(6)) exp(z(7)) exp(z(8)) exp(z(9))];
z0 = [log(0.1) log(0.5) 0.125 0 0 0 log(0.5) log(0.2) log(0.2)];
opt = optimset('MaxFunEvals', 20000, 'MaxIter', 20000, 'TolX', 1e-7, 'TolFun', 1e-7, 'Display', 'off');
zhat = z0;
for rep = 1:3, zhat = fminsearch(@(z) -nk_post(th(z), Y), zhat, opt); end
"""

EXERCISES["13-bayesian-estimation.md"] = [
dict(title="The posterior mode of a small New Keynesian model",
data="`us_quarterly.csv` — HP output gap, GDP-deflator inflation and the federal funds rate, 1984Q1–2007Q4 "
     "(96 quarters).",
task="""
Estimate the three-equation New Keynesian model with interest-rate smoothing,
\\(i_t=\\rho_i i_{t-1}+(1-\\rho_i)(\\phi_\\pi\\pi_t+\\phi_y\\tilde y_t)+\\varepsilon^v_t\\), AR(1) natural-rate and cost-push
shocks, and \\(\\beta=0.99\\), \\(\\sigma=1\\) fixed.

1. Write the log posterior as a function (`nk_post.m`): solve the model with `solve_lre`, evaluate the likelihood with
   `kalman_loglik`, and add the log prior (Gamma for \\(\\kappa\\), normal for the policy coefficients, Beta for the
   autocorrelations, uniform for the shock standard deviations).
2. Find the posterior mode with `fminsearch`, reparameterising so that every parameter is unbounded.
3. Compare the mode with the prior means. Which parameters do the data move most?
""",
starter=[("nk_post.m", """
function [lp, ll] = nk_post(th, Y)
% th = [kappa phi_pi phi_y rho_i rho_r rho_u sd_r sd_u sd_v]
% TODO: log prior; matrices A, B, C, D for y = [x pi i r u]; solve_lre; kalman_loglik with H = [eye(3) zeros(3,2)]
end
"""), ("ex13_1.m", """
T = readtable('us_quarterly.csv');
% TODO: build Y (demeaned, 1984Q1-2007Q4); transform parameters; fminsearch on -nk_post
""")],
solution=[("nk_post.m", NK_POST), ("ex13_1.m", NK_DATA + """
names = {'kappa', 'phi_pi', 'phi_y', 'rho_i', 'rho_r', 'rho_u', 'sd_r', 'sd_u', 'sd_v'};
pmean = [0.1 1.5 0.125 0.5 0.5 0.5 2.5 2.5 2.5];
[lp, ll] = nk_post(th(zhat), Y); m = th(zhat);
fprintf('observations: %d; log posterior at mode %.2f, log likelihood %.2f\\n\\n', size(Y, 1), lp, ll);
fprintf('parameter   prior mean   posterior mode\\n');
for j = 1:9, fprintf('%-8s     %7.3f       %8.4f\\n', names{j}, pmean(j), m(j)); end
""")],
notes="""The data move several parameters far from their priors. Interest-rate smoothing is strong (\\(\\rho_i\\approx0.8\\)),
the response to the output gap is much larger than the prior mean, and the natural-rate shock is very persistent
(\\(\\rho_r\\approx0.9\\)), because the model needs a slow-moving force to explain the persistent output gap. The
Phillips-curve slope is pushed far below its prior mean, to about 0.015: inflation barely responded to the output gap
over 1984–2007, the flat Phillips curve of Lecture 4. The long-run inflation response stays close to its prior mean of
1.5, a sign that the data say little about it once the Taylor principle is imposed. In Dynare the same estimation needs
`varobs`, an `estimated_params` block and one command, `estimation(datafile=..., mode_compute=4, mh_replic=0)`;
writing it by hand shows what that command does.
"""),

dict(title="Random-walk Metropolis–Hastings",
data="As in Exercise 13.1.",
task="""
1. Starting at the posterior mode, compute the Hessian of the log posterior (in the unbounded parameters) by finite
   differences and use \\(c^2(-H)^{-1}\\) with \\(c=2.38/\\sqrt{9}\\) as the proposal covariance.
2. Run 20,000 random-walk Metropolis–Hastings draws, discard the first 5,000 and report the acceptance rate, posterior
   means and 90% credible intervals. Remember the Jacobian of the reparameterisation.
3. For which parameters is the posterior much tighter than the prior? What would you check before trusting the
   results (convergence, identification)?
""",
starter=[("ex13_2.m", NK_DATA + """
% TODO: numerical Hessian of f(z) = nk_post(th(z), Y) + log|J(z)| at zhat
% TODO: RWMH: z' = z + chol(Sigma)' * randn(9, 1); accept with probability exp(f(z') - f(z))
""")],
solution=[("nk_post.m", NK_POST), ("ex13_2.m", NK_DATA + """
logJ = @(z) z(1) + z(2) + sum(log(lg(z(4:6)) .* (1 - lg(z(4:6))))) + sum(z(7:9));   % log |d theta / d z|
f = @(z) nk_post(th(z), Y) + logJ(z);
n = 9; h = 1e-3; Hs = zeros(n); E = eye(n) * h; f0 = f(zhat);
for a = 1:n
  for b = a:n
    Hs(a, b) = (f(zhat + E(a,:) + E(b,:)) - f(zhat + E(a,:) - E(b,:)) - f(zhat - E(a,:) + E(b,:)) ...
              + f(zhat - E(a,:) - E(b,:))) / (4 * h^2);
    Hs(b, a) = Hs(a, b);
  end
end
Sig = (2.38 / sqrt(n))^2 * inv(-Hs); Sig = (Sig + Sig') / 2; L = chol(Sig)';
rand('seed', 1); randn('seed', 1);
N = 20000; burn = 5000; Z = zeros(N, n); z = zhat; fz = f(z); acc = 0;
for d = 1:N
  zp = z + (L * randn(n, 1))'; fp = f(zp);
  if log(rand) < fp - fz, z = zp; fz = fp; acc = acc + 1; end
  Z(d, :) = z;
end
TH = zeros(N - burn, n); for d = 1:N - burn, TH(d, :) = th(Z(burn + d, :)); end
names = {'kappa', 'phi_pi', 'phi_y', 'rho_i', 'rho_r', 'rho_u', 'sd_r', 'sd_u', 'sd_v'};
fprintf('acceptance rate: %.1f%%\\n\\n', 100 * acc / N);
fprintf('parameter   posterior mean    5%%      95%%\\n');
for j = 1:n
  q = sort(TH(:, j)); fprintf('%-8s       %7.3f     %7.3f  %7.3f\\n', names{j}, mean(q), q(round(0.05 * end)), q(round(0.95 * end)));
end
""")],
notes="""An acceptance rate of about 25% indicates a well-tuned proposal. Most posteriors are much tighter than their priors:
the smoothing parameter, the output-gap response, the shock persistences and standard deviations, and the Phillips
slope, whose 90% interval (roughly 0.01–0.04) excludes the prior mean of 0.1. The exception is the inflation response,
whose posterior interval is about as wide as the prior: \\(\\phi_\\pi\\) is weakly identified in this model, a
well-known problem when policy has kept inflation stable (Cochrane, 2011). Before trusting results you would run
several chains from dispersed starting points and compare them (Gelman–Rubin \\(\\hat R\\)), check trace plots and
effective sample sizes, compare prior and posterior densities for each parameter to detect weak identification
(Iskrev, 2010), and test robustness to the output-gap measure and the sample. Your exact numbers will differ slightly
because random-number generators differ across MATLAB and Octave versions.
"""),
]

# --------------------------------------------------------------------------------------------- Lecture 14
EXERCISES["14-identifying-monetary-shocks.md"] = [
dict(title="A recursive VAR and the price puzzle",
data="`us_monthly.csv` — industrial production, CPI and the federal funds rate, 1965–2007.",
task="""
Estimate a VAR(12) in \\(100\\log IP_t\\), \\(100\\log CPI_t\\) and the federal funds rate, with a constant, on
1965M1–2007M12. Identify the monetary-policy shock recursively (Cholesky, with the funds rate ordered last).

1. Compute the impulse responses to a 25 bp funds-rate shock for 48 months.
2. Report the responses of output and prices at 6, 12, 24, 36 and 48 months.
3. Do prices initially rise after a tightening? What is the usual explanation, and how does adding commodity prices
   change the result?
""",
starter=[("ex14_1.m", """
D = readtable('us_monthly.csv');
k = D.year >= 1965 & D.year <= 2007;
Y = [100 * log(D.indpro(k)) 100 * log(D.cpi(k)) D.ffr(k)];
p = 12;
% TODO: X = [1, Y(t-1), ..., Y(t-p)]; B = X \\ Y(p+1:end,:); Sigma; chol; companion-form IRFs
""")],
solution=[("ex14_1.m", """
D = readtable('us_monthly.csv');
k = D.year >= 1965 & D.year <= 2007;
Y = [100 * log(D.indpro(k)) 100 * log(D.cpi(k)) D.ffr(k)];
p = 12; [T, n] = size(Y);
X = ones(T - p, 1); for l = 1:p, X = [X Y(p+1-l:T-l, :)]; end
B = X \\ Y(p+1:end, :); U = Y(p+1:end, :) - X * B; Sigma = U' * U / (T - p - size(X, 2));
S = chol(Sigma)';                                   % lower triangular: IP, CPI, FFR
shock = S(:, 3) * 0.25 / S(3, 3);                   % 25 bp funds-rate shock
H = 49; IR = zeros(n, H); Acomp = [B(2:end, :)'; eye(n * (p - 1)) zeros(n * (p - 1), n)];
s = [shock; zeros(n * (p - 1), 1)];
for h = 1:H, IR(:, h) = s(1:n); s = Acomp * s; end
fprintf('months after shock   %s\\n', sprintf('%7d', [0 6 12 24 36 48]));
lab = {'industrial prod.', 'CPI             ', 'funds rate      '};
for v = 1:3, fprintf('%s     %s\\n', lab{v}, sprintf('%7.3f', IR(v, [0 6 12 24 36 48] + 1))); end
""")],
notes="""After a 25 bp tightening the funds rate stays elevated for over a year and industrial production falls gradually,
reaching its trough after one to two years. The CPI, however, rises for a long time instead of falling: the "price
puzzle" (Sims, 1992). The usual interpretation is misidentification. The Fed raises rates when it expects inflation,
using information the VAR does not contain, so the "shock" is partly a response to anticipated inflation. Adding
commodity prices or Greenbook forecasts reduces the puzzle, and narrative or high-frequency identification
(Exercise 14.2) removes most of it. Note also the sensitivity to the sample: the 1970s, with oil shocks and
accommodative policy, drive much of the puzzle.
"""),

dict(title="Local projections with Romer–Romer shocks",
data="`mp_shocks_monthly.csv` (Romer–Romer narrative shocks, updated by Acosta, 1969–2019) and `us_monthly.csv`.",
task="""
Romer and Romer (2004) purge the Fed's intended rate changes of the Greenbook forecasts the Fed reacted to. Estimate
\\[ y_{t+h}-y_{t-1}=a_h+\\beta_h\\,RR_t+\\sum_{j=1}^{12}\\gamma_{h,j}\\Delta y_{t-j}+\\sum_{j=1}^{12}\\delta_{h,j}RR_{t-j}+u_{t+h} \\]
for \\(y\\) = 100 log IP and 100 log CPI, \\(h=0,\\dots,48\\), on 1970–2007, with Newey–West standard errors.

1. Report \\(\\beta_h\\) per 1 percentage point shock at 6, 12, 24, 36 and 48 months.
2. Compare the shapes with the recursive VAR of Exercise 14.1. Is there a price puzzle?
3. Why are local projections less efficient but more robust than a VAR?
""",
starter=[("ex14_2.m", """
D = readtable('us_monthly.csv'); S = readtable('mp_shocks_monthly.csv');
% TODO: align on year/month; build lags of dy and RR; loop over h with ols_nw(lhs, X, h+1)
""")],
solution=[("ex14_2.m", """
D = readtable('us_monthly.csv'); S = readtable('mp_shocks_monthly.csv');
key = @(y, m) 12 * y + m;
[~, ia, ib] = intersect(key(D.year, D.month), key(S.year, S.month));
yr = D.year(ia); rr = S.rr(ib); T = numel(yr);
L = @(v, j) [NaN(j, 1); v(1:end-j)];
hs = [0 6 12 24 36 48]; lab = {'industrial prod.', 'CPI             '};
fprintf('months after shock   %s\\n', sprintf('%7d', hs));
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
  fprintf('%s     %s\\n', lab{v}, sprintf('%7.2f', b));
  fprintf('   (s.e.)            %s\\n', sprintf('%7.2f', se));
end
""")],
notes="""With narrative shocks, a 1 pp tightening is followed by a large fall in industrial production of about 2% after two
years, roughly twice the VAR response once both are scaled to a 1 pp rate change. Prices still rise for one to two years —
a short-lived price puzzle that is statistically significant here — before falling steadily, by about 2.5% after four
years. Romer and Romer (2004) also find that prices respond only after a long delay. Local projections estimate each
horizon by a separate regression, so they do not impose the VAR's dynamic restrictions: they are robust to
misspecification at long horizons but have larger standard errors, as the widening intervals show. Plagborg-Møller and
Wolf (2021) show that VARs and LPs estimate the same impulse responses in population; the choice is a bias–variance
trade-off (Li, Plagborg-Møller and Wolf, 2024).
"""),
]

# --------------------------------------------------------------------------------------------- Lecture 15
EXERCISES["15-central-banking-practice.md"] = [
dict(title="Did inflation targeting lower inflation? A Ball–Sheridan comparison",
data="`cpi_inflation_wdi.csv` — World Bank CPI inflation for about 190 countries, 1960–2024, with inflation-targeting "
     "adoption years (`it_year`).",
task="""
1. For each inflation-targeting country that adopted between 1990 and 2016, compute average inflation in the five
   years before and the five years after adoption. For non-targeters, use the same windows around 1998.
   Drop countries whose pre-period inflation exceeds 40%.
2. Compare the average change in inflation for adopters and non-adopters.
3. Following Ball and Sheridan (2005), regress the change on an IT dummy and pre-period inflation. What happens to the
   IT coefficient once you control for initial inflation, and why?
""",
starter=[("ex15_1.m", """
W = readtable('cpi_inflation_wdi.csv');
codes = unique(W.code);
% TODO: for each country: adoption year (or 1998 for non-IT), pre and post averages; regression with ols_nw(..., 0)
""")],
solution=[("ex15_1.m", """
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
  fprintf('%s n = %3d   before %5.1f%%   after %5.1f%%   change %+5.1f\\n', nm, sum(m), mean(R(m, 2)), mean(R(m, 3)), mean(d(m)));
end
[b1, s1] = ols_nw(d, [ones(size(d)) R(:, 1)], 0);
[b2, s2] = ols_nw(d, [ones(size(d)) R(:, 1) R(:, 2)], 0);
fprintf('\\nchange = %.2f %+.2f IT                       (IT s.e. %.2f)\\n', b1(1), b1(2), s1(2));
fprintf('change = %.2f %+.2f IT %+.2f pre-inflation    (IT s.e. %.2f)\\n', b2(1), b2(2), b2(3), s2(2));
""")],
notes="""Inflation fell by about 5.5 points in adopters and 4.6 points in non-adopters over the same kind of window: adopters
started from slightly higher inflation, and inflation fell almost everywhere in the 1990s. The raw difference of under
one point is small and statistically insignificant. Once the change is regressed on initial inflation, the strong
negative coefficient on pre-period inflation captures regression to the mean and the global disinflation, and the IT
coefficient shrinks further towards zero — Ball and Sheridan's (2005) finding. This does not show that inflation
targeting is useless: adoption is endogenous and often comes at the end of a reform process (central-bank
independence, ending exchange-rate pegs), which is why staggered difference-in-differences and event-study designs
that model the timing of adoption give more credible answers.
"""),

dict(title="Inflation at risk: do monetary frameworks thin the upper tail?",
data="`cpi_inflation_wdi.csv`, 1985–2024; observations with inflation above 50% are excluded.",
task="""
Estimate pooled quantile regressions of next year's inflation on current inflation, an inflation-targeting indicator
and their interaction,
\\[ Q_\\tau(\\pi_{i,t+1}\\mid\\cdot)=a_\\tau+b_\\tau\\pi_{it}+c_\\tau IT_{it}+d_\\tau IT_{it}\\pi_{it}, \\]
for \\(\\tau\\in\\{0.1,0.5,0.9\\}\\), using `qreg`.

1. Report the coefficients.
2. Predict the 10th, 50th and 90th percentiles of next year's inflation when current inflation is 3% and when it is 8%,
   with and without inflation targeting.
3. Is the effect of the framework larger in the upper tail? What are the limits of this exercise?
""",
starter=[("ex15_2.m", """
W = readtable('cpi_inflation_wdi.csv');
% TODO: build pi_t and pi_{t+1} pairs within countries; IT = year >= it_year; qreg at tau = 0.1, 0.5, 0.9
""")],
solution=[("ex15_2.m", """
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
fprintf('observations: %d (country-years), IT share %.0f%%\\n\\n', numel(Y), 100 * mean(X(:, 3)));
taus = [0.1 0.5 0.9]; Bq = zeros(4, 3);
fprintf('quantile    const    pi_t     IT     IT*pi_t\\n');
for j = 1:3
  Bq(:, j) = qreg(Y, X, taus(j));
  fprintf('  %.1f     %6.2f   %6.3f  %6.2f   %6.3f\\n', taus(j), Bq(:, j));
end
fprintf('\\npredicted quantiles of next year''s inflation (%%)\\n');
fprintf('current infl   framework    q10     q50     q90\\n');
for p0 = [3 8]
  for it = [0 1]
    q = [1 p0 it it * p0] * Bq;
    if it, nm = 'IT     '; else, nm = 'non-IT '; end
    fprintf('     %d%%        %s   %6.2f  %6.2f  %6.2f\\n', p0, nm, q);
  end
end
""")],
notes="""Persistence rises across the distribution: the coefficient on current inflation is about 0.3 at the 10th percentile,
0.7 at the median and about 1 at the 90th, so high inflation raises the risk of even higher inflation more than it
raises the median. Under inflation targeting the median is almost unchanged, but the distribution is compressed from
both sides: the 90th percentile is about 2 points lower and the 10th percentile is higher, which reduces both
inflation and deflation risk. The exercise is descriptive: IT countries differ in many ways (income, institutions,
exchange-rate regimes), adoption is endogenous, pooling assumes common coefficients across very different economies,
and the 2021–23 global surge showed that frameworks offer limited protection against large common shocks. Adding
income and central-bank independence as controls, or country fixed effects, is a natural extension.
"""),
]
