INTRO = """
These exercises use **real market data** in Python: monthly Fama–French factor returns (1926–2018), daily S&P 500 and
NASDAQ prices (1999–2018), the VIX, Moody's AAA and BAA corporate bond yields (1919–2018), all bundled with the `arch`
package, plus U.S. Treasury yields and household data from the `wooldridge` package. Install once with
`pip install arch wooldridge statsmodels scipy numpy pandas`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.
"""

FF = """from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns
"""

EXERCISES = {}

EXERCISES["01-returns-risk.md"] = [
dict(title="Ninety years of U.S. stock returns: arithmetic, geometric and compounded",
data="Fama–French monthly market excess return `Mkt-RF` and T-bill rate `RF`, July 1926 – November 2018.",
task="""
1. Build monthly total market returns \\(R_m=\\)`Mkt-RF`+`RF`. Report annualized arithmetic and geometric mean returns for stocks and bills, and the annualized volatility of stocks.
2. Check the approximation \\(\\text{geometric}\\approx\\text{arithmetic}-\\sigma^2/2\\).
3. Compute the value in 2018 of \\$1 invested in July 1926 in stocks and in bills.
4. Report the annualized Sharpe ratio with its standard error \\(\\sqrt{(1+SR^2/2)/T}\\) (Lo, 2002, i.i.d. case).
""",
starter="""
import numpy as np, pandas as pd
""" + FF + """
rm, rf = ff["Mkt-RF"] + ff["RF"], ff["RF"]
# TODO: 12 * mean, (prod(1+r))^(12/T) - 1, sqrt(12) * std, cumulative wealth
""",
solution="""
import numpy as np, pandas as pd
""" + FF + """
rm, rf, ex = ff["Mkt-RF"] + ff["RF"], ff["RF"], ff["Mkt-RF"]; T = len(rm)
for lab, r in (("stocks", rm), ("T-bills", rf)):
    arith, geo, vol = 12 * r.mean(), np.prod(1 + r)**(12 / T) - 1, np.sqrt(12) * r.std()
    print(f"{lab:7s}: arithmetic {100*arith:5.2f}%, geometric {100*geo:5.2f}%, volatility {100*vol:5.2f}%, arith - vol^2/2 = {100*(arith - vol**2/2):5.2f}%, $1 -> ${np.prod(1 + r):,.0f}")
sr_m = ex.mean() / ex.std(); sr = np.sqrt(12) * sr_m; se = np.sqrt(12) * np.sqrt((1 + sr_m**2 / 2) / T)
print(f"annualized equity premium {100*12*ex.mean():.2f}%, Sharpe ratio {sr:.3f} (se {se:.3f})")
print(f"sample: {ff.index[0]} to {ff.index[-1]}, {T} months")
""",
notes="""Over 92 years, U.S. stocks returned about 11.2% a year on average (arithmetic) and 9.9% compounded, against 3.3% for
T-bills. The geometric mean is close to arithmetic minus \\(\\sigma^2/2\\): volatility drag. Compounding turns the gap into
an enormous difference in wealth — \\$1 grew to about \\$6,400 in stocks versus \\$21 in bills. The Sharpe ratio of about 0.43
has a standard error of 0.10, even with 1,109 months: expected returns are hard to estimate precisely, which is a recurring
theme of this course. (Bills look almost riskless in nominal terms here; in real terms they are not.)
"""),

dict(title="Daily S&P 500 returns: fat tails and volatility clustering",
data="Daily S&P 500 adjusted close, 1999–2018 (`arch.data.sp500`).",
task="""
1. Compute daily log returns. Report mean, standard deviation, skewness and excess kurtosis, and run a Jarque–Bera test.
2. Count the days with moves larger than 4 standard deviations, and compare with the number a normal distribution predicts.
3. Compare the autocorrelation of returns with the autocorrelation of **squared** returns (lags 1–5). What does the contrast say about predictability of returns versus predictability of risk?
""",
starter="""
import numpy as np, pandas as pd
from arch.data import sp500
from scipy import stats
p = sp500.load()["Adj Close"]; r = 100 * np.log(p).diff().dropna()
# TODO: stats.skew, stats.kurtosis, stats.jarque_bera; |r - mean| > 4 sd; acf of r and r^2
""",
solution="""
import numpy as np, pandas as pd
from arch.data import sp500
from scipy import stats
from statsmodels.tsa.stattools import acf
p = sp500.load()["Adj Close"]; r = 100 * np.log(p).diff().dropna()
jb = stats.jarque_bera(r)
print(f"daily log returns (%): mean {r.mean():.4f}, sd {r.std():.3f}, skewness {stats.skew(r):.3f}, excess kurtosis {stats.kurtosis(r):.2f}")
print(f"Jarque-Bera = {jb.statistic:,.0f}, p = {jb.pvalue:.1e}")
big = (np.abs(r - r.mean()) > 4 * r.std()).sum()
print(f"days beyond 4 sd: {big} observed vs {len(r) * 2 * stats.norm.sf(4):.2f} expected under normality ({len(r)} days)")
print("autocorrelation of returns,         lags 1-5:", acf(r, nlags=5)[1:].round(3))
print("autocorrelation of squared returns, lags 1-5:", acf(r**2, nlags=5)[1:].round(3))
print("largest daily losses:", r.nsmallest(3).round(2).to_dict())
""",
notes="""Daily returns are far from normal: excess kurtosis is about 8, and there were 32 days with moves beyond four standard
deviations, where a normal distribution predicts 0.3. The three worst days all fell in autumn 2008. Returns themselves have
small (slightly negative) autocorrelations, but **squared** returns are strongly autocorrelated: large moves cluster in time.
Returns are close to unpredictable; risk is highly predictable. This is the motivation for GARCH models (Lectures 13 and 15)
and for fat-tailed risk measures.
"""),
]

EXERCISES["02-expected-utility.md"] = [
dict(title="How much in stocks? Expected-utility portfolio choice with historical returns",
data="Fama–French monthly market and T-bill returns, compounded to annual returns, 1927–2017.",
task="""
A CRRA investor chooses the share \\(w\\) of wealth in stocks (the rest in bills) to maximize \\(E[u(1+r_f+w\\,r^e)]\\), using the empirical distribution of annual returns.

1. Solve numerically for \\(w^*\\) with \\(\\gamma=2,5,10\\), allowing \\(0\\le w\\le2\\).
2. Compare with the Merton/mean–variance approximation \\(w=\\mu_e/(\\gamma\\sigma^2)\\).
3. Compute the certainty-equivalent gain from holding \\(w^*\\) rather than all bills.
""",
starter="""
import numpy as np, pandas as pd
from scipy import optimize
""" + FF + """
a = (1 + ff).groupby(ff.index.year).prod() - 1                    # annual compounded returns
a = a.loc[1927:2017]; rf = a["RF"]; rex = (1 + a["Mkt-RF"] + a["RF"]) / (1 + rf) - 1  # approx
# TODO: maximize mean(u(1 + rf + w * (rm - rf))) over w
""",
solution="""
import numpy as np, pandas as pd
from scipy import optimize
""" + FF + """
m = ff.assign(RM=ff["Mkt-RF"] + ff["RF"])
a = (1 + m[["RM", "RF"]]).groupby(m.index.year).prod() - 1
a = a.loc[1927:2017]; rm, rf = a["RM"].to_numpy(), a["RF"].to_numpy(); re = rm - rf
mu, s2 = re.mean(), re.var()
print(f"annual excess return: mean {100*mu:.2f}%, sd {100*np.sqrt(s2):.2f}%, {len(re)} years")
u = lambda W, g: W**(1 - g) / (1 - g)
for g in (2, 5, 10):
    EU = lambda w: -np.mean(u(1 + rf + w * re, g))
    w = optimize.minimize_scalar(EU, bounds=(0, 2), method="bounded").x
    ce = lambda w: ((1 - g) * np.mean(u(1 + rf + w * re, g)))**(1 / (1 - g))
    print(f"gamma = {g:2d}: optimal share {w:.3f};  mean-variance mu/(g s2) = {mu/(g*s2):.3f};  certainty-equivalent gain vs bills {100*(ce(w) - ce(0)):.2f}% per year")
""",
notes="""With the historical annual premium of about 8.5% and volatility of about 20%, an investor with \\(\\gamma=5\\) would hold about
40% in stocks, and one with \\(\\gamma=2\\) almost everything. The exact expected-utility solution is very close to the
mean–variance rule \\(\\mu/(\\gamma\\sigma^2)\\): with annual horizons, higher moments matter little. The certainty-equivalent
gains from participating are large (1–4% a year), which makes the empirical fact that many households hold no stocks at all
a puzzle ("non-participation"). These calculations treat historical averages as known — Exercise 3.2 shows the cost of that
assumption.
"""),

dict(title="The equity premium puzzle in numbers",
data="Fama–French annual market excess returns and U.S. real per-capita consumption growth (`macrodata`, 1960–2008).",
task="""
With CRRA utility, the consumption CAPM implies \\(E[R^e]\\approx\\gamma\\,\\operatorname{Cov}(\\Delta\\log c,R^e)\\).

1. For 1960–2008, compute the average annual equity premium, the volatility of consumption growth, and the covariance between consumption growth and excess returns.
2. Solve for the implied coefficient of relative risk aversion \\(\\gamma\\).
3. Also compute the Hansen–Jagannathan bound: the volatility of the SDF must be at least \\(SR\\), and with CRRA, \\(\\sigma(m)\\approx\\gamma\\sigma(\\Delta\\log c)\\). What \\(\\gamma\\) does the bound require?
""",
starter="""
import numpy as np, pandas as pd, statsmodels.api as sm
""" + FF + """
mac = sm.datasets.macrodata.load_pandas().data
c = (mac["realcons"] / mac["pop"]).groupby(mac["year"]).mean()
# TODO: annual excess returns 1960-2008; dc = diff(log c); gamma = mean(re) / cov(dc, re)
""",
solution="""
import numpy as np, pandas as pd, statsmodels.api as sm
""" + FF + """
mac = sm.datasets.macrodata.load_pandas().data
c = (mac["realcons"] / mac["pop"]).groupby(mac["year"]).mean()
dc = np.log(c).diff().loc[1960:2008]
m = ff.assign(RM=ff["Mkt-RF"] + ff["RF"])
a = (1 + m[["RM", "RF"]]).groupby(m.index.year).prod() - 1
re = (a["RM"] - a["RF"]).loc[1960:2008]
cov = np.cov(dc, re)[0, 1]; sr = re.mean() / re.std()
print(f"1960-2008: equity premium {100*re.mean():.2f}%, sd of excess return {100*re.std():.1f}%, Sharpe ratio {sr:.3f}")
print(f"consumption growth: mean {100*dc.mean():.2f}%, sd {100*dc.std():.2f}%, correlation with excess returns {np.corrcoef(dc, re)[0,1]:.3f}")
print(f"implied gamma from E[Re] = gamma Cov(dc, Re): {re.mean()/cov:.1f}")
print(f"Hansen-Jagannathan: gamma >= SR / sd(dc) = {sr/dc.std():.1f}")
""",
notes="""Consumption growth is smooth (SD about 1.7%) and only weakly correlated with stock returns (0.13), so the covariance between
the two is tiny. Matching a 5% equity premium then requires a coefficient of relative risk aversion of over 130. Even the
Hansen–Jagannathan bound, which ignores the low correlation, requires \\(\\gamma\\ge17\\). Economists find risk aversion above
about 10 implausible, so this is the equity premium puzzle (Mehra and Prescott, 1985). Habit formation, long-run risks and
rare disasters are the main proposed resolutions.
"""),
]

EXERCISES["03-portfolio-theory.md"] = [
dict(title="The tangency portfolio of the market, size and value factors",
data="Fama–French monthly excess returns `Mkt-RF`, `SMB`, `HML`, 1963–2018.",
task="""
Treat the three factor portfolios as investable excess returns.

1. Estimate the mean vector and covariance matrix on 1963–1990. Compute the tangency (maximum-Sharpe) weights \\(w\\propto\\Sigma^{-1}\\mu\\), normalized to sum to one, and the in-sample Sharpe ratio.
2. Hold these weights fixed and evaluate the portfolio **out of sample** on 1991–2018. Compare its Sharpe ratio with the market's.
3. Why do out-of-sample Sharpe ratios of estimated optimal portfolios usually fall short of in-sample ones?
""",
starter="""
import numpy as np, pandas as pd
""" + FF + """
X = ff[["Mkt-RF", "SMB", "HML"]]
ins, oos = X.loc["1963-07":"1990-12"], X.loc["1991-01":]
# TODO: w = inv(Sigma) mu / sum(...); Sharpe = sqrt(12) * mean / sd
""",
solution="""
import numpy as np, pandas as pd
""" + FF + """
X = ff[["Mkt-RF", "SMB", "HML"]]
ins, oos = X.loc["1963-07":"1990-12"], X.loc["1991-01":]
mu, S = ins.mean().to_numpy(), ins.cov().to_numpy()
w = np.linalg.solve(S, mu); w = w / w.sum()
SR = lambda r: np.sqrt(12) * r.mean() / r.std()
print("tangency weights (Mkt, SMB, HML):", w.round(3))
print(f"in-sample 1963-1990:  tangency Sharpe {SR(ins @ w):.3f}, market Sharpe {SR(ins['Mkt-RF']):.3f}")
print(f"out-of-sample 1991-2018: tangency Sharpe {SR(oos @ w):.3f}, market Sharpe {SR(oos['Mkt-RF']):.3f}")
print("factor Sharpe ratios in each period:")
print(pd.DataFrame({"1963-1990": ins.apply(SR), "1991-2018": oos.apply(SR)}).round(3).to_string())
""",
notes="""In 1963–90, the optimal combination puts most weight on HML and raises the Sharpe ratio from 0.26 (market) to 0.79. But held
fixed for 1991–2018, it earns a Sharpe ratio of 0.56, **below** the market's 0.59: the value premium weakened and the market
did unusually well. This is the classic out-of-sample problem. Estimated optimal portfolios load heavily on the assets with
the best past performance, and past means are poor predictors of future means, so in-sample Sharpe ratios are biased upward
(DeMiguel, Garlappi and Uppal, 2009, find that naive 1/N often wins).
"""),

dict(title="Estimation risk in optimal weights: a bootstrap",
data="Fama–French monthly excess returns, 1963–2018.",
task="""
1. Compute the full-sample tangency weights for `(Mkt-RF, SMB, HML)` and for the global minimum-variance portfolio \\(w\\propto\\Sigma^{-1}\\mathbf 1\\).
2. Bootstrap the monthly returns (1,000 draws, resampling months) and report the 5th–95th percentile range of each weight for both portfolios.
3. Which portfolio's weights are more stable, and why? (Hint: which inputs does each depend on?)
""",
starter="""
import numpy as np, pandas as pd
""" + FF + """
X = ff.loc["1963-07":, ["Mkt-RF", "SMB", "HML"]].to_numpy()
# TODO: tangency and GMV weights; bootstrap rows of X
""",
solution="""
import numpy as np, pandas as pd
""" + FF + """
X = ff.loc["1963-07":, ["Mkt-RF", "SMB", "HML"]].to_numpy(); T = len(X)
def weights(Y):
    S = np.cov(Y, rowvar=False); one = np.ones(3)
    t = np.linalg.solve(S, Y.mean(0)); g = np.linalg.solve(S, one)
    return t / t.sum(), g / g.sum()
wt, wg = weights(X)
rng = np.random.default_rng(0); B = [weights(X[rng.integers(0, T, T)]) for _ in range(1000)]
bt, bg = np.array([b[0] for b in B]), np.array([b[1] for b in B])
for lab, w, b in (("tangency", wt, bt), ("min-variance", wg, bg)):
    lo, hi = np.percentile(b, 5, axis=0), np.percentile(b, 95, axis=0)
    print(f"{lab:12s}: weights {w.round(3)};  90% bootstrap ranges: " + ", ".join(f"[{l:.2f}, {h:.2f}]" for l, h in zip(lo, hi)))
""",
notes="""The tangency weights are uncertain: the 90% range for the market weight is about 0.17 to 0.45, and the SMB weight could be
zero. The minimum-variance weights are much more stable (ranges about four times narrower), because they depend only on the
covariance matrix, which is estimated precisely from monthly data, while tangency weights also depend on mean returns,
which are not. That is why many practitioners use minimum-variance or risk-parity portfolios, or shrink estimated means
toward a model (Black–Litterman).
"""),
]

EXERCISES["04-capm-factors.md"] = [
dict(title="Does the CAPM price size and value? Alphas and the GRS test",
data="Fama–French monthly `SMB`, `HML` and `Mkt-RF`, July 1963 – November 2018.",
task="""
1. Run time-series CAPM regressions \\(R^e_{it}=\\alpha_i+\\beta_i\\,(R_m-R_f)_t+\\varepsilon_{it}\\) for SMB and HML. Report annualized alphas, betas and HAC \\(t\\)-statistics.
2. Test \\(\\alpha_{SMB}=\\alpha_{HML}=0\\) jointly with the Gibbons–Ross–Shanken (GRS) statistic.
3. Interpret: what does a significant HML alpha say about the CAPM?
""",
starter="""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
""" + FF + """
d = ff.loc["1963-07":]
# TODO: OLS of SMB and HML on Mkt-RF; GRS = (T-N-K)/N * a' S^-1 a / (1 + mu' O^-1 mu) ~ F(N, T-N-K)
""",
solution="""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
""" + FF + """
d = ff.loc["1963-07":]; X = sm.add_constant(d["Mkt-RF"]); T, N, K = len(d), 2, 1
alphas, resid = [], []
for y in ("SMB", "HML"):
    r = sm.OLS(d[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    alphas.append(r.params["const"]); resid.append(r.resid)
    print(f"{y}: alpha {1200*r.params['const']:.2f}% per year (HAC t = {r.tvalues['const']:.2f}), beta {r.params['Mkt-RF']:.3f}, R2 {r.rsquared:.3f}")
a = np.array(alphas); Sig = np.cov(np.array(resid)); mu = d["Mkt-RF"].mean(); om = d["Mkt-RF"].var()
grs = (T - N - K) / N * (a @ np.linalg.solve(Sig, a)) / (1 + mu**2 / om)
print(f"GRS = {grs:.2f}, p = {stats.f.sf(grs, N, T - N - K):.4f}")
""",
notes="""SMB's CAPM alpha is small and insignificant in 1963–2018, but HML earns an alpha of about 5% a year (\\(t\\approx3.1\\)) — with a
**negative** beta, so its average return cannot be explained as compensation for market risk. The GRS test rejects that both
alphas are zero. This is the anomaly behind the Fama–French three-factor model: the CAPM fails to price value stocks.
Whether HML is a risk factor (distress risk) or mispricing (overreaction) remains debated (Lecture 14).
"""),

dict(title="Has the value premium disappeared?",
data="Fama–French monthly `HML` and `Mkt-RF`, 1926–2018.",
task="""
1. Compute HML's average return and CAPM alpha (annualized) in four periods: 1926–1962, 1963–1990, 1991–2006, 2007–2018.
2. Compute a 10-year rolling CAPM beta of HML.
3. Is there statistical evidence that the value premium after 2007 differs from its pre-2007 average? Test with a dummy-variable regression and HAC standard errors.
""",
starter="""
import numpy as np, pandas as pd, statsmodels.api as sm
""" + FF + """
# TODO: loop over periods; rolling cov/var for beta; regression HML ~ post2007
""",
solution="""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
""" + FF + """
d = ff.rename(columns={"Mkt-RF": "mkt"}).copy()
for lo, hi in (("1926-07", "1962-12"), ("1963-01", "1990-12"), ("1991-01", "2006-12"), ("2007-01", "2018-11")):
    s = d.loc[lo:hi]; r = sm.OLS(s["HML"], sm.add_constant(s["mkt"])).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    print(f"{lo[:4]}-{hi[:4]}: mean HML {1200*s['HML'].mean():5.2f}%/yr, CAPM alpha {1200*r.params['const']:5.2f}% (t = {r.tvalues['const']:.2f}), beta {r.params['mkt']:.2f}")
beta = d["HML"].rolling(120).cov(d["mkt"]) / d["mkt"].rolling(120).var()
print(f"rolling 10-year beta of HML: min {beta.min():.2f} ({beta.idxmin()}), max {beta.max():.2f} ({beta.idxmax()}), latest {beta.iloc[-1]:.2f}")
d["post"] = (d.index >= pd.Period("2007-01", "M")).astype(int)
t = smf.ols("HML ~ post", data=d.loc["1963-07":]).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
print(f"1963-2018: HML = {1200*t.params['Intercept']:.2f}% + {1200*t.params['post']:.2f}% x post-2007  (t = {t.tvalues['post']:.2f}, p = {t.pvalues['post']:.3f})")
""",
notes="""The value premium was strong from 1963 to 2006, with large CAPM alphas, but **negative** from 2007 to 2018 (about \\(-2.5\\%\\) a
year). A dummy regression finds the post-2007 drop statistically significant (\\(p\\approx0.01\\)). HML's market beta also
changes sign over time, so its CAPM alpha and its raw return can tell different stories. Possible explanations include
arbitrage after publication (McLean and Pontiff, 2016), the rise of intangible capital that book values miss, and bad luck.
Twelve years of negative returns are still consistent with a positive long-run premium at these volatilities.
"""),
]

EXERCISES["05-arbitrage-sdf.md"] = [
dict(title="A linear stochastic discount factor that prices the factors",
data="Fama–French monthly excess returns `Mkt-RF`, `SMB`, `HML`, 1963–2018.",
task="""
Look for an SDF of the form \\(m_t=1-b'(f_t-E f)\\) such that \\(E[m_tR^e_t]=0\\) for the three factor returns.

1. Show that the moment conditions give \\(b=\\Sigma_f^{-1}\\mu\\), and compute \\(b\\).
2. Verify that the pricing errors \\(E[mR^e]\\) are zero in sample.
3. Compute \\(\\sigma(m)\\) and compare it with the maximum Sharpe ratio of the three factors (the Hansen–Jagannathan bound). Why are they equal?
4. Use \\(b\\) estimated on 1963–1990 to price the 1991–2018 returns. Report the out-of-sample pricing errors (annualized).
""",
starter="""
import numpy as np, pandas as pd
""" + FF + """
F = ff.loc["1963-07":, ["Mkt-RF", "SMB", "HML"]].to_numpy()
# TODO: b = solve(cov(F), mean(F)); m = 1 - (F - mean) @ b; check mean(m[:,None] * F)
""",
solution="""
import numpy as np, pandas as pd
""" + FF + """
X = ff.loc["1963-07":, ["Mkt-RF", "SMB", "HML"]]
def sdf(Y):
    mu, S = Y.mean(0), np.cov(Y, rowvar=False, ddof=0); return np.linalg.solve(S, mu), mu
F = X.to_numpy(); b, mu = sdf(F); m = 1 - (F - mu) @ b
print("b =", b.round(3))
print("in-sample pricing errors E[m Re] (per month):", (m[:, None] * F).mean(0).round(8))
maxSR = np.sqrt(mu @ np.linalg.solve(np.cov(F, rowvar=False, ddof=0), mu))
print(f"sd(m) = {m.std():.4f} per month;  max Sharpe ratio = {maxSR:.4f} per month ({np.sqrt(12)*maxSR:.3f} annualized)")
ins, oos = X.loc[:"1990-12"].to_numpy(), X.loc["1991-01":].to_numpy()
b1, mu1 = sdf(ins); m_oos = 1 - (oos - mu1) @ b1
print("out-of-sample pricing errors, 1991-2018 (% per year):", (1200 * (m_oos[:, None] * oos).mean(0)).round(2))
""",
notes="""The SDF coefficients are the (unnormalized) mean–variance weights, \\(b=\\Sigma^{-1}\\mu\\), and they price all three factors
exactly in sample. The standard deviation of \\(m\\) equals the maximum Sharpe ratio of the factors (0.20 per month, 0.70 a
year): the SDF built from the tangency portfolio is the minimum-variance SDF, so it attains the Hansen–Jagannathan bound. Out
of sample, the SDF estimated on 1963–90 misprices the 1991–2018 returns by several percent a year — it overprices value (HML
earned less than the SDF required). The same overfitting problem as in Exercise 3.1, in SDF language.
"""),

dict(title="Arbitrage bounds in the term structure of T-bills",
data="`intqrt` — U.S. quarterly yields on 3-, 6- and 12-month Treasury bills, 1960s–1990s.",
task="""
With no arbitrage, the price of a 6-month bill must equal the price of rolling two 3-month bills **in expectation, under the risk-neutral measure**, and forward rates are pinned down by spot rates.

1. Compute the implied 3-month forward rate three months ahead from the 3- and 6-month yields, \\(f_t=2y^{(6)}_t-y^{(3)}_t\\) (annualized, simple approximation).
2. Compare the forward rate with the 3-month yield actually realized three months later. Compute the average forecast error (the term premium) and its HAC standard error.
3. Regress the realized change in the 3-month rate on the forward–spot spread (Fama, 1984). Under the pure expectations hypothesis, the slope is 1. What do you find?
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("intqrt")
d["fwd"] = 2 * d["r6"] - d["r3"]; d["r3_next"] = d["r3"].shift(-1)
# TODO: error = r3_next - fwd; regression (r3_next - r3) ~ (fwd - r3)
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("intqrt")
d["fwd"] = 2 * d["r6"] - d["r3"]; d["r3_next"] = d["r3"].shift(-1)
d["err"] = d["r3_next"] - d["fwd"]; d["dr"] = d["r3_next"] - d["r3"]; d["spread"] = d["fwd"] - d["r3"]
s = d.dropna(subset=["err"])
e = smf.ols("err ~ 1", data=s).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
print(f"average forward-rate forecast error {e.params.iloc[0]:.3f} pp (HAC se {e.bse.iloc[0]:.3f}) -> term premium ~ {-e.params.iloc[0]:.3f} pp")
r = smf.ols("dr ~ spread", data=s).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
print(f"Fama regression: change in 3m rate = {r.params['Intercept']:.3f} + {r.params['spread']:.3f} x (forward - spot)  (HAC se {r.bse['spread']:.3f})")
t = r.t_test("spread = 1")
print(f"test slope = 1: t = {float(np.squeeze(t.tvalue)):.2f}, p = {float(t.pvalue):.4f};  R2 = {r.rsquared:.3f};  n = {int(r.nobs)}")
""",
notes="""Forward rates systematically **overpredict** future short rates by about 0.6 percentage points: the gap is a term premium.
The Fama regression slope is about \\(-0.1\\), far below the value of 1 implied by the pure expectations hypothesis, and the
\\(R^2\\) is close to zero: forward–spot spreads contain almost no information about future rate changes at this horizon.
Arbitrage pins forward rates down by spot rates, but it does not make them unbiased forecasts — that requires an assumption
about risk premia.
"""),
]

EXERCISES["06-fixed-income.md"] = [
dict(title="Corporate credit spreads and implied default probabilities",
data="Moody's AAA and BAA corporate bond yields, monthly, 1919–2018 (`arch.data.default`).",
task="""
1. Compute the BAA–AAA spread. Report its mean, minimum and maximum, and the dates of the three highest monthly values.
2. Under risk neutrality, a bond with annual default probability \\(p\\) and recovery rate \\(R\\) must offer a spread that compensates for expected losses: approximately \\(\\text{spread}\\approx p(1-R)\\). With \\(R=40\\%\\), compute the implied annual default probability at the median spread and at the 2008–09 peak.
3. Historical BAA default rates are about 0.2–0.5% per year. What does the gap tell you about the **credit risk premium** (the "credit spread puzzle")?
""",
starter="""
import numpy as np, pandas as pd
from arch.data import default
d = default.load(); d["spread"] = d["BAA"] - d["AAA"]
# TODO: describe; nlargest; p = spread / (1 - R)
""",
solution="""
import numpy as np, pandas as pd
from arch.data import default
d = default.load(); d["spread"] = d["BAA"] - d["AAA"]
s = d["spread"]
print(f"BAA-AAA spread (pp): mean {s.mean():.2f}, median {s.median():.2f}, min {s.min():.2f}, max {s.max():.2f}")
print("three highest months:", {k.strftime('%Y-%m'): round(v, 2) for k, v in s.nlargest(3).items()})
print("peaks in later crises:", {k: round(s.loc[k].max(), 2) for k in ("1982", "2001", "2009")})
R = 0.4
for lab, v in (("median", s.median()), ("2008-09 peak", s.loc["2008":"2009"].max())):
    print(f"implied risk-neutral default probability at the {lab} spread: {v/(1-R):.2f}% per year")
""",
notes="""The BAA–AAA spread averaged about 1.2 points, with peaks in the Great Depression (above 5.5 points in 1932), 1982 and the
2008–09 crisis (about 3.1 points). With 40% recovery, the median spread implies a risk-neutral default probability of about
1.6% a year, and the 2008–09 peak about 5.6%. Actual BAA default rates average only about 0.2–0.5% a year, so most of the
spread is compensation for risk and illiquidity rather than expected losses — the "credit spread puzzle" (Huang and Huang,
2012). Risk-neutral probabilities are not physical probabilities.
"""),

dict(title="Testing the expectations hypothesis with T-bill holding yields",
data="`intqrt` — quarterly holding yield on 6-month bills held for 3 months (`hy6`), 3-month yield (`r3`).",
task="""
As in a classic example in Wooldridge's *Introductory Econometrics*: if the expectations hypothesis holds, the excess holding return of a
6-month bill over three months, \\(hy6_t-r3_{t-1}\\), should be unpredictable from information at \\(t-1\\).

1. Regress \\(hy6_t\\) on \\(hy3_{t-1}\\) and the lagged spread \\((r6-r3)_{t-1}\\), as in Wooldridge's example.
2. Test whether the coefficient on \\(hy3_{t-1}\\) is 1 and the spread coefficient is 0.
3. What does a significant spread coefficient imply about time-varying term premia?
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("intqrt")
# TODO: hy6 ~ hy3_1 + spr63_1 (both are in the dataset)
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("intqrt")
r = smf.ols("hy6 ~ hy3_1 + spr63_1", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
print(r.params.round(4).to_string()); print("HAC se:", r.bse.round(4).to_dict())
w = r.wald_test("hy3_1 = 1, spr63_1 = 0", scalar=True)
print(f"joint test (hy3_1 = 1, spread = 0): chi2(2) = {w.statistic:.2f}, p = {w.pvalue:.4f};  n = {int(r.nobs)}")
print(f"average excess holding return hy6 - hy3_1: {(d['hy6'] - d['hy3_1']).mean():.3f} pp per year")
""",
notes="""The coefficient on the lagged 3-month holding yield is close to one, but the lagged spread has a coefficient of about 0.48
(\\(t\\approx4.5\\)), and the joint test rejects the expectations hypothesis decisively. When the yield curve is steep, the
6-month bill earns a higher excess return over the next quarter: the term premium is **time-varying** and predictable
from the slope of the curve. This is the short-maturity version of Campbell and Shiller's (1991) and Cochrane and
Piazzesi's (2005) findings for longer bonds.
"""),
]

EXERCISES["07-duration-term-structure.md"] = [
dict(title="Duration and convexity against historical yield changes",
data="Moody's AAA corporate bond yield, monthly, 1919–2018.",
task="""
Take a 20-year bond with a 6% annual coupon, priced at the AAA yield.

1. Compute its price, modified duration and convexity at the December 2018 AAA yield.
2. For every historical **monthly** change in the AAA yield, compute the exact price change and the duration and duration-plus-convexity approximations.
3. Report the mean absolute approximation error of each, and the error for the largest yield increase and decrease in the sample.
""",
starter="""
import numpy as np, pandas as pd
from arch.data import default
y = default.load()["AAA"] / 100; dy = y.diff().dropna()
t = np.arange(1, 21); cf = np.r_[np.full(19, 6.0), 106.0]
P = lambda r: np.sum(cf / (1 + r)**t)
# TODO: D = -P'(y)/P, C = P''(y)/P; dP/P ~ -D dy + 0.5 C dy^2
""",
solution="""
import numpy as np, pandas as pd
from arch.data import default
y = default.load()["AAA"] / 100; dy = y.diff().dropna()
t = np.arange(1, 21); cf = np.r_[np.full(19, 6.0), 106.0]
P = lambda r: np.sum(cf / (1 + r)**t)
y0 = y.iloc[-1]; p0 = P(y0)
D = np.sum(t * cf / (1 + y0)**(t + 1)) / p0; C = np.sum(t * (t + 1) * cf / (1 + y0)**(t + 2)) / p0
print(f"AAA yield Dec 2018 {100*y0:.2f}%: price {p0:.2f}, modified duration {D:.2f}, convexity {C:.1f}")
exact = np.array([P(y0 + h) / p0 - 1 for h in dy]); d1 = -D * dy.to_numpy(); d2 = d1 + 0.5 * C * dy.to_numpy()**2
print(f"mean |error|: duration {100*np.mean(np.abs(exact - d1)):.4f} pp, duration + convexity {100*np.mean(np.abs(exact - d2)):.5f} pp")
for lab, k in (("largest increase", dy.idxmax()), ("largest decrease", dy.idxmin())):
    i = dy.index.get_loc(k)
    print(f"{lab} ({k.strftime('%Y-%m')}, {100*dy.iloc[i]:+.2f} pp): exact {100*exact[i]:+.3f}%, duration {100*d1[i]:+.3f}%, + convexity {100*d2[i]:+.3f}%")
""",
notes="""At a 4% yield, the 20-year 6% bond has a modified duration of about 12.6 and convexity of about 214. Over all monthly yield
changes since 1919, the duration approximation misses by 0.03 percentage points on average, and adding convexity cuts the
error by a factor of about 30. For the largest moves (around ±1.2–1.3 points in 1980–81), duration alone is off by about
1.6 points of price, always **overstating** losses and **understating** gains; with convexity, the error falls to about 0.1
points. Convexity is valuable to bondholders for exactly this reason.
"""),

dict(title="Immunizing a liability with two bonds",
data="Moody's AAA yields (for the level of rates and historical yield shocks).",
task="""
A pension fund owes \\$1 million in 10 years. It can buy a 2-year zero-coupon bond and a 30-year zero-coupon bond, priced at a flat yield equal to the December 2018 AAA yield.

1. Choose holdings so that the portfolio's present value and duration match the liability's (two equations, two unknowns).
2. Shock the yield curve with every historical 12-month change in the AAA yield (parallel shifts). Report the distribution of the surplus (assets minus liability, in present value) for the immunized portfolio and for an unhedged portfolio holding only the 2-year bond.
3. Why does the immunized surplus stay **positive** for both large rises and large falls? (Hint: compare convexities.)
""",
starter="""
import numpy as np
from arch.data import default
y = default.load()["AAA"] / 100; y0 = y.iloc[-1]; shocks = (y - y.shift(12)).dropna().to_numpy()
pv = lambda T, r: (1 + r)**-T
# TODO: solve [pv2, pv30; 2 pv2, 30 pv30] [x2, x30] = [L pv10, 10 L pv10]
""",
solution="""
import numpy as np
from arch.data import default
y = default.load()["AAA"] / 100; y0 = y.iloc[-1]; shocks = (y - y.shift(12)).dropna().to_numpy()
pv = lambda T, r: (1 + r)**-T; L = 1e6
A = np.array([[pv(2, y0), pv(30, y0)], [2 * pv(2, y0), 30 * pv(30, y0)]])
x2, x30 = np.linalg.solve(A, [L * pv(10, y0), 10 * L * pv(10, y0)])
print(f"yield {100*y0:.2f}%: buy face ${x2:,.0f} of the 2-year zero and ${x30:,.0f} of the 30-year zero (liability PV ${L*pv(10, y0):,.0f})")
x2_only = L * pv(10, y0) / pv(2, y0)
for lab, f in (("immunized", lambda r: x2 * pv(2, r) + x30 * pv(30, r)), ("2-year only", lambda r: x2_only * pv(2, r))):
    surplus = np.array([f(y0 + s) - L * pv(10, y0 + s) for s in shocks])
    print(f"{lab:11s}: surplus over {len(shocks)} historical 12-month shocks: min ${surplus.min():,.0f}, median ${np.median(surplus):,.0f}, max ${surplus.max():,.0f}")
""",
notes="""Matching present value and duration with a 2-year and a 30-year zero ("barbell") immunizes the liability: across all 1,188
historical 12-month parallel shocks, the surplus never falls below zero, and it rises for large moves in either direction.
The barbell has more convexity than the 10-year liability, so it gains more than the liability when rates fall and loses
less when they rise. The unhedged 2-year position can lose over \\$230,000. Real immunization must also handle
non-parallel shifts (slope and curvature), which a barbell is exposed to.
"""),
]

EXERCISES["08-derivatives.md"] = [
dict(title="Black–Scholes with the VIX, and the variance risk premium",
data="Daily S&P 500 (1999–2018) and VIX (2014–2018), from `arch.data`.",
task="""
1. Price a one-month at-the-money call and put on the S&P 500 on the last trading day of 2018, using the VIX as the implied volatility and a 2.4% risk-free rate. Verify put–call parity.
2. Compute delta, gamma and vega analytically and check delta by finite differences.
3. For 2014–2018, compare the VIX with the **realized** volatility of the S&P 500 over the following 21 trading days. What is the average gap, and what does it measure?
""",
starter="""
import numpy as np, pandas as pd
from scipy.stats import norm
from arch.data import sp500, vix
S = sp500.load()["Adj Close"]; V = vix.load()["vix"]
def bs(S, K, T, r, sig, call=True):
    d1 = (np.log(S / K) + (r + sig**2 / 2) * T) / (sig * np.sqrt(T)); d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) if call else K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
# TODO: price, parity, greeks; realized vol = sqrt(252) * rolling std of future 21 daily log returns
""",
solution="""
import numpy as np, pandas as pd
from scipy.stats import norm
from arch.data import sp500, vix
S = sp500.load()["Adj Close"]; V = vix.load()["vix"] / 100
def bs(S, K, T, r, sig, call=True):
    d1 = (np.log(S / K) + (r + sig**2 / 2) * T) / (sig * np.sqrt(T)); d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) if call else K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
s0, sig, T, r = S.iloc[-1], V.loc[:"2018-12-31"].iloc[-1], 21 / 252, 0.024; K = s0
c, p = bs(s0, K, T, r, sig), bs(s0, K, T, r, sig, call=False)
print(f"S = {s0:.2f}, VIX = {100*sig:.1f}%: call {c:.2f}, put {p:.2f};  C - P = {c - p:.4f} vs S - K e^(-rT) = {s0 - K*np.exp(-r*T):.4f}")
d1 = (np.log(s0 / K) + (r + sig**2 / 2) * T) / (sig * np.sqrt(T))
delta, gamma, vega = norm.cdf(d1), norm.pdf(d1) / (s0 * sig * np.sqrt(T)), s0 * norm.pdf(d1) * np.sqrt(T)
h = 0.01; fd = (bs(s0 + h, K, T, r, sig) - bs(s0 - h, K, T, r, sig)) / (2 * h)
print(f"delta {delta:.4f} (finite difference {fd:.4f}), gamma {gamma:.5f}, vega {vega:.2f} per unit vol")
lr = np.log(S).diff()
rv = lr[::-1].rolling(21).std()[::-1].shift(-1) * np.sqrt(252)        # realized vol over the next 21 days
df = pd.concat([V.rename("vix"), rv.rename("rv")], axis=1).dropna().loc["2014":"2018"]
print(f"2014-2018: mean VIX {100*df.vix.mean():.1f}%, mean subsequent realized vol {100*df.rv.mean():.1f}%, VIX above realized on {np.mean(df.vix > df.rv):.0%} of days")
print(f"average variance risk premium (VIX^2 - RV^2): {1e4*np.mean(df.vix**2 - df.rv**2):.1f} (squared percentage points)")
""",
notes="""At the end of 2018, with the VIX at 25.4%, the one-month at-the-money call is worth about 76 index points and the put about
71; put–call parity holds to four decimals, because it is a no-arbitrage relation independent of the model. The analytic
delta matches the finite difference. Over 2014–2018, the VIX exceeded the volatility realized over the following month on
80% of days, by about 3 points on average: option buyers pay a **variance risk premium** to insure against volatility
spikes. Option sellers earn it most of the time — and lose heavily in crises.
"""),

dict(title="Binomial trees: convergence to Black–Scholes and the value of early exercise",
data="Inputs from the previous exercise: the S&P 500 level and VIX on 31 December 2018, one-month and one-year maturities.",
task="""
1. Implement a Cox–Ross–Rubinstein binomial tree. Price the European call with 10, 50, 200 and 1,000 steps and show convergence to Black–Scholes.
2. Price a one-year **American put** at the money and 10% in the money, and compute the early-exercise premium over the European put.
3. Why is early exercise of an American call on a non-dividend-paying stock never optimal, while it can be optimal for a put?
""",
starter="""
import numpy as np
from scipy.stats import norm
def crr(S, K, T, r, sig, n, call=True, american=False):
    dt = T / n; u = np.exp(sig * np.sqrt(dt)); d = 1 / u; q = (np.exp(r * dt) - d) / (u - d)
    # TODO: terminal payoffs, backward induction with optional early exercise
    ...
""",
solution="""
import numpy as np
from scipy.stats import norm
from arch.data import sp500, vix
s0 = float(sp500.load()["Adj Close"].iloc[-1]); sig = float(vix.load()["vix"].loc[:"2018-12-31"].iloc[-1]) / 100; r = 0.024
def crr(S, K, T, r, sig, n, call=True, american=False):
    dt = T / n; u = np.exp(sig * np.sqrt(dt)); d = 1 / u; q = (np.exp(r * dt) - d) / (u - d); disc = np.exp(-r * dt)
    ST = S * u**np.arange(n, -1, -1) * d**np.arange(0, n + 1)
    V = np.maximum(ST - K, 0) if call else np.maximum(K - ST, 0)
    for k in range(n - 1, -1, -1):
        ST = S * u**np.arange(k, -1, -1) * d**np.arange(0, k + 1)
        V = disc * (q * V[:-1] + (1 - q) * V[1:])
        if american: V = np.maximum(V, (ST - K) if call else (K - ST))
    return V[0]
def bs(S, K, T, r, sig, call=True):
    d1 = (np.log(S / K) + (r + sig**2 / 2) * T) / (sig * np.sqrt(T)); d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) if call else K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
T = 21 / 252
print(f"Black-Scholes one-month ATM call: {bs(s0, s0, T, r, sig):.4f}")
for n in (10, 50, 200, 1000):
    print(f"  CRR with {n:4d} steps: {crr(s0, s0, T, r, sig, n):.4f}")
for lab, K in (("at the money", s0), ("10% in the money", 1.1 * s0)):
    eu, am = crr(s0, K, 1.0, r, sig, 1000, call=False), crr(s0, K, 1.0, r, sig, 1000, call=False, american=True)
    print(f"one-year put {lab}: European {eu:.2f}, American {am:.2f}, early-exercise premium {am - eu:.2f} ({(am-eu)/eu:.2%})")
print(f"one-year ATM call: European {crr(s0, s0, 1, r, sig, 1000):.2f}, American {crr(s0, s0, 1, r, sig, 1000, american=True):.2f}")
""",
notes="""The CRR price converges to Black–Scholes as the number of steps grows, with the oscillating, first-order convergence
typical of binomial trees. Early exercise is worth about 2.4% of the European price for an at-the-money one-year put and
about 3% when the put is 10% in the money: when the put is deep in the money, receiving the strike now and earning
interest beats waiting. For a call on a non-dividend-paying asset, exercising early forfeits the time value and the interest
on the strike, so the American and European calls have the same price — as the output confirms.
"""),
]

EXERCISES["09-capital-budgeting.md"] = [
dict(title="A CAPM cost of capital and NPV sensitivity",
data="Fama–French market excess returns (for the market risk premium) and the AAA yield (for the cost of debt), December 2018.",
task="""
A firm with equity beta 1.2, a target debt-to-value ratio of 30% and a 21% tax rate considers a project costing \\$100 million that generates \\$9 million in year 1, growing 3% a year for 20 years, with no salvage value.

1. Estimate the market risk premium as the 1926–2018 average excess return, and alternatively as the 1963–2018 average. Use the December 2018 T-bill rate (annualized) as \\(r_f\\) and the AAA yield as the pre-tax cost of debt.
2. Compute the cost of equity (CAPM), the WACC and the project NPV for each premium estimate.
3. Plot (tabulate) NPV against the discount rate and find the IRR. How much does the choice of sample for the risk premium matter?
""",
starter="""
import numpy as np, pandas as pd
from scipy import optimize
from arch.data import default
""" + FF + """
# TODO: MRP = 12 * mean(Mkt-RF); re = rf + 1.2 MRP; wacc = 0.7 re + 0.3 rd (1 - 0.21); NPV with growing annuity
""",
solution="""
import numpy as np, pandas as pd
from scipy import optimize
from arch.data import default
""" + FF + """
rf = 12 * ff["RF"].iloc[-1]; rd = default.load()["AAA"].iloc[-1] / 100
cf = 9 * 1.03**np.arange(20); t = np.arange(1, 21)
npv = lambda r: -100 + np.sum(cf / (1 + r)**t)
for lab, lo in (("1926-2018", "1926-07"), ("1963-2018", "1963-07")):
    mrp = 12 * ff.loc[lo:, "Mkt-RF"].mean(); re = rf + 1.2 * mrp; w = 0.7 * re + 0.3 * rd * (1 - 0.21)
    print(f"MRP {lab}: {100*mrp:.2f}% -> cost of equity {100*re:.2f}%, WACC {100*w:.2f}%, NPV ${npv(w):.1f}m")
irr = optimize.brentq(npv, -0.5, 1.0)
print(f"rf = {100*rf:.2f}%, cost of debt {100*rd:.2f}%;  project IRR = {100*irr:.2f}%")
for r in (0.06, 0.08, 0.10, 0.12):
    print(f"  NPV at {100*r:.0f}%: ${npv(r):6.1f}m")
""",
notes="""The project's IRR (about 9.2%) is very close to the WACC, so the investment decision hinges on the market risk premium.
With the 1926–2018 premium (7.9%), the WACC is 9.1% and the NPV barely positive; with the 1963–2018 premium (6.3%), the WACC
is 7.8% and the NPV is about \\$12 million. The NPV falls by roughly \\$8 million for each percentage point added to the
discount rate. A choice of sample that looks technical changes the decision. Sensitivity analysis on the cost of capital
should be standard practice.
"""),

dict(title="Monte Carlo NPV with cash flows that move with the economy",
data="`macrodata` — U.S. annual real GDP growth (mean and volatility used to simulate cash-flow growth).",
task="""
Suppose the project's cash-flow growth each year equals 1.5 times U.S. real GDP growth minus 1.5 percentage points (a cyclical project), with GDP growth drawn from a normal distribution with the historical mean and standard deviation.

1. Simulate 20,000 cash-flow paths for the project of the previous exercise (initial cash flow \\$9m) and compute the NPV distribution at a 9% discount rate.
2. Report the mean NPV, the probability that NPV is negative and the 5th percentile.
3. Compare the mean NPV with the NPV of the **mean** cash flows. Why are they nearly the same here, and when would they differ?
""",
starter="""
import numpy as np, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
g = np.log(m.groupby("year")["realgdp"].mean()).diff().dropna()
mu, sd = g.mean(), g.std()
# TODO: simulate growth = 1.5*gdp - 0.015; cash flows cumulative product; NPV distribution
""",
solution="""
import numpy as np, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
g = np.log(m.groupby("year")["realgdp"].mean()).diff().dropna().loc[:2008]
mu, sd = g.mean(), g.std()
print(f"U.S. real GDP growth 1960-2008: mean {100*mu:.2f}%, sd {100*sd:.2f}%")
rng = np.random.default_rng(42); n, T, r = 20000, 20, 0.09
growth = 1.5 * rng.normal(mu, sd, (n, T - 1)) - 0.015
cf = 9 * np.hstack([np.ones((n, 1)), np.cumprod(1 + growth, axis=1)])
npv = -100 + cf @ (1 + r)**-np.arange(1, T + 1)
print(f"mean NPV ${npv.mean():.1f}m, sd ${npv.std():.1f}m, P(NPV < 0) = {np.mean(npv < 0):.3f}, 5th percentile ${np.percentile(npv, 5):.1f}m")
mean_cf = 9 * np.r_[1, np.cumprod(np.full(T - 1, 1 + 1.5 * mu - 0.015))]
print(f"NPV of the mean cash-flow path: ${-100 + mean_cf @ (1 + r)**-np.arange(1, T + 1):.1f}m")
""",
notes="""With growth uncertainty calibrated to U.S. GDP, the NPV has a mean of about \\$4 million but a standard deviation of \\$6
million: there is about a 27% chance the project destroys value, and a 5% chance of losing \\$6 million or more. The mean NPV
equals the NPV of the mean cash-flow path here, because cash flows are close to linear in the growth shocks over this range
and the discount rate is fixed. They would differ with real options (the ability to abandon or expand), which make NPV a
**convex** function of cash flows — and with a discount rate that should itself reflect the project's cyclicality.
"""),
]

EXERCISES["10-capital-structure.md"] = [
dict(title="Leverage, risk and return across firms",
data="`return` — 142 U.S. firms: stock return 1990–1994 `return` (%), return on equity `roe`, debt-to-capital ratio `dkr`, earnings per share `eps`.",
task="""
Modigliani–Miller (Proposition II) says that the expected return on equity rises with leverage, \\(r_E=r_A+(r_A-r_D)D/E\\).

1. Describe leverage across firms. Regress the 1990–94 stock return on `dkr` and on `roe`, with robust standard errors.
2. Is there evidence that higher-leverage firms earned higher returns? How much variation do the regressors explain?
3. Discuss why a single five-year return per firm is a very noisy measure of **expected** returns, and what MM II would predict about the **riskiness** of returns.
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("return")
# TODO: describe dkr; return ~ dkr + roe
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("return")
print(d[["return", "dkr", "roe", "eps"]].describe().round(2).loc[["mean", "std", "min", "50%", "max"]].to_string())
r = smf.ols("Q('return') ~ dkr + roe", data=d).fit(cov_type="HC1")
print(r.params.round(3).to_dict(), "robust t:", r.tvalues.round(2).to_dict(), f"R2 = {r.rsquared:.3f}, n = {int(r.nobs)}")
d["lev_group"] = np.where(d["dkr"] > d["dkr"].median(), "high leverage", "low leverage")
print(d.groupby("lev_group")["return"].agg(["mean", "std", "size"]).round(2).to_string())
""",
notes="""Leverage varies widely across firms (debt-to-capital from 0 to 80%). The regression finds no significant relation between
leverage and 1990–94 returns, and the regressors explain about 3% of the variation. Higher-leverage firms did earn more on
average (about −1% against −7%), with more dispersion, but a single five-year realized return per firm is an extremely noisy
proxy for expected returns: the cross-sectional SD of returns is about 40 points. MM II predicts that leverage raises the
**riskiness** and the expected return of equity; the higher dispersion in the high-leverage group is consistent with the
first part.
"""),

dict(title="The WACC curve: MM with taxes and costly distress",
data="Market risk premium from the Fama–French data (1963–2018); risk-free rate and the BAA–AAA spread (December 2018) to calibrate distress costs.",
task="""
An all-equity firm has asset beta 0.9. Debt is risk-free up to 20% debt-to-value; beyond that, the cost of debt rises with the BAA–AAA spread scaled by leverage, and expected distress costs grow quadratically.

1. For debt-to-value ratios from 0 to 70%, compute the levered equity beta (Hamada with taxes), the cost of equity, the cost of debt, the WACC and firm value for a perpetual pre-tax cash flow of \\$100m (tax rate 21%).
2. Find the value-maximizing leverage.
3. Explain which ingredient makes the WACC curve U-shaped. What would MM without taxes predict?
""",
starter="""
import numpy as np, pandas as pd
from arch.data import default
""" + FF + """
# TODO: grid over L = D/V; beta_E = beta_A (1 + (1-t) D/E); r_D = rf + spread * f(L); V = EBIT(1-t)/WACC - distress
""",
solution="""
import numpy as np, pandas as pd
from arch.data import default
""" + FF + """
mrp = 12 * ff.loc["1963-07":, "Mkt-RF"].mean(); rf = 12 * ff["RF"].iloc[-1]
dd = default.load(); spread = (dd["BAA"] - dd["AAA"]).iloc[-1] / 100
tax, bA, ebit = 0.21, 0.9, 100.0
rows = []
for L in np.arange(0, 0.71, 0.05):
    DE = L / (1 - L) if L < 1 else np.inf
    bE = bA * (1 + (1 - tax) * DE)
    rE = rf + bE * mrp
    rD = rf + (0 if L <= 0.2 else spread * ((L - 0.2) / 0.1)**1.5)
    wacc = (1 - L) * rE + L * rD * (1 - tax)
    distress = 400 * max(L - 0.3, 0)**2                  # PV of expected distress costs, $m
    V = ebit * (1 - tax) / wacc - distress
    rows.append((L, bE, rE, rD, wacc, V))
t = pd.DataFrame(rows, columns=["D/V", "equity beta", "r_E", "r_D", "WACC", "value $m"])
print(f"MRP {100*mrp:.2f}%, rf {100*rf:.2f}%, BAA-AAA spread {100*spread:.2f}%")
print(t.iloc[::2].round(4).to_string(index=False))
best = t.loc[t["value $m"].idxmax()]
print(f"value-maximizing leverage: D/V = {best['D/V']:.2f}, WACC {100*best['WACC']:.2f}%, value ${best['value $m']:.0f}m")
""",
notes="""As leverage rises, the equity beta rises (MM II), but cheap tax-deductible debt lowers the WACC up to about 25% debt, where
firm value peaks at about \\$1.05 billion, 4% above the all-equity value. Beyond that, the rising cost of risky debt and
expected distress costs push the WACC up steeply. Without taxes and distress costs, MM Proposition I says the WACC and firm
value are independent of leverage: the curve would be flat. The U-shape comes from trading off the tax shield against
distress costs — the trade-off theory. The specific optimum depends entirely on the assumed distress-cost function.
"""),
]

EXERCISES["11-market-efficiency.md"] = [
dict(title="Weak-form efficiency: autocorrelations and variance ratios",
data="Weekly NYSE index returns (`nyse`, Wooldridge), and daily S&P 500 log returns 1999–2018.",
task="""
1. Estimate the AR(1) of weekly NYSE returns (Wooldridge, 2020, Example 11.4). Is last week's return a useful predictor?
2. Compute Lo–MacKinlay variance ratios \\(VR(q)=\\operatorname{Var}(r_t+\\dots+r_{t-q+1})/(q\\operatorname{Var}(r_t))\\) for the daily S&P 500 at \\(q=2,5,10,20\\), with heteroskedasticity-robust \\(z\\)-statistics.
3. Interpret. Do the results contradict market efficiency? What would trading costs do to any apparent predictability?
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
from arch.data import sp500
n = wd.data("nyse")
r = smf.ols("Q('return') ~ return_1", data=n).fit()
# TODO: variance ratios from overlapping q-day sums; robust z (Lo-MacKinlay 1988)
""",
solution="""
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
from arch.data import sp500
n = wd.data("nyse")
r = smf.ols("Q('return') ~ return_1", data=n).fit(cov_type="HC1")
print(f"weekly NYSE: return_t = {r.params.iloc[0]:.3f} + {r.params['return_1']:.4f} return_(t-1)  (robust t = {r.tvalues['return_1']:.2f}, R2 = {r.rsquared:.4f})")
x = np.log(sp500.load()["Adj Close"]).diff().dropna().to_numpy(); T = len(x); mu = x.mean(); s2 = np.mean((x - mu)**2)
for q in (2, 5, 10, 20):
    sq = np.convolve(x, np.ones(q), "valid")
    vr = np.mean((sq - q * mu)**2) / (q * s2)
    e2 = (x - mu)**2
    theta = sum((2 * (q - j) / q)**2 * np.sum(e2[j:] * e2[:-j]) / np.sum(e2)**2 * T for j in range(1, q))
    print(f"q = {q:2d}: VR = {vr:.3f}, robust z = {(vr - 1) / np.sqrt(theta / T):.2f}")
""",
notes="""Weekly NYSE returns show a small, insignificant positive autocorrelation (\\(R^2\\) below 0.4%), as in Wooldridge's example.
For the daily S&P 500 in 1999–2018, variance ratios are significantly **below one** at all horizons: daily returns show mild
negative serial correlation (short-term reversals), concentrated in volatile periods. The predictability is small, and it
would be very hard to exploit after trading costs and the bid–ask bounce. Statistical rejection of a random walk is not the
same as a profitable trading rule, which is what market efficiency rules out.
"""),

dict(title="Event studies of market-wide shocks",
data="Daily S&P 500 log returns, 1999–2018.",
task="""
Measure the market's reaction to three events, using a constant-mean model estimated over the 250 trading days ending 30 days before each event:
the reopening of markets after 9/11 (17 September 2001), Lehman Brothers' bankruptcy filing (15 September 2008) and the Brexit referendum result (24 June 2016).

1. Compute abnormal returns on the event day and cumulative abnormal returns (CAR) over days 0 to +4.
2. Compute \\(t\\)-statistics using the estimation-window standard deviation.
3. Did the market fully react on day 0, or did prices drift? What does a drift imply for semi-strong efficiency, and why are market-wide events harder to study than firm-specific ones?
""",
starter="""
import numpy as np, pandas as pd
from arch.data import sp500
r = 100 * np.log(sp500.load()["Adj Close"]).diff().dropna()
events = {"9/11 reopening": "2001-09-17", "Lehman": "2008-09-15", "Brexit": "2016-06-24"}
# TODO: estimation window [-280, -31]; AR = r - mean; CAR(0, 4); t = CAR / (sd * sqrt(5))
""",
solution="""
import numpy as np, pandas as pd
from arch.data import sp500
r = 100 * np.log(sp500.load()["Adj Close"]).diff().dropna()
events = {"9/11 reopening": "2001-09-17", "Lehman": "2008-09-15", "Brexit": "2016-06-24"}
for name, day in events.items():
    i = r.index.get_loc(pd.Timestamp(day))
    est = r.iloc[i - 280:i - 30]; mu, sd = est.mean(), est.std()
    ar = r.iloc[i:i + 5] - mu
    car = ar.cumsum()
    print(f"{name:15s} ({day}): AR day 0 = {ar.iloc[0]:6.2f}% (t = {ar.iloc[0]/sd:6.2f});  CAR(0,1) = {car.iloc[1]:6.2f}%;  CAR(0,4) = {car.iloc[4]:6.2f}% (t = {car.iloc[4]/(sd*np.sqrt(5)):6.2f});  estimation sd {sd:.2f}%")
""",
notes="""All three events produced day-0 abnormal returns of about −3.7% to −5% (\\(t\\) around −3.5). After the 9/11 reopening, prices
kept falling all week (5-day CAR about −12%). After Lehman and Brexit, the 5-day CARs were close to zero: the market partly
reversed the initial drop. Persistent drift would suggest under-reaction, and reversal over-reaction, but with market-wide
events there is no control group: the "abnormal" return is just the market return minus its usual mean, and other news
arrives on the same days. Firm-level event studies, with the market as a benchmark, are much more powerful.
"""),
]

EXERCISES["12-banking.md"] = [
dict(title="Credit scoring and the Boston Fed mortgage study",
data="`loanapp` — 1,989 mortgage applications in Boston, 1990 (Munnell et al., 1996): approval `approve`, `white`, housing-expense and total-debt ratios `hrat`, `obrat`, loan-to-value `loanprc`, credit history `cosign`, `chist`, `pubrec`, `mortlat1`, `mortlat2`, `unem`, `male`, `married`, `dep`, `sch`.",
task="""
1. Fit a logit of approval on the financial and credit-history variables (without race). Compute the in-sample AUC (area under the ROC curve).
2. Add `white`. Report the average marginal effect of being white on the approval probability, with a robust standard error.
3. Interpret the coefficient as evidence on discrimination. What omitted-variable concerns did critics raise, and why does a high AUC not settle them?
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score     # or compute AUC by ranks
d = wd.data("loanapp")
X = "hrat + obrat + loanprc + unem + male + married + dep + sch + cosign + chist + pubrec + mortlat1 + mortlat2 + vr"
# TODO: logit without and with white; AUC; get_margeff
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
from scipy.stats import rankdata
d = wd.data("loanapp")
X = "hrat + obrat + loanprc + unem + male + married + dep + sch + cosign + chist + pubrec + mortlat1 + mortlat2 + vr"
d = d.dropna(subset=["approve", "white"] + X.split(" + "))
def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
m0 = smf.logit(f"approve ~ {X}", data=d).fit(disp=0)
m1 = smf.logit(f"approve ~ white + {X}", data=d).fit(disp=0, cov_type="HC0")
print(f"n = {len(d)}, approval rate {d['approve'].mean():.3f} (white {d.loc[d.white == 1, 'approve'].mean():.3f}, non-white {d.loc[d.white == 0, 'approve'].mean():.3f})")
print(f"AUC without race {auc(d['approve'].to_numpy(), m0.predict()):.3f}; with race {auc(d['approve'].to_numpy(), m1.predict()):.3f}")
me = m1.get_margeff().summary_frame().loc["white"]
print(f"average marginal effect of white on approval: {me['dy/dx']:.3f} (se {me['Std. Err.']:.3f}, p = {me['Pr(>|z|)']:.4f})")
print(f"obrat AME: {m1.get_margeff().summary_frame().loc['obrat', 'dy/dx']:.4f} per percentage point")
""",
notes="""The approval rate was 91% for white applicants and 70% for others. Financial and credit-history variables discriminate well
(AUC about 0.78), and adding race barely changes the AUC. Yet, controlling for those variables, being white raises the
approval probability by about 8 percentage points (\\(p<0.001\\)), close to Munnell et al.'s (1996) estimate. Critics argued
that the regression omits variables lenders observe (data errors, unobserved credit quality). A high AUC means the
**included** variables predict approval well; it says nothing about whether an omitted variable correlated with race
explains the gap. The burden is on that omitted variable to be large enough.
"""),

dict(title="Portfolio credit risk: the Vasicek model behind Basel capital",
data="Default probabilities calibrated to corporate credit (1% and 2% a year); asset correlation 0.15 as in the Basel IRB formula; loss given default 45%.",
task="""
In the one-factor Vasicek model, the fraction of a large loan portfolio that defaults, conditional on the common factor \\(Z\\), is \\(\\Phi\\big((\\Phi^{-1}(PD)-\\sqrt\\rho Z)/\\sqrt{1-\\rho}\\big)\\).

1. Simulate the portfolio loss distribution (one million draws of \\(Z\\)) for \\(PD=1\\%\\) and 2%.
2. Compute expected loss, the 99.9% quantile of losses, and the Basel "unexpected loss" capital charge (99.9% quantile minus expected loss), and check against the closed form.
3. Show how the capital charge changes when \\(\\rho\\) rises from 0.12 to 0.24. Why is correlation, not just PD, central to bank capital?
""",
starter="""
import numpy as np
from scipy.stats import norm
rng = np.random.default_rng(0); Z = rng.standard_normal(1_000_000); LGD = 0.45
loss = lambda pd_, rho: LGD * norm.cdf((norm.ppf(pd_) - np.sqrt(rho) * Z) / np.sqrt(1 - rho))
# TODO: EL, quantile, closed form: LGD * Phi((Phi^-1(PD) + sqrt(rho) Phi^-1(0.999)) / sqrt(1 - rho))
""",
solution="""
import numpy as np
from scipy.stats import norm
rng = np.random.default_rng(0); Z = rng.standard_normal(1_000_000); LGD = 0.45
loss = lambda p, rho: LGD * norm.cdf((norm.ppf(p) - np.sqrt(rho) * Z) / np.sqrt(1 - rho))
closed = lambda p, rho: LGD * norm.cdf((norm.ppf(p) + np.sqrt(rho) * norm.ppf(0.999)) / np.sqrt(1 - rho))
for p in (0.01, 0.02):
    for rho in (0.12, 0.15, 0.24):
        L = loss(p, rho); q = np.quantile(L, 0.999)
        print(f"PD {p:.0%}, rho {rho:.2f}: expected loss {100*L.mean():.2f}%, 99.9% loss {100*q:.2f}% (closed form {100*closed(p, rho):.2f}%), capital = UL {100*(q - L.mean()):.2f}% of exposure")
""",
notes="""Simulated 99.9% losses match the closed-form Vasicek quantile. With a 1% PD and Basel's correlation range, the capital charge
(unexpected loss) is about 4–5% of exposure, roughly ten times expected loss. Doubling the asset correlation from 0.12 to
0.24 doubles the capital requirement at a given PD: correlated defaults fatten the loss tail. This is why the Basel formula
assigns correlations by asset class, and why underestimated default correlation (as in mortgage-backed securities before
2008) left banks undercapitalized.
"""),
]

EXERCISES["13-risk-management.md"] = [
dict(title="Value-at-Risk and Expected Shortfall for the S&P 500, with backtests",
data="Daily S&P 500 log returns, 1999–2018.",
task="""
1. Using a rolling 500-day window, compute one-day 99% VaR each day by (a) historical simulation, (b) a normal distribution with the window's mean and SD, and (c) RiskMetrics EWMA volatility (\\(\\lambda=0.94\\)) with normal quantiles.
2. Count VaR violations over 2001–2018 and run Kupiec's unconditional-coverage likelihood-ratio test for each method.
3. Report the violations in 2008 alone. Why does historical simulation over a long window react slowly to a crisis?
4. Compute the full-sample 99% Expected Shortfall by historical simulation and compare it with VaR.
""",
starter="""
import numpy as np, pandas as pd
from scipy.stats import norm, chi2
from arch.data import sp500
r = 100 * np.log(sp500.load()["Adj Close"]).diff().dropna()
# TODO: rolling quantile, rolling std, EWMA variance; violations; Kupiec LR
""",
solution="""
import numpy as np, pandas as pd
from scipy.stats import norm, chi2
from arch.data import sp500
r = 100 * np.log(sp500.load()["Adj Close"]).diff().dropna()
hs = -r.rolling(500).quantile(0.01).shift(1)
nv = -(r.rolling(500).mean() - norm.ppf(0.99) * r.rolling(500).std()).shift(1)
ew = np.zeros(len(r)); ew[0] = r.iloc[:50].var()
for t in range(1, len(r)): ew[t] = 0.94 * ew[t - 1] + 0.06 * r.iloc[t - 1]**2
em = pd.Series(norm.ppf(0.99) * np.sqrt(ew), index=r.index)
def kupiec(v, p=0.01):
    n, x = len(v), v.sum(); ph = x / n
    return -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - ph) - x * np.log(ph))
for lab, var in (("historical", hs), ("normal", nv), ("EWMA", em)):
    s = pd.concat([r, var], axis=1, keys=["r", "var"]).loc["2001":].dropna(); v = (s.r < -s["var"])
    lr = kupiec(v)
    print(f"{lab:10s}: {v.sum():3d} violations in {len(v)} days ({v.mean():.2%}, expected 1%), Kupiec LR = {lr:.2f} (p = {chi2.sf(lr, 1):.3f}); in 2008: {v.loc['2008'].sum()}")
q = r.quantile(0.01)
print(f"full sample: 99% VaR {-q:.2f}%, 99% Expected Shortfall {-r[r <= q].mean():.2f}%")
""",
notes="""All three methods produce too many violations over 2001–2018, and Kupiec's test rejects correct coverage for each. The
normal model is worst (2.5% violations), because daily returns have fat tails. Historical simulation does better on average
but clusters its violations: 21 in 2008 alone, because a 500-day window adapts slowly to a sudden jump in volatility. EWMA
reacts quickly (only 9 violations in 2008) but uses normal quantiles, so it fails in the tails overall. Expected Shortfall
(4.8%) is much larger than VaR (3.4%): the average loss beyond the VaR threshold is large, which is why Basel III moved to ES.
"""),

dict(title="GARCH volatility and filtered historical simulation",
data="Daily S&P 500 log returns, 1999–2018.",
task="""
1. Estimate a GARCH(1,1) with Student-\\(t\\) errors using the `arch` package. Report the persistence \\(\\alpha+\\beta\\), the half-life of volatility shocks and the degrees of freedom.
2. Compare the conditional volatility at its 2008 peak and at the end of 2018 with the unconditional volatility.
3. Compute a one-day 99% VaR for 2 January 2019 by **filtered historical simulation**: scale the empirical quantile of standardized residuals by the forecast volatility. Compare with the GARCH-\\(t\\) quantile.
""",
starter="""
import numpy as np
from arch import arch_model
from arch.data import sp500
r = 100 * np.log(sp500.load()["Adj Close"]).diff().dropna()
res = arch_model(r, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
# TODO: persistence, half-life, conditional_volatility, forecast(horizon=1), std_resid quantile
""",
solution="""
import numpy as np
from scipy import stats
from arch import arch_model
from arch.data import sp500
r = 100 * np.log(sp500.load()["Adj Close"]).diff().dropna()
res = arch_model(r, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
a, b, nu = res.params["alpha[1]"], res.params["beta[1]"], res.params["nu"]
print(res.params.round(4).to_dict())
print(f"persistence alpha + beta = {a + b:.4f}, half-life {np.log(0.5)/np.log(a + b):.0f} days, t degrees of freedom {nu:.1f}")
cv = res.conditional_volatility; unc = np.sqrt(res.params["omega"] / (1 - a - b))
print(f"conditional vol: 2008 peak {cv.loc['2008'].max():.2f}% ({cv.loc['2008'].idxmax().date()}), end-2018 {cv.iloc[-1]:.2f}%, unconditional {unc:.2f}% (sample sd {r.std():.2f}%)")
f = np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0])
z = res.std_resid.dropna()
var_fhs = -(res.params["mu"] + f * np.quantile(z, 0.01))
var_t = -(res.params["mu"] + f * stats.t.ppf(0.01, nu) * np.sqrt((nu - 2) / nu))
print(f"forecast vol for the next day {f:.2f}%: 99% VaR by filtered historical simulation {var_fhs:.2f}%, GARCH-t {var_t:.2f}%")
""",
notes="""Volatility is extremely persistent: \\(\\alpha+\\beta\\approx0.9996\\), close to integrated GARCH, so the implied "unconditional"
volatility (5%) is unreliable — far above the sample SD of 1.2%. In practice, shocks to volatility decay over months. The
Student-\\(t\\) degrees of freedom (about 6.5) confirm fat tails even after conditioning on volatility. Conditional volatility
peaked at about 5.6% a day in October 2008. Filtered historical simulation gives a slightly higher VaR than the GARCH-\\(t\\)
quantile, because the empirical standardized residuals have a heavier left tail than a symmetric \\(t\\) distribution.
"""),
]

EXERCISES["14-behavioral-finance.md"] = [
dict(title="How do people invest their retirement money?",
data="`pension` — 194 U.S. workers with self-directed pension accounts (1989): share invested in stocks `pctstck` (0, 50 or 100), whether they chose their allocation `choice`, age, education, gender, family income and wealth.",
task="""
1. Tabulate the distribution of `pctstck`. What does the clustering at 0, 50 and 100 suggest about how people make allocation decisions?
2. Regress `pctstck` on `choice`, `age`, `educ`, `female`, `black`, `married`, income dummies and `prftshr`. Interpret the coefficients on age and gender.
3. Relate the results to the "1/n" heuristic (Benartzi and Thaler, 2001) and to life-cycle portfolio advice.
""",
starter="""
import wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("pension")
# TODO: value_counts; OLS with robust se
""",
solution="""
import wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("pension")
print("distribution of % in stocks:", d["pctstck"].value_counts().sort_index().to_dict())
r = smf.ols("pctstck ~ choice + age + educ + female + black + married + finc25 + finc35 + finc50 + finc75 + finc100 + finc101 + prftshr",
            data=d).fit(cov_type="HC1")
for v in ("choice", "age", "educ", "female", "prftshr"):
    print(f"  {v:8s}: {r.params[v]:7.2f} (robust se {r.bse[v]:.2f})")
print(f"R2 = {r.rsquared:.3f}, n = {int(r.nobs)}")
print("mean % in stocks by choice:", d.groupby("choice")["pctstck"].mean().round(1).to_dict())
""",
notes="""Allocations cluster at 0, 50 and 100% in stocks, and 50% is the most common choice: many people seem to split evenly rather
than optimize — consistent with the "1/n" heuristic of Benartzi and Thaler (2001). Workers who chose their own allocation
hold about 12 points more in stocks. Each year of age lowers the stock share by about 1.7 points, in line with life-cycle
advice. The gender coefficient is small and insignificant here. With 194 observations and \\(R^2=0.10\\), most of the
variation in allocations is unexplained by these characteristics — another sign of heuristics and inertia.
"""),

dict(title="Was the NASDAQ a bubble? A test for explosive prices",
data="Weekly NASDAQ composite closing prices, 1999–2002 (`arch.data.nasdaq`).",
task="""
Phillips, Wu and Yu (2011) detect bubbles with right-tailed unit-root tests: an explosive root (\\(\\rho>1\\)) in log prices indicates prices rising faster than any random walk.

1. Compute the forward-recursive sup-ADF statistic: ADF \\(t\\)-statistics on samples starting at the first observation and ending at each later week (minimum window 20% of the sample), and take the maximum.
2. Simulate the 95% critical value of the sup-ADF statistic under a driftless random walk (500 replications).
3. Date the episode: simulate the 95% critical value of the recursive ADF statistic **for each end point**, and find the weeks in which the statistic exceeds it. Compare with the March 2000 peak.
""",
starter="""
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller
from arch.data import nasdaq
p = np.log(nasdaq.load()["Adj Close"].loc["1999":"2002"].resample("W").last())
# TODO: recursive adfuller(p[:k], maxlag=1, regression="c", autolag=None)[0] for k >= 0.2 n
""",
solution="""
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller
from arch.data import nasdaq
p = np.log(nasdaq.load()["Adj Close"].loc["1999":"2002"].resample("W").last()).dropna()
n = len(p); r0 = int(0.2 * n)
def recursive(x):
    return np.array([adfuller(x[:k], maxlag=1, regression="c", autolag=None)[0] for k in range(r0, len(x) + 1)])
stat = recursive(p.to_numpy()); sadf = stat.max()
rng = np.random.default_rng(1)
sims = np.array([recursive(np.cumsum(rng.standard_normal(n))) for _ in range(500)])
cv = np.quantile(sims.max(1), 0.95); cv_t = np.quantile(sims, 0.95, axis=0)        # sup-ADF and pointwise critical values
print(f"{n} weeks; sup-ADF = {sadf:.2f}, simulated 95% critical value {cv:.2f}  -> explosive episode detected: {sadf > cv}")
ends = p.index[r0 - 1:]
flag = pd.Series(stat > cv_t, index=ends)
runs = (flag != flag.shift()).cumsum()[flag]
for _, w in flag[flag].groupby(runs):
    print(f"episode: {w.index[0].date()} to {w.index[-1].date()} ({len(w)} week{'s' if len(w) > 1 else ''} above the pointwise 95% critical value)")
print(f"NASDAQ peak week: {p.idxmax().date()} (level {np.exp(p.max()):,.0f}); end-2002 level {np.exp(p.iloc[-1]):,.0f} ({np.exp(p.iloc[-1] - p.max()) - 1:.0%})")
""",
notes="""The sup-ADF statistic exceeds its simulated 95% critical value, so the test detects explosive behavior. Date-stamping finds
the main episode from November 1999 to April 2000, bracketing the March 2000 peak, after which the index fell 74%. A few
short episodes during the collapse (a single week after 9/11 and several weeks in mid-2002) reflect the persistent slide,
which a right-tailed test with a fixed starting point can also flag; isolated one-week crossings are best treated as noise. The test detects prices rising faster than a random walk; calling that a "bubble" requires the
additional argument that fundamentals did not grow explosively. The full Phillips–Shi–Yu (2015) GSADF procedure also varies
the starting point.
"""),
]

EXERCISES["15-financial-econometrics.md"] = [
dict(title="Do dividend yields predict stock returns? Long horizons and Stambaugh bias",
data="`volat` — monthly S&P 500 index and dividend yield `divyld`, 1947–1993.",
task="""
1. Compute log returns over the next 1, 12, 24 and 48 months (price returns plus the dividend yield accrued) and regress them on the current dividend yield, with Newey–West standard errors (lags = horizon).
2. Report slopes, \\(t\\)-statistics and \\(R^2\\). How do they change with the horizon?
3. Estimate the persistence of the dividend yield (AR(1)) and the correlation between its innovations and return innovations. Use Stambaugh's formula to compute the small-sample bias of the one-month slope: \\(E[\\widehat b-b]\\approx\\frac{\\sigma_{uv}}{\\sigma_v^2}E[\\widehat\\rho-\\rho]\\) with \\(E[\\widehat\\rho-\\rho]\\approx-(1+3\\rho)/T\\).
""",
starter="""
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
v = wd.data("volat")
lp = np.log(v["sp500"]); dy = v["divyld"] / 100
# TODO: future h-month return = lp.shift(-h) - lp + sum of dy/12 over the next h months; regress on dy
""",
solution="""
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
v = wd.data("volat")
lp = np.log(v["sp500"]); dy = v["divyld"] / 100
d = pd.DataFrame({"dy": 100 * dy})
for h in (1, 12, 24, 48):
    d[f"r{h}"] = 100 * (lp.shift(-h) - lp + (dy / 12)[::-1].rolling(h).sum()[::-1].shift(-1))
    s = d.dropna(subset=[f"r{h}"])
    r = smf.ols(f"r{h} ~ dy", data=s).fit(cov_type="HAC", cov_kwds={"maxlags": h})
    print(f"horizon {h:2d} months: slope {r.params['dy']:6.2f} (NW t = {r.tvalues['dy']:5.2f}), R2 = {r.rsquared:.3f}")
s = d.dropna(subset=["r1"]); T = len(s)
ar = smf.ols("dy ~ dy_lag", data=pd.DataFrame({"dy": s["dy"].iloc[1:].to_numpy(), "dy_lag": s["dy"].iloc[:-1].to_numpy()})).fit()
rho = ar.params["dy_lag"]; v_ = ar.resid.to_numpy()
u = smf.ols("r1 ~ dy", data=s).fit().resid.to_numpy()[:-1]
suv, sv2 = np.cov(u, v_)[0, 1], v_.var(ddof=1)
bias = suv / sv2 * (-(1 + 3 * rho) / T)
print(f"dividend-yield AR(1) rho = {rho:.4f}; corr(return, yield innovations) = {np.corrcoef(u, v_)[0,1]:.3f}")
print(f"Stambaugh bias in the 1-month slope ~ {bias:.3f} (vs estimate {smf.ols('r1 ~ dy', data=s).fit().params['dy']:.3f})")
""",
notes="""The one-month slope is significant but tiny in \\(R^2\\) (1.4%). The slopes and \\(R^2\\) grow almost proportionally with the
horizon, reaching 60% at four years. This is the famous long-horizon predictability. But the dividend yield is almost a unit
root (\\(\\rho\\approx0.99\\)), and its innovations are strongly negatively correlated with returns (\\(-0.89\\)), so the
one-month slope is biased upward: Stambaugh's formula attributes about a third of it to small-sample bias. Overlapping
long-horizon regressions add no independent information — they mostly cumulate the short-horizon slope — and their
\\(t\\)-statistics are badly oversized even with Newey–West errors (Boudoukh, Richardson and Whitelaw, 2008).
"""),

dict(title="Asymmetric volatility: GJR-GARCH and the leverage effect",
data="Daily S&P 500 log returns 1999–2018, and the VIX 2014–2018.",
task="""
1. Estimate GARCH(1,1) and GJR-GARCH(1,1,1) models with Student-\\(t\\) errors. Compare log-likelihoods and BIC. Is the asymmetry term \\(\\gamma\\) significant?
2. Compute the "news impact" — next-day variance after a −2% and a +2% return — implied by the GJR model.
3. Compare the GJR one-day-ahead volatility forecast (annualized) with the VIX over 2014–2018: correlation and average gap. Why is the VIX systematically higher?
""",
starter="""
import numpy as np, pandas as pd
from arch import arch_model
from arch.data import sp500, vix
r = 100 * np.log(sp500.load()["Adj Close"]).diff().dropna()
g = arch_model(r, vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
# TODO: compare with o=0; news impact; conditional_volatility * sqrt(252) vs vix
""",
solution="""
import numpy as np, pandas as pd
from arch import arch_model
from arch.data import sp500, vix
r = 100 * np.log(sp500.load()["Adj Close"]).diff().dropna()
g0 = arch_model(r, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
g1 = arch_model(r, vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
print(f"GARCH: logL {g0.loglikelihood:.1f}, BIC {g0.bic:.1f};  GJR: logL {g1.loglikelihood:.1f}, BIC {g1.bic:.1f}")
P = g1.params
print(f"GJR: alpha {P['alpha[1]']:.4f}, gamma {P['gamma[1]']:.4f} (t = {g1.tvalues['gamma[1]']:.1f}), beta {P['beta[1]']:.4f}")
s2 = g1.conditional_volatility.iloc[-1]**2
for shock in (-2.0, 2.0):
    e = shock - P["mu"]
    nxt = P["omega"] + (P["alpha[1]"] + P["gamma[1]"] * (e < 0)) * e**2 + P["beta[1]"] * s2
    print(f"next-day volatility after a {shock:+.0f}% day: {np.sqrt(nxt):.2f}% (from {np.sqrt(s2):.2f}%)")
vol = g1.conditional_volatility * np.sqrt(252)
df = pd.concat([vol.rename("garch"), vix.load()["vix"].rename("vix")], axis=1).dropna().loc["2014":"2018"]
print(f"2014-2018: corr(GJR vol, VIX) = {df.corr().iloc[0,1]:.3f}; mean GJR vol {df.garch.mean():.1f}%, mean VIX {df.vix.mean():.1f}%")
""",
notes="""The GJR model improves the fit substantially (BIC falls by over 160). The asymmetry term is large and highly significant,
while the symmetric \\(\\alpha\\) is essentially zero: only negative returns raise future volatility. After a −2% day, volatility
rises; after a +2% day, it falls. This is the leverage (or volatility-feedback) effect. The GJR volatility forecast tracks the
VIX closely (correlation 0.84) but is about 2 points lower on average: the VIX is a risk-neutral expectation and includes the
variance risk premium measured in Lecture 8.
"""),
]
