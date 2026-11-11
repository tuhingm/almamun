INTRO = """
These exercises use **real data** in Python. Install once with
`pip install wooldridge statsmodels scipy numpy pandas`. The `wooldridge` package holds survey, experimental and
administrative datasets; `scipy.stats` and `statsmodels` provide the inference tools. Try the starter code first, then
open the solution. Every printed result below comes from running the solution code exactly as shown.
"""

EXERCISES = {}

EXERCISES["01-probability-models.md"] = [
dict(title="Bayes' rule with real data: smoking and low birth weight",
data="`bwght` — 1,388 U.S. births, 1988 National Health Interview Survey (`bwght` in ounces, `cigs` = cigarettes per day during pregnancy).",
task="""
Define the events \\(S\\) = "mother smoked during pregnancy" (`cigs > 0`) and \\(L\\) = "low birth weight" (below 88 ounces, about 2.5 kg).

1. Estimate \\(P(S)\\), \\(P(L)\\), \\(P(L\\mid S)\\) and \\(P(L\\mid S^c)\\).
2. Use Bayes' rule to compute \\(P(S\\mid L)\\), and check it against the direct sample frequency.
3. Compute the relative risk \\(P(L\\mid S)/P(L\\mid S^c)\\) and its 95% confidence interval on the log scale.
4. Are \\(S\\) and \\(L\\) independent? Test with a \\(\\chi^2\\) test of independence.
""",
starter="""
import numpy as np, pandas as pd, wooldridge as wd
from scipy import stats
d = wd.data("bwght")
S, L = d["cigs"] > 0, d["bwght"] < 88
# TODO: probabilities, Bayes' rule, relative risk, stats.chi2_contingency(pd.crosstab(S, L))
""",
solution="""
import numpy as np, pandas as pd, wooldridge as wd
from scipy import stats
d = wd.data("bwght")
S, L = d["cigs"] > 0, d["bwght"] < 88
pS, pL = S.mean(), L.mean()
pL_S, pL_nS = L[S].mean(), L[~S].mean()
print(f"P(S) = {pS:.4f}   P(L) = {pL:.4f}   P(L|S) = {pL_S:.4f}   P(L|not S) = {pL_nS:.4f}")
bayes = pL_S * pS / (pL_S * pS + pL_nS * (1 - pS))
print(f"P(S|L) by Bayes' rule = {bayes:.4f};  directly = {S[L].mean():.4f}")
a, b, c, e = (S & L).sum(), (S & ~L).sum(), (~S & L).sum(), (~S & ~L).sum()
rr = (a / (a + b)) / (c / (c + e)); se = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+e))
print(f"relative risk = {rr:.2f}, 95% CI [{rr*np.exp(-1.96*se):.2f}, {rr*np.exp(1.96*se):.2f}]  (counts: {a} smokers with low weight)")
chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(S, L))
print(f"chi-square test of independence: chi2({dof}) = {chi2:.2f}, p = {p:.4f}")
""",
notes="""About 15% of mothers smoked, and 6% of babies had low birth weight. Bayes' rule and the direct frequency agree exactly,
as they must: \\(P(S\\mid L)\\approx0.24\\), so a quarter of low-weight babies had smoking mothers, against 15% of all babies.
The relative risk is about 1.8, with an interval from about 1.1 to 2.9, and the \\(\\chi^2\\) test rejects independence at 5%.
Only 20 smokers had low-weight babies, so the interval is wide. This is an association: smoking mothers differ in income,
education and other risk factors, so a causal claim needs the tools of Econometrics Lecture 13.
"""),

dict(title="The law of total variance in a wage distribution",
data="`wage1` — 526 workers, CPS 1976.",
task="""
1. Verify numerically that \\(\\operatorname{Var}(wage)=E[\\operatorname{Var}(wage\\mid educ)]+\\operatorname{Var}(E[wage\\mid educ])\\), using sample moments with divisor \\(n\\).
2. What share of wage variance is "explained" by education cells? Compare with the \\(R^2\\) of a linear regression on `educ`, and explain why the cell-based share is larger.
3. Repeat with `lwage`. Why does the share change?
""",
starter="""
import wooldridge as wd
d = wd.data("wage1")
g = d.groupby("educ")["wage"]
# TODO: within = (g.var(ddof=0) * g.size()).sum() / n ;  between = ((g.mean() - mean)**2 * g.size()).sum() / n
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("wage1"); n = len(d)
for y in ("wage", "lwage"):
    g = d.groupby("educ")[y]
    total = d[y].var(ddof=0)
    within = (g.var(ddof=0) * g.size()).sum() / n
    between = ((g.mean() - d[y].mean())**2 * g.size()).sum() / n
    r2 = smf.ols(f"{y} ~ educ", data=d).fit().rsquared
    print(f"{y:6s}: total {total:8.4f} = within {within:8.4f} + between {between:7.4f}  (check {within + between:8.4f})")
    print(f"        share explained by education cells {between/total:.3f}   vs linear R^2 {r2:.3f}")
""",
notes="""The decomposition holds exactly (the "check" column reproduces the total). Education cells account for about 22% of the
variance of wages, more than the linear \\(R^2\\) of 16.5%, because cell means capture the **nonlinear** CEF, while the
regression line can capture only its linear part. Logs raise both shares a little, because the log transformation makes
the CEF closer to linear and dampens the influence of very high wages on the total variance.
"""),
]

EXERCISES["02-distributions-multivariate.md"] = [
dict(title="Which distribution fits wages? Lognormal versus gamma by maximum likelihood",
data="`wage1` — 526 hourly wages, CPS 1976.",
task="""
1. Fit lognormal and gamma distributions to `wage` by maximum likelihood (`scipy.stats`, location fixed at 0).
2. Compare them by log-likelihood and AIC.
3. Compare fitted and empirical quantiles (10th, 50th, 90th, 99th percentiles).
4. Run a Kolmogorov–Smirnov test for each. Why are the KS p-values only approximate here?
""",
starter="""
import numpy as np, wooldridge as wd
from scipy import stats
w = wd.data("wage1")["wage"].to_numpy()
ln = stats.lognorm.fit(w, floc=0); ga = stats.gamma.fit(w, floc=0)
# TODO: log-likelihoods via .logpdf(w, *params).sum(); AIC = 2k - 2 logL; quantiles via .ppf
""",
solution="""
import numpy as np, wooldridge as wd
from scipy import stats
w = wd.data("wage1")["wage"].to_numpy()
fits = {"lognormal": (stats.lognorm, stats.lognorm.fit(w, floc=0)), "gamma": (stats.gamma, stats.gamma.fit(w, floc=0))}
q = [0.10, 0.50, 0.90, 0.99]
print("empirical quantiles:      ", np.quantile(w, q).round(2))
for name, (dist, par) in fits.items():
    ll = dist.logpdf(w, *par).sum()
    ks = stats.kstest(w, dist.cdf, args=par)
    print(f"{name:9s}: logL = {ll:8.2f}, AIC = {2*2 - 2*ll:7.2f}, quantiles {dist.ppf(q, *par).round(2)}, KS = {ks.statistic:.3f} (p = {ks.pvalue:.3f})")
""",
notes="""The lognormal fits much better than the gamma by log-likelihood and AIC (a difference of over 60 AIC points), and it
tracks the middle and upper quantiles more closely. Both understate the 99th percentile, so the right tail of wages is
heavier than either model allows. Both KS tests reject — with 526 observations, even moderate misfit is detected. The KS
p-values are only approximate because the parameters were estimated from the same data (a Lilliefors-type correction, or
a parametric bootstrap, gives exact p-values).
"""),

dict(title="Conditional normal distribution and the regression line",
data="`econmath` — 856 students in an introductory economics course: ACT math score `actmth` and college GPA `colgpa`.",
task="""
If \\((X,Y)\\) is bivariate normal, \\(E[Y\\mid X=x]=\\mu_Y+\\frac{\\sigma_{XY}}{\\sigma_X^2}(x-\\mu_X)\\) and \\(\\operatorname{Var}(Y\\mid X)=\\sigma_Y^2(1-\\rho^2)\\).

1. Estimate the mean vector and covariance matrix of `(actmth, colgpa)` (drop missing values).
2. Compute the implied conditional mean and SD of GPA at ACT math scores 20, 25 and 30.
3. Compare the conditional-mean slope with the OLS slope, and the conditional SD with the regression's residual SD.
4. Check the normality assumption for GPA with a skewness statistic. Is a GPA above 4.0 possible under the fitted normal model?
""",
starter="""
import numpy as np, wooldridge as wd
d = wd.data("econmath")[["actmth", "colgpa"]].dropna()
mu = d.mean().to_numpy(); S = d.cov().to_numpy()
# TODO: slope = S[0,1]/S[0,0]; cond_sd = sqrt(S[1,1]*(1-rho^2))
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
from scipy import stats
d = wd.data("econmath")[["actmth", "colgpa"]].dropna()
mu = d.mean().to_numpy(); S = d.cov().to_numpy(); rho = S[0, 1] / np.sqrt(S[0, 0] * S[1, 1])
slope = S[0, 1] / S[0, 0]; csd = np.sqrt(S[1, 1] * (1 - rho**2))
print(f"n = {len(d)}, means {mu.round(3)}, correlation {rho:.3f}")
for x in (20, 25, 30):
    m = mu[1] + slope * (x - mu[0])
    print(f"  ACT math {x}: E[GPA|x] = {m:.3f}, SD = {csd:.3f}, P(GPA > 4 | x) = {stats.norm.sf(4, m, csd):.4f}")
ols = smf.ols("colgpa ~ actmth", data=d).fit()
print(f"conditional-mean slope {slope:.5f} vs OLS slope {ols.params['actmth']:.5f};  conditional SD {csd:.4f} vs residual SD {np.sqrt(ols.mse_resid):.4f}")
print(f"skewness of GPA: {stats.skew(d['colgpa']):.3f}  (max observed {d['colgpa'].max():.2f})")
""",
notes="""The slope implied by the bivariate normal formula, \\(\\sigma_{XY}/\\sigma_X^2\\), is identical to the OLS slope: the
population regression of \\(Y\\) on \\(X\\) is the normal conditional mean. The conditional SD matches the residual SD
(up to the degrees-of-freedom correction). The correlation of 0.30 means ACT math explains only 9% of GPA variance.
The normal model assigns small but positive probability to a GPA above 4.0, which is impossible — GPA is bounded and
slightly left-skewed, so the normal model is an approximation that fails in the tails.
"""),
]

EXERCISES["03-sampling-asymptotics.md"] = [
dict(title="The CLT with a heavily skewed real population: CEO salaries",
data="`ceosal1` — 1990 salaries (thousands of dollars) of 209 CEOs. We treat these 209 values as a population.",
task="""
1. Describe the population: mean, median, skewness.
2. Draw 10,000 samples with replacement of size \\(n=5, 30, 200\\) and record the sample mean. For each \\(n\\), report the skewness of the sampling distribution and the coverage of the nominal 95% \\(t\\)-interval for the population mean.
3. Relate the results to the Berry–Esseen idea that the normal approximation error shrinks like \\(\\text{skewness}/\\sqrt n\\).
""",
starter="""
import numpy as np, wooldridge as wd
from scipy import stats
pop = wd.data("ceosal1")["salary"].to_numpy(); mu = pop.mean()
rng = np.random.default_rng(0)
# TODO: for n in (5, 30, 200): samples = rng.choice(pop, (10000, n)); compute means, t-intervals, coverage
""",
solution="""
import numpy as np, wooldridge as wd
from scipy import stats
pop = wd.data("ceosal1")["salary"].to_numpy(); mu = pop.mean()
print(f"population: mean {mu:.1f}, median {np.median(pop):.1f}, skewness {stats.skew(pop):.2f}")
rng = np.random.default_rng(0)
for n in (5, 30, 200):
    s = rng.choice(pop, (10000, n))
    m, sd = s.mean(1), s.std(1, ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    cover = np.mean(np.abs(m - mu) <= half)
    print(f"n = {n:3d}: skewness of sample mean {stats.skew(m):5.2f}, t-interval coverage {cover:.3f} (MC se {np.sqrt(cover*(1-cover)/10000):.3f})")
""",
notes="""CEO salaries are extremely skewed (skewness 6.9, with one salary above \\$14 million). The sampling distribution of the
mean is still visibly skewed at \\(n=200\\), and the \\(t\\)-interval **undercovers badly at every sample size** — 79–89%
instead of 95%. Coverage is not even monotone in \\(n\\): at \\(n=5\\), most samples miss the extreme salaries entirely,
which produces short intervals that happen to cover the mean more often than at \\(n=30\\). The Berry–Esseen bound says the
error shrinks like skewness\\(/\\sqrt n\\): with skewness near 7, "large \\(n\\)" means thousands. Use logs, a bootstrap-t
interval, or report the median.
"""),

dict(title="Delta method and bootstrap for a ratio: the 401(k) participation rate",
data="`401k` — 1,534 U.S. 401(k) plans: number of participants `totpart` and eligible employees `totelg`.",
task="""
The aggregate participation rate is the ratio \\(R=E[totpart]/E[totelg]\\).

1. Estimate \\(R\\) and its standard error by the delta method, using the gradient \\((1/\\mu_e,\\,-\\mu_p/\\mu_e^2)\\) and the sample covariance matrix of the means.
2. Estimate the standard error with a nonparametric bootstrap (2,000 draws, seed 1).
3. Compare \\(R\\) with the average of plan-level rates \\(E[prate]/100\\). Why do they differ?
""",
starter="""
import numpy as np, wooldridge as wd
d = wd.data("401k"); p, e = d["totpart"].to_numpy(), d["totelg"].to_numpy(); n = len(d)
R = p.mean() / e.mean()
# TODO: V = np.cov(np.vstack([p, e])) / n; se = sqrt(g' V g)
""",
solution="""
import numpy as np, wooldridge as wd
d = wd.data("401k"); p, e = d["totpart"].to_numpy(float), d["totelg"].to_numpy(float); n = len(d)
R = p.mean() / e.mean()
V = np.cov(np.vstack([p, e])) / n
g = np.array([1 / e.mean(), -p.mean() / e.mean()**2])
print(f"aggregate participation rate R = {R:.4f}, delta-method se = {np.sqrt(g @ V @ g):.4f}")
rng = np.random.default_rng(1); idx = rng.integers(0, n, (2000, n))
boot = p[idx].mean(1) / e[idx].mean(1)
print(f"bootstrap se = {boot.std(ddof=1):.4f}, 95% percentile CI [{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}]")
print(f"average plan-level rate = {d['prate'].mean()/100:.4f}")
print(f"correlation between plan size (totelg) and plan rate: {np.corrcoef(e, d['prate'])[0,1]:.3f}")
""",
notes="""The delta-method and bootstrap standard errors agree closely (about 0.016), because the ratio of two means is smooth and
\\(n\\) is large. The aggregate rate (83%) is lower than the average plan rate (87%): the aggregate weights plans by their
number of eligible employees, and larger plans have slightly lower participation (negative correlation). The two numbers
answer different questions — "what share of eligible workers participate?" versus "what is the typical plan's rate?".
"""),
]

EXERCISES["04-point-estimation.md"] = [
dict(title="Poisson maximum likelihood for arrest counts, by hand",
data="`crime1` — 2,725 young men in California; `narr86` = number of arrests in 1986.",
task="""
1. Write the Poisson log-likelihood for \\(\\lambda\\) and maximize it numerically with `scipy.optimize`. Confirm \\(\\widehat\\lambda=\\bar X\\).
2. Compute the standard error from (a) the observed information (numerical second derivative) and (b) the sandwich formula.
3. Compare the sample variance with the mean. What does over-dispersion imply for the two standard errors?
4. Compare observed and Poisson-predicted frequencies of 0, 1, 2 and 3+ arrests.
""",
starter="""
import numpy as np, wooldridge as wd
from scipy import optimize, stats
x = wd.data("crime1")["narr86"].to_numpy()
negll = lambda lam: -np.sum(stats.poisson.logpmf(x, lam))
# TODO: optimize.minimize_scalar(negll, bounds=(1e-6, 5), method="bounded")
""",
solution="""
import numpy as np, wooldridge as wd
from scipy import optimize, stats
x = wd.data("crime1")["narr86"].to_numpy(); n = len(x)
negll = lambda lam: -np.sum(stats.poisson.logpmf(x, lam))
lam = optimize.minimize_scalar(negll, bounds=(1e-6, 5), method="bounded", options={"xatol": 1e-10}).x
print(f"MLE lambda = {lam:.5f};  sample mean = {x.mean():.5f}")
h = 1e-4; info = (negll(lam + h) - 2 * negll(lam) + negll(lam - h)) / h**2
score_var = np.sum((x / lam - 1)**2)
print(f"se from observed information = {np.sqrt(1/info):.5f};  sandwich se = {np.sqrt(score_var)/info:.5f}")
print(f"sample variance {x.var(ddof=1):.4f} vs mean {x.mean():.4f}  (ratio {x.var(ddof=1)/x.mean():.2f})")
obs = [np.mean(x == k) for k in (0, 1, 2)] + [np.mean(x >= 3)]
fit = [stats.poisson.pmf(k, lam) for k in (0, 1, 2)] + [stats.poisson.sf(2, lam)]
print("P(0,1,2,3+) observed:", np.round(obs, 4), " Poisson:", np.round(fit, 4))
""",
notes="""The numerical MLE equals the sample mean, as theory says. The variance is 1.8 times the mean, so the Poisson assumption
\\(\\operatorname{Var}=E\\) fails. The sandwich standard error is about 35% larger than the information-based one, because
the information equality no longer holds. The Poisson model underpredicts zeros and, especially, three or more arrests
(2.8% observed against 0.8% predicted): the classic signature of overdispersion, typically from unobserved heterogeneity
in arrest risk. Lecture 9's negative binomial model addresses this.
"""),

dict(title="Method of moments versus maximum likelihood for a gamma model",
data="`wage1` — hourly wages.",
task="""
For a gamma distribution with shape \\(k\\) and scale \\(\\theta\\), \\(E[X]=k\\theta\\) and \\(\\operatorname{Var}(X)=k\\theta^2\\).

1. Derive and compute the method-of-moments estimators \\(\\widehat\\theta=s^2/\\bar x\\), \\(\\widehat k=\\bar x^2/s^2\\).
2. Compute the MLE with `scipy.stats.gamma.fit`.
3. Bootstrap both estimators (1,000 draws) and compare their standard deviations. Which is more efficient, and is that what theory predicts?
""",
starter="""
import numpy as np, wooldridge as wd
from scipy import stats
w = wd.data("wage1")["wage"].to_numpy()
# TODO: MoM, MLE, bootstrap both
""",
solution="""
import numpy as np, wooldridge as wd
from scipy import stats
w = wd.data("wage1")["wage"].to_numpy(); n = len(w)
mom = lambda x: (x.mean()**2 / x.var(ddof=1), x.var(ddof=1) / x.mean())
mle = lambda x: stats.gamma.fit(x, floc=0)[::2]
print("MoM  (shape, scale):", np.round(mom(w), 4))
print("MLE  (shape, scale):", np.round(mle(w), 4))
rng = np.random.default_rng(3); B = np.array([[*mom(s), *mle(s)] for s in (w[rng.integers(0, n, n)] for _ in range(1000))])
print(f"bootstrap SD of shape:  MoM {B[:,0].std():.4f}   MLE {B[:,2].std():.4f}")
print(f"bootstrap SD of scale:  MoM {B[:,1].std():.4f}   MLE {B[:,3].std():.4f}")
""",
notes="""The two estimators give quite different parameters (shape 2.5 against 3.5), which is itself a sign that the gamma model
fits poorly: under correct specification both would be consistent for the same values. MLE is noticeably more precise for
the scale parameter, but not for the shape. Theory says the MLE is asymptotically efficient **if the model is correct**;
when the model is misspecified, the two estimators target different pseudo-true values and the efficiency ranking need not
hold. Compare with the previous exercise, where the gamma lost clearly to the lognormal.
"""),
]

EXERCISES["05-interval-testing.md"] = [
dict(title="Did job training raise earnings? Welch t-test, permutation test and a confidence interval",
data="`jtrain2` — NSW randomized experiment, 445 men; `re78` = 1978 earnings (thousands of dollars).",
task="""
1. Compute the difference in mean earnings between trainees and controls, with Welch and pooled-variance \\(t\\)-tests.
2. Run a permutation test of the sharp null of no effect (10,000 permutations, seed 0).
3. Report a 95% Welch confidence interval and interpret it in dollars.
4. Earnings are highly skewed with many zeros. Which of the three procedures relies least on distributional assumptions?
""",
starter="""
import numpy as np, wooldridge as wd
from scipy import stats
d = wd.data("jtrain2"); y1, y0 = d.loc[d.train == 1, "re78"], d.loc[d.train == 0, "re78"]
# TODO: stats.ttest_ind(..., equal_var=False); permutation loop
""",
solution="""
import numpy as np, wooldridge as wd
from scipy import stats
d = wd.data("jtrain2"); y = d["re78"].to_numpy(); t = d["train"].to_numpy().astype(bool)
diff = y[t].mean() - y[~t].mean()
w, p = stats.ttest_ind(y[t], y[~t], equal_var=False), stats.ttest_ind(y[t], y[~t], equal_var=True)
print(f"difference in means = {diff:.3f} thousand dollars")
print(f"Welch t = {w.statistic:.3f} (p = {w.pvalue:.4f});  pooled t = {p.statistic:.3f} (p = {p.pvalue:.4f})")
rng = np.random.default_rng(0); perm = np.empty(10000)
for b in range(10000):
    s = rng.permutation(t); perm[b] = y[s].mean() - y[~s].mean()
print(f"permutation p-value (two-sided) = {np.mean(np.abs(perm) >= abs(diff)):.4f}")
v1, v0, n1, n0 = y[t].var(ddof=1), y[~t].var(ddof=1), t.sum(), (~t).sum()
se = np.sqrt(v1/n1 + v0/n0); df = se**4 / ((v1/n1)**2/(n1-1) + (v0/n0)**2/(n0-1))
q = stats.t.ppf(0.975, df)
print(f"95% Welch CI: [{diff - q*se:.3f}, {diff + q*se:.3f}]  (df = {df:.1f})")
print(f"share with zero earnings: treated {np.mean(y[t] == 0):.3f}, control {np.mean(y[~t] == 0):.3f}")
""",
notes="""Trainees earned about \\$1,800 more in 1978. All three procedures agree on significance (p < 0.01). The permutation test
uses only the random assignment — no normality, no equal variances — so it is the most assumption-light, and it is the
natural test for a randomized experiment. The Welch interval runs from about \\$470 to \\$3,100: the effect is clearly
positive but imprecisely measured. A third of controls and a quarter of trainees had zero earnings, so the distribution is
far from normal; the \\(t\\)-tests are rescued by the sample size of 445.
"""),

dict(title="Wald, likelihood-ratio and score tests for a proportion",
data="`pntsprd` — 553 college basketball games; `favwin` = 1 if the betting favorite won.",
task="""
Test whether favorites win 75% of games, \\(H_0:p=0.75\\).

1. Compute the Wald, likelihood-ratio and score statistics and their p-values.
2. Compute the Wald (normal) and Wilson (score-inverted) 95% intervals for \\(p\\).
3. Why do the three tests give similar answers here, and when would they diverge?
""",
starter="""
import numpy as np, wooldridge as wd
from scipy import stats
x = wd.data("pntsprd")["favwin"].to_numpy(); n, k = len(x), x.sum(); p0 = 0.75
# TODO: Wald = (ph-p0)^2 / (ph(1-ph)/n); score uses p0 in the variance; LR = 2(logL(ph) - logL(p0))
""",
solution="""
import numpy as np, wooldridge as wd
from scipy import stats
x = wd.data("pntsprd")["favwin"].to_numpy(); n, k = len(x), int(x.sum()); p0 = 0.75; ph = k / n
ll = lambda p: k * np.log(p) + (n - k) * np.log(1 - p)
W = (ph - p0)**2 / (ph * (1 - ph) / n); S = (ph - p0)**2 / (p0 * (1 - p0) / n); LR = 2 * (ll(ph) - ll(p0))
print(f"favorites won {k} of {n} games: p-hat = {ph:.4f}")
for name, s in (("Wald", W), ("LR", LR), ("score", S)):
    print(f"  {name:5s}: chi2(1) = {s:.3f}, p = {stats.chi2.sf(s, 1):.4f}")
z = 1.96; se = np.sqrt(ph * (1 - ph) / n)
c = (ph + z**2/(2*n)) / (1 + z**2/n); h = z * np.sqrt(ph*(1-ph)/n + z**2/(4*n**2)) / (1 + z**2/n)
print(f"Wald 95% CI [{ph - z*se:.4f}, {ph + z*se:.4f}]   Wilson 95% CI [{c - h:.4f}, {c + h:.4f}]")
""",
notes="""Favorites won 76.3% of games. None of the three tests rejects \\(p=0.75\\), and they give nearly identical statistics,
because \\(n\\) is large and \\(\\widehat p\\) is close to \\(p_0\\): the log-likelihood is close to quadratic over that range.
They diverge when \\(n\\) is small or \\(p\\) is near 0 or 1. The Wilson interval is then much better behaved than the
Wald interval, which can extend outside \\([0,1]\\) and undercovers.
"""),
]

EXERCISES["06-bayesian-inference.md"] = [
dict(title="Beta–binomial updating: response rates in a charity mail campaign",
data="`charity` — 4,268 people who received a charity's mail appeal; `respond` = 1 if they gave.",
task="""
1. With a uniform \\(\\text{Beta}(1,1)\\) prior, compute the posterior for the response rate after the first 50, 500 and all 4,268 letters (in the data's order). Report posterior means and 95% credible intervals.
2. Repeat with an informative prior \\(\\text{Beta}(2,18)\\) (prior mean 10%, worth 20 letters). When does the prior stop mattering?
3. Compute the posterior predictive probability that at least 400 of the next 1,000 letters get a response.
""",
starter="""
import numpy as np, wooldridge as wd
from scipy import stats
r = wd.data("charity")["respond"].to_numpy()
# TODO: posterior Beta(a + k, b + n - k); stats.beta.ppf([0.025, 0.975], ...); betabinom for prediction
""",
solution="""
import numpy as np, wooldridge as wd
from scipy import stats
r = wd.data("charity")["respond"].to_numpy()
for a, b, lab in ((1, 1, "uniform Beta(1,1)"), (2, 18, "informative Beta(2,18)")):
    print(lab)
    for n in (50, 500, len(r)):
        k = r[:n].sum(); post = stats.beta(a + k, b + n - k)
        lo, hi = post.ppf([0.025, 0.975])
        print(f"   n = {n:5d}, responses {k:5d}: posterior mean {post.mean():.4f}, 95% interval [{lo:.4f}, {hi:.4f}]")
k, n = r.sum(), len(r)
pred = stats.betabinom(1000, 1 + k, 1 + n - k)
print(f"P(at least 400 of next 1000 respond | data, uniform prior) = {pred.sf(399):.4f}")
""",
notes="""After 50 letters, the posterior is wide and depends on the prior: the informative prior pulls the estimate toward 10%. By
500 letters, the two posteriors differ by about one percentage point; with all 4,268, they are practically identical —
the data swamp a prior worth 20 observations. The first 50 letters are not representative: only 14% responded, against
40% overall, so the data are sorted in some way. That is a warning that sequential updating assumes exchangeability. The
posterior predictive probability of at least 400 responses in 1,000 is about one half, because 400 is almost exactly the
expected number.
"""),

dict(title="A random-walk Metropolis sampler for a lognormal wage model",
data="`wage1` — hourly wages.",
task="""
Model \\(\\log wage_i\\sim N(\\mu,\\sigma^2)\\) with priors \\(\\mu\\sim N(0,10^2)\\) and \\(\\log\\sigma\\sim N(0,1)\\).

1. Write the log posterior and a random-walk Metropolis sampler on \\((\\mu,\\log\\sigma)\\). Run 4 chains of 5,000 draws from dispersed starting values; discard the first 1,000 of each.
2. Report the acceptance rate, the posterior means and 95% credible intervals, and the split-\\(\\widehat R\\) for each parameter.
3. Compare with the frequentist estimates. Then report the posterior for the **median** wage \\(e^{\\mu}\\) and the **mean** wage \\(e^{\\mu+\\sigma^2/2}\\).
""",
starter="""
import numpy as np, wooldridge as wd
from scipy import stats
y = np.log(wd.data("wage1")["wage"].to_numpy())
def logpost(m, ls):
    return stats.norm.logpdf(y, m, np.exp(ls)).sum() + stats.norm.logpdf(m, 0, 10) + stats.norm.logpdf(ls, 0, 1)
# TODO: chains; proposal theta + 0.05 * normal; accept with prob exp(new - old)
""",
solution="""
import numpy as np, wooldridge as wd
from scipy import stats
y = np.log(wd.data("wage1")["wage"].to_numpy())
def logpost(th):
    m, ls = th
    return stats.norm.logpdf(y, m, np.exp(ls)).sum() + stats.norm.logpdf(m, 0, 10) + stats.norm.logpdf(ls, 0, 1)
rng = np.random.default_rng(7); chains, acc = [], 0
for start in ([0.0, 1.0], [3.0, -2.0], [-1.0, 0.0], [2.0, 0.5]):
    th = np.array(start); lp = logpost(th); out = []
    for i in range(5000):
        prop = th + rng.normal(0, [0.04, 0.05]); lpp = logpost(prop)
        if np.log(rng.random()) < lpp - lp:
            th, lp = prop, lpp; acc += 1
        out.append(th.copy())
    chains.append(np.array(out[1000:]))
C = np.array(chains)                                   # 4 x 4000 x 2
print(f"acceptance rate {acc / 20000:.3f}")
def split_rhat(x):
    h = x.shape[1] // 2; s = np.vstack([x[:, :h], x[:, h:]])
    W = s.var(1, ddof=1).mean(); B = h * s.mean(1).var(ddof=1)
    return np.sqrt(((h - 1) / h * W + B / h) / W)
for j, name in enumerate(("mu", "log sigma")):
    d = C[:, :, j].ravel()
    print(f"{name:9s}: mean {d.mean():.4f}, 95% [{np.percentile(d, 2.5):.4f}, {np.percentile(d, 97.5):.4f}], split R-hat {split_rhat(C[:, :, j]):.4f}")
print(f"frequentist: mean log wage {y.mean():.4f} (se {y.std(ddof=1)/np.sqrt(len(y)):.4f}), log sd {np.log(y.std(ddof=0)):.4f}")
mu, sig = C[:, :, 0].ravel(), np.exp(C[:, :, 1].ravel())
for lab, v in (("median wage e^mu", np.exp(mu)), ("mean wage e^(mu+s^2/2)", np.exp(mu + sig**2 / 2))):
    print(f"{lab:23s}: posterior mean ${v.mean():.2f}, 95% [{np.percentile(v, 2.5):.2f}, {np.percentile(v, 97.5):.2f}]")
""",
notes="""The acceptance rate of about 36% is in the efficient range for a random-walk sampler. All four chains, from very
different starting points, converge to the same distribution: split-\\(\\widehat R\\) is about 1.00. With 526 observations
and weak priors, the posterior means equal the frequentist estimates to two or three decimals. The posterior of a
**function** of parameters is immediate from the draws: the median wage is about \\$5.07 and the mean wage about \\$5.84,
reflecting the right skew of a lognormal.
"""),
]

EXERCISES["07-experimental-design.md"] = [
dict(title="Checking a randomized experiment: balance and randomization inference",
data="`jtrain2` — NSW experiment.",
task="""
1. Build a balance table of pre-treatment covariates (`age`, `educ`, `black`, `hisp`, `married`, `nodegree`, `re74`, `re75`) by treatment arm, with standardized differences.
2. Run an omnibus test of balance: regress treatment on all covariates and test that all slopes are zero.
3. Compute a **randomization-inference** p-value for the effect on `re78` using the difference in means as the statistic (5,000 re-randomizations that keep 185 treated).
""",
starter="""
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("jtrain2")
X = ["age", "educ", "black", "hisp", "married", "nodegree", "re74", "re75"]
# TODO: means by arm, standardized difference (m1 - m0) / sqrt((v1 + v0)/2); smf.ols("train ~ ...").f_test
""",
solution="""
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("jtrain2")
X = ["age", "educ", "black", "hisp", "married", "nodegree", "re74", "re75"]
g = d.groupby("train")[X]
tab = pd.DataFrame({"control": g.mean().loc[0], "treated": g.mean().loc[1]})
tab["std diff"] = (tab["treated"] - tab["control"]) / np.sqrt((g.var().loc[0] + g.var().loc[1]) / 2)
print(tab.round(3).to_string())
r = smf.ols("train ~ " + " + ".join(X), data=d).fit(cov_type="HC1")
w = r.wald_test(", ".join(f"{x} = 0" for x in X), scalar=True)
print(f"\\nomnibus balance test: chi2({len(X)}) = {w.statistic:.2f}, p = {w.pvalue:.3f}")
y, t = d["re78"].to_numpy(), d["train"].to_numpy().astype(bool)
obs = y[t].mean() - y[~t].mean(); rng = np.random.default_rng(11); n1 = t.sum(); hits = 0
for _ in range(5000):
    s = np.zeros(len(y), bool); s[rng.choice(len(y), n1, replace=False)] = True
    hits += abs(y[s].mean() - y[~s].mean()) >= abs(obs)
print(f"randomization-inference p-value for re78: {hits/5000:.4f}  (observed difference {obs:.3f})")
""",
notes="""Most standardized differences are small, but `nodegree` differs by 0.30 SD and `hisp` by 0.18, and the omnibus test
rejects balance at 5% (p ≈ 0.02). In a genuinely randomized experiment, imbalance on some covariate happens by chance;
it is a reason to adjust for these covariates for precision, not evidence that randomization failed. The randomization
inference p-value (0.004) is valid by design, with no distributional assumptions.
"""),

dict(title="Does blocking help? A design simulation using real outcomes",
data="`jtrain2` controls (260 men) — their actual 1978 earnings serve as untreated potential outcomes.",
task="""
Use the 260 control men as a "population" with known \\(Y(0)\\). Add a constant treatment effect of 1 (thousand dollars) to get \\(Y(1)\\).

1. Simulate 5,000 **completely randomized** experiments (130 treated) and record the difference-in-means estimate.
2. Simulate 5,000 **blocked** experiments, randomizing within blocks defined by 1975 earnings (zero vs positive) and `nodegree`, and estimate the effect as the block-size-weighted average of within-block differences.
3. Compare the bias and standard deviation of the two designs. Why does blocking help only if the blocking variables predict the outcome?
""",
starter="""
import numpy as np, wooldridge as wd
c = wd.data("jtrain2").query("train == 0").reset_index(drop=True)
y0 = c["re78"].to_numpy(); y1 = y0 + 1.0
block = (c["re75"] > 0).astype(int) * 2 + c["nodegree"]
# TODO: simulate complete and blocked randomization
""",
solution="""
import numpy as np, wooldridge as wd
c = wd.data("jtrain2").query("train == 0").reset_index(drop=True)
y0 = c["re78"].to_numpy(); y1 = y0 + 1.0; n = len(y0)
block = ((c["re75"] > 0).astype(int) * 2 + c["nodegree"]).to_numpy()
rng = np.random.default_rng(5); cr, bl = [], []
for _ in range(5000):
    t = np.zeros(n, bool); t[rng.choice(n, n // 2, replace=False)] = True
    cr.append(y1[t].mean() - y0[~t].mean())
    est = 0.0
    for b in np.unique(block):
        idx = np.where(block == b)[0]; tb = np.zeros(len(idx), bool); tb[rng.choice(len(idx), len(idx) // 2, replace=False)] = True
        est += len(idx) / n * (y1[idx][tb].mean() - y0[idx][~tb].mean())
    bl.append(est)
for lab, v in (("complete randomization", cr), ("blocked randomization", bl)):
    print(f"{lab:23s}: mean estimate {np.mean(v):.3f} (true 1.000), SD {np.std(v):.3f}")
print(f"variance reduction from blocking: {1 - np.var(bl)/np.var(cr):.1%}")
print("block sizes:", np.bincount(block), " block means of Y(0):", [round(y0[block == b].mean(), 2) for b in range(4)])
""",
notes="""Both designs are unbiased. Here blocking does **not** reduce variance — in this simulation it is even slightly worse —
because the blocking variables barely predict 1978 earnings (block means range from 4.2 to 5.3, while the within-block SD
is about 5.5). Blocking pays off only when blocks are homogeneous in the outcome; blocking on weak predictors costs a
little precision through smaller within-block samples. In a real design, choose blocks by their predictive power for the
outcome, for example from pilot data.
"""),
]

EXERCISES["08-anova-multiple-comparisons.md"] = [
dict(title="One-way ANOVA and Tukey comparisons: wages by occupation",
data="`wage1` — CPS 1976, with indicators `profocc` (professional), `clerocc` (clerical), `servocc` (service); the rest are grouped as \"other\".",
task="""
1. Build an `occ` factor with four levels and compute group means and SDs of `lwage`.
2. Run a one-way ANOVA (\\(F\\) test). Report the ANOVA table and \\(\\eta^2\\).
3. Check equal variances (Levene). Run Welch's ANOVA as a robust alternative.
4. Run Tukey's HSD to see which pairs differ, controlling the family-wise error rate.
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf, statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
d = wd.data("wage1")
d["occ"] = np.select([d.profocc == 1, d.clerocc == 1, d.servocc == 1], ["professional", "clerical", "service"], "other")
# TODO: sm.stats.anova_lm(smf.ols("lwage ~ C(occ)", d).fit()); scipy.stats.levene; pairwise_tukeyhsd
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway
d = wd.data("wage1")
d["occ"] = np.select([d.profocc == 1, d.clerocc == 1, d.servocc == 1], ["professional", "clerical", "service"], "other")
print(d.groupby("occ")["lwage"].agg(["size", "mean", "std"]).round(3).to_string())
a = sm.stats.anova_lm(smf.ols("lwage ~ C(occ)", data=d).fit())
print(a.round(4).to_string()); print(f"eta-squared = {a['sum_sq'].iloc[0] / a['sum_sq'].sum():.3f}")
lev = stats.levene(*[g["lwage"] for _, g in d.groupby("occ")])
wel = anova_oneway(d["lwage"], d["occ"], use_var="unequal")
print(f"Levene: W = {lev.statistic:.2f}, p = {lev.pvalue:.4f};  Welch ANOVA: F = {wel.statistic:.2f}, p = {wel.pvalue:.2e}")
print(pairwise_tukeyhsd(d["lwage"], d["occ"]).summary())
""",
notes="""Occupation explains about 24% of the variance of log wages (\\(\\eta^2\\)), and the \\(F\\) test rejects equal means
overwhelmingly. Levene's test rejects equal variances — professional wages are more dispersed — so the Welch ANOVA is the
safer test; it gives the same conclusion. Tukey's HSD shows that every pair differs except clerical versus "other". The
professional premium over service work is about 0.73 log points, roughly double.
"""),

dict(title="Two-way ANOVA with interaction: gender, marriage and wages",
data="`wage1`.",
task="""
1. Fit the two-way model `lwage ~ female * married` and produce a type-II ANOVA table.
2. Interpret the interaction: compute the four cell means and the "marriage premium" for men and for women.
3. Because the design is unbalanced, explain why type-I (sequential) and type-II sums of squares differ.
""",
starter="""
import wooldridge as wd, statsmodels.formula.api as smf, statsmodels.api as sm
d = wd.data("wage1")
m = smf.ols("lwage ~ C(female) * C(married)", data=d).fit()
# TODO: sm.stats.anova_lm(m, typ=2) and typ=1; cell means with groupby
""",
solution="""
import wooldridge as wd, statsmodels.formula.api as smf, statsmodels.api as sm
d = wd.data("wage1")
m = smf.ols("lwage ~ C(female) * C(married)", data=d).fit()
print("type II:\\n", sm.stats.anova_lm(m, typ=2).round(4).to_string())
print("type I (female first):\\n", sm.stats.anova_lm(m, typ=1).round(4).to_string())
cm = d.groupby(["female", "married"])["lwage"].agg(["mean", "size"]).round(3)
print(cm.to_string())
mu = d.groupby(["female", "married"])["lwage"].mean()
print(f"marriage premium (log points): men {mu[(0,1)] - mu[(0,0)]:.3f}, women {mu[(1,1)] - mu[(1,0)]:.3f}")
""",
notes="""The interaction is highly significant: marriage is associated with a 0.43 log-point (about 53%) wage premium for men but
almost none for women (0.05). Because the cells are unbalanced (86 to 188 observations), type-I sums of squares depend on
the order of terms: the sum of squares for `female` is larger when it enters first, because it absorbs variation shared
with `married`. Type-II sums of squares test each main effect adjusted for the other. These are associations — the male
marriage premium partly reflects selection of higher earners into marriage.
"""),
]

EXERCISES["09-regression-glm.md"] = [
dict(title="Logistic regression: married women's labor force participation",
data="`mroz` — 753 married women, PSID 1975; `inlf` = in the labor force.",
task="""
1. Fit a logit of `inlf` on `nwifeinc`, `educ`, `exper`, `expersq`, `age`, `kidslt6`, `kidsge6`.
2. Report odds ratios with 95% intervals.
3. Compute average marginal effects, and compare with a linear probability model.
4. What is the predicted probability of participation for a 30-year-old woman with 12 years of education, 10 years of experience, no young children, one older child and non-wife income of 20?
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("mroz")
f = "inlf ~ nwifeinc + educ + exper + expersq + age + kidslt6 + kidsge6"
lg = smf.logit(f, data=d).fit()
# TODO: np.exp(lg.conf_int()); lg.get_margeff().summary(); smf.ols(f, d).fit(cov_type="HC1")
""",
solution="""
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("mroz")
f = "inlf ~ nwifeinc + educ + exper + expersq + age + kidslt6 + kidsge6"
lg = smf.logit(f, data=d).fit(disp=0); lpm = smf.ols(f, data=d).fit(cov_type="HC1")
ci = np.exp(lg.conf_int())
me = lg.get_margeff().summary_frame()
print(pd.DataFrame({"odds ratio": np.exp(lg.params), "OR 2.5%": ci[0], "OR 97.5%": ci[1],
                    "logit AME": me["dy/dx"], "LPM coef": lpm.params}).drop("Intercept").round(4).to_string())
new = pd.DataFrame({"nwifeinc": [20], "educ": [12], "exper": [10], "expersq": [100], "age": [30], "kidslt6": [0], "kidsge6": [1]})
print(f"\\npredicted participation: logit {lg.predict(new).iloc[0]:.3f}, LPM {lpm.predict(new).iloc[0]:.3f}")
print(f"LPM predictions outside [0,1]: {np.mean((lpm.fittedvalues < 0) | (lpm.fittedvalues > 1)):.3f};  pseudo R2 (McFadden) {lg.prsquared:.3f}")
""",
notes="""A child under six reduces the odds of participation by about 76% (odds ratio 0.24) and the probability by about 26
percentage points on average. Logit average marginal effects and LPM coefficients are very close — typical when
probabilities are not near 0 or 1. The LPM predicts probabilities outside \\([0,1]\\) for 4% of women, which the logit avoids.
The predicted participation probability for the described woman is about 0.86.
"""),

dict(title="Count data: Poisson, overdispersion and negative binomial for arrests",
data="`crime1` — 2,725 young men; `narr86` = arrests in 1986.",
task="""
1. Fit a Poisson regression of `narr86` on `pcnv` (proportion of prior arrests leading to conviction), `avgsen`, `tottime`, `ptime86`, `qemp86`, `inc86`, `black`, `hispan`, `born60`.
2. Test for overdispersion (Cameron–Trivedi auxiliary regression).
3. Fit a negative binomial model and compare coefficients and standard errors. Report Poisson standard errors that are robust to overdispersion as well.
4. Interpret the coefficient on `black` as an incidence-rate ratio.
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("crime1")
f = "narr86 ~ pcnv + avgsen + tottime + ptime86 + qemp86 + inc86 + black + hispan + born60"
po = smf.poisson(f, data=d).fit()
# TODO: overdispersion test: regress ((y-mu)^2 - y)/mu on mu without constant; smf.negativebinomial(f, d)
""",
solution="""
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("crime1")
f = "narr86 ~ pcnv + avgsen + tottime + ptime86 + qemp86 + inc86 + black + hispan + born60"
po = smf.poisson(f, data=d).fit(disp=0); por = smf.poisson(f, data=d).fit(disp=0, cov_type="HC0")
mu = po.predict(); y = d["narr86"]
ct = smf.ols("aux ~ mu - 1", data=pd.DataFrame({"aux": ((y - mu)**2 - y) / mu, "mu": mu})).fit()
print(f"overdispersion test: alpha = {ct.params['mu']:.3f} (t = {ct.tvalues['mu']:.2f});  var/mean of y = {y.var()/y.mean():.2f}")
nb = smf.negativebinomial(f, data=d).fit(disp=0)
cols = ["pcnv", "ptime86", "qemp86", "inc86", "black", "hispan"]
print(pd.DataFrame({"Poisson": po.params[cols], "se": po.bse[cols], "robust se": por.bse[cols],
                    "NegBin": nb.params[cols], "NB se": nb.bse[cols]}).round(4).to_string())
print(f"NB dispersion alpha = {nb.params['alpha']:.3f};  IRR for black (Poisson) = {np.exp(po.params['black']):.3f}")
""",
notes="""The overdispersion test rejects strongly. The Poisson point estimates remain consistent if the conditional mean is
correctly specified, but the conventional Poisson standard errors are too small; the robust (sandwich) standard errors are
15–35% larger and close to the negative binomial ones. NB coefficients are similar to Poisson. The incidence-rate ratio
for `black` is about 1.9: black men in this sample had nearly twice the expected number of arrests, other covariates
held fixed — a descriptive association that reflects policing as well as behavior.
"""),
]

EXERCISES["10-nonparametric.md"] = [
dict(title="Rank tests and distributional comparisons for skewed earnings",
data="`jtrain2` — NSW experiment.",
task="""
1. Compare 1978 earnings of trainees and controls with the Mann–Whitney (Wilcoxon rank-sum) test and the two-sample Kolmogorov–Smirnov test.
2. Compare the quantiles of the two distributions at the 25th, 50th, 75th and 90th percentiles. Where is the training effect concentrated?
3. Explain why the Mann–Whitney test is not a test of equal means or medians in general.
""",
starter="""
import numpy as np, wooldridge as wd
from scipy import stats
d = wd.data("jtrain2"); y1, y0 = d.loc[d.train == 1, "re78"], d.loc[d.train == 0, "re78"]
# TODO: stats.mannwhitneyu, stats.ks_2samp, np.quantile
""",
solution="""
import numpy as np, wooldridge as wd
from scipy import stats
d = wd.data("jtrain2"); y1, y0 = d.loc[d.train == 1, "re78"].to_numpy(), d.loc[d.train == 0, "re78"].to_numpy()
mw = stats.mannwhitneyu(y1, y0, alternative="two-sided"); ks = stats.ks_2samp(y1, y0)
print(f"Mann-Whitney U = {mw.statistic:.0f}, p = {mw.pvalue:.4f};  P(Y1 > Y0) estimate = {mw.statistic/(len(y1)*len(y0)):.3f}")
print(f"Kolmogorov-Smirnov D = {ks.statistic:.3f}, p = {ks.pvalue:.4f}")
q = [0.25, 0.5, 0.75, 0.9]
print("quantile:       ", q)
print("treated:        ", np.quantile(y1, q).round(2))
print("control:        ", np.quantile(y0, q).round(2))
print("difference:     ", (np.quantile(y1, q) - np.quantile(y0, q)).round(2))
""",
notes="""Both tests reject at 5%. The Mann–Whitney statistic estimates \\(P(Y_1>Y_0)\\approx0.57\\): a randomly chosen trainee
out-earns a randomly chosen control 57% of the time. The quantile differences grow along the distribution, from about
\\$500 at the 25th percentile to \\$3,200 at the 90th, so training raised earnings throughout, most at the top. Mann–Whitney
tests \\(P(Y_1>Y_0)=1/2\\), which coincides with equal medians or means only under extra assumptions such as a pure location
shift.
"""),

dict(title="Kernel density and a bootstrap interval for a median",
data="`wage1` — hourly wages by gender.",
task="""
1. Estimate kernel densities of `lwage` for men and women (Gaussian kernel, Silverman bandwidth) and report the modes.
2. Compute the difference in median wages (women − men) with a percentile bootstrap 95% interval (2,000 draws, resampling within each group).
3. Compare with a 95% interval for the difference in means. Which is more informative for a skewed variable, and why?
""",
starter="""
import numpy as np, wooldridge as wd
from scipy import stats
d = wd.data("wage1")
# TODO: stats.gaussian_kde(..., bw_method="silverman"); bootstrap medians
""",
solution="""
import numpy as np, wooldridge as wd
from scipy import stats
d = wd.data("wage1"); grid = np.linspace(-0.5, 3.5, 801)
for g, lab in ((0, "men"), (1, "women")):
    k = stats.gaussian_kde(d.loc[d.female == g, "lwage"], bw_method="silverman")
    print(f"{lab:5s}: KDE mode of log wage {grid[np.argmax(k(grid))]:.3f} (= ${np.exp(grid[np.argmax(k(grid))]):.2f}), bandwidth factor {k.factor:.3f}")
m, f = d.loc[d.female == 0, "wage"].to_numpy(), d.loc[d.female == 1, "wage"].to_numpy()
rng = np.random.default_rng(2)
bm = np.array([np.median(rng.choice(f, len(f))) - np.median(rng.choice(m, len(m))) for _ in range(2000)])
print(f"median difference (women - men) ${np.median(f) - np.median(m):.2f}, bootstrap 95% CI [{np.percentile(bm, 2.5):.2f}, {np.percentile(bm, 97.5):.2f}]")
t = stats.ttest_ind(f, m, equal_var=False); ci = t.confidence_interval()
print(f"mean difference   (women - men) ${f.mean() - m.mean():.2f}, Welch 95% CI [{ci.low:.2f}, {ci.high:.2f}]")
""",
notes="""The density of log wages peaks around \\$5.56 for men and \\$3.29 for women. The median gap is \\$2.25 with a bootstrap
interval of roughly \\$1.74 to \\$2.74; the mean gap is a little larger (\\$2.51), because the right tail is thicker for men.
For skewed variables, the median describes the typical worker and is robust to a few very high wages; the mean matters for
totals (for example, the aggregate wage bill). Report both and say which question each answers.
"""),
]

EXERCISES["11-power-sample-size.md"] = [
dict(title="Planning a job-training trial from pilot data",
data="`jtrain2` — use the NSW control group as pilot data on the outcome's variability.",
task="""
1. Using the control-group SD of `re78`, compute the sample size per arm needed to detect an effect of \\$1,000 with 80% power at \\(\\alpha=0.05\\) (two-sided), analytically with `statsmodels`.
2. Compute the minimum detectable effect (MDE) for the actual NSW design (185 treated, 260 controls).
3. Simulate the power of the Welch \\(t\\)-test for a \\$1,800 effect in the actual design by resampling control outcomes (2,000 replications). Compare with the analytic power. Why might they differ?
""",
starter="""
import numpy as np, wooldridge as wd
from statsmodels.stats.power import TTestIndPower
c = wd.data("jtrain2").query("train == 0")["re78"].to_numpy(); sd = c.std(ddof=1)
pw = TTestIndPower()
# TODO: pw.solve_power(effect_size=1/sd, power=0.8, alpha=0.05); MDE with nobs1=185, ratio=260/185
""",
solution="""
import numpy as np, wooldridge as wd
from scipy import stats
from statsmodels.stats.power import TTestIndPower
c = wd.data("jtrain2").query("train == 0")["re78"].to_numpy(); sd = c.std(ddof=1)
pw = TTestIndPower()
n = pw.solve_power(effect_size=1.0 / sd, power=0.8, alpha=0.05)
print(f"control SD of re78 = {sd:.3f} thousand;  n per arm for a $1,000 effect at 80% power: {np.ceil(n):.0f}")
es = pw.solve_power(nobs1=185, ratio=260/185, power=0.8, alpha=0.05)
print(f"MDE for the NSW design (185 vs 260): {es*sd:.3f} thousand dollars ({es:.3f} SD)")
print(f"analytic power for a $1,800 effect: {pw.power(effect_size=1.8/sd, nobs1=185, ratio=260/185, alpha=0.05):.3f}")
rng = np.random.default_rng(9); rej = 0
for _ in range(2000):
    y0 = rng.choice(c, 260); y1 = rng.choice(c, 185) + 1.8
    rej += stats.ttest_ind(y1, y0, equal_var=False).pvalue < 0.05
print(f"simulated power for a $1,800 effect: {rej/2000:.3f} (MC se {np.sqrt(rej/2000*(1-rej/2000)/2000):.3f})")
""",
notes="""Earnings are very noisy (SD of about \\$5,500), so detecting a \\$1,000 effect needs about 474 men per arm. The actual NSW
design could detect effects of about \\$1,500 (0.27 SD) with 80% power, and had about 93% power for the \\$1,800 effect it
found. The simulation reproduces the analytic power closely here. Simulated power is more trustworthy in general, because
it uses the real, skewed, zero-inflated outcome distribution instead of a normal approximation.
"""),

dict(title="Power under clustering: the design effect",
data="`wagepan` — 545 workers observed for 8 years; outcome `lwage`.",
task="""
Suppose a wage subsidy is randomized **by worker** and wages are measured in all 8 years.

1. Estimate the intra-class correlation (ICC) of `lwage` within workers using a one-way random-effects ANOVA.
2. Compute the design effect \\(1+(m-1)\\rho\\) with \\(m=8\\) and the effective sample size.
3. How many workers are needed to detect a 5% wage effect with 80% power, accounting for clustering? Compare with the naive calculation that treats person-years as independent.
""",
starter="""
import numpy as np, wooldridge as wd
d = wd.data("wagepan")
g = d.groupby("nr")["lwage"]; m = 8; k = d["nr"].nunique()
# TODO: MSB = m * var(group means) ; MSW = pooled within variance; ICC = (MSB - MSW) / (MSB + (m-1) MSW)
""",
solution="""
import numpy as np, wooldridge as wd
from statsmodels.stats.power import TTestIndPower
d = wd.data("wagepan"); m = 8; k = d["nr"].nunique()
gm = d.groupby("nr")["lwage"].mean()
msb = m * ((gm - d["lwage"].mean())**2).sum() / (k - 1)
msw = ((d["lwage"] - d.groupby("nr")["lwage"].transform("mean"))**2).sum() / (k * (m - 1))
icc = (msb - msw) / (msb + (m - 1) * msw)
deff = 1 + (m - 1) * icc; sd = d["lwage"].std()
print(f"ICC of log wages within workers = {icc:.3f};  design effect = {deff:.2f};  effective n = {len(d)/deff:.0f} of {len(d)} person-years")
n_naive = TTestIndPower().solve_power(effect_size=0.05 / sd, power=0.8, alpha=0.05)
print(f"naive: {np.ceil(n_naive):.0f} person-years per arm = {np.ceil(n_naive/m):.0f} workers per arm")
print(f"with clustering: {np.ceil(n_naive*deff/m):.0f} workers per arm")
""",
notes="""Almost half of the variance in log wages is between workers (ICC 0.47), so 8 observations per worker carry the information
of only about 2.3 independent ones: the design effect is 4.3. Ignoring clustering would suggest 223 workers per arm;
accounting for it requires about 959 per arm — more than four times as many. When treatment is assigned at the worker level,
repeated measurements of the same worker add much less information than new workers.
"""),
]

EXERCISES["12-mixed-models.md"] = [
dict(title="Random intercepts and slopes for wage growth",
data="`wagepan` — 545 men, 1980–1987.",
task="""
1. Fit a linear mixed model `lwage ~ exper + educ + black + hisp + union` with a random intercept for each worker (REML).
2. Add a random slope on `exper`. Report the variance components and the correlation between intercepts and slopes.
3. Compare the two models by AIC (refit with ML). Interpret the random-slope variance: how much do individual wage-growth rates differ?
4. Compare the fixed-effect estimate of `union` with the FE (within) estimator.
""",
starter="""
import wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("wagepan")
f = "lwage ~ exper + educ + black + hisp + union"
ri = smf.mixedlm(f, d, groups=d["nr"]).fit(reml=True)
# TODO: rs = smf.mixedlm(f, d, groups=d["nr"], re_formula="~exper").fit()
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("wagepan")
f = "lwage ~ exper + educ + black + hisp + union"
ri = smf.mixedlm(f, d, groups=d["nr"]).fit(reml=True)
rs = smf.mixedlm(f, d, groups=d["nr"], re_formula="~exper").fit(reml=True, method="lbfgs")
icc = ri.cov_re.iloc[0, 0] / (ri.cov_re.iloc[0, 0] + ri.scale)
print(f"random intercept: sd(intercept) = {np.sqrt(ri.cov_re.iloc[0,0]):.3f}, residual sd = {np.sqrt(ri.scale):.3f}, ICC = {icc:.3f}")
V = rs.cov_re.to_numpy()
print(f"random slope:     sd(intercept) = {np.sqrt(V[0,0]):.3f}, sd(exper slope) = {np.sqrt(V[1,1]):.4f}, corr = {V[0,1]/np.sqrt(V[0,0]*V[1,1]):.3f}")
print(f"  mean exper slope {rs.fe_params['exper']:.4f}; middle 95% of individual slopes [{rs.fe_params['exper'] - 1.96*np.sqrt(V[1,1]):.4f}, {rs.fe_params['exper'] + 1.96*np.sqrt(V[1,1]):.4f}]")
a1 = smf.mixedlm(f, d, groups=d["nr"]).fit(reml=False).aic
a2 = smf.mixedlm(f, d, groups=d["nr"], re_formula="~exper").fit(reml=False, method="lbfgs").aic
print(f"AIC (ML): random intercept {a1:.1f}, random slope {a2:.1f}")
fe = smf.ols("lwage ~ exper + union + C(nr)", data=d).fit()
print(f"union: mixed model {ri.fe_params['union']:.4f}, within (FE) {fe.params['union']:.4f}")
""",
notes="""Almost half of the variance is between workers (ICC 0.47). Allowing individual experience slopes improves the fit
substantially (AIC falls by about 170). The SD of individual slopes (0.054) is almost as large as the mean slope (0.063):
the middle 95% of workers have wage growth between about \\(-4\\%\\) and \\(+17\\%\\) per year of experience. The negative
intercept–slope correlation means workers who start low catch up. The mixed-model union effect (0.11) lies between pooled
OLS and FE (0.09), because the random-intercept model assumes worker effects uncorrelated with union status — the
assumption the Mundlak test rejected in Econometrics Lecture 7.
"""),

dict(title="Shrinkage: partial pooling of school means",
data="`meap93` — 408 Michigan high schools; `math10` = % of 10th graders passing a math test. We randomly divide the schools into 20 groups of very different sizes (4 to 78 schools) to mimic clusters with unequal information.",
task="""
Treat each group as a "cluster" whose mean pass rate we want to estimate. Because groups are formed at random, every group's true mean is the same — so any spread in the raw group means is pure noise.

1. Compute the raw group means and their sizes.
2. Fit a random-intercept model `math10 ~ 1` with groups and compute the **empirical Bayes (BLUP)** group means.
3. Show that small groups are shrunk more toward the grand mean, and compute each group's shrinkage factor \\(\\tau^2/(\\tau^2+\\sigma^2/n_j)\\).
""",
starter="""
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("meap93")
d = d.sample(frac=1, random_state=4).reset_index(drop=True)
sizes = np.r_[np.repeat(4, 8), np.repeat(10, 6), np.repeat(40, 4), np.repeat(78, 2)]
d["grp"] = np.repeat(np.arange(20), sizes)
# TODO: smf.mixedlm("math10 ~ 1", d, groups=d["grp"]).fit(); random_effects
""",
solution="""
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("meap93")
d = d.sample(frac=1, random_state=4).reset_index(drop=True)
sizes = np.r_[np.repeat(4, 8), np.repeat(10, 6), np.repeat(40, 4), np.repeat(78, 2)]   # unequal cluster sizes, total 408
d["grp"] = np.repeat(np.arange(20), sizes)
m = smf.mixedlm("math10 ~ 1", d, groups=d["grp"]).fit(reml=True)
tau2, s2, mu = m.cov_re.iloc[0, 0], m.scale, m.fe_params["Intercept"]
g = d.groupby("grp")["math10"].agg(n="size", raw="mean")
g["blup"] = mu + np.array([m.random_effects[k].iloc[0] for k in g.index])
g["shrink"] = tau2 / (tau2 + s2 / g["n"])
print(f"grand mean {mu:.2f}, between-group sd {np.sqrt(tau2):.2f}, within sd {np.sqrt(s2):.2f}")
print(g.iloc[[0, 1, 2, 8, 9, 14, 15, 18, 19]].round(3).to_string())
""",
notes="""Groups were formed at random, so all true group means equal the grand mean; the spread of raw means is noise. The
between-group SD is estimated at only about 1.5 points against a within-group SD of 10.4. Small groups (4 schools) are
shrunk about 93% of the way to the grand mean — a raw mean of 17.5 becomes 23.8 — while the largest groups keep about 60%
of their deviation. Empirical Bayes estimates are biased toward the mean but have lower mean squared error; this is the
logic behind value-added models for teachers and schools.
"""),
]

EXERCISES["13-multivariate.md"] = [
dict(title="Principal components of the U.S. yield curve",
data="`intqrt` — U.S. quarterly Treasury bill yields at 3, 6 and 12 months (`r3`, `r6`, `r12`).",
task="""
1. Compute the correlation matrix of the three yields.
2. Run PCA on the standardized yields. Report the eigenvalues, the share of variance explained, and the loadings.
3. Interpret the first two components (the "level" and "slope" factors of the term structure).
4. How well does the first component alone reconstruct the 12-month yield?
""",
starter="""
import numpy as np, wooldridge as wd
d = wd.data("intqrt")[["r3", "r6", "r12"]].dropna()
Z = (d - d.mean()) / d.std()
# TODO: eigenvalues/eigenvectors of the correlation matrix (np.linalg.eigh), sorted descending
""",
solution="""
import numpy as np, pandas as pd, wooldridge as wd
d = wd.data("intqrt")[["r3", "r6", "r12"]].dropna()
print(d.corr().round(4).to_string())
Z = ((d - d.mean()) / d.std()).to_numpy()
val, vec = np.linalg.eigh(np.corrcoef(Z, rowvar=False)); o = val.argsort()[::-1]; val, vec = val[o], vec[:, o]
vec = vec * np.sign(vec[0])                               # sign convention: first loading positive
print("\\neigenvalues:", val.round(4), " share explained:", (val / val.sum()).round(4))
print(pd.DataFrame(vec, index=d.columns, columns=["PC1", "PC2", "PC3"]).round(3).to_string())
pc1 = Z @ vec[:, 0]
r12_hat = np.outer(pc1, vec[:, 0])[:, 2] * d["r12"].std() + d["r12"].mean()
print(f"\\nRMSE of 12-month yield reconstructed from PC1 alone: {np.sqrt(np.mean((r12_hat - d['r12'])**2)):.3f} percentage points (yield sd {d['r12'].std():.2f})")
""",
notes="""The three yields are almost perfectly correlated, and the first component explains 99.6% of their variance. Its loadings
are equal across maturities: it is the **level** factor. The second component has opposite signs at short and long
maturities: the **slope** factor. The third, with a hump-shaped loading, is curvature. Reconstructing the 12-month yield from
the level factor alone leaves an error of only 0.22 percentage points against a standard deviation of 3.07. This is
Litterman and Scheinkman's (1991) finding in miniature.
"""),

dict(title="PCA and a crime index across U.S. states",
data="`statecrime` (statsmodels) — 51 U.S. states and DC: violent crime and murder rates, % high-school graduates, % below poverty, % single-parent households, % white, % urban.",
task="""
1. Run PCA on the seven standardized variables. How many components have eigenvalue above one (Kaiser rule)?
2. Interpret PC1 using the loadings.
3. Regress the violent crime rate on the scores of the first two components of the five **non-crime** variables only (principal-components regression). Compare its \\(R^2\\) with the regression on all five covariates, and check how collinear the covariates actually are. What does the comparison teach about PCR?
""",
starter="""
import numpy as np, statsmodels.api as sm
d = sm.datasets.statecrime.load_pandas().data
# TODO: standardize, eigen-decompose the correlation matrix; principal-components regression
""",
solution="""
import numpy as np, pandas as pd, statsmodels.api as sm
d = sm.datasets.statecrime.load_pandas().data
def pca(X):
    Z = (X - X.mean()) / X.std(); val, vec = np.linalg.eigh(Z.corr()); o = val.argsort()[::-1]
    vec = vec[:, o] * np.sign(vec[:, o].sum(0))           # sign convention for readability
    return val[o], pd.DataFrame(vec, index=X.columns), Z.to_numpy() @ vec
val, load, _ = pca(d)
print("eigenvalues:", val.round(3), " -> components with eigenvalue > 1:", int((val > 1).sum()))
print("PC1 loadings:", load[0].round(3).to_dict())
X = d[["hs_grad", "poverty", "single", "white", "urban"]]
v2, l2, S = pca(X)
pcr = sm.OLS(d["violent"], sm.add_constant(S[:, :2])).fit()
full = sm.OLS(d["violent"], sm.add_constant(X)).fit()
print(f"\\nPC regression (2 components of 5 covariates): R2 = {pcr.rsquared:.3f};  full regression on 5 covariates: R2 = {full.rsquared:.3f}")
print(f"condition number of the 5 standardized covariates: {np.sqrt(v2.max()/v2.min()):.1f}")
""",
notes="""Two components have eigenvalues above one. PC1 loads positively on violence, murder, single-parent households and poverty
and negatively on high-school graduation and the white share: a social-disadvantage index. Two components of the five covariates explain about 55% of the variation in violent crime,
against 68% for the regression on all five. The covariates are only mildly collinear (condition number 4.5), so reducing
dimension buys little stability here and costs some fit. More fundamentally, the leading components summarize the
**covariates**, not the **outcome**: PCR can discard a low-variance direction that matters for prediction. Partial least
squares and penalized regression (ridge, lasso) use the outcome when choosing directions. PCR earns its keep when the
number of covariates is large relative to \\(n\\) or the covariates are nearly collinear, as with the yields in the previous
exercise.
"""),
]

EXERCISES["14-reproducible-statistics.md"] = [
dict(title="A reproducible analysis function with data validation",
data="`wage1` and `wage2`.",
task="""
Good statistical code fails loudly on bad data and gives identical results on every run.

1. Write a function `analyze(df)` that (a) validates the input — required columns present, no missing values, wages positive, education between 0 and 20 — and raises an informative error otherwise; (b) fits `log(wage) ~ educ + exper + expersq` with HC1 standard errors; and (c) returns a tidy table.
2. Run it on `wage1`. Then run it on `wage2` (after constructing `expersq`) and compare.
3. Deliberately corrupt one value and show that validation catches it.
""",
starter="""
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
REQUIRED = ["wage", "educ", "exper"]
def analyze(df):
    # TODO: assertions, then fit and return a DataFrame of estimates
    ...
""",
solution="""
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
REQUIRED = ["wage", "educ", "exper"]
def validate(df):
    missing = [c for c in REQUIRED if c not in df]
    if missing: raise ValueError(f"missing columns: {missing}")
    if df[REQUIRED].isna().any().any(): raise ValueError("missing values in required columns")
    if (df["wage"] <= 0).any(): raise ValueError(f"{(df['wage'] <= 0).sum()} non-positive wages")
    if not df["educ"].between(0, 20).all(): raise ValueError("education outside [0, 20]")
def analyze(df, name):
    validate(df)
    d = df.assign(expersq=df["exper"]**2)
    r = smf.ols("np.log(wage) ~ educ + exper + expersq", data=d).fit(cov_type="HC1")
    return pd.DataFrame({"data": name, "n": int(r.nobs), "coef": r.params, "robust se": r.bse}).drop("Intercept")
print(pd.concat([analyze(wd.data("wage1"), "wage1"), analyze(wd.data("wage2"), "wage2")]).round(4).to_string())
bad = wd.data("wage1").copy(); bad.loc[10, "wage"] = -3.0
try:
    analyze(bad, "corrupted")
except ValueError as e:
    print("\\nvalidation caught a problem:", e)
""",
notes="""The function produces the same tidy table on both datasets, and it stops with an informative message when a single wage is
corrupted, instead of silently producing a missing log or a wrong estimate. The return to education is 9.0% in the 1976
CPS and 7.8% in the 1980 NLS men; the experience profile is much flatter in `wage2`, which covers men of a narrow age range.
Validation checks are cheap, and they turn data errors into loud failures rather than wrong results.
"""),

dict(title="A simulation study with a fixed seed: which interval covers the mean of skewed data?",
data="`ceosal1` CEO salaries, treated as a population.",
task="""
Design a small, reproducible simulation study.

1. Using `numpy.random.SeedSequence`, create independent random streams for each sample size \\(n\\in\\{10,30,100\\}\\).
2. For each \\(n\\), draw 2,000 samples and compute three 95% intervals for the population mean: the \\(t\\)-interval, the bootstrap percentile interval and the bootstrap-\\(t\\) interval (199 resamples each).
3. Report coverage with Monte Carlo standard errors, and the average interval width. Run the script twice to confirm identical output.
""",
starter="""
import numpy as np, wooldridge as wd
pop = wd.data("ceosal1")["salary"].to_numpy(float); mu = pop.mean()
streams = [np.random.default_rng(s) for s in np.random.SeedSequence(2026).spawn(3)]
# TODO: loop over n and stream; compute the three intervals; tabulate coverage and width
""",
solution="""
import numpy as np, pandas as pd, wooldridge as wd
from scipy import stats
pop = wd.data("ceosal1")["salary"].to_numpy(float); mu = pop.mean()
streams = [np.random.default_rng(s) for s in np.random.SeedSequence(2026).spawn(3)]
rows = []
for n, rng in zip((10, 30, 100), streams):
    hit = np.zeros(3); width = np.zeros(3); R, B = 2000, 199
    for _ in range(R):
        x = rng.choice(pop, n); m, s = x.mean(), x.std(ddof=1); se = s / np.sqrt(n)
        q = stats.t.ppf(0.975, n - 1); ints = [(m - q * se, m + q * se)]
        bs = x[rng.integers(0, n, (B, n))]; bm = bs.mean(1); bse = bs.std(1, ddof=1) / np.sqrt(n)
        ints.append(tuple(np.percentile(bm, [2.5, 97.5])))
        tstar = (bm - m) / bse; lo, hi = np.percentile(tstar, [97.5, 2.5])
        ints.append((m - lo * se, m - hi * se))
        for j, (a, b) in enumerate(ints):
            hit[j] += a <= mu <= b; width[j] += b - a
    for j, lab in enumerate(("t-interval", "percentile", "bootstrap-t")):
        p = hit[j] / R
        rows.append((n, lab, round(p, 3), round(np.sqrt(p * (1 - p) / R), 3), round(width[j] / R, 1)))
print(pd.DataFrame(rows, columns=["n", "interval", "coverage", "MC se", "avg width"]).to_string(index=False))
""",
notes="""Because each sample size has its own seeded stream, the table is reproduced exactly on every run, and adding a new design
would not change the existing results. With such a skewed population, **all** three intervals undercover. The percentile
bootstrap is worst at small \\(n\\), and the bootstrap-\\(t\\) is best, at the cost of much wider intervals. Even at
\\(n=100\\), none reaches 90%. The Monte Carlo standard errors (about 0.01) show that differences of 3–4 points are real,
not simulation noise.
"""),
]
