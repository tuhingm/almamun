INTRO = """
These exercises use **real data** and run in Python. Install the packages once with
`pip install wooldridge statsmodels linearmodels scipy`. The `wooldridge` package contains the datasets
from Wooldridge's *Introductory Econometrics*; `statsmodels` ships U.S. and Danish macroeconomic series.
Try the starter code first, then open the solution. Every printed result below was produced by running the solution code exactly as shown.
"""

EXERCISES = {}

# ---------------------------------------------------------------- Lecture 1
EXERCISES["01-probability-regression.md"] = [
dict(title="The CEF of wages given schooling, and its best linear approximation",
data="`wage1` — 526 U.S. workers, Current Population Survey 1976 (hourly wage, years of education).",
task="""
1. Estimate the **conditional expectation function** \\(E[wage\\mid educ]\\) nonparametrically, as the mean wage in each education cell.
2. Estimate the **linear projection** of `wage` on `(1, educ)` by OLS.
3. Show numerically that OLS on the individual data equals weighted least squares of the **cell means** on `educ`, with cell sizes as weights. This is the "regression approximates the CEF" result.
4. Where does the linear approximation fit worst? What does that say about the true shape of the CEF?
""",
starter="""
import wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wage1")
cef = df.groupby("educ")["wage"].agg(["mean", "size"])
ols = smf.ols("wage ~ educ", data=df).fit()
# TODO: weighted regression of cef['mean'] on educ with weights cef['size']; compare slopes
""",
solution="""
import wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wage1")

cef = df.groupby("educ")["wage"].agg(mean="mean", n="size").reset_index()
ols = smf.ols("wage ~ educ", data=df).fit()
wls = smf.wls("mean ~ educ", data=cef, weights=cef["n"]).fit()
print(f"OLS on individuals:   intercept {ols.params['Intercept']:.4f}, slope {ols.params['educ']:.4f}")
print(f"WLS on cell means:    intercept {wls.params['Intercept']:.4f}, slope {wls.params['educ']:.4f}")

cef["blp"] = ols.params["Intercept"] + ols.params["educ"] * cef["educ"]
cef["gap"] = cef["mean"] - cef["blp"]
print(cef[cef["n"] >= 10].round(2).to_string(index=False))
""",
notes="""
The two slopes are identical: OLS on the micro data is exactly a weighted regression of the CEF on `educ`
(Angrist and Pischke, 2009, §3.1). The gaps are **positive at the extremes and negative in the middle** (for example,
the CEF is below the line around 9–11 years and above it at 16–18 years). The CEF is **convex** in schooling. The linear
projection is still well defined, but it understates returns at high education. That is one motivation for the log
specification in Lecture 3.
"""),

dict(title="Frisch–Waugh–Lovell and leverage, by hand",
data="`wage2` — 935 men from the National Longitudinal Survey, 1980 (monthly earnings, education, experience, tenure, IQ).",
task="""
1. Estimate \\(\\log(wage)=\\beta_0+\\beta_1 educ+\\beta_2 exper+\\beta_3 tenure+u\\) with matrix algebra (`numpy`, no regression package).
2. Verify FWL: partial `exper` and `tenure` (and the constant) out of both `educ` and `log(wage)`, and regress residual on residual.
3. Compute the leverage values \\(h_{ii}\\), check that they sum to \\(k\\), and list the three highest-leverage workers. Why are they unusual?
""",
starter="""
import numpy as np, wooldridge as wd
df = wd.data("wage2")
y = np.log(df["wage"].to_numpy())
X = np.column_stack([np.ones(len(df)), df[["educ", "exper", "tenure"]].to_numpy()])
# TODO: beta = solve(X'X, X'y); FWL with X1 = [1, exper, tenure]; leverage h = diag(X (X'X)^-1 X')
""",
solution="""
import numpy as np, wooldridge as wd
df = wd.data("wage2")
y = np.log(df["wage"].to_numpy())
X = np.column_stack([np.ones(len(df)), df[["educ", "exper", "tenure"]].to_numpy()])
beta = np.linalg.solve(X.T @ X, X.T @ y)
print("full regression beta:", beta.round(5))

X1 = X[:, [0, 2, 3]]                                     # constant, exper, tenure
resid = lambda z: z - X1 @ np.linalg.lstsq(X1, z, rcond=None)[0]
r_educ, r_y = resid(X[:, 1]), resid(y)
print("FWL slope on educ:   ", round(r_educ @ r_y / (r_educ @ r_educ), 5))

h = np.einsum("ij,ji->i", X, np.linalg.solve(X.T @ X, X.T))   # diag of the hat matrix, without forming n x n
print("sum of leverages:", round(h.sum(), 6), " mean:", round(h.mean(), 5), " max:", round(h.max(), 5))
top = np.argsort(h)[-3:][::-1]
print(df.iloc[top][["wage", "educ", "exper", "tenure"]].assign(leverage=h[top].round(4)).to_string())
""",
notes="""
The FWL slope matches the full-regression coefficient on `educ` to every printed digit. The leverages sum to
\\(k=4\\), as \\(\\operatorname{tr}(P_X)=k\\) requires. The highest-leverage workers have unusual **combinations** of regressors, not extreme wages.
Two of the three report **tenure longer than total experience** (16 years with the employer but 4 years of experience),
which is logically impossible: leverage diagnostics are a quick way to find data errors. Leverage depends on \\(X\\) only,
which is why it enters the HC2/HC3 corrections of Lecture 4.
"""),
]

# ---------------------------------------------------------------- Lecture 2
EXERCISES["02-ols-inference.md"] = [
dict(title="Testing a hedonic price model: classical versus robust inference",
data="`hprice1` — 88 house sales in Boston-area communities, 1990 (price, lot size, square footage, bedrooms).",
task="""
1. Estimate \\(\\log(price)=\\beta_0+\\beta_1\\log(lotsize)+\\beta_2\\log(sqrft)+\\beta_3 bdrms+u\\).
2. Report classical and HC1 standard errors side by side.
3. Test \\(H_0:\\beta_1+\\beta_2=1\\) (a 1% increase in both lot size and floor space raises price by 1%) with a robust Wald test, and build a 95% confidence interval for \\(\\beta_1+\\beta_2\\).
4. Test the joint null \\(\\beta_1=\\beta_3=0\\).
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("hprice1")
m = smf.ols("np.log(price) ~ np.log(lotsize) + np.log(sqrft) + bdrms", data=df)
# TODO: fit with cov_type="nonrobust" and cov_type="HC1"; use .t_test() and .wald_test()
""",
solution="""
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("hprice1")
m = smf.ols("np.log(price) ~ np.log(lotsize) + np.log(sqrft) + bdrms", data=df)
cl, rb = m.fit(), m.fit(cov_type="HC1")
print(pd.DataFrame({"coef": cl.params, "se classical": cl.bse, "se HC1": rb.bse}).round(4))

t = rb.t_test("np.log(lotsize) + np.log(sqrft) = 1")
est = float(np.squeeze(t.effect)); se = float(np.squeeze(t.sd))
print(f"\\nbeta1 + beta2 = {est:.4f} (robust se {se:.4f});  t = {float(np.squeeze(t.tvalue)):.3f}, p = {float(t.pvalue):.4f}")
print(f"95% CI for beta1 + beta2: [{est - 1.96*se:.3f}, {est + 1.96*se:.3f}]")
w = rb.wald_test("np.log(lotsize) = 0, bdrms = 0", scalar=True)
print(f"joint test beta1 = beta3 = 0: chi2(2) = {w.statistic:.3f}, p = {w.pvalue:.4f}")
""",
notes="""
The point estimate of \\(\\beta_1+\\beta_2\\) is about 0.87, but the robust 95% interval runs from about 0.66 to 1.08 and
contains one, so the test does **not** reject proportionality. With 88 houses, the data cannot distinguish a size
elasticity of 0.87 from 1. The combined standard error uses the estimated covariance between the two coefficients,
which a calculation from the two separate standard errors would ignore. Robust standard errors are 8–20% larger than
classical ones here. The joint test that lot size and bedrooms both have no effect is rejected decisively, driven by
lot size.
"""),

dict(title="Delta method versus bootstrap for a turning point",
data="`wage1` — 526 workers, CPS 1976.",
task="""
1. Estimate \\(\\log(wage)=\\beta_0+\\beta_1 educ+\\beta_2 exper+\\beta_3 exper^2+u\\).
2. Compute the experience level at which predicted log wages peak, \\(x^*=-\\beta_2/(2\\beta_3)\\), with a delta-method standard error (robust covariance).
3. Compute a pairs-bootstrap 95% percentile interval (999 draws, seed 42) and compare.
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wage1")
f = "np.log(wage) ~ educ + exper + I(exper**2)"
res = smf.ols(f, data=df).fit(cov_type="HC1")
# TODO: gradient of g(b) = -b2/(2 b3); se = sqrt(g' V g); bootstrap rows with rng.integers
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wage1")
f = "np.log(wage) ~ educ + exper + I(exper**2)"
res = smf.ols(f, data=df).fit(cov_type="HC1")
b2, b3 = res.params["exper"], res.params["I(exper ** 2)"]
xstar = -b2 / (2 * b3)
V = res.cov_params().loc[["exper", "I(exper ** 2)"], ["exper", "I(exper ** 2)"]].to_numpy()
g = np.array([-1 / (2 * b3), b2 / (2 * b3**2)])
se = np.sqrt(g @ V @ g)
print(f"turning point x* = {xstar:.2f} years, delta-method se = {se:.2f}")
print(f"delta-method 95% CI: [{xstar - 1.96*se:.1f}, {xstar + 1.96*se:.1f}]")

rng = np.random.default_rng(42); draws = []
for _ in range(999):
    p = smf.ols(f, data=df.iloc[rng.integers(0, len(df), len(df))]).fit().params
    draws.append(-p["exper"] / (2 * p["I(exper ** 2)"]))
lo, hi = np.percentile(draws, [2.5, 97.5])
print(f"bootstrap 95% CI:    [{lo:.1f}, {hi:.1f}]  (bootstrap se {np.std(draws, ddof=1):.2f})")
print("share of sample with exper above x*:", round((df["exper"] > xstar).mean(), 3))
""",
notes="""
Predicted log wages peak after about 29 years of experience. The bootstrap interval is shifted to the **right**
of the symmetric delta-method interval and is wider, reflecting the ratio form of \\(x^*\\): the bootstrap standard error is
about 15% larger. With a less precise quadratic term, the delta method would be much less reliable (Lecture 2,
Problem 3). About a quarter of the sample lies beyond the estimated peak, so the decline at high experience is
supported by data rather than pure extrapolation — but a quadratic forces the profile to turn down; a more flexible
specification (splines) would test that.
"""),
]

# ---------------------------------------------------------------- Lecture 3
EXERCISES["03-functional-form-ovb.md"] = [
dict(title="Ability bias in the return to schooling, with IQ as a proxy",
data="`wage2` — 935 men, NLS 1980; `IQ` is an IQ test score.",
task="""
1. Estimate the short regression of `log(wage)` on `educ`, `exper`, `tenure`, `married`, `south`, `urban`, `black`.
2. Add `IQ` (the long regression). How much does the return to schooling fall?
3. Verify the omitted-variable bias identity \\(\\widehat\\beta^S_{educ}=\\widehat\\beta_{educ}+\\widehat\\gamma\\,\\widehat\\beta_{IQ}\\) exactly, where \\(\\widehat\\gamma\\) is the coefficient on `educ` in the auxiliary regression of `IQ` on all the short-regression regressors.
4. Express the bias as a percentage of the short-regression estimate.
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wage2")
ctrl = "exper + tenure + married + south + urban + black"
short = smf.ols(f"np.log(wage) ~ educ + {ctrl}", data=df).fit()
# TODO: long regression adds IQ; auxiliary regression IQ ~ educ + ctrl
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wage2")
ctrl = "exper + tenure + married + south + urban + black"
short = smf.ols(f"np.log(wage) ~ educ + {ctrl}", data=df).fit()
long_ = smf.ols(f"np.log(wage) ~ educ + IQ + {ctrl}", data=df).fit()
aux = smf.ols(f"IQ ~ educ + {ctrl}", data=df).fit()
bS, bL, bIQ, gam = short.params["educ"], long_.params["educ"], long_.params["IQ"], aux.params["educ"]
print(f"short-regression return to educ: {bS:.5f}")
print(f"long-regression return to educ:  {bL:.5f}")
print(f"IQ coefficient {bIQ:.5f};  auxiliary slope of IQ on educ {gam:.4f}")
print(f"bL + gamma * bIQ = {bL + gam * bIQ:.5f}   (identity holds: {abs(bL + gam*bIQ - bS) < 1e-10})")
print(f"omitted-variable bias = {gam * bIQ:.5f}, i.e. {100 * gam * bIQ / bS:.1f}% of the short estimate")
""",
notes="""
Each year of schooling is associated with a few more IQ points, and IQ raises wages, so omitting IQ biases the return
to schooling **upward**. The identity holds to machine precision, as Theorem 3.1 says it must. Controlling for IQ lowers
the return from about 6.5% to 5.4% per year — the bias is about one sixth of the short estimate. IQ is only a proxy for ability: if it measures ability with error, part of the bias
remains, and the true ability-adjusted return could be lower still. Remember the offsetting attenuation bias from
measurement error in schooling (Lecture 3, worked example).
"""),

dict(title="Exact percentage effects, interactions and RESET",
data="`wage1` — 526 workers, CPS 1976.",
task="""
1. Estimate `log(wage)` on `female`, `educ`, `exper`, `exper²`, `tenure`. Report the **exact** percentage gender gap \\(100(e^{\\delta}-1)\\) with a delta-method standard error.
2. Add `female × educ`. Does the return to education differ by gender? What is the gender gap at 12 and at 16 years of schooling?
3. Run a RESET test (add \\(\\widehat y^2\\) and \\(\\widehat y^3\\)) for the model of part 1, using robust inference.
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wage1")
base = "np.log(wage) ~ female + educ + exper + I(exper**2) + tenure"
r1 = smf.ols(base, data=df).fit(cov_type="HC1")
# TODO: exact gap 100*(exp(d)-1); interaction model; RESET with fitted values squared and cubed
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wage1")
base = "np.log(wage) ~ female + educ + exper + I(exper**2) + tenure"
r1 = smf.ols(base, data=df).fit(cov_type="HC1")
d, sd = r1.params["female"], r1.bse["female"]
print(f"female coefficient {d:.4f}: approx gap {100*d:.1f}%, exact gap {100*(np.exp(d)-1):.1f}% (se {100*np.exp(d)*sd:.1f} pp)")

r2 = smf.ols(base + " + female:educ", data=df).fit(cov_type="HC1")
print(f"female:educ = {r2.params['female:educ']:.4f} (robust se {r2.bse['female:educ']:.4f}, p = {r2.pvalues['female:educ']:.3f})")
for e in (12, 16):
    t = r2.t_test(f"female + {e}*female:educ = 0")
    print(f"  log gap at educ = {e}: {float(np.squeeze(t.effect)):.4f} (se {float(np.squeeze(t.sd)):.4f})")

df["yh"] = smf.ols(base, data=df).fit().fittedvalues
r3 = smf.ols(base + " + I(yh**2) + I(yh**3)", data=df).fit(cov_type="HC1")
w = r3.wald_test("I(yh ** 2) = 0, I(yh ** 3) = 0", scalar=True)
print(f"RESET (robust): chi2(2) = {w.statistic:.3f}, p = {w.pvalue:.3f}")
""",
notes="""
The exact gap is a few points smaller in absolute value than the approximate one, because \\(e^{\\delta}-1>\\delta\\) for
negative \\(\\delta\\). The interaction is small and statistically insignificant: in 1976 the return to education was
similar for women and men, so the gender gap is roughly constant across schooling levels. The \\(t\\)-test on `female`
alone in the interaction model would be the gap at **zero** years of education — not a meaningful quantity. RESET **rejects** strongly
(p < 0.001): the specification misses some nonlinearity — candidates include a quadratic in tenure or interactions of
experience with education. RESET signals that the conditional mean is misspecified; it does not say how.
"""),
]

# ---------------------------------------------------------------- Lecture 4
EXERCISES["04-heteroskedasticity-robust-inference.md"] = [
dict(title="Heteroskedasticity in house prices: tests, HC0–HC3 and WLS",
data="`hprice1` — 88 house sales.",
task="""
1. Estimate `price` (in levels) on `lotsize`, `sqrft`, `bdrms`. Run the Breusch–Pagan and White tests.
2. Compare classical, HC0, HC1, HC2 and HC3 standard errors.
3. Re-estimate in logs. Does heteroskedasticity remain? What does this suggest about functional form?
4. Estimate by WLS with weights \\(1/sqrft\\) and report **robust** standard errors.
""",
starter="""
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
df = wd.data("hprice1")
lev = smf.ols("price ~ lotsize + sqrft + bdrms", data=df).fit()
# TODO: het_breuschpagan(lev.resid, lev.model.exog); compare lev.get_robustcov_results(cov_type=...)
""",
solution="""
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
df = wd.data("hprice1")
lev = smf.ols("price ~ lotsize + sqrft + bdrms", data=df).fit()
bp, wh = het_breuschpagan(lev.resid, lev.model.exog), het_white(lev.resid, lev.model.exog)
print(f"levels: Breusch-Pagan LM = {bp[0]:.2f} (p = {bp[1]:.4f});  White LM = {wh[0]:.2f} (p = {wh[1]:.4f})")

ses = {"classical": lev.bse}
for c in ("HC0", "HC1", "HC2", "HC3"):
    ses[c] = pd.Series(lev.get_robustcov_results(cov_type=c).bse, index=lev.params.index)
print(pd.DataFrame(ses).round(4))

lg = smf.ols("np.log(price) ~ np.log(lotsize) + np.log(sqrft) + bdrms", data=df).fit()
bp2 = het_breuschpagan(lg.resid, lg.model.exog)
print(f"\\nlogs:   Breusch-Pagan LM = {bp2[0]:.2f} (p = {bp2[1]:.4f})")

w = smf.wls("price ~ lotsize + sqrft + bdrms", data=df, weights=1 / df["sqrft"]).fit(cov_type="HC1")
print("\\nWLS (weights 1/sqrft), robust se:")
print(pd.DataFrame({"coef": w.params, "robust se": w.bse}).round(4))
""",
notes="""
In levels, both tests reject homoskedasticity: the spread of prices grows with house size. Robust standard errors are
larger than classical ones, and the HC3 standard error on `lotsize` is far larger than the classical one, because a few
very large lots have high leverage. In logs, the Breusch–Pagan test no longer rejects: the log model is the more natural
specification for prices, and it also fixes most of the heteroskedasticity. WLS changes the point estimates, which is a
sign that the CEF is not exactly linear in levels (Problem 5): WLS and OLS then estimate different weighted projections.
"""),

dict(title="Clustering in a worker panel",
data="`wagepan` — 545 young men followed annually 1980–1987 (NLSY), 4,360 person-years.",
task="""
1. Estimate pooled OLS of `lwage` on `union`, `educ`, `black`, `hisp`, `exper`, `expersq`, `married` and year dummies.
2. Compare the standard error on `union` under (a) classical, (b) HC1 and (c) clustering by worker (`nr`).
3. Estimate the within-worker correlation of the residuals and of `union`, and compare the ratio of the clustered to the HC1 variance with the Moulton approximation \\(1+(m-1)\\rho_x\\rho_u\\), with \\(m=8\\).
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wagepan")
f = "lwage ~ union + educ + black + hisp + exper + expersq + married + C(year)"
# TODO: fit(cov_type="cluster", cov_kwds={"groups": df["nr"]})
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wagepan")
f = "lwage ~ union + educ + black + hisp + exper + expersq + married + C(year)"
m = smf.ols(f, data=df)
se = {"classical": m.fit().bse["union"], "HC1": m.fit(cov_type="HC1").bse["union"],
      "cluster(nr)": m.fit(cov_type="cluster", cov_kwds={"groups": df["nr"]}).bse["union"]}
print({k: round(v, 4) for k, v in se.items()}, " coef:", round(m.fit().params["union"], 4))

def within_corr(v):
    w = v.to_numpy().reshape(-1, 8)                 # 545 workers x 8 years (sorted by nr, year)
    w = w - w.mean()
    num = sum(np.sum(np.outer(r, r)) - np.sum(r**2) for r in w)
    return num / (7 * np.sum(w**2))
df["res"] = m.fit().resid
rho_u, rho_x = within_corr(df["res"]), within_corr(df["union"].astype(float))
print(f"within-worker correlation: residual {rho_u:.3f}, union {rho_x:.3f}")
print(f"variance ratio cluster/HC1 = {(se['cluster(nr)']/se['HC1'])**2:.2f};  Moulton 1+(m-1)rho_x rho_u = {1 + 7*rho_x*rho_u:.2f}")
""",
notes="""
Both union status and the wage residual are highly persistent within workers (within-worker correlations of about
0.5), so clustering raises the standard error on `union` by about 70% — the variance by a factor of 2.8. The simple
Moulton formula, 2.7, approximates the ratio well. The HC1 standard error treats 4,360 strongly
correlated observations as independent. Inference here rests on 545 worker clusters — the level of the sampling design —
and on the remaining issue that pooled OLS ignores the unobserved worker effect, which Exercise 7.1 addresses.
"""),
]

# ---------------------------------------------------------------- Lecture 5
EXERCISES["05-endogeneity-iv.md"] = [
dict(title="Card (1995): college proximity as an instrument for schooling",
data="`card` — 3,010 men from the NLS Young Men cohort, earnings in 1976; `nearc4` = grew up near a four-year college.",
task="""
1. Estimate OLS of `lwage` on `educ`, `exper`, `expersq`, `black`, `smsa`, `south`, `smsa66` and regional dummies `reg662`–`reg669`.
2. Estimate the first stage (`educ` on `nearc4` and the controls). Report the robust first-stage \\(F\\).
3. Estimate 2SLS with `linearmodels` (robust standard errors), treating `exper` and `expersq` as exogenous as in Card's simplest specification.
4. Compute the **Anderson–Rubin** 95% confidence set by inverting the AR test over a grid of \\(\\beta_0\\).
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
df = wd.data("card")
ctrl = "exper + expersq + black + smsa + south + smsa66 + " + " + ".join(f"reg66{i}" for i in range(2, 10))
iv = IV2SLS.from_formula(f"lwage ~ 1 + {ctrl} + [educ ~ nearc4]", data=df).fit(cov_type="robust")
# TODO: OLS, first stage F, AR test: regress (lwage - b0*educ) on nearc4 + controls for a grid of b0
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
df = wd.data("card")
ctrl = "exper + expersq + black + smsa + south + smsa66 + " + " + ".join(f"reg66{i}" for i in range(2, 10))
ols = smf.ols(f"lwage ~ educ + {ctrl}", data=df).fit(cov_type="HC1")
fs = smf.ols(f"educ ~ nearc4 + {ctrl}", data=df).fit(cov_type="HC1")
iv = IV2SLS.from_formula(f"lwage ~ 1 + {ctrl} + [educ ~ nearc4]", data=df).fit(cov_type="robust")
print(f"OLS  return to educ: {ols.params['educ']:.4f} (se {ols.bse['educ']:.4f})")
print(f"first stage: nearc4 -> educ = {fs.params['nearc4']:.4f}, robust F = {fs.tvalues['nearc4']**2:.1f}")
print(f"2SLS return to educ: {iv.params['educ']:.4f} (se {iv.std_errors['educ']:.4f})")

acc = []
for b0 in np.arange(-0.10, 0.60, 0.0025):
    df["ytil"] = df["lwage"] - b0 * df["educ"]
    t = smf.ols(f"ytil ~ nearc4 + {ctrl}", data=df).fit(cov_type="HC1").tvalues["nearc4"]
    if abs(t) < 1.96:
        acc.append(b0)
print(f"Anderson-Rubin 95% set: [{min(acc):.3f}, {max(acc):.3f}]  vs Wald: [{iv.params['educ']-1.96*iv.std_errors['educ']:.3f}, {iv.params['educ']+1.96*iv.std_errors['educ']:.3f}]")
""",
notes="""
The 2SLS return (about 13%) is close to double the OLS estimate (7.5%), with a standard error about fifteen times larger. The first-stage
\\(F\\) is above 10 but well below the ~105 needed for the usual \\(t\\)-test to have exactly correct size (Lee et al., 2022).
The Anderson–Rubin set is therefore wider than the Wald interval and **asymmetric**, extending further to the right.
It is valid regardless of instrument strength. Interpretation: a LATE for men whose schooling responded to living near
a college, who plausibly face higher marginal returns. The exclusion restriction is open to challenge — proximity to a
college correlates with urban labor markets and family background, which the controls only partly capture.
"""),

dict(title="Returns to education for married women: over-identification and endogeneity tests",
data="`mroz` — 753 married women, PSID 1975 (428 in the labor force); `fatheduc`, `motheduc` = parents' education.",
task="""
1. For women in the labor force (`inlf == 1`), estimate `lwage` on `educ`, `exper`, `expersq` by OLS and by 2SLS using `fatheduc` and `motheduc` as instruments for `educ`.
2. Report the first-stage \\(F\\) for the two excluded instruments.
3. Test the over-identifying restriction (robust Hansen \\(J\\) / Wooldridge score test).
4. Test the endogeneity of `educ` with the control-function regression.
""",
starter="""
import wooldridge as wd, statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
df = wd.data("mroz").query("inlf == 1").copy()
iv = IV2SLS.from_formula("lwage ~ 1 + exper + expersq + [educ ~ fatheduc + motheduc]", data=df).fit(cov_type="robust")
# TODO: iv.first_stage, iv.wooldridge_overid, control function: add first-stage residual to OLS
""",
solution="""
import wooldridge as wd, statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
df = wd.data("mroz").query("inlf == 1").copy()
ols = smf.ols("lwage ~ educ + exper + expersq", data=df).fit(cov_type="HC1")
iv = IV2SLS.from_formula("lwage ~ 1 + exper + expersq + [educ ~ fatheduc + motheduc]", data=df).fit(cov_type="robust")
print(f"OLS  educ: {ols.params['educ']:.4f} (se {ols.bse['educ']:.4f})")
print(f"2SLS educ: {iv.params['educ']:.4f} (se {iv.std_errors['educ']:.4f})")

fs = smf.ols("educ ~ fatheduc + motheduc + exper + expersq", data=df).fit(cov_type="HC1")
print(f"first-stage robust F (2 instruments): {fs.wald_test('fatheduc = 0, motheduc = 0', scalar=True).statistic/2:.1f}")
o = iv.wooldridge_overid
print(f"over-identification (Wooldridge score): stat = {o.stat:.3f}, p = {o.pval:.3f}")

df["vhat"] = smf.ols("educ ~ fatheduc + motheduc + exper + expersq", data=df).fit().resid
cf = smf.ols("lwage ~ educ + exper + expersq + vhat", data=df).fit(cov_type="HC1")
print(f"control function: educ = {cf.params['educ']:.4f} (equals 2SLS), vhat t = {cf.tvalues['vhat']:.2f}, p = {cf.pvalues['vhat']:.3f}")
""",
notes="""
The 2SLS estimate is smaller than OLS and much less precise. The instruments are strong, and the over-identification
test does not reject, so the two parental-education instruments give mutually consistent answers. That does **not**
prove both are valid: if both are correlated with unobserved ability in the same direction, the test has no power. The
control-function coefficient on `educ` equals 2SLS exactly, and the endogeneity test is not significant at 10% (p ≈ 0.11), so
the data cannot sharply distinguish OLS from 2SLS here. Note also the sample selection: wages are observed only for
women who work, so these regressions condition on participation (Heckman, 1979).
"""),
]

# ---------------------------------------------------------------- Lecture 6
EXERCISES["06-simultaneous-equations.md"] = [
dict(title="Labor supply and wage offers of married women as a simultaneous system",
data="`mroz` — working married women (428 observations).",
task="""
Wooldridge (2020, Example 16.5) writes a two-equation system:

- **labor supply:** `hours` on `lwage`, `educ`, `age`, `kidslt6`, `nwifeinc`;
- **wage offer:** `lwage` on `hours`, `educ`, `exper`, `expersq`.

1. Check the order condition for each equation.
2. Estimate each equation by OLS and by 2SLS, using all exogenous variables in the system as instruments.
3. Interpret the implied labor supply elasticity at the mean of `hours`.
4. Check the rank condition empirically: in the first stage for `lwage`, are `exper` and `expersq` jointly significant?
""",
starter="""
import wooldridge as wd, statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
df = wd.data("mroz").query("inlf == 1").copy()
supply = IV2SLS.from_formula("hours ~ 1 + educ + age + kidslt6 + nwifeinc + [lwage ~ exper + expersq]", data=df).fit(cov_type="robust")
# TODO: wage-offer equation instruments hours with age, kidslt6, nwifeinc
""",
solution="""
import wooldridge as wd, statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
df = wd.data("mroz").query("inlf == 1").copy()
s_ols = smf.ols("hours ~ lwage + educ + age + kidslt6 + nwifeinc", data=df).fit(cov_type="HC1")
s_iv = IV2SLS.from_formula("hours ~ 1 + educ + age + kidslt6 + nwifeinc + [lwage ~ exper + expersq]", data=df).fit(cov_type="robust")
w_ols = smf.ols("lwage ~ hours + educ + exper + expersq", data=df).fit(cov_type="HC1")
w_iv = IV2SLS.from_formula("lwage ~ 1 + educ + exper + expersq + [hours ~ age + kidslt6 + nwifeinc]", data=df).fit(cov_type="robust")
print(f"supply: hours on lwage   OLS {s_ols.params['lwage']:8.1f}   2SLS {s_iv.params['lwage']:8.1f} (se {s_iv.std_errors['lwage']:.1f})")
print(f"offer:  lwage on hours   OLS {w_ols.params['hours']:8.5f}   2SLS {w_iv.params['hours']:8.5f} (se {w_iv.std_errors['hours']:.5f})")
print(f"implied labor supply elasticity at mean hours ({df['hours'].mean():.0f}): {s_iv.params['lwage']/df['hours'].mean():.2f}")
fs = smf.ols("lwage ~ exper + expersq + educ + age + kidslt6 + nwifeinc", data=df).fit(cov_type="HC1")
print(f"rank check, first stage for lwage: F(exper, expersq) = {fs.wald_test('exper = 0, expersq = 0', scalar=True).statistic/2:.2f}")
""",
notes="""
Each equation excludes enough exogenous variables: supply excludes `exper` and `expersq` (just enough for one
endogenous regressor, plus one over-identifying restriction), and the wage-offer equation excludes three. OLS finds a
**negative** effect of wages on hours, because of simultaneity bias. 2SLS reverses the sign and implies a labor supply
elasticity of about 1.3. The first-stage \\(F\\) for the supply equation's instruments is only about 6 — below the rule of
thumb of 10 — so the estimate is imprecise and weak-instrument-robust inference (Lecture 5) is needed before taking the
elasticity seriously. The exclusion of experience from the
supply equation is itself debatable.
"""),

dict(title="Inflation and openness: a cross-country simultaneous system (Romer, 1993)",
data="`openness` — 114 countries, averages 1973–1990: inflation `inf`, imports as share of GDP `open`, log land area `lland`, log per capita income `lpcinc`.",
task="""
Romer (1993) argues that more open economies have less incentive to inflate. Openness is plausibly endogenous, but
small countries are mechanically more open, so land area can shift openness without affecting inflation directly.

1. Estimate `inf` on `open` and `lpcinc` by OLS.
2. Estimate the first stage of `open` on `lland` and `lpcinc`.
3. Estimate 2SLS with `lland` instrumenting `open`. Compare.
4. Discuss the exclusion restriction.
""",
starter="""
import wooldridge as wd, statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
df = wd.data("openness").rename(columns={"inf": "infl"})   # "inf" would be read as infinity in a formula
# TODO: OLS, first stage and IV2SLS.from_formula("infl ~ 1 + lpcinc + [open ~ lland]", ...)
""",
solution="""
import wooldridge as wd, statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
df = wd.data("openness")
df = df.rename(columns={"inf": "infl"})
ols = smf.ols("infl ~ open + lpcinc", data=df).fit(cov_type="HC1")
fs = smf.ols("open ~ lland + lpcinc", data=df).fit(cov_type="HC1")
iv = IV2SLS.from_formula("infl ~ 1 + lpcinc + [open ~ lland]", data=df).fit(cov_type="robust")
print(f"OLS : effect of openness on inflation {ols.params['open']:.3f} (se {ols.bse['open']:.3f})")
print(f"first stage: lland -> open {fs.params['lland']:.3f}, robust F = {fs.tvalues['lland']**2:.1f}")
print(f"2SLS: effect of openness on inflation {iv.params['open']:.3f} (se {iv.std_errors['open']:.3f})")
print(f"a 10-point rise in openness changes inflation by {10*iv.params['open']:.2f} percentage points (2SLS)")
""",
notes="""
Land area is a strong instrument: larger countries trade less. Both OLS and 2SLS give a negative effect of openness on
inflation, and 2SLS is larger in absolute value, consistent with Romer's argument. Exclusion requires that country size
affects inflation only through openness. Size could also matter through the political economy of central banking or
seigniorage needs, so the restriction is an assumption to be argued, not a fact.
"""),
]

# ---------------------------------------------------------------- Lecture 7
EXERCISES["07-panel-data.md"] = [
dict(title="The union wage premium: pooled OLS, random effects, fixed effects and first differences",
data="`wagepan` — 545 men, 1980–1987.",
task="""
1. Estimate the union premium in `lwage` by pooled OLS, RE, FE and FD, with year effects and the controls `married`, `exper²` (and `educ`, `black`, `hisp` where identified). Use standard errors clustered by worker.
2. Run the Mundlak (correlated random effects) test of RE against FE, adding worker means of the time-varying regressors.
3. Interpret the ranking of the estimates.
""",
starter="""
import wooldridge as wd
from linearmodels.panel import PooledOLS, RandomEffects, PanelOLS, FirstDifferenceOLS
df = wd.data("wagepan").set_index(["nr", "year"])
fe = PanelOLS.from_formula("lwage ~ union + married + expersq + TimeEffects + EntityEffects", data=df).fit(cov_type="clustered", cluster_entity=True)
# TODO: pooled, RE, FD; Mundlak: add group means of union, married, expersq to a RE/pooled regression
""",
solution="""
import pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
from linearmodels.panel import PooledOLS, RandomEffects, PanelOLS, FirstDifferenceOLS
raw = wd.data("wagepan")
df = raw.set_index(["nr", "year"])
yd = " + ".join(f"d8{i}" for i in range(1, 8))
cl = dict(cov_type="clustered", cluster_entity=True)
po = PooledOLS.from_formula(f"lwage ~ 1 + union + married + expersq + educ + black + hisp + {yd}", data=df).fit(**cl)
re = RandomEffects.from_formula(f"lwage ~ 1 + union + married + expersq + educ + black + hisp + {yd}", data=df).fit(**cl)
fe = PanelOLS.from_formula("lwage ~ union + married + expersq + TimeEffects + EntityEffects", data=df).fit(**cl)
fd = FirstDifferenceOLS.from_formula(f"lwage ~ union + married + expersq + {yd}", data=df).fit(**cl)
tab = pd.DataFrame({m: [r.params["union"], r.std_errors["union"]] for m, r in
                    [("pooled", po), ("RE", re), ("FE", fe), ("FD", fd)]}, index=["union", "cluster se"]).round(4)
print(tab)

for v in ("union", "married", "expersq"):
    raw[v + "_bar"] = raw.groupby("nr")[v].transform("mean")
mk = smf.ols(f"lwage ~ union + married + expersq + union_bar + married_bar + expersq_bar + educ + black + hisp + {yd}",
             data=raw).fit(cov_type="cluster", cov_kwds={"groups": raw["nr"]})
print(f"\\nMundlak: union coefficient {mk.params['union']:.4f} (= FE)")
w = mk.wald_test("union_bar = 0, married_bar = 0, expersq_bar = 0", scalar=True)
print(f"robust test of RE vs FE (xi = 0): chi2(3) = {w.statistic:.2f}, p = {w.pvalue:.4f}")
""",
notes="""
Pooled OLS gives the largest premium, RE an intermediate one, and FE and FD the smallest. The Mundlak regression
reproduces the FE coefficient exactly, and its cluster-robust test rejects \\(\\xi=0\\) decisively, so the worker effect is
correlated with the regressors and RE is inconsistent. The drop from pooled OLS to FE is consistent with positive
selection of higher-wage workers into union jobs. Part of it may also be attenuation, because misreported switches in
union status loom large in within-worker variation (Lecture 7, worked example). The difference between FE and FD hints at
remaining measurement error or a failure of strict exogeneity.
"""),

dict(title="Crime and deterrence across North Carolina counties",
data="`crime4` — 90 North Carolina counties, 1981–1987 (Cornwell and Trumbull, 1994).",
task="""
1. Estimate `lcrmrte` (log crimes per person) on log probability of arrest `lprbarr`, conviction `lprbconv`, prison `lprbpris`, log average sentence `lavgsen` and log police per capita `lpolpc`, with year effects, by pooled OLS and by county fixed effects. Cluster by county.
2. Compare the deterrence elasticities. Why could FE still be biased for `lpolpc`?
""",
starter="""
import wooldridge as wd
from linearmodels.panel import PanelOLS, PooledOLS
df = wd.data("crime4").set_index(["county", "year"])
x = "lprbarr + lprbconv + lprbpris + lavgsen + lpolpc"
# TODO: PooledOLS with year dummies vs PanelOLS with EntityEffects + TimeEffects
""",
solution="""
import pandas as pd, wooldridge as wd
from linearmodels.panel import PanelOLS, PooledOLS
df = wd.data("crime4").set_index(["county", "year"])
x = "lprbarr + lprbconv + lprbpris + lavgsen + lpolpc"
yd = " + ".join(f"d8{i}" for i in range(2, 8))
cl = dict(cov_type="clustered", cluster_entity=True)
po = PooledOLS.from_formula(f"lcrmrte ~ 1 + {x} + {yd}", data=df).fit(**cl)
fe = PanelOLS.from_formula(f"lcrmrte ~ {x} + EntityEffects + TimeEffects", data=df).fit(**cl)
names = x.split(" + ")
print(pd.DataFrame({"pooled": po.params[names], "pooled se": po.std_errors[names],
                    "FE": fe.params[names], "FE se": fe.std_errors[names]}).round(3))
print(f"\\nwithin R2 (FE): {fe.rsquared_within:.3f};  counties: {df.index.get_level_values(0).nunique()}")
""",
notes="""
Fixed effects shrink the deterrence elasticities for arrest and conviction probabilities substantially compared with
pooled OLS: counties with persistently low crime also have high arrest rates for reasons that FE removes. The police
coefficient is **positive** in both models. That is the classic sign of reverse causality — counties hire police when
crime rises — which county fixed effects cannot fix because the feedback is time-varying. This is exactly the failure of
strict exogeneity discussed in Lecture 7, and it calls for an instrument or a dynamic model.
"""),
]

# ---------------------------------------------------------------- Lecture 8
EXERCISES["08-dynamic-panels.md"] = [
dict(title="Wage persistence: the OLS–FE bracket and Anderson–Hsiao IV",
data="`wagepan` — 545 men, 1980–1987.",
task="""
Model \\(lwage_{it}=\\rho\\,lwage_{i,t-1}+\\beta\\,union_{it}+\\lambda_t+c_i+u_{it}\\).

1. Estimate \\(\\rho\\) by pooled OLS and by fixed effects. Which way should each be biased?
2. Estimate the first-differenced equation by 2SLS, instrumenting \\(\\Delta lwage_{i,t-1}\\) with the level \\(lwage_{i,t-2}\\) (Anderson–Hsiao). Report the first-stage \\(F\\).
3. Does the IV estimate fall inside the OLS–FE bracket?
4. Test for second-order serial correlation in the differenced residuals.
""",
starter="""
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wagepan").sort_values(["nr", "year"])
g = df.groupby("nr")
df["l1"] = g["lwage"].shift(1); df["l2"] = g["lwage"].shift(2)
df["dy"] = df["lwage"] - df["l1"]; df["dl1"] = df["l1"] - df["l2"]
# TODO: pooled OLS and FE of lwage on l1; 2SLS of dy on dl1 (instrument: l2 = lwage two years back)
""",
solution="""
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS
df = wd.data("wagepan").sort_values(["nr", "year"])
g = df.groupby("nr")
df["l1"], df["l2"], df["l3"] = g["lwage"].shift(1), g["lwage"].shift(2), g["lwage"].shift(3)
df["dy"], df["dl1"] = df["lwage"] - df["l1"], df["l1"] - df["l2"]
df["dunion"] = df["union"] - g["union"].shift(1)

po = smf.ols("lwage ~ l1 + union + C(year)", data=df.dropna(subset=["l1"])).fit(cov_type="cluster", cov_kwds={"groups": df.dropna(subset=["l1"])["nr"]})
fe = PanelOLS.from_formula("lwage ~ l1 + union + TimeEffects + EntityEffects", data=df.dropna(subset=["l1"]).set_index(["nr", "year"])).fit(cov_type="clustered", cluster_entity=True)
d = df.dropna(subset=["dy", "dl1", "l2", "dunion"]).copy()
ah = IV2SLS.from_formula("dy ~ 1 + dunion + C(year) + [dl1 ~ l2]", data=d).fit(cov_type="clustered", clusters=d["nr"])
fs = smf.ols("dl1 ~ l2 + dunion + C(year)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["nr"]})
print(f"pooled OLS rho = {po.params['l1']:.3f}   FE rho = {fe.params['l1']:.3f}   Anderson-Hsiao rho = {ah.params['dl1']:.3f} (se {ah.std_errors['dl1']:.3f})")
print(f"AH first stage: coefficient on l2 = {fs.params['l2']:.3f}, clustered F = {fs.tvalues['l2']**2:.1f}")
print(f"union effect: pooled {po.params['union']:.3f}, FE {fe.params['union']:.3f}, AH {ah.params['dunion']:.3f}")

d["e"] = ah.resids
d["e_lag2"] = d.groupby("nr")["e"].shift(2)
dd = d.dropna(subset=["e_lag2"])
ar2 = smf.ols("e ~ e_lag2", data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd["nr"]})
print(f"AR(2) check: corr of differenced residuals at lag 2 = {dd['e'].corr(dd['e_lag2']):.3f}, cluster t = {ar2.tvalues['e_lag2']:.2f}")
""",
notes="""
Pooled OLS is biased **up** (the lag contains the worker effect) and FE is biased **down** (Nickell bias with only
\\(T=8\\)). They bracket the true \\(\\rho\\) widely. The Anderson–Hsiao estimate lies inside the bracket, which is a reassuring
sanity check. Its first stage is strong here because wages are only moderately persistent once worker effects are
removed. The AR(2) check is what licenses the \\(t-2\\) instrument: a significant lag-2 correlation of the differenced
residuals would mean serial correlation in \\(u_{it}\\), and deeper lags would be needed. Arellano–Bond GMM uses all
available lags; try `pydynpd` and compare, reporting the instrument count.
"""),

dict(title="Grunfeld investment: how instrument count changes GMM",
data="`grunfeld` (statsmodels) — investment, market value and capital of 11 large U.S. firms, 1935–1954.",
task="""
With only \\(N=11\\) firms and \\(T=20\\), the Arellano–Bond instrument count explodes relative to \\(N\\).

1. Estimate \\(invest_{it}=\\rho\\,invest_{i,t-1}+\\beta_1 value_{it}+\\beta_2 capital_{it}+c_i+u_{it}\\) by pooled OLS and FE.
2. Count how many Arellano–Bond moment conditions "all lags" would use for the lagged dependent variable alone.
3. Estimate the differenced equation by just-identified IV (lag-2 level) and by an over-identified IV with lags 2–4 (collapsed). Compare.
4. Explain why "panel GMM" is a poor tool for this dataset.
""",
starter="""
import statsmodels.api as sm
df = sm.datasets.grunfeld.load_pandas().data.sort_values(["firm", "year"])
# TODO: create lags by firm; pooled OLS, FE; IV on first differences with y_{t-2}, y_{t-3}, y_{t-4}
""",
solution="""
import statsmodels.api as sm, statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
df = sm.datasets.grunfeld.load_pandas().data.sort_values(["firm", "year"]).reset_index(drop=True)
g = df.groupby("firm")
for k in (1, 2, 3, 4):
    df[f"i{k}"] = g["invest"].shift(k)
df["dy"], df["dl1"] = df["invest"] - df["i1"], df["i1"] - df["i2"]
df["dv"], df["dk"] = g["value"].diff(), g["capital"].diff()

d1 = df.dropna(subset=["i1"])
po = smf.ols("invest ~ i1 + value + capital", data=d1).fit()
fe = smf.ols("invest ~ i1 + value + capital + C(firm)", data=d1).fit()
print(f"pooled OLS rho = {po.params['i1']:.3f}    FE rho = {fe.params['i1']:.3f}")
T = df["year"].nunique()
print(f"Arellano-Bond 'all lags' moments for the lagged dependent variable: {sum(t - 1 for t in range(2, T))} with N = {df['firm'].nunique()} firms")

d2 = df.dropna(subset=["dy", "dl1", "i2", "i3", "i4", "dv", "dk"])
cl = dict(cov_type="clustered", clusters=d2["firm"].astype("category").cat.codes)
ji = IV2SLS.from_formula("dy ~ 0 + dv + dk + [dl1 ~ i2]", data=d2).fit(**cl)
oi = IV2SLS.from_formula("dy ~ 0 + dv + dk + [dl1 ~ i2 + i3 + i4]", data=d2).fit(**cl)
print(f"IV, lag 2 only:   rho = {ji.params['dl1']:.3f} (se {ji.std_errors['dl1']:.3f})")
print(f"IV, lags 2-4:     rho = {oi.params['dl1']:.3f} (se {oi.std_errors['dl1']:.3f})")
""",
notes="""
"All lags" would use 171 moment conditions for the lagged dependent variable alone — with 11 firms, the GMM weight matrix
cannot even be estimated. The two IV estimates differ wildly (about 0.01 with one instrument, 0.80 with three), and both have
standard errors above 0.2: the instruments carry little reliable information. With \\(T=20\\), the Nickell bias of FE is modest (order \\(1/T\\)), so FE is already a reasonable
estimator here; difference-GMM methods are designed for **large \\(N\\), small \\(T\\)**, the opposite of this dataset.
The clustered standard errors rest on only 11 clusters, so even they should not be trusted (Lecture 4). Choosing the estimator to fit the dimensions of the data is part of the econometrics.
"""),
]

# ---------------------------------------------------------------- Lecture 9
EXERCISES["09-time-series.md"] = [
dict(title="Forecasting U.S. inflation: AR(p) by BIC against a random walk",
data="`macrodata` (statsmodels) — U.S. quarterly macro series, 1959Q1–2009Q3; `infl` = annualized CPI inflation.",
task="""
1. Plot the ACF and PACF of inflation and choose an AR order by BIC on the 1960–1984 sample.
2. Produce pseudo out-of-sample one-quarter-ahead forecasts for 1985Q1–2009Q3 from (a) an AR(\\(p\\)) re-estimated each quarter on an expanding window and (b) a random walk (the forecast is the last observed value).
3. Report the RMSFE of each and a Diebold–Mariano test with a HAC standard error.
""",
starter="""
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
y = sm.datasets.macrodata.load_pandas().data["infl"].to_numpy()
# observations 4..103 are 1960Q1..1984Q4; 104..202 are 1985Q1..2009Q3
# TODO: ar_select_order(y[4:104], maxlag=8, ic="bic"); loop over t, fit AutoReg(y[:t], p), forecast y[t]
""",
solution="""
import numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.stattools import acf, pacf
y = sm.datasets.macrodata.load_pandas().data["infl"].to_numpy()
print("ACF  lags 1-6:", acf(y[4:104], nlags=6)[1:].round(2))
print("PACF lags 1-6:", pacf(y[4:104], nlags=6)[1:].round(2))
p = len(ar_select_order(y[4:104], maxlag=8, ic="bic").ar_lags)
print("BIC-selected AR order on 1960-1984:", p)

e_ar, e_rw = [], []
for t in range(104, len(y)):
    fit = AutoReg(y[:t], lags=p, trend="c").fit()
    e_ar.append(y[t] - fit.predict(start=t, end=t)[0])
    e_rw.append(y[t] - y[t - 1])
e_ar, e_rw = np.array(e_ar), np.array(e_rw)
print(f"RMSFE 1985Q1-2009Q3:  AR({p}) {np.sqrt(np.mean(e_ar**2)):.3f}   random walk {np.sqrt(np.mean(e_rw**2)):.3f}")
d = e_ar**2 - e_rw**2
dm = smf.ols("d ~ 1", data={"d": d}).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
print(f"Diebold-Mariano: mean loss differential {d.mean():.3f}, HAC t = {dm.tvalues.iloc[0]:.2f}, p = {dm.pvalues.iloc[0]:.3f}")
""",
notes="""
Inflation in 1960–84 is highly persistent: the ACF decays slowly and the PACF cuts off after a few lags, pointing to a
low-order AR. Out of sample, after 1985, the AR model **beats** the random walk, because quarterly CPI inflation became
noisy and much less persistent during the Great Moderation; the random walk then carries every transitory spike forward.
That is the opposite of the 1970s pattern, and a reminder that forecasting rankings depend on the sample (Stock and
Watson, 2007). The DM test uses HAC standard errors; the sharp 2008Q4 deflation spike dominates the loss differential —
check how sensitive the result is to excluding it.
"""),

dict(title="The persistence of unemployment: AR(1), half-life and the Kendall bias",
data="`phillips` — U.S. annual unemployment and inflation, 1948–2003.",
task="""
1. Estimate an AR(1) for the unemployment rate `unem`. Compute the half-life of a shock.
2. Correct \\(\\widehat\\rho\\) for the Kendall bias \\(-(1+3\\rho)/T\\) and recompute the half-life.
3. Check the residuals for remaining autocorrelation with a Ljung–Box test.
4. Forecast unemployment for 2004–2008 with 95% intervals (ignoring parameter uncertainty).
""",
starter="""
import numpy as np, wooldridge as wd
from statsmodels.tsa.ar_model import AutoReg
u = wd.data("phillips")["unem"].to_numpy()
fit = AutoReg(u, lags=1, trend="c").fit()
# TODO: half-life = ln(0.5)/ln(rho); bias-corrected rho_bc = rho + (1+3 rho)/T
""",
solution="""
import numpy as np, wooldridge as wd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
ph = wd.data("phillips")
u = ph["unem"].to_numpy(); T = len(u) - 1
fit = AutoReg(u, lags=1, trend="c").fit()
c, rho = fit.params
hl = lambda r: np.log(0.5) / np.log(r)
rho_bc = rho + (1 + 3 * rho) / T
print(f"rho = {rho:.3f} (se {fit.bse[1]:.3f}), mean = {c/(1-rho):.2f}%, half-life = {hl(rho):.2f} years")
print(f"bias-corrected rho = {rho_bc:.3f}, half-life = {hl(rho_bc):.2f} years")
lb = acorr_ljungbox(fit.resid, lags=[4], return_df=True)
print(f"Ljung-Box(4) on residuals: Q = {lb['lb_stat'].iloc[0]:.2f}, p = {lb['lb_pvalue'].iloc[0]:.3f}")
s2 = fit.sigma2; last = u[-1]; mu = c / (1 - rho)
for h in range(1, 6):
    f = mu + rho**h * (last - mu); v = s2 * sum(rho**(2*j) for j in range(h))
    print(f"  {int(ph['year'].iloc[-1]) + h}: {f:.2f}%  [{f - 1.96*np.sqrt(v):.2f}, {f + 1.96*np.sqrt(v):.2f}]")
""",
notes="""
Annual unemployment is persistent: \\(\\widehat\\rho\\approx0.74\\), a half-life of about 2.3 years. Because OLS is biased
downward in a sample of 56 years, the bias-corrected estimate (about 0.80) implies a half-life of about 3 years — a large
difference for a policy question such as how long a recession's unemployment lingers. The
Ljung–Box test does not detect remaining autocorrelation, but with annual data it has little power, and a business-cycle
AR(2) is worth trying. The intervals widen quickly toward the unconditional variance. Actual U.S. unemployment in 2004–2008
was 5.5, 5.1, 4.6, 4.6 and 5.8% — all inside the intervals. But 2009's 9.3% would have been far above them: an AR(1) cannot
anticipate a financial crisis.
"""),
]

# ---------------------------------------------------------------- Lecture 10
EXERCISES["10-unit-roots-breaks.md"] = [
dict(title="Is U.S. real GDP trend-stationary?",
data="`macrodata` — U.S. quarterly real GDP, 1959Q1–2009Q3.",
task="""
1. Run ADF (with constant and trend, AIC lag choice) and KPSS (trend) tests on `log(realgdp)`.
2. Run the same tests on GDP growth, \\(\\Delta\\log(realgdp)\\), with a constant only.
3. Run the Zivot–Andrews test that allows a break in the trend.
4. What do the four results say together?
""",
starter="""
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
y = np.log(sm.datasets.macrodata.load_pandas().data["realgdp"].to_numpy())
# TODO: adfuller(y, regression="ct", autolag="AIC"); kpss(y, regression="ct", nlags="auto")
""",
solution="""
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
m = sm.datasets.macrodata.load_pandas().data
y = np.log(m["realgdp"].to_numpy()); dy = np.diff(y)
a = adfuller(y, regression="ct", autolag="AIC"); k = kpss(y, regression="ct", nlags="auto")
print(f"log GDP:   ADF(ct) t = {a[0]:.2f} (5% cv {a[4]['5%']:.2f}, p = {a[1]:.3f}, lags {a[2]});  KPSS(ct) = {k[0]:.3f} (5% cv {k[3]['5%']})")
a2 = adfuller(dy, regression="c", autolag="AIC"); k2 = kpss(dy, regression="c", nlags="auto")
print(f"growth:    ADF(c)  t = {a2[0]:.2f} (p = {a2[1]:.4f});  KPSS(c) = {k2[0]:.3f} (5% cv {k2[3]['5%']})")
za = zivot_andrews(y, regression="ct", autolag="AIC")
bp = int(za[4]); yr = m['year'].iloc[bp]; q = m['quarter'].iloc[bp]
print(f"Zivot-Andrews (break in level and trend): stat = {za[0]:.2f}, p = {za[1]:.3f}, break at {int(yr)}Q{int(q)}")
""",
notes="""
The ADF test does not reject a unit root in log GDP, and KPSS rejects trend-stationarity: both point to a stochastic
trend. Growth is clearly stationary by both tests, so log GDP behaves as I(1). Allowing for one endogenous break in
level and trend, Zivot–Andrews still does not reject the unit root at 5% in this sample. The conclusion "GDP has a unit
root" is therefore robust to a single break here, but remember the low power of these tests against roots of
0.95–0.98 in 200 quarters (Lecture 10, lab), and that the sample ends in the middle of the Great Recession.
"""),

dict(title="A structural break in the Nile: Chow, sup-F and the effect on unit-root tests",
data="`nile` (statsmodels) — annual flow of the Nile at Aswan, 1871–1970.",
task="""
The first Aswan dam was completed in 1902; hydrologists date a drop in flow to about 1898.

1. Compute the Chow statistic for a break in the mean at every candidate year (15% trimming) and find the sup-\\(F\\) and its date.
2. Compare the sup-\\(F\\) with Andrews' 5% critical value for one parameter (about 8.7).
3. Run an ADF test on the raw series and on the series demeaned with a separate mean before and after the estimated break. Explain the difference.
""",
starter="""
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
d = sm.datasets.nile.load_pandas().data
y = d["volume"].to_numpy(); years = d["year"].astype(int).to_numpy()
# TODO: for each break b, F = (SSR_r - SSR_u) / (SSR_u / (n - 2)) with a one-mean vs two-means model
""",
solution="""
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
d = sm.datasets.nile.load_pandas().data
y = d["volume"].to_numpy(); years = d["year"].astype(int).to_numpy(); n = len(y)
ssr_r = ((y - y.mean())**2).sum()
F = {}
for b in range(int(0.15 * n), int(0.85 * n)):
    ssr_u = ((y[:b] - y[:b].mean())**2).sum() + ((y[b:] - y[b:].mean())**2).sum()
    F[b] = (ssr_r - ssr_u) / (ssr_u / (n - 2))
b = max(F, key=F.get)
print(f"sup-F = {F[b]:.1f} at a break starting in {years[b]} (Andrews 5% cv ~ 8.7)")
print(f"mean flow before: {y[:b].mean():.0f}, after: {y[b:].mean():.0f}")
a1 = adfuller(y, regression="c", autolag="AIC")
z = y.copy().astype(float); z[:b] -= y[:b].mean(); z[b:] -= y[b:].mean()
a2 = adfuller(z, regression="n", autolag="AIC")
print(f"ADF raw series:            t = {a1[0]:.2f}, p = {a1[1]:.3f}")
print(f"ADF break-adjusted series: t = {a2[0]:.2f}, p = {a2[1]:.4f}")
""",
notes="""
The sup-\\(F\\) statistic is far above Andrews' critical value, and the estimated break falls at the end of the 1890s, matching
the historical record: mean flow drops by roughly a fifth. On the raw series, the ADF test rejects less strongly than on
the break-adjusted series, where the evidence of stationarity is overwhelming. This is Perron's (1989) point in miniature:
an ignored level shift makes a stationary series look more persistent. (Strictly, after estimating the break date, the
second ADF test needs adjusted critical values, as in Zivot–Andrews.)
"""),
]

# ---------------------------------------------------------------- Lecture 11
EXERCISES["11-var-svar.md"] = [
dict(title="A three-variable monetary VAR for the U.S.",
data="`macrodata` — U.S. quarterly data, 1959Q1–2009Q3: real GDP, CPI inflation, 3-month T-bill rate.",
task="""
1. Build a VAR in GDP growth (annualized, %), inflation, and the T-bill rate. Choose the lag by AIC and BIC (max 8).
2. Check stability (eigenvalues of the companion matrix).
3. Test whether the T-bill rate Granger-causes GDP growth.
4. Compute recursive (Cholesky) impulse responses to a one-standard-deviation interest-rate shock, ordering growth, inflation, rate. Report the response of GDP growth and inflation over 12 quarters.
5. Report the 12-quarter forecast-error variance decomposition of GDP growth.
""",
starter="""
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.api import VAR
m = sm.datasets.macrodata.load_pandas().data
df = pd.DataFrame({"growth": 400 * np.log(m["realgdp"]).diff(), "infl": m["infl"], "rate": m["tbilrate"]}).dropna()
model = VAR(df)
# TODO: model.select_order(8); res = model.fit(p); res.is_stable(); res.test_causality(...); res.irf(12); res.fevd(12)
""",
solution="""
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.api import VAR
m = sm.datasets.macrodata.load_pandas().data
df = pd.DataFrame({"growth": 400 * np.log(m["realgdp"]).diff(), "infl": m["infl"], "rate": m["tbilrate"]}).dropna()
model = VAR(df.to_numpy())
sel = model.select_order(8)
print("lag order by AIC:", sel.aic, " by BIC:", sel.bic)
res = model.fit(4)
companion = np.vstack([np.hstack(res.coefs), np.eye(3 * 3, 3 * 4)])      # [A1 A2 A3 A4; I 0]
print("stable:", res.is_stable(), " max |eigenvalue| of companion:", round(max(abs(np.linalg.eigvals(companion))), 3))
gc = res.test_causality(0, [2], kind="wald")
print(f"Granger: rate -> growth  chi2 = {gc.test_statistic:.2f}, p = {gc.pvalue:.4f}")
irf = res.irf(12).orth_irfs
print("response to a 1-s.d. rate shock (Cholesky: growth, infl, rate):")
print(pd.DataFrame({"growth": irf[:, 0, 2], "infl": irf[:, 1, 2], "rate": irf[:, 2, 2]}).iloc[[0, 1, 2, 4, 6, 8, 12]].round(3).to_string())
fe = res.fevd(12).decomp[0, -1]
print("FEVD of growth at h=12 (shares of growth, infl, rate shocks):", fe.round(3))
""",
notes="""
AIC and BIC disagree on the lag length, a typical outcome; we use four lags (one year), a common compromise for quarterly
data. The system is stable. Interest rates Granger-cause GDP growth strongly — but as Problem 3 explains, that may reflect
the Fed reacting to information about future growth, not a causal effect of policy. After a recursive interest-rate
shock, output growth first *rises* and then falls in the second quarter, and **inflation rises** for several quarters: the
**price puzzle** of Lecture 11. The recursive ordering attributes to "policy shocks" some of the Fed's response to anticipated inflation. The
FEVD shows that interest-rate shocks explain only about 4% of the variance of growth at a 3-year horizon; almost 90% is
due to growth's own shocks.
"""),

dict(title="How much does the ordering matter?",
data="`macrodata`, same VAR as above.",
task="""
1. Re-estimate the Cholesky responses with the rate ordered **first** (the policy rate cannot respond within the quarter to output and inflation).
2. Compare the impact responses and the 8-quarter responses of growth to the rate shock under the two orderings.
3. Compute the correlation of the reduced-form residuals. Why does the ordering matter more when this correlation is large?
""",
starter="""
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.api import VAR
m = sm.datasets.macrodata.load_pandas().data
df = pd.DataFrame({"growth": 400 * np.log(m["realgdp"]).diff(), "infl": m["infl"], "rate": m["tbilrate"]}).dropna()
# TODO: fit VAR(4) on df and on df[["rate", "growth", "infl"]]; compare orth_irfs
""",
solution="""
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.api import VAR
m = sm.datasets.macrodata.load_pandas().data
df = pd.DataFrame({"growth": 400 * np.log(m["realgdp"]).diff(), "infl": m["infl"], "rate": m["tbilrate"]}).dropna()
a = VAR(df).fit(4); b = VAR(df[["rate", "growth", "infl"]]).fit(4)
ia, ib = a.irf(12).orth_irfs, b.irf(12).orth_irfs
print("response of growth to the rate shock")
print(pd.DataFrame({"rate ordered last": ia[:, 0, 2], "rate ordered first": ib[:, 1, 0]}).iloc[[0, 1, 2, 4, 8]].round(3).to_string())
print("\\nreduced-form residual correlations:")
print(pd.DataFrame(a.resid).corr().round(3).to_string())
""",
notes="""
With the rate ordered last, growth cannot respond on impact by construction. With the rate ordered first, the impact
response is whatever the residual correlation implies. That correlation is positive: in the data, unexpected rate rises
come with unexpectedly strong growth, because the Fed responds within the quarter. Ordering the rate first therefore
attributes that co-movement to the policy shock, and growth appears to **rise** on impact after a tightening — an
implausible result. Identification is an economic assumption about timing, and the data alone cannot settle it. External
instruments (high-frequency surprises) are the modern response.
"""),
]

# ---------------------------------------------------------------- Lecture 12
EXERCISES["12-cointegration-ardl.md"] = [
dict(title="Danish money demand: Johansen's classic dataset",
data="`danish_data` (statsmodels) — Danish quarterly data, 1974Q1–1987Q3 (Johansen and Juselius, 1990): log real money `lrm`, log real income `lry`, bond rate `ibo`, deposit rate `ide`.",
task="""
1. Run the Johansen trace test on `(lrm, lry, ibo, ide)` with a constant and one lagged difference.
2. Assuming rank 1, estimate the VECM and normalize the cointegrating vector on money.
3. Interpret the income elasticity and the interest-rate coefficients, and the adjustment coefficients.
""",
starter="""
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
d = sm.datasets.danish_data.load_pandas().data[["lrm", "lry", "ibo", "ide"]]
j = coint_johansen(d, det_order=0, k_ar_diff=1)
# TODO: print j.lr1 against j.cvt[:,1]; VECM(d, k_ar_diff=1, coint_rank=1, deterministic="ci").fit()
""",
solution="""
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
d = sm.datasets.danish_data.load_pandas().data[["lrm", "lry", "ibo", "ide"]]
j = coint_johansen(d, det_order=0, k_ar_diff=1)
print(pd.DataFrame({"H0: r <=": range(4), "trace": j.lr1.round(2), "5% cv": j.cvt[:, 1]}).to_string(index=False))
v = VECM(d, k_ar_diff=1, coint_rank=1, deterministic="ci").fit()
beta = v.beta[:, 0] / v.beta[0, 0]
print("\\ncointegrating vector (normalized on lrm):", dict(zip(d.columns, beta.round(3))))
print("long run: lrm =", " + ".join(f"{-b:.3f}*{c}" for b, c in zip(beta[1:], d.columns[1:])), "+ const")
print("adjustment coefficients alpha:", dict(zip(d.columns, v.alpha[:, 0].round(3))))
""",
notes="""
The trace test rejects \\(r=0\\) only narrowly (48.8 against a critical value of 47.9) and does not reject \\(r\\le1\\) in this
short sample of 55 quarters, so one cointegrating relation — as Johansen and Juselius (1990) found with seasonal dummies
and a restricted constant. Under
rank one, the long-run income elasticity of money demand is close to one, and the coefficients on the two interest rates
have opposite signs: the bond rate (the opportunity cost of holding money) lowers money demand, and the deposit rate (the
own rate on money) raises it — with similar magnitudes, so the spread matters. Money is the variable that adjusts to
disequilibrium, which supports reading the relation as **money demand**. The sensitivity of the rank test to the
deterministic terms is part of the lesson.
"""),

dict(title="The consumption–income relation: Engle–Granger, DOLS and the error-correction model",
data="`macrodata` — U.S. quarterly real consumption `realcons` and real disposable income `realdpi`, 1959–2009.",
task="""
1. Test both logs for a unit root (ADF with trend).
2. Run the Engle–Granger cointegration test of `log(realcons)` on `log(realdpi)`.
3. Estimate the cointegrating slope by static OLS and by DOLS with 2 leads and lags (HAC standard errors).
4. Estimate the error-correction model \\(\\Delta c_t=\\alpha+\\gamma\\,\\widehat z_{t-1}+\\delta\\,\\Delta y_t+\\varepsilon_t\\). How fast does consumption close the gap?
""",
starter="""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.stattools import coint, adfuller
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"c": np.log(m["realcons"]), "y": np.log(m["realdpi"])})
# TODO: coint(d.c, d.y); DOLS: add dy leads/lags; ECM with lagged residual
""",
solution="""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.stattools import coint, adfuller
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"c": np.log(m["realcons"]), "y": np.log(m["realdpi"])})
for v in ("c", "y"):
    a = adfuller(d[v], regression="ct", autolag="AIC"); print(f"ADF(ct) {v}: t = {a[0]:.2f}, p = {a[1]:.3f}")
t, p, cv = coint(d["c"], d["y"])
print(f"Engle-Granger: t = {t:.2f}, p = {p:.4f}, 5% cv = {cv[1]:.2f}")

static = smf.ols("c ~ y", data=d).fit()
d["dy"] = d["y"].diff()
for k in (-2, -1, 1, 2):
    d[f"dy_{k}"] = d["dy"].shift(k)
d = d.rename(columns={"dy_-1": "dy_m1", "dy_-2": "dy_m2"})         # leads of dy (k = -1, -2)
dols = smf.ols("c ~ y + dy + dy_1 + dy_2 + dy_m1 + dy_m2", data=d.dropna()).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
print(f"static OLS slope {static.params['y']:.4f};  DOLS slope {dols.params['y']:.4f} (HAC se {dols.bse['y']:.4f})")
w = dols.t_test("y = 1")
print(f"DOLS test of slope = 1: t = {float(np.squeeze(w.tvalue)):.2f}")

d["z"] = static.resid; d["dc"] = d["c"].diff(); d["z1"] = d["z"].shift(1)
ecm = smf.ols("dc ~ z1 + dy", data=d.dropna(subset=["dc", "z1", "dy"])).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
g = ecm.params["z1"]
print(f"ECM: adjustment {g:.4f} (t = {ecm.tvalues['z1']:.2f}), short-run income effect {ecm.params['dy']:.3f}")
print(f"half-life of a deviation from equilibrium: {np.log(0.5)/np.log(1+g):.1f} quarters")
""",
notes="""
Both series are I(1). The Engle–Granger test, with its special critical values, does **not** reject the null of no
cointegration (p ≈ 0.26). That is a common finding for consumption and income in levels: the consumption–income ratio
drifted upward from the 1980s as the saving rate fell, so the gap looks persistent. The remaining steps are therefore
conditional on cointegration — exactly the situation where a researcher must say so explicitly. The long-run elasticity is close to but significantly above one in the DOLS regression, which uses
valid HAC inference unlike the static regression. The error-correction coefficient is negative, as theory requires, but
small: consumption closes only a few percent of the gap each quarter. Short-run consumption responds much less than
one-for-one to income growth — the consumption smoothing that permanent-income theory predicts.
"""),
]

# ---------------------------------------------------------------- Lecture 13
EXERCISES["13-causal-inference.md"] = [
dict(title="LaLonde's job training data: can observational methods recover an experimental benchmark?",
data="`jtrain2` — the NSW randomized experiment (185 treated, 260 controls); `jtrain3` — the same treated men with 2,490 non-experimental controls from the PSID. Outcome: 1978 earnings `re78` (thousands of dollars).",
task="""
1. Estimate the experimental effect of training on `re78` (difference in means and with covariates).
2. Using `jtrain3`, estimate the effect by (a) a naive difference, (b) regression adjustment, and (c) a doubly robust AIPW estimator with a logit propensity score. Covariates: `age`, `agesq`, `educ`, `black`, `hisp`, `married`, `re74`, `re75`, `unem74`, `unem75`.
3. Check overlap: what share of PSID controls have propensity scores below 0.01? Trim to \\(0.01\\le\\widehat e\\le0.99\\) and re-estimate.
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
exp_, obs = wd.data("jtrain2"), wd.data("jtrain3")
X = "age + agesq + educ + black + hisp + married + re74 + re75 + unem74 + unem75"
# TODO: experimental ATE; then on obs: ps = smf.logit(f"train ~ {X}", obs).fit().predict()
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
exp_, obs = wd.data("jtrain2"), wd.data("jtrain3")
exp_["agesq"] = exp_["age"]**2
X = "age + agesq + educ + black + hisp + married + re74 + re75 + unem74 + unem75"
e1 = smf.ols("re78 ~ train", data=exp_).fit(cov_type="HC1"); e2 = smf.ols(f"re78 ~ train + {X}", data=exp_).fit(cov_type="HC1")
print(f"EXPERIMENT   diff in means {e1.params['train']:.3f} (se {e1.bse['train']:.3f});  with covariates {e2.params['train']:.3f}")

def aipw(d):
    ps = smf.logit(f"train ~ {X}", data=d).fit(disp=0).predict(d)
    m1 = smf.ols(f"re78 ~ {X}", data=d[d.train == 1]).fit().predict(d)
    m0 = smf.ols(f"re78 ~ {X}", data=d[d.train == 0]).fit().predict(d)
    D, Y = d["train"], d["re78"]
    ra_att = (Y - m0)[D == 1].mean()                                     # regression adjustment, ATT
    w = ps / (1 - ps)                                                    # doubly robust ATT
    dr_att = ((D * (Y - m0)).sum() - ((1 - D) * w * (Y - m0)).sum()) / D.sum()
    return ps, ra_att, dr_att

naive = obs.loc[obs.train == 1, "re78"].mean() - obs.loc[obs.train == 0, "re78"].mean()
ps, ra, dr = aipw(obs)
print(f"PSID CONTROLS naive {naive:.3f};  regression adjustment ATT {ra:.3f};  doubly robust ATT {dr:.3f}")
print(f"share of PSID controls with propensity < 0.01: {(ps[obs.train == 0] < 0.01).mean():.3f}")
trim = obs[(ps >= 0.01) & (ps <= 0.99)].copy()
_, ra_t, dr_t = aipw(trim)
print(f"TRIMMED (n = {len(trim)}, treated {int(trim.train.sum())}): regression adjustment {ra_t:.3f}; doubly robust {dr_t:.3f}")
""",
notes="""
The experiment gives an effect of about \\$1,800 on 1978 earnings. With PSID controls, the naive comparison is hugely
**negative**: trainees were far poorer than the average PSID household — classic selection bias. Regression adjustment and
the doubly robust estimator move a long way toward the benchmark, but the answer depends on modeling choices, and most
PSID controls have propensity scores near zero: there is almost no overlap. Trimming to the region of common support
changes the estimates noticeably. This is LaLonde's (1986) lesson, refined by Dehejia and Wahba (1999): observational
methods can work only when comparable controls exist, and overlap diagnostics must come first.
"""),

dict(title="Difference-in-differences: an incinerator and house prices",
data="`kielmc` — 321 house sales in North Andover, Massachusetts, in 1978 (before rumors of an incinerator) and 1981 (during construction); `nearinc` = within 3 miles of the site.",
task="""
1. Compute the 2×2 DiD estimate of the incinerator's effect on real house prices `rprice` from the four group means.
2. Estimate it by regression, `rprice ~ nearinc * y81`, with robust standard errors.
3. Add house characteristics (`age`, `agesq`, `intst`, `land`, `area`, `rooms`, `baths`). How does the estimate change, and why?
4. Use `log(rprice)` to express the effect in percent. Which scale makes parallel trends more plausible?
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("kielmc")
means = df.groupby(["nearinc", "y81"])["rprice"].mean().unstack()
# TODO: DiD from means; regression with interaction; add controls
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("kielmc")
mu = df.groupby(["nearinc", "y81"])["rprice"].mean().unstack()
print(mu.round(0).rename(index={0: "far", 1: "near"}, columns={0: "1978", 1: "1981"}))
did = (mu.loc[1, 1] - mu.loc[1, 0]) - (mu.loc[0, 1] - mu.loc[0, 0])
print(f"\\nDiD from means: {did:,.0f} dollars")
r1 = smf.ols("rprice ~ nearinc * y81", data=df).fit(cov_type="HC1")
print(f"regression:       {r1.params['nearinc:y81']:,.0f} (robust se {r1.bse['nearinc:y81']:,.0f})")
ctrl = "age + I(age**2) + intst + land + area + rooms + baths"
r2 = smf.ols(f"rprice ~ nearinc * y81 + {ctrl}", data=df).fit(cov_type="HC1")
print(f"with controls:    {r2.params['nearinc:y81']:,.0f} (robust se {r2.bse['nearinc:y81']:,.0f})")
r3 = smf.ols(f"np.log(rprice) ~ nearinc * y81 + {ctrl}", data=df).fit(cov_type="HC1")
b = r3.params["nearinc:y81"]
print(f"log specification with controls: {b:.3f} -> {100*(np.exp(b)-1):.1f}% (robust se {r3.bse['nearinc:y81']:.3f})")
""",
notes="""
Houses near the site were already cheaper in 1978. The DiD removes that level difference and attributes the change in
the gap — a fall of roughly \\$12,000 in real (1978) dollars — to the incinerator. Without controls the estimate is
imprecise. Adding house characteristics makes it larger and more precise, because the mix of houses sold near the site
changed between the two years. In logs with controls, the effect is about \\(-9\\%\\), but with a standard error of 0.06 it is not statistically
significant: the evidence is suggestive rather than conclusive with 321 sales.
Parallel trends is an assumption about counterfactual price paths; the log scale is often more plausible for prices,
because it allows near and far houses to share common *percentage* appreciation. With only two periods, pre-trends cannot
be checked — a key limitation (Kiel and McClain, 1995; Wooldridge, Example 13.3).
"""),
]

# ---------------------------------------------------------------- Lecture 14
EXERCISES["14-research-replication.md"] = [
dict(title="Wild cluster bootstrap with few clusters",
data="`wagepan` — 545 workers, 1980–1987; the dummies `occ1`–`occ9` give each person-year's occupation (9 groups).",
task="""
Many applied papers have few clusters. Build an `occupation` variable from the dummies `occ1`–`occ9` and use it (9 groups) as the cluster:

1. Regress `lwage` on `union`, `educ`, `exper`, `expersq` and year dummies; report the cluster-robust \\(t\\)-statistic on `union` with 9 occupation clusters.
2. Implement a **wild cluster bootstrap-t** with Rademacher weights and the null \\(\\beta_{union}=0\\) imposed (999 draws, seed 1). Compare p-values.
3. Why can the conventional cluster-robust p-value be misleading here?
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wagepan")
df["occupation"] = df[[f"occ{k}" for k in range(1, 10)]].to_numpy().argmax(1) + 1
f = "lwage ~ union + educ + exper + expersq + C(year)"
# TODO: restricted model without union; y* = fitted_r + v_g * resid_r; refit unrestricted; cluster t each draw
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wagepan").reset_index(drop=True)
df["occupation"] = df[[f"occ{k}" for k in range(1, 10)]].to_numpy().argmax(1) + 1
f = "lwage ~ union + educ + exper + expersq + C(year)"
g = df["occupation"].to_numpy(); G = len(np.unique(g))
fit = lambda d: smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": g, "use_correction": True})
base = fit(df); t0 = base.tvalues["union"]
from scipy import stats
print(f"{G} clusters: union = {base.params['union']:.4f}, cluster t = {t0:.2f}, p (normal) = {2*stats.norm.sf(abs(t0)):.4f}, p (t_{G-1}) = {2*stats.t.sf(abs(t0), G-1):.4f}")

r = smf.ols(f.replace("union + ", ""), data=df).fit()
rng = np.random.default_rng(1); ts = []
groups = np.unique(g)
for _ in range(999):
    v = dict(zip(groups, rng.choice([-1.0, 1.0], size=G)))
    d = df.copy(); d["lwage"] = r.fittedvalues + r.resid * np.vectorize(v.get)(g)
    ts.append(fit(d).tvalues["union"])
p_wcb = np.mean(np.abs(ts) >= abs(t0))
print(f"wild cluster bootstrap-t p-value: {p_wcb:.3f}   (with G = {G}, Rademacher weights give only 2^{G} = {2**G} distinct draws)")
""",
notes="""
With only 9 clusters, the cluster-robust variance estimator is noisy and biased downward, so normal (or even \\(t_{G-1}\\))
critical values overstate significance. The wild cluster bootstrap with the null imposed gives a more honest p-value —
here noticeably larger than the conventional one. With very few clusters, Rademacher weights allow only \\(2^G\\) distinct
bootstrap samples; Webb's six-point weights are recommended for \\(G<12\\). The deeper question, as in Lecture 4, is
whether occupation is the right clustering level for this design at all (Abadie et al., 2023).
"""),

dict(title="Multiple outcomes and multiple testing in the NSW experiment",
data="`jtrain2` — NSW randomized experiment.",
task="""
1. Estimate the effect of training on six outcomes: `re78`, `unem78`, an indicator of positive 1978 earnings, earnings above \\$5,000, above \\$10,000, and the change `re78 − re75`.
2. Report unadjusted p-values and Holm, Bonferroni and Benjamini–Hochberg adjustments.
3. Build a standardized summary index (Kling, Liebman and Katz, 2007): standardize each outcome by the control-group mean and SD (signing `unem78` so higher is better), average them, and test the treatment effect on the index.
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
d = wd.data("jtrain2")
d["pos78"] = (d.re78 > 0).astype(int); d["gt5"] = (d.re78 > 5).astype(int); d["gt10"] = (d.re78 > 10).astype(int); d["chg"] = d.re78 - d.re75
# TODO: loop over outcomes, collect robust p-values, apply multipletests
""",
solution="""
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
d = wd.data("jtrain2")
d["pos78"] = (d.re78 > 0).astype(int); d["gt5"] = (d.re78 > 5).astype(int)
d["gt10"] = (d.re78 > 10).astype(int); d["chg"] = d.re78 - d.re75
outs = ["re78", "unem78", "pos78", "gt5", "gt10", "chg"]
rows = []
for y in outs:
    r = smf.ols(f"{y} ~ train", data=d).fit(cov_type="HC1")
    rows.append((y, r.params["train"], r.pvalues["train"]))
tab = pd.DataFrame(rows, columns=["outcome", "effect", "p"]).set_index("outcome")
for meth, lab in (("bonferroni", "Bonferroni"), ("holm", "Holm"), ("fdr_bh", "BH (FDR)")):
    tab[lab] = multipletests(tab["p"], method=meth)[1]
print(tab.round(4).to_string())

c = d[d.train == 0]; z = []
for y in outs:
    s = (d[y] - c[y].mean()) / c[y].std()
    z.append(-s if y == "unem78" else s)
d["index"] = np.mean(z, axis=0)
ri = smf.ols("index ~ train", data=d).fit(cov_type="HC1")
print(f"\\nsummary index: effect {ri.params['train']:.3f} control-SD units (robust se {ri.bse['train']:.3f}, p = {ri.pvalues['train']:.4f})")
""",
notes="""
Four of the six outcomes are individually significant at 5%. After adjusting for six tests, only 1978 earnings survives
Bonferroni and Holm (Holm's adjusted p-values are smaller, but here they cross 5% at the same point), while the
Benjamini–Hochberg FDR procedure keeps four. Notice that `unem78` and `pos78` give identical results: they are almost the
same variable (positive earnings = employed), which is a reminder to define the outcome family carefully before testing. The summary index answers the
question "did training improve labor-market outcomes overall?" with a single test and more power than any one outcome.
Pre-specify which family of outcomes will be adjusted and which index will be primary *before* seeing the data
(Lecture 14, §1).
"""),
]
