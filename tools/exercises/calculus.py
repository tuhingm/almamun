INTRO = """
These exercises apply the calculus of this lecture to **real economic data** in Python, mixing symbolic (`sympy`),
numerical (`scipy`) and statistical (`statsmodels`) tools. Install once with
`pip install sympy scipy numpy pandas statsmodels wooldridge`. Try the starter code first, then open the solution.
Every printed result below comes from running the solution code exactly as shown.
"""

EXERCISES = {}

EXERCISES["01-limits-continuity.md"] = [
dict(title="From discrete to continuous compounding with real interest rates",
data="`intqrt` — U.S. 3-month Treasury bill yields (`r3`, % per year), quarterly.",
task="""
1. Take the average T-bill yield \\(r\\) in the sample. Compute the growth factor of \\$1 after one year, \\((1+r/n)^n\\), for \\(n=1,2,4,12,365,10^6\\).
2. Show numerically that the sequence converges to \\(e^r\\), and that the error behaves like \\(r^2e^r/(2n)\\).
3. Use `sympy` to prove the limit symbolically.
4. Why does the choice between annual and continuous compounding matter more when rates are high (as in 1981)?
""",
starter="""
import numpy as np, sympy as sp, wooldridge as wd
r = wd.data("intqrt")["r3"].mean() / 100
for n in (1, 2, 4, 12, 365, 10**6):
    pass  # TODO: (1 + r/n)**n and the gap to np.exp(r)
x, n = sp.symbols("x n", positive=True)
# TODO: sp.limit((1 + x/n)**n, n, sp.oo)
""",
solution="""
import numpy as np, sympy as sp, wooldridge as wd
d = wd.data("intqrt"); r = d["r3"].mean() / 100
print(f"average 3-month T-bill yield: {100*r:.3f}% per year")
for n in (1, 2, 4, 12, 365, 10**6):
    g = (1 + r / n)**n
    print(f"n = {n:>7}: (1+r/n)^n = {g:.8f}   gap to e^r = {np.exp(r) - g:.2e}   approx r^2 e^r/(2n) = {r**2*np.exp(r)/(2*n):.2e}")
x, n = sp.symbols("x n", positive=True)
print("sympy limit:", sp.limit((1 + x / n)**n, n, sp.oo))
rmax = d["r3"].max() / 100
print(f"at the sample's peak yield {100*rmax:.2f}%: annual {1+rmax:.5f} vs continuous {np.exp(rmax):.5f} (gap {100*(np.exp(rmax)-1-rmax):.2f} pp)")
""",
notes="""At the sample's average yield of 6.45%, the growth factor rises from 1.0645 (annual compounding) toward \\(e^{0.0645}=1.0666\\).
The gap shrinks in proportion to \\(1/n\\), exactly as the second-order expansion \\(r^2e^r/(2n)\\) predicts: doubling \\(n\\)
halves the error. `sympy` confirms \\(\\lim_{n\\to\\infty}(1+x/n)^n=e^x\\). The difference between annual and continuous
compounding is of order \\(r^2/2\\), so it grows quadratically with the rate: at the 16.5% peak of 1981 it was 1.4 percentage
points a year — enough to matter when quoting or comparing yields.
"""),

dict(title="Arc elasticity converges to point elasticity: demand for eco-labeled apples",
data="`apple` — 660 households in a 1990s survey that randomly assigned prices of eco-labeled (`ecoprc`) and regular (`regprc`) apples and recorded pounds demanded (`ecolbs`).",
task="""
1. Estimate an exponential demand curve \\(E[q\\mid p]=\\exp(\\beta_0+\\beta_1p_{eco}+\\beta_2p_{reg})\\) by Poisson regression (which handles the many zero purchases).
2. At the mean prices, compute the **arc elasticity** \\(\\frac{\\Delta q/q}{\\Delta p/p}\\) for price changes of 20, 10, 1 and 0.1 cents. Show that it converges to the point elasticity \\(\\beta_1p\\) as \\(\\Delta p\\to0\\), and that the error is roughly proportional to \\(\\Delta p\\).
3. Why would the same exercise with a **linear** demand curve show no convergence at all?
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
a = wd.data("apple")
r = smf.poisson("ecolbs ~ ecoprc + regprc", data=a).fit()
q = lambda p: np.exp(r.params["Intercept"] + r.params["ecoprc"] * p + r.params["regprc"] * a["regprc"].mean())
# TODO: arc elasticities at p0 = a.ecoprc.mean() for dp in (0.2, 0.1, 0.01, 0.001); point elasticity b1 * p0
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
a = wd.data("apple")
r = smf.poisson("ecolbs ~ ecoprc + regprc", data=a).fit(disp=0, cov_type="HC0")
b0, b1, b2 = r.params; p0, preg = a["ecoprc"].mean(), a["regprc"].mean()
q = lambda p: np.exp(b0 + b1 * p + b2 * preg)
print(f"E[q|p] = exp({b0:.3f} {b1:+.3f} p_eco {b2:+.3f} p_reg)   (robust se on p_eco {r.bse['ecoprc']:.3f})")
point = b1 * p0
for dp in (0.20, 0.10, 0.01, 0.001):
    arc = ((q(p0 + dp) - q(p0)) / q(p0)) / (dp / p0)
    print(f"  dp = {dp:5.3f}: arc elasticity {arc:.5f}   error {arc - point:+.5f}   error/dp {(arc - point)/dp:+.3f}")
print(f"point elasticity b1 * p = {point:.5f}")
lin = smf.ols("ecolbs ~ ecoprc + regprc", data=a).fit()
ql = lambda p: lin.params["Intercept"] + lin.params["ecoprc"] * p + lin.params["regprc"] * preg
print("linear demand, arc elasticity for dp = 0.2 and 0.001:", round(((ql(p0+.2)-ql(p0))/ql(p0))/(.2/p0), 5), round(((ql(p0+.001)-ql(p0))/ql(p0))/(.001/p0), 5))
""",
notes="""At mean prices, the point elasticity of demand for eco-labeled apples is about \\(-2.2\\): demand is elastic. The arc
elasticity approaches it as the price step shrinks, and the error divided by \\(\\Delta p\\) settles near a constant: the error
is first order in \\(\\Delta p\\), as a Taylor expansion of \\(e^{\\beta_1\\Delta p}\\) predicts. For a linear demand curve, the
one-sided arc elasticity \\((\\Delta q/q)/(\\Delta p/p)=\\beta_1p/q\\) is exact for every step size, so there is nothing to
converge — the last line shows the same number for 20 cents and 0.1 cent. Prices were randomized in this survey, so the
slope can be read as a causal demand response for stated purchases.
"""),
]

EXERCISES["02-sequences-series.md"] = [
dict(title="Log differences versus percent changes: a Taylor series for growth rates",
data="`macrodata` (statsmodels) — U.S. quarterly real GDP and CPI, 1959–2009.",
task="""
Economists often measure growth as \\(\\Delta\\log y\\) instead of \\(\\Delta y/y\\). Since \\(\\log(1+g)=g-g^2/2+g^3/3-\\dots\\),
the two differ at second order.

1. Compute quarterly percent changes and log differences of real GDP and of the CPI. Report the largest gap between them.
2. Show that adding the second-order term, \\(g-g^2/2\\), removes most of the gap.
3. Annualize quarterly growth in two ways — \\(4\\times\\) the quarterly rate and \\((1+g)^4-1\\) — and compare for the most extreme quarter.
""",
starter="""
import numpy as np, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
y = m["realgdp"].to_numpy()
g = y[1:] / y[:-1] - 1; lg = np.diff(np.log(y))
# TODO: max |g - lg|; max |g - g**2/2 - lg|; annualization
""",
solution="""
import numpy as np, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
for name in ("realgdp", "cpi"):
    y = m[name].to_numpy(); g = y[1:] / y[:-1] - 1; lg = np.diff(np.log(y))
    k = np.argmax(np.abs(g))
    print(f"{name}: largest |quarterly change| {100*g[k]:.3f}% (log diff {100*lg[k]:.3f}%)")
    print(f"   max |g - log| = {np.max(np.abs(g - lg))*100:.4f} pp;  max |g - g^2/2 - log| = {np.max(np.abs(g - g**2/2 - lg))*100:.5f} pp;  max |g - g^2/2 + g^3/3 - log| = {np.max(np.abs(g - g**2/2 + g**3/3 - lg))*100:.6f} pp")
    print(f"   annualized at that quarter: 4g = {400*g[k]:.2f}%,  (1+g)^4 - 1 = {100*((1+g[k])**4-1):.2f}%,  4*log diff = {400*lg[k]:.2f}%")
""",
notes="""Quarterly changes are small, so log differences and percent changes differ by at most 0.08 percentage points. Adding the
\\(-g^2/2\\) term cuts the gap by a factor of about 40, and the cubic term by another 30 — the Taylor series converges fast
for small \\(g\\). Annualization matters more: for GDP's strongest quarter, simple multiplication by four gives 15.7%, true
compounding 16.7%, and four times the log difference 15.4%. Always state which convention a reported growth rate uses.
"""),

dict(title="Present values as geometric series: pricing a perpetuity and a bond",
data="`intqrt` — U.S. Treasury yields; we use the sample's average 12-month yield `r12` as a discount rate.",
task="""
1. A perpetuity pays \\$100 per year forever. Write its value as a geometric series and sum it with `sympy`.
2. Compute the partial sums for 10, 30 and 100 years. How many years of payments make up 90% of the perpetuity's value?
3. Price a 10-year bond with a 5% annual coupon and face value \\$1,000 at the average yield, using the closed form for a finite geometric series. Check it with a direct sum.
""",
starter="""
import numpy as np, sympy as sp, wooldridge as wd
r = wd.data("intqrt")["r12"].mean() / 100
t, R = sp.symbols("t R", positive=True)
# TODO: sp.summation(100 / (1 + R)**t, (t, 1, sp.oo)); partial sums; bond price
""",
solution="""
import numpy as np, sympy as sp, wooldridge as wd
r = wd.data("intqrt")["r12"].mean() / 100
t, R = sp.symbols("t R", positive=True)
pv = sp.simplify(sp.summation(100 / (1 + R)**t, (t, 1, sp.oo)))
print(f"discount rate r = {100*r:.3f}%;  symbolic perpetuity value: {pv}  ->  ${float(pv.subs(R, r)):,.2f}")
total = 100 / r
for T in (10, 30, 100):
    s = sum(100 / (1 + r)**k for k in range(1, T + 1))
    print(f"  first {T:3d} years: ${s:,.2f} ({s/total:.1%} of the perpetuity)")
T90 = np.log(10) / np.log(1 + r)
print(f"years needed for 90% of the value: {T90:.1f}")
c, F, n = 50, 1000, 10
closed = c * (1 - (1 + r)**-n) / r + F / (1 + r)**n
direct = sum(c / (1 + r)**k for k in range(1, n + 1)) + F / (1 + r)**n
print(f"10-year 5% coupon bond: closed form ${closed:,.2f}, direct sum ${direct:,.2f}")
""",
notes="""`sympy` sums the geometric series to \\(100/R\\), about \\$1,444 at a 6.9% discount rate. The first 10 years contribute less
than half of that value, and it takes about 34 years of payments to reach 90%: distant payments matter a lot when rates are
moderate, which is why long-duration assets are so sensitive to interest rates. The closed form for the finite geometric
series prices the 5% coupon bond at \\$864, below par because the coupon is below the yield.
"""),
]

EXERCISES["03-differentiation.md"] = [
dict(title="Marginal effects and elasticities from a fitted R&D model",
data="`rdchem` — 32 U.S. chemical firms: R&D spending `rd` and `sales` (millions of dollars).",
task="""
1. Fit \\(rd=\\beta_0+\\beta_1 sales+\\beta_2 sales^2+u\\).
2. Differentiate the fitted function symbolically with `sympy` to get the marginal effect \\(d\\,rd/d\\,sales\\) and the elasticity \\(\\frac{d\\,rd}{d\\,sales}\\frac{sales}{rd}\\).
3. Evaluate both at the median firm and at the largest firm. Where does the marginal effect turn negative, and is that inside the data?
4. Check the symbolic derivative against a central finite difference.
""",
starter="""
import sympy as sp, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("rdchem")
r = smf.ols("rd ~ sales + I(sales**2)", data=d).fit()
S = sp.symbols("S")
f = r.params["Intercept"] + r.params["sales"] * S + r.params["I(sales ** 2)"] * S**2
# TODO: sp.diff(f, S); elasticity; solve df/dS = 0
""",
solution="""
import numpy as np, sympy as sp, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("rdchem")
r = smf.ols("rd ~ sales + I(sales**2)", data=d).fit()
S = sp.symbols("S", positive=True)
f = r.params["Intercept"] + r.params["sales"] * S + r.params["I(sales ** 2)"] * S**2
fp = sp.diff(f, S); el = sp.simplify(fp * S / f)
print("fitted rd(S) =", sp.N(f, 4)); print("rd'(S) =", sp.N(fp, 4))
for lab, s in (("median firm", d["sales"].median()), ("largest firm", d["sales"].max())):
    print(f"  {lab:12s} (sales {s:,.0f}): marginal effect {float(fp.subs(S, s)):.4f}, elasticity {float(el.subs(S, s)):.3f}")
peak = float(sp.solve(sp.Eq(fp, 0), S)[0])
print(f"marginal effect turns negative at sales = {peak:,.0f}; firms above it: {(d['sales'] > peak).sum()} of {len(d)}")
fn = sp.lambdify(S, f); h = 1e-3; s0 = d["sales"].median()
print(f"finite difference at median: {(fn(s0 + h) - fn(s0 - h)) / (2*h):.6f}  vs symbolic {float(fp.subs(S, s0)):.6f}")
""",
notes="""The fitted R&D function is slightly concave. The marginal effect of \\$1 of sales on R&D falls from about 6 cents at the
median firm to about 1 cent at the largest firm, and the elasticity falls from about 2 to 0.3. The marginal effect would
turn negative only at sales of about \\$49 billion, beyond every firm in the data — so the "turning point" is an artefact of
the quadratic, not a finding. The symbolic derivative agrees with the finite difference to six decimals. The elasticity
above 1 at the median is driven by the negative intercept and should not be over-interpreted with 32 firms.
"""),

dict(title="Newton's method for an internal rate of return",
data="`macrodata` — annual averages of U.S. real GDP, 2000–2008, used to give a stylized project realistic cash flows that grow with the economy.",
task="""
A project costs \\$1,000 today and returns cash flows \\(C_1,\\dots,C_9\\) proportional to annual U.S. real GDP growth paths. The IRR solves \\(NPV(r)=-1000+\\sum_t C_t/(1+r)^t=0\\).

1. Derive \\(NPV'(r)\\) analytically.
2. Implement Newton's method \\(r_{k+1}=r_k-NPV(r_k)/NPV'(r_k)\\) from \\(r_0=0\\). Print each iteration and show quadratic convergence (the number of correct digits roughly doubles).
3. Check with `scipy.optimize.brentq`.
""",
starter="""
import numpy as np, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
y = m.groupby("year")["realgdp"].mean().loc[2000:2008].to_numpy()
C = 150 * y / y[0]                       # cash flows that grow with GDP
npv = lambda r: -1000 + sum(c / (1 + r)**t for t, c in enumerate(C, 1))
# TODO: dnpv = lambda r: -sum(t * c / (1 + r)**(t+1) ...); Newton loop
""",
solution="""
import numpy as np, statsmodels.api as sm
from scipy import optimize
m = sm.datasets.macrodata.load_pandas().data
y = m.groupby("year")["realgdp"].mean().loc[2000:2008].to_numpy()
C = 150 * y / y[0]
npv = lambda r: -1000 + sum(c / (1 + r)**t for t, c in enumerate(C, 1))
dnpv = lambda r: -sum(t * c / (1 + r)**(t + 1) for t, c in enumerate(C, 1))
print("cash flows:", C.round(1))
root = optimize.brentq(npv, -0.5, 1.0, xtol=1e-15)
rk = 0.0
for k in range(6):
    print(f"iteration {k}: r = {rk:.15f}   error = {abs(rk - root):.2e}")
    rk = rk - npv(rk) / dnpv(rk)
print(f"brentq IRR = {root:.15f}")
""",
notes="""Newton's method converges to the IRR of about 8.27% in five iterations. The errors fall from \\(10^{-2}\\) to \\(10^{-3}\\),
\\(10^{-6}\\), \\(10^{-11}\\) and machine precision: the number of correct digits roughly doubles each step, which is
quadratic convergence. It works so well here because \\(NPV(r)\\) is smooth, decreasing and convex. With cash flows that
change sign more than once, NPV can have several roots and Newton's method can converge to any of them, or fail;
bracketing methods such as Brent's are safer.
"""),
]

EXERCISES["04-taylor-convexity-optimization.md"] = [
dict(title="The Arrow–Pratt approximation with real stock returns",
data="`volat` — monthly S&P 500 returns (`rsp500`, % per year) and 3-month T-bill yields (`i3`), 1947–1993.",
task="""
An investor with CRRA utility \\(u(w)=w^{1-\\gamma}/(1-\\gamma)\\) puts all wealth in the S&P 500 for one month.

1. Using the empirical distribution of monthly gross returns \\(R\\), compute the **exact** certainty equivalent \\(CE=u^{-1}(E[u(R)])\\) and risk premium \\(E[R]-CE\\) for \\(\\gamma=2,5,10\\).
2. Compare with the second-order Taylor (Arrow–Pratt) approximation \\(\\pi\\approx\\tfrac12\\gamma\\operatorname{Var}(R)/E[R]\\) (relative risk aversion form).
3. When does the approximation break down, and why (think of the third-order term and skewness)?
""",
starter="""
import numpy as np, wooldridge as wd
v = wd.data("volat").dropna(subset=["rsp500"])
R = 1 + v["rsp500"].to_numpy() / 1200         # monthly gross return
u = lambda w, g: w**(1 - g) / (1 - g)
# TODO: CE = ((1-g) * mean(u(R)))**(1/(1-g)); compare with 0.5*g*var/mean
""",
solution="""
import numpy as np, wooldridge as wd
from scipy import stats
v = wd.data("volat").dropna(subset=["rsp500"])
R = 1 + v["rsp500"].to_numpy() / 1200
m, var = R.mean(), R.var()
print(f"monthly gross return: mean {m:.5f}, sd {np.sqrt(var):.5f}, skewness {stats.skew(R):.3f}")
for g in (2, 5, 10, 50):
    EU = np.mean(R**(1 - g) / (1 - g))
    ce = ((1 - g) * EU)**(1 / (1 - g))
    exact, approx = m - ce, 0.5 * g * var / m
    print(f"gamma = {g:2d}: exact premium {1e4*exact:7.3f} bp/month, Arrow-Pratt {1e4*approx:7.3f} bp, relative error {approx/exact - 1:+.2%}")
""",
notes="""For moderate risk aversion, the Arrow–Pratt approximation is excellent: the error is under 2% at \\(\\gamma=2\\) and under
8% at \\(\\gamma=10\\). It always understates the premium here, because monthly stock returns are **negatively skewed**:
the third-order Taylor term, which a CRRA investor dislikes, adds to the premium. At \\(\\gamma=50\\) the approximation
misses almost 40%, because higher-order terms grow with \\(\\gamma\\) and the distribution's tails dominate. Second-order
approximations are a local tool.
"""),

dict(title="A revenue-maximizing price from an estimated demand curve",
data="`apple` — randomized prices and demand for eco-labeled apples.",
task="""
1. Using the linear demand from Lecture 1's exercise, write revenue \\(TR(p)=p\\,q(p)\\) at the mean regular-apple price.
2. Find the revenue-maximizing price from the first-order condition, and verify the second-order condition (concavity).
3. Propagate estimation uncertainty: compute a delta-method standard error for the optimal price.
4. Is the optimal price inside the range of prices used in the survey?
""",
starter="""
import sympy as sp, wooldridge as wd, statsmodels.formula.api as smf
a = wd.data("apple")
r = smf.ols("ecolbs ~ ecoprc + regprc", data=a).fit(cov_type="HC1")
p, b0, b1, b2, pr = sp.symbols("p b0 b1 b2 pr")
TR = p * (b0 + b1 * p + b2 * pr)
# TODO: solve dTR/dp = 0 for p; second derivative; gradient of p* w.r.t. (b0, b1, b2)
""",
solution="""
import numpy as np, sympy as sp, wooldridge as wd, statsmodels.formula.api as smf
a = wd.data("apple")
r = smf.ols("ecolbs ~ ecoprc + regprc", data=a).fit(cov_type="HC1")
p, b0, b1, b2, pr = sp.symbols("p b0 b1 b2 pr")
TR = p * (b0 + b1 * p + b2 * pr)
pstar = sp.solve(sp.diff(TR, p), p)[0]
print("p* =", pstar, "   d2TR/dp2 =", sp.diff(TR, p, 2))
vals = {b0: r.params["Intercept"], b1: r.params["ecoprc"], b2: r.params["regprc"], pr: a["regprc"].mean()}
ps = float(pstar.subs(vals))
grad = np.array([float(sp.diff(pstar, s).subs(vals)) for s in (b0, b1, b2)])
se = np.sqrt(grad @ r.cov_params().to_numpy() @ grad)
print(f"revenue-maximizing eco price: ${ps:.3f} (delta-method se {se:.3f}); second derivative = 2*b1 = {2*vals[b1]:.3f} < 0")
print(f"revenue at p*: {float(TR.subs(vals).subs(p, ps)):.3f} $/household;  survey prices ranged ${a['ecoprc'].min():.2f}-${a['ecoprc'].max():.2f}")
""",
notes="""With linear demand, revenue is a concave quadratic in price (second derivative \\(2\\beta_1<0\\)), so the first-order
condition gives the unique maximum: about 79 cents per pound, where the demand elasticity equals \\(-1\\) (compare
Exercise 1.2). The delta-method standard error of about 4 cents reflects uncertainty in the estimated demand curve. The
optimal price is inside the range of prices the survey randomized, so we are not extrapolating.
"""),
]

EXERCISES["05-integration-fundamental-theorem.md"] = [
dict(title="Consumer surplus as an integral under an estimated demand curve",
data="`apple` — randomized prices and demand for eco-labeled apples.",
task="""
1. With the estimated linear demand (regular price at its mean), find the choke price where demand reaches zero.
2. Compute consumer surplus at the mean eco price, \\(CS=\\int_{p_0}^{p_{choke}}q(p)\\,dp\\), symbolically and numerically (`scipy.integrate.quad`).
3. Compute the change in consumer surplus if the eco price rises by 20 cents, using the fundamental theorem of calculus, and compare with the approximation \\(-q(p_0)\\Delta p\\).
""",
starter="""
import sympy as sp, wooldridge as wd, statsmodels.formula.api as smf
from scipy import integrate
a = wd.data("apple"); r = smf.ols("ecolbs ~ ecoprc + regprc", data=a).fit()
p = sp.symbols("p")
q = r.params["Intercept"] + r.params["ecoprc"] * p + r.params["regprc"] * a["regprc"].mean()
# TODO: choke = sp.solve(q, p)[0]; CS = sp.integrate(q, (p, p0, choke))
""",
solution="""
import numpy as np, sympy as sp, wooldridge as wd, statsmodels.formula.api as smf
from scipy import integrate
a = wd.data("apple"); r = smf.ols("ecolbs ~ ecoprc + regprc", data=a).fit()
p = sp.symbols("p")
q = r.params["Intercept"] + r.params["ecoprc"] * p + r.params["regprc"] * a["regprc"].mean()
p0 = a["ecoprc"].mean(); choke = float(sp.solve(q, p)[0])
cs_sym = float(sp.integrate(q, (p, p0, choke)))
cs_num = integrate.quad(sp.lambdify(p, q), p0, choke)[0]
print(f"choke price ${choke:.3f}; demand at mean price ${p0:.3f}: {float(q.subs(p, p0)):.3f} lbs")
print(f"consumer surplus: symbolic {cs_sym:.5f}, numerical {cs_num:.5f} dollars per household")
dCS = -float(sp.integrate(q, (p, p0, p0 + 0.20)))
print(f"20-cent price rise: exact change in CS {dCS:.5f};  first-order approximation -q(p0)*dp = {-float(q.subs(p, p0))*0.20:.5f}")
""",
notes="""The choke price, where predicted demand reaches zero, is about \\$1.59 — the top of the survey's price range. Symbolic and
numerical integration agree: consumer surplus at the mean price is about 37 cents per household. A 20-cent price rise
lowers surplus by about 24 cents. The first-order approximation \\(-q\\Delta p\\) overstates the loss (29 cents), because it
ignores that households buy less as the price rises — the triangle term \\(\\tfrac12\\Delta q\\Delta p\\), which is large here
because demand is elastic.
"""),

dict(title="The Lorenz curve and Gini coefficient by numerical integration",
data="`401ksubs` — 9,275 U.S. households (1991 SIPP): family income `inc` (thousands of dollars).",
task="""
1. Construct the Lorenz curve \\(L(p)\\): the share of total income received by the poorest fraction \\(p\\) of households.
2. Compute the Gini coefficient \\(G=1-2\\int_0^1L(p)\\,dp\\) with the trapezoidal rule.
3. Check it against the mean-difference formula \\(G=\\frac{E|X-Y|}{2E[X]}\\) computed from the sorted data.
4. What income shares do the bottom 50% and the top 10% receive?
""",
starter="""
import numpy as np, wooldridge as wd
x = np.sort(wd.data("401ksubs")["inc"].to_numpy()); n = len(x)
L = np.concatenate([[0], np.cumsum(x) / x.sum()]); P = np.arange(n + 1) / n
# TODO: gini = 1 - 2 * np.trapezoid(L, P)
""",
solution="""
import numpy as np, wooldridge as wd
x = np.sort(wd.data("401ksubs")["inc"].to_numpy()); n = len(x)
L = np.concatenate([[0], np.cumsum(x) / x.sum()]); P = np.arange(n + 1) / n
g_int = 1 - 2 * np.trapezoid(L, P)
i = np.arange(1, n + 1)
g_md = np.sum((2 * i - n - 1) * x) / (n * x.sum())          # mean-difference formula on sorted data
print(f"Gini by integrating the Lorenz curve: {g_int:.5f};  by the mean-difference formula: {g_md:.5f}")
print(f"income share of bottom 50%: {L[n//2]:.3f};  top 10%: {1 - L[int(0.9*n)]:.3f}")
print(f"mean income ${x.mean():.1f}k, median ${np.median(x):.1f}k")
""",
notes="""Integrating the Lorenz curve with the trapezoidal rule gives a Gini coefficient of about 0.32, identical to the
mean-difference formula — the two are algebraically the same for a step-function Lorenz curve. The poorest half of
households receive 28% of income, and the richest tenth 24%. These are 1991 SIPP data on households **eligible for 401(k)
plans**, with incomes capped at \\$200k, so inequality is lower than in the full U.S. income distribution.
"""),
]

EXERCISES["06-multivariable-functions-limits.md"] = [
dict(title="The CES production function and its Cobb–Douglas limit",
data="We calibrate to the U.S. labor share of about 2/3 (capital share \\(\\alpha=1/3\\)) and evaluate at the capital–output ratio implied by `macrodata` investment data.",
task="""
The CES function is \\(Y=\\big(\\alpha K^{\\rho}+(1-\\alpha)L^{\\rho}\\big)^{1/\\rho}\\), with elasticity of substitution \\(\\sigma=1/(1-\\rho)\\).

1. Show with `sympy` that \\(\\lim_{\\rho\\to0}Y=K^{\\alpha}L^{1-\\alpha}\\) (take logs and use L'Hôpital's rule).
2. Evaluate the CES function at \\(K=3\\), \\(L=1\\) for \\(\\rho\\) approaching 0 from both sides, and show convergence to the Cobb–Douglas value.
3. Compute the capital share \\(\\frac{\\partial Y}{\\partial K}\\frac{K}{Y}\\) as \\(\\rho\\) varies. Why does the Cobb–Douglas case imply a constant factor share, and what does the U.S. evidence on a falling labor share suggest about \\(\\sigma\\)?
""",
starter="""
import sympy as sp
K, L, a, r = sp.symbols("K L alpha rho", positive=True)
logY = sp.log(a * K**r + (1 - a) * L**r) / r
# TODO: sp.limit(logY, r, 0)
""",
solution="""
import numpy as np, sympy as sp
K, L, a = sp.symbols("K L alpha", positive=True); r = sp.symbols("rho", real=True)
logY = sp.log(a * K**r + (1 - a) * L**r) / r
lim = sp.simplify(sp.limit(logY, r, 0))
print("lim_{rho->0} log Y =", lim, " -> Y = K^alpha L^(1-alpha)")
ces = lambda rho, k=3.0, l=1.0, al=1/3: (al * k**rho + (1 - al) * l**rho)**(1 / rho)
print(f"Cobb-Douglas value at K=3, L=1: {3**(1/3):.8f}")
for rho in (-0.5, -0.1, -0.01, -0.001, 0.001, 0.01, 0.1, 0.5):
    Y = ces(rho); share = (1/3) * 3**rho / ((1/3) * 3**rho + 2/3)
    print(f"  rho = {rho:+.3f} (sigma = {1/(1-rho):.3f}): Y = {Y:.8f}, capital share = {share:.4f}")
""",
notes="""`sympy` evaluates the 0/0 limit of \\(\\log Y\\) and returns the Cobb–Douglas form, and the numerical values converge to
\\(3^{1/3}\\) from both sides. The capital share is constant at \\(\\alpha=1/3\\) only in the Cobb–Douglas case: with
\\(\\sigma<1\\) (\\(\\rho<0\\)), a high capital–labor ratio lowers capital's share, and with \\(\\sigma>1\\) it raises it. A falling
U.S. labor share alongside capital deepening is therefore consistent with \\(\\sigma>1\\) — one explanation debated in the
literature (Karabarbounis and Neiman, 2014); others stress markups and automation.
"""),

dict(title="Level sets of an estimated two-input investment function",
data="`grunfeld` (statsmodels) — investment, firm market value and capital stock of 11 large U.S. firms, 1935–1954.",
task="""
1. Estimate the log-linear investment function \\(\\log I=\\beta_0+\\beta_1\\log V+\\beta_2\\log K\\) (pooled OLS).
2. The fitted function \\(I(V,K)=e^{\\beta_0}V^{\\beta_1}K^{\\beta_2}\\) is a function of two variables. Find the level curve on which predicted investment equals the sample median, solving for \\(K\\) as a function of \\(V\\).
3. Compute the slope \\(dK/dV\\) of that level curve at the median firm-year (the marginal rate of substitution between market value and capital in generating investment).
""",
starter="""
import numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
g = sm.datasets.grunfeld.load_pandas().data.query("invest > 0 and capital > 0")
r = smf.ols("np.log(invest) ~ np.log(value) + np.log(capital)", data=g).fit()
# TODO: K(V) on the level set: K = (I0 / (e^b0 V^b1))^(1/b2); slope -(b1 K)/(b2 V)
""",
solution="""
import numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
g = sm.datasets.grunfeld.load_pandas().data.query("invest > 0 and capital > 0")
r = smf.ols("np.log(invest) ~ np.log(value) + np.log(capital)", data=g).fit()
b0, b1, b2 = r.params
print(f"log I = {b0:.3f} + {b1:.3f} log V + {b2:.3f} log K   (R2 {r.rsquared:.3f})")
I0, V0, K0 = g["invest"].median(), g["value"].median(), g["capital"].median()
Kcurve = lambda V: (I0 / (np.exp(b0) * V**b1))**(1 / b2)
for V in (0.5 * V0, V0, 2 * V0):
    print(f"  on the level set I = {I0:.1f}: V = {V:8.1f} -> K = {Kcurve(V):8.1f}")
print(f"slope dK/dV of the level curve at V = {V0:.1f}: {-(b1 * Kcurve(V0)) / (b2 * V0):.4f}")
""",
notes="""The fitted function explains 88% of the variation in log investment. On the level curve through the median investment,
doubling market value lets predicted investment stay constant with much less capital: the curve is steep and convex, as for
any Cobb–Douglas level set. The slope at the median value says that \\$1 more market value substitutes for about \\$0.87
less capital in generating the same predicted investment. This is a descriptive relation across firms, not a production
technology.
"""),
]

EXERCISES["07-partial-derivatives-chain-rule.md"] = [
dict(title="Growth accounting with the chain rule",
data="`macrodata` — U.S. quarterly real GDP, population and CPI.",
task="""
1. Use the chain rule on \\(\\log Y=\\log(Y/N)+\\log N\\) to decompose real GDP growth into per-capita growth and population growth, decade by decade.
2. Similarly decompose **nominal** GDP growth into real growth and inflation (CPI-based), using \\(\\log(PY)=\\log P+\\log Y\\).
3. Verify that the decompositions add up exactly in logs, and show the small error when using percent changes instead.
""",
starter="""
import numpy as np, pandas as pd, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
# TODO: annual averages; log differences of Y, Y/N, N, P; group by decade
""",
solution="""
import numpy as np, pandas as pd, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
a = m.groupby("year")[["realgdp", "pop", "cpi"]].mean().loc[1960:2008]
lg = np.log(a).diff().dropna() * 100
lg["percap"] = lg["realgdp"] - lg["pop"]; lg["nominal"] = lg["realgdp"] + lg["cpi"]
lg["decade"] = (lg.index // 10) * 10
t = lg.groupby("decade")[["realgdp", "percap", "pop", "cpi", "nominal"]].mean().round(2)
print("average annual log growth (%), by decade:"); print(t.to_string())
pc = a.pct_change().dropna() * 100
err = (pc["realgdp"] - ((1 + pc["realgdp"]/100) / (1 + pc["pop"]/100) - 1) * 100 - pc["pop"]).abs().max()
print(f"\\nmax error if percent changes are simply added (per-capita + population = total): {err:.3f} pp")
""",
notes="""In logs, the decomposition is exact: real GDP growth equals per-capita growth plus population growth, and nominal growth
equals real growth plus inflation. The table tells a familiar story: the 1960s had the fastest real growth; the 1970s had
nominal growth above 10% driven by inflation of almost 7%; and per-capita growth slowed steadily to about 1.4% in the
2000s. Adding percent changes instead of log changes introduces a small error (up to 0.07 percentage points a year), from
the cross term in \\((1+a)(1+b)=1+a+b+ab\\).
"""),

dict(title="Partial elasticities and a total derivative in a hedonic price function",
data="`hprice1` — 88 house sales.",
task="""
1. Fit \\(\\log(price)=\\beta_0+\\beta_1\\log(sqrft)+\\beta_2 bdrms+\\beta_3\\log(lotsize)\\).
2. Compute the partial effect of one more bedroom, holding square footage fixed. Why can it be small or even negative?
3. In the data, bigger houses have more bedrooms: estimate the auxiliary relation \\(bdrms=\\gamma_0+\\gamma_1\\log(sqrft)\\). Use the chain rule to compute the **total** derivative \\(\\frac{d\\log price}{d\\log sqrft}=\\beta_1+\\beta_2\\gamma_1\\) and compare it with the slope from regressing \\(\\log price\\) on \\(\\log sqrft\\) and \\(\\log lotsize\\) only.
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
h = wd.data("hprice1")
full = smf.ols("np.log(price) ~ np.log(sqrft) + bdrms + np.log(lotsize)", data=h).fit()
# TODO: aux = bdrms ~ log(sqrft) + log(lotsize); total = b1 + b2 * g1
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
h = wd.data("hprice1")
full = smf.ols("np.log(price) ~ np.log(sqrft) + bdrms + np.log(lotsize)", data=h).fit()
aux = smf.ols("bdrms ~ np.log(sqrft) + np.log(lotsize)", data=h).fit()
short = smf.ols("np.log(price) ~ np.log(sqrft) + np.log(lotsize)", data=h).fit()
b1, b2, g1 = full.params["np.log(sqrft)"], full.params["bdrms"], aux.params["np.log(sqrft)"]
print(f"partial effects: d log price / d log sqrft = {b1:.4f};  one more bedroom (same size) = {b2:.4f} ({100*(np.exp(b2)-1):.1f}%)")
print(f"auxiliary: d bdrms / d log sqrft = {g1:.4f}")
print(f"total derivative b1 + b2*g1 = {b1 + b2*g1:.4f};  short-regression slope = {short.params['np.log(sqrft)']:.4f}")
""",
notes="""Holding size fixed, one more bedroom raises price by only about 4% (and insignificantly): at a fixed floor area, an extra
bedroom means smaller rooms. But bigger houses have more bedrooms, so the **total** derivative of log price with respect to
log size, \\(\\beta_1+\\beta_2\\gamma_1\\approx0.76\\), exceeds the partial effect of 0.70. It equals the short-regression slope
exactly — the chain rule here is the omitted-variable bias formula of Econometrics Lecture 3, read the other way round.
"""),
]

EXERCISES["08-hessian-multivariate-optimization.md"] = [
dict(title="Least squares as unconstrained optimization: gradient and Hessian checks",
data="`wage1` — 526 workers.",
task="""
1. Write the sum of squared residuals \\(SSR(b)=(y-Xb)'(y-Xb)\\) for \\(\\log wage\\) on \\((1, educ, exper, tenure)\\).
2. Minimize it numerically with BFGS from \\(b=0\\), supplying the analytic gradient \\(-2X'(y-Xb)\\).
3. At the optimum, check that the gradient is (numerically) zero and that the Hessian \\(2X'X\\) is positive definite (eigenvalues). Report the condition number and explain what it means for numerical accuracy.
4. Compare with the closed-form solution.
""",
starter="""
import numpy as np, wooldridge as wd
from scipy import optimize
d = wd.data("wage1")
y = d["lwage"].to_numpy(); X = np.column_stack([np.ones(len(d)), d[["educ", "exper", "tenure"]]])
ssr = lambda b: np.sum((y - X @ b)**2); grad = lambda b: -2 * X.T @ (y - X @ b)
# TODO: optimize.minimize(ssr, np.zeros(4), jac=grad, method="BFGS")
""",
solution="""
import numpy as np, wooldridge as wd
from scipy import optimize
d = wd.data("wage1")
y = d["lwage"].to_numpy(); X = np.column_stack([np.ones(len(d)), d[["educ", "exper", "tenure"]]])
ssr = lambda b: np.sum((y - X @ b)**2); grad = lambda b: -2 * X.T @ (y - X @ b)
res = optimize.minimize(ssr, np.zeros(4), jac=grad, method="BFGS", options={"gtol": 1e-10})
closed = np.linalg.solve(X.T @ X, X.T @ y)
print("BFGS:       ", res.x.round(6), f"({res.nit} iterations)")
print("closed form:", closed.round(6))
H = 2 * X.T @ X; ev = np.linalg.eigvalsh(H)
print(f"max |gradient| at optimum: {np.abs(grad(res.x)).max():.2e}")
print(f"Hessian eigenvalues: {ev.round(2)}  -> positive definite: {bool(np.all(ev > 0))};  condition number {ev.max()/ev.min():.0f}")
Xs = (X[:, 1:] - X[:, 1:].mean(0)) / X[:, 1:].std(0); Hs = 2 * np.column_stack([np.ones(len(d)), Xs]).T @ np.column_stack([np.ones(len(d)), Xs])
print(f"condition number after standardizing the regressors: {np.linalg.cond(Hs):.1f}")
""",
notes="""BFGS converges to the closed-form OLS solution in five iterations, and the gradient is zero to 12 decimals. The Hessian
\\(2X'X\\) is positive definite, so the SSR is strictly convex and the minimum is unique. Its condition number of about
18,000 comes from regressors on very different scales (a constant, years of education, years of tenure). Standardizing
the regressors reduces it to about 4, which is why numerical optimizers and ridge-type penalties work better on
standardized data. Here the problem is still well conditioned enough for double-precision arithmetic.
"""),

dict(title="Newton–Raphson for a logit likelihood, by hand",
data="`mroz` — 753 married women; `inlf` = labor force participation.",
task="""
1. Write the logit log-likelihood \\(\\ell(\\beta)=\\sum_i[y_i x_i'\\beta-\\log(1+e^{x_i'\\beta})]\\), its gradient \\(X'(y-p)\\) and Hessian \\(-X'WX\\) with \\(W=\\text{diag}(p_i(1-p_i))\\).
2. Run Newton–Raphson from \\(\\beta=0\\) with regressors \\((1, educ, exper, age, kidslt6, nwifeinc)\\). Print the log-likelihood and the gradient norm at each iteration.
3. Show that the Hessian is negative definite everywhere (so \\(\\ell\\) is globally concave and the maximum is unique), and compare with `statsmodels`.
""",
starter="""
import numpy as np, wooldridge as wd
d = wd.data("mroz")
y = d["inlf"].to_numpy(); X = np.column_stack([np.ones(len(d)), d[["educ", "exper", "age", "kidslt6", "nwifeinc"]]])
b = np.zeros(X.shape[1])
# TODO: loop: p = 1/(1+exp(-Xb)); g = X'(y-p); H = -X' diag(p(1-p)) X; b = b - solve(H, g)
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.api as sm
d = wd.data("mroz")
y = d["inlf"].to_numpy(); X = np.column_stack([np.ones(len(d)), d[["educ", "exper", "age", "kidslt6", "nwifeinc"]]])
b = np.zeros(X.shape[1])
for k in range(8):
    p = 1 / (1 + np.exp(-X @ b))
    ll = np.sum(y * (X @ b) - np.log1p(np.exp(X @ b)))
    g = X.T @ (y - p); H = -(X * (p * (1 - p))[:, None]).T @ X
    print(f"iter {k}: logL = {ll:.8f}   |gradient| = {np.linalg.norm(g):.2e}   max eigenvalue of H = {np.linalg.eigvalsh(H).max():.3f}")
    b = b - np.linalg.solve(H, g)
print("Newton estimates:     ", b.round(5))
print("statsmodels estimates:", sm.Logit(y, X).fit(disp=0).params.round(5))
""",
notes="""Starting from zero, Newton–Raphson reaches the maximum in about five iterations: the gradient norm falls from 2,700 to
\\(10^{-10}\\), with the quadratic convergence typical near the optimum. The largest eigenvalue of the Hessian stays
negative at every iterate: the logit log-likelihood is globally concave, because \\(-X'WX\\) is negative definite whenever
\\(X\\) has full rank. The maximum is therefore unique, and the estimates match `statsmodels` (which uses the same
algorithm) to every printed digit.
"""),
]

EXERCISES["09-lagrange-multipliers.md"] = [
dict(title="Cobb–Douglas demand and the multiplier as marginal utility of income",
data="`engel` (statsmodels) — 235 Belgian working-class households (Engel's 19th-century data): annual income and food expenditure.",
task="""
A household maximizes \\(u(f,o)=\\alpha\\log f+(1-\\alpha)\\log o\\) subject to \\(f+o=m\\) (prices normalized to 1).

1. Solve the Lagrangian with `sympy` for the demands and the multiplier \\(\\lambda\\).
2. Calibrate \\(\\alpha\\) to the average food budget share in Engel's data and compute the demands for the median household.
3. Verify the envelope theorem numerically: \\(dV/dm=\\lambda\\), where \\(V(m)\\) is maximized utility.
4. Cobb–Douglas implies a constant budget share. Check this against the data by regressing the food share on log income. What does Engel's law say?
""",
starter="""
import numpy as np, sympy as sp, statsmodels.api as sm
e = sm.datasets.engel.load_pandas().data
f, o, lam, a, m = sp.symbols("f o lambda alpha m", positive=True)
Lag = a * sp.log(f) + (1 - a) * sp.log(o) + lam * (m - f - o)
# TODO: sp.solve([diff(Lag, f), diff(Lag, o), diff(Lag, lam)], [f, o, lam])
""",
solution="""
import numpy as np, sympy as sp, statsmodels.api as sm, statsmodels.formula.api as smf
e = sm.datasets.engel.load_pandas().data
f, o, lam, a, m = sp.symbols("f o lambda alpha m", positive=True)
Lag = a * sp.log(f) + (1 - a) * sp.log(o) + lam * (m - f - o)
sol = sp.solve([sp.diff(Lag, v) for v in (f, o, lam)], [f, o, lam], dict=True)[0]
print("demands and multiplier:", sol)
alpha = (e["foodexp"] / e["income"]).mean(); mmed = e["income"].median()
vals = {a: alpha, m: mmed}
print(f"alpha = mean food share = {alpha:.4f};  median income {mmed:.1f}: food {float(sol[f].subs(vals)):.1f}, other {float(sol[o].subs(vals)):.1f}, lambda = {float(sol[lam].subs(vals)):.6f}")
V = sp.lambdify(m, (a * sp.log(sol[f]) + (1 - a) * sp.log(sol[o])).subs(a, alpha))
h = 1e-3
print(f"envelope check: dV/dm numerically = {(V(mmed + h) - V(mmed - h)) / (2*h):.6f}  vs lambda = {1/mmed:.6f}")
r = smf.ols("I(foodexp/income) ~ np.log(income)", data=e).fit(cov_type="HC1")
print(f"food share on log income: slope {r.params.iloc[1]:.4f} (robust se {r.bse.iloc[1]:.4f})")
""",
notes="""The Lagrangian gives the familiar Cobb–Douglas demands \\(f=\\alpha m\\) and \\(o=(1-\\alpha)m\\), with \\(\\lambda=1/m\\). The
envelope theorem holds to six decimals: the multiplier is the marginal utility of income. Calibrated to Engel's data, the
average household spends 66% of income on food. But Cobb–Douglas implies a **constant** budget share, and the data reject
it: the food share falls by about 8.6 percentage points for each log point of income. That is Engel's law — food is a
necessity — and it calls for non-homothetic preferences such as Stone–Geary.
"""),

dict(title="Restricted least squares via a Lagrangian: constant returns in investment",
data="`grunfeld` (statsmodels) — 11 U.S. firms, 1935–1954.",
task="""
In \\(\\log I=\\beta_0+\\beta_1\\log V+\\beta_2\\log K+u\\), impose \\(\\beta_1+\\beta_2=1\\).

1. Write the Lagrangian for minimizing SSR subject to \\(R\\beta=r\\) and derive \\(\\tilde\\beta=\\widehat\\beta-(X'X)^{-1}R'[R(X'X)^{-1}R']^{-1}(R\\widehat\\beta-r)\\).
2. Compute \\(\\tilde\\beta\\) and the multiplier, and check the result by substituting the constraint into the regression.
3. Compute the \\(F\\) statistic from the restricted and unrestricted SSRs. Interpret the size of the multiplier.
""",
starter="""
import numpy as np, statsmodels.api as sm
g = sm.datasets.grunfeld.load_pandas().data.query("invest > 0 and capital > 0")
y = np.log(g["invest"].to_numpy()); X = np.column_stack([np.ones(len(g)), np.log(g["value"]), np.log(g["capital"])])
R = np.array([[0, 1, 1]]); r = np.array([1.0])
# TODO: restricted estimator, multiplier = 2 [R (X'X)^-1 R']^-1 (R b - r)
""",
solution="""
import numpy as np, statsmodels.api as sm
from scipy import stats
g = sm.datasets.grunfeld.load_pandas().data.query("invest > 0 and capital > 0")
y = np.log(g["invest"].to_numpy()); X = np.column_stack([np.ones(len(g)), np.log(g["value"]), np.log(g["capital"])])
R = np.array([[0.0, 1.0, 1.0]]); r = np.array([1.0]); n, k = X.shape
XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y
mult = 2 * np.linalg.solve(R @ XtXi @ R.T, R @ b - r)
bt = b - XtXi @ R.T @ np.linalg.solve(R @ XtXi @ R.T, R @ b - r)
print(f"unrestricted: {b.round(4)}  (b1 + b2 = {b[1] + b[2]:.4f})")
print(f"restricted:   {bt.round(4)}  (b1 + b2 = {bt[1] + bt[2]:.4f});  Lagrange multiplier = {mult[0]:.3f}")
ys = y - X[:, 2]; Xs = np.column_stack([np.ones(n), X[:, 1] - X[:, 2]])          # substitute b2 = 1 - b1
c = np.linalg.lstsq(Xs, ys, rcond=None)[0]
print(f"substitution method: b0 = {c[0]:.4f}, b1 = {c[1]:.4f}, b2 = {1 - c[1]:.4f}")
ssr_u, ssr_r = np.sum((y - X @ b)**2), np.sum((y - X @ bt)**2)
F = (ssr_r - ssr_u) / (ssr_u / (n - k))
print(f"F(1, {n-k}) = {F:.2f}, p = {stats.f.sf(F, 1, n - k):.2e}")
""",
notes="""The unrestricted elasticities sum to 1.18. Imposing \\(\\beta_1+\\beta_2=1\\) through the Lagrangian reproduces the
substitution method exactly. The large multiplier says the constraint is costly in terms of fit, and the \\(F\\) test rejects
it decisively (\\(F\\approx36\\)): investment rises more than proportionally with the scale of the firm. The Lagrange
multiplier and the \\(F\\) statistic carry the same information, which is the logic of the LM test.
"""),
]

EXERCISES["10-kkt-inequality-constraints.md"] = [
dict(title="Stocks or bills? A portfolio choice with no borrowing and no short sales",
data="`volat` — monthly S&P 500 returns and 3-month T-bill yields, 1947–1993.",
task="""
A mean–variance investor chooses the share \\(w\\) in stocks (the rest in T-bills) to maximize \\(w\\mu_e-\\tfrac{\\gamma}{2}w^2\\sigma^2\\), where \\(\\mu_e\\) and \\(\\sigma^2\\) are the mean and variance of the monthly **excess** return, subject to \\(0\\le w\\le1\\).

1. Write the KKT conditions with multipliers for both bounds.
2. Estimate \\(\\mu_e\\) and \\(\\sigma\\) from the data. Solve for \\(w^*\\) and the multipliers for \\(\\gamma=1,2,5,10\\).
3. For which \\(\\gamma\\) is the borrowing constraint \\(w\\le1\\) binding, and what is its shadow value? Check complementary slackness.
""",
starter="""
import numpy as np, wooldridge as wd
v = wd.data("volat").dropna(subset=["rsp500"])
ex = (v["rsp500"] - v["i3"]).to_numpy() / 1200          # monthly excess return
mu, s2 = ex.mean(), ex.var()
# TODO: interior solution w = mu/(gamma s2); clip to [0,1]; multipliers from the FOC
""",
solution="""
import numpy as np, wooldridge as wd
v = wd.data("volat").dropna(subset=["rsp500"])
ex = (v["rsp500"] - v["i3"]).to_numpy() / 1200
mu, s2 = ex.mean(), ex.var()
print(f"monthly excess return: mean {100*mu:.3f}%, sd {100*np.sqrt(s2):.3f}%  (annualized Sharpe ratio {mu/np.sqrt(s2)*np.sqrt(12):.2f})")
for g in (1, 2, 5, 10):
    w_int = mu / (g * s2); w = min(max(w_int, 0.0), 1.0)
    foc = mu - g * w * s2                       # = lam_upper - lam_lower at the optimum
    lam_up, lam_lo = max(foc, 0.0), max(-foc, 0.0)
    cs = lam_up * (1 - w) + lam_lo * w
    print(f"gamma = {g:2d}: unconstrained w = {w_int:5.2f}, optimal w = {w:.3f}, multiplier on w<=1 = {lam_up:.6f}, on w>=0 = {lam_lo:.6f}, complementary slackness {cs:.1e}")
""",
notes="""Over 1947–1993, stocks earned about 0.6% a month over T-bills, with a Sharpe ratio of about 0.6 a year. That premium is
so large relative to the risk that a mean–variance investor with \\(\\gamma\\le5\\) would like to **borrow** to hold more than
100% in stocks; with borrowing ruled out, the constraint \\(w\\le1\\) binds. Its multiplier is the marginal gain from relaxing
it and falls as risk aversion rises. At \\(\\gamma=10\\), the interior solution of about 52% is feasible, and both multipliers
are zero. Complementary slackness holds in every case. (The large equity premium relative to risk is the "equity premium
puzzle" of Mehra and Prescott, 1985.)
"""),

dict(title="Synthetic weights as a KKT problem",
data="`statecrime` (statsmodels) — U.S. states: % high-school graduates, % in poverty, % single-parent households, % urban.",
task="""
A synthetic-control-style problem: find non-negative weights summing to one so that a weighted average of other states matches California's characteristics as closely as possible:
\\(\\min_w\\|x_{CA}-X_0w\\|^2\\) s.t. \\(w\\ge0\\), \\(\\sum w=1\\).

1. Solve it with `scipy.optimize.minimize` (SLSQP), using standardized characteristics.
2. Report the states with positive weight, and the fit.
3. Check the KKT conditions: for states with \\(w_j=0\\), the gradient component must be at least as large as for states with \\(w_j>0\\) (whose gradient components are all equal, to the multiplier on the adding-up constraint).
""",
starter="""
import numpy as np, statsmodels.api as sm
from scipy import optimize
d = sm.datasets.statecrime.load_pandas().data
cols = ["hs_grad", "poverty", "single", "urban"]
Z = (d[cols] - d[cols].mean()) / d[cols].std()
x1 = Z.loc["California"].to_numpy(); X0 = Z.drop("California").to_numpy().T
# TODO: minimize ||x1 - X0 w||^2 with bounds (0,1) and constraint sum(w) = 1
""",
solution="""
import numpy as np, statsmodels.api as sm
from scipy import optimize
d = sm.datasets.statecrime.load_pandas().data
cols = ["hs_grad", "poverty", "single", "urban"]
Z = (d[cols] - d[cols].mean()) / d[cols].std()
x1 = Z.loc["California"].to_numpy(); donors = Z.drop("California"); X0 = donors.to_numpy().T; J = X0.shape[1]
obj = lambda w: np.sum((x1 - X0 @ w)**2); grad = lambda w: -2 * X0.T @ (x1 - X0 @ w)
res = optimize.minimize(obj, np.full(J, 1 / J), jac=grad, method="SLSQP", bounds=[(0, 1)] * J,
                        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"ftol": 1e-14, "maxiter": 500})
w = res.x; pos = w > 1e-6
print("positive weights:", {s: round(x, 3) for s, x in zip(donors.index[pos], w[pos])})
print("California (std units):", x1.round(3), "  synthetic:", (X0 @ w).round(3), f"  loss {obj(w):.4f}")
gr = grad(w)
print(f"gradient on positive-weight states: {gr[pos].round(4)}  (equal -> common multiplier {gr[pos].mean():.4f})")
print(f"min gradient on zero-weight states: {gr[~pos].min():.4f}  (KKT requires >= {gr[pos].mean():.4f})")
""",
notes="""Only two states get positive weight — Texas (57%) and Rhode Island (43%). Corner solutions with most weights at zero are
typical of synthetic control. The synthetic unit matches California's poverty and single-parent shares well but not its
very high urban share or low high-school graduation, because no convex combination of other states reaches those
extremes. The KKT conditions hold: the gradient components of the two active states are equal (the common multiplier on
the adding-up constraint), and every inactive state's gradient component is at least as large, so moving weight to it
would not lower the loss.
"""),
]

EXERCISES["11-implicit-inverse-function-theorems.md"] = [
dict(title="Comparative statics of the Solow steady state with the implicit function theorem",
data="`macrodata` — U.S. investment share of GDP and population growth, used to calibrate the model.",
task="""
In the Solow model, steady-state capital per worker \\(k^*\\) solves \\(F(k,s)=s\\,k^{\\alpha}-(n+\\delta)k=0\\).

1. Calibrate \\(s\\) as the average ratio of real investment to real GDP and \\(n\\) as average annual population growth; set \\(\\alpha=1/3\\), \\(\\delta=0.05\\).
2. Solve for \\(k^*\\) numerically (`brentq`), without using the closed form.
3. Use the implicit function theorem, \\(dk^*/ds=-F_s/F_k\\), to compute the effect of a higher saving rate, and the elasticity \\(\\frac{dk^*}{ds}\\frac{s}{k^*}\\). Compare with a finite difference and with the closed-form elasticity \\(1/(1-\\alpha)\\).
""",
starter="""
import numpy as np, statsmodels.api as sm
from scipy import optimize
m = sm.datasets.macrodata.load_pandas().data
s = (m["realinv"] / m["realgdp"]).mean(); n = np.log(m["pop"]).diff().mean() * 4
a, dl = 1/3, 0.05
F = lambda k, s: s * k**a - (n + dl) * k
# TODO: brentq for k*, IFT derivative: -F_s/F_k with F_s = k^a, F_k = s a k^(a-1) - (n+dl)
""",
solution="""
import numpy as np, statsmodels.api as sm
from scipy import optimize
m = sm.datasets.macrodata.load_pandas().data
s = (m["realinv"] / m["realgdp"]).mean(); n = np.log(m["pop"]).diff().mean() * 4
a, dl = 1/3, 0.05
F = lambda k, s: s * k**a - (n + dl) * k
kstar = lambda s: optimize.brentq(F, 1e-6, 1e6, args=(s,), xtol=1e-14)
k = kstar(s)
print(f"calibration: s = {s:.4f}, n = {n:.4f}, alpha = {a:.3f}, delta = {dl}")
print(f"steady state k* = {k:.4f}  (closed form {(s/(n+dl))**(1/(1-a)):.4f})")
Fs, Fk = k**a, s * a * k**(a - 1) - (n + dl)
dk = -Fs / Fk
h = 1e-6; fd = (kstar(s + h) - kstar(s - h)) / (2 * h)
print(f"IFT: dk*/ds = {dk:.4f}  vs finite difference {fd:.4f};  F_k = {Fk:.4f} < 0 (stable)")
print(f"elasticity dk*/ds * s/k* = {dk * s / k:.4f}  vs closed form 1/(1-alpha) = {1/(1-a):.4f}")
""",
notes="""Calibrated to the U.S. (investment share 13%, population growth 1.1%), the numerical root matches the closed-form steady
state. The implicit function theorem gives \\(dk^*/ds=-F_s/F_k\\) without solving for \\(k^*\\) explicitly, and it agrees with
the finite difference. \\(F_k<0\\) at the steady state, which is both the condition for the IFT to apply and the condition
for stability. The elasticity of steady-state capital with respect to the saving rate is exactly \\(1/(1-\\alpha)=1.5\\).
"""),

dict(title="Yield to maturity as an inverse function, and duration",
data="`intqrt` — U.S. 12-month Treasury yield (average, used as a reference yield).",
task="""
The price of a bond is a decreasing function of its yield, \\(P(y)=\\sum_t c_t/(1+y)^t\\). Its inverse \\(y(P)\\) is the yield to maturity.

1. For a 10-year bond with a 6% annual coupon and face value 100, compute \\(P\\) at the average 12-month yield.
2. Invert numerically: find \\(y\\) from \\(P\\), and check that the round trip recovers the yield.
3. By the inverse function theorem, \\(dy/dP=1/P'(y)\\). Compute it, and the modified duration \\(-P'(y)/P\\). Use duration to approximate the price change from a 1-point yield rise and compare with the exact change.
""",
starter="""
import numpy as np, wooldridge as wd
from scipy import optimize
y0 = wd.data("intqrt")["r12"].mean() / 100
cf = np.r_[np.full(9, 6.0), 106.0]; t = np.arange(1, 11)
P = lambda y: np.sum(cf / (1 + y)**t); dP = lambda y: -np.sum(t * cf / (1 + y)**(t + 1))
# TODO: price, inverse via brentq, dy/dP = 1/dP, modified duration
""",
solution="""
import numpy as np, wooldridge as wd
from scipy import optimize
y0 = wd.data("intqrt")["r12"].mean() / 100
cf = np.r_[np.full(9, 6.0), 106.0]; t = np.arange(1, 11)
P = lambda y: np.sum(cf / (1 + y)**t); dP = lambda y: -np.sum(t * cf / (1 + y)**(t + 1))
p0 = P(y0)
yinv = optimize.brentq(lambda y: P(y) - p0, -0.5, 1.0, xtol=1e-15)
print(f"yield {100*y0:.4f}% -> price {p0:.4f};  inverted yield {100*yinv:.10f}%")
print(f"dy/dP = 1/P'(y) = {1/dP(y0):.6f}  (yield change per $1 of price)")
D = -dP(y0) / p0
print(f"modified duration {D:.4f} years")
print(f"1-point yield rise: duration approximation {-D*0.01*100:.3f}% of price, exact {100*(P(y0+0.01)/p0 - 1):.3f}%")
""",
notes="""Because the price is strictly decreasing in the yield, the inverse function exists, and the numerical inversion recovers
the yield to ten decimals. The inverse function theorem gives \\(dy/dP=1/P'(y)\\): a \\$1 price fall raises the yield by about
15 basis points. Modified duration (7.2 years) predicts a 7.2% price fall for a 1-point yield rise, against an exact fall
of 6.9%. The gap is convexity: the price–yield curve is convex, so duration overstates losses and understates gains (Finance,
Lecture 7).
"""),
]

EXERCISES["12-multiple-integrals-jacobians.md"] = [
dict(title="A bivariate normal probability by double integration",
data="`econmath` — ACT math scores and college GPA of 814 students.",
task="""
1. Fit a bivariate normal to `(actmth, colgpa)`.
2. Compute \\(P(\\text{ACT math}\\ge28,\\ GPA\\ge3.5)\\) by numerical double integration of the density (`scipy.integrate.dblquad`).
3. Check it with the bivariate normal CDF and with Monte Carlo integration (one million draws).
4. Compare with the empirical frequency in the data. What does any gap say about the normal model?
""",
starter="""
import numpy as np, wooldridge as wd
from scipy import stats, integrate
d = wd.data("econmath")[["actmth", "colgpa"]].dropna()
mvn = stats.multivariate_normal(d.mean(), d.cov())
# TODO: integrate.dblquad(lambda g, a: mvn.pdf([a, g]), 28, inf, 3.5, inf)
""",
solution="""
import numpy as np, wooldridge as wd
from scipy import stats, integrate
d = wd.data("econmath")[["actmth", "colgpa"]].dropna()
mu, S = d.mean().to_numpy(), d.cov().to_numpy(); mvn = stats.multivariate_normal(mu, S)
dbl, err = integrate.dblquad(lambda g, a: mvn.pdf([a, g]), 28, mu[0] + 10 * np.sqrt(S[0, 0]), 3.5, mu[1] + 10 * np.sqrt(S[1, 1]))
cdf = 1 - stats.norm.cdf(28, mu[0], np.sqrt(S[0, 0])) - stats.norm.cdf(3.5, mu[1], np.sqrt(S[1, 1])) + mvn.cdf([28, 3.5])
mc = np.mean(np.all(np.random.default_rng(0).multivariate_normal(mu, S, 1_000_000) >= [28, 3.5], axis=1))
print(f"double integral {dbl:.5f} (error estimate {err:.1e});  via CDF {cdf:.5f};  Monte Carlo {mc:.5f}")
emp = np.mean((d["actmth"] >= 28) & (d["colgpa"] >= 3.5))
print(f"empirical frequency: {emp:.5f} ({int(emp*len(d))} of {len(d)} students)")
""",
notes="""Double integration, the CDF formula and Monte Carlo all give about 2.2% for the joint event. In the data, 3.2% of students
(26 of 814) have both an ACT math score of 28 or more and a GPA of 3.5 or more — about 45% more than the normal model
predicts. The bivariate normal understates the joint upper tail: strong students are more concentrated at the top than a
normal distribution with correlation 0.3 allows (and GPA is capped at 4). Joint tail probabilities are exactly where normal
approximations tend to fail.
"""),

dict(title="Change of variables: from log wages to wages",
data="`wage1` — hourly wages.",
task="""
If \\(Z=\\log W\\sim N(\\mu,\\sigma^2)\\), the density of \\(W=e^Z\\) follows from the change-of-variables formula \\(f_W(w)=f_Z(\\log w)\\,|dz/dw|=f_Z(\\log w)/w\\).

1. Estimate \\(\\mu,\\sigma\\) from log wages, and derive \\(f_W\\) with `sympy` using the Jacobian.
2. Check numerically that \\(f_W\\) integrates to one, and that \\(\\int w f_W(w)\\,dw=e^{\\mu+\\sigma^2/2}\\).
3. Compare the implied mean wage with the sample mean, and the implied probability of earning less than \\$3/hour with the sample share.
""",
starter="""
import numpy as np, sympy as sp, wooldridge as wd
z = np.log(wd.data("wage1")["wage"].to_numpy())
w, mu, s = sp.symbols("w mu sigma", positive=True)
fz = lambda x: sp.exp(-(x - mu)**2 / (2 * s**2)) / (s * sp.sqrt(2 * sp.pi))
# TODO: fw = fz(sp.log(w)) * sp.diff(sp.log(w), w)
""",
solution="""
import numpy as np, sympy as sp, wooldridge as wd
from scipy import integrate
wage = wd.data("wage1")["wage"].to_numpy(); z = np.log(wage)
w, mu, s = sp.symbols("w mu sigma", positive=True)
fz = lambda x: sp.exp(-(x - mu)**2 / (2 * s**2)) / (s * sp.sqrt(2 * sp.pi))
fw = fz(sp.log(w)) * sp.diff(sp.log(w), w)
print("f_W(w) =", sp.simplify(fw))
vals = {mu: z.mean(), s: z.std(ddof=1)}
f = sp.lambdify(w, fw.subs(vals))
tot = integrate.quad(f, 0, np.inf)[0]; mean = integrate.quad(lambda x: x * f(x), 0, np.inf)[0]
print(f"mu = {vals[mu]:.4f}, sigma = {vals[s]:.4f}")
print(f"integral of f_W = {tot:.8f};  E[W] by integration = {mean:.4f}, formula e^(mu+s^2/2) = {np.exp(vals[mu] + vals[s]**2/2):.4f}, sample mean = {wage.mean():.4f}")
print(f"P(W < 3): lognormal {integrate.quad(f, 0, 3)[0]:.4f}, sample share {np.mean(wage < 3):.4f}")
""",
notes="""`sympy` produces the lognormal density through the Jacobian \\(1/w\\). Numerically it integrates to one, and its mean
equals \\(e^{\\mu+\\sigma^2/2}\\approx\\$5.84\\), close to the sample mean of \\$5.90. The lognormal overstates the share of very
low wages (16% below \\$3 against 12% in the data), because the 1976 minimum wage of \\$2.30 truncates the lower tail of
actual wages. The change of variables is exact; the model's fit is the empirical question.
"""),
]

EXERCISES["13-differential-equations-dynamics.md"] = [
dict(title="The continuous-time Solow model calibrated to the U.S.",
data="`macrodata` — U.S. investment share and population growth (calibration as in Lecture 11).",
task="""
Capital per worker follows \\(\\dot k=s\\,k^{\\alpha}-(n+\\delta)k\\).

1. Solve the ODE numerically with `solve_ivp` starting from half the steady-state capital.
2. Linearize around \\(k^*\\): \\(\\dot k\\approx-\\lambda(k-k^*)\\) with \\(\\lambda=(1-\\alpha)(n+\\delta)\\). Compute the implied half-life \\(\\ln2/\\lambda\\).
3. Measure the half-life in the numerical solution. Why is it close but not identical to the linear prediction?
""",
starter="""
import numpy as np, statsmodels.api as sm
from scipy.integrate import solve_ivp
m = sm.datasets.macrodata.load_pandas().data
s = (m["realinv"] / m["realgdp"]).mean(); n = np.log(m["pop"]).diff().mean() * 4; a, dl = 1/3, 0.05
kss = (s / (n + dl))**(1 / (1 - a))
# TODO: solve_ivp(lambda t, k: s*k**a - (n+dl)*k, (0, 150), [kss/2], dense_output=True)
""",
solution="""
import numpy as np, statsmodels.api as sm
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
m = sm.datasets.macrodata.load_pandas().data
s = (m["realinv"] / m["realgdp"]).mean(); n = np.log(m["pop"]).diff().mean() * 4; a, dl = 1/3, 0.05
kss = (s / (n + dl))**(1 / (1 - a))
sol = solve_ivp(lambda t, k: s * k**a - (n + dl) * k, (0, 150), [kss / 2], dense_output=True, rtol=1e-10, atol=1e-12)
lam = (1 - a) * (n + dl)
print(f"k* = {kss:.4f}; convergence rate lambda = {lam:.4f} per year; linear half-life = {np.log(2)/lam:.2f} years")
gap = lambda t: (kss - sol.sol(t)[0]) - 0.5 * (kss - kss / 2)
print(f"numerical half-life (gap halves): {brentq(gap, 0.1, 100):.2f} years")
for t in (0, 10, 20, 40, 80):
    print(f"  t = {t:3d}: k = {sol.sol(t)[0]:.4f} ({sol.sol(t)[0]/kss:.1%} of k*)")
""",
notes="""Starting at half the steady state, capital per worker closes 90% of the gap after about 40 years. Linearization predicts a
half-life of 17.1 years; the numerical solution takes about 18.5. The linear approximation is accurate only near \\(k^*\\):
starting far below, the true dynamics are slower at the beginning in terms of the **proportional** gap, because
\\(\\dot k/k=sk^{\\alpha-1}-(n+\\delta)\\) is not linear in \\(k\\). The calibrated convergence rate of about 4% a year is in
the range of empirical estimates (2–3%) only roughly; a higher capital share brings it down.
"""),

dict(title="Exponential growth: fitting \\(\\dot y=gy\\) to U.S. real GDP",
data="`macrodata` — U.S. quarterly real GDP, 1959–2009.",
task="""
1. The solution of \\(\\dot y=gy\\) is \\(y(t)=y_0e^{gt}\\), so \\(\\log y\\) is linear in \\(t\\). Estimate \\(g\\) by regressing \\(\\log GDP\\) on time (in years).
2. Compute the implied doubling time \\(\\ln2/g\\) and compare with the actual doubling time in the data.
3. Plot (or tabulate) the deviation of log GDP from the fitted exponential path at business-cycle troughs (1975Q1, 1982Q4, 1991Q1, 2009Q2). Is the growth rate constant over 1959–2009? Compare \\(g\\) for 1959–1983 and 1984–2009.
""",
starter="""
import numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
m = sm.datasets.macrodata.load_pandas().data
m["t"] = m["year"] + (m["quarter"] - 1) / 4 - 1959
# TODO: smf.ols("np.log(realgdp) ~ t", m).fit()
""",
solution="""
import numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
m = sm.datasets.macrodata.load_pandas().data
m["t"] = m["year"] + (m["quarter"] - 1) / 4 - 1959
r = smf.ols("np.log(realgdp) ~ t", data=m).fit()
g = r.params["t"]
print(f"g = {100*g:.3f}% per year;  implied doubling time {np.log(2)/g:.1f} years")
y0 = m["realgdp"].iloc[0]; k = np.argmax(m["realgdp"].to_numpy() >= 2 * y0)
print(f"actual first doubling: {m['t'].iloc[k]:.2f} years (reached in {int(m['year'].iloc[k])}Q{int(m['quarter'].iloc[k])})")
for yq in ((1975, 1), (1982, 4), (1991, 1), (2009, 2)):
    row = m[(m.year == yq[0]) & (m.quarter == yq[1])]
    if len(row): print(f"  {yq[0]}Q{yq[1]}: log deviation from exponential trend {100*r.resid[row.index[0]]:+.2f}%")
for lo, hi in ((1959, 1983), (1984, 2009)):
    s = m[(m.year >= lo) & (m.year <= hi)]
    print(f"g over {lo}-{hi}: {100*smf.ols('np.log(realgdp) ~ t', data=s).fit().params['t']:.3f}% per year")
""",
notes="""Real GDP grew at about 3.2% a year on average, implying a doubling time of about 22 years. It actually doubled faster, in
18.5 years (by 1977), because growth was higher early in the sample: 3.4% a year in 1959–83 against 3.0% in 1984–2009. The
constant-growth model is a first approximation. The deviations from trend at recession troughs grow over time, ending
10.6% below trend in 2009 — partly because of the deep recession and partly because the trend slowed, which a single
exponential cannot capture.
"""),
]

EXERCISES["14-difference-equations-discrete-dynamics.md"] = [
dict(title="A first-order difference equation for unemployment",
data="`phillips` — U.S. annual unemployment, 1948–2003.",
task="""
1. Estimate \\(u_t=a+bu_{t-1}+e_t\\) by OLS. Find the fixed point \\(u^*=a/(1-b)\\) and check stability (\\(|b|<1\\)).
2. Write the general solution \\(u_t=u^*+b^t(u_0-u^*)\\) and simulate the deterministic path from the 2003 level for 10 years.
3. Plot the phase diagram: \\(u_t\\) against \\(u_{t-1}\\) with the 45° line. How many years until the gap halves?
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
p = wd.data("phillips")
p["u1"] = p["unem"].shift(1)
r = smf.ols("unem ~ u1", data=p).fit()
# TODO: fixed point, path u_t = u* + b^t (u_0 - u*)
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
p = wd.data("phillips"); p["u1"] = p["unem"].shift(1)
r = smf.ols("unem ~ u1", data=p).fit()
a, b = r.params["Intercept"], r.params["u1"]; us = a / (1 - b)
print(f"u_t = {a:.3f} + {b:.3f} u_(t-1);  fixed point u* = {us:.3f}%;  stable: {abs(b) < 1}")
u0 = p["unem"].iloc[-1]
print(f"start: {int(p['year'].iloc[-1])} unemployment {u0:.1f}%")
for t in range(0, 11, 2):
    print(f"  {int(p['year'].iloc[-1]) + t}: {us + b**t * (u0 - us):.3f}%")
print(f"half-life of the gap: {np.log(0.5)/np.log(b):.2f} years")
""",
notes="""The fitted equation has a fixed point at about 5.8% and a root of 0.74, so it is stable, and any gap closes by 26% a
year — a half-life of 2.3 years. Starting from 6.0% in 2003, the deterministic path converges slowly to 5.8%. The
deterministic solution describes the expected path only; the actual path is hit by new shocks every year (Econometrics
Lecture 9).
"""),

dict(title="Business cycles from a second-order difference equation",
data="`macrodata` — U.S. quarterly real GDP growth, 1959–2009.",
task="""
Samuelson's multiplier–accelerator model produces a second-order difference equation, \\(y_t=a_1y_{t-1}+a_2y_{t-2}\\).

1. Estimate an AR(2) for quarterly GDP growth (in deviations from the mean).
2. Find the roots of the characteristic equation \\(\\lambda^2-a_1\\lambda-a_2=0\\). Are they real or complex? Is the system stable?
3. If complex, compute the implied cycle period \\(2\\pi/\\theta\\) where \\(\\lambda=re^{\\pm i\\theta}\\), and the damping factor \\(r\\). Compute the impulse response for 16 quarters.
4. Repeat for an AR(2) in the **level** of the unemployment rate (`unemp`). Which series shows a clearer cycle?
""",
starter="""
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
m = sm.datasets.macrodata.load_pandas().data
g = 400 * np.diff(np.log(m["realgdp"].to_numpy()))
fit = AutoReg(g, lags=2, trend="c").fit()
# TODO: roots of lambda^2 - a1 lambda - a2; period 2 pi / angle
""",
solution="""
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
m = sm.datasets.macrodata.load_pandas().data
series = {"GDP growth": 400 * np.diff(np.log(m["realgdp"].to_numpy())), "unemployment rate": m["unemp"].to_numpy()}
for name, x in series.items():
    fit = AutoReg(x, lags=2, trend="c").fit(); a1, a2 = fit.params[1], fit.params[2]
    roots = np.roots([1, -a1, -a2]); mod = np.abs(roots)
    print(f"{name}: a1 = {a1:.3f}, a2 = {a2:.3f}; roots {np.round(roots, 3)}; |roots| {mod.round(3)}; stable {bool(np.all(mod < 1))}")
    if np.iscomplexobj(roots) and abs(roots[0].imag) > 1e-12:
        th = abs(np.angle(roots[0])); print(f"   complex roots: period {2*np.pi/th:.1f} quarters, damping r = {mod[0]:.3f}")
    else:
        print("   real roots: no oscillation, monotone adjustment")
    irf = [1.0, a1]
    for _ in range(14): irf.append(a1 * irf[-1] + a2 * irf[-2])
    print("   impulse response:", np.round(irf[:16:3], 3))
""",
notes="""For GDP growth, both roots are real and small: shocks die out within two or three quarters with no oscillation. Growth
rates have little persistence. The **unemployment rate** behaves differently: its roots are complex with modulus 0.83, so a
shock produces a hump-shaped response that peaks after a few quarters and then decays with a cycle. The implied period of
about 58 quarters is long, and the damping is strong, so the "cycle" is barely visible. Samuelson's oscillations exist,
but the data favor damped, mostly monotone dynamics.
"""),
]
