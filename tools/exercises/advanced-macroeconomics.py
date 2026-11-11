INTRO = """
These exercises combine **real U.S. macroeconomic data** (the quarterly `macrodata` series in `statsmodels`, 1959–2009,
and annual series from the `wooldridge` package) with **numerical solutions of the models** in this lecture. Install once
with `pip install numpy scipy pandas statsmodels wooldridge`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.
"""

EXERCISES = {}

EXERCISES["01-dynamic-household.md"] = [
dict(title="Estimating the elasticity of intertemporal substitution from the Euler equation",
data="`macrodata` — U.S. quarterly real consumption `realcons` and ex-post real interest rate `realint`, 1959–2009.",
task="""
The log-linearized Euler equation is \\(\\Delta\\log c_{t+1}=\\text{const}+\\psi\\,r_t+\\varepsilon_{t+1}\\), where \\(\\psi\\) is the
elasticity of intertemporal substitution (EIS). The realized real rate is measured with error and correlated with the
expectational error, so use instrumental variables (Hall, 1988).

1. Estimate \\(\\psi\\) by OLS.
2. Estimate it by 2SLS, instrumenting \\(r_t\\) with variables dated \\(t-2\\) and \\(t-3\\) (lagged real rate, inflation and consumption growth). Report the first-stage \\(F\\).
3. What does the estimate imply for the coefficient of relative risk aversion if preferences are CRRA (\\(\\gamma=1/\\psi\\))?
""",
starter="""
import numpy as np, pandas as pd, statsmodels.api as sm
from linearmodels.iv import IV2SLS
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"dc": 400 * np.log(m["realcons"]).diff().shift(-1), "r": m["realint"], "infl": m["infl"]})
# TODO: lags dated t-2, t-3; IV2SLS.from_formula("dc ~ 1 + [r ~ r_l2 + r_l3 + infl_l2 + dc_l2]", d)
""",
solution="""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"dc": 400 * np.log(m["realcons"]).diff().shift(-1), "r": m["realint"], "infl": m["infl"]})
d["dc0"] = 400 * np.log(m["realcons"]).diff()
for v in ("r", "infl", "dc0"):
    for k in (2, 3):
        d[f"{v}_l{k}"] = d[v].shift(k)
d = d.dropna()
ols = smf.ols("dc ~ r", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
iv = IV2SLS.from_formula("dc ~ 1 + [r ~ r_l2 + r_l3 + infl_l2 + infl_l3 + dc0_l2 + dc0_l3]", data=d).fit(cov_type="kernel")
fs = smf.ols("r ~ r_l2 + r_l3 + infl_l2 + infl_l3 + dc0_l2 + dc0_l3", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
F = fs.wald_test("r_l2 = 0, r_l3 = 0, infl_l2 = 0, infl_l3 = 0, dc0_l2 = 0, dc0_l3 = 0", scalar=True).statistic / 6
print(f"OLS  EIS = {ols.params['r']:.3f} (HAC se {ols.bse['r']:.3f})")
print(f"2SLS EIS = {iv.params['r']:.3f} (HAC se {iv.std_errors['r']:.3f});  first-stage F = {F:.1f}")
lo, hi = iv.params["r"] - 1.96 * iv.std_errors["r"], iv.params["r"] + 1.96 * iv.std_errors["r"]
print(f"95% CI for the EIS: [{lo:.3f}, {hi:.3f}]")
""",
notes="""OLS and 2SLS both give an EIS of about 0.2–0.3, and the 2SLS confidence interval includes zero. This is Hall's (1988)
classic finding: aggregate consumption growth responds little to the real interest rate, so the EIS is small (and, under
CRRA, risk aversion \\(1/\\psi\\) would be large, around 4 or more). The first-stage \\(F\\) of about 13 is only moderately
strong. Instruments dated \\(t-2\\) and earlier are used because time aggregation of quarterly consumption makes \\(t-1\\)
instruments invalid. Many macro models assume an EIS near 0.5–1, which is hard to reconcile with this evidence.
"""),

dict(title="Is consumption a random walk? The Campbell–Mankiw test",
data="`macrodata` — U.S. quarterly real consumption and real disposable income.",
task="""
Hall's (1978) permanent-income hypothesis implies that consumption growth is unpredictable. Campbell and Mankiw (1989)
nest it in \\(\\Delta\\log c_t=\\mu+\\lambda\\,\\Delta\\log y_t+\\varepsilon_t\\), where \\(\\lambda\\) is the share of income going to
"rule-of-thumb" (hand-to-mouth) consumers.

1. Estimate \\(\\lambda\\) by OLS and by 2SLS, using lags 2–4 of income growth, consumption growth and the change in the T-bill rate as instruments.
2. Test \\(\\lambda=0\\). What does the estimate say about the importance of hand-to-mouth households — a central ingredient of TANK and HANK models (Lectures 12–13)?
""",
starter="""
import numpy as np, pandas as pd, statsmodels.api as sm
from linearmodels.iv import IV2SLS
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"dc": 100 * np.log(m["realcons"]).diff(), "dy": 100 * np.log(m["realdpi"]).diff(), "di": m["tbilrate"].diff()})
# TODO: lags 2-4 as instruments; IV2SLS
""",
solution="""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"dc": 100 * np.log(m["realcons"]).diff(), "dy": 100 * np.log(m["realdpi"]).diff(), "di": m["tbilrate"].diff()})
z = []
for v in ("dy", "dc", "di"):
    for k in (2, 3, 4):
        d[f"{v}_l{k}"] = d[v].shift(k); z.append(f"{v}_l{k}")
d = d.dropna()
ols = smf.ols("dc ~ dy", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
iv = IV2SLS.from_formula(f"dc ~ 1 + [dy ~ {' + '.join(z)}]", data=d).fit(cov_type="kernel")
fs = smf.ols(f"dy ~ {' + '.join(z)}", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
F = fs.wald_test(", ".join(f"{v} = 0" for v in z), scalar=True).statistic / len(z)
print(f"OLS  lambda = {ols.params['dy']:.3f} (se {ols.bse['dy']:.3f})")
print(f"2SLS lambda = {iv.params['dy']:.3f} (se {iv.std_errors['dy']:.3f}), t = {iv.tstats['dy']:.2f};  first-stage F = {F:.1f}")
print(f"overidentification (Sargan) p-value: {iv.sargan.pval:.3f}")
""",
notes="""The 2SLS estimate of \\(\\lambda\\) is about 0.65 and strongly significant, so consumption growth is far from unpredictable:
predictable income growth passes into consumption. Taken at face value, around half or more of income goes to
rule-of-thumb consumers — Campbell and Mankiw (1989) found about 0.5. Two warnings: the first-stage \\(F\\) of about 4 signals
**weak instruments** (Lecture 5 of Econometrics), so the 2SLS point estimate is unreliable and weak-IV-robust inference is
needed; and the over-identification test rejects, so the instruments disagree. The qualitative conclusion — a sizable
hand-to-mouth share — is what motivates TANK and HANK models.
"""),
]

EXERCISES["02-production-investment.md"] = [
dict(title="Building a capital stock and a Solow residual for the U.S.",
data="`macrodata` — U.S. quarterly real GDP, real investment, unemployment rate and population, 1959–2009.",
task="""
1. Build a capital stock with the perpetual-inventory method \\(K_{t+1}=(1-\\delta)K_t+I_t\\), \\(\\delta=0.025\\) per quarter, initializing \\(K_0=I_0/(g+\\delta)\\) with \\(g\\) the average investment growth rate.
2. Proxy labor input by employment, \\(N_t=pop_t(1-u_t/100)\\), and compute the Solow residual \\(\\log z_t=\\log Y_t-\\alpha\\log K_t-(1-\\alpha)\\log N_t\\) with \\(\\alpha=1/3\\).
3. Decompose average annual growth of output per worker into capital deepening and TFP growth, before and after 1973.
""",
starter="""
import numpy as np, pandas as pd, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
Y, I = m["realgdp"].to_numpy(), m["realinv"].to_numpy(); d = 0.025; a = 1/3
# TODO: perpetual inventory; employment; Solow residual; growth accounting by period
""",
solution="""
import numpy as np, pandas as pd, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
Y, I = m["realgdp"].to_numpy(), m["realinv"].to_numpy(); dl, a = 0.025, 1/3
g = np.mean(np.diff(np.log(I)))
K = np.empty_like(Y); K[0] = I[0] / (g + dl)
for t in range(len(Y) - 1):
    K[t + 1] = (1 - dl) * K[t] + I[t]
N = m["pop"].to_numpy() * (1 - m["unemp"].to_numpy() / 100)
logz = np.log(Y) - a * np.log(K) - (1 - a) * np.log(N)
df = pd.DataFrame({"year": m["year"], "y": np.log(Y / N), "k": np.log(K / N), "z": logz})
print(f"capital-output ratio (annual): 1959 {K[0]/Y[0]/4:.2f}, 1973 {K[56]/Y[56]/4:.2f}, 2009 {K[-1]/Y[-1]/4:.2f}")
for lo, hi in ((1960, 1973), (1973, 2007)):
    s = df[(df.year >= lo) & (df.year <= hi)]; T = (len(s) - 1) / 4
    gy = 100 * (s.y.iloc[-1] - s.y.iloc[0]) / T; gk = 100 * a * (s.k.iloc[-1] - s.k.iloc[0]) / T; gz = 100 * (s.z.iloc[-1] - s.z.iloc[0]) / T
    print(f"{lo}-{hi}: output/worker {gy:.2f}%/yr = capital deepening {gk:.2f} + TFP {gz:.2f}")
np.save("/tmp/us_logz.npy", logz)
""",
notes="""The perpetual-inventory capital stock rises relative to output over the sample. Growth accounting shows the post-1973
productivity slowdown: output per worker growth fell from about 2.8% to 1.9% a year, and almost all of the decline is
TFP (from about 1.7% to 0.9%), while the contribution of capital deepening was stable at about 1 point. This replicates
the classic finding with a crude employment measure; using hours and a capital measure that excludes residential
structures changes the levels but not the story.
"""),

dict(title="Tobin's q: does market value predict investment?",
data="`grunfeld` (statsmodels) — investment, market value and capital stock of 11 large U.S. firms, 1935–1954.",
task="""
With adjustment costs, investment depends on Tobin's \\(q\\). Grunfeld's data give firm value \\(V\\) and capital \\(K\\); use
\\(V/K\\) as a (crude) average-\\(q\\) proxy.

1. Regress the investment rate \\(I_{it}/K_{it}\\) on lagged \\(V/K\\), with firm fixed effects.
2. Interpret the coefficient as \\(1/\\phi\\) for quadratic adjustment costs \\(\\frac{\\phi}{2}(I/K)^2K\\), and report the implied \\(\\phi\\).
3. Why does the \\(q\\) theory typically explain investment poorly, and what does measurement error in \\(q\\) do to the coefficient?
""",
starter="""
import numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
g = sm.datasets.grunfeld.load_pandas().data.sort_values(["firm", "year"])
g["ik"] = g["invest"] / g["capital"]; g["q"] = g["value"] / g["capital"]
# TODO: lag q by firm; ik ~ q_lag + C(firm)
""",
solution="""
import numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
g = sm.datasets.grunfeld.load_pandas().data.sort_values(["firm", "year"])
g = g[g["capital"] > 5].copy()
g["ik"] = g["invest"] / g["capital"]; g["q"] = g["value"] / g["capital"]
g["q_lag"] = g.groupby("firm")["q"].shift(1)
d = g.dropna(subset=["q_lag"])
r = smf.ols("ik ~ q_lag + C(firm)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["firm"]})
b = r.params["q_lag"]
print(f"investment rate on lagged V/K (firm FE): {b:.4f} (cluster se {r.bse['q_lag']:.4f}), within R2 approx {r.rsquared:.3f}")
print(f"implied adjustment-cost parameter phi = 1/b = {1/b:.1f}")
print(f"median I/K {d['ik'].median():.3f}; median V/K {d['q'].median():.2f}")
""",
notes="""The coefficient on lagged \\(V/K\\) is positive and significant, but small: it implies an adjustment-cost parameter of about
34, so firms would take decades to close a gap between \\(q\\) and one. That is the well-known failure of \\(q\\) theory —
implausibly slow adjustment. One reason is measurement: \\(V/K\\) is **average** q, the relevant concept is **marginal** q,
and market values are noisy, so the coefficient is attenuated (Erickson and Whited, 2000). With 11 firms, the clustered
standard error also rests on few clusters.
"""),
]

EXERCISES["03-solow-ramsey.md"] = [
dict(title="The Ramsey model: steady state, saddle path and transition",
data="Calibrated to standard annual U.S. values: \\(\\alpha=1/3\\), \\(\\beta=0.96\\), \\(\\delta=0.08\\), \\(\\sigma=2\\).",
task="""
In discrete time, the resource constraint is \\(k_{t+1}=k_t^{\\alpha}+(1-\\delta)k_t-c_t\\) and the Euler equation is \\(c_t^{-\\sigma}=\\beta c_{t+1}^{-\\sigma}(\\alpha k_{t+1}^{\\alpha-1}+1-\\delta)\\).

1. Compute the steady state \\((k^*,c^*)\\) and the saving rate.
2. Find the saddle path by **shooting**: from \\(k_0=0.5k^*\\), search for the initial \\(c_0\\) such that the path converges to the steady state (bisection on \\(c_0\\)).
3. Report the transition: how many years until capital closes half of the gap? How does the saving rate evolve along the path?
""",
starter="""
import numpy as np
a, b, d, s = 1/3, 0.96, 0.08, 2.0
kss = (a / (1/b - 1 + d))**(1 / (1 - a)); css = kss**a - d * kss
def path(c0, k0, T=200):
    # TODO: iterate k' from the resource constraint, c' from the Euler equation; return arrays (stop if k<0 or c<0)
    ...
""",
solution="""
import numpy as np
a, b, d, s = 1/3, 0.96, 0.08, 2.0
kss = (a / (1/b - 1 + d))**(1 / (1 - a)); css = kss**a - d * kss
print(f"steady state: k* = {kss:.4f}, c* = {css:.4f}, y* = {kss**a:.4f}, saving rate = {d*kss/kss**a:.3f}")
def path(c0, k0, T=150):
    k, c = [k0], [c0]
    for _ in range(T):
        k1 = k[-1]**a + (1 - d) * k[-1] - c[-1]
        if k1 <= 0: return np.array(k), np.array(c), "crash"
        c1 = c[-1] * (b * (a * k1**(a - 1) + 1 - d))**(1 / s)
        k.append(k1); c.append(c1)
        if c1 > 3 * css: return np.array(k), np.array(c), "explode"
    return np.array(k), np.array(c), "ok"
k0 = 0.5 * kss; lo, hi = 1e-6, k0**a + (1 - d) * k0
for _ in range(200):
    c0 = 0.5 * (lo + hi); k, c, status = path(c0, k0)
    if status == "crash" or (status == "ok" and k[-1] < kss): hi = c0      # consumed too much -> capital falls
    else: lo = c0                                                          # consumed too little -> overshoot
k, c, _ = path(c0, k0, 60)
print(f"saddle-path initial consumption c0 = {c0:.6f} ({c0/css:.1%} of c*)")
half = np.argmax(kss - k <= 0.5 * (kss - k0))
print(f"capital closes half of the gap after {half} years")
for t in (0, 5, 10, 20, 40):
    y = k[t]**a
    print(f"  t = {t:2d}: k/k* = {k[t]/kss:.3f}, c/c* = {c[t]/css:.3f}, saving rate = {(y - c[t])/y:.3f}")
""",
notes="""The steady-state saving rate is about 22%. Shooting finds the unique initial consumption (about 75% of its steady-state
level) that puts the economy on the saddle path; any other choice leads to capital exhaustion or oversaving. Capital closes
half the gap in about 9 years. Along the path, the saving rate **falls** from 26.5% toward 22%: with \\(\\sigma=2\\), poor
households save more while the return to capital is high. Unlike in Solow, the saving rate is an outcome, not a parameter.
"""),

dict(title="Is the U.S. saving too much or too little? The golden rule",
data="`macrodata` — U.S. gross investment as a share of GDP; capital stock from the perpetual-inventory method of Lecture 2.",
task="""
In the Solow model, steady-state consumption is maximized when the marginal product of capital net of depreciation equals the growth rate, \\(\\alpha Y/K-\\delta=n+g\\) — equivalently, when the saving rate equals the capital share \\(\\alpha\\).

1. Compute the U.S. average investment share and the average capital–output ratio from the perpetual-inventory capital stock.
2. Compute the implied net marginal product of capital \\(\\alpha Y/K-\\delta\\) (annual) and compare it with average real GDP growth.
3. Is the U.S. dynamically efficient (below the golden rule)? What would happen to steady-state consumption if the saving rate rose toward \\(\\alpha\\)?
""",
starter="""
import numpy as np, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
Y, I = m["realgdp"].to_numpy(), m["realinv"].to_numpy()
# TODO: perpetual inventory (delta = 0.025 per quarter); s = mean(I/Y); K/Y; MPK
""",
solution="""
import numpy as np, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
Y, I = m["realgdp"].to_numpy(), m["realinv"].to_numpy(); dl_q, a = 0.025, 1/3
K = np.empty_like(Y); K[0] = I[0] / (np.mean(np.diff(np.log(I))) + dl_q)
for t in range(len(Y) - 1): K[t + 1] = (1 - dl_q) * K[t] + I[t]
s = np.mean(I / Y); KY = np.mean(K[40:] / (4 * Y[40:]))          # annual K/Y, skipping the initialization decade
mpk = a / KY - 4 * dl_q
g = 4 * np.mean(np.diff(np.log(Y)))
print(f"investment share s = {s:.3f};  golden-rule saving rate = alpha = {a:.3f}")
print(f"capital-output ratio {KY:.2f} (annual);  net MPK = alpha*Y/K - delta = {100*mpk:.1f}%;  GDP growth = {100*g:.1f}%")
print(f"dynamically efficient (net MPK > growth): {mpk > g}")
n_g = g; ky_gold = a / (n_g + 4 * dl_q)
print(f"golden-rule K/Y would be {ky_gold:.2f} versus {KY:.2f} today")
""",
notes="""The U.S. investment share (13%) is far below the capital share (1/3), and the net return to capital comfortably exceeds the
growth rate, so the U.S. is **dynamically efficient**: it is below the golden rule, and higher saving would eventually raise
steady-state consumption, after a period of lower consumption. The levels here depend on the crude capital measure: with a
10% annual depreciation rate, the capital–output ratio is about 1, much lower than the official U.S. figure of about 3,
which includes structures with low depreciation. A larger \\(K/Y\\) lowers the net MPK, but it remains above the growth rate
(Abel et al., 1989).
"""),
]

EXERCISES["04-rbc.md"] = [
dict(title="U.S. business-cycle facts with the HP filter",
data="`macrodata` — U.S. quarterly real GDP, consumption, investment and government spending, 1959–2009.",
task="""
1. HP-filter the logs of real GDP, consumption, investment and government spending (\\(\\lambda=1600\\)).
2. Report for each the standard deviation (%), the standard deviation relative to output, the correlation with output, and the first-order autocorrelation.
3. Which facts must a business-cycle model reproduce? Which of them does the basic RBC model struggle with?
""",
starter="""
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
# TODO: cycles = {v: hpfilter(np.log(m[v]), 1600)[0] * 100 for v in [...]}
""",
solution="""
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
names = {"realgdp": "output", "realcons": "consumption", "realinv": "investment", "realgovt": "government"}
cyc = pd.DataFrame({lab: hpfilter(np.log(m[v]), 1600)[0] * 100 for v, lab in names.items()})
cyc["employment"] = hpfilter(np.log(m["pop"] * (1 - m["unemp"] / 100)), 1600)[0] * 100
out = pd.DataFrame({"sd %": cyc.std(), "rel. sd": cyc.std() / cyc["output"].std(),
                    "corr with output": cyc.corr()["output"], "autocorr": [cyc[c].autocorr() for c in cyc]})
print(out.round(2).to_string())
""",
notes="""These are the classic stylized facts: investment is about 4.7 times as volatile as output and strongly procyclical;
consumption is smoother (0.8 times as volatile) and procyclical; employment is procyclical and persistent; government
spending is essentially acyclical; all series are persistent (autocorrelations of 0.8–0.9). A business-cycle model must
reproduce these relative volatilities and co-movements. The basic RBC model does well on investment and consumption, but
it generates too little employment volatility and too much correlation between productivity and output.
"""),

dict(title="Simulating the stochastic growth model with full depreciation",
data="TFP process estimated from the U.S. Solow residual of Lecture 2 (rebuilt here); model calibration \\(\\alpha=1/3\\), \\(\\beta=0.99\\).",
task="""
With log utility and full depreciation (Brock–Mirman), the policy functions are known in closed form:
\\(k_{t+1}=\\alpha\\beta z_tk_t^{\\alpha}\\) and \\(c_t=(1-\\alpha\\beta)z_tk_t^{\\alpha}\\).

1. Re-estimate an AR(1) for the (linearly detrended) U.S. log Solow residual.
2. Simulate the model for 10,000 quarters, HP-filter simulated output, consumption and investment, and compute the same moments as in the previous exercise.
3. Compare with the data. Why does the model make consumption **exactly** as volatile as output, and what feature of the full RBC model (with \\(\\delta<1\\)) fixes this?
""",
starter="""
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
# TODO: rebuild log z (Lecture 2 exercise), detrend, AR(1); simulate k' = a b z k^a
""",
solution="""
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
Y, I = m["realgdp"].to_numpy(), m["realinv"].to_numpy(); dl, a, b = 0.025, 1/3, 0.99
K = np.empty_like(Y); K[0] = I[0] / (np.mean(np.diff(np.log(I))) + dl)
for t in range(len(Y) - 1): K[t + 1] = (1 - dl) * K[t] + I[t]
N = m["pop"].to_numpy() * (1 - m["unemp"].to_numpy() / 100)
lz = np.log(Y) - a * np.log(K) - (1 - a) * np.log(N)
tt = np.arange(len(lz)); res = lz - np.polyval(np.polyfit(tt, lz, 1), tt)
rho = np.corrcoef(res[1:], res[:-1])[0, 1]; sig = np.std(res[1:] - rho * res[:-1])
print(f"TFP: rho = {rho:.3f}, sigma = {100*sig:.3f}%")
rng = np.random.default_rng(1); T = 10000
z = np.zeros(T); k = np.zeros(T); k[0] = (a * b)**(1 / (1 - a))
for t in range(1, T): z[t] = rho * z[t - 1] + sig * rng.normal()
yv = np.zeros(T)
for t in range(T):
    yv[t] = np.exp(z[t]) * k[t]**a
    if t + 1 < T: k[t + 1] = a * b * yv[t]
c, inv = (1 - a * b) * yv, a * b * yv
cyc = pd.DataFrame({v: hpfilter(np.log(x), 1600)[0] * 100 for v, x in (("output", yv), ("consumption", c), ("investment", inv))})
print(pd.DataFrame({"sd %": cyc.std(), "rel. sd": cyc.std() / cyc["output"].std(), "corr with output": cyc.corr()["output"],
                    "autocorr": [cyc[x].autocorr() for x in cyc]}).round(2).to_string())
""",
notes="""The estimated TFP process (\\(\\rho\\approx0.96\\), \\(\\sigma\\approx0.7\\%\\)) is close to the standard RBC calibration. But with full
depreciation and log utility, consumption and investment are **constant shares** of output, so all three series have exactly
the same volatility and are perfectly correlated — far from the data, where investment is about six times as volatile as
consumption. With \\(\\delta<1\\), capital is a durable stock: households smooth consumption by letting investment absorb shocks,
which generates the observed volatility ranking. The simple model is useful as a check on numerical methods (Lecture 10),
not as a model of the cycle.
"""),
]

EXERCISES["05-nominal-rigidities.md"] = [
dict(title="From price stickiness to the slope of the Phillips curve",
data="Calibration: \\(\\beta=0.99\\), \\(\\sigma=1\\), \\(\\varphi=1\\) (Galí, 2015, ch. 3, without strategic complementarities).",
task="""
In the Calvo model, a share \\(\\theta\\) of firms cannot reset prices each quarter, so the average price duration is \\(1/(1-\\theta)\\), and the NKPC slope on the output gap is
\\(\\kappa=\\frac{(1-\\theta)(1-\\beta\\theta)}{\\theta}(\\sigma+\\varphi)\\).

1. Compute \\(\\kappa\\) for average durations of 2, 3, 4, 6 and 8 quarters.
2. Micro evidence (Bils and Klenow, 2004; Nakamura and Steinsson, 2008) suggests durations of 4–11 months for regular prices. What range of \\(\\kappa\\) does that imply?
3. Invert the mapping numerically: what duration is implied by a slope of \\(\\kappa=0.01\\), roughly what many macro estimates find?
""",
starter="""
import numpy as np
from scipy import optimize
b, s, phi = 0.99, 1.0, 1.0
kappa = lambda th: (1 - th) * (1 - b * th) / th * (s + phi)
# TODO: durations D -> theta = 1 - 1/D; invert kappa(theta) = 0.01 with brentq
""",
solution="""
import numpy as np
from scipy import optimize
b, s, phi = 0.99, 1.0, 1.0
kappa = lambda th: (1 - th) * (1 - b * th) / th * (s + phi)
for D in (2, 3, 4, 6, 8):
    th = 1 - 1 / D
    print(f"duration {D} quarters: theta = {th:.3f}, kappa = {kappa(th):.4f}")
lo, hi = kappa(1 - 1 / (4 / 3)), kappa(1 - 1 / (11 / 3))
print(f"durations of 4-11 months (1.33-3.67 quarters): kappa between {hi:.3f} and {lo:.3f}")
th = optimize.brentq(lambda t: kappa(t) - 0.01, 0.5, 0.9999)
print(f"kappa = 0.01 requires theta = {th:.4f}, i.e. an average price duration of {1/(1-th):.1f} quarters ({1/(1-th)/4:.1f} years)")
""",
notes="""The slope \\(\\kappa\\) falls steeply with price duration: from 1.0 at 2 quarters to 0.17 at one year and 0.04 at two years.
Micro evidence on regular price durations of 4–11 months implies \\(\\kappa\\) of 0.2 or more. Yet a slope of 0.01, typical of
aggregate estimates, would require prices to stay fixed for almost four years on average. This micro–macro gap is why
models add strategic complementarities (real rigidities), which flatten the Phillips curve for a given frequency of price
changes.
"""),

dict(title="Estimating a New Keynesian Phillips curve by GMM",
data="`macrodata` — U.S. quarterly CPI inflation and unemployment, 1960–2007.",
task="""
Estimate the hybrid NKPC \\(\\pi_t=\\gamma_f E_t\\pi_{t+1}+\\gamma_b\\pi_{t-1}+\\kappa\\,x_t+\\varepsilon_t\\), with the output gap proxied by the negative unemployment gap \\(x_t=-(u_t-\\bar u_t)\\), \\(\\bar u_t\\) the HP trend of unemployment.

1. Replace \\(E_t\\pi_{t+1}\\) with realized \\(\\pi_{t+1}\\) and estimate by 2SLS with instruments dated \\(t-1\\) and earlier (lags 1–4 of inflation, lags 1–2 of the gap and of the T-bill rate).
2. Report \\(\\gamma_f\\), \\(\\gamma_b\\) and \\(\\kappa\\). Is the curve flat? Is inflation mostly forward- or backward-looking?
3. Discuss why weak instruments plague NKPC estimation (Mavroeidis, Plagborg-Møller and Stock, 2014).
""",
starter="""
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
from linearmodels.iv import IV2SLS
m = sm.datasets.macrodata.load_pandas().data
# TODO: x = -(u - HP trend of u); pi_lead = pi.shift(-1); instruments: lags
""",
solution="""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.filters.hp_filter import hpfilter
from linearmodels.iv import IV2SLS
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"year": m["year"], "pi": m["infl"], "x": -hpfilter(m["unemp"], 1600)[0], "i": m["tbilrate"]})
d["pi_f"], d["pi_b"] = d["pi"].shift(-1), d["pi"].shift(1)
Z = []
for k in (2, 3, 4):
    d[f"pi_l{k}"] = d["pi"].shift(k); Z.append(f"pi_l{k}")
for v in ("x", "i"):
    for k in (1, 2):
        d[f"{v}_l{k}"] = d[v].shift(k); Z.append(f"{v}_l{k}")
d = d[(d.year >= 1960) & (d.year <= 2007)].dropna()
iv = IV2SLS.from_formula(f"pi ~ 1 + pi_b + x + [pi_f ~ {' + '.join(Z)}]", data=d).fit(cov_type="kernel")
print(iv.params[["pi_f", "pi_b", "x"]].round(3).to_string())
print("HAC se:", iv.std_errors[["pi_f", "pi_b", "x"]].round(3).to_dict())
fs = smf.ols(f"pi_f ~ pi_b + x + {' + '.join(Z)}", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
F = fs.wald_test(", ".join(f"{z} = 0" for z in Z), scalar=True).statistic / len(Z)
print(f"first-stage F for pi_(t+1) on excluded instruments: {F:.1f};  gamma_f + gamma_b = {iv.params['pi_f'] + iv.params['pi_b']:.3f}")
""",
notes="""The estimates put most weight on expected future inflation (\\(\\gamma_f\\approx0.85\\)) and little on lagged inflation, with the
two summing to slightly below one. The slope on the unemployment gap is positive but imprecise — not significantly
different from zero: the Phillips curve is flat and poorly identified. The first-stage \\(F\\) of about 13 looks acceptable,
but NKPC estimates are notoriously sensitive to the instrument set, the gap measure and the sample, because lagged variables
predict future inflation only weakly once current inflation is controlled for (Mavroeidis, Plagborg-Møller and Stock, 2014).
Use weak-instrument-robust methods before drawing strong conclusions.
"""),
]

EXERCISES["06-three-equation-nk.md"] = [
dict(title="Solving the three-equation NK model by undetermined coefficients",
data="Calibration following Galí (2015, ch. 3): \\(\\beta=0.99\\), \\(\\sigma=1\\), \\(\\kappa=0.17\\), \\(\\phi_\\pi=1.5\\), \\(\\phi_y=0.125\\), monetary shock persistence \\(\\rho_v=0.5\\).",
task="""
The model: \\(x_t=E_tx_{t+1}-\\frac1\\sigma(i_t-E_t\\pi_{t+1})\\); \\(\\pi_t=\\beta E_t\\pi_{t+1}+\\kappa x_t\\); \\(i_t=\\phi_\\pi\\pi_t+\\phi_yx_t+v_t\\), \\(v_t=\\rho_vv_{t-1}+\\varepsilon_t\\).

1. Guess \\(x_t=\\psi_xv_t\\), \\(\\pi_t=\\psi_\\pi v_t\\), and solve for the coefficients with `sympy`.
2. Compute impulse responses of the output gap, inflation, the nominal rate and the real rate to a 25 basis-point shock (annualized: 1 percentage point).
3. Repeat for shock persistence \\(\\rho_v=0.8\\) and \\(0.9\\). Why can the nominal rate **fall** after a contractionary monetary shock, while the real rate rises?
""",
starter="""
import sympy as sp
px, pp, b, s, k, fp, fy, r = sp.symbols("psi_x psi_pi beta sigma kappa phi_pi phi_y rho")
# x = px v, pi = pp v, E v' = r v
eqs = [sp.Eq(px, r * px - (1 / s) * (fp * pp + fy * px + 1 - r * pp)), sp.Eq(pp, b * r * pp + k * px)]
# TODO: sp.solve(eqs, [px, pp])
""",
solution="""
import numpy as np, sympy as sp
px, pp, b, s, k, fp, fy, r = sp.symbols("psi_x psi_pi beta sigma kappa phi_pi phi_y rho")
eqs = [sp.Eq(px, r * px - (1 / s) * (fp * pp + fy * px + 1 - r * pp)), sp.Eq(pp, b * r * pp + k * px)]
sol = sp.solve(eqs, [px, pp], dict=True)[0]
print("psi_x  =", sp.simplify(sol[px])); print("psi_pi =", sp.simplify(sol[pp]))
cal = {b: 0.99, s: 1.0, k: 0.17, fp: 1.5, fy: 0.125, r: 0.5}
cx, cp = float(sol[px].subs(cal)), float(sol[pp].subs(cal))
v = 0.25 * 0.5**np.arange(7)                                     # 25 bp shock, quarterly rates
x, pi = cx * v, cp * v; i = 1.5 * pi + 0.125 * x + v; rr = i - np.r_[pi[1:], 0.5 * pi[-1]]
print("quarter | output gap % | inflation (ann. %) | nominal rate (ann. %) | real rate (ann. %)")
for t in range(7):
    print(f"   {t}    |   {x[t]:7.3f}    |      {4*pi[t]:7.3f}       |       {4*i[t]:7.3f}         |     {4*rr[t]:7.3f}")
for rv in (0.5, 0.8, 0.9):
    c2 = {**cal, r: rv}; x0, p0 = 0.25 * float(sol[px].subs(c2)), 0.25 * float(sol[pp].subs(c2))
    i0 = 1.5 * p0 + 0.125 * x0 + 0.25
    print(f"rho_v = {rv}: impact nominal rate {4*i0:+.3f} pp, real rate {4*(i0 - rv*p0):+.3f} pp, output gap {x0:+.3f}%")
""",
notes="""A 25 basis-point (1 point annualized) contractionary shock lowers the output gap by about 0.26% and annualized inflation by
about 0.35 points on impact; both responses decay at the shock's rate \\(\\rho_v\\). The real rate rises by more than the nominal
rate because expected inflation falls. With more persistent shocks, the fall in inflation and output is so large that the
Taylor rule's endogenous response **lowers** the nominal rate on impact even though policy is tighter: with
\\(\\rho_v=0.8\\), the nominal rate falls by about 0.5 points while the real rate still rises. The nominal interest rate is a
poor indicator of the policy stance.
"""),

dict(title="The Taylor principle and determinacy",
data="Same calibration (\\(\\beta=0.99\\), \\(\\sigma=1\\), \\(\\kappa=0.17\\)).",
task="""
Write the model without shocks as \\(E_tz_{t+1}=Mz_t\\) with \\(z_t=(x_t,\\pi_t)'\\). With two forward-looking variables, the equilibrium is determinate when both eigenvalues of \\(M\\) lie outside the unit circle.

1. Derive \\(M\\) and compute its eigenvalues on a grid of \\(\\phi_\\pi\\in[0,3]\\) for \\(\\phi_y=0\\) and \\(\\phi_y=0.5\\).
2. Find the critical \\(\\phi_\\pi\\) numerically and compare with the analytical condition \\(\\kappa(\\phi_\\pi-1)+(1-\\beta)\\phi_y>0\\) (Bullard and Mitra, 2002).
3. Interpret: why does the central bank have to raise the nominal rate more than one-for-one with inflation?
""",
starter="""
import numpy as np
b, s, k = 0.99, 1.0, 0.17
def M(fp, fy):
    return np.array([[1 + fy / s + k / (s * b), (fp - 1 / b) / s], [-k / b, 1 / b]])
# TODO: eigenvalues over a grid; find where min |eig| crosses 1
""",
solution="""
import numpy as np
from scipy import optimize
b, s, k = 0.99, 1.0, 0.17
def M(fp, fy):
    return np.array([[1 + fy / s + k / (s * b), (fp - 1 / b) / s], [-k / b, 1 / b]])
determinate = lambda fp, fy: np.min(np.abs(np.linalg.eigvals(M(fp, fy)))) > 1
for fy in (0.0, 0.5):
    grid = np.linspace(0, 3, 3001); det = np.array([determinate(f, fy) for f in grid])
    crit = grid[np.argmax(det)]
    analytic = 1 - (1 - b) * fy / k
    print(f"phi_y = {fy}: determinate for phi_pi > {crit:.3f} (grid);  Bullard-Mitra threshold {analytic:.3f}")
for fp in (0.8, 1.0, 1.2, 1.5):
    print(f"  phi_pi = {fp}, phi_y = 0: |eigenvalues| = {np.abs(np.linalg.eigvals(M(fp, 0))).round(4)}, determinate: {determinate(fp, 0)}")
""",
notes="""With \\(\\phi_y=0\\), determinacy requires \\(\\phi_\\pi>1\\) — the Taylor principle — and the numerical boundary matches the
Bullard–Mitra condition. Responding to the output gap relaxes it slightly: with \\(\\phi_y=0.5\\), \\(\\phi_\\pi>0.97\\) suffices,
because in the long run a permanent rise in inflation also raises the output gap by \\((1-\\beta)/\\kappa\\). Below the threshold,
one eigenvalue lies inside the unit circle and sunspot equilibria exist: self-fulfilling inflation expectations are not ruled
out, because the real rate falls when inflation rises.
"""),
]

EXERCISES["07-monetary-policy.md"] = [
dict(title="Taylor rules before and after Volcker",
data="`macrodata` — U.S. quarterly 3-month T-bill rate, CPI inflation and real GDP.",
task="""
Clarida, Galí and Gertler (2000) argue that U.S. policy was "passive" (\\(\\phi_\\pi<1\\)) before 1979 and "active" (\\(\\phi_\\pi>1\\)) afterwards.

1. Construct four-quarter inflation and the HP output gap.
2. Estimate a Taylor rule with interest-rate smoothing, \\(i_t=(1-\\rho)(c+\\phi_\\pi\\pi_t+\\phi_xx_t)+\\rho i_{t-1}+e_t\\), by nonlinear least squares for 1960–1979 and 1987–2007.
3. Report the long-run responses \\(\\phi_\\pi,\\phi_x\\) with delta-method standard errors, and relate them to the determinacy condition of Lecture 6.
""",
starter="""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
# TODO: pi4 = rolling 4-quarter mean of infl; gap = 100*HP cycle of log GDP; OLS of i on pi4, gap, i_lag -> recover long-run coefficients
""",
solution="""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"year": m["year"], "i": m["tbilrate"], "pi": m["infl"].rolling(4).mean(),
                  "x": 100 * hpfilter(np.log(m["realgdp"]), 1600)[0]})
d["i_lag"] = d["i"].shift(1)
for lo, hi in ((1960, 1979), (1987, 2007)):
    s = d[(d.year >= lo) & (d.year <= hi)].dropna()
    r = smf.ols("i ~ pi + x + i_lag", data=s).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    rho = r.params["i_lag"]; V = r.cov_params()
    out = {}
    for v in ("pi", "x"):
        est = r.params[v] / (1 - rho)
        g = np.zeros(len(r.params)); g[list(r.params.index).index(v)] = 1 / (1 - rho); g[list(r.params.index).index("i_lag")] = r.params[v] / (1 - rho)**2
        out[v] = (est, np.sqrt(g @ V.to_numpy() @ g))
    print(f"{lo}-{hi}: smoothing rho = {rho:.3f};  long-run phi_pi = {out['pi'][0]:.2f} (se {out['pi'][1]:.2f}), phi_x = {out['x'][0]:.2f} (se {out['x'][1]:.2f})")
""",
notes="""For 1960–79, the long-run inflation response is about 0.56, well below one: policy accommodated inflation, consistent with
Clarida, Galí and Gertler (2000). For 1987–2007, this simple specification does **not** find \\(\\phi_\\pi>1\\): the point estimate
is similar but has a huge standard error (0.73), because inflation varied little after the mid-1980s and smoothing is high, so
the inflation response is poorly identified. CGG use expected future inflation (GMM), the GDP deflator, the federal funds
rate and a different gap measure, and find about 2. Specification choices matter a lot for this famous result — an honest
lesson.
"""),

dict(title="The effective lower bound in 2009: what did a Taylor rule prescribe?",
data="`macrodata` — U.S. data through 2009Q3.",
task="""
1. Compute the rate prescribed by Taylor's (1993) original rule, \\(i_t=1+1.5\\pi_t+0.5x_t\\), for each quarter of 2008–2009, using an output gap whose trend is estimated only through 2007 (to avoid the end-point problem of the HP filter).
2. Compare it with the actual T-bill rate and with the rule estimated on 1987–2007 in the previous exercise. In which quarter did the prescribed rate first fall below zero, and how negative did it get?
3. What tools did the Fed use when the rule called for negative rates (forward guidance, asset purchases), and how does a "shadow rate" summarize them?
""",
starter="""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
# TODO: estimate on 1987-2007 as before; prescribed = c + phi_pi * pi + phi_x * x for 2008-2009
""",
solution="""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"year": m["year"], "q": m["quarter"], "i": m["tbilrate"], "pi": m["infl"].rolling(4).mean(),
                  "x": 100 * hpfilter(np.log(m["realgdp"].loc[:195]), 1600)[0].reindex(m.index)})
# HP trend estimated through 2007Q4 only and extended linearly, to avoid the end-point problem of the filter
tr = np.log(m["realgdp"]) - d["x"] / 100
slope = tr.loc[190:195].diff().mean()
d.loc[196:, "x"] = 100 * (np.log(m["realgdp"].loc[196:]) - (tr.loc[195] + slope * np.arange(1, len(m) - 195)))
d["i_lag"] = d["i"].shift(1)
s = d[(d.year >= 1987) & (d.year <= 2007)].dropna()
r = smf.ols("i ~ pi + x + i_lag", data=s).fit()
rho = r.params["i_lag"]; c, fpi, fx = (r.params[v] / (1 - rho) for v in ("Intercept", "pi", "x"))
print(f"estimated 1987-2007 rule: i = {c:.2f} + {fpi:.2f} pi + {fx:.2f} x;   Taylor (1993): i = 1 + 1.5 pi + 0.5 x")
for _, row in d[d.year >= 2008].iterrows():
    print(f"  {int(row.year)}Q{int(row.q)}: inflation {row.pi:5.2f}, gap {row.x:6.2f} -> Taylor 1993 {1 + 1.5*row.pi + 0.5*row.x:6.2f}%, estimated rule {c + fpi*row.pi + fx*row.x:7.2f}%, actual T-bill {row.i:.2f}%")
""",
notes="""Taylor's 1993 rule would have called for a negative policy rate from 2008Q4, reaching about −5.8% in 2009Q2 — close to
estimates in the literature (Rudebusch, 2009). The rule estimated on 1987–2007 gives absurd values (below −18%), because its
large output-gap coefficient was estimated in a period of small gaps — an extrapolation warning. The T-bill rate was stuck
near zero. With conventional policy constrained, the Fed turned to forward guidance and large-scale asset purchases;
"shadow rate" estimates (Wu and Xia, 2016) translate their effect into an equivalent negative policy rate of a few percent.
"""),
]

EXERCISES["08-fiscal-debt.md"] = [
dict(title="Debt dynamics and the debt-stabilizing primary surplus",
data="`intdef` — U.S. federal receipts and outlays (% of GDP), 3-month T-bill rate and inflation, 1948–2003; `macrodata` for real GDP growth.",
task="""
Government debt evolves as \\(b_{t+1}=\\frac{1+r}{1+g}b_t-s_t\\), where \\(b\\) is debt/GDP, \\(r\\) the real interest rate, \\(g\\) real growth and \\(s\\) the primary surplus/GDP.

1. Compute the average real T-bill rate \\(r\\) and average real GDP growth \\(g\\). What is \\(r-g\\), and why does its sign matter?
2. Compute the debt-stabilizing primary surplus \\(s^*=\\frac{r-g}{1+g}b\\) at debt ratios of 60% and 100%.
3. Simulate debt for 30 years from \\(b_0=60\\%\\) under (a) \\(s=0\\), (b) \\(s=-2\\%\\) (a primary deficit), for \\(r-g\\) equal to its historical value and to \\(+2\\) percentage points.
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.api as sm
d = wd.data("intdef")
r = (d["i3"] - d["inf"]).mean() / 100
m = sm.datasets.macrodata.load_pandas().data; g = 4 * np.mean(np.diff(np.log(m["realgdp"])))
# TODO: s* = (r-g)/(1+g) b; simulate b' = (1+r)/(1+g) b - s
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.api as sm
d = wd.data("intdef")
r = (d["i3"] - d["inf"]).mean() / 100
m = sm.datasets.macrodata.load_pandas().data; g = 4 * np.mean(np.diff(np.log(m["realgdp"])))
print(f"average real T-bill rate r = {100*r:.2f}%, average real growth g = {100*g:.2f}%, r - g = {100*(r-g):.2f} pp")
print(f"average federal deficit (outlays - receipts, including interest) = {(d['out'] - d['rec']).mean():.2f}% of GDP")
for b in (0.6, 1.0):
    print(f"debt-stabilizing primary surplus at b = {b:.0%}: {100*(r-g)/(1+g)*b:+.2f}% of GDP")
for rg_label, rr in (("historical r-g", r), ("r-g = +2 pp", g + 0.02)):
    for s in (0.0, -0.02):
        b = 0.6
        for _ in range(30): b = (1 + rr) / (1 + g) * b - s
        print(f"  {rg_label:15s}, primary surplus {100*s:+.0f}%: debt after 30 years = {100*b:.1f}% of GDP")
""",
notes="""Over this sample, the average real T-bill rate (about 1%) was well below real growth (about 3%), so \\(r-g\\approx-2\\) points.
With \\(r<g\\), debt stabilizes even with a primary **deficit**: about 1.2% of GDP at a 60% debt ratio. With a balanced primary
budget, debt falls from 60% to about 33% in 30 years. If \\(r-g\\) rises to +2 points, the same policies make debt explode — a
2% primary deficit takes it to almost 190%. The sign of \\(r-g\\) is central to the debate on fiscal space (Blanchard, 2019).
(The T-bill rate understates the average cost of federal debt, which has longer maturities.)
"""),

dict(title="A government-spending multiplier from a recursive VAR",
data="`macrodata` — U.S. quarterly real government spending, real GDP and real consumption, per capita, 1959–2009.",
task="""
Following Blanchard and Perotti (2002), assume government spending does not respond to output within the quarter, so
spending is ordered first in a Cholesky VAR.

1. Estimate a VAR(4) in log real per-capita government spending, GDP and consumption, with a linear trend.
2. Compute the impulse responses to a spending shock, and convert them into **dollar multipliers** using the sample average ratios \\(Y/G\\) and \\(C/G\\).
3. Report the impact and cumulative 8-quarter multipliers \\(\\sum\\Delta Y/\\sum\\Delta G\\). Does consumption rise or fall?
""",
starter="""
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.api import VAR
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"g": np.log(m["realgovt"] / m["pop"]), "y": np.log(m["realgdp"] / m["pop"]), "c": np.log(m["realcons"] / m["pop"])})
# TODO: VAR(d).fit(4, trend="ct"); irf(12).orth_irfs; multipliers = response_y / response_g * (Y/G)
""",
solution="""
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.api import VAR
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"g": np.log(m["realgovt"] / m["pop"]), "y": np.log(m["realgdp"] / m["pop"]), "c": np.log(m["realcons"] / m["pop"])})
res = VAR(d).fit(4, trend="ct"); ir = res.irf(12).orth_irfs[:, :, 0]          # responses to the spending shock
YG, CG = (m["realgdp"] / m["realgovt"]).mean(), (m["realcons"] / m["realgovt"]).mean()
dG, dY, dC = ir[:, 0], ir[:, 1] * YG, ir[:, 2] * CG                             # in units of G
print(f"average Y/G = {YG:.2f}, C/G = {CG:.2f}")
for h in (0, 1, 2, 4, 8):
    print(f"  h = {h}: dG {dG[h]:.4f}, dY/dG(0) = {dY[h]/dG[0]:.2f}, dC/dG(0) = {dC[h]/dG[0]:.2f}")
print(f"impact multiplier dY/dG = {dY[0]/dG[0]:.2f};  cumulative 8-quarter multiplier = {dY[:9].sum()/dG[:9].sum():.2f};  consumption: {dC[:9].sum()/dG[:9].sum():.2f}")
""",
notes="""On impact, output rises by about \\$0.50 per dollar of spending, but the effect turns negative within a few quarters, and the
cumulative 8-quarter multiplier is slightly negative, with consumption falling. That is lower than Blanchard and Perotti's
(2002) estimates (around 1), which use tax data, quarterly fiscal series and a different trend specification. Ramey (2011)
argues that recursive VARs mismeasure the timing of spending shocks, because spending is anticipated (military buildups are
announced in advance), and that narrative shocks give multipliers between 0.6 and 1.2. Multiplier estimates are fragile.
"""),
]

EXERCISES["09-open-economy.md"] = [
dict(title="Current-account responses to temporary and persistent income shocks",
data="`macrodata` — U.S. quarterly real GDP; the persistence of its cyclical component calibrates the income process.",
task="""
In a small open economy with the permanent-income hypothesis, consumption responds to the annuity value of an income
shock. If detrended income follows an AR(1) with persistence \\(\\rho\\), a unit shock raises consumption by
\\(\\frac{r}{1+r-\\rho}\\), so the current account changes by \\(CA=1-\\frac{r}{1+r-\\rho}\\) on impact.

1. Estimate \\(\\rho\\) for the quarterly HP cycle of U.S. log real GDP and for its linearly detrended log.
2. With \\(r=1\\%\\) per quarter, compute the impact responses of consumption and the current account for both persistence estimates and for \\(\\rho\\in\\{0,0.5,0.9,0.99,1\\}\\).
3. Interpret: why do countries run current-account **surpluses** after temporary income gains, but not after permanent ones?
""",
starter="""
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data; ly = np.log(m["realgdp"].to_numpy())
# TODO: rho from AR(1) of the HP cycle and of linear-detrended log GDP; responses r/(1+r-rho)
""",
solution="""
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data; ly = np.log(m["realgdp"].to_numpy())
ar1 = lambda x: np.corrcoef(x[1:], x[:-1])[0, 1]
hp = hpfilter(ly, 1600)[0]; t = np.arange(len(ly)); lin = ly - np.polyval(np.polyfit(t, ly, 1), t)
r = 0.01
cases = {"HP cycle": ar1(hp), "linear detrend": ar1(lin), "rho = 0": 0.0, "rho = 0.5": 0.5, "rho = 0.9": 0.9, "rho = 0.99": 0.99, "random walk": 1.0}
for lab, rho in cases.items():
    dc = r / (1 + r - rho)
    print(f"{lab:15s}: rho = {rho:.3f} -> consumption +{dc:.3f}, current account {1 - dc:+.3f} per unit income shock")
""",
notes="""With the HP cycle's persistence (0.86), a unit income shock raises consumption by only 0.07 and the current account by
0.93: households save almost all of a temporary gain abroad. With the linearly detrended series (0.97), consumption rises by
a quarter. As persistence approaches one, the consumption response approaches one and the current account response
vanishes. That is why the current account should absorb temporary shocks (a harvest failure, a temporary oil price boom) but
not permanent ones. The estimated persistence depends heavily on the detrending method — the same issue as in the RBC
calibration.
"""),

dict(title="Dornbusch overshooting, solved and simulated",
data="Stylized calibration of the Dornbusch (1976) model: money-demand semi-elasticity to the quarterly interest rate \\(\\lambda=4\\); price-level convergence speed \\(\\nu\\) per quarter varied from 0.05 to 1; long-run PPP.",
task="""
With sticky prices, uncovered interest parity and money demand \\(m-p=\\phi y-\\lambda i\\), a permanent money increase raises
the long-run price level and exchange rate one-for-one. On the saddle path, the exchange rate satisfies
\\(e_t-\\bar e=-\\frac{1}{\\lambda\\nu}(p_t-\\bar p)\\), where \\(\\nu\\) is the speed at which prices converge,
\\(p_{t+1}-p_t=\\nu(\\bar p-p_t)\\).

1. For a permanent 10% money increase, compute the impact depreciation and the overshooting ratio \\(1+\\frac{1}{\\lambda\\nu}\\) for several values of \\(\\nu\\).
2. Simulate prices, the exchange rate and the interest differential (from money-market equilibrium, \\(i-i^*=(p-m)/\\lambda\\)) for 20 quarters with \\(\\nu=0.25\\). Check that the differential and the expected appreciation satisfy UIP.
3. How does overshooting depend on price stickiness?
""",
starter="""
import numpy as np
lam, dm = 4.0, 10.0
# impact: e0 = dm * (1 + 1/(lam*nu)); then p' = p + nu (dm - p), e = dm + (dm - p)/(lam*nu)
""",
solution="""
import numpy as np
lam, dm = 4.0, 10.0
for nu in (0.05, 0.10, 0.25, 0.50, 1.00):
    jump = dm * (1 + 1 / (lam * nu))
    print(f"nu = {nu:.2f} (half-life of price gap {np.log(0.5)/np.log(1-nu) if nu < 1 else 0:.1f} q): impact depreciation {jump:.1f}% (overshooting ratio {jump/dm:.2f})")
nu, p = 0.25, 0.0; e_prev = None
print("quarter | price level | exchange rate | i - i* (pp/quarter) | expected depreciation next quarter")
for t in range(0, 21):
    e = dm + (dm - p) / (lam * nu); idiff = (p - dm) / lam
    p_next = p + nu * (dm - p); e_next = dm + (dm - p_next) / (lam * nu)
    if t % 4 == 0:
        print(f"  {t:2d}    |   {p:6.2f}    |    {e:6.2f}     |       {idiff:6.3f}        |        {e_next - e:6.3f}")
    p = p_next
""",
notes="""On impact, prices are fixed and the exchange rate jumps by more than the 10% long-run depreciation: by 20% when the price gap
closes 25% a quarter, and by 60% when prices are very sticky. Higher real balances lower the domestic interest rate, and
uncovered interest parity then requires an expected **appreciation** of exactly the same size — the two columns of the
simulation coincide. The more sluggish prices are, the larger the overshooting. This explains why exchange rates are so much
more volatile than the fundamentals that drive them.
"""),
]

EXERCISES["10-dsge-solution-methods.md"] = [
dict(title="Linearize the Ramsey model and solve it with the QZ decomposition",
data="Calibration as in Lecture 3: \\(\\alpha=1/3\\), \\(\\beta=0.96\\), \\(\\delta=0.08\\), \\(\\sigma=2\\).",
task="""
1. Write the two equilibrium conditions \\(f(k_{t+1},c_{t+1},k_t,c_t)=0\\) and linearize them **numerically** (finite differences) around the steady state into \\(A\\,E_ty_{t+1}=B\\,y_t\\), with \\(y=(\\hat k,\\hat c)\\).
2. Use `scipy.linalg.ordqz` to compute the generalized eigenvalues. Check the Blanchard–Kahn condition: one unstable root for one jump variable (\\(c\\)).
3. Recover the policy function \\(\\hat c_t=h\\,\\hat k_t\\) and the law of motion \\(\\hat k_{t+1}=g\\,\\hat k_t\\). Compare the linear transition with the nonlinear shooting solution of Lecture 3 after 10 years.
""",
starter="""
import numpy as np
from scipy.linalg import ordqz
a, b, d, s = 1/3, 0.96, 0.08, 2.0
kss = (a / (1/b - 1 + d))**(1 / (1 - a)); css = kss**a - d * kss
def f(k1, c1, k0, c0):
    return np.array([k1 - (k0**a + (1 - d) * k0 - c0), c0**(-s) - b * c1**(-s) * (a * k1**(a - 1) + 1 - d)])
# TODO: Jacobians by finite differences; A = df/d(k1,c1) * ss, B = -df/d(k0,c0) * ss (log-deviations)
""",
solution="""
import numpy as np
from scipy.linalg import ordqz
a, b, d, s = 1/3, 0.96, 0.08, 2.0
kss = (a / (1/b - 1 + d))**(1 / (1 - a)); css = kss**a - d * kss
def f(x):                                     # x = (k1, c1, k0, c0) in levels
    k1, c1, k0, c0 = x
    return np.array([k1 - (k0**a + (1 - d) * k0 - c0), c0**(-s) - b * c1**(-s) * (a * k1**(a - 1) + 1 - d)])
x0 = np.array([kss, css, kss, css]); h = 1e-7; J = np.zeros((2, 4))
for j in range(4):
    e = np.zeros(4); e[j] = h * x0[j]; J[:, j] = (f(x0 + e) - f(x0 - e)) / (2 * h)      # derivative w.r.t. log deviations
A, B = J[:, :2], -J[:, 2:]                     # A E[y_{t+1}] = B y_t
AA, BB, alpha, beta_, Q, Z = ordqz(B, A, sort="iuc", output="real")    # stable generalized eigenvalues first
lam = alpha / beta_
print("generalized eigenvalues:", np.round(lam, 4), " -> unstable roots:", int(np.sum(np.abs(lam) > 1)), "(Blanchard-Kahn needs 1)")
Z11, Z21 = Z[0, 0], Z[1, 0]
hc = Z21 / Z11                                 # c_hat = h k_hat on the stable subspace
g = float(np.real(lam[np.abs(lam) < 1][0]))
print(f"policy: c_hat = {hc:.4f} k_hat;  law of motion: k_hat' = {g:.4f} k_hat  (half-life {np.log(0.5)/np.log(g):.1f} years)")
k0 = np.log(0.5)
print(f"linear solution from k0 = 0.5 k*: k/k* after 10 years = {np.exp(g**10 * k0):.3f}, initial c/c* = {np.exp(hc * k0):.3f}")
""",
notes="""The linearized system has one stable and one unstable generalized eigenvalue, matching one predetermined variable (capital)
and one jump variable (consumption): the Blanchard–Kahn condition holds and the saddle path is unique. The stable root, 0.918,
implies a half-life of about 8 years, and the policy function sets consumption 0.43% above steady state for each 1% of extra
capital. Started at half the steady-state capital, the linear solution gives \\(k/k^*=0.745\\) after 10 years against 0.775 in the
nonlinear shooting solution: linearization is accurate near the steady state but less so far from it.
"""),

dict(title="Value function iteration versus the exact solution",
data="Brock–Mirman model: log utility, full depreciation, \\(\\alpha=1/3\\), \\(\\beta=0.96\\) — the exact policy \\(k'=\\alpha\\beta k^{\\alpha}\\) is known.",
task="""
1. Solve the deterministic model \\(V(k)=\\max_{k'}\\{\\log(k^{\\alpha}-k')+\\beta V(k')\\}\\) by value function iteration on grids of 100, 300 and 1,000 points (vectorized).
2. Report the number of iterations, computing time, and the maximum error of the policy function relative to the exact \\(k'=\\alpha\\beta k^{\\alpha}\\).
3. Why does the error fall only slowly with the number of grid points, and what methods (interpolation, endogenous grids — Lecture 12) do better?
""",
starter="""
import numpy as np, time
a, b = 1/3, 0.96
kss = (a * b)**(1 / (1 - a))
# TODO: grid around kss; utility matrix U[i, j] = log(k_i^a - k_j); iterate V = max_j (U + b V_j)
""",
solution="""
import numpy as np, time
a, b = 1/3, 0.96
kss = (a * b)**(1 / (1 - a))
for n in (100, 300, 1000):
    k = np.linspace(0.2 * kss, 2.0 * kss, n)
    c = k[:, None]**a - k[None, :]
    U = np.where(c > 0, np.log(np.maximum(c, 1e-300)), -1e10)
    V = np.zeros(n); t0 = time.time()
    for it in range(2000):
        W = U + b * V[None, :]; V1 = W.max(1)
        if np.max(np.abs(V1 - V)) < 1e-8: break
        V = V1
    pol = k[W.argmax(1)]
    inner = (k > 0.3 * kss) & (k < 1.8 * kss)
    err = np.max(np.abs(pol - a * b * k**a)[inner])
    print(f"grid {n:5d}: {it:4d} iterations, {time.time() - t0:6.2f} s, max policy error {err:.2e} (grid spacing {k[1]-k[0]:.2e})")
""",
notes="""VFI converges in about 450 iterations regardless of the grid, because the contraction rate is \\(\\beta\\). The policy error falls
roughly in proportion to the grid spacing (first order), because the maximization is restricted to grid points: a 10 times
finer grid gives about 10 times smaller errors, at about 100 times the computing time, since the vectorized search is
\\(O(n^2)\\). Interpolating the value function, using first-order conditions (policy-function iteration), or the endogenous grid
method (Lecture 12) reach much higher accuracy at a fraction of the cost.
"""),
]

EXERCISES["11-dsge-estimation.md"] = [
dict(title="Estimating the TFP process that drives an RBC model",
data="`macrodata` — U.S. Solow residual constructed as in Lecture 2.",
task="""
1. Rebuild the quarterly log Solow residual, and estimate an AR(1) after removing a linear trend, by OLS and by exact Gaussian maximum likelihood (`statsmodels` ARIMA).
2. Report \\(\\rho\\) and \\(\\sigma_\\varepsilon\\) with standard errors, and the half-life of a TFP shock.
3. Compare with the standard RBC calibration \\(\\rho=0.95\\), \\(\\sigma=0.007\\) (Cooley and Prescott, 1995). How sensitive is \\(\\rho\\) to the detrending method (linear trend versus HP filter)?
""",
starter="""
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
# TODO: rebuild log z; detrend; ARIMA(z, order=(1,0,0), trend="n").fit()
""",
solution="""
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
Y, I = m["realgdp"].to_numpy(), m["realinv"].to_numpy(); dl, a = 0.025, 1/3
K = np.empty_like(Y); K[0] = I[0] / (np.mean(np.diff(np.log(I))) + dl)
for t in range(len(Y) - 1): K[t + 1] = (1 - dl) * K[t] + I[t]
N = m["pop"].to_numpy() * (1 - m["unemp"].to_numpy() / 100)
lz = (np.log(Y) - a * np.log(K) - (1 - a) * np.log(N))[20:]              # drop the capital-initialization years
t = np.arange(len(lz))
for lab, z in (("linear trend", lz - np.polyval(np.polyfit(t, lz, 1), t)), ("HP filter", hpfilter(lz, 1600)[0])):
    fit = ARIMA(z, order=(1, 0, 0), trend="n").fit()
    rho, sig2 = fit.params[0], fit.params[1]
    ols_rho = np.sum(z[1:] * z[:-1]) / np.sum(z[:-1]**2)
    print(f"{lab:12s}: MLE rho = {rho:.3f} (se {fit.bse[0]:.3f}), sigma = {100*np.sqrt(sig2):.3f}%  | OLS rho = {ols_rho:.3f} | half-life {np.log(0.5)/np.log(rho):.1f} quarters")
""",
notes="""With a linear trend, U.S. TFP is highly persistent (\\(\\rho\\approx0.94\\), a half-life of almost three years), with innovations of
about 0.7% — close to the standard RBC calibration of 0.95 and 0.007. With the HP filter, persistence drops to 0.80 and the
half-life to about three quarters, because the filter removes low-frequency movements. Since the model's implied volatility
and persistence of output depend heavily on \\(\\rho\\), the choice of detrending method is effectively a model assumption;
estimating the model and its shocks jointly (full-information Bayesian estimation) makes that assumption explicit.
"""),

dict(title="A Kalman filter by hand: trend inflation as a latent state",
data="`macrodata` — U.S. quarterly CPI inflation (annualized), 1959–2009.",
task="""
State-space model: observed inflation \\(\\pi_t=\\tau_t+e_t\\), with latent persistent component \\(\\tau_t=\\mu+\\rho(\\tau_{t-1}-\\mu)+u_t\\), \\(e_t\\sim N(0,\\sigma_e^2)\\), \\(u_t\\sim N(0,\\sigma_u^2)\\).

1. Write the Kalman filter recursions and the prediction-error log-likelihood by hand.
2. Maximize the likelihood over \\((\\mu,\\rho,\\log\\sigma_u,\\log\\sigma_e)\\) with `scipy.optimize`.
3. Check against `statsmodels` (`SARIMAX` with `measurement_error=True`).
4. Report the signal-to-noise ratio and the filtered estimate of the persistent component in 1980Q1 and 2008Q4. This is exactly how DSGE likelihoods are evaluated, with a larger state vector.
""",
starter="""
import numpy as np, statsmodels.api as sm
from scipy import optimize
y = sm.datasets.macrodata.load_pandas().data["infl"].to_numpy()
def loglik(th):
    mu, rho, lsu, lse = th; su2, se2 = np.exp(2 * lsu), np.exp(2 * lse)
    # TODO: initialize x = mu, P = su2/(1-rho^2); loop: predict, update, accumulate -0.5*(log 2 pi F + v^2/F)
    ...
""",
solution="""
import numpy as np, statsmodels.api as sm
from scipy import optimize
from statsmodels.tsa.statespace.sarimax import SARIMAX
m = sm.datasets.macrodata.load_pandas().data; y = m["infl"].to_numpy()
def kfilter(th, keep=False):
    mu, rho, lsu, lse = th; su2, se2 = np.exp(2 * lsu), np.exp(2 * lse)
    x, P = mu, su2 / max(1 - rho**2, 1e-6); ll = 0.0; filt = []
    for yt in y:
        F = P + se2; v = yt - x
        ll += -0.5 * (np.log(2 * np.pi * F) + v**2 / F)
        K = P / F; x, P = x + K * v, P * (1 - K)             # update
        filt.append(x)
        x, P = mu + rho * (x - mu), rho**2 * P + su2         # predict
    return (ll, np.array(filt)) if keep else ll
res = optimize.minimize(lambda th: -kfilter(th), x0=[3.0, 0.9, 0.0, 0.5], method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10})
mu, rho, lsu, lse = res.x
print(f"hand-coded Kalman MLE: mu = {mu:.3f}, rho = {rho:.4f}, sigma_u = {np.exp(lsu):.3f}, sigma_e = {np.exp(lse):.3f}, logL = {-res.fun:.3f}")
sm_fit = SARIMAX(y, order=(1, 0, 0), trend="c", measurement_error=True).fit(disp=0)
print(f"statsmodels SARIMAX logL = {sm_fit.llf:.3f};  AR coefficient {sm_fit.params[1]:.4f}")
print(f"signal-to-noise ratio sigma_u^2 / sigma_e^2 = {np.exp(2*lsu)/np.exp(2*lse):.3f}")
_, filt = kfilter(res.x, keep=True)
for yr, q in ((1980, 1), (2008, 4)):
    i = int(m.index[(m.year == yr) & (m.quarter == q)][0])
    print(f"  {yr}Q{q}: observed inflation {y[i]:6.2f}%, filtered persistent component {filt[i]:6.2f}%")
""",
notes="""The hand-coded Kalman likelihood reproduces the `statsmodels` log-likelihood to three decimals, which validates the code.
Quarterly CPI inflation is mostly noise: the signal-to-noise ratio is about 0.3. The persistent component is highly persistent
(\\(\\rho\\approx0.93\\)). In 1980Q1, the filter attributes most of the 14.6% observed inflation to the persistent component
(12.7%); in 2008Q4, it treats the −8.8% reading mostly as noise, with the persistent component near −2%. A DSGE likelihood is
the same calculation with a larger state vector generated by the model's solution.
"""),
]

EXERCISES["12-heterogeneous-agents.md"] = [
dict(title="An Aiyagari household solved by the endogenous grid method",
data="Calibration (annual): \\(\\beta=0.96\\), CRRA \\(\\sigma=2\\), \\(r=0.03\\), borrowing limit 0; log labor income AR(1) with \\(\\rho=0.9\\), \\(\\sigma=0.2\\) (discretized by Rouwenhorst with 7 states).",
task="""
1. Discretize income with the Rouwenhorst method.
2. Solve the household's consumption policy by the **endogenous grid method** (Carroll, 2006): invert the Euler equation on a grid of next-period assets, then handle the borrowing constraint.
3. Compute the stationary wealth distribution (Young's histogram method). Report the share of households at the borrowing constraint, the wealth Gini, and the average marginal propensity to consume (MPC) out of a small windfall.
""",
starter="""
import numpy as np
b, s, r, rho, sig, n = 0.96, 2.0, 0.03, 0.9, 0.2, 7
# TODO: Rouwenhorst; asset grid a' in [0, 50] (dense near 0); EGM loop; Young's method for the distribution
""",
solution="""
import numpy as np
b, s, r, rho, sig, n = 0.96, 2.0, 0.03, 0.9, 0.2, 7
p = (1 + rho) / 2; P = np.array([[p, 1 - p], [1 - p, p]])
for m_ in range(3, n + 1):
    Pn = np.zeros((m_, m_))
    Pn[:-1, :-1] += p * P; Pn[:-1, 1:] += (1 - p) * P; Pn[1:, :-1] += (1 - p) * P; Pn[1:, 1:] += p * P
    Pn[1:-1] /= 2; P = Pn
psi = sig / np.sqrt(1 - rho**2) * np.sqrt(n - 1)
z = np.exp(np.linspace(-psi, psi, n)); pi = np.ones(n) / n
for _ in range(2000): pi = pi @ P
z = z / (pi @ z)                                          # normalize mean income to 1
a = 50 * np.linspace(0, 1, 500)**2                         # asset grid, dense near the constraint
up = lambda c: c**(-s); upinv = lambda u: u**(-1 / s)
C = (1 + r) * a[:, None] + z[None, :] - a[0]               # initial guess: consume income plus a little
for it in range(5000):
    Emu = (up(C) @ P.T) * b * (1 + r)                      # E[u'(c')] for each (a', z)
    c_endo = upinv(Emu); a_endo = (c_endo + a[:, None] - z[None, :]) / (1 + r)
    Cn = np.empty_like(C)
    for j in range(n):
        Cn[:, j] = np.interp(a, a_endo[:, j], c_endo[:, j])
        bind = a < a_endo[0, j]; Cn[bind, j] = (1 + r) * a[bind] + z[j] - a[0]     # constraint binds
    if np.max(np.abs(Cn - C)) < 1e-9: break
    C = Cn
ap = (1 + r) * a[:, None] + z[None, :] - C
D = np.ones((len(a), n)) / (len(a) * n)
idx = np.clip(np.searchsorted(a, ap) - 1, 0, len(a) - 2); w = np.clip((a[idx + 1] - ap) / (a[idx + 1] - a[idx]), 0, 1)
for _ in range(5000):
    Dn = np.zeros_like(D)
    for j in range(n):
        np.add.at(Dn[:, j], idx[:, j], w[:, j] * D[:, j]); np.add.at(Dn[:, j], idx[:, j] + 1, (1 - w[:, j]) * D[:, j])
    Dn = Dn @ P
    if np.max(np.abs(Dn - D)) < 1e-12: break
    D = Dn
wealth = D.sum(1); order = np.argsort(a); cw = np.cumsum(wealth[order]); cwa = np.cumsum(wealth[order] * a[order])
gini = 1 - np.sum((cwa[1:] + cwa[:-1]) * np.diff(cw)) / cwa[-1] - cwa[0] * cw[0] / cwa[-1]
mpc = (C[1:] - C[:-1]) / ((1 + r) * (a[1:] - a[:-1]))[:, None]
avg_mpc = np.sum(D[:-1] * mpc) / D[:-1].sum()
print(f"EGM converged in {it} iterations; mean assets {np.sum(D.sum(1) * a):.2f} (mean income 1)")
print(f"share at the borrowing constraint: {D[0].sum():.3f};  wealth Gini: {gini:.3f};  average annual MPC: {avg_mpc:.3f}")
np.save("/tmp/aiyagari_D.npy", D); np.save("/tmp/aiyagari_mpc.npy", mpc); np.save("/tmp/aiyagari_a.npy", a)
""",
notes="""EGM converges in a few hundred iterations without any root-finding: inverting the Euler equation on a grid of next-period
assets gives consumption directly. The stationary distribution has a wealth Gini of about 0.49 — far below the U.S. value of
about 0.8 — and only 1.6% of households at the borrowing constraint. The average annual MPC is 0.08, close to the permanent-income
benchmark. One-asset models with these parameters generate too few constrained households and too little wealth inequality;
HANK models fix this with two assets (Kaplan, Moll and Violante, 2018) or with a lower discount factor.
"""),

dict(title="Hand-to-mouth households and wealth inequality in U.S. survey data",
data="`401ksubs` — 9,275 U.S. households, 1991 SIPP: family income `inc` and net total financial assets `nettfa` (thousands of dollars).",
task="""
Kaplan, Violante and Weidner (2014) classify households as hand-to-mouth (HtM) if their liquid wealth is below about half a month of income.

1. Compute the share of households with non-positive financial assets, and the share with financial assets below half a month of income.
2. Compute the Gini coefficient of financial wealth (allowing negative values) and the share of financial wealth held by the top 10%.
3. Break the HtM share down by income quintile. Why does the presence of HtM households at **all** income levels matter for fiscal policy?
""",
starter="""
import numpy as np, pandas as pd, wooldridge as wd
d = wd.data("401ksubs")
htm = d["nettfa"] < d["inc"] / 24
# TODO: shares; Gini via mean absolute difference; by pd.qcut(d.inc, 5)
""",
solution="""
import numpy as np, pandas as pd, wooldridge as wd
d = wd.data("401ksubs")
d["htm"] = d["nettfa"] < d["inc"] / 24
x = np.sort(d["nettfa"].to_numpy()); n = len(x); i = np.arange(1, n + 1)
gini = np.sum((2 * i - n - 1) * x) / (n * x.sum())
print(f"share with nettfa <= 0: {np.mean(d['nettfa'] <= 0):.3f};  hand-to-mouth (nettfa < half a month of income): {d['htm'].mean():.3f}")
print(f"Gini of financial wealth: {gini:.3f};  top 10% share: {x[int(0.9*n):].sum()/x.sum():.3f};  median ${np.median(x):.1f}k vs mean ${x.mean():.1f}k")
d["quintile"] = pd.qcut(d["inc"], 5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
print(d.groupby("quintile", observed=True).agg(income=("inc", "mean"), htm_share=("htm", "mean"), median_nettfa=("nettfa", "median")).round(3).to_string())
""",
notes="""In these 1991 data, 35% of households have non-positive financial assets, and 46% hold less than half a month of income in
financial assets — very close to Kaplan, Violante and Weidner's (2014) estimates of about one third using liquid wealth. Financial
wealth is extremely concentrated (Gini 0.98, top 10% share 72%; the mean is ten times the median). The hand-to-mouth share
falls with income but remains above 20% even in the top income quintile. Those are the "wealthy hand-to-mouth" — households
with illiquid wealth (housing, pensions) but little cash — which is why transfers raise spending even among middle-income
households.
"""),
]

EXERCISES["13-hank-distribution.md"] = [
dict(title="Who spends a transfer? MPCs across the wealth distribution",
data="The stationary Aiyagari distribution from Lecture 12's exercise (re-solved here so the code is self-contained).",
task="""
1. Re-solve the Aiyagari household and its stationary distribution (same calibration).
2. Compute the average MPC by wealth quintile.
3. Compare the aggregate first-year spending response to a uniform \\$1 transfer to everyone with a transfer **targeted** at the poorest 20% of households by wealth. By how much does targeting raise the spending response per dollar?
""",
starter="""
# Reuse the EGM and distribution code of Lecture 12, Exercise 12.1, then:
# TODO: sort grid points by assets, assign quintiles by the cumulative distribution, average the MPC within each
""",
solution="""
import numpy as np
b, s, r, rho, sig, n = 0.96, 2.0, 0.03, 0.9, 0.2, 7
p = (1 + rho) / 2; P = np.array([[p, 1 - p], [1 - p, p]])
for m_ in range(3, n + 1):
    Pn = np.zeros((m_, m_)); Pn[:-1, :-1] += p * P; Pn[:-1, 1:] += (1 - p) * P; Pn[1:, :-1] += (1 - p) * P; Pn[1:, 1:] += p * P
    Pn[1:-1] /= 2; P = Pn
psi = sig / np.sqrt(1 - rho**2) * np.sqrt(n - 1); z = np.exp(np.linspace(-psi, psi, n)); pi = np.ones(n) / n
for _ in range(2000): pi = pi @ P
z = z / (pi @ z); a = 50 * np.linspace(0, 1, 500)**2
C = (1 + r) * a[:, None] + z[None, :]
for it in range(5000):
    c_endo = (b * (1 + r) * (C**(-s) @ P.T))**(-1 / s); a_endo = (c_endo + a[:, None] - z[None, :]) / (1 + r)
    Cn = np.empty_like(C)
    for j in range(n):
        Cn[:, j] = np.interp(a, a_endo[:, j], c_endo[:, j]); bind = a < a_endo[0, j]; Cn[bind, j] = (1 + r) * a[bind] + z[j]
    if np.max(np.abs(Cn - C)) < 1e-9: break
    C = Cn
ap = (1 + r) * a[:, None] + z[None, :] - C
idx = np.clip(np.searchsorted(a, ap) - 1, 0, len(a) - 2); w = np.clip((a[idx + 1] - ap) / (a[idx + 1] - a[idx]), 0, 1)
D = np.ones((len(a), n)) / (len(a) * n)
for _ in range(5000):
    Dn = np.zeros_like(D)
    for j in range(n):
        np.add.at(Dn[:, j], idx[:, j], w[:, j] * D[:, j]); np.add.at(Dn[:, j], idx[:, j] + 1, (1 - w[:, j]) * D[:, j])
    Dn = Dn @ P
    if np.max(np.abs(Dn - D)) < 1e-12: break
    D = Dn
mpc = np.vstack([(C[1:] - C[:-1]) / ((1 + r) * (a[1:] - a[:-1]))[:, None], ((C[-1] - C[-2]) / ((1 + r) * (a[-1] - a[-2])))[None, :]])
wa = D.sum(1); cum = np.cumsum(wa)
q = np.minimum((cum / cum[-1] * 5).astype(int), 4)
q_mass = np.array([wa[q == k].sum() for k in range(5)])
for k in range(5):
    sel = q == k
    print(f"wealth quintile {k+1}: assets {a[sel].min():6.2f}-{a[sel].max():6.2f}, mass {q_mass[k]:.3f}, average MPC {np.sum(D[sel] * mpc[sel]) / D[sel].sum():.3f}")
uni = np.sum(D * mpc)
poor = q == 0
tgt = np.sum(D[poor] * mpc[poor]) / D[poor].sum()
print(f"spending per $1 transferred: uniform {uni:.3f}, targeted to the poorest-wealth group {tgt:.3f} ({tgt/uni:.1f}x)")
""",
notes="""MPCs fall steeply with wealth: about 0.20 in the poorest wealth quintile and about 0.04 at the top. A transfer targeted at the
poorest fifth generates about 2.6 times as much first-year spending per dollar as a uniform transfer. In a HANK model this
"distributional channel" matters for fiscal and monetary transmission. This one-asset model still understates MPCs — the
empirical average is about 0.2–0.3 per quarter for windfalls (Parker et al., 2013) — because it has too few constrained
households, as the previous lecture's exercise showed.
"""),

dict(title="A TANK multiplier with a data-based hand-to-mouth share",
data="`401ksubs` — the share of hand-to-mouth households from Lecture 12's exercise.",
task="""
In a two-agent New Keynesian (TANK) model with a share \\(\\lambda\\) of hand-to-mouth households who consume their income,
and Ricardian households who smooth consumption, a deficit-financed transfer \\(T\\) given equally to all households has a
first-round output effect of approximately \\(\\Delta Y=\\frac{\\lambda}{1-\\lambda\\,\\mu}T\\) under a fixed real rate, where
\\(\\mu\\) is the share of income that HtM households receive relative to their population share (set \\(\\mu=1\\) for simplicity) —
a simple "Keynesian cross" among HtM households.

1. Compute the HtM share \\(\\lambda\\) from the data (half-month-of-income rule) and from the stricter "non-positive wealth" rule.
2. Compute the transfer multiplier \\(\\Delta Y/T\\) for each, and for \\(\\lambda\\in\\{0,0.1,0.2,0.3,0.4\\}\\).
3. Compare with the Campbell–Mankiw \\(\\lambda\\) estimated in Lecture 1's exercise. Why do HANK models with *wealthy* hand-to-mouth households give larger multipliers than this simple TANK calculation?
""",
starter="""
import numpy as np, wooldridge as wd
d = wd.data("401ksubs")
lam1 = (d["nettfa"] < d["inc"] / 24).mean(); lam2 = (d["nettfa"] <= 0).mean()
mult = lambda lam: lam / (1 - lam)
# TODO: print multipliers
""",
solution="""
import numpy as np, wooldridge as wd
d = wd.data("401ksubs")
cases = {"half-month-of-income rule": (d["nettfa"] < d["inc"] / 24).mean(), "non-positive wealth rule": (d["nettfa"] <= 0).mean()}
mult = lambda lam: lam / (1 - lam)
for lab, lam in cases.items():
    print(f"{lab:26s}: lambda = {lam:.3f} -> transfer multiplier {mult(lam):.3f}")
for lam in (0.0, 0.1, 0.2, 0.3, 0.4):
    print(f"   lambda = {lam:.1f}: multiplier {mult(lam):.3f}")
""",
notes="""With the half-month-of-income rule (\\(\\lambda\\approx0.46\\)), the stylized transfer multiplier is about 0.87; with the stricter rule
(\\(\\lambda\\approx0.35\\)), about 0.54. The multiplier is convex in \\(\\lambda\\) because hand-to-mouth households spend the income
created by each other's spending. Campbell and Mankiw's \\(\\lambda\\) (Lecture 1) points to a similar range. HANK models produce
larger and more persistent effects than this calculation because many households are only **partly** constrained, MPCs are
high for several quarters, and the "intertemporal Keynesian cross" works through future income too (Auclert, Rognlie and
Straub, 2024).
"""),
]

EXERCISES["14-empirical-policy.md"] = [
dict(title="Local projections of a monetary shock measured as a Taylor-rule residual",
data="`macrodata` — U.S. quarterly data, 1960–2007.",
task="""
1. Estimate a Taylor rule with smoothing on 1960–2007 (as in Lecture 7) including two lags of inflation and the output gap, and treat its residual as a monetary policy shock (a Romer–Romer-style shortcut).
2. Estimate local projections of the unemployment rate and inflation on the shock for \\(h=0,\\dots,12\\) quarters, controlling for four lags of unemployment, inflation and the T-bill rate. Use HAC standard errors.
3. Report the responses to a 100 bp shock. Do you find a price puzzle? What identification assumption is doing the work?
""",
starter="""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
# TODO: rule regression -> shock = residual; for h: y_{t+h} - y_{t-1} on shock_t + controls
""",
solution="""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"year": m["year"], "i": m["tbilrate"], "pi": m["infl"].rolling(4).mean(), "u": m["unemp"],
                  "x": 100 * hpfilter(np.log(m["realgdp"]), 1600)[0]})
for v in ("i", "pi", "x", "u"):
    for k in (1, 2, 3, 4): d[f"{v}{k}"] = d[v].shift(k)
s = d[(d.year >= 1960) & (d.year <= 2007)].dropna().copy()
rule = smf.ols("i ~ pi + x + pi1 + x1 + i1 + i2", data=s).fit()
s["shock"] = rule.resid
print(f"policy-rule R2 {rule.rsquared:.3f}; shock sd {s['shock'].std():.3f} pp")
ctrl = " + ".join(f"{v}{k}" for v in ("u", "pi", "i") for k in (1, 2, 3, 4))
rows = []
for h in range(0, 13):
    for v in ("u", "pi"):
        s[f"lead"] = s[v].shift(-h) - s[f"{v}1"]
        r = smf.ols(f"lead ~ shock + {ctrl}", data=s.dropna(subset=["lead"])).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        rows.append((h, v, r.params["shock"], r.bse["shock"]))
out = pd.DataFrame(rows, columns=["h", "var", "coef", "se"]).pivot(index="h", columns="var", values=["coef", "se"])
print(out.loc[[0, 2, 4, 6, 8, 10, 12]].round(3).to_string())
""",
notes="""The Taylor-rule residual is a crude shock: the rule explains 92% of the T-bill rate, and the residual still contains the Fed's
reaction to information not in the rule. The responses show it. Unemployment **falls** for the first year after a "tightening"
and rises only after about two years, peaking near 0.12 points; inflation is flat at first and then falls by about 0.6 points after
three years. The early unemployment response has the wrong sign, a symptom of residual endogeneity — the same problem as the
price puzzle. Narrative (Romer and Romer, 2004) or high-frequency shocks (Gertler and Karadi, 2015) give cleaner identification.
The identification assumption here is that the residual is uncorrelated with all other shocks affecting the outcomes.
"""),

dict(title="Okun's law: estimation and parameter stability",
data="`okun` — U.S. annual real GDP growth `pcrgdp` and the change in the unemployment rate `cunem`, 1959–2005.",
task="""
1. Estimate Okun's law \\(\\Delta u_t=a+b\\,g_t+e_t\\). Compute the implied "normal" growth rate \\(-a/b\\) at which unemployment is stable, with a delta-method standard error.
2. Test whether \\(b=-0.5\\) (the textbook rule of thumb).
3. Test for a break in 1983 with a Chow test, and estimate the two subsamples. Has the relationship changed?
""",
starter="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
o = wd.data("okun")
r = smf.ols("cunem ~ pcrgdp", data=o).fit()
# TODO: -a/b with delta method; t_test("pcrgdp = -0.5"); Chow with post-1983 interactions
""",
solution="""
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
o = wd.data("okun")
r = smf.ols("cunem ~ pcrgdp", data=o).fit(cov_type="HC1")
a, b = r.params["Intercept"], r.params["pcrgdp"]; V = r.cov_params().to_numpy()
g = np.array([-1 / b, a / b**2]); se = np.sqrt(g @ V @ g)
print(f"Okun: d(unemployment) = {a:.3f} {b:+.3f} * growth;  normal growth -a/b = {-a/b:.2f}% (se {se:.2f})")
t = r.t_test("pcrgdp = -0.5")
print(f"test b = -0.5: t = {float(np.squeeze(t.tvalue)):.2f}, p = {float(t.pvalue):.3f}")
o["post"] = (o["year"] >= 1983).astype(int)
c = smf.ols("cunem ~ pcrgdp * post", data=o).fit(cov_type="HC1")
w = c.wald_test("post = 0, pcrgdp:post = 0", scalar=True)
print(f"Chow test for a break in 1983: chi2(2) = {w.statistic:.2f}, p = {w.pvalue:.3f}")
for lab, sub in (("1959-1982", o[o.post == 0]), ("1983-2005", o[o.post == 1])):
    rs = smf.ols("cunem ~ pcrgdp", data=sub).fit()
    print(f"  {lab}: b = {rs.params['pcrgdp']:.3f}, normal growth {-rs.params['Intercept']/rs.params['pcrgdp']:.2f}%")
""",
notes="""Each extra point of GDP growth lowers unemployment by about 0.38 points, and unemployment is stable when growth is about 3.3%.
The textbook coefficient of −0.5 is rejected. The Chow test detects a break in 1983: the slope is similar across subperiods, but
the growth rate needed to keep unemployment stable fell from about 3.9% to 2.9%, reflecting slower labor-force and productivity
growth. The break is in the **intercept**, which is why rule-of-thumb versions of Okun's law need periodic updating.
"""),
]
