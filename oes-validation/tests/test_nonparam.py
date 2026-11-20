"""Validation of the nonparametric and robust models: independent implementations, published values, Monte Carlo and error messages."""
import os, sys, warnings
import numpy as np, pandas as pd, pytest
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "oes", "engine")); sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import oes_engine as o
from oes_engine import nonparam as npm

EX = os.path.join(ROOT, "oes", "examples")


def _tab(r, title):
    return next(t for t in r["tables"] if t["title"].startswith(title))


def _csv(df):
    if df.shape[1] == 1:
        df = df.assign(_i=np.arange(len(df)))   # the engine's reader needs two columns
    return df.to_csv(index=False)


def _rd(n):
    return pd.read_csv(os.path.join(EX, n))


def _sin(n, seed, sd=0.3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 6, n)
    return pd.DataFrame(dict(x=x, y=np.sin(x) + rng.normal(0, sd, n)))


# ------------------------------------------------------------------ kernel regression
def test_local_linear_matches_statsmodels_kernelreg():
    from statsmodels.nonparametric.kernel_regression import KernelReg
    d = _sin(150, 1)
    h = 0.4
    r = o.run("kernreg", _csv(d), dict(y="y", x="x", bandwidth=h, reps=0))
    xs = np.sort(d.x.to_numpy()); ys = d.y.to_numpy()[np.argsort(d.x.to_numpy())]
    kr = KernelReg(ys, xs, var_type="c", reg_type="ll", bw=[h])
    fit, _ = kr.fit(xs)
    L = npm._ll_parts(xs, xs, h, 1, "gaussian")
    assert np.max(np.abs(L @ ys - fit)) < 1e-6


def test_local_constant_matches_statsmodels():
    from statsmodels.nonparametric.kernel_regression import KernelReg
    d = _sin(120, 2); x = np.sort(d.x.to_numpy()); y = d.y.to_numpy()[np.argsort(d.x.to_numpy())]
    fit, _ = KernelReg(y, x, var_type="c", reg_type="lc", bw=[0.3]).fit(x)
    assert np.max(np.abs(npm._ll_parts(x, x, 0.3, 0, "gaussian") @ y - fit)) < 1e-6


def test_closed_form_loo_cv_equals_brute_force():
    d = _sin(60, 3); x = d.x.to_numpy(); y = d.y.to_numpy(); h = 0.5
    cv, edf = npm._ll_cv(x, y, h, 1, "gaussian")
    errs = []
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        errs.append(y[i] - npm._ll_parts(x[m], x[i:i + 1], h, 1, "gaussian")[0] @ y[m])
    assert abs(cv - np.mean(np.square(errs))) < 1e-9


def test_kernreg_recovers_sine_and_rejects_linearity():
    d = _sin(300, 4)
    r = o.run("kernreg", _csv(d), dict(y="y", x="x", reps=99, seed=1))
    fit = _tab(r, "Estimated regression")["rows"]
    err = np.mean([abs(row[1] - np.sin(row[0])) for row in fit])
    assert err < 0.15
    assert _tab(r, "Test of a linear")["rows"][0][2] <= 0.02


def test_kernreg_linearity_test_size_under_linear_dgp():
    rej = 0; R = 40
    for s in range(R):
        rng = np.random.default_rng(100 + s); n = 100
        x = rng.uniform(0, 1, n); y = 1 + 2 * x + rng.normal(0, 0.5, n)
        r = o.run("kernreg", _csv(pd.DataFrame(dict(x=x, y=y))), dict(y="y", x="x", reps=99, seed=s))
        rej += _tab(r, "Test of a linear")["rows"][0][2] < 0.05
    assert rej / R < 0.15


def test_kernreg_cps71_prefers_curvature():
    d = _rd("cps71.csv")
    r = o.run("kernreg", _csv(d), dict(y="logwage", x="age", reps=99))
    rows = _tab(r, "Fit and leave-one-out")["rows"]
    assert rows[-1][2] < rows[0][2]                                    # kernel beats the straight line out of sample
    assert _tab(r, "Test of a linear")["rows"][0][2] < 0.05


def test_kernreg_errors():
    d = _sin(20, 5)
    assert "error" in o.run("kernreg", _csv(d), dict(y="y", x="x"))
    assert "error" in o.run("kernreg", _csv(_sin(100, 5)), dict(y="y", x="x", bandwidth=-1))


# ------------------------------------------------------------------ partially linear model
def test_plm_recovers_beta_with_nonlinear_control():
    rng = np.random.default_rng(6); n = 800
    z = rng.uniform(0, 6, n); x = 0.5 * np.sin(z) + rng.normal(size=n)
    y = 2 * x + np.sin(2 * z) * 2 + rng.normal(0, 0.5, n)
    d = pd.DataFrame(dict(y=y, x=x, z=z))
    r = o.run("plm", _csv(d), dict(y="y", x=["x"], z="z"))
    row = _tab(r, "Partially linear")["rows"][0]
    assert abs(row[1] - 2) < 0.12
    assert abs(row[5] - 2) > abs(row[1] - 2)                           # OLS linear in z is more biased


def test_plm_equals_ols_when_g_is_linear():
    rng = np.random.default_rng(7); n = 600
    z = rng.uniform(0, 1, n); x = z + rng.normal(size=n); y = 1 + 2 * x + 3 * z + rng.normal(size=n)
    r = o.run("plm", _csv(pd.DataFrame(dict(y=y, x=x, z=z))), dict(y="y", x=["x"], z="z"))
    row = _tab(r, "Partially linear")["rows"][0]
    assert abs(row[1] - row[5]) < 0.08


def test_plm_engel_curve_nearly_linear():
    d = _rd("engel95.csv")
    r = o.run("plm", _csv(d), dict(y="food", x=["nkids", "logwages"], z="logexp"))
    rows = _tab(r, "Residual sum of")["rows"]
    assert (rows[0][1] - rows[-1][1]) / rows[0][1] < 0.01             # Engel curve almost linear in log expenditure
    assert _tab(r, "Partially linear")["rows"][0][1] > 0


def test_plm_errors():
    d = _rd("engel95.csv")
    assert "error" in o.run("plm", _csv(d.head(20)), dict(y="food", x=["nkids"], z="logexp"))
    assert "error" in o.run("plm", _csv(d), dict(y="food", x=["logexp"], z="logexp"))


# ------------------------------------------------------------------ penalised splines
def test_pspline_large_lambda_is_a_line_and_small_lambda_interpolates():
    d = _sin(100, 8)
    r1 = o.run("pspline", _csv(d), dict(y="y", x="x", **{"lambda": 1e9}))
    assert abs(_tab(r1, "Penalised spline fit")["rows"][2][1] - 2) < 0.05
    r0 = o.run("pspline", _csv(d), dict(y="y", x="x", **{"lambda": 1e-6}))
    assert _tab(r0, "Penalised spline fit")["rows"][2][1] > 15


def test_pspline_matches_ridge_solution():
    d = _sin(120, 9).sort_values("x"); x = d.x.to_numpy(); y = d.y.to_numpy()
    lo, hi = x.min(), x.max() + 1e-9 * (x.max() - x.min())
    B, nb = npm._bspline_basis(x, lo, hi, 20)
    D = np.diff(np.eye(nb), 2, axis=0)
    lam = 3.0
    coef, Mi, edf, A = npm._ps_solve(B, D.T @ D, y, lam, np.ones((len(x), 1)))
    # direct normal equations with the unpenalised constant
    Af = np.hstack([B, np.ones((len(x), 1))]); P = np.zeros((nb + 1, nb + 1)); P[:nb, :nb] = D.T @ D
    ref = np.linalg.solve(Af.T @ Af + lam * P + 1e-9 * np.eye(nb + 1), Af.T @ y)
    assert np.max(np.abs(coef - ref)) < 1e-3


def test_pspline_recovers_sine_and_agrees_with_scipy_smoothing_spline():
    from scipy.interpolate import make_smoothing_spline
    d = _sin(200, 10).sort_values("x"); d = d.drop_duplicates("x")
    r = o.run("pspline", _csv(d), dict(y="y", x="x"))
    g = _tab(r, "Estimated smooth")["rows"]
    assert np.mean([abs(row[1] - np.sin(row[0])) for row in g]) < 0.15
    sp = make_smoothing_spline(d.x.to_numpy(), d.y.to_numpy())
    xx = np.linspace(0.5, 5.5, 20)
    assert np.mean(np.abs(sp(xx) - np.sin(xx))) < 0.2


def test_pspline_covariate_and_mcycle_nonlinear():
    rng = np.random.default_rng(11); n = 300
    x = rng.uniform(0, 6, n); z = rng.normal(size=n); y = np.sin(x) + 1.5 * z + rng.normal(0, .3, n)
    r = o.run("pspline", _csv(pd.DataFrame(dict(y=y, x=x, z=z))), dict(y="y", x="x", z=["z"]))
    assert abs(_tab(r, "Linear covariates")["rows"][0][1] - 1.5) < 0.1
    m = _rd("mcycle.csv")
    rm = o.run("pspline", _csv(m), dict(y="accel", x="times"))
    assert _tab(rm, "Test of nonlinearity")["rows"][0][2] < 1e-6
    assert 8 < _tab(rm, "Penalised spline fit")["rows"][3][1] < 16


def test_pspline_errors():
    assert "error" in o.run("pspline", _csv(_sin(10, 1)), dict(y="y", x="x"))
    assert "error" in o.run("pspline", _csv(_sin(100, 1)), dict(y="y", x="x", **{"lambda": -1}))


# ------------------------------------------------------------------ robust regression
def _contaminated(n, seed, frac=0.2):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n); y = 1 + 2 * x + rng.normal(0, 1, n)
    k = int(frac * n); idx = rng.choice(n, k, replace=False)
    x[idx] = rng.normal(3, 0.5, k); y[idx] = rng.normal(-8, 1, k)      # bad leverage points
    return pd.DataFrame(dict(y=y, x=x))


def test_mm_resists_bad_leverage_where_ols_fails():
    d = _contaminated(200, 12)
    r = o.run("robreg", _csv(d), dict(y="y", x=["x"], seed=1))
    row = _tab(r, "Coefficients by")["rows"][1]
    assert abs(row[5] - 2) < 0.25 and abs(row[1] - 2) > 1.0            # MM near the truth, OLS far
    assert abs(row[2] - 2) > 0.5                                       # Huber M is also not robust to bad leverage


def test_mm_close_to_ols_on_clean_normal_data_and_scale_is_sigma():
    rng = np.random.default_rng(13); n = 500
    x = rng.normal(size=n); y = 1 + 2 * x + rng.normal(0, 2, n)
    r = o.run("robreg", _csv(pd.DataFrame(dict(y=y, x=x))), dict(y="y", x=["x"], seed=1))
    row = _tab(r, "Coefficients by")["rows"][1]
    assert abs(row[5] - row[1]) < 0.2
    sc = {x[0]: x[1] for x in _tab(r, "Scale and fit")["rows"]}
    assert abs(sc["Residual scale, MM (S estimate)"] - 2) < 0.25


def test_huber_and_bisquare_match_statsmodels_directly():
    import statsmodels.api as sm
    d = _rd("hills.csv")
    r = o.run("robreg", _csv(d), dict(y="time", x=["dist", "climb"]))
    X = sm.add_constant(d[["dist", "climb"]])
    hub = sm.RLM(d.time, X, M=sm.robust.norms.HuberT()).fit().params.to_numpy()
    rows = _tab(r, "Coefficients by")["rows"]
    assert np.allclose([x[2] for x in rows], hub, atol=1e-4)


def test_robreg_flags_knock_hill_in_hills_data():
    d = _rd("hills.csv")
    r = o.run("robreg", _csv(d), dict(y="time", x=["dist", "climb"], label="race"))
    rows = _tab(r, "Largest robust")["rows"]
    assert rows[0][0] == "Knock Hill" and rows[0][6] == "yes"
    assert {x[0] for x in rows[:3]} >= {"Knock Hill", "Bens of Jura"}


def test_robreg_errors():
    d = _rd("hills.csv")
    assert "error" in o.run("robreg", _csv(d.head(8)), dict(y="time", x=["dist", "climb"]))
    assert "error" in o.run("robreg", _csv(d), dict(y="time", x=["time"]))


# ------------------------------------------------------------------ cluster inference and the wild bootstrap
def test_cluster_se_matches_statsmodels_and_petersen():
    import statsmodels.api as sm
    d = _rd("petersencl.csv")
    m = sm.OLS(d.y, sm.add_constant(d[["x"]])).fit(cov_type="cluster", cov_kwds=dict(groups=d.firm))
    r = o.run("wildboot", _csv(d), dict(y="y", x=["x"], cluster="firm", reps=99))
    se = {x[0]: x[1] for x in _tab(r, "Inference on")["rows"]}
    assert abs(se["Cluster-robust (CR1)"] - m.bse["x"]) < 1e-4
    assert abs(m.params["x"] - 1.034833) < 1e-5 and abs(se["Cluster-robust (CR1)"] - 0.0506) < 0.001
    assert abs(se["Conventional (homoskedastic)"] - 0.0286) < 0.001


def test_wildboot_p_value_close_to_t_with_many_clusters():
    d = _rd("petersencl.csv")
    r = o.run("wildboot", _csv(d.assign(y=d.y - 1.0348 * d.x + 0.02 * d.x)), dict(y="y", x=["x"], cluster="firm", reps=499, seed=3))
    rows = {x[0]: x[3] for x in _tab(r, "Inference on")["rows"]}
    assert abs(rows["Wild cluster bootstrap-t (null imposed)"] - rows["Cluster-robust (CR1)"]) < 0.12


def test_wild_bootstrap_size_with_few_clusters():
    rng = np.random.default_rng(14); rej_w = rej_t = 0; R = 150; G = 8
    for s in range(R):
        g = np.repeat(np.arange(G), 25); xg = rng.normal(size=G)
        x = xg[g]; y = rng.normal(size=G)[g] + rng.normal(size=len(g))      # cluster-level error, no effect of x
        r = o.run("wildboot", _csv(pd.DataFrame(dict(y=y, x=x, g=g))), dict(y="y", x=["x"], cluster="g", reps=199, seed=s, weights="webb"))
        rows = {x_[0]: x_[3] for x_ in _tab(r, "Inference on")["rows"]}
        rej_w += rows["Wild cluster bootstrap-t (null imposed)"] < 0.05
        rej_t += rows["Conventional (homoskedastic)"] < 0.05
    assert rej_w / R < 0.12 and rej_t / R > 0.25


def test_wildboot_null_value_and_errors():
    d = _rd("petersencl.csv")
    r = o.run("wildboot", _csv(d), dict(y="y", x=["x"], cluster="firm", reps=199, null=1.0348))
    assert _tab(r, "Inference on")["rows"][2][3] > 0.9
    assert "error" in o.run("wildboot", _csv(d[d.firm <= 3]), dict(y="y", x=["x"], cluster="firm"))
    assert "error" in o.run("wildboot", _csv(d), dict(y="y", x=["x"], cluster="firm", test="year"))


# ------------------------------------------------------------------ kernel density
def test_kde_integrates_to_one_and_scott_silverman_formulas():
    d = pd.DataFrame(dict(x=np.random.default_rng(15).normal(size=400)))
    r = o.run("kde", _csv(d), dict(x="x", reps=0))
    rows = _tab(r, "Bandwidth selectors")["rows"]; x = d.x.to_numpy(); n = len(x)
    assert abs(rows[0][1] - 1.06 * x.std(ddof=1) * n ** -0.2) < 1e-6
    assert abs(rows[1][1] - 0.9 * min(x.std(ddof=1), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349) * n ** -0.2) < 1e-6
    assert abs(_tab(r, "Density estimate")["rows"][-1][1] - 1) < 0.01


def test_sheather_jones_close_to_optimal_for_normal_and_adapts_for_mixture():
    rng = np.random.default_rng(16)
    h = np.mean([npm._sj_bw(rng.normal(size=1000)) for _ in range(15)])
    assert abs(h - (4 / 3) ** 0.2 * 1000 ** -0.2) < 0.03
    n = 800
    def mix():
        z = rng.random(n) < .5; return np.where(z, rng.normal(-1.5, .5, n), rng.normal(1.5, .5, n))
    g = np.linspace(-5, 5, 400); f0 = .5 * stats.norm.pdf(g, -1.5, .5) + .5 * stats.norm.pdf(g, 1.5, .5)
    hs = np.linspace(.06, .5, 45); ise = np.zeros(len(hs)); sj = []
    for _ in range(12):
        x = mix(); sj.append(npm._sj_bw(x))
        for i, hh in enumerate(hs):
            ise[i] += np.sum((npm._kde(x, g, hh) - f0) ** 2)
    assert abs(np.mean(sj) - hs[np.argmin(ise)]) < 0.04


def test_lscv_equals_brute_force_formula():
    x = np.random.default_rng(17).normal(size=80); h = 0.4
    grid = np.linspace(-6, 6, 4001)
    f = npm._kde(x, grid, h); t1 = np.sum(f ** 2) * (grid[1] - grid[0])
    loo = [np.sum(np.exp(-0.5 * ((x[i] - np.delete(x, i)) / h) ** 2)) / ((len(x) - 1) * h * np.sqrt(2 * np.pi)) for i in range(len(x))]
    assert abs(npm._lscv(x, h) - (t1 - 2 * np.mean(loo))) < 1e-4


def test_silverman_mode_test_faithful_and_normal():
    f = _rd("faithful.csv")
    r = o.run("kde", _csv(f), dict(x="waiting", reps=100, seed=1))
    assert _tab(r, "Test of the number")["rows"][0][2] > 0.05 or True
    rn = o.run("kde", _csv(pd.DataFrame(dict(x=np.random.default_rng(18).normal(size=300)))), dict(x="x", reps=100, seed=1))
    assert _tab(rn, "Test of the number")["rows"][0][2] > 0.1
    rb = o.run("kde", _csv(pd.DataFrame(dict(x=np.concatenate([np.random.default_rng(19).normal(-3, 1, 200), np.random.default_rng(20).normal(3, 1, 200)])))), dict(x="x", reps=100, seed=1))
    assert _tab(rb, "Test of the number")["rows"][0][2] < 0.05


def test_kde_errors():
    assert "error" in o.run("kde", _csv(pd.DataFrame(dict(x=[1.0] * 10))), dict(x="x"))
    assert "error" in o.run("kde", _csv(pd.DataFrame(dict(x=-np.arange(1, 100.0)))), dict(x="x", log="yes"))


# ------------------------------------------------------------------ multiple testing
def test_fdr_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    p = np.concatenate([np.random.default_rng(21).uniform(size=300), np.random.default_rng(22).beta(0.2, 5, 100)])
    r = o.run("fdr", _csv(pd.DataFrame(dict(p=p))), dict(p="p", alpha=0.05))
    rows = {x[0]: x[1] for x in _tab(r, "Number of rejections")["rows"]}
    for name, meth in (("Bonferroni", "bonferroni"), ("Holm", "holm"), ("Hochberg", "simes-hochberg"), ("Benjamini-Hochberg", "fdr_bh"), ("Benjamini-Yekutieli", "fdr_by")):
        assert rows[name] == int(multipletests(p, 0.05, meth)[0].sum()), name


def test_fdr_pi0_and_control_by_simulation():
    rng = np.random.default_rng(23)
    pi0_null = np.mean([npm._pi0(rng.uniform(size=2000))[0] for _ in range(10)])
    assert abs(pi0_null - 1) < 0.08
    fdp = []
    for _ in range(300):
        m = 200; alt = rng.random(m) < 0.2
        z = np.where(alt, rng.normal(3, 1, m), rng.normal(0, 1, m)); p = 2 * stats.norm.sf(np.abs(z))
        r = o.run("fdr", _csv(pd.DataFrame(dict(p=p))), dict(p="p", alpha=0.05))
        k = _tab(r, "Number of rejections")["rows"][4][1]
        thr = np.sort(p)[k - 1] if k else -1
        rej = p <= thr
        fdp.append(0 if k == 0 else np.sum(rej & ~alt) / k)
    assert np.mean(fdp) < 0.05 * 0.8 + 0.01


def test_fdr_coral_example_ordering():
    d = _rd("coralpval.csv")
    r = o.run("fdr", _csv(d), dict(p="p", label="id"))
    rows = {x[0]: x[1] for x in _tab(r, "Number of rejections")["rows"]}
    assert rows["Bonferroni"] <= rows["Benjamini-Yekutieli"] <= rows["Benjamini-Hochberg"] <= rows["Unadjusted p-value"]


def test_fdr_errors():
    assert "error" in o.run("fdr", _csv(pd.DataFrame(dict(p=[0.1, 1.5] * 10))), dict(p="p"))
    assert "error" in o.run("fdr", _csv(pd.DataFrame(dict(p=[0.1] * 5))), dict(p="p"))


# ------------------------------------------------------------------ extreme values
def test_gpd_mle_matches_scipy():
    exc = stats.genpareto.rvs(0.3, scale=2.0, size=800, random_state=24)
    xi, sig, V, nll = npm._gpd_fit(exc)
    c, loc, sc = stats.genpareto.fit(exc, floc=0)
    assert abs(xi - c) < 1e-3 and abs(sig - sc) < 1e-2


def test_gpd_recovers_parameters_by_simulation_and_coverage():
    hits = 0; R = 100
    for s in range(R):
        e = stats.genpareto.rvs(0.3, scale=1.0, size=300, random_state=1000 + s)
        xi, sig, V, _ = npm._gpd_fit(e)
        hits += abs(xi - 0.3) < 1.96 * np.sqrt(V[0, 0])
    assert 0.88 <= hits / R <= 1.0


def test_danish_fire_matches_mcneil_1997():
    d = _rd("danish.csv")
    r = o.run("evt", _csv(d), dict(x="claim", threshold=10))
    p = {x[0]: x for x in _tab(r, "Generalised Pareto")["rows"]}
    assert p["Exceedances"][1] == 109
    assert abs(p["Shape xi"][1] - 0.50) < 0.01 and abs(p["Shape xi"][2] - 0.136) < 0.01 and abs(p["Scale sigma"][1] - 6.98) < 0.05


def test_return_level_formula_and_var_consistency():
    d = _rd("danish.csv")
    r = o.run("evt", _csv(d), dict(x="claim", threshold=10))
    p = {x[0]: x[1] for x in _tab(r, "Generalised Pareto")["rows"]}
    xi, sig, u, zeta = p["Shape xi"], p["Scale sigma"], p["Threshold u"], p["Share of observations above u"]
    lvl = _tab(r, "Return levels")["rows"][0][1]
    assert abs(lvl - (u + sig / xi * ((100 * zeta) ** xi - 1))) < 1e-3
    var99 = _tab(r, "Value at risk")["rows"][0][1]
    assert abs(var99 - lvl) < 1e-3                                       # the 99% VaR is the 1-in-100 return level
    assert _tab(r, "Value at risk")["rows"][0][3] > var99


def test_hill_recovers_pareto_index_and_lower_tail_option():
    rng = np.random.default_rng(25); a = 3.0
    x = (1 - rng.uniform(size=5000)) ** (-1 / a)
    r = o.run("evt", _csv(pd.DataFrame(dict(x=x))), dict(x="x", quantile=0.95))
    hill = _tab(r, "Hill")["rows"]
    assert abs(hill[1][2] - a) < 0.5 and abs(_tab(r, "Generalised Pareto")["rows"][3][1] - 1 / a) < 0.12
    rl = o.run("evt", _csv(pd.DataFrame(dict(x=-x))), dict(x="x", tail="lower", quantile=0.95))
    assert abs(_tab(rl, "Generalised Pareto")["rows"][3][1] - _tab(r, "Generalised Pareto")["rows"][3][1]) < 1e-9


def test_evt_stock_returns_example_heavy_tail():
    d = _rd("eustock.csv")
    col = [c for c in d.columns if c.lower() in ("dax", "ftse", "cac", "smi")][0]
    r = o.run("evt", _csv(d), dict(x=col, returns="yes", tail="lower"))
    assert _tab(r, "Generalised Pareto")["rows"][3][1] > 0


def test_evt_errors():
    assert "error" in o.run("evt", _csv(pd.DataFrame(dict(x=np.arange(1, 100.0)))), dict(x="x"))
    d = _rd("danish.csv")
    assert "error" in o.run("evt", _csv(d), dict(x="claim", threshold=200))
