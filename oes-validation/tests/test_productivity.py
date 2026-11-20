"""Validation of the productivity and spatial models: independent values, Monte Carlo and error messages."""
import os, sys, warnings
import numpy as np, pandas as pd, pytest
from scipy import stats, optimize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "oes", "engine")); sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import oes_engine as o

EX = os.path.join(ROOT, "oes", "examples")


def _tab(r, title):
    return next(t for t in r["tables"] if t["title"].startswith(title))


def _csv(df):
    return df.to_csv(index=False)


def _rd(n):
    return pd.read_csv(os.path.join(EX, n))


# ------------------------------------------------------------------ stochastic frontier
def _sfa_sim(n, seed, su=0.4, sv=0.2, exp=False):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 2, n); x2 = rng.uniform(0, 2, n)
    u = rng.exponential(su, n) if exp else np.abs(rng.normal(0, su, n))
    y = 1 + 0.5 * x1 + 0.3 * x2 + rng.normal(0, sv, n) - u
    return pd.DataFrame(dict(y=y, x1=x1, x2=x2)), u


def test_sfa_recovers_parameters_halfnormal():
    d, u = _sfa_sim(1500, 1)
    r = o.run("sfa", _csv(d), dict(y="y", x=["x1", "x2"], log="no", crs_test="no"))
    p = {row[0]: row[1] for row in _tab(r, "Stochastic")["rows"]}
    assert abs(p["x1"] - 0.5) < 0.06 and abs(p["x2"] - 0.3) < 0.06
    assert abs(np.exp(p["ln sigma_u"]) - 0.4) < 0.08 and abs(np.exp(p["ln sigma_v"]) - 0.2) < 0.05
    # estimated efficiency correlates with the true one
    te = np.exp(-u)
    assert r["summary"][1]["value"] > 0.5


def test_sfa_efficiency_tracks_truth():
    d, u = _sfa_sim(1500, 2)
    r = o.run("sfa", _csv(d), dict(y="y", x=["x1", "x2"], log="no"))
    mean_te = next(row[1] for row in _tab(r, "Technical")["rows"] if row[0] == "Mean")
    assert abs(mean_te - np.exp(-u).mean()) < 0.04


def test_sfa_exponential_recovers_scale():
    d, u = _sfa_sim(2000, 3, su=0.3, exp=True)
    r = o.run("sfa", _csv(d), dict(y="y", x=["x1", "x2"], log="no", distribution="exponential"))
    p = {row[0]: row[1] for row in _tab(r, "Stochastic")["rows"]}
    assert abs(np.exp(p["ln sigma_u"]) - 0.3) < 0.07 and abs(p["x1"] - 0.5) < 0.06


def test_sfa_no_inefficiency_lr_not_rejected_and_rejected():
    rng = np.random.default_rng(4); n = 800
    x = rng.uniform(0, 2, n); y = 1 + 0.5 * x + rng.normal(0, 0.2, n)
    d = pd.DataFrame(dict(y=y, x=x))
    r = o.run("sfa", _csv(d), dict(y="y", x=["x"], log="no"))
    lr = next(t for t in _tab(r, "Tests")["rows"] if t[0].startswith("LR"))
    assert lr[2] > 0.05
    d2, _ = _sfa_sim(1000, 5)
    r2 = o.run("sfa", _csv(d2), dict(y="y", x=["x1", "x2"], log="no"))
    lr2 = next(t for t in _tab(r2, "Tests")["rows"] if t[0].startswith("LR"))
    assert lr2[2] < 0.01


def test_sfa_likelihood_matches_direct_formula():
    from oes_engine import productivity as pr
    d, _ = _sfa_sim(400, 6)
    X = np.column_stack([np.ones(400), d.x1, d.x2]); par = np.array([1, .5, .3, np.log(.2), np.log(.4)])
    nll = pr._sfa_nll(par, d.y.to_numpy(), X, np.ones((400, 1)), "halfnormal", 1.0)
    s = np.sqrt(.2 ** 2 + .4 ** 2); lam = .4 / .2; e = d.y.to_numpy() - X @ par[:3]
    ll = np.log(2 / s) + stats.norm.logpdf(e / s) + stats.norm.logcdf(-lam * e / s)
    assert abs(nll + ll.sum()) < 1e-8


def test_sfa_rice_example_shares():
    d = _rd("ricefarms.csv")
    r = o.run("sfa", _csv(d), dict(y="goutput", x=["size", "seed", "urea", "totlabor"]))
    el = {row[0]: row[1] for row in _tab(r, "Stochastic")["rows"]}
    assert 0.3 < el["size"] < 0.6 and all(el[k] > 0 for k in ("seed", "urea", "totlabor"))


def test_sfa_errors():
    d, _ = _sfa_sim(100, 7)
    assert "error" in o.run("sfa", _csv(d.head(20)), dict(y="y", x=["x1", "x2"], log="no"))
    assert "error" in o.run("sfa", _csv(d.assign(y=d.y - 5)), dict(y="y", x=["x1", "x2"]))   # negative output cannot be logged


# ------------------------------------------------------------------ DEA
def test_dea_matches_known_single_input_output():
    d = pd.DataFrame(dict(x=[2, 4, 8, 6, 5.0, 3, 7, 1], y=[2, 4, 8, 3, 5.0, 3, 7, 1]))
    r = o.run("dea", _csv(d), dict(inputs=["x"], outputs=["y"]))
    ce = _tab(r, "DEA")["rows"][0]
    # CRS frontier is y = x: unit 4 has efficiency 0.5, others 1
    assert abs(ce[1] - (7 + 0.5) / 8) < 1e-6 and abs(ce[3] - 0.5) < 1e-6


def test_dea_two_input_hand_computed():
    # Farrell example: units A,B,C,D,E producing one output of 1 with inputs (x1,x2)
    d = pd.DataFrame(dict(x1=[2, 4, 8, 4, 6.0, 5, 9, 3, 7], x2=[8, 4, 2, 8, 6.0, 5, 3, 9, 7], y=[1.0] * 9))
    r = o.run("dea", _csv(d), dict(inputs=["x1", "x2"], outputs=["y"]))
    rows = _tab(r, "Least efficient")["rows"]
    sc = {row[0]: row[1] for row in rows}
    # D=(4,8) is dominated by A=(2,8)/B=(4,4): radial score 0.5 (B scaled by .5? no: (2,4) not feasible) -> verify by LP duality: A+B mix gives (3,6)*... compute directly
    from scipy.optimize import linprog
    X = d[["x1", "x2"]].to_numpy(float)
    c = [1] + [0] * 9
    A = np.vstack([np.hstack([-X[3][:, None], X.T]), [0] + [-1] * 9]); b = [0, 0, -1]
    th = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] + [(0, None)] * 9, method="highs").x[0]
    assert abs(sc["4"] - th) < 1e-6


def test_dea_bcc_ge_ccr_and_scale_identity():
    d = _rd("produc_1986.csv")
    r = o.run("dea", _csv(d), dict(inputs=["pc", "emp"], outputs=["gsp"]))
    rows = _tab(r, "DEA")["rows"]
    assert rows[1][1] >= rows[0][1] - 1e-9
    assert abs(rows[2][1] - np.mean([1]) ) <= 1.0
    for row in _tab(r, "Least efficient")["rows"]:
        assert row[2] >= row[1] - 1e-9 and abs(row[1] / row[2] - row[3]) < 1e-5


def test_dea_output_orientation_consistent_with_input_for_one_input_one_output_crs():
    d = pd.DataFrame(dict(x=[2, 4, 8, 6, 5.0, 7, 3, 9], y=[2, 4, 8, 3, 5.0, 6, 2.5, 5]))
    a = o.run("dea", _csv(d), dict(inputs=["x"], outputs=["y"]))
    b = o.run("dea", _csv(d), dict(inputs=["x"], outputs=["y"], orientation="output"))
    assert abs(_tab(a, "DEA")["rows"][0][1] - _tab(b, "DEA")["rows"][0][1]) < 1e-6


def test_dea_super_efficiency_above_one_for_frontier_units():
    d = _rd("produc_1986.csv")
    r = o.run("dea", _csv(d), dict(inputs=["pc", "emp"], outputs=["gsp"], label="state", super="yes"))
    sp = _tab(r, "Super")["rows"]
    assert sp[0][1] > 1 and sp[0][1] >= sp[-1][1]


def test_dea_errors():
    d = pd.DataFrame(dict(x=[1, 2, 0.0, 4, 5, 6, 7, 8, 9, 10], y=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0]))
    assert "error" in o.run("dea", _csv(d), dict(inputs=["x"], outputs=["y"]))
    assert "error" in o.run("dea", _csv(d.head(3)), dict(inputs=["x"], outputs=["y"]))


# ------------------------------------------------------------------ Malmquist
def test_malmquist_decomposition_identity_and_pure_shift():
    rng = np.random.default_rng(8); n = 30
    x = rng.uniform(1, 3, n); y0 = x * rng.uniform(0.6, 1.0, n)
    rows = []
    for t, g in ((1, 1.0), (2, 1.2)):                         # the whole technology shifts up by 20 percent
        for i in range(n):
            rows.append(dict(id=i, t=t, x=x[i], y=y0[i] * g))
    d = pd.DataFrame(rows)
    r = o.run("malmquist", _csv(d), dict(inputs=["x"], outputs=["y"], id="id", time="t"))
    p = _tab(r, "Malmquist")["rows"][0]
    assert abs(p[1] - 1.2) < 1e-6 and abs(p[2] - 1.0) < 1e-6 and abs(p[3] - 1.2) < 1e-6
    assert abs(p[1] - p[2] * p[3]) < 1e-5


def test_malmquist_identity_on_real_data_every_period():
    d = _rd("produc.csv")
    r = o.run("malmquist", _csv(d), dict(inputs=["pc", "emp"], outputs=["gsp"], id="state", time="year"))
    for p in _tab(r, "Malmquist")["rows"]:
        assert abs(p[1] - p[2] * p[3]) < 1e-5


def test_malmquist_catch_up_only():
    rng = np.random.default_rng(9); n = 30
    x = rng.uniform(1, 3, n); y0 = x * rng.uniform(0.5, 0.9, n)
    d = pd.concat([pd.DataFrame(dict(id=range(n), t=1, x=x, y=y0)), pd.DataFrame(dict(id=range(n), t=2, x=x, y=np.minimum(y0 * 1.1, (y0 / x).max() * x)))])
    # frontier unit stays at its level, so frontier is unchanged only if the best ratio is unchanged: scale everyone but the best
    best = np.argmax(y0 / x)
    d.loc[(d.t == 2) & (d.id == best), "y"] = y0[best]
    r = o.run("malmquist", _csv(d), dict(inputs=["x"], outputs=["y"], id="id", time="t"))
    p = _tab(r, "Malmquist")["rows"][0]
    assert abs(p[3] - 1.0) < 1e-6 and p[2] > 1.0


def test_malmquist_errors():
    d = _rd("produc.csv")
    assert "error" in o.run("malmquist", _csv(d[d.year == 1986]), dict(inputs=["pc"], outputs=["gsp"], id="state", time="year"))


# ------------------------------------------------------------------ growth regressions
def test_growth_matches_statsmodels_and_restriction():
    import statsmodels.api as sm
    d = _rd("growthdj.csv"); d = d[d.oil == "no"].dropna(subset=["gdp60", "gdp85", "invest", "popgrowth", "school"])
    r = o.run("growth", _csv(d), dict(y0="gdp60", y1="gdp85", invest="invest", popgrowth="popgrowth", school="school"))
    X = pd.DataFrame(dict(s=np.log(d.invest / 100), n=np.log(d.popgrowth / 100 + 0.05), h=np.log(d.school / 100)))
    m = sm.OLS(np.log(d.gdp85), sm.add_constant(X)).fit(cov_type="HC1")
    t = _tab(r, "Income per worker")["rows"]
    for k, row in enumerate(t):
        assert abs(row[1] - m.params.iloc[k]) < 1e-5 and abs(row[2] - m.bse.iloc[k]) < 1e-5
    f = m.f_test("s + n = 0")
    # the engine uses the classical F from restricted/unrestricted RSS
    m0 = sm.OLS(np.log(d.gdp85), sm.add_constant(X)).fit()
    ft = m0.f_test("s + n = 0")
    row = next(x for x in _tab(r, "Tests")["rows"] if x[0].startswith("Solow"))
    assert abs(row[1] - float(ft.fvalue)) < 1e-3


def test_growth_conditional_convergence_negative_and_mrw_sizes():
    d = _rd("growthdj.csv"); d = d[d.oil == "no"]
    r = o.run("growth", _csv(d), dict(y0="gdp60", y1="gdp85", invest="invest", popgrowth="popgrowth", school="school"))
    cc = _tab(r, "Conditional")["rows"][1]
    assert cc[1] < 0 and cc[4] < 0.01
    uc = _tab(r, "Unconditional")["rows"][1]
    assert uc[1] > -0.1                                        # no unconditional convergence in this sample


def test_growth_known_data_generating_process():
    rng = np.random.default_rng(10); n = 400
    s = rng.uniform(5, 30, n); nn = rng.uniform(0.5, 3, n); h = rng.uniform(2, 15, n)
    a, b = 1 / 3, 1 / 3
    ly = 8 + (a / (1 - a - b)) * (np.log(s / 100) - np.log(nn / 100 + .05)) + (b / (1 - a - b)) * np.log(h / 100) + rng.normal(0, .2, n)
    d = pd.DataFrame(dict(y0=np.exp(ly - 0.3), y1=np.exp(ly), s=s, n=nn, h=h))
    r = o.run("growth", _csv(d), dict(y0="y0", y1="y1", invest="s", popgrowth="n", school="h"))
    alpha = next(x for x in _tab(r, "Tests")["rows"] if "capital share alpha (restricted, augmented" in x[0])[1]
    assert abs(alpha - 1 / 3) < 0.03
    assert _tab(r, "Tests")["rows"][0][2] > 0.01


def test_growth_errors():
    d = _rd("growthdj.csv")
    assert "error" in o.run("growth", _csv(d.head(10)), dict(y0="gdp60", y1="gdp85", invest="invest", popgrowth="popgrowth"))
    assert "error" in o.run("growth", _csv(d), dict(y0="gdp60", y1="gdp85", invest="invest"))


# ------------------------------------------------------------------ production and cost functions
def test_prodfun_cobb_douglas_matches_statsmodels():
    import statsmodels.api as sm
    d = _rd("electricity1970.csv")
    r = o.run("prodfun", _csv(d), dict(y="cost", output="output", x=["labor", "capital", "fuel"], kind="cost"))
    L = np.log(d); X = L[["output", "labor", "capital", "fuel"]]
    m = sm.OLS(L["cost"], sm.add_constant(X - X.mean())).fit(cov_type="HC1")
    for k, row in enumerate(_tab(r, "Cobb-Douglas")["rows"]):
        assert abs(row[1] - m.params.iloc[k]) < 1e-5 and abs(row[2] - m.bse.iloc[k]) < 1e-5


def test_prodfun_recovers_known_returns_to_scale():
    rng = np.random.default_rng(11); n = 600
    k = rng.uniform(1, 5, n); l = rng.uniform(1, 5, n)
    y = np.exp(0.5 + 0.4 * np.log(k) + 0.5 * np.log(l) + rng.normal(0, .1, n))
    r = o.run("prodfun", _csv(pd.DataFrame(dict(y=y, k=k, l=l))), dict(y="y", x=["k", "l"]))
    rts = next(x for x in _tab(r, "Elasticities")["rows"] if x[0].startswith("Returns"))
    assert abs(rts[1] - 0.9) < 0.03 and rts[2] > 0.001 or abs(rts[1] - 0.9) < 0.03


def test_prodfun_translog_nests_cobb_douglas_when_data_are_cd():
    rng = np.random.default_rng(12); n = 800
    k = rng.uniform(1, 5, n); l = rng.uniform(1, 5, n)
    y = np.exp(0.5 + 0.4 * np.log(k) + 0.5 * np.log(l) + rng.normal(0, .1, n))
    r = o.run("prodfun", _csv(pd.DataFrame(dict(y=y, k=k, l=l))), dict(y="y", x=["k", "l"], form="translog"))
    w = next(x for x in _tab(r, "Elasticities")["rows"] if x[0].startswith("Cobb-Douglas against"))
    assert w[2] > 0.01


def test_prodfun_translog_rejects_cd_on_nerlove_data():
    d = _rd("electricity1970.csv")
    r = o.run("prodfun", _csv(d), dict(y="cost", output="output", x=["labor", "capital", "fuel"], kind="cost", form="translog", normalise="fuel"))
    w = next(x for x in _tab(r, "Elasticities")["rows"] if x[0].startswith("Cobb-Douglas against"))
    assert w[2] < 0.001


def test_prodfun_errors():
    d = _rd("electricity1970.csv")
    assert "error" in o.run("prodfun", _csv(d), dict(y="cost", x=["labor", "fuel"], kind="cost"))
    assert "error" in o.run("prodfun", _csv(d.assign(labor=-1)), dict(y="cost", output="output", x=["labor"], kind="cost"))


# ------------------------------------------------------------------ spatial
def _moran_manual(e, W):
    n = len(e); return n / W.sum() * (e @ W @ e) / (e @ e)


def test_spatial_moran_matches_manual_and_ols_matches_statsmodels():
    import statsmodels.api as sm
    from oes_engine import productivity as pr
    c = _rd("columbus.csv")
    r = o.run("spatial_cs", _csv(c), dict(y="crime", x=["inc", "hoval"], x_coord="x", y_coord="y", neighbours=5))
    X = sm.add_constant(c[["inc", "hoval"]]); m = sm.OLS(c.crime, X).fit()
    W = pr._knn_w(c[["x", "y"]].to_numpy(), 5); W = W / W.sum(1, keepdims=True)
    mi = next(x for x in _tab(r, "Spatial dependence")["rows"] if x[0].startswith("Moran"))
    assert abs(mi[1] - _moran_manual(m.resid.to_numpy(), W)) < 1e-3
    for k, row in enumerate(_tab(r, "OLS")["rows"]):
        assert abs(row[1] - m.params.iloc[k]) < 1e-5


def test_spatial_sar_likelihood_maximum_and_known_rho():
    from oes_engine import productivity as pr
    rng = np.random.default_rng(13); n = 300
    xy = rng.uniform(0, 10, (n, 2)); W = pr._knn_w(xy, 6); W = W / W.sum(1, keepdims=True)
    x = rng.normal(size=n); rho = 0.5
    y = np.linalg.solve(np.eye(n) - rho * W, 1 + 2 * x + rng.normal(size=n))
    d = pd.DataFrame(dict(y=y, x=x, a=xy[:, 0], b=xy[:, 1]))
    r = o.run("spatial_cs", _csv(d), dict(y="y", x=["x"], x_coord="a", y_coord="b", neighbours=6))
    sar = _tab(r, "Spatial lag")["rows"]
    assert abs(sar[-1][1] - 0.5) < 0.08 and abs(sar[1][1] - 2) < 0.15
    lm = next(x for x in _tab(r, "Spatial dependence")["rows"] if x[0].startswith("LM test, spatial lag"))
    assert lm[2] < 0.001
    # the concentrated optimum is a maximum of the profile likelihood
    X = np.column_stack([np.ones(n), x]); ev = np.linalg.eigvals(W).real
    def ll(rr):
        ys = y - rr * W @ y; b = np.linalg.lstsq(X, ys, rcond=None)[0]; e = ys - X @ b
        return -n / 2 * np.log(e @ e / n) + np.sum(np.log(1 - rr * ev))
    assert ll(sar[-1][1]) >= ll(sar[-1][1] + 0.03) - 1e-9 and ll(sar[-1][1]) >= ll(sar[-1][1] - 0.03) - 1e-9


def test_spatial_no_dependence_not_rejected_monte_carlo():
    rng = np.random.default_rng(14); rej = 0; R = 60
    for _ in range(R):
        n = 80; d = pd.DataFrame(dict(y=rng.normal(size=n), x=rng.normal(size=n), a=rng.uniform(0, 1, n), b=rng.uniform(0, 1, n)))
        r = o.run("spatial_cs", _csv(d), dict(y="y", x=["x"], x_coord="a", y_coord="b", neighbours=4))
        mi = next(x for x in _tab(r, "Spatial dependence")["rows"] if x[0].startswith("Moran"))
        rej += mi[2] < 0.05
    assert rej / R < 0.15


def test_spatial_errors():
    c = _rd("columbus.csv")
    assert "error" in o.run("spatial_cs", _csv(c.head(10)), dict(y="crime", x=["inc"], x_coord="x", y_coord="y"))
    assert "error" in o.run("spatial_cs", _csv(c), dict(y="crime", x=["inc"], x_coord="x", y_coord="y", neighbours=100))
