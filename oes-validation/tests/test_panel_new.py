"""Validation of panel models added after the first release: Hausman-Taylor."""
import os, sys, warnings
import numpy as np, pandas as pd, pytest
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "oes", "engine")); sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import oes_engine as o
from oes_engine import panel_extra

PD = os.path.join(ROOT, "panel-data-guide", "examples")
SPEC = dict(y="lwage", x1=["occ", "south", "smsa", "ind"], x2=["exp", "exp2", "wks", "ms", "union"], z1=["fem", "blk"], z2=["ed"], unit="id", time="year")
def psid(): return pd.read_csv(os.path.join(PD, "psid_wages.csv"))
def coefs(r): return {x[0]: float(x[2]) for x in r["tables"][0]["rows"]}


def test_ht_registered_with_guide_and_example():
    m = o.BY_ID["hausman_taylor"]
    assert m["group"] == "Panel data" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_ht_matches_published_psid_estimates():
    # Baltagi and Khanti-Akom (1990) / Stata xthtaylor on the Cornwell-Rupert PSID data
    pub = dict(const=2.9127, occ=-0.0207, south=0.0074, smsa=-0.0418, ind=0.0136, exp=0.1131, exp2=-0.0004, wks=0.0008, ms=-0.0299, union=0.0328, fem=-0.1309, blk=-0.2858, ed=0.1379)
    c = coefs(panel_extra.hausman_taylor(psid(), SPEC))
    for k, v in pub.items():
        assert c[k] == pytest.approx(v, abs=1.1e-3), k


def test_ht_standard_errors_match_published():
    r = panel_extra.hausman_taylor(psid(), SPEC)
    se = {x[0]: float(x[3]) for x in r["tables"][0]["rows"]}
    assert se["ed"] == pytest.approx(0.0212, abs=2e-3) and se["exp"] == pytest.approx(0.0025, abs=3e-4)


def test_ht_time_varying_close_to_within():
    r = panel_extra.hausman_taylor(psid(), SPEC)
    w = {x[0]: float(x[1]) for x in r["tables"][1]["rows"]}; h = {x[0]: float(x[3]) for x in r["tables"][1]["rows"]}
    assert w["exp"] == pytest.approx(h["exp"], abs=2e-3)


def sim(seed=0, N=400, T=5):
    rng = np.random.default_rng(seed); u = rng.normal(size=N)
    z1 = rng.normal(size=N); ez = rng.normal(size=N); z2 = 0.7 * u + ez; m1 = 0.8 * ez                       # z2 correlated with the unit effect
    rows = []
    for i in range(N):
        for t in range(T):
            x1 = m1[i] + rng.normal(); x2 = 0.6 * u[i] + rng.normal()
            y = 1 + 0.5 * x1 + 1.0 * x2 + 0.8 * z1[i] + 1.5 * z2[i] + u[i] + rng.normal()
            rows.append((i, t, y, x1, x2, z1[i], z2[i]))
    return pd.DataFrame(rows, columns=["id", "t", "y", "x1", "x2", "z1", "z2"])


def test_ht_recovers_parameters_where_re_is_biased():
    d = sim()
    c = coefs(panel_extra.hausman_taylor(d, dict(y="y", x1=["x1"], x2=["x2"], z1=["z1"], z2=["z2"], unit="id", time="t")))
    assert c["x2"] == pytest.approx(1.0, abs=0.08) and c["z2"] == pytest.approx(1.5, abs=0.25) and c["z1"] == pytest.approx(0.8, abs=0.2)
    # naive pooled OLS is biased on z2
    import statsmodels.api as sm
    b = sm.OLS(d.y, sm.add_constant(d[["x1", "x2", "z1", "z2"]])).fit().params
    assert abs(b["z2"] - 1.5) > abs(c["z2"] - 1.5)


def test_ht_under_identified_error():
    d = sim()
    r = o.run("hausman_taylor", d.to_csv(index=False), dict(y="y", x1=[], x2=["x1", "x2"], z1=["z1"], z2=["z2"], unit="id", time="t"))
    assert "not identified" in r["error"]


def test_ht_misclassified_variable_error():
    d = sim()
    r = o.run("hausman_taylor", d.to_csv(index=False), dict(y="y", x1=["z1"], x2=["x2"], z1=[], z2=["z2"], unit="id", time="t"))
    assert "does not vary within units" in r["error"]
    r = o.run("hausman_taylor", d.to_csv(index=False), dict(y="y", x1=["x1"], x2=[], z1=["x2"], z2=[], unit="id", time="t"))
    assert "varies within units" in r["error"]


# ---- panel unit-root tests ---------------------------------------------------------------------------------
def infl(): return pd.read_csv(os.path.join(PD, "inflation_panel.csv"))
def trows(r): return {x[0]: x for x in r["tables"][0]["rows"]}
def panel_sim(kind, N=15, T=40, seed=0, rho=0.5):
    rng = np.random.default_rng(seed); rows = []
    for i in range(N):
        e = rng.standard_normal(T + 30)
        if kind == "rw": y = np.cumsum(e)[30:]
        else:
            y = np.zeros(T + 30)
            for t in range(1, T + 30): y[t] = rho * y[t - 1] + e[t]
            y = y[30:] + rng.normal()
        rows += [(i, t, v) for t, v in enumerate(y)]
    return pd.DataFrame(rows, columns=["u", "t", "y"])
PS = dict(y="y", unit="u", time="t", sims=100, maxlag=1)


def test_punitroot_registered():
    m = o.BY_ID["punitroot"]
    assert m["group"] == "Panel data" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_punitroot_individual_adf_matches_statsmodels():
    from statsmodels.tsa.stattools import adfuller, mackinnonp
    d = infl(); r = panel_extra.punitroot(d, dict(y="inflation", unit="country", time="year", maxlag=0, sims=50))
    rows = r["tables"][1]["rows"]; x = d[d.country == rows[3][0]].inflation.to_numpy(float)
    a = adfuller(x, maxlag=0, regression="c", autolag=None)
    assert float(rows[3][3]) == pytest.approx(a[0], abs=1e-3) and float(rows[3][4]) == pytest.approx(a[1], abs=1e-3)


def test_punitroot_fisher_matches_individual_pvalues():
    d = infl(); r = panel_extra.punitroot(d, dict(y="inflation", unit="country", time="year", maxlag=1, sims=50))
    pv = np.array([float(x[4]) for x in r["tables"][1]["rows"]]); N = len(pv)
    P = float(trows(r)["Fisher-type, inverse chi-squared (P)"][1])
    assert P == pytest.approx(-2 * np.log(pv).sum(), rel=0.02)
    assert float(trows(r)["Fisher-type, inverse chi-squared (P)"][3]) == pytest.approx(stats.chi2.sf(P, 2 * N), abs=1e-3)


def test_hadri_matches_direct_computation():
    d = panel_sim("ar"); Y = d.pivot(index="t", columns="u", values="y").to_numpy(float); T, N = Y.shape
    E = Y - Y.mean(axis=0); S = np.cumsum(E, axis=0); lm = np.mean((S ** 2).sum(axis=0) / T ** 2 / ((E ** 2).sum() / (N * T)))
    r = panel_extra.punitroot(d, PS); assert float(trows(r)["Hadri LM (stationarity)"][1]) == pytest.approx(lm, abs=1e-4)
    assert float(trows(r)["Hadri LM (stationarity)"][2]) == pytest.approx(np.sqrt(N) * (lm - 1 / 6) / np.sqrt(1 / 45), abs=1e-2)


def test_punitroot_rejects_unit_root_in_stationary_panel():
    r = panel_extra.punitroot(panel_sim("ar", rho=0.4), PS); t = trows(r)
    for k in ["Levin-Lin-Chu type (common unit root)", "Im-Pesaran-Shin t-bar (W)", "Fisher-type, inverse normal (Z)"]:
        assert float(t[k][3]) < 0.01, k
    # Hadri's statistic has no long-run variance correction here, so check its size on serially uncorrelated data
    assert float(trows(panel_extra.punitroot(panel_sim("ar", rho=0.0), PS))["Hadri LM (stationarity)"][3]) > 0.05


def test_punitroot_size_and_hadri_power_under_random_walks():
    rej = 0
    for s in range(12):
        t = trows(panel_extra.punitroot(panel_sim("rw", seed=s, N=10, T=40), dict(PS, sims=60)))
        rej += float(t["Im-Pesaran-Shin t-bar (W)"][3]) < 0.05
    assert rej <= 4                                                    # nominal 5% (12 panels): far fewer than a power result
    t = trows(panel_extra.punitroot(panel_sim("rw", seed=1), PS)); assert float(t["Hadri LM (stationarity)"][3]) < 0.01


def test_punitroot_errors():
    d = panel_sim("rw", N=6, T=8)
    assert "at least" in o.run("punitroot", d.to_csv(index=False), PS)["error"]
    assert "at least 4 units" in o.run("punitroot", panel_sim("rw", N=3).to_csv(index=False), PS)["error"]
    assert "Deterministic terms" in o.run("punitroot", panel_sim("rw").to_csv(index=False), dict(PS, trend="x"))["error"]


# ---- Dumitrescu-Hurlin panel Granger causality -----------------------------------------------------------------
def garrett(): return pd.read_csv(os.path.join(PD, "garrett.csv"))
GS = dict(y="gdp_growth", x="unem", unit="country", time="year", lags=1, diff_x="yes")
def dh_panel(N=10, T=30, seed=0, beta=0.0, share=1.0):
    rng = np.random.default_rng(seed); rows = []
    for i in range(N):
        e = rng.standard_normal((T + 20, 2)); y = np.zeros(T + 20); x = np.zeros(T + 20); b = beta if i < share * N else 0.0
        for t in range(1, T + 20): x[t] = 0.5 * x[t - 1] + e[t, 1]; y[t] = 0.4 * y[t - 1] + b * x[t - 1] + e[t, 0]
        rows += [(i, t, y[t + 20], x[t + 20]) for t in range(T)]
    return pd.DataFrame(rows, columns=["u", "t", "y", "x"])


def test_dh_registered():
    m = o.BY_ID["dh_causality"]; assert m["group"] == "Panel data" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_dh_unit_tests_match_statsmodels():
    from statsmodels.tsa.stattools import grangercausalitytests
    d = garrett(); r = panel_extra.dh_causality(d, dict(GS, lags=2)); row = r["tables"][1]["rows"][2]
    g = d[d.country == row[0]].sort_values("year"); x = np.diff(g.unem.to_numpy(float)); y = g.gdp_growth.to_numpy(float)[1:]
    F = grangercausalitytests(np.column_stack([y, x]), maxlag=2, verbose=False)[2][0]["ssr_ftest"]
    assert float(row[2]) == pytest.approx(F[0], abs=2e-3) and float(row[3]) == pytest.approx(F[1], abs=2e-3)


def test_dh_statistics_match_formulas():
    K = 2; r = panel_extra.dh_causality(garrett(), dict(GS, lags=K)); W = np.array([float(x[1]) for x in r["tables"][1]["rows"]]); N, T = len(W), 24
    Wb = W.mean(); zb = np.sqrt(N / (2 * K)) * (Wb - K)
    zt = np.sqrt(N / (2 * K) * (T - 3 * K - 5) / (T - 2 * K - 3)) * ((T - 3 * K - 3) / (T - 3 * K - 1) * Wb - K)
    row = r["tables"][0]["rows"][0]
    assert float(row[1]) == pytest.approx(Wb, abs=2e-3) and float(row[2]) == pytest.approx(zb, abs=2e-3) and float(row[4]) == pytest.approx(zt, abs=2e-3)


def test_dh_size_small_sample_statistic_better_than_large_t():
    rz, rt, R = 0, 0, 150
    for s in range(R):
        row = panel_extra.dh_causality(dh_panel(seed=s, T=20), dict(y="y", x="x", unit="u", time="t", lags=1))["tables"][0]["rows"][0]
        rz += float(row[3]) < 0.05; rt += float(row[5]) < 0.05
    assert rt / R <= 0.13 and rz > rt


def test_dh_power_and_direction():
    r = panel_extra.dh_causality(dh_panel(beta=0.6, share=0.6, T=40), dict(y="y", x="x", unit="u", time="t", lags=1)); rows = r["tables"][0]["rows"]
    assert float(rows[0][5]) < 1e-6 and float(rows[1][5]) > 0.001 and float(rows[1][5]) > 1000 * float(rows[0][5])


def test_dh_errors():
    d = dh_panel(); sp = dict(y="y", x="x", unit="u", time="t")
    assert "balanced" in o.run("dh_causality", d.iloc[1:].to_csv(index=False), sp)["error"]
    assert "periods" in o.run("dh_causality", dh_panel(T=10).to_csv(index=False), dict(sp, lags=3))["error"]
    assert "at least 3 units" in o.run("dh_causality", dh_panel(N=2).to_csv(index=False), sp)["error"]
    assert "Lags" in o.run("dh_causality", d.to_csv(index=False), dict(sp, lags=9))["error"]


# ---- correlated random effects (Mundlak) ------------------------------------------------------------------------
def wage(): return pd.read_csv(os.path.join(PD, "wagepan_full.csv"))
WS = dict(y="lwage", x=["union", "married", "expersq"], z=["educ", "black", "hisp"], unit="nr", time="year")
def crow(r, tab, var): return next(x for x in r["tables"][tab]["rows"] if x[0] == var)


def test_cre_registered():
    m = o.BY_ID["cre"]; assert m["group"] == "Panel data" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_cre_within_coefficients_match_published_fixed_effects():
    # Wooldridge (2010), Table 14.2 (wagepan, with year dummies): FE union .0800, married .0467, expersq -.0052
    r = panel_extra.cre(wage(), WS)
    assert float(crow(r, 1, "union")[1]) == pytest.approx(0.0800, abs=6e-4) and float(crow(r, 1, "married")[1]) == pytest.approx(0.0467, abs=6e-4) and float(crow(r, 1, "expersq")[1]) == pytest.approx(-0.0052, abs=6e-5)


def test_cre_equals_fe_on_time_varying_and_between_minus_within_on_means():
    d = wage(); r = panel_extra.cre(d, dict(y="lwage", x=["union", "married"], unit="nr", time="year", time_effects="no"))
    cr = {x[0]: float(x[1]) for x in r["tables"][0]["rows"]}; fe = {x[0]: float(x[1]) for x in r["tables"][1]["rows"]}
    g = d.groupby("nr"); xb = g[["union", "married"]].mean(); yb = g.lwage.mean()
    bb = np.linalg.lstsq(np.column_stack([np.ones(len(xb)), xb.to_numpy()]), yb.to_numpy(), rcond=None)[0][1:]
    dw = d[["union", "married"]] - g[["union", "married"]].transform("mean"); bw = np.linalg.lstsq(dw.to_numpy(), (d.lwage - g.lwage.transform("mean")).to_numpy(), rcond=None)[0]
    for i, v in enumerate(["union", "married"]):
        assert cr[v] == pytest.approx(bw[i], abs=2e-4) and cr["mean of " + v] == pytest.approx(bb[i] - bw[i], abs=2e-4)


def test_cre_random_effects_matches_re_model():
    d = wage(); r = panel_extra.cre(d, dict(WS, time_effects="no"))
    re = o.run("re", d.to_csv(index=False), dict(y="lwage", x=["union", "married", "expersq", "educ", "black", "hisp"], unit="nr", time="year"))
    for i, v in enumerate(["union", "married", "expersq"]):
        row = next(x for x in re["tables"][0]["rows"] if x[0] == v)
        assert float(crow(r, 1, v)[3]) == pytest.approx(row[1], abs=2e-4)


def test_cre_mundlak_test_matches_statsmodels_cluster_wald():
    import statsmodels.api as sm
    d = wage(); r = panel_extra.cre(d, dict(y="lwage", x=["union", "married"], unit="nr", time="year", time_effects="no"))
    X = d[["union", "married"]].copy(); X["mu"] = d.groupby("nr").union.transform("mean"); X["mm"] = d.groupby("nr").married.transform("mean")
    m = sm.OLS(d.lwage, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds=dict(groups=d.nr))
    b = m.params[["mu", "mm"]].to_numpy(); V = m.cov_params().loc[["mu", "mm"], ["mu", "mm"]].to_numpy(); W = float(b @ np.linalg.solve(V, b))
    assert float(r["tables"][2]["rows"][0][1]) == pytest.approx(W, rel=0.02)


def cre_sim(corr, seed=0, N=300, T=6):
    rng = np.random.default_rng(seed); u = rng.normal(size=N); rows = []
    for i in range(N):
        for t in range(T):
            x = corr * u[i] + rng.normal(); rows.append((i, t, 1 + 0.5 * x + u[i] + rng.normal(), x))
    return pd.DataFrame(rows, columns=["id", "t", "y", "x"])


def test_cre_detects_correlated_effects_and_corrects_re():
    sp = dict(y="y", x=["x"], unit="id", time="t", time_effects="no")
    r = panel_extra.cre(cre_sim(1.0), sp); assert float(r["tables"][2]["rows"][0][2]) < 0.001
    assert abs(float(crow(r, 1, "x")[1]) - 0.5) < 0.05 and abs(float(crow(r, 1, "x")[3]) - 0.5) > 0.1          # FE near truth, RE biased
    assert float(panel_extra.cre(cre_sim(0.0, seed=3), sp)["tables"][2]["rows"][0][2]) > 0.05


def test_cre_errors():
    d = wage()
    assert "does not vary within units" in o.run("cre", d.to_csv(index=False), dict(WS, x=["educ"], z=[]))["error"]
    assert "varies within units" in o.run("cre", d.to_csv(index=False), dict(WS, x=["union", "expersq"], z=["married"]))["error"]
    assert "Period dummies" in o.run("cre", d.to_csv(index=False), dict(WS, time_effects="x"))["error"]


# ---- interactive fixed effects (Bai 2009) ----------------------------------------------------------------------
def ife_sim(seed=0, N=60, T=50, beta=(1.0, -0.5), r=1):
    rng = np.random.default_rng(seed); F = rng.normal(size=(T, r)); L = rng.normal(size=(N, r)); a = rng.normal(size=N); g = rng.normal(size=T)
    X1 = 0.8 * (F @ L.T) + rng.normal(size=(T, N)) + 0.5 * a; X2 = 0.6 * (F @ rng.normal(size=(r, N))) + rng.normal(size=(T, N))
    Y = beta[0] * X1 + beta[1] * X2 + F @ L.T + a + g[:, None] + 0.5 * rng.normal(size=(T, N))
    rows = [(i, t, Y[t, i], X1[t, i], X2[t, i]) for i in range(N) for t in range(T)]
    return pd.DataFrame(rows, columns=["u", "t", "y", "x1", "x2"]), F
IS = dict(y="y", x=["x1", "x2"], unit="u", time="t")
def ifecoef(r): return np.array([float(x[1]) for x in r["tables"][0]["rows"]])


def test_ife_registered():
    m = o.BY_ID["ife"]; assert m["group"] == "Panel data" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_ife_without_factors_is_two_way_fixed_effects():
    d, _ = ife_sim(); r0 = panel_extra.ife(d, dict(IS, factors=0))
    Y = d.pivot(index="t", columns="u", values="y").to_numpy(); X = [d.pivot(index="t", columns="u", values=v).to_numpy() for v in ["x1", "x2"]]
    dm = lambda A: A - A.mean(0) - A.mean(1, keepdims=True) + A.mean()
    b = np.linalg.lstsq(np.column_stack([dm(a).ravel() for a in X]), dm(Y).ravel(), rcond=None)[0]
    assert ifecoef(r0) == pytest.approx(b, abs=1e-3)
    import statsmodels.api as sm                                             # same estimate from least squares with unit and period dummies
    m = sm.OLS(d.y, pd.concat([d[['x1', 'x2']], pd.get_dummies(d.u, prefix='u', drop_first=True).astype(float), pd.get_dummies(d.t, prefix='t', drop_first=True).astype(float)], axis=1).assign(c=1.0)).fit()
    assert ifecoef(r0) == pytest.approx(m.params[['x1', 'x2']].to_numpy(), abs=1e-3)


def test_ife_matches_concentrated_objective_minimiser():
    from scipy.optimize import minimize
    d, _ = ife_sim(seed=2, N=30, T=30); r = panel_extra.ife(d, dict(IS, factors=1))
    Y = d.pivot(index="t", columns="u", values="y").to_numpy(); X = [d.pivot(index="t", columns="u", values=v).to_numpy() for v in ["x1", "x2"]]
    dm = lambda A: A - A.mean(0) - A.mean(1, keepdims=True) + A.mean(); Y = dm(Y); X = [dm(a) for a in X]
    def ssr(b):
        E = Y - b[0] * X[0] - b[1] * X[1]; ev = np.linalg.eigvalsh(E.T @ E); return float(ev[:-1].sum())          # sum of all but the largest eigenvalue
    m = minimize(ssr, np.zeros(2), method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-10, maxiter=4000))
    assert ifecoef(r) == pytest.approx(m.x, abs=2e-3)


def test_ife_recovers_beta_where_two_way_fe_is_biased():
    d, F = ife_sim(); b1 = ifecoef(panel_extra.ife(d, dict(IS, factors=1))); b0 = ifecoef(panel_extra.ife(d, dict(IS, factors=0)))
    assert np.abs(b1 - np.array([1.0, -0.5])).max() < 0.05 and np.abs(b0 - np.array([1.0, -0.5])).max() > 0.1


def test_ife_chooses_true_number_of_factors_in_clean_design():
    d, _ = ife_sim(seed=4, r=2); r = panel_extra.ife(d, dict(IS, factors=2, rmax=4))
    assert next(x["value"] for x in r["summary"] if x["label"] == "IC chooses") == 2


def test_ife_reduces_residual_cross_section_dependence():
    d, _ = ife_sim(); r = panel_extra.ife(d, dict(IS, factors=1)); rows = r["tables"][2]["rows"]
    assert float(rows[1][3]) < 0.5 * float(rows[0][3])


def test_ife_errors():
    d, _ = ife_sim()
    assert "balanced" in o.run("ife", d.iloc[1:].to_csv(index=False), IS)["error"]
    assert "at least 10" in o.run("ife", ife_sim(N=8)[0].to_csv(index=False), IS)["error"]
    assert "Number of factors" in o.run("ife", d.to_csv(index=False), dict(IS, factors=9))["error"]
    assert "Additive effects" in o.run("ife", d.to_csv(index=False), dict(IS, effects="x"))["error"]
