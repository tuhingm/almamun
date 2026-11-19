"""Validation of causal-inference models added after the first release: weak-instrument-robust IV inference."""
import os, re, sys, warnings
import numpy as np, pandas as pd, pytest
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "oes", "engine")); sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import oes_engine as o
from oes_engine import causal_extra

CX = os.path.join(ROOT, "causal-inference-guide", "examples")
def card(): return pd.read_csv(os.path.join(CX, "card_full.csv"))
CTRL = ["exper", "expersq", "black", "smsa", "south"]
def wspec(zs, **kw): return dict(outcome="lwage", treat="educ", instruments=zs, controls=CTRL, **kw)
def est(r, name): return next(x for x in r["tables"][0]["rows"] if x[0].startswith(name))
def interval(txt): return [tuple(float(v) for v in m) for m in re.findall(r"\[(-?[\d.]+|-inf), (-?[\d.]+|inf)\]", txt)]


def test_weakiv_registered():
    m = o.BY_ID["weakiv"]; assert m["group"] == "Causal inference" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_weakiv_2sls_matches_iv_model():
    d = card(); r = causal_extra.weakiv(d, wspec(["nearc4", "nearc2"], sims=200))
    iv = o.run("iv", d.to_csv(index=False), dict(outcome="lwage", treat="educ", instruments=["nearc4", "nearc2"], controls=CTRL))
    assert float(est(r, "2SLS")[1]) == pytest.approx(float(iv["tables"][0]["rows"][0][1]), abs=2e-4)


def test_weakiv_ar_set_endpoints_have_pvalue_alpha():
    import statsmodels.api as sm
    d = card(); r = causal_extra.weakiv(d, wspec(["nearc4"], sims=200)); lo, hi = interval(est(r, "Anderson-Rubin")[3])[0]
    X = sm.add_constant(d[CTRL + ["nearc4"]])
    for b0 in (lo, hi):
        F = sm.OLS(d.lwage - b0 * d.educ, X).fit().f_test(np.eye(X.shape[1])[-1:]).fvalue
        assert stats.chi2.sf(float(F), 1) == pytest.approx(0.05, abs=0.006)


def test_weakiv_single_instrument_all_sets_agree_and_liml_equals_2sls():
    r = causal_extra.weakiv(card(), wspec(["nearc4"], sims=200))
    assert est(r, "Anderson-Rubin")[3] == est(r, "Lagrange")[3] == est(r, "Conditional")[3]
    assert float(est(r, "LIML")[1]) == pytest.approx(float(est(r, "2SLS")[1]), abs=1e-6)


def test_weakiv_strong_instruments_sets_close_to_wald():
    r = causal_extra.weakiv(card(), wspec(["fatheduc", "motheduc"], sims=400)); w = interval(est(r, "2SLS")[3])[0]; c = interval(est(r, "Conditional")[3])[0]
    assert abs(c[0] - w[0]) < 0.01 and abs(c[1] - w[1]) < 0.01


def test_weakiv_weak_instruments_give_wide_or_unbounded_sets():
    r = causal_extra.weakiv(card(), wspec(["nearc4", "nearc2"], sims=400))
    assert "unbounded" in est(r, "Lagrange")[3] or interval(est(r, "Lagrange")[3])[0][1] - interval(est(r, "Lagrange")[3])[0][0] > 1
    w = interval(est(r, "2SLS")[3])[0]; a = interval(est(r, "Anderson-Rubin")[3])[0]; assert a[1] - a[0] > 2 * (w[1] - w[0])


def weak_sim(seed, pi=0.08, n=250, k=2, beta=1.0, rho=0.8, bad=0.0):
    rng = np.random.default_rng(seed); Z = rng.normal(size=(n, k)); v = rng.normal(size=n); u = rho * v + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    x = Z @ np.full(k, pi) + v; y = beta * x + u + bad * Z[:, 0]
    return pd.DataFrame(np.column_stack([y, x, Z]), columns=["y", "x"] + [f"z{i}" for i in range(k)])


def test_weakiv_coverage_robust_sets_vs_wald_under_weak_instruments():
    cov = dict(w=0, ar=0, clr=0); R = 150
    for s in range(R):
        r = causal_extra.weakiv(weak_sim(s), dict(outcome="y", treat="x", instruments=["z0", "z1"], sims=400))
        for key, name in (("w", "2SLS"), ("ar", "Anderson-Rubin"), ("clr", "Conditional")):
            txt = est(r, name)[3]; cov[key] += ("unbounded" in txt) or any(a <= 1.0 <= b for a, b in interval(txt))
    assert cov["ar"] / R >= 0.90 and cov["clr"] / R >= 0.90 and cov["w"] / R < cov["clr"] / R


def test_weakiv_overidentification_test_flags_invalid_instrument():
    ok = causal_extra.weakiv(weak_sim(1, pi=0.5, n=800), dict(outcome="y", treat="x", instruments=["z0", "z1"], sims=200))
    bad = causal_extra.weakiv(weak_sim(1, pi=0.5, n=800, bad=0.5), dict(outcome="y", treat="x", instruments=["z0", "z1"], sims=200))
    assert float(ok["tables"][2]["rows"][1][2]) > 0.01 and float(bad["tables"][2]["rows"][1][2]) < 0.001


def test_weakiv_errors():
    d = card()
    assert "Column(s) not found" in o.run("weakiv", d.to_csv(index=False), wspec(["nope"]))["error"]
    assert "collinear" in o.run("weakiv", d.assign(z2=d.nearc4).to_csv(index=False), wspec(["nearc4", "z2"]))["error"]
    assert "confidence level" in o.run("weakiv", d.to_csv(index=False), wspec(["nearc4"], level=0.2))["error"]


# ---- randomised experiments ------------------------------------------------------------------------------------
def nsw(): return pd.read_csv(os.path.join(CX, "nsw_experiment.csv"))
COV = ["age", "educ", "black", "hisp", "married", "nodegree", "re74", "re75"]
ES = dict(outcome="re78", treat="train", covariates=COV, draws=1000)
def erow(r, name): return next(x for x in r["tables"][0]["rows"] if x[0].startswith(name))


def test_experiment_registered():
    m = o.BY_ID["experiment"]; assert m["group"] == "Causal inference" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_experiment_difference_in_means_matches_published_nsw():
    # Dehejia and Wahba (1999) experimental sample: 1.794 (0.671) in thousands of 1978 dollars
    r = causal_extra.experiment(nsw(), ES); row = erow(r, "Difference")
    assert float(row[1]) == pytest.approx(1.794, abs=1e-3) and float(row[2]) == pytest.approx(0.671, abs=1e-3)


def test_experiment_regression_estimators_match_statsmodels():
    import statsmodels.api as sm
    d = nsw(); r = causal_extra.experiment(d, ES); Xc = d[COV] - d[COV].mean()
    a = sm.OLS(d.re78, sm.add_constant(pd.concat([d.train, Xc], axis=1))).fit(cov_type="HC2")
    assert float(erow(r, "Regression adjustment")[1]) == pytest.approx(a.params["train"], abs=1e-3) and float(erow(r, "Regression adjustment")[2]) == pytest.approx(a.bse["train"], abs=1e-3)
    inter = pd.concat([d.train, Xc, Xc.mul(d.train, axis=0).add_suffix("_x")], axis=1)
    l = sm.OLS(d.re78, sm.add_constant(inter)).fit(cov_type="HC2")
    assert float(erow(r, "Lin")[1]) == pytest.approx(l.params["train"], abs=1e-3) and float(erow(r, "Lin")[2]) == pytest.approx(l.bse["train"], abs=1e-3)


def test_experiment_randomisation_inference_matches_scipy_permutation():
    d = nsw(); r = causal_extra.experiment(d, dict(outcome="re78", treat="train", draws=4000))
    res = stats.permutation_test((d.re78[d.train == 1].to_numpy(), d.re78[d.train == 0].to_numpy()), lambda a, b: a.mean() - b.mean(), n_resamples=4000, random_state=1)
    assert float(r["tables"][1]["rows"][0][2]) == pytest.approx(res.pvalue, abs=0.012)


def test_experiment_balance_joint_test_matches_statsmodels():
    import statsmodels.api as sm
    d = nsw(); r = causal_extra.experiment(d, ES); m = sm.OLS(d.train, sm.add_constant(d[COV])).fit()
    note = next(t for t in r["tables"] if t["title"] == "Covariate balance")["note"]; F = float(re.search(r"F = ([\d.]+)", note).group(1))
    assert F == pytest.approx(m.fvalue, abs=0.01)


def exp_sim(seed, n=200, tau=0.0, prog=2.0):
    rng = np.random.default_rng(seed); x = rng.normal(size=n); T = np.zeros(n); T[rng.permutation(n)[: n // 2]] = 1
    y = prog * x + tau * T + rng.normal(size=n); return pd.DataFrame(dict(y=y, t=T, x=x))


def test_experiment_neyman_coverage_ri_size_and_lin_precision():
    cov = rej = 0; se_d = se_l = 0.0; R = 300
    for s in range(R):
        r = causal_extra.experiment(exp_sim(s, tau=1.0), dict(outcome="y", treat="t", covariates=["x"], draws=200))
        e, se = float(erow(r, "Difference")[1]), float(erow(r, "Difference")[2]); cov += abs(e - 1.0) <= 1.96 * se; se_d += se; se_l += float(erow(r, "Lin")[2])
        rej += float(causal_extra.experiment(exp_sim(s, tau=0.0), dict(outcome="y", treat="t", draws=200))["tables"][1]["rows"][0][2]) < 0.05
    assert 0.92 <= cov / R <= 0.985 and 0.02 <= rej / R <= 0.09 and se_l < 0.6 * se_d


def test_experiment_errors():
    d = nsw()
    assert "0/1" in o.run("experiment", d.assign(train=d.train * 2).to_csv(index=False), dict(ES))["error"]
    assert "Column(s) not found" in o.run("experiment", d.to_csv(index=False), dict(ES, covariates=["nope"]))["error"]
    few = d.iloc[:60].assign(train=[1.0] * 3 + [0.0] * 57)
    assert "at least 5" in o.run("experiment", few.to_csv(index=False), dict(outcome="re78", treat="train"))["error"]
