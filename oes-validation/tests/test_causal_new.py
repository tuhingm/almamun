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
