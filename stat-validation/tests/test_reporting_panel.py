"""Validation of the static-panel report against statsmodels (pooled, LSDV) and linearmodels (FE, RE, Hausman-free pieces)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "oes", "engine"))
from oes_engine import rep_panel  # noqa: E402

DF = pd.read_csv(os.path.join(ROOT, "panel-data-guide", "examples", "wagepan.csv"))
SPEC = dict(y="lwage", x=["union", "married", "expersq"], unit="nr", time="year", cov="clustered")
D = DF.dropna(subset=["lwage", "union", "married", "expersq"]).sort_values(["nr", "year"])


@pytest.fixture(scope="module")
def rep():
    return rep_panel.build(DF, SPEC, "fe")


@pytest.fixture(scope="module")
def reptw():
    return rep_panel.build(DF, dict(SPEC, time_effects="yes"), "fe")


def tab(rep, key):
    return next(t for s in rep.steps for t in s.tables if t.key == key)


def col(rep, sub):
    return next(c for c in tab(rep, "t_est").raw["columns"] if c["label"] and True and sub)  # placeholder, replaced below


def cols(rep):
    t = tab(rep, "t_est")
    subs = [h for h in t.header[1][1:]]
    return dict(zip(subs, t.raw["columns"]))


def test_structure(rep):
    assert [s.key for s in rep.steps] == ["data", "estimation", "tests", "resid", "effects"]
    assert sum(len(s.figures) for s in rep.steps) >= 9


def test_data(rep):
    st = tab(rep, "t_struct").raw
    assert st["units"] == D.nr.nunique() and st["periods"] == D.year.nunique() and st["n"] == len(D)
    r = tab(rep, "t_var").raw["rows"][0]
    a = D.lwage.to_numpy()
    assert r["overall"] == pytest.approx(a.std(ddof=1))
    assert r["between"] == pytest.approx(D.groupby("nr").lwage.mean().std(ddof=1))


def test_pooled_matches_statsmodels(rep):
    c = cols(rep)["Pooled OLS"]
    m = sm.OLS(D.lwage, sm.add_constant(D[SPEC["x"]])).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(D.nr)[0]}, use_t=True)
    for k in ["const"] + SPEC["x"]:
        assert c["params"][k] == pytest.approx(m.params[k], rel=1e-8)


def test_fe_matches_lsdv(rep):
    c = cols(rep)["Fixed effects"]
    X = pd.concat([D[SPEC["x"]], pd.get_dummies(D.nr, drop_first=True, dtype=float)], axis=1)
    m = sm.OLS(D.lwage, sm.add_constant(X)).fit()
    for k in SPEC["x"]:
        assert c["params"][k] == pytest.approx(m.params[k], rel=1e-8)


def test_against_linearmodels(rep):
    lm = pytest.importorskip("linearmodels")
    from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
    P = D.set_index(["nr", "year"])
    X = sm.add_constant(P[SPEC["x"]])
    c = cols(rep)
    fe = PanelOLS(P.lwage, P[SPEC["x"]], entity_effects=True, drop_absorbed=False).fit(cov_type="clustered", cluster_entity=True)
    re = RandomEffects(P.lwage, X).fit(cov_type="clustered", cluster_entity=True)
    for k in SPEC["x"]:
        assert c["Fixed effects"]["params"][k] == pytest.approx(fe.params[k], rel=1e-6)
        assert c["Fixed effects"]["se"][k] == pytest.approx(fe.std_errors[k], rel=5e-4)  # linearmodels 7.x differs from the engine in the 4th digit of clustered SEs (small-sample factor)
    for k in ["const"] + SPEC["x"]:
        assert c["Random effects"]["params"][k] == pytest.approx(re.params[k], rel=1e-5)
        assert c["Random effects"]["se"][k] == pytest.approx(re.std_errors[k], rel=5e-4)


def test_two_way(reptw):
    c = cols(reptw)["Two-way FE"]
    X = pd.concat([D[SPEC["x"]], pd.get_dummies(D.nr, drop_first=True, dtype=float), pd.get_dummies(D.year, drop_first=True, dtype=float)], axis=1)
    m = sm.OLS(D.lwage, sm.add_constant(X)).fit()
    for k in SPEC["x"]:
        assert c["params"][k] == pytest.approx(m.params[k], rel=1e-7)


def test_tests(rep):
    t = {x.name.split(",")[0].split(" (")[0]: x for s in rep.steps for x in s.tests}
    X = pd.concat([D[SPEC["x"]], pd.get_dummies(D.nr, drop_first=True, dtype=float)], axis=1)
    mu = sm.OLS(D.lwage, sm.add_constant(X)).fit()
    mr = sm.OLS(D.lwage, sm.add_constant(D[SPEC["x"]])).fit()
    N = D.nr.nunique()
    F, p, _ = mu.compare_f_test(mr)
    f = t["F test of unit effects"]
    assert f.statistic == pytest.approx(F, rel=1e-8) and f.p_value == pytest.approx(p, abs=1e-12)
    # Breusch-Pagan LM from its definition
    e = mr.resid.to_numpy()
    u = pd.factorize(D.nr)[0]
    Ti = np.bincount(u)
    s = pd.Series(e).groupby(u).sum().to_numpy()
    n = len(e)
    LM = n ** 2 / (2 * (Ti * (Ti - 1)).sum()) * ((s ** 2).sum() / (e @ e) - 1) ** 2
    assert t["Breusch-Pagan LM test of unit effects"].statistic == pytest.approx(LM, rel=1e-8)
    # Jarque-Bera on the FE residuals
    ef = mu.resid.to_numpy()
    jb = stats.jarque_bera(ef)
    assert t["Jarque-Bera test of residual normality"].statistic == pytest.approx(jb[0], rel=1e-6)


def test_effects(rep):
    al = np.array(tab(rep, "t_eff").raw["effects"])
    X = pd.concat([D[SPEC["x"]], pd.get_dummies(D.nr, drop_first=True, dtype=float)], axis=1)
    m = sm.OLS(D.lwage, sm.add_constant(X)).fit()
    b = m.params[SPEC["x"]].to_numpy()
    expected = (D.lwage.to_numpy() - D[SPEC["x"]].to_numpy() @ b)
    expected = pd.Series(expected).groupby(pd.factorize(D.nr)[0]).mean().to_numpy()
    assert al == pytest.approx(expected, rel=1e-8)


def test_flags(rep):
    assert all(s.validated and s.validated != "not validated" for s in rep.steps)
