"""Validation of panel models added after the first release: Hausman-Taylor."""
import os, sys, warnings
import numpy as np, pandas as pd, pytest

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
