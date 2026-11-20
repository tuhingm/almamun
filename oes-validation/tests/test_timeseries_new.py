"""Validation of the time-series models added to OES after the first release: unit-root and stationarity tests.
Checks against (1) statsmodels and the `arch` package, (2) simulated series with a known answer and (3) error messages."""
import os, sys, warnings
import numpy as np, pandas as pd, pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "oes", "engine")); sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import oes_engine as o
from oes_engine import ts_extra

OX = os.path.join(ROOT, "oes", "examples")
def macro(): return pd.read_csv(os.path.join(OX, "us_macro.csv"))
def table(r, prefix): return next(t for t in r["tables"] if t["title"].startswith(prefix))
def urow(r, test, form, det): return next(x for x in table(r, "Unit-root")["rows"] if x[0] == test and x[1] == form and x[2] == det)


def test_unitroot_registered_with_guide_and_example():
    m = o.BY_ID["unitroot"]
    assert m["group"] == "Time series" and os.path.exists(os.path.join(ROOT, m["guide"]))
    assert os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_adf_matches_statsmodels():
    from statsmodels.tsa.stattools import adfuller
    x = macro().tbilrate.to_numpy(float)
    r = ts_extra.unitroot(macro(), dict(y="tbilrate", time="date", maxlag=8, ic="bic", sims=50))
    a = adfuller(x, maxlag=8, regression="c", autolag="BIC")
    row = urow(r, "ADF", "level", "constant")
    assert row[3] == pytest.approx(a[0], abs=1e-3) and row[4] == a[2] and row[5] == pytest.approx(a[1], abs=1e-4)


def test_dfgls_and_phillips_perron_match_arch():
    arch = pytest.importorskip("arch.unitroot")
    x = macro().tbilrate.to_numpy(float)
    for tr in ("c", "ct"):
        for k in (1, 4):
            assert ts_extra._dfgls_stat(x, tr, k) == pytest.approx(arch.DFGLS(x, lags=k, trend=tr).stat, abs=1e-6)
        q = int(4 * (len(x) / 100) ** (2 / 9))
        assert ts_extra._pp(x, tr, q)[0] == pytest.approx(arch.PhillipsPerron(x, lags=q, trend=tr, test_type="tau").stat, abs=1e-6)


def test_regression_values_for_us_tbill_rate():
    # stored values for the example (guards against accidental changes): DF-GLS with 1 lag and PP with 4 lags
    x = macro().tbilrate.to_numpy(float)
    assert ts_extra._dfgls_stat(x, "c", 1) == pytest.approx(-1.8027, abs=5e-4)
    assert ts_extra._pp(x, "c", 4)[0] == pytest.approx(-2.0741, abs=5e-4)


def test_random_walk_is_not_rejected_and_ar1_is():
    rng = np.random.default_rng(3); n = 300
    rw = np.cumsum(rng.standard_normal(n)); e = rng.standard_normal(n); ar = np.zeros(n)
    for t in range(1, n): ar[t] = .5 * ar[t - 1] + e[t]
    r1 = ts_extra.unitroot(pd.DataFrame({"y": rw}), dict(y="y", sims=100)); r2 = ts_extra.unitroot(pd.DataFrame({"y": ar}), dict(y="y", sims=100))
    assert "I(1)" in next(s["value"] for s in r1["summary"] if s["label"] == "Conclusion")
    assert "I(0)" in next(s["value"] for s in r2["summary"] if s["label"] == "Conclusion")
    assert urow(r1, "ADF", "level", "constant")[-1] == "no" and urow(r2, "ADF", "level", "constant")[-1] == "yes"


def test_size_of_dfgls_simulated_critical_values():
    # under a unit root the simulated 5% critical value should reject about 5% of the time
    rng = np.random.default_rng(7); rej = 0; N = 60
    cv = np.percentile([ts_extra._dfgls_stat(np.cumsum(rng.standard_normal(150)), "c", 2) for _ in range(400)], 5)
    for _ in range(N):
        rej += ts_extra._dfgls_stat(np.cumsum(rng.standard_normal(150)), "c", 2) < cv
    assert rej / N < 0.20


def test_zivot_andrews_finds_a_break_in_a_shifted_series():
    rng = np.random.default_rng(11); n = 240; e = rng.standard_normal(n); y = np.zeros(n)
    for t in range(1, n): y[t] = .5 * y[t - 1] + e[t]
    y[120:] += 6
    r = ts_extra.unitroot(pd.DataFrame({"y": y}), dict(y="y", sims=50))
    z = table(r, "Zivot-Andrews")["rows"][0]
    assert abs(int(z[4]) - 121) <= 12


def test_unitroot_errors():
    short = pd.DataFrame({"y": np.arange(20.0) ** 1.1, "t": np.arange(20)})
    assert "40" in o.run("unitroot", short.to_csv(index=False), dict(y="y"))["error"]
    assert "series" in o.run("unitroot", macro().to_csv(index=False), dict())["error"].lower()
    assert "BIC" in o.run("unitroot", macro().to_csv(index=False), dict(y="tbilrate", ic="hq"))["error"]
