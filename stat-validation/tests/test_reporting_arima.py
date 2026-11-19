"""Validation of the ARIMA report against definitions, statsmodels' separate ARIMA class, and direct computations."""
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "oes", "engine"))
import oes_engine  # noqa: E402
from oes_engine import rep_arima  # noqa: E402
from oes_engine.rep_var import adf_by_ols  # noqa: E402

DF = pd.read_csv(os.path.join(ROOT, "oes", "examples", "pce_inflation.csv"))
SPEC = dict(y="infl", time="date", d="0", max_p=2, max_q=2, holdout=24)
X = DF.sort_values("date")["infl"].dropna().to_numpy(float)


@pytest.fixture(scope="module")
def rep():
    return rep_arima.build(DF, SPEC)


def tab(rep, key):
    return next(t for s in rep.steps for t in s.tables if t.key == key)


def tmap(rep):
    return {t.name: t for s in rep.steps for t in s.tests}


def test_structure(rep):
    assert [s.key for s in rep.steps] == ["data", "unitroot", "order", "estimation", "diag", "forecast"]
    assert sum(len(s.figures) for s in rep.steps) >= 12


def test_acf_pacf():
    a = rep_arima.acf(X, 12)
    assert a == pytest.approx(sm_acf(X, nlags=12, fft=False), abs=1e-10)
    assert rep_arima.pacf_yw(X, 12) == pytest.approx(sm_pacf(X, nlags=12, method="ywm"), abs=1e-8)


def test_adf_kpss(rep):
    raw = tab(rep, "t_ur").raw
    stat, lags = raw["adf_level"]
    assert stat == pytest.approx(adf_by_ols(X, lags), rel=1e-8)
    stat_d, lags_d = raw["adf_diff"]
    assert stat_d == pytest.approx(adf_by_ols(np.diff(X), lags_d), rel=1e-8)
    k, L = raw["kpss_level"]
    e = X - X.mean()
    n = len(e)
    S = np.cumsum(e)
    lrv = e @ e / n + 2 * sum((1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n for l in range(1, L + 1))
    assert k == pytest.approx((S ** 2).sum() / (n ** 2 * lrv), rel=1e-8)


def test_order_ic(rep):
    r = tab(rep, "t_order").raw
    assert r["grid"][0][3] == min(g[3] for g in r["grid"])
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    p, q = r["chosen"]
    m = SARIMAX(X[:-24], order=(p, 0, q), trend="c").fit(disp=False, maxiter=200)
    k = len(m.params)
    assert m.bic == pytest.approx(-2 * m.llf + k * math.log(m.nobs), rel=1e-6)
    assert r["grid"][0][3] == pytest.approx(m.bic, rel=1e-6)


def test_estimation_vs_arima_class(rep):
    from statsmodels.tsa.arima.model import ARIMA
    r = tab(rep, "t_est").raw
    p, q = tab(rep, "t_order").raw["chosen"]
    m = ARIMA(X, order=(p, 0, q), trend="c").fit()
    assert r["llf"] == pytest.approx(m.llf, rel=1e-4)
    mp = dict(zip(m.param_names, m.params))
    for nm, b in zip(r["names"], r["params"]):
        if nm == "intercept":       # SARIMAX reports the equation constant, ARIMA the process mean
            ar = sum(v for k_, v in mp.items() if k_.startswith("ar."))
            assert b == pytest.approx(mp["const"] * (1 - ar), abs=5e-3, rel=2e-2)
        else:
            assert b == pytest.approx(mp[nm], abs=2e-3, rel=2e-2)


def test_diagnostics(rep):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    p, q = tab(rep, "t_order").raw["chosen"]
    m = SARIMAX(X[:-24], order=(p, 0, q), trend="c").fit(disp=False, maxiter=200)
    e = np.asarray(m.resid)[max(p, 1):]
    n = len(e)
    ec = e - e.mean()
    r = np.array([(ec[k:] * ec[:n - k]).sum() / (ec ** 2).sum() for k in range(13)])
    Q = n * (n + 2) * sum(r[k] ** 2 / (n - k) for k in range(1, 13))
    t = tmap(rep)["Ljung-Box (12 lags)"]
    assert t.statistic == pytest.approx(Q, rel=1e-8)
    assert t.p_value == pytest.approx(stats.chi2.sf(Q, 12), rel=1e-6)
    jb = n / 6 * (stats.skew(e) ** 2 + (stats.kurtosis(e, fisher=False) - 3) ** 2 / 4)
    assert tmap(rep)["Jarque-Bera"].statistic == pytest.approx(jb, rel=1e-8)
    # ARCH LM: n R^2 of squared residuals on 6 lags
    e2 = e ** 2
    Z = np.column_stack([np.ones(n - 6)] + [e2[6 - k:n - k] for k in range(1, 7)])
    b = np.linalg.lstsq(Z, e2[6:], rcond=None)[0]
    res = e2[6:] - Z @ b
    r2 = 1 - res @ res / ((e2[6:] - e2[6:].mean()) ** 2).sum()
    assert tmap(rep)["ARCH LM (6 lags)"].statistic == pytest.approx((n - 6) * r2, rel=1e-6)


def test_forecast_accuracy(rep):
    r = tab(rep, "t_acc").raw
    act, fc = np.array(r["actual"]), np.array(r["forecast"])
    bench = np.concatenate([[X[-25]], act[:-1]])
    assert r["rmse"][0] == pytest.approx(np.sqrt(np.nanmean((act - fc) ** 2)), rel=1e-10)
    assert r["rmse"][1] == pytest.approx(np.sqrt(np.mean((act - bench) ** 2)), rel=1e-10)


def test_matches_engine(rep):
    out = oes_engine.run("arima", DF.to_csv(index=False), SPEC)
    eng = next(t for t in out["tables"] if "Hold-out" in t["title"])
    assert float(eng["rows"][0][1]) == pytest.approx(tab(rep, "t_acc").raw["rmse"][0], abs=5e-5)


def test_flags(rep):
    assert all(s.validated != "not validated" for s in rep.steps)
