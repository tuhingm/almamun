"""Validation of the VAR report: every number is compared with an independent computation (direct OLS, linear algebra, definitions)."""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "oes", "engine"))
import oes_engine  # noqa: E402
from oes_engine import rep_text, rep_var  # noqa: E402

PRICES = pd.read_csv(os.path.join(ROOT, "time-series-guide", "examples", "prices.csv"))
SPEC = dict(vars=["log_sp500", "log_indpro"], time="date", transform="diff", max_lags=12, lags=2)
NAMES = SPEC["vars"]


@pytest.fixture(scope="module")
def rep():
    return rep_var.build(PRICES, SPEC)


@pytest.fixture(scope="module")
def mats():
    d = PRICES.sort_values("date")[NAMES].astype(float).diff().dropna().to_numpy()
    p = 2
    Y = d[p:]
    X = np.column_stack([np.ones(len(Y))] + [d[p - l:len(d) - l] for l in range(1, p + 1)])
    B = np.linalg.solve(X.T @ X, X.T @ Y)
    E = Y - X @ B
    return d, p, X, Y, B, E


def table(rep, key):
    return next(t for s in rep.steps for t in s.tables if t.key == key)


def step_tests(rep, step):
    return {t.name: t for s in rep.steps if s.key == step for t in s.tests}


def test_var_data(rep, mats):
    d = mats[0]
    for r in table(rep, "t_desc").raw["rows"]:
        x = d[:, NAMES.index(r["name"])]
        assert r["mean"] == pytest.approx(x.mean(), rel=1e-12) and r["sd"] == pytest.approx(x.std(ddof=1), rel=1e-12)


def test_var_adf(rep, mats):
    d = mats[0]
    for r in table(rep, "t_adf").raw["rows"]:
        x = d[:, NAMES.index(r["var"])]
        assert r["stat"] == pytest.approx(rep_var.adf_by_ols(x, r["lags"]), rel=1e-8)


def test_var_estimation(rep, mats):
    d, p, X, Y, B, E = mats
    raw = table(rep, "t_var").raw["columns"]
    s2 = (E ** 2).sum(axis=0) / (len(Y) - X.shape[1])
    V = np.linalg.inv(X.T @ X)
    for i, col in enumerate(raw):
        for j, nm in enumerate(["const"] + [f"L{l}.{v}" for l in range(1, p + 1) for v in NAMES]):
            assert col["params"][nm] == pytest.approx(B[j, i], rel=1e-9)
            assert col["se"][nm] == pytest.approx(math.sqrt(s2[i] * V[j, j]), rel=1e-9)
            t = B[j, i] / math.sqrt(s2[i] * V[j, j])
            assert col["p"][nm] == pytest.approx(2 * stats.t.sf(abs(t), len(Y) - X.shape[1]), rel=1e-6, abs=1e-12)


def test_var_matches_engine():
    csv = PRICES.to_csv(index=False)
    a = oes_engine.run("var", csv, dict(SPEC))
    b = oes_engine.run("var", csv, dict(SPEC, report=False))
    assert "report" in a and {k: v for k, v in a.items() if k != "report"} == b
    coef = next(t for t in b["tables"] if t["title"] == f"Equation: {NAMES[0]}")
    raw = next(t for s in a["report"]["steps"] for t in s["tables"] if t["key"] == "t_var")["raw"]["columns"][0]
    for row in coef["rows"]:
        assert raw["params"][row[0]] == pytest.approx(row[1], abs=2e-6)


def test_var_lags(rep, mats):
    d, p, X, Y, B, E = mats
    maxl = 12                                              # select_order uses a common sample: the first max_lags observations are presample
    Yc = d[maxl:]
    Xc = np.column_stack([np.ones(len(Yc))] + [d[maxl - l:len(d) - l] for l in range(1, p + 1)])
    Ec = Yc - Xc @ np.linalg.solve(Xc.T @ Xc, Xc.T @ Yc)
    T_, K = Ec.shape
    sig = Ec.T @ Ec / T_
    aic = math.log(np.linalg.det(sig)) + 2 * (p * K * K + K) / T_
    raw = table(rep, "t_lags").raw
    # the report's table holds the criteria of every lag; the entry at the chosen lag equals the direct computation
    assert raw["aic"][p] == pytest.approx(aic, rel=1e-8)
    bic = math.log(np.linalg.det(sig)) + math.log(T_) / T_ * (p * K * K + K)
    assert raw["bic"][p] == pytest.approx(bic, rel=1e-8)


def test_var_stability(rep, mats):
    d, p, X, Y, B, E = mats
    K = len(NAMES)
    A = np.zeros((K * p, K * p))
    A[:K, :] = B[1:].T
    A[K:, :-K] = np.eye(K * (p - 1))
    mine = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
    assert np.allclose(table(rep, "t_roots").raw["moduli"], mine, rtol=1e-9)


def test_var_diag(rep, mats):
    d, p, X, Y, B, E = mats
    t = step_tests(rep, "diag")
    n = len(E)
    for i, v in enumerate(NAMES):
        e = E[:, i]
        jb = n / 6 * (stats.skew(e) ** 2 + (stats.kurtosis(e, fisher=False) - 3) ** 2 / 4)
        assert t[f"Jarque-Bera, {v}"].statistic == pytest.approx(jb, rel=1e-8)
        m = min(10, n // 5)
        ac = rep_var.acf(e, m)
        q = n * (n + 2) * sum(ac[j] ** 2 / (n - j) for j in range(1, m + 1))
        assert t[f"Ljung-Box Q({m}), {v}"].statistic == pytest.approx(q, rel=1e-8)
        e2 = e ** 2
        Z = np.column_stack([np.ones(n - 4)] + [e2[4 - l:n - l] for l in range(1, 5)])
        b = np.linalg.lstsq(Z, e2[4:], rcond=None)[0]
        r2 = 1 - ((e2[4:] - Z @ b) ** 2).sum() / ((e2[4:] - e2[4:].mean()) ** 2).sum()
        assert t[f"ARCH LM (4 lags), {v}"].statistic == pytest.approx((n - 4) * r2, rel=1e-6)
    h = max(p + 1, min(12, len(d) // 5))
    q, dfq = rep_var.portmanteau(E, h, p)
    key = [k for k in t if k.startswith("Portmanteau")][0]
    assert t[key].statistic == pytest.approx(q, rel=1e-8)
    assert t[key].p_value == pytest.approx(stats.chi2.sf(q, dfq), rel=1e-6)


def test_var_granger(rep, mats):
    d, p, X, Y, B, E = mats
    t = step_tests(rep, "granger")
    for i, caused in enumerate(NAMES):
        for j, causing in enumerate(NAMES):
            if i == j:
                continue
            keep = [0] + [1 + (l - 1) * 2 + c for l in range(1, p + 1) for c in range(2) if c != j]
            Xr = X[:, keep]
            rss1 = ((Y[:, i] - X @ np.linalg.lstsq(X, Y[:, i], rcond=None)[0]) ** 2).sum()
            rss0 = ((Y[:, i] - Xr @ np.linalg.lstsq(Xr, Y[:, i], rcond=None)[0]) ** 2).sum()
            df2 = len(Y) - X.shape[1]
            F = ((rss0 - rss1) / p) / (rss1 / df2)
            got = t[f"{causing} → {caused}"]
            assert got.statistic == pytest.approx(F, rel=1e-8)
            assert got.p_value == pytest.approx(stats.f.sf(F, p, 2 * df2), rel=1e-6)    # statsmodels: F(p, K * residual d.f.)


def test_var_irf(rep, mats):
    d, p, X, Y, B, E = mats
    K = 2
    A = np.zeros((K * p, K * p))
    A[:K, :] = B[1:].T
    A[K:, :-K] = np.eye(K * (p - 1))
    sig = E.T @ E / (len(E) - X.shape[1])
    P = np.linalg.cholesky(sig)
    raw = table(rep, "t_irf").raw
    phi = rep_var.ma_matrices(A, K, len(raw["orth_irf"]) - 1)
    mine = np.array([f @ P for f in phi])
    assert np.allclose(raw["orth_irf"], mine, rtol=1e-7, atol=1e-10)
    H = int(table(rep, "t_fevd").title.split("horizon ")[1].split(" ")[0])
    th = mine[:H] ** 2                                  # steps 0..H-1, as statsmodels' fevd(H)
    share = th.sum(axis=0) / th.sum(axis=0).sum(axis=1, keepdims=True)
    assert np.allclose(np.array(table(rep, "t_fevd").raw["fevd"])[:, -1, :], share, rtol=1e-7)


def test_var_text_and_validation_flags(rep):
    for p in rep.methods + rep.limitations:
        assert rep_text.check_numbers(p) == [], p.text
    # the bands are explicitly marked as not validated in the irf step text
    irf = next(s for s in rep.steps if s.key == "irf")
    assert "not validated" in irf.validated and "not validated" in irf.figures[0].note
    assert 6 <= sum(len(s.figures) for s in rep.steps) <= 15
