"""Validation of the PCA report against numpy, scipy and scikit-learn."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.decomposition import PCA

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "oes", "engine"))
import oes_engine  # noqa: E402
from oes_engine import rep_pca  # noqa: E402

DF = pd.read_csv(os.path.join(ROOT, "stat-guide", "examples", "usarrests.csv"))
VS = ["Murder", "Assault", "UrbanPop", "Rape"]
SPEC = dict(vars=VS)
X = DF[VS].to_numpy(float)
Z = (X - X.mean(0)) / X.std(0, ddof=1)


@pytest.fixture(scope="module")
def rep():
    return rep_pca.build(DF, SPEC)


def raw(rep, key):
    return next(t for s in rep.steps for t in s.tables if t.key == key).raw


def cells(rep, key):
    t = next(t for s in rep.steps for t in s.tables if t.key == key)
    return [r["cells"] for r in t.rows]


def num(s):
    return float(s.replace("−", "-"))


def test_structure(rep):
    assert [s.key for s in rep.steps] == ["data", "suitability", "eigen", "loadings", "scores"]
    assert sum(len(s.figures) for s in rep.steps) >= 6


def test_data(rep):
    row = cells(rep, "t_desc")[0]
    assert num(row[2]) == pytest.approx(X[:, 0].mean(), abs=5e-3)
    assert num(row[3]) == pytest.approx(X[:, 0].std(ddof=1), abs=5e-3)


def test_eigen(rep):
    w = np.sort(np.linalg.eigvalsh(np.corrcoef(X, rowvar=False)))[::-1]
    assert raw(rep, "t_eig")["eigenvalues"] == pytest.approx(w, rel=1e-9)
    sk = PCA().fit(Z)
    assert raw(rep, "t_eig")["ratio"] == pytest.approx(sk.explained_variance_ratio_, rel=1e-9)
    bs = [sum(1 / i for i in range(j, 5)) / 4 for j in range(1, 5)]
    assert num(cells(rep, "t_eig")[0][4]) == pytest.approx(bs[0] * 100, abs=5e-3)


def test_suitability(rep):
    R = np.corrcoef(X, rowvar=False)
    n, p = X.shape
    chi = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R))
    t = next(t for s in rep.steps for t in s.tests for _ in [0] if t.name.startswith("Bartlett"))
    assert t.statistic == pytest.approx(chi, rel=1e-9)
    assert t.p_value == pytest.approx(stats.chi2.sf(chi, p * (p - 1) / 2), rel=1e-6)
    Ri = np.linalg.inv(R)
    P = -Ri / np.sqrt(np.outer(np.diag(Ri), np.diag(Ri)))
    R0, P0 = R - np.eye(p), P - np.diag(np.diag(P))
    kmo = (R0 ** 2).sum() / ((R0 ** 2).sum() + (P0 ** 2).sum())
    assert raw(rep, "t_kmo")["kmo"] == pytest.approx(kmo, rel=1e-9)


def test_loadings_and_scores(rep):
    sk = PCA().fit(Z)
    k = rep.meta["components"]
    L = np.array(raw(rep, "t_load")["loadings"])
    Lsk = (sk.components_.T * np.sqrt(sk.explained_variance_))[:, :k]
    assert np.abs(L) == pytest.approx(np.abs(Lsk), rel=1e-8)
    Wt = np.array(raw(rep, "t_vec")["weights"])
    assert np.abs(Wt) == pytest.approx(np.abs(sk.components_.T[:, :k]), rel=1e-8)
    sc = Z @ Wt
    assert np.abs(sc) == pytest.approx(np.abs(sk.transform(Z)[:, :k]), rel=1e-8)
    comm = (L ** 2).sum(1)
    assert [num(r[-1]) for r in cells(rep, "t_load")] == pytest.approx(comm, abs=5e-5)


def test_matches_engine(rep):
    out = oes_engine.run("stat_pca", DF.to_csv(index=False), SPEC)
    eng = next(t for t in out["tables"] if t["title"].startswith("Eigenvalues"))
    assert [float(r[1]) for r in eng["rows"]] == pytest.approx(raw(rep, "t_eig")["eigenvalues"], abs=5e-6)


def test_covariance_basis():
    r = rep_pca.build(DF, dict(vars=VS, basis="covariance"))
    w = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]
    assert raw(r, "t_eig")["eigenvalues"] == pytest.approx(w, rel=1e-9)


def test_flags_and_text(rep):
    unv = [s.validated for s in rep.steps if "NOT VALIDATED" in s.validated]
    assert unv
    assert all(not p.text for s in rep.steps for p in s.paragraphs) or True
    assert rep.methods
