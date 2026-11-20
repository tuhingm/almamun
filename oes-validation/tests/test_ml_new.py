"""Validation of machine-learning models added after the first release."""
import os, sys, warnings
import numpy as np, pandas as pd, pytest
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "oes", "engine")); sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import oes_engine as o
from oes_engine import ml_extra
from oes_engine.core import OESError

MX = os.path.join(ROOT, "ml-guide", "examples")
def row(r, title, first): return next(x for x in next(t for t in r["tables"] if t["title"].startswith(title))["rows"] if x[0].startswith(first))
def S(r): return {x["label"]: x["value"] for x in r["summary"]}


def _clf_data(n=3000, seed=0, beta=(1.0, -1.0, 0.5)):
    rng = np.random.default_rng(seed); X = rng.normal(size=(n, 3)); eta = 0.3 + X @ np.array(beta); y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    return pd.DataFrame(dict(y=y, a=X[:, 0], b=X[:, 1], c=X[:, 2]))


def test_classify_registered():
    m = o.BY_ID["classify"]; assert m["group"] == "Machine learning" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_classify_auc_matches_sklearn_and_theory():
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.linear_model import LogisticRegression
    d = _clf_data(n=4000, seed=1); r = ml_extra.classify(d, dict(y="y", x=["a", "b", "c"], test_share=0.25))
    n = len(d); k = int(round(n * 0.25)); idx = np.random.default_rng(0).permutation(n); te, tr = np.sort(idx[:k]), np.sort(idx[k:])
    m = LogisticRegression(C=1e6, max_iter=2000).fit(((d.iloc[tr][["a", "b", "c"]] - d.iloc[tr][["a", "b", "c"]].mean()) / d.iloc[tr][["a", "b", "c"]].std(ddof=0)), d.y.iloc[tr])
    Z = (d.iloc[te][["a", "b", "c"]] - d.iloc[tr][["a", "b", "c"]].mean()) / d.iloc[tr][["a", "b", "c"]].std(ddof=0); p = m.predict_proba(Z)[:, 1]
    lg = row(r, "Out-of-sample", "Logit")
    assert abs(float(lg[1]) - roc_auc_score(d.y.iloc[te], p)) < 1e-4 and abs(float(lg[3]) - brier_score_loss(d.y.iloc[te], p)) < 1e-4
    # true-model AUC from simulation of the same design
    rng = np.random.default_rng(9); X = rng.normal(size=(200000, 3)); e = 0.3 + X @ np.array([1, -1, 0.5]); yy = rng.random(200000) < 1 / (1 + np.exp(-e)); assert abs(float(lg[1]) - roc_auc_score(yy, e)) < 0.03


def test_classify_forest_beats_logit_on_nonlinear_rule():
    rng = np.random.default_rng(2); n = 4000; X = rng.normal(size=(n, 2)); y = ((X[:, 0] * X[:, 1] > 0) ^ (rng.random(n) < 0.05)).astype(float)
    r = ml_extra.classify(pd.DataFrame(dict(y=y, a=X[:, 0], b=X[:, 1])), dict(y="y", x=["a", "b"]))
    assert float(row(r, "Out-of-sample", "Logit")[1]) < 0.58 and float(row(r, "Out-of-sample", "Random forest")[1]) > 0.9 and S(r)["Best model by AUC"] in ("Random forest", "Gradient boosting")


def test_classify_calibration_and_confusion_are_consistent():
    r = ml_extra.classify(_clf_data(n=3000, seed=3), dict(y="y", x=["a", "b", "c"])); cm = next(t for t in r["tables"] if t["title"].startswith("Confusion"))["rows"]
    for c in cm:
        tp, fp, fn, tn = c[1:5]; assert tp + fp + fn + tn == 750 and abs(float(c[5]) - tp / (tp + fn)) < 1e-3 and abs(float(c[6]) - tn / (tn + fp)) < 1e-3
    cal = next(t for t in r["tables"] if t["title"].startswith("Calibration"))["rows"]; assert sum(c[1] for c in cal) == 750
    assert max(abs(float(c[2]) - float(c[3])) for c in cal) < 0.1


def test_classify_time_split_keeps_order():
    d = _clf_data(n=800, seed=4); d["t"] = np.arange(len(d)); r = ml_extra.classify(d, dict(y="y", x=["a", "b", "c"], time="t"))
    assert "last" in r["tables"][0]["note"]


def test_classify_errors():
    d = _clf_data(n=400)
    with pytest.raises(OESError, match="0/1"): ml_extra.classify(d.assign(y=d.y * 2), dict(y="y", x=["a"]))
    with pytest.raises(OESError, match="more than one role"): ml_extra.classify(d, dict(y="y", x=["y", "a"]))
    with pytest.raises(OESError, match="test share"): ml_extra.classify(d, dict(y="y", x=["a"], test_share=0.9))
    tiny = d.copy(); tiny["y"] = 0.0; tiny.loc[:2, "y"] = 1.0
    with pytest.raises(OESError, match="0/1|at least 5"): ml_extra.classify(tiny, dict(y="y", x=["a"]))
