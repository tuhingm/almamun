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


# ------------------------------------------------------------------ interpretation
def _lin_data(n=1500, seed=0, kind="additive"):
    rng = np.random.default_rng(seed); X = rng.normal(size=(n, 4)); e = rng.normal(size=n) * 0.3
    y = {"additive": 2 * X[:, 0] + np.sin(2 * X[:, 1]) + e, "interact": 2 * X[:, 0] * X[:, 1] + e, "linear": 3 * X[:, 0] + e}[kind]
    return pd.DataFrame(dict(y=y, x1=X[:, 0], x2=X[:, 1], x3=X[:, 2], x4=X[:, 3]))


def _irun(d, **kw): return ml_extra.interpret(d, dict(y="y", x=["x1", "x2", "x3", "x4"], repeats=15, **kw))


def test_interpret_registered():
    m = o.BY_ID["interpret"]; assert m["group"] == "Machine learning" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_interpret_importance_finds_relevant_variables():
    r = _irun(_lin_data()); rows = r["tables"][0]["rows"]; names = [x[0] for x in rows]
    assert set(names[:2]) == {"x1", "x2"} and float(rows[0][1]) > 0.3
    assert all(abs(float(x[1])) < 0.03 for x in rows if x[0] in ("x3", "x4"))


def test_interpret_partial_dependence_recovers_linear_slope():
    r = _irun(_lin_data(kind="linear", n=2500)); pd_row = next(x for x in r["tables"][1]["rows"] if x[0] == "x1")
    lo, hi = float(pd_row[1]), float(pd_row[2]); rng_pd = float(pd_row[5])
    assert abs(rng_pd - 3 * (hi - lo)) < 0.6 and float(pd_row[6]) > 0.97


def test_interpret_interaction_strength_distinguishes_additive_from_product():
    h_add = float(_irun(_lin_data(kind="additive", seed=1), top=2)["tables"][-1]["rows"][0][1]); h_int = float(_irun(_lin_data(kind="interact", seed=1), top=2)["tables"][-1]["rows"][0][1])
    assert h_add < 0.12 and h_int > 0.6


def test_pd_at_matches_double_loop():
    from sklearn.ensemble import RandomForestRegressor
    d = _lin_data(n=300, seed=2); X = d[["x1", "x2", "x3", "x4"]].to_numpy(); m = RandomForestRegressor(n_estimators=30, random_state=0).fit(X, d.y); Xs = X[:25]
    fast = ml_extra._pd_at(m, Xs, [0, 1]); slow = []
    for i in range(len(Xs)):
        Z = Xs.copy(); Z[:, 0] = Xs[i, 0]; Z[:, 1] = Xs[i, 1]; slow.append(m.predict(Z).mean())
    assert np.allclose(fast, slow)


def test_interpret_boosting_option_and_errors():
    r = _irun(_lin_data(n=800), model="gb"); assert "boosting" in r["title"]
    d = _lin_data(n=300)
    with pytest.raises(OESError, match="rf"): ml_extra.interpret(d, dict(y="y", x=["x1", "x2"], model="svm"))
    with pytest.raises(OESError, match="at least two"): ml_extra.interpret(d, dict(y="y", x=["x1"]))
    with pytest.raises(OESError, match="more than one role"): ml_extra.interpret(d, dict(y="y", x=["y", "x1"]))
