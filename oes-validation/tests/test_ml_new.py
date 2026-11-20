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


# ------------------------------------------------------------------ stacking
def _stack_data(kind, n=1200, seed=0):
    rng = np.random.default_rng(seed); X = rng.normal(size=(n, 4)); e = rng.normal(size=n) * 0.4
    y = {"linear": 1 + X @ np.array([1.0, -0.5, 0.3, 0.0]) + e, "nonlinear": np.sin(2 * X[:, 0]) * 2 + (X[:, 1] > 0) * 1.5 + X[:, 2] * X[:, 3] + e}[kind]
    return pd.DataFrame(dict(y=y, x1=X[:, 0], x2=X[:, 1], x3=X[:, 2], x4=X[:, 3]))


def _wts(r): return {x[0]: float(x[1]) for x in r["tables"][0]["rows"][:6]}


def test_stacking_registered():
    m = o.BY_ID["stacking"]; assert m["group"] == "Machine learning" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_nnls_weights_recover_known_mixture():
    rng = np.random.default_rng(0); P = rng.normal(size=(500, 3)); y = 0.7 * P[:, 0] + 0.3 * P[:, 1]; w = ml_extra._nnls_weights(P, y)
    assert np.allclose(w, [0.7, 0.3, 0.0], atol=1e-8) and abs(w.sum() - 1) < 1e-12 and (w >= 0).all()


def test_stacking_weights_follow_the_truth():
    lin = _wts(ml_extra.stacking(_stack_data("linear"), dict(y="y", x=["x1", "x2", "x3", "x4"]))); non = _wts(ml_extra.stacking(_stack_data("nonlinear"), dict(y="y", x=["x1", "x2", "x3", "x4"])))
    assert lin["OLS"] + lin["LASSO"] + lin["Ridge"] > 0.7 and non["Random forest"] + non["Gradient boosting"] + non["k nearest neighbours"] > 0.85 and abs(sum(non.values()) - 1) < 1e-3


def test_stacking_is_close_to_best_single_learner_out_of_sample():
    ratio = []
    for s in range(4):
        r = ml_extra.stacking(_stack_data("nonlinear", n=1500, seed=10 + s), dict(y="y", x=["x1", "x2", "x3", "x4"])); sm = {x["label"]: float(x["value"]) for x in r["summary"] if x["label"].startswith("Test RMSE")}
        ratio.append(sm["Test RMSE, super learner"] / sm["Test RMSE, best single learner (chosen on the test set)"])
    assert np.mean(ratio) < 1.05


def test_stacking_equal_weights_row_and_correlation_matrix():
    r = ml_extra.stacking(_stack_data("linear", n=600), dict(y="y", x=["x1", "x2", "x3", "x4"], folds=4)); C = np.array([[float(v) for v in x[1:]] for x in r["tables"][1]["rows"]])
    assert np.allclose(C, C.T, atol=1e-3) and np.allclose(np.diag(C), 1) and r["tables"][0]["rows"][-1][0] == "Equal weights"


def test_stacking_errors():
    d = _stack_data("linear", n=200)
    with pytest.raises(OESError, match="folds"): ml_extra.stacking(d, dict(y="y", x=["x1"], folds=20))
    with pytest.raises(OESError, match="test share"): ml_extra.stacking(d, dict(y="y", x=["x1"], test_share=0.8))
    with pytest.raises(OESError, match="more than one role"): ml_extra.stacking(d, dict(y="y", x=["y"]))
    with pytest.raises(OESError, match="too small"): ml_extra.stacking(d.iloc[:60], dict(y="y", x=["x1"], folds=10))


# ------------------------------------------------------------------ heterogeneity (BLP, GATES, CLAN)
def _het_data(n=2000, tau=lambda x: 1 + 2 * x[:, 0], seed=0, p=0.5):
    rng = np.random.default_rng(seed); X = rng.normal(size=(n, 3)); D = (rng.random(n) < p).astype(float)
    Y = 0.5 * X[:, 1] + X[:, 2] + D * tau(X) + rng.normal(size=n); return pd.DataFrame(dict(y=Y, d=D, x1=X[:, 0], x2=X[:, 1], x3=X[:, 2]))


def _hrun(d, **kw): return ml_extra.hetero(d, dict(y="y", treat="d", x=["x1", "x2", "x3"], **kw))
def hrow(r, i, name): return next(x for x in r["tables"][i]["rows"] if x[0].startswith(name))


def test_hetero_registered():
    m = o.BY_ID["hetero"]; assert m["group"] == "Machine learning" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_hetero_detects_and_sorts_true_heterogeneity():
    r = _hrun(_het_data(n=3000), splits=10); b1, b2 = hrow(r, 0, "Average"), hrow(r, 0, "Heterogeneity")
    assert abs(float(b1[1]) - 1.0) < 0.2 and 0.6 < float(b2[1]) < 1.4 and float(b2[5]) < 0.01
    g = [float(x[1]) for x in r["tables"][1]["rows"][:4]]; assert g == sorted(g) and g[3] - g[0] > 2.0 and float(r["tables"][1]["rows"][4][5]) < 0.01
    cl = hrow(r, 2, "x1"); assert float(cl[3]) > 1.0 and float(cl[5]) < 0.01 and abs(float(hrow(r, 2, "x3")[3])) < 0.4


def test_hetero_no_heterogeneity_gives_loading_near_zero_and_rare_rejection():
    rej, cover = 0, 0
    for s in range(30):
        r = _hrun(_het_data(n=500, tau=lambda x: np.ones(len(x)) * 1.0, seed=50 + s), splits=5, seed=s); rej += float(hrow(r, 0, "Heterogeneity")[5]) < 0.05
        b = hrow(r, 0, "Average"); cover += float(b[3]) < 1.0 < float(b[4])
    assert rej <= 4 and cover >= 24


def test_hetero_average_effect_matches_difference_in_means():
    d = _het_data(n=3000, seed=3); r = _hrun(d, splits=10); dim = d.y[d.d == 1].mean() - d.y[d.d == 0].mean(); assert abs(float(hrow(r, 0, "Average")[1]) - dim) < 0.25


def test_hetero_unbalanced_treatment_and_reproducible():
    d = _het_data(n=1500, seed=4, p=0.25); a = _hrun(d, splits=5, seed=2); b = _hrun(d, splits=5, seed=2)
    assert a["tables"] == b["tables"] and abs(a["summary"][1]["value"] - 0.25) < 0.03


def test_hetero_errors():
    d = _het_data(n=300)
    with pytest.raises(OESError, match="0/1"): ml_extra.hetero(d.assign(d=d.d * 2), dict(y="y", treat="d", x=["x1"]))
    with pytest.raises(OESError, match="more than one role"): ml_extra.hetero(d, dict(y="y", treat="d", x=["d", "x1"]))
    with pytest.raises(OESError, match="at least"): ml_extra.hetero(d.iloc[:50], dict(y="y", treat="d", x=["x1"], groups=4))
    with pytest.raises(OESError, match="not found"): ml_extra.hetero(d, dict(y="y", treat="d", x=["zz"]))


# ------------------------------------------------------------------ double-selection LASSO
def _ds_sim(n=200, p=100, seed=0, alpha=0.5):
    rng = np.random.default_rng(seed); idx = np.arange(p); Sg = 0.5 ** np.abs(idx[:, None] - idx[None, :]); X = rng.multivariate_normal(np.zeros(p), Sg, size=n)
    th = 1.0 / (idx + 1) ** 2; d = X @ th * 3 + rng.normal(size=n); y = alpha * d + X @ (th * 3) + rng.normal(size=n)
    return X, d, y


def test_dsel_registered():
    m = o.BY_ID["dsel"]; assert m["group"] == "Machine learning" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_dsel_reproduces_published_hdm_growth_result():
    # Belloni, Chernozhukov and Hansen (2014) / hdm::rlassoEffect(method = "double selection") on GrowthData: -0.05001 (s.e. 0.01579)
    d = pd.read_csv(os.path.join(MX, "growth.csv")); r = ml_extra.dsel(d, dict(y="growth", treat="gdpsh465", x=[c for c in d.columns if c not in ("growth", "gdpsh465")]))
    row = row_(r, "Double selection (plug-in")
    assert abs(float(row[1]) + 0.05001) < 6e-5 and abs(float(row[2]) - 0.01579) < 6e-5 and int(row[5]) == 7


def row_(r, name): return next(x for x in r["tables"][0]["rows"] if x[0].startswith(name))


def test_dsel_matches_manual_post_selection_ols():
    X, d, y = _ds_sim(seed=3); df = pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])]); df["d"], df["y"] = d, y
    r = ml_extra.dsel(df, dict(y="y", treat="d", x=list(df.columns[:-2])))
    sel = sorted(set(ml_extra._rlasso(y, X)) | set(ml_extra._rlasso(d, X))); Z = np.column_stack([np.ones(len(y)), d, X[:, sel]]); b = np.linalg.lstsq(Z, y, rcond=None)[0]
    assert abs(float(row_(r, "Double selection (plug-in")[1]) - b[1]) < 1e-5 and int(row_(r, "Double selection (plug-in")[5]) == len(sel)


def test_dsel_monte_carlo_coverage_and_naive_failure():
    cov_ds = cov_nc = 0; R = 60
    for s in range(R):
        X, d, y = _ds_sim(n=200, p=100, seed=500 + s); df = pd.DataFrame(X, columns=[f"x{j}" for j in range(100)]); df["d"], df["y"] = d, y
        r = ml_extra.dsel(df, dict(y="y", treat="d", x=list(df.columns[:100])))
        a = row_(r, "Double selection (plug-in"); b = row_(r, "No controls")
        cov_ds += float(a[3]) < 0.5 < float(a[4]); cov_nc += float(b[3]) < 0.5 < float(b[4])
    assert cov_ds / R >= 0.85 and cov_nc / R <= 0.1


def test_dsel_recovers_effect_and_selects_signal_variables():
    X, d, y = _ds_sim(n=400, p=60, seed=7, alpha=1.0); df = pd.DataFrame(X, columns=[f"x{j}" for j in range(60)]); df["d"], df["y"] = d, y
    r = ml_extra.dsel(df, dict(y="y", treat="d", x=list(df.columns[:60]))); b = row_(r, "Double selection (plug-in"); assert abs(float(b[1]) - 1.0) < 0.15
    sel = [x[0] for x in r["tables"][1]["rows"]]; assert "x0" in sel and len(sel) < 20


def test_dsel_p_larger_than_n_has_no_full_ols_and_errors():
    X, d, y = _ds_sim(n=60, p=120, seed=5); df = pd.DataFrame(X, columns=[f"x{j}" for j in range(120)]); df["d"], df["y"] = d, y
    r = ml_extra.dsel(df, dict(y="y", treat="d", x=list(df.columns[:120]))); assert row_(r, "All controls")[1] == "n/a"
    with pytest.raises(OESError, match="more than one role"): ml_extra.dsel(df, dict(y="y", treat="d", x=["d", "x0"]))
    with pytest.raises(OESError, match="penalty constant"): ml_extra.dsel(df, dict(y="y", treat="d", x=["x0"], c=9))
    with pytest.raises(OESError, match="not found"): ml_extra.dsel(df, dict(y="y", treat="d", x=["nope"]))
    with pytest.raises(OESError, match="at least 30"): ml_extra.dsel(df.iloc[:20], dict(y="y", treat="d", x=["x0", "x1"]))


# ------------------------------------------------------------------ clustering
def _blobs(n=300, k=3, sep=6.0, seed=0, noise=1.0):
    rng = np.random.default_rng(seed); cent = (np.eye(k, 3) * sep if k <= 3 else rng.normal(size=(k, 3)) * sep); lab = rng.integers(0, k, n)
    X = cent[lab] + rng.normal(size=(n, 3)) * noise; d = pd.DataFrame(X, columns=["a", "b", "c"]); return d, lab


def _crun(d, **kw): return ml_extra.cluster(d, dict(x=["a", "b", "c"], boot=20, **kw))
def crow(r, i): return r["tables"][i]["rows"]


def test_cluster_registered():
    m = o.BY_ID["cluster"]; assert m["group"] == "Machine learning" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


@pytest.mark.parametrize("method", ["kmeans", "gmm", "ward"])
def test_cluster_recovers_known_clusters(method):
    from sklearn.metrics import adjusted_rand_score
    d, lab = _blobs(); r = _crun(d, method=method); s = S(r)
    assert s["Clusters"] == 3 and float(s["Average silhouette"]) > 0.6 and float(s["Mean adjusted Rand index (stability)"]) > 0.9
    sizes = [x[1] for x in crow(r, 1)[:-1]]; assert sizes == sorted(sizes, reverse=True) and sum(sizes) == 300
    assert sorted(sizes) == sorted(np.bincount(lab).tolist())


def test_cluster_statistics_match_sklearn():
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    d, _ = _blobs(seed=2, sep=3.0); r = _crun(d, method="kmeans", k=4); Z = ((d - d.mean()) / d.std(ddof=0)).to_numpy(); km = KMeans(4, n_init=10, random_state=0).fit(Z)
    row = next(x for x in crow(r, 0) if x[0] == 4)
    assert abs(float(row[1]) - km.inertia_) < 0.02 and abs(float(row[2]) - silhouette_score(Z, km.labels_)) < 1e-4 and abs(float(row[3]) - calinski_harabasz_score(Z, km.labels_)) < 0.02
    assert S(r)["Clusters"] == 4


def test_cluster_weak_structure_is_flagged_by_silhouette_and_stability():
    rng = np.random.default_rng(3); d = pd.DataFrame(rng.uniform(size=(300, 3)), columns=["a", "b", "c"]); r = _crun(d, method="kmeans"); good = _crun(_blobs()[0], method="kmeans")
    assert float(S(r)["Average silhouette"]) < 0.4 and float(S(good)["Average silhouette"]) > 0.6 and float(S(r)["Mean adjusted Rand index (stability)"]) < float(S(good)["Mean adjusted Rand index (stability)"])


def test_cluster_reproducible_and_errors():
    d, _ = _blobs(seed=5); assert _crun(d)["tables"] == _crun(d)["tables"]
    with pytest.raises(OESError, match="method"): _crun(d, method="dbscan")
    with pytest.raises(OESError, match="between 2"): _crun(d, k=50)
    with pytest.raises(OESError, match="whole number"): _crun(d, k="many")
    with pytest.raises(OESError, match="at least two"): ml_extra.cluster(d, dict(x=["a"]))
    with pytest.raises(OESError, match="constant"): ml_extra.cluster(d.assign(c=1.0), dict(x=["a", "b", "c"]))
    with pytest.raises(OESError, match="at least"): ml_extra.cluster(d.iloc[:20], dict(x=["a", "b", "c"]))


# ------------------------------------------------------------------ neural network
def _nn_data(kind, n=900, seed=0):
    rng = np.random.default_rng(seed); X = rng.normal(size=(n, 3)); e = rng.normal(size=n) * 0.2
    y = {"nonlinear": np.sin(2.5 * X[:, 0]) + X[:, 1] ** 2 + e, "linear": 1 + X @ np.array([1.0, -0.7, 0.4]) + e}[kind]
    return pd.DataFrame(dict(y=y, a=X[:, 0], b=X[:, 1], c=X[:, 2]))


def _nrun(d, **kw): return ml_extra.nnet(d, dict(y="y", x=["a", "b", "c"], **kw))
def nrow(r, name): return next(x for x in r["tables"][1]["rows"] if x[0].startswith(name))


def test_nnet_registered():
    m = o.BY_ID["nnet"]; assert m["group"] == "Machine learning" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_nnet_learns_nonlinear_function_and_matches_ols_on_linear():
    r = _nrun(_nn_data("nonlinear"), seeds=3); assert float(nrow(r, "Neural network, ensemble")[2]) > float(nrow(r, "OLS")[2]) + 0.4 and float(nrow(r, "Neural network, ensemble")[2]) > 0.8
    r = _nrun(_nn_data("linear", seed=1), seeds=3); assert float(nrow(r, "Neural network, ensemble")[2]) > float(nrow(r, "OLS")[2]) - 0.03


def test_nnet_ensemble_never_worse_than_average_single_network():
    r = _nrun(_nn_data("nonlinear", seed=2), seeds=4); assert float(nrow(r, "Neural network, ensemble")[1]) <= float(nrow(r, "Neural network, single (mean")[1]) + 1e-9
    worst, best = [float(v) for v in nrow(r, "Neural network, single (worst")[1].split(" / ")]; assert best <= worst


def test_nnet_selects_lowest_cv_architecture_and_is_reproducible():
    d = _nn_data("nonlinear", n=500, seed=3); r = _nrun(d, seeds=2); rows = r["tables"][0]["rows"]; chosen = [x for x in rows if x[3] == "yes"]
    assert len(chosen) == 1 and float(chosen[0][2]) == min(float(x[2]) for x in rows) and r["tables"] == _nrun(d, seeds=2)["tables"]
    assert S(r)["Architecture"].startswith("(")


def test_nnet_time_split_and_errors():
    d = _nn_data("linear", n=400); d["t"] = np.arange(len(d)); assert "last" in _nrun(d, time="t", seeds=1)["tables"][1]["note"]
    with pytest.raises(OESError, match="at least 60"): _nrun(d.iloc[:70], seeds=1)
    with pytest.raises(OESError, match="whole number"): _nrun(d, seeds="x")
    with pytest.raises(OESError, match="between 1 and 20"): _nrun(d, seeds=50)
    with pytest.raises(OESError, match="more than one role"): ml_extra.nnet(d, dict(y="y", x=["y", "a"]))
