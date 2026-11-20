"""Further machine-learning models: classification, interpretation, stacking, effect heterogeneity, double selection, clustering and neural networks."""
import numpy as np
import pandas as pd
from scipy import stats

from . import _ml_core as M
from .core import OESError, as_list, chart, distinct, f, float_option, int_option, need, result, stat, table, test_row, tests_table, time_order
from .ml import _data, _split


def _xy(df, spec, binary=False):
    y = need(spec, "y", "outcome to predict"); x = as_list(need(spec, "x", "predictors")); distinct([y], x)
    d, ts = _data(df, spec, [y] + x)
    if binary and (d[y].nunique() != 2 or not set(np.unique(d[y])) <= {0.0, 1.0}):
        raise OESError(f"The outcome {y} must be a 0/1 variable with both values present.")
    return d, y, x, ts


# ------------------------------------------------------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------------------------------------------------------
def _auc(y, p):
    r = stats.rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _roc(y, p):
    o = np.argsort(-p); ys = y[o]; tp = np.cumsum(ys) / ys.sum(); fp = np.cumsum(1 - ys) / (1 - ys).sum()
    return np.concatenate([[0], fp]), np.concatenate([[0], tp]), np.concatenate([[1.0], p[o]])


def classify(df, spec):
    d, y, x, ts = _xy(df, spec, binary=True)
    share = float_option(spec, "test_share", 0.25, "Test share")
    if not 0.05 <= share <= 0.5:
        raise OESError("The test share must be between 0.05 and 0.5.")
    X, Y = d[x].astype(float), d[y].astype(float).to_numpy(); tr, te = _split(len(d), ts, share)
    if min(Y[tr].sum(), len(tr) - Y[tr].sum(), Y[te].sum(), len(te) - Y[te].sum()) < 5:
        raise OESError("Each class needs at least 5 observations in both the training and the test sample.")
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    n_est = int_option(spec, "n_estimators", 200, 50, 2000, "Number of trees")
    models = {"Logit": make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=2000)),
              "L1 logit (LASSO)": make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, penalty="l1", solver="liblinear", Cs=10, max_iter=1000, scoring="neg_log_loss")),
              "Random forest": RandomForestClassifier(n_estimators=n_est, min_samples_leaf=5, max_features="sqrt", random_state=0, n_jobs=1),
              "Gradient boosting": GradientBoostingClassifier(n_estimators=n_est, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=0)}
    prob, rows, fitted = {}, [], {}
    base = max(Y[te].mean(), 1 - Y[te].mean())
    for name, m in models.items():
        m.fit(X.iloc[tr], Y[tr]); p = np.clip(m.predict_proba(X.iloc[te])[:, 1], 1e-6, 1 - 1e-6); prob[name] = p; fitted[name] = m
        ll = -float(np.mean(Y[te] * np.log(p) + (1 - Y[te]) * np.log(1 - p)))
        rows.append([name, f(_auc(Y[te], p), 4), f(float(np.mean((p > 0.5) == (Y[te] == 1))), 4), f(float(np.mean((p - Y[te]) ** 2)), 4), f(ll, 4)])
    best = max(prob, key=lambda k: _auc(Y[te], prob[k]))
    acc = table("Out-of-sample classification accuracy", ["Model", "AUC", "Accuracy (cut-off 0.5)", "Brier score", "Log loss"], rows,
                f"Test set: {'the last' if ts else 'a random'} {int(round(share * 100))}% of the observations ({len(te)}). Always predicting the more common class gives accuracy {base:.4f} and AUC 0.5.")
    p = prob[best]; fpr, tpr, thr = _roc(Y[te], p); j = int(np.argmax(tpr - fpr)); cut = float(thr[j])
    def conf(c):
        pr = p >= c; tp = int(((pr) & (Y[te] == 1)).sum()); fp = int((pr & (Y[te] == 0)).sum()); fn = int((~pr & (Y[te] == 1)).sum()); tn = int((~pr & (Y[te] == 0)).sum())
        return [f(c, 3), tp, fp, fn, tn, f(tp / max(tp + fn, 1), 4), f(tn / max(tn + fp, 1), 4)]
    cm = table(f"Confusion matrix of the best model ({best}) at two cut-offs", ["Cut-off", "True positives", "False positives", "False negatives", "True negatives", "Sensitivity", "Specificity"], [conf(0.5), conf(cut)],
               "The second row uses the cut-off that maximises sensitivity plus specificity (Youden's J) on the test set; choose the cut-off from the costs of the two kinds of error instead if you can.")
    q = np.quantile(p, np.linspace(0, 1, 6)); q[-1] += 1e-9; cal = []
    for a, b in zip(q[:-1], q[1:]):
        k = (p >= a) & (p < b); cal.append([f"{a:.3f} to {min(b, 1):.3f}", int(k.sum()), f(float(p[k].mean()), 4), f(float(Y[te][k].mean()), 4)])
    ctab = table(f"Calibration of {best}: predicted against observed share, by quintile of predicted probability", ["Predicted probability", "Observations", "Mean predicted", "Observed share"], cal,
                 "A calibrated model has the two columns close. Discrimination (AUC) and calibration are different properties.")
    tabs = [acc, cm, ctab]
    l1 = fitted["L1 logit (LASSO)"][-1]; co = l1.coef_[0]
    tabs.append(table("L1 logit coefficients (standardised predictors)", ["Predictor", "Coefficient", "Status"], [[v, f(c), "kept" if abs(c) > 1e-10 else "dropped"] for v, c in sorted(zip(x, co), key=lambda z: -abs(z[1]))],
                      "Log-odds per standard deviation of the predictor."))
    charts = [chart("ROC curve of the best model", [float(v) for v in fpr[::max(1, len(fpr) // 100)]], [(f"{best}", [float(v) for v in tpr[::max(1, len(tpr) // 100)]]), ("No skill", [float(v) for v in fpr[::max(1, len(fpr) // 100)]])], "False positive rate", "True positive rate", kind="line")]
    return result("classify", f"Classification of {y}", len(d), [stat("Observations", len(d)), stat("Predictors", len(x)), stat("Share of ones", float(Y.mean()), 4), stat("Best model by AUC", best), stat("Best AUC", _auc(Y[te], p), 4), stat("Youden cut-off", cut, 3)],
                  tabs, charts, ["AUC is the probability that a random positive receives a higher score than a random negative. Accuracy depends on the cut-off and on the class balance; report AUC, Brier score and calibration together.",
                                 "The hold-out test set is used once. Tuning (the LASSO penalty) is done by cross-validation on the training set."])


# ------------------------------------------------------------------------------------------------------------------
# Interpretation of a fitted model: permutation importance, partial dependence and interaction strength
# ------------------------------------------------------------------------------------------------------------------
def _pd_curve(model, Xr, j, grid):
    out = []
    for v in grid:
        Z = Xr.copy(); Z[:, j] = v; out.append(float(model.predict(Z).mean()))
    return np.array(out)


def _pd_at(model, Xs, cols):
    """Partial dependence of the columns in cols, evaluated at each row of Xs (one prediction call)."""
    n = len(Xs); Z = np.tile(Xs, (n, 1))
    for c in cols:
        Z[:, c] = np.repeat(Xs[:, c], n)
    return model.predict(Z).reshape(n, n).mean(1)


def interpret(df, spec):
    d, y, x, ts = _xy(df, spec)
    kind = str(spec.get("model") or "rf")
    if kind not in ("rf", "gb"):
        raise OESError("The model must be rf (random forest) or gb (gradient boosting).")
    share = float_option(spec, "test_share", 0.25, "Test share")
    if not 0.05 <= share <= 0.5:
        raise OESError("The test share must be between 0.05 and 0.5.")
    reps = int_option(spec, "repeats", 20, 5, 200, "Permutation repeats"); top = int_option(spec, "top", 3, 1, 6, "Number of variables to profile")
    if len(x) < 2:
        raise OESError("Choose at least two predictors.")
    top = min(top, len(x))
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    X, Y = d[x].astype(float).to_numpy(), d[y].astype(float).to_numpy(); tr, te = _split(len(d), ts, share)
    m = (RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=1) if kind == "rf"
         else GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=0)).fit(X[tr], Y[tr])
    def r2(Xt): e = Y[te] - m.predict(Xt); return 1 - float(e @ e) / float(((Y[te] - Y[tr].mean()) ** 2).sum())
    base = r2(X[te]); rng = np.random.default_rng(0); imp = np.zeros((len(x), reps))
    nt = len(te); den = float(((Y[te] - Y[tr].mean()) ** 2).sum())
    for j in range(len(x)):
        Zs = np.tile(X[te], (reps, 1))                                     # all repetitions in one prediction call
        for r in range(reps):
            Zs[r * nt:(r + 1) * nt, j] = rng.permutation(X[te][:, j])
        pr = m.predict(Zs).reshape(reps, nt); imp[j] = base - (1 - ((Y[te][None, :] - pr) ** 2).sum(1) / den)
    mi, si = imp.mean(1), imp.std(1, ddof=1); order = np.argsort(-mi)
    import statsmodels.api as sm
    Z = (X[tr] - X[tr].mean(0)) / np.where(X[tr].std(0) == 0, 1, X[tr].std(0)); ols = sm.OLS(Y[tr], sm.add_constant(Z)).fit(); beta = ols.params[1:]
    rho = float(stats.spearmanr(mi, np.abs(beta))[0]) if len(x) > 2 else float("nan")
    tabs = [table(f"Permutation importance on the test set ({'random forest' if kind == 'rf' else 'gradient boosting'}; test R-squared {base:.4f})", ["Predictor", "Importance (fall in test R-squared)", "Std. dev. over repeats", "Share of total", "Standardised OLS coefficient"],
                  [[x[j], f(mi[j], 4), f(si[j], 4), f(float(max(mi[j], 0) / max(mi.clip(0).sum(), 1e-12)), 4), f(float(beta[j]), 4)] for j in order],
                  f"Each predictor is shuffled on the test set {reps} times; importance is how much the fit falls. Correlated predictors share importance, so a small value does not mean a variable is unrelated to the outcome.")]
    sub = rng.choice(len(tr), size=min(80, len(tr)), replace=False); Xs = X[tr][sub]; charts = []; prof = []; pdc = {}
    for j in order[:top]:
        grid = np.quantile(X[tr][:, j], np.linspace(0.05, 0.95, 10)); grid = np.unique(grid); pdv = _pd_curve(m, Xs, j, grid); pdc[j] = (grid, pdv)
        slope = float(np.polyfit(grid, pdv, 1)[0]) if len(grid) > 2 else float("nan")
        lin = float(np.corrcoef(grid, pdv)[0, 1] ** 2) if len(grid) > 2 and pdv.std() > 0 else float("nan")
        prof.append([x[j], f(float(grid[0]), 3), f(float(grid[-1]), 3), f(float(pdv[0]), 4), f(float(pdv[-1]), 4), f(float(pdv.max() - pdv.min()), 4), f(lin, 3)])
        charts.append(chart(f"Partial dependence of {y} on {x[j]}", [float(v) for v in grid], [("Partial dependence", [float(v) for v in pdv])], x[j], y, kind="line"))
    tabs.append(table("Partial dependence profiles (5th to 95th percentile of each predictor)", ["Predictor", "From", "To", "Prediction at start", "Prediction at end", "Range", "R-squared of a straight line"], prof,
                      "The average prediction when the predictor is set to each value and the others keep their observed values. An R-squared of a straight line near 1 means the effect is close to linear."))
    inter = []
    if top >= 2:
        idx = list(order[:top]); pj = {}
        for j in idx:
            pj[j] = _pd_at(m, Xs, [j])
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                j, k = idx[a], idx[b]; pjk = _pd_at(m, Xs, [j, k])
                c = lambda v: v - v.mean(); den = float(np.sum(c(pjk) ** 2))
                h2 = float(np.sum((c(pjk) - c(pj[j]) - c(pj[k])) ** 2) / den) if den > 0 else float("nan")
                inter.append([f"{x[j]} and {x[k]}", f(h2, 4), f(float(np.sqrt(max(h2, 0))), 4)])
        tabs.append(table("Interaction strength between the leading predictors (Friedman's H statistic)", ["Pair", "H-squared", "H"], inter,
                          f"The share of the variance of the joint partial dependence that is not explained by the two separate partial dependences, on {len(Xs)} training observations. 0 means the two effects add up."))
    summary = [stat("Observations", len(d)), stat("Predictors", len(x)), stat("Test R-squared", base, 4), stat("Most important predictor", x[int(order[0])]), stat("Rank correlation with |OLS coefficient|", rho, 3)]
    return result("interpret", f"Interpretation of the {'forest' if kind == 'rf' else 'boosting'} model for {y}", len(d), summary, tabs, charts,
                  ["Importance, partial dependence and interaction strength describe the fitted model, not the economy. They are causal only if the model is a causal one.", "Permutation importance is computed on data the model has not seen, so it does not reward overfitting."])


# ------------------------------------------------------------------------------------------------------------------
# Stacking: the super learner (van der Laan, Polley and Hubbard 2007)
# ------------------------------------------------------------------------------------------------------------------
def _base_learners():
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {"OLS": make_pipeline(StandardScaler(), LinearRegression()),
            "LASSO": make_pipeline(StandardScaler(), LassoCV(cv=5, n_alphas=40, max_iter=5000, tol=1e-3, random_state=0)),
            "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
            "Random forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=1),
            "Gradient boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=0),
            "k nearest neighbours": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10, weights="distance"))}


def _nnls_weights(P, y):
    from scipy.optimize import nnls
    w, _ = nnls(P, y); s = w.sum()
    return w / s if s > 0 else np.ones(P.shape[1]) / P.shape[1]


def stacking(df, spec):
    d, y, x, ts = _xy(df, spec)
    share = float_option(spec, "test_share", 0.25, "Test share")
    if not 0.05 <= share <= 0.5:
        raise OESError("The test share must be between 0.05 and 0.5.")
    K = int_option(spec, "folds", 5, 3, 10, "Number of folds")
    from sklearn.base import clone
    from sklearn.model_selection import KFold
    X, Y = d[x].astype(float).to_numpy(), d[y].astype(float).to_numpy(); tr, te = _split(len(d), ts, share)
    if len(tr) < 5 * K:
        raise OESError("The training sample is too small for this number of folds.")
    L = _base_learners(); names = list(L); P = np.zeros((len(tr), len(L)))
    for a, b in KFold(K, shuffle=not ts, random_state=None if ts else 0).split(tr):
        for j, nm in enumerate(names):
            P[b, j] = clone(L[nm]).fit(X[tr][a], Y[tr][a]).predict(X[tr][b])
    cvr = np.sqrt(((Y[tr][:, None] - P) ** 2).mean(0)); w = _nnls_weights(P, Y[tr])
    cv_ens = float(np.sqrt(((Y[tr] - P @ w) ** 2).mean())); best = int(np.argmin(cvr))
    T = np.zeros((len(te), len(L)))
    for j, nm in enumerate(names):
        T[:, j] = clone(L[nm]).fit(X[tr], Y[tr]).predict(X[te])
    den = float(((Y[te] - Y[tr].mean()) ** 2).sum()); rm = lambda p: float(np.sqrt(np.mean((Y[te] - p) ** 2))); r2 = lambda p: 1 - float(((Y[te] - p) ** 2).sum()) / den
    ens = T @ w
    rows = [[nm, f(float(w[j]), 4), f(float(cvr[j]), 4), f(rm(T[:, j]), 4), f(r2(T[:, j]), 4)] for j, nm in enumerate(names)]
    rows += [["Super learner (weighted average)", "1.0000", f(cv_ens, 4), f(rm(ens), 4), f(r2(ens), 4)], [f"Discrete super learner ({names[best]})", "", f(float(cvr[best]), 4), f(rm(T[:, best]), 4), f(r2(T[:, best]), 4)],
             ["Equal weights", f(1 / len(names), 4), "", f(rm(T.mean(1)), 4), f(r2(T.mean(1)), 4)]]
    tab = table("Learner weights, cross-validated risk and test performance", ["Learner", "Weight", f"{K}-fold CV RMSE (training)", "Test RMSE", "Test R-squared"], rows,
                "Weights solve a non-negative least-squares problem on the cross-validated predictions and are rescaled to sum to one. The CV RMSE of the weighted average is optimistic because the weights were fitted on the same predictions; the test columns are not.")
    corr = np.corrcoef(P.T); cr = [[names[i]] + [f(float(corr[i, j]), 3) for j in range(len(names))] for i in range(len(names))]
    ctab = table("Correlation of the cross-validated predictions", ["Learner"] + names, cr, "Highly correlated learners are substitutes: the weights are shared among them, and adding a similar learner changes little.")
    summary = [stat("Observations", len(d)), stat("Learners", len(L)), stat("Test RMSE, super learner", rm(ens), 4), stat("Test RMSE, best single learner (chosen on the test set)", min(rm(T[:, j]) for j in range(len(names))), 4), stat("Test R-squared, super learner", r2(ens), 4), stat("Largest weight", names[int(np.argmax(w))])]
    ch = [chart("Super learner weights", names, [("Weight", [float(v) for v in w])], "Learner", "Weight", kind="bar")]
    return result("stacking", f"Stacking (super learner) for {y}", len(d), summary, [tab, ctab], ch, ["The super learner is asymptotically as good as the best weighted combination of its learners (an oracle inequality), so it protects against picking the wrong model, not against omitting a useful learner.", "Report the test results, not the cross-validated risk of the combination."])


# ------------------------------------------------------------------------------------------------------------------
# Heterogeneous treatment effects in an experiment: BLP, GATES and CLAN (Chernozhukov, Demirer, Duflo and Fernandez-Val 2018)
# ------------------------------------------------------------------------------------------------------------------
def _wls_hc(Z, y, w):
    sw = np.sqrt(w); Zw, yw = Z * sw[:, None], y * sw
    A = np.linalg.pinv(Zw.T @ Zw); b = A @ (Zw.T @ yw); e = yw - Zw @ b; n, k = Zw.shape
    V = A @ (Zw.T * (e ** 2)) @ Zw @ A * n / max(n - k, 1)
    return b, np.sqrt(np.clip(np.diag(V), 0, None)), V


def hetero(df, spec):
    y = need(spec, "y", "outcome"); t = need(spec, "treat", "treatment (0/1)"); x = as_list(need(spec, "x", "covariates")); distinct([y], [t], x)
    G = int_option(spec, "groups", 4, 2, 10, "Number of groups"); S = int_option(spec, "splits", 20, 5, 200, "Number of data splits"); seed = int_option(spec, "seed", 0, 0, 10 ** 6, "Random seed")
    d = df[[c for c in [y, t] + x if c in df.columns]]
    miss = [c for c in [y, t] + x if c not in df.columns]
    if miss:
        raise OESError("Column(s) not found in the data: " + ", ".join(miss))
    d = d.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True); n = len(d)
    if d[t].nunique() != 2 or not set(np.unique(d[t])) <= {0.0, 1.0}:
        raise OESError("The treatment must be a 0/1 indicator with both values present.")
    if n < 20 * G:
        raise OESError(f"With {G} groups the data need at least {20 * G} observations; {n} were found.")
    from sklearn.ensemble import RandomForestRegressor
    Y, D, X = d[y].to_numpy(float), d[t].to_numpy(float), d[x].to_numpy(float); p = float(D.mean()); alpha = 0.025
    rng = np.random.default_rng(seed); z = stats.norm.ppf(1 - alpha / 2)
    blp = np.zeros((S, 2, 3)); gat = np.zeros((S, G + 1, 3)); clan = np.zeros((S, len(x), 3)); cl_means = np.zeros((S, len(x), 2))      # estimate, se, p
    for s in range(S):
        idx = rng.permutation(n); half = n // 2; A, Mn = idx[:half], idx[half:]
        # stratify so that both arms are present in each half
        if D[A].sum() < 5 or (1 - D[A]).sum() < 5:
            A, Mn = idx[::2], idx[1::2]
        rf = lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.5, random_state=int(rng.integers(1 << 30)), n_jobs=1)
        m1 = rf().fit(X[A][D[A] == 1], Y[A][D[A] == 1]); m0 = rf().fit(X[A][D[A] == 0], Y[A][D[A] == 0])
        B = m0.predict(X[Mn]); Sx = m1.predict(X[Mn]) - B
        if Sx.std() < 1e-10:
            Sx = Sx + rng.normal(scale=1e-6, size=len(Sx))
        Dm, Ym = D[Mn], Y[Mn]; dd = Dm - p; w = np.full(len(Mn), 1 / (p * (1 - p)))
        Z = np.column_stack([np.ones(len(Mn)), B, Sx, dd, dd * (Sx - Sx.mean())]); b, se, _ = _wls_hc(Z, Ym, w)
        for k, j in enumerate((3, 4)):
            blp[s, k] = [b[j], se[j], 2 * stats.norm.sf(abs(b[j] / se[j])) if se[j] > 0 else 1.0]
        ranks = stats.rankdata(Sx, method="ordinal"); grp = np.minimum(((ranks - 1) * G // len(Sx)).astype(int), G - 1)
        Gm = np.column_stack([(grp == k).astype(float) * dd for k in range(G)]); Z2 = np.column_stack([np.ones(len(Mn)), B, Sx, Gm]); b2, se2, V2 = _wls_hc(Z2, Ym, w)
        for k in range(G):
            gat[s, k] = [b2[3 + k], se2[3 + k], 2 * stats.norm.sf(abs(b2[3 + k] / se2[3 + k])) if se2[3 + k] > 0 else 1.0]
        c = np.zeros(len(b2)); c[3 + G - 1], c[3] = 1, -1; dif = float(c @ b2); sed = float(np.sqrt(max(c @ V2 @ c, 0)))
        gat[s, G] = [dif, sed, 2 * stats.norm.sf(abs(dif / sed)) if sed > 0 else 1.0]
        for j in range(len(x)):
            hi, lo = X[Mn][grp == G - 1, j], X[Mn][grp == 0, j]; df_ = float(hi.mean() - lo.mean()); sd = float(np.sqrt(hi.var(ddof=1) / len(hi) + lo.var(ddof=1) / len(lo)))
            clan[s, j] = [df_, sd, 2 * stats.norm.sf(abs(df_ / sd)) if sd > 0 else 1.0]; cl_means[s, j] = [hi.mean(), lo.mean()]
    def agg(a):  # median estimate, median se, adjusted p-value (twice the median p, capped at one)
        m = np.median(a[..., 0], axis=0); se = np.median(a[..., 1], axis=0); pv = np.minimum(1.0, 2 * np.median(a[..., 2], axis=0)); return m, se, pv
    m1_, s1_, p1_ = agg(blp); mg, sg, pg = agg(gat); mc, sc, pc = agg(clan)
    ci = lambda m, s_: (f(float(m - z * s_), 4), f(float(m + z * s_), 4))
    nm = ["Average treatment effect (beta1)", "Heterogeneity loading (beta2)"]
    btab = table(f"Best linear predictor of the effect of {t} on {y}", ["Coefficient", "Estimate", "Std. error", "95% CI low", "95% CI high", "Adjusted p-value"],
                 [[nm[k], f(float(m1_[k]), 4), f(float(s1_[k]), 4), *ci(m1_[k], s1_[k]), f(float(p1_[k]), 4)] for k in range(2)],
                 f"Medians over {S} random splits of the data; the p-value is twice the median p-value (Chernozhukov et al. 2018). beta2 = 1 means the machine-learning proxy of the effect is perfect, and beta2 = 0 means it carries no information about heterogeneity.")
    glab = [f"Group {k + 1} (least affected)" if k == 0 else (f"Group {k + 1} (most affected)" if k == G - 1 else f"Group {k + 1}") for k in range(G)] + [f"Group {G} minus group 1"]
    gtab = table("Sorted group average treatment effects (GATES)", ["Group", "Effect", "Std. error", "95% CI low", "95% CI high", "Adjusted p-value"],
                 [[glab[k], f(float(mg[k]), 4), f(float(sg[k]), 4), *ci(mg[k], sg[k]), f(float(pg[k]), 4)] for k in range(G + 1)],
                 f"Groups are quantiles of the predicted effect, formed in the estimation half of each split; medians over {S} splits.")
    order = np.argsort(pc)
    ctab = table(f"Classification analysis (CLAN): covariate means in the most and least affected groups", ["Covariate", "Group " + str(G) + " mean", "Group 1 mean", "Difference", "Std. error", "Adjusted p-value"],
                 [[x[j], f(float(np.median(cl_means[:, j, 0])), 4), f(float(np.median(cl_means[:, j, 1])), 4), f(float(mc[j]), 4), f(float(sc[j]), 4), f(float(pc[j]), 4)] for j in order],
                 "Which covariates separate the people with the largest predicted effects from those with the smallest. Medians over splits; p-values are not adjusted for the number of covariates.")
    charts = [chart("Group average treatment effects", list(range(1, G + 1)), [("GATES", [float(v) for v in mg[:G]]), ("Average effect", [float(m1_[0])] * G)], "Group (by predicted effect)", "Effect", kind="line")]
    summary = [stat("Observations", n), stat("Treated share", p, 4), stat("Average effect", float(m1_[0]), 4), stat("Heterogeneity loading", float(m1_[1]), 4), stat("p-value of heterogeneity loading", float(p1_[1]), 4), stat(f"Group {G} minus group 1", float(mg[G]), 4), stat("p-value of the difference", float(pg[G]), 4)]
    return result("hetero", f"Heterogeneity of the effect of {t} on {y}", n, summary, [btab, gtab, ctab], charts,
                  ["The treatment must be randomised, or ignorable with a known propensity score: the estimates here assume the treatment share is the same for everyone.", "The groups are formed from a predicted effect, so the results are about the proxy; the data splitting makes the inference valid whatever the quality of the proxy."])


# ------------------------------------------------------------------------------------------------------------------
# Double-selection LASSO (Belloni, Chernozhukov and Hansen 2014) with the plug-in penalty of the "rigorous LASSO" (hdm package)
# ------------------------------------------------------------------------------------------------------------------
def _rlasso(y, X, c=1.1, K=15):
    """Indices of the controls selected by the rigorous LASSO: penalty 2 c sqrt(n) Phi^-1(1 - g / (2 p)) with g = 0.1 / log(n), penalty loadings
    from iterated post-LASSO residuals (heteroskedasticity-robust), as in the R package hdm."""
    from sklearn.linear_model import Lasso
    n, p = X.shape
    gam = 0.1 / np.log(n); lam0 = 2 * c * np.sqrt(n) * stats.norm.ppf(1 - gam / (2 * p))
    Xc, yc = X - X.mean(0), y - y.mean()
    sd = Xc.std(0); ok = np.where(sd > 1e-12)[0]
    if len(ok) == 0:
        return np.array([], int)
    cor = np.array([abs(np.corrcoef(Xc[:, j], yc)[0, 1]) if sd[j] > 1e-12 else 0.0 for j in range(p)])
    Z = np.column_stack([np.ones(n), Xc[:, np.argsort(-cor)[:5]]]); e = yc - Z @ np.linalg.lstsq(Z, yc, rcond=None)[0]
    ups = np.sqrt((Xc ** 2 * e[:, None] ** 2).mean(0)); ups[ups < 1e-12] = 1e-12; old = None; sel = np.array([], int)
    for _ in range(K):
        m = Lasso(alpha=lam0 / (2 * n), fit_intercept=False, max_iter=100000, tol=1e-10).fit(Xc / ups, yc); sel = np.where(np.abs(m.coef_) > 1e-10)[0]
        if len(sel) == 0:
            e = yc
        else:
            Zs = Xc[:, sel]; e = (yc - Zs @ np.linalg.lstsq(Zs, yc, rcond=None)[0]) * np.sqrt(n / max(n - len(sel), 1))
        if old is not None and set(sel) == set(old):
            break
        ups = np.sqrt((Xc ** 2 * e[:, None] ** 2).mean(0)); ups[ups < 1e-12] = 1e-12; old = sel
    return sel


def _ols_effect(y, d, X, cols):
    """OLS of y on d and the controls in cols; heteroskedasticity-robust standard error with the degrees-of-freedom factor n / (n - k) of the hdm package (k = target plus controls)."""
    n = len(y); Z = np.column_stack([np.ones(n), d] + ([X[:, cols]] if len(cols) else []))
    A = np.linalg.pinv(Z.T @ Z); b = A @ (Z.T @ y); e = y - Z @ b; V = A @ (Z.T * e ** 2) @ Z @ A * n / max(n - 1 - len(cols), 1)
    return float(b[1]), float(np.sqrt(max(V[1, 1], 0)))


def dsel(df, spec):
    y = need(spec, "y", "outcome"); t = need(spec, "treat", "treatment (target) variable"); x = as_list(need(spec, "x", "potential controls")); distinct([y], [t], x)
    miss = [c for c in [y, t] + x if c not in df.columns]
    if miss:
        raise OESError("Column(s) not found in the data: " + ", ".join(miss))
    d = df[[y, t] + x].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True); n = len(d)
    if n < 30:
        raise OESError(f"Double selection needs at least 30 complete observations; {n} were found.")
    cc = float_option(spec, "c", 1.1, "Penalty constant c")
    if not 0.5 <= cc <= 3:
        raise OESError("The penalty constant c must be between 0.5 and 3 (1.1 is the standard choice).")
    Y, D, X = d[y].to_numpy(float), d[t].to_numpy(float), d[x].to_numpy(float); p = len(x)
    sy, sd_ = _rlasso(Y, X, cc), _rlasso(D, X, cc); un = sorted(set(sy) | set(sd_))
    from sklearn.linear_model import LassoCV
    def cvsel(v):
        m = LassoCV(cv=5, n_alphas=60, max_iter=20000, tol=1e-4, random_state=0).fit((X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1), v - v.mean()); return np.where(np.abs(m.coef_) > 1e-10)[0]
    cy, cd = cvsel(Y), cvsel(D); cun = sorted(set(cy) | set(cd))
    rows = []
    def add(name, cols, feasible=True):
        if not feasible:
            rows.append([name, "n/a", "", "", "", "too many controls"]); return None
        b, s = _ols_effect(Y, D, X, list(cols)); rows.append([name, f(b, 5), f(s, 5), f(b - 1.96 * s, 5), f(b + 1.96 * s, 5), len(cols)]); return b, s
    add("No controls", []); add("All controls (OLS)", list(range(p)), feasible=p + 2 < n - 2)
    add("Single selection: controls chosen for the outcome only", list(sy)); add("Single selection: controls chosen for the treatment only", list(sd_))
    ds = add("Double selection (plug-in LASSO)", un); add("Double selection with cross-validated LASSO (for comparison)", cun)
    tab = table(f"Effect of {t} on {y}", ["Estimator", "Estimate", "Robust std. error", "95% CI low", "95% CI high", "Controls"], rows,
                "Robust standard errors with the n/(n - k) factor of the R package hdm. The plug-in penalty has a theoretical justification; the cross-validated row is shown because it is common practice, but its selection is not covered by the theory.")
    sel_rows = [[x[j], "yes" if j in sy else "", "yes" if j in sd_ else "", "yes" if j in un else ""] for j in un]
    stab = table("Selected controls (plug-in penalty)", ["Control", "Selected for the outcome", "Selected for the treatment", "In the final regression"], sel_rows or [["none", "", "", ""]],
                 f"{len(sy)} controls selected in the outcome equation, {len(sd_)} in the treatment equation, {len(un)} in the union out of {p}.")
    b0, s0 = ds
    summary = [stat("Observations", n), stat("Potential controls", p), stat("Selected for the outcome", len(sy)), stat("Selected for the treatment", len(sd_)), stat("Union (final controls)", len(un)),
               stat("Double-selection estimate", b0, 5), stat("Robust std. error", s0, 5), stat("t-statistic", b0 / s0 if s0 > 0 else float("nan"), 3), stat("p-value", float(2 * stats.norm.sf(abs(b0 / s0))) if s0 > 0 else float("nan"), 4)]
    ch = [chart("Estimates and 95% intervals", [r[0].split(" (")[0][:28] for r in rows if r[1] != "n/a"], [("Estimate", [float(r[1]) for r in rows if r[1] != "n/a"])], "Estimator", "Estimate", kind="bar")]
    return result("dsel", f"Double-selection LASSO: effect of {t} on {y}", n, summary, [tab, stab], ch,
                  ["Selecting controls for the outcome alone omits controls that matter for the treatment and biases the estimate; the union of the two selections protects against this (Belloni, Chernozhukov and Hansen 2014).", "The theory needs approximately sparse effects of the controls on both the outcome and the treatment, and a treatment that is not too well predicted by the controls."])
