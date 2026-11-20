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
