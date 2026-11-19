"""Machine-learning models of the Machine Learning Research Guide.

Prediction: LASSO, ridge, elastic net, random forest, XGBoost, support vector regression and kernel-based regularized
least squares (KRLS), each with cross-validated tuning, a hold-out test set and an OLS benchmark. Causal: double machine learning (partially linear, interactive and IV models) using the
guide's cross-fitting routine (_ml_core.crossfit), and a causal forest built as a doubly robust forest (DR-learner).
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import _ml_core as M
from .core import OESError, as_list, chart, coef_table, distinct, f, float_option, int_option, need, numeric, result, stat, table, tests_table, test_row, time_order


def _data(df, spec, cols):
    tcol = spec.get("time")
    if tcol:
        df = time_order(df, tcol)
    d = numeric(df, cols).reset_index(drop=True)
    if len(d) < 30:
        raise OESError(f"Machine-learning models need at least 30 complete observations; {len(d)} were found.")
    return d, bool(tcol)


class XGBoostRegressor:
    """Minimal scikit-learn-compatible wrapper around xgboost.train (XGBRegressor is not compatible with the
    scikit-learn 1.6 shipped with Pyodide 0.27)."""

    def __init__(self, n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                 min_child_weight=1.0, reg_lambda=1.0, random_state=0):
        self.n_estimators, self.learning_rate, self.max_depth = n_estimators, learning_rate, max_depth
        self.subsample, self.colsample_bytree, self.min_child_weight = subsample, colsample_bytree, min_child_weight
        self.reg_lambda, self.random_state = reg_lambda, random_state

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in ("n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree",
                                              "min_child_weight", "reg_lambda", "random_state")}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        import xgboost
        params = {"objective": "reg:squarederror", "eta": self.learning_rate, "max_depth": int(self.max_depth),
                  "subsample": self.subsample, "colsample_bytree": self.colsample_bytree, "min_child_weight": self.min_child_weight,
                  "lambda": self.reg_lambda, "seed": self.random_state, "nthread": 1, "tree_method": "hist"}
        self.booster_ = xgboost.train(params, xgboost.DMatrix(np.asarray(X, float), label=np.asarray(y, float)), int(self.n_estimators))
        return self

    def predict(self, X):
        import xgboost
        return self.booster_.predict(xgboost.DMatrix(np.asarray(X, float)))

    def score(self, X, y):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))


class KRLSRegressor:
    """Kernel-based regularized least squares (Hainmueller and Hazlett 2014), scikit-learn compatible.

    Predictors and outcome are standardised, the Gaussian kernel is k(x, z) = exp(-|x - z|^2 / sigma2) with sigma2 = number of
    predictors by default, the weights solve (K + lambda I) c = y, and lambda is chosen by leave-one-out cross-validation
    (golden-section search over the bounds of the KRLS package). Pointwise marginal effects, average marginal effects with their
    standard errors (and first differences for 0/1 predictors) follow the formulas of the paper."""

    def __init__(self, sigma=None, lam=None):
        self.sigma, self.lam = sigma, lam

    def get_params(self, deep=True):
        return {"sigma": self.sigma, "lam": self.lam}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    @staticmethod
    def _kernel(A, B, sigma2):
        d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-np.maximum(d2, 0) / sigma2)

    @staticmethod
    def _loo_loss(lam, d, U, Uty):
        inv = 1.0 / (d + lam)
        c = U @ (Uty * inv)
        return float(np.sum((c / ((U ** 2) @ inv)) ** 2))

    @staticmethod
    def _bounds(d, n):
        """Bounds of the lambda search used by the KRLS package: the upper bound keeps the effective degrees of freedom at one or more, the lower
        bound keeps them at or below the number of eigenvalues above 1/1000 of the largest."""
        up = float(n)
        while np.sum(d / (d + up)) < 1 and up > 1e-8:
            up -= 1.0
            if up <= 0:
                up = 1e-3
                break
        dd = np.sort(d)[::-1]
        q = int(np.argmin(np.abs(dd - dd[0] / 1000.0))) + 1
        lo = np.finfo(float).eps
        if np.sum(d / (d + lo)) > q:
            a, b = lo, up
            for _ in range(80):
                mid = 0.5 * (a + b)
                a, b = (mid, b) if np.sum(d / (d + mid)) > q else (a, mid)
            lo = b
        return min(lo, up * 0.5), up

    @staticmethod
    def _lambda_search(d, U, Uty, n):
        """Golden-section search for the leave-one-out minimiser between the bounds of _bounds."""
        lo, up = KRLSRegressor._bounds(d, n)
        tol = 1e-3 * n
        x1, x2 = lo + 0.381966 * (up - lo), up - 0.381966 * (up - lo)
        s1, s2 = KRLSRegressor._loo_loss(x1, d, U, Uty), KRLSRegressor._loo_loss(x2, d, U, Uty)
        for _ in range(100):
            if abs(s1 - s2) <= tol:
                break
            if s1 < s2:
                up, x2, s2 = x2, x1, s1
                x1 = lo + 0.381966 * (up - lo)
                s1 = KRLSRegressor._loo_loss(x1, d, U, Uty)
            else:
                lo, x1, s1 = x1, x2, s2
                x2 = up - 0.381966 * (up - lo)
                s2 = KRLSRegressor._loo_loss(x2, d, U, Uty)
        return x1 if s1 < s2 else x2

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, p = X.shape
        self.xm_, self.xs_ = X.mean(0), X.std(0, ddof=1)
        self.xs_[self.xs_ == 0] = 1.0
        self.ym_, self.ys_ = float(y.mean()), float(y.std(ddof=1)) or 1.0
        self.Z_ = (X - self.xm_) / self.xs_
        t = (y - self.ym_) / self.ys_
        self.sigma2_ = float(p) if self.sigma in (None, "") else float(self.sigma)
        K = self._kernel(self.Z_, self.Z_, self.sigma2_)
        d, U = np.linalg.eigh(K)
        d = np.maximum(d, 0.0)
        Uty = U.T @ t
        self.lambda_ = float(self.lam) if self.lam not in (None, "") else float(self._lambda_search(d, U, Uty, n))
        self.coef_ = U @ (Uty / (d + self.lambda_))
        self.d_, self.U_, self.K_ = d, U, K
        fit = K @ self.coef_
        self.sigma_eps2_ = float(np.mean((t - fit) ** 2))             # residual variance in standardised units
        self.df_eff_ = float(np.sum(d / (d + self.lambda_)))
        self.r2_ = 1 - float(np.sum((t - fit) ** 2)) / float(np.sum(t ** 2))
        self.binary_ = np.array([set(np.unique(X[:, j])) <= {0.0, 1.0} and len(np.unique(X[:, j])) == 2 for j in range(p)])
        return self

    def predict(self, X):
        Z = (np.asarray(X, float) - self.xm_) / self.xs_
        return self.ym_ + self.ys_ * (self._kernel(Z, self.Z_, self.sigma2_) @ self.coef_)

    def score(self, X, y):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))

    def _quad(self, w):
        """w' Var(c) w with Var(c) = sigma_eps2 U diag((d + lambda)^-2) U' (Hainmueller and Hazlett 2014)."""
        v = (self.U_.T @ w) / (self.d_ + self.lambda_)
        return self.sigma_eps2_ * float(v @ v)

    def effects(self):
        """Pointwise marginal effects (original units: derivatives, or first differences for 0/1 predictors), their average and its standard error."""
        Z, n, p = self.Z_, *self.Z_.shape
        pointwise, avg, se = np.zeros((n, p)), np.zeros(p), np.zeros(p)
        for j in range(p):
            if self.binary_[j]:
                Z1, Z0 = Z.copy(), Z.copy()
                Z1[:, j], Z0[:, j] = (1.0 - self.xm_[j]) / self.xs_[j], (0.0 - self.xm_[j]) / self.xs_[j]
                D = self._kernel(Z1, Z, self.sigma2_) - self._kernel(Z0, Z, self.sigma2_)
                scale = self.ys_
            else:
                D = (-2.0 / self.sigma2_) * self.K_ * (Z[:, j][:, None] - Z[:, j][None, :])
                scale = self.ys_ / self.xs_[j]
            pointwise[:, j] = scale * (D @ self.coef_)
            avg[j] = pointwise[:, j].mean()
            se[j] = scale * np.sqrt(self._quad(D.sum(0)) / n ** 2)
        return dict(pointwise=pointwise, avg=avg, se=se)


def _svr(p, kernel="rbf", epsilon=0.1):
    """Support vector regression with the outcome and the predictors standardised; C (and gamma) chosen by 3-fold cross-validation."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    grid = {"svr__C": [0.5, 5.0, 50.0]}
    if kernel == "rbf":
        grid["svr__gamma"] = [0.25 / p, 1.0 / p, 4.0 / p]
    inner = GridSearchCV(make_pipeline(StandardScaler(), SVR(kernel=kernel, epsilon=epsilon)), grid,
                         cv=KFold(3, shuffle=True, random_state=0), scoring="neg_mean_squared_error", n_jobs=1)
    return TransformedTargetRegressor(regressor=inner, transformer=StandardScaler())


def _learners(binary, trees):
    """Cross-fitting learners. Same specification as the guide's, with lighter settings so they run in the browser."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV, LogisticRegressionCV
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    rf = dict(min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=1, n_estimators=trees)
    return {"lasso": dict(reg=make_pipeline(StandardScaler(), LassoCV(cv=5, n_alphas=50, random_state=0, max_iter=5000, tol=1e-3)),
                          clf=make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, penalty="l1", solver="liblinear", Cs=10, max_iter=1000))),
            "rf": dict(reg=RandomForestRegressor(**rf), clf=RandomForestClassifier(**rf))}


def _split(n, ts, share, seed=0):
    k = max(5, int(round(n * share)))
    if ts:
        return np.arange(n - k), np.arange(n - k, n)
    idx = np.random.default_rng(seed).permutation(n)
    return np.sort(idx[k:]), np.sort(idx[:k])


def _fit_stats(model, X, y, tr, te, ts):
    m = model.fit(X.iloc[tr], y.iloc[tr])
    pred = m.predict(X.iloc[te])
    resid = y.iloc[te].to_numpy() - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    r2 = 1 - float(resid @ resid) / float(((y.iloc[te] - y.iloc[tr].mean()) ** 2).sum())
    return m, pred, rmse, r2


def _predict(df, spec, kind):
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance
    y = need(spec, "y", "outcome to predict")
    x = as_list(need(spec, "x", "predictors"))
    distinct([y], x)
    d, ts = _data(df, spec, [y] + x)
    share = float_option(spec, "test_share", 0.2, "Test share")
    if not 0.05 <= share <= 0.5:
        raise OESError("The test share must be between 0.05 and 0.5.")
    X, Y = d[x].astype(float), d[y].astype(float)
    tr, te = _split(len(d), ts, share)
    models = M.models()
    if kind == "xgb":
        models["xgb"] = XGBoostRegressor(n_estimators=int_option(spec, "n_estimators", 400, 50, 3000, "Number of trees"),
                                         learning_rate=float_option(spec, "learning_rate", 0.05, "Learning rate"),
                                         max_depth=int_option(spec, "max_depth", 4, 1, 12, "Maximum depth"))
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        models["rf"] = RandomForestRegressor(n_estimators=int_option(spec, "n_estimators", 300, 50, 2000, "Number of trees"),
                                             min_samples_leaf=int_option(spec, "min_leaf", 5, 1, 100, "Minimum leaf size"),
                                             max_features=0.5, random_state=0, n_jobs=1)
    if kind == "svr":
        kernel = str(spec.get("kernel") or "rbf")
        if kernel not in ("rbf", "linear"):
            raise OESError("The kernel must be rbf or linear.")
        eps = float_option(spec, "epsilon", 0.1, "Epsilon")
        if not 0 < eps <= 1:
            raise OESError("Epsilon (in standard deviations of the outcome) must be between 0 and 1.")
        models["svr"] = _svr(len(x), kernel, eps)
    if kind == "krls":
        if len(tr) > 2000:
            raise OESError(f"KRLS builds an n x n kernel matrix and the training sample has {len(tr)} observations; above 2,000 it is too slow in the browser. "
                           "Use a smaller sample or support vector regression.")
        sg, lm = float_option(spec, "sigma", None, "Kernel bandwidth"), float_option(spec, "lambda", None, "Lambda")
        if (sg is not None and sg <= 0) or (lm is not None and lm <= 0):
            raise OESError("The kernel bandwidth and lambda must be positive.")
        models["krls"] = KRLSRegressor(sigma=sg, lam=lm)
    main = models[kind]
    m, pred, rmse, r2 = _fit_stats(clone(main), X, Y, tr, te, ts)
    cv = M.cv_rmse(clone(main), X.iloc[tr], Y.iloc[tr], ts)
    bench = []
    for name, key in (("OLS", "ols"), ("LASSO", "lasso"), ("Random forest", "rf")):
        if key == kind:
            continue
        _, _, r_, q_ = _fit_stats(clone(models[key]), X, Y, tr, te, ts)
        bench.append([name, f(r_, 4), f(q_, 4)])
    label = {"lasso": "LASSO", "ridge": "Ridge", "enet": "Elastic net", "rf": "Random forest", "xgb": "XGBoost",
             "svr": "Support vector regression", "krls": "KRLS"}[kind]
    tabs = [table("Out-of-sample accuracy", ["Model", "Test RMSE", "Test R-squared"], [[label, f(rmse, 4), f(r2, 4)]] + bench,
                  ("Test set: the last " if ts else "Test set: a random ") + f"{int(round(share * 100))}% of the observations"
                  + (" (time order kept; validation is rolling-origin)." if ts else "."))]
    summary = [stat("Observations", len(d)), stat("Predictors", len(x)), stat("CV RMSE (training)", cv), stat("Test RMSE", rmse), stat("Test R-squared", r2)]
    notes = []
    if kind in ("lasso", "ridge", "enet"):
        est = m[-1]
        coef = est.coef_
        alpha = float(est.alpha_)
        rows = [[v, f(c), "kept" if abs(c) > 1e-10 else "dropped"] for v, c in sorted(zip(x, coef), key=lambda z: -abs(z[1]))]
        tabs.append(table(f"{label} coefficients (standardised predictors)", ["Predictor", "Coefficient", "Status"], rows,
                          "Coefficients are per standard deviation of each predictor. Penalised estimates have no standard errors."))
        summary.append(stat("Penalty (alpha)", alpha, 6))
        if kind == "enet":
            summary.append(stat("L1 ratio", float(est.l1_ratio_), 2))
        if kind == "lasso":
            keep, a1 = M.lasso_1se(X.iloc[tr], Y.iloc[tr], ts)
            summary.append(stat("Kept at min CV", int(sum(abs(c) > 1e-10 for c in coef))))
            summary.append(stat("Kept at 1-s.e. rule", len(keep)))
            notes.append("One-standard-error rule keeps: " + (", ".join(keep) if keep else "no predictor") + ".")
    extra_charts = []
    if kind == "svr":
        gs = m.regressor_
        best, est = gs.best_params_, gs.best_estimator_[-1]
        nsv = len(est.support_)
        rows = [["Kernel", "Gaussian (radial basis)" if est.kernel == "rbf" else "Linear"], ["C (cost), chosen by cross-validation", f(best["svr__C"], 4)]]
        if est.kernel == "rbf":
            rows.append(["Gamma, chosen by cross-validation", f(best["svr__gamma"], 4)])
            summary.append(stat("Gamma", float(best["svr__gamma"]), 4))
        rows += [["Epsilon (standard deviations of the outcome)", f(est.epsilon, 4)], ["Support vectors", f"{nsv} of {len(tr)} training observations ({f(100 * nsv / len(tr), 1)}%)"]]
        tabs.append(table("Selected hyperparameters", ["Parameter", "Value"], rows,
                          "C and gamma are chosen from a grid by 3-fold cross-validation on the training set; the outcome and the predictors are standardised first."))
        summary.append(stat("C (cost)", float(best["svr__C"]), 3))
        summary.append(stat("Support vectors (share)", nsv / len(tr), 3))
        notes.append("Epsilon-insensitive support vector regression: errors smaller than epsilon are not penalised, so only the support vectors shape the fit. "
                     "Like every kernel method it gives predictions, not marginal effects; use KRLS when you need the effect of each predictor.")
    if kind == "krls":
        ef = m.effects()
        dfe = max(len(tr) - m.df_eff_, 1.0)
        tv = ef["avg"] / ef["se"]
        pv = 2 * stats.t.sf(np.abs(tv), dfe)
        pw = ef["pointwise"]
        order = np.argsort(-np.abs(np.nan_to_num(tv)))
        rows = [[x[j], "first difference (0 to 1)" if m.binary_[j] else "derivative", f(ef["avg"][j]), f(ef["se"][j]), f(tv[j], 3), f(pv[j], 4),
                 f(np.percentile(pw[:, j], 25)), f(np.percentile(pw[:, j], 50)), f(np.percentile(pw[:, j], 75))] for j in order]
        tabs.append(table("Average marginal effects (training sample)", ["Predictor", "Effect of", "Average", "Std. err.", "t", "p-value", "25th pct", "Median", "75th pct"], rows,
                          f"Effect on {y} of one unit of the predictor (or of a switch from 0 to 1), averaged over the training observations; the last three columns show how the pointwise "
                          f"effects vary. Standard errors condition on lambda and the bandwidth; t has {f(dfe, 1)} degrees of freedom (n minus effective df)."))
        summary += [stat("Lambda", m.lambda_, 4), stat("Kernel bandwidth sigma2", m.sigma2_, 3), stat("Effective df", m.df_eff_, 1), stat("R-squared (training)", m.r2_, 4)]
        for j in [j for j in order if not m.binary_[j]][:3]:
            extra_charts.append(chart(f"Marginal effect of {x[j]} across observations", X.iloc[tr][x[j]].to_numpy(), [("marginal effect", pw[:, j])], x[j], f"effect on {y}", kind="scatter"))
        notes.append("Kernel-based regularized least squares (Hainmueller and Hazlett 2014): a flexible, nonparametric fit with a Gaussian kernel, lambda chosen by leave-one-out "
                     "cross-validation, and marginal effects that are free to vary across observations. The averages are comparable to regression coefficients; "
                     "the spread of the pointwise effects shows nonlinearity and interaction.")
        if len(tr) > 1000:
            notes.append("With more than 1,000 training observations KRLS can take a minute or more in the browser.")
    pi = permutation_importance(m, X.iloc[te], Y.iloc[te], n_repeats=10, random_state=0, scoring="neg_root_mean_squared_error")
    order = np.argsort(-pi.importances_mean)
    tabs.append(table("Permutation importance on the test set", ["Predictor", "RMSE increase", "Std."],
                      [[x[i], f(pi.importances_mean[i]), f(pi.importances_std[i])] for i in order]))
    top = order[: min(15, len(order))]
    charts = [chart("Permutation importance", [x[i] for i in top], [("RMSE increase", pi.importances_mean[top])], "", "", kind="bar"),
              chart("Test set: actual and predicted", list(range(1, len(te) + 1)), [("actual", Y.iloc[te].values), ("predicted", pred)], "test observation", y)] + extra_charts
    return result(kind, f"{label}: predicting {y}", len(d), summary, tabs, charts, notes)


def lasso(df, spec):
    return _predict(df, spec, "lasso")


def ridge(df, spec):
    return _predict(df, spec, "ridge")


def enet(df, spec):
    return _predict(df, spec, "enet")


def rf(df, spec):
    return _predict(df, spec, "rf")


def xgb(df, spec):
    return _predict(df, spec, "xgb")


def svr(df, spec):
    return _predict(df, spec, "svr")


def krls(df, spec):
    return _predict(df, spec, "krls")


# ------------------------------------------------------------------------------------------------ double machine learning
def _causal_data(df, spec, with_instr=False):
    y = need(spec, "y", "outcome")
    t = need(spec, "treat", "treatment")
    x = as_list(need(spec, "x", "controls"))
    z = need(spec, "instrument", "instrument") if with_instr else None
    distinct([y], [t], x, [z] if z else [])
    d, _ = _data(df, {}, [y, t] + x + ([z] if z else []))
    return d, y, t, x, z


def _ols_bench(d, y, t, x):
    import statsmodels.api as sm
    o = sm.OLS(d[y].astype(float), sm.add_constant(d[[t] + x].astype(float))).fit(cov_type="HC1")
    return float(o.params[t]), float(o.bse[t])


def _dml(df, spec, kind):
    d, y, t, x, z = _causal_data(df, spec, kind == "PLIV")
    folds = int_option(spec, "folds", 5, 2, 10, "Number of folds")
    reps = int_option(spec, "reps", 2, 1, 10, "Repetitions")
    binary = bool(M.is_binary(d[t]))
    if kind == "IRM" and not binary:
        raise OESError("The interactive model needs a binary (0/1) treatment. Use the partially linear model for a continuous treatment.")
    trees = int_option(spec, "trees", 100, 20, 1000, "Trees in the random forests")
    L = _learners(binary, trees)
    which = spec.get("learner", "both")
    keys = ["lasso", "rf"] if which == "both" else [which]
    rows, est = [], {}
    for k in keys:
        th, se = M.crossfit(d[y], d[t], d[x], L[k], "IRM" if kind == "IRM" else "PLR", Z=d[z] if z else None, folds=folds, reps=reps)
        est[k] = (th, se)
        rows.append([{"lasso": "DML with LASSO", "rf": "DML with random forest"}[k], th, se])
    ob, os_ = _ols_bench(d, y, t, x)
    rows.append(["OLS with controls (HC1)", ob, os_])
    title = {"PLR": "partially linear model", "IRM": "interactive model (ATE)", "PLIV": "partially linear IV model"}[kind]
    tabs = [coef_table([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], dist="z", title=f"Effect of {t} on {y}: {title}",
                       note=f"Cross-fitting with {folds} folds, {reps} repetition(s), median aggregation (Chernozhukov et al. 2018); "
                            f"random forests with {trees} trees.")]
    checks = []
    if len(est) == 2:
        g = abs(est["lasso"][0] - est["rf"][0]) / max(est["lasso"][1], est["rf"][1])
        checks.append(["Learner gap (in standard errors)", f(g, 3), None, "above 1: the learners disagree; nonlinearity matters, prefer the forest"])
    if kind == "IRM":
        from sklearn.model_selection import cross_val_predict
        ps = cross_val_predict(L["rf"]["clf"], d[x], d[t].astype(int), cv=5, method="predict_proba")[:, 1]
        out = float(np.mean((ps < 0.05) | (ps > 0.95)))
        checks.append(["Propensity score range", f"{f(ps.min(), 3)} to {f(ps.max(), 3)}", None, f"share outside [0.05, 0.95]: {f(out, 4)}"])
    if kind == "PLIV":
        import statsmodels.api as sm
        fs = sm.OLS(d[t].astype(float), sm.add_constant(d[[z] + x].astype(float))).fit(cov_type="HC1")
        F = float((fs.params[z] / fs.bse[z]) ** 2)
        checks.append(test_row("First-stage F (robust, linear)", F, float(fs.pvalues[z]), "below 10: weak instrument"))
    if checks:
        tabs.append(tests_table(checks, "Checks"))
    main = est.get("rf") or est[keys[0]]
    summary = [stat("Observations", len(d)), stat("Controls", len(x)), stat("Estimate", main[0]), stat("Std. err.", main[1]),
               stat("OLS estimate", ob)]
    return result({"PLR": "dml_plr", "IRM": "dml_irm", "PLIV": "dml_iv"}[kind], f"Double machine learning: {title}", len(d), summary, tabs)


def dml_plr(df, spec):
    return _dml(df, spec, "PLR")


def dml_irm(df, spec):
    return _dml(df, spec, "IRM")


def dml_iv(df, spec):
    return _dml(df, spec, "PLIV")


def causal_forest(df, spec):
    """Doubly robust forest (DR-learner): cross-fitted AIPW scores, then a random forest of the scores on the controls."""
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    import statsmodels.api as sm
    d, y, t, x, _ = _causal_data(df, spec)
    if not M.is_binary(d[t]):
        raise OESError("The causal forest here needs a binary (0/1) treatment.")
    folds = int_option(spec, "folds", 5, 2, 10, "Number of folds")
    Y, D, X = d[y].to_numpy(float), d[t].to_numpy(float), d[x].to_numpy(float)
    L = _learners(True, int_option(spec, "trees", 100, 20, 1000, "Trees in the nuisance forests"))["rf"]
    g0, g1, m = np.zeros_like(Y), np.zeros_like(Y), np.zeros_like(Y)
    kf = KFold(folds, shuffle=True, random_state=0)
    for tr, te in kf.split(X):
        for v, arr in ((0, g0), (1, g1)):
            idx = tr[D[tr] == v]
            arr[te] = clone(L["reg"]).fit(X[idx], Y[idx]).predict(X[te])
        m[te] = clone(L["clf"]).fit(X[tr], D[tr]).predict_proba(X[te])[:, 1]
    m = np.clip(m, 0.01, 0.99)
    psi = g1 - g0 + D * (Y - g1) / m - (1 - D) * (Y - g0) / (1 - m)
    ate, se = float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(Y)))
    leaf = int_option(spec, "min_leaf", 20, 5, 500, "Minimum leaf size")
    forest = RandomForestRegressor(n_estimators=int_option(spec, "n_estimators", 300, 50, 2000, "Number of trees"), min_samples_leaf=leaf,
                                   max_features=0.5, oob_score=True, random_state=0, n_jobs=1).fit(X, psi)
    tau = forest.oob_prediction_
    # best linear predictor test (Chernozhukov, Demirer, Duflo and Fernandez-Val): psi on 1 and (tau - mean tau)
    blp = sm.OLS(psi, sm.add_constant(tau - tau.mean())).fit(cov_type="HC1")
    q = pd.qcut(pd.Series(tau).rank(method="first"), 4, labels=False)
    gates = []
    for g in range(4):
        s = psi[q.values == g]
        gates.append([f"Quartile {g + 1} of predicted effect", f(float(np.mean(tau[q.values == g]))), f(float(s.mean())), f(float(s.std(ddof=1) / np.sqrt(len(s)))), int(len(s))])
    imp = sorted(zip(x, forest.feature_importances_), key=lambda z: -z[1])
    tabs = [coef_table(["Average treatment effect (doubly robust)"], [ate], [se], dist="z", title=f"Effect of {t} on {y}",
                       note="AIPW scores from cross-fitted random forests."),
            coef_table(["Average effect (intercept)", "Heterogeneity loading (slope on predicted effect)"], blp.params, blp.bse, dist="z",
                       title="Best linear predictor test of heterogeneity",
                       note="A slope significantly above 0 (ideally near 1) means the forest finds real differences in the effect."),
            table("Group average treatment effects (GATES)", ["Group", "Mean predicted effect", "Estimated effect", "Std. err.", "Obs."], gates),
            table("Which controls drive the differences", ["Variable", "Importance"], [[v, f(i, 4)] for v, i in imp])]
    summary = [stat("Observations", len(d)), stat("ATE", ate), stat("Std. err.", se),
               stat("Predicted effects: 10th pct", float(np.percentile(tau, 10))), stat("Predicted effects: 90th pct", float(np.percentile(tau, 90)))]
    notes = ["This is a doubly robust (DR-learner) forest: honest cross-fitted scores with out-of-bag predictions. It targets the same "
             "conditional effects as EconML's CausalForestDML but is not the same algorithm; EconML is not available in the browser."]
    if len(d) < 2000:
        notes.append("Effects by subgroup are noisy with fewer than about 2,000 observations.")
    hist, edges = np.histogram(tau, bins=20)
    charts = [chart("Distribution of predicted effects", [f((a + b) / 2, 4) for a, b in zip(edges[:-1], edges[1:])], [("count", hist)], "predicted effect", "count", kind="bar")]
    return result("causal_forest", f"Causal forest: effect of {t} on {y}", len(d), summary, tabs, charts, notes)
