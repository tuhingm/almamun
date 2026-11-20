"""Microeconometric models of the Microeconometrics Guide: binary-outcome (logit and probit) regression.

Estimators are statsmodels maximum-likelihood fits; the marginal effects, odds ratios, classification table, ROC curve and the
specification tests (likelihood ratio, Hosmer-Lemeshow, Pregibon link test) are computed here so that every number on the page
can be reproduced with the code in the guide.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .core import OESError, as_list, chart, coef_table, distinct, f, float_option, need, numeric, result, stat, table, test_row, tests_table


def _choice(spec, key, options, default, label):
    v = spec.get(key, default)
    if v in (None, ""):
        v = default
    if v not in options:
        raise OESError(f"{label} must be one of: {', '.join(options)}.")
    return v


def _fit_binary(Y, X, link, cov, groups=None):
    model = sm.Logit(Y, X) if link == "logit" else sm.Probit(Y, X)
    kw = dict(disp=0, maxiter=200)
    if cov == "robust":
        kw.update(cov_type="HC1")
    elif cov == "cluster":
        kw.update(cov_type="cluster", cov_kwds={"groups": groups})
    try:
        res = model.fit(method="newton", **kw)
    except Exception as e:  # perfect separation, singular Hessian
        if "perfect" in str(e).lower() or "singular" in str(e).lower():
            raise OESError("The maximum-likelihood estimate does not exist (perfect or quasi-perfect separation, or collinear regressors). "
                           "Drop the regressor that predicts the outcome perfectly, or combine categories.")
        raise OESError(f"The model did not converge ({e}).")
    if not res.mle_retvals.get("converged", True):
        raise OESError("The maximum-likelihood iterations did not converge. Check for separation or near-collinear regressors.")
    return res


def _auc(y, p):
    r = stats.rankdata(p)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _roc_curve(y, p, grid=21):
    """True-positive rate at grid false-positive rates, by linear interpolation of the empirical ROC curve."""
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    last = np.r_[ps[1:] != ps[:-1], True]      # one point per distinct threshold
    tpr = np.r_[0, tp[last] / tp[-1]]
    fpr = np.r_[0, fp[last] / fp[-1]]
    g = np.linspace(0, 1, grid)
    return g, np.interp(g, fpr, tpr)


def _hosmer_lemeshow(y, p, groups):
    cut = pd.qcut(pd.Series(p).rank(method="first"), groups, labels=False)
    chi, used = 0.0, 0
    for g in range(groups):
        m = (cut == g).to_numpy()
        n = int(m.sum())
        if n == 0:
            continue
        o, pbar = float(y[m].sum()), float(p[m].mean())
        if 0 < pbar < 1:
            chi += (o - n * pbar) ** 2 / (n * pbar * (1 - pbar))
        used += 1
    df = max(used - 2, 1)
    return chi, df, float(stats.chi2.sf(chi, df))


def _link_test(Y, xb, link):
    """Pregibon (1980) link test: refit on the linear index and its square; a significant square signals misspecification."""
    Z = sm.add_constant(np.column_stack([xb, xb ** 2]))
    model = sm.Logit(Y, Z) if link == "logit" else sm.Probit(Y, Z)
    try:
        r = model.fit(method="newton", disp=0, maxiter=100)
        return float(r.tvalues[2]), float(r.pvalues[2])
    except Exception:
        return None, None


def binary(df, spec):
    """Logit or probit regression for a 0/1 outcome, with marginal effects, odds ratios, classification and ROC."""
    y = need(spec, "y", "binary outcome variable")
    xs = as_list(spec.get("x"))
    if not xs:
        raise OESError("Choose at least one explanatory variable.")
    link = _choice(spec, "link", ("logit", "probit"), "logit", "Link")
    cov = _choice(spec, "cov", ("oim", "robust", "cluster"), "oim", "Standard errors")
    me = _choice(spec, "me", ("ame", "mem"), "ame", "Marginal effects")
    cutoff = float_option(spec, "cutoff", 0.5, "The classification cut-off")
    if not 0 < cutoff < 1:
        raise OESError("The classification cut-off must be between 0 and 1.")
    cl = spec.get("cluster") or None
    if cov == "cluster" and not cl:
        raise OESError("Choose the cluster variable.")
    distinct(y, xs, cl)
    d = numeric(df, [y] + xs, keep=([cl] if cl else [])).reset_index(drop=True)
    if set(d[y].unique()) - {0, 1}:
        raise OESError(f"The outcome {y} must be coded 0 and 1.")
    if d[y].nunique() < 2:
        raise OESError(f"The outcome {y} takes only one value, so the model cannot be estimated.")
    n, k = len(d), len(xs) + 1
    if n < 3 * k:
        raise OESError(f"{n} observations are too few for {k} parameters; use at least three times as many observations as parameters.")
    Y = d[y].to_numpy(float)
    X = sm.add_constant(d[xs].astype(float), has_constant="add")
    names = list(X.columns)
    groups = pd.factorize(d[cl])[0] if cov == "cluster" else None
    res = _fit_binary(Y, X, link, cov, groups)
    ll, ll0 = float(res.llf), float(res.llnull)
    lr = 2 * (ll - ll0)
    lr_df = k - 1
    p = np.asarray(res.predict(X), float)
    xb = np.asarray(X, float) @ np.asarray(res.params, float)
    yhat = (p >= cutoff).astype(int)
    correct = float((yhat == Y).mean())
    auc = _auc(Y, p)
    covtxt = {"oim": "conventional (inverse observed information)", "robust": "heteroskedasticity-robust (HC1)", "cluster": f"clustered by {cl}"}[cov]
    coefs = coef_table(names, res.params, res.bse, dist="z",
                       note=f"{link.capitalize()} maximum likelihood. Standard errors: {covtxt}. The coefficients are changes in the "
                            f"{'log odds' if link == 'logit' else 'probit index (z-score)'}, not in the probability; read the marginal effects below.")
    tabs = [coefs]
    # marginal effects: discrete change for 0/1 regressors, derivative otherwise
    dummies = [c for c in xs if set(d[c].unique()) <= {0, 1}]
    mm = res.get_margeff(at="overall" if me == "ame" else "mean", method="dydx", dummy=True)
    fr = mm.summary_frame()
    rows = [[name, f(r["dy/dx"]), f(r["Std. Err."]), f(r["z"], 4), f(r["Pr(>|z|)"], 4), f(r["Conf. Int. Low"]), f(r["Cont. Int. Hi."]),
             "discrete change 0 to 1" if name in dummies else "derivative"] for name, r in fr.iterrows()]
    label = "Average marginal effects (change in Pr(y = 1))" if me == "ame" else "Marginal effects at the means (change in Pr(y = 1))"
    tabs.append(table(label, ["Variable", "dy/dx", "Std. err.", "z", "p-value", "95% CI low", "95% CI high", "Type"], rows,
                      "Delta-method standard errors. For a 0/1 regressor the effect is the discrete change in the probability from 0 to 1, with the other regressors "
                      + ("at their observed values and averaged over the sample." if me == "ame" else "held at their sample means.")))
    if link == "logit":
        ci = np.exp(res.conf_int().to_numpy())
        orr = [[names[i], f(np.exp(res.params.iloc[i])), f(np.exp(res.params.iloc[i]) * res.bse.iloc[i]), f(ci[i, 0]), f(ci[i, 1])] for i in range(1, k)]
        tabs.append(table("Odds ratios", ["Variable", "Odds ratio", "Std. err.", "95% CI low", "95% CI high"], orr,
                          "exp(coefficient): the factor by which the odds of y = 1 change when the regressor rises by one unit. The standard error is by the delta method."))
    # fit and classification
    n1 = int(Y.sum())
    sens = float(((yhat == 1) & (Y == 1)).sum() / n1)
    spec_ = float(((yhat == 0) & (Y == 0)).sum() / (n - n1))
    tabs.append(table(f"Classification at cut-off {cutoff:g}", ["Actual", "Predicted 0", "Predicted 1", "Total"],
                      [["0", int(((yhat == 0) & (Y == 0)).sum()), int(((yhat == 1) & (Y == 0)).sum()), n - n1],
                       ["1", int(((yhat == 0) & (Y == 1)).sum()), int(((yhat == 1) & (Y == 1)).sum()), n1]],
                      f"Sensitivity (share of actual 1s predicted 1) = {sens:.3f}; specificity (share of actual 0s predicted 0) = {spec_:.3f}. "
                      "With rare outcomes use a cut-off near the sample share of 1s, not 0.5."))
    g = min(10, max(n // 5, 3))
    hl, hl_df, hl_p = _hosmer_lemeshow(Y, p, g)
    lt, lt_p = _link_test(Y, xb, link)
    drows = [test_row(f"Likelihood-ratio test, all slopes zero ({lr_df} df)", lr, stats.chi2.sf(lr, lr_df), "H0: the model is no better than a constant"),
             test_row(f"Hosmer-Lemeshow goodness of fit ({g} groups, {hl_df} df)", hl, hl_p, "H0: predicted and observed frequencies agree; low power in small samples")]
    if lt is not None:
        drows.append(test_row("Pregibon link test (z on the squared index)", lt, lt_p, "H0: the link function is correctly specified"))
    tabs.append(tests_table(drows))
    # charts: ROC and the predicted probabilities by outcome
    gx, tpr = _roc_curve(Y, p)
    roc = chart("ROC curve", [f"{v:.2f}" for v in gx], [("Model", tpr), ("Chance", gx)], "False-positive rate", "True-positive rate")
    edges = np.linspace(0, 1, 11)
    h0 = np.histogram(p[Y == 0], bins=edges)[0] / max((Y == 0).sum(), 1)
    h1 = np.histogram(p[Y == 1], bins=edges)[0] / max((Y == 1).sum(), 1)
    dist = chart("Predicted probability by actual outcome", [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in range(10)],
                 [(f"{y} = 0", h0), (f"{y} = 1", h1)], "Predicted Pr(y = 1)", "Share of observations", kind="bar")
    summary = [stat("Observations", n), stat("Log-likelihood", ll), stat("LR chi-squared", lr), stat("McFadden pseudo R-squared", 1 - ll / ll0),
               stat("Share correctly classified", correct, 3), stat("Area under the ROC curve", auc, 3)]
    notes = [f"{link.capitalize()} regression of {y} on {', '.join(xs)}, {n} observations, share of ones = {Y.mean():.3f}.",
             "Logit and probit give almost the same fitted probabilities; the logit coefficients are about 1.6 times the probit coefficients, so compare marginal effects, not coefficients, across the two."]
    if n1 < 10 or (n - n1) < 10:
        notes.append("Fewer than 10 observations in one outcome class: maximum-likelihood standard errors and the Hosmer-Lemeshow test are unreliable.")
    if n < 10 * k:
        notes.append(f"{n} observations for {k} parameters is below the rule of thumb of 10 observations per parameter in the rarer class.")
    return result("logit_probit", f"{link.capitalize()} regression: {y}", n, summary, tabs, [roc, dist], notes)


# ---------------------------------------------------------------------------------------------------------------------
# ordered logit and probit
# ---------------------------------------------------------------------------------------------------------------------
def _cdf(link):
    return (lambda z: 1 / (1 + np.exp(-z))) if link == "logit" else stats.norm.cdf


def _pdf(link):
    return (lambda z: np.exp(z) / (1 + np.exp(z)) ** 2) if link == "logit" else stats.norm.pdf


def _ord_probs(X, beta, cuts, link):
    """n x J matrix of category probabilities: Pr(y = j) = F(c_j - x'b) - F(c_{j-1} - x'b)."""
    F = _cdf(link)
    xb = X @ beta
    cum = np.column_stack([np.zeros(len(xb))] + [F(c - xb) for c in cuts] + [np.ones(len(xb))])
    return np.diff(cum, axis=1)


def _cuts_from_raw(raw):
    """statsmodels parametrisation (first cut-point, then log increments) to ordered cut-points."""
    return raw[0] + np.r_[0.0, np.cumsum(np.exp(raw[1:]))]


def _num_jac(fun, x, h=1e-5):
    x = np.asarray(x, float)
    f0 = np.atleast_1d(fun(x))
    J = np.zeros((len(f0), len(x)))
    for i in range(len(x)):
        e = np.zeros(len(x)); e[i] = h * max(1.0, abs(x[i]))
        J[:, i] = (np.atleast_1d(fun(x + e)) - np.atleast_1d(fun(x - e))) / (2 * e[i])
    return J


def _brant(Y, X, levels, k):
    """Brant (1990) Wald test of the parallel-lines (proportional odds) assumption for the ordered logit."""
    J = len(levels)
    betas, ps = [], []
    for j in range(J - 1):
        yb = (Y > levels[j]).astype(float)
        fit = sm.Logit(yb, sm.add_constant(X, has_constant="add")).fit(disp=0, maxiter=100)
        betas.append(np.asarray(fit.params)[1:])
        ps.append(np.asarray(fit.predict(sm.add_constant(X, has_constant="add"))))
    Xc = sm.add_constant(X, has_constant="add")
    inv = [np.linalg.inv(Xc.T @ (Xc * (p * (1 - p))[:, None])) for p in ps]
    m = J - 1
    V = np.zeros((m * k, m * k))
    for a in range(m):
        for b in range(a, m):
            W = ps[b] * (1 - ps[a])                       # Pr(y > b)(1 - Pr(y > a)) for a <= b
            blk = (inv[a] @ (Xc.T @ (Xc * W[:, None])) @ inv[b])[1:, 1:]
            V[a * k:(a + 1) * k, b * k:(b + 1) * k] = blk
            V[b * k:(b + 1) * k, a * k:(a + 1) * k] = blk.T
    D = np.zeros(((m - 1) * k, m * k))
    for a in range(m - 1):
        D[a * k:(a + 1) * k, a * k:(a + 1) * k] = np.eye(k)
        D[a * k:(a + 1) * k, (a + 1) * k:(a + 2) * k] = -np.eye(k)
    bv = np.concatenate(betas)
    dv = D @ bv
    W = float(dv @ np.linalg.solve(D @ V @ D.T, dv))
    return W, (m - 1) * k, float(stats.chi2.sf(W, (m - 1) * k))


def ordered(df, spec):
    """Ordered logit or probit for an ordinal outcome with 3 to 10 ordered categories."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    y = need(spec, "y", "ordinal outcome variable")
    xs = as_list(spec.get("x"))
    if not xs:
        raise OESError("Choose at least one explanatory variable.")
    link = _choice(spec, "link", ("logit", "probit"), "logit", "Link")
    cov = _choice(spec, "cov", ("oim", "robust"), "oim", "Standard errors")
    distinct(y, xs)
    d = numeric(df, [y] + xs).reset_index(drop=True)
    levels = np.sort(d[y].unique())
    J = len(levels)
    if J < 3:
        raise OESError(f"The outcome {y} has {J} distinct values; use the logit/probit model for a binary outcome.")
    if J > 10:
        raise OESError(f"The outcome {y} has {J} distinct values. An ordered model suits 3 to 10 ordered categories; for a continuous outcome use a regression.")
    n, k = len(d), len(xs)
    counts = np.array([(d[y] == v).sum() for v in levels])
    if counts.min() < 5:
        raise OESError("Every category needs at least 5 observations. Combine the sparse categories and try again.")
    if n < 10 * k:
        raise OESError(f"{n} observations are too few for {k} regressors; use at least ten observations per regressor.")
    X = d[xs].astype(float)
    if np.linalg.matrix_rank(X.to_numpy()) < k or (X.std() == 0).any():
        raise OESError("One regressor is constant, or the regressors are perfectly collinear. An ordered model has no intercept, because the cut-points play that role.")
    ycat = pd.Categorical(d[y], categories=levels, ordered=True)
    mod = OrderedModel(ycat, X, distr=link)
    try:
        kw = dict(method="bfgs", disp=0, maxiter=1000)
        if cov == "robust":
            kw.update(cov_type="HC1")
        res = mod.fit(**kw)
    except Exception as e:
        if "constant" in str(e).lower():
            raise OESError("The regressors contain a constant (for example dummies that add up to one). Drop one of them: the cut-points play the role of the intercept.")
        raise OESError(f"The model did not converge ({e}).")
    raw = np.asarray(res.params, float)
    beta, rawc = raw[:k], raw[k:]
    cuts = _cuts_from_raw(rawc)
    Vraw = np.asarray(res.cov_params(), float)
    # cut-points and their standard errors by the delta method
    Jc = _num_jac(lambda r: _cuts_from_raw(r), rawc)
    Vc = Jc @ Vraw[k:, k:] @ Jc.T
    Vfull = np.zeros((k + J - 1, k + J - 1))
    Vfull[:k, :k] = Vraw[:k, :k]
    Vfull[:k, k:] = Vraw[:k, k:] @ Jc.T
    Vfull[k:, :k] = Jc @ Vraw[k:, :k]
    Vfull[k:, k:] = Vc
    se = np.sqrt(np.diag(Vfull))
    ll = float(res.llf)
    ll0 = float((counts * np.log(counts / n)).sum())
    lr = 2 * (ll - ll0)
    Xn = X.to_numpy()
    probs = _ord_probs(Xn, beta, cuts, link)
    ycode = np.searchsorted(levels, d[y].to_numpy())
    pred = probs.argmax(axis=1)
    correct = float((pred == ycode).mean())
    covtxt = "conventional (inverse observed information)" if cov == "oim" else "heteroskedasticity-robust (HC1)"
    tabs = [coef_table(xs, beta, se[:k], dist="z", note=f"Ordered {link}, maximum likelihood. Standard errors: {covtxt}. "
                       "A positive coefficient raises the probability of the higher categories and lowers that of the lowest; "
                       "the size is not a change in probability, so read the odds ratios and marginal effects."),
            table("Cut-points (thresholds)", ["Cut-point", "Estimate", "Std. err.", "z"],
                  [[f"{f(levels[i], 6)} | {f(levels[i + 1], 6)}", f(cuts[i]), f(se[k + i]), f(cuts[i] / se[k + i], 4)] for i in range(J - 1)],
                  "Pr(y <= j) = F(cut-point j - x'b). The cut-points are ordered by construction and are the model's intercepts.")]
    if link == "logit":
        ci = np.exp(np.column_stack([beta - 1.96 * se[:k], beta + 1.96 * se[:k]]))
        tabs.append(table("Odds ratios (proportional odds)", ["Variable", "Odds ratio", "Std. err.", "95% CI low", "95% CI high"],
                          [[xs[i], f(np.exp(beta[i])), f(np.exp(beta[i]) * se[i]), f(ci[i, 0]), f(ci[i, 1])] for i in range(k)],
                          "exp(coefficient): the factor by which the odds of being in a HIGHER category (rather than in or below the current one) change for a one-unit rise. "
                          "The same factor applies at every cut-point, which is the proportional-odds assumption."))
    dummy = [(X[c].isin([0, 1]).all()) for c in xs]

    def ame_vec(theta):
        b, c = theta[:k], theta[k:]
        out = np.zeros((k, J))
        base_p = _ord_probs(Xn, b, c, link)
        for i in range(k):
            if dummy[i]:
                X1, X0 = Xn.copy(), Xn.copy()
                X1[:, i], X0[:, i] = 1, 0
                out[i] = (_ord_probs(X1, b, c, link) - _ord_probs(X0, b, c, link)).mean(axis=0)
            else:
                dens = _pdf(link)
                xb = Xn @ b
                fz = np.column_stack([np.zeros(n)] + [dens(cj - xb) for cj in c] + [np.zeros(n)])
                out[i] = (-b[i] * np.diff(fz, axis=1)).mean(axis=0)
        return out.ravel()
    theta = np.r_[beta, cuts]
    ame = ame_vec(theta)
    Jm = _num_jac(ame_vec, theta)
    se_ame = np.sqrt(np.maximum(np.diag(Jm @ Vfull @ Jm.T), 0))
    rows = []
    for i in range(k):
        for j in range(J):
            a, s_ = ame[i * J + j], se_ame[i * J + j]
            rows.append([xs[i], f"y = {f(levels[j], 6)}", f(a), f(s_), f(a / s_, 4) if s_ > 0 else None, f(2 * stats.norm.sf(abs(a / s_)), 4) if s_ > 0 else None,
                         "discrete change 0 to 1" if dummy[i] else "derivative"])
    tabs.append(table("Average marginal effects on each category probability", ["Variable", "Category", "dPr/dx", "Std. err.", "z", "p-value", "Type"], rows,
                      "Delta-method standard errors. For each regressor the effects across categories add to zero: what the highest category gains, the lower ones lose."))
    drows = [test_row(f"Likelihood-ratio test, all slopes zero ({k} df)", lr, stats.chi2.sf(lr, k), "H0: the model is no better than the category frequencies")]
    if link == "logit":
        try:
            bw, bdf, bp = _brant(d[y].to_numpy(), Xn, levels, k)
            drows.append(test_row(f"Brant test of proportional odds ({bdf} df)", bw, bp, "H0: the slopes are the same at every cut-point (parallel lines)"))
        except Exception:
            pass
    tabs.append(tests_table(drows))
    obs = counts / n
    prd = probs.mean(axis=0)
    ch = chart("Observed and predicted category shares", [f"{f(v, 6)}" for v in levels], [("Observed", obs), ("Predicted (average)", prd)], y, "Share of observations", kind="bar")
    summary = [stat("Observations", n), stat("Categories", J), stat("Log-likelihood", ll), stat("LR chi-squared", lr), stat("McFadden pseudo R-squared", 1 - ll / ll0),
               stat("Share predicted in the right category", correct, 3)]
    notes = [f"Ordered {link} of {y} ({J} categories: {', '.join(str(f(v, 6)) for v in levels)}) on {', '.join(xs)}, {n} observations.",
             "There is no intercept: the J - 1 cut-points play that role. The ordering of the categories must be meaningful, otherwise use a multinomial model."]
    if link == "logit" and len(drows) > 1 and drows[1][2] is not None and drows[1][2] < 0.05:
        notes.append("The Brant test rejects proportional odds: the effect of some regressor differs between cut-points. Consider a generalised ordered logit, or collapse categories.")
    return result("ordered", f"Ordered {link}: {y}", n, summary, tabs, [ch], notes)


# ---------------------------------------------------------------------------------------------------------------------
# multinomial logit
# ---------------------------------------------------------------------------------------------------------------------
def _softmax_probs(X1, B):
    """X1: n x (k+1) with a constant, B: (k+1) x (J-1) coefficients of the non-base categories. Returns n x J (base first)."""
    z = np.column_stack([np.zeros(len(X1)), X1 @ B])
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _mnl_fit(Yc, X1, J):
    import statsmodels.api as sm_
    res = sm_.MNLogit(Yc, X1).fit(method="newton", disp=0, maxiter=200)
    B = np.asarray(res.params, float)
    return res, B


def multinomial(df, spec):
    """Multinomial logit for an unordered outcome with 3 to 10 categories."""
    y = need(spec, "y", "outcome variable (category code)")
    xs = as_list(spec.get("x"))
    if not xs:
        raise OESError("Choose at least one explanatory variable.")
    distinct(y, xs)
    d = numeric(df, [y] + xs).reset_index(drop=True)
    levels = np.sort(d[y].unique())
    J = len(levels)
    if J < 3:
        raise OESError(f"The outcome {y} has {J} distinct values; use the logit/probit model for a binary outcome.")
    if J > 10:
        raise OESError(f"The outcome {y} has {J} distinct values. A multinomial logit suits 3 to 10 categories.")
    base = spec.get("base")
    base = levels[0] if base in (None, "") else float(base)
    if base not in levels:
        raise OESError(f"The base category {base:g} is not a value of {y}.")
    order = [base] + [v for v in levels if v != base]
    n, k = len(d), len(xs)
    counts = np.array([(d[y] == v).sum() for v in order])
    if counts.min() < 10:
        raise OESError("Every category needs at least 10 observations. Combine the sparse categories and try again.")
    if n < 10 * k * (J - 1) / 2:
        raise OESError(f"{n} observations are too few for {k * (J - 1)} slopes; reduce the regressors or the categories.")
    code = {v: i for i, v in enumerate(order)}
    Yc = d[y].map(code).to_numpy()
    X1 = sm.add_constant(d[xs].astype(float), has_constant="add")
    names = ["const"] + xs
    try:
        res, B = _mnl_fit(Yc, X1, J)
    except Exception as e:
        raise OESError("The multinomial logit did not converge: " + ("a regressor predicts a category perfectly (separation) or the regressors are collinear." if "singular" in str(e).lower() or "perfect" in str(e).lower() else str(e)))
    if not res.mle_retvals.get("converged", True):
        raise OESError("The maximum-likelihood iterations did not converge. Check for separation or near-collinear regressors.")
    Vfull = np.asarray(res.cov_params(), float)           # ordered equation by equation
    se = np.sqrt(np.diag(Vfull)).reshape(J - 1, k + 1).T
    ll = float(res.llf)
    ll0 = float((counts * np.log(counts / n)).sum())
    lr = 2 * (ll - ll0)
    lr_df = k * (J - 1)
    X1n = np.asarray(X1, float)
    probs = _softmax_probs(X1n, B)
    pred = probs.argmax(axis=1)
    tabs = []
    for j in range(1, J):
        lab = f"{f(order[j], 6)} vs base {f(base, 6)}"
        tabs.append(coef_table(names, B[:, j - 1], se[:, j - 1], dist="z", title=f"Equation: {y} = {lab}",
                               note="Log of the relative risk ratio of the category against the base category. Compare categories through the marginal effects, not coefficient by coefficient." if j == 1 else None))
    rr = []
    for j in range(1, J):
        for i in range(1, k + 1):
            b_, s_ = B[i, j - 1], se[i, j - 1]
            rr.append([f"{f(order[j], 6)} vs {f(base, 6)}", xs[i - 1], f(np.exp(b_)), f(np.exp(b_) * s_), f(np.exp(b_ - 1.96 * s_)), f(np.exp(b_ + 1.96 * s_))])
    tabs.append(table("Relative risk ratios", ["Category vs base", "Variable", "RRR", "Std. err.", "95% CI low", "95% CI high"], rr,
                      "exp(coefficient): the factor by which the probability of the category relative to the base changes for a one-unit rise in the regressor."))
    dummy = [bool(set(d[c].unique()) <= {0, 1}) for c in xs]

    def ame_vec(theta):
        Bt = theta.reshape(J - 1, k + 1).T
        out = np.zeros((k, J))
        for i in range(k):
            if dummy[i]:
                A1, A0 = X1n.copy(), X1n.copy()
                A1[:, i + 1], A0[:, i + 1] = 1, 0
                out[i] = (_softmax_probs(A1, Bt) - _softmax_probs(A0, Bt)).mean(axis=0)
            else:
                P = _softmax_probs(X1n, Bt)
                slope = np.column_stack([np.zeros(n)] + [np.full(n, Bt[i + 1, j]) for j in range(J - 1)])
                out[i] = (P * (slope - (P * slope).sum(axis=1, keepdims=True))).mean(axis=0)
        return out.ravel()
    theta = B.T.ravel()
    ame = ame_vec(theta)
    Jm = _num_jac(ame_vec, theta)
    se_ame = np.sqrt(np.maximum(np.diag(Jm @ Vfull @ Jm.T), 0))
    rows = []
    for i in range(k):
        for j in range(J):
            a, s_ = ame[i * J + j], se_ame[i * J + j]
            rows.append([xs[i], f"{y} = {f(order[j], 6)}", f(a), f(s_), f(a / s_, 4) if s_ > 0 else None, f(2 * stats.norm.sf(abs(a / s_)), 4) if s_ > 0 else None,
                         "discrete change 0 to 1" if dummy[i] else "derivative"])
    tabs.append(table("Average marginal effects on each category probability", ["Variable", "Category", "dPr/dx", "Std. err.", "z", "p-value", "Type"], rows,
                      "Delta-method standard errors. For each regressor the effects across categories add to zero."))
    # classification table
    conf = np.zeros((J, J), int)
    for a_, b_ in zip(Yc, pred):
        conf[a_, b_] += 1
    tabs.append(table("Classification (rows: actual, columns: predicted)", ["Actual"] + [f"Predicted {f(v, 6)}" for v in order],
                      [[f(order[i], 6)] + [int(c) for c in conf[i]] for i in range(J)],
                      f"Share correctly classified = {np.trace(conf) / n:.3f}; the benchmark of always predicting the largest category is {counts.max() / n:.3f}."))
    drows = [test_row(f"Likelihood-ratio test, all slopes zero ({lr_df} df)", lr, stats.chi2.sf(lr, lr_df), "H0: the model is no better than the category frequencies")]
    # Hausman-McFadden test of independence of irrelevant alternatives: drop one category at a time
    iia = []
    if J >= 3:
        for drop in range(1, J):
            keep = [c for c in range(J) if c != drop]
            m_ = np.isin(Yc, keep)
            remap = {c: i for i, c in enumerate(keep)}
            Yr = np.array([remap[c] for c in Yc[m_]])
            try:
                rr_, Br = _mnl_fit(Yr, X1n[m_], J - 1)
                Vr = np.asarray(rr_.cov_params(), float)
            except Exception:
                continue
            kept_eq = [c - 1 for c in keep if c != 0]              # equations (non-base) that remain, in the full model
            idx_f = np.concatenate([np.arange(e * (k + 1), (e + 1) * (k + 1)) for e in kept_eq])
            diff = Br.T.ravel() - theta[idx_f]
            Vd = Vr - Vfull[np.ix_(idx_f, idx_f)]
            try:
                H = float(diff @ np.linalg.pinv(Vd) @ diff)
                rk = int(np.linalg.matrix_rank(Vd))
                if H < 0:
                    iia.append((f(order[drop], 6), None, None, "Hausman statistic is negative (a known problem of the test)"))
                else:
                    iia.append((f(order[drop], 6), H, float(stats.chi2.sf(H, rk)), f"{rk} df"))
            except Exception:
                continue
    for lab, H, p_, det in iia:
        drows.append(test_row(f"Hausman-McFadden IIA test, category {lab} dropped", H, p_, det) if H is not None else [f"Hausman-McFadden IIA test, category {lab} dropped", None, None, det])
    tabs.append(tests_table(drows, note="IIA: the odds of two categories do not depend on the other categories available. The Hausman-McFadden test is known to have poor size and can be negative; treat it as a warning, not proof."))
    obs = counts / n
    prd = probs.mean(axis=0)
    ch = chart("Observed and predicted category shares", [f(v, 6) if not isinstance(f(v, 6), float) or f(v, 6) % 1 else str(int(v)) for v in order], [("Observed", obs), ("Predicted (average)", prd)], y, "Share of observations", kind="bar")
    summary = [stat("Observations", n), stat("Categories", J), stat("Log-likelihood", ll), stat("LR chi-squared", lr), stat("McFadden pseudo R-squared", 1 - ll / ll0),
               stat("Share correctly classified", float(np.trace(conf) / n), 3)]
    notes = [f"Multinomial logit of {y} ({J} categories, base = {f(base, 6)}) on {', '.join(xs)}, {n} observations.",
             "The coefficients of each equation compare one category with the base; changing the base changes the coefficients but not the fitted probabilities or the marginal effects."]
    return result("multinomial", f"Multinomial logit: {y}", n, summary, tabs, [ch], notes)
