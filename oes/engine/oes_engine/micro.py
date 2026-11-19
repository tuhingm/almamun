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


# ---------------------------------------------------------------------------------------------------------------------
# Tobit (censored regression)
# ---------------------------------------------------------------------------------------------------------------------
def _tobit_nll(t, X, y, lo, hi):
    b, s = t[:-1], np.exp(t[-1])
    xb = X @ b
    ll = np.zeros(len(y))
    cl = y <= lo if lo is not None else np.zeros(len(y), bool)
    cu = y >= hi if hi is not None else np.zeros(len(y), bool)
    mid = ~(cl | cu)
    if cl.any():
        ll[cl] = stats.norm.logcdf((lo - xb[cl]) / s)
    if cu.any():
        ll[cu] = stats.norm.logsf((hi - xb[cu]) / s)
    ll[mid] = stats.norm.logpdf((y[mid] - xb[mid]) / s) - t[-1]
    return -ll.sum()


def _tobit_means(xb, s, lo, hi):
    """E[y*], E[y | uncensored] and E[y] for the Tobit model with limits lo and hi (None = no limit)."""
    a = np.full_like(xb, -np.inf) if lo is None else (lo - xb) / s
    b = np.full_like(xb, np.inf) if hi is None else (hi - xb) / s
    Pa, Pb = stats.norm.cdf(a), stats.norm.cdf(b)
    pa = np.where(np.isfinite(a), stats.norm.pdf(a), 0.0)
    pb = np.where(np.isfinite(b), stats.norm.pdf(b), 0.0)
    prob = np.maximum(Pb - Pa, 1e-300)
    cond = xb + s * (pa - pb) / prob
    total = (0.0 if lo is None else Pa * lo) + prob * cond + (0.0 if hi is None else (1 - Pb) * hi)
    return xb, cond, total, prob


def tobit(df, spec):
    """Tobit regression for an outcome censored from below and/or above at known limits."""
    from statsmodels.tools.numdiff import approx_hess
    y = need(spec, "y", "outcome variable")
    xs = as_list(spec.get("x"))
    if not xs:
        raise OESError("Choose at least one explanatory variable.")
    distinct(y, xs)
    lo = float_option(spec, "lower", 0.0, "The lower limit") if spec.get("lower", 0) not in (None, "") else None
    hi = float_option(spec, "upper", None, "The upper limit") if spec.get("upper") not in (None, "") else None
    if lo is None and hi is None:
        raise OESError("Give a lower limit, an upper limit or both: a Tobit model needs censoring.")
    if lo is not None and hi is not None and lo >= hi:
        raise OESError("The lower limit must be below the upper limit.")
    d = numeric(df, [y] + xs).reset_index(drop=True)
    Y = d[y].to_numpy(float)
    if lo is not None and (Y < lo).any():
        raise OESError(f"{y} has values below the lower limit {lo:g}; a censored outcome cannot go below its limit.")
    if hi is not None and (Y > hi).any():
        raise OESError(f"{y} has values above the upper limit {hi:g}.")
    cl = (Y <= lo) if lo is not None else np.zeros(len(Y), bool)
    cu = (Y >= hi) if hi is not None else np.zeros(len(Y), bool)
    n, k = len(d), len(xs)
    if not (cl | cu).any():
        raise OESError("No observation is at a limit, so the data are not censored: use OLS.")
    if (~(cl | cu)).sum() < 3 * (k + 2):
        raise OESError("Too few uncensored observations to estimate the model.")
    X = sm.add_constant(d[xs].astype(float), has_constant="add")
    names = list(X.columns)
    Xn = np.asarray(X, float)
    keep = ~(cl | cu)
    b0 = np.linalg.lstsq(Xn[keep], Y[keep], rcond=None)[0]
    s0 = max(float(np.std(Y[keep] - Xn[keep] @ b0)), 1e-3)
    from scipy import optimize
    fun = lambda t: _tobit_nll(t, Xn, Y, lo, hi)
    opt = optimize.minimize(fun, np.r_[b0, np.log(s0)], method="BFGS", options={"gtol": 1e-7, "maxiter": 2000})
    opt = optimize.minimize(fun, opt.x, method="BFGS", options={"gtol": 1e-8, "maxiter": 2000})
    if not np.isfinite(opt.fun) or np.linalg.norm(opt.jac) > 1e-2 * max(1, n):
        raise OESError("The Tobit likelihood did not converge. Check the limits and the scale of the variables.")
    t = opt.x
    H = approx_hess(t, fun)
    try:
        V = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        raise OESError("The information matrix is singular: the regressors are collinear or a regressor predicts the censoring perfectly.")
    if np.any(np.diag(V) <= 0):
        raise OESError("The information matrix is not positive definite; the estimate is not a maximum.")
    se_all = np.sqrt(np.diag(V))
    beta, s = t[:-1], float(np.exp(t[-1]))
    se_sigma = s * se_all[-1]                                   # delta method for exp(log sigma)
    ll = -float(opt.fun)
    nll0 = optimize.minimize(lambda u: _tobit_nll(u, Xn[:, :1], Y, lo, hi), np.r_[Y.mean(), np.log(Y.std() + 1e-6)], method="BFGS")
    lr = 2 * (ll + float(nll0.fun))
    cof = coef_table(names, beta, se_all[:-1], dist="z", note="Tobit maximum likelihood. The coefficients are effects on the LATENT (uncensored) outcome; "
                     "the marginal effects below are effects on the observed outcome.")
    cof["rows"].append(["sigma", f(s), f(se_sigma), f(s / se_sigma, 4), f(2 * stats.norm.sf(abs(s / se_sigma)), 4), f(s - 1.96 * se_sigma), f(s + 1.96 * se_sigma)])
    tabs = [cof]
    dummy = [bool(set(d[c].unique()) <= {0, 1}) for c in xs]

    def me_vec(theta):
        b_, s_ = theta[:-1], np.exp(theta[-1])
        out = np.zeros((k, 3))
        for i in range(k):
            if dummy[i]:
                A1, A0 = Xn.copy(), Xn.copy(); A1[:, i + 1], A0[:, i + 1] = 1, 0
                m1, m0 = _tobit_means(A1 @ b_, s_, lo, hi), _tobit_means(A0 @ b_, s_, lo, hi)
                out[i] = [b_[i + 1], (m1[1] - m0[1]).mean(), (m1[2] - m0[2]).mean()]
            else:
                h = 1e-4 * max(Xn[:, i + 1].std(), 1e-8)
                A1, A0 = Xn.copy(), Xn.copy(); A1[:, i + 1] += h; A0[:, i + 1] -= h
                m1, m0 = _tobit_means(A1 @ b_, s_, lo, hi), _tobit_means(A0 @ b_, s_, lo, hi)
                out[i] = [b_[i + 1], ((m1[1] - m0[1]) / (2 * h)).mean(), ((m1[2] - m0[2]) / (2 * h)).mean()]
        return out.ravel()
    me = me_vec(t)
    Jm = _num_jac(me_vec, t)
    sem = np.sqrt(np.maximum(np.diag(Jm @ V @ Jm.T), 0))
    rows = []
    for i in range(k):
        for j, lab in enumerate(["Latent outcome E[y*]", "Uncensored outcome E[y | not censored]", "Observed outcome E[y]"]):
            a, s_ = me[i * 3 + j], sem[i * 3 + j]
            rows.append([xs[i], lab, f(a), f(s_), f(a / s_, 4) if s_ > 0 else None, f(2 * stats.norm.sf(abs(a / s_)), 4) if s_ > 0 else None])
    tabs.append(table("Average marginal effects (McDonald-Moffitt decomposition)", ["Variable", "Effect on", "dE/dx", "Std. err.", "z", "p-value"], rows,
                      "The effect on the observed outcome splits into an effect on the probability of being uncensored and an effect on the level of the uncensored outcome. Delta-method standard errors."))
    ols = sm.OLS(Y, X).fit(cov_type="HC1")
    tabs.append(table("Comparison with OLS on all observations", ["Variable", "Tobit coef.", "OLS coef.", "Tobit / OLS"],
                      [[names[i], f(beta[i]), f(ols.params.iloc[i]), f(beta[i] / ols.params.iloc[i], 3)] for i in range(1, k + 1)],
                      "OLS on a censored outcome is attenuated toward zero; the Tobit coefficient is larger in absolute value. The Tobit coefficient is the effect on the latent outcome, OLS estimates something between that and the effect on the observed outcome."))
    # generalised-residual (score) test of normality / homoskedasticity is omitted: report fit of the censoring share instead
    xb = Xn @ beta
    pr_cens = (stats.norm.cdf((lo - xb) / s) if lo is not None else 0.0) + (stats.norm.sf((hi - xb) / s) if hi is not None else 0.0)
    drows = [test_row(f"Likelihood-ratio test, all slopes zero ({k} df)", lr, stats.chi2.sf(lr, k), "H0: the slopes are all zero (constant and sigma free)"),
             ["Share censored: observed against predicted", f(float((cl | cu).mean())), f(float(np.mean(pr_cens))), "the two should be close"]]
    tabs.append(tests_table(drows))
    mt = _tobit_means(xb, s, lo, hi)[2]
    q = pd.qcut(pd.Series(xb).rank(method="first"), 10, labels=False)
    ch = chart("Observed and predicted mean outcome by decile of the linear index", [str(i + 1) for i in range(10)],
               [("Observed", [Y[q == g].mean() for g in range(10)]), ("Predicted E[y]", [mt[q == g].mean() for g in range(10)])], "Decile of x'b", y, kind="bar")
    summary = [stat("Observations", n), stat("Censored at the limits", int((cl | cu).sum())), stat("Log-likelihood", ll), stat("Sigma", s), stat("LR chi-squared", lr)]
    notes = [f"Tobit of {y} on {', '.join(xs)}: {n} observations, {int((cl | cu).sum())} censored ({(cl | cu).mean():.0%}), limits: lower {lo if lo is not None else 'none'}, upper {hi if hi is not None else 'none'}.",
             "The Tobit model assumes that the same process decides whether the outcome is above the limit and how large it is, and that the latent errors are normal and homoskedastic. "
             "If a separate decision creates the zeros (for example, whether to have an affair at all), a hurdle or Heckman model is more appropriate."]
    return result("tobit", f"Tobit regression: {y}", n, summary, tabs, [ch], notes)


def _heckman_nll(t, Z, X, y, s, kz, kx):
    g, b = t[:kz], t[kz:kz + kx]
    sig, rho = np.exp(t[kz + kx]), np.tanh(t[kz + kx + 1])
    zg = Z @ g
    u = (y[s] - X[s] @ b) / sig
    ll = stats.norm.logcdf(-zg[~s]).sum()
    ll += (stats.norm.logpdf(u) - np.log(sig) + stats.norm.logcdf((zg[s] + rho * u) / np.sqrt(1 - rho ** 2))).sum()
    return -ll


def heckman(df, spec):
    """Heckman sample-selection model: full-information maximum likelihood and the two-step estimator."""
    from statsmodels.tools.numdiff import approx_hess
    from scipy import optimize
    y = need(spec, "y", "outcome variable (blank for people who are not selected)")
    sel = need(spec, "sel", "selection indicator (1 = outcome observed, 0 = not)")
    xs, zs = as_list(spec.get("x")), as_list(spec.get("z"))
    if not xs:
        raise OESError("Choose at least one explanatory variable for the outcome equation.")
    if not zs:
        raise OESError("Choose at least one explanatory variable for the selection equation.")
    distinct(y, sel, xs)
    if y in zs or sel in zs:
        raise OESError("The outcome and the selection indicator cannot be explanatory variables.")
    check = numeric(df, [sel] + zs)
    if not set(check[sel].unique()) <= {0, 1}:
        raise OESError(f"The selection indicator {sel} must contain only 0 and 1.")
    d = df.copy()
    for c in [y, sel] + xs + zs:
        if c not in d.columns:
            raise OESError("Column(s) not found in the data: " + c)
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[sel] + zs).reset_index(drop=True)
    s = d[sel].to_numpy(float) == 1
    if s.sum() == 0 or (~s).sum() == 0:
        raise OESError("The selection indicator needs both selected (1) and non-selected (0) observations.")
    if d.loc[s, [y] + xs].isna().any().any():
        raise OESError(f"{y} and the outcome regressors must be observed for every selected observation.")
    if d.loc[~s, y].notna().any():
        pass                                           # an outcome recorded for non-selected people is simply not used
    n, n1 = len(d), int(s.sum())
    kx, kz = len(xs) + 1, len(zs) + 1
    if n1 < 3 * (kx + 2):
        raise OESError("Too few selected observations to estimate the outcome equation.")
    Z = sm.add_constant(d[zs].astype(float), has_constant="add")
    Xf = d[xs].copy()
    Xf = Xf.fillna(Xf.loc[s].mean())                  # placeholder for the unselected rows, never used in the likelihood
    X = sm.add_constant(Xf.astype(float), has_constant="add")
    zn, xn, zn_names, xn_names = np.asarray(Z, float), np.asarray(X, float), list(Z.columns), list(X.columns)
    yv = d[y].fillna(0).to_numpy(float)
    try:
        pr = sm.Probit(s.astype(float), zn).fit(disp=0, maxiter=200)
    except Exception as e:
        raise OESError("The selection probit could not be estimated (perfect prediction or collinear variables): " + str(e))
    if not pr.mle_retvals.get("converged", True):
        raise OESError("The selection probit did not converge.")
    zg = zn @ pr.params
    lam = stats.norm.pdf(zg) / stats.norm.cdf(zg)
    # ---- two-step
    X2 = np.column_stack([xn[s], lam[s]])
    if np.linalg.matrix_rank(X2) < X2.shape[1]:
        raise OESError("The inverse Mills ratio is collinear with the outcome regressors: add a variable to the selection equation that is not in the outcome equation.")
    b2 = np.linalg.lstsq(X2, yv[s], rcond=None)[0]
    e = yv[s] - X2 @ b2
    dl = lam[s] * (lam[s] + zg[s])
    bl = b2[-1]
    s2 = float(e @ e / n1 + bl ** 2 * dl.mean())
    rho2 = bl ** 2 / s2
    A = np.linalg.inv(X2.T @ X2)
    Dm = np.diag(dl)
    Q = rho2 * (X2.T @ Dm @ zn[s]) @ np.asarray(pr.cov_params()) @ (zn[s].T @ Dm @ X2)
    V2 = s2 * A @ (X2.T @ (np.eye(n1) - rho2 * Dm) @ X2 + Q) @ A
    se2 = np.sqrt(np.maximum(np.diag(V2), 0))
    sigma2 = float(np.sqrt(s2))
    # ---- full-information ML
    rho0 = float(np.clip(bl / sigma2, -0.9, 0.9))
    t0 = np.r_[pr.params, b2[:-1], np.log(sigma2), np.arctanh(rho0)]
    fun = lambda t: _heckman_nll(t, zn, xn, yv, s, kz, kx)
    opt = optimize.minimize(fun, t0, method="BFGS", options={"gtol": 1e-7, "maxiter": 3000})
    opt = optimize.minimize(fun, opt.x, method="BFGS", options={"gtol": 1e-8, "maxiter": 3000})
    if not np.isfinite(opt.fun) or np.linalg.norm(opt.jac) > 1e-2 * max(1, n):
        raise OESError("The maximum-likelihood iterations did not converge. Check the scale of the variables and the exclusion restriction.")
    t = opt.x
    try:
        V = np.linalg.inv(approx_hess(t, fun))
    except np.linalg.LinAlgError:
        raise OESError("The information matrix is singular: the model is not identified (add an exclusion restriction) or the regressors are collinear.")
    if np.any(np.diag(V) <= 0):
        raise OESError("The information matrix is not positive definite; the estimate is not a maximum. Try removing the weakest variable.")
    se = np.sqrt(np.diag(V))
    g, b = t[:kz], t[kz:kz + kx]
    sig, rho = float(np.exp(t[kz + kx])), float(np.tanh(t[kz + kx + 1]))
    se_sig, se_rho = sig * se[kz + kx], (1 - rho ** 2) * se[kz + kx + 1]
    # delta method for lambda = rho * sigma
    gl = np.zeros(len(t)); gl[kz + kx] = rho * sig; gl[kz + kx + 1] = sig * (1 - rho ** 2)
    lam_ml, se_lam = rho * sig, float(np.sqrt(gl @ V @ gl))
    ll = -float(opt.fun)
    # ---- independence benchmark (rho = 0): probit + OLS on the selected observations
    ols = sm.OLS(yv[s], xn[s]).fit()
    s_ml = float(np.sqrt(ols.ssr / n1))
    ll0 = pr.llf + float(np.sum(stats.norm.logpdf(ols.resid / s_ml) - np.log(s_ml)))
    lr = max(2 * (ll - ll0), 0.0)
    out = coef_table(xn_names, b, se[kz:kz + kx], dist="z", title="Outcome equation (maximum likelihood)",
                     note=f"Dependent variable: {y}, observed for {n1} of {n} observations. Full-information maximum likelihood.")
    for lab, v, sv in [("sigma", sig, se_sig), ("rho", rho, se_rho), ("lambda = rho x sigma", lam_ml, se_lam)]:
        out["rows"].append([lab, f(v), f(sv), f(v / sv, 4), f(2 * stats.norm.sf(abs(v / sv)), 4), f(v - 1.96 * sv), f(v + 1.96 * sv)])
    tabs = [out, coef_table(zn_names, g, se[:kz], dist="z", title=f"Selection equation (probit for {sel} = 1)",
                            note="Maximum-likelihood probit coefficients from the joint model.")]
    cmp_rows = []
    for i, nm in enumerate(xn_names):
        cmp_rows.append([nm, f(b2[i]), f(se2[i]), f(b[i]), f(se[kz + i]), f(ols.params[i])])
    cmp_rows.append(["lambda (inverse Mills ratio)", f(bl), f(se2[-1]), f(lam_ml), f(se_lam), None])
    tabs.append(table("Two-step estimator, maximum likelihood and OLS on the selected sample",
                      ["Variable", "Two-step coef.", "Two-step s.e.", "ML coef.", "ML s.e.", "OLS (selected only)"], cmp_rows,
                      "Two-step standard errors are corrected for the estimated inverse Mills ratio (Heckman 1979; Greene). OLS ignores selection."))
    tt = bl / se2[-1]
    drows = [test_row("Likelihood-ratio test of independent equations, rho = 0 (1 df)", lr, stats.chi2.sf(lr, 1), "H0: no selection bias; if not rejected, OLS on the selected sample is consistent"),
             test_row("Wald test, rho = 0 (ML)", (rho / se_rho) ** 2, stats.chi2.sf((rho / se_rho) ** 2, 1), "H0: rho = 0"),
             test_row("t test, coefficient on the inverse Mills ratio (two-step)", tt, 2 * stats.norm.sf(abs(tt)), "H0: no selection bias (two-step version)")]
    tabs.append(tests_table(drows))
    xz_only = [c for c in zs if c not in xs]
    q = pd.qcut(pd.Series(zn @ g).rank(method="first"), 10, labels=False)
    pg = stats.norm.cdf(zn @ g)
    ch = chart("Share selected by decile of the selection index: observed and predicted", [str(i + 1) for i in range(10)],
               [("Observed", [float(s[q == k].mean()) for k in range(10)]), ("Predicted", [float(pg[q == k].mean()) for k in range(10)])],
               "Decile of z'g", "Share with the outcome observed", kind="bar")
    summary = [stat("Observations", n), stat("Selected (outcome observed)", n1), stat("Log-likelihood", ll), stat("rho", rho), stat("sigma", sig), stat("LR test of rho = 0", lr)]
    notes = [f"Heckman selection model: {y} observed for {n1} of {n} observations (selection equation: {sel}).",
             ("Exclusion restriction(s): " + ", ".join(xz_only) + " enter the selection equation only.") if xz_only else
             "WARNING: every selection variable is also in the outcome equation. The model is then identified only by the nonlinearity of the inverse Mills ratio, which is fragile: add a variable that affects selection but not the outcome.",
             "The model assumes that the errors of the two equations are jointly normal. If rho is close to zero, selection is ignorable and OLS on the selected sample is consistent."]
    return result("heckman", f"Heckman selection model: {y}", n, summary, tabs, [ch], notes)


def _count_pmf(mu, family, alpha, kmax):
    ks = np.arange(kmax + 1)[:, None]
    if family == "poisson":
        return stats.poisson.pmf(ks, mu[None, :])
    r = 1.0 / alpha
    return stats.nbinom.pmf(ks, r, r / (r + mu[None, :]))


def count(df, spec):
    """Poisson and negative-binomial (NB2) regression for a count outcome, with incidence-rate ratios, marginal effects and dispersion tests."""
    y = need(spec, "y", "count outcome variable")
    xs = as_list(spec.get("x"))
    if not xs:
        raise OESError("Choose at least one explanatory variable.")
    fam = _choice(spec, "family", ["poisson", "negbin"], "poisson", "The family")
    cov = _choice(spec, "cov", ["model", "robust"], "robust", "The standard errors")
    expo = spec.get("exposure") if spec.get("exposure") not in (None, "") else None
    distinct(y, xs, [expo] if expo else [])
    d = numeric(df, [y] + xs + ([expo] if expo else [])).reset_index(drop=True)
    Y = d[y].to_numpy(float)
    if (Y < 0).any() or not np.allclose(Y, np.round(Y)):
        raise OESError(f"{y} must contain non-negative whole numbers (counts).")
    if Y.sum() == 0:
        raise OESError(f"{y} is zero for every observation.")
    off = None
    if expo:
        if (d[expo] <= 0).any():
            raise OESError("The exposure variable must be positive.")
        off = np.log(d[expo].to_numpy(float))
    n, k = len(d), len(xs)
    if n < 5 * (k + 1):
        raise OESError("Too few observations for this number of variables.")
    X = sm.add_constant(d[xs].astype(float), has_constant="add")
    names = list(X.columns)
    Xn = np.asarray(X, float)
    kw = dict(cov_type="HC0") if cov == "robust" else {}
    try:
        pois = sm.GLM(Y, X, family=sm.families.Poisson(), offset=off).fit(**kw)
    except Exception as e:
        raise OESError(f"The Poisson model could not be estimated ({e}). Check for collinear or constant variables.")
    if fam == "poisson":
        res, beta, V, alpha = pois, np.asarray(pois.params), np.asarray(pois.cov_params()), None
        mu = np.asarray(pois.mu)
        ll = float(sm.GLM(Y, X, family=sm.families.Poisson(), offset=off).fit().llf)
        se = np.asarray(pois.bse)
    else:
        try:
            m = sm.NegativeBinomial(Y, X, loglike_method="nb2", offset=off)
            start = np.r_[np.asarray(pois.params), 0.5]
            res = m.fit(start_params=start, method="bfgs", maxiter=2000, disp=0, **kw)
            res = m.fit(start_params=res.params, method="newton", maxiter=200, disp=0, **kw) if False else res
        except Exception as e:
            raise OESError(f"The negative-binomial model did not converge ({e}).")
        if not res.mle_retvals.get("converged", True):
            raise OESError("The negative-binomial likelihood did not converge. Try fewer variables or rescale them.")
        beta, alpha = np.asarray(res.params)[:-1], float(np.asarray(res.params)[-1])
        V = np.asarray(res.cov_params())
        se = np.sqrt(np.diag(V))
        mu = np.exp(Xn @ beta + (off if off is not None else 0))
        ll = float(res.llf)
    if not np.all(np.isfinite(se)):
        raise OESError("Standard errors could not be computed (collinear variables or perfect prediction).")
    nb = len(beta)
    cof = coef_table(names + (["alpha (dispersion)"] if fam == "negbin" else []), np.r_[beta, [alpha] if fam == "negbin" else []],
                     se, dist="z", note=("Poisson" if fam == "poisson" else "Negative binomial (NB2)") + " regression; the coefficient is the change in the log expected count. "
                     + ("Robust (sandwich) standard errors." if cov == "robust" else "Model-based standard errors."))
    tabs = [cof]
    # incidence-rate ratios
    irr = np.exp(beta)
    tabs.append(table("Incidence-rate ratios exp(b)", ["Variable", "IRR", "Std. err.", "95% CI low", "95% CI high"],
                      [[names[i], f(irr[i]), f(irr[i] * se[i]), f(np.exp(beta[i] - 1.96 * se[i])), f(np.exp(beta[i] + 1.96 * se[i]))] for i in range(1, nb)],
                      "An IRR of 1.10 means that a one-unit increase raises the expected count by 10%. Delta-method standard errors."))
    # average marginal effects
    dummy = [bool(set(d[c].unique()) <= {0, 1}) for c in xs]
    ofs = off if off is not None else 0

    def ame(b):
        out = np.zeros(k)
        for i in range(k):
            if dummy[i]:
                A1, A0 = Xn.copy(), Xn.copy(); A1[:, i + 1], A0[:, i + 1] = 1, 0
                out[i] = (np.exp(A1 @ b + ofs) - np.exp(A0 @ b + ofs)).mean()
            else:
                out[i] = (b[i + 1] * np.exp(Xn @ b + ofs)).mean()
        return out
    am = ame(beta)
    Vb = V[:nb, :nb]
    J = _num_jac(ame, beta)
    sa = np.sqrt(np.maximum(np.diag(J @ Vb @ J.T), 0))
    tabs.append(table("Average marginal effects on the expected count", ["Variable", "dE[y]/dx", "Std. err.", "z", "p-value"],
                      [[xs[i], f(am[i]), f(sa[i]), f(am[i] / sa[i], 4) if sa[i] > 0 else None, f(2 * stats.norm.sf(abs(am[i] / sa[i]))) if sa[i] > 0 else None] for i in range(k)],
                      "Average of the observation-level effects. Discrete change for 0/1 variables."))
    # dispersion diagnostics from the Poisson fit
    mu_p = np.asarray(pois.mu)
    pear = float(((Y - mu_p) ** 2 / mu_p).sum())
    dev = float(pois.deviance)
    dfr = n - (k + 1)
    aux = ((Y - mu_p) ** 2 - Y) / mu_p
    ct = sm.OLS(aux, mu_p).fit()                       # Cameron-Trivedi (1990) test of Var = mu against Var = mu + alpha mu^2
    drows = [test_row("Cameron-Trivedi overdispersion test (Poisson)", float(ct.tvalues[0]), 2 * stats.norm.sf(abs(float(ct.tvalues[0]))), "H0: variance = mean (equidispersion)"),
             ["Pearson chi-squared / df (Poisson)", f(pear / dfr), None, "about 1 under equidispersion; much larger means overdispersion"],
             ["Deviance / df (Poisson)", f(dev / dfr), None, "about 1 under a correct Poisson model"]]
    if fam == "negbin":
        lr = max(2 * (ll - float(sm.GLM(Y, X, family=sm.families.Poisson(), offset=off).fit().llf)), 0.0)
        drows.append(["LR test alpha = 0 (NB against Poisson)", f(lr, 4), f(0.5 * stats.chi2.sf(lr, 1), 4), "boundary test: the p-value is halved"])
    drows.append(["Share of zeros: observed against predicted", f(float((Y == 0).mean())), f(float(_count_pmf(mu, fam, alpha, 0)[0].mean())),
                  "an excess of observed zeros points to a zero-inflated or hurdle model"])
    tabs.append(tests_table(drows))
    kmax = int(min(max(Y.max(), 5), 10))
    obs = [float((Y == j).mean()) for j in range(kmax)] + [float((Y >= kmax).mean())]
    pm = _count_pmf(mu, fam, alpha, kmax - 1).mean(axis=1)
    pred = list(pm) + [float(1 - pm.sum())]
    ch = chart("Observed and predicted distribution of the count", [str(j) for j in range(kmax)] + [f"{kmax}+"], [("Observed", obs), ("Predicted", pred)], y, "Share of observations", kind="bar")
    summary = [stat("Observations", n), stat("Log-likelihood", ll), stat("Mean of the count", float(Y.mean())), stat("Variance of the count", float(Y.var(ddof=1)))]
    if fam == "negbin":
        summary.append(stat("alpha", alpha))
    notes = [f"{'Poisson' if fam == 'poisson' else 'Negative-binomial'} regression of {y} on {', '.join(xs)}: {n} observations; mean {Y.mean():.3g}, variance {Y.var(ddof=1):.3g}.",
             "Poisson regression is consistent for the mean whenever the mean is correctly specified, so use robust standard errors; it is efficient only if variance equals mean. "
             "The negative binomial adds a dispersion parameter alpha and is the standard model for overdispersed counts."]
    return result("count", f"{'Poisson' if fam == 'poisson' else 'Negative-binomial'} regression: {y}", n, summary, tabs, [ch], notes)
