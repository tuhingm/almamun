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
