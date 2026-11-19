"""OLS publication report: the steps of a regression analysis in the order a researcher works.

1 Data and descriptive statistics   2 Correlations   3 Estimation (robust SE options, specifications side by side)
4 Effect sizes and marginal effects   5 Multicollinearity   6 Residual diagnostics (serial correlation, heteroskedasticity,
normality, functional form)   7 Parameter stability (CUSUM)   8 Influence   9 Added-variable (partial regression) plots.

The estimation calls are exactly those of timeseries.ols (same data preparation, same statsmodels fit, same covariance option),
so the coefficients of the report equal the coefficients of the standard OES result; the layer only adds steps around them.
"""
import math

import numpy as np
import pandas as pd
from scipy import stats

from . import rep_tables as T
from . import rep_text as X
from .core import OESError, as_list, distinct, int_option, need
from .rep_model import Figure, Report, Step, TestResult
from .rep_tables import MINUS, fnum, fp, make_table, regression_table, tests_table

REFS = {
    'white1980': 'White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. Econometrica, 48(4), 817-838.',
    'mackinnon1985': 'MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. Journal of Econometrics, 29(3), 305-325.',
    'neweywest1987': 'Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. Econometrica, 55(3), 703-708.',
    'breuschpagan1979': 'Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. Econometrica, 47(5), 1287-1294.',
    'breusch1978': 'Breusch, T. S. (1978). Testing for autocorrelation in dynamic linear models. Australian Economic Papers, 17(31), 334-355.',
    'godfrey1978': 'Godfrey, L. G. (1978). Testing against general autoregressive and moving average error models when the regressors include lagged dependent variables. Econometrica, 46(6), 1293-1301.',
    'engle1982': 'Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. Econometrica, 50(4), 987-1007.',
    'jarquebera1980': 'Jarque, C. M., & Bera, A. K. (1980). Efficient tests for normality, homoscedasticity and serial independence of regression residuals. Economics Letters, 6(3), 255-259.',
    'ramsey1969': 'Ramsey, J. B. (1969). Tests for specification errors in classical linear least-squares regression analysis. Journal of the Royal Statistical Society: Series B, 31(2), 350-371.',
    'durbinwatson1950': 'Durbin, J., & Watson, G. S. (1950). Testing for serial correlation in least squares regression: I. Biometrika, 37(3-4), 409-428.',
    'brown1975': 'Brown, R. L., Durbin, J., & Evans, J. M. (1975). Techniques for testing the constancy of regression relationships over time. Journal of the Royal Statistical Society: Series B, 37(2), 149-192.',
    'ploberger1992': 'Ploberger, W., & Kramer, W. (1992). The CUSUM test with OLS residuals. Econometrica, 60(2), 271-285.',
    'cook1977': 'Cook, R. D. (1977). Detection of influential observation in linear regression. Technometrics, 19(1), 15-18.',
    'belsley1980': 'Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). Regression Diagnostics: Identifying Influential Data and Sources of Collinearity. Wiley.',
    'frischwaugh': 'Frisch, R., & Waugh, F. V. (1933). Partial time regressions as compared with individual trends. Econometrica, 1(4), 387-401.',
    'shapirowilk1965': 'Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). Biometrika, 52(3-4), 591-611.',
    'ljungbox1978': 'Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models. Biometrika, 65(2), 297-303.',
    "seabold2010": 'Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. Proceedings of the 9th Python in Science Conference, 92-96.',
    "wooldridge2020": 'Wooldridge, J. M. (2020). Introductory Econometrics: A Modern Approach (7th ed.). Cengage.',
}

COV_LABEL = {
    "nonrobust": "conventional (homoskedastic) standard errors",
    "HC0": "heteroskedasticity-robust (HC0) standard errors",
    "HC1": "heteroskedasticity-robust (HC1) standard errors",
    "HC2": "heteroskedasticity-robust (HC2) standard errors",
    "HC3": "heteroskedasticity-robust (HC3) standard errors",
    "HAC": "Newey-West heteroskedasticity- and autocorrelation-consistent standard errors",
}


def recursive_residuals(y, X):
    """Brown-Durbin-Evans recursive residuals w_t, t = k+1..T, by the recursive least-squares update."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    T_, k = X.shape
    XtX = X[:k].T @ X[:k]
    P = np.linalg.inv(XtX)
    b = P @ X[:k].T @ y[:k]
    w = []
    for t in range(k, T_):
        x = X[t]
        f = 1.0 + x @ P @ x
        w.append((y[t] - x @ b) / math.sqrt(f))
        Px = P @ x
        b = b + Px * (y[t] - x @ b) / f
        P = P - np.outer(Px, Px) / f
    return np.array(w)


def cusum_path(w):
    s = math.sqrt((w ** 2).sum() / len(w))
    return np.cumsum(w) / s


def cusum_bounds(n_w, k, a=0.948):
    r = np.arange(1, n_w + 1)
    return a * (math.sqrt(n_w) + 2 * r / math.sqrt(n_w))


def acf(e, nlags):
    e = np.asarray(e, float) - np.mean(e)
    d = (e ** 2).sum()
    return np.array([(e[k:] * e[:len(e) - k]).sum() / d for k in range(nlags + 1)])


def vif_manual(X):
    out = []
    for j in range(X.shape[1]):
        yj = X[:, j]
        Z = np.delete(X, j, axis=1)
        Z = np.column_stack([np.ones(len(Z)), Z])
        r2 = 1 - ((yj - Z @ np.linalg.lstsq(Z, yj, rcond=None)[0]) ** 2).sum() / ((yj - yj.mean()) ** 2).sum()
        out.append(1 / (1 - r2))
    return np.array(out)


def cooks_manual(X, e):
    h = np.einsum("ij,jk,ik->i", X, np.linalg.inv(X.T @ X), X)
    n, k = X.shape
    s2 = (e ** 2).sum() / (n - k)
    return e ** 2 * h / (k * s2 * (1 - h) ** 2), h


def _cov_kw(cov, hac):
    from .timeseries import _cov_kw as ck
    return ck(cov, hac)


def _xticks(labels, n=6):
    m = len(labels)
    idx = sorted(set(int(round(i)) for i in np.linspace(0, m - 1, min(n, m))))
    return [[float(i), labels[i]] for i in idx]


def build(df, spec):
    import sys
    import statsmodels
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox, breaks_cusumolsresid, het_arch, het_breuschpagan, het_white, linear_reset
    from statsmodels.stats.outliers_influence import OLSInfluence
    from statsmodels.stats.stattools import durbin_watson, jarque_bera
    from .stat_core import qq_points, thin
    from .timeseries import _series_frame
    from . import __version__

    y = need(spec, "y", "dependent variable")
    xs = as_list(need(spec, "x", "explanatory variables"))
    distinct([y], xs)
    d, labels = _series_frame(df, spec, [y] + xs)
    k = len(xs) + 1
    n = len(d)
    if n < k + 3:
        raise OESError("Too few complete observations for this regression.")
    cov = spec.get("cov", "HC1") or "HC1"
    hac = int_option(spec, "hac_lags", 4, 0, 48, "HAC lags")
    names = ["const"] + xs
    if np.linalg.matrix_rank(d[xs].astype(float).assign(_c=1.0).to_numpy()) < k:
        raise OESError("The regressors are perfectly collinear (or one is constant), so the step-by-step diagnostics, which need an invertible regressor matrix, cannot be computed. Remove the redundant variable to get the full report.")
    Xd = sm.add_constant(d[xs].astype(float), has_constant="add")
    yv = d[y].astype(float)
    base = sm.OLS(yv, Xd).fit()
    res = sm.OLS(yv, Xd).fit(**_cov_kw(cov, hac))
    robust = cov not in ("nonrobust", None, "")
    Xm = Xd.to_numpy()
    yy = yv.to_numpy()
    e = np.asarray(base.resid)
    fit = np.asarray(base.fittedvalues)
    cov_name = COV_LABEL.get(cov, cov)
    if cov == "HAC":
        cov_name = f"Newey-West heteroskedasticity- and autocorrelation-consistent standard errors with {hac} lags"
    obs = np.arange(n, dtype=float)
    xt = _xticks(labels)
    per = f" from {labels[0]} to {labels[-1]}" if spec.get("time") else ""
    xl_obs = "Observation" if not spec.get("time") else ""

    rep = Report("ols", "OLS regression of " + y + " on " + ", ".join(xs), references=REFS)
    rep.meta = dict(y=y, x=xs, nobs=n, nobs_input=int(len(df)), cov=cov, engine=__version__, statsmodels=statsmodels.__version__,
                    python=".".join(map(str, sys.version_info[:3])), period=(labels[0] + " to " + labels[-1]) if spec.get("time") else "")
    rep.meta["extra_refs"] = ["seabold2010", "wooldridge2020"]
    allv = [y] + xs

    # ------------------------------------------------------------------ 1 data
    s1 = Step("data", "Data and descriptive statistics", "Sample, scale and distribution of every variable before any model is estimated.",
              validated="pandas/scipy: mean, sd, quartiles, bias-adjusted skewness and kurtosis recomputed independently (test_descriptives)", refs=["wooldridge2020"])
    s1.tables.append(T.descriptives_table("t_desc", d, allv, "Descriptive statistics"))
    npan = len(allv)
    s1.figures.append(Figure("f_series", "Time series of the variables" if spec.get("time") else "The variables in observation order",
                             {"ncols": 2 if npan > 1 else 1, "panels": [{"title": v, "xlabel": xl_obs, "ylabel": "", "xticks": xt, "zero": False,
                                                                       "layers": [{"type": "line", "x": obs.tolist(), "y": d[v].astype(float).tolist(), "k": 0, "lw": 0.8}]} for v in allv]},
                             "Each panel plots one variable in the units of the data.", "Source: author's calculations from the data supplied.", 6.5, 2.1 * math.ceil(npan / 2) + 0.5))
    panels = []
    for v in allv:
        x = d[v].astype(float).to_numpy()
        nb = int(min(30, max(6, round(math.sqrt(n)))))
        cnt, edges = np.histogram(x, bins=nb, density=True)
        mid = (edges[:-1] + edges[1:]) / 2
        lay = [{"type": "bar", "x": mid.tolist(), "y": cnt.tolist(), "width": float(edges[1] - edges[0]), "fill": "#c8c8c8"}]
        if x.std(ddof=1) > 0:
            g = np.linspace(edges[0], edges[-1], 80)
            lay.append({"type": "line", "x": g.tolist(), "y": stats.norm.pdf(g, x.mean(), x.std(ddof=1)).tolist(), "k": 2, "lw": 0.9})
        panels.append({"title": v, "xlabel": "", "ylabel": "Density", "layers": lay})
    s1.figures.append(Figure("f_hist", "Distribution of the variables", {"ncols": 2 if npan > 1 else 1, "panels": panels},
                             "Histograms; the dashed line is the normal density with the sample mean and standard deviation.", "Source: author's calculations.", 6.5, 2.1 * math.ceil(npan / 2) + 0.5))
    miss = int(len(df) - n)
    s1.paragraphs.append(X.P("result", "The estimation sample has {n} observations{per}. {miss_txt}The dependent variable {y} has a mean of {m:.3f} and a standard deviation of {s:.3f}.",
                             n=n, per=per, y=y, m=float(yy.mean()), s=float(yy.std(ddof=1)), miss=miss,
                             miss_txt=(f"{miss} rows of the file were dropped because a selected variable was missing or not numeric. " if miss > 0 else "")))
    rep.steps.append(s1)

    # ------------------------------------------------------------------ 2 correlations
    s2 = Step("corr", "Correlations", "Pairwise linear association, including a first look at collinearity among the regressors.",
              validated="scipy.stats.pearsonr recomputed with numpy (test_correlations)", refs=["wooldridge2020"])
    s2.tables.append(T.correlation_table("t_corr", d, allv, "Correlation matrix"))
    C = d[allv].astype(float).corr().to_numpy()
    s2.figures.append(Figure("f_corr", "Correlation matrix of the variables",
                             {"ncols": 1, "panels": [{"layers": [{"type": "heatmap", "matrix": C.tolist(), "xlabels": allv, "ylabels": allv}], "ml": 70}]},
                             "Cells show Pearson correlations. Blue is negative, orange positive.", "Source: author's calculations.", 4.6, 4.0))
    best = max((abs(C[0, j]), C[0, j], allv[j]) for j in range(1, len(allv)))
    txt = "The regressor most strongly correlated with {y} is {v} (r = {r:.3f}). "
    kw = dict(y=y, v=best[2], r=float(best[1]))
    if len(xs) > 1:
        pr = max((abs(C[i, j]), C[i, j], allv[i], allv[j]) for i in range(1, len(allv)) for j in range(1, i))
        txt += "Among the regressors, the largest pairwise correlation is {rr:.3f} (between {a} and {b})."
        kw.update(rr=float(pr[1]), a=pr[2], b=pr[3])
    s2.paragraphs.append(X.P("result", txt, **kw))
    rep.steps.append(s2)

    # ------------------------------------------------------------------ 3 estimation
    s3 = Step("estimation", "Estimation", "Coefficients with the chosen covariance estimator, robustness of inference to the covariance choice, and nested specifications.",
              validated="statsmodels OLS vs closed-form (X'X)^-1 X'y, HC0-HC3 sandwich and Newey-West formulas recomputed in numpy (test_estimation); equals oes_engine.run('ols') (test_matches_engine)",
              refs=["white1980", "mackinnon1985", "neweywest1987", "seabold2010", "wooldridge2020"])

    def colstats(b):
        return dict(n=int(b.nobs), r2=float(b.rsquared), ar2=float(b.rsquared_adj), F=float(b.fvalue), pF=float(b.f_pvalue), ll=float(b.llf), aic=float(b.aic), bic=float(b.bic),
                    rmse=float(math.sqrt(b.mse_resid)))

    stat_rows = [("Observations", "n", "int"), ("R²", "r2", "num3"), ("Adjusted R²", "ar2", "num3"), ("F statistic", "F", "num2"), ("Prob (F)", "pF", "p"),
                 ("Log likelihood", "ll", "num2"), ("AIC", "aic", "num2"), ("BIC", "bic", "num2"), ("Residual std. error", "rmse", "num3")]
    t_main = regression_table("t_reg", "OLS estimates: dependent variable " + y,
                              [dict(label="(1)", sub="OLS", params=res.params, se=res.bse, p=res.pvalues, stats=colstats(base))], names, stat_rows=stat_rows, dep_label="Dependent variable: " + y)
    t_main.notes = ["Dependent variable: " + y + ".", "Standard errors in parentheses are " + cov_name + ".",
                    "R², adjusted R², F, information criteria and the residual standard error come from the conventional fit; they do not depend on the covariance option.", T.STAR_NOTE]
    s3.tables.append(t_main)

    covs = [("Conventional", "nonrobust"), ("HC1", "HC1"), ("HC3", "HC3"), (f"HAC({hac})", "HAC")]
    fits = {}
    cc = []
    for i, (lab_, cv_) in enumerate(covs):
        r_ = sm.OLS(yv, Xd).fit(**_cov_kw(cv_, hac))
        fits[cv_] = r_
        cc.append(dict(label=f"({i + 1})", sub=lab_, params=r_.params, se=r_.bse, p=r_.pvalues, stats=colstats(base)))
    t_cov = regression_table("t_cov", "Sensitivity of inference to the covariance estimator", cc, names, stat_rows=[("Observations", "n", "int")], dep_label="Dependent variable: " + y)
    t_cov.notes = ["Dependent variable: " + y + ".", "Standard errors in parentheses.", "The coefficients are identical in every column; only the standard errors, and so the stars, change.", T.STAR_NOTE]
    s3.tables.append(t_cov)
    if len(xs) >= 2:
        nc = []
        for j in range(1, len(xs) + 1):
            sub = sm.OLS(yv, sm.add_constant(d[xs[:j]].astype(float), has_constant="add")).fit(**_cov_kw(cov, hac))
            nc.append(dict(label=f"({j})", sub="", params=sub.params, se=sub.bse, p=sub.pvalues, stats=colstats(sub)))
        t_nest = regression_table("t_nested", "Specifications adding one regressor at a time", nc, names, stat_rows=stat_rows[:3], dep_label="Dependent variable: " + y)
        t_nest.notes = ["Dependent variable: " + y + ".", "Standard errors are " + cov_name + ", in parentheses.", "Regressors enter in the order selected.", T.STAR_NOTE]
        s3.tables.append(t_nest)
    ci = np.asarray(res.conf_int())
    ypos = [float(len(xs) - 1 - i) for i in range(len(xs))]
    ml_ = 20 + 5.2 * min(24, max(len(v) for v in xs))
    s3.figures.append(Figure("f_coef", "Coefficient estimates with 95 per cent confidence intervals",
                             {"ncols": 1, "panels": [{"xlabel": "Coefficient on the regressor (dependent variable: " + y + ")", "yticks": [[ypos[i], xs[i]] for i in range(len(xs))],
                                                      "ylim": [-0.7, len(xs) - 0.3], "ml": ml_,
                                                      "layers": [{"type": "errh", "y": ypos, "x": [float(res.params[v]) for v in xs], "lo": [float(ci[i + 1, 0]) for i in range(len(xs))],
                                                                  "hi": [float(ci[i + 1, 1]) for i in range(len(xs))]}, {"type": "vline", "x": 0.0}]}]},
                             "Points are OLS estimates; bars are 95 per cent intervals based on the " + cov_name + ". The intercept is omitted.", "Source: author's calculations.", 5.2, max(2.2, 0.45 * len(xs) + 1.2)))
    s3.figures.append(Figure("f_fit", "Actual and fitted values",
                             {"ncols": 1, "panels": [{"xlabel": xl_obs, "ylabel": y, "xticks": xt, "legend": True,
                                                      "layers": [{"type": "line", "x": obs.tolist(), "y": yy.tolist(), "k": 0, "label": "Actual", "lw": 0.9},
                                                                 {"type": "line", "x": obs.tolist(), "y": fit.tolist(), "k": 2, "label": "Fitted", "lw": 0.9}]}]},
                             "Fitted values are from the full specification.", "Source: author's calculations.", 6.5, 3.0))
    for j, v in enumerate(xs[:6], 1):
        b = float(res.params[v])
        lo_, hi_ = float(ci[j, 0]), float(ci[j, 1])
        pj = float(res.pvalues[v])
        s3.paragraphs.append(X.P("result", "{c} has a {dr} coefficient of {b:.3f} (SE = {se:.3f}; 95% CI [{lo:.3f}, {hi:.3f}]; {pv}), which {sg} at the 5% level. A one-unit higher {c} is associated with a {b2:.3f}-unit {chg} in {y}, holding the other regressors constant.",
                                 c=v, dr=X.direction(b), b=b, se=float(res.bse[v]), lo=lo_, hi=hi_, pv=X.pv(pj), sg=X.sig_phrase(pj), b2=abs(b), chg="increase" if b > 0 else "decrease", y=y))
    s3.paragraphs.append(X.P("result", "The regression explains {r2:.3f} of the variance of {y} (adjusted R² = {ar2:.3f}; F({k1}, {df}) = {F:.2f}, {pF}).",
                             r2=float(base.rsquared), y=y, ar2=float(base.rsquared_adj), k1=int(base.df_model), df=int(base.df_resid), F=float(base.fvalue), pF=X.pv(float(base.f_pvalue))))
    if base.rsquared < 0.1:
        s3.paragraphs.append(X.P("warning", "The fit is weak (R² = {r2:.3f}): the regressors account for little of the variation in {y}, so predictions from this model will be imprecise.", r2=float(base.rsquared), y=y))
    cmp_name = cov if robust else "HC3"
    cmp_fit = res if robust else fits["HC3"]
    for v in xs:
        p0 = float(fits["nonrobust"].pvalues[v])
        p1 = float(cmp_fit.pvalues[v])
        if (p0 < 0.05) != (p1 < 0.05):
            s3.paragraphs.append(X.P("warning", "Statistical significance of {c} at the 5% level depends on the covariance estimator (conventional p = {p0:.3f}; {cv} p = {p1:.3f}); report the robust result and treat the conclusion with care.",
                                     c=v, p0=p0, cv=cmp_name, p1=p1))
    rep.steps.append(s3)

    # ------------------------------------------------------------------ 4 effects
    s4 = Step("effects", "Effect sizes and marginal effects", "Size of each association in practical terms: per standard deviation, standardised, and the predicted outcome over the range of each regressor.",
              validated="standardised coefficients, partial R-squared and prediction standard errors recomputed in numpy; prediction CI equals the delta-method quadratic form x'Vx (test_effects)", refs=["wooldridge2020"])
    sdy = float(yy.std(ddof=1))
    erows, trows = [], []
    for v in xs:
        bj = float(res.params[v])
        sdx = float(d[v].astype(float).std(ddof=1))
        tj = float(base.tvalues[v])
        pr2 = tj ** 2 / (tj ** 2 + base.df_resid)
        erows.append(dict(var=v, b=bj, sd_x=sdx, effect_1sd=bj * sdx, beta_std=bj * sdx / sdy, partial_r2=pr2))
        trows.append([v, fnum(bj, 3), fnum(sdx, 3), fnum(bj * sdx, 3), fnum(bj * sdx / sdy, 3), fnum(pr2, 3)])
    s4.tables.append(make_table("t_eff", "Effect sizes", ["Variable", "Coefficient", "Std. dev. of x", "Effect of +1 s.d. of x", "Standardised coefficient", "Partial R²"], trows,
                                ["Effect of +1 s.d. is in the units of " + y + "; the standardised coefficient is that effect divided by the standard deviation of " + y + f" ({sdy:.3f}). Partial R² = t²/(t²+d.f.) from the conventional t-statistic.",
                                 "The model is linear without interactions or transformations of the regressors, so the marginal effect of each regressor equals its coefficient at every point."], raw={"rows": erows}))
    V = np.asarray(res.cov_params())
    crit = stats.norm.ppf(0.975) if robust else stats.t.ppf(0.975, base.df_resid)
    means_x = Xm.mean(axis=0)
    pp = []
    for j, v in enumerate(xs, 1):
        lo_x, hi_x = np.percentile(Xm[:, j], [2.5, 97.5])
        if hi_x <= lo_x:
            lo_x, hi_x = float(Xm[:, j].min()), float(Xm[:, j].max()) + 1e-9
        g = np.linspace(lo_x, hi_x, 40)
        Z = np.tile(means_x, (40, 1))
        Z[:, j] = g
        mu = Z @ np.asarray(res.params)
        se = np.sqrt(np.einsum("ij,jk,ik->i", Z, V, Z))
        pp.append({"title": v, "xlabel": v, "ylabel": "Predicted " + y,
                   "layers": [{"type": "band", "x": g.tolist(), "lo": (mu - crit * se).tolist(), "hi": (mu + crit * se).tolist()}, {"type": "line", "x": g.tolist(), "y": mu.tolist(), "k": 0}]})
    nc_ = 2 if len(xs) > 1 else 1
    s4.figures.append(Figure("f_pred", "Predicted value of the dependent variable over the range of each regressor", {"ncols": nc_, "panels": pp},
                             "Other regressors are held at their sample means; the horizontal range is the 2.5th to 97.5th percentile of the regressor; the band is a 95 per cent confidence interval for the mean prediction.",
                             "Source: author's calculations.", 6.5, 2.3 * math.ceil(len(xs) / nc_) + 0.4))
    top = max(erows, key=lambda r: abs(r["beta_std"]))
    s4.paragraphs.append(X.P("result", "In standardised terms the largest association is that of {c}: a one-standard-deviation higher {c} is associated with a {e:.3f}-unit change in {y}, i.e. {bs:.3f} standard deviations of {y}.",
                             c=top["var"], e=top["effect_1sd"], y=y, bs=top["beta_std"]))
    rep.steps.append(s4)

    # ------------------------------------------------------------------ 5 multicollinearity
    if len(xs) > 1:
        s5 = Step("vif", "Multicollinearity", "Variance inflation factors and the condition number of the scaled regressor matrix.",
                  validated="VIF recomputed from auxiliary regressions (1/(1-R2_j)) in numpy; condition number from the SVD (test_vif)", refs=["belsley1980"])
        vifs = vif_manual(Xm[:, 1:])
        Zs = Xm / np.sqrt((Xm ** 2).sum(axis=0))
        sv = np.linalg.svd(Zs, compute_uv=False)
        cn = float(sv[0] / sv[-1])
        s5.tables.append(make_table("t_vif", "Variance inflation factors", ["Variable", "VIF", "Tolerance (1/VIF)"], [[xs[i], fnum(vifs[i], 2), fnum(1 / vifs[i], 3)] for i in range(len(xs))],
                                    ["Condition number of the column-scaled regressor matrix including the constant: " + f"{cn:.1f}" + ". VIF above 10 or a condition number above 30 are the usual warning levels (Belsley, Kuh and Welsch 1980)."],
                                    raw={"vif": {xs[i]: float(vifs[i]) for i in range(len(xs))}, "condition_number": cn}))
        s5.figures.append(Figure("f_vif", "Variance inflation factor",
                                 {"ncols": 1, "panels": [{"xlabel": "VIF", "yticks": [[ypos[i], xs[i]] for i in range(len(xs))], "ylim": [-0.7, len(xs) - 0.3],
                                                          "xlim": [0, max(11.0, float(vifs.max()) * 1.1)], "ml": ml_,
                                                          "layers": [{"type": "errh", "y": ypos, "x": vifs.tolist(), "lo": [1.0] * len(xs), "hi": vifs.tolist()}, {"type": "vline", "x": 10.0}]}]},
                                 "Bars run from 1 to the VIF; the dashed line marks the conventional threshold of 10.", "Source: author's calculations.", 5.2, max(2.2, 0.45 * len(xs) + 1.2)))
        iv = int(np.argmax(vifs))
        if vifs[iv] > 10:
            s5.paragraphs.append(X.P("warning", "Multicollinearity is a concern: the VIF of {c} is {v:.2f}, above the usual threshold of 10, so its standard error is inflated and individual coefficients are imprecisely estimated.", c=xs[iv], v=float(vifs[iv])))
        else:
            s5.paragraphs.append(X.P("result", "Multicollinearity does not appear to be a problem: the largest VIF is {v:.2f} ({c}), below 10, and the condition number is {cn:.1f}.", v=float(vifs[iv]), c=xs[iv], cn=cn))
        rep.steps.append(s5)

    # ------------------------------------------------------------------ 6 diagnostics
    s6 = Step("diag", "Residual diagnostics", "Tests of the classical assumptions on the conventional OLS residuals: no serial correlation, constant variance, normal errors, correct functional form.",
              validated="Breusch-Godfrey, Breusch-Pagan, White, ARCH LM, Jarque-Bera, Durbin-Watson and Ljung-Box recomputed from their definitions with numpy/scipy; RESET compared with an auxiliary-regression F-test (test_diagnostics)",
              refs=["breusch1978", "godfrey1978", "breuschpagan1979", "white1980", "engle1982", "jarquebera1980", "shapirowilk1965", "ramsey1969", "durbinwatson1950", "ljungbox1978"])
    tests = []

    def add(name, st_, p_, null, dfm="", ref="", problem=True):
        tests.append(TestResult(name, float(st_), float(p_), null, dfm, bool(p_ < 0.05), problem, ref))

    L = 4
    skipped = []
    try:
        bg = acorr_breusch_godfrey(base, nlags=L)
        add(f"Breusch-Godfrey LM ({L} lags)", bg[0], bg[1], "No serial correlation", str(L), "breusch1978")
    except Exception:
        skipped.append("Breusch-Godfrey")
    m = min(10, n // 5)
    try:
        lb = acorr_ljungbox(e, lags=[m], return_df=True)
        add(f"Ljung-Box Q ({m} lags)", lb["lb_stat"].iloc[0], lb["lb_pvalue"].iloc[0], "No residual autocorrelation", str(m), "ljungbox1978")
    except Exception:
        skipped.append("Ljung-Box")
    try:
        bp = het_breuschpagan(e, Xm)
        add("Breusch-Pagan LM", bp[0], bp[1], "Constant variance", str(k - 1), "breuschpagan1979")
    except Exception:
        skipped.append("Breusch-Pagan")
    try:
        wh = het_white(e, Xm)
        add("White test LM", wh[0], wh[1], "Constant variance", "", "white1980")
    except Exception:
        skipped.append("White")
    try:
        ar = het_arch(e, nlags=L)
        add(f"ARCH LM ({L} lags)", ar[0], ar[1], "No ARCH effects", str(L), "engle1982")
    except Exception:
        skipped.append("ARCH")
    jb = jarque_bera(e)
    add("Jarque-Bera", jb[0], jb[1], "Normal errors", "2", "jarquebera1980")
    if n <= 5000 and np.ptp(e) > 0:
        sw = stats.shapiro(e)
        add("Shapiro-Wilk W", sw.statistic, sw.pvalue, "Normal errors", "", "shapirowilk1965")
    try:
        rs = linear_reset(base, power=3, test_type="fitted", use_f=True)
        add("Ramsey RESET (powers 2-3)", float(np.squeeze(rs.fvalue)), float(np.squeeze(rs.pvalue)), "Correct functional form", f"2, {int(base.df_resid) - 2}", "ramsey1969")
    except Exception:
        skipped.append("RESET")
    dw = float(durbin_watson(e))
    tests.append(TestResult("Durbin-Watson d", dw, None, "No first-order autocorrelation (d near 2)", "", None, False, "durbinwatson1950"))
    s6.tests = tests
    s6.tables.append(tests_table("t_diag", "Residual diagnostic tests", tests,
                                 ["Tests use the residuals of the conventional OLS fit. 'Reject' means the null hypothesis is rejected at the 5% level; for these tests rejection signals a violated assumption. The Durbin-Watson statistic has no p-value here: values near 2 indicate no first-order autocorrelation."]
                                 + ([f"Not computed (numerical failure): {', '.join(skipped)}."] if skipped else [])))
    h = np.einsum("ij,jk,ik->i", Xm, np.linalg.inv(Xm.T @ Xm), Xm)
    s_ = math.sqrt(e @ e / (n - k))
    stud_int = e / (s_ * np.sqrt(1 - h))
    fx, ex = thin(fit.tolist(), e.tolist(), 300)
    fx2, sx2 = thin(fit.tolist(), np.sqrt(np.abs(stud_int)).tolist(), 300)
    s6.figures.append(Figure("f_resid1", "Residuals against fitted values, and scale-location",
                             {"ncols": 2, "panels": [{"title": "(a) Residuals vs fitted", "xlabel": "Fitted value", "ylabel": "Residual", "zero": True, "layers": [{"type": "scatter", "x": fx, "y": ex}]},
                                                     {"title": "(b) Scale-location", "xlabel": "Fitted value", "ylabel": "sqrt(|studentized residual|)", "layers": [{"type": "scatter", "x": fx2, "y": sx2}]}]},
                             "A funnel shape in (a) or an upward trend in (b) indicates heteroskedasticity; curvature indicates a misspecified functional form.", "Source: author's calculations.", 6.5, 2.9))
    z, rs_ = qq_points(e)
    qx, qy = thin(z.tolist(), rs_.tolist(), 250)
    q1, q3 = np.percentile(rs_, [25, 75])
    z1, z3 = stats.norm.ppf([0.25, 0.75])
    slope = (q3 - q1) / (z3 - z1)
    icpt = q1 - slope * z1
    zl = [float(z.min()), float(z.max())]
    nb = int(min(30, max(8, round(math.sqrt(n)))))
    cnt, edges = np.histogram(e / e.std(ddof=1), bins=nb, density=True)
    mid = (edges[:-1] + edges[1:]) / 2
    g = np.linspace(edges[0], edges[-1], 80)
    s6.figures.append(Figure("f_resid2", "Normal quantile plot and histogram of the residuals",
                             {"ncols": 2, "panels": [{"title": "(a) Normal Q-Q", "xlabel": "Theoretical quantile", "ylabel": "Standardised residual",
                                                      "layers": [{"type": "scatter", "x": qx, "y": qy}, {"type": "line", "x": zl, "y": [icpt + slope * zl[0], icpt + slope * zl[1]], "k": 2}]},
                                                     {"title": "(b) Histogram", "xlabel": "Standardised residual", "ylabel": "Density",
                                                      "layers": [{"type": "bar", "x": mid.tolist(), "y": cnt.tolist(), "width": float(edges[1] - edges[0]), "fill": "#c8c8c8"},
                                                                 {"type": "line", "x": g.tolist(), "y": stats.norm.pdf(g).tolist(), "k": 2}]}]},
                             "In (a) the line passes through the first and third quartiles; points far from it in the tails indicate non-normal errors. In (b) the dashed line is the standard normal density.",
                             "Source: author's calculations.", 6.5, 2.9))
    mlag = min(20, max(4, n // 4))
    ac = acf(e, mlag)
    bnd = 1.96 / math.sqrt(n)
    s6.figures.append(Figure("f_resid3", "Residual autocorrelation function and residuals over time",
                             {"ncols": 2, "panels": [{"title": "(a) Autocorrelation of residuals", "xlabel": "Lag", "ylabel": "Autocorrelation", "zero": True,
                                                      "layers": [{"type": "stem", "x": [float(i) for i in range(1, mlag + 1)], "y": ac[1:].tolist()}, {"type": "hline", "y": bnd}, {"type": "hline", "y": -bnd}]},
                                                     {"title": "(b) Residuals", "xlabel": xl_obs, "ylabel": "Residual", "xticks": xt, "zero": True,
                                                      "layers": [{"type": "line", "x": obs.tolist(), "y": e.tolist(), "k": 0, "lw": 0.8}]}]},
                             "Dashed lines in (a) are approximate 95 per cent bounds, plus or minus 1.96/sqrt(N).", "Source: author's calculations.", 6.5, 2.9))
    byn = {t.name: t for t in tests}
    bgt = byn.get(f"Breusch-Godfrey LM ({L} lags)")
    if bgt:
        if bgt.reject:
            s6.paragraphs.append(X.P("warning", "The Breusch-Godfrey test rejects the null of no serial correlation (LM = {s:.3f}, {pv}), so the errors appear to be autocorrelated. Conventional standard errors are then invalid; use Newey-West (HAC) standard errors or model the dynamics.", s=bgt.statistic, pv=X.pv(bgt.p_value)))
        else:
            s6.paragraphs.append(X.P("result", "The Breusch-Godfrey test does not reject the null of no serial correlation (LM = {s:.3f}, {pv}).", s=bgt.statistic, pv=X.pv(bgt.p_value)))
    bpt = byn.get("Breusch-Pagan LM")
    if bpt:
        if bpt.reject:
            s6.paragraphs.append(X.P("warning", "The Breusch-Pagan test rejects constant variance (LM = {s:.3f}, {pv}), indicating heteroskedasticity.{e}", s=bpt.statistic, pv=X.pv(bpt.p_value),
                                     e=(" The standard errors in the main table are conventional, so they should not be relied on; re-estimate with a robust covariance option." if not robust else "")))
        else:
            s6.paragraphs.append(X.P("result", "The Breusch-Pagan test does not reject constant variance (LM = {s:.3f}, {pv}).", s=bpt.statistic, pv=X.pv(bpt.p_value)))
    jbt = byn["Jarque-Bera"]
    if jbt.reject:
        s6.paragraphs.append(X.P("warning", "The Jarque-Bera test rejects normality of the residuals (JB = {s:.3f}, {pv}; skewness = {sk:.3f}, kurtosis = {ku:.3f}). With a large sample the coefficient tests remain approximately valid, but small-sample inference and prediction intervals rely on normality.",
                                 s=jbt.statistic, pv=X.pv(jbt.p_value), sk=float(stats.skew(e)), ku=float(stats.kurtosis(e, fisher=False))))
    else:
        s6.paragraphs.append(X.P("result", "The Jarque-Bera test does not reject normality of the residuals (JB = {s:.3f}, {pv}).", s=jbt.statistic, pv=X.pv(jbt.p_value)))
    rst = byn.get("Ramsey RESET (powers 2-3)")
    if rst:
        if rst.reject:
            s6.paragraphs.append(X.P("warning", "The RESET test rejects the null of correct functional form (F = {s:.3f}, {pv}); nonlinear terms, interactions or omitted variables may be needed.", s=rst.statistic, pv=X.pv(rst.p_value)))
        else:
            s6.paragraphs.append(X.P("result", "The RESET test does not reject the linear functional form (F = {s:.3f}, {pv}).", s=rst.statistic, pv=X.pv(rst.p_value)))
    rep.steps.append(s6)

    # ------------------------------------------------------------------ 7 stability
    s7 = Step("stability", "Parameter stability", "Whether the regression relationship is constant over the sample (CUSUM of recursive residuals; OLS-CUSUM test).",
              validated="recursive residuals, CUSUM path and bounds recomputed with a direct numpy implementation and compared with statsmodels recursive_olsresiduals; OLS-CUSUM statistic recomputed from its definition (test_stability)",
              refs=["brown1975", "ploberger1992"])
    cs = breaks_cusumolsresid(e, ddof=k)
    cus = TestResult("OLS-CUSUM (Ploberger-Kramer)", float(cs[0]), float(cs[1]), "Stable parameters", "", bool(cs[1] < 0.05), True, "ploberger1992")
    s7.tests = [cus]
    w = recursive_residuals(yy, Xm)
    path = cusum_path(w)
    bd = cusum_bounds(len(w), k)
    crossed = bool(np.any(np.abs(path) > bd))
    s7.tables.append(tests_table("t_stab", "Parameter stability tests", [cus], ["The recursive CUSUM path crosses its 5% bounds: " + ("yes" if crossed else "no") + "."]))
    tt = [float(i) for i in range(k, n)]
    s7.figures.append(Figure("f_cusum", "Recursive CUSUM with 5 per cent significance bounds",
                             {"ncols": 1, "panels": [{"xlabel": xl_obs, "ylabel": "CUSUM", "xticks": xt, "zero": True, "legend": True,
                                                      "layers": [{"type": "line", "x": tt, "y": path.tolist(), "k": 0, "label": "CUSUM"},
                                                                 {"type": "line", "x": tt, "y": bd.tolist(), "k": 2, "label": "5% bounds"},
                                                                 {"type": "line", "x": tt, "y": (-bd).tolist(), "k": 2}]}]},
                             "The relationship is unstable if the CUSUM path leaves the bounds.", "Source: author's calculations; bounds from Brown, Durbin and Evans (1975).", 6.5, 3.0))
    if cus.reject or crossed:
        s7.paragraphs.append(X.P("warning", "There is evidence of parameter instability: the OLS-CUSUM test gives {s:.3f} ({pv}) and the recursive CUSUM path {cross} its 5% bounds. A single set of coefficients may not describe the whole sample; consider sub-samples or a regime-switching or time-varying-parameter model.",
                                 s=cus.statistic, pv=X.pv(cus.p_value), cross="leaves" if crossed else "stays within"))
    else:
        s7.paragraphs.append(X.P("result", "The data give no evidence of parameter instability: the OLS-CUSUM test gives {s:.3f} ({pv}) and the recursive CUSUM path stays within its 5% bounds.", s=cus.statistic, pv=X.pv(cus.p_value)))
    rep.steps.append(s7)

    # ------------------------------------------------------------------ 8 influence
    s8 = Step("influence", "Influential observations", "Leverage, studentised residuals and Cook's distance; a few observations may drive the estimates.",
              validated="leverage and Cook's distance recomputed from the hat matrix in numpy; DFFITS and studentised residuals compared with statsmodels (test_influence)", refs=["cook1977", "belsley1980"])
    infl = OLSInfluence(base)
    cd, hh = cooks_manual(Xm, e)
    stud = np.asarray(infl.resid_studentized_external)
    dff = np.asarray(infl.dffits[0])
    order = [int(i) for i in np.argsort(-cd)[:5]]
    irows = [[labels[i], fnum(hh[i], 4), fnum(stud[i], 3), fnum(cd[i], 4), fnum(dff[i], 3)] for i in order]
    s8.tables.append(make_table("t_infl", "The five most influential observations", ["Observation", "Leverage", "Studentised residual", "Cook's D", "DFFITS"], irows,
                                [f"Rules of thumb: leverage above 2k/N = {2 * k / n:.4f}; Cook's D above 4/N = {4 / n:.4f}; |studentised residual| above 2."], raw={"order": order}))
    lx, ly = thin(hh.tolist(), stud.tolist(), 300)
    s8.figures.append(Figure("f_infl", "Leverage and Cook's distance",
                             {"ncols": 2, "panels": [{"title": "(a) Leverage vs studentised residual", "xlabel": "Leverage", "ylabel": "Studentised residual", "zero": True,
                                                      "layers": [{"type": "scatter", "x": lx, "y": ly}, {"type": "vline", "x": float(2 * k / n)}]},
                                                     {"title": "(b) Cook's distance", "xlabel": xl_obs, "ylabel": "Cook's D", "xticks": xt,
                                                      "layers": [{"type": "stem", "x": obs.tolist(), "y": cd.tolist()}, {"type": "hline", "y": float(4 / n)}]}]},
                             "Dashed lines mark 2k/N for leverage and 4/N for Cook's distance.", "Source: author's calculations.", 6.5, 2.9))
    i0 = int(np.argmax(cd))
    if cd[i0] > 1:
        s8.paragraphs.append(X.P("warning", "Observation {o} is highly influential (Cook's D = {cd:.3f}, above 1): the estimates may change materially without it. Re-estimate excluding it before relying on the result.", o=labels[i0], cd=float(cd[i0])))
    else:
        s8.paragraphs.append(X.P("result", "No single observation dominates the estimates: the largest Cook's distance is {cd:.3f} (observation {o}), well below 1; {m} of {n} observations exceed the 4/N rule of thumb.",
                                 cd=float(cd[i0]), o=labels[i0], m=int((cd > 4 / n).sum()), n=n))
    rep.steps.append(s8)

    # ------------------------------------------------------------------ 9 added-variable plots
    s9 = Step("avplots", "Added-variable (partial regression) plots", "The relationship between y and each regressor after removing the linear effect of all other regressors (Frisch-Waugh-Lovell).",
              validated="slope of each plot equals the OLS coefficient (Frisch-Waugh-Lovell theorem), checked numerically (test_avplots)", refs=["frischwaugh"])
    pan = []
    for j, v in enumerate(xs, 1):
        Zo = np.delete(Xm, j, axis=1)
        ry = yy - Zo @ np.linalg.lstsq(Zo, yy, rcond=None)[0]
        rx = Xm[:, j] - Zo @ np.linalg.lstsq(Zo, Xm[:, j], rcond=None)[0]
        bj = float((rx @ ry) / (rx @ rx))
        px, py = thin(rx.tolist(), ry.tolist(), 250)
        xl = [float(rx.min()), float(rx.max())]
        pan.append({"title": v, "xlabel": v + " | others", "ylabel": y + " | others", "zero": True,
                    "layers": [{"type": "scatter", "x": px, "y": py}, {"type": "line", "x": xl, "y": [bj * xl[0], bj * xl[1]], "k": 2}]})
    s9.figures.append(Figure("f_av", "Added-variable plots", {"ncols": nc_, "panels": pan},
                             "Each axis shows the residual from regressing the variable on the intercept and all other regressors; the line has the OLS coefficient as its slope.",
                             "Source: author's calculations.", 6.5, 2.6 * math.ceil(len(xs) / nc_) + 0.3))
    rep.steps.append(s9)

    # ------------------------------------------------------------------ methods and limitations
    rep.methods.append(X.P("method", "We estimate a linear regression of {y} on {kx} explanatory variable{pl} ({xl}) by ordinary least squares, using {n} complete observations{per}. Inference uses {opts}. Serial correlation, heteroskedasticity, normality and functional form are checked with the Breusch-Godfrey, Breusch-Pagan, Jarque-Bera and RESET tests, parameter stability with the CUSUM tests, and influential observations with Cook's distance. Estimation was carried out with the Open Econometric Systems engine (version {ver}) using statsmodels (version {sv}) in Python {py}.",
                           y=y, kx=len(xs), pl="" if len(xs) == 1 else "s", xl=X.join(xs), n=n, per=per, opts=cov_name, ver=rep.meta["engine"], sv=rep.meta["statsmodels"], py=rep.meta["python"]))
    bad = [t for s in rep.steps for t in s.tests if t.reject and t.problem]
    if bad:
        rep.limitations.append(X.P("warning", "The following diagnostic tests reject their null hypothesis at the 5% level: {names}. The corresponding assumptions appear violated, so conclusions that rely on them should be treated with caution.", names=X.join([t.name for t in bad])))
    else:
        rep.limitations.append(X.P("note", "None of the diagnostic tests in this report signals a violated assumption at the 5% level; this does not prove the assumptions hold, and the tests have limited power in small samples."))
    rep.limitations.append(X.P("note", "The regression describes conditional associations in this sample. Interpreting a coefficient as a causal effect needs further assumptions (no omitted variables, no reverse causation) that these data cannot test."))
    if base.rsquared < 0.1:
        rep.limitations.append(X.P("warning", "The fit is weak (R² = {r2:.3f}); the regressors explain little of the variation in {y}.", r2=float(base.rsquared), y=y))
    return rep.finalize()
