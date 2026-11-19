"""Vector autoregression (VAR) publication report.

Steps: 1 data, 2 stationarity (ADF), 3 lag-order selection, 4 estimation (equation by equation), 5 stability (companion roots),
6 residual diagnostics, 7 Granger causality, 8 impulse responses (orthogonalised, Cholesky), 9 forecast-error variance decomposition.
The estimation calls are those of timeseries.var (same preparation, same statsmodels fit, same lag rule), so the numbers equal the
standard OES result; this layer adds the steps around them.
"""
import math

import numpy as np
from scipy import stats

from . import rep_tables as T
from . import rep_text as X
from .core import OESError, as_list, distinct, int_option, need
from .rep_model import Figure, Report, Step, TestResult
from .rep_ols import _xticks, acf
from .rep_tables import fnum, fint, fp, make_table, regression_table, tests_table

MAXPAN = 4          # impulse-response grid: at most this many shocks and responses are drawn
FIGSTYLE = "Source: author's calculations."


def companion(params, K, p):
    """Companion matrix of the VAR from the statsmodels coefficient frame (rows: const, L1.v1..., L2...; columns: equations)."""
    A = np.zeros((K * p, K * p))
    B = np.asarray(params)[1:].T            # K x (K p), blocks [A1 ... Ap]
    A[:K, :] = B
    if p > 1:
        A[K:, :-K] = np.eye(K * (p - 1))
    return A


def ma_matrices(A, K, h):
    out = []
    M = np.eye(A.shape[0])
    for _ in range(h + 1):
        out.append(M[:K, :K].copy())
        M = M @ A
    return np.array(out)                    # Phi_0 .. Phi_h


def portmanteau(resid, h, p):
    """Q = T sum_{j=1..h} tr(C_j' C_0^-1 C_j C_0^-1), C_j = (1/T) sum u_t u_{t-j}'; chi2 with K^2 (h-p) d.f. (Luetkepohl 2005, 4.4.21)."""
    u = np.asarray(resid, float)
    T_, K = u.shape
    C0 = u.T @ u / T_
    C0i = np.linalg.inv(C0)
    q = 0.0
    for j in range(1, h + 1):
        Cj = u[j:].T @ u[:-j] / T_
        q += np.trace(Cj.T @ C0i @ Cj @ C0i)
    return T_ * q, K * K * (h - p)


def adf_by_ols(x, lags):
    """Augmented Dickey-Fuller t statistic (constant, `lags` lagged differences) from a direct OLS regression."""
    x = np.asarray(x, float)
    dx = np.diff(x)
    n = len(dx) - lags
    Y = dx[lags:]
    cols = [x[lags:-1][:n] if lags else x[:-1], np.ones(n)]
    for i in range(1, lags + 1):
        cols.append(dx[lags - i:len(dx) - i])
    Z = np.column_stack(cols)
    b = np.linalg.lstsq(Z, Y, rcond=None)[0]
    e = Y - Z @ b
    s2 = e @ e / (n - Z.shape[1])
    V = s2 * np.linalg.inv(Z.T @ Z)
    return b[0] / math.sqrt(V[0, 0])


def build(df, spec):
    import sys
    import statsmodels
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.stats.stattools import jarque_bera
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller
    from .stat_core import qq_points, thin
    from .timeseries import _series_frame
    from . import __version__

    names = as_list(need(spec, "vars", "variables (at least two)"))
    if len(names) < 2:
        raise OESError("A VAR needs at least two variables.")
    distinct(names)
    d, labels = _series_frame(df, spec, names)
    data = d[names].astype(float)
    diffed = spec.get("transform", "levels") == "diff"
    if diffed:
        data = data.diff().dropna()
        labels = labels[1:]
    K = len(names)
    maxl = int_option(spec, "max_lags", 8, 1, 24, "Maximum lag")
    ic = spec.get("ic", "aic")
    horizon = int_option(spec, "horizon", 12, 1, 60, "IRF horizon")
    model = VAR(data.reset_index(drop=True))
    maxl = min(maxl, max(1, (len(data) - 10) // (K + 1)))
    so = model.select_order(maxl)
    p = spec.get("lags")
    p = int_option(spec, "lags", 1, 1, 24, "VAR lag order") if p not in (None, "", "auto") else max(1, int(getattr(so, ic)))
    res = model.fit(p)
    n = int(res.nobs)
    xt_all = _xticks(labels)
    obs = np.arange(len(data), dtype=float)
    xl_obs = "Observation" if not spec.get("time") else ""
    mat = np.asarray(data)
    eres = np.asarray(res.resid)
    efit = mat[p:] - eres

    rep = Report("var", "Vector autoregression of " + ", ".join(names), references={})
    rep.meta = dict(vars=names, nobs=n, lags=p, engine=__version__, statsmodels=statsmodels.__version__, python=".".join(map(str, sys.version_info[:3])),
                    transform="first differences" if diffed else "levels", ic=ic.upper())

    # ------------------------------------------------------------------ 1 data
    s1 = Step("data", "Data and descriptive statistics", "Sample and distribution of every variable" + (" after first-differencing." if diffed else "."),
              validated="pandas/scipy recomputation (test_var_data)")
    s1.tables.append(T.descriptives_table("t_desc", data, names, "Descriptive statistics" + (" (first differences)" if diffed else "")))
    s1.tables.append(T.correlation_table("t_corr", data, names, "Correlation matrix"))
    s1.figures.append(Figure("f_series", "Time series of the variables" + (" (first differences)" if diffed else ""),
                             {"ncols": 2, "panels": [{"title": v, "xlabel": xl_obs, "xticks": xt_all, "layers": [{"type": "line", "x": obs.tolist(), "y": data[v].tolist(), "k": 0, "lw": 0.8}]} for v in names]},
                             "Each panel plots one variable in the units used in estimation.", FIGSTYLE, 6.5, 2.1 * math.ceil(K / 2) + 0.5))
    C = data.corr().to_numpy()
    s1.figures.append(Figure("f_corr", "Correlation matrix of the variables", {"ncols": 1, "panels": [{"layers": [{"type": "heatmap", "matrix": C.tolist(), "xlabels": names, "ylabels": names}], "ml": 70}]},
                             "Cells show Pearson correlations. Blue is negative, orange positive.", FIGSTYLE, 4.6, 4.0))
    rep.steps.append(s1)

    # ------------------------------------------------------------------ 2 stationarity
    s2 = Step("adf", "Stationarity of the variables", "Augmented Dickey-Fuller tests with a constant; the lag length of each test is chosen by AIC.",
              validated="ADF statistic recomputed from a direct OLS regression with the reported lag length; the p-value is statsmodels' MacKinnon approximation and has no independent reference here (test_var_adf)")
    adf_tests = []
    rows = []
    for v in names:
        x = data[v].to_numpy()
        a = adfuller(x, regression="c", autolag="AIC")
        adf_tests.append(TestResult(f"ADF, {v}", float(a[0]), float(a[1]), "The series has a unit root", f"{a[2]} lags", bool(a[1] < 0.05), False, ""))
        rows.append(dict(var=v, stat=float(a[0]), p=float(a[1]), lags=int(a[2]), crit=a[4]))
    s2.tests = adf_tests
    s2.tables.append(tests_table("t_adf", "Augmented Dickey-Fuller unit-root tests", adf_tests,
                                 ["Rejecting the null of a unit root at the 5% level is evidence of stationarity; failing to reject does not prove a unit root, because the test has low power in short samples.",
                                  "Critical values (1%, 5%, 10%): " + "; ".join(f"{r['var']}: " + ", ".join(fnum(r['crit'][k], 2) for k in ('1%', '5%', '10%')) for r in rows[:1]) + " (same for every variable of the same sample size up to rounding)."]))
    s2.tables[-1].raw = {"rows": rows}
    rep.steps.append(s2)

    # ------------------------------------------------------------------ 3 lag selection
    s3 = Step("lags", "Lag-order selection", f"Information criteria for lag orders 0 to {maxl}; the model uses {p} lag(s).",
              validated="AIC recomputed as ln|Sigma_mle| + 2 (p K^2 + K)/T from a direct per-equation OLS fit at the chosen lag (test_var_lags); the other criteria use the same statsmodels formulas")
    srows = []
    for lag in range(len(so.ics["aic"])):
        srows.append([str(lag)] + [fnum(so.ics[c][lag], 4) + ("*" if lag == int(getattr(so, c)) else "") for c in ("aic", "bic", "hqic", "fpe")])
    s3.tables.append(make_table("t_lags", "Lag-order selection", ["Lags", "AIC", "BIC", "HQIC", "FPE"], srows,
                                [f"* marks the minimum of each criterion. The estimated model uses {p} lag(s), chosen " + ("by the user." if spec.get("lags") not in (None, "", "auto") else f"by {ic.upper()}.")],
                                raw={"aic": [float(v) for v in so.ics["aic"]], "bic": [float(v) for v in so.ics["bic"]], "hqic": [float(v) for v in so.ics["hqic"]], "fpe": [float(v) for v in so.ics["fpe"]], "selected": p}))
    lg = [float(i) for i in range(len(so.ics["aic"]))]
    s3.figures.append(Figure("f_ic", "Information criteria by lag order",
                             {"ncols": 3, "panels": [{"title": c.upper(), "xlabel": "Lags", "xticks": [[x, str(int(x))] for x in lg][::max(1, len(lg) // 6)],
                                                      "layers": [{"type": "line", "x": lg, "y": [float(v) for v in so.ics[c]], "k": 0}, {"type": "scatter", "x": [float(int(getattr(so, c)))], "y": [float(so.ics[c][int(getattr(so, c))])], "r": 2.8, "fill": "#000000", "alpha": 1.0}]}
                                                     for c in ("aic", "bic", "hqic")]},
                             "The dot marks the lag order that minimises each criterion.", FIGSTYLE, 6.5, 2.4))
    rep.steps.append(s3)

    # ------------------------------------------------------------------ 4 estimation
    s4 = Step("estimation", "Estimation", f"VAR({p}) estimated equation by equation by OLS with a constant; standard errors are the conventional ones of the system fit.",
              validated="coefficients, standard errors and R-squared equal a direct per-equation OLS with the closed form (X'X)^-1 X'y (test_var_estimation); equals oes_engine.run('var')")
    plabels = {nm: nm.replace("L", "L", 1) for nm in res.params.index}
    cols, r2 = [], {}
    for i, eq in enumerate(names):
        y_ = mat[p:, i]
        r2[eq] = float(1 - (eres[:, i] ** 2).sum() / ((y_ - y_.mean()) ** 2).sum())
        tv = res.params[eq] / res.stderr[eq]
        pv_ = tv.copy()
        pv_[:] = 2 * stats.t.sf(np.abs(tv.to_numpy()), res.df_resid)
        cols.append(dict(label=f"({i + 1})", sub=eq, params=res.params[eq], se=res.stderr[eq], p=pv_,
                         stats=dict(n=n, r2=r2[eq], rmse=float(math.sqrt((eres[:, i] ** 2).sum() / res.df_resid)))))
    tr = regression_table("t_var", f"VAR({p}) estimates", cols, list(res.params.index), stat_rows=[("Observations", "n", "int"), ("R²", "r2", "num3"), ("Residual std. error", "rmse", "num3")],
                          dep_label="Columns are the equations; the column heading is the dependent variable.")
    tr.notes = ["Columns are the equations; the second heading row is the dependent variable. L1.x is the first lag of x.", "Standard errors in parentheses; p-values and stars use the t distribution with the residual degrees of freedom.", T.STAR_NOTE]
    s4.tables.append(tr)
    s4.figures.append(Figure("f_fit", "Actual and fitted values by equation",
                             {"ncols": 2 if K > 1 else 1, "panels": [{"title": v, "xlabel": xl_obs, "xticks": [[x - p, l] for x, l in xt_all if x >= p] or xt_all, "legend": i == 0,
                                                                     "layers": [{"type": "line", "x": obs[p:].tolist(), "y": mat[p:, i].tolist(), "k": 0, "label": "Actual", "lw": 0.8},
                                                                                {"type": "line", "x": obs[p:].tolist(), "y": efit[:, i].tolist(), "k": 2, "label": "Fitted", "lw": 0.8}]} for i, v in enumerate(names)]},
                             "Fitted values are the one-step-ahead predictions of each equation.", FIGSTYLE, 6.5, 2.2 * math.ceil(K / 2) + 0.5))
    rep.steps.append(s4)

    # ------------------------------------------------------------------ 5 stability
    s5 = Step("stability", "Stability of the system", "The VAR is stable (stationary) if every eigenvalue of the companion matrix lies inside the unit circle.",
              validated="eigenvalues of the companion matrix built from the estimated coefficients and compared with statsmodels' roots (test_var_stability)")
    A = companion(res.params, K, p)
    eig = np.linalg.eigvals(A)
    mod = np.sort(np.abs(eig))[::-1]
    s5.tables.append(make_table("t_roots", "Eigenvalues of the companion matrix", ["Eigenvalue", "Real part", "Imaginary part", "Modulus"],
                                [[str(i + 1), fnum(z.real, 4), fnum(z.imag, 4), fnum(abs(z), 4)] for i, z in enumerate(sorted(eig, key=lambda z: -abs(z)))],
                                ["The system is stable if every modulus is below 1. The largest modulus is " + fnum(mod[0], 4) + "."],
                                raw={"moduli": [float(m) for m in mod], "stable": bool(mod[0] < 1)}))
    th = np.linspace(0, 2 * math.pi, 121)
    s5.figures.append(Figure("f_roots", "Eigenvalues of the companion matrix and the unit circle",
                             {"ncols": 1, "panels": [{"xlabel": "Real part", "ylabel": "Imaginary part", "xlim": [-1.25, 1.25], "ylim": [-1.25, 1.25],
                                                      "layers": [{"type": "line", "x": np.cos(th).tolist(), "y": np.sin(th).tolist(), "k": 2}, {"type": "scatter", "x": eig.real.tolist(), "y": eig.imag.tolist(), "r": 2.6, "fill": "#000000", "alpha": 1.0}]}]},
                             "Points inside the dashed unit circle indicate a stable system; a point on or outside it signals a unit root or an explosive root.", FIGSTYLE, 3.8, 3.8))
    rep.steps.append(s5)

    # ------------------------------------------------------------------ 6 residual diagnostics
    s6 = Step("diag", "Residual diagnostics", "Tests on the residuals of each equation and of the system: autocorrelation, ARCH effects, normality.",
              validated="portmanteau statistic recomputed from residual autocovariances; Jarque-Bera, Ljung-Box and ARCH LM of each equation recomputed from their definitions (test_var_diag)")
    tests = []
    h_p = max(p + 1, min(12, len(data) // 5))
    try:
        w = res.test_whiteness(nlags=h_p)
        tests.append(TestResult(f"Portmanteau ({h_p} lags)", float(w.test_statistic), float(w.pvalue), "No residual autocorrelation (system)", f"{int(w.df)}", bool(w.pvalue < 0.05), True, ""))
    except Exception:
        pass
    try:
        nm = res.test_normality()
        tests.append(TestResult("Jarque-Bera (system)", float(nm.test_statistic), float(nm.pvalue), "Normal residuals (system)", f"{int(nm.df)}", bool(nm.pvalue < 0.05), True, ""))
    except Exception:
        pass
    m_lb = min(10, n // 5)
    for i, v in enumerate(names):
        e = eres[:, i]
        jb = jarque_bera(e)
        tests.append(TestResult(f"Jarque-Bera, {v}", float(jb[0]), float(jb[1]), "Normal residuals", "2", bool(jb[1] < 0.05), True, ""))
        lb = acorr_ljungbox(e, lags=[m_lb], return_df=True)
        tests.append(TestResult(f"Ljung-Box Q({m_lb}), {v}", float(lb["lb_stat"].iloc[0]), float(lb["lb_pvalue"].iloc[0]), "No residual autocorrelation", str(m_lb), bool(lb["lb_pvalue"].iloc[0] < 0.05), True, ""))
        ar = het_arch(e, nlags=4)
        tests.append(TestResult(f"ARCH LM (4 lags), {v}", float(ar[0]), float(ar[1]), "No ARCH effects", "4", bool(ar[1] < 0.05), True, ""))
    s6.tests = tests
    s6.tables.append(tests_table("t_diag", "Residual diagnostic tests", tests,
                                 ["'Reject' means the null hypothesis is rejected at the 5% level; for these tests rejection signals a violated assumption.", "System tests use the multivariate residuals; the others use each equation's residuals."]))
    ob = np.arange(len(eres), dtype=float)
    s6.figures.append(Figure("f_resid", "Residuals by equation",
                             {"ncols": 2 if K > 1 else 1, "panels": [{"title": v, "xlabel": xl_obs, "xticks": [[x - p, l] for x, l in xt_all if x >= p] or xt_all, "zero": True,
                                                                    "layers": [{"type": "line", "x": obs[p:].tolist(), "y": eres[:, i].tolist(), "k": 0, "lw": 0.8}]} for i, v in enumerate(names)]},
                             "Residuals of each equation over the estimation sample.", FIGSTYLE, 6.5, 2.1 * math.ceil(K / 2) + 0.5))
    mlag = min(20, max(4, n // 4))
    bnd = 1.96 / math.sqrt(n)
    s6.figures.append(Figure("f_acf", "Residual autocorrelation by equation",
                             {"ncols": 2 if K > 1 else 1, "panels": [{"title": v, "xlabel": "Lag", "ylabel": "Autocorrelation", "zero": True,
                                                                     "layers": [{"type": "stem", "x": [float(j) for j in range(1, mlag + 1)], "y": acf(eres[:, i], mlag)[1:].tolist()}, {"type": "hline", "y": bnd}, {"type": "hline", "y": -bnd}]} for i, v in enumerate(names)]},
                             "Dashed lines are approximate 95 per cent bounds, plus or minus 1.96/sqrt(N).", FIGSTYLE, 6.5, 2.2 * math.ceil(K / 2) + 0.5))
    qp = []
    for i, v in enumerate(names):
        z, r_ = qq_points(eres[:, i])
        qx, qy = thin(z.tolist(), r_.tolist(), 200)
        qp.append({"title": v, "xlabel": "Theoretical quantile", "ylabel": "Standardised residual", "layers": [{"type": "scatter", "x": qx, "y": qy}, {"type": "line", "x": [float(z.min()), float(z.max())], "y": [float(z.min()), float(z.max())], "k": 2}]})
    s6.figures.append(Figure("f_qq", "Normal quantile plots of the residuals", {"ncols": 2 if K > 1 else 1, "panels": qp},
                             "Points near the dashed 45-degree line indicate normally distributed residuals.", FIGSTYLE, 6.5, 2.4 * math.ceil(K / 2) + 0.4))
    rep.steps.append(s6)

    # ------------------------------------------------------------------ 7 Granger causality
    s7 = Step("granger", "Granger causality", "Does the lagged history of one variable help to predict another, beyond the other's own lags and the remaining variables?",
              validated="the statistic equals the F statistic of the restricted vs unrestricted regression of the equation; the p-value uses statsmodels' F(p, K x residual d.f.) reference distribution, checked against that definition (test_var_granger)")
    gtests = []
    for caused in names:
        for causing in names:
            if caused == causing:
                continue
            t = res.test_causality(caused, [causing], kind="f")
            gtests.append(TestResult(f"{causing} → {caused}", float(t.test_statistic), float(t.pvalue), f"{causing} does not Granger-cause {caused}", f"{int(t.df[0])}, {int(t.df[1])}", bool(t.pvalue < 0.05), False, ""))
    s7.tests = gtests
    s7.tables.append(tests_table("t_granger", "Granger causality tests", gtests,
                                 ["Each row tests that all lags of the first variable have zero coefficients in the equation of the second variable (F test). Granger causality is about predictive content, not about causal effects."]))
    rep.steps.append(s7)

    # ------------------------------------------------------------------ 8 impulse responses
    s8 = Step("irf", "Impulse responses", f"Orthogonalised responses over {horizon} periods to a one-standard-deviation shock; Cholesky ordering as the variables were listed.",
              validated="responses recomputed as Phi_h P (powers of the companion matrix times the Cholesky factor of the residual covariance) (test_var_irf). The +/-1.96 bands are statsmodels' asymptotic standard errors and have no independent reference here: not validated")
    irf = res.irf(horizon)
    resp = irf.orth_irfs
    try:
        se = irf.stderr(orth=True)
    except Exception:
        se = None
    kk = min(K, MAXPAN)
    xh = [float(i) for i in range(horizon + 1)]
    panels = []
    for j in range(kk):
        for i in range(kk):
            y = resp[:, i, j]
            lay = []
            if se is not None:
                lay.append({"type": "band", "x": xh, "lo": (y - 1.96 * se[:, i, j]).tolist(), "hi": (y + 1.96 * se[:, i, j]).tolist()})
            lay.append({"type": "line", "x": xh, "y": y.tolist(), "k": 0})
            panels.append({"title": f"{names[i]} ← {names[j]}", "xlabel": "Horizon", "zero": True, "layers": lay})
    s8.figures.append(Figure("f_irf", "Orthogonalised impulse responses", {"ncols": kk, "panels": panels},
                             "Each panel shows the response of the first variable to a one-standard-deviation shock in the second. Bands are plus or minus 1.96 asymptotic standard errors (not validated against a reference)." +
                             (f" Only the first {kk} variables are drawn." if K > kk else ""), FIGSTYLE, 6.5, 1.9 * kk + 0.4))
    rows = []
    for j in range(K):
        for i in range(K):
            rows.append([f"{names[i]} ← {names[j]}"] + [fnum(resp[h, i, j], 4) for h in sorted(set([0, 1, 2, 4, 8, horizon]))])
    hs = sorted(set([0, 1, 2, 4, 8, horizon]))
    s8.tables.append(make_table("t_irf", "Orthogonalised impulse responses at selected horizons", ["Response ← shock"] + [f"h = {h}" for h in hs], rows,
                                ["Cholesky ordering: " + " → ".join(names) + ". A shock in a variable listed earlier can affect later variables contemporaneously, not the reverse; results depend on this ordering."],
                                raw={"hs": hs, "orth_irf": resp.tolist()}))
    rep.steps.append(s8)

    # ------------------------------------------------------------------ 9 FEVD
    s9 = Step("fevd", "Forecast-error variance decomposition", f"Share of each variable's forecast-error variance at horizon {horizon} explained by each shock.",
              validated="shares recomputed from the orthogonalised responses (test_var_irf)")
    fe = res.fevd(horizon).decomp
    s9.tables.append(make_table("t_fevd", f"Forecast-error variance decomposition at horizon {horizon} (%)", ["Variable"] + names,
                                [[names[i]] + [fnum(fe[i, -1, j] * 100, 2) for j in range(K)] for i in range(K)],
                                ["Rows: the variable being forecast; columns: the shock. Each row sums to 100. Cholesky ordering as in the previous step."], raw={"fevd": fe.tolist()}))
    fp_ = []
    hx = [float(i + 1) for i in range(horizon)]
    for i in range(kk):
        fp_.append({"title": names[i], "xlabel": "Horizon", "ylabel": "Share", "legend": i == 0, "ylim": [0, 1], "yticks": [[0, "0"], [0.5, "0.5"], [1, "1"]],
                    "layers": [{"type": "line", "x": hx, "y": fe[i, :, j].tolist(), "k": j, "label": names[j]} for j in range(kk)]})
    s9.figures.append(Figure("f_fevd", "Forecast-error variance decomposition over the horizon", {"ncols": 2 if kk > 1 else 1, "panels": fp_},
                             "Each panel shows, for one variable, the share of its forecast-error variance due to each shock; every line has its own style." + (f" Only the first {kk} variables are drawn." if K > kk else ""),
                             FIGSTYLE, 6.5, 2.4 * math.ceil(kk / 2) + 0.5))
    rep.steps.append(s9)

    # ------------------------------------------------------------------ methods and limitations
    rep.methods.append(X.P("method", "We estimate a {k}-variable vector autoregression ({v}) with {p} lag(s) and a constant by equation-by-equation OLS on {n} observations{tr}. The lag order was {how}. Stationarity is checked with ADF tests, residuals with portmanteau, Ljung-Box, ARCH LM and Jarque-Bera tests, and stability with the eigenvalues of the companion matrix. Impulse responses and variance decompositions use a Cholesky ordering of the variables as listed. Estimation used the Open Econometric Systems engine (version {ver}) with statsmodels (version {sv}) in Python {py}.",
                           k=K, v=X.join(names), p=p, n=n, tr=(" after first-differencing" if diffed else ""), how=("set by the user" if spec.get("lags") not in (None, "", "auto") else f"chosen by {ic.upper()}"),
                           ver=__version__, sv=statsmodels.__version__, py=rep.meta["python"]))
    bad = [t.name for s in rep.steps for t in s.tests if t.reject and t.problem]
    if bad:
        rep.limitations.append(X.P("warning", "The following diagnostic tests reject their null hypothesis at the 5% level: {names}. The corresponding assumptions appear violated; inference from the estimated system should be treated with caution.", names=X.join(bad)))
    if mod[0] >= 1:
        rep.limitations.append(X.P("warning", "The system is not stable: the largest companion eigenvalue has modulus {m:.4f}, which is not below 1. Impulse responses and variance decompositions are then not meaningful in the usual sense.", m=float(mod[0])))
    unit = [t.name for t in adf_tests if not t.reject]
    if unit and not diffed:
        rep.limitations.append(X.P("note", "The ADF test does not reject a unit root for: {names}. A VAR in levels with non-stationary variables can give unreliable inference; consider differences or a VECM.", names=X.join(unit)))
    rep.limitations.append(X.P("note", "Impulse responses depend on the Cholesky ordering; the 95% bands rest on asymptotic standard errors and are not validated against a reference implementation. Granger causality is about predictive content, not a causal effect."))
    return rep.finalize()
