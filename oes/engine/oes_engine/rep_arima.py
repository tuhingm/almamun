"""ARIMA / SARIMA publication report.

Steps: 1 data and autocorrelation, 2 unit-root tests, 3 order selection, 4 estimation, 5 residual diagnostics, 6 forecasts.
Same preparation, order grid (BIC on the training sample), SARIMAX fit and hold-out scheme as ts_extra.arima, so the numbers equal
the standard OES result; this layer adds the steps and figures around them.
"""
import math

import numpy as np
import pandas as pd
from scipy import stats

from . import rep_tables as T
from . import rep_text as X
from .core import OESError, int_option, need
from .rep_model import Figure, Report, Step, TestResult
from .rep_ols import _xticks, acf
from .rep_tables import fnum, fp, make_table, tests_table

FIGSTYLE = "Source: author's calculations."


def pacf_yw(x, nlags):
    """Partial autocorrelations from the Yule-Walker equations (Levinson-Durbin on the biased autocovariances)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    g = np.array([(x[k:] * x[:n - k]).sum() / n for k in range(nlags + 1)])
    out = [1.0]
    for k in range(1, nlags + 1):
        R = np.array([[g[abs(i - j)] for j in range(k)] for i in range(k)])
        phi = np.linalg.solve(R, g[1:k + 1])
        out.append(float(phi[-1]))
    return np.array(out)


def _corr_fig(key, title, vals, n, lagmax, ylabel):
    lags = np.arange(1, lagmax + 1, dtype=float)
    band = 1.96 / math.sqrt(n)
    return Figure(key, title, {"ncols": 1, "panels": [{"xlabel": "Lag", "ylabel": ylabel, "ylim": [-1, 1],
                                                       "layers": [{"type": "stem", "x": lags.tolist(), "y": [float(v) for v in vals[1:lagmax + 1]]}, {"type": "hline", "y": band, "dash": (3, 2)}, {"type": "hline", "y": -band, "dash": (3, 2)}, {"type": "hline", "y": 0.0}]}]},
                  "Dashed lines are approximate 95% bounds, plus and minus 1.96 over the square root of the sample size.", FIGSTYLE, 4.6, 2.8)


def build(df, spec):
    import sys
    import statsmodels
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from .stat_core import qq_points
    from .timeseries import _series_frame
    from . import __version__

    y = need(spec, "y", "series")
    d, labels = _series_frame(df, spec, [y])
    x = d[y].to_numpy(float)
    n = len(x)
    s = int_option(spec, "season", 0, 0, 52, "Seasonal period")
    maxp = int_option(spec, "max_p", 3, 0, 5, "Maximum p")
    maxq = int_option(spec, "max_q", 3, 0, 5, "Maximum q")
    h = int_option(spec, "holdout", 12, 4, 60, "Hold-out periods")
    dsel = spec.get("d") or "auto"
    if n < max(60, 3 * h):
        raise OESError("The series is too short for this hold-out. Use at least 60 observations.")
    a0, k0 = adfuller(x, autolag="AIC"), kpss(x, regression="c", nlags="auto")
    a1, k1 = adfuller(np.diff(x), autolag="AIC"), kpss(np.diff(x), regression="c", nlags="auto")
    dd = int(dsel) if dsel != "auto" else (0 if a0[1] < .10 else 1)
    train, test = x[:-h], x[-h:]
    seas = (1, 1 if spec.get("sdiff") == "yes" else 0, 0, s) if s > 1 else (0, 0, 0, 0)
    trend = "c" if dd == 0 and seas[1] == 0 else "n"

    def fit(p, q, data):
        return SARIMAX(data, order=(p, dd, q), seasonal_order=seas, trend=trend).fit(disp=False, maxiter=200)
    grid = []
    for p in range(maxp + 1):
        for q in range(maxq + 1):
            try:
                r = fit(p, q, train)
                grid.append((p, q, r.aic, r.bic))
            except Exception:
                continue
    if not grid:
        raise OESError("No ARIMA model could be estimated.")
    g = pd.DataFrame(grid, columns=["p", "q", "AIC", "BIC"]).sort_values("BIC")
    p, q = int(g.iloc[0].p), int(g.iloc[0].q)
    m = fit(p, q, train)
    full = fit(p, q, x)
    skip = max(dd + p, 1)
    e = np.asarray(m.resid)[skip:]
    obs = np.arange(n, dtype=float)
    xt = _xticks(labels)

    rep = Report("arima", f"ARIMA({p},{dd},{q}) model of {y}", references={})
    rep.meta = dict(y=y, nobs=n, order=[p, dd, q], seasonal=list(seas), holdout=h, engine=__version__, statsmodels=statsmodels.__version__, python=".".join(map(str, sys.version_info[:3])))

    # 1 data
    s1 = Step("data", "Data and autocorrelation", "Distribution, time path and autocorrelation of the series and its first difference.", validated="pandas/scipy recomputation; ACF from its definition; PACF from the Yule-Walker equations (test_arima_data)")
    s1.tables.append(T.descriptives_table("t_desc", d, [y], "Descriptive statistics"))
    s1.figures.append(Figure("f_series", f"{y} over time", {"ncols": 1, "panels": [{"xlabel": "", "ylabel": y, "xticks": xt, "layers": [{"type": "line", "x": obs.tolist(), "y": x.tolist(), "k": 0, "lw": 0.8}]}]}, "The series as used in estimation.", FIGSTYLE, 6.0, 2.8))
    dx = np.diff(x)
    s1.figures.append(Figure("f_diff", f"First difference of {y}", {"ncols": 1, "panels": [{"xlabel": "", "ylabel": "Change", "xticks": xt, "layers": [{"type": "line", "x": obs[1:].tolist(), "y": dx.tolist(), "k": 0, "lw": 0.8}, {"type": "hline", "y": 0.0}]}]}, "Period-to-period change of the series.", FIGSTYLE, 6.0, 2.6))
    L = min(24, n // 4)
    s1.figures.append(_corr_fig("f_acf", "Autocorrelation function of the series", acf(x, L), n, L, "Autocorrelation"))
    s1.figures.append(_corr_fig("f_pacf", "Partial autocorrelation function of the series", pacf_yw(x, L), n, L, "Partial autocorrelation"))
    rep.steps.append(s1)

    # 2 unit roots
    s2 = Step("unitroot", "Unit-root and stationarity tests", "ADF (null: unit root) and KPSS (null: stationary) with a constant, for the level and the first difference.",
              validated="ADF statistic recomputed by OLS at the chosen lag length; KPSS statistic recomputed from its definition with the Bartlett-kernel long-run variance at the reported lag; p-values are statsmodels' interpolated tables and have no independent reference (test_arima_unitroot)")
    tt = [TestResult("ADF, level", float(a0[0]), float(a0[1]), "Unit root", f"{a0[2]} lags", bool(a0[1] < .05), False), TestResult("KPSS, level", float(k0[0]), float(k0[1]), "Stationary", f"{k0[2]} lags", bool(k0[1] < .05), False),
          TestResult("ADF, first difference", float(a1[0]), float(a1[1]), "Unit root", f"{a1[2]} lags", bool(a1[1] < .05), False), TestResult("KPSS, first difference", float(k1[0]), float(k1[1]), "Stationary", f"{k1[2]} lags", bool(k1[1] < .05), False)]
    s2.tests = tt
    s2.tables.append(tests_table("t_ur", "Unit-root and stationarity tests", tt, [f"Differencing order used: d = {dd}" + (" (chosen from the ADF test of the level at the 10% level)." if dsel == "auto" else " (set by the user).")]))
    s2.tables[-1].raw = {"adf_level": [float(a0[0]), int(a0[2])], "adf_diff": [float(a1[0]), int(a1[2])], "kpss_level": [float(k0[0]), int(k0[2])]}
    rep.steps.append(s2)

    # 3 order
    s3 = Step("order", "Order selection", f"ARMA orders chosen by BIC on the training sample (all but the last {h} observations).", validated="information criteria recomputed from the log-likelihood and parameter count (test_arima_order)")
    s3.tables.append(make_table("t_order", "Order selection (training sample)", ["p", "q", "AIC", "BIC"], [[str(int(a)), str(int(b)), fnum(c, 2), fnum(e_, 2)] for a, b, c, e_ in g.head(8).values],
                                [f"Best eight by BIC out of {len(g)} models, d = {dd}" + (f", seasonal period {s}" if s > 1 else "") + ". The first row is the model used."], raw={"grid": g.values.tolist(), "chosen": [p, q]}))
    top = g.head(8)
    s3.figures.append(Figure("f_order", "BIC of the best candidate models", {"ncols": 1, "panels": [{"xlabel": "Model (p, q)", "ylabel": "BIC", "xticks": [[float(i), f"({int(r.p)},{int(r.q)})"] for i, r in enumerate(top.itertuples())],
                                                                                         "layers": [{"type": "scatter", "x": [float(i) for i in range(len(top))], "y": top.BIC.tolist(), "r": 2.6, "fill": "#000000", "alpha": 1.0}]}]}, "Lower is better; models are ordered from best to worst.", FIGSTYLE, 5.5, 2.8))
    rep.steps.append(s3)

    # 4 estimation
    s4 = Step("estimation", "Estimation", f"ARIMA({p},{dd},{q}) estimated by maximum likelihood (state-space form) on the full sample.", validated="parameters agree with statsmodels ARIMA (a separate implementation) for d = 0; the engine's own estimates are those reported (test_arima_estimation)")
    names = list(full.param_names)
    se = np.asarray(full.bse)
    pr = np.asarray(full.params)
    s4.tables.append(make_table("t_est", f"ARIMA({p},{dd},{q}) estimates", ["Parameter", "Estimate", "Std. err.", "z", "p-value"],
                                [[nm, fnum(b, 4), fnum(a, 4), fnum(b / a, 2), fp(2 * stats.norm.sf(abs(b / a)))] for nm, b, a in zip(names, pr, se)] + [["Log-likelihood", fnum(full.llf, 2), "", "", ""], ["AIC", fnum(full.aic, 2), "", "", ""], ["BIC", fnum(full.bic, 2), "", "", ""], ["Observations", str(n), "", "", ""]],
                                ["Standard errors are from the observed information matrix; sigma2 is the innovation variance."], raw={"names": names, "params": pr.tolist(), "se": se.tolist(), "llf": float(full.llf)}))
    fv = np.asarray(full.fittedvalues)
    s4.figures.append(Figure("f_fit", "Actual and fitted values", {"ncols": 1, "panels": [{"xlabel": "", "ylabel": y, "xticks": xt, "legend": True, "layers": [{"type": "line", "x": obs[skip:].tolist(), "y": x[skip:].tolist(), "k": 0, "label": "Actual", "lw": 0.8}, {"type": "line", "x": obs[skip:].tolist(), "y": fv[skip:].tolist(), "k": 2, "label": "Fitted", "lw": 0.8}]}]},
                             "Fitted values are one-step-ahead predictions.", FIGSTYLE, 6.0, 2.9))
    rep.steps.append(s4)

    # 5 diagnostics
    s5 = Step("diag", "Residual diagnostics", "Tests on the training-sample residuals of the selected model.", validated="Ljung-Box from its definition; ARCH LM from an auxiliary OLS regression; Jarque-Bera from sample moments (test_arima_diag)")
    lb = acorr_ljungbox(e, lags=[6, 12, 24], return_df=True)
    dg = [TestResult(f"Ljung-Box ({Lg} lags)", float(lb.lb_stat.iloc[i]), float(lb.lb_pvalue.iloc[i]), "No serial correlation", str(Lg), bool(lb.lb_pvalue.iloc[i] < .05), True) for i, Lg in enumerate((6, 12, 24))]
    ar = het_arch(e, nlags=6)
    dg.append(TestResult("ARCH LM (6 lags)", float(ar[0]), float(ar[1]), "No ARCH effects", "6", bool(ar[1] < .05), True))
    jb = stats.jarque_bera(e)
    dg.append(TestResult("Jarque-Bera", float(jb[0]), float(jb[1]), "Normal residuals", "2", bool(jb[1] < .05), True))
    s5.tests = dg
    s5.tables.append(tests_table("t_diag", "Residual diagnostics", dg))
    s5.figures.append(Figure("f_res", "Residuals over time", {"ncols": 1, "panels": [{"xlabel": "", "ylabel": "Residual", "xticks": xt, "layers": [{"type": "line", "x": obs[:len(train)][skip:].tolist(), "y": e.tolist(), "k": 0, "lw": 0.7}, {"type": "hline", "y": 0.0}]}]}, "Residuals of the training-sample fit.", FIGSTYLE, 6.0, 2.6))
    s5.figures.append(_corr_fig("f_racf", "Autocorrelation of the residuals", acf(e, L), len(e), L, "Autocorrelation"))
    z, rs = qq_points(e)
    lim = float(max(abs(z.min()), abs(z.max())))
    s5.figures.append(Figure("f_qq", "Normal Q-Q plot of the residuals", {"ncols": 1, "panels": [{"xlabel": "Theoretical quantiles", "ylabel": "Standardised residuals", "layers": [{"type": "scatter", "x": z.tolist(), "y": rs.tolist()}, {"type": "line", "x": [-lim, lim], "y": [-lim, lim], "k": 2}]}]}, "Points near the line indicate approximately normal residuals.", FIGSTYLE, 4.2, 3.6))
    hc, hn = np.histogram(e, bins=20)
    s5.figures.append(Figure("f_hist", "Distribution of the residuals", {"ncols": 1, "panels": [{"xlabel": "Residual", "ylabel": "Count", "layers": [{"type": "bar", "x": ((hn[:-1] + hn[1:]) / 2).tolist(), "y": hc.tolist(), "width": float(hn[1] - hn[0]), "fill": "#8a8a8a"}]}]}, "Histogram with 20 bins.", FIGSTYLE, 4.6, 3.0))
    lg = list(range(1, 25 if len(e) > 60 else 13))
    lbp = acorr_ljungbox(e, lags=lg, return_df=True)
    s5.figures.append(Figure("f_lbp", "Ljung-Box p-values by lag", {"ncols": 1, "panels": [{"xlabel": "Lag", "ylabel": "p-value", "ylim": [0, 1], "layers": [{"type": "scatter", "x": [float(v) for v in lg], "y": lbp.lb_pvalue.tolist(), "r": 2.2, "fill": "#000000", "alpha": 1.0}, {"type": "hline", "y": 0.05, "dash": (3, 2)}]}]},
                             "Points below the dashed 5% line signal remaining serial correlation.", FIGSTYLE, 4.8, 2.8))
    rep.steps.append(s5)

    # 6 forecasts
    s6 = Step("forecast", "Forecasts", f"Rolling one-step forecasts over the last {h} observations against a random-walk benchmark, and a {h}-step forecast with intervals.",
              validated="RMSE and the Diebold-Mariano statistic recomputed from the forecast errors with the same bandwidth rule; the forecasts themselves come from re-estimating SARIMAX at each origin (test_arima_forecast)")
    fc, bench = [], []
    for i in range(h):
        data = x[:n - h + i]
        try:
            fc.append(float(fit(p, q, data).forecast(1)[0]))
        except Exception:
            fc.append(np.nan)
        bench.append(float(data[-1]))
    fc, bench = np.array(fc), np.array(bench)
    e1, e0 = test - fc, test - bench
    rm1, rm0 = float(np.sqrt(np.nanmean(e1 ** 2))), float(np.sqrt(np.mean(e0 ** 2)))
    dm = e1 ** 2 - e0 ** 2
    ok = np.isfinite(dm)
    dms = dmp = np.nan
    if ok.sum() > 4:
        v = dm[ok]
        bw = max(int(round(ok.sum() ** (1 / 3))), 1)
        u_ = v - v.mean()
        lrv = u_ @ u_ / len(v) + 2 * sum((1 - l / (bw + 1)) * (u_[l:] @ u_[:-l]) / len(v) for l in range(1, bw + 1))
        dms = v.mean() / np.sqrt(max(lrv, 1e-12) / len(v))
        dmp = 2 * stats.norm.sf(abs(dms))
    s6.tests = [TestResult("Diebold-Mariano (ARIMA vs random walk)", float(dms), float(dmp), "Equal squared forecast errors", f"{int(ok.sum())} forecasts", bool(dmp < .05) if np.isfinite(dmp) else None, False)]
    s6.tables.append(make_table("t_acc", "Hold-out forecast accuracy (rolling one-step)", ["Forecast", "RMSE"], [[f"ARIMA({p},{dd},{q})", fnum(rm1, 4)], ["Random walk (last value)", fnum(rm0, 4)]],
                                [f"Diebold-Mariano statistic {fnum(dms, 3)}, p = {fp(dmp)} ({int(ok.sum())} forecasts)."], raw={"rmse": [rm1, rm0], "dm": [float(dms), float(dmp)], "forecast": fc.tolist(), "actual": test.tolist()}))
    ho = obs[n - h:]
    s6.figures.append(Figure("f_hold", "Hold-out forecasts against actual values", {"ncols": 1, "panels": [{"xlabel": "", "ylabel": y, "xticks": [[float(i), labels[i]] for i in range(n - h, n, max(1, h // 5))], "legend": True,
                                                                                                         "layers": [{"type": "line", "x": ho.tolist(), "y": test.tolist(), "k": 0, "label": "Actual"}, {"type": "line", "x": ho.tolist(), "y": fc.tolist(), "k": 2, "label": "ARIMA"}, {"type": "line", "x": ho.tolist(), "y": bench.tolist(), "k": 3, "label": "Random walk"}]}]},
                               "Each forecast uses only data before its date; the model is re-estimated at every origin.", FIGSTYLE, 5.6, 3.0))
    s6.figures.append(Figure("f_err", "Hold-out forecast errors", {"ncols": 1, "panels": [{"xlabel": "", "ylabel": "Error", "legend": True, "layers": [{"type": "line", "x": ho.tolist(), "y": e1.tolist(), "k": 2, "label": "ARIMA"}, {"type": "line", "x": ho.tolist(), "y": e0.tolist(), "k": 3, "label": "Random walk"}, {"type": "hline", "y": 0.0}]}]},
                               "Actual minus forecast.", FIGSTYLE, 5.6, 2.8))
    fo = full.get_forecast(h)
    fm, ci = np.asarray(fo.predicted_mean), np.asarray(fo.conf_int(alpha=.05))
    fx = np.arange(n, n + h, dtype=float)
    k0_ = max(0, n - 60)
    s6.figures.append(Figure("f_fan", f"{h}-step forecast with 95% interval", {"ncols": 1, "panels": [{"xlabel": "", "ylabel": y, "xticks": [[float(i), labels[i]] for i in range(k0_, n, max(1, (n - k0_) // 5))],
                                                                                                    "layers": [{"type": "band", "x": fx.tolist(), "lo": ci[:, 0].tolist(), "hi": ci[:, 1].tolist()}, {"type": "line", "x": obs[k0_:].tolist(), "y": x[k0_:].tolist(), "k": 0}, {"type": "line", "x": [obs[-1]] + fx.tolist(), "y": [float(x[-1])] + fm.tolist(), "k": 2}]}]},
                               "Last 60 observations, then the forecast (dashed) with its 95% interval from the model's forecast variance.", FIGSTYLE, 6.0, 3.0))
    rep.steps.append(s6)

    rep.methods.append(X.P("method", "We fit an ARIMA({p},{d},{q}) model{sea} to {y} ({n} observations). Orders p and q were chosen by BIC on the first {tr} observations, and the last {h} observations were held out for rolling one-step forecasts against a random walk. Stationarity was assessed with ADF and KPSS tests and residuals with Ljung-Box, ARCH LM and Jarque-Bera tests. Estimation used the Open Econometric Systems engine (version {ver}) with statsmodels (version {sv}) in Python {py}.",
                           p=p, d=dd, q=q, sea=(f" with seasonal period {s}" if s > 1 else ""), y=y, n=n, tr=n - h, h=h, ver=__version__, sv=statsmodels.__version__, py=rep.meta["python"]))
    bad = [t.name for st_ in rep.steps for t in st_.tests if t.reject and t.problem]
    if bad:
        rep.limitations.append(X.P("warning", "The following diagnostic tests reject their null hypothesis at the 5% level: {names}. The corresponding assumptions appear violated; interval forecasts in particular may be unreliable.", names=X.join(bad)))
    rep.limitations.append(X.P("note", "Forecast intervals assume normal, serially uncorrelated innovations with constant variance. The ADF and KPSS p-values are interpolated from published tables and are not validated against a second implementation."))
    return rep.finalize()
