"""Static panel-data publication report (pols, fe, re, fd, fedk, twfe).

Steps: 1 data and panel structure, 2 estimation (pooled OLS, fixed effects, random effects side by side), 3 specification tests,
4 residual diagnostics, 5 estimated effects. Estimates come from the same functions as oes_engine.panel (numpy versions that
reproduce linearmodels), so the numbers equal the standard OES result; this layer adds the steps and figures around them.
"""
import math

import numpy as np
import pandas as pd
from scipy import stats

from . import rep_tables as T
from . import rep_text as X
from .core import OESError, as_list, need
from .rep_anova1 import boxstats
from .rep_model import Figure, Report, Step, TestResult
from .rep_tables import fnum, fint, fp, make_table, regression_table, tests_table

FIGSTYLE = "Source: author's calculations."
MAXU = 25


def _pv(b, V, df):
    se = np.sqrt(np.diag(V))
    return se, 2 * stats.t.sf(np.abs(b / se), df)


def build(df, spec, model_id="fe"):
    import sys
    from . import panel as P
    from . import _panel_tests
    from .stat_core import qq_points, thin
    from . import __version__

    y = need(spec, "y", "dependent variable")
    x = as_list(need(spec, "x", "regressors"))
    d, unit, time, N, Tn = P._panel(df, spec, [y] + x)
    n = len(d)
    kind, bw = P._cov_args(spec, "dk" if model_id == "fedk" else "clustered")
    te = str(spec.get("time_effects", "no")).lower() in ("yes", "true", "1")
    u, ti = d["_u"].to_numpy(), d["_ti"].to_numpy()
    Y = d[y].to_numpy(float)
    Xr = d[x].to_numpy(float)
    X1 = np.column_stack([np.ones(n), Xr])

    rep = Report(model_id, f"Panel regression of {y} on " + ", ".join(x), references={})
    Ti = np.bincount(u)
    rep.meta = dict(y=y, x=x, unit=unit, time=time, nobs=n, units=N, periods=Tn, balanced=bool((Ti == Tn).all()), cov=P.COV_LABEL[kind], engine=__version__, python=".".join(map(str, sys.version_info[:3])))

    # ---- estimates
    bp_ = np.linalg.lstsq(X1, Y, rcond=None)[0]
    ep = Y - X1 @ bp_
    Vp = P._sandwich(X1, ep, u, ti, kind, bw)
    sep, pp = _pv(bp_, Vp, n - X1.shape[1])
    cols, fitted, resid, mainlab = [], {}, {}, None
    r2p = 1 - float(ep @ ep) / float(((Y - Y.mean()) ** 2).sum())
    names = ["const"] + x
    cols.append(dict(label="(1)", sub="Pooled OLS", params=pd.Series(bp_, names), se=pd.Series(sep, names), p=pd.Series(pp, names), stats=dict(n=n, r2=r2p, units=N)))
    fitted["pooled"], resid["pooled"] = X1 @ bp_, ep
    fixed_ok = True
    try:
        bf, Vf, ef, yd, extra = P._fe_fit(d, y, x, N, Tn, False, kind, bw)
        df2 = n - extra - len(x)
        sef, pf = _pv(bf, Vf, df2)
        cols.append(dict(label="(2)", sub="Fixed effects", params=pd.Series(bf, x), se=pd.Series(sef, x), p=pd.Series(pf, x), stats=dict(n=n, r2=1 - float(ef @ ef) / float(yd @ yd), units=N)))
        resid["fe"] = ef
    except OESError:
        fixed_ok = False
    if te:
        bt, Vt, et, ydt, extrat = P._fe_fit(d, y, x, N, Tn, True, kind, bw)
        set_, pt = _pv(bt, Vt, n - extrat - len(x))
        cols.append(dict(label=f"({len(cols) + 1})", sub="Two-way FE", params=pd.Series(bt, x), se=pd.Series(set_, x), p=pd.Series(pt, x), stats=dict(n=n, r2=1 - float(et @ et) / float(ydt @ ydt), units=N)))
        resid["twfe"] = et
    br, Xs, es, s2u, s2e, theta = P._re_fit(d, y, x, N)
    Vr = P._sandwich(Xs, es, u, ti, kind, bw)
    ser, pr = _pv(br, Vr, n - len(x) - 1)
    cols.append(dict(label=f"({len(cols) + 1})", sub="Random effects", params=pd.Series(br, names), se=pd.Series(ser, names), p=pd.Series(pr, names), stats=dict(n=n, r2=None, units=N)))
    resid["re"] = es
    if model_id in ("fd", "fedk"):
        g = d.groupby("_u")
        D = d[[y] + x].astype(float) - g[[y] + x].shift(1).astype(float)
        D = D[(d["_ti"] - g["_ti"].shift(1)) == 1]
        dd = d.loc[D.index]
        Yd = D[y].to_numpy(float)
        Xd = np.column_stack([np.ones(len(D)), D[x].to_numpy(float)])
        bd = np.linalg.lstsq(Xd, Yd, rcond=None)[0]
        ed = Yd - Xd @ bd
        Vd = P._sandwich(Xd, ed, dd["_u"].to_numpy(), dd["_ti"].to_numpy(), kind, bw)
        sed, pd_ = _pv(bd, Vd, len(Yd) - Xd.shape[1])
        nm = ["const"] + ["D." + v for v in x]
        cols.append(dict(label=f"({len(cols) + 1})", sub="First differences", params=pd.Series(bd, nm), se=pd.Series(sed, nm), p=pd.Series(pd_, nm),
                         stats=dict(n=len(Yd), r2=1 - float(ed @ ed) / float(((Yd - Yd.mean()) ** 2).sum()), units=N)))
    allnames = names + [("D." + v) for v in x] if model_id in ("fd", "fedk") else names

    # ------------------------------------------------------------------ 1 data
    s1 = Step("data", "Data and panel structure", "Sample, balance and the split of each variable's variation into between and within units.",
              validated="pandas/numpy recomputation of descriptives, panel counts and variance decomposition (test_panel_data)")
    s1.tables.append(T.descriptives_table("t_desc", d, [y] + x, "Descriptive statistics"))
    s1.tables.append(make_table("t_struct", "Panel structure", ["Item", "Value"], [["Units", fint(N)], ["Periods", fint(Tn)], ["Observations", fint(n)],
                                                                                     ["Observations per unit (min / mean / max)", f"{Ti.min()} / {fnum(Ti.mean(), 2)} / {Ti.max()}"],
                                                                                     ["Balanced", "Yes" if rep.meta["balanced"] else "No"]],
                                raw=dict(units=N, periods=Tn, n=n, tmin=int(Ti.min()), tmax=int(Ti.max()), balanced=rep.meta["balanced"])))
    vr = []
    for v in [y] + x:
        a = d[v].to_numpy(float)
        within = a - pd.Series(a).groupby(u).transform("mean").to_numpy() + a.mean()
        between = pd.Series(a).groupby(u).mean().to_numpy()
        vr.append((v, a.std(ddof=1), between.std(ddof=1), within.std(ddof=1)))
    s1.tables.append(make_table("t_var", "Variation overall, between and within units", ["Variable", "Overall s.d.", "Between s.d.", "Within s.d."], [[v] + [fnum(a_, 4) for a_ in r] for v, *r in [(q[0], q[1], q[2], q[3]) for q in vr]],
                                ["Between: standard deviation of unit means. Within: standard deviation after removing unit means and adding back the grand mean."], raw={"rows": [dict(v=q[0], overall=q[1], between=q[2], within=q[3]) for q in vr]}))
    tgrid = np.sort(d["_ti"].unique())
    tl = d.groupby("_ti")["_t"].first().sort_index().to_numpy()
    xt = [[float(i), (str(int(tl[i])) if float(tl[i]).is_integer() else f"{tl[i]:.4g}")] for i in range(0, len(tl), max(1, len(tl) // 7))]
    sel = np.unique(u)[:MAXU]
    lay = [{"type": "line", "x": d.loc[u == s, "_ti"].astype(float).tolist(), "y": d.loc[u == s, y].tolist(), "k": 0, "lw": 0.5} for s in sel]
    s1.figures.append(Figure("f_spag", f"{y} over time by unit", {"ncols": 1, "panels": [{"xlabel": time, "ylabel": y, "xticks": xt, "layers": lay}]},
                             "Each line is one unit" + (f" (the first {MAXU} are drawn)." if N > MAXU else "."), FIGSTYLE, 6.0, 3.4))
    gm = d.groupby("_ti")[[y] + x].mean()
    k4 = min(len(x) + 1, 4)
    s1.figures.append(Figure("f_means", "Cross-sectional means over time", {"ncols": 2 if k4 > 1 else 1, "panels": [{"title": v, "xlabel": time, "xticks": xt, "layers": [{"type": "line", "x": gm.index.astype(float).tolist(), "y": gm[v].tolist(), "k": 0}]} for v in ([y] + x)[:k4]]},
                             "Mean across units in each period.", FIGSTYLE, 6.5, 2.1 * math.ceil(k4 / 2) + 0.5))
    vn = [q[0] for q in vr]
    s1.figures.append(Figure("f_var", "Between and within variation", {"ncols": 1, "panels": [{"ylabel": "Standard deviation", "xticks": [[float(i) + 0.5, v] for i, v in enumerate(vn)], "legend": True,
                                                                                               "layers": [{"type": "bar", "x": [float(i) for i in range(len(vn))], "y": [q[2] for q in vr], "width": 0.4, "fill": "#8a8a8a", "label": "Between"},
                                                                                                          {"type": "bar", "x": [float(i) + 0.45 for i in range(len(vn))], "y": [q[3] for q in vr], "width": 0.4, "fill": "#d9d9d9", "label": "Within"}]}]},
                             "Standard deviation of unit means (between) and of deviations from unit means (within), by variable.", FIGSTYLE, 6.0, 2.8))
    rep.steps.append(s1)

    # ------------------------------------------------------------------ 2 estimation
    s2 = Step("estimation", "Estimation", "Pooled OLS, fixed-effects and random-effects estimates side by side with " + P.COV_LABEL[kind] + " standard errors.",
              validated="pooled OLS equals statsmodels OLS; fixed effects equals least squares with unit dummies (LSDV); random effects equals the Swamy-Arora GLS of the engine, which is checked against linearmodels where installed (test_panel_estimation)")
    tr = regression_table("t_est", "Panel regression estimates", cols, allnames, stat_rows=[("Observations", "n", "int"), ("Units", "units", "int"), ("R²", "r2", "num3")], dep_label=f"Dependent variable: {y}")
    tr.notes += ["Fixed-effects R² is the within R²; random-effects R² is not reported." + (" The first-differences column explains the change in the dependent variable." if model_id in ("fd", "fedk") else "")]
    s2.tables.append(tr)
    cs = [c for c in cols if c["sub"] in ("Pooled OLS", "Fixed effects", "Two-way FE", "Random effects")]
    ncs = len(cs)
    s2.figures.append(Figure("f_coef", "Coefficient estimates across estimators", {"ncols": 1, "panels": [{"xlabel": "Coefficient (95% confidence interval)", "yticks": [[float(i), x[i]] for i in range(len(x))], "ylim": [-0.6, len(x) - 0.4], "legend": True,
                                                                                                          "layers": [{"type": "errh", "y": [float(i) + (j - (ncs - 1) / 2) * 0.18 for i in range(len(x))], "x": [float(c["params"][v]) for v in x],
                                                                                                                      "lo": [float(c["params"][v] - stats.norm.ppf(0.975) * c["se"][v]) for v in x], "hi": [float(c["params"][v] + stats.norm.ppf(0.975) * c["se"][v]) for v in x]} for j, c in enumerate(cs)] + [{"type": "vline", "x": 0.0}]}]},
                               "Estimates with normal-approximation 95% intervals, one row of dots per estimator in the order of the table (top to bottom within each variable).", FIGSTYLE, 6.0, 1.0 + 0.8 * len(x)))
    rep.steps.append(s2)

    # ------------------------------------------------------------------ 3 tests
    s3 = Step("tests", "Specification tests", "Tests of unit effects, of fixed versus random effects, and of cross-sectional dependence.",
              validated="F test from the restricted and unrestricted sums of squares of statsmodels fits; Breusch-Pagan LM, Hausman and Mundlak from their definitions; Pesaran CD from the pairwise residual correlations (test_panel_tests)")
    tests = []
    ssr_p = float(ep @ ep)
    if fixed_ok:
        ssr_f = float(ef @ ef)
        df2 = n - N - len(x)
        F = ((ssr_p - ssr_f) / (N - 1)) / (ssr_f / df2)
        tests.append(TestResult(f"F test of unit effects, F({N - 1}, {df2})", F, float(stats.f.sf(F, N - 1, df2)), "No unit effects (pooled OLS)", None, bool(stats.f.sf(F, N - 1, df2) < 0.05), False))
        h = P._hausman_fe_re(d, y, x, N)
        if h:
            if isinstance(h[2], str) or h[2] is None:
                pass
            hv = None
            bf0, Vf0, *_ = P._fe_fit(d, y, x, N, Tn, False, "unadjusted", None)
            Vr0 = P._sandwich(Xs, es, kind="unadjusted")[1:, 1:]
            dl = bf0 - br[1:]
            H = float(dl @ np.linalg.pinv(Vf0 - Vr0) @ dl)
            if H >= 0:
                ph = float(stats.chi2.sf(H, len(x)))
                tests.append(TestResult(f"Hausman test, FE against RE ({len(x)} df)", H, ph, "Random effects is consistent", None, bool(ph < 0.05), True))
    eb = pd.Series(ep).groupby(u)
    LM = n ** 2 / (2 * float((Ti * (Ti - 1)).sum())) * (float((eb.sum() ** 2).sum()) / ssr_p - 1) ** 2
    tests.append(TestResult("Breusch-Pagan LM test of unit effects", float(LM), float(stats.chi2.sf(LM, 1)), "No unit effects", "1", bool(stats.chi2.sf(LM, 1) < 0.05), False))
    tv = [v for v in x if d.groupby("_u")[v].std().fillna(0).max() > 1e-10]
    if tv:
        M = d[tv].groupby(d["_u"]).transform("mean").to_numpy(float)
        Xm = np.column_stack([np.ones(n), Xr, M])
        bm = np.linalg.lstsq(Xm, Y, rcond=None)[0]
        Vm = P._sandwich(Xm, Y - Xm @ bm, u, kind="clustered")
        idx = slice(1 + len(x), None)
        W = float(bm[idx] @ np.linalg.pinv(Vm[idx, idx]) @ bm[idx])
        tests.append(TestResult("Mundlak test (clustered Wald)", W, float(stats.chi2.sf(W, len(tv))), "Unit means are irrelevant", str(len(tv)), bool(stats.chi2.sf(W, len(tv)) < 0.05), True))
    rmain = resid["twfe"] if te else resid.get("fe", ep)
    Wd = pd.DataFrame({"u": u, "t": ti, "e": rmain}).pivot(index="t", columns="u", values="e")
    cd, pcd, rbar = _panel_tests._cd(Wd.to_numpy(float))
    if cd is not None:
        tests.append(TestResult("Pesaran CD test of cross-sectional dependence", float(cd), float(pcd), "No cross-sectional dependence", None, bool(pcd < 0.05), True))
    s3.tests = tests
    s3.tables.append(tests_table("t_tests", "Specification tests", tests, ["Residuals of the fixed-effects model are used for the CD test (pooled OLS if fixed effects cannot be estimated). Where the Hausman statistic is negative it is omitted; rely on the Mundlak test then."]))
    rep.steps.append(s3)

    # ------------------------------------------------------------------ 4 residuals
    key = "twfe" if te else ("fe" if fixed_ok else "pooled")
    if key == "pooled":
        fit_main = X1 @ bp_
    else:
        yd_ = P._within(d, y, x, te)[1] @ (bt if te else bf)
        fit_main = yd_
    e = rmain
    s4 = Step("resid", "Residual diagnostics", f"Residuals of the {'two-way fixed-effects' if te else 'fixed-effects' if fixed_ok else 'pooled'} model.",
              validated="residuals recomputed by LSDV; normality and moments from scipy (test_panel_resid)")
    a, b_ = thin(fit_main.tolist(), e.tolist(), 600)
    s4.figures.append(Figure("f_rvf", "Residuals against fitted values", {"ncols": 1, "panels": [{"xlabel": "Fitted (within-transformed) value", "ylabel": "Residual", "layers": [{"type": "scatter", "x": a, "y": b_}, {"type": "hline", "y": 0.0}]}]},
                             "At most 600 points are drawn.", FIGSTYLE, 4.6, 3.3))
    z, rs = qq_points(e)
    zz, rr = thin(z.tolist(), rs.tolist(), 400)
    lim = float(max(abs(min(zz)), abs(max(zz))))
    s4.figures.append(Figure("f_qq", "Normal Q-Q plot of the residuals", {"ncols": 1, "panels": [{"xlabel": "Theoretical quantiles", "ylabel": "Standardised residuals", "layers": [{"type": "scatter", "x": zz, "y": rr}, {"type": "line", "x": [-lim, lim], "y": [-lim, lim], "k": 2}]}]},
                             "Points near the line indicate approximately normal residuals.", FIGSTYLE, 4.2, 3.6))
    hc, hn = np.histogram(e, bins=25)
    s4.figures.append(Figure("f_hist", "Distribution of the residuals", {"ncols": 1, "panels": [{"xlabel": "Residual", "ylabel": "Count", "layers": [{"type": "bar", "x": ((hn[:-1] + hn[1:]) / 2).tolist(), "y": hc.tolist(), "width": float(hn[1] - hn[0]), "fill": "#8a8a8a"}]}]},
                             "Histogram with 25 bins.", FIGSTYLE, 4.6, 3.0))
    sel2 = np.unique(u)[:MAXU]
    bx = [boxstats(e[u == s]) for s in sel2]
    s4.figures.append(Figure("f_ubox", "Residuals by unit", {"ncols": 1, "panels": [{"xlabel": "Unit (in sample order)", "ylabel": "Residual", "xticks": [[float(i), str(i + 1)] for i in range(len(sel2))][::max(1, len(sel2) // 12)], "xlim": [-0.6, len(sel2) - 0.4], "layers": [{"type": "box", "x": [float(i) for i in range(len(sel2))], "boxes": bx}, {"type": "hline", "y": 0.0}]}]},
                             "Boxes span the quartiles; whiskers reach 1.5 interquartile ranges." + (f" The first {MAXU} units are drawn." if N > MAXU else ""), FIGSTYLE, 6.5, 3.0))
    mt = pd.Series(e).groupby(ti).mean()
    s4.figures.append(Figure("f_rtime", "Mean residual by period", {"ncols": 1, "panels": [{"xlabel": time, "ylabel": "Mean residual", "xticks": xt, "layers": [{"type": "line", "x": mt.index.astype(float).tolist(), "y": mt.tolist(), "k": 0}, {"type": "hline", "y": 0.0}]}]},
                             "Systematic movement over time points to omitted common shocks; consider time effects.", FIGSTYLE, 6.0, 2.8))
    jb, pjb = stats.jarque_bera(e)
    s4.tests = [TestResult("Jarque-Bera test of residual normality", float(jb), float(pjb), "Residuals are normal", "2", bool(pjb < 0.05), True)]
    s4.tables.append(tests_table("t_resid", "Residual normality", s4.tests, [f"Skewness {fnum(stats.skew(e), 3)}, kurtosis {fnum(stats.kurtosis(e, fisher=False), 3)}."]))
    rep.steps.append(s4)

    # ------------------------------------------------------------------ 5 effects
    s5 = Step("effects", "Estimated unit effects", "Unit effects implied by the fixed-effects estimates.", validated="alpha_i = ybar_i - xbar_i b, compared with LSDV dummy coefficients (test_panel_effects)")
    if fixed_ok:
        bb = bt if te else bf
        al = pd.Series(Y - Xr @ bb).groupby(u).mean().to_numpy()
        order = np.argsort(al)
        s5.tables.append(make_table("t_eff", "Distribution of the estimated unit effects", ["Statistic", "Value"], [["Mean", fnum(al.mean(), 4)], ["Std. dev.", fnum(al.std(ddof=1), 4)], ["Min", fnum(al.min(), 4)], ["Median", fnum(np.median(al), 4)], ["Max", fnum(al.max(), 4)]],
                                    ["Effects are ybar_i minus xbar_i times the slope estimates" + (" (with time effects the unit means are taken from the unit-demeaned series, so the level is a convention)." if te else ".")], raw={"effects": al.tolist()}))
        s5.figures.append(Figure("f_eff", "Estimated unit effects, sorted", {"ncols": 1, "panels": [{"xlabel": "Unit rank", "ylabel": "Unit effect", "layers": [{"type": "scatter", "x": list(map(float, range(1, N + 1))), "y": al[order].tolist(), "r": 2.0}, {"type": "hline", "y": float(al.mean())}]}]},
                                 "Units sorted from the smallest to the largest effect; the line is the mean.", FIGSTYLE, 5.0, 3.2))
        eh, en = np.histogram(al, bins=min(20, max(5, N // 3)))
        s5.figures.append(Figure("f_effh", "Distribution of the unit effects", {"ncols": 1, "panels": [{"xlabel": "Unit effect", "ylabel": "Count", "layers": [{"type": "bar", "x": ((en[:-1] + en[1:]) / 2).tolist(), "y": eh.tolist(), "width": float(en[1] - en[0]), "fill": "#8a8a8a"}]}]},
                                 "Histogram of the estimated unit effects.", FIGSTYLE, 4.6, 3.0))
    else:
        s5.tables.append(make_table("t_eff", "Unit effects", ["Note"], [["Fixed effects cannot be estimated because some regressors do not vary within units."]]))
    s5.tables.append(make_table("t_re", "Random-effects variance components", ["Component", "Estimate"], [["Unit effect variance", fnum(s2u, 5)], ["Idiosyncratic variance", fnum(s2e, 5)], ["Share of variance due to unit effects (rho)", fnum(s2u / (s2u + s2e), 4)], ["Median theta", fnum(float(np.median(theta)), 4)]],
                                raw=dict(s2u=s2u, s2e=s2e, rho=s2u / (s2u + s2e))))
    rep.steps.append(s5)

    rep.methods.append(X.P("method", "We estimate pooled OLS, fixed-effects and random-effects (Swamy-Arora) regressions of {y} on {k} regressor(s) for {N} units over {T} periods ({n} observations). Standard errors are {cov}. The fixed-effects estimator removes unit means{tw}. Specification is assessed with an F test and a Breusch-Pagan test of unit effects, a Hausman test, a Mundlak test and the Pesaran CD test. Estimation used the Open Econometric Systems engine (version {ver}) in Python {py}.",
                           y=y, k=len(x), N=N, T=Tn, n=n, cov=P.COV_LABEL[kind], tw=" and, in the two-way model, period means" if te else "", ver=__version__, py=rep.meta["python"]))
    bad = [t.name for s in rep.steps for t in s.tests if t.reject and t.problem]
    if bad:
        rep.limitations.append(X.P("warning", "The following diagnostic tests reject their null hypothesis at the 5% level: {names}. The corresponding assumptions appear violated; inference should be treated with caution.", names=X.join(bad)))
    if not fixed_ok:
        rep.limitations.append(X.P("warning", "Fixed effects could not be estimated because at least one regressor does not vary within units."))
    rep.limitations.append(X.P("note", "The estimates are associations conditional on the included controls and effects; they have a causal reading only under the usual exogeneity assumptions. Normal-approximation intervals in the coefficient figure ignore the small-sample t correction used in the table."))
    return rep.finalize()
