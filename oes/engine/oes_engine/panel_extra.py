"""Two-way fixed effects with cluster inference, dynamic panel threshold regression, panel cointegration tests,
panel quantile regression and spatial panel models (SAR, SEM, SDM). Same result format as the other models (core.py)."""
import numpy as np
import pandas as pd
from scipy import optimize, stats

from . import _panel_tests
from .core import OESError, as_list, chart, coef_table, distinct, f, int_option, float_option, need, result, stat, table, test_row, tests_table
from .panel import _panel


def _codes(d):
    return d["_u"].to_numpy(int), d["_ti"].to_numpy(int)


def _tw_demean(a, u, t, N, T, iters=200):
    """Remove unit and time means by alternating projections (one pass if balanced)."""
    a = np.asarray(a, float).copy()
    nu = np.bincount(u, minlength=N).astype(float); nt = np.bincount(t, minlength=T).astype(float)
    for _ in range(iters):
        old = a.copy()
        if a.ndim == 1:
            a -= (np.bincount(u, a, N) / nu)[u]; a -= (np.bincount(t, a, T) / nt)[t]
        else:
            for j in range(a.shape[1]):
                a[:, j] -= (np.bincount(u, a[:, j], N) / nu)[u]; a[:, j] -= (np.bincount(t, a[:, j], T) / nt)[t]
        if np.max(np.abs(a - old)) < 1e-11:
            break
    return a


def _cluster_v(X, e, groups, p_total):
    n, k = X.shape; XtXi = np.linalg.pinv(X.T @ X)
    G = int(groups.max()) + 1; S = np.zeros((k, k))
    sc = X * e[:, None]
    acc = np.zeros((G, k)); np.add.at(acc, groups, sc)
    S = acc.T @ acc
    return XtXi @ S @ XtXi * G / (G - 1) * (n - 1) / (n - p_total)


def _balanced(d, N, T, what):
    if len(d) != N * T:
        raise OESError(f"{what} needs a balanced panel (every unit observed in every period). This panel has {N} units, {T} periods and {len(d)} rows.")


# ---------------------------------------------------------------------------------------------------------------
# Two-way fixed effects with the choice of standard errors, wild cluster bootstrap and placebo experiment
# ---------------------------------------------------------------------------------------------------------------
def twfe(df, spec):
    y = need(spec, "y", "dependent variable"); xs = as_list(need(spec, "x", "regressors")); distinct([y], xs)
    d, unit, time, N, T = _panel(df, spec, [y] + xs)
    n = len(d); k = len(xs); u, t = _codes(d)
    yv = d[y].to_numpy(float); Xv = d[xs].to_numpy(float)
    yt = _tw_demean(yv, u, t, N, T); Xt = _tw_demean(Xv, u, t, N, T)
    b = np.linalg.lstsq(Xt, yt, rcond=None)[0]; e = yt - Xt @ b
    p_full = k + (N - 1) + (T - 1) + 1
    XtXi = np.linalg.inv(Xt.T @ Xt)
    iid = XtXi * (e @ e) / (n - p_full)
    hc = XtXi @ ((Xt * e[:, None]).T @ (Xt * e[:, None])) @ XtXi * n / (n - p_full)
    Vu = _cluster_v(Xt, e, u, p_full); Vt = _cluster_v(Xt, e, t, p_full)
    Vut = _cluster_v(Xt, e, np.arange(n), p_full)
    V2 = Vu + Vt - Vut
    ses = {"Default (iid)": iid, "Robust (HC1)": hc, "Cluster by unit": Vu, "Cluster by time": Vt, "Two-way (unit, time)": V2}
    srow = []
    for nm, V in ses.items():
        sd = np.sqrt(np.clip(np.diag(V), 0, None))
        srow.append([nm] + [v for j in range(k) for v in (f(sd[j], 4), f(b[j] / sd[j], 2))])
    cols = ["Method"] + [c for x in xs for c in (f"s.e. {x}", f"t {x}")]
    # comparison with pooled and unit-effects-only estimates
    Xp = np.column_stack([np.ones(n), Xv]); bp = np.linalg.lstsq(Xp, yv, rcond=None)[0]
    ud = _tw_demean_1way(yv, u, N); Xud = np.column_stack([_tw_demean_1way(Xv[:, j], u, N) for j in range(k)])
    bu = np.linalg.lstsq(Xud, ud, rcond=None)[0]
    comp = [[x, f(bp[j + 1], 4), f(bu[j], 4), f(b[j], 4)] for j, x in enumerate(xs)]
    # variance shares
    vs = []
    for nm, a in zip([y] + xs, [yv] + [Xv[:, j] for j in range(k)]):
        tot = np.var(a)
        vs.append([nm, f(np.var(pd.Series(a).groupby(u).transform("mean")) / tot, 3), f(np.var(pd.Series(a).groupby(t).transform("mean")) / tot, 3),
                   f(np.var(_tw_demean(a, u, t, N, T)) / tot, 3)])
    # serial correlation of residuals within units
    ed = pd.DataFrame({"u": u, "e": e}); ed["l"] = ed.groupby("u").e.shift(1); m = ed.dropna()
    rho = float((m.e * m.l).sum() / (m.l ** 2).sum())
    # wild cluster bootstrap-t for the first regressor, null imposed
    R = int_option(spec, "boot", 499, 99, 2000, "Bootstrap draws"); rng = np.random.default_rng(int(spec.get("seed") or 0))
    def tstat(ys):
        bb = np.linalg.lstsq(Xt, ys, rcond=None)[0]; ee = _tw_demean(ys - Xt @ bb, u, t, N, T) if False else ys - Xt @ bb
        V = _cluster_v(Xt, ee, u, p_full); return bb[0], bb[0] / np.sqrt(V[0, 0])
    b_hat, t_hat = tstat(yt)
    X0 = Xt[:, 1:]
    if k > 1:
        b0 = np.linalg.lstsq(X0, yt, rcond=None)[0]; fit0 = X0 @ b0
    else:
        fit0 = np.zeros(n)
    e0 = yt - fit0; tb = []
    for _ in range(R):
        w = rng.choice([-1.0, 1.0], N)[u]
        ys = _tw_demean(fit0 + e0 * w, u, t, N, T)
        tb.append(tstat(ys)[1])
    tb = np.array(tb); p_boot = (np.abs(tb) >= abs(t_hat)).mean()
    p_norm = 2 * stats.norm.sf(abs(t_hat)); p_t = 2 * stats.t.sf(abs(t_hat), N - 1)
    # placebo experiment
    Rp = int_option(spec, "placebo", 200, 0, 1000, "Placebo draws"); rej = {"Default (iid)": 0, "Robust (HC1)": 0, "Cluster by unit": 0}
    prow = None
    if Rp > 0:
        for _ in range(Rp):
            xm = np.zeros((N, T)); uu = rng.standard_normal((N, T))
            for s in range(1, T):
                xm[:, s] = .8 * xm[:, s - 1] + uu[:, s]
            xf = _tw_demean(xm[u, t], u, t, N, T)
            bb = (xf @ yt) / (xf @ xf); ee = yt - bb * xf
            si = np.sqrt(ee @ ee / (n - N - T) / (xf @ xf)); sh = np.sqrt(((xf * ee) ** 2).sum()) / (xf @ xf)
            sc = np.bincount(u, xf * ee, N); sc_ = np.sqrt((sc ** 2).sum() * N / (N - 1)) / (xf @ xf)
            for nm, s_ in zip(rej, (si, sh, sc_)):
                rej[nm] += abs(bb / s_) > 1.96
        prow = [[nm, f(v / Rp, 3)] for nm, v in rej.items()]
    tabs = [table("Variance shares", ["Variable", "Between units", "Between periods", "Left after two-way demeaning"], vs),
            table("Coefficients under three sets of effects", ["Variable", "Pooled OLS", "Unit effects", "Unit + time effects"], comp),
            table("Standard errors for the two-way fixed-effects model", cols, srow,
                  f"Residual autocorrelation within units: {rho:.2f}. {N} unit clusters and {T} period clusters. Two-way clustering follows Cameron, Gelbach and Miller (2011)."),
            table(f"Wild cluster bootstrap for {xs[0]}", ["Item", "Value"],
                  [["Coefficient", f(b_hat, 4)], ["Cluster-robust t", f(t_hat, 3)], ["p-value, normal", f(p_norm, 4)], [f"p-value, t({N - 1})", f(p_t, 4)],
                   [f"Wild cluster bootstrap-t p-value ({R} draws)", f(p_boot, 4) if p_boot > 0 else f"< {1 / R:.4f}"],
                   ["Bootstrap 2.5% and 97.5% quantiles of t", f"{np.percentile(tb, 2.5):.2f}, {np.percentile(tb, 97.5):.2f}"]],
                  "Rademacher weights by unit, null imposed on the first regressor.")]
    if prow:
        tabs.append(table("Placebo experiment: rejection rate of a true 5% null", ["Standard error", "Share rejecting"], prow,
                          f"{Rp} persistent AR(1) (0.8) regressors unrelated to the outcome; a correct test rejects about 5% of the time."))
    ch = [chart("t-statistic: wild cluster bootstrap distribution", list(range(len(tb))), [("bootstrap t (sorted)", np.sort(tb))], "", "t", kind="line")]
    summary = [stat("Observations", n), stat("Units", N), stat("Periods", T), stat(f"Coefficient on {xs[0]}", float(b[0]), 4), stat("Cluster s.e.", float(np.sqrt(Vu[0, 0])), 4), stat("Bootstrap p", float(p_boot), 4)]
    notes = ["Unit and period effects are removed by demeaning (identical coefficients to dummy variables). For staggered treatment timing, two-way fixed effects can be biased by negative weights; use the event-study or synthetic DiD models."]
    return result("twfe", f"Two-way fixed effects with clustered inference: {y}", n, summary, tabs, ch, notes)


def _tw_demean_1way(a, u, N):
    a = np.asarray(a, float); nu = np.bincount(u, minlength=N).astype(float)
    return a - (np.bincount(u, a, N) / nu)[u]


# ---------------------------------------------------------------------------------------------------------------
# Dynamic panel threshold regression (first-difference IV, Seo-Shin style)
# ---------------------------------------------------------------------------------------------------------------
def _tsls(y, X, Z):
    Xh = Z @ (np.linalg.pinv(Z.T @ Z) @ (Z.T @ X))
    b = np.linalg.solve(Xh.T @ X, Xh.T @ y); return b, y - X @ b


def dpthresh(df, spec):
    y = need(spec, "y", "dependent variable"); xr = need(spec, "x", "regressor with the threshold effect"); ctrl = as_list(spec.get("controls"))
    qv = spec.get("tvar") or xr
    distinct([y], [xr], ctrl)
    d, unit, time, N, T = _panel(df, spec, [y, xr] + ctrl + ([qv] if qv not in [y, xr] + ctrl else []))
    if T < 6:
        raise OESError("The dynamic threshold model needs at least 6 periods per unit.")
    d = d.sort_values(["_u", "_ti"]).reset_index(drop=True)
    g = d.groupby("_u")
    def lag(v, L):
        out = g[v].shift(L); okc = (g["_ti"].shift(L) == d["_ti"] - L); return out.where(okc)
    d["dy"] = d[y] - lag(y, 1); d["dy_l1"] = lag(y, 1) - lag(y, 2); d["y_l2"] = lag(y, 2); d["y_l3"] = lag(y, 3)
    d["dx"] = d[xr] - lag(xr, 1); d["q"] = lag(qv, 1)
    for c in ctrl:
        d["d_" + c] = d[c] - lag(c, 1)
    need_cols = ["dy", "dy_l1", "dx", "q", "y_l2", "y_l3"] + ["d_" + c for c in ctrl]
    use = d.dropna(subset=need_cols).copy()
    if len(use) < 100:
        raise OESError("Too few usable observations after differencing and lags (needs consecutive periods).")
    ids = pd.factorize(use["_u"])[0]; G = ids.max() + 1
    yrs = pd.get_dummies(use["_ti"], drop_first=True).astype(float).to_numpy()
    q = use["q"].to_numpy(float); yy = use["dy"].to_numpy(float); X0 = use[["dy_l1"]].to_numpy(float); Xx = use[["dx"]].to_numpy(float)
    K = use[["d_" + c for c in ctrl]].to_numpy(float) if ctrl else np.zeros((len(use), 0)); Zn = use[["y_l2", "y_l3"]].to_numpy(float)
    nobs = len(use)
    # lagged dependent variable under three estimators
    dd = d.dropna(subset=[y]).copy(); dd["yl"] = lag(y, 1).reindex(dd.index); dd = dd.dropna(subset=["yl"])
    ols = np.polyfit(dd["yl"], dd[y], 1)[0]
    ym = dd[y] - dd.groupby("_u")[y].transform("mean"); xm = dd["yl"] - dd.groupby("_u")["yl"].transform("mean"); fe = float((xm @ ym) / (xm @ xm))
    Xl = np.column_stack([X0, Xx, K, yrs]); Zl = np.column_stack([Zn, Xx, K, yrs])
    b_lin, e_lin = _tsls(yy, Xl, Zl); ssr0 = float(e_lin @ e_lin)
    def fit(gam, y_=None):
        lo = (q <= gam).astype(float)[:, None]; hi = 1 - lo
        X = np.hstack([X0, Xx * lo, Xx * hi, K, yrs]); Z = np.hstack([Zn, Xx * lo, Xx * hi, K, yrs])
        b, e = _tsls(yy if y_ is None else y_, X, Z); return b, e, X, Z
    trim = float_option(spec, "trim", .15, "Trimming")
    if not 0.05 <= trim <= 0.4:
        raise OESError("Trimming must be between 0.05 and 0.4.")
    ng = int_option(spec, "grid", 71, 11, 101, "Grid points")
    grid = np.unique(np.quantile(q, np.linspace(trim, 1 - trim, ng)))
    ssr = np.array([float((lambda r: r[1] @ r[1])(fit(gm))) for gm in grid]); kk = int(np.argmin(ssr)); gam = float(grid[kk])
    sig2 = ssr[kk] / (nobs - Xl.shape[1] - 1)
    F_obs = (ssr0 - ssr[kk]) / sig2
    R = int_option(spec, "boot", 99, 19, 499, "Bootstrap draws"); rng = np.random.default_rng(int(spec.get("seed") or 0)); fitted = Xl @ b_lin; Fb = []
    for _ in range(R):
        w = rng.choice([-1.0, 1.0], G)[ids]; ys = fitted + e_lin * w
        s0 = float((lambda r: r[1] @ r[1])(_tsls(ys, Xl, Zl))); s1 = min(float((lambda r: r[1] @ r[1])(fit(gm, ys))) for gm in grid)
        Fb.append((s0 - s1) / (s1 / (nobs - Xl.shape[1] - 1)))
    Fb = np.array(Fb); pboot = float((Fb >= F_obs).mean())
    LR = (ssr - ssr[kk]) / sig2; ok = grid[LR <= 7.35]
    b, e, X, Z = fit(gam)
    Xh = Z @ (np.linalg.pinv(Z.T @ Z) @ (Z.T @ X)); A = np.linalg.inv(Xh.T @ Xh)
    acc = np.zeros((G, X.shape[1])); np.add.at(acc, ids, Xh * e[:, None]); V = A @ (acc.T @ acc) @ A * G / (G - 1)
    se = np.sqrt(np.diag(V)); nm = [f"lagged {y} (change)", f"{xr}, low regime", f"{xr}, high regime"] + [f"{c} (change)" for c in ctrl]
    keep = list(range(3 + len(ctrl)))
    ct = table("Estimates at the estimated threshold", ["Variable", "Coef.", "Cluster s.e.", "t"], [[nm[i], f(b[i], 4), f(se[i], 4), f(b[i] / se[i], 2)] for i in keep])
    dv = float(np.array([1, -1]) @ V[1:3, 1:3] @ np.array([1, -1]))
    gi = np.array([Zn[ids == i].T @ e[ids == i] for i in range(G)])
    try:
        Jv = float(gi.sum(0) @ np.linalg.inv(gi.T @ gi) @ gi.sum(0))
    except np.linalg.LinAlgError:
        Jv = float("nan")
    df_ = pd.DataFrame({"u": ids, "e": e}); df_["e2"] = df_.groupby("u").e.shift(2); m2 = df_.dropna()
    ar2 = float((m2.e * m2.e2).sum() / np.sqrt(((m2.e * m2.e2) ** 2).sum()))
    tests = [test_row("Regime difference in the slope of " + xr, (b[1] - b[2]) / np.sqrt(dv), 2 * stats.norm.sf(abs((b[1] - b[2]) / np.sqrt(dv))), f"difference {b[1] - b[2]:.3f}, s.e. {np.sqrt(dv):.3f}"),
             test_row(f"Threshold effect (sup-F, {R} bootstrap draws)", F_obs, pboot, "H0: no threshold; wild bootstrap by unit, regressors held fixed"),
             test_row("AR(2) in the differenced residuals", ar2, 2 * stats.norm.sf(abs(ar2)), "H0: no second-order autocorrelation"),
             test_row("Hansen J (lagged-level instruments, 1 df)", Jv, stats.chi2.sf(Jv, 1), "H0: instruments valid")]
    tabs = [table("Persistence under three estimators", ["Estimator", f"Coefficient on lagged {y}"], [["Pooled OLS", f(ols, 3)], ["Within (fixed effects)", f(fe, 3)], ["First-difference IV (linear)", f(float(b_lin[0]), 3)]],
                  "OLS is biased upward and the within estimator downward in short panels (Nickell bias); the IV estimate is the one used below."),
            table("Threshold", ["Item", "Value"], [["Estimated threshold", f(gam, 4)], ["Approximate 95% interval", f"[{ok.min():.3f}, {ok.max():.3f}]"],
                                                  ["Share of observations at or below it", f((q <= gam).mean(), 3)], ["IV sum of squares, linear model", f(ssr0, 2)], ["IV sum of squares, threshold model", f(ssr[kk], 2)],
                                                  [f"Bootstrap 90th, 95th, 99th percentile of F", ", ".join(f"{v:.2f}" for v in np.percentile(Fb, [90, 95, 99]))]],
                  f"Threshold variable: lagged {qv}. Search over {len(grid)} grid values between the {int(trim * 100)}th and {int((1 - trim) * 100)}th percentiles. Interval: values with LR below 7.35 (Hansen)."),
            ct, tests_table(tests)]
    ch = [chart("IV residual sum of squares by candidate threshold", grid, [("Threshold model", ssr), ("Linear model", [ssr0] * len(grid))], f"Candidate threshold (lagged {qv})", "Sum of squares")]
    summary = [stat("Observations", nobs), stat("Units", int(G)), stat("Threshold", gam, 4), stat("Bootstrap p", pboot, 3), stat("Regime slopes", f"{b[1]:.3f} / {b[2]:.3f}")]
    notes = ["First-difference instrumental variables: the lagged change is instrumented by the second and third lags of the level; the regressor and controls enter in differences and are treated as exogenous. The threshold variable is predetermined (lagged level). Year dummies are included."]
    return result("dpthresh", f"Dynamic panel threshold: {y}", nobs, summary, tabs, ch, notes)


# ---------------------------------------------------------------------------------------------------------------
# Panel cointegration tests
# ---------------------------------------------------------------------------------------------------------------
def _ols_b(y, X):
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    return b, e, np.sqrt(np.diag(XtXi) * (e @ e) / (len(y) - X.shape[1]))


def _adf_t(e, k=1):
    de = np.diff(e); n = len(de)
    cols = [e[k:-1]] + [de[k - j:n - j] for j in range(1, k + 1)]
    b, _, se = _ols_b(de[k:], np.column_stack(cols)); return b[0] / se[0]


def _group_adf(Y, Xs, k=1):
    ts = []
    for i in range(Y.shape[1]):
        X = np.column_stack([np.ones(len(Y))] + [x[:, i] for x in Xs]); ts.append(_adf_t(_ols_b(Y[:, i], X)[1], k))
    return float(np.mean(ts)), np.array(ts)


def _westerlund_gt(Y, Xs, k=0):
    out = []; T_ = len(Y)
    for i in range(Y.shape[1]):
        y = Y[:, i]; x = np.column_stack([a[:, i] for a in Xs]); dy = np.diff(y); idx = np.arange(k + 1, T_); dx = np.diff(x, axis=0)
        cols = [np.ones(len(idx)), y[idx - 1]] + [x[idx - 1, j] for j in range(x.shape[1])]
        cols += [dy[idx - 1 - L] for L in range(1, k + 1)]
        cols += [dx[idx - 1 - L, j] for L in range(0, k + 1) for j in range(x.shape[1])]
        b, _, se = _ols_b(dy[idx - 1], np.column_stack(cols)); out.append(b[1] / se[1])
    return float(np.mean(out)), np.array(out)


def pcoint(df, spec):
    import statsmodels.api as sm
    y = need(spec, "y", "dependent variable"); xs = as_list(need(spec, "x", "regressors")); distinct([y], xs)
    d, unit, time, N, T = _panel(df, spec, [y] + xs)
    _balanced(d, N, T, "Panel cointegration tests")
    if T < 12 or N < 5:
        raise OESError("Panel cointegration tests need at least 5 units and 12 periods.")
    names = d.drop_duplicates("_u").sort_values("_u")[unit].astype(str).tolist()
    Wl = {v: d.pivot(index="_ti", columns="_u", values=v).sort_index() for v in [y] + xs}
    for v in Wl: Wl[v].columns = names
    rows = []
    for v in [y] + xs:
        lev, dif = Wl[v], Wl[v].diff().dropna()
        c1, c2 = _panel_tests.cips_test(lev), _panel_tests.cips_test(dif); f1, f2 = _panel_tests.fisher_adf_test(lev, maxlag=1), _panel_tests.fisher_adf_test(dif, maxlag=1)
        g = lambda dct, k: (dct or {}).get(k)
        rows.append([v, g(c1, "stat"), g(c1, "p"), g(f1, "Z_p"), g(c2, "stat"), g(c2, "p"), g(f2, "Z_p")])
    Yv = Wl[y].to_numpy(float); Xl = [Wl[v].to_numpy(float) for v in xs]
    kao = _panel_tests.kao_test(d, unit, y, xs) or {}
    g_stat, g_i = _group_adf(Yv, Xl)
    R = int_option(spec, "sims", 200, 50, 1000, "Simulation replications"); rng = np.random.default_rng(int(spec.get("seed") or 0))
    def simulate(fn, demean=False):
        cs = (lambda A: A - A.mean(1, keepdims=True)) if demean else (lambda A: A)
        out = []
        for _ in range(R):
            Ys = np.cumsum(rng.standard_normal(Yv.shape), 0); Xss = [np.cumsum(rng.standard_normal(Yv.shape), 0) for _ in xs]
            out.append(fn(cs(Ys), [cs(a) for a in Xss]))
        return np.array(out)
    tests = [("Pedroni-type group ADF", lambda a, b: _group_adf(a, b)[0]), ("Westerlund Gt, no lags", lambda a, b: _westerlund_gt(a, b, 0)[0]), ("Westerlund Gt, 1 lag", lambda a, b: _westerlund_gt(a, b, 1)[0])]
    res = []
    for lab, fn in tests:
        obs = fn(Yv, Xl); sim = simulate(fn); cv = np.percentile(sim, [5, 10]); res.append([lab, f(obs, 3), f(cv[0], 3), f(cv[1], 3), f(((sim <= obs).sum() + 1) / (len(sim) + 1), 4)])
    # cross-sectional dependence of the residuals and cross-sectional demeaning
    E = np.column_stack([_ols_b(Yv[:, i], np.column_stack([np.ones(T)] + [a[:, i] for a in Xl]))[1] for i in range(N)])
    ed = pd.DataFrame(E).diff().dropna(); cd = _panel_tests.cd_test(ed)
    cs = lambda A: A - A.mean(1, keepdims=True)
    res2 = []
    for lab, fn in tests[:2]:
        obs = fn(cs(Yv), [cs(a) for a in Xl]); sim = simulate(fn, True); res2.append([lab + ", cross-section demeaned", f(obs, 3), f(np.percentile(sim, 5), 3), f(((sim <= obs).sum() + 1) / (len(sim) + 1), 4)])
    # DOLS mean group against fixed effects
    kx = len(xs)
    def dols(i, ll=1):
        yv = Yv[:, i]; x = np.column_stack([a[:, i] for a in Xl]); dx = np.diff(x, axis=0); idx = np.arange(ll + 1, T - ll)
        cols = [np.ones(len(idx))] + [x[idx, j] for j in range(kx)] + [dx[idx - 1 + s, j] for s in range(-ll + 1, ll + 1) for j in range(kx)]
        return _ols_b(yv[idx], np.column_stack(cols))[0][1:1 + kx]
    B = np.array([dols(i) for i in range(N)]); mg = B.mean(0); mg_se = B.std(0, ddof=1) / np.sqrt(N)
    ym = d[y] - d.groupby("_u")[y].transform("mean"); xm = d[xs] - d.groupby("_u")[xs].transform("mean")
    fe = sm.OLS(ym, xm).fit(cov_type="cluster", cov_kwds={"groups": d["_u"].to_numpy()})
    tabs = [table("Panel unit-root tests", ["Variable", "CIPS level", "p", "Fisher p", "CIPS difference", "p ", "Fisher p "], rows,
                  "CIPS allows common shocks; Fisher-ADF assumes independent units. H0: unit root (non-stationary)."),
            table("Residual-based tests of no cointegration", ["Test", "Statistic", "p-value"],
                  [["Kao ADF (common slopes)", kao.get("stat"), kao.get("p")], ["Group-mean ADF on unit residuals", f(g_stat, 3), None]], "Country-by-country ADF t-statistics: " + ", ".join(f"{n}: {v:.1f}" for n, v in zip(names, g_i))),
            table(f"Simulated critical values and p-values ({R} replications)", ["Test", "Statistic", "5% cv", "10% cv", "p-value"], res,
                  "Null: independent random walks, no cointegration. p = (count + 1)/(R + 1). Reject for small (negative) statistics."),
            table("Cross-sectional dependence and robustness", ["Test", "Statistic", "5% cv", "p-value"], res2,
                  f"Pesaran CD test on differenced residuals: CD = {f(cd[0], 2)}, p = {f(cd[1], 4)}, mean absolute correlation {f(cd[2], 2)}." if cd[0] is not None else None),
            table("Long-run coefficients", ["Variable", "Fixed effects (levels)", "FE cluster s.e.", "DOLS mean group", "MG s.e.", "MG median"],
                  [[x, f(float(np.asarray(fe.params)[j]), 3), f(float(np.asarray(fe.bse)[j]), 3), f(mg[j], 3), f(mg_se[j], 3), f(np.median(B[:, j]), 3)] for j, x in enumerate(xs)],
                  "DOLS with one lead and one lag by unit, averaged over units.")]
    ch = [chart("Country ADF t-statistics on the cointegrating residuals", names, [("ADF t", g_i)], "", "t-statistic", kind="bar")]
    summary = [stat("Units", N), stat("Periods", T), stat("Kao p", kao.get("p")), stat("Group ADF", g_stat, 3), stat("Westerlund Gt p", float(res[1][-1]), 3)]
    return result("pcoint", f"Panel cointegration: {y}", len(d), summary, tabs, ch,
                  ["Westerlund Gt is a simplified error-correction version (its own statistic, simulated critical values), not the packaged Westerlund (2007) bootstrap; treat p-values as approximate."])


# ---------------------------------------------------------------------------------------------------------------
# Panel quantile regression: pooled, Canay two-step and MM-QR
# ---------------------------------------------------------------------------------------------------------------
def _pq_estimators(Y, X, taus):
    from statsmodels.regression.quantile_regression import QuantReg
    n_, T_, k_ = X.shape
    Xc = np.column_stack([np.ones(n_ * T_), X.reshape(-1, k_)])
    out = {"pooled": [], "canay": [], "mmqr": []}
    Xm = X - X.mean(1, keepdims=True); ym = Y - Y.mean(1, keepdims=True)
    b = np.linalg.lstsq(Xm.reshape(-1, k_), ym.ravel(), rcond=None)[0]
    alpha = Y.mean(1) - X.mean(1) @ b; ystar = (Y - alpha[:, None]).ravel(); res = (ym - Xm @ b).ravel()
    a_abs = np.abs(ym - Xm @ b); am = a_abs - a_abs.mean(1, keepdims=True)
    gam = np.linalg.lstsq(Xm.reshape(-1, k_), am.ravel(), rcond=None)[0]; delta = a_abs.mean(1) - X.mean(1) @ gam
    scale = (delta[:, None] + X @ gam).ravel(); pos = scale > 1e-3
    for tau in taus:
        out["pooled"].append(QuantReg(Y.ravel(), Xc).fit(q=tau, max_iter=3000).params[1:])
        out["canay"].append(QuantReg(ystar, Xc).fit(q=tau, max_iter=3000).params[1:])
        qt = QuantReg(res[pos], scale[pos]).fit(q=tau, max_iter=3000).params[0]
        out["mmqr"].append(b + gam * qt)
    return {m: np.array(v) for m, v in out.items()}, int((~pos).sum()), b


def pquant(df, spec):
    y = need(spec, "y", "dependent variable"); xs = as_list(need(spec, "x", "regressors")); distinct([y], xs)
    d, unit, time, N, T = _panel(df, spec, [y] + xs)
    _balanced(d, N, T, "Panel quantile regression")
    d = d.sort_values(["_u", "_ti"]); k = len(xs)
    taus = [float(v) for v in str(spec.get("taus") or "0.1,0.25,0.5,0.75,0.9").replace(" ", "").split(",") if v]
    if not taus or any(not 0 < t < 1 for t in taus) or len(taus) < 3:
        raise OESError("Give at least three quantiles strictly between 0 and 1, separated by commas.")
    taus = sorted(taus)
    twoway = str(spec.get("time_effects", "yes")) == "yes"
    Xw = d[xs].copy()
    if twoway and T > 2:
        yd = pd.get_dummies(d["_ti"], drop_first=True).astype(float); yd.columns = [f"_t{c}" for c in yd.columns]; Xw = pd.concat([Xw, yd], axis=1)
    Yarr = d[y].to_numpy(float).reshape(N, T); Xarr = Xw.to_numpy(float).reshape(N, T, -1)
    est, nbad, b_fe = _pq_estimators(Yarr, Xarr, taus)
    R = int_option(spec, "boot", 100, 20, 500, "Bootstrap replications"); rng = np.random.default_rng(int(spec.get("seed") or 0)); boot = {m: [] for m in est}
    for _ in range(R):
        idx = rng.integers(0, N, N)
        try:
            e_, _, _ = _pq_estimators(Yarr[idx], Xarr[idx], taus)
        except Exception:
            continue
        for m in boot: boot[m].append(e_[m])
    boot = {m: np.array(v) for m, v in boot.items()}
    if len(boot["pooled"]) < 10:
        raise OESError("The bootstrap failed too often; check for constant regressors within the resampled units.")
    lab = {"pooled": "Pooled quantile regression", "canay": "Canay two-step fixed-effects QR", "mmqr": "MM-QR (location-scale)"}
    tabs = []
    first = xs[0]
    rows = []
    for m in est:
        for j, tau in enumerate(taus):
            lo, hi = np.percentile(boot[m][:, j, 0], [5, 95]); rows.append([lab[m], tau, f(est[m][j, 0], 4), f(boot[m][:, j, 0].std(ddof=1), 4), f(lo, 4), f(hi, 4)])
    tabs.append(table(f"Coefficient on {first} by quantile", ["Method", "Quantile", "Coef.", "Bootstrap s.e.", "90% low", "90% high"], rows,
                      f"Bootstrap over units ({len(boot['pooled'])} replications). Worker fixed-effects (mean) coefficient: {b_fe[0]:.4f}."))
    full = []
    for m in est:
        for jx, x in enumerate(xs):
            full.append([lab[m], x] + [f(est[m][j, jx], 3) for j in range(len(taus))])
    tabs.append(table("All coefficients by quantile", ["Method", "Variable"] + [f"q{t:g}" for t in taus], full))
    drows = []; i0, i1 = 0, len(taus) - 1
    for m in est:
        dd = boot[m][:, i1, 0] - boot[m][:, i0, 0]; e_ = est[m][i1, 0] - est[m][i0, 0]; lo, hi = np.percentile(dd, [2.5, 97.5]); p = (np.abs(dd - dd.mean()) >= abs(e_)).mean()
        drows.append([lab[m], f(e_, 4), f(lo, 4), f(hi, 4), f(p, 4)])
    tabs.append(table(f"Difference in the {first} coefficient: q{taus[i1]:g} minus q{taus[i0]:g}", ["Method", "Difference", "95% low", "95% high", "p-value (bootstrap)"], drows))
    sd_alpha = float(np.std(Yarr.mean(1) - Xarr.mean(1) @ b_fe)); sd_w = float(np.std((Yarr - Yarr.mean(1, keepdims=True)) - (Xarr - Xarr.mean(1, keepdims=True)) @ b_fe))
    tabs.append(table("Unit effects and scale", ["Item", "Value"], [["SD of the estimated unit effects", f(sd_alpha, 3)], ["SD of the within residuals", f(sd_w, 3)],
                                                                  ["Observations with non-positive fitted scale (dropped in MM-QR)", nbad]]))
    ch = [chart(f"Coefficient on {first} by quantile", taus, [(lab[m], est[m][:, 0]) for m in est] + [("Mean (fixed effects)", [b_fe[0]] * len(taus))], "Quantile", "Coefficient")]
    summary = [stat("Observations", len(d)), stat("Units", N), stat("Periods", T), stat(f"{first}: Canay q{taus[0]:g} to q{taus[-1]:g}", f"{est['canay'][0, 0]:.3f} to {est['canay'][-1, 0]:.3f}")]
    notes = ["Canay (2011): estimate unit effects from a within regression, subtract them, then run pooled quantile regression. MM-QR (Machado and Santos Silva 2019): location-scale model with unit effects in both location and scale. Pooled quantile regression ignores unit effects."]
    return result("pquant", f"Panel quantile regression: {y}", len(d), summary, tabs, ch, notes)


# ---------------------------------------------------------------------------------------------------------------
# Spatial panel: SAR, SEM, SDM with a weights matrix from a neighbour list or coordinates
# ---------------------------------------------------------------------------------------------------------------
def _weights(d, spec, names):
    unit = spec["unit"]; N = len(names); idx = {str(n): i for i, n in enumerate(names)}
    A = np.zeros((N, N))
    ncol = spec.get("neighbours")
    if ncol:
        if ncol not in d.columns:
            raise OESError(f"Column {ncol} not found.")
        first = d.drop_duplicates("_u").sort_values("_u")
        for i, v in zip(first["_u"], first[ncol]):
            if pd.isna(v): continue
            for tok in str(v).replace(",", ";").replace(" ", ";").split(";"):
                tok = tok.strip()
                if not tok: continue
                if tok not in idx:
                    raise OESError(f"Neighbour '{tok}' of unit {names[i]} is not a unit in the data.")
                A[i, idx[tok]] = 1
        A = np.maximum(A, A.T)
    else:
        la, lo = spec.get("lat"), spec.get("lon")
        if not (la and lo):
            raise OESError("Choose either a neighbour-list column or both latitude and longitude columns to build the weights matrix.")
        first = d.drop_duplicates("_u").sort_values("_u")
        P = first[[la, lo]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if np.isnan(P).any():
            raise OESError("Latitude and longitude must be numeric for every unit.")
        k = int_option(spec, "knn", 5, 1, max(N - 1, 1), "Nearest neighbours")
        la_r, lo_r = np.radians(P[:, 0]), np.radians(P[:, 1])
        dl = np.sin((la_r[:, None] - la_r[None]) / 2) ** 2 + np.cos(la_r)[:, None] * np.cos(la_r)[None] * np.sin((lo_r[:, None] - lo_r[None]) / 2) ** 2
        D = 2 * 6371 * np.arcsin(np.sqrt(np.clip(dl, 0, 1))); np.fill_diagonal(D, np.inf)
        for i in range(N):
            A[i, np.argsort(D[i])[:k]] = 1
        A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    if (A.sum(1) == 0).any():
        raise OESError("Some units have no neighbours: " + ", ".join(str(names[i]) for i in np.where(A.sum(1) == 0)[0][:5]))
    return A


def spatial(df, spec):
    y = need(spec, "y", "dependent variable"); xs = as_list(need(spec, "x", "regressors")); distinct([y], xs)
    d, unit, time, N, T = _panel(df, spec, [y] + xs, extra=[c for c in (spec.get("lat"), spec.get("lon")) if c] if not spec.get("neighbours") else [])
    if spec.get("neighbours"):
        d = d.merge(df[[unit, spec["neighbours"]]].drop_duplicates(unit), on=unit, how="left") if spec["neighbours"] not in d.columns else d
    _balanced(d, N, T, "Spatial panel models")
    if N < 10:
        raise OESError("Spatial models need at least 10 units.")
    d = d.sort_values(["_ti", "_u"]).reset_index(drop=True)
    names = d.drop_duplicates("_u").sort_values("_u")[unit].astype(str).tolist()
    A = _weights(d, dict(spec, unit=unit), names); W = A / A.sum(1, keepdims=True)
    om = np.linalg.eigvals(W); k = len(xs)
    lo_b = (1 / om.real.min() if om.real.min() < 0 else -.99) + 1e-3; hi_b = 1 - 1e-3
    Y = d[y].to_numpy(float).reshape(T, N); X = np.stack([d[v].to_numpy(float).reshape(T, N) for v in xs], -1)
    twoway = lambda B: B - B.mean(1, keepdims=True) - B.mean(0, keepdims=True) + B.mean((0, 1))
    Ys = twoway(Y); Xs = np.stack([twoway(X[:, :, j]) for j in range(k)], -1)
    WY = Ys @ W.T; WX = np.stack([Xs[:, :, j] @ W.T for j in range(k)], -1)
    Z = Xs.reshape(-1, k); yv = Ys.ravel(); n_obs = N * T
    b_ols = np.linalg.lstsq(Z, yv, rcond=None)[0]; e = yv - Z @ b_ols; s2 = e @ e / n_obs; E = e.reshape(T, N)
    I_t = np.array([E[t] @ W @ E[t] / (E[t] @ E[t]) for t in range(T)]); I_pool = float((E * (E @ W.T)).sum() / (E ** 2).sum()); EI = -1 / (N - 1)
    tr = float(np.trace(W.T @ W + W @ W)); We = (E @ W.T).ravel(); Wy = WY.ravel(); WXb = WX.reshape(-1, k) @ b_ols
    M = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    D = float((WXb @ M(WXb)) / s2 + T * tr); g_err, g_lag = float(e @ We / s2), float(e @ Wy / s2)
    lm_err = g_err ** 2 / (T * tr); lm_lag = g_lag ** 2 / D
    rlm_err = (g_err - T * tr / D * g_lag) ** 2 / (T * tr - (T * tr) ** 2 / D); rlm_lag = (g_lag - g_err) ** 2 / (D - T * tr)
    ld = lambda r: float(np.sum(np.log(np.abs(1 - r * om))))
    const = -(n_obs / 2) * (np.log(2 * np.pi) + 1)
    def sar_fit(Zm):
        b0 = np.linalg.lstsq(Zm, yv, rcond=None)[0]; b1 = np.linalg.lstsq(Zm, Wy, rcond=None)[0]; e0, e1 = yv - Zm @ b0, Wy - Zm @ b1
        nll = lambda r: (n_obs / 2) * np.log(((e0 - r * e1) @ (e0 - r * e1)) / n_obs) - T * ld(r)
        r = optimize.minimize_scalar(nll, bounds=(lo_b, hi_b), method="bounded").x
        return b0 - r * b1, float(r), float(-nll(r) + const)
    WZs = WX.reshape(-1, k)
    def sem_fit():
        def parts(l):
            yt, Zt = yv - l * Wy, Z - l * WZs; b = np.linalg.lstsq(Zt, yt, rcond=None)[0]; return b, yt - Zt @ b
        nll = lambda l: (n_obs / 2) * np.log((lambda ee: ee @ ee)(parts(l)[1]) / n_obs) - T * ld(l)
        l = optimize.minimize_scalar(nll, bounds=(lo_b, hi_b), method="bounded").x
        return parts(l)[0], float(l), float(-nll(l) + const)
    def hess(fn, p0):
        h = 1e-4 * np.maximum(np.abs(p0), 1e-2); kk = len(p0); H = np.zeros((kk, kk))
        for i in range(kk):
            for j in range(i, kk):
                a = p0.copy(); a[i] += h[i]; a[j] += h[j]; b_ = p0.copy(); b_[i] += h[i]; b_[j] -= h[j]; c = p0.copy(); c[i] -= h[i]; c[j] += h[j]; d_ = p0.copy(); d_[i] -= h[i]; d_[j] -= h[j]
                H[i, j] = H[j, i] = (fn(a) - fn(b_) - fn(c) + fn(d_)) / (4 * h[i] * h[j])
        return np.linalg.inv(-H)
    def sar_ll(p, Zm):
        kk = Zm.shape[1]; b, r, ls2 = p[:kk], p[kk], p[kk + 1]; ee = yv - r * Wy - Zm @ b
        return -(n_obs / 2) * (np.log(2 * np.pi) + ls2) + T * ld(r) - ee @ ee / (2 * np.exp(ls2))
    def sem_ll(p):
        b, l, ls2 = p[:k], p[k], p[k + 1]; ee = (yv - l * Wy) - (Z - l * WZs) @ b
        return -(n_obs / 2) * (np.log(2 * np.pi) + ls2) + T * ld(l) - ee @ ee / (2 * np.exp(ls2))
    Zd = np.concatenate([Xs, WX], -1).reshape(-1, 2 * k)
    b_sar, r_sar, ll_sar = sar_fit(Z); b_sdm, r_sdm, ll_sdm = sar_fit(Zd); b_sem, l_sem, ll_sem = sem_fit()
    ll_ols = -(n_obs / 2) * (np.log(2 * np.pi) + 1 + np.log(s2))
    rs = yv - r_sar * Wy - Z @ b_sar; rd = yv - r_sdm * Wy - Zd @ b_sdm; ytm = yv - l_sem * Wy; ee = ytm - (Z - l_sem * WZs) @ b_sem
    V_sar = hess(lambda p: sar_ll(p, Z), np.r_[b_sar, r_sar, np.log(rs @ rs / n_obs)])
    V_sdm = hess(lambda p: sar_ll(p, Zd), np.r_[b_sdm, r_sdm, np.log(rd @ rd / n_obs)])
    V_sem = hess(sem_ll, np.r_[b_sem, l_sem, np.log(ee @ ee / n_obs)])
    se_ = lambda V, kk: np.sqrt(np.clip(np.diag(V), 0, None))[:kk + 1]
    lab = xs + ["W x " + v for v in xs]
    def cell(b, V, i, kk): return f"{b[i]:.3f} ({se_(V, kk)[i]:.3f})"
    trow = []
    for i, nm in enumerate(lab):
        trow.append([nm, f"{b_ols[i]:.3f}" if i < k else "", cell(b_sar, V_sar, i, k) if i < k else "", cell(b_sem, V_sem, i, k) if i < k else "", cell(b_sdm, V_sdm, i, 2 * k)])
    trow.append(["Spatial coefficient (rho or lambda)", "", f"{r_sar:.3f} ({se_(V_sar, k)[k]:.3f})", f"{l_sem:.3f} ({se_(V_sem, k)[k]:.3f})", f"{r_sdm:.3f} ({se_(V_sdm, 2 * k)[2 * k]:.3f})"])
    trow.append(["Log-likelihood", f"{ll_ols:.1f}", f"{ll_sar:.1f}", f"{ll_sem:.1f}", f"{ll_sdm:.1f}"])
    trow.append(["AIC", f"{2 * (k + 1) - 2 * ll_ols:.1f}", f"{2 * (k + 2) - 2 * ll_sar:.1f}", f"{2 * (k + 2) - 2 * ll_sem:.1f}", f"{2 * (2 * k + 2) - 2 * ll_sdm:.1f}"])
    lr = lambda a, b, df_: (2 * (a - b), stats.chi2.sf(2 * (a - b), df_))
    lrt = []
    for nm, (s_, p_) in [("SAR vs non-spatial (rho = 0)", lr(ll_sar, ll_ols, 1)), ("SEM vs non-spatial (lambda = 0)", lr(ll_sem, ll_ols, 1)), ("SDM vs SAR (theta = 0)", lr(ll_sdm, ll_sar, k)), ("SDM vs SEM (common-factor restriction)", lr(ll_sdm, ll_sem, k))]:
        lrt.append(test_row(nm, s_, p_, "likelihood ratio"))
    diag = [test_row(f"Moran's I of residuals (pooled {I_pool:.3f}; expected {EI:.3f})", I_pool, float("nan"), f"positive in {int((I_t > 0).sum())} of {T} periods"),
            test_row("LM error", lm_err, stats.chi2.sf(lm_err, 1)), test_row("LM lag", lm_lag, stats.chi2.sf(lm_lag, 1)),
            test_row("Robust LM error", rlm_err, stats.chi2.sf(rlm_err, 1)), test_row("Robust LM lag", rlm_lag, stats.chi2.sf(rlm_lag, 1))]
    # direct, indirect and total effects with simulated standard errors
    rng = np.random.default_rng(int(spec.get("seed") or 0)); I_N = np.eye(N); Rs = int_option(spec, "sims", 500, 100, 3000, "Simulation draws")
    def effects(beta, theta, rho):
        S = np.linalg.inv(I_N - rho * W); out = []
        for j in range(k):
            Sk = S @ (beta[j] * I_N + theta[j] * W); dr = np.trace(Sk) / N; tot = Sk.sum() / N; out.append((dr, tot - dr, tot))
        return np.array(out)
    erows = []
    for nm, bb, th, rho, V, kk in [("SAR", b_sar, np.zeros(k), r_sar, V_sar, k), ("SDM", b_sdm[:k], b_sdm[k:], r_sdm, V_sdm, 2 * k)]:
        est_e = effects(bb, th, rho)
        mean = np.r_[b_sar if kk == k else b_sdm, rho]; cov = V[:kk + 1, :kk + 1]; draws = rng.multivariate_normal(mean, cov, Rs); sims = []
        for dr in draws:
            if lo_b < dr[-1] < hi_b:
                sims.append(effects(dr[:k], dr[k:2 * k] if kk == 2 * k else np.zeros(k), dr[-1]))
        sd = np.array(sims).std(0)
        for j, v in enumerate(xs):
            erows.append([nm, v] + [f"{est_e[j, m]:.3f} (t = {est_e[j, m] / sd[j, m]:.1f})" for m in range(3)])
    tabs = [table("Spatial weights", ["Item", "Value"], [["Units", N], ["Neighbour pairs", int(A.sum() / 2)], ["Neighbours per unit: mean, min, max", f"{A.sum(1).mean():.1f}, {int(A.sum(1).min())}, {int(A.sum(1).max())}"],
                                                         ["Bounds for the spatial coefficient", f"{lo_b:.2f} to 1"]], "Row-standardised weights from the neighbour list (symmetrised) or from the nearest neighbours by great-circle distance."),
            table("Spatial dependence in the non-spatial fixed-effects residuals", ["Test", "Statistic", "p-value", "Detail"], [[r[0], r[1], r[2], r[3]] for r in diag], "Anselin et al. (1996) Lagrange multiplier tests, panel version."),
            table("Coefficients (standard errors) after removing unit and time effects", ["Variable", "Non-spatial", "SAR", "SEM", "SDM"], trow),
            tests_table(lrt, "Likelihood-ratio tests"),
            table("Direct, indirect and total effects (simulated t)", ["Model", "Variable", "Direct", "Indirect (spillover)", "Total"], erows,
                  f"In the SAR and SDM the coefficient is not the marginal effect: LeSage and Pace effects, {Rs} draws from the estimated distribution.")]
    ch = [chart("Log-likelihood by spatial coefficient (SAR)", np.linspace(lo_b, hi_b, 40), [("Log-likelihood", [ (lambda r: -((n_obs / 2) * np.log(((lambda a, b_: (a - r * b_) @ (a - r * b_))(yv - Z @ np.linalg.lstsq(Z, yv, rcond=None)[0], Wy - Z @ np.linalg.lstsq(Z, Wy, rcond=None)[0])) / n_obs) - T * ld(r)) + const)(r) for r in np.linspace(lo_b, hi_b, 40)])], "rho", "Log-likelihood")]
    summary = [stat("Observations", n_obs), stat("Units", N), stat("Periods", T), stat("SAR rho", r_sar, 3), stat("SEM lambda", l_sem, 3), stat("Moran's I (residuals)", I_pool, 3)]
    notes = ["Fixed effects are removed by two-way demeaning and the concentrated likelihood is maximised over the spatial coefficient (Lee and Yu 2010 style, without the bias correction). Standard errors come from the numerical Hessian."]
    return result("spatial", f"Spatial panel models: {y}", n_obs, summary, tabs, ch, notes)


# ---------------------------------------------------------------------------------------------------------------
# Time-series cross-section (TSCS) estimators: panel-corrected standard errors (Beck and Katz 1995) and
# feasible GLS with panel heteroskedasticity, contemporaneous correlation and AR(1) errors (Parks 1967, Kmenta 1986)
# ---------------------------------------------------------------------------------------------------------------
def _choice(spec, key, options, default, label):
    v = str(spec.get(key) or default).lower()
    if v not in options:
        raise OESError(f"{label} must be one of: " + ", ".join(options) + ".")
    return v


def _tscs_data(df, spec):
    """Panel, design matrix (constant, regressors, optional unit dummies) and index vectors shared by PCSE and FGLS."""
    y = need(spec, "y", "dependent variable"); xs = as_list(need(spec, "x", "regressors")); distinct([y], xs)
    d, unit, time, N, T = _panel(df, spec, [y] + xs)
    ue = str(spec.get("unit_effects", "no")).lower() in ("yes", "true", "1")
    Y = d[y].to_numpy(float)
    X = np.column_stack([np.ones(len(d)), d[xs].to_numpy(float)])
    km = X.shape[1]
    if ue:
        X = np.column_stack([X, pd.get_dummies(d["_u"], drop_first=True).to_numpy(float)])
    if len(d) <= X.shape[1] + 1:
        raise OESError("There are too few observations for the number of regressors.")
    u, t = _codes(d)
    return d, y, xs, N, T, Y, X, km, u, t, ue


def _has_prev(u, t):
    """True where the previous row is the same unit in the previous period."""
    return np.r_[False, (u[1:] == u[:-1]) & (t[1:] - t[:-1] == 1)]


def _ar1_rho(e, u, t, N, by_unit):
    """AR(1) coefficient of the residuals, sum e_t e_(t-1) / sum e_(t-1)^2 over consecutive pairs: one value, or one per unit."""
    idx = np.where(_has_prev(u, t))[0]
    if len(idx) == 0:
        raise OESError("The AR(1) correction needs units observed in consecutive periods.")
    num, den = e[idx] * e[idx - 1], e[idx - 1] ** 2
    if by_unit:
        rho = np.bincount(u[idx], num, N) / np.maximum(np.bincount(u[idx], den, N), 1e-300)
        rho[np.bincount(u[idx], minlength=N) < 3] = 0.0
    else:
        rho = np.full(N, num.sum() / den.sum())
    return np.clip(rho, -0.99, 0.99), bool(np.any(np.abs(rho) >= 0.99))


def _prais_winsten(Y, X, u, t, rho):
    """Prais-Winsten transformation: y_t - rho y_(t-1) after the first period of a run, sqrt(1 - rho^2) y_t at the first."""
    r = rho[u]
    prev = _has_prev(u, t)
    Yl = np.r_[0.0, Y[:-1]]
    Xl = np.vstack([np.zeros((1, X.shape[1])), X[:-1]])
    Ys = np.where(prev, Y - r * Yl, np.sqrt(1 - r ** 2) * Y)
    Xs = np.where(prev[:, None], X - r[:, None] * Xl, np.sqrt(1 - r ** 2)[:, None] * X)
    return Ys, Xs


def _contemp(e, u, t, N, T):
    """Contemporaneous covariance of the residuals, sum_t e_it e_jt / T_ij over the periods both units are observed, and the counts T_ij."""
    E = np.zeros((T, N)); Mk = np.zeros((T, N))
    E[t, u] = e; Mk[t, u] = 1.0
    num, cnt = E.T @ E, Mk.T @ Mk
    return np.divide(num, cnt, out=np.zeros_like(num), where=cnt > 0), cnt


def _cross_corr(S, cnt):
    """Breusch-Pagan LM test of cross-sectional independence from the residual correlations, and the correlations themselves."""
    sd = np.sqrt(np.maximum(np.diag(S), 1e-300))
    R = S / np.outer(sd, sd)
    iu = np.triu_indices(S.shape[0], 1)
    ok = cnt[iu] >= 3
    r = R[iu][ok]
    lm = float((cnt[iu][ok] * r ** 2).sum())
    return lm, int(ok.sum()), r


def _pcse_vcov(Xs, es, u, t, N, T, diag_only):
    S, _ = _contemp(es, u, t, N, T)
    if diag_only:
        S = np.diag(np.diag(S))
    Xz = np.zeros((T, N, Xs.shape[1])); Xz[t, u] = Xs
    meat = np.einsum("tik,ij,tjl->kl", Xz, S, Xz, optimize=True)
    XtXi = np.linalg.pinv(Xs.T @ Xs)
    return XtXi @ meat @ XtXi


def _wooldridge_ar(d, y, xs):
    """Wooldridge (2002) / Drukker (2003) test for first-order serial correlation: residuals of the first-differenced model on their lag, H0 slope = -0.5."""
    g = d.groupby("_u")
    cols = [y] + list(xs)
    D = d[cols].astype(float) - g[cols].shift(1).astype(float)
    D = D[((d["_ti"] - g["_ti"].shift(1)) == 1).to_numpy()]
    dd = d.loc[D.index]
    b = np.linalg.lstsq(D[xs].to_numpy(float), D[y].to_numpy(float), rcond=None)[0]
    e = pd.Series(D[y].to_numpy(float) - D[xs].to_numpy(float) @ b, index=D.index)
    lag = e.groupby(dd["_u"].values).shift(1)
    ok = lag.notna().to_numpy()
    if ok.sum() < 10:
        return None
    ev, lv, gv = e.to_numpy()[ok], lag.to_numpy()[ok], dd["_u"].to_numpy()[ok]
    rho = float(lv @ ev / (lv @ lv))
    res = ev - rho * lv
    G = np.unique(gv).size
    if G < 3:
        return None
    sc = np.bincount(np.searchsorted(np.unique(gv), gv), lv * res, G)
    V = float((sc ** 2).sum()) / float(lv @ lv) ** 2 * G / (G - 1)
    F = (rho + 0.5) ** 2 / V
    return F, float(stats.f.sf(F, 1, G - 1)), rho, G


def _tscs_diagnostics(d, y, xs, es, S, cnt, u, t, N, T):
    rows = []
    lm, npair, r = _cross_corr(S, cnt)
    if npair:
        rows.append(test_row(f"Breusch-Pagan LM of cross-sectional independence ({npair} df)", lm, stats.chi2.sf(lm, npair),
                             f"H0: no contemporaneous correlation; mean |r| = {f(float(np.abs(r).mean()), 3)}, largest {f(float(np.abs(r).max()), 3)}"))
    W = pd.DataFrame({"u": u, "t": t, "e": es}).pivot(index="t", columns="u", values="e")
    cd, p, rr = _panel_tests._cd(W.to_numpy(float))
    if cd is not None:
        rows.append(test_row("Pesaran CD on residuals", cd, p, f"H0: no cross-sectional dependence; mean |rho| = {f(rr, 3)}"))
    try:
        w = _wooldridge_ar(d, y, xs)
    except Exception:
        w = None
    if w:
        rows.append(test_row(f"Wooldridge test for serial correlation F(1, {w[3] - 1}) (before any correction)", w[0], w[1], f"H0: no first-order autocorrelation; first-difference residual coefficient {f(w[2], 3)} (-0.5 under H0)"))
    return rows, r


def _tscs_charts(S, r):
    sd = np.sqrt(np.diag(S))
    show = min(len(sd), 40)
    ch = [chart("Residual standard deviation by unit", [f"unit {i + 1}" for i in range(show)], [("s.d.", sd[:show])], "", "", kind="bar")]
    if len(r) >= 10:
        hist, edges = np.histogram(r, bins=np.linspace(-1, 1, 21))
        ch.append(chart("Contemporaneous correlations between units' residuals", [f((a + b) / 2, 2) for a, b in zip(edges[:-1], edges[1:])], [("pairs", hist)], "correlation", "number of unit pairs", kind="bar"))
    return ch


def _unit_rho_table(rho, d, unit):
    names = d.groupby("_u")[unit].first().astype(str).to_list()
    return table("Panel-specific AR(1) coefficients", ["Unit", "rho"], [[n, f(v, 4)] for n, v in zip(names, rho)])


def pcse(df, spec):
    d, y, xs, N, T, Y, X, km, u, t, ue = _tscs_data(df, spec)
    corr = _choice(spec, "corr", ("none", "ar1", "psar1"), "none", "Serial correlation")
    cov = _choice(spec, "cov", ("full", "hetonly"), "full", "Covariance")
    n, k = X.shape
    b0 = np.linalg.lstsq(X, Y, rcond=None)[0]
    e0 = Y - X @ b0
    rho, clipped = np.zeros(N), False
    if corr != "none":
        rho, clipped = _ar1_rho(e0, u, t, N, corr == "psar1")
        Ys, Xs = _prais_winsten(Y, X, u, t, rho)
        b = np.linalg.lstsq(Xs, Ys, rcond=None)[0]
        es = Ys - Xs @ b
    else:
        Ys, Xs, b, es = Y, X, b0, e0
    V = _pcse_vcov(Xs, es, u, t, N, T, cov == "hetonly")
    if np.any(np.diag(V)[:km] <= 0):
        raise OESError("The panel-corrected covariance matrix is not positive definite (too few overlapping periods between units). Choose panel heteroskedasticity only.")
    se = np.sqrt(np.diag(V))
    names = ["const"] + list(xs)
    # standard errors compared: conventional and clustered OLS on the same (transformed) data, and PCSE
    XtXi = np.linalg.pinv(Xs.T @ Xs)
    se_ols = np.sqrt(np.diag(XtXi) * float(es @ es) / (n - k))
    se_cl = np.sqrt(np.diag(_cluster_v(Xs, es, u, k)))
    comp = table("Standard errors compared", ["Variable", "Coef.", "Conventional", "Clustered by unit", "Panel-corrected", "PCSE / conventional"],
                 [[names[i], f(b[i]), f(se_ols[i]), f(se_cl[i]), f(se[i]), f(se[i] / se_ols[i], 3)] for i in range(km)],
                 "Same coefficients, three ways of estimating their precision. Conventional errors assume independent, homoskedastic residuals; clustered errors allow serial correlation within units but not correlation between units.")
    S, cnt = _contemp(es, u, t, N, T)
    drows, r = _tscs_diagnostics(d, y, xs, es, S, cnt, u, t, N, T)
    bs, Vs = b[1:km], V[1:km, 1:km]
    W = float(bs @ np.linalg.pinv(Vs) @ bs)
    drows.insert(0, test_row(f"Wald test that all slopes are zero ({km - 1} df)", W, stats.chi2.sf(W, km - 1), "H0: all slope coefficients are zero (panel-corrected covariance)"))
    fitted = X @ b
    r2 = 1 - float(((Y - fitted) ** 2).sum()) / float(((Y - Y.mean()) ** 2).sum())
    covtxt = "contemporaneous correlation and panel heteroskedasticity" if cov == "full" else "panel heteroskedasticity only"
    est = {"none": "pooled OLS", "ar1": "Prais-Winsten with a common AR(1) coefficient", "psar1": "Prais-Winsten with panel-specific AR(1) coefficients"}[corr]
    tabs = [coef_table(names, b[:km], se[:km], dist="z", note=f"Estimator: {est}. Standard errors robust to {covtxt} (Beck and Katz 1995), no small-sample correction."
                       + (" Unit dummies are included but not shown." if ue else "")), comp, tests_table(drows)]
    if corr == "psar1":
        tabs.insert(2, _unit_rho_table(rho, d, spec.get("unit")))
    summary = [stat("Observations", n), stat("Units", N), stat("Periods", T), stat("R-squared", r2), stat("Wald chi-squared", W)]
    if corr == "ar1":
        summary.append(stat("AR(1) rho", float(rho[0]), 3))
    notes = ["Panel-corrected standard errors (Beck and Katz 1995) keep the pooled OLS (or Prais-Winsten) coefficients and replace the standard errors by a sandwich "
             "with the N x N contemporaneous covariance matrix of the residuals, estimated from T periods. They work best when T is at least about 15; "
             "serial correlation is better handled in the model (AR(1) correction or a lagged dependent variable) than in the standard errors."]
    if T < N:
        notes.append(f"T = {T} is smaller than N = {N}: the contemporaneous covariance matrix is singular, which PCSE tolerates, but its estimate is noisy.")
    if len(d) != N * T:
        notes.append("The panel is unbalanced: each covariance is computed over the periods in which both units are observed.")
    if clipped:
        notes.append("An estimated AR(1) coefficient was truncated at 0.99 in absolute value.")
    return result("pcse", f"Panel-corrected standard errors (PCSE): {y}", n, summary, tabs, _tscs_charts(S, r), notes)


def fgls(df, spec):
    d, y, xs, N, T, Y, X, km, u, t, ue = _tscs_data(df, spec)
    panels = _choice(spec, "panels", ("iid", "het", "corr"), "het", "Error structure across units")
    corr = _choice(spec, "corr", ("none", "ar1", "psar1"), "none", "Serial correlation")
    n, k = X.shape
    if panels == "corr":
        _balanced(d, N, T, "FGLS with contemporaneous correlation")
        if T < N:
            raise OESError(f"FGLS with contemporaneous correlation needs at least as many periods as units (T = {T}, N = {N}): the N x N covariance matrix cannot be estimated. "
                           "Choose panel heteroskedasticity only, or use panel-corrected standard errors.")
    b0 = np.linalg.lstsq(X, Y, rcond=None)[0]
    e0 = Y - X @ b0
    rho, clipped = np.zeros(N), False
    if corr != "none":
        rho, clipped = _ar1_rho(e0, u, t, N, corr == "psar1")
        Ys, Xs = _prais_winsten(Y, X, u, t, rho)
    else:
        Ys, Xs = Y, X
    bt = np.linalg.lstsq(Xs, Ys, rcond=None)[0]
    es = Ys - Xs @ bt
    Ti = np.bincount(u, minlength=N).astype(float)
    s2i = np.bincount(u, es ** 2, N) / Ti
    s2 = float(es @ es) / n
    S, cnt = _contemp(es, u, t, N, T)
    if panels == "corr":
        if np.linalg.cond(S) > 1e12:
            raise OESError("The estimated contemporaneous covariance matrix is numerically singular (T is too close to N or units are collinear). Choose panel heteroskedasticity only.")
        Si = np.linalg.inv(S)
        Xz = np.zeros((T, N, k)); Xz[t, u] = Xs
        Yz = np.zeros((T, N)); Yz[t, u] = Ys
        A = np.einsum("tik,ij,tjl->kl", Xz, Si, Xz, optimize=True)
        b = np.linalg.solve(A, np.einsum("tik,ij,tj->k", Xz, Si, Yz, optimize=True))
        V = np.linalg.inv(A)
    else:
        w = 1.0 / s2i[u] if panels == "het" else np.full(n, 1.0 / s2)
        A = (Xs * w[:, None]).T @ Xs
        b = np.linalg.solve(A, (Xs * w[:, None]).T @ Ys)
        V = np.linalg.inv(A)
    se = np.sqrt(np.diag(V))
    names = ["const"] + list(xs)
    eg = Ys - Xs @ b                                  # GLS residuals on the transformed data
    drows, r = _tscs_diagnostics(d, y, xs, es, S, cnt, u, t, N, T)
    lr = n * np.log(s2) - float((Ti * np.log(np.maximum(s2i, 1e-300))).sum())
    drows.insert(0, test_row(f"Likelihood-ratio test of panel heteroskedasticity ({N - 1} df)", lr, stats.chi2.sf(lr, N - 1), "H0: the error variance is the same in every unit"))
    bs, Vs = b[1:km], V[1:km, 1:km]
    W = float(bs @ np.linalg.pinv(Vs) @ bs)
    drows.insert(0, test_row(f"Wald test that all slopes are zero ({km - 1} df)", W, stats.chi2.sf(W, km - 1), "H0: all slope coefficients are zero"))
    # comparison with pooled OLS: conventional and panel-corrected standard errors
    Vp = _pcse_vcov(X, e0, u, t, N, T, False)
    se_ols = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * float(e0 @ e0) / (n - k))
    se_pc = np.sqrt(np.maximum(np.diag(Vp), 0))
    comp = table("FGLS compared with pooled OLS", ["Variable", "OLS coef.", "OLS conventional s.e.", "OLS panel-corrected s.e.", "FGLS coef.", "FGLS s.e.", "FGLS s.e. / panel-corrected"],
                 [[names[i], f(b0[i]), f(se_ols[i]), f(se_pc[i]), f(b[i]), f(se[i]), f(se[i] / se_pc[i], 3)] for i in range(km)],
                 "Beck and Katz (1995) show that the Parks FGLS standard errors can be far too small when T is not much larger than N: a ratio well below 1 in the last column is that warning.")
    structure = {"iid": "homoskedastic and independent errors", "het": "panel-specific error variances", "corr": "panel-specific variances and contemporaneous correlation between units"}[panels]
    ar = {"none": "no serial correlation", "ar1": "a common AR(1) coefficient (Prais-Winsten)", "psar1": "panel-specific AR(1) coefficients (Prais-Winsten)"}[corr]
    tabs = [coef_table(names, b[:km], se[:km], dist="z", note=f"Feasible GLS with {structure} and {ar}; z statistics." + (" Unit dummies are included but not shown." if ue else "")), comp, tests_table(drows)]
    if corr == "psar1":
        tabs.insert(2, _unit_rho_table(rho, d, spec.get("unit")))
    r2 = 1 - float(((Y - X @ b) ** 2).sum()) / float(((Y - Y.mean()) ** 2).sum())
    summary = [stat("Observations", n), stat("Units", N), stat("Periods", T), stat("R-squared", r2), stat("Wald chi-squared", W)]
    if corr == "ar1":
        summary.append(stat("AR(1) rho", float(rho[0]), 3))
    notes = ["Feasible GLS in the steps of Parks (1967) and Kmenta (1986): pooled OLS, the AR(1) coefficient from its residuals, a Prais-Winsten transformation, the error covariance "
             "(per unit, and between units) from the transformed OLS residuals, then GLS. The covariance matrix of the estimates is the GLS one, (X' Omega^-1 X)^-1, with no sandwich."]
    if panels == "corr" and T < 3 * N:
        notes.append(f"T = {T} is less than three times N = {N}: the Parks standard errors are likely too small (Beck and Katz 1995). Compare them with the panel-corrected ones in the table above.")
    if clipped:
        notes.append("An estimated AR(1) coefficient was truncated at 0.99 in absolute value.")
    return result("fgls", f"Feasible GLS (Parks-Kmenta): {y}", n, summary, tabs, _tscs_charts(S, r), notes)


# ---------------------------------------------------------------------------------------------------------------
# Hausman-Taylor estimator: random-effects efficiency with endogenous regressors, time-invariant ones included
# ---------------------------------------------------------------------------------------------------------------
def hausman_taylor(df, spec):
    y = need(spec, "y", "dependent variable")
    x1 = as_list(spec.get("x1") or []); x2 = as_list(spec.get("x2") or []); z1 = as_list(spec.get("z1") or []); z2 = as_list(spec.get("z2") or [])
    if not (x1 or x2):
        raise OESError("Give at least one time-varying regressor (exogenous or endogenous).")
    allv = x1 + x2 + z1 + z2
    d, unit, time, N, T = _panel(df, spec, [y] + allv)
    X = x1 + x2
    if len(x1) < len(z2):
        raise OESError(f"The model is not identified: the number of exogenous time-varying regressors ({len(x1)}) must be at least the number of endogenous time-invariant regressors ({len(z2)}).")
    u = d["_u"].to_numpy(int); n = len(d)
    for v in z1 + z2:
        if d.groupby("_u")[v].std().fillna(0).max() > 1e-10:
            raise OESError(f"{v} varies within units; list it as a time-varying regressor.")
    for v in X:
        if d.groupby("_u")[v].std().fillna(0).max() <= 1e-10:
            raise OESError(f"{v} does not vary within units; list it as a time-invariant regressor.")
    Ti = np.bincount(u).astype(float)
    if (Ti < 2).any():
        raise OESError("Every unit needs at least two periods.")
    gm = lambda M: pd.DataFrame(np.asarray(M, float)).groupby(u).transform("mean").to_numpy()
    Y = d[y].to_numpy(float); Xm = d[X].to_numpy(float); Xw = Xm - gm(Xm); Yw = Y - gm(Y[:, None])[:, 0]
    bw = np.linalg.lstsq(Xw, Yw, rcond=None)[0]; ew = Yw - Xw @ bw
    se2 = float(ew @ ew) / (n - N - len(X))
    # between step: group-mean residuals on the time-invariant regressors, instrumented by the means of the exogenous time-varying ones
    ob = d.groupby("_u")[[y] + X + z1 + z2].mean(); di = ob[y].to_numpy() - ob[X].to_numpy() @ bw
    if z1 or z2:
        Zm = np.column_stack([np.ones(N), ob[z1 + z2].to_numpy()]); Am = np.column_stack([np.ones(N), ob[x1].to_numpy() if x1 else np.zeros((N, 0)), ob[z1].to_numpy() if z1 else np.zeros((N, 0))])
        Pz = Am @ np.linalg.lstsq(Am, Zm, rcond=None)[0]; gam = np.linalg.solve(Pz.T @ Zm, Pz.T @ di); rs = di - Zm @ gam
    else:
        rs = di - di.mean()
    su2 = float(np.mean(rs ** 2 - se2 / Ti))
    if su2 <= 0:
        su2 = 1e-8
    theta = 1 - np.sqrt(se2 / (se2 + Ti * su2))                       # weight on the unit mean that is removed (theta_i = 1 - sqrt(s_e2 / (s_e2 + T_i s_u2)))
    cols = X + z1 + z2; W = np.column_stack([np.ones(n), d[cols].to_numpy(float)])
    th = theta[u][:, None]; Wt = W - th * gm(W); Yt = Y - th[:, 0] * gm(Y[:, None])[:, 0]
    IV = np.column_stack([np.ones(n), Xw] + ([gm(d[x1].to_numpy(float))] if x1 else []) + ([d[z1].to_numpy(float)] if z1 else []))
    PW = IV @ np.linalg.lstsq(IV, Wt, rcond=None)[0]; b = np.linalg.solve(PW.T @ Wt, PW.T @ Yt)
    e = Yt - Wt @ b; s2e = float(e @ e) / (n - W.shape[1])
    V = s2e * np.linalg.inv(PW.T @ Wt); se = np.sqrt(np.diag(V)); names = ["const"] + cols
    # Hausman test of HT against the within estimator, on the time-varying coefficients
    bh = b[1:1 + len(X)]; Vh = V[1:1 + len(X), 1:1 + len(X)]; Vw = se2 * np.linalg.inv(Xw.T @ Xw); dd = bw - bh; Vd = Vw - Vh
    ev, evec = np.linalg.eigh((Vd + Vd.T) / 2); pos = ev > 1e-12; H = float(((evec[:, pos].T @ dd) ** 2 / ev[pos]).sum()); dfh = int(pos.sum())
    # over-identification: HT against the within estimator is the test of the extra instruments (the means of x1 and the z1 variables)
    role = {**{v: "time-varying, exogenous" for v in x1}, **{v: "time-varying, endogenous" for v in x2}, **{v: "time-invariant, exogenous" for v in z1}, **{v: "time-invariant, endogenous" for v in z2}, "const": ""}
    rows = [[nm, role[nm], f(b[i], 4), f(se[i], 4), f(b[i] / se[i], 2), f(2 * stats.norm.sf(abs(b[i] / se[i])), 4)] for i, nm in enumerate(names)]
    tabs = [table(f"Hausman-Taylor estimates (dependent variable: {y})", ["Variable", "Role", "Coefficient", "s.e.", "z", "p-value"], rows,
                  "Instruments: the deviations of all time-varying regressors from their unit means, the unit means of the exogenous time-varying regressors, and the exogenous time-invariant regressors. Standard errors assume the random-effects error structure.")]
    wrows = [[nm, f(bw[i], 4), f(np.sqrt(Vw[i, i]), 4), f(bh[i], 4), f(np.sqrt(Vh[i, i]), 4)] for i, nm in enumerate(X)]
    tabs.append(table("Time-varying coefficients: within (fixed effects) and Hausman-Taylor", ["Variable", "Within", "s.e.", "Hausman-Taylor", "s.e."], wrows, "The within estimator ignores the time-invariant regressors and is consistent whatever the correlation with the unit effect."))
    tabs.append(tests_table([test_row(f"Hausman test: within against Hausman-Taylor ({dfh} df)", H, stats.chi2.sf(H, max(dfh, 1)), "H0: the Hausman-Taylor classification is valid (the extra instruments are exogenous)")],
                            title="Specification test"))
    tabs.append(table("Variance components", ["Component", "Estimate"], [["sigma_e (idiosyncratic)", f(np.sqrt(se2), 4)], ["sigma_u (unit effect)", f(np.sqrt(su2), 4)], ["rho = sigma_u^2 / (sigma_u^2 + sigma_e^2)", f(su2 / (su2 + se2), 4)],
                                                      ["theta (weight removed from unit means)", f(float(theta.mean()), 4)]], "A large share of the variance is between units; theta near 1 means the estimator is close to the within estimator."))
    return result("hausman_taylor", f"Hausman-Taylor: {y}", n, [stat("Observations", n), stat("Units", N), stat("Hausman test p-value", float(stats.chi2.sf(H, max(dfh, 1))), 4), stat("rho", su2 / (su2 + se2), 3)], tabs, [],
                  ["The classification of each variable as exogenous or endogenous is an assumption; the Hausman test checks it only partly.", "Time-invariant endogenous variables are identified only because some time-varying variables are assumed uncorrelated with the unit effect."])


# ---------------------------------------------------------------------------------------------------------------
# Panel unit-root and stationarity tests: Levin-Lin-Chu type, Im-Pesaran-Shin, Fisher-type, Hadri
# ---------------------------------------------------------------------------------------------------------------
def _pur_adf(y, p, det):
    """ADF regression of dy on y(-1), p lagged differences and deterministic terms. Returns t, e-tilde parts (for LLC) and sigma."""
    dy = np.diff(y); n = len(dy) - p
    if n < p + 6:
        return None
    Z = [np.ones(n)] + ([np.arange(1, n + 1, dtype=float)] if det == "ct" else []) + [dy[p - j: len(dy) - j] for j in range(1, p + 1)]
    Z = np.column_stack(Z); yt = dy[p:]; yl = y[p: len(y) - 1]
    X = np.column_stack([yl, Z]); b = np.linalg.lstsq(X, yt, rcond=None)[0]; e = yt - X @ b
    dfree = n - X.shape[1]; s2 = float(e @ e) / dfree
    try:
        se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
    except np.linalg.LinAlgError:
        return None
    Q = np.eye(n) - Z @ np.linalg.pinv(Z)
    ea, va = Q @ yt, Q @ yl
    return dict(t=float(b[0] / se), rho=float(b[0]), n=n, s=float(np.sqrt(s2)), e=ea / np.sqrt(s2), v=va / np.sqrt(s2), bic=n * np.log(float(e @ e) / n) + X.shape[1] * np.log(n))


def _pur_select(y, maxlag, det, ic):
    best = None
    nmax = len(y) - 1 - maxlag
    for p in range(maxlag + 1):
        # common effective sample for comparing lag lengths
        yy = y[maxlag - p:]
        r = _pur_adf(yy, p, det)
        if r is None:
            continue
        k = 1 + (2 if det == "ct" else 1) + p
        n = r["n"]; rss = (r["s"] ** 2) * (n - k)
        val = n * np.log(rss / n) + k * (np.log(n) if ic == "bic" else 2)
        if best is None or val < best[0]:
            best = (val, p)
    if best is None:
        return None, None
    return best[1], _pur_adf(y, best[1], det)


def _pur_llc(parts):
    e = np.concatenate([q["e"] for q in parts]); v = np.concatenate([q["v"] for q in parts])
    d = float(v @ e) / float(v @ v); r = e - d * v; s2 = float(r @ r) / (len(e) - 1)
    return d / np.sqrt(s2 / float(v @ v)), len(e)


def _pur_hadri(Y, det, hetero):
    T, N = Y.shape; tt = np.arange(1, T + 1, dtype=float)
    Xd = np.column_stack([np.ones(T)] + ([tt] if det == "ct" else []))
    E = Y - Xd @ np.linalg.lstsq(Xd, Y, rcond=None)[0]
    S = np.cumsum(E, axis=0); num = (S ** 2).sum(axis=0) / T ** 2
    s2 = (E ** 2).sum(axis=0) / T if hetero else np.full(N, (E ** 2).sum() / (N * T))
    LM = float(np.mean(num / s2)); xi, z2 = (1 / 6, 1 / 45) if det == "c" else (1 / 15, 11 / 6300)
    return LM, float(np.sqrt(N) * (LM - xi) / np.sqrt(z2))


def _pur_sim_moments(T, maxlag, ic, det, reps, rng, cache):
    """Mean and variance of the individual ADF t statistic under a random walk, for a series of length T, with the lag order chosen by the same criterion (cached)."""
    key = (T, maxlag, ic, det)
    if key not in cache:
        ts = []
        for _ in range(reps):
            r = _pur_select(np.cumsum(rng.standard_normal(T + 50))[50:], maxlag, det, ic)[1]
            if r is not None:
                ts.append(r["t"])
        cache[key] = (float(np.mean(ts)), float(np.var(ts)))
    return cache[key]


def punitroot(df, spec):
    from statsmodels.tsa.stattools import mackinnonp
    y = need(spec, "y", "variable to test")
    d, unit, time, N, T = _panel(df, spec, [y])
    det = _choice(spec, "trend", ["c", "ct"], "c", "Deterministic terms"); ic = _choice(spec, "ic", ["aic", "bic"], "aic", "Lag criterion")
    maxlag = int_option(spec, "maxlag", 2, 0, 8, "Maximum lags"); reps = int_option(spec, "sims", 200, 50, 2000, "Simulation replications")
    demean = _choice(spec, "demean", ["no", "yes"], "no", "Cross-sectional demeaning"); seed = int_option(spec, "seed", 0, 0, 10 ** 6, "Random seed")
    names = d.drop_duplicates("_u").sort_values("_u")[unit].astype(str).tolist()
    series = [g.sort_values("_ti")[y].to_numpy(float) for _, g in d.groupby("_u", sort=True)]
    if demean == "yes":
        if len({len(s) for s in series}) != 1:
            raise OESError("Cross-sectional demeaning needs a balanced panel.")
        M = np.column_stack(series); series = list((M - M.mean(axis=1, keepdims=True)).T)
    mint = 10 + maxlag
    if min(len(s) for s in series) < mint:
        raise OESError(f"Panel unit-root tests need at least {mint} periods for every unit with {maxlag} lags (shortest unit has {min(len(s) for s in series)}).")
    if N < 4:
        raise OESError("Panel unit-root tests need at least 4 units.")
    rng = np.random.default_rng(seed); cache = {}
    ind, parts, lags = [], [], []
    for s in series:
        p, r = _pur_select(s, maxlag, det, ic)
        if r is None:
            raise OESError("A unit has too few usable observations for the chosen number of lags.")
        ind.append(r); parts.append(r); lags.append(p)
    tt = np.array([r["t"] for r in ind]); Ts = [len(s) for s in series]
    pv = np.array([mackinnonp(t, regression=det, N=1) for t in tt])
    # Im-Pesaran-Shin t-bar, standardised with simulated moments for each unit's length and lag order
    mom = np.array([_pur_sim_moments(Ts[i], maxlag, ic, det, max(reps, 200), rng, cache) for i in range(N)])
    tbar = float(tt.mean()); Wt = float(np.sqrt(N) * (tbar - mom[:, 0].mean()) / np.sqrt(mom[:, 1].mean()))
    p_ips = float(stats.norm.cdf(Wt))
    # Fisher-type combinations of the individual p-values
    P = float(-2 * np.log(np.clip(pv, 1e-12, 1)).sum()); p_P = float(stats.chi2.sf(P, 2 * N))
    Z = float(stats.norm.ppf(np.clip(pv, 1e-12, 1 - 1e-12)).sum() / np.sqrt(N)); p_Z = float(stats.norm.cdf(Z))
    # Levin-Lin-Chu pooled t, calibrated by simulation of the same statistic under independent random walks
    tl, nl = _pur_llc(parts)
    sims = []
    for _ in range(reps):
        pr = []
        for i in range(N):
            r = _pur_select(np.cumsum(rng.standard_normal(Ts[i] + 50))[50:], maxlag, det, ic)[1]
            if r is not None:
                pr.append(r)
        sims.append(_pur_llc(pr)[0])
    sims = np.array(sims); z_llc = float((tl - sims.mean()) / sims.std(ddof=1)); p_llc = float(stats.norm.cdf(z_llc))
    # Hadri stationarity test (null: all panels stationary)
    hetero = _choice(spec, "hadri_variance", ["common", "unit"], "common", "Hadri variance") == "unit"
    Hl = None
    if len({len(s) for s in series}) == 1:
        Yb = np.column_stack(series); LM, Zh = _pur_hadri(Yb, det, hetero); Hl = (LM, Zh, float(stats.norm.sf(Zh)))
    lab = "constant" if det == "c" else "constant and trend"
    rows = [["Levin-Lin-Chu type (common unit root)", f(tl, 3), f(z_llc, 3), f(p_llc, 4), "H0: all series have a unit root; H1: all are stationary with a common rho"],
            ["Im-Pesaran-Shin t-bar (W)", f(tbar, 3), f(Wt, 3), f(p_ips, 4), "H0: all series have a unit root; H1: some are stationary"],
            ["Fisher-type, inverse chi-squared (P)", f(P, 2), f(P, 2), f(p_P, 4), "H0: all series have a unit root; H1: some are stationary"],
            ["Fisher-type, inverse normal (Z)", f(Z, 3), f(Z, 3), f(p_Z, 4), "H0: all series have a unit root; H1: some are stationary"]]
    if Hl:
        rows.append(["Hadri LM (stationarity)", f(Hl[0], 4), f(Hl[1], 3), f(Hl[2], 4), "H0: all series are stationary; H1: some have a unit root"])
    tabs = [table(f"Panel unit-root and stationarity tests: {y} ({lab}, {N} units)", ["Test", "Statistic", "Standardised (z)", "p-value", "Hypotheses"], rows,
                  "The first four tests have a unit root under the null (reject for stationarity); Hadri has stationarity under the null. Null distributions of the Levin-Lin-Chu type and IPS statistics are simulated for the panel's own dimensions and lag orders, assuming independence across units.")]
    ur = [[names[i], Ts[i], lags[i], f(tt[i], 3), f(pv[i], 4), "stationary" if pv[i] < 0.05 else "unit root"] for i in range(N)]
    tabs.append(table("Unit-by-unit augmented Dickey-Fuller tests", ["Unit", "T", "Lags", "t-statistic", "p-value (MacKinnon)", "5% decision"], ur,
                      "Lags chosen by " + ic.upper() + f" up to {maxlag}. The decision uses the individual p-value only and ignores the multiplicity of {N} tests."))
    share = float((pv < 0.05).mean())
    ch = [chart("Individual ADF p-values by unit", names, [("p-value", pv)], "", "p-value", kind="bar")]
    short = [f"The Fisher-type tests use asymptotic MacKinnon p-values, which are unreliable when T is as short as {min(Ts)}; the IPS and Levin-Lin-Chu type tests use null distributions simulated for the panel's own length, so prefer them here."] if min(Ts) < 30 else []
    notes = short + ["The tests assume independence across units; common shocks make them over-reject. Cross-sectional demeaning removes a common time effect but not other forms of dependence.",
             "A rejection by IPS or Fisher means at least one unit is stationary, not that all are."]
    return result("punitroot", f"Panel unit-root tests: {y}", sum(Ts), [stat("Units", N), stat("Share of units rejecting at 5%", share, 3), stat("IPS W-statistic", Wt, 3), stat("IPS p-value", p_ips, 4)], tabs, ch, notes)


# ---------------------------------------------------------------------------------------------------------------
# Dumitrescu-Hurlin Granger non-causality test for heterogeneous panels
# ---------------------------------------------------------------------------------------------------------------
def _dh_unit(y, x, K):
    """Wald (F-form) statistic that the K lags of x do not help predict y in the unit's regression of y on K lags of y and of x."""
    T = len(y); n = T - K
    cols = [np.ones(n)] + [y[K - j: T - j] for j in range(1, K + 1)] + [x[K - j: T - j] for j in range(1, K + 1)]
    X = np.column_stack(cols); yt = y[K:]
    b = np.linalg.lstsq(X, yt, rcond=None)[0]; rss1 = float(((yt - X @ b) ** 2).sum())
    X0 = X[:, : 1 + K]; b0 = np.linalg.lstsq(X0, yt, rcond=None)[0]; rss0 = float(((yt - X0 @ b0) ** 2).sum())
    dfd = n - X.shape[1]
    if dfd < 1 or rss1 <= 0:
        return None
    F = ((rss0 - rss1) / K) / (rss1 / dfd)
    return dict(W=K * F, F=F, p=float(stats.f.sf(F, K, dfd)), n=n, dfd=dfd)


def _dh_stats(W, T, K):
    N = len(W); Wb = float(np.mean(W))
    Zbar = float(np.sqrt(N / (2 * K)) * (Wb - K))
    Zt = None
    if T - 3 * K - 5 > 0:
        Zt = float(np.sqrt(N / (2 * K) * (T - 3 * K - 5) / (T - 2 * K - 3)) * ((T - 3 * K - 3) / (T - 3 * K - 1) * Wb - K))
    return Wb, Zbar, Zt


def dh_causality(df, spec):
    y = need(spec, "y", "variable that may be caused"); x = need(spec, "x", "variable that may cause it"); distinct([y], [x])
    d, unit, time, N, T = _panel(df, spec, [y, x])
    K = int_option(spec, "lags", 1, 1, 6, "Lags"); dy = _choice(spec, "diff_y", ["no", "yes"], "no", "Difference the caused variable") == "yes"; dx = _choice(spec, "diff_x", ["no", "yes"], "no", "Difference the causing variable") == "yes"
    _balanced(d, N, T, "The Dumitrescu-Hurlin test")
    if N < 3:
        raise OESError("The test needs at least 3 units.")
    names = d.drop_duplicates("_u").sort_values("_u")[unit].astype(str).tolist()
    ser = {k: np.column_stack([g.sort_values("_ti")[k].to_numpy(float) for _, g in d.groupby("_u", sort=True)]) for k in (y, x)}
    ser = {y: np.diff(ser[y], axis=0) if dy else ser[y][1:] if dx else ser[y], x: np.diff(ser[x], axis=0) if dx else ser[x][1:] if dy else ser[x]}
    Te = ser[y].shape[0]
    if Te < 3 * K + 8:
        raise OESError(f"Each unit needs at least {3 * K + 8} periods with {K} lags (this panel has {Te} after any differencing).")
    tabs = []; summ = []; out = {}
    for lab, (a, b) in {f"{x} does not Granger-cause {y}": (y, x), f"{y} does not Granger-cause {x}": (x, y)}.items():
        res = [_dh_unit(ser[a][:, i], ser[b][:, i], K) for i in range(N)]
        if any(r is None for r in res):
            raise OESError("A unit has too few observations for the chosen number of lags.")
        W = np.array([r["W"] for r in res]); Wb, Zb, Zt = _dh_stats(W, Te, K)
        out[lab] = (res, Wb, Zb, Zt)
    rows = []
    for lab, (res, Wb, Zb, Zt) in out.items():
        rows.append([lab, f(Wb, 3), f(Zb, 3), f(2 * stats.norm.sf(abs(Zb)), 4), f(Zt, 3) if Zt is not None else "n/a", f(2 * stats.norm.sf(abs(Zt)), 4) if Zt is not None else "n/a"])
    tabs.append(table(f"Dumitrescu-Hurlin Granger non-causality test ({K} lag{'s' if K > 1 else ''}, {N} units, T = {Te})", ["Null hypothesis", "W-bar", "Z-bar", "p-value", "Z-tilde", "p-value"], rows,
                      "H0: x does not Granger-cause y for any unit. H1: it does for at least one unit (causality may differ across units). Z-bar is for large T, Z-tilde is the standardisation for small T and is the one to use when T is below about 50 or N is larger than T."))
    res = out[f"{x} does not Granger-cause {y}"][0]
    ur = [[names[i], f(res[i]["W"], 3), f(res[i]["F"], 3), f(res[i]["p"], 4), "rejects" if res[i]["p"] < 0.05 else "does not reject"] for i in range(N)]
    tabs.append(table(f"Unit-by-unit tests: {x} -> {y}", ["Unit", "Wald statistic", "F statistic", "p-value", "5% decision"], ur, "F test with " + f"{K} and the unit's residual degrees of freedom."))
    ch = [chart(f"Unit-by-unit p-values: does {x} Granger-cause {y}?", names, [("p-value", [r["p"] for r in res])], "", "p-value", kind="bar")]
    Wb, Zb, Zt = out[f"{x} does not Granger-cause {y}"][1:]
    notes = ["Both variables must be stationary: difference them first if they have unit roots.", "Granger causality is predictive content, not a causal effect: an omitted variable that moves both series produces it too.",
             "The test is for the null of no causality in every unit; a rejection does not say in which units causality holds."]
    return result("dh_causality", f"Panel Granger causality: {x} and {y}", int(N * Te), [stat("Units", N), stat("T per unit", Te), stat(f"W-bar ({x} to {y})", Wb, 3), stat("Z-tilde p-value", 2 * stats.norm.sf(abs(Zt)) if Zt is not None else None, 4)], tabs, ch, notes)


# ---------------------------------------------------------------------------------------------------------------
# Correlated random effects (Mundlak / Chamberlain): random effects with unit means, and the robust Mundlak test
# ---------------------------------------------------------------------------------------------------------------
def _cluster_ols(X, y, g):
    n, k = X.shape; XtXi = np.linalg.pinv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    G = len(np.unique(g)); S = np.zeros((k, k))
    for gi in np.unique(g):
        m = g == gi; s = X[m].T @ e[m]; S += np.outer(s, s)
    V = XtXi @ S @ XtXi * (G / (G - 1)) * ((n - 1) / (n - k))
    return b, V, e


def cre(df, spec):
    y = need(spec, "y", "dependent variable"); xs = as_list(need(spec, "x", "time-varying regressors")); zs = as_list(spec.get("z") or [])
    distinct([y], xs + zs)
    d, unit, time, N, T = _panel(df, spec, [y] + xs + zs)
    tw = _choice(spec, "time_effects", ["yes", "no"], "yes", "Period dummies")
    u = d["_u"].to_numpy(int)
    for v in xs:
        if d.groupby("_u")[v].std().fillna(0).max() <= 1e-10:
            raise OESError(f"{v} does not vary within units; list it among the time-invariant regressors.")
    for v in zs:
        if d.groupby("_u")[v].std().fillna(0).max() > 1e-10:
            raise OESError(f"{v} varies within units; list it among the time-varying regressors.")
    gm = lambda M: pd.DataFrame(np.asarray(M, float)).groupby(u).transform("mean").to_numpy()
    X = d[xs].to_numpy(float); Y = d[y].to_numpy(float); Xb = gm(X)
    dum = pd.get_dummies(d["_ti"], drop_first=True).to_numpy(float) if tw == "yes" and T > 1 else np.zeros((len(d), 0))
    dn = [f"period {int(v)}" for v in sorted(d["_ti"].unique())[1:]] if dum.shape[1] else []
    Z = d[zs].to_numpy(float) if zs else np.zeros((len(d), 0))
    one = np.ones((len(d), 1))
    if len(d) < 3 * (1 + len(xs) * 2 + len(zs)):
        raise OESError("Too few observations for this number of regressors.")
    # CRE: pooled OLS on x, unit means of x, z and period dummies, clustered by unit
    Wc = np.column_stack([one, X, Xb, Z, dum]); bc, Vc, _ = _cluster_ols(Wc, Y, u)
    nm = ["const"] + xs + [f"mean of {v}" for v in xs] + zs + dn
    k = len(xs); seC = np.sqrt(np.diag(Vc))
    # Mundlak test: joint significance of the unit means (cluster-robust Wald)
    bm = bc[1 + k: 1 + 2 * k]; Vm = Vc[1 + k: 1 + 2 * k, 1 + k: 1 + 2 * k]; Wm = float(bm @ np.linalg.solve(Vm, bm)); pm = float(stats.chi2.sf(Wm, k))
    # fixed effects (within) with period dummies, clustered
    Dm = np.column_stack([X, dum]); Dw = Dm - gm(Dm); Yw = Y - gm(Y[:, None])[:, 0]
    bf, Vf, ef = _cluster_ols(Dw, Yw, u); seF = np.sqrt(np.diag(Vf))
    # random effects (Swamy-Arora), clustered
    ew = Yw - Dw @ bf; se2 = float(ew @ ew) / (len(d) - N - Dm.shape[1])
    Ti = np.bincount(u).astype(float); Wr = np.column_stack([one, X, Z, dum])
    ob = pd.DataFrame(np.column_stack([Y, Wr])).groupby(u).mean().to_numpy(); bb = np.linalg.lstsq(ob[:, 1:], ob[:, 0], rcond=None)[0]; rb = ob[:, 0] - ob[:, 1:] @ bb
    su2 = max(float(rb @ rb) / max(N - ob.shape[1], 1) - se2 / float(np.mean(Ti)), 0.0)
    th = (1 - np.sqrt(se2 / (se2 + Ti * su2)))[u][:, None]
    Wt = Wr - th * gm(Wr); Yt = Y - th[:, 0] * gm(Y[:, None])[:, 0]; br, Vr, _ = _cluster_ols(Wt, Yt, u); seR = np.sqrt(np.diag(Vr))
    # classical Hausman-type contrast on the time-varying coefficients, cluster-robust via the Mundlak equivalence is the test above
    tab = []
    for i, nme in enumerate(nm):
        tab.append([nme, f(bc[i], 4), f(seC[i], 4), f(bc[i] / seC[i], 2), f(2 * stats.norm.sf(abs(bc[i] / seC[i])), 4)])
    cre_t = table(f"Correlated random effects (Mundlak) estimates, dependent variable: {y}", ["Variable", "Coefficient", "Cluster s.e.", "z", "p-value"], tab,
                  "The coefficients on the time-varying variables equal the fixed-effects estimates; the coefficient on the unit mean is the between effect minus the within effect. The coefficients on the time-invariant variables are identified from between-unit variation.")
    cmp_rows = []
    for i, v in enumerate(xs):
        cmp_rows.append([v, f(bf[i], 4), f(seF[i], 4), f(br[1 + i], 4), f(seR[1 + i], 4), f(bc[1 + i], 4), f(seC[1 + i], 4), f(bc[1 + k + i], 4), f(seC[1 + k + i], 4)])
    cmp_t = table("Time-varying coefficients: fixed effects, random effects, correlated random effects", ["Variable", "FE", "s.e.", "RE", "s.e.", "CRE (within)", "s.e.", "Mean of x (between - within)", "s.e."], cmp_rows,
                  "All standard errors are cluster-robust by unit. If the unit means have zero coefficients, random effects and fixed effects estimate the same thing.")
    tests = tests_table([test_row(f"Mundlak test: unit means of the time-varying regressors are jointly zero ({k} df)", Wm, pm, "H0: the unit effect is uncorrelated with the regressors, so random effects is consistent (cluster-robust Wald)")], title="Specification test")
    var_t = table("Variance components (random effects)", ["Component", "Estimate"], [["sigma_e (idiosyncratic)", f(np.sqrt(se2), 4)], ["sigma_u (unit effect)", f(np.sqrt(su2), 4)], ["rho", f(su2 / (su2 + se2 + 1e-300), 4)], ["theta (mean)", f(float(th.mean()), 4)]])
    ch = [chart("Within and random-effects coefficients", xs, [("Fixed effects", bf[:k]), ("Random effects", br[1: 1 + k])], "", "Coefficient", kind="bar")]
    notes = ["The Mundlak test is robust to heteroskedasticity and serial correlation, unlike the classical Hausman test.", "A rejection means the random-effects estimates of the time-varying coefficients are inconsistent; use the within (CRE) estimates for them."]
    return result("cre", f"Correlated random effects: {y}", len(d), [stat("Observations", len(d)), stat("Units", N), stat("Mundlak chi-squared", Wm, 3), stat("Mundlak p-value", pm, 4)], [cre_t, cmp_t, tests, var_t], ch, notes)


# ---------------------------------------------------------------------------------------------------------------
# Interactive fixed effects (Bai 2009): regression with unobserved common factors that load differently on each unit
# ---------------------------------------------------------------------------------------------------------------
def _ife_fit(Y, Xs, r, tol=1e-9, maxit=1000):
    T, N = Y.shape; k = len(Xs)
    XX = np.column_stack([x.reshape(-1) for x in Xs]); b = np.linalg.lstsq(XX, Y.reshape(-1), rcond=None)[0]; fit = np.zeros_like(Y); F = np.zeros((T, 0)); L = np.zeros((N, 0))
    for it in range(maxit):
        E = Y - sum(b[j] * Xs[j] for j in range(k))
        if r > 0:
            w, V = np.linalg.eigh(E @ E.T / (N * T)); F = np.sqrt(T) * V[:, ::-1][:, :r]; L = E.T @ F / T; fit = F @ L.T
        bn = np.linalg.lstsq(XX, (Y - fit).reshape(-1), rcond=None)[0]
        if np.max(np.abs(bn - b)) < tol:
            b = bn; break
        b = bn
    E = Y - sum(b[j] * Xs[j] for j in range(k)) - fit
    return b, E, F, L, it + 1


def _ife_var(Xs, E, F, L, r):
    T, N = E.shape; k = len(Xs)
    MF = np.eye(T) - (F @ np.linalg.solve(F.T @ F, F.T) if r else 0); ML = np.eye(N) - (L @ np.linalg.solve(L.T @ L, L.T) if r else 0)
    Z = [MF @ x @ ML for x in Xs]; D = np.array([[float((Z[a] * Z[c]).sum()) for c in range(k)] for a in range(k)]) / (N * T)
    s2 = float((E ** 2).sum()) / (N * T - r * (N + T) + r * r - k)
    return s2 * np.linalg.inv(D) / (N * T)


def _pesaran_cd(E):
    T, N = E.shape; C = np.corrcoef(E.T); iu = np.triu_indices(N, 1)
    return float(np.sqrt(2 * T / (N * (N - 1))) * C[iu].sum()), float(np.abs(C[iu]).mean())


def ife(df, spec):
    y = need(spec, "y", "dependent variable"); xs = as_list(need(spec, "x", "regressors")); distinct([y], xs)
    d, unit, time, N, T = _panel(df, spec, [y] + xs)
    _balanced(d, N, T, "The interactive fixed effects estimator")
    r = int_option(spec, "factors", 2, 0, 6, "Number of factors"); eff = _choice(spec, "effects", ["twoway", "unit", "none"], "twoway", "Additive effects")
    rmax = int_option(spec, "rmax", 5, 1, 8, "Maximum number of factors for the comparison")
    if min(N, T) <= r + len(xs) + 2 or min(N, T) < 10:
        raise OESError("The number of units and periods must both be at least 10 and larger than the number of factors plus regressors.")
    piv = lambda v: d.pivot(index="_ti", columns="_u", values=v).sort_index().to_numpy(float)
    Y = piv(y); Xs = [piv(v) for v in xs]
    def dm(A):
        if eff == "twoway": return A - A.mean(0, keepdims=True) - A.mean(1, keepdims=True) + A.mean()
        if eff == "unit": return A - A.mean(0, keepdims=True)
        return A
    Y = dm(Y); Xs = [dm(a) for a in Xs]
    names = d.drop_duplicates("_u").sort_values("_u")[unit].astype(str).tolist()
    res = {}
    for q in range(0, max(r, rmax) + 1):
        b, E, F, L, it = _ife_fit(Y, Xs, q); V = _ife_var(Xs, E, F, L, q); res[q] = (b, E, F, L, V, it)
    b, E, F, L, V, it = res[r]; se = np.sqrt(np.diag(V)); k = len(xs)
    dfv = N * T - r * (N + T) + r * r - k
    rows = [[xs[j], f(b[j], 4), f(se[j], 4), f(b[j] / se[j], 2), f(2 * stats.norm.sf(abs(b[j] / se[j])), 4)] for j in range(k)]
    tabs = [table(f"Interactive fixed effects estimates ({r} factor{'s' if r != 1 else ''}, {eff} additive effects), dependent variable: {y}", ["Variable", "Coefficient", "Std. err.", "z", "p-value"], rows,
                  "Standard errors are the homoskedastic asymptotic ones of Bai (2009) without bias correction; they are reliable when T and N are both large and the errors are close to independent. Check that the conclusions do not move with the number of factors.")]
    ic = []
    for q in range(0, max(r, rmax) + 1):
        s2 = float((res[q][1] ** 2).sum()) / (N * T); pen = q * (N + T) / (N * T) * np.log(N * T / (N + T)); ic.append((q, s2, np.log(s2) + pen))
    qbest = min(ic, key=lambda z: z[2])[0]
    cd0, ac0 = _pesaran_cd(res[0][1]); cdr, acr = _pesaran_cd(E)
    tabs.append(table("Choosing the number of factors (Bai-Ng IC_p1 on the residuals)", ["Factors", "Residual variance"] + [f"b({v})" for v in xs] + ["IC"],
                      [[q, f(s2, 5)] + [f(res[q][0][j], 4) for j in range(k)] + [f"{c:.4f}" + (" *" if q == qbest else "")] for q, s2, c in ic], "* minimises the criterion. With no factors the estimator is the fixed-effects (two-way) regression."))
    tabs.append(table("Cross-section dependence of the residuals (Pesaran CD)", ["Model", "CD statistic", "p-value", "Mean |pairwise correlation|"],
                      ([["No factors", f(cd0, 2), f(2 * stats.norm.sf(abs(cd0)), 4), f(ac0, 3)]] + ([[f"{r} factor{'s' if r != 1 else ''}", f(cdr, 2), f(2 * stats.norm.sf(abs(cdr)), 4), f(acr, 3)]] if r > 0 else [])),
                      "H0: errors are cross-sectionally independent. The CD statistic is a signed average and can stay large when the remaining correlations are of mixed sign, so look at the mean absolute correlation too."))
    ch = [chart("Estimated common factors", [str(t) for t in sorted(d["_ti"].unique())], [(f"Factor {j + 1}", F[:, j]) for j in range(min(r, 3))], "Period", "Factor", kind="line")] if r > 0 else []
    ch.append(chart("Coefficient by number of factors", list(range(0, max(r, rmax) + 1)), [(v, [res[q][0][j] for q in range(0, max(r, rmax) + 1)]) for j, v in enumerate(xs)], "Number of factors", "Coefficient", kind="line"))
    notes = ["The factors and loadings are identified only up to rotation; interpret the estimated factors as a space, not as individual series.", "Bai's (2009) bias terms for heteroskedastic or serially correlated errors are not corrected."]
    return result("ife", f"Interactive fixed effects: {y}", N * T, [stat("Units", N), stat("Periods", T), stat("Factors", r), stat("Iterations", it), stat("IC chooses", qbest)], tabs, ch, notes)
