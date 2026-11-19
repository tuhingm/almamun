"""Event study with Sun-Abraham and sensitivity analysis, synthetic difference-in-differences, matching / IPW / doubly robust estimation and
shift-share (Bartik) instrumental variables. Same result format as the other models (core.py)."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog, minimize

from .core import (test_row, tests_table, OESError, as_list, chart, check_columns, distinct, f, float_option, int_option, need, numeric, result, stat, table, time_order)


def _panel(df, spec, y, extra=()):
    unit, tcol = need(spec, "unit", "unit identifier"), need(spec, "time", "time identifier")
    check_columns(df, [unit, tcol])
    d = numeric(df, [y] + list(extra), keep=[unit, tcol]); d = d.rename(columns={unit: "_u", tcol: "_t"})
    d["_u"] = d["_u"].astype(str); d["_t"] = pd.to_numeric(d["_t"], errors="coerce")
    if d["_t"].isna().any():
        raise OESError("The time identifier must be numeric (a year or a period number).")
    if d.duplicated(["_u", "_t"]).any():
        raise OESError("Each unit and period may appear only once.")
    return d.sort_values(["_u", "_t"]).reset_index(drop=True), unit, tcol


# ------------------------------------------------------------------------------------------------------------------
def _fe_dummies(dd):
    return pd.concat([pd.get_dummies(dd._u).astype(float), pd.get_dummies(dd._t, drop_first=True).astype(float)], axis=1)


def _twfe_es(dd, ycol, lo, hi):
    lc = dd.l.clip(lo, hi); ev = [l for l in range(lo, hi + 1) if l != -1]
    X = pd.concat([_fe_dummies(dd)] + [pd.Series(((dd.g > 0) & (lc == l)).astype(float), name=l) for l in ev], axis=1)
    b = np.linalg.lstsq(X.values, dd[ycol].values, rcond=None)[0]
    return pd.Series(b[-len(ev):], index=ev)


def _sa_es(dd, ycol, cohorts, window):
    names, cols = [], [_fe_dummies(dd)]
    for g in cohorts:
        for l in range(int(dd._t.min() - g), int(dd._t.max() - g) + 1):
            if l == -1:
                continue
            cols.append(pd.Series(((dd.g == g) & (dd.l == l)).astype(float), name=(g, l))); names.append((g, l))
    X = pd.concat(cols, axis=1)
    b = pd.Series(np.linalg.lstsq(X.values, dd[ycol].values, rcond=None)[0][-len(names):], index=pd.MultiIndex.from_tuples(names))
    n = dd[dd.g > 0].groupby("g")._u.nunique(); out = {}
    for l in window:
        if l == -1:
            continue
        gs = [g for g in cohorts if (g, l) in b.index and n.get(g, 0) > 0]
        if not gs:
            continue
        wt = np.array([n[g] for g in gs], float); wt /= wt.sum(); out[l] = sum(w_ * b[(g, l)] for w_, g in zip(wt, gs))
    return pd.Series(out)


def eventstudy(df, spec):
    y = need(spec, "y", "outcome"); treat = need(spec, "treat", "treatment indicator (0/1)"); distinct([y], [treat])
    pre = int_option(spec, "pre", 5, 2, 12, "Pre-treatment periods"); post = int_option(spec, "post", 3, 1, 10, "Post-treatment periods")
    R = int_option(spec, "boot", 150, 50, 1000, "Bootstrap replications")
    d, unit, tcol = _panel(df, spec, y, [treat])
    first = d[d[treat] == 1].groupby("_u")._t.min()
    d["g"] = d._u.map(first).fillna(0).astype(int); d["l"] = np.where(d.g > 0, d._t - d.g, np.nan)
    cohorts = sorted(d.g[d.g > 0].unique())
    if not cohorts:
        raise OESError("No unit is ever treated.")
    if (d.g == 0).sum() == 0:
        raise OESError("A never-treated group is needed for the interaction-weighted estimator.")
    window = range(-pre, post + 1); units = sorted(d._u.unique()); size = d[d.g > 0].groupby("g")._u.nunique()
    b_tw = _twfe_es(d, y, -pre, post); b_sa = _sa_es(d, y, cohorts, window)
    rng = np.random.default_rng(int(spec.get("seed") or 0)); draws = []
    for _ in range(R):
        pick = rng.choice(np.array(units), len(units)); parts = []
        for k, s_ in enumerate(pick):
            x = d[d._u == s_].copy(); x["_u"] = f"b{k}"; parts.append(x)
        draws.append(_sa_es(pd.concat(parts, ignore_index=True), y, cohorts, window))
    draws = pd.DataFrame(draws).reindex(columns=b_sa.index); se = draws.std()
    rng_l = {int(g): (int(d._t.min() - g), int(d._t.max() - g)) for g in cohorts}
    ncoh = {l: int(sum(rng_l[int(g)][0] <= l <= rng_l[int(g)][1] for g in cohorts)) for l in b_sa.index}
    rows = [[int(l), f(b_tw.get(l)), f(b_sa[l]), f(se[l]), f(b_sa[l] / se[l], 3), f(2 * stats.norm.sf(abs(b_sa[l] / se[l])), 4), ncoh[l]] for l in b_sa.index]
    tabs = [table("Event-study coefficients (reference period: the year before adoption)", ["Event time", "Conventional TWFE", "Sun-Abraham", "Bootstrap s.e.", "t", "p-value", "Cohorts observed"], rows,
                  f"Interaction-weighted estimator with never-treated controls; bootstrap over units ({R} draws). End points of the conventional regression are binned.")]
    prel = [l for l in b_sa.index if l < -1]; postl = [l for l in b_sa.index if l >= 0]
    bp = b_sa[prel].values; Sp = np.atleast_2d(np.cov(draws[prel].values.T))
    wald = float(bp @ np.linalg.solve(Sp, bp)); pw = stats.chi2.sf(wald, len(prel))
    tau = np.array([l + 1 for l in prel], float); ncp = float(tau @ np.linalg.solve(Sp, tau)); crit = stats.chi2.ppf(.95, len(prel))
    slopes = np.linspace(0, 2 * max(np.abs(bp).max(), 1e-3), 600); power = 1 - stats.ncx2.cdf(crit, len(prel), slopes ** 2 * ncp)
    s80 = float(slopes[np.argmax(power >= .8)]) if (power >= .8).any() else float("nan"); slope_est = float((tau @ bp) / (tau @ tau))
    tabs.append(table("Pre-trend test and its power", ["Item", "Value"], [["Wald test that the pre-treatment coefficients are zero", f(wald, 3)], ["p-value", f(pw, 4)],
                                                                          ["Slope of a linear pre-trend detected with 80% power (per period)", f(s80, 4)], ["Least-squares slope of the estimated pre-coefficients", f(slope_est, 4)]]))
    # smoothness-bounded sensitivity (Rambachan and Roth 2023): second differences of the bias are at most M
    Ls = list(range(-pre, post + 1)); idx = {l: i for i, l in enumerate(Ls)}
    b_pre_full = np.array([b_sa[l] if l != -1 else 0.0 for l in Ls[:pre + 1]]); M_hat = float(np.abs(np.diff(b_pre_full, 2)).max()) if pre >= 2 else 0.0

    def bounds(k, M):
        n = len(Ls); c = np.zeros(n); c[idx[k]] = 1; box = [(None, None)] * n
        for l in Ls[:pre + 1]:
            box[idx[l]] = (0.0, 0.0) if l == -1 else (b_sa[l] - 1.96 * se[l], b_sa[l] + 1.96 * se[l])
        A, bb = [], []
        for l in Ls[1:-1]:
            row = np.zeros(n); row[idx[l + 1]] += 1; row[idx[l]] -= 2; row[idx[l - 1]] += 1; A += [row, -row]; bb += [M, M]
        r1 = linprog(c, A_ub=A, b_ub=bb, bounds=box); r2 = linprog(-c, A_ub=A, b_ub=bb, bounds=box)
        if r1.status != 0 or r2.status != 0:
            return np.nan, np.nan
        return r1.fun, -r2.fun
    srows = []
    for mult in (0, .5, 1, 2):
        for k in [l for l in postl if l <= 2]:
            blo, bhi = bounds(k, mult * M_hat); lo_set, hi_set = b_sa[k] - bhi, b_sa[k] - blo
            lo_ci, hi_ci = lo_set - 1.96 * se[k], hi_set + 1.96 * se[k]
            srows.append([mult, int(k), f(b_sa[k]), f(lo_set), f(hi_set), f(lo_ci), f(hi_ci), "yes" if (lo_ci > 0 or hi_ci < 0) else "no"])
    tabs.append(table("Sensitivity to violations of parallel trends (smoothness bound M = multiple of the largest pre-trend curvature)", ["M / M_hat", "Event time", "Estimate", "Set low", "Set high", "CI low", "CI high", "Excludes 0"], srows,
                      f"M_hat = {M_hat:.4f}. M = 0 allows only a linear post-treatment trend continuing the pre-trend; larger M allows more curvature."))
    ls = list(b_sa.index)
    ch = [chart("Event-study coefficients", ls, [("Sun-Abraham", b_sa.values), ("Conventional TWFE", b_tw.reindex(ls).values)], "Periods since adoption", "Effect", kind="line",
                band=(b_sa.values - 1.96 * se.values, b_sa.values + 1.96 * se.values))]
    summary = [stat("Observations", len(d)), stat("Units", len(units)), stat("Cohorts", len(cohorts)), stat("Pre-trend Wald p-value", pw, 4),
               stat("Average post effect", float(b_sa[postl].mean()), 4)]
    notes = ["Event study with the Sun and Abraham (2021) interaction-weighted estimator, cohort-by-event-time effects averaged with cohort-size weights, never-treated units as controls.",
             "Pre-trend power and smoothness-bounded sensitivity follow Roth (2022) and Rambachan and Roth (2023)."]
    return result("eventstudy", f"Event study: {y}", len(d), summary, tabs, ch, notes)


# ------------------------------------------------------------------------------------------------------------------
def _simplex(A, b, ridge, intercept):
    n = A.shape[1]

    def fun(x):
        r = A @ x[:n] + (x[n] if intercept else 0) - b
        return r @ r + ridge * (x[:n] @ x[:n])
    x0 = np.r_[np.ones(n) / n, 0.0] if intercept else np.ones(n) / n
    r = minimize(fun, x0, method="SLSQP", bounds=[(0, 1)] * n + ([(None, None)] if intercept else []),
                 constraints=[{"type": "eq", "fun": lambda x: x[:n].sum() - 1}], options={"maxiter": 1000, "ftol": 1e-12})
    return r.x[:n]


def _sdid(Yc, yt, T0, scale, zeta_mult=1.0, ntr=1):
    N0, T = Yc.shape; T1 = T - T0
    sig = np.std(np.diff(Yc[:, :T0], axis=1)); sig = sig if sig > 0 else 1.0
    zeta = zeta_mult * (ntr * T1) ** .25 * sig
    s_ = sig                                   # the data are divided by the noise level before the weights are fitted
    om = _simplex(Yc[:, :T0].T / s_, yt[:T0] / s_, (zeta / s_) ** 2 * T0, True)
    lam = _simplex(Yc[:, :T0] / s_, Yc[:, T0:].mean(1) / s_, (1e-6) ** 2 * N0, True)
    t_post, t_pre = yt[T0:].mean(), lam @ yt[:T0]; c_post, c_pre = om @ Yc[:, T0:].mean(1), om @ (Yc[:, :T0] @ lam)
    tau = (t_post - t_pre) - (c_post - c_pre); gap = (yt - t_pre) - (om @ Yc - c_pre)
    return tau, om, lam, gap


def _synth(Yc, yt, T0, scale):
    w = _simplex(Yc[:, :T0].T / scale, yt[:T0] / scale, 0.0, False); return (yt[T0:] - w @ Yc[:, T0:]).mean(), w


def sdid(df, spec):
    y = need(spec, "y", "outcome"); treat = need(spec, "treat", "treatment indicator (0/1)"); distinct([y], [treat])
    d, unit, tcol = _panel(df, spec, y, [treat])
    Y = d.pivot(index="_u", columns="_t", values=y); D = d.pivot(index="_u", columns="_t", values=treat)
    if Y.isna().any().any():
        raise OESError("Synthetic DiD needs a balanced panel: every unit observed in every period.")
    years = list(Y.columns); treated = [u for u in Y.index if D.loc[u].max() == 1]
    if not treated or len(treated) >= len(Y) - 2:
        raise OESError("Need at least one treated unit and at least three control units.")
    t_first = min(D.loc[treated[0]][D.loc[treated[0]] == 1].index); T0 = sum(t < t_first for t in years)
    if any(D.loc[u][D.loc[u] == 1].index.min() != t_first for u in treated):
        raise OESError("Synthetic DiD needs a common adoption period for all treated units (use the staggered DiD model otherwise).")
    if T0 < 3 or len(years) - T0 < 1:
        raise OESError("Need at least three pre-treatment periods and one post-treatment period.")
    ctrl = [u for u in Y.index if u not in treated]; Yc = Y.loc[ctrl].values.astype(float); yt = Y.loc[treated].values.astype(float).mean(0); N0 = len(ctrl); ntr = len(treated)
    scale = float(np.std(Y.values)) or 1.0
    did = (yt[T0:].mean() - yt[:T0].mean()) - (Yc[:, T0:].mean() - Yc[:, :T0].mean())
    sc_est, w_sc = _synth(Yc, yt, T0, scale); tau, om, lam, gap = _sdid(Yc, yt, T0, scale, 1.0, ntr)
    nrep = int_option(spec, "placebos", 100, 20, 500, "Placebo draws"); rng = np.random.default_rng(int(spec.get("seed") or 0))
    plac = []
    sets = [[k] for k in range(N0)] if ntr == 1 else [list(rng.choice(N0, ntr, replace=False)) for _ in range(nrep)]
    for S in sets[:max(nrep, N0 if ntr == 1 else nrep)]:
        keep = [i for i in range(N0) if i not in S]
        if len(keep) < 3:
            continue
        ytp = Yc[S].mean(0); Yp = Yc[keep]
        d_ = (ytp[T0:].mean() - ytp[:T0].mean()) - (Yp[:, T0:].mean() - Yp[:, :T0].mean())
        plac.append((d_, _synth(Yp, ytp, T0, scale)[0], _sdid(Yp, ytp, T0, scale, 1.0, ntr)[0]))
    plac = np.array(plac); est = np.array([did, sc_est, tau]); pse = plac.std(0, ddof=1)
    rows = [[n_, f(e), f(s_), f(e / s_, 3), f(float((np.abs(plac[:, j]) >= abs(e)).mean()), 3)] for j, (n_, e, s_) in enumerate(zip(["Difference-in-differences", "Synthetic control", "Synthetic DiD"], est, pse))]
    tabs = [table("Estimates and placebo inference", ["Estimator", "Effect", "Placebo s.e.", "t", "Share of placebos at least as large"], rows,
                  f"{len(plac)} placebo assignments of the treatment to control units; the placebo standard error is approximate and probably understated when the treated unit is large."),
            table("Largest unit weights", ["Unit", "SDID weight", "Synthetic-control weight"], [[ctrl[i], f(om[i], 4), f(w_sc[i], 4)] for i in np.argsort(-om)[:8]],
                  f"Effective number of units: {1 / np.sum(om ** 2):.1f} (SDID), {1 / np.sum(w_sc ** 2):.1f} (synthetic control)."),
            table("Time weights (pre-treatment periods)", ["Period", "Weight"], [[str(int(t)), f(w, 4)] for t, w in zip(years[:T0], lam)]),
            table("Gap by period (treated minus its synthetic path)", ["Period", "Gap"], [[str(int(t)), f(g_)] for t, g_ in zip(years, gap)], "Pre-treatment gaps are measured relative to the time-weighted pre-treatment level.")]
    rob = []
    for mult in (.25, 1, 4, 16):
        t_, o_, l_, _ = _sdid(Yc, yt, T0, scale, mult, ntr); rob.append([f"regularisation x {mult:g}", f(t_), f(1 / np.sum(o_ ** 2), 2), f(l_.max(), 3)])
    tabs.append(table("Sensitivity to the regularisation", ["Specification", "SDID estimate", "Effective units", "Largest time weight"], rob))
    cf = om @ Yc + (yt[:T0] @ lam - om @ (Yc[:, :T0] @ lam))
    ch = [chart("Treated against the synthetic DiD counterfactual", [str(int(t)) for t in years], [("Treated", yt), ("Counterfactual", cf)], "", y),
          chart("Gap by period", [str(int(t)) for t in years], [("Gap", gap)], "", "Effect", kind="bar")]
    summary = [stat("Units", len(Y)), stat("Treated units", ntr), stat("Pre-treatment periods", T0), stat("Synthetic DiD effect", tau, 4), stat("Placebo s.e.", pse[2], 4)]
    notes = ["Synthetic difference-in-differences (Arkhangelsky et al. 2021): unit weights and time weights on the simplex with an intercept, ridge penalty on the unit weights, weighted DiD.",
             "Time weights often concentrate on the last pre-treatment period; check the sensitivity and the pre-treatment gaps."]
    return result("sdid", f"Synthetic difference-in-differences: {y}", len(d), summary, tabs, ch, notes)


# ------------------------------------------------------------------------------------------------------------------
def _pscore(Xm, D):
    import statsmodels.api as sm
    m = sm.Logit(D, sm.add_constant(Xm)).fit(disp=0, maxiter=200); return m.predict(sm.add_constant(Xm)), m


def _match_estimators(y, D, Xm, trim=None):
    import statsmodels.api as sm
    e, _ = _pscore(Xm, D)
    keep = np.ones(len(y), bool) if trim is None else (e > trim) & (e < 1 - trim)
    y, D, Xm, e = y[keep], D[keep], Xm[keep], e[keep]; out = {}
    t = np.where(D == 1)[0]; c = np.where(D == 0)[0]
    idx = np.abs(e[t][:, None] - e[c][None, :]).argmin(1); out["Matching (1 nearest neighbour), ATT"] = (y[t] - y[c][idx]).mean()
    w1 = D / e; w0 = (1 - D) / (1 - e); out["IPW, ATE"] = (w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum()
    wa = np.where(D == 1, 1.0, e / (1 - e)); out["IPW, ATT"] = y[D == 1].mean() - np.average(y[D == 0], weights=wa[D == 0])
    Z = sm.add_constant(Xm); m1 = sm.OLS(y[D == 1], Z[D == 1]).fit(); m0 = sm.OLS(y[D == 0], Z[D == 0]).fit(); mu1, mu0 = m1.predict(Z), m0.predict(Z)
    out["Regression adjustment, ATE"] = (mu1 - mu0).mean(); out["Regression adjustment, ATT"] = (y[D == 1] - mu0[D == 1]).mean()
    psi = mu1 - mu0 + D * (y - mu1) / e - (1 - D) * (y - mu0) / (1 - e); out["AIPW (doubly robust), ATE"] = psi.mean()
    out["AIPW (doubly robust), ATT"] = (D * (y - mu0) - (1 - D) * e / (1 - e) * (y - mu0)).sum() / D.sum()
    return out


def _smd(x, D, w=None):
    w = np.ones(len(x)) if w is None else w
    m1 = np.average(x[D == 1], weights=w[D == 1]); m0 = np.average(x[D == 0], weights=w[D == 0])
    return (m1 - m0) / np.sqrt((x[D == 1].var(ddof=1) + x[D == 0].var(ddof=1)) / 2)


def matching(df, spec):
    import statsmodels.api as sm
    y = need(spec, "y", "outcome"); treat = need(spec, "treat", "treatment indicator (0/1)"); x = as_list(spec.get("x")); distinct([y], [treat], x)
    if not x:
        raise OESError("Choose at least one covariate.")
    d, labels = numeric(df, [y, treat] + x), None
    if not set(d[treat].unique()) <= {0, 1}:
        raise OESError("The treatment must be a 0/1 variable.")
    yv = d[y].to_numpy(float); D = d[treat].to_numpy(int); Xm = d[x].to_numpy(float)
    if D.sum() < 20 or (1 - D).sum() < 20:
        raise OESError("Need at least 20 treated and 20 untreated observations.")
    B = int_option(spec, "boot", 200, 50, 1000, "Bootstrap replications"); e, lg = _pscore(Xm, D)
    est = _match_estimators(yv, D, Xm); rng = np.random.default_rng(int(spec.get("seed") or 0)); boot = []
    for _ in range(B):
        ix = rng.integers(0, len(yv), len(yv))
        try:
            boot.append(_match_estimators(yv[ix], D[ix], Xm[ix]))
        except Exception:
            continue
    boot = pd.DataFrame(boot); se = boot.std(ddof=1)
    rows = [[k, f(v), f(se[k]), f(v - 1.96 * se[k]), f(v + 1.96 * se[k])] for k, v in est.items()]
    tabs = [table("Treatment-effect estimates", ["Estimator", "Estimate", "Bootstrap s.e.", "95% CI low", "95% CI high"], rows,
                  f"Raw difference in means: {yv[D == 1].mean() - yv[D == 0].mean():.4f}. The propensity score is re-estimated in every bootstrap draw ({len(boot)} draws)."),
            table("Propensity score", ["Statistic", "Treated", "Untreated"], [["Mean", f(e[D == 1].mean(), 4), f(e[D == 0].mean(), 4)], ["Minimum", f(e[D == 1].min(), 4), f(e[D == 0].min(), 4)], ["Maximum", f(e[D == 1].max(), 4), f(e[D == 0].max(), 4)]])]
    w_att = np.where(D == 1, 1.0, e / (1 - e))
    tabs.append(table("Covariate balance (standardised mean differences)", ["Covariate", "Raw", "After ATT weighting"], [[c, f(_smd(d[c].to_numpy(float), D), 3), f(_smd(d[c].to_numpy(float), D, w_att), 3)] for c in x],
                      f"|value| below 0.1 is the usual threshold. Effective number of untreated units after weighting: {w_att[D == 0].sum() ** 2 / (w_att[D == 0] ** 2).sum():.0f} of {(D == 0).sum()}; largest weight {w_att[D == 0].max():.2f}."))
    trows = []
    for tr in (None, .02, .05, .10):
        try:
            o = _match_estimators(yv, D, Xm, tr)
            trows.append(["no trimming" if tr is None else f"scores outside [{tr}, {1 - tr}] dropped", f(o["IPW, ATE"]), f(o["AIPW (doubly robust), ATE"]), f(o["Matching (1 nearest neighbour), ATT"])])
        except Exception:
            continue
    tabs.append(table("Sensitivity to trimming", ["Sample", "IPW ATE", "AIPW ATE", "Matching ATT"], trows))
    ols = sm.OLS(yv, sm.add_constant(np.column_stack([D, Xm]))).fit(); tstat = ols.tvalues[1]; fq = abs(tstat) / np.sqrt(ols.df_resid); rv = .5 * (np.sqrt(fq ** 4 + 4 * fq ** 2) - fq ** 2)
    tabs.append(table("Sensitivity to an omitted confounder (Cinelli and Hazlett 2020)", ["Item", "Value"], [["Regression estimate", f(ols.params[1])], ["t statistic", f(tstat, 2)], ["Robustness value", f(rv, 4)]],
                      "An omitted confounder would have to explain at least this share of the residual variance of both treatment and outcome to bring the estimate to zero."))
    ch = [chart("Propensity score by treatment status", np.round(np.linspace(0, 1, 21), 2), [("Untreated", np.histogram(e[D == 0], bins=np.linspace(0, 1, 22), density=True)[0]), ("Treated", np.histogram(e[D == 1], bins=np.linspace(0, 1, 22), density=True)[0])], "Propensity score", "Density", kind="bar")]
    summary = [stat("Observations", len(d)), stat("Treated", int(D.sum())), stat("AIPW ATE", est["AIPW (doubly robust), ATE"], 4), stat("AIPW ATT", est["AIPW (doubly robust), ATT"], 4), stat("Robustness value", rv, 4)]
    notes = ["Selection on observables: the estimates are valid if treatment is as good as random given the covariates (unconfoundedness) and treated and untreated units overlap.", "Balance is necessary, not sufficient."]
    return result("matching", f"Matching, IPW and doubly robust estimation: {y}", len(d), summary, tabs, ch, notes)


# ------------------------------------------------------------------------------------------------------------------
def _tsls(y, x, z, W=None):
    n = len(y); W = np.ones((n, 1)) if W is None else np.column_stack([np.ones(n), W])
    Z = np.column_stack([W, z]); X = np.column_stack([W, x]); xh = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    b = np.linalg.lstsq(xh, y, rcond=None)[0]; u = y - X @ b; A = np.linalg.inv(xh.T @ xh); V = A @ (xh.T * u ** 2) @ xh @ A
    return b[-1], np.sqrt(V[-1, -1])


def bartik(df, spec):
    import statsmodels.api as sm
    y = need(spec, "y", "outcome"); x = need(spec, "x", "endogenous variable"); shares = as_list(spec.get("shares")); controls = as_list(spec.get("controls")); distinct([y], [x], shares, controls)
    if len(shares) < 5:
        raise OESError("Choose at least five share columns (one per industry).")
    raw = spec.get("shocks"); 
    try:
        shocks = np.array([float(v) for v in (raw if isinstance(raw, list) else str(raw).replace(";", ",").split(","))])
    except Exception:
        raise OESError("Enter the industry shocks as a comma-separated list of numbers.")
    if len(shocks) != len(shares):
        raise OESError(f"{len(shares)} share columns need {len(shares)} shocks (found {len(shocks)}).")
    d = numeric(df, [y, x] + shares + controls); S = d[shares].to_numpy(float); yv = d[y].to_numpy(float); xv = d[x].to_numpy(float); W = d[controls].to_numpy(float) if controls else None
    B = S @ shocks; L, N = S.shape; sn = S.sum(0) / S.sum()
    Zc = np.column_stack([np.ones(L), B] + ([W] if W is not None else [])); fs = sm.OLS(xv, Zc).fit(cov_type="HC1")
    ols = sm.OLS(yv, np.column_stack([np.ones(L), xv] + ([W] if W is not None else []))).fit(cov_type="HC1")
    b_iv, se_iv = _tsls(yv, xv, B, W); rf = sm.OLS(yv, Zc).fit(cov_type="HC1")
    # Rotemberg weights (without controls partialled out when none are given)
    Zs = S - S.mean(0); xt = xv - xv.mean(); yt = yv - yv.mean()
    if W is not None:
        M = np.eye(L) - np.column_stack([np.ones(L), W]) @ np.linalg.pinv(np.column_stack([np.ones(L), W])); Zs = M @ S; xt = M @ xv; yt = M @ yv
    b_n = (Zs.T @ yt) / (Zs.T @ xt); alpha = shocks * (Zs.T @ xt); alpha = alpha / alpha.sum()
    order = np.argsort(-np.abs(alpha)); top = order[:10]
    # industry-level (Borusyak, Hull and Jaravel) standard error
    s_n = S.sum(0); y_n = (S * yv[:, None]).sum(0) / s_n; x_n = (S * xv[:, None]).sum(0) / s_n; Wn = np.sqrt(s_n)
    Zm = np.column_stack([np.ones(N), shocks]) * Wn[:, None]; Xm_ = np.column_stack([np.ones(N), x_n]) * Wn[:, None]; ym = y_n * Wn
    xh = Zm @ np.linalg.lstsq(Zm, Xm_, rcond=None)[0]; bb = np.linalg.lstsq(xh, ym, rcond=None)[0]; u = ym - Xm_ @ bb
    A = np.linalg.inv(xh.T @ xh); V = A @ (xh.T * u ** 2) @ xh @ A; se_bhj = float(np.sqrt(V[1, 1]))
    tabs = [table("Estimates", ["Estimator", "Coefficient", "Std. err."], [["OLS", f(ols.params[1]), f(ols.bse[1])], ["2SLS with the shift-share instrument (region-robust s.e.)", f(b_iv), f(se_iv)],
                                                                          ["Industry-level IV (shock-level robust s.e.)", f(bb[1]), f(se_bhj)], ["Reduced form", f(rf.params[1]), f(rf.bse[1])]],
                  "With controls, the industry-level coefficient is not identical to the region-level 2SLS."),
            table("First stage", ["Statistic", "Value"], [["Coefficient on the instrument", f(fs.params[1])], ["t statistic", f(fs.tvalues[1], 2)], ["F statistic", f(fs.tvalues[1] ** 2, 2)], ["R-squared", f(fs.rsquared, 3)]]),
            table("Rotemberg weights: industries with the largest absolute weights", ["Industry", "Weight", "Shock", "Industry-specific IV estimate"], [[shares[i], f(alpha[i], 4), f(shocks[i], 4), f(b_n[i], 4)] for i in top],
                  f"Share of absolute weight on the top 5: {np.abs(alpha[order[:5]]).sum() / np.abs(alpha).sum():.0%}; negative weights: {int((alpha < 0).sum())} industries ({alpha[alpha < 0].sum():.3f} of total). Effective number of industries: {1 / np.sum(sn ** 2):.1f}.")]
    rows = []
    for k in (0, 1, 3, 5):
        keep = [i for i in range(N) if i not in list(order[:k])]; Bk = S[:, keep] @ shocks[keep]; bk, sk = _tsls(yv, xv, Bk, W); rows.append([f"drop top {k} industries" if k else "all industries", f(bk), f(sk)])
    tabs.append(table("Leave-out sensitivity (drop the highest-weight industries from the instrument)", ["Instrument", "2SLS", "Std. err."], rows))
    ch = [chart("Rotemberg weights (top 10)", [shares[i] for i in top], [("Weight", alpha[top])], "", "Weight", kind="bar")]
    summary = [stat("Regions", L), stat("Industries", N), stat("First-stage F", fs.tvalues[1] ** 2, 1), stat("2SLS", b_iv, 4), stat("OLS", ols.params[1], 4), stat("Shock-level s.e.", se_bhj, 4)]
    notes = ["Shift-share instrument: initial industry shares times national industry shocks. Identification rests on share exogeneity (Goldsmith-Pinkham, Sorkin and Swift 2020) or shock exogeneity (Borusyak, Hull and Jaravel 2022); the diagnostics differ.",
             "Report which assumption you rely on and use the shock-level standard error."]
    return result("bartik", f"Shift-share IV: {y} on {x}", L, summary, tabs, ch, notes)


# ------------------------------------------------------------------------------------------------------------------
# Weak-instrument-robust inference: Anderson-Rubin, LM and conditional likelihood ratio confidence sets, LIML
# ------------------------------------------------------------------------------------------------------------------
def _partial(M, C):
    """Residuals of the columns of M after regressing on C (which includes the constant)."""
    return M - C @ np.linalg.lstsq(C, M, rcond=None)[0]


def _sqrtm_inv(A):
    w, V = np.linalg.eigh(A); return V @ np.diag(1 / np.sqrt(w)) @ V.T


def _wiv_parts(y, x, Z, b0, Om, ZZi_half, Pi_hat_num):
    """S and T vectors of Andrews, Moreira and Stock (2006) for the null beta = b0."""
    bv = np.array([1.0, -b0]); av = np.array([b0, 1.0]); Omi = np.linalg.inv(Om)
    S = Pi_hat_num @ bv / np.sqrt(bv @ Om @ bv); T = Pi_hat_num @ Omi @ av / np.sqrt(av @ Omi @ av)
    return S, T


def _clr_p(S, T, Sd):
    """Conditional likelihood ratio p-value of Moreira (2003): the null distribution of the statistic given T, simulated with common draws Sd of S ~ N(0, I)."""
    qT = float(T @ T); qS = float(S @ S); qST = float(S @ T)
    lr = lambda QS, QST: 0.5 * (QS - qT + np.sqrt(np.maximum((QS + qT) ** 2 - 4 * (QS * qT - QST ** 2), 0)))
    obs = lr(qS, qST); QS = (Sd ** 2).sum(1); QST = Sd @ T
    return float(np.mean(lr(QS, QST) >= obs))


def weakiv(df, spec):
    y = need(spec, "outcome", "outcome"); x = need(spec, "treat", "endogenous regressor"); zs = as_list(need(spec, "instruments", "instrument(s)")); cs = as_list(spec.get("controls") or [])
    distinct([y, x], zs, cs)
    miss = [c for c in [y, x] + zs + cs if c not in df.columns]
    if miss:
        raise OESError("Column(s) not found in the data: " + ", ".join(miss))
    d = numeric(df, [y, x] + zs + cs); n = len(d); k = len(zs)
    level = float(spec.get("level") or 0.95)
    if not 0.5 <= level < 1:
        raise OESError("The confidence level must be between 0.5 and 1.")
    reps = int_option(spec, "sims", 1000, 200, 20000, "Simulation replications"); seed = int_option(spec, "seed", 0, 0, 10 ** 6, "Random seed")
    if n < 3 * (k + len(cs) + 2):
        raise OESError("Too few observations for this number of instruments and controls.")
    one = np.ones((n, 1)); C = np.column_stack([one] + ([d[cs].to_numpy(float)] if cs else []))
    Y = _partial(d[[y, x]].to_numpy(float), C); Z = _partial(d[zs].to_numpy(float), C)
    if np.linalg.matrix_rank(Z) < k:
        raise OESError("The instruments are collinear (after partialling out the controls).")
    dof = n - C.shape[1] - k
    ZZ = Z.T @ Z; Pi = np.linalg.solve(ZZ, Z.T @ Y)                       # reduced form (k x 2)
    E = Y - Z @ Pi; Om = E.T @ E / dof
    ZZh = _sqrtm_inv(ZZ); Num = ZZh @ Z.T @ Y                          # (Z'Z)^{-1/2} Z'Y, k x 2, for the S and T vectors
    # 2SLS and LIML
    PZ = Z @ np.linalg.solve(ZZ, Z.T); xh = PZ @ Y[:, 1]
    b2 = float(xh @ Y[:, 0] / (xh @ Y[:, 1])); u2 = Y[:, 0] - b2 * Y[:, 1]; s2 = float(u2 @ u2) / (n - C.shape[1] - 1)
    se2 = np.sqrt(s2 / float(xh @ Y[:, 1]))
    Wm = np.column_stack([Y[:, 0], Y[:, 1]]); A = Wm.T @ Wm; B = Wm.T @ (Wm - PZ @ Wm)
    kappa = float(np.min(np.real(np.linalg.eigvals(np.linalg.solve(B, A)))))                                    # LIML k-class
    Mk = np.eye(n) - kappa * (np.eye(n) - PZ); bl = float(Y[:, 1] @ Mk @ Y[:, 0] / (Y[:, 1] @ Mk @ Y[:, 1])); ul = Y[:, 0] - bl * Y[:, 1]
    sel = float(ul @ ul) / (n - C.shape[1] - 1); sel_se = float(np.sqrt(sel / (Y[:, 1] @ Mk @ Y[:, 1])))
    # first stage
    fs = Y[:, 1] - Z @ Pi[:, 1]; Ffs = float((Pi[:, 1] @ ZZ @ Pi[:, 1]) / k / (float(fs @ fs) / dof)); pF = float(stats.f.sf(Ffs, k, dof))
    # grid inversion for AR, LM and CLR
    se0 = se2 if np.isfinite(se2) and se2 > 0 else 1.0
    half = max(40 * se0, 5 * abs(b2), 1.0); lo, hi = b2 - half, b2 + half; grid = np.linspace(lo, hi, 1201)
    rng = np.random.default_rng(seed); Sd = rng.standard_normal((reps, k)); ar = np.empty(len(grid)); lm = np.empty(len(grid)); clr = np.empty(len(grid))
    for i, b0 in enumerate(grid):
        S, T = _wiv_parts(None, None, Z, b0, Om, ZZh, Num)
        ar[i] = stats.chi2.sf(float(S @ S), k)
        lmv = float((S @ T) ** 2 / (T @ T)); lm[i] = stats.chi2.sf(lmv, 1)
        clr[i] = _clr_p(S, T, Sd) if k > 1 else lm[i]
    a = 1 - level
    def cset(p):
        ok = p > a
        if ok.all(): return "unbounded (the whole grid)"
        if not ok.any(): return "empty"
        idx = np.where(ok)[0]; segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        out = []
        for sg in segs:
            l = "-inf" if sg[0] == 0 else f"{grid[sg[0]]:.3f}"; h = "inf" if sg[-1] == len(grid) - 1 else f"{grid[sg[-1]]:.3f}"; out.append(f"[{l}, {h}]")
        return " U ".join(out)
    z = stats.norm.ppf(0.5 + level / 2)
    est = table(f"Effect of {x} on {y}: point estimates and weak-instrument-robust confidence sets", ["Method", "Estimate", "Std. err.", f"{int(round(level * 100))}% confidence set"],
                [["2SLS (Wald interval)", f(b2, 4), f(se2, 4), f"[{b2 - z * se2:.3f}, {b2 + z * se2:.3f}]"], ["LIML (Wald interval)", f(bl, 4), f(sel_se, 4), f"[{bl - z * sel_se:.3f}, {bl + z * sel_se:.3f}]"],
                 ["Anderson-Rubin", "", "", cset(ar)], ["Lagrange multiplier (Kleibergen)", "", "", cset(lm)], ["Conditional likelihood ratio (Moreira)" if k > 1 else "Conditional likelihood ratio (= LM with one instrument)", "", "", cset(clr)]],
                "Homoskedastic errors. The AR, LM and CLR sets are valid whatever the strength of the instruments; the Wald intervals are not. The grid is wide (at least 40 standard errors either side of 2SLS); a set that reaches its edge is reported with an infinite end.")
    fst = table("First stage", ["Statistic", "Value"], [["First-stage F (joint significance of the instruments)", f(Ffs, 2)], ["p-value", f(pF, 4)], ["Number of instruments", k], ["Partial R-squared", f(float((Pi[:, 1] @ ZZ @ Pi[:, 1]) / float(Y[:, 1] @ Y[:, 1])), 4)],
                                                      ["Rule of thumb", "F < 10: weak (Staiger-Stock); AR, LM or CLR sets should be reported"]])
    S0 = _wiv_parts(None, None, Z, 0.0, Om, ZZh, Num)[0]; AR0 = float(S0 @ S0)
    rows_t = [test_row("Anderson-Rubin test of beta = 0", AR0, float(stats.chi2.sf(AR0, k)), f"chi-squared({k}); valid with weak instruments")]
    if k > 1:
        from scipy.optimize import minimize_scalar
        mn = minimize_scalar(lambda b0: float((lambda S: S @ S)(_wiv_parts(None, None, Z, b0, Om, ZZh, Num)[0])), bounds=(grid[0], grid[-1]), method="bounded", options=dict(xatol=1e-9))
        arm = min(float(mn.fun), float(np.min([stats.chi2.isf(v, k) if v > 0 else np.inf for v in ar])))
        rows_t.append(test_row(f"Overidentification (minimum of the AR statistic, {k - 1} df)", arm, float(stats.chi2.sf(arm, k - 1)), "H0: all instruments are valid; a rejection makes the AR set narrow or empty"))
    tests = tests_table(rows_t, title="Tests")
    ch = [chart("p-value of the test of each value of the coefficient", list(grid), [("Anderson-Rubin", ar), ("LM", lm)] + ([("CLR", clr)] if k > 1 else []), f"Value of the coefficient on {x}", "p-value", kind="line")]
    notes = ["The sets are for homoskedastic errors; with heteroskedasticity use a robust version (for example in Stata, weakiv or ivreg2 with robust options).",
             "With more instruments than endogenous regressors, the AR test also has power against invalid instruments, so a rejection does not by itself show an effect."]
    return result("weakiv", f"Weak-instrument-robust inference: effect of {x} on {y}", n, [stat("Observations", n), stat("2SLS estimate", b2, 4), stat("LIML estimate", bl, 4), stat("First-stage F", Ffs, 2), stat("Instruments", k)], [est, fst, tests], ch, notes)


# ------------------------------------------------------------------------------------------------------------------
# Randomised experiments: balance, Neyman and regression-adjusted effects, Lin (2013) estimator, randomisation inference, power
# ------------------------------------------------------------------------------------------------------------------
def _hc2(X, y):
    n, k = X.shape; XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtXi, X); h = np.minimum(h, 1 - 1e-8)
    V = XtXi @ (X.T * (e ** 2 / (1 - h))) @ X @ XtXi
    return b, V, e


def experiment(df, spec):
    y = need(spec, "outcome", "outcome"); t = need(spec, "treat", "treatment indicator"); xs = as_list(spec.get("covariates") or []); distinct([y, t], xs)
    miss = [c for c in [y, t] + xs if c not in df.columns]
    if miss:
        raise OESError("Column(s) not found in the data: " + ", ".join(miss))
    d = numeric(df, [y, t] + xs); n = len(d)
    if not set(np.unique(d[t])) <= {0.0, 1.0} or d[t].nunique() != 2:
        raise OESError("The treatment must be a 0/1 indicator with both values present.")
    draws = int_option(spec, "draws", 2000, 200, 50000, "Randomisation draws"); seed = int_option(spec, "seed", 0, 0, 10 ** 6, "Random seed")
    T = d[t].to_numpy(float); Y = d[y].to_numpy(float); n1 = int(T.sum()); n0 = n - n1
    if min(n1, n0) < 5:
        raise OESError("Each arm needs at least 5 units.")
    # Neyman difference in means
    ate = float(Y[T == 1].mean() - Y[T == 0].mean()); se_n = float(np.sqrt(Y[T == 1].var(ddof=1) / n1 + Y[T == 0].var(ddof=1) / n0))
    # regression adjustment (additive) and Lin (2013) interacted estimator, HC2 standard errors
    rows = [["Difference in means (Neyman)", f(ate, 4), f(se_n, 4), f(ate / se_n, 2), f(2 * stats.norm.sf(abs(ate / se_n)), 4)]]
    est = {"diff": (ate, se_n)}
    if xs:
        Xc = d[xs].to_numpy(float); Xc = Xc - Xc.mean(0)
        A = np.column_stack([np.ones(n), T, Xc]); b, V, _ = _hc2(A, Y); se = float(np.sqrt(V[1, 1])); est["adj"] = (float(b[1]), se)
        rows.append(["Regression adjustment (covariates, HC2)", f(b[1], 4), f(se, 4), f(b[1] / se, 2), f(2 * stats.norm.sf(abs(b[1] / se)), 4)])
        L = np.column_stack([np.ones(n), T, Xc, Xc * T[:, None]]); bl, Vl, _ = _hc2(L, Y); sel = float(np.sqrt(Vl[1, 1])); est["lin"] = (float(bl[1]), sel)
        rows.append(["Lin (2013) interacted adjustment (HC2)", f(bl[1], 4), f(sel, 4), f(bl[1] / sel, 2), f(2 * stats.norm.sf(abs(bl[1] / sel)), 4)])
    key = "lin" if xs else "diff"
    # randomisation inference under the sharp null of no effect for any unit (complete randomisation)
    rng = np.random.default_rng(seed)
    if xs:
        stat_fn = lambda tt: _lin_t(Y, tt, Xc)
        tobs = _lin_t(Y, T, Xc)
    else:
        stat_fn = lambda tt: float(Y[tt == 1].mean() - Y[tt == 0].mean()); tobs = ate
    sims = np.empty(draws)
    for i in range(draws):
        sims[i] = stat_fn(rng.permutation(T))
    p_ri = float((np.sum(np.abs(sims) >= abs(tobs) - 1e-12) + 1) / (draws + 1))
    lab = "Lin-adjusted effect" if xs else "difference in means"
    tabs = [table(f"Average treatment effect of {t} on {y}", ["Estimator", "Estimate", "Std. err.", "t", "p-value"], rows,
                  f"{n1} treated and {n0} control units. Neyman variance for the difference in means; HC2 (leverage-corrected) for the regression estimators, which are consistent for the average effect even if the effects differ across units.")]
    tabs.append(tests_table([test_row(f"Randomisation inference: sharp null of no effect ({draws} draws, {lab})", tobs, p_ri, "p-value of the two-sided test against the permutation distribution")], title="Randomisation test"))
    # balance
    if xs:
        brows = []
        for v in xs:
            x = d[v].to_numpy(float); m1, m0 = x[T == 1].mean(), x[T == 0].mean(); sd = np.sqrt((x[T == 1].var(ddof=1) + x[T == 0].var(ddof=1)) / 2)
            tt = (m1 - m0) / np.sqrt(x[T == 1].var(ddof=1) / n1 + x[T == 0].var(ddof=1) / n0)
            brows.append([v, f(m1, 3), f(m0, 3), f((m1 - m0) / sd if sd > 0 else 0.0, 3), f(2 * stats.norm.sf(abs(tt)), 4)])
        Xa = np.column_stack([np.ones(n), d[xs].to_numpy(float)]); bt = np.linalg.lstsq(Xa, T, rcond=None)[0]; e = T - Xa @ bt
        k = len(xs); rss = float(e @ e); tss = float(((T - T.mean()) ** 2).sum()); Fj = ((tss - rss) / k) / (rss / (n - k - 1)); pF = float(stats.f.sf(Fj, k, n - k - 1))
        tabs.append(table("Covariate balance", ["Covariate", "Mean, treated", "Mean, control", "Standardised difference", "p-value"], brows, "A joint F test of the covariates predicting treatment: F = " + f"{Fj:.2f}, p = {pF:.3f}. Standardised differences above 0.1 deserve a look, though some imbalance is expected by chance."))
    # power
    sdp = float(np.sqrt(((n1 - 1) * Y[T == 1].var(ddof=1) + (n0 - 1) * Y[T == 0].var(ddof=1)) / (n - 2))); se_eq = sdp * np.sqrt(1 / n1 + 1 / n0)
    mde = float((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * se_eq)
    tabs.append(table("Precision and power", ["Quantity", "Value"], [["Control-group mean of the outcome", f(float(Y[T == 0].mean()), 4)], ["Pooled standard deviation of the outcome", f(sdp, 4)], ["Minimum detectable effect (80% power, 5% test)", f(mde, 4)], ["MDE as a share of the control mean", f(mde / abs(Y[T == 0].mean()), 3) if Y[T == 0].mean() != 0 else "n/a"]],
                      "Computed for the difference in means with equal variances; covariate adjustment lowers the standard error and the MDE when the covariates predict the outcome."))
    ch = [chart("Randomisation distribution of the test statistic", list(range(len(sims))), [("Permutation draws (sorted)", np.sort(sims)), ("Observed", [tobs] * len(sims))], "Draw", "Statistic", kind="line")]
    notes = ["The randomisation test assumes complete randomisation of the observed number of treated units; with stratified or clustered designs, permute within strata or clusters.", "Covariates must be measured before treatment; adjusting for post-treatment variables can bias the effect."]
    return result("experiment", f"Randomised experiment: effect of {t} on {y}", n, [stat("Observations", n), stat("Treated", n1), stat("Effect", est[key][0], 4), stat("Std. err.", est[key][1], 4), stat("Randomisation p-value", p_ri, 4)], tabs, ch, notes)


def _lin_t(Y, T, Xc):
    n = len(Y); L = np.column_stack([np.ones(n), T, Xc, Xc * T[:, None]])
    return float(np.linalg.lstsq(L, Y, rcond=None)[0][1])


# ------------------------------------------------------------------------------------------------------------------
# Causal mediation analysis (linear models, Imai-Keele-Tingley) with bootstrap intervals and sensitivity to sequential ignorability
# ------------------------------------------------------------------------------------------------------------------
def _med_fit(Y, T, M, X, interact):
    n = len(Y); one = np.ones(n)
    Am = np.column_stack([one, T, X]); gm = np.linalg.lstsq(Am, M, rcond=None)[0]; e1 = M - Am @ gm
    Ay = np.column_stack([one, T, M] + ([T * M] if interact else []) + [X]); gy = np.linalg.lstsq(Ay, Y, rcond=None)[0]; e2 = Y - Ay @ gy
    a = float(gm[1]); c = float(gy[1]); b = float(gy[2]); kap = float(gy[3]) if interact else 0.0
    Mbar = {t: float(gm[0] + a * t + (X.mean(0) @ gm[2:] if X.shape[1] else 0.0)) for t in (0.0, 1.0)}
    acme = {t: a * (b + kap * t) for t in (0.0, 1.0)}; ade = {t: c + kap * Mbar[t] for t in (0.0, 1.0)}
    return dict(a=a, b=b, c=c, kap=kap, acme=acme, ade=ade, s1=float(np.sqrt(e1 @ e1 / (n - Am.shape[1]))), s2=float(np.sqrt(e2 @ e2 / (n - Ay.shape[1]))))


def mediation(df, spec):
    y = need(spec, "outcome", "outcome"); t = need(spec, "treat", "treatment (0/1)"); m = need(spec, "mediator", "mediator"); xs = as_list(spec.get("covariates") or []); distinct([y, t, m], xs)
    miss = [c for c in [y, t, m] + xs if c not in df.columns]
    if miss:
        raise OESError("Column(s) not found in the data: " + ", ".join(miss))
    d = numeric(df, [y, t, m] + xs); n = len(d)
    if not set(np.unique(d[t])) <= {0.0, 1.0} or d[t].nunique() != 2:
        raise OESError("The treatment must be a 0/1 indicator with both values present.")
    interact = str(spec.get("interaction") or "no").lower() == "yes"; B = int_option(spec, "boot", 500, 99, 5000, "Bootstrap replications"); seed = int_option(spec, "seed", 0, 0, 10 ** 6, "Random seed")
    if n < 5 * (len(xs) + 4):
        raise OESError("Too few observations for this number of covariates.")
    Y, T, M = d[y].to_numpy(float), d[t].to_numpy(float), d[m].to_numpy(float); X = d[xs].to_numpy(float) if xs else np.zeros((n, 0))
    if d[m].nunique() < 2:
        raise OESError("The mediator does not vary.")
    fit = _med_fit(Y, T, M, X, interact)
    rng = np.random.default_rng(seed); draws = []
    for _ in range(B):
        i = rng.integers(0, n, n); fb = _med_fit(Y[i], T[i], M[i], X[i], interact)
        draws.append([fb["acme"][0.0], fb["acme"][1.0], fb["ade"][0.0], fb["ade"][1.0], fb["a"], fb["b"]])
    D = np.array(draws)
    def ci(v): return np.percentile(v, [2.5, 97.5])
    acme_avg = 0.5 * (fit["acme"][0.0] + fit["acme"][1.0]); ade_avg = 0.5 * (fit["ade"][0.0] + fit["ade"][1.0]); tot = acme_avg + ade_avg
    ca = 0.5 * (D[:, 0] + D[:, 1]); cd = 0.5 * (D[:, 2] + D[:, 3]); ct = ca + cd; prop = ca / ct
    def prow(lab, est, v): lo, hi = ci(v); return [lab, f(est, 4), f(float(np.std(v, ddof=1)), 4), f(lo, 4), f(hi, 4), f(float(max(2 * min(np.mean(v <= 0), np.mean(v >= 0)), 1.0 / (B + 1))), 4)]
    rows = []
    if interact:
        rows += [prow("ACME, control (t = 0)", fit["acme"][0.0], D[:, 0]), prow("ACME, treated (t = 1)", fit["acme"][1.0], D[:, 1]), prow("ADE, control (t = 0)", fit["ade"][0.0], D[:, 2]), prow("ADE, treated (t = 1)", fit["ade"][1.0], D[:, 3])]
    rows += [prow("ACME (average mediation effect)", acme_avg, ca), prow("ADE (average direct effect)", ade_avg, cd), prow("Total effect", tot, ct)]
    pm = float(np.median(prop)) if tot != 0 else float("nan"); plo, phi = np.percentile(prop, [2.5, 97.5])
    rows.append(["Proportion mediated (ACME / total)", f(acme_avg / tot if tot != 0 else float("nan"), 4), f(float(np.std(prop, ddof=1)), 4), f(plo, 4), f(phi, 4), ""])
    tabs = [table(f"Mediation of the effect of {t} on {y} through {m}", ["Effect", "Estimate", "Boot. s.e.", "95% CI low", "95% CI high", "Boot. p-value"], rows,
                  f"Linear models for the mediator and the outcome{' with a treatment-by-mediator interaction' if interact else ''}; percentile intervals from {B} bootstrap resamples. The proportion mediated is unstable when the total effect is near zero.")]
    ab = table("Path coefficients", ["Path", "Coefficient"], [[f"{t} -> {m} (a)", f(fit["a"], 4)], [f"{m} -> {y}, holding treatment fixed (b)", f(fit["b"], 4)], [f"{t} -> {y}, holding the mediator fixed (c')", f(fit["c"], 4)]] + ([["Treatment-by-mediator interaction", f(fit["kap"], 4)]] if interact else []))
    tabs.append(ab)
    # sensitivity to a violation of sequential ignorability: correlation rho between the errors of the mediator and outcome models
    grid = np.round(np.linspace(-0.9, 0.9, 19), 2); b0 = fit["b"]; s1, s2 = fit["s1"], fit["s2"]; a = fit["a"]
    bro = lambda r: b0 - r * s2 / (s1 * np.sqrt(1 - r ** 2))
    acm = np.array([a * bro(r) for r in grid]); rstar = (b0 * s1 / s2) / np.sqrt(1 + (b0 * s1 / s2) ** 2)
    srows = [[f(r, 2), f(float(a * bro(r)), 4)] for r in grid]
    tabs.append(table("Sensitivity of the ACME to correlation between the mediator and outcome errors", ["rho", "ACME at that rho"], srows,
                      f"At rho = 0 the estimate relies on sequential ignorability (no unmeasured confounder of the mediator-outcome relationship, given treatment and covariates). The ACME is zero at rho = {rstar:.3f}. The no-interaction ACME formula is used here."))
    ch = [chart("ACME as a function of the error correlation rho", list(grid), [("ACME", acm), ("Zero", [0.0] * len(grid))], "rho", "ACME", kind="line")]
    notes = ["The treatment must be randomised or ignorable given the covariates, and the mediator must be ignorable given treatment and covariates (sequential ignorability); the second assumption cannot be ensured by randomising the treatment.", "A mediator measured after treatment and affected by it can confound the mediator-outcome relationship through variables that are also affected by treatment; check the sensitivity table."]
    return result("mediation", f"Causal mediation: {t} -> {m} -> {y}", n, [stat("Observations", n), stat("ACME", acme_avg, 4), stat("ADE", ade_avg, 4), stat("Total effect", tot, 4), stat("Proportion mediated", acme_avg / tot if tot != 0 else None, 3), stat("rho at which ACME = 0", float(rstar), 3)], tabs, ch, notes)


# ------------------------------------------------------------------------------------------------------------------
# Lee (2009) trimming bounds for sample selection and attrition, with Manski worst-case bounds for comparison
# ------------------------------------------------------------------------------------------------------------------
def _lee(Y, T, S):
    """Lee bounds on the effect for the always-selected. Y may be NaN where S = 0."""
    q1 = S[T == 1].mean(); q0 = S[T == 0].mean()
    y1 = Y[(T == 1) & (S == 1)]; y0 = Y[(T == 0) & (S == 1)]
    if q1 >= q0:
        p = (q1 - q0) / q1 if q1 > 0 else 0.0; arm = 1; ys = np.sort(y1)
    else:
        p = (q0 - q1) / q0 if q0 > 0 else 0.0; arm = 0; ys = np.sort(y0)
    k = int(np.floor(p * len(ys) + 1e-9)); frac = p * len(ys) - k
    # trim exactly p of the selected in the arm with the higher selection rate (fractional observation at the edge)
    def trimmed_mean(v, from_top):
        v = np.sort(v)
        if p <= 0: return float(v.mean())
        w = np.ones(len(v));
        idx = np.arange(len(v) - 1, -1, -1) if from_top else np.arange(len(v))
        rem = p * len(v)
        for j in idx:
            take = min(1.0, rem); w[j] -= take; rem -= take
            if rem <= 1e-12: break
        return float(np.sum(w * v) / np.sum(w))
    if arm == 1:
        lo = trimmed_mean(y1, True) - y0.mean(); hi = trimmed_mean(y1, False) - y0.mean()
    else:
        lo = y1.mean() - trimmed_mean(y0, False); hi = y1.mean() - trimmed_mean(y0, True)
    return lo, hi, p, arm, float(y1.mean() - y0.mean()), q1, q0


def leebounds(df, spec):
    y = need(spec, "outcome", "outcome"); t = need(spec, "treat", "treatment (0/1)"); s = need(spec, "selected", "selection indicator (1 if the outcome is observed)"); distinct([y, t, s])
    miss = [c for c in [y, t, s] if c not in df.columns]
    if miss:
        raise OESError("Column(s) not found in the data: " + ", ".join(miss))
    d = df[[y, t, s]].apply(pd.to_numeric, errors="coerce"); d = d.dropna(subset=[t, s]).reset_index(drop=True); n = len(d)
    if not set(np.unique(d[t])) <= {0.0, 1.0} or d[t].nunique() != 2:
        raise OESError("The treatment must be a 0/1 indicator with both values present.")
    if not set(np.unique(d[s])) <= {0.0, 1.0} or d[s].nunique() != 2:
        raise OESError("The selection indicator must be 0/1 with both values present.")
    if d.loc[d[s] == 1, y].isna().any():
        raise OESError("The outcome is missing for some units marked as selected.")
    B = int_option(spec, "boot", 500, 99, 5000, "Bootstrap replications"); seed = int_option(spec, "seed", 0, 0, 10 ** 6, "Random seed")
    T = d[t].to_numpy(float); S = d[s].to_numpy(float); Y = d[y].to_numpy(float)
    if min(((T == 1) & (S == 1)).sum(), ((T == 0) & (S == 1)).sum()) < 10:
        raise OESError("Each arm needs at least 10 units with an observed outcome.")
    lo, hi, p, arm, naive, q1, q0 = _lee(Y, T, S)
    rng = np.random.default_rng(seed); bl, bh = [], []
    for _ in range(B):
        i = rng.integers(0, n, n); Tb, Sb, Yb = T[i], S[i], Y[i]
        if min(((Tb == 1) & (Sb == 1)).sum(), ((Tb == 0) & (Sb == 1)).sum()) < 3: continue
        l_, h_, *_ = _lee(Yb, Tb, Sb); bl.append(l_); bh.append(h_)
    bl, bh = np.array(bl), np.array(bh)
    ci_lo, ci_hi = float(np.percentile(bl, 2.5)), float(np.percentile(bh, 97.5))
    # Manski worst-case bounds for the effect on the outcome of all units, filling missing outcomes with the extremes
    ymin = float(spec["ymin"]) if spec.get("ymin") not in (None, "") else float(np.nanmin(Y[S == 1])); ymax = float(spec["ymax"]) if spec.get("ymax") not in (None, "") else float(np.nanmax(Y[S == 1]))
    def arm_mean(tt, fill): v = np.where(S[T == tt] == 1, Y[T == tt], fill); return float(np.mean(v))
    m_lo = arm_mean(1, ymin) - arm_mean(0, ymax); m_hi = arm_mean(1, ymax) - arm_mean(0, ymin)
    sel_tab = table("Selection (outcome observed) by treatment arm", ["Arm", "Units", "Share with an observed outcome", "Mean outcome among the selected"],
                    [["Control", int((T == 0).sum()), f(q0, 4), f(float(Y[(T == 0) & (S == 1)].mean()), 4)], ["Treated", int((T == 1).sum()), f(q1, 4), f(float(Y[(T == 1) & (S == 1)].mean()), 4)]],
                    f"The treatment changes the selection rate by {100 * (q1 - q0):.1f} percentage points. Lee bounds assume that treatment moves selection in one direction only (monotonicity).")
    res = table(f"Effect of {t} on {y} for units that would be selected under either arm", ["Estimate", "Value", "95% CI low", "95% CI high"],
                [["Naive difference among the selected (biased)", f(naive, 4), "", ""], ["Lee lower bound", f(lo, 4), f(ci_lo, 4), ""], ["Lee upper bound", f(hi, 4), "", f(ci_hi, 4)],
                 ["Lee bounds: confidence interval for the effect", f"[{lo:.3f}, {hi:.3f}]", f(ci_lo, 4), f(ci_hi, 4)], [f"Share trimmed ({'treated' if arm == 1 else 'control'} arm)", f(p, 4), "", ""],
                 ["Manski worst-case bounds (all units, outcome in [{:.2f}, {:.2f}])".format(ymin, ymax), f"[{m_lo:.3f}, {m_hi:.3f}]", "", ""]],
                "Bootstrap percentile intervals with " + f"{len(bl)} resamples. The confidence interval for the effect takes the lower end of the lower bound and the upper end of the upper bound.")
    ch = [chart("Bootstrap distributions of the Lee bounds", list(range(len(bl))), [("Lower bound (sorted)", np.sort(bl)), ("Upper bound (sorted)", np.sort(bh))], "Resample", "Effect", kind="line")]
    notes = ["The bounds are for the always-selected: units that would have an observed outcome whether treated or not. They are not the effect for the whole sample.", "Monotonicity (treatment cannot push some units into selection and others out) is an assumption; covariates can tighten the bounds."]
    return result("leebounds", f"Lee bounds: effect of {t} on {y}", n, [stat("Observations", n), stat("Share trimmed", p, 4), stat("Lee lower bound", lo, 4), stat("Lee upper bound", hi, 4), stat("Naive effect", naive, 4)], [sel_tab, res], ch, notes)
