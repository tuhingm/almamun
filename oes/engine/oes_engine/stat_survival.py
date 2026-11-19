"""Category D of the Statistics Division: survival analysis.

Kaplan-Meier with log-rank type tests, the Cox proportional hazards model (Efron or Breslow ties, Schoenfeld-residual test of
proportionality, Harrell's concordance), accelerated failure time models (Weibull, log-normal, log-logistic, exponential), the
Fine-Gray subdistribution hazard model for competing risks, and a comparison of parametric survival distributions. Everything is
implemented here with numpy and scipy; no survival package is needed, so it runs in the browser."""
import numpy as np
import pandas as pd
from scipy import optimize, stats

from .core import OESError, as_list, chart, distinct, f, float_option, need, result, stat, table, tests_table
from .stat_core import clean_frame, dummy_cols, levels


# ---------------------------------------------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------------------------------------------
def _alpha(spec):
    a = float_option(spec, "alpha", 0.05, "Significance level")
    if not 0 < a < 0.5:
        raise OESError("The significance level must be between 0 and 0.5.")
    return a


def _surv_data(df, spec, covariates=(), factors=(), group=None, status_codes=False):
    """Clean frame with the time and event columns validated. Returns (frame, dropped)."""
    tcol = need(spec, "time", "follow-up time")
    ecol = need(spec, "event", "event indicator")
    covs, facs = as_list(covariates), as_list(factors)
    grp = [group] if group else []
    distinct(tcol, ecol, covs, facs, grp)
    d, dropped = clean_frame(df, numeric=[tcol, ecol] + covs, factors=facs + grp)
    if (d[tcol] < 0).any():
        raise OESError("Follow-up times must not be negative.")
    ev = d[ecol]
    if status_codes:
        if not np.allclose(ev, np.round(ev)) or (ev < 0).any():
            raise OESError("The status column must hold whole-number codes: 0 = censored, 1, 2, ... = event types.")
    elif not ev.isin([0, 1]).all():
        raise OESError("The event indicator must be coded 0 = censored, 1 = event.")
    return d, dropped


def _design(d, covs, facs, intercept=False):
    cols, names = [], []
    if intercept:
        cols.append(np.ones(len(d)))
        names.append("(Intercept)")
    for c in covs:
        v = d[c].to_numpy(float)
        if np.ptp(v) == 0:
            raise OESError(f"The covariate {c} is constant.")
        cols.append(v)
        names.append(c)
    for c in facs:
        X, lv = dummy_cols(d[c])
        if len(lv) < 2:
            raise OESError(f"The factor {c} has only one level.")
        for j, l in enumerate(lv[1:]):
            cols.append(X[:, j])
            names.append(f"{c}[{l}] vs {lv[0]}")
    if not cols:
        return np.zeros((len(d), 0)), []
    return np.column_stack(cols), names


def _ci_label(alpha):
    return f"{int(round((1 - alpha) * 100))}% CI"


# ---------------------------------------------------------------------------------------------------------------
# Kaplan-Meier
# ---------------------------------------------------------------------------------------------------------------
def km_curve(t, e, ci="loglog", alpha=0.05):
    """Kaplan-Meier estimate with Greenwood variance. Returns the event-time table and arrays."""
    t, e = np.asarray(t, float), np.asarray(e, int)
    ut = np.unique(t[e == 1])
    n_at = np.array([np.sum(t >= u) for u in ut], float)
    d = np.array([np.sum((t == u) & (e == 1)) for u in ut], float)
    c = np.array([np.sum((t == u) & (e == 0)) for u in ut], float)
    S = np.cumprod(1 - d / n_at)
    with np.errstate(divide="ignore", invalid="ignore"):
        gw = np.cumsum(np.where(n_at > d, d / (n_at * (n_at - d)), np.inf))
    se = S * np.sqrt(gw)
    z = stats.norm.ppf(1 - alpha / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        if ci == "loglog":
            se_ll = np.sqrt(gw) / np.abs(np.log(S))
            lo = S ** np.exp(z * se_ll)
            hi = S ** np.exp(-z * se_ll)
        elif ci == "log":
            lo = S * np.exp(-z * np.sqrt(gw))
            hi = np.minimum(1.0, S * np.exp(z * np.sqrt(gw)))
        else:
            lo = np.maximum(0.0, S - z * se)
            hi = np.minimum(1.0, S + z * se)
    lo = np.where(S >= 1, 1.0, lo)
    hi = np.where(S >= 1, 1.0, hi)
    lo = np.where(S <= 0, 0.0, lo)
    hi = np.where(S <= 0, 0.0, hi)
    return dict(t=ut, n=n_at, d=d, c=c, S=S, se=se, lo=lo, hi=hi, gw=gw)


def _first_below(times, curve, level):
    idx = np.where(curve <= level + 1e-12)[0]
    return float(times[idx[0]]) if len(idx) else float("nan")


def _step_eval(times, vals, grid, before=1.0):
    """Value of a right-continuous step function at grid points."""
    pos = np.searchsorted(times, grid, side="right") - 1
    out = np.where(pos >= 0, vals[np.clip(pos, 0, None)], before)
    return out


def _rmst(km, tau):
    ts = np.concatenate([[0.0], km["t"][km["t"] < tau], [tau]])
    Ss = np.concatenate([[1.0], km["S"][km["t"] < tau]])
    return float(np.sum(Ss * np.diff(ts)))


def _rmst_se(km, tau):
    """Greenwood-type standard error of the restricted mean."""
    m = km["t"] < tau
    ts, S, n, d = km["t"][m], km["S"][m], km["n"][m], km["d"][m]
    if len(ts) == 0:
        return 0.0
    grid = np.concatenate([ts, [tau]])
    area_after = np.zeros(len(ts))
    # integral of S from t_j to tau
    for j in range(len(ts)):
        seg_t = np.concatenate([ts[j:], [tau]])
        seg_S = S[j:]
        area_after[j] = np.sum(seg_S * np.diff(seg_t))
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(n > d, area_after ** 2 * d / (n * (n - d)), 0.0)
    return float(np.sqrt(np.sum(terms)))


def logrank_family(t, e, g, weight="logrank"):
    """K-sample weighted log-rank test (Fleming-Harrington family). Returns chi-square, df, p, observed, expected."""
    t, e = np.asarray(t, float), np.asarray(e, int)
    lv = list(dict.fromkeys(g))
    K = len(lv)
    gi = np.array([lv.index(v) for v in g])
    ut = np.unique(t[e == 1])
    O = np.zeros(K)
    E = np.zeros(K)
    V = np.zeros((K, K))
    # pooled left-continuous Peto-Peto survival estimate prod(1 - d/(n+1)) for the Peto weight
    S_prev = 1.0
    for u in ut:
        at = t >= u
        n = float(at.sum())
        d = float(np.sum((t == u) & (e == 1)))
        nk = np.array([np.sum(at & (gi == k)) for k in range(K)], float)
        dk = np.array([np.sum((t == u) & (e == 1) & (gi == k)) for k in range(K)], float)
        S_prev *= 1 - d / (n + 1) if weight == 'peto' else 1.0
        w = 1.0 if weight == "logrank" else n if weight == "gehan" else S_prev
        O += w * dk
        E += w * d * nk / n
        if n > 1:
            fac = d * (n - d) / (n - 1) / n
            V += w * w * fac * (np.diag(nk) - np.outer(nk, nk) / n)
    keep = np.arange(K - 1)
    diff = (O - E)[keep]
    Vk = V[np.ix_(keep, keep)]
    try:
        chi = float(diff @ np.linalg.solve(Vk, diff))
    except np.linalg.LinAlgError:
        raise OESError("The log-rank variance matrix is singular; a group may have no events or no one at risk.")
    return chi, K - 1, float(stats.chi2.sf(chi, K - 1)), O, E, lv


def kaplan_meier(df, spec):
    alpha = _alpha(spec)
    gcol = spec.get("group") or None
    ci = spec.get("ci_type") or "loglog"
    weight = spec.get("test") or "logrank"
    d, dropped = _surv_data(df, spec, group=gcol)
    tcol, ecol = spec["time"], spec["event"]
    t, e = d[tcol].to_numpy(float), d[ecol].to_numpy(int)
    if e.sum() == 0:
        raise OESError("There are no events in the data, so the survival curve cannot be estimated.")
    groups = [("All subjects", np.ones(len(d), bool))]
    if gcol:
        lv = levels(d[gcol])
        if len(lv) < 2:
            raise OESError("The grouping variable has only one level.")
        if len(lv) > 12:
            raise OESError("The grouping variable has more than 12 levels; collapse it first.")
        groups = [(f"{gcol} = {v}", (d[gcol] == v).to_numpy()) for v in lv]
    lab = _ci_label(alpha)
    rows_sum, tabs, curves = [], [], []
    for name, m in groups:
        if e[m].sum() == 0:
            km = dict(t=np.array([]), S=np.array([]), lo=np.array([]), hi=np.array([]), n=np.array([]), d=np.array([]), c=np.array([]), se=np.array([]))
            med = lo_m = hi_m = float("nan")
            tau = float(t[m].max())
            rm = tau
            rm_se = 0.0
        else:
            km = km_curve(t[m], e[m], ci, alpha)
            med = _first_below(km["t"], km["S"], 0.5)
            lo_m = _first_below(km["t"], km["lo"], 0.5)
            hi_m = _first_below(km["t"], km["hi"], 0.5)
            tau = float(t[m].max())
            rm = _rmst(km, tau)
            rm_se = _rmst_se(km, tau)
        curves.append((name, km))
        rows_sum.append([name, int(m.sum()), int(e[m].sum()), int(m.sum() - e[m].sum()), f(med, 4), f(lo_m, 4), f(hi_m, 4), f(rm, 4), f(rm_se, 4)])
    tabs.append(table("Summary by group", ["Group", "n", "Events", "Censored", "Median survival", f"Median {lab} low", "Median CI high", "Restricted mean", "SE"], rows_sum,
                      "The median is the first time the curve reaches 0.5 or below; the interval is the range of times at which the confidence band for S(t) covers 0.5. "
                      "The restricted mean is the area under the curve up to the largest observed time in the group, with a Greenwood-type standard error. 'None' means the curve never reaches 0.5."))
    summary = [stat("Subjects", len(d)), stat("Events", int(e.sum()))]
    notes = []
    if gcol:
        chi, df_, p, O, E, lv2 = logrank_family(t, e, d[gcol].tolist(), weight)
        wname = {"logrank": "Log-rank", "gehan": "Gehan-Breslow (Wilcoxon)", "peto": "Peto-Peto"}[weight]
        tabs.append(table(f"{wname} test of equal survival curves", ["Group", "Observed events", "Expected events", "(O-E)^2 / E"],
                          [[f"{gcol} = {v}", f(O[i], 3), f(E[i], 3), f((O[i] - E[i]) ** 2 / E[i], 4) if E[i] > 0 else None] for i, v in enumerate(lv2)] +
                          [["Chi-square", f(chi, 4), f"df = {df_}", f(p, 6)]],
                          "The last row holds the chi-square statistic, its degrees of freedom and the p-value."))
        summary += [stat(f"{wname} chi-square", chi), stat("p-value", p, 6)]
        notes.append("The log-rank test is most powerful when hazards are proportional; the Gehan-Breslow and Peto-Peto weights emphasise early differences.")
    # life table at chosen times
    all_t = np.sort(np.unique(t[e == 1]))
    qs = np.unique(np.round(np.quantile(t, [0.25, 0.5, 0.75]), 6))
    rows = []
    for name, km in curves:
        for q in qs:
            if len(km["t"]) == 0:
                continue
            s = float(_step_eval(km["t"], km["S"], np.array([q]))[0])
            lo = float(_step_eval(km["t"], km["lo"], np.array([q]))[0])
            hi = float(_step_eval(km["t"], km["hi"], np.array([q]))[0])
            m_ = dict(groups)[name]
            rows.append([name, f(q, 4), int(np.sum(t[m_] >= q)), f(s, 5), f(lo, 5), f(hi, 5)])
    tabs.append(table("Survival probability at the quartiles of follow-up time", ["Group", "Time", "At risk", "S(t)", f"{lab} low", "CI high"], rows,
                      f"Confidence intervals use the {'log-log' if ci == 'loglog' else 'log' if ci == 'log' else 'plain'} transformation of the Greenwood variance."))
    # the full table for the first group (all subjects) or each group, truncated
    big = curves[0][1]
    full_rows = []
    for name, km in curves:
        for i in range(min(len(km["t"]), 60)):
            full_rows.append([name, f(km["t"][i], 4), int(km["n"][i]), int(km["d"][i]), int(km["c"][i]), f(km["S"][i], 5), f(km["se"][i], 5), f(km["lo"][i], 5), f(km["hi"][i], 5)])
    tabs.append(table("Kaplan-Meier life table (event times)", ["Group", "Time", "At risk", "Events", "Censored before next", "S(t)", "SE (Greenwood)", "CI low", "CI high"], full_rows,
                      "Only the first 60 event times of each group are listed; the chart shows them all." if any(len(k["t"]) > 60 for _, k in curves) else None))
    grid = np.concatenate([[0.0], all_t])
    if len(grid) > 300:
        grid = grid[np.unique(np.linspace(0, len(grid) - 1, 300).astype(int))]
    series = [(name, _step_eval(km["t"], km["S"], grid) if len(km["t"]) else np.ones(len(grid))) for name, km in curves]
    charts = [chart("Kaplan-Meier survival curves", grid, series, "time", "survival probability", "line")]
    if not gcol and len(curves[0][1]["t"]):
        k0 = curves[0][1]
        charts[0] = chart("Kaplan-Meier survival curve with confidence band", grid, [("S(t)", series[0][1])], "time", "survival probability", "band",
                          band=(_step_eval(k0["t"], k0["lo"], grid), _step_eval(k0["t"], k0["hi"], grid)))
    if dropped:
        notes.append(f"{dropped} row(s) with missing values were dropped.")
    return result("stat_km", "Kaplan-Meier survival analysis", len(d), summary, tabs, charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# Cox proportional hazards
# ---------------------------------------------------------------------------------------------------------------
def _cox_terms(time, event, X, beta, ties):
    """Log partial likelihood, gradient and Hessian (Efron or Breslow ties)."""
    n, p = X.shape
    order = np.argsort(-time, kind="stable")
    t, ev, Xs = time[order], event[order], X[order]
    eta = Xs @ beta
    eta -= eta.max()
    r = np.exp(eta)
    c0 = np.cumsum(r)
    c1 = np.cumsum(r[:, None] * Xs, axis=0)
    c2 = np.cumsum(r[:, None, None] * Xs[:, :, None] * Xs[:, None, :], axis=0)
    ll = 0.0
    g = np.zeros(p)
    H = np.zeros((p, p))
    # group boundaries in the descending order
    bounds = np.flatnonzero(np.r_[True, t[1:] != t[:-1]])
    ends = np.r_[bounds[1:], n]
    for a, b in zip(bounds, ends):
        idx = np.arange(a, b)
        dead = idx[ev[idx] == 1]
        d = len(dead)
        if d == 0:
            continue
        S0, S1, S2 = c0[b - 1], c1[b - 1], c2[b - 1]
        s0d, s1d, s2d = r[dead].sum(), (r[dead, None] * Xs[dead]).sum(0), np.einsum("i,ij,ik->jk", r[dead], Xs[dead], Xs[dead])
        ll += eta[dead].sum()
        g += Xs[dead].sum(0)
        for l in range(d):
            cfrac = l / d if ties == "efron" else 0.0
            den = S0 - cfrac * s0d
            n1 = (S1 - cfrac * s1d) / den
            n2 = (S2 - cfrac * s2d) / den
            ll -= np.log(den)
            g -= n1
            H -= n2 - np.outer(n1, n1)
    return ll, g, H


def _cox_fit(time, event, X, ties, max_iter=60):
    n, p = X.shape
    beta = np.zeros(p)
    ll, g, H = _cox_terms(time, event, X, beta, ties)
    ll0 = ll
    g0, H0 = g.copy(), H.copy()
    for it in range(max_iter):
        try:
            step = np.linalg.solve(-H, g)
        except np.linalg.LinAlgError:
            raise OESError("The covariates are perfectly collinear, so the Cox model cannot be estimated.")
        s = 1.0
        while True:
            nb = beta + s * step
            nll, ng, nH = _cox_terms(time, event, X, nb, ties)
            if np.isfinite(nll) and nll >= ll - 1e-12:
                break
            s /= 2
            if s < 1e-8:
                break
        if np.max(np.abs(nb)) > 50:
            raise OESError("The partial likelihood has no finite maximum (a covariate perfectly separates the events); estimates diverge.")
        done = abs(nll - ll) < 1e-10 * (1 + abs(nll)) and np.max(np.abs(s * step)) < 1e-8
        beta, ll, g, H = nb, nll, ng, nH
        if done:
            break
    else:
        raise OESError("The Newton-Raphson iterations did not converge.")
    cov = np.linalg.inv(-H)
    return dict(beta=beta, ll=ll, ll0=ll0, g0=g0, H0=H0, cov=cov, H=H, iters=it + 1)


def harrell_c(time, event, score):
    """Harrell's concordance: score is a risk score (higher = shorter expected survival)."""
    time, event, score = np.asarray(time, float), np.asarray(event, int), np.asarray(score, float)
    conc = ties = comp = 0.0
    for i in np.flatnonzero(event == 1):
        m = time > time[i]
        comp += m.sum()
        conc += np.sum(score[m] < score[i])
        ties += np.sum(score[m] == score[i])
    if comp == 0:
        return float("nan"), 0
    return float((conc + 0.5 * ties) / comp), int(comp)


def _schoenfeld(time, event, X, beta, ties):
    """Schoenfeld residuals at the death times (Efron expectations for ties)."""
    eta = X @ beta
    eta -= eta.max()
    r = np.exp(eta)
    ut = np.unique(time[event == 1])
    rows, tt = [], []
    for u in ut:
        risk = time >= u
        dead = (time == u) & (event == 1)
        d = int(dead.sum())
        S0, S1 = r[risk].sum(), (r[risk, None] * X[risk]).sum(0)
        s0d, s1d = r[dead].sum(), (r[dead, None] * X[dead]).sum(0)
        ex = np.zeros(X.shape[1])
        for l in range(d):
            cf = l / d if ties == "efron" else 0.0
            ex += (S1 - cf * s1d) / (S0 - cf * s0d)
        ex /= d
        for i in np.flatnonzero(dead):
            rows.append(X[i] - ex)
            tt.append(u)
    return np.array(rows), np.array(tt)


def ph_test(time, event, X, beta, cov, ties, transform="rank"):
    """Grambsch-Therneau test of proportional hazards for each covariate and globally."""
    R, tt = _schoenfeld(time, event, X, beta, ties)
    D = len(tt)
    g = stats.rankdata(tt) if transform == "rank" else tt if transform == "identity" else np.log(tt) if transform == "log" else stats.rankdata(tt)
    gc = g - g.mean()
    u = gc @ R
    uV = u @ cov
    sxx = float(np.sum(gc ** 2))
    per = D * uV ** 2 / (np.diag(cov) * sxx)
    glob = float(D * (u @ cov @ u) / sxx)
    return per, glob, R, tt


def cox_ph(df, spec):
    alpha = _alpha(spec)
    ties = spec.get("ties") or "efron"
    covs, facs = as_list(spec.get("covariates")), as_list(spec.get("factors"))
    if not covs and not facs:
        raise OESError("Choose at least one covariate.")
    d, dropped = _surv_data(df, spec, covs, facs)
    X, names = _design(d, covs, facs)
    t, e = d[spec["time"]].to_numpy(float), d[spec["event"]].to_numpy(int)
    n, p = X.shape
    nev = int(e.sum())
    if nev < 2:
        raise OESError("At least two events are needed.")
    if nev < 5 * p:
        pass
    fit = _cox_fit(t, e, X, ties)
    beta, cov = fit["beta"], fit["cov"]
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pv = 2 * stats.norm.sf(np.abs(z))
    zc = stats.norm.ppf(1 - alpha / 2)
    lab = _ci_label(alpha)
    coef_rows = [[names[j], f(beta[j], 5), f(np.exp(beta[j]), 5), f(se[j], 5), f(z[j], 4), f(pv[j], 6), f(np.exp(beta[j] - zc * se[j]), 5), f(np.exp(beta[j] + zc * se[j]), 5)] for j in range(p)]
    lr = 2 * (fit["ll"] - fit["ll0"])
    wald = float(beta @ np.linalg.solve(cov, beta))
    score = float(fit["g0"] @ np.linalg.solve(-fit["H0"], fit["g0"]))
    mt = [["Likelihood ratio", f(lr, 4), p, f(stats.chi2.sf(lr, p), 6)], ["Wald", f(wald, 4), p, f(stats.chi2.sf(wald, p), 6)], ["Score (log-rank)", f(score, 4), p, f(stats.chi2.sf(score, p), 6)]]
    risk = X @ beta
    C, npairs = harrell_c(t, e, risk)
    per, glob, R, tt = ph_test(t, e, X, beta, cov, ties)
    ph_rows = [[names[j], f(per[j], 4), 1, f(stats.chi2.sf(per[j], 1), 4)] for j in range(p)]
    ph_rows.append(["GLOBAL", f(glob, 4), p, f(stats.chi2.sf(glob, p), 4)])
    # baseline cumulative hazard (Breslow) at the covariate means
    xbar = X.mean(0)
    eta = X @ beta
    shift = eta.max()
    r = np.exp(eta - shift)
    ut = np.unique(t[e == 1])
    dj = np.array([np.sum((t == u) & (e == 1)) for u in ut], float)
    S0 = np.array([r[t >= u].sum() for u in ut])
    H0 = np.cumsum(dj / S0) * np.exp(-shift)
    Sbar = np.exp(-H0 * np.exp(xbar @ beta))
    grid = np.concatenate([[0.0], ut])
    if len(grid) > 300:
        grid = grid[np.unique(np.linspace(0, len(grid) - 1, 300).astype(int))]
    charts = [chart("Estimated survival curve at the mean of the covariates", grid, [("S(t | mean covariates)", _step_eval(ut, Sbar, grid))], "time", "survival probability", "line")]
    # hazard-ratio forest as a bar chart of log hazard ratios
    charts.append(chart("Log hazard ratios", names, [("log HR", beta)], "", "log hazard ratio", "bar"))
    tabs = [table("Cox proportional hazards coefficients", ["Covariate", "coef (log HR)", "HR = exp(coef)", "SE", "z", "p-value", f"HR {lab} low", "HR CI high"], coef_rows,
                  f"Partial likelihood with {'Efron' if ties == 'efron' else 'Breslow'} handling of tied event times; standard errors from the inverse observed information."),
            table("Overall model tests", ["Test", "Chi-square", "df", "p-value"], mt),
            table("Test of proportional hazards (Schoenfeld residuals, rank time)", ["Covariate", "Chi-square", "df", "p-value"], ph_rows,
                  "Grambsch-Therneau score test of a time trend in the scaled Schoenfeld residuals. A small p-value signals that the hazard ratio changes over time.")]
    summary = [stat("Subjects", n), stat("Events", nev), stat("Concordance (Harrell C)", C), stat("Likelihood-ratio chi-square", lr), stat("LR p-value", float(stats.chi2.sf(lr, p)), 6)]
    notes = []
    if nev < 10 * p:
        notes.append(f"Only {nev} events for {p} covariate coefficient(s); the usual guideline is about ten events per coefficient, so estimates may be unstable.")
    if np.any(per > 0) and glob and stats.chi2.sf(glob, p) < alpha:
        notes.append("The global proportional-hazards test rejects at the chosen level; consider stratifying, adding time interactions or using an AFT model.")
    if dropped:
        notes.append(f"{dropped} row(s) with missing values were dropped.")
    return result("stat_cox", "Cox proportional hazards regression", n, summary, tabs, charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# Accelerated failure time and parametric survival
# ---------------------------------------------------------------------------------------------------------------
_DISTS = ["weibull", "lognormal", "loglogistic", "exponential"]
_DNAME = {"weibull": "Weibull", "lognormal": "Log-normal", "loglogistic": "Log-logistic", "exponential": "Exponential"}


def _eps(dist, z):
    """log f, d log f / dz, log S, d log S / dz of the standardised error distribution."""
    if dist in ("weibull", "exponential"):
        ez = np.exp(np.minimum(z, 700))
        return z - ez, 1 - ez, -ez, -ez
    if dist == "lognormal":
        lf = -0.5 * z * z - 0.5 * np.log(2 * np.pi)
        ls = stats.norm.logsf(z)
        return lf, -z, ls, -np.exp(lf - ls)
    sg = 1 / (1 + np.exp(-z))
    sp = np.logaddexp(0, z)
    return z - 2 * sp, 1 - 2 * sg, -sp, -sg


def _aft_ll(theta, lt, ev, X, dist, grad=False):
    p = X.shape[1]
    beta = theta[:p]
    ls = 0.0 if dist == "exponential" else theta[p]
    sig = np.exp(ls)
    z = (lt - X @ beta) / sig
    lf, a, lS, b = _eps(dist, z)
    ll = np.sum(np.where(ev == 1, lf - ls - lt, lS))
    if not grad:
        return ll
    w = np.where(ev == 1, a, b)
    gb = (-1 / sig) * (X.T @ w)
    if dist == "exponential":
        return ll, gb
    gs = -np.sum(np.where(ev == 1, 1 + z * a, z * b))
    return ll, np.r_[gb, gs]


def _aft_fit(t, ev, X, dist):
    lt = np.log(t)
    n, p = X.shape
    if p:
        b0 = np.linalg.lstsq(X, lt, rcond=None)[0]
        res = lt - X @ b0
    else:
        b0, res = np.zeros(0), lt - lt.mean()
    k = p + (0 if dist == "exponential" else 1)
    s0 = np.log(max(res.std(), 0.1))
    th0 = np.r_[b0, [] if dist == "exponential" else [s0]]
    if dist == "weibull":
        pass

    def negll(th):
        v = _aft_ll(th, lt, ev, X, dist)
        return -v if np.isfinite(v) else 1e30

    def negg(th):
        return -_aft_ll(th, lt, ev, X, dist, True)[1]

    best = None
    for start in (th0, th0 + np.r_[np.zeros(p), [] if dist == "exponential" else [-0.7]]):
        r = optimize.minimize(negll, start, jac=negg, method="BFGS", options=dict(gtol=1e-8, maxiter=2000))
        if best is None or r.fun < best.fun:
            best = r
    th = best.x
    # polish with Newton steps on the numerical Hessian of the analytic gradient
    for _ in range(25):
        H = _num_hess(th, lt, ev, X, dist)
        g = _aft_ll(th, lt, ev, X, dist, True)[1]
        try:
            st = np.linalg.solve(-H, g)
        except np.linalg.LinAlgError:
            break
        nth = th + st
        if _aft_ll(nth, lt, ev, X, dist) >= _aft_ll(th, lt, ev, X, dist) - 1e-12:
            th = nth
        if np.max(np.abs(st)) < 1e-9:
            break
    H = _num_hess(th, lt, ev, X, dist)
    try:
        cov = np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        raise OESError("The information matrix is singular; the model is not identified (check for collinear covariates).")
    if np.any(np.diag(cov) <= 0):
        raise OESError("The accelerated failure time model did not converge to a proper maximum.")
    ll = _aft_ll(th, lt, ev, X, dist)
    return dict(theta=th, cov=cov, ll=ll, k=k, p=p, dist=dist, n=n)


def _num_hess(th, lt, ev, X, dist):
    k = len(th)
    H = np.zeros((k, k))
    for j in range(k):
        h = 1e-5 * max(1.0, abs(th[j]))
        tp, tm = th.copy(), th.copy()
        tp[j] += h
        tm[j] -= h
        H[:, j] = (_aft_ll(tp, lt, ev, X, dist, True)[1] - _aft_ll(tm, lt, ev, X, dist, True)[1]) / (2 * h)
    return (H + H.T) / 2


def _aft_surv(fit, X, grid):
    """Survival function S(t | x_i) for every subject on a grid: returns (n_subjects, len(grid))."""
    p = fit["p"]
    beta = fit["theta"][:p]
    sig = 1.0 if fit["dist"] == "exponential" else np.exp(fit["theta"][p])
    mu = X @ beta if p else np.zeros(X.shape[0])
    z = (np.log(np.maximum(grid, 1e-300))[None, :] - mu[:, None]) / sig
    d = fit["dist"]
    if d in ("weibull", "exponential"):
        return np.exp(-np.exp(np.minimum(z, 700)))
    if d == "lognormal":
        return stats.norm.sf(z)
    return 1 / (1 + np.exp(z))


def _clean_times(d, tcol):
    if (d[tcol] <= 0).any():
        raise OESError("Accelerated failure time models need strictly positive follow-up times.")


def aft(df, spec):
    alpha = _alpha(spec)
    dist = spec.get("distribution") or "weibull"
    if dist not in _DISTS:
        raise OESError("Unknown distribution.")
    covs, facs = as_list(spec.get("covariates")), as_list(spec.get("factors"))
    if not covs and not facs:
        raise OESError("Choose at least one covariate.")
    d, dropped = _surv_data(df, spec, covs, facs)
    _clean_times(d, spec["time"])
    X, names = _design(d, covs, facs, intercept=True)
    t, e = d[spec["time"]].to_numpy(float), d[spec["event"]].to_numpy(int)
    n, p = X.shape
    if e.sum() < p + 1:
        raise OESError("There are too few events for the number of parameters.")
    fit = _aft_fit(t, e, X, dist)
    th, cov = fit["theta"], fit["cov"]
    se = np.sqrt(np.diag(cov))
    zc = stats.norm.ppf(1 - alpha / 2)
    lab = _ci_label(alpha)
    rows = []
    for j in range(p):
        z = th[j] / se[j]
        rows.append([names[j], f(th[j], 5), f(se[j], 5), f(z, 4), f(2 * stats.norm.sf(abs(z)), 6), f(th[j] - zc * se[j], 5), f(th[j] + zc * se[j], 5),
                     f(np.exp(th[j]), 5) if j > 0 else None, f(np.exp(th[j] - zc * se[j]), 5) if j > 0 else None, f(np.exp(th[j] + zc * se[j]), 5) if j > 0 else None])
    tabs = [table(f"{_DNAME[dist]} accelerated failure time model", ["Term", "Coefficient (log time)", "SE", "z", "p-value", f"{lab} low", "CI high", "Time ratio exp(coef)", "TR CI low", "TR CI high"], rows,
                  "The model is log(T) = X b + sigma * error. A time ratio above 1 means longer survival; below 1 means shorter. The intercept has no time ratio.")]
    summary = [stat("Subjects", n), stat("Events", int(e.sum())), stat("Log-likelihood", fit["ll"], 3)]
    k = fit["k"]
    aic, bic = -2 * fit["ll"] + 2 * k, -2 * fit["ll"] + k * np.log(n)
    summary += [stat("AIC", aic, 3), stat("BIC", bic, 3)]
    prow = []
    if dist != "exponential":
        ls = th[p]
        sig = float(np.exp(ls))
        se_sig = sig * float(np.sqrt(cov[p, p]))
        prow.append(["Scale sigma", f(sig, 5), f(se_sig, 5), "log-time error scale"])
        if dist == "weibull":
            rho = 1 / sig
            se_rho = rho * float(np.sqrt(cov[p, p]))
            prow.append(["Weibull shape rho = 1/sigma", f(rho, 5), f(se_rho, 5), "rho > 1: hazard rises with time; rho < 1: falls"])
        if dist == "loglogistic":
            prow.append(["Log-logistic shape = 1/sigma", f(1 / sig, 5), f(1 / sig * float(np.sqrt(cov[p, p])), 5), "shape > 1: hazard rises then falls"])
        tabs.append(table("Distribution parameters", ["Parameter", "Estimate", "SE", "Meaning"], prow))
    # null model for likelihood ratio
    if p > 1:
        f0 = _aft_fit(t, e, X[:, :1], dist)
        lr = 2 * (fit["ll"] - f0["ll"])
        tabs.append(table("Likelihood-ratio test against the intercept-only model", ["Statistic", "df", "p-value"], [[f(lr, 4), p - 1, f(stats.chi2.sf(lr, p - 1), 6)]]))
        summary.append(stat("LR chi-square", lr))
    # Cox-Snell residual check: exp(-log S) should look like a unit exponential sample with censoring
    sig = 1.0 if dist == "exponential" else np.exp(th[p])
    z = (np.log(t) - X @ th[:p]) / sig
    cs = -_eps(dist, z)[2]
    kmc = km_curve(cs, e, "loglog", alpha)
    if len(kmc["t"]):
        Hhat = -np.log(kmc["S"])
        grid = kmc["t"]
        if len(grid) > 250:
            ix = np.unique(np.linspace(0, len(grid) - 1, 250).astype(int))
            grid, Hhat = grid[ix], Hhat[ix]
        charts = [chart("Cox-Snell residual check (points should follow the 45-degree line)", grid, [("Estimated cumulative hazard of the residuals", Hhat), ("Unit exponential (45-degree line)", grid)], "Cox-Snell residual", "cumulative hazard", "line")]
    else:
        charts = []
    gridt = np.linspace(0, float(t.max()), 120)[1:]
    xm = X.mean(0, keepdims=True)
    Sm = _aft_surv(fit, xm, gridt)[0]
    charts.insert(0, chart("Fitted survival curve at the mean covariates", gridt, [("S(t | mean covariates)", Sm)], "time", "survival probability", "line"))
    notes = ["The exponential model fixes sigma = 1; the Weibull model nests it. The fitted model's validity is judged with the Cox-Snell chart and by comparing distributions with the parametric survival method."]
    if dropped:
        notes.append(f"{dropped} row(s) with missing values were dropped.")
    return result("stat_aft", f"Accelerated failure time model ({_DNAME[dist]})", n, summary, tabs, charts, notes)


def parametric(df, spec):
    alpha = _alpha(spec)
    covs, facs = as_list(spec.get("covariates")), as_list(spec.get("factors"))
    d, dropped = _surv_data(df, spec, covs, facs)
    _clean_times(d, spec["time"])
    X, names = _design(d, covs, facs, intercept=True)
    t, e = d[spec["time"]].to_numpy(float), d[spec["event"]].to_numpy(int)
    n = len(d)
    if e.sum() < 5:
        raise OESError("At least five events are needed to compare distributions.")
    fits = {}
    for dist in _DISTS:
        try:
            fits[dist] = _aft_fit(t, e, X, dist)
        except OESError:
            continue
    if not fits:
        raise OESError("No distribution could be fitted to these data.")
    for dist, ft in fits.items():
        ft["aic"] = -2 * ft["ll"] + 2 * ft["k"]
        ft["bic"] = -2 * ft["ll"] + ft["k"] * np.log(n)
    best = min(fits, key=lambda k_: fits[k_]["aic"])
    amin = fits[best]["aic"]
    w = {k_: np.exp(-(v["aic"] - amin) / 2) for k_, v in fits.items()}
    sw = sum(w.values())
    cmp_rows = [[_DNAME[k_], fits[k_]["k"], f(fits[k_]["ll"], 3), f(fits[k_]["aic"], 3), f(fits[k_]["bic"], 3), f(w[k_] / sw, 4), "best by AIC" if k_ == best else ""] for k_ in fits]
    tabs = [table("Comparison of parametric survival distributions", ["Distribution", "Parameters", "Log-likelihood", "AIC", "BIC", "Akaike weight", ""], cmp_rows,
                  "Lower AIC and BIC are better. The exponential distribution is a Weibull with shape 1; the Weibull and the exponential are proportional-hazards as well as AFT models.")]
    # parameter table for each distribution
    prow = []
    zc = stats.norm.ppf(1 - alpha / 2)
    for k_, ft in fits.items():
        th, cov = ft["theta"], ft["cov"]
        se = np.sqrt(np.diag(cov))
        p = ft["p"]
        for j in range(p):
            prow.append([_DNAME[k_], names[j], f(th[j], 5), f(se[j], 5), f(2 * stats.norm.sf(abs(th[j] / se[j])), 6)])
        if k_ != "exponential":
            sig = float(np.exp(th[p]))
            prow.append([_DNAME[k_], "scale sigma", f(sig, 5), f(sig * se[p], 5), None])
    tabs.append(table("Parameter estimates (log-time scale)", ["Distribution", "Term", "Estimate", "SE", "p-value"], prow))
    grid = np.unique(np.concatenate([[0.0], np.sort(np.unique(t[e == 1]))]))
    if len(grid) > 250:
        grid = grid[np.unique(np.linspace(0, len(grid) - 1, 250).astype(int))]
    km = km_curve(t, e, "loglog", alpha)
    series = [("Kaplan-Meier", _step_eval(km["t"], km["S"], grid))]
    gerr = []
    qrows = []
    fine = np.linspace(0, float(t.max()) * 1.5, 600)[1:]
    for k_, ft in fits.items():
        Sp = _aft_surv(ft, X, np.maximum(grid, 1e-9)).mean(0)
        Sp[grid == 0] = 1.0
        series.append((_DNAME[k_], Sp))
        gerr.append([_DNAME[k_], f(float(np.max(np.abs(Sp - series[0][1]))), 4), f(float(np.sqrt(np.mean((Sp - series[0][1]) ** 2))), 4)])
        Sf = _aft_surv(ft, X, fine).mean(0)
        qs = []
        for q in (0.25, 0.5, 0.75):
            ix = np.where(Sf <= 1 - q)[0]
            qs.append(float(fine[ix[0]]) if len(ix) else float("nan"))
        qrows.append([_DNAME[k_]] + [f(v, 4) for v in qs])
    kmq = [_first_below(km["t"], km["S"], 1 - q) for q in (0.25, 0.5, 0.75)]
    qrows.insert(0, ["Kaplan-Meier"] + [f(v, 4) for v in kmq])
    tabs.append(table("Distance from the Kaplan-Meier curve (population-average fitted survival)", ["Distribution", "Max abs difference", "Root mean squared difference"], gerr,
                      "Fitted curves are averaged over the subjects' covariate values, so they are comparable to the single Kaplan-Meier curve."))
    tabs.append(table("Survival-time quantiles (time by which 25%, 50%, 75% have had the event)", ["Source", "25%", "50% (median)", "75%"], qrows,
                      "Quantiles of the fitted distributions are taken from the population-average curve and are limited to 1.5 times the longest observed time."))
    charts = [chart("Kaplan-Meier curve against fitted parametric survival functions", grid, series, "time", "survival probability", "line")]
    # hazard shape for the best distribution (intercept-only view at mean covariates)
    ft = fits[best]
    tg = np.linspace(float(t.max()) / 120, float(t.max()), 120)
    xm = X.mean(0, keepdims=True)
    S1 = _aft_surv(ft, xm, tg)[0]
    S2 = _aft_surv(ft, xm, tg * 1.0001)[0]
    haz = -(np.log(S2) - np.log(S1)) / (tg * 0.0001)
    charts.append(chart(f"Hazard function of the best-fitting model ({_DNAME[best]}) at the mean covariates", tg, [("hazard", haz)], "time", "hazard rate", "line"))
    summary = [stat("Subjects", n), stat("Events", int(e.sum())), stat("Best by AIC", _DNAME[best]), stat("Best by BIC", _DNAME[min(fits, key=lambda k_: fits[k_]["bic"])])]
    notes = ["With covariates the models are accelerated failure time regressions; with none they describe a single survival distribution."]
    if dropped:
        notes.append(f"{dropped} row(s) with missing values were dropped.")
    return result("stat_parametric", "Parametric survival distributions", n, summary, tabs, charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# Fine-Gray subdistribution hazards (competing risks)
# ---------------------------------------------------------------------------------------------------------------
def _fg_weights(time, status, cause, ut):
    """IPCW matrix W[j, i]: weight of subject i in the risk set at event time ut[j]."""
    cens = (status == 0).astype(int)
    kmc = km_curve(time, cens, "plain", 0.05)
    ct, cS = kmc["t"], kmc["S"]

    def G(x):
        return _step_eval(ct, cS, np.atleast_1d(x), before=1.0)

    Gt = G(ut)
    Gi = G(time)
    competing = (status != 0) & (status != cause)
    W = np.zeros((len(ut), len(time)))
    for j, u in enumerate(ut):
        at = time >= u
        W[j, at] = 1.0
        late = competing & (time < u)
        with np.errstate(divide="ignore", invalid="ignore"):
            W[j, late] = np.where(Gi[late] > 0, Gt[j] / Gi[late], 0.0)
    return W


def _fg_terms(X, ev_of_interest, time, W, ut, beta):
    n, p = X.shape
    eta = X @ beta
    eta = eta - eta.max()
    r = np.exp(eta)
    Wr = W * r[None, :]
    S0 = Wr.sum(1)
    S1 = Wr @ X
    S2 = np.einsum("ji,ik,il->jkl", Wr, X, X)
    dj = np.array([np.sum((time == u) & ev_of_interest) for u in ut], float)
    ll = float(eta[ev_of_interest].sum()) - float(np.sum(dj * np.log(S0)))
    m1 = S1 / S0[:, None]
    g = X[ev_of_interest].sum(0) - (dj[:, None] * m1).sum(0)
    H = -np.sum(dj[:, None, None] * (S2 / S0[:, None, None] - m1[:, :, None] * m1[:, None, :]), axis=0)
    return ll, g, H


def _aalen_johansen(time, status, cause, grid):
    """Cumulative incidence of the cause of interest and of the competing events."""
    anyev = (status != 0).astype(int)
    ut = np.unique(time[status != 0])
    S_prev = 1.0
    F1 = np.zeros(len(ut))
    F2 = np.zeros(len(ut))
    a1 = a2 = 0.0
    for i, u in enumerate(ut):
        n = float(np.sum(time >= u))
        d1 = float(np.sum((time == u) & (status == cause)))
        d2 = float(np.sum((time == u) & (status != 0) & (status != cause)))
        a1 += S_prev * d1 / n
        a2 += S_prev * d2 / n
        S_prev *= 1 - (d1 + d2) / n
        F1[i], F2[i] = a1, a2
    return _step_eval(ut, F1, grid, 0.0), _step_eval(ut, F2, grid, 0.0)


def fine_gray(df, spec):
    alpha = _alpha(spec)
    covs, facs = as_list(spec.get("covariates")), as_list(spec.get("factors"))
    if not covs and not facs:
        raise OESError("Choose at least one covariate.")
    cause = int(float_option(spec, "cause", 1, "Cause of interest"))
    d, dropped = _surv_data(df, {"time": spec.get("time"), "event": spec.get("event")}, covs, facs, status_codes=True)
    X, names = _design(d, covs, facs)
    t, st = d[spec["time"]].to_numpy(float), d[spec["event"]].to_numpy(int)
    if not np.any(st == cause):
        raise OESError(f"No subject has status code {cause}; check the cause of interest.")
    comp = np.sum((st != 0) & (st != cause))
    if comp == 0:
        raise OESError("There are no competing events in the data; use the Cox model instead.")
    n, p = X.shape
    ev = st == cause
    ut = np.unique(t[ev])
    W = _fg_weights(t, st, cause, ut)
    beta = np.zeros(p)
    ll, g, H = _fg_terms(X, ev, t, W, ut, beta)
    ll0 = ll
    for it in range(60):
        try:
            step = np.linalg.solve(-H, g)
        except np.linalg.LinAlgError:
            raise OESError("The covariates are collinear, so the model cannot be estimated.")
        s = 1.0
        while True:
            nb = beta + s * step
            nll, ng, nH = _fg_terms(X, ev, t, W, ut, nb)
            if np.isfinite(nll) and nll >= ll - 1e-10:
                break
            s /= 2
            if s < 1e-8:
                break
        done = abs(nll - ll) < 1e-10 * (1 + abs(nll)) and np.max(np.abs(s * step)) < 1e-8
        beta, ll, g, H = nb, nll, ng, nH
        if np.max(np.abs(beta)) > 50:
            raise OESError("The estimates diverge; a covariate may perfectly predict the event.")
        if done:
            break
    else:
        raise OESError("The Newton-Raphson iterations did not converge.")
    cov = np.linalg.inv(-H)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pv = 2 * stats.norm.sf(np.abs(z))
    zc = stats.norm.ppf(1 - alpha / 2)
    lab = _ci_label(alpha)
    rows = [[names[j], f(beta[j], 5), f(np.exp(beta[j]), 5), f(se[j], 5), f(z[j], 4), f(pv[j], 6), f(np.exp(beta[j] - zc * se[j]), 5), f(np.exp(beta[j] + zc * se[j]), 5)] for j in range(p)]
    lr = 2 * (ll - ll0)
    tabs = [table(f"Fine-Gray subdistribution hazard model (cause {cause})", ["Covariate", "coef", "sHR = exp(coef)", "SE", "z", "p-value", f"sHR {lab} low", "sHR CI high"], rows,
                  "sHR is the subdistribution hazard ratio: it compares the instantaneous rate of the event of interest among subjects who have not yet had that event (those who had a competing event stay in the risk set, weighted). A value above 1 means a higher cumulative incidence of the cause of interest."),
            table("Event counts", ["Group", "Subjects"], [["Event of interest (code " + str(cause) + ")", int(ev.sum())], ["Competing events", int(comp)], ["Censored", int(np.sum(st == 0))]])]
    # baseline and cumulative incidence
    eta = X @ beta
    shift = eta.max()
    r = np.exp(eta - shift)
    Wr = W * r[None, :]
    S0 = Wr.sum(1)
    dj = np.array([np.sum((t == u) & ev) for u in ut], float)
    H0 = np.cumsum(dj / S0) * np.exp(-shift)
    grid = np.concatenate([[0.0], ut])
    if len(grid) > 300:
        grid = grid[np.unique(np.linspace(0, len(grid) - 1, 300).astype(int))]
    xbar = X.mean(0)
    Fm = 1 - np.exp(-_step_eval(ut, H0, grid, 0.0) * np.exp(xbar @ beta))
    aj1, aj2 = _aalen_johansen(t, st, cause, grid)
    charts = [chart("Cumulative incidence of the cause of interest", grid, [("Aalen-Johansen (nonparametric)", aj1), ("Fine-Gray model at mean covariates", Fm), ("Competing events (Aalen-Johansen)", aj2)], "time", "cumulative incidence", "line")]
    qt = np.unique(np.round(np.quantile(t[ev], [0.25, 0.5, 0.75]), 6))
    tabs.append(table("Cumulative incidence at selected times", ["Time", "Aalen-Johansen, cause of interest", "Fine-Gray at mean covariates", "Aalen-Johansen, competing events"],
                      [[f(q, 4), f(float(_aalen_johansen(t, st, cause, np.array([q]))[0][0]), 5), f(float(1 - np.exp(-_step_eval(ut, H0, np.array([q]), 0.0)[0] * np.exp(xbar @ beta))), 5),
                        f(float(_aalen_johansen(t, st, cause, np.array([q]))[1][0]), 5)] for q in qt]))
    summary = [stat("Subjects", n), stat(f"Events (cause {cause})", int(ev.sum())), stat("Competing events", int(comp)), stat("Score-type LR chi-square", lr), stat("p-value", float(stats.chi2.sf(lr, p)), 6)]
    notes = ["Standard errors come from the inverse of the weighted partial-likelihood information and do not include the extra variance from estimating the censoring distribution (the correction in R's cmprsk). They are close when censoring is light and can be somewhat too small when it is heavy.",
             "Censoring is assumed independent of the covariates and the event times."]
    if dropped:
        notes.append(f"{dropped} row(s) with missing values were dropped.")
    return result("stat_finegray", "Fine-Gray competing-risks regression", n, summary, tabs, charts, notes)
