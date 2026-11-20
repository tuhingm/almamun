"""Productivity and spatial models: stochastic frontier analysis, data envelopment analysis, Malmquist productivity indexes,
growth regressions (Mankiw-Romer-Weil), production and cost functions, spatial regression models.
Pure NumPy and SciPy. Same result format as the other models (core.py)."""
import numpy as np
import pandas as pd
from scipy import optimize, stats

from .core import (OESError, as_list, chart, check_columns, distinct, f, float_option, int_option, need, numeric, result, stat, table, test_row, tests_table)


def _num_hessian(fn, x, eps=1e-4):
    k = len(x)
    H = np.zeros((k, k))
    h = eps * np.maximum(1.0, np.abs(x))
    f0 = fn(x)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ej = np.zeros(k); ei[i] = h[i]; ej[j] = h[j]
            if i == j:
                H[i, i] = (fn(x + ei) - 2 * f0 + fn(x - ei)) / h[i] ** 2
            else:
                H[i, j] = H[j, i] = (fn(x + ei + ej) - fn(x + ei - ej) - fn(x - ei + ej) + fn(x - ei - ej)) / (4 * h[i] * h[j])
    return H


# ================================================================================================ stochastic frontier
def _sfa_nll(par, y, X, Z, dist, sgn):
    k = X.shape[1]
    b = par[:k]
    lsv = par[k]
    nz = Z.shape[1]
    d = par[k + 1:k + 1 + nz]
    lsu = Z @ d
    mu = par[k + 1 + nz] if dist == "truncated" else 0.0
    e = sgn * (y - X @ b)                                # e = v - u for a production frontier
    sv = np.exp(lsv)
    su = np.exp(np.clip(lsu, -20, 20))
    if dist == "exponential":
        z1 = -e / sv - sv / su
        ll = -np.log(su) + e / su + sv ** 2 / (2 * su ** 2) + stats.norm.logcdf(z1)
    else:
        s2 = sv ** 2 + su ** 2
        s = np.sqrt(s2)
        mt = (-su ** 2 * e + mu * sv ** 2) / s2
        st = su * sv / s
        ll = -np.log(s) + stats.norm.logpdf((e + mu) / s) + stats.norm.logcdf(mt / st) - stats.norm.logcdf(mu / su)
    v = -np.sum(ll)
    return v if np.isfinite(v) else 1e12


def _sfa_eff(par, y, X, Z, dist, sgn):
    k = X.shape[1]
    nz = Z.shape[1]
    b = par[:k]; sv = np.exp(par[k]); su = np.exp(np.clip(Z @ par[k + 1:k + 1 + nz], -20, 20))
    mu = par[k + 1 + nz] if dist == "truncated" else 0.0
    e = sgn * (y - X @ b)
    if dist == "exponential":
        mt = -e - sv ** 2 / su
        st = np.full_like(e, sv)
    else:
        s2 = sv ** 2 + su ** 2
        mt = (-su ** 2 * e + mu * sv ** 2) / s2
        st = su * sv / np.sqrt(s2)
    r = mt / st
    Eu = mt + st * stats.norm.pdf(r) / np.maximum(stats.norm.cdf(r), 1e-300)
    te_bc = np.exp(-mt + st ** 2 / 2) * stats.norm.cdf(r - st) / np.maximum(stats.norm.cdf(r), 1e-300)
    return Eu, te_bc, e


def _sfa_fit(y, X, Z, dist, sgn):
    n, k = X.shape
    nz = Z.shape[1]
    b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    r0 = sgn * (y - X @ b0)
    s2 = r0.var()
    m3 = np.mean((r0 - r0.mean()) ** 3)
    # moment starting values (Aigner, Lovell and Schmidt 1977): the third moment identifies sigma_u
    su0 = np.cbrt(max(-m3 / (np.sqrt(2 / np.pi) * (1 - 4 / np.pi)), 1e-6)) if m3 < 0 else 0.1 * np.sqrt(s2)
    su0 = min(su0, 0.95 * np.sqrt(s2))
    sv0 = np.sqrt(max(s2 - (1 - 2 / np.pi) * su0 ** 2, 0.05 * s2))
    if dist == "exponential":
        su0 = su0 * 0.8
    b_start = b0.copy()
    b_start[0] += sgn * (np.sqrt(2 / np.pi) * su0 if dist != "exponential" else su0) * (-1)
    best = None
    for scale in (1.0, 0.5, 2.0):
        x0 = np.concatenate([b_start, [np.log(sv0)], [np.log(su0 * scale)], np.zeros(nz - 1), [0.0] if dist == "truncated" else []])
        for meth in ("BFGS", "Nelder-Mead"):
            try:
                if meth == "BFGS":
                    o = optimize.minimize(_sfa_nll, x0, args=(y, X, Z, dist, sgn), method="BFGS", options=dict(maxiter=500, gtol=1e-6))
                else:
                    o = optimize.minimize(_sfa_nll, x0, args=(y, X, Z, dist, sgn), method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-7, fatol=1e-9, adaptive=True))
                    o = optimize.minimize(_sfa_nll, o.x, args=(y, X, Z, dist, sgn), method="BFGS", options=dict(maxiter=500, gtol=1e-6))
            except Exception:
                continue
            if best is None or o.fun < best.fun:
                best = o
            if meth == "BFGS" and o.success:
                break
        if best is not None and best.success:
            break
    if best is None or best.fun >= 1e11:
        raise OESError("The stochastic frontier likelihood could not be maximised; try another distribution or fewer inputs.")
    ll_ols = float(np.sum(stats.norm.logpdf(y - X @ b0, scale=np.sqrt(np.mean((y - X @ b0) ** 2)))))
    return best, b0, ll_ols


def sfa(df, spec):
    yv = need(spec, "y", "output (in logs, or choose log = yes)")
    xs = as_list(need(spec, "x", "inputs"))
    zs = as_list(spec.get("z"))
    grp = spec.get("group") or None
    distinct([yv], xs)
    kind = str(spec.get("frontier", "production"))
    dist_main = str(spec.get("distribution", "half-normal")).replace("_", "-")
    if dist_main == "half-normal":
        dist_main = "halfnormal"
    take_log = str(spec.get("log", "yes")).lower() in ("yes", "true", "1")
    cols_ = [yv] + xs + zs
    d = numeric(df, cols_, keep=[grp] if grp else [])
    if take_log:
        for c in [yv] + xs:
            if (d[c] <= 0).any():
                raise OESError(f"Column {c} has zeros or negative values: choose log = no, or remove them.")
    Y = np.log(d[yv].to_numpy(float)) if take_log else d[yv].to_numpy(float)
    Xr = np.log(d[xs].to_numpy(float)) if take_log else d[xs].to_numpy(float)
    n = len(Y)
    if n < 30 + 10 * len(xs):
        raise OESError(f"Too few observations ({n}) for {len(xs)} inputs: at least {30 + 10 * len(xs)} are needed.")
    X = np.column_stack([np.ones(n), Xr])
    Z = np.column_stack([np.ones(n)] + ([d[zs].to_numpy(float)] if zs else []))
    sgn = 1.0 if kind == "production" else -1.0
    fits = {}
    for dist in ("halfnormal", "exponential", "truncated"):
        if zs and dist != dist_main:
            continue
        try:
            o, b0, ll_ols = _sfa_fit(Y, X, Z, dist, sgn)
            fits[dist] = o
        except OESError:
            continue
    if dist_main not in fits:
        raise OESError("The chosen distribution could not be estimated on these data; choose another one.")
    o = fits[dist_main]
    par = o.x
    k = X.shape[1]; nz = Z.shape[1]
    names = ["Constant"] + xs
    try:
        H = _num_hessian(lambda p: _sfa_nll(p, Y, X, Z, dist_main, sgn), par)
        V = np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(V), 0, None))
    except Exception:
        V = None
        se = np.full(len(par), np.nan)
    rows = []
    for i in range(k):
        rows.append([names[i], par[i], se[i], par[i] / se[i] if se[i] else None, 2 * stats.norm.sf(abs(par[i] / se[i])) if se[i] else None])
    sv, su0 = np.exp(par[k]), np.exp(par[k + 1])
    if nz == 1:
        lam = su0 / sv
        rows.append(["ln sigma_v", par[k], se[k], par[k] / se[k] if se[k] else None, None])
        rows.append(["ln sigma_u", par[k + 1], se[k + 1], par[k + 1] / se[k + 1] if se[k + 1] else None, None])
    else:
        rows.append(["ln sigma_v", par[k], se[k], par[k] / se[k] if se[k] else None, None])
        for j in range(nz):
            nm = "Constant (ln sigma_u)" if j == 0 else f"{zs[j - 1]} (ln sigma_u)"
            rows.append([nm, par[k + 1 + j], se[k + 1 + j], par[k + 1 + j] / se[k + 1 + j] if se[k + 1 + j] else None,
                         2 * stats.norm.sf(abs(par[k + 1 + j] / se[k + 1 + j])) if se[k + 1 + j] else None])
    if dist_main == "truncated":
        rows.append(["mu (mean of the truncated normal)", par[-1], se[-1], par[-1] / se[-1] if se[-1] else None, 2 * stats.norm.sf(abs(par[-1] / se[-1])) if se[-1] else None])
    ptab = table(f"Stochastic {kind} frontier: {dict(halfnormal='half-normal', exponential='exponential', truncated='truncated-normal')[dist_main]} inefficiency", ["Parameter", "Estimate", "s.e.", "z", "p"], rows,
                 f"Maximum likelihood, {n} observations; inputs and output in {'logarithms' if take_log else 'levels'}.")
    # variance decomposition
    if dist_main == "exponential":
        var_u = su0 ** 2
    elif dist_main == "truncated":
        mu = par[-1]; a = mu / su0
        var_u = su0 ** 2 * (1 - a * stats.norm.pdf(a) / stats.norm.cdf(a) - (stats.norm.pdf(a) / stats.norm.cdf(a)) ** 2)
    else:
        var_u = (1 - 2 / np.pi) * su0 ** 2
    gamma = var_u / (var_u + sv ** 2)
    # tests
    ll_sfa = -o.fun
    lr = max(2 * (ll_sfa - ll_ols), 0.0)
    nextra = 1 + (nz - 1) + (1 if dist_main == "truncated" else 0)
    r_ols = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
    r_ols_s = sgn * r_ols
    m2 = np.mean((r_ols_s - r_ols_s.mean()) ** 2); m3 = np.mean((r_ols_s - r_ols_s.mean()) ** 3)
    zsk = m3 / np.sqrt(6 * m2 ** 3 / n)
    p_lr = 0.5 * stats.chi2.sf(lr, 1) if nextra == 1 else 0.5 * stats.chi2.sf(lr, nextra) + 0.5 * stats.chi2.sf(lr, nextra - 1)
    trows = [test_row("Skewness of the OLS residuals (Schmidt-Lin)", zsk, stats.norm.cdf(zsk) if True else 0, "Expected negative: p is the one-sided probability of a value this low or lower"),
             test_row("LR test of no inefficiency (against OLS)", lr, p_lr, "Mixed chi-square (Coelli 1995); H0: sigma_u = 0")]
    if str(spec.get("crs_test", "yes")) == "yes" and take_log and V is not None and kind == "production":
        w = np.zeros(len(par)); w[1:k] = 1
        diff = w @ par - 1.0
        wald = diff ** 2 / (w @ V @ w)
        trows.append(test_row("Constant returns to scale: sum of input elasticities = 1 (Wald)", wald, stats.chi2.sf(wald, 1), f"sum = {w @ par:.3f}"))
    ttab = tests_table(trows, "Tests")
    # efficiency
    Eu, te_bc, eps = _sfa_eff(par, Y, X, Z, dist_main, sgn)
    te_j = np.exp(-Eu)
    q = np.percentile(te_bc, [10, 25, 50, 75, 90])
    etab = table("Technical efficiency (Battese-Coelli, E[exp(-u)|e])", ["Statistic", "Value"],
                 [["Mean", float(te_bc.mean())], ["Standard deviation", float(te_bc.std(ddof=1))], ["Minimum", float(te_bc.min())], ["10th percentile", float(q[0])], ["25th percentile", float(q[1])],
                  ["Median", float(q[2])], ["75th percentile", float(q[3])], ["90th percentile", float(q[4])], ["Maximum", float(te_bc.max())], ["Share below 0.7", float((te_bc < 0.7).mean())],
                  ["Mean of exp(-E[u|e]) (Jondrow et al. 1982)", float(te_j.mean())], ["Correlation of the two measures", float(np.corrcoef(te_bc, te_j)[0, 1])]],
                 "Efficiency is the ratio of observed output to the frontier output (production) or of the frontier cost to observed cost (cost).")
    tabs = [ptab, table("Variance decomposition", ["Quantity", "Value"],
                        [["sigma_v (noise)", float(sv)], ["sigma_u (inefficiency scale)", float(su0)], ["lambda = sigma_u / sigma_v", float(su0 / sv)], ["Share of the error variance due to inefficiency", float(gamma)],
                         ["Mean inefficiency E[u]", float(np.mean(Eu))], ["Log-likelihood", float(ll_sfa)], ["Log-likelihood of OLS", float(ll_ols)]]), ttab, etab]
    # comparison of distributions
    cmp_rows = []
    if not zs:
        for dn, lab in (("halfnormal", "Half-normal"), ("exponential", "Exponential"), ("truncated", "Truncated normal")):
            if dn in fits:
                of = fits[dn]
                kk = len(of.x)
                _, tb, _ = _sfa_eff(of.x, Y, X, Z, dn, sgn)
                cmp_rows.append([lab, float(-of.fun), kk, float(2 * kk + 2 * of.fun), float(np.log(n) * kk + 2 * of.fun), float(tb.mean()), float(of.x[1] if k > 1 else 0)])
        tabs.append(table("Comparison of inefficiency distributions", ["Distribution", "Log-likelihood", "Parameters", "AIC", "BIC", "Mean efficiency", "Elasticity of the first input"], cmp_rows,
                           "The three models are not nested except the half-normal in the truncated normal; compare by AIC and BIC, and check how much the mean efficiency changes."))
    if grp:
        g = d[grp].astype(str)
        gr = []
        for lv in sorted(g.unique()):
            m_ = (g == lv).to_numpy()
            if m_.sum() >= 5:
                gr.append([lv, int(m_.sum()), float(te_bc[m_].mean()), float(np.median(te_bc[m_])), float(te_bc[m_].min())])
        tabs.append(table(f"Efficiency by {grp}", [grp, "Observations", "Mean", "Median", "Minimum"], gr))
    # histogram
    bins = np.linspace(max(0, te_bc.min() - 1e-9), 1.0, 11)
    hist = np.histogram(te_bc, bins=bins)[0]
    ch = [chart("Distribution of technical efficiency", [f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(10)], [("Observations", hist)], "Efficiency", "Count", "bar")]
    ordr = np.argsort(te_bc)
    return result("sfa", "Stochastic frontier analysis", n, [stat("Observations", n), stat("Mean efficiency", float(te_bc.mean()), 3), stat("lambda", float(su0 / sv), 3), stat("Inefficiency share of the variance", float(gamma), 3),
                                                              stat("Log-likelihood", float(ll_sfa), 2)], tabs, ch,
                  ["Aigner, Lovell and Schmidt (1977) and Meeusen and van den Broeck (1977): the error is v - u (production) or v + u (cost), with v normal noise and u a non-negative inefficiency.",
                   "Efficiency is estimated for each observation from the composed residual (Jondrow et al. 1982; Battese and Coelli 1988); with a cross-section its estimate is not consistent for any one unit, only its distribution is informative."])


# ================================================================================================ DEA
def _dea_one(X, Y, x0, y0, rts, orient, exclude=None):
    """Efficiency of (x0,y0) against reference set (X,Y) (rows = units). Returns theta (input: <=1; output: >=1)."""
    n, m = X.shape
    s = Y.shape[1]
    idx = np.arange(n) if exclude is None else np.array([i for i in range(n) if i != exclude])
    Xr, Yr = X[idx], Y[idx]
    k = len(idx)
    if orient == "input":
        c = np.concatenate([[1.0], np.zeros(k)])
        A = np.vstack([np.hstack([-x0[:, None], Xr.T]), np.hstack([np.zeros((s, 1)), -Yr.T])])
        b = np.concatenate([np.zeros(m), -y0])
    else:
        c = np.concatenate([[-1.0], np.zeros(k)])
        A = np.vstack([np.hstack([np.zeros((m, 1)), Xr.T]), np.hstack([y0[:, None], -Yr.T])])
        b = np.concatenate([x0, np.zeros(s)])
    Aeq = np.concatenate([[0.0], np.ones(k)])[None, :] if rts == "vrs" else None
    beq = [1.0] if rts == "vrs" else None
    o = optimize.linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=[(None, None)] + [(0, None)] * k, method="highs")
    if o.status != 0:
        return np.nan, None
    return (o.x[0] if orient == "input" else -o.fun * 1.0 if False else o.x[0]), o.x[1:]


def dea(df, spec):
    xs = as_list(need(spec, "inputs", "inputs"))
    ys = as_list(need(spec, "outputs", "outputs"))
    lab = spec.get("label") or None
    distinct(xs, ys)
    orient = str(spec.get("orientation", "input"))
    sup = str(spec.get("super", "no")) == "yes"
    d = numeric(df, xs + ys, keep=[lab] if lab else [])
    if (d[xs + ys] <= 0).any().any():
        raise OESError("DEA needs strictly positive inputs and outputs: remove or recode zeros and negatives.")
    n = len(d)
    if n < 2 * (len(xs) + len(ys)) + 3:
        raise OESError(f"Too few units ({n}) for {len(xs)} inputs and {len(ys)} outputs; a common rule is n >= 2(m+s)+3.")
    X = d[xs].to_numpy(float); Y = d[ys].to_numpy(float)
    names = d[lab].astype(str).tolist() if lab else [str(i + 1) for i in range(n)]
    sc = {}
    for rts in ("crs", "vrs"):
        sc[rts] = np.array([_dea_one(X, Y, X[i], Y[i], rts, orient)[0] for i in range(n)])
    se = sc["crs"] / sc["vrs"] if orient == "input" else sc["vrs"] / sc["crs"]
    # in the output orientation theta>=1 is the expansion factor; report efficiency = 1/theta
    ce = sc["crs"] if orient == "input" else 1 / sc["crs"]
    ve = sc["vrs"] if orient == "input" else 1 / sc["vrs"]
    scale = ce / ve
    # returns to scale: NIRS
    ne = np.array([_dea_one_nirs(X, Y, X[i], Y[i], orient) for i in range(n)])
    ne = ne if orient == "input" else 1 / ne
    rtsl = np.where(np.isclose(ce, ve, atol=1e-6), "constant", np.where(np.isclose(ve, ne, atol=1e-6), "decreasing", "increasing"))
    summ = lambda a: [float(a.mean()), float(a.std(ddof=1)), float(a.min()), float(np.median(a)), float(a.max()), float((a >= 1 - 1e-7).mean())]
    tabs = [table(f"DEA efficiency scores ({orient}-oriented)", ["Score", "Mean", "Std. dev.", "Minimum", "Median", "Maximum", "Share efficient"],
                  [["Technical efficiency, constant returns (CCR)"] + summ(ce), ["Pure technical efficiency, variable returns (BCC)"] + summ(ve), ["Scale efficiency (CCR / BCC)"] + summ(scale)],
                  f"{n} decision-making units, {len(xs)} inputs, {len(ys)} outputs. A score of 1 means the unit is on the frontier.")]
    tabs.append(table("Returns to scale", ["Type", "Units", "Share"], [[t, int((rtsl == t).sum()), float((rtsl == t).mean())] for t in ("increasing", "constant", "decreasing")],
                      "Constant where CCR = BCC; decreasing where the non-increasing-returns frontier equals BCC; increasing otherwise."))
    # peers / lambda for the least efficient units
    order = np.argsort(ce)
    rows = []
    for i in order[:min(15, n)]:
        _, lam = _dea_one(X, Y, X[i], Y[i], "crs", orient)
        peers = ", ".join(f"{names[j]} ({lam[j]:.2f})" for j in np.argsort(-lam)[:3] if lam[j] > 1e-6)
        rows.append([names[i], float(ce[i]), float(ve[i]), float(scale[i]), str(rtsl[i]), peers])
    tabs.append(table("Least efficient units and their reference peers (CCR)", ["Unit", "CCR", "BCC", "Scale", "Returns to scale", "Peers (weights)"], rows))
    # slacks-free input targets: average potential saving
    if orient == "input":
        tabs.append(table("Average potential input saving (CCR)", ["Input", "Mean observed", "Mean at the frontier", "Saving"],
                          [[xs[j], float(X[:, j].mean()), float((ce * X[:, j]).mean()), float(1 - (ce * X[:, j]).mean() / X[:, j].mean())] for j in range(len(xs))],
                          "Radial targets only; slacks are not added."))
    else:
        tabs.append(table("Average potential output gain (CCR)", ["Output", "Mean observed", "Mean at the frontier", "Gain"],
                          [[ys[j], float(Y[:, j].mean()), float((Y[:, j] / ce).mean()), float((1 / ce).mean() - 1)] for j in range(len(ys))], "Radial targets only."))
    if sup:
        sp = np.array([_dea_one(X, Y, X[i], Y[i], "crs", orient, exclude=i)[0] for i in range(n)])
        sp = sp if orient == "input" else 1 / sp
        top = np.argsort(-np.where(np.isnan(sp), 0, sp))[:10]
        tabs.append(table("Super-efficiency (Andersen-Petersen, CCR)", ["Unit", "Super-efficiency"], [[names[i], float(sp[i])] for i in top],
                          "Each unit is excluded from its own reference set; values above 1 rank the efficient units."))
    bins = np.linspace(max(0, ce.min() - 1e-9), 1.0001, 11)
    h = np.histogram(ce, bins=bins)[0]
    ch = [chart("Distribution of CCR efficiency", [f"{bins[i]:.2f}-{min(bins[i + 1], 1):.2f}" for i in range(10)], [("Units", h)], "Efficiency", "Units", "bar")]
    return result("dea", "Data envelopment analysis", n, [stat("Units", n), stat("Mean CCR efficiency", float(ce.mean()), 3), stat("Mean BCC efficiency", float(ve.mean()), 3), stat("Mean scale efficiency", float(scale.mean()), 3),
                                                       stat("Units on the CCR frontier", int((ce >= 1 - 1e-7).sum()))], tabs, ch,
                  ["Charnes, Cooper and Rhodes (1978) and Banker, Charnes and Cooper (1984). DEA has no error term: noise and measurement error count as inefficiency, and the scores depend on the sample and on the inputs and outputs chosen.",
                   "With few units and many inputs and outputs, most units come out efficient; do not read a high share efficient as good performance."])


def _dea_one_nirs(X, Y, x0, y0, orient):
    n, m = X.shape
    s = Y.shape[1]
    if orient == "input":
        c = np.concatenate([[1.0], np.zeros(n)])
        A = np.vstack([np.hstack([-x0[:, None], X.T]), np.hstack([np.zeros((s, 1)), -Y.T]), np.concatenate([[0.0], np.ones(n)])[None, :]])
        b = np.concatenate([np.zeros(m), -y0, [1.0]])
    else:
        c = np.concatenate([[-1.0], np.zeros(n)])
        A = np.vstack([np.hstack([np.zeros((m, 1)), X.T]), np.hstack([y0[:, None], -Y.T]), np.concatenate([[0.0], np.ones(n)])[None, :]])
        b = np.concatenate([x0, np.zeros(s), [1.0]])
    o = optimize.linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] + [(0, None)] * n, method="highs")
    return o.x[0] if o.status == 0 else np.nan


# ================================================================================================ Malmquist
def malmquist(df, spec):
    xs = as_list(need(spec, "inputs", "inputs"))
    ys = as_list(need(spec, "outputs", "outputs"))
    uid = need(spec, "id", "unit identifier")
    tv = need(spec, "time", "time variable")
    distinct(xs, ys)
    orient = str(spec.get("orientation", "input"))
    d = numeric(df, xs + ys + [tv], keep=[uid])
    if (d[xs + ys] <= 0).any().any():
        raise OESError("The Malmquist index needs strictly positive inputs and outputs.")
    periods = sorted(d[tv].unique())
    if len(periods) < 2:
        raise OESError("Need at least two time periods.")
    cnt = d.groupby(uid)[tv].nunique()
    full = cnt[cnt == len(periods)].index
    if len(full) < 2 * (len(xs) + len(ys)) + 3:
        raise OESError("Too few units observed in every period for a balanced Malmquist index.")
    d = d[d[uid].isin(full)]
    dat = {p: d[d[tv] == p].sort_values(uid) for p in periods}
    ids = dat[periods[0]][uid].astype(str).tolist()

    def D(pref, pobs):
        Xr = dat[pref][xs].to_numpy(float); Yr = dat[pref][ys].to_numpy(float)
        Xo = dat[pobs][xs].to_numpy(float); Yo = dat[pobs][ys].to_numpy(float)
        out = []
        for i in range(len(Xo)):
            th = _dea_one(Xr, Yr, Xo[i], Yo[i], "crs", orient)[0]
            out.append(th if orient == "input" else 1 / th)   # distance-function value; <=1 when inside the technology
        return np.array(out)
    rows = []; yr_rows = []; unit_acc = {}
    cache = {}
    for a, b in zip(periods[:-1], periods[1:]):
        Dtt = cache.get((a, a)) if (a, a) in cache else D(a, a); cache[(a, a)] = Dtt
        Dbb = D(b, b); cache[(b, b)] = Dbb
        Dab = D(a, b)       # period b observation against period a technology
        Dba = D(b, a)
        ec = Dbb / Dtt
        tc = np.sqrt((Dab / Dbb) * (Dtt / Dba))
        m = ec * tc
        if orient == "input":
            pass
        # input orientation: distance D = theta (<=1 efficient=1); Malmquist = sqrt(D_a(b)/D_a(a) * D_b(b)/D_b(a)); >1 = productivity growth for output orientation.
        # for input orientation productivity growth corresponds to a ratio below 1 of input distance; use inverse so that >1 always means growth
        if orient == "input":
            ec = Dbb / Dtt
            tc = np.sqrt((Dab / Dbb) * (Dtt / Dba))
            m = np.sqrt(Dab / Dtt * Dbb / Dba)
        g = lambda v: float(np.exp(np.mean(np.log(v))))
        yr_rows.append([f"{a} to {b}", g(m), g(ec), g(tc), float((m > 1).mean())])
        for i, u in enumerate(ids):
            unit_acc.setdefault(u, []).append((m[i], ec[i], tc[i]))
    gm = lambda v: float(np.exp(np.mean(np.log(v))))
    allm = np.array([x for v in unit_acc.values() for x in v])
    tabs = [table("Malmquist productivity index by period (geometric mean over units)", ["Period", "TFP change", "Efficiency change (catch-up)", "Technical change (frontier shift)", "Share of units with growth"], yr_rows,
                  "Values above 1 mean improvement. TFP change = efficiency change x technical change (Fare et al. 1994); constant-returns frontiers per period."),
            table("Cumulative change over the whole sample", ["Component", "Geometric mean per period", "Cumulative"],
                  [["TFP", gm(allm[:, 0]), float(np.prod([r[1] for r in yr_rows]))], ["Efficiency change", gm(allm[:, 1]), float(np.prod([r[2] for r in yr_rows]))],
                   ["Technical change", gm(allm[:, 2]), float(np.prod([r[3] for r in yr_rows]))]], f"{len(periods) - 1} adjacent period pairs, {len(ids)} units.")]
    ub = sorted(((u, gm(np.array(v)[:, 0]), gm(np.array(v)[:, 1]), gm(np.array(v)[:, 2])) for u, v in unit_acc.items()), key=lambda r: -r[1])
    tabs.append(table("Units with the fastest and slowest TFP growth", ["Unit", "TFP change", "Efficiency change", "Technical change"], [list(r) for r in ub[:5] + ub[-5:]], "Geometric mean per period."))
    ch = [chart("Malmquist components by period", [r[0] for r in yr_rows], [("TFP", [r[1] for r in yr_rows]), ("Efficiency change", [r[2] for r in yr_rows]), ("Technical change", [r[3] for r in yr_rows])], "Period", "Index (1 = no change)", "line")]
    return result("malmquist", "Malmquist productivity index", len(d), [stat("Units", len(ids)), stat("Periods", len(periods)), stat("Mean TFP change", gm(allm[:, 0]), 4),
                                                                       stat("Mean efficiency change", gm(allm[:, 1]), 4), stat("Mean technical change", gm(allm[:, 2]), 4)], tabs, ch,
                  ["Caves, Christensen and Diewert (1982); Fare, Grosskopf, Norris and Zhang (1994). Computed with constant-returns DEA frontiers for each period; only units observed in every period are used.",
                   "Like DEA scores, the index has no sampling error: it describes this sample and is sensitive to outliers."])


# ================================================================================================ growth regression
def _ols(y, X, cluster=None):
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    meat = (X * e[:, None]).T @ (X * e[:, None]) * n / (n - k)
    V = XtXi @ meat @ XtXi
    return b, V, e


def growth(df, spec):
    y0 = need(spec, "y0", "initial income column")
    y1 = need(spec, "y1", "final income column")
    s_ = need(spec, "invest", "investment rate column (percent of GDP)")
    n_ = need(spec, "popgrowth", "population growth column (percent a year)")
    h_ = spec.get("school") or None
    grp = spec.get("sample") or None
    T = float_option(spec, "years", 25.0, "years between the two income observations")
    gd = float_option(spec, "gd", 0.05, "g + delta (default 0.05)")
    cols_ = [y0, y1, s_, n_] + ([h_] if h_ else [])
    d = numeric(df, cols_, keep=[grp] if grp else [])
    d = d[(d[[y0, y1, s_]] > 0).all(axis=1)]
    if grp:
        keep = str(spec.get("sample_value", "")).strip()
        if keep:
            d = d[d[grp].astype(str) == keep]
    n = len(d)
    if n < 25:
        raise OESError(f"Only {n} countries left: at least 25 are needed.")
    ly1 = np.log(d[y1].to_numpy(float)); ly0 = np.log(d[y0].to_numpy(float))
    ls = np.log(d[s_].to_numpy(float) / 100); lnd = np.log(d[n_].to_numpy(float) / 100 + gd)
    lh = np.log(d[h_].to_numpy(float) / 100) if h_ else None
    one = np.ones(n)
    tabs = []
    # 1 Solow levels
    cols_s = [one, ls, lnd] + ([lh] if h_ else [])
    names = ["Constant", "ln(saving rate)", "ln(n + g + delta)"] + (["ln(human capital)"] if h_ else [])
    Xs = np.column_stack(cols_s)
    b, V, e = _ols(ly1, Xs)
    se = np.sqrt(np.diag(V))
    r2 = 1 - e @ e / np.sum((ly1 - ly1.mean()) ** 2)
    tabs.append(table("Income per worker in levels (Mankiw-Romer-Weil)", ["Variable", "Coefficient", "Robust s.e.", "t", "p"], [[names[i], float(b[i]), float(se[i]), float(b[i] / se[i]), float(2 * stats.t.sf(abs(b[i] / se[i]), n - len(b)))] for i in range(len(b))],
                      f"Dependent variable: ln {y1}; {n} countries; R-squared {r2:.3f}."))
    # restricted: coefficient on ls = -coef on lnd
    Xr_ = np.column_stack([one, ls - lnd] + ([lh] if h_ else []))
    br, Vr, er = _ols(ly1, Xr_)
    rss_u = e @ e; rss_r = er @ er
    q = 1; Fst = ((rss_r - rss_u) / q) / (rss_u / (n - len(b)))
    # implied capital share
    if h_:
        a_ = br[1] / (1 + br[1] + br[2]) if False else None
    sums = b[1] + b[2]
    alpha = b[1] / (1 + b[1]) if True else None
    rows = [test_row("Solow restriction: coefficients on saving and n + g + delta sum to zero (F)", float(Fst), float(stats.f.sf(Fst, q, n - len(b))), f"unrestricted sum = {sums:.3f}"),
            ]
    if h_:
        rows.append(test_row("Implied capital share alpha (restricted, augmented model)", float(br[1] / (1 + br[1] + br[2])), None, "alpha = c_s / (1 + c_s + c_h)"))
        rows.append(test_row("Implied human capital share beta (restricted, augmented model)", float(br[2] / (1 + br[1] + br[2])), None, "beta = c_h / (1 + c_s + c_h)"))
    else:
        rows.append(test_row("Implied capital share alpha (restricted, textbook model)", float(br[1] / (1 + br[1])), None, "alpha = c / (1 + c) for the coefficient c on ln(s) - ln(n+g+delta)"))
    tabs.append(tests_table(rows, "Tests and implied parameters"))
    tabs.append(table("Restricted regression", ["Variable", "Coefficient", "Robust s.e."], [["Constant", float(br[0]), float(np.sqrt(Vr[0, 0]))], ["ln(s) - ln(n+g+delta)", float(br[1]), float(np.sqrt(Vr[1, 1]))]] +
                      ([["ln(human capital)", float(br[2]), float(np.sqrt(Vr[2, 2]))]] if h_ else []), f"R-squared {1 - er @ er / np.sum((ly1 - ly1.mean()) ** 2):.3f}."))
    # 2 convergence
    gr = (ly1 - ly0)
    Xc = np.column_stack([one, ly0]); namec = ["Constant", f"ln {y0}"]
    bc, Vc, ec = _ols(gr, Xc); sec = np.sqrt(np.diag(Vc))
    lam = -np.log(1 + bc[1] * 1.0) / T if 1 + bc[1] > 0 else np.nan
    Xc2 = np.column_stack([one, ly0, ls, lnd] + ([lh] if h_ else [])); namec2 = ["Constant", f"ln {y0}", "ln(saving rate)", "ln(n + g + delta)"] + (["ln(human capital)"] if h_ else [])
    bc2, Vc2, ec2 = _ols(gr, Xc2); sec2 = np.sqrt(np.diag(Vc2))
    lam2 = -np.log(1 + bc2[1]) / T if 1 + bc2[1] > 0 else np.nan
    for ttl, nm, bb, ss, ee, lm in (("Unconditional convergence", namec, bc, sec, ec, lam), ("Conditional convergence", namec2, bc2, sec2, ec2, lam2)):
        tabs.append(table(ttl, ["Variable", "Coefficient", "Robust s.e.", "t", "p"], [[nm[i], float(bb[i]), float(ss[i]), float(bb[i] / ss[i]), float(2 * stats.t.sf(abs(bb[i] / ss[i]), n - len(bb)))] for i in range(len(bb))],
                          f"Dependent variable: ln({y1}/{y0}). Speed of convergence lambda = {lm:.4f} a year; half-life {np.log(2) / lm:.1f} years." if lm and lm > 0 else "Dependent variable: growth; the coefficient implies no convergence."))
    ch = [chart("Growth against initial income", [f"{v:.2f}" for v in np.sort(ly0)], [("Growth, ln(y1/y0)", gr[np.argsort(ly0)]), ("Fitted, unconditional", (Xc @ bc)[np.argsort(ly0)])], f"ln {y0}", "Growth", "line")]
    return result("growth", "Growth regressions (Solow and convergence)", n, [stat("Countries", n), stat("Capital share (restricted)", float(br[1] / (1 + br[1] + (br[2] if h_ else 0))), 3), stat("Unconditional beta", float(bc[1]), 4),
                                                                        stat("Conditional beta", float(bc2[1]), 4), stat("Conditional speed lambda", float(lam2), 4)], tabs, ch,
                  ["Mankiw, Romer and Weil (1992): the Solow model implies income depends on ln(s) and ln(n+g+delta) with equal and opposite coefficients; adding human capital tests the augmented model. g + delta is fixed at 0.05 by convention.",
                   "Convergence: Barro and Sala-i-Martin; the speed is lambda = -ln(1 + beta)/T. Cross-country regressions identify correlations, not causal effects, and the variables are endogenous."])


# ================================================================================================ production / cost function
def prodfun(df, spec):
    yv = need(spec, "y", "output (or total cost for a cost function)")
    xs = as_list(need(spec, "x", "inputs (or input prices)"))
    kind = str(spec.get("kind", "production"))
    form = str(spec.get("form", "cobb-douglas"))
    norm = spec.get("normalise") or None           # price used to impose homogeneity in a cost function
    distinct([yv], xs)
    out = spec.get("output") or None                 # output quantity for a cost function
    d = numeric(df, [yv] + xs + ([out] if out else []))
    if (d <= 0).any().any():
        raise OESError("All variables must be positive: they enter in logarithms.")
    n = len(d)
    if kind == "cost" and not out:
        raise OESError("A cost function needs the output column (setting 'output').")
    L = np.log(d)
    one = np.ones(n)
    if kind == "production":
        Z = [L[c].to_numpy() for c in xs]; zn = xs
        dep = L[yv].to_numpy(); depn = f"ln {yv}"
    else:
        Z = [L[out].to_numpy()] + [L[c].to_numpy() for c in xs]; zn = [out] + xs
        dep = L[yv].to_numpy(); depn = f"ln {yv}"
        if norm:
            if norm not in xs:
                raise OESError(f"normalise = {norm} must be one of the price columns.")
            dep = dep - L[norm].to_numpy()
            depn = f"ln({yv}/{norm})"
            keep = [i for i, c in enumerate(zn) if c != norm]
            Z = [Z[i] - (L[norm].to_numpy() if zn[i] != out else 0) for i in keep]
            zn = [zn[i] for i in keep]
    mean = [z.mean() for z in Z]
    Zc = [z - m for z, m in zip(Z, mean)]               # centre so first-order terms are elasticities at the means
    cols = [one] + Zc
    names = ["Constant"] + [f"ln {c}" + (f" (price/{norm})" if (norm and c != out) else "") for c in zn]
    if form == "translog":
        k0 = len(Zc)
        for i in range(k0):
            for j in range(i, k0):
                cols.append(Zc[i] * Zc[j] * (0.5 if i == j else 1.0))
                names.append(f"1/2 ln {zn[i]}^2" if i == j else f"ln {zn[i]} x ln {zn[j]}")
    X = np.column_stack(cols)
    if n < X.shape[1] + 15:
        raise OESError(f"Too few observations ({n}) for {X.shape[1]} parameters.")
    b, V, e = _ols(dep, X)
    se = np.sqrt(np.diag(V))
    r2 = 1 - e @ e / np.sum((dep - dep.mean()) ** 2)
    kk = len(b)
    tabs = [table(f"{'Cobb-Douglas' if form != 'translog' else 'Translog'} {kind} function", ["Term", "Coefficient", "Robust s.e.", "t", "p"],
                  [[names[i], float(b[i]), float(se[i]), float(b[i] / se[i]), float(2 * stats.t.sf(abs(b[i] / se[i]), n - kk))] for i in range(kk)],
                  f"Dependent variable: {depn}; {n} observations; R-squared {r2:.3f}. Variables are centred at their means, so first-order terms are elasticities at the sample means.")]
    rows = []
    if kind == "production":
        idx = list(range(1, 1 + len(xs)))
        rts = float(b[idx].sum()); w = np.zeros(kk); w[idx] = 1
        var = float(w @ V @ w); wald = (rts - 1) ** 2 / var
        rows.append(test_row("Returns to scale (sum of input elasticities at the means)", rts, float(stats.chi2.sf(wald, 1)), "p tests constant returns (H0: sum = 1)"))
        for i in idx:
            rows.append(test_row(f"Elasticity of {names[i][3:]}", float(b[i]), float(2 * stats.t.sf(abs(b[i] / se[i]), n - kk)), ""))
    else:
        be = float(b[1])
        rows.append(test_row(f"Cost elasticity of output at the means ({out})", be, float(2 * stats.t.sf(abs((be - 1) / se[1]), n - kk)), "p tests the elasticity equal to 1 (constant returns to scale)"))
        rows.append(test_row("Returns to scale = 1 / cost elasticity", float(1 / be), None, "above 1 means economies of scale"))
        if not norm:
            pidx = list(range(2, 2 + len(xs)))
            s_ = float(b[pidx].sum()); w = np.zeros(kk); w[pidx] = 1
            wald = (s_ - 1) ** 2 / float(w @ V @ w)
            rows.append(test_row("Homogeneity of degree one in input prices (sum of price elasticities = 1)", s_, float(stats.chi2.sf(wald, 1)), "theory requires 1"))
    if form == "translog":
        k0 = len(Zc)
        nq = k0 * (k0 + 1) // 2
        R = np.zeros((nq, kk)); R[np.arange(nq), np.arange(1 + k0, 1 + k0 + nq)] = 1
        W = (R @ b) @ np.linalg.solve(R @ V @ R.T, R @ b)
        rows.append(test_row("Cobb-Douglas against translog (all second-order terms zero, Wald)", float(W), float(stats.chi2.sf(W, nq)), f"{nq} restrictions"))
    tabs.append(tests_table(rows, "Elasticities and tests"))
    ch = []
    if kind == "production" or out:
        key = xs[0] if kind == "production" else out
        ii = names.index(f"ln {key}") if f"ln {key}" in names else 1
        order = np.argsort(Zc[ii - 1])
        ch.append(chart("Fitted against observed", [f"{v:.2f}" for v in dep[order]], [("Observed", dep[order]), ("Fitted", (X @ b)[order])], depn, "", "line")) if False else None
    ch = [c for c in ch if c]
    head = [stat("Observations", n), stat("R-squared", float(r2), 3)]
    if kind == "production":
        head.append(stat("Returns to scale", float(b[1:1 + len(xs)].sum()), 3))
    else:
        head.append(stat("Returns to scale", float(1 / b[1]), 3))
    return result("prodfun", "Production and cost functions", n, head, tabs, ch,
                  ["Cobb-Douglas (constant elasticities) or translog (Christensen, Jorgenson and Lau 1973). Estimated by OLS with robust standard errors; inputs are treated as exogenous, which fails if they are chosen with knowledge of productivity (Marschak and Andrews 1944).",
                   "Cost function: Nerlove (1963) imposes homogeneity of degree one in input prices by dividing cost and prices by one price (option 'normalise')."])


# ================================================================================================ spatial
def _knn_w(coords, k):
    n = len(coords)
    D = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    W = np.zeros((n, n))
    nn = np.argsort(D, axis=1)[:, :k]
    for i in range(n):
        W[i, nn[i]] = 1.0
    return W


def _sp_fit(y, X, W, kind):
    n, k = X.shape
    Wsym = (W + W.T) / 2
    ev = np.linalg.eigvals(W).real if False else np.linalg.eigvals(W)
    ev = ev.real
    lo, hi = 1 / ev.min() + 1e-6, 1 / ev.max() - 1e-6
    lo = max(lo, -0.999)
    I = np.eye(n)
    if kind == "sar":
        def conc(rho):
            ys = y - rho * (W @ y)
            b = np.linalg.lstsq(X, ys, rcond=None)[0]
            e = ys - X @ b
            return -(-n / 2 * np.log(e @ e / n) + np.sum(np.log(1 - rho * ev)))
    else:
        def conc(lam):
            ys = y - lam * (W @ y); Xs = X - lam * (W @ X)
            b = np.linalg.lstsq(Xs, ys, rcond=None)[0]
            e = ys - Xs @ b
            return -(-n / 2 * np.log(e @ e / n) + np.sum(np.log(1 - lam * ev)))
    o = optimize.minimize_scalar(conc, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-8))
    r = float(o.x)
    if kind == "sar":
        ys = y - r * (W @ y); b = np.linalg.lstsq(X, ys, rcond=None)[0]; e = ys - X @ b
    else:
        ys = y - r * (W @ y); Xs = X - r * (W @ X); b = np.linalg.lstsq(Xs, ys, rcond=None)[0]; e = ys - Xs @ b
    s2 = float(e @ e / n)
    ll = float(-conc(r) - n / 2 * np.log(2 * np.pi) - n / 2)
    # full-information variance: numerical Hessian of the log-likelihood over (b, rho, s2)
    def nll(p):
        bb, rr, ls2 = p[:k], p[k], p[k + 1]
        if kind == "sar":
            u = y - rr * (W @ y) - X @ bb
        else:
            u = (y - rr * (W @ y)) - (X - rr * (W @ X)) @ bb
        return -(-n / 2 * np.log(2 * np.pi) - n / 2 * ls2 - 0.5 * (u @ u) / np.exp(ls2) + np.sum(np.log(np.clip(1 - rr * ev, 1e-12, None))))
    p0 = np.concatenate([b, [r, np.log(s2)]])
    H = _num_hessian(nll, p0, eps=1e-4)
    try:
        Vp = np.linalg.inv(H)
    except Exception:
        Vp = np.full_like(H, np.nan)
    return b, r, s2, ll, np.sqrt(np.clip(np.diag(Vp), 0, None)), e


def spatial(df, spec):
    yv = need(spec, "y", "dependent variable")
    xs = as_list(need(spec, "x", "regressors"))
    cx = need(spec, "x_coord", "first coordinate column (x or longitude)")
    cy = need(spec, "y_coord", "second coordinate column (y or latitude)")
    kn = int_option(spec, "neighbours", 5, 1, 30, "number of nearest neighbours")
    distinct([yv], xs)
    d = numeric(df, [yv] + xs + [cx, cy])
    n = len(d)
    if n < 25:
        raise OESError(f"Only {n} areas: at least 25 are needed for spatial models.")
    if kn >= n:
        raise OESError("Number of neighbours must be below the number of areas.")
    y = d[yv].to_numpy(float); X = np.column_stack([np.ones(n), d[xs].to_numpy(float)])
    coords = d[[cx, cy]].to_numpy(float)
    Wb = _knn_w(coords, kn)
    W = Wb / Wb.sum(1, keepdims=True)
    S0 = Wb.sum()
    names = ["Constant"] + xs
    k = X.shape[1]
    b0, V0, e0 = _ols(y, X)
    # OLS classical for tests
    s2 = e0 @ e0 / n
    # Moran
    I = (n / W.sum()) * (e0 @ W @ e0) / (e0 @ e0)
    M = np.eye(n) - X @ np.linalg.solve(X.T @ X, X.T)
    T1 = np.trace(M @ W); WtW = np.trace(W.T @ W + W @ W)
    EI = (n / W.sum()) * T1 / (n - k)
    VI_num = np.trace(M @ W @ M @ W.T) + np.trace(M @ W @ M @ W) + T1 ** 2
    VI = (n / W.sum()) ** 2 * (VI_num / ((n - k) * (n - k + 2)) - (T1 / (n - k)) ** 2)
    zI = (I - EI) / np.sqrt(VI)
    # LM tests (Anselin 1988)
    T = np.trace(W.T @ W + W @ W)
    lm_e = (e0 @ W @ e0 / s2) ** 2 / T
    WXb = W @ X @ b0
    J = (WXb @ M @ WXb / s2 + T) 
    lm_l = (e0 @ W @ y / s2) ** 2 / J
    rlm_e = (e0 @ W @ e0 / s2 - T * (e0 @ W @ y / s2) / J) ** 2 / (T * (1 - T / J))
    rlm_l = (e0 @ W @ y / s2 - e0 @ W @ e0 / s2) ** 2 / (J - T)
    tests = [test_row("Moran's I of the OLS residuals", float(I), float(2 * stats.norm.sf(abs(zI))), f"z = {zI:.2f}; expected under no autocorrelation {EI:.3f}"),
             test_row("LM test, spatial error", float(lm_e), float(stats.chi2.sf(lm_e, 1)), "H0: no spatial error dependence"),
             test_row("LM test, spatial lag", float(lm_l), float(stats.chi2.sf(lm_l, 1)), "H0: no spatially lagged dependent variable"),
             test_row("Robust LM test, spatial error", float(rlm_e), float(stats.chi2.sf(rlm_e, 1)), "valid with a spatial lag present"),
             test_row("Robust LM test, spatial lag", float(rlm_l), float(stats.chi2.sf(rlm_l, 1)), "valid with spatial error present")]
    tabs = [table("OLS", ["Variable", "Coefficient", "Robust s.e.", "t", "p"], [[names[i], float(b0[i]), float(np.sqrt(V0[i, i])), float(b0[i] / np.sqrt(V0[i, i])), float(2 * stats.t.sf(abs(b0[i] / np.sqrt(V0[i, i])), n - k))] for i in range(k)], f"{n} areas; weights: {kn} nearest neighbours, row-standardised."),
            tests_table(tests, "Spatial dependence tests on the OLS residuals")]
    res = {}
    for kind, lab, pn in (("sar", "Spatial lag model (SAR)", "rho"), ("sem", "Spatial error model (SEM)", "lambda")):
        b, r, s2_, ll, se, e = _sp_fit(y, X, W, kind)
        rows = [[names[i], float(b[i]), float(se[i]), float(b[i] / se[i]), float(2 * stats.norm.sf(abs(b[i] / se[i])))] for i in range(k)]
        rows.append([pn, r, float(se[k]), float(r / se[k]), float(2 * stats.norm.sf(abs(r / se[k])))])
        tabs.append(table(lab, ["Variable", "Coefficient", "s.e.", "z", "p"], rows, f"Maximum likelihood; log-likelihood {ll:.2f}; sigma^2 {s2_:.3f}."))
        res[kind] = (b, r, ll, se)
    ll0 = float(-n / 2 * (np.log(2 * np.pi * (e0 @ e0 / n)) + 1))
    cmp = [["OLS", ll0, k + 1, float(2 * (k + 1) - 2 * ll0), float(np.log(n) * (k + 1) - 2 * ll0)]]
    for kind, lab in (("sar", "SAR"), ("sem", "SEM")):
        ll = res[kind][2]
        cmp.append([lab, ll, k + 2, float(2 * (k + 2) - 2 * ll), float(np.log(n) * (k + 2) - 2 * ll)])
    tabs.append(table("Model comparison", ["Model", "Log-likelihood", "Parameters", "AIC", "BIC"], cmp, "Lower AIC and BIC are better."))
    # SAR impacts
    b, r, ll, se = res["sar"]
    A = np.linalg.inv(np.eye(n) - r * W)
    imp = []
    for j in range(1, k):
        S = A * b[j]
        direct = float(np.trace(S) / n); total = float(S.sum() / n)
        imp.append([names[j], direct, total - direct, total])
    tabs.append(table("SAR direct, indirect and total effects", ["Variable", "Direct", "Indirect (spillover)", "Total"], imp,
                      "LeSage and Pace (2009): with a spatial lag the coefficient is not the marginal effect; effects are averages of the matrix (I - rho W)^-1 b."))
    ch = [chart("OLS residuals against their spatial lag", [f"{v:.1f}" for v in np.sort(e0)], [("Spatial lag of the residuals", (W @ e0)[np.argsort(e0)])], "Residual", "W x residual", "line")]
    return result("spatial_cs", "Spatial regression (SAR and SEM)", n, [stat("Areas", n), stat("Moran's I of residuals", float(I), 3), stat("SAR rho", float(res["sar"][1]), 3), stat("SEM lambda", float(res["sem"][1]), 3),
                                                                   stat("Best model by AIC", min(cmp, key=lambda r: r[3])[0])], tabs, ch,
                  ["Anselin (1988). Weights are k nearest neighbours by Euclidean distance, row-standardised; the results depend on that choice. Maximum likelihood with the Jacobian from the eigenvalues of W; standard errors from the numerical information matrix.",
                   "The robust LM tests guide the choice between lag and error models (Anselin et al. 1996)."])
