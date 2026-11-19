"""Category E of the Statistics Division: mixed-effects and longitudinal methods.

Linear mixed models (REML or ML, random intercept and an optional random slope, Satterthwaite degrees of freedom), generalised
linear mixed models (binomial and Poisson with a random intercept, adaptive Gauss-Hermite quadrature), generalised estimating
equations (independence, exchangeable and AR(1) working correlation, robust sandwich errors, QIC) and intraclass correlation
coefficients (the six Shrout-Fleiss forms with confidence intervals, plus Cronbach's alpha). Only numpy and scipy are used."""
import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .core import OESError, as_list, chart, distinct, f, float_option, int_option, need, result, stat, table, tests_table
from .stat_core import clean_frame, dummy_cols, levels, normality_tests, thin


def _alpha(spec):
    a = float_option(spec, "alpha", 0.05, "Significance level")
    if not 0 < a < 0.5:
        raise OESError("The significance level must be between 0 and 0.5.")
    return a


def _lab(alpha):
    return f"{int(round((1 - alpha) * 100))}% CI"


def _fixed_design(d, covs, facs):
    cols, names = [np.ones(len(d))], ["(Intercept)"]
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
    X = np.column_stack(cols)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise OESError("The fixed-effect covariates are perfectly collinear.")
    return X, names


# ---------------------------------------------------------------------------------------------------------------
# Linear mixed model
# ---------------------------------------------------------------------------------------------------------------
class _LMM:
    """y = X b + Z u + e with u_i ~ N(0, s^2 D(theta)), e ~ N(0, s^2 I); D = L L' with L lower triangular (relative factor)."""

    def __init__(self, y, X, Z, groups):
        self.N, self.p = X.shape
        self.q = Z.shape[1]
        self.idx = [np.flatnonzero(groups == g) for g in pd.unique(groups)]
        self.G = len(self.idx)
        self.XtX = X.T @ X
        self.Xty = X.T @ y
        self.yty = float(y @ y)
        self.ZtZ = np.array([Z[i].T @ Z[i] for i in self.idx])
        self.ZtX = np.array([Z[i].T @ X[i] for i in self.idx])
        self.Zty = np.array([Z[i].T @ y[i] for i in self.idx])
        self.y, self.X, self.Z, self.groups = y, X, Z, groups

    def lam(self, theta):
        q = self.q
        L = np.zeros((q, q))
        L[np.tril_indices(q)] = theta
        return L

    def parts(self, theta):
        """Sums that define the profiled criteria for a given relative factor."""
        L = self.lam(theta)
        D = L @ L.T
        q = self.q
        M = np.eye(q)[None] + self.ZtZ @ D[None]
        K = D[None] @ np.linalg.inv(M)
        K = (K + np.transpose(K, (0, 2, 1))) / 2
        logdet = float(np.sum(np.linalg.slogdet(M)[1]))
        Axx = self.XtX - np.einsum("gqi,gqr,grj->ij", self.ZtX, K, self.ZtX)
        Axy = self.Xty - np.einsum("gqi,gqr,gr->i", self.ZtX, K, self.Zty)
        ayy = self.yty - float(np.einsum("gq,gqr,gr->", self.Zty, K, self.Zty))
        return Axx, Axy, ayy, logdet

    def profile(self, theta, reml):
        Axx, Axy, ayy, logdet = self.parts(theta)
        try:
            beta = np.linalg.solve(Axx, Axy)
        except np.linalg.LinAlgError:
            return 1e30, None
        r = ayy - Axy @ beta
        if r <= 0:
            return 1e30, None
        N, p = self.N, self.p
        if reml:
            dev = (N - p) * (1 + np.log(2 * np.pi * r / (N - p))) + logdet + np.linalg.slogdet(Axx)[1]
        else:
            dev = N * (1 + np.log(2 * np.pi * r / N)) + logdet
        return dev, dict(beta=beta, r=r, Axx=Axx)

    def full_dev(self, phi, reml):
        """-2 log likelihood (up to a constant) as a function of (theta, log sigma)."""
        theta, ls = phi[:-1], phi[-1]
        Axx, Axy, ayy, logdet = self.parts(theta)
        beta = np.linalg.solve(Axx, Axy)
        r = ayy - Axy @ beta
        N, p = self.N, self.p
        s2 = np.exp(2 * ls)
        if reml:
            return (N - p) * np.log(s2) + logdet + np.linalg.slogdet(Axx)[1] + r / s2
        return N * np.log(s2) + logdet + r / s2


def _fit_lmm(y, X, Z, groups, reml):
    m = _LMM(y, X, Z, groups)
    q = m.q
    nt = q * (q + 1) // 2
    diag = np.array([i * (i + 1) // 2 + i for i in range(q)])
    lower = np.full(nt, -np.inf)
    lower[diag] = 0.0
    bounds = [(lower[k], None) for k in range(nt)]
    sd0 = max(float(np.std(y)), 1e-3)
    starts = []
    for sc in (0.5, 1.0):
        t0 = np.zeros(nt)
        t0[diag] = sc
        starts.append(t0)
    best = None
    for t0 in starts:
        r = optimize.minimize(lambda th: m.profile(th, reml)[0], t0, method="L-BFGS-B", bounds=bounds, options=dict(ftol=1e-14, gtol=1e-9, maxiter=1000))
        r2 = optimize.minimize(lambda th: m.profile(th, reml)[0], r.x, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-10, maxiter=600))
        cand = r2 if r2.fun <= r.fun else r
        th = cand.x.copy()
        th[diag] = np.abs(th[diag])
        v = m.profile(th, reml)[0]
        if best is None or v < best[1] - 1e-10:
            best = (th, v)
    theta = best[0]
    dev, info = m.profile(theta, reml)
    if info is None:
        raise OESError("The mixed model could not be fitted (the random-effects structure is not supported by the data).")
    N, p = m.N, m.p
    s2 = info["r"] / ((N - p) if reml else N)
    return m, theta, s2, info, dev


def _satterthwaite(m, theta, s2, reml, L):
    """Satterthwaite degrees of freedom for the contrast rows of L (numerical derivatives in the variance parameters)."""
    phi0 = np.r_[theta, 0.5 * np.log(s2)]
    k = len(phi0)

    def varb(phi):
        Axx = m.parts(phi[:-1])[0]
        return np.exp(2 * phi[-1]) * np.einsum("ij,jk,ik->i", L, np.linalg.inv(Axx), L)

    H = np.zeros((k, k))
    hs = 1e-3 * np.maximum(1.0, np.abs(phi0))
    for a in range(k):
        for b in range(a, k):
            def fv(sa, sb):
                ph = phi0.copy()
                ph[a] += sa
                ph[b] += sb
                return m.full_dev(ph, reml)
            if a == b:
                H[a, a] = (fv(hs[a], 0) - 2 * fv(0, 0) + fv(-hs[a], 0)) / hs[a] ** 2
            else:
                H[a, b] = H[b, a] = (fv(hs[a], hs[b]) - fv(hs[a], -hs[b]) - fv(-hs[a], hs[b]) + fv(-hs[a], -hs[b])) / (4 * hs[a] * hs[b])
    g = np.zeros((L.shape[0], k))
    for a in range(k):
        pp, pm = phi0.copy(), phi0.copy()
        pp[a] += hs[a]
        pm[a] -= hs[a]
        g[:, a] = (varb(pp) - varb(pm)) / (2 * hs[a])
    v0 = varb(phi0)
    try:
        A = 2 * np.linalg.inv(H)
        den = np.einsum("ij,jk,ik->i", g, A, g)
        df = 2 * v0 ** 2 / den
    except np.linalg.LinAlgError:
        return None
    df = np.where((df > 0) & np.isfinite(df), df, np.nan)
    return df


def lmm(df, spec):
    alpha = _alpha(spec)
    ycol, gcol = need(spec, "y", "outcome"), need(spec, "group", "grouping (cluster) variable")
    covs, facs = as_list(spec.get("covariates")), as_list(spec.get("factors"))
    slope = spec.get("slope") or None
    method = spec.get("method") or "reml"
    corr = (spec.get("correlated") or "yes") == "yes"
    distinct(ycol, gcol, covs, facs)
    if slope and slope in (ycol, gcol):
        raise OESError("The random-slope variable must differ from the outcome and the grouping variable.")
    if slope and slope in facs:
        raise OESError("The random-slope variable must be numeric.")
    d, dropped = clean_frame(df, numeric=list(dict.fromkeys([ycol] + covs + ([slope] if slope else []))), factors=facs + [gcol])
    X, names = _fixed_design(d, covs, facs)
    y = d[ycol].to_numpy(float)
    groups = d[gcol].to_numpy()
    ng = len(pd.unique(groups))
    N, p = X.shape
    if ng < 2:
        raise OESError("There is only one group.")
    if N <= p + 1:
        raise OESError("There are not enough observations for the fixed effects.")
    Z = np.ones((N, 1))
    if slope:
        sv = d[slope].to_numpy(float)
        if np.ptp(sv) == 0:
            raise OESError("The random-slope variable is constant.")
        Z = np.column_stack([np.ones(N), sv])
        if N <= 2 * ng:
            raise OESError("A random slope needs more observations than twice the number of groups.")
    reml = method == "reml"
    m, theta, s2, info, dev = _fit_lmm(y, X, Z, groups, reml)
    if slope and not corr:
        # re-fit with the intercept-slope covariance fixed at zero (diagonal relative factor)
        m2 = _LMM(y, X, Z, groups)
        wrap = lambda th2: m2.profile(np.array([abs(th2[0]), 0.0, abs(th2[1])]), reml)[0]
        r = optimize.minimize(wrap, np.array([theta[0], theta[2]]), method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-10, maxiter=600))
        theta = np.array([abs(r.x[0]), 0.0, abs(r.x[1])])
        dev, info = m2.profile(theta, reml)
        s2 = info["r"] / ((N - p) if reml else N)
        m = m2
    beta = info["beta"]
    cov_b = s2 * np.linalg.inv(info["Axx"])
    se = np.sqrt(np.diag(cov_b))
    L = np.eye(p)
    dfs = _satterthwaite(m, theta, s2, reml, L)
    if dfs is None:
        dfs = np.full(p, np.nan)
    tv = beta / se
    pv = np.array([2 * stats.t.sf(abs(tv[j]), dfs[j]) if np.isfinite(dfs[j]) else 2 * stats.norm.sf(abs(tv[j])) for j in range(p)])
    crit = np.array([stats.t.ppf(1 - alpha / 2, dfs[j]) if np.isfinite(dfs[j]) else stats.norm.ppf(1 - alpha / 2) for j in range(p)])
    lab = _lab(alpha)
    fx = [[names[j], f(beta[j], 5), f(se[j], 5), f(dfs[j], 2), f(tv[j], 4), f(pv[j], 6), f(beta[j] - crit[j] * se[j], 5), f(beta[j] + crit[j] * se[j], 5)] for j in range(p)]
    Lm = m.lam(theta)
    G = s2 * (Lm @ Lm.T)
    q = Z.shape[1]
    vc = [[f"Group ({gcol}): intercept", f(G[0, 0], 5), f(np.sqrt(G[0, 0]), 5), None]]
    if slope:
        vc.append([f"Group ({gcol}): slope of {slope}", f(G[1, 1], 5), f(np.sqrt(G[1, 1]), 5), f(G[0, 1] / np.sqrt(G[0, 0] * G[1, 1]), 4) if G[0, 0] > 0 and G[1, 1] > 0 else None])
    vc.append(["Residual", f(s2, 5), f(np.sqrt(s2), 5), None])
    ll_ml = None
    # random effects (BLUPs) and residuals
    D = Lm @ Lm.T
    blup = []
    fitted_fixed = X @ beta
    cond_fit = fitted_fixed.copy()
    for ii, i in enumerate(m.idx):
        ri = y[i] - fitted_fixed[i]
        Zi = Z[i]
        Vi = np.eye(len(i)) + Zi @ D @ Zi.T
        b = D @ Zi.T @ np.linalg.solve(Vi, ri)
        blup.append((groups[i][0], b))
        cond_fit[i] += Zi @ b
    resid = y - cond_fit
    # ICC and R-squared (random intercept only)
    tabs = [table("Fixed effects", ["Term", "Estimate", "SE", "df (Satterthwaite)", "t", "p-value", f"{lab} low", "CI high"], fx,
                  "Degrees of freedom are Satterthwaite approximations computed from the curvature of the likelihood in the variance parameters; they are not reported when that curvature is singular (boundary fits), where normal p-values are shown."),
            table("Random effects and residual variance", ["Component", "Variance", "SD", "Correlation with intercept"], vc)]
    summary = [stat("Observations", N), stat("Groups", ng), stat("REML criterion" if reml else "-2 log-likelihood", dev, 3)]
    fit_rows = [["Estimation", "REML" if reml else "Maximum likelihood"], ["Observations", N], ["Groups", ng], ["Observations per group (min / mean / max)", f"{min(len(i) for i in m.idx)} / {N / ng:.2f} / {max(len(i) for i in m.idx)}"]]
    if reml:
        fit_rows.append(["REML criterion (-2 restricted log-likelihood)", f(dev, 4)])
    else:
        k = p + q * (q + 1) // 2 + 1
        if slope and not corr:
            k -= 1
        ll_ml = -dev / 2
        fit_rows += [["Log-likelihood", f(ll_ml, 4)], ["AIC", f(-2 * ll_ml + 2 * k, 3)], ["BIC", f(-2 * ll_ml + k * np.log(N), 3)]]
    tabs.append(table("Model fit", ["Quantity", "Value"], fit_rows))
    notes = []
    if not slope:
        icc = G[0, 0] / (G[0, 0] + s2)
        summary.append(stat("ICC (intraclass correlation)", icc))
        var_f = float(np.var(fitted_fixed))
        r2m = var_f / (var_f + G[0, 0] + s2)
        r2c = (var_f + G[0, 0]) / (var_f + G[0, 0] + s2)
        tabs.append(table("Variance explained", ["Quantity", "Value"], [["ICC: share of variance between groups", f(icc, 4)], ["Marginal R-squared (fixed effects)", f(r2m, 4)], ["Conditional R-squared (fixed + random)", f(r2c, 4)]],
                          "Nakagawa-Schielzeth R-squared for a random-intercept model."))
    else:
        notes.append("With a random slope the between-group share of variance depends on the value of the slope variable, so no single intraclass correlation is reported.")
    if G[0, 0] < 1e-8 * max(s2, 1e-12) or (slope and G[1, 1] < 1e-8 * max(s2, 1e-12)):
        notes.append("A variance estimate is at the boundary (zero): the data do not support that random effect. Consider dropping it.")
    diag = normality_tests(resid)
    if diag:
        tabs.append(tests_table(diag, "Residual diagnostics"))
    bl_rows = sorted(([str(g), f(b[0], 4)] + ([f(b[1], 4)] if slope else []) for g, b in blup), key=lambda r: r[0])
    tabs.append(table("Predicted random effects (conditional modes)", ["Group", "Intercept"] + (["Slope"] if slope else []), bl_rows[:60],
                      "Deviations of each group from the average intercept (and slope)." + (" Only the first 60 groups are listed." if len(bl_rows) > 60 else "")))
    fx_, rs_ = thin(list(cond_fit), list(resid), 300)
    charts = [chart("Conditional residuals against fitted values", fx_, [("Residual", rs_)], "fitted value (including random effects)", "residual", "scatter")]
    order = sorted(blup, key=lambda t: t[1][0])
    charts.append(chart("Random intercepts by group", [str(g) for g, _ in order][:60], [("Intercept deviation", [b[0] for _, b in order][:60])], "group", "deviation", "bar"))
    if dropped:
        notes.append(f"{dropped} row(s) with missing values were dropped.")
    return result("stat_lmm", "Linear mixed-effects model", N, summary, tabs, charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# Generalised linear mixed model (random intercept, adaptive Gauss-Hermite quadrature)
# ---------------------------------------------------------------------------------------------------------------
def _glm_irls(X, y, n, fam, iters=100):
    p = X.shape[1]
    beta = np.zeros(p)
    if fam == "binomial":
        pr = (y.sum() + 0.5) / (n.sum() + 1.0)
        beta[0] = np.log(pr / (1 - pr))
    else:
        beta[0] = np.log(max(y.mean(), 1e-3))
    for _ in range(iters):
        eta = X @ beta
        if fam == "binomial":
            mu = 1 / (1 + np.exp(-eta))
            w = n * mu * (1 - mu)
            z = eta + (y - n * mu) / np.maximum(w, 1e-12)
        else:
            mu = np.exp(eta)
            w = mu
            z = eta + (y - mu) / np.maximum(mu, 1e-12)
        W = np.maximum(w, 1e-12)
        nb = np.linalg.solve((X * W[:, None]).T @ X, (X * W[:, None]).T @ z)
        if np.max(np.abs(nb - beta)) < 1e-10:
            beta = nb
            break
        beta = nb
    return beta


def _obs_ll(eta, y, n, fam):
    if fam == "binomial":
        return y * eta - n * np.logaddexp(0, eta)
    return y * eta - np.exp(np.minimum(eta, 700))


def _const(y, n, fam):
    if fam == "binomial":
        return float(np.sum(special.gammaln(n + 1) - special.gammaln(y + 1) - special.gammaln(n - y + 1)))
    return float(-np.sum(special.gammaln(y + 1)))


class _GLMM:
    def __init__(self, X, y, n, g, fam, nodes):
        self.X, self.y, self.n, self.fam = X, y, n, fam
        _, self.g = np.unique(g, return_inverse=True)
        self.G = int(self.g.max()) + 1
        self.zk, self.wk = np.polynomial.hermite.hermgauss(nodes) if nodes > 1 else (np.array([0.0]), np.array([np.sqrt(np.pi)]))
        self.nodes = nodes
        self.const = _const(y, n, fam)

    def _d(self, eta):
        """first and second derivatives of the observation log-likelihood with respect to eta"""
        if self.fam == "binomial":
            p = 1 / (1 + np.exp(-eta))
            return self.y - self.n * p, self.n * p * (1 - p)
        mu = np.exp(np.minimum(eta, 700))
        return self.y - mu, mu

    def loglik(self, theta, return_modes=False):
        p = self.X.shape[1]
        beta, s = theta[:p], np.exp(theta[p])
        xb = self.X @ beta
        u = np.zeros(self.G)
        for _ in range(60):
            eta = xb + s * u[self.g]
            sc, w = self._d(eta)
            h1 = s * np.bincount(self.g, sc, self.G) - u
            h2 = -(s * s) * np.bincount(self.g, w, self.G) - 1
            step = h1 / h2
            step = np.clip(step, -5, 5)
            u = u - step
            if np.max(np.abs(step)) < 1e-10:
                break
        eta = xb + s * u[self.g]
        sc, w = self._d(eta)
        c = (s * s) * np.bincount(self.g, w, self.G) + 1.0
        tot = np.zeros((self.nodes, self.G))
        for k in range(self.nodes):
            uk = u + np.sqrt(2.0) * self.zk[k] / np.sqrt(c)
            e = xb + s * uk[self.g]
            tot[k] = np.bincount(self.g, _obs_ll(e, self.y, self.n, self.fam), self.G) - uk ** 2 / 2
        a = np.log(self.wk)[:, None] + self.zk[:, None] ** 2 + tot
        mx = a.max(0)
        li = mx + np.log(np.exp(a - mx).sum(0)) + 0.5 * np.log(2.0 / c) - 0.5 * np.log(2 * np.pi)
        ll = float(li.sum() + self.const)
        if return_modes:
            return ll, u, c
        return ll


def glmm(df, spec):
    alpha = _alpha(spec)
    fam = spec.get("family") or "binomial"
    if fam not in ("binomial", "poisson"):
        raise OESError("Choose the binomial or Poisson family.")
    ycol, gcol = need(spec, "y", "outcome"), need(spec, "group", "grouping (cluster) variable")
    tcol = spec.get("trials") or None
    covs, facs = as_list(spec.get("covariates")), as_list(spec.get("factors"))
    nodes = int_option(spec, "nodes", 15, 1, 51, "Quadrature points")
    distinct(ycol, gcol, [tcol] if tcol else [], covs, facs)
    d, dropped = clean_frame(df, numeric=[ycol] + covs + ([tcol] if tcol else []), factors=facs + [gcol])
    y = d[ycol].to_numpy(float)
    if (y < 0).any() or not np.allclose(y, np.round(y)):
        raise OESError("The outcome must be a non-negative whole number (a count or a 0/1 indicator).")
    if fam == "binomial":
        n = d[tcol].to_numpy(float) if tcol else np.ones(len(d))
        if (n < 1).any() or not np.allclose(n, np.round(n)):
            raise OESError("The number of trials must be a whole number of at least 1.")
        if (y > n).any():
            raise OESError("The outcome exceeds the number of trials in some rows.")
        if y.sum() == 0 or y.sum() == n.sum():
            raise OESError("The outcome has no variation (all failures or all successes).")
    else:
        if tcol:
            raise OESError("The number of trials applies to the binomial family only.")
        n = np.ones(len(d))
    X, names = _fixed_design(d, covs, facs)
    N, p = X.shape
    ng = d[gcol].nunique()
    if ng < 5:
        raise OESError("At least five groups are needed to estimate a random-intercept variance.")
    beta0 = _glm_irls(X, y, n, fam)
    m = _GLMM(X, y, n, d[gcol].to_numpy(), fam, nodes)
    # GLM log-likelihood for the likelihood-ratio test of the random intercept
    ll_glm = float(np.sum(_obs_ll(X @ beta0, y, n, fam)) + m.const)
    negll = lambda th: -m.loglik(th)
    best = None
    for s0 in (-0.7, -1.5, 0.0):
        r = optimize.minimize(negll, np.r_[beta0, s0], method="BFGS", options=dict(gtol=1e-7, maxiter=500))
        r2 = optimize.minimize(negll, r.x, method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-11, maxiter=3000))
        c = r2 if r2.fun < r.fun else r
        if best is None or c.fun < best.fun:
            best = c
    th = best.x
    ll, u, cc = m.loglik(th, True)
    k = len(th)
    H = np.zeros((k, k))
    hs = 1e-3 * np.maximum(1.0, np.abs(th))
    f0 = m.loglik(th)
    for a in range(k):
        for b in range(a, k):
            def fv(sa, sb):
                t2 = th.copy()
                t2[a] += sa
                t2[b] += sb
                return m.loglik(t2)
            if a == b:
                H[a, a] = (fv(hs[a], 0) - 2 * f0 + fv(-hs[a], 0)) / hs[a] ** 2
            else:
                H[a, b] = H[b, a] = (fv(hs[a], hs[b]) - fv(hs[a], -hs[b]) - fv(-hs[a], hs[b]) + fv(-hs[a], -hs[b])) / (4 * hs[a] * hs[b])
    try:
        cov = np.linalg.inv(-H)
        if np.any(np.diag(cov) <= 0):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        raise OESError("The information matrix is singular; the model is not identified (the random-intercept variance may be zero).")
    beta, ls = th[:p], th[p]
    s = float(np.exp(ls))
    se = np.sqrt(np.diag(cov))
    zc = stats.norm.ppf(1 - alpha / 2)
    lab = _lab(alpha)
    eff = "OR" if fam == "binomial" else "IRR"
    fx = []
    for j in range(p):
        z = beta[j] / se[j]
        fx.append([names[j], f(beta[j], 5), f(se[j], 5), f(z, 4), f(2 * stats.norm.sf(abs(z)), 6), f(np.exp(beta[j]), 5), f(np.exp(beta[j] - zc * se[j]), 5), f(np.exp(beta[j] + zc * se[j]), 5)])
    lr = max(2 * (ll - ll_glm), 0.0)
    p_lr = 0.5 * float(stats.chi2.sf(lr, 1)) if lr > 0 else 1.0
    sd_se = s * se[p]
    var = s * s
    icc_row = []
    if fam == "binomial":
        icc = var / (var + np.pi ** 2 / 3)
        mor = float(np.exp(np.sqrt(2 * var) * stats.norm.ppf(0.75)))
        icc_row = [["Latent-scale ICC, sigma^2 / (sigma^2 + pi^2/3)", f(icc, 4)], ["Median odds ratio between two random groups", f(mor, 4)]]
        summary_extra = [stat("Latent ICC", icc)]
    else:
        mrr = float(np.exp(np.sqrt(2 * var) * stats.norm.ppf(0.75)))
        icc_row = [["Median rate ratio between two random groups", f(mrr, 4)]]
        summary_extra = [stat("Median rate ratio", mrr)]
    tabs = [table("Fixed effects (conditional on the group)", ["Term", "Estimate", "SE", "z", "p-value", eff, f"{eff} {lab} low", f"{eff} CI high"], fx,
                  f"Wald tests from the observed information of the marginal likelihood, integrated with {nodes}-point adaptive Gauss-Hermite quadrature" + (" (the Laplace approximation)." if nodes == 1 else ".")),
            table("Random intercept", ["Quantity", "Estimate", "SE"], [["Group SD (sigma)", f(s, 5), f(sd_se, 5)], ["Group variance (sigma^2)", f(var, 5), f(2 * var * se[p], 5)]] + [[a, b, None] for a, b in icc_row]),
            table("Likelihood-ratio test of the random intercept (against the ordinary GLM)", ["Statistic", "p-value"], [[f(lr, 4), f(p_lr, 6)]],
                  "The null value of the variance is on the boundary, so the p-value is half the chi-square(1) tail (a 50:50 mixture of chi-square(0) and chi-square(1))."),
            table("Model fit", ["Quantity", "Value"], [["Log-likelihood", f(ll, 4)], ["AIC", f(-2 * ll + 2 * k, 3)], ["BIC", f(-2 * ll + k * np.log(len(y)), 3)], ["Observations", len(y)], ["Groups", ng], ["Quadrature points", nodes],
                                                      ["Log-likelihood of the GLM without random intercept", f(ll_glm, 4)]])]
    summary = [stat("Observations", len(y)), stat("Groups", ng), stat("Log-likelihood", ll, 3), stat("Group SD", s)] + summary_extra
    order = np.argsort(u)
    gl = pd.unique(d[gcol].to_numpy())
    _, inv = np.unique(d[gcol].to_numpy(), return_index=True)
    labels = sorted(pd.unique(d[gcol].to_numpy()).tolist(), key=lambda v: str(v))
    labels_u = list(np.unique(d[gcol].to_numpy()))
    modes = [(str(labels_u[i]), float(s * u[i])) for i in range(len(u))]
    modes.sort(key=lambda t: t[1])
    charts = [chart("Conditional modes of the group effects (log scale), sorted", [a for a, _ in modes][:60], [("Group effect", [b for _, b in modes][:60])], "group", "effect on the linear predictor", "bar")]
    tabs.append(table("Predicted group effects (conditional modes)", ["Group", "Effect (log scale)"], [[a, f(b, 4)] for a, b in modes[:60]]))
    notes = []
    if s < 0.02:
        notes.append("The estimated group SD is close to zero: the data show almost no between-group variation beyond the covariates.")
    if dropped:
        notes.append(f"{dropped} row(s) with missing values were dropped.")
    if nodes > 1:
        notes.append("Increase the number of quadrature points to check that the estimates are stable; for binary outcomes with small clusters the Laplace approximation (1 point) can be biased.")
    return result("stat_glmm", f"Generalised linear mixed model ({'logistic' if fam == 'binomial' else 'Poisson'}, random intercept)", len(y), summary, tabs, charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# Generalised estimating equations
# ---------------------------------------------------------------------------------------------------------------
def _gee_family(fam):
    if fam == "gaussian":
        return (lambda eta: eta, lambda mu: np.ones_like(mu), lambda mu: np.ones_like(mu), lambda y, mu: -0.5 * (y - mu) ** 2)
    if fam == "binomial":
        return (lambda eta: 1 / (1 + np.exp(-eta)), lambda mu: mu * (1 - mu), lambda mu: mu * (1 - mu),
                lambda y, mu: y * np.log(np.clip(mu, 1e-300, 1)) + (1 - y) * np.log(np.clip(1 - mu, 1e-300, 1)))
    return (lambda eta: np.exp(np.minimum(eta, 700)), lambda mu: mu, lambda mu: mu, lambda y, mu: y * np.log(np.clip(mu, 1e-300, None)) - mu)


def _wcorr(kind, alpha, times):
    k = len(times)
    if kind == "independence":
        return np.eye(k)
    if kind == "exchangeable":
        R = np.full((k, k), alpha)
        np.fill_diagonal(R, 1.0)
        return R
    lag = np.abs(times[:, None] - times[None, :])
    return alpha ** lag


def gee(df, spec):
    alpha_l = _alpha(spec)
    fam = spec.get("family") or "gaussian"
    corr = spec.get("corr") or "exchangeable"
    if fam not in ("gaussian", "binomial", "poisson"):
        raise OESError("Unknown family.")
    ycol, gcol = need(spec, "y", "outcome"), need(spec, "group", "cluster (subject) identifier")
    tcol = spec.get("time") or None
    covs, facs = as_list(spec.get("covariates")), as_list(spec.get("factors"))
    fixed_a = spec.get("fixed_alpha")
    fixed_a = None if fixed_a in (None, "") else float_option(spec, "fixed_alpha", 0.0, "Fixed correlation parameter")
    if fixed_a is not None and not -0.99 <= fixed_a <= 0.99:
        raise OESError("The fixed correlation parameter must be between -0.99 and 0.99.")
    distinct(ycol, gcol, covs, facs)
    if tcol and tcol in (ycol, gcol):
        raise OESError("The time variable must differ from the outcome and the cluster identifier.")
    d, dropped = clean_frame(df, numeric=list(dict.fromkeys([ycol] + covs + ([tcol] if tcol else []))), factors=facs + [gcol])
    X, names = _fixed_design(d, covs, facs)
    y = d[ycol].to_numpy(float)
    if fam == "binomial" and not np.isin(y, [0, 1]).all():
        raise OESError("A binomial outcome must be coded 0/1.")
    if fam == "poisson" and ((y < 0).any() or not np.allclose(y, np.round(y))):
        raise OESError("A Poisson outcome must be a non-negative whole number.")
    N, p = X.shape
    gv = d[gcol].to_numpy()
    tv = d[tcol].to_numpy(float) if tcol else np.zeros(N)
    clusters = []
    for g in pd.unique(gv):
        idx = np.flatnonzero(gv == g)
        if tcol:
            idx = idx[np.argsort(tv[idx], kind="stable")]
            if len(np.unique(tv[idx])) != len(idx):
                raise OESError(f"Cluster {g} has repeated time values.")
        clusters.append(idx)
    G = len(clusters)
    if G < 10:
        raise OESError("GEE needs at least 10 clusters for robust standard errors to be trustworthy.")
    if max(len(i) for i in clusters) == 1:
        raise OESError("Every cluster has a single observation; there is nothing to correlate.")
    inv_link, varf, dmu, qfun = _gee_family(fam)
    deta_link = {"gaussian": lambda mu: np.ones_like(mu), "binomial": lambda mu: mu * (1 - mu), "poisson": lambda mu: mu}[fam]
    times = [(tv[i] - tv[i][0]) if tcol else np.arange(len(i), dtype=float) for i in clusters]
    # start from the independence GLM
    beta = np.zeros(p)
    if fam == "binomial":
        beta[0] = 0.0
    elif fam == "poisson":
        beta[0] = np.log(max(y.mean(), 1e-3))
    else:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    a = 0.0
    scale = 1.0

    def moments(beta_):
        mu = inv_link(X @ beta_)
        return mu, (y - mu) / np.sqrt(varf(mu))

    for it in range(200):
        mu, r = moments(beta)
        scale = float(np.sum(r ** 2) / (N - p))
        if fixed_a is not None and corr != "independence":
            a = fixed_a
        elif corr == "exchangeable":
            num = 0.0
            npairs = 0
            for i in clusters:
                ri = r[i]
                s1 = ri.sum()
                num += (s1 * s1 - np.sum(ri * ri)) / 2
                npairs += len(i) * (len(i) - 1) / 2
            a = num / (scale * (npairs - p)) if npairs > p else 0.0
            a = float(np.clip(a, -0.99, 0.99))
        elif corr == "ar1":
            num = 0.0
            cnt = 0
            for i in clusters:
                if len(i) > 1:
                    ri = r[i]
                    num += float(np.sum(ri[:-1] * ri[1:]))
                    cnt += len(i) - 1
            a = num / (scale * (cnt - p)) if cnt > p else 0.0
            a = float(np.clip(a, -0.99, 0.99))
        B = np.zeros((p, p))
        U = np.zeros(p)
        for i, tm in zip(clusters, times):
            mi = mu[i]
            sdi = np.sqrt(varf(mi))
            Di = X[i] * deta_link(mi)[:, None]
            R = _wcorr(corr, a, tm)
            Vi = sdi[:, None] * R * sdi[None, :]
            Vinv = np.linalg.inv(Vi)
            B += Di.T @ Vinv @ Di
            U += Di.T @ Vinv @ (y[i] - mi)
        step = np.linalg.solve(B, U)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    else:
        raise OESError("The GEE iterations did not converge.")
    mu, r = moments(beta)
    scale = float(np.sum(r ** 2) / (N - p))
    B = np.zeros((p, p))
    M = np.zeros((p, p))
    for i, tm in zip(clusters, times):
        mi = mu[i]
        sdi = np.sqrt(varf(mi))
        Di = X[i] * deta_link(mi)[:, None]
        R = _wcorr(corr, a, tm)
        Vinv = np.linalg.inv(sdi[:, None] * R * sdi[None, :])
        B += Di.T @ Vinv @ Di
        ui = Di.T @ Vinv @ (y[i] - mi)
        M += np.outer(ui, ui)
    Binv = np.linalg.inv(B)
    robust = Binv @ M @ Binv
    naive = Binv * scale
    se_r, se_n = np.sqrt(np.diag(robust)), np.sqrt(np.diag(naive))
    # QIC (Pan 2001): quasi-likelihood under independence, trace term from the independence model
    Bi = np.zeros((p, p))
    for i in clusters:
        mi = mu[i]
        Di = X[i] * deta_link(mi)[:, None]
        Bi += (Di / varf(mi)[:, None]).T @ Di
    qs = scale if fam == "gaussian" else 1.0   # the binomial and Poisson quasi-likelihoods are defined with dispersion 1
    Q = float(np.sum(qfun(y, mu))) / qs
    trace = float(np.trace(Bi @ robust)) / qs
    qic = -2 * Q + 2 * trace
    qicu = -2 * Q + 2 * p
    zc = stats.norm.ppf(1 - alpha_l / 2)
    lab = _lab(alpha_l)
    z = beta / se_r
    eff = {"binomial": "OR", "poisson": "IRR", "gaussian": None}[fam]
    cols = ["Term", "Estimate", "Robust SE", "z", "p-value", f"{lab} low", "CI high", "Naive SE"]
    rows = [[names[j], f(beta[j], 5), f(se_r[j], 5), f(z[j], 4), f(2 * stats.norm.sf(abs(z[j])), 6), f(beta[j] - zc * se_r[j], 5), f(beta[j] + zc * se_r[j], 5), f(se_n[j], 5)] for j in range(p)]
    if eff:
        cols += [eff, f"{eff} CI low", f"{eff} CI high"]
        for j in range(p):
            rows[j] += [f(np.exp(beta[j]), 5), f(np.exp(beta[j] - zc * se_r[j]), 5), f(np.exp(beta[j] + zc * se_r[j]), 5)]
    sizes = [len(i) for i in clusters]
    tabs = [table("GEE coefficients (population-average effects)", cols, rows,
                  "Inference uses the robust (sandwich) covariance, which is valid even if the working correlation is wrong. The naive standard error assumes it is right."),
            table("Working correlation and fit", ["Quantity", "Value"],
                  [["Family", fam], ["Working correlation", corr], ["Estimated correlation parameter", f(a, 5) if corr != "independence" else None], ["Scale (dispersion)", f(scale, 5)],
                   ["QIC", f(qic, 3)], ["QICu (for comparing covariate sets)", f(qicu, 3)], ["Clusters", G], ["Observations", N], ["Cluster size (min / mean / max)", f"{min(sizes)} / {N / G:.2f} / {max(sizes)}"]],
                  "QIC compares working-correlation structures for the same mean model (lower is better); QICu compares different covariate sets. For the binomial and Poisson families the quasi-likelihood uses dispersion 1; the dispersion above is the Pearson estimate and is used for the correlation parameter only.")]
    tr_, fr_ = thin(list(mu), list(r), 300)
    charts = [chart("Pearson residuals against fitted means", tr_, [("Pearson residual", fr_)], "fitted mean", "residual", "scatter")]
    summary = [stat("Observations", N), stat("Clusters", G), stat("Working correlation", f(a, 4) if corr != "independence" else "independence"), stat("QIC", qic, 3)]
    notes = []
    if G < 30:
        notes.append(f"Only {G} clusters: sandwich standard errors tend to be too small with fewer than about 30-40 clusters.")
    if corr == "ar1" and not tcol:
        notes.append("No time variable was chosen, so the observations in each cluster are taken to be equally spaced in the order they appear in the data.")
    if dropped:
        notes.append(f"{dropped} row(s) with missing values were dropped.")
    return result("stat_gee", "Generalised estimating equations", N, summary, tabs, charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# Intraclass correlation and Cronbach's alpha
# ---------------------------------------------------------------------------------------------------------------
def icc_forms(Y, alpha=0.05):
    """The six Shrout-Fleiss intraclass correlations of an n x k matrix (subjects x raters), with McGraw-Wong confidence intervals."""
    n, k = Y.shape
    gm = Y.mean()
    ssr = k * np.sum((Y.mean(1) - gm) ** 2)
    ssc = n * np.sum((Y.mean(0) - gm) ** 2)
    sst = np.sum((Y - gm) ** 2)
    sse = sst - ssr - ssc
    ssw = sst - ssr
    msr, msc, mse, msw = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1)), ssw / (n * (k - 1))
    I1 = (msr - msw) / (msr + (k - 1) * msw)
    I2 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    I3 = (msr - mse) / (msr + (k - 1) * mse)
    I1k, I2k, I3k = (msr - msw) / msr, (msr - mse) / (msr + (msc - mse) / n), (msr - mse) / msr
    fu = lambda a, b: stats.f.isf(alpha / 2, a, b)
    out = {}
    # single-measure and average-measure forms; the "average" intervals are transformed from the single ones
    F1 = msr / msw
    FL, FU = F1 / fu(n - 1, n * (k - 1)), F1 * fu(n * (k - 1), n - 1)
    out["ICC1"] = (I1, (FL - 1) / (FL + k - 1), (FU - 1) / (FU + k - 1), F1, n - 1, n * (k - 1))
    out["ICC1k"] = (I1k, 1 - 1 / FL, 1 - 1 / FU, F1, n - 1, n * (k - 1))
    F3 = msr / mse
    FL3, FU3 = F3 / fu(n - 1, (n - 1) * (k - 1)), F3 * fu((n - 1) * (k - 1), n - 1)
    out["ICC3"] = (I3, (FL3 - 1) / (FL3 + k - 1), (FU3 - 1) / (FU3 + k - 1), F3, n - 1, (n - 1) * (k - 1))
    out["ICC3k"] = (I3k, 1 - 1 / FL3, 1 - 1 / FU3, F3, n - 1, (n - 1) * (k - 1))
    FJ = msc / mse
    vn = (k - 1) * (n - 1) * (k * I2 * FJ + n * (1 + (k - 1) * I2) - k * I2) ** 2
    vd = (n - 1) * k ** 2 * I2 ** 2 * FJ ** 2 + (n * (1 + (k - 1) * I2) - k * I2) ** 2
    v = vn / vd
    F2U, F2L = fu(n - 1, v), fu(v, n - 1)
    lo2 = n * (msr - F2U * mse) / (F2U * (k * msc + (k * n - k - n) * mse) + n * msr)
    hi2 = n * (F2L * msr - mse) / (k * msc + (k * n - k - n) * mse + n * F2L * msr)
    out["ICC2"] = (I2, lo2, hi2, msr / mse, n - 1, (n - 1) * (k - 1))
    out["ICC2k"] = (I2k, lo2 * k / (1 + lo2 * (k - 1)), hi2 * k / (1 + hi2 * (k - 1)), msr / mse, n - 1, (n - 1) * (k - 1))
    return out, dict(msr=msr, msc=msc, mse=mse, msw=msw, ssr=ssr, ssc=ssc, sse=sse, ssw=ssw, n=n, k=k)


def icc(df, spec):
    alpha = _alpha(spec)
    cols = as_list(spec.get("raters"))
    if len(cols) < 2:
        raise OESError("Choose at least two rater (measurement) columns.")
    d, dropped = clean_frame(df, numeric=cols)
    Y = d[cols].to_numpy(float)
    n, k = Y.shape
    if n < 3:
        raise OESError("At least three subjects (rows) are needed.")
    if np.ptp(Y) == 0:
        raise OESError("All ratings are identical, so the ICC is undefined.")
    forms, ms = icc_forms(Y, alpha)
    lab = _lab(alpha)
    names = {"ICC1": ("ICC(1,1)", "Single rater, one-way random effects (each subject rated by a different random set of raters)"),
             "ICC2": ("ICC(2,1)", "Single rater, two-way random effects, absolute agreement"),
             "ICC3": ("ICC(3,1)", "Single rater, two-way mixed effects, consistency"),
             "ICC1k": (f"ICC(1,{k})", f"Mean of {k} raters, one-way random effects"),
             "ICC2k": (f"ICC(2,{k})", f"Mean of {k} raters, two-way random effects, absolute agreement"),
             "ICC3k": (f"ICC(3,{k})", f"Mean of {k} raters, two-way mixed effects, consistency")}
    rows = []
    for key in ["ICC1", "ICC2", "ICC3", "ICC1k", "ICC2k", "ICC3k"]:
        est, lo, hi, F, d1, d2 = forms[key]
        rows.append([names[key][0], names[key][1], f(est, 5), f(F, 4), d1, d2, f(stats.f.sf(F, d1, d2), 6), f(lo, 4), f(hi, 4)])
    tabs = [table("Intraclass correlation coefficients (Shrout and Fleiss 1979)", ["Form", "Model", "ICC", "F", "df1", "df2", "p-value", f"{lab} low", "CI high"], rows,
                  "Use the single-rater forms when one rater's score will be used, the average forms when scores are averaged over all raters. 'Absolute agreement' penalises raters who differ systematically; 'consistency' does not.")]
    tabs.append(table("Mean squares", ["Source", "Sum of squares", "df", "Mean square"],
                      [["Between subjects", f(ms["ssr"], 5), n - 1, f(ms["msr"], 5)], ["Between raters", f(ms["ssc"], 5), k - 1, f(ms["msc"], 5)], ["Residual (subject x rater)", f(ms["sse"], 5), (n - 1) * (k - 1), f(ms["mse"], 5)],
                       ["Within subjects (raters + residual)", f(ms["ssw"], 5), n * (k - 1), f(ms["msw"], 5)]]))
    # Cronbach's alpha (Feldt interval)
    item_var = Y.var(axis=0, ddof=1).sum()
    tot_var = Y.sum(axis=1).var(ddof=1)
    alpha_c = k / (k - 1) * (1 - item_var / tot_var)
    d1, d2 = n - 1, (n - 1) * (k - 1)
    lo_a = 1 - (1 - alpha_c) * stats.f.isf(alpha / 2, d1, d2)
    hi_a = 1 - (1 - alpha_c) * stats.f.isf(1 - alpha / 2, d1, d2)
    tabs.append(table("Internal consistency", ["Measure", "Estimate", f"{lab} low", "CI high"], [["Cronbach's alpha", f(alpha_c, 5), f(lo_a, 4), f(hi_a, 4)]],
                      "Cronbach's alpha equals ICC(3,k) for these data; the interval is Feldt's F-based interval."))
    rater_rows = [[c, n, f(Y[:, j].mean(), 4), f(Y[:, j].std(ddof=1), 4), f(float(np.corrcoef(Y[:, j], np.delete(Y, j, axis=1).mean(1))[0, 1]), 4)] for j, c in enumerate(cols)]
    tabs.append(table("Raters", ["Rater", "n", "Mean", "SD", "Correlation with the mean of the others"], rater_rows))
    keys = ["ICC1", "ICC2", "ICC3", "ICC1k", "ICC2k", "ICC3k"]
    charts = [chart("Intraclass correlations with their confidence limits", [names[q][0] for q in keys], [("ICC", [forms[q][0] for q in keys]), ("CI low", [forms[q][1] for q in keys]), ("CI high", [forms[q][2] for q in keys])], "form", "ICC", "bar")]
    e3 = forms["ICC2"][0]
    interp = "poor" if e3 < 0.5 else "moderate" if e3 < 0.75 else "good" if e3 < 0.9 else "excellent"
    summary = [stat("Subjects", n), stat("Raters", k), stat("ICC(2,1)", forms["ICC2"][0]), stat("ICC(3,1)", forms["ICC3"][0]), stat("Cronbach's alpha", alpha_c)]
    notes = [f"By the Koo and Li (2016) rule of thumb the single-rater absolute-agreement reliability is {interp} (below 0.5 poor, 0.5-0.75 moderate, 0.75-0.9 good, above 0.9 excellent). Judge it against the confidence interval, not only the point estimate."]
    if dropped:
        notes.append(f"{dropped} subject(s) with a missing rating were dropped; the ICC needs complete rows.")
    return result("stat_icc", "Intraclass correlation and reliability", n, summary, tabs, charts, notes)
