"""Nonparametric and robust models: kernel regression, partially linear models, penalised splines, robust regression,
cluster-robust inference with the wild cluster bootstrap, kernel density estimation, multiple testing, extreme value analysis.
NumPy, SciPy and statsmodels. Same result format as the other models (core.py)."""
import numpy as np
import pandas as pd
from scipy import optimize, stats

from .core import (OESError, as_list, chart, check_columns, distinct, f, float_option, int_option, need, numeric, result, stat, table, test_row, tests_table)


def _mad(r):
    return np.median(np.abs(r - np.median(r))) / 0.6744897501960817


# ================================================================================================ local polynomial smoothers
def _ll_parts(x, xe, h, degree, kernel):
    """Smoother weights for evaluation points xe: returns the matrix L (len(xe) x n) of a local polynomial of degree 0 or 1."""
    d = x[None, :] - xe[:, None]
    u = d / h
    if kernel == "epanechnikov":
        K = np.where(np.abs(u) < 1, 0.75 * (1 - u ** 2), 0.0)
    else:
        K = np.exp(-0.5 * u ** 2)
    S0 = K.sum(1)
    if degree == 0:
        S0 = np.where(S0 < 1e-300, np.nan, S0)
        return K / S0[:, None]
    S1 = (K * d).sum(1); S2 = (K * d ** 2).sum(1)
    den = S0 * S2 - S1 ** 2
    den = np.where(np.abs(den) < 1e-300, np.nan, den)
    return K * (S2[:, None] - d * S1[:, None]) / den[:, None]


def _ll_fit(x, y, xe, h, degree=1, kernel="gaussian"):
    L = _ll_parts(x, xe, h, degree, kernel)
    return L @ y, L


def _ll_cv(x, y, h, degree, kernel):
    """Leave-one-out CV score and effective degrees of freedom, by the hat matrix diagonal."""
    L = _ll_parts(x, x, h, degree, kernel)
    fit = L @ y
    li = np.diag(L)
    ok = np.isfinite(fit) & np.isfinite(li) & (np.abs(1 - li) > 1e-8)
    if ok.sum() < len(x) * 0.9:
        return np.inf, np.nan
    loo = (y - fit) / (1 - li)
    return float(np.mean(loo[ok] ** 2)), float(np.nansum(li))


def _bw_grid(x):
    s = min(x.std(ddof=1), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349)
    base = 1.06 * s * len(x) ** (-0.2)
    return base * np.exp(np.linspace(np.log(0.15), np.log(4), 30)), base


def _choose_bw(x, y, degree, kernel):
    grid, base = _bw_grid(x)
    if kernel == "epanechnikov":
        grid = grid * 2.2
    sc = [_ll_cv(x, y, h, degree, kernel)[0] for h in grid]
    i = int(np.nanargmin(sc))
    if i in (0, len(grid) - 1):
        # refine outward once
        pass
    return float(grid[i]), grid, np.array(sc), base


def _polyfit_cv(x, y, deg):
    X = np.vander(x - x.mean(), deg + 1)
    H = X @ np.linalg.pinv(X)
    fit = H @ y
    loo = (y - fit) / (1 - np.diag(H))
    return float(np.mean(loo ** 2)), float(np.mean((y - fit) ** 2)), fit


def kernreg(df, spec):
    yv = need(spec, "y", "outcome")
    xv = need(spec, "x", "regressor")
    distinct([yv], [xv])
    degree = int(spec.get("degree", 1)) if str(spec.get("degree", 1)) in ("0", "1") else 1
    kernel = str(spec.get("kernel", "gaussian"))
    reps = int_option(spec, "reps", 199, 0, 2000, "bootstrap replications")
    seed = int_option(spec, "seed", 0, 0, 10 ** 9, "seed")
    d = numeric(df, [yv, xv]).sort_values(xv).reset_index(drop=True)
    x = d[xv].to_numpy(float); y = d[yv].to_numpy(float)
    n = len(x)
    if n < 30:
        raise OESError(f"Only {n} observations: at least 30 are needed for kernel regression.")
    if n > 4000:
        raise OESError("More than 4,000 observations: kernel regression here builds an n by n matrix; use a random subsample.")
    bw = spec.get("bandwidth")
    if bw not in (None, ""):
        h = float(bw)
        if h <= 0:
            raise OESError("The bandwidth must be positive.")
        grid, base = _bw_grid(x); sc = np.array([_ll_cv(x, y, g, degree, kernel)[0] for g in grid])
    else:
        h, grid, sc, base = _choose_bw(x, y, degree, kernel)
    cv_h, edf = _ll_cv(x, y, h, degree, kernel)
    L = _ll_parts(x, x, h, degree, kernel)
    fit = L @ y
    res = y - fit
    tr1 = np.trace(L); tr2 = np.sum(L ** 2)
    IL = np.eye(n) - L
    d1 = np.trace(IL.T @ IL)
    s2 = float(res @ res / d1)
    ng = 60
    xg = np.linspace(np.percentile(x, 2), np.percentile(x, 98), ng)
    Lg = _ll_parts(x, xg, h, degree, kernel)
    fg = Lg @ y
    se = np.sqrt(s2 * np.sum(Lg ** 2, 1))
    # derivative by local linear slope
    if degree == 1:
        dd = x[None, :] - xg[:, None]; u = dd / h
        K = np.exp(-0.5 * u ** 2) if kernel == "gaussian" else np.where(np.abs(u) < 1, 0.75 * (1 - u ** 2), 0)
        S0 = K.sum(1); S1 = (K * dd).sum(1); S2 = (K * dd ** 2).sum(1)
        wts = K * (S0[:, None] * dd - S1[:, None]) / (S0 * S2 - S1 ** 2)[:, None]
        slope = wts @ y
    else:
        slope = np.gradient(fg, xg)
    tabs = []
    rows = []
    for k, lab in ((1, "Linear"), (2, "Quadratic"), (3, "Cubic"), (4, "Quartic")):
        cvs, mse, _ = _polyfit_cv(x, y, k)
        rows.append([f"Polynomial of degree {k} ({lab.lower()})", k + 1, cvs, mse])
    rows.append([f"Local {'linear' if degree == 1 else 'constant'} ({kernel}, h = {h:.4g})", float(tr1), cv_h, float(np.mean(res ** 2))])
    tabs.append(table("Fit and leave-one-out cross-validation", ["Model", "Parameters (effective)", "CV error (MSE)", "In-sample MSE"], rows,
                      "The effective number of parameters of the smoother is the trace of its hat matrix. A lower CV error means better prediction of held-out points."))
    j = int(np.nanargmin(sc))
    tabs.append(table("Bandwidth", ["Quantity", "Value"], [["Selected bandwidth h", float(h)], ["Chosen by", "user" if bw not in (None, "") else "least-squares cross-validation"], ["Rule-of-thumb bandwidth (1.06 s n^-1/5)", float(base)],
                                                         ["CV minimiser on the grid", float(grid[j])], ["Smallest bandwidth tried", float(grid[0])], ["Largest bandwidth tried", float(grid[-1])], ["Residual standard deviation", float(np.sqrt(s2))],
                                                         ["R-squared (in sample)", float(1 - res @ res / np.sum((y - y.mean()) ** 2))]],
                      "A selected bandwidth at either end of the grid means the search should be widened."))
    # linearity test: Hardle-Mammen wild bootstrap with bias-matched statistic
    olsb = np.polyfit(x, y, 1); yl = np.polyval(olsb, x); el = y - yl
    T = float(np.sum((fit - L @ yl) ** 2))
    pval = None
    if reps > 0:
        rng = np.random.default_rng(seed)
        a1 = (1 - np.sqrt(5)) / 2; a2 = (1 + np.sqrt(5)) / 2; pa = (5 + np.sqrt(5)) / 10
        cnt = 0
        Lyl = L @ yl
        for _ in range(reps):
            v = np.where(rng.random(n) < pa, a1, a2)
            ys = yl + el * v
            b = np.polyfit(x, ys, 1); ysl = np.polyval(b, x)
            Tb = np.sum((L @ ys - L @ ysl) ** 2)
            cnt += Tb >= T
        pval = (cnt + 1) / (reps + 1)
    tabs.append(tests_table([test_row("Linear regression against the kernel fit (Hardle-Mammen wild bootstrap)", T, pval, f"{reps} replications; H0: the regression function is linear")] if pval is not None else
                            [test_row("Linear against kernel (not run: reps = 0)", T, None, "")], "Test of a linear specification"))
    tabs.append(table("Estimated regression function", [xv, "Fitted", "Lower 95%", "Upper 95%", "Slope"], [[float(xg[i]), float(fg[i]), float(fg[i] - 1.96 * se[i]), float(fg[i] + 1.96 * se[i]), float(slope[i])] for i in range(0, ng, 5)],
                      "Pointwise 95% bands ignore the smoothing bias; they describe the variance of the estimate at the chosen bandwidth."))
    ch = [chart("Local regression with 95% bands", [f"{v:.3g}" for v in xg], [("Fitted", fg), ("Lower 95%", fg - 1.96 * se), ("Upper 95%", fg + 1.96 * se), ("Linear fit", np.polyval(olsb, xg))], xv, yv, "line"),
          chart("Cross-validation error by bandwidth", [f"{v:.3g}" for v in grid], [("CV error", np.where(np.isfinite(sc), sc, np.nan))], "Bandwidth", "CV error", "line")]
    return result("kernreg", "Kernel and local linear regression", n, [stat("Observations", n), stat("Bandwidth", float(h), 4), stat("Effective parameters", float(tr1), 2), stat("CV error", float(cv_h), 4),
                                                                        stat("Linearity test p-value", pval if pval is not None else "n/a", 3)], tabs, ch,
                  ["Local polynomial regression (Fan and Gijbels 1996): at each point a weighted regression with kernel weights. The bandwidth is chosen by leave-one-out cross-validation, using the closed form from the hat matrix.",
                   "The linearity test follows Hardle and Mammen (1993): the kernel fit is compared with the kernel smooth of the linear fit, which removes the smoothing bias, and the null distribution is found by the wild bootstrap."])


def _cv_bandwidth_resid(z, v, kernel="gaussian"):
    grid, base = _bw_grid(z)
    sc = [_ll_cv(z, v, h, 1, kernel)[0] for h in grid]
    return float(grid[int(np.nanargmin(sc))])


def plm(df, spec):
    yv = need(spec, "y", "outcome")
    xs = as_list(need(spec, "x", "parametric regressors"))
    zv = need(spec, "z", "nonparametric variable")
    distinct([yv, zv], xs)
    d = numeric(df, [yv, zv] + xs).sort_values(zv).reset_index(drop=True)
    n = len(d)
    if n < 50:
        raise OESError(f"Only {n} observations: at least 50 are needed for a partially linear model.")
    if n > 4000:
        raise OESError("More than 4,000 observations: use a random subsample.")
    y = d[yv].to_numpy(float); X = d[xs].to_numpy(float); z = d[zv].to_numpy(float)
    names = xs
    k = len(xs)
    # Robinson: residualise y and each x on z
    hu = float(spec["bandwidth"]) if spec.get("bandwidth") not in (None, "") else None
    if hu is not None and hu <= 0:
        raise OESError("The bandwidth must be positive.")
    def sm(v, h=None):
        h = h or hu or _cv_bandwidth_resid(z, v)
        return _ll_parts(z, z, h, 1, "gaussian") @ v, h
    ey, hy = sm(y)
    EX = []; hx = []
    for j in range(k):
        e, h = sm(X[:, j]); EX.append(e); hx.append(h)
    Ry = y - ey; RX = X - np.column_stack(EX)
    b = np.linalg.lstsq(RX, Ry, rcond=None)[0]
    u = Ry - RX @ b
    XtXi = np.linalg.inv(RX.T @ RX)
    V = XtXi @ ((RX * u[:, None]).T @ (RX * u[:, None])) @ XtXi * n / (n - k)
    se = np.sqrt(np.diag(V))
    # comparison: OLS with polynomial in z
    cmp = []
    for deg in (1, 2, 3):
        Z = np.vander(z - z.mean(), deg + 1)
        Xf = np.column_stack([X, Z])
        bf = np.linalg.lstsq(Xf, y, rcond=None)[0]
        ef = y - Xf @ bf
        Vf = np.linalg.inv(Xf.T @ Xf) @ ((Xf * ef[:, None]).T @ (Xf * ef[:, None])) @ np.linalg.inv(Xf.T @ Xf) * n / (n - Xf.shape[1])
        cmp.append((deg, bf[:k], np.sqrt(np.diag(Vf))[:k], float(ef @ ef / n)))
    rows = []
    for j in range(k):
        rows.append([names[j], float(b[j]), float(se[j]), float(b[j] / se[j]), float(2 * stats.norm.sf(abs(b[j] / se[j]))), float(cmp[0][1][j]), float(cmp[1][1][j]), float(cmp[2][1][j])])
    tabs = [table("Partially linear model (Robinson 1988)", ["Variable", "Coefficient", "Robust s.e.", "z", "p", "OLS, linear in z", "OLS, quadratic in z", "OLS, cubic in z"], rows,
                  f"Dependent variable: {yv}; nonparametric in {zv}; {n} observations. Bandwidths for the residualising smooths: {yv} {hy:.3g}, " + ", ".join(f"{names[j]} {hx[j]:.3g}" for j in range(k)) + ".")]
    # g(z)
    hg = _cv_bandwidth_resid(z, y - X @ b)
    zg = np.linspace(np.percentile(z, 2), np.percentile(z, 98), 60)
    Lg = _ll_parts(z, zg, hg, 1, "gaussian")
    g = Lg @ (y - X @ b)
    resid = y - X @ b - _ll_parts(z, z, hg, 1, "gaussian") @ (y - X @ b)
    s2 = float(resid @ resid / n)
    sg = np.sqrt(s2 * np.sum(Lg ** 2, 1))
    d1 = np.gradient(g, zg)
    tabs.append(table(f"Nonparametric part g({zv})", [zv, "g", "Lower 95%", "Upper 95%", "Slope"], [[float(zg[i]), float(g[i]), float(g[i] - 1.96 * sg[i]), float(g[i] + 1.96 * sg[i]), float(d1[i])] for i in range(0, 60, 6)],
                      f"Local linear fit of y - X b on {zv}, bandwidth {hg:.3g}; bands ignore the smoothing bias."))
    # specification: compare nonparametric to polynomials by residual sum of squares
    rss_np = float(np.sum((y - X @ b - _ll_parts(z, z, hg, 1, "gaussian") @ (y - X @ b)) ** 2))
    comp = [[f"Polynomial of degree {c[0]} in {zv}", c[3] * n] for c in cmp]
    comp.append([f"Nonparametric in {zv}", rss_np])
    tabs.append(table("Residual sum of squares", ["Specification", "RSS"], comp, "The nonparametric part is flexible, so its in-sample RSS is lower by construction; the question is whether it is much lower than the best polynomial."))
    ch = [chart(f"The nonparametric part of the model, g({zv})", [f"{v:.3g}" for v in zg], [("g", g), ("Lower 95%", g - 1.96 * sg), ("Upper 95%", g + 1.96 * sg)], zv, "g", "line")]
    return result("plm", "Partially linear model (Robinson)", n, [stat("Observations", n)] + [stat(f"{names[j]} coefficient", float(b[j]), 4) for j in range(min(k, 3))] + [stat("Bandwidth for g", float(hg), 4)], tabs, ch,
                  ["Robinson (1988): y - E[y|z] is regressed on x - E[x|z], so the parametric coefficients are estimated at the root-n rate without a functional form for the effect of z. Conditional means are estimated by local linear smoothing with cross-validated bandwidths.",
                   "Standard errors are heteroskedasticity-robust; the confidence bands for g ignore smoothing bias."])


# ================================================================================================ penalised spline
def _bspline_basis(x, lo, hi, nk, deg=3):
    from scipy.interpolate import BSpline
    dx = (hi - lo) / nk
    t = np.concatenate([lo - dx * np.arange(deg, 0, -1), lo + dx * np.arange(nk + 1), hi + dx * np.arange(1, deg + 1)])
    nb = len(t) - deg - 1
    B = BSpline.design_matrix(np.clip(x, lo, hi - 1e-12 * (hi - lo)), t, deg).toarray()
    return B, nb


def _ps_solve(B, P, y, lam, Z=None):
    n, nb = B.shape
    if Z is not None:
        A = np.hstack([B, Z])
        Pf = np.zeros((A.shape[1], A.shape[1])); Pf[:nb, :nb] = P
    else:
        A, Pf = B, P
    M = A.T @ A + lam * Pf + 1e-9 * np.eye(A.shape[1])
    Mi = np.linalg.inv(M)
    coef = Mi @ A.T @ y
    H = A @ Mi @ A.T
    return coef, Mi, float(np.trace(H)), A


def pspline(df, spec):
    yv = need(spec, "y", "outcome")
    xv = need(spec, "x", "smooth variable")
    zs = as_list(spec.get("z"))
    distinct([yv, xv], zs)
    nk = int_option(spec, "knots", 20, 5, 60, "number of knots")
    d = numeric(df, [yv, xv] + zs).sort_values(xv).reset_index(drop=True)
    n = len(d)
    if n < 30:
        raise OESError(f"Only {n} observations: at least 30 are needed.")
    x = d[xv].to_numpy(float); y = d[yv].to_numpy(float)
    Z = np.column_stack([np.ones(n)] + [d[c].to_numpy(float) for c in zs]) if zs else np.ones((n, 1))
    lo, hi = x.min(), x.max() + 1e-9 * (x.max() - x.min())
    B, nb = _bspline_basis(x, lo, hi, nk)
    D = np.diff(np.eye(nb), 2, axis=0)
    P = D.T @ D
    lam_user = spec.get("lambda")
    grid = 10.0 ** np.linspace(-4, 7, 45)
    gcv = []
    for lam in grid:
        coef, Mi, edf, A = _ps_solve(B, P, y, lam, Z)
        r = y - A @ coef
        gcv.append(n * float(r @ r) / (n - edf) ** 2)
    gcv = np.array(gcv)
    lam = float(lam_user) if lam_user not in (None, "") else float(grid[int(np.argmin(gcv))])
    if lam <= 0:
        raise OESError("lambda must be positive.")
    coef, Mi, edf, A = _ps_solve(B, P, y, lam, Z)
    fit = A @ coef; res = y - fit
    nz = Z.shape[1]
    s2 = float(res @ res / (n - edf))
    xg = np.linspace(x.min(), x.max() - 1e-9 * (x.max() - x.min()), 80)
    Bg, _ = _bspline_basis(xg, lo, hi, nk)
    Ag = np.hstack([Bg, np.zeros((80, nz))])
    Ag[:, nb:] = Z.mean(0)
    # remove the intercept ambiguity: show the smooth plus the mean of covariates
    gfit = Ag @ coef
    Vc = s2 * Mi
    gse = np.sqrt(np.einsum("ij,jk,ik->i", Ag, Vc, Ag))
    # linear alternative
    Xl = np.column_stack([Z, x]); bl = np.linalg.lstsq(Xl, y, rcond=None)[0]; rl = y - Xl @ bl
    rss_l = float(rl @ rl); rss = float(res @ res)
    edf_m = edf - nz            # effective df of the smooth part
    F = ((rss_l - rss) / max(edf - (nz + 1), 1e-6)) / (rss / (n - edf))
    pF = float(stats.f.sf(F, max(edf - (nz + 1), 1e-6), n - edf))
    # polynomial comparison by cross-validation
    rows = []
    for k in (1, 2, 3, 4):
        cvs, mse, _ = _polyfit_cv(x, y, k) if not zs else (np.nan, np.nan, None)
        rows.append([f"Polynomial of degree {k}", k + 1 + len(zs), cvs, mse])
    H = A @ Mi @ A.T
    loo = res / (1 - np.diag(H))
    rows.append([f"Penalised spline ({nk} knots, lambda = {lam:.3g})", float(edf), float(np.mean(loo ** 2)), float(np.mean(res ** 2))])
    tabs = [table("Penalised spline fit (P-spline, Eilers and Marx 1996)", ["Quantity", "Value"],
                  [["Smoothing parameter lambda", lam], ["Chosen by", "user" if lam_user not in (None, "") else "generalised cross-validation"], ["Effective degrees of freedom (total)", float(edf)], ["Effective degrees of freedom of the smooth", float(edf_m)],
                   ["Residual standard deviation", float(np.sqrt(s2))], ["R-squared", float(1 - rss / np.sum((y - y.mean()) ** 2))], ["GCV score", float(gcv.min() if lam_user in (None, "") else n * rss / (n - edf) ** 2)]],
                  f"Cubic B-splines with {nk} equally spaced knots and a second-order difference penalty. Effective degrees of freedom of 2 would be a straight line."),
            table("Comparison with polynomials", ["Model", "Parameters (effective)", "CV error (MSE)", "In-sample MSE"], rows, "Polynomial CV errors are shown for models without extra covariates."),
            tests_table([test_row("Linear against smooth (approximate F test)", float(F), pF, f"F with {max(edf - (nz + 1), 1e-6):.1f} and {n - edf:.1f} degrees of freedom; H0: the effect of {xv} is linear")], "Test of nonlinearity")]
    if zs:
        zb = coef[nb:]
        zse = np.sqrt(np.diag(Vc)[nb:])
        tabs.append(table("Linear covariates", ["Variable", "Coefficient", "s.e.", "t"], [[(["Constant"] + zs)[j], float(zb[j]), float(zse[j]), float(zb[j] / zse[j])] for j in range(1, nz)] if nz > 1 else [["(none)", 0.0, 0.0, 0.0]]))
    tabs.append(table("Estimated smooth function", [xv, "Fitted", "Lower 95%", "Upper 95%"], [[float(xg[i]), float(gfit[i]), float(gfit[i] - 1.96 * gse[i]), float(gfit[i] + 1.96 * gse[i])] for i in range(0, 80, 8)], "Bayesian credible bands (Wahba 1983)."))
    ch = [chart("Penalised spline with 95% bands", [f"{v:.3g}" for v in xg], [("Fitted", gfit), ("Lower 95%", gfit - 1.96 * gse), ("Upper 95%", gfit + 1.96 * gse)], xv, yv, "line"),
          chart("Generalised cross-validation by smoothing parameter", [f"{np.log10(v):.1f}" for v in grid], [("GCV", gcv)], "log10 lambda", "GCV", "line")]
    return result("pspline", "Penalised splines and additive models", n, [stat("Observations", n), stat("Effective df", float(edf), 2), stat("Lambda", lam, 4), stat("R-squared", float(1 - rss / np.sum((y - y.mean()) ** 2)), 3),
                                                                           stat("Nonlinearity test p-value", pF, 4)], tabs, ch,
                  ["Penalised B-spline regression (Eilers and Marx 1996; Ruppert, Wand and Carroll 2003): a flexible basis with a roughness penalty, the smoothing parameter chosen by generalised cross-validation. Linear covariates enter unpenalised (a partially linear additive model).",
                   "Bands are Bayesian credible intervals for the smooth plus the covariate means; they treat lambda as known. The F test for nonlinearity is approximate."])


# ================================================================================================ robust regression
def _bisq_rho(u, c):
    a = np.abs(u) / c
    return np.where(a < 1, 1 - (1 - a ** 2) ** 3, 1.0)


def _mscale(r, c=1.547, b=0.5, s0=None):
    s = s0 if s0 else _mad(r)
    if s <= 0:
        return 0.0
    for _ in range(100):
        s_new = s * np.sqrt(np.mean(_bisq_rho(r / s, c)) / b)
        if abs(s_new - s) < 1e-10 * s:
            s = s_new; break
        s = s_new
    return float(s)


def _irls(X, y, b0, scale, c, fixed_scale=True, maxit=200):
    b = b0.copy()
    for _ in range(maxit):
        r = y - X @ b
        s = scale if fixed_scale else _mad(r)
        u = r / s
        w = np.where(np.abs(u) < c, (1 - (u / c) ** 2) ** 2, 0.0)
        if w.sum() < X.shape[1]:
            break
        sw = np.sqrt(w)
        bn = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        if np.max(np.abs(bn - b)) < 1e-9 * (1 + np.max(np.abs(b))):
            b = bn; break
        b = bn
    return b


def _s_estimate(X, y, rng, nsub=500):
    n, p = X.shape
    best = (np.inf, None, None)
    cands = []
    for _ in range(nsub):
        idx = rng.choice(n, p, replace=False)
        try:
            b = np.linalg.solve(X[idx], y[idx])
        except np.linalg.LinAlgError:
            continue
        r = y - X @ b
        s = _mscale(r)
        cands.append((s, b))
    cands.sort(key=lambda t: t[0])
    for s, b in cands[:10]:
        bb = b.copy(); ss = s
        for _ in range(30):                              # refine: a few S-step IRLS iterations
            r = y - X @ bb
            u = r / ss
            w = np.where(np.abs(u) < 1.547, (1 - (u / 1.547) ** 2) ** 2, 0.0)
            if w.sum() < p:
                break
            sw = np.sqrt(w)
            bn = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
            sn = _mscale(y - X @ bn, s0=ss)
            if sn <= ss:
                bb, ss = bn, sn
            else:
                break
        if ss < best[0]:
            best = (ss, bb, None)
    return best[1], best[0]


def robreg(df, spec):
    import statsmodels.api as sm
    yv = need(spec, "y", "outcome")
    xs = as_list(need(spec, "x", "regressors"))
    distinct([yv], xs)
    seed = int_option(spec, "seed", 0, 0, 10 ** 9, "seed")
    thr = float_option(spec, "flag", 2.5, "cut-off for flagging outliers")
    d = numeric(df, [yv] + xs)
    labels = None
    if spec.get("label"):
        d = numeric(df, [yv] + xs, keep=[spec["label"]]); labels = d[spec["label"]].astype(str).tolist()
    n, p = len(d), len(xs) + 1
    if n < 3 * p + 10:
        raise OESError(f"Only {n} observations for {p} parameters: at least {3 * p + 10} are needed.")
    y = d[yv].to_numpy(float); X = np.column_stack([np.ones(n), d[xs].to_numpy(float)])
    names = ["Constant"] + xs
    rng = np.random.default_rng(seed)
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    hub = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    bis = sm.RLM(y, X, M=sm.robust.norms.TukeyBiweight()).fit()
    lad = sm.QuantReg(y, X).fit(q=0.5)
    b_s, s_s = _s_estimate(X, y, rng)
    b_mm = _irls(X, y, b_s, s_s, 4.685, fixed_scale=True)
    r_mm = y - X @ b_mm
    # MM covariance (Yohai 1987 asymptotic): s^2 * a / b^2 * (X'X)^-1 , weights version
    u = r_mm / s_s; c = 4.685
    psi = np.where(np.abs(u) < c, u * (1 - (u / c) ** 2) ** 2, 0.0)
    dpsi = np.where(np.abs(u) < c, (1 - (u / c) ** 2) * (1 - 5 * (u / c) ** 2), 0.0)
    a = np.mean(psi ** 2); bb = np.mean(dpsi)
    V_mm = s_s ** 2 * a / bb ** 2 * np.linalg.inv(X.T @ X)
    se_mm = np.sqrt(np.diag(V_mm))
    res_ols = y - X @ ols
    s_ols = float(np.sqrt(res_ols @ res_ols / (n - p)))
    se_ols = np.sqrt(np.diag(s_ols ** 2 * np.linalg.inv(X.T @ X)))
    rows = []
    for j in range(p):
        rows.append([names[j], float(ols[j]), float(hub.params[j]), float(bis.params[j]), float(lad.params[j]), float(b_mm[j]), float(se_mm[j]), float(se_ols[j])])
    tabs = [table("Coefficients by estimator", ["Variable", "OLS", "Huber M", "Bisquare M", "Median (LAD)", "MM", "s.e. (MM)", "s.e. (OLS)"], rows,
                  "Huber and bisquare use iteratively reweighted least squares with the MAD scale, started from OLS; MM starts from a high-breakdown S estimate (50% breakdown) and reaches 95% efficiency at the normal.")]
    # outlier diagnostics
    zr = r_mm / s_s
    w_mm = np.where(np.abs(zr) < c, (1 - (zr / c) ** 2) ** 2, 0.0)
    flag = np.abs(zr) > thr
    ids = np.argsort(-np.abs(zr))[:min(10, n)]
    lab = labels or [str(i + 1) for i in range(n)]
    tabs.append(table(f"Largest robust residuals (flag: |residual / scale| > {thr:g})", ["Observation", yv, "Fitted (MM)", "Robust standardised residual", "MM weight", "OLS standardised residual", "Flagged"],
                      [[lab[i], float(y[i]), float((X @ b_mm)[i]), float(zr[i]), float(w_mm[i]), float(res_ols[i] / s_ols), "yes" if flag[i] else "no"] for i in ids],
                      f"{int(flag.sum())} of {n} observations are flagged. The robust scale (S estimate) is {s_s:.4g}; the OLS residual standard deviation is {s_ols:.4g}."))
    # classical vs robust R2 and scale
    tabs.append(table("Scale and fit", ["Quantity", "Value"], [["Residual scale, OLS", s_ols], ["Residual scale, MM (S estimate)", float(s_s)], ["Share of observations with weight below 0.5", float((w_mm < 0.5).mean())],
                                                              ["Share with zero weight", float((w_mm == 0).mean())], ["Largest change in a coefficient, MM vs OLS (in OLS s.e.)", float(np.max(np.abs(b_mm - ols) / se_ols))]],
                      "A large gap between the two scales, or a large coefficient shift, signals that outliers matter."))
    # leverage
    Hh = X @ np.linalg.solve(X.T @ X, X.T); lev = np.diag(Hh)
    lv = np.argsort(-lev)[:min(5, n)]
    tabs.append(table("Highest-leverage observations", ["Observation", "Leverage", "Rule of thumb 2p/n", "Robust standardised residual"], [[lab[i], float(lev[i]), float(2 * p / n), float(zr[i])] for i in lv],
                      "Leverage measures the position in x-space only. A bad leverage point has high leverage and a large residual."))
    ch = [chart("Robust standardised residuals", [lab[i] if n <= 60 else str(i + 1) for i in range(n)], [("MM residual / scale", zr)], "Observation", "Standardised residual", "bar")]
    return result("robreg", "Robust regression (M and MM estimators)", n, [stat("Observations", n), stat("Flagged outliers", int(flag.sum())), stat("MM scale", float(s_s), 4), stat("OLS scale", s_ols, 4),
                                                                           stat("Largest MM weight loss", float(1 - w_mm.min()), 3)], tabs, ch,
                  ["Huber (1964) M estimation; Yohai (1987) MM estimation: an S estimate of scale (Rousseeuw and Yohai 1984) from 500 random elemental subsets, then a bisquare M step at 95% efficiency. The standard errors of the MM estimator are the asymptotic ones.",
                   "Robust regression downweights observations; it does not say they are errors. Examine the flagged observations before deciding what they are."])


# ================================================================================================ cluster-robust inference and the wild cluster bootstrap
def _cluster_meat(X, e, codes, G):
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, codes, X * e[:, None])
    return S.T @ S, S


def wildboot(df, spec):
    yv = need(spec, "y", "outcome")
    xs = as_list(need(spec, "x", "regressors"))
    cl = need(spec, "cluster", "cluster identifier")
    distinct([yv], xs)
    tv = spec.get("test") or xs[0]
    if tv not in xs:
        raise OESError("The tested variable must be one of the regressors.")
    reps = int_option(spec, "reps", 999, 99, 20000, "bootstrap replications")
    seed = int_option(spec, "seed", 0, 0, 10 ** 9, "seed")
    wtype = str(spec.get("weights", "rademacher"))
    nulls = float_option(spec, "null", 0.0, "value under the null")
    d = numeric(df, [yv] + xs, keep=[cl])
    codes_raw = d[cl].astype(str).to_numpy()
    uniq, codes = np.unique(codes_raw, return_inverse=True)
    G = len(uniq); n = len(d); k = len(xs) + 1
    if G < 5:
        raise OESError(f"Only {G} clusters: at least 5 are needed.")
    if wtype == "rademacher" and G < 12 and 2 ** G < reps:
        reps = min(reps, 2 ** G)
    X = np.column_stack([np.ones(n), d[xs].to_numpy(float)]); y = d[yv].to_numpy(float)
    names = ["Constant"] + xs
    j = names.index(tv)
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y; e = y - X @ b
    # variance estimators
    s2 = e @ e / (n - k)
    V_ols = s2 * XtXi
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    V_hc1 = XtXi @ ((X * e[:, None]).T @ (X * e[:, None])) * n / (n - k) @ XtXi
    M, S = _cluster_meat(X, e, codes, G)
    c1 = G / (G - 1) * (n - 1) / (n - k)
    V_cr1 = c1 * XtXi @ M @ XtXi
    # CR3: delete-one-cluster jackknife
    bj = np.zeros((G, k))
    for g in range(G):
        m_ = codes != g
        bj[g] = np.linalg.lstsq(X[m_], y[m_], rcond=None)[0]
    dev = bj - bj.mean(0)
    V_cr3 = (G - 1) / G * dev.T @ dev
    se = {"Conventional (homoskedastic)": np.sqrt(V_ols[j, j]), "Heteroskedasticity-robust (HC1)": np.sqrt(V_hc1[j, j]), "Cluster-robust (CR1)": np.sqrt(V_cr1[j, j]), "Cluster jackknife (CR3)": np.sqrt(V_cr3[j, j])}
    bj_ = b[j]
    t0 = (bj_ - nulls) / se["Cluster-robust (CR1)"]
    # wild cluster bootstrap-t, null imposed (WCR), Cameron, Gelbach and Miller (2008)
    rng = np.random.default_rng(seed)
    others = [i for i in range(k) if i != j]
    Xr = X[:, others]
    yr = y - nulls * X[:, j]
    br = np.linalg.lstsq(Xr, yr, rcond=None)[0]
    er = yr - Xr @ br
    fr = Xr @ br + nulls * X[:, j]
    if wtype == "webb":
        vals = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
    else:
        vals = np.array([-1.0, 1.0])
    tb = np.empty(reps)
    for r in range(reps):
        if wtype == "rademacher" and 2 ** G <= reps:                  # enumerate when few clusters
            v = np.where(((r >> np.arange(G)) & 1) == 1, 1.0, -1.0)
        else:
            v = vals[rng.integers(0, len(vals), G)]
        ys = fr + er * v[codes]
        bs = XtXi @ (X.T @ ys)
        es = ys - X @ bs
        Mb, _ = _cluster_meat(X, es, codes, G)
        Vb = c1 * XtXi @ Mb @ XtXi
        tb[r] = (bs[j] - nulls) / np.sqrt(Vb[j, j])
    p_wild = float(np.mean(np.abs(tb) >= abs(t0) - 1e-12))
    p_ols = float(2 * stats.t.sf(abs((bj_ - nulls) / se["Conventional (homoskedastic)"]), n - k))
    p_hc = float(2 * stats.t.sf(abs((bj_ - nulls) / se["Heteroskedasticity-robust (HC1)"]), n - k))
    p_cr1 = float(2 * stats.t.sf(abs(t0), G - 1))
    p_cr3 = float(2 * stats.t.sf(abs((bj_ - nulls) / se["Cluster jackknife (CR3)"]), G - 1))
    # bootstrap-t confidence interval (unrestricted)
    tu = np.empty(reps)
    e_u = e
    for r in range(reps):
        v = vals[rng.integers(0, len(vals), G)]
        ys = X @ b + e_u * v[codes]
        bs = XtXi @ (X.T @ ys); es = ys - X @ bs
        Mb, _ = _cluster_meat(X, es, codes, G)
        Vb = c1 * XtXi @ Mb @ XtXi
        tu[r] = (bs[j] - b[j]) / np.sqrt(Vb[j, j])
    q = np.quantile(np.abs(tu), 0.95)
    ci_w = (bj_ - q * se["Cluster-robust (CR1)"], bj_ + q * se["Cluster-robust (CR1)"])
    tcrit = stats.t.ppf(0.975, G - 1)
    sizes = np.bincount(codes)
    rows = [[lab, float(sv), float((bj_ - nulls) / sv), pv] for (lab, sv), pv in zip(se.items(), (p_ols, p_hc, p_cr1, p_cr3))]
    rows.append(["Wild cluster bootstrap-t (null imposed)", float(se["Cluster-robust (CR1)"]), float(t0), p_wild])
    tabs = [table(f"Inference on {tv}", ["Method", "Standard error", "t", "p-value"], rows,
                  f"Coefficient {bj_:.4f}; {n} observations in {G} clusters (sizes {sizes.min()} to {sizes.max()}). p-values use t with {n - k} degrees of freedom for the first two, t with {G - 1} for the cluster methods, and {reps} bootstrap draws ({wtype} weights) for the last."),
            table(f"95% confidence intervals for {tv}", ["Method", "Lower", "Upper", "Width"],
                  [["Conventional", float(bj_ - stats.t.ppf(.975, n - k) * se["Conventional (homoskedastic)"]), float(bj_ + stats.t.ppf(.975, n - k) * se["Conventional (homoskedastic)"]), float(2 * stats.t.ppf(.975, n - k) * se["Conventional (homoskedastic)"])],
                   ["Cluster-robust CR1, t(G-1)", float(bj_ - tcrit * se["Cluster-robust (CR1)"]), float(bj_ + tcrit * se["Cluster-robust (CR1)"]), float(2 * tcrit * se["Cluster-robust (CR1)"])],
                   ["Wild cluster bootstrap-t", float(ci_w[0]), float(ci_w[1]), float(ci_w[1] - ci_w[0])]]),
            table("Clusters", ["Quantity", "Value"], [["Number of clusters", G], ["Smallest cluster", int(sizes.min())], ["Largest cluster", int(sizes.max())], ["Largest share of observations in one cluster", float(sizes.max() / n)],
                                                       ["Ratio of the cluster SE to the conventional SE", float(se["Cluster-robust (CR1)"] / se["Conventional (homoskedastic)"])],
                                                       ["Within-cluster cross-products of the scores relative to their squares (0 if independent)", float(np.sum(S[:, j] ** 2 - np.bincount(codes, weights=(X[:, j] * e) ** 2)) / max(np.sum((X[:, j] * e) ** 2), 1e-300))]],
                  "The variance of the estimate rises with cluster size and with the correlation of both the regressor and the error within clusters."),
            table("Coefficients", ["Variable", "Coefficient", "Cluster-robust s.e. (CR1)"], [[names[i], float(b[i]), float(np.sqrt(V_cr1[i, i]))] for i in range(k)])]
    hist_edges = np.linspace(np.percentile(tb, 0.5), np.percentile(tb, 99.5), 21)
    hh = np.histogram(np.clip(tb, hist_edges[0], hist_edges[-1]), bins=hist_edges)[0]
    ch = [chart("Distribution of the wild bootstrap t statistic", [f"{v:.2f}" for v in hist_edges[:-1]], [("Draws", hh)], "t", "Count", "bar")]
    return result("wildboot", "Cluster-robust inference and the wild cluster bootstrap", n, [stat("Observations", n), stat("Clusters", G), stat(f"{tv} coefficient", float(bj_), 4), stat("CR1 p-value", p_cr1, 4), stat("Wild bootstrap p-value", p_wild, 4)], tabs, ch,
                  ["Cameron, Gelbach and Miller (2008): the wild cluster bootstrap-t imposes the null, flips the sign of each cluster's residuals at random (Rademacher or Webb weights) and recomputes the cluster-robust t statistic. It is more reliable than the t(G-1) test when there are few clusters or one cluster is large.",
                   "With fewer than about 12 clusters the Rademacher draws are enumerated; use Webb weights for fewer than 10 (Webb 2014). Clusters must be defined by the level at which treatment or the regressor varies and errors are correlated (Abadie, Athey, Imbens and Wooldridge 2023)."])


# ================================================================================================ kernel density estimation
def _sj_bw(x):
    x = np.sort(x)
    n = len(x)
    s = min(x.std(ddof=1), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349)
    # work on binned pairwise differences
    nb = 1000
    lo, hi = x.min(), x.max()
    dgrid = (hi - lo) / nb
    idx = np.floor((x - lo) / dgrid).astype(int).clip(0, nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(float)
    # pairwise count by lag
    lags = np.arange(nb)
    pc = np.zeros(nb)
    for l in range(nb):
        pc[l] = np.dot(cnt[:nb - l], cnt[l:]) if l else (np.dot(cnt, cnt) - n) / 1.0
    delta = lags * dgrid
    phi0 = 1 / np.sqrt(2 * np.pi)

    def phi4(t): return (t ** 4 - 6 * t ** 2 + 3) * np.exp(-t ** 2 / 2) * phi0
    def phi6(t): return (t ** 6 - 15 * t ** 4 + 45 * t ** 2 - 15) * np.exp(-t ** 2 / 2) * phi0
    def SD(h):
        t = delta / h
        return (pc[0] * phi4(0.0) + 2 * np.sum(pc[1:] * phi4(t[1:]))) / (n * (n - 1) * h ** 5)
    def TD(h):
        t = delta / h
        return -(pc[0] * phi6(0.0) + 2 * np.sum(pc[1:] * phi6(t[1:]))) / (n * (n - 1) * h ** 7)
    a = 1.24 * s * n ** (-1 / 7); b = 0.7 * s * n ** (-1 / 9)
    c1 = 1 / (2 * np.sqrt(np.pi) * n)
    td_b = TD(b); sd_a = SD(a)
    if not (td_b > 0):
        td_b = 15 / (16 * np.sqrt(np.pi) * s ** 7)             # normal-reference value of the sixth-derivative functional
    if not (sd_a > 0):
        sd_a = 3 / (8 * np.sqrt(np.pi) * s ** 5)
    alph2 = 1.357 * (sd_a / td_b) ** (1 / 7)
    def fsd(h):
        v = SD(alph2 * h ** (5 / 7))
        if not (v > 0):
            v = 3 / (8 * np.sqrt(np.pi) * s ** 5)
        return (c1 / v) ** (1 / 5) - h
    lo_b = 0.05 * s * n ** (-0.2); hi_b = 1.5 * 1.144 * s * n ** (-0.2)
    try:
        return float(optimize.brentq(fsd, lo_b, hi_b))
    except ValueError:
        return float(optimize.minimize_scalar(lambda h: abs(fsd(h)), bounds=(lo_b, hi_b), method="bounded").x)


def _lscv(x, h):
    n = len(x)
    d = (x[:, None] - x[None, :])
    t = d / h
    term1 = np.sum(np.exp(-t ** 2 / 4)) / (2 * np.sqrt(np.pi))
    term2 = np.sum(np.exp(-t ** 2 / 2)) / np.sqrt(2 * np.pi) - n / np.sqrt(2 * np.pi)
    return term1 / (n ** 2 * h) - 2 * term2 / (n * (n - 1) * h)


def _kde(x, grid, h):
    return np.exp(-0.5 * ((grid[:, None] - x[None, :]) / h) ** 2).sum(1) / (len(x) * h * np.sqrt(2 * np.pi))


def _modes(dens):
    return int(np.sum((dens[1:-1] > dens[:-2]) & (dens[1:-1] > dens[2:])))


def kde(df, spec):
    xv = need(spec, "x", "variable")
    reps = int_option(spec, "reps", 100, 0, 1000, "bootstrap replications")
    seed = int_option(spec, "seed", 0, 0, 10 ** 9, "seed")
    take_log = str(spec.get("log", "no")) == "yes"
    d = numeric(df, [xv])
    x = d[xv].to_numpy(float)
    if take_log:
        if (x <= 0).any():
            raise OESError("Logs need positive values.")
        x = np.log(x)
    n = len(x)
    if n < 30:
        raise OESError(f"Only {n} observations: at least 30 are needed.")
    if n > 5000:
        raise OESError("More than 5,000 observations: use a random subsample for bandwidth selection.")
    s = x.std(ddof=1); iqr = np.percentile(x, 75) - np.percentile(x, 25)
    bws = {"Scott (1.06 s n^-1/5)": 1.06 * s * n ** (-0.2), "Silverman (0.9 min(s, IQR/1.349) n^-1/5)": 0.9 * min(s, iqr / 1.349) * n ** (-0.2)}
    bws["Sheather-Jones plug-in"] = _sj_bw(x)
    hg = np.exp(np.linspace(np.log(0.05 * s), np.log(1.2 * s), 60))
    lc = np.array([_lscv(x, h) for h in hg])
    bws["Least-squares cross-validation"] = float(hg[int(np.argmin(lc))])
    user = spec.get("bandwidth")
    h = float(user) if user not in (None, "") else bws["Sheather-Jones plug-in"]
    if h <= 0:
        raise OESError("The bandwidth must be positive.")
    grid = np.linspace(x.min() - 3 * h, x.max() + 3 * h, 400)
    dens = _kde(x, grid, h)
    area = float(np.sum(dens) * (grid[1] - grid[0]))
    mode_pos = [float(grid[i + 1]) for i in range(len(grid) - 2) if dens[i + 1] > dens[i] and dens[i + 1] > dens[i + 2]]
    rows = []
    for lab, bw in bws.items():
        dd = _kde(x, grid, bw)
        rows.append([lab, float(bw), _modes(dd), float(np.max(dd))])
    tabs = [table("Bandwidth selectors", ["Selector", "Bandwidth", "Modes", "Peak density"], rows,
                  f"Gaussian kernel, {n} observations{' (log scale)' if take_log else ''}. Normal-reference rules (Scott, Silverman) over-smooth multimodal data; Sheather-Jones and cross-validation adapt to the shape.")]
    # Silverman test for the number of modes
    p_modes = None; hcrit = None
    kk = int(spec.get("modes", 1))
    if reps > 0:
        gr = np.linspace(x.min() - 3 * s * 0.3, x.max() + 3 * s * 0.3, 300)
        def crit(sample, kmodes):
            lo, hi = 0.01 * sample.std(ddof=1), 2 * sample.std(ddof=1)
            for _ in range(40):
                mid = (lo * hi) ** 0.5
                if _modes(_kde(sample, gr, mid)) > kmodes:
                    lo = mid
                else:
                    hi = mid
            return hi
        hcrit = crit(x, kk)
        rng = np.random.default_rng(seed)
        cnt = 0
        for _ in range(reps):
            xb = x[rng.integers(0, n, n)] + hcrit * rng.normal(size=n)
            xb = x.mean() + (xb - x.mean()) / np.sqrt(1 + hcrit ** 2 / x.var())
            if _modes(_kde(xb, gr, hcrit)) > kk:
                cnt += 1
        p_modes = (cnt) / reps
        tabs.append(tests_table([test_row(f"Silverman test: at most {kk} mode{'s' if kk > 1 else ''}", float(hcrit), float(p_modes), f"Critical bandwidth h_crit = {hcrit:.4g}; p is the share of {reps} smoothed bootstrap samples that need more than {kk} mode(s) at h_crit")], "Test of the number of modes"))
    tabs.append(table("Density estimate and modes", ["Quantity", "Value"], [["Bandwidth used", float(h)], ["Modes at this bandwidth", len(mode_pos)]] + [[f"Mode {i + 1} at", m] for i, m in enumerate(mode_pos[:6])] +
                      [["Mean", float(x.mean())], ["Median", float(np.median(x))], ["Skewness", float(stats.skew(x))], ["Excess kurtosis", float(stats.kurtosis(x))], ["Area under the estimate (check)", area]]))
    xs = np.linspace(x.min(), x.max(), 100)
    series = [("Selected bandwidth", np.interp(xs, grid, dens))]
    for lab in ("Scott (1.06 s n^-1/5)", "Least-squares cross-validation"):
        series.append((lab.split(" (")[0], np.interp(xs, grid, _kde(x, grid, bws[lab]))))
    series.append(("Normal with the same mean and s.d.", stats.norm.pdf(xs, x.mean(), s)))
    ch = [chart("Kernel density estimates", [f"{v:.3g}" for v in xs], series, xv, "Density", "line")]
    return result("kde", "Kernel density estimation", n, [stat("Observations", n), stat("Bandwidth", float(h), 4), stat("Modes", len(mode_pos)), stat("Sheather-Jones bandwidth", float(bws["Sheather-Jones plug-in"]), 4),
                                                         stat("Modes test p-value", p_modes if p_modes is not None else "n/a", 3)], tabs, ch,
                  ["Gaussian kernel density estimate with the bandwidth chosen by the Sheather-Jones (1991) solve-the-equation plug-in, which is the recommended default; Scott and Silverman rules and least-squares cross-validation are shown for comparison.",
                   "The test of the number of modes is Silverman (1981): the critical bandwidth is the smallest at which the estimate has at most k modes, and its significance is judged by the smoothed bootstrap (with the variance correction of Efron and Tibshirani)."])


# ================================================================================================ multiple testing
def _pi0(p, lam=None):
    if lam is None:
        lams = np.arange(0.05, 0.951, 0.05)
        est = np.array([min(1.0, np.mean(p > l) / (1 - l)) for l in lams])
        # smooth with a cubic polynomial in lambda and read the value at the largest lambda (Storey and Tibshirani 2003)
        coef = np.polyfit(lams, est, 3)
        return float(min(max(np.polyval(coef, lams[-1]), 0.0), 1.0)), lams, est
    return float(min(1.0, np.mean(p > lam) / (1 - lam))), None, None


def fdr(df, spec):
    pv = need(spec, "p", "column of p-values")
    alpha = float_option(spec, "alpha", 0.05, "level (alpha)")
    lab = spec.get("label") or None
    d = numeric(df, [pv], keep=[lab] if lab else [])
    p = d[pv].to_numpy(float)
    if ((p < 0) | (p > 1)).any():
        raise OESError("The p-values must lie between 0 and 1.")
    m = len(p)
    if m < 10:
        raise OESError(f"Only {m} p-values: at least 10 are needed.")
    if not 0 < alpha < 1:
        raise OESError("alpha must be between 0 and 1.")
    order = np.argsort(p); ps = p[order]
    ranks = np.arange(1, m + 1)
    bonf = np.minimum(ps * m, 1.0)
    holm = np.minimum(np.maximum.accumulate((m - ranks + 1) * ps), 1.0)
    hoch = np.minimum.accumulate(np.minimum(((m - ranks + 1) * ps)[::-1], 1.0))[::-1]
    bh = np.minimum.accumulate((ps * m / ranks)[::-1])[::-1]; bh = np.minimum(bh, 1.0)
    cm = np.sum(1.0 / ranks)
    by = np.minimum.accumulate((ps * m * cm / ranks)[::-1])[::-1]; by = np.minimum(by, 1.0)
    pi0, lams, est = _pi0(p)
    qv = np.minimum.accumulate((ps * m * pi0 / ranks)[::-1])[::-1]; qv = np.minimum(qv, 1.0)
    adj = {"Bonferroni": bonf, "Holm": holm, "Hochberg": hoch, "Benjamini-Hochberg": bh, "Benjamini-Yekutieli": by, "Storey q-value": qv}
    ctl = {"Bonferroni": "family-wise error", "Holm": "family-wise error", "Hochberg": "family-wise error (independent or positively dependent tests)", "Benjamini-Hochberg": "false discovery rate (independent or positively dependent tests)",
           "Benjamini-Yekutieli": "false discovery rate (any dependence)", "Storey q-value": "false discovery rate (estimated null proportion)"}
    rows = [["Unadjusted p-value", int((p <= alpha).sum()), "none"]] + [[k, int((v <= alpha).sum()), ctl[k]] for k, v in adj.items()]
    tabs = [table(f"Number of rejections at level {alpha:g}", ["Method", "Rejections", "Error rate controlled"], rows, f"{m} tests. A rejection at level alpha means an adjusted p-value (or q-value) at most alpha.")]
    levels = [0.01, 0.05, 0.10, 0.20]
    tabs.append(table("Rejections at several levels", ["Method"] + [f"alpha = {a:g}" for a in levels], [["Unadjusted"] + [int((p <= a).sum()) for a in levels]] + [[k] + [int((v <= a).sum()) for a in levels] for k, v in adj.items()]))
    exp_fp = alpha * m
    tabs.append(table("Null proportion and false discoveries", ["Quantity", "Value"],
                      [["Estimated share of true null hypotheses (pi0, Storey)", pi0], ["Estimated number of true null hypotheses", float(pi0 * m)], ["Estimated number of false hypotheses", float((1 - pi0) * m)],
                       [f"False rejections expected with no correction at alpha = {alpha:g} if all were null", float(exp_fp)], [f"Unadjusted rejections at alpha = {alpha:g}", int((p <= alpha).sum())],
                       ["Expected false discoveries among the BH rejections (alpha x rejections)", float(alpha * (bh <= alpha).sum())]],
                      "pi0 is the height of the p-value histogram where it is flat (near 1), estimated by smoothing 1 - F(lambda) over lambda."))
    names = (d[lab].astype(str).to_numpy()[order] if lab else np.array([str(i + 1) for i in order]))
    top = min(15, m)
    tabs.append(table("The smallest p-values", ["Test", "p-value", "Bonferroni", "Holm", "BH", "BY", "q-value"], [[names[i], float(ps[i]), float(bonf[i]), float(holm[i]), float(bh[i]), float(by[i]), float(qv[i])] for i in range(top)]))
    nb = 20
    hist = np.histogram(p, bins=np.linspace(0, 1, nb + 1))[0]
    ch = [chart("Histogram of p-values", [f"{i / nb:.2f}" for i in range(nb)], [("Tests", hist), ("Expected under the null (pi0 x m / bins)", np.full(nb, pi0 * m / nb))], "p-value", "Count", "bar"),
          chart("Sorted p-values and the Benjamini-Hochberg line", [str(i) for i in range(1, min(m, 400) + 1)], [("Sorted p-value", ps[:400]), (f"alpha x rank / m ({alpha:g})", alpha * ranks[:400] / m)], "Rank", "p-value", "line")]
    return result("fdr", "Multiple testing: family-wise error and false discovery rate", m, [stat("Tests", m), stat("Unadjusted rejections", int((p <= alpha).sum())), stat("Bonferroni", int((bonf <= alpha).sum())), stat("Benjamini-Hochberg", int((bh <= alpha).sum())),
                                                                                         stat("Estimated pi0", float(pi0), 3)], tabs, ch,
                  ["Bonferroni, Holm (1979) and Hochberg (1988) control the probability of any false rejection; Benjamini and Hochberg (1995) and Benjamini and Yekutieli (2001) control the expected share of false rejections among the rejections; Storey (2002) q-values use the estimated share of true nulls.",
                   "The p-values must come from tests that are valid one by one. Under strong dependence only the Benjamini-Yekutieli adjustment is guaranteed."])


# ================================================================================================ extreme value analysis
def _gpd_nll(par, y):
    xi, ls = par
    s = np.exp(ls)
    if abs(xi) < 1e-8:
        return float(len(y) * ls + np.sum(y) / s)
    z = 1 + xi * y / s
    if np.any(z <= 0):
        return 1e12
    return float(len(y) * ls + (1 + 1 / xi) * np.sum(np.log(z)))


def _gpd_fit(y):
    best = None
    for x0 in ([0.1, np.log(np.mean(y))], [0.5, np.log(np.mean(y))], [0.0, np.log(np.mean(y))], [-0.2, np.log(np.max(y))]):
        o = optimize.minimize(_gpd_nll, x0, args=(y,), method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-11, maxiter=4000))
        if best is None or o.fun < best.fun:
            best = o
    xi, ls = best.x
    s = float(np.exp(ls))
    # observed information in (xi, sigma)
    def nll2(p):
        return _gpd_nll([p[0], np.log(p[1])], y)
    from .productivity import _num_hessian
    H = _num_hessian(nll2, np.array([xi, s]), eps=1e-4)
    try:
        V = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        V = np.full((2, 2), np.nan)
    return float(xi), s, V, float(best.fun)


def _hill(xs_desc, k):
    return float(np.mean(np.log(xs_desc[:k])) - np.log(xs_desc[k]))


def evt(df, spec):
    xv = need(spec, "x", "variable")
    tail = str(spec.get("tail", "upper"))
    take_ret = str(spec.get("returns", "no")) == "yes"
    d = numeric(df, [xv])
    x = d[xv].to_numpy(float)
    if take_ret:
        if (x <= 0).any():
            raise OESError("Log returns need positive prices.")
        x = np.diff(np.log(x))
    if tail == "lower":
        x = -x
    n = len(x)
    if n < 200:
        raise OESError(f"Only {n} observations: at least 200 are needed for a tail analysis.")
    thr_user = spec.get("threshold")
    q = float_option(spec, "quantile", 0.95, "threshold quantile")
    u = float(thr_user) if thr_user not in (None, "") else float(np.quantile(x, q))
    exc = x[x > u] - u
    nu = len(exc)
    if nu < 30:
        raise OESError(f"Only {nu} exceedances over the threshold {u:.4g}: lower the threshold (at least 30 are needed).")
    xi, sig, V, nll = _gpd_fit(exc)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    zeta = nu / n
    rows = [["Threshold u", u, None], ["Exceedances", nu, None], ["Share of observations above u", float(zeta), None], ["Shape xi", xi, float(se[0])], ["Scale sigma", sig, float(se[1])], ["Log-likelihood", -nll, None]]
    tabs = [table("Generalised Pareto fit to the exceedances", ["Quantity", "Estimate", "s.e."], rows,
                  f"{'Lower' if tail == 'lower' else 'Upper'} tail of {xv}{' (log returns)' if take_ret else ''}; {n} observations. A positive xi means a heavy (power-law) tail with tail index 1/xi = {1 / xi:.2f}." if xi > 0 else f"{'Lower' if tail == 'lower' else 'Upper'} tail of {xv}; xi is not positive, so the tail is not heavy.")]
    # threshold stability
    qs = [0.85, 0.90, 0.925, 0.95, 0.975, 0.99]
    st = []
    for qq in qs:
        uu = float(np.quantile(x, qq)); e2 = x[x > uu] - uu
        if len(e2) >= 20:
            try:
                x2, s2, V2, _ = _gpd_fit(e2)
                st.append([f"{qq:g}", uu, len(e2), x2, float(np.sqrt(max(V2[0, 0], 0))), s2 - x2 * (uu - u) if False else s2])
            except Exception:
                pass
    tabs.append(table("Stability of the shape across thresholds", ["Threshold quantile", "Threshold", "Exceedances", "xi", "s.e. of xi", "Scale sigma"], st, "A shape estimate that settles as the threshold rises supports the choice of threshold; a drift means the generalised Pareto approximation is poor."))
    # Hill
    xs_desc = np.sort(x[x > 0])[::-1]
    ks = [k for k in (20, 40, 60, 100, 150, 250) if k < len(xs_desc)]
    tabs.append(table("Hill estimator of the tail index", ["Top k observations", "Hill estimate of xi", "Tail index alpha = 1/xi"], [[k, _hill(xs_desc, k), 1 / _hill(xs_desc, k)] for k in ks],
                      "Valid for positive-valued heavy-tailed data; it needs a flat region in k."))
    # return levels and VaR/ES
    def rl(m_):
        return u + sig / xi * ((m_ * zeta) ** xi - 1) if abs(xi) > 1e-8 else u + sig * np.log(m_ * zeta)
    def rl_se(m_):
        g = np.array([-(sig / xi ** 2) * ((m_ * zeta) ** xi - 1) + (sig / xi) * (m_ * zeta) ** xi * np.log(m_ * zeta), (1 / xi) * ((m_ * zeta) ** xi - 1)])
        v = float(g @ V @ g) + (sig * (m_ * zeta) ** xi / zeta) ** 2 * zeta * (1 - zeta) / n
        return float(np.sqrt(max(v, 0)))
    horizons = [100, 250, 1000, 2500, 10000]
    rrows = []
    for m_ in horizons:
        if m_ * zeta > 1:
            r_ = rl(m_); s_ = rl_se(m_)
            rrows.append([f"1 in {m_} observations", float(r_), float(s_), float(r_ - 1.96 * s_), float(r_ + 1.96 * s_), int((x > r_).sum())])
    tabs.append(table("Return levels (delta-method 95% intervals)", ["Return period", "Level", "s.e.", "Lower", "Upper", "Observed exceedances"], rrows, "The level exceeded once on average in the stated number of observations. Intervals for long return periods are wide and symmetric by construction; profile likelihood intervals would be asymmetric."))
    vrows = []
    for pr in (0.99, 0.995, 0.999):
        if (1 - pr) < zeta:
            var = u + sig / xi * (((1 - pr) / zeta) ** (-xi) - 1) if abs(xi) > 1e-8 else u - sig * np.log((1 - pr) / zeta)
            es = (var + sig - xi * u) / (1 - xi) if xi < 1 else np.nan
            emp = float(np.quantile(x, pr))
            vrows.append([f"{pr:g}", float(var), float(emp), float(es) if np.isfinite(es) else None, float(x[x > var].mean()) if (x > var).any() else None])
    tabs.append(table("Value at risk and expected shortfall from the tail model", ["Level", "VaR (GPD)", "VaR (empirical quantile)", "ES (GPD)", "ES (empirical)"], vrows, "McNeil and Frey (2000). Expected shortfall is infinite when xi is 1 or more."))
    # goodness of fit: Anderson-Darling-type via probability integral transform and KS
    pit = 1 - (1 + xi * np.sort(exc) / sig) ** (-1 / xi) if abs(xi) > 1e-8 else 1 - np.exp(-np.sort(exc) / sig)
    ks_stat, ks_p = stats.kstest(pit, "uniform")
    eg = np.sort(exc)
    tabs.append(tests_table([test_row("Kolmogorov-Smirnov test of the generalised Pareto fit", float(ks_stat), float(ks_p), "p-value is approximate: the parameters are estimated")], "Goodness of fit"))
    # mean excess
    ths = np.quantile(x, np.linspace(0.5, 0.99, 30))
    me = [np.mean(x[x > t] - t) if (x > t).sum() > 5 else np.nan for t in ths]
    pp = (np.arange(1, nu + 1) - 0.5) / nu
    qth = u + sig / xi * ((1 - pp) ** (-xi) - 1) if abs(xi) > 1e-8 else u - sig * np.log(1 - pp)
    ch = [chart("Mean excess function", [f"{t:.3g}" for t in ths], [("Mean excess", me)], "Threshold", "Mean excess", "line"),
          chart("Exceedances against the fitted generalised Pareto quantiles", [f"{v:.3g}" for v in qth], [("Observed (sorted)", u + eg), ("Model", qth)], "Model quantile", "Observed", "line")]
    return result("evt", "Extreme value analysis (peaks over threshold)", n, [stat("Observations", n), stat("Threshold", u, 4), stat("Exceedances", nu), stat("Shape xi", xi, 3), stat("Scale sigma", sig, 3)], tabs, ch,
                  ["Peaks over threshold (Pickands 1975; Balkema and de Haan 1974): exceedances of a high threshold follow a generalised Pareto distribution, fitted by maximum likelihood, with standard errors from the observed information.",
                   "Return levels follow McNeil (1997): x_m = u + (sigma/xi)((m zeta_u)^xi - 1), where zeta_u is the share of observations above the threshold. Extrapolation far beyond the data rests on the model and is uncertain."])
