"""Category C of the Statistics Division: multivariate analysis.

PCA, exploratory factor analysis (maximum likelihood with varimax, promax and oblimin rotations), confirmatory factor analysis
(maximum likelihood covariance structure), linear discriminant analysis, canonical correlation analysis and correspondence
analysis. All decompositions are written out with NumPy; optimisation uses SciPy's L-BFGS-B and BFGS."""
import re

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

from .core import OESError, as_list, chart, distinct, f, float_option, int_option, need, result, stat, table, tests_table
from .stat_core import clean_frame, levels, thin


def _alpha(spec):
    a = float_option(spec, "alpha", 0.05, "Significance level")
    if not 0 < a < 0.5:
        raise OESError("The significance level must be between 0 and 0.5.")
    return a


def _ci(a):
    return f"{int(round((1 - a) * 100))}%"


def _num_matrix(df, vars_, min_vars=2, min_rows=None):
    vars_ = as_list(vars_)
    if len(vars_) < min_vars:
        raise OESError(f"Choose at least {min_vars} variables.")
    distinct(vars_)
    d, dropped = clean_frame(df, numeric=vars_)
    X = d[vars_].to_numpy(float)
    if min_rows and len(X) < min_rows:
        raise OESError(f"At least {min_rows} complete rows are needed; {len(X)} remain.")
    if (X.std(0, ddof=1) == 0).any():
        raise OESError("A variable is constant: " + ", ".join(v for v, s in zip(vars_, X.std(0, ddof=1)) if s == 0))
    return d, X, vars_, dropped


def _kmo(R):
    Ri = np.linalg.pinv(R)
    d = np.sqrt(np.abs(np.diag(Ri)))
    Q = -Ri / np.outer(d, d)
    np.fill_diagonal(Q, 0)
    R0 = R.copy()
    np.fill_diagonal(R0, 0)
    r2, q2 = (R0 ** 2).sum(0), (Q ** 2).sum(0)
    return float(r2.sum() / (r2.sum() + q2.sum())), r2 / (r2 + q2)


def _bartlett_sphericity(R, n):
    p = R.shape[0]
    ld = np.linalg.slogdet(R)[1]
    chi = -(n - 1 - (2 * p + 5) / 6.0) * ld
    dfb = p * (p - 1) / 2
    return float(chi), dfb, float(stats.chi2.sf(chi, dfb))


def _parallel(n, p, sims, seed, basis_corr=True):
    rng = np.random.default_rng(seed)
    ev = np.empty((sims, p))
    for s in range(sims):
        Z = rng.standard_normal((n, p))
        ev[s] = np.sort(np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False)))[::-1]
    return np.percentile(ev, 95, axis=0)


# ---------------------------------------------------------------------------------------------------------------
# C1. PCA
# ---------------------------------------------------------------------------------------------------------------
def pca(df, spec):
    d, X, vs, dropped = _num_matrix(df, need(spec, "vars", "variables"), 2, 3)
    n, p = X.shape
    basis = spec.get("basis") or "correlation"
    seed = int_option(spec, "seed", 0, 0, 10 ** 9, "Random seed")
    Xc = X - X.mean(0)
    sd = X.std(0, ddof=1)
    Z = Xc / sd if basis == "correlation" else Xc
    C = np.cov(Z, rowvar=False)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = np.clip(w[order], 0, None), V[:, order]
    for j in range(p):                                  # sign convention: largest loading positive
        if V[np.argmax(np.abs(V[:, j])), j] < 0:
            V[:, j] = -V[:, j]
    tot = w.sum()
    ratio = w / tot
    cum = np.cumsum(ratio)
    bs = np.array([sum(1.0 / i for i in range(j, p + 1)) / p for j in range(1, p + 1)])
    pa = _parallel(n, p, 200, seed) if basis == "correlation" else None
    k_kaiser = int((w > (1 if basis == "correlation" else w.mean())).sum())
    k_pa = int(np.sum(w > pa)) if pa is not None else None
    k_80 = int(np.searchsorted(cum, 0.80) + 1)
    k = spec.get("n_components")
    if k in (None, ""):
        k = max(1, k_pa if k_pa else k_kaiser)
    k = int(k)
    if not 1 <= k <= p:
        raise OESError(f"The number of components must be between 1 and {p}.")
    L = V * np.sqrt(w)
    scores = Z @ V
    rows = [[f"PC{i + 1}", f(w[i], 5), f(ratio[i] * 100, 3), f(cum[i] * 100, 3), f(bs[i] * 100, 3), f(pa[i], 4) if pa is not None else None, "retain" if i < k else ""] for i in range(p)]
    t_ev = table("Eigenvalues and variance explained", ["Component", "Eigenvalue", "% of variance", "Cumulative %", "Broken-stick %", "Parallel-analysis 95th pct.", f"Retained (k = {k})"], rows,
                 "Retention rules: Kaiser (eigenvalue above " + ("1" if basis == "correlation" else "the mean") + f") suggests {k_kaiser}; Horn's parallel analysis suggests {k_pa if k_pa is not None else 'n/a (covariance basis)'}; 80% of the variance needs {k_80}.")
    t_load = table(f"Loadings (correlation of each variable with each of the first {k} components)" if basis == "correlation" else f"Loadings (eigenvectors scaled by the root of the eigenvalues), first {k} components",
                   ["Variable"] + [f"PC{i + 1}" for i in range(k)] + ["Communality"], [[v] + [f(L[i, j], 4) for j in range(k)] + [f((L[i, :k] ** 2).sum() / (1 if basis == "correlation" else C[i, i]), 4)] for i, v in enumerate(vs)])
    t_vec = table("Eigenvectors (component weights)", ["Variable"] + [f"PC{i + 1}" for i in range(k)], [[v] + [f(V[i, j], 5) for j in range(k)] for i, v in enumerate(vs)])
    diag = []
    if basis == "correlation":
        R = np.corrcoef(X, rowvar=False)
        kmo, msa = _kmo(R)
        diag.append(["Kaiser-Meyer-Olkin measure of sampling adequacy", f(kmo, 4), None, "above 0.6 is acceptable for components or factors"])
        chi, dfb, pb = _bartlett_sphericity(R, n)
        diag.append(["Bartlett's test of sphericity", f(chi, 4), f(pb, 6), f"df = {dfb:.0f}; H0: the correlation matrix is the identity (no structure to summarise)"])
        diag.append(["Determinant of the correlation matrix", f(float(np.linalg.det(R)), 6), None, "close to zero signals multicollinearity"])
    diag.append(["Condition number of the analysed matrix", f(float(w[0] / max(w[-1], 1e-300)), 2), None, "very large values point to near-collinear variables"])
    tabs = [t_ev, t_load, t_vec, tests_table(diag, "Suitability and conditioning")]
    sc1, sc2 = thin(list(scores[:, 0]), list(scores[:, 1] if p > 1 else scores[:, 0]), 400)
    charts = [chart("Scree plot", [f"PC{i + 1}" for i in range(p)], [("Eigenvalue", w)] + ([("Parallel analysis", pa)] if pa is not None else []), "Component", "Eigenvalue"),
              chart("Component scores: PC1 against PC2", sc1, [("PC2", sc2)], "PC1", "PC2", kind="scatter"),
              chart("Cumulative variance explained (%)", [f"PC{i + 1}" for i in range(p)], [("Cumulative %", cum * 100)], "Component", "%", kind="bar")]
    notes = [f"{n} observations, {p} variables, {basis} matrix" + (f"; {dropped} row(s) with missing values were dropped." if dropped else "."),
             "Components are linear combinations of the variables that successively maximise variance subject to orthogonality. Standardise (correlation basis) when the variables have different units."]
    summary = [stat("PC1 variance %", ratio[0] * 100, 3), stat("Cumulative % (retained)", cum[k - 1] * 100, 3), stat("Components retained", k), stat("KMO", _kmo(np.corrcoef(X, rowvar=False))[0], 3) if basis == "correlation" else stat("Variables", p)]
    return result("stat_pca", "Principal component analysis (PCA)", n, summary, tabs, charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# C2. Exploratory factor analysis (maximum likelihood)
# ---------------------------------------------------------------------------------------------------------------
def _fa_objective(psi, S, m):
    sc = 1.0 / np.sqrt(psi)
    Ss = S * np.outer(sc, sc)
    ev = np.linalg.eigvalsh(Ss)[::-1]
    e = ev[m:]
    return float(-(np.sum(np.log(e) - e) - m + S.shape[0]))


def _fa_gradient(psi, S, m):
    sc = 1.0 / np.sqrt(psi)
    Ss = S * np.outer(sc, sc)
    w, V = np.linalg.eigh(Ss)
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    load = V[:, :m] * np.sqrt(np.maximum(w[:m] - 1, 0))
    load = np.sqrt(psi)[:, None] * load
    g = load @ load.T + np.diag(psi) - S
    return np.diag(g) / psi ** 2


def ml_factor_analysis(R, m, n):
    """Maximum-likelihood factor analysis of a correlation matrix (Joreskog 1967), as in R's factanal. Returns loadings, uniquenesses, objective."""
    p = R.shape[0]
    start = (1 - 0.5 * m / p) / np.diag(np.linalg.inv(R))
    res = optimize.minimize(_fa_objective, start, args=(R, m), jac=_fa_gradient, method="L-BFGS-B", bounds=[(0.005, 1.0)] * p, options={"maxiter": 1000})
    psi = res.x
    sc = 1.0 / np.sqrt(psi)
    Ss = R * np.outer(sc, sc)
    w, V = np.linalg.eigh(Ss)
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    L = np.sqrt(psi)[:, None] * (V[:, :m] * np.sqrt(np.maximum(w[:m] - 1, 0)))
    return L, psi, float(res.fun), bool(res.success)


def varimax(A, gamma=1.0, q=1000, tol=1e-5, normalize=True):
    A = np.asarray(A, float)
    p, k = A.shape
    if k < 2:
        return A, np.eye(k)
    h = np.sqrt((A ** 2).sum(1)) if normalize else np.ones(p)
    h[h == 0] = 1
    X = A / h[:, None]
    T = np.eye(k)
    d = 0.0
    for _ in range(q):
        Lm = X @ T
        u, s, vt = np.linalg.svd(X.T @ (Lm ** 3 - (gamma / p) * Lm @ np.diag((Lm ** 2).sum(0))))
        T = u @ vt
        dpast = d
        d = s.sum()
        if d < dpast * (1 + tol):
            break
    return (X @ T) * h[:, None], T


def promax(A, power=4):
    xx, Tv = varimax(A)
    Q = xx * np.abs(xx) ** (power - 1)
    U = np.linalg.lstsq(xx, Q, rcond=None)[0]
    dd = np.diag(np.linalg.inv(U.T @ U))
    U = U * np.sqrt(dd)
    z = xx @ U
    U = Tv @ U
    return z, U


def oblimin(A, gamma=0.0, maxit=1000, eps=1e-5):
    """Direct oblimin rotation by gradient projection (Bernaards and Jennrich 2005). Returns pattern loadings and the factor correlation matrix."""
    p, k = A.shape
    if k < 2:
        return A, np.eye(k)
    Tm = np.eye(k)

    def vgq(L):
        X = (L ** 2) @ (1 - np.eye(k))
        if gamma != 0:
            X = (np.eye(p) - gamma / p * np.ones((p, p))) @ X
        return float(np.sum(L ** 2 * X) / 4), L * X

    L = A @ np.linalg.inv(Tm).T
    fv, Gq = vgq(L)
    G = -(L.T @ Gq @ np.linalg.inv(Tm)).T
    al = 1.0
    for _ in range(maxit):
        Gp = G - Tm @ np.diag((Tm * G).sum(0))
        s = np.sqrt(np.sum(Gp ** 2))
        if s < eps:
            break
        al *= 2
        for _i in range(11):
            X = Tm - al * Gp
            v = 1 / np.sqrt((X ** 2).sum(0))
            Tmt = X * v
            Lt = A @ np.linalg.inv(Tmt).T
            ft, Gqt = vgq(Lt)
            if fv - ft > 0.5 * s ** 2 * al:
                break
            al /= 2
        Tm, L, fv, Gq = Tmt, Lt, ft, Gqt
        G = -(L.T @ Gq @ np.linalg.inv(Tm)).T
    return L, Tm.T @ Tm


def _rmsea_ci(chi, dfm, n, level=0.90):
    if dfm <= 0:
        return None, None, None
    est = np.sqrt(max((chi - dfm) / (dfm * (n - 1)), 0.0))
    a = (1 - level) / 2
    lo = hi = 0.0
    try:
        if stats.ncx2.cdf(chi, dfm, 0) > 1 - a:
            lo = optimize.brentq(lambda l: stats.ncx2.cdf(chi, dfm, l) - (1 - a), 1e-9, max(chi * 5, 100))
        if stats.ncx2.cdf(chi, dfm, 0) > a:
            hi = optimize.brentq(lambda l: stats.ncx2.cdf(chi, dfm, l) - a, 1e-9, max(chi * 10, 200))
    except ValueError:
        pass
    return est, float(np.sqrt(lo / (dfm * (n - 1)))), float(np.sqrt(hi / (dfm * (n - 1))))


def efa(df, spec):
    d, X, vs, dropped = _num_matrix(df, need(spec, "vars", "variables"), 3, 20)
    n, p = X.shape
    m = int_option(spec, "n_factors", 2, 1, max(1, p - 1), "Number of factors")
    rot = spec.get("rotation") or "varimax"
    seed = int_option(spec, "seed", 0, 0, 10 ** 9, "Random seed")
    dof = ((p - m) ** 2 - (p + m)) / 2
    if dof < 0:
        raise OESError(f"{m} factors for {p} variables leaves negative degrees of freedom; the model is not identified. Use fewer factors.")
    R = np.corrcoef(X, rowvar=False)
    if np.linalg.cond(R) > 1e12:
        raise OESError("The correlation matrix is singular: some variables are exact linear combinations of the others.")
    L, psi, obj, ok = ml_factor_analysis(R, m, n)
    chi = (n - 1 - (2 * p + 5) / 6.0 - 2 * m / 3.0) * obj
    pval = float(stats.chi2.sf(chi, dof)) if dof > 0 else None
    # sign convention before rotation
    for j in range(m):
        if L[:, j].sum() < 0:
            L[:, j] = -L[:, j]
    Phi = np.eye(m)
    if rot == "varimax":
        Lr, T = varimax(L)
    elif rot == "promax":
        Lr, T = promax(L)
        Phi = np.linalg.inv(T.T @ T)
    elif rot == "oblimin":
        Lr, Phi = oblimin(L)
    elif rot == "none":
        Lr = L
    else:
        raise OESError("Unknown rotation.")
    for j in range(m):
        if Lr[:, j].sum() < 0:
            Lr[:, j] = -Lr[:, j]
            Phi[j, :] *= -1
            Phi[:, j] *= -1
    comm = 1 - psi
    S_load = Lr @ Phi                                                    # structure matrix
    ssl = np.sum(Lr * S_load, axis=0) if rot in ("promax", "oblimin") else np.sum(Lr ** 2, axis=0)
    order = np.argsort(-ssl)
    Lr, ssl, S_load = Lr[:, order], ssl[order], S_load[:, order]
    Phi = Phi[np.ix_(order, order)]
    # fit indices
    S0 = R
    chi0 = (n - 1 - (2 * p + 5) / 6.0) * (-np.linalg.slogdet(R)[1])
    df0 = p * (p - 1) / 2
    tli = ((chi0 / df0) - (chi / dof)) / ((chi0 / df0) - 1) if dof > 0 else None
    est, lo, hi = _rmsea_ci(chi, dof, n)
    kmo, msa = _kmo(R)
    pa = _parallel(n, p, 200, seed)
    ev = np.sort(np.linalg.eigvalsh(R))[::-1]
    load_rows = [[v] + [f(Lr[i, j], 4) for j in range(m)] + [f(comm[i], 4), f(psi[i], 4), f(msa[i], 3)] for i, v in enumerate(vs)]
    t_load = table(f"Factor loadings ({rot} rotation" + (", pattern matrix" if rot in ("promax", "oblimin") else "") + ")", ["Variable"] + [f"Factor {j + 1}" for j in range(m)] + ["Communality", "Uniqueness", "MSA"], load_rows,
                   "Loadings with absolute value below 0.30 are usually read as negligible. MSA is the variable's sampling adequacy (Kaiser-Meyer-Olkin).")
    t_var = table("Variance explained by the factors", ["Factor", "SS loadings", "Proportion of variance", "Cumulative"], [[f"Factor {j + 1}", f(ssl[j], 4), f(ssl[j] / p, 4), f(ssl[:j + 1].sum() / p, 4)] for j in range(m)])
    tabs = [t_load, t_var]
    if rot in ("promax", "oblimin"):
        tabs.append(table("Factor correlations", [""] + [f"Factor {j + 1}" for j in range(m)], [[f"Factor {i + 1}"] + [f(Phi[i, j], 4) for j in range(m)] for i in range(m)]))
        tabs.append(table("Structure matrix (correlations of the variables with the factors)", ["Variable"] + [f"Factor {j + 1}" for j in range(m)], [[v] + [f(S_load[i, j], 4) for j in range(m)] for i, v in enumerate(vs)]))
    fit = [["Chi-square test of exact fit", f(chi, 4), f(pval, 6), f"df = {dof:.0f}; H0: {m} factors reproduce the correlations exactly"] if dof > 0 else ["Chi-square test of exact fit", None, None, "saturated model (df = 0)"]]
    if dof > 0:
        fit.append(["RMSEA (90% CI)", f(est, 4), None, f"[{lo:.4f}, {hi:.4f}]; below 0.08 is acceptable, below 0.05 is good"])
        fit.append(["Tucker-Lewis index (TLI)", f(tli, 4), None, "above 0.90 is acceptable, above 0.95 is good"])
        fit.append(["BIC (chi-square - df x ln n)", f(chi - dof * np.log(n), 3), None, "smaller is better; compare numbers of factors"])
    fit.append(["Kaiser-Meyer-Olkin measure", f(kmo, 4), None, "above 0.6 is acceptable"])
    chb, dfb, pb = _bartlett_sphericity(R, n)
    fit.append(["Bartlett's test of sphericity", f(chb, 4), f(pb, 6), "H0: the correlation matrix is the identity"])
    fit.append(["Heywood case (uniqueness at the lower bound)", int((psi <= 0.0051).sum()), None, "a boundary solution; too many factors or too few variables per factor" if (psi <= 0.0051).any() else "none"])
    fit.append(["Optimiser converged", "yes" if ok else "no", None, ""])
    tabs.append(tests_table(fit, "Model fit and diagnostics"))
    k_pa = int(np.sum(ev > pa))
    tabs.append(table("Eigenvalues of the correlation matrix and parallel analysis", ["Component", "Eigenvalue", "Parallel-analysis 95th pct."], [[i + 1, f(ev[i], 4), f(pa[i], 4)] for i in range(p)], f"Parallel analysis suggests {k_pa} factor(s); the Kaiser rule suggests {int((ev > 1).sum())}."))
    # Thomson regression scores
    Fs = ((X - X.mean(0)) / X.std(0, ddof=1)) @ np.linalg.solve(R, S_load)
    s1, s2 = thin(list(Fs[:, 0]), list(Fs[:, 1] if m > 1 else Fs[:, 0]), 400)
    charts = [chart("Scree plot with parallel analysis", list(range(1, p + 1)), [("Eigenvalue", ev), ("Parallel analysis", pa)], "Component", "Eigenvalue"),
              chart("Factor scores: factor 1 against factor 2", s1, [("Factor 2", s2)], "Factor 1", "Factor 2", kind="scatter")]
    notes = [f"{n} observations, {p} variables, {m} factors, {rot} rotation" + (f"; {dropped} row(s) with missing values were dropped." if dropped else "."),
             "Maximum-likelihood factor analysis assumes multivariate normal variables; the chi-square test is sensitive to sample size, so judge fit with RMSEA and TLI as well."]
    summary = [stat("Chi-square", chi, 3), stat("df", int(dof)), stat("RMSEA", est, 4) if est is not None else stat("Factors", m), stat("KMO", kmo, 3)]
    return result("stat_efa", "Exploratory factor analysis (EFA)", n, summary, tabs, charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# C3. Confirmatory factor analysis
# ---------------------------------------------------------------------------------------------------------------
def parse_cfa_model(text, columns):
    """Parse lavaan-style measurement syntax: 'visual =~ x1 + x2 + x3' (one factor per line or separated by semicolons)."""
    fac = {}
    for part in re.split(r"[;\n]+", str(text)):
        part = part.strip()
        if not part or part.startswith("#"):
            continue
        m = re.match(r"^\s*([A-Za-z_][\w.]*)\s*=~\s*(.+)$", part)
        if not m:
            raise OESError(f"Cannot read the model line '{part}'. Write one factor per line, for example: visual =~ x1 + x2 + x3")
        name = m.group(1)
        items = [t.strip() for t in m.group(2).split("+") if t.strip()]
        if len(items) < 2:
            raise OESError(f"Factor {name} needs at least two indicators.")
        if name in fac:
            raise OESError(f"Factor {name} is defined twice.")
        miss = [t for t in items if t not in columns]
        if miss:
            raise OESError(f"Variable(s) not found in the data: {', '.join(miss)}")
        fac[name] = items
    if not fac:
        raise OESError("Write the measurement model, for example: visual =~ x1 + x2 + x3")
    seen = [v for items in fac.values() for v in items]
    if len(set(seen)) != len(seen):
        raise OESError("Each indicator may load on one factor only (cross-loadings are not part of a simple-structure CFA).")
    return fac


class _CFA:
    def __init__(self, fac, order, std_lv, correlated):
        self.order = list(order)
        self.p = len(order)
        self.names = list(fac)
        self.k = len(fac)
        pos = {v: i for i, v in enumerate(order)}
        self.pattern = np.zeros((self.p, self.k), bool)
        for j, (nm, items) in enumerate(fac.items()):
            for v in items:
                self.pattern[pos[v], j] = True
        self.fixed_loading = np.zeros((self.p, self.k), bool)
        if not std_lv:
            for j, items in enumerate(fac.values()):
                self.fixed_loading[pos[items[0]], j] = True
        self.free_load = self.pattern & ~self.fixed_loading
        self.std_lv = std_lv
        self.correlated = correlated
        self.tri = [(i, j) for i in range(self.k) for j in range(i + 1) if (i == j or correlated)]
        self.nl = int(self.free_load.sum())
        self.nphi = len(self.tri) if not std_lv else sum(1 for (i, j) in self.tri if i != j)
        self.npar = self.nl + self.nphi + self.p if not std_lv else self.nl + self.nphi + self.p

    def unpack(self, th):
        L = np.zeros((self.p, self.k))
        L[self.fixed_loading] = 1.0
        L[self.free_load] = th[:self.nl]
        i0 = self.nl
        Phi = np.eye(self.k)
        if self.std_lv:
            for (i, j) in self.tri:
                if i != j:
                    Phi[i, j] = Phi[j, i] = th[i0]
                    i0 += 1
        else:
            for (i, j) in self.tri:
                Phi[i, j] = Phi[j, i] = th[i0]
                i0 += 1
        Theta = np.diag(th[i0:i0 + self.p])
        return L, Phi, Theta

    def sigma(self, th):
        L, Phi, Theta = self.unpack(th)
        return L @ Phi @ L.T + Theta

    def fit_fn(self, th, S):
        Sg = self.sigma(th)
        sign, ld = np.linalg.slogdet(Sg)
        if sign <= 0:
            return 1e10
        return float(ld + np.trace(np.linalg.solve(Sg, S)) - np.linalg.slogdet(S)[1] - self.p)

    def grad(self, th, S):
        L, Phi, Theta = self.unpack(th)
        Sg = L @ Phi @ L.T + Theta
        try:
            Si = np.linalg.inv(Sg)
        except np.linalg.LinAlgError:
            return np.zeros_like(th)
        G = Si @ (Sg - S) @ Si
        gL = 2 * G @ L @ Phi
        gPhi = L.T @ G @ L
        out = list(gL[self.free_load])
        for (i, j) in self.tri:
            if self.std_lv and i == j:
                continue
            out.append(gPhi[i, i] if i == j else 2 * gPhi[i, j])
        out += list(np.diag(G))
        return np.array(out)


def cfa(df, spec):
    txt = need(spec, "model", "measurement model")
    alpha = _alpha(spec)
    corr = (spec.get("correlated") or "yes") == "yes"
    ident = spec.get("identification") or "marker"
    fac = parse_cfa_model(txt, list(df.columns))
    order = [v for items in fac.values() for v in items]
    d, X, vs, dropped = _num_matrix(df, order, 3, 20)
    n, p = X.shape
    S = np.cov(X, rowvar=False) * (n - 1) / n
    mdl = _CFA(fac, order, ident == "std", corr)
    q = mdl.npar
    nstat = p * (p + 1) / 2
    dfm = nstat - q
    if dfm < 0:
        raise OESError(f"The model has {q} free parameters but the data give only {int(nstat)} variances and covariances (negative degrees of freedom).")
    for nm, items in fac.items():
        if len(items) == 2 and mdl.k == 1:
            raise OESError("A single factor with two indicators is not identified.")
    # start values
    th0 = []
    sdv = np.sqrt(np.diag(S))
    for j in range(mdl.k):
        for i in range(p):
            if mdl.free_load[i, j]:
                th0.append(1.0 if mdl.std_lv else 0.8)
    if mdl.std_lv:
        th0 = [sdv[i] * 0.7 for j in range(mdl.k) for i in range(p) if mdl.free_load[i, j]]
    th0 += [0.3 if True else 0 for (i, j) in mdl.tri if i != j] if mdl.std_lv else [float(np.mean(np.diag(S)) * 0.7) if i == j else 0.1 * float(np.mean(np.diag(S))) for (i, j) in mdl.tri]
    th0 += list(0.5 * np.diag(S))
    th0 = np.array(th0, float)
    # parameter vector in the same order as unpack/grad: loadings, factor (co)variances, unique variances
    res = optimize.minimize(mdl.fit_fn, th0, args=(S,), jac=mdl.grad, method="BFGS", options={"gtol": 1e-8, "maxiter": 2000})
    for _ in range(2):
        if res.success:
            break
        res = optimize.minimize(mdl.fit_fn, res.x, args=(S,), jac=mdl.grad, method="BFGS", options={"gtol": 1e-8, "maxiter": 2000})
    th = res.x
    Fmin = max(res.fun, 0.0)
    chi = n * Fmin
    pchi = float(stats.chi2.sf(chi, dfm)) if dfm > 0 else None
    # standard errors from the expected information of the ML fit function: cov(theta) = (2 / N) [J' (Sigma^-1 x Sigma^-1) J]^-1,
    # with J the Jacobian of vec(Sigma) with respect to the free parameters (central differences)
    Sg0 = mdl.sigma(th)
    J = np.zeros((p * p, q))
    for a_ in range(q):
        e = np.zeros(q)
        e[a_] = 1e-6 * max(1.0, abs(th[a_]))
        J[:, a_] = ((mdl.sigma(th + e) - mdl.sigma(th - e)) / (2 * e[a_])).ravel()
    Si0 = np.linalg.inv(Sg0)
    info = J.T @ np.kron(Si0, Si0) @ J
    try:
        cov_th = 2.0 / n * np.linalg.inv(info)
        se = np.sqrt(np.clip(np.diag(cov_th), 0, None))
        info_ok = bool(np.all(np.diag(cov_th) > 0)) and bool(np.all(np.linalg.eigvalsh((info + info.T) / 2) > 0))
    except np.linalg.LinAlgError:
        se = np.full(q, np.nan)
        info_ok = False
    L, Phi, Theta = mdl.unpack(th)
    Sg = mdl.sigma(th)
    # baseline (independence) model
    ld0 = np.linalg.slogdet(np.diag(np.diag(S)))[1]
    F0 = ld0 - np.linalg.slogdet(S)[1]
    chi0, df0 = n * F0, nstat - p
    cfi = 1 - max(chi - dfm, 0) / max(chi - dfm, chi0 - df0, 1e-12) if dfm > 0 else 1.0
    tli = ((chi0 / df0) - (chi / dfm)) / ((chi0 / df0) - 1) if dfm > 0 else 1.0
    est, lo, hi = _rmsea_ci(chi, dfm, n + 1)
    D = np.diag(1 / np.sqrt(np.diag(Sg)))
    resid = (D @ (S - Sg) @ D)
    srmr = float(np.sqrt(np.sum(np.tril(resid) ** 2) / nstat))
    loglik = -0.5 * n * (p * np.log(2 * np.pi) + np.linalg.slogdet(Sg)[1] + np.trace(np.linalg.solve(Sg, S)))
    aic, bic = -2 * loglik + 2 * q, -2 * loglik + q * np.log(n)
    # parameter table
    zc = stats.norm.ppf(1 - alpha / 2)
    rows, k0 = [], 0
    stdL = L / np.sqrt(np.diag(Sg))[:, None]
    for j, nm in enumerate(mdl.names):
        for i in range(p):
            if mdl.pattern[i, j]:
                if mdl.fixed_loading[i, j]:
                    rows.append([f"{nm} =~ {order[i]}", 1.0, None, None, None, None, None, f(stdL[i, j], 4)])
                else:
                    b, s_ = th[k0], se[k0]
                    rows.append([f"{nm} =~ {order[i]}", f(b, 5), f(s_, 5), f(b / s_, 3), f(2 * stats.norm.sf(abs(b / s_)), 6), f(b - zc * s_, 4), f(b + zc * s_, 4), f(stdL[i, j], 4)])
                    k0 += 1
    t_load = table("Factor loadings", ["Parameter", "Estimate", "Std. err.", "z", "p-value", f"{_ci(alpha)} CI low", "CI high", "Standardised"], rows,
                   "The first loading of each factor is fixed to 1 to set the scale of the factor (marker-variable identification)." if not mdl.std_lv else "Factor variances are fixed to 1 (standardised latent variables).")
    prow = []
    ph = list(th[mdl.nl:mdl.nl + mdl.nphi])
    ph_se = list(se[mdl.nl:mdl.nl + mdl.nphi])
    kk = 0
    for (i, j) in mdl.tri:
        if mdl.std_lv and i == j:
            continue
        b, s_ = ph[kk], ph_se[kk]
        kk += 1
        lab = f"{mdl.names[i]} variance" if i == j else f"{mdl.names[i]} ~~ {mdl.names[j]}"
        std = b if (mdl.std_lv or i == j) is False else None
        corr_ij = Phi[i, j] / np.sqrt(Phi[i, i] * Phi[j, j]) if i != j else None
        prow.append([lab, f(b, 5), f(s_, 5), f(b / s_, 3), f(2 * stats.norm.sf(abs(b / s_)), 6), f(b - zc * s_, 4), f(b + zc * s_, 4), f(corr_ij, 4)])
    t_fac = table("Factor variances and covariances", ["Parameter", "Estimate", "Std. err.", "z", "p-value", f"{_ci(alpha)} CI low", "CI high", "Correlation"], prow) if prow else None
    ub = list(th[mdl.nl + mdl.nphi:])
    ub_se = list(se[mdl.nl + mdl.nphi:])
    t_unq = table("Unique (error) variances and reliability of each indicator", ["Indicator", "Estimate", "Std. err.", "z", "p-value", "R-squared"],
                  [[order[i], f(ub[i], 5), f(ub_se[i], 5), f(ub[i] / ub_se[i], 3), f(2 * stats.norm.sf(abs(ub[i] / ub_se[i])), 6), f(1 - ub[i] / Sg[i, i], 4)] for i in range(p)])
    # composite reliability and AVE per factor
    rel = []
    for j, nm in enumerate(mdl.names):
        idx = np.where(mdl.pattern[:, j])[0]
        lam = stdL[idx, j]
        thv = 1 - lam ** 2
        omega = lam.sum() ** 2 / (lam.sum() ** 2 + thv.sum())
        ave = (lam ** 2).mean()
        rel.append([nm, len(idx), f(omega, 4), f(ave, 4), "yes" if ave >= 0.5 else "no"])
    t_rel = table("Composite reliability (omega) and average variance extracted", ["Factor", "Indicators", "Omega", "AVE", "AVE above 0.5"], rel)
    fitrows = [["Chi-square (N x F_ML)", f(chi, 4), f(pchi, 6), f"df = {dfm:.0f}; H0: the model reproduces the covariance matrix exactly"],
               ["Baseline chi-square", f(chi0, 4), f(float(stats.chi2.sf(chi0, df0)), 6), f"df = {df0:.0f}; independence model"],
               ["Comparative fit index (CFI)", f(cfi, 4), None, "above 0.90 acceptable, above 0.95 good"],
               ["Tucker-Lewis index (TLI)", f(tli, 4), None, "above 0.90 acceptable, above 0.95 good"],
               ["RMSEA (90% CI)", f(est, 4), None, f"[{lo:.4f}, {hi:.4f}]; below 0.08 acceptable, below 0.05 good" if est is not None else ""],
               ["SRMR", f(srmr, 4), None, "below 0.08 is good"],
               ["Log-likelihood", f(loglik, 3), None, ""], ["AIC", f(aic, 3), None, ""], ["BIC", f(bic, 3), None, ""],
               ["Free parameters", q, None, ""], ["Optimiser converged", "yes" if res.success else "no", None, f"{res.nit} iterations; gradient norm {np.linalg.norm(mdl.grad(th, S)):.2e}"],
               ["Negative error variance (Heywood case)", int((np.array(ub) < 0).sum()), None, "a negative unique variance signals misspecification or a small sample" if (np.array(ub) < 0).any() else "none"],
               ["Information matrix positive definite", "yes" if info_ok else "no", None, "the standard errors need this"]]
    t_fit = tests_table(fitrows, "Model fit")
    rr = [[order[i]] + [f(resid[i, j], 3) for j in range(p)] for i in range(p)]
    t_res = table("Residual correlations (observed minus model-implied)", ["Variable"] + order, rr, "Absolute values above 0.10 point to local misfit.")
    tabs = [t_fit, t_load] + ([t_fac] if t_fac else []) + [t_unq, t_rel, t_res]
    xs = list(range(1, p + 1))
    charts = [chart("Standardised loadings", [order[i] for i in range(p)], [("Standardised loading", [float(stdL[i][mdl.pattern[i]][0]) for i in range(p)])], "Indicator", "Loading", kind="bar"),
              chart("Observed and model-implied variances", order, [("Observed", np.diag(S)), ("Model-implied", np.diag(Sg))], "Indicator", "Variance", kind="bar")]
    notes = [f"{n} observations, {p} indicators, {mdl.k} factor(s)" + (f"; {dropped} row(s) with missing values were dropped." if dropped else "."),
             "Maximum-likelihood estimation assumes multivariate normal indicators. Standard errors come from the expected information matrix; the chi-square uses N times the minimised fit function.",
             "Each indicator loads on one factor and error variances are uncorrelated (simple structure)."]
    summary = [stat("Chi-square", chi, 3), stat("df", int(dfm)), stat("CFI", cfi, 4), stat("RMSEA", est, 4) if est is not None else stat("Factors", mdl.k), stat("SRMR", srmr, 4)]
    return result("stat_cfa", "Confirmatory factor analysis (CFA)", n, summary, tabs, charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# C4. Linear discriminant analysis
# ---------------------------------------------------------------------------------------------------------------
def _lda_fit(X, y_idx, k, priors):
    n, p = X.shape
    means = np.array([X[y_idx == g].mean(0) for g in range(k)])
    Xc = X - means[y_idx]
    W = Xc.T @ Xc / (n - k)
    return means, W


def _lda_predict(X, means, W, priors):
    Wi = np.linalg.inv(W)
    lin = X @ Wi @ means.T - 0.5 * np.sum((means @ Wi) * means, axis=1) + np.log(priors)
    lin -= lin.max(1, keepdims=True)
    post = np.exp(lin)
    post /= post.sum(1, keepdims=True)
    return post


def lda(df, spec):
    xs, g = as_list(need(spec, "vars", "predictors")), need(spec, "group", "class variable")
    distinct(xs, g)
    pri = spec.get("priors") or "proportional"
    d, dropped = clean_frame(df, numeric=xs, factors=[g])
    lv = levels(d[g])
    k = len(lv)
    if k < 2:
        raise OESError("The class variable needs at least two classes.")
    X = d[xs].to_numpy(float)
    n, p = X.shape
    yi = d[g].map({v: i for i, v in enumerate(lv)}).to_numpy()
    n_i = np.bincount(yi, minlength=k)
    if (n_i <= 1).any() or n - k <= p:
        raise OESError("Too few observations: each class needs at least two and the total must exceed the number of predictors plus classes.")
    priors = n_i / n if pri == "proportional" else np.full(k, 1.0 / k)
    means, W = _lda_fit(X, yi, k, priors)
    if np.linalg.cond(W) > 1e12:
        raise OESError("The pooled within-class covariance matrix is singular: remove predictors that are collinear.")
    gm = (priors[:, None] * means).sum(0) if pri != "proportional" else X.mean(0)
    Bm = (means - X.mean(0))
    Bmat = (np.sqrt(n_i)[:, None] * Bm).T @ (np.sqrt(n_i)[:, None] * Bm) / (n - k)
    nd = min(k - 1, p)
    # discriminant directions: eigen-problem of W^-1 B with the within-class variance scaled to 1
    ev_w, ev_v = np.linalg.eigh(W)
    Wh = ev_v @ np.diag(ev_w ** -0.5) @ ev_v.T
    M = Wh @ Bmat @ Wh
    lam, U = np.linalg.eigh((M + M.T) / 2)
    order = np.argsort(lam)[::-1][:nd]
    lam, U = lam[order], U[:, order]
    A = Wh @ U                                                       # coefficients: LD = (x - mean) A has unit pooled within-class variance
    for j in range(nd):
        if A[np.argmax(np.abs(A[:, j])), j] < 0:
            A[:, j] = -A[:, j]
    lam_tot = np.sum(np.clip(np.linalg.eigvalsh((M + M.T) / 2), 0, None))
    Z = (X - X.mean(0)) @ A
    cancor = np.sqrt(lam / (1 + lam))
    # Wilks lambda tests of the discriminant functions (Bartlett's chi-square)
    ev_all = np.clip(np.sort(np.linalg.eigvalsh((M + M.T) / 2))[::-1], 0, None)
    ev_ssp = ev_all                      # the eigenvalues of E^-1 H equal those of W^-1 B / (n - k), because E = (n - k) W
    wil_rows = []
    for j in range(nd):
        lamb = float(np.prod(1.0 / (1 + ev_ssp[j:nd])))
        chi = -(n - 1 - (p + k) / 2.0) * np.log(lamb)
        dfj = (p - j) * (k - 1 - j)
        wil_rows.append([f"Functions {j + 1} to {nd}", f(lamb, 5), f(chi, 3), dfj, f(float(stats.chi2.sf(chi, dfj)), 6)])
    post = _lda_predict(X, means, W, priors)
    pred = post.argmax(1)
    acc = float((pred == yi).mean())
    # cross-validated accuracy: leave-one-out with the closed form when feasible, otherwise 10 folds
    cv_pred = np.empty(n, int)
    if n <= 1500:
        for i in range(n):
            mask = np.arange(n) != i
            mi, Wi_ = _lda_fit(X[mask], yi[mask], k, priors)
            pr = n_i - np.eye(k, dtype=int)[yi[i]]
            pri_ = pr / pr.sum() if pri == "proportional" else priors
            cv_pred[i] = _lda_predict(X[i:i + 1], mi, Wi_, pri_).argmax(1)[0]
        cv_label = "Leave-one-out"
    else:
        rng = np.random.default_rng(0)
        folds = rng.permutation(n) % 10
        for fo in range(10):
            te = folds == fo
            mi, Wi_ = _lda_fit(X[~te], yi[~te], k, priors)
            cv_pred[te] = _lda_predict(X[te], mi, Wi_, priors).argmax(1)
        cv_label = "10-fold"
    cv_acc = float((cv_pred == yi).mean())
    cm = np.zeros((k, k), int)
    for a, b in zip(yi, pred):
        cm[a, b] += 1
    cmcv = np.zeros((k, k), int)
    for a, b in zip(yi, cv_pred):
        cmcv[a, b] += 1
    t_coef = table("Discriminant function coefficients (unstandardised; scaled to unit pooled within-class variance)", ["Variable"] + [f"LD{j + 1}" for j in range(nd)], [[v] + [f(A[i, j], 5) for j in range(nd)] for i, v in enumerate(xs)])
    sd_w = np.sqrt(np.diag(W))
    t_std = table("Standardised coefficients and structure matrix", ["Variable"] + [f"LD{j + 1} std. coef." for j in range(nd)] + [f"LD{j + 1} structure" for j in range(nd)],
                  [[v] + [f(A[i, j] * sd_w[i], 4) for j in range(nd)] + [f(float(np.corrcoef(X[:, i], Z[:, j])[0, 1]), 4) for j in range(nd)] for i, v in enumerate(xs)],
                  "The structure matrix holds the total correlations between each predictor and each discriminant score.")
    t_eig = table("Discriminant functions", ["Function", "Eigenvalue", "% of trace", "Cumulative %", "Canonical correlation"], [[f"LD{j + 1}", f(lam[j], 5), f(100 * lam[j] / lam_tot, 3), f(100 * lam[:j + 1].sum() / lam_tot, 3), f(cancor[j], 4)] for j in range(nd)])
    t_wil = table("Wilks' lambda tests of the discriminant functions", ["Test", "Wilks' lambda", "Chi-square", "df", "p-value"], wil_rows)
    t_cls = table("Class summary", ["Class", "n", "Prior"] + [f"Mean {v}" for v in xs], [[lv[i], int(n_i[i]), f(priors[i], 4)] + [f(means[i, j], 4) for j in range(p)] for i in range(k)])
    t_acc = table("Classification accuracy", ["Measure", "Value"], [["Resubstitution accuracy", f(acc, 4)], [f"{cv_label} cross-validated accuracy", f(cv_acc, 4)], ["Accuracy of always predicting the largest class", f(float(n_i.max() / n), 4)]])
    t_cm = table(f"Confusion matrix, {cv_label.lower()} (rows = true class, columns = predicted)", ["True class"] + lv + ["Recall"], [[lv[i]] + [int(cmcv[i, j]) for j in range(k)] + [f(cmcv[i, i] / cmcv[i].sum(), 4)] for i in range(k)])
    # Mahalanobis distances between classes
    Wi = np.linalg.inv(W)
    dist = [[f(float((means[a] - means[b]) @ Wi @ (means[a] - means[b])), 4) for b in range(k)] for a in range(k)]
    t_md = table("Squared Mahalanobis distances between class means", [""] + lv, [[lv[a]] + dist[a] for a in range(k)])
    diag = []
    from .stat_anova import _box_m
    try:
        bm = _box_m([X[yi == i] for i in range(k)])
        diag.append(["Box's M equality of covariance matrices", f(bm[0], 4), f(bm[2], 4), "LDA assumes a common covariance matrix; if rejected consider quadratic discriminant analysis"])
    except Exception:
        pass
    from .stat_anova import _mardia
    zz1, dfz, pz1, b2, z2, pz2 = _mardia(X - means[yi])
    diag.append(["Mardia multivariate skewness (within-class residuals)", f(zz1, 4), f(pz1, 4), "H0: multivariate normal within classes"])
    diag.append(["Mardia multivariate kurtosis", f(z2, 4), f(pz2, 4), ""])
    t_diag = tests_table(diag, "Assumption checks")
    tabs = [t_eig, t_wil, t_coef, t_std, t_cls, t_acc, t_cm, t_md, t_diag]
    x_ld = list(Z[:, 0])
    ser = []
    for i in range(k):
        yy = list(Z[:, 1]) if nd > 1 else [0.0] * n
        ser.append((lv[i], [yy[r] if yi[r] == i else None for r in range(n)]))
    sub = np.unique(np.linspace(0, n - 1, min(n, 500)).astype(int))
    charts = [chart("Discriminant scores: LD1 against LD2" if nd > 1 else "Discriminant scores on LD1", [x_ld[r] for r in sub], [(nm, [y_[r] for r in sub]) for nm, y_ in ser], "LD1", "LD2" if nd > 1 else "", kind="scatter"),
              chart("Share of the between-class trace", [f"LD{j + 1}" for j in range(nd)], [("% of trace", 100 * lam / lam_tot)], "Function", "%", kind="bar")]
    notes = [f"{n} observations, {k} classes, {p} predictors" + (f"; {dropped} row(s) with missing values were dropped." if dropped else "."),
             "Assumes multivariate normal predictors with a common covariance matrix in every class. Resubstitution accuracy is optimistic; use the cross-validated figure."]
    summary = [stat("Resubstitution accuracy", acc, 4), stat("Cross-validated accuracy", cv_acc, 4), stat("LD1 % of trace", 100 * lam[0] / lam_tot, 3), stat("Classes", k)]
    return result("stat_lda", "Linear discriminant analysis (LDA)", n, summary, tabs, charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# C5. Canonical correlation analysis
# ---------------------------------------------------------------------------------------------------------------
def _inv_sqrt(C):
    w, V = np.linalg.eigh(C)
    if w.min() <= 1e-12 * w.max():
        raise OESError("A covariance matrix within one variable set is singular: remove collinear variables.")
    return V @ np.diag(w ** -0.5) @ V.T


def cca(df, spec):
    xv, yv = as_list(need(spec, "x_vars", "first variable set")), as_list(need(spec, "y_vars", "second variable set"))
    distinct(xv, yv)
    if len(xv) < 1 or len(yv) < 1:
        raise OESError("Each variable set needs at least one variable.")
    alpha = _alpha(spec)
    nperm = int_option(spec, "perms", 999, 0, 20000, "Permutations")
    seed = int_option(spec, "seed", 0, 0, 10 ** 9, "Random seed")
    d, dropped = clean_frame(df, numeric=xv + yv)
    X, Y = d[xv].to_numpy(float), d[yv].to_numpy(float)
    n, p, q = len(X), len(xv), len(yv)
    if n <= p + q:
        raise OESError("There must be more observations than variables in both sets together.")
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    Cxx, Cyy, Cxy = Xc.T @ Xc / (n - 1), Yc.T @ Yc / (n - 1), Xc.T @ Yc / (n - 1)
    Kx, Ky = _inv_sqrt(Cxx), _inv_sqrt(Cyy)
    U, s, Vt = np.linalg.svd(Kx @ Cxy @ Ky)
    r = np.clip(s, 0, 1)
    m = min(p, q)
    A = Kx @ U[:, :m]                                # coefficients for the standardised-scale scores: scores = Xc @ A
    B = Ky @ Vt.T[:, :m]
    for j in range(m):
        if A[:, j].sum() < 0:
            A[:, j], B[:, j] = -A[:, j], -B[:, j]
    SX, SY = Xc @ A, Yc @ B
    sdx, sdy = np.sqrt(np.diag(Cxx)), np.sqrt(np.diag(Cyy))
    # sequential Wilks tests with Bartlett's chi-square and Rao's F
    rows = []
    for j in range(m):
        lamb = float(np.prod(1 - r[j:m] ** 2))
        chi = -(n - 1 - (p + q + 1) / 2.0) * np.log(lamb)
        dfj = (p - j) * (q - j)
        pp, qq = p - j, q - j
        tt = np.sqrt((pp ** 2 * qq ** 2 - 4) / (pp ** 2 + qq ** 2 - 5)) if pp ** 2 + qq ** 2 - 5 > 0 else 1.0
        df1 = pp * qq
        df2 = ((n - 1) - (pp + qq + 3) / 2.0) * tt - (pp * qq - 2) / 2.0
        Fr = (1 - lamb ** (1 / tt)) / lamb ** (1 / tt) * df2 / df1 if lamb > 0 else np.inf
        rows.append([j + 1, f(r[j], 5), f(r[j] ** 2, 5), f(lamb, 5), f(chi, 3), dfj, f(float(stats.chi2.sf(chi, dfj)), 6), f(Fr, 3), f(df1, 1), f(df2, 2), f(float(stats.f.sf(Fr, df1, df2)), 6)])
    t_cor = table("Canonical correlations and sequential tests", ["Pair", "Canonical correlation", "Squared", "Wilks' lambda (this and later pairs)", "Bartlett chi-square", "df", "p-value", "Rao F", "df1", "df2", "F p-value"], rows,
                  "Each row tests that this canonical correlation and all later ones are zero.")
    # permutation test of the first canonical correlation (rows of Y permuted)
    tabs = [t_cor]
    if nperm:
        rng = np.random.default_rng(seed)
        cnt = 0
        for _ in range(nperm):
            Yp = Yc[rng.permutation(n)]
            Cxy_p = Xc.T @ Yp / (n - 1)
            if np.linalg.svd(Kx @ Cxy_p @ Ky, compute_uv=False)[0] >= r[0] - 1e-12:
                cnt += 1
        pperm = (cnt + 1) / (nperm + 1)
        tabs.append(table("Permutation test of the first canonical correlation", ["Statistic", "Value"], [["Observed first canonical correlation", f(r[0], 5)], ["Permutations", nperm], ["Permutation p-value", f(pperm, 6)]],
                          "Rows of the second set are shuffled; the p-value is (count + 1) / (permutations + 1) and does not rely on normality."))
    else:
        pperm = None
    tabs.append(table("Standardised canonical coefficients, first set", ["Variable"] + [f"U{j + 1}" for j in range(m)], [[v] + [f(A[i, j] * sdx[i], 4) for j in range(m)] for i, v in enumerate(xv)]))
    tabs.append(table("Standardised canonical coefficients, second set", ["Variable"] + [f"V{j + 1}" for j in range(m)], [[v] + [f(B[i, j] * sdy[i], 4) for j in range(m)] for i, v in enumerate(yv)]))
    SXs, SYs = SX / SX.std(0, ddof=1), SY / SY.std(0, ddof=1)
    LX = np.array([[np.corrcoef(Xc[:, i], SX[:, j])[0, 1] for j in range(m)] for i in range(p)])
    LY = np.array([[np.corrcoef(Yc[:, i], SY[:, j])[0, 1] for j in range(m)] for i in range(q)])
    tabs.append(table("Canonical loadings (structure correlations)", ["Variable", "Set"] + [f"Pair {j + 1}" for j in range(m)], [[v, "first"] + [f(LX[i, j], 4) for j in range(m)] for i, v in enumerate(xv)] + [[v, "second"] + [f(LY[i, j], 4) for j in range(m)] for i, v in enumerate(yv)]))
    red_x = (LX ** 2).mean(0) * r[:m] ** 2
    red_y = (LY ** 2).mean(0) * r[:m] ** 2
    tabs.append(table("Variance extracted and redundancy", ["Pair", "Share of first-set variance (own)", "Redundancy of first set", "Share of second-set variance (own)", "Redundancy of second set"],
                      [[j + 1, f((LX[:, j] ** 2).mean(), 4), f(red_x[j], 4), f((LY[:, j] ** 2).mean(), 4), f(red_y[j], 4)] for j in range(m)],
                      "Redundancy is the share of one set's variance that the other set's canonical variate explains."))
    diag = []
    from .stat_anova import _mardia
    z1, dfz, pz1, b2, z2, pz2 = _mardia(np.hstack([Xc, Yc]))
    diag.append(["Mardia multivariate skewness (both sets)", f(z1, 4), f(pz1, 4), "H0: joint multivariate normality (the Wilks tests assume it)"])
    diag.append(["Mardia multivariate kurtosis", f(z2, 4), f(pz2, 4), ""])
    diag.append(["Observations per variable", f(n / (p + q), 2), None, "at least 10 per variable is a common minimum for stable canonical weights"])
    tabs.append(tests_table(diag, "Assumption checks"))
    sub = np.unique(np.linspace(0, n - 1, min(n, 400)).astype(int))
    charts = [chart("First pair of canonical variates", [float(SX[i, 0]) for i in sub], [("V1 against U1", [float(SY[i, 0]) for i in sub])], "U1 (first set)", "V1 (second set)", kind="scatter"),
              chart("Canonical correlations", [f"Pair {j + 1}" for j in range(m)], [("Canonical correlation", r[:m])], "Pair", "Correlation", kind="bar")]
    notes = [f"{n} observations, {p} variables in the first set and {q} in the second" + (f"; {dropped} row(s) with missing values were dropped." if dropped else "."),
             "Canonical correlations are the correlations between the linear combinations of the two sets that are most strongly related, and are uncorrelated across pairs."]
    summary = [stat("First canonical correlation", r[0], 4), stat("Wilks' lambda", float(np.prod(1 - r[:m] ** 2)), 4), stat("Bartlett p-value (all pairs)", float(stats.chi2.sf(-(n - 1 - (p + q + 1) / 2.0) * np.log(np.prod(1 - r[:m] ** 2)), p * q)), 6), stat("Pairs", m)]
    return result("stat_cca", "Canonical correlation analysis (CCA)", n, summary, tabs, charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# C6. Correspondence analysis
# ---------------------------------------------------------------------------------------------------------------
def ca(df, spec):
    rv = need(spec, "row_var", "row variable")
    cv = spec.get("col_var") or None
    counts = as_list(spec.get("counts"))
    alpha = _alpha(spec)
    if counts:
        distinct([rv], counts)
        d, dropped = clean_frame(df, numeric=counts, factors=[rv])
        N = d.groupby(rv, sort=False)[counts].sum()
        N = N.loc[levels(pd.Series(N.index))]
        rows_l, cols_l = list(N.index), list(counts)
        T = N.to_numpy(float)
    elif cv:
        distinct(rv, cv)
        d, dropped = clean_frame(df, factors=[rv, cv])
        ct = pd.crosstab(d[rv], d[cv])
        ct = ct.loc[levels(pd.Series(ct.index)), levels(pd.Series(ct.columns))]
        rows_l, cols_l, T = list(ct.index), list(ct.columns), ct.to_numpy(float)
    else:
        raise OESError("Give either a column variable (one row per case) or the count columns of a contingency table.")
    if (T < 0).any():
        raise OESError("Counts cannot be negative.")
    T = T[T.sum(1) > 0][:, T.sum(0) > 0] if ((T.sum(1) == 0).any() or (T.sum(0) == 0).any()) else T
    I, J = T.shape
    if I < 2 or J < 2:
        raise OESError("The table needs at least two rows and two columns with non-zero totals.")
    n = float(T.sum())
    P = T / n
    r_, c_ = P.sum(1), P.sum(0)
    E = np.outer(r_, c_)
    S = (P - E) / np.sqrt(E)
    U, sv, Vt = np.linalg.svd(S, full_matrices=False)
    K = min(I, J) - 1
    sv, U, V = sv[:K], U[:, :K], Vt.T[:, :K]
    inertia = sv ** 2
    tot = float(inertia.sum())
    chi = n * tot
    dfc = (I - 1) * (J - 1)
    pchi = float(stats.chi2.sf(chi, dfc))
    cramer = float(np.sqrt(chi / (n * min(I - 1, J - 1))))
    F = (U / np.sqrt(r_)[:, None]) * sv                       # row principal coordinates
    G = (V / np.sqrt(c_)[:, None]) * sv                       # column principal coordinates
    for j in range(K):
        if F[:, j].sum() < 0:
            F[:, j], G[:, j] = -F[:, j], -G[:, j]
    t_dim = table("Dimensions: singular values and inertia", ["Dimension", "Singular value (canonical correlation)", "Inertia (squared)", "% of inertia", "Cumulative %", "Chi-square share"],
                  [[j + 1, f(sv[j], 5), f(inertia[j], 5), f(100 * inertia[j] / tot, 3), f(100 * inertia[:j + 1].sum() / tot, 3), f(inertia[j] * n, 3)] for j in range(K)],
                  "Total inertia equals the Pearson chi-square statistic divided by the number of observations.")
    t_test = table("Test of independence", ["Statistic", "Value"], [["Observations", f(n, 6)], ["Pearson chi-square", f(chi, 4)], ["df", dfc], ["p-value", f(pchi, 6)], ["Total inertia (phi-squared)", f(tot, 5)], ["Cramer's V", f(cramer, 4)]])
    Kd = min(K, 2)
    mass_r, mass_c = r_, c_
    row_inertia = np.sum(((P / r_[:, None] - c_) ** 2) / c_, axis=1) * r_
    col_inertia = np.sum(((P / c_[None, :] - r_[:, None]) ** 2) / r_[:, None], axis=0) * c_
    def coord_table(title, labels, mass, inert, C, ctr_mass, sing):
        ctr = (mass[:, None] * C ** 2) / (sing ** 2)[None, :]
        qual = (C ** 2) / np.maximum((C ** 2).sum(1, keepdims=True), 1e-300)
        qual_tot = (C ** 2).sum(1) / np.maximum(inert / mass, 1e-300)
        cols = ["Label", "Mass", "Inertia share"] + [f"Dim {j + 1}" for j in range(Kd)] + [f"Contribution to dim {j + 1}" for j in range(Kd)] + [f"Quality (cos2) dim {j + 1}" for j in range(Kd)]
        out = []
        for i, lab in enumerate(labels):
            out.append([str(lab), f(mass[i], 4), f(inert[i] / tot, 4)] + [f(C[i, j], 4) for j in range(Kd)] + [f(ctr[i, j], 4) for j in range(Kd)] + [f(qual[i, j] * qual_tot[i], 4) for j in range(Kd)])
        return table(title, cols, out)
    t_row = coord_table("Row points (principal coordinates)", rows_l, mass_r, row_inertia, F, mass_r, sv)
    t_col = coord_table("Column points (principal coordinates)", cols_l, mass_c, col_inertia, G, mass_c, sv)
    resid = (T - n * E) / np.sqrt(n * E)
    t_res = table("Pearson residuals (observed - expected) / sqrt(expected)", [""] + [str(c) for c in cols_l], [[str(rows_l[i])] + [f(resid[i, j], 3) for j in range(J)] for i in range(I)], "Absolute values above 2 mark cells that depart from independence.")
    exp_low = int((n * E < 5).sum())
    diag = [["Cells with expected count below 5", exp_low, None, "the chi-square test is reliable when this is small (under 20% of cells)" if exp_low else "none"],
            ["Smallest expected count", f(float((n * E).min()), 3), None, ""]]
    tabs = [t_dim, t_test, t_row, t_col, t_res, tests_table(diag, "Checks")]
    x_all = [float(v) for v in F[:, 0]] + [float(v) for v in G[:, 0]]
    y_r = [float(v) for v in (F[:, 1] if K > 1 else np.zeros(I))] + [None] * J
    y_c = [None] * I + [float(v) for v in (G[:, 1] if K > 1 else np.zeros(J))]
    charts = [chart("Symmetric map: rows and columns on dimensions 1 and 2", x_all, [("Rows", y_r), ("Columns", y_c)], f"Dimension 1 ({100 * inertia[0] / tot:.1f}%)", f"Dimension 2 ({100 * inertia[1] / tot:.1f}%)" if K > 1 else "", kind="scatter"),
              chart("Inertia by dimension (%)", [f"Dim {j + 1}" for j in range(K)], [("% of inertia", 100 * inertia / tot)], "Dimension", "%", kind="bar")]
    notes = [f"{I} rows, {J} columns, {n:g} observations" + (f"; {dropped} row(s) with missing values were dropped." if (not counts) and dropped else "."),
             "Correspondence analysis decomposes the chi-square distance from independence. In the symmetric map, rows (or columns) close together have similar profiles; distances between a row and a column are not directly interpretable."]
    summary = [stat("Total inertia", tot, 5), stat("Chi-square", chi, 3), stat("p-value", pchi, 6), stat("Dim 1 inertia %", 100 * inertia[0] / tot, 3), stat("Cramer's V", cramer, 4)]
    return result("stat_ca", "Correspondence analysis (CA)", int(round(n)), summary, tabs, charts, notes)
