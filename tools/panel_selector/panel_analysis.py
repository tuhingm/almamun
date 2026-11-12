"""Model-selection analysis for the Panel-Data Research Guide.

analyze(csv_text) reads a long-format panel CSV: entity identifier, time identifier, then the dependent variable and the
regressors (one column per variable). It reports the panel structure, summary statistics (overall, between, within),
correlations and then one of two test sequences:

* short panels (median T below LONG_T): F test for unit effects, Breusch-Pagan LM test, Hausman and Mundlak tests,
  with pooled OLS, random-effects and fixed-effects estimates;
* long panels: Pesaran CD tests, the Pesaran-Yamagata slope-homogeneity test, panel unit-root tests (Fisher-ADF and
  CIPS) on levels and differences, and, when every variable is I(1), Kao and Fisher-Engle-Granger cointegration tests.

Returns a JSON string. The same file runs in the browser (Pyodide) for uploaded data and offline for the examples.
"""
import io, json, math, warnings
import numpy as np
import pandas as pd

LONG_T = 15
CIPS_Q = {}  # filled in by the page builder: simulated quantiles of the CIPS statistic (see simulate_cips.py)
ID_NAMES = ("id", "entity", "unit", "country", "code", "iso", "iso3", "state", "province", "region", "county", "city",
            "firm", "company", "bank", "industry", "sector", "nr", "person", "pid", "hhid", "household", "individual", "panel")
TIME_NAMES = ("year", "time", "t", "period", "date", "month", "quarter", "yr", "wave")


def _f(x, nd=4):
    try:
        x = float(x)
    except Exception:
        return None
    return None if not np.isfinite(x) else round(x, nd)


def _ncdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def _ninv(p):
    from scipy.stats import norm
    return float(norm.ppf(p))


def _chi2_sf(x, df):
    from scipy.stats import chi2
    return float(chi2.sf(x, df))


def _f_sf(x, d1, d2):
    from scipy.stats import f
    return float(f.sf(x, d1, d2))


def _ols(y, X):
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    return b, e, XtX_inv


def _cluster_vcov(X, e, XtX_inv, groups, k_used):
    n = len(e)
    G = len(np.unique(groups))
    order = np.argsort(groups, kind="stable")
    g, Xs, es = groups[order], X[order], e[order]
    cut = np.flatnonzero(np.diff(g)) + 1
    meat = np.zeros((X.shape[1], X.shape[1]))
    for Xg, eg in zip(np.split(Xs, cut), np.split(es, cut)):
        s = Xg.T @ eg
        meat += np.outer(s, s)
    c = (G / (G - 1)) * ((n - 1) / max(n - k_used, 1)) if G > 1 else 1.0
    return c * XtX_inv @ meat @ XtX_inv


def _demean(a, codes, ng):
    a = np.asarray(a, float)
    if a.ndim == 1:
        m = np.bincount(codes, a, ng) / np.bincount(codes, minlength=ng)
        return a - m[codes]
    out = np.empty_like(a)
    cnt = np.bincount(codes, minlength=ng)
    for j in range(a.shape[1]):
        out[:, j] = a[:, j] - (np.bincount(codes, a[:, j], ng) / cnt)[codes]
    return out


def _gmeans(a, codes, ng):
    cnt = np.bincount(codes, minlength=ng)
    if a.ndim == 1:
        return (np.bincount(codes, a, ng) / cnt)
    return np.column_stack([np.bincount(codes, a[:, j], ng) / cnt for j in range(a.shape[1])])


def _independent_cols(X, tol=1e-8):
    """Indices of a maximal set of linearly independent columns (keeps the earlier column when two are collinear)."""
    keep = []
    for j in range(X.shape[1]):
        cand = keep + [j]
        sub = X[:, cand]
        if np.linalg.matrix_rank(sub, tol=tol * max(1.0, np.abs(sub).max())) == len(cand):
            keep.append(j)
    return keep


# ---------------------------------------------------------------- reading and structure

def _read(csv_text, max_vars):
    try:
        raw = pd.read_csv(io.StringIO(csv_text), sep=None, engine="python")
    except Exception as e:
        return None, f"Could not read the file as CSV ({e})."
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.loc[:, [c for c in raw.columns if not c.lower().startswith("unnamed")]]
    if raw.shape[1] < 4:
        return None, "The file needs at least four columns: entity, time, the dependent variable and one regressor."
    cols = list(raw.columns)
    low = [c.lower() for c in cols]
    ent = next((cols[i] for i, c in enumerate(low) if c in ID_NAMES), None)
    tim = next((cols[i] for i, c in enumerate(low) if c in TIME_NAMES and cols[i] != ent), None)
    if ent is None or tim is None:
        ent, tim = cols[0], cols[1]
    num = raw.drop(columns=[ent, tim]).apply(pd.to_numeric, errors="coerce")
    keep = [c for c in num.columns if num[c].notna().mean() > 0.8 and num[c].nunique() >= 2]
    skipped = [c for c in num.columns if c not in keep]
    if keep and num[keep[0]].nunique() <= 2:
        return None, f"The first variable after the identifiers ({keep[0]}) is treated as the dependent variable, but it takes only two values. This guide covers continuous outcomes; see the binary and count models section below."
    keep = keep[:max_vars]
    if len(keep) < 2:
        return None, "The file needs a dependent variable and at least one numeric regressor after the entity and time columns."
    df = pd.concat([raw[[ent, tim]], num[keep]], axis=1).dropna().copy()
    df[ent] = df[ent].astype(str).str.strip()
    tnum = pd.to_numeric(df[tim], errors="coerce")
    df["_t"] = tnum if tnum.notna().all() else df[tim].astype(str).str.strip()
    if df.duplicated([ent, "_t"]).any():
        d = df[df.duplicated([ent, "_t"], keep=False)].iloc[0]
        return None, f"Entity “{d[ent]}” appears more than once in period {d[tim]}. Each entity–period pair must be a single row (long format)."
    df = df.sort_values([ent, "_t"]).reset_index(drop=True)
    periods = sorted(df["_t"].unique())
    tmap = {p: i for i, p in enumerate(periods)}
    df["_ti"] = df["_t"].map(tmap).astype(int)
    cnt = df.groupby(ent)[tim].transform("size")
    df = df[cnt >= 2].reset_index(drop=True)
    return dict(df=df, ent=ent, tim=tim, vars=keep, skipped=skipped, periods=periods), None


# ---------------------------------------------------------------- short-panel (static) tests

def _short_panel(df, ent, dep, xs):
    codes, uniq = pd.factorize(df[ent])
    ng = len(uniq)
    y = df[dep].to_numpy(float)
    Xr = df[xs].to_numpy(float)
    n = len(y)
    tcodes = df["_ti"].to_numpy()
    tlev = np.unique(tcodes)
    D = (tcodes[:, None] == tlev[None, 1:]).astype(float)           # time dummies (first period omitted)
    out = {"time_effects": True}

    # regressors usable in FE: varying within units and not collinear with unit and time effects
    Xw, Dw = _demean(Xr, codes, ng), _demean(D, codes, ng)
    within_sd = Xw.std(axis=0)
    tinv = [x for x, s in zip(xs, within_sd) if s < 1e-10]
    cand = [j for j, s in enumerate(within_sd) if s >= 1e-10]
    if cand:
        Z = np.column_stack([Dw, Xw[:, cand]]) if D.shape[1] else Xw[:, cand]
        ind = _independent_cols(Z)
        ok = [cand[i - D.shape[1]] for i in ind if i >= D.shape[1]]
    else:
        ok = []
    coll = [xs[j] for j in cand if j not in ok]
    fx = [xs[j] for j in ok]
    out["fe_dropped"] = dict(time_invariant=tinv, collinear=coll)
    if not fx:
        return {"error": "No regressor varies within units once unit and time effects are removed, so the fixed-effects and Hausman comparisons cannot be run."}

    # pooled OLS with time dummies (all regressors that are not collinear with the time dummies)
    Xp_full = np.column_stack([np.ones(n), Xr, D])
    pk = _independent_cols(Xp_full)
    px = [xs[j - 1] for j in pk if 1 <= j <= len(xs)]
    Xp = Xp_full[:, pk]
    bp, ep, Ip = _ols(y, Xp)
    Vp = _cluster_vcov(Xp, ep, Ip, codes, Xp.shape[1])
    ssr_p = float(ep @ ep)

    # fixed effects (two-way: unit effects by demeaning, time effects by dummies)
    Xf = np.column_stack([Xw[:, ok], Dw]) if D.shape[1] else Xw[:, ok]
    yw = _demean(y, codes, ng)
    bf, ef, If = _ols(yw, Xf)
    kf = Xf.shape[1]
    dof_f = n - ng - kf
    s2e = float(ef @ ef) / dof_f
    Vf_conv = s2e * If
    Vf = _cluster_vcov(Xf, ef, If, codes, kf)
    ssr_f = float(ef @ ef)

    # F test for unit effects: FE (with the same regressors) against pooled OLS
    Xp2 = np.column_stack([np.ones(n), Xr[:, ok], D])
    _, ep2, _ = _ols(y, Xp2)
    ssr_p2 = float(ep2 @ ep2)
    F = ((ssr_p2 - ssr_f) / (ng - 1)) / (ssr_f / dof_f)
    out["ftest"] = dict(stat=_f(F, 2), df1=int(ng - 1), df2=int(dof_f), p=_f(_f_sf(F, ng - 1, dof_f), 4))

    # Breusch-Pagan LM test (Baltagi-Li version, valid for unbalanced panels), from pooled OLS residuals
    Ti = np.bincount(codes, minlength=ng).astype(float)
    se_i = np.bincount(codes, ep, ng)
    lm = (n ** 2 / (2 * float((Ti * (Ti - 1)).sum()))) * ((se_i ** 2).sum() / ssr_p - 1) ** 2
    out["lm"] = dict(stat=_f(lm, 2), p=_f(_chi2_sf(lm, 1), 4))

    # random effects (Swamy-Arora variance components; unbalanced: harmonic mean of T_i)
    Xb = np.column_stack([np.ones(ng), _gmeans(Xr[:, ok], codes, ng), _gmeans(D, codes, ng)])
    yb = _gmeans(y, codes, ng)
    kb = _independent_cols(Xb)
    _, eb, _ = _ols(yb, Xb[:, kb])
    s2b = float(eb @ eb) / max(ng - len(kb), 1)
    Th = ng / float((1 / Ti).sum())
    s2u = max(s2b - s2e / Th, 0.0)
    theta = 1 - np.sqrt(s2e / (Ti * s2u + s2e))
    th = theta[codes]
    Xre_raw = np.column_stack([np.ones(n), Xr[:, ok], D])
    mX = _gmeans(Xre_raw, codes, ng)[codes]
    Xre = Xre_raw - th[:, None] * mX
    yre = y - th * _gmeans(y, codes, ng)[codes]
    br, er, Ir = _ols(yre, Xre)
    s2re = float(er @ er) / (n - Xre.shape[1])
    Vr_conv = s2re * Ir
    Vr = _cluster_vcov(Xre, er, Ir, codes, Xre.shape[1])
    out["re"] = dict(sigma_u=_f(math.sqrt(s2u), 4), sigma_e=_f(math.sqrt(s2e), 4),
                     rho=_f(s2u / (s2u + s2e), 3), theta_mean=_f(float(theta.mean()), 3))

    # conventional Hausman test on the regressors common to FE and RE
    kk = len(ok)
    d = bf[:kk] - br[1:1 + kk]
    Vd = Vf_conv[:kk, :kk] - Vr_conv[1:1 + kk, 1:1 + kk]
    try:
        w, U = np.linalg.eigh((Vd + Vd.T) / 2)
        pos = w > 1e-12 * max(1, abs(w).max())
        H = float(d @ (U[:, pos] @ np.diag(1 / w[pos]) @ U[:, pos].T) @ d)
        hdf = int(pos.sum())
        out["hausman"] = dict(stat=_f(H, 2), df=hdf, p=_f(_chi2_sf(H, hdf), 4), pd=bool(pos.all()))
    except Exception:
        out["hausman"] = None

    # Mundlak / correlated random effects: pooled OLS on X, unit means of X and time dummies, cluster-robust Wald test
    Xm = np.column_stack([np.ones(n), Xr[:, ok], _gmeans(Xr[:, ok], codes, ng)[codes], D])
    bm, em, Im = _ols(y, Xm)
    Vm = _cluster_vcov(Xm, em, Im, codes, Xm.shape[1])
    sl = slice(1 + kk, 1 + 2 * kk)
    gm = bm[sl]
    W = float(gm @ np.linalg.pinv(Vm[sl, sl]) @ gm)
    out["mundlak"] = dict(stat=_f(W, 2), df=kk, p=_f(_chi2_sf(W, kk), 4))

    # coefficient table (cluster-robust standard errors by entity)
    rows = []
    for j, x in enumerate(xs):
        r = dict(var=x)
        if x in px:
            i = 1 + px.index(x)
            r["pols"], r["pols_se"] = _f(bp[i]), _f(math.sqrt(max(Vp[i, i], 0)))
        if x in fx:
            i = fx.index(x)
            r["re"], r["re_se"] = _f(br[1 + i]), _f(math.sqrt(max(Vr[1 + i, 1 + i], 0)))
            r["fe"], r["fe_se"] = _f(bf[i]), _f(math.sqrt(max(Vf[i, i], 0)))
        rows.append(r)
    out["coefs"] = rows
    out["within_r2"] = _f(1 - ssr_f / float(yw @ yw), 3)

    fp, lp, mp = out["ftest"]["p"], out["lm"]["p"], out["mundlak"]["p"]
    if fp >= 0.05 and lp >= 0.05:
        model = "POLS"
    elif mp < 0.05:
        model = "FE"
    elif lp < 0.05:
        model = "RE"
    else:
        model = "FE"
    out["static_model"] = model
    return out


# ---------------------------------------------------------------- long-panel tests

def _wide(df, ent, v):
    return df.pivot(index="_ti", columns=ent, values=v).sort_index()


def _cd(W):
    """Pesaran (2004/2015) CD test on a T x N wide array (NaN allowed). Returns CD, p, mean |rho|."""
    A = np.asarray(W, float)
    N = A.shape[1]
    s, sabs, npairs = 0.0, 0.0, 0
    M = ~np.isnan(A)
    for i in range(N - 1):
        for j in range(i + 1, N):
            m = M[:, i] & M[:, j]
            Tij = int(m.sum())
            if Tij < 3:
                continue
            a, b = A[m, i], A[m, j]
            a, b = a - a.mean(), b - b.mean()
            den = math.sqrt(float(a @ a) * float(b @ b))
            if den == 0:
                continue
            r = float(a @ b) / den
            s += math.sqrt(Tij) * r
            sabs += abs(r)
            npairs += 1
    if npairs == 0:
        return None, None, None
    cd = math.sqrt(2.0 / (N * (N - 1))) * s
    return cd, 2 * (1 - _ncdf(abs(cd))), sabs / npairs


def _fisher_adf(W, maxlag):
    from statsmodels.tsa.stattools import adfuller
    ps = []
    for c in W.columns:
        x = W[c].dropna().to_numpy(float)
        if len(x) < 8 or np.ptp(x) == 0:
            continue
        ml = max(0, min(maxlag, len(x) // 3 - 2))
        try:
            ps.append(adfuller(x, maxlag=ml, autolag="AIC" if ml > 0 else None, regression="c")[1])
        except Exception:
            pass
    if len(ps) < 2:
        return None
    ps = np.clip(np.array(ps), 1e-12, 1 - 1e-12)
    N = len(ps)
    P = -2 * np.log(ps).sum()
    Z = sum(_ninv(p) for p in ps) / math.sqrt(N)
    return dict(P=_f(P, 2), P_p=_f(_chi2_sf(P, 2 * N), 4), Z=_f(Z, 3), Z_p=_f(_ncdf(Z), 4), n=int(N))


def _cips(W, p=1):
    """Pesaran (2007) CIPS with an intercept and p lags on a T x N wide array (NaN allowed)."""
    A = np.asarray(W, float)
    yb = np.nanmean(A, axis=1)
    dA, dyb = np.diff(A, axis=0), np.diff(yb)
    T = A.shape[0]
    ts = []
    for i in range(A.shape[1]):
        rows_y, rows_X = [], []
        for t in range(p + 1, T):             # dy at time t uses index t-1 in the diff arrays
            k = t - 1
            vals = [dA[k, i], A[t - 1, i], yb[t - 1], dyb[k]] + [dyb[k - j] for j in range(1, p + 1)] + [dA[k - j, i] for j in range(1, p + 1)]
            if np.all(np.isfinite(vals)):
                rows_y.append(vals[0]); rows_X.append([1.0] + vals[1:])
        L, kx = len(rows_y), 4 + 2 * p
        if L < kx + 5:
            continue
        X, yv = np.array(rows_X), np.array(rows_y)
        b, e, I = _ols(yv, X)
        s2 = float(e @ e) / (L - kx)
        se = math.sqrt(s2 * I[1, 1])
        if se > 0:
            ts.append(b[1] / se)
    if len(ts) < 2:
        return None
    stat = float(np.mean(ts))
    Nn, Tn = len(ts), int(np.isfinite(A).any(axis=1).sum())
    return dict(stat=_f(stat, 3), p=_f(_cips_p(stat, Nn, Tn), 4), n=Nn, T=Tn)


def _cips_p(stat, N, T):
    Q = CIPS_Q
    if not Q:
        return None
    Ns, Ts, probs, q = np.array(Q["N"], float), np.array(Q["T"], float), np.array(Q["probs"]), np.array(Q["q"])
    def interp_axis(vals, x):
        x = min(max(x, vals[0]), vals[-1])
        j = int(np.searchsorted(vals, x))
        j = min(max(j, 1), len(vals) - 1)
        w = (math.log(x) - math.log(vals[j - 1])) / (math.log(vals[j]) - math.log(vals[j - 1]))
        return j, w
    i, wi = interp_axis(Ns, N)
    j, wj = interp_axis(Ts, T)
    qq = ((1 - wi) * ((1 - wj) * q[i - 1, j - 1] + wj * q[i - 1, j]) + wi * ((1 - wj) * q[i, j - 1] + wj * q[i, j]))
    if stat <= qq[0]:
        return float(probs[0]) / 2
    if stat >= qq[-1]:
        return min(1.0, float(probs[-1]) + (1 - probs[-1]) / 2)
    return float(np.interp(stat, qq, probs))


def _py_delta(df, ent, dep, xs):
    """Pesaran-Yamagata (2008) slope-homogeneity test, model y_it = a_i + x_it' b_i + e_it."""
    k = len(xs)
    units = []
    for _, g in df.groupby(ent, sort=False):
        if len(g) <= k + 2:
            continue
        y = g[dep].to_numpy(float); X = g[xs].to_numpy(float)
        y = y - y.mean(); X = X - X.mean(axis=0)
        if np.linalg.matrix_rank(X) < k:
            continue
        units.append((y, X))
    N = len(units)
    if N < 5:
        return None
    XX = sum(X.T @ X for _, X in units); Xy = sum(X.T @ y for y, X in units)
    bfe = np.linalg.solve(XX, Xy)
    s2 = [float((y - X @ bfe) @ (y - X @ bfe)) / (len(y) - 1) for y, X in units]
    A = sum(X.T @ X / s for (y, X), s in zip(units, s2)); c = sum(X.T @ y / s for (y, X), s in zip(units, s2))
    bw = np.linalg.solve(A, c)
    S = 0.0
    for (y, X), s in zip(units, s2):
        bi = np.linalg.solve(X.T @ X, X.T @ y)
        d = bi - bw
        S += float(d @ (X.T @ X) @ d) / s
    T = float(np.mean([len(y) for y, _ in units]))
    dl = math.sqrt(N) * (S / N - k) / math.sqrt(2 * k)
    da = math.sqrt(N) * (S / N - k) / math.sqrt(2 * k * (T - k - 1) / (T + 1))
    return dict(delta=_f(dl, 3), delta_p=_f(1 - _ncdf(dl), 4), delta_adj=_f(da, 3), delta_adj_p=_f(1 - _ncdf(da), 4), n=N)


def _lrcov(W, bw):
    T = W.shape[0]
    G = W.T @ W / T
    for j in range(1, bw + 1):
        Gj = W[j:].T @ W[:-j] / T
        G = G + (1 - j / (bw + 1)) * (Gj + Gj.T)
    return G


def _kao(df, ent, dep, xs, p=1):
    """Kao (1999) residual-based ADF test of no cointegration (homogeneous slopes, unit intercepts)."""
    codes, uniq = pd.factorize(df[ent]); ng = len(uniq)
    y = df[dep].to_numpy(float); X = df[xs].to_numpy(float)
    b, _, _ = _ols(_demean(y, codes, ng), _demean(X, codes, ng))
    df = df.assign(_e=_demean(y - X @ b, codes, ng))
    Sig, Om, rows_y, rows_X = [], [], [], []
    for _, g in df.groupby(ent, sort=False):
        e = g["_e"].to_numpy(); Xi = g[xs].to_numpy(float)
        if len(e) < p + 5:
            continue
        de = np.diff(e)
        for t in range(p + 1, len(e)):
            rows_y.append(e[t]); rows_X.append([e[t - 1]] + [de[t - 1 - j] for j in range(1, p + 1)])
        Wm = np.column_stack([e[1:], np.diff(Xi, axis=0)])
        Wm = Wm - Wm.mean(axis=0)
        Sig.append(Wm.T @ Wm / len(Wm)); Om.append(_lrcov(Wm, max(1, int(4 * (len(Wm) / 100) ** (2 / 9)))))
    if len(Sig) < 5:
        return None
    N = len(Sig)
    Sg, O = np.mean(Sig, axis=0), np.mean(Om, axis=0)
    sv2 = Sg[0, 0] - Sg[0, 1:] @ np.linalg.pinv(Sg[1:, 1:]) @ Sg[1:, 0]
    s0v2 = O[0, 0] - O[0, 1:] @ np.linalg.pinv(O[1:, 1:]) @ O[1:, 0]
    if sv2 <= 0 or s0v2 <= 0:
        return None
    Xa, ya = np.array(rows_X), np.array(rows_y)
    bb, ee, I = _ols(ya, Xa)
    s2 = float(ee @ ee) / (len(ya) - Xa.shape[1])
    t_adf = (bb[0] - 1) / math.sqrt(s2 * I[0, 0])
    sv, s0v = math.sqrt(sv2), math.sqrt(s0v2)
    stat = (t_adf + math.sqrt(6 * N) * sv / (2 * s0v)) / math.sqrt(s0v2 / (2 * sv2) + 3 * sv2 / (10 * s0v2))
    return dict(stat=_f(stat, 3), p=_f(_ncdf(stat), 4), n=N)


def _fisher_eg(df, ent, dep, xs):
    """Maddala-Wu (1999) Fisher combination of unit-by-unit Engle-Granger tests (heterogeneous slopes)."""
    from statsmodels.tsa.stattools import coint
    ps = []
    for _, g in df.groupby(ent, sort=False):
        if len(g) < 12 + 2 * len(xs):
            continue
        try:
            ps.append(coint(g[dep].to_numpy(float), g[xs].to_numpy(float), trend="c", maxlag=1, autolag=None)[1])
        except Exception:
            pass
    if len(ps) < 5:
        return None
    ps = np.clip(np.array(ps), 1e-12, 1 - 1e-12)
    N = len(ps)
    P = -2 * np.log(ps).sum(); Z = sum(_ninv(p) for p in ps) / math.sqrt(N)
    return dict(P=_f(P, 2), P_p=_f(_chi2_sf(P, 2 * N), 4), Z=_f(Z, 3), Z_p=_f(_ncdf(Z), 4), n=int(N))


def _long_panel(df, ent, dep, xs, T):
    out = {}
    V = [dep] + xs
    maxlag = max(1, min(4, int(12 * (T / 100) ** 0.25)))
    # cross-sectional dependence of each variable and of the fixed-effects residuals
    cds = []
    for v in V:
        cd, p, ar = _cd(_wide(df, ent, v).to_numpy())
        cds.append(dict(var=v, cd=_f(cd, 2), p=_f(p, 4), absr=_f(ar, 3)))
    out["cd"] = cds
    codes, uniq = pd.factorize(df[ent]); ng = len(uniq)
    yw = _demean(df[dep].to_numpy(float), codes, ng); Xw = _demean(df[xs].to_numpy(float), codes, ng)
    b, e, I = _ols(yw, Xw)
    cd, p, ar = _cd(df.assign(_r=e).pivot(index="_ti", columns=ent, values="_r").to_numpy())
    out["cd_resid"] = dict(cd=_f(cd, 2), p=_f(p, 4), absr=_f(ar, 3))
    out["slope"] = _py_delta(df, ent, dep, xs)

    # panel unit-root tests on levels, first and (if needed) second differences
    ur, orders = [], []
    for v, c in zip(V, cds):
        W = _wide(df, ent, v)
        use = "CIPS" if (c["p"] is not None and c["p"] < 0.05) else "Fisher"
        res = {"var": v, "test": use}
        order = None
        for d, lab in ((0, "lev"), (1, "d1"), (2, "d2")):
            Wd = W if d == 0 else W.diff(d) if d == 1 else W.diff().diff()
            fi, ci = _fisher_adf(Wd, maxlag), _cips(Wd.to_numpy())
            res[lab] = dict(fisher=fi, cips=ci)
            pv = (ci or {}).get("p") if use == "CIPS" else (fi or {}).get("Z_p")
            if pv is None:
                pv = (fi or {}).get("Z_p") if use == "CIPS" else (ci or {}).get("p")
            res[lab]["p"] = pv
            if pv is not None and pv < 0.05:
                order = d
                break
        res["order"] = 3 if order is None else order
        orders.append(res["order"])
        ur.append(res)
    out["unitroot"] = ur
    out["orders"] = orders

    cdr = out["cd_resid"]["p"] is not None and out["cd_resid"]["p"] < 0.05
    het = out["slope"] is not None and out["slope"]["delta_adj_p"] is not None and out["slope"]["delta_adj_p"] < 0.05
    out["cd_flag"], out["het_flag"] = bool(cdr), bool(het)
    if max(orders) >= 2:
        model = "I2"
    elif all(o == 0 for o in orders):
        model = "MG" if het else "FEDK"
    else:
        if all(o == 1 for o in orders):
            k1, f1 = _kao(df, ent, dep, xs), _fisher_eg(df, ent, dep, xs)
            used = "fisher" if het else "kao"
            pv = (f1 or {}).get("Z_p") if used == "fisher" else (k1 or {}).get("p")
            if pv is None:
                used = "kao" if used == "fisher" else "fisher"
                pv = (f1 or {}).get("Z_p") if used == "fisher" else (k1 or {}).get("p")
            coint = pv is not None and pv < 0.05
            out["coint"] = dict(kao=k1, fisher=f1, used=used, p=pv, coint=bool(coint))
            if not coint:
                model = "DIFF"
            else:
                model = "CSARDL" if cdr else "PMG"
        else:
            model = "CSARDL" if cdr else "PMG"
    out["model"] = model
    return out


# ---------------------------------------------------------------- main entry point

def analyze(csv_text, max_vars=6):
    warnings.simplefilter("ignore")
    rd, err = _read(csv_text, max_vars)
    if err:
        return json.dumps({"error": err})
    df, ent, tim, V, periods = rd["df"], rd["ent"], rd["tim"], rd["vars"], rd["periods"]
    dep, xs = V[0], V[1:]
    Ti = df.groupby(ent).size()
    N, Tn, n = int(len(Ti)), int(len(periods)), int(len(df))
    if N < 5:
        return json.dumps({"error": f"Only {N} entities with at least two periods were found. A panel needs several entities; with one or two, use the Time-Series Guide."})
    if n < 30:
        return json.dumps({"error": f"Only {n} complete entity–period rows were found. These tests need at least 30."})
    Tmed = float(Ti.median())
    out = dict(n=n, N=N, T=Tn, Tmin=int(Ti.min()), Tmax=int(Ti.max()), Tmed=_f(Tmed, 1), balanced=bool(Ti.min() == Ti.max() == Tn),
               entity=ent, time=tim, first=str(periods[0]), last=str(periods[-1]), vars=V, dep=dep, skipped=rd["skipped"],
               branch="long" if Tmed >= LONG_T else "short", long_t=LONG_T)

    # charts: cross-sectional mean over time with the 10th-90th percentile band, and the same for first differences
    g = df.groupby("_ti")
    def band(s):
        q = s.groupby(df["_ti"]).quantile([0.1, 0.9]).unstack()
        return dict(mean=[_f(v, 5) for v in s.groupby(df["_ti"]).mean().values], lo=[_f(v, 5) for v in q[0.1].values], hi=[_f(v, 5) for v in q[0.9].values])
    out["series"] = {v: band(df[v]) for v in V}
    dfd = df.copy()
    for v in V:
        dfd[v] = df.groupby(ent)[v].diff()
    out["dseries"] = {v: [_f(x, 5) for x in dfd.groupby("_ti")[v].mean().values[1:]] for v in V}
    # a few individual units for the spaghetti chart
    top = Ti.sort_values(ascending=False).index[:6]
    out["units"] = {str(u): [_f(x, 5) for x in df[df[ent] == u].set_index("_ti")[dep].reindex(range(Tn)).values] for u in top}

    summ = []
    codes, uniq = pd.factorize(df[ent])
    for v in V:
        x = df[v].to_numpy(float)
        m = _gmeans(x, codes, len(uniq))
        w = x - m[codes] + x.mean()
        summ.append(dict(var=v, mean=_f(x.mean()), sd=_f(x.std(ddof=1)), between=_f(m.std(ddof=1)), within=_f(w.std(ddof=1)), min=_f(x.min()), max=_f(x.max())))
    out["summary"] = summ
    out["corr"] = [[_f(c, 3) for c in row] for row in df[V].corr().values]

    if out["branch"] == "short":
        r = _short_panel(df, ent, dep, xs)
        if "error" in r:
            return json.dumps({"error": r["error"]})
        out["static"] = r
        out["model"] = r["static_model"]
    else:
        out.update(_long_panel(df, ent, dep, xs, Tn))
    return json.dumps(out)


# ---------------------------------------------------------------- public functions for use in your own scripts
# Long-format data must be sorted by entity and time. A "wide" frame has one row per period and one column per unit,
# for example wide = df.pivot(index="year", columns="country", values="log_gdp").

def cd_test(wide):
    """Pesaran CD test. Returns (CD statistic, p-value, mean absolute correlation)."""
    return _cd(wide.to_numpy(float))


def cips_test(wide, p=1):
    """Pesaran (2007) CIPS unit-root test with an intercept and p lags (p-value from simulated quantiles, p=1)."""
    return _cips(wide.to_numpy(float), p)


def fisher_adf_test(wide, maxlag=4):
    """Maddala-Wu / Choi Fisher-type ADF unit-root test (combines unit-by-unit ADF p-values)."""
    return _fisher_adf(wide, maxlag)


def slope_test(df, entity, y, xs):
    """Pesaran-Yamagata (2008) slope-homogeneity test (delta and adjusted delta, upper-tail p-values)."""
    return _py_delta(df.sort_values(entity, kind="stable"), entity, y, list(xs))


def kao_test(df, entity, y, xs):
    """Kao (1999) ADF test of no cointegration."""
    return _kao(df, entity, y, list(xs))


def fisher_eg_test(df, entity, y, xs):
    """Maddala-Wu Fisher combination of unit-by-unit Engle-Granger cointegration tests."""
    return _fisher_eg(df, entity, y, list(xs))
