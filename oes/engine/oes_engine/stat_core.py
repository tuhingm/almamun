"""Shared helpers of the Statistics Division: factor coding, least squares, linear hypothesis tests, effect-size
intervals and assumption checks. Pure NumPy and SciPy. Same result format as every other OES model (core.py)."""
import numpy as np
import pandas as pd
from scipy import optimize, stats

from .core import OESError, as_list, check_columns, f


def level_of(v):
    """Text label of a factor level (integers stored as floats print without the trailing .0)."""
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def numeric_col(df, col, label=None):
    check_columns(df, [col])
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().sum() == 0:
        raise OESError(f"{label or col} has no numeric values.")
    return s


def clean_frame(df, numeric=(), factors=(), keep=()):
    """Copy with the numeric columns converted, the factor columns as text, and rows with a missing value dropped."""
    cols = list(dict.fromkeys(list(numeric) + list(factors) + list(keep)))
    check_columns(df, cols)
    d = df[cols].copy()
    for c in numeric:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)
    n0 = len(d)
    d = d.dropna(subset=cols)
    for c in factors:
        d[c] = d[c].map(level_of)
    if len(d) == 0:
        raise OESError("No complete rows remain for the selected variables.")
    return d.reset_index(drop=True), n0 - len(d)


def levels(series):
    """Levels in the order they first appear when they are not all numbers, otherwise in numeric order."""
    u = list(pd.unique(series))
    try:
        return sorted(u, key=float)
    except (TypeError, ValueError):
        return u


def effect_cols(series, lv=None):
    """Sum-to-zero (effect) coding: k-1 columns; the last level is coded -1 in every column."""
    lv = lv or levels(series)
    k = len(lv)
    idx = series.map({v: i for i, v in enumerate(lv)}).to_numpy()
    X = np.zeros((len(series), k - 1))
    for j in range(k - 1):
        X[:, j] = (idx == j).astype(float)
    X[idx == k - 1, :] = -1.0
    return X, lv


def dummy_cols(series, lv=None):
    lv = lv or levels(series)
    idx = series.map({v: i for i, v in enumerate(lv)}).to_numpy()
    X = np.zeros((len(series), len(lv) - 1))
    for j in range(1, len(lv)):
        X[:, j - 1] = (idx == j).astype(float)
    return X, lv


def lstsq(X, Y):
    """Least squares with rank detection. Returns coefficients, residuals, rank and (X'X)^+."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    rank = int(np.linalg.matrix_rank(X))
    XtXp = np.linalg.pinv(X.T @ X)
    return beta, Y - X @ beta, rank, XtXp


def rss_rank(X, y):
    if X.shape[1] == 0:
        return float(np.sum(np.asarray(y, float) ** 2)), 0
    _, e, r, _ = lstsq(X, y)
    return float(np.sum(e ** 2)), r


def term_tests(y, base, terms, kind="III"):
    """Sequential-free sums of squares of the model terms.

    base: columns always in the model (intercept and, for ANCOVA, the covariates).
    terms: ordered dict name -> (frozenset of factor names, design columns).
    kind: "II" (each term adjusted for the terms that do not contain it) or "III" (adjusted for all other terms).
    Returns {name: (SS, df)} and the residual (SS, df) of the full model."""
    full_cols = [base] + [m for _, m in terms.values()]
    Xf = np.hstack(full_cols)
    rss_f, rank_f = rss_rank(Xf, y)
    n = len(y)
    out = {}
    for name, (fs, M) in terms.items():
        if kind == "III":
            keep = [m for nm, (_, m) in terms.items() if nm != name]
        else:
            keep = [m for nm, (g, m) in terms.items() if nm != name and not fs <= g]
        Xr = np.hstack([base] + keep) if keep else base
        if kind == "II":
            Xfull_t = np.hstack([Xr, M])
            rss_full, r_full = rss_rank(Xfull_t, y)
            rss_red, r_red = rss_rank(Xr, y)
            out[name] = (rss_red - rss_full, r_full - r_red)
        else:
            rss_red, r_red = rss_rank(Xr, y)
            out[name] = (rss_red - rss_f, rank_f - r_red)
    return out, (rss_f, n - rank_f)


def mlm_fit(Y, X):
    """Multivariate linear model Y = X B + E. Returns B, residual SSP matrix E, rank, (X'X)^+."""
    B, R, rank, XtXp = lstsq(X, Y)
    return B, R.T @ R, rank, XtXp


def mlm_hypothesis(B, E, XtXp, L, df_e):
    """H = (LB)' [L (X'X)^+ L']^-1 (LB) for the hypothesis L B = 0 and the residual SSP matrix E."""
    LB = L @ B
    M = L @ XtXp @ L.T
    H = LB.T @ np.linalg.pinv(M) @ LB
    return H, int(np.linalg.matrix_rank(L))


def mv_statistics(H, E, q, df_e):
    """Wilks, Pillai, Hotelling-Lawley and Roy statistics with their F approximations (SAS/R conventions).

    H: hypothesis SSP matrix, E: error SSP matrix, q: hypothesis df, df_e: error df. p is the number of responses."""
    p = E.shape[0]
    Einv_H = np.linalg.solve(E, H)
    ev = np.sort(np.clip(np.linalg.eigvals(Einv_H).real, 0, None))[::-1][:min(p, q)]
    s = min(p, q)
    m = (abs(p - q) - 1) / 2.0
    nn = (df_e - p - 1) / 2.0
    out = {}
    # Wilks' lambda, Rao's F approximation
    lam = float(np.prod(1.0 / (1.0 + ev)))
    r_ = df_e - (p - q + 1) / 2.0
    u = (p * q - 2) / 4.0
    t = np.sqrt((p ** 2 * q ** 2 - 4) / (p ** 2 + q ** 2 - 5)) if p ** 2 + q ** 2 - 5 > 0 else 1.0
    df1 = p * q
    df2 = r_ * t - 2 * u
    Fw = ((1 - lam ** (1 / t)) / lam ** (1 / t)) * (df2 / df1) if lam > 0 else np.inf
    out["Wilks' lambda"] = (lam, Fw, df1, df2)
    # Pillai's trace
    V = float(np.sum(ev / (1 + ev)))
    d1 = s * (2 * m + s + 1)
    d2 = s * (2 * nn + s + 1)
    Fp = (2 * nn + s + 1) / (2 * m + s + 1) * V / (s - V) if s - V > 0 else np.inf
    out["Pillai's trace"] = (V, Fp, d1, d2)
    # Hotelling-Lawley trace
    Ut = float(np.sum(ev))
    if nn > 0:
        d2h = 2 * (s * nn + 1)
        Fh = d2h * Ut / (s * s * (2 * m + s + 1))
        out["Hotelling-Lawley trace"] = (Ut, Fh, s * (2 * m + s + 1), d2h)
    else:
        d2h = s * (df_e - q - 1) + 2
        Fh = d2h * Ut / (s * s * (2 * m + s + 1))
        out["Hotelling-Lawley trace"] = (Ut, Fh, s * (2 * m + s + 1), d2h)
    # Roy's largest root (upper bound on F)
    th = float(ev[0]) if len(ev) else 0.0
    r_roy = max(p, q)
    Fr = th * (df_e - r_roy + q) / r_roy
    out["Roy's largest root"] = (th, Fr, r_roy, df_e - r_roy + q)
    return out, ev


def noncentral_ci(stat_value, df1, df2, level=0.95):
    """Confidence limits for the noncentrality parameter of an F statistic (inversion of the noncentral F)."""
    a = (1 - level) / 2
    F = max(float(stat_value), 0.0)
    if F <= 0 or not np.isfinite(F):
        return 0.0, 0.0
    cdf = lambda lam: stats.ncf.cdf(F, df1, df2, lam)
    hi_cap = max(100.0, 20 * F * df1)
    if cdf(1e-9) < 1 - a:
        lo = 0.0
    else:
        lo = optimize.brentq(lambda l: cdf(l) - (1 - a), 1e-9, hi_cap)
    if cdf(1e-9) < a:
        hi = 0.0
    else:
        try:
            hi = optimize.brentq(lambda l: cdf(l) - a, 1e-9, hi_cap * 10)
        except ValueError:
            hi = np.inf
    return float(lo), float(hi)


def eta2_ci(F, df1, df2, n, level=0.95):
    lo, hi = noncentral_ci(F, df1, df2, level)
    return lo / (lo + n), (hi / (hi + n) if np.isfinite(hi) else 1.0)


def normality_tests(resid):
    """Shapiro-Wilk (n <= 5000) and D'Agostino-Pearson on residuals; rows for the diagnostics table."""
    r = np.asarray(resid, float)
    rows = []
    if 3 <= len(r) <= 5000 and np.ptp(r) > 0:
        w = stats.shapiro(r)
        rows.append(["Shapiro-Wilk normality of residuals", f(w.statistic, 4), f(w.pvalue, 4), "H0: residuals are normal"])
    if len(r) >= 20 and np.ptp(r) > 0:
        k = stats.normaltest(r)
        rows.append(["D'Agostino-Pearson normality", f(k.statistic, 4), f(k.pvalue, 4), "H0: residuals are normal"])
    return rows


def qq_points(resid):
    r = np.sort(np.asarray(resid, float))
    n = len(r)
    z = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    sd = r.std(ddof=1) or 1.0
    return z, (r - r.mean()) / sd


def thin(x, y, k=200):
    """At most k evenly spaced points, so that large samples keep the chart light."""
    n = len(x)
    if n <= k:
        return list(x), list(y)
    idx = np.unique(np.linspace(0, n - 1, k).astype(int))
    return [x[i] for i in idx], [y[i] for i in idx]


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def bonferroni(p):
    p = np.asarray(p, float)
    return np.minimum(1.0, p * len(p))


def benjamini_hochberg(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    run = 1.0
    for k, i in enumerate(order):
        rank = m - k
        run = min(run, p[i] * m / rank)
        adj[i] = run
    return adj
