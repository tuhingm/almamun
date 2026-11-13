"""Tests behind the Bayesian and Markov-switching model selectors. The same file runs offline (to precompute the example
datasets) and in the browser under Pyodide (for uploaded data), so both give the same numbers.

    analyse(csv_text, time=None, unit=None, y=None, aim="var") -> dict of facts that the page's rules read.
"""
import io, json, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
TIME_NAMES = ("date", "time", "year", "quarter", "period", "month", "t")


def _is_num(s):
    return pd.api.types.is_numeric_dtype(s)


def detect(df, time=None, unit=None):
    cols = list(df.columns)
    if time in (None, "", "none"):
        time = next((c for c in cols if str(c).lower() in TIME_NAMES), None)
        if time is None:
            time = next((c for c in cols if not _is_num(df[c]) and df[c].nunique() == len(df)), None)
    if unit in (None, ""):
        unit = None
        for c in cols:
            if c != time and not _is_num(df[c]) and 1 < df[c].nunique() < len(df) / 2:
                unit = c
                break
    elif unit == "none":
        unit = None
    return time, unit


def _ds(v, n=90):
    v = np.asarray(v, float)
    if len(v) > n:
        idx = np.linspace(0, len(v) - 1, n).round().astype(int)
        v = v[idx]
    return [None if not np.isfinite(a) else round(float(a), 4) for a in v]


def adf_kpss(x):
    from statsmodels.tsa.stattools import adfuller, kpss
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 20 or np.nanstd(x) == 0:
        return dict(adf_p=None, kpss_p=None, order="n/a")
    p = adfuller(x, regression="c", autolag="AIC")[1]
    pd_ = adfuller(np.diff(x), regression="c", autolag="AIC")[1]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k = kpss(x, regression="c", nlags="auto")[1]
    except Exception:
        k = None
    if p < 0.05 and (k is None or k > 0.05):
        o = "I(0)"
    elif p >= 0.05 and pd_ < 0.05:
        o = "I(1)"
    elif p < 0.05:
        o = "I(0)"            # ADF rejects; KPSS disagrees: treat as stationary but borderline
    else:
        o = "I(1) or I(2)"
    return dict(adf_p=float(p), kpss_p=None if k is None else float(k), order=o)


def _lag_design(Y, p=1):
    T = len(Y)
    X = np.column_stack([np.ones(T - p)] + [Y[p - j - 1:T - j - 1] for j in range(p)])
    return Y[p:], X


def _ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return b, e, float(e @ e)


def volatility(Y, names):
    from statsmodels.stats.diagnostic import het_arch
    out = {}
    for i, n in enumerate(names):
        y, X = _lag_design(Y, 1)
        _, e, _ = _ols(y[:, i], X)
        h = len(e) // 2
        v1, v2 = float(np.var(e[:h])), float(np.var(e[h:]))
        try:
            arch_p = float(het_arch(e, nlags=4)[1])
        except Exception:
            arch_p = None
        lev = float(stats.levene(e[:h], e[h:])[1])
        out[n] = dict(ratio=v2 / v1 if v1 > 0 else None, levene_p=lev, arch_p=arch_p)
    return out


def instability(Y, names):
    from statsmodels.stats.diagnostic import breaks_cusumolsresid
    out = {}
    y, X = _lag_design(Y, 1)
    for i, n in enumerate(names):
        b, e, _ = _ols(y[:, i], X)
        try:
            p = float(breaks_cusumolsresid(e, ddof=X.shape[1])[1])
        except Exception:
            p = None
        # Chow-type sup-F over the middle 70% of the sample
        k, T = X.shape[1], len(y)
        best, at = 0.0, None
        _, _, ssr = _ols(y[:, i], X)
        for s in range(int(.15 * T), int(.85 * T), max(1, T // 60)):
            try:
                _, _, s1 = _ols(y[:s, i], X[:s]); _, _, s2 = _ols(y[s:, i], X[s:])
            except Exception:
                continue
            F = ((ssr - s1 - s2) / k) / ((s1 + s2) / (T - 2 * k))
            if F > best:
                best, at = F, s
        out[n] = dict(cusum_p=p, supF=float(best), k=int(k))
    return out


def threshold_test(Y, names, nboot=199, seed=0):
    """Hansen (1996)-type sup-LR test of one threshold in a VAR(1) (all equations, common threshold) for every candidate
    threshold variable; the p-value comes from a fixed-regressor residual bootstrap."""
    rng = np.random.default_rng(seed)
    y, X = _lag_design(Y, 1)
    T, k = X.shape
    n = y.shape[1]
    B0 = np.linalg.lstsq(X, y, rcond=None)[0]
    E0 = y - X @ B0
    ld0 = np.linalg.slogdet(E0.T @ E0 / T)[1]
    res = {}
    for j, zn in enumerate(names):
        z = Y[:-1, j]
        grid = np.unique(np.quantile(z, np.linspace(.15, .85, 21)))

        def sup(yv, E, ld):
            m, at = 0.0, None
            for g in grid:
                lo = z <= g
                if lo.sum() < 2 * k or (~lo).sum() < 2 * k:
                    continue
                S = 0
                for msk in (lo, ~lo):
                    Bm = np.linalg.lstsq(X[msk], yv[msk], rcond=None)[0]
                    Em = yv[msk] - X[msk] @ Bm
                    S = S + Em.T @ Em
                lr = T * (ld - np.linalg.slogdet(S / T)[1])
                if lr > m:
                    m, at = lr, g
            return m, at
        L0, g0 = sup(y, E0, ld0)
        cnt = 0
        for _ in range(nboot):
            ys = X @ B0 + E0 * rng.standard_normal((T, 1))
            Bs = np.linalg.lstsq(X, ys, rcond=None)[0]
            Es = ys - X @ Bs
            cnt += sup(ys, Es, np.linalg.slogdet(Es.T @ Es / T)[1])[0] >= L0
        res[zn] = dict(supLR=float(L0), threshold=None if g0 is None else float(g0), p=float((cnt + 1) / (nboot + 1)))
    best = min(res, key=lambda q: (res[q]["p"], -res[q]["supLR"]))
    return dict(by_variable=res, best=best)


def ms_evidence(x):
    """Two-regime Markov-switching mean model for one series against the linear (constant) model."""
    import statsmodels.api as sm
    x = pd.Series(np.asarray(x, float)).dropna().reset_index(drop=True)
    if len(x) < 40:
        return None
    best = None
    for sv in (True,):
        try:
            mod = sm.tsa.MarkovRegression(x, k_regimes=2, trend="c", switching_variance=sv)
            for sd in range(3):
                np.random.seed(sd)
                r = mod.fit(disp=False, **({"search_reps": 15} if sd else {}))
                if np.isfinite(r.llf) and (best is None or r.llf > best.llf):
                    best = r
        except Exception:
            continue
    if best is None:
        return None
    n = len(x)
    ll1 = float(np.sum(stats.norm.logpdf(x, x.mean(), x.std(ddof=0))))
    k1, k2 = 2, len(best.params)
    lr = 2 * (float(best.llf) - ll1)
    sm_ = np.asarray(best.smoothed_marginal_probabilities)
    share = sm_.argmax(1)
    cnt = np.bincount(share, minlength=2)
    p = np.asarray(best.params)
    return dict(lr=float(lr), aic_linear=float(2 * k1 - 2 * ll1), aic_ms=float(best.aic),
                bic_linear=float(k1 * np.log(n) - 2 * ll1), bic_ms=float(best.bic),
                min_regime=int(cnt.min()), n=int(n), min_share=float(cnt.min() / n))


def analyse(csv_text, time=None, unit=None, y=None, aim="var", vars=None, fast=False):
    warnings.simplefilter("ignore")
    df = pd.read_csv(io.StringIO(csv_text))
    time, unit = detect(df, time, unit)
    num = [c for c in df.columns if _is_num(df[c]) and c not in (time, unit)]
    if vars:
        num = [c for c in vars if c in num] or num
    F = dict(rows=int(len(df)), time=time, unit=unit, vars=num, aim=aim)
    if len(num) == 0:
        F["error"] = "No numeric columns found."
        return F
    # frequency pattern -------------------------------------------------------------------------------------------
    share = {c: float(df[c].notna().mean()) for c in num}
    sparse = [c for c in num if share[c] < 0.5]
    dense = [c for c in num if share[c] > 0.9]
    F["share_obs"] = share
    F["mixed_freq"] = bool(sparse and dense and not unit)
    F["sparse_vars"] = sparse
    if F["mixed_freq"]:
        num_t = dense
    else:
        num_t = num
    F["unit_count"] = int(df[unit].nunique()) if unit else 1
    F["periods"] = int(df[time].nunique()) if time else (int(len(df) / F["unit_count"]))
    if unit:
        d = df.copy()
        d[num_t] = d[num_t].astype(float) - d.groupby(unit)[num_t].transform("mean")
        avg = d.groupby(time)[num_t].mean() if time else d.groupby(d.groupby(unit).cumcount())[num_t].mean()
        S = avg.dropna()
    else:
        S = df[num_t].dropna()
    F["tested_vars"] = num_t
    F["series"] = {c: _ds(S[c].values) for c in num_t[:6]}
    if aim == "regression":
        yv = y if y in num else num[0]
        xs = [c for c in num if c != yv]
        d = df[[yv] + xs].dropna()
        X = d[xs].values.astype(float)
        Z = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
        vif = []
        for j in range(Z.shape[1]):
            o = np.delete(Z, j, axis=1)
            if o.shape[1] == 0:
                vif.append(1.0); continue
            b, e, ss = _ols(Z[:, j], np.column_stack([np.ones(len(o)), o]))
            r2 = 1 - ss / float(((Z[:, j] - Z[:, j].mean()) ** 2).sum())
            vif.append(float(1 / max(1 - r2, 1e-6)))
        F["reg"] = dict(y=yv, n=int(len(d)), k=int(len(xs)), n_per_k=float(len(d) / max(len(xs), 1)), vif_max=float(max(vif) if vif else 1.0), time_series=bool(time))
        F["stationarity"] = {c: adf_kpss(df[c].dropna().values) for c in ([yv] if time else [])}
        F["ms"] = ms_evidence(df[yv].values) if time else None
        F["y"] = yv
        F["series"] = {yv: _ds(df[yv].dropna().values)}
        return F
    # stationarity (cross-unit average for a panel, as in the OES panel unit-root table) ----------------------------
    F["stationarity"] = {c: adf_kpss((df.groupby(time)[c].mean() if unit and time else df[c]).dropna().values) for c in num}
    Y = S.values.astype(float)
    if len(Y) < 30:
        F["too_short"] = True
        return F
    ycol = y if y in num_t else num_t[0]
    order = [ycol] + [c for c in num_t if c != ycol]
    Yo = S[order].values
    F["y"] = ycol
    if not unit and not F["mixed_freq"] and len(order) >= 1:
        F["volatility"] = volatility(Yo, order)
        F["instability"] = instability(Yo, order)
        F["threshold"] = threshold_test(Yo, order, nboot=99 if fast else 199)
    F["ms"] = ms_evidence(Yo[:, 0] if not unit else Yo[:, 0])
    F["n_series"] = len(num_t)
    return F


def analyse_json(csv_text, time="", unit="", y="", aim="var", vars=""):
    import json
    kw = dict(time=time or None, unit=unit or None, y=y or None, aim=aim or "var",
              vars=[v for v in vars.split("\x1f") if v] or None)
    try:
        return json.dumps(analyse(csv_text, **kw), default=float)
    except Exception as e:  # shown to the user
        return json.dumps(dict(error="The tests could not run on this file: %s" % e))
