"""ARIMA/SARIMA, GARCH family, quantile regression, cointegration tests and wavelet coherence.
Pure NumPy, SciPy and statsmodels (the `arch` package is not available in the browser). Same result format as the other models (core.py)."""
import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .core import (OESError, as_list, chart, distinct, f, int_option, float_option, need, result, stat, table, test_row, tests_table)
from .timeseries import _series_frame


def _nums(v, default, label):
    if v in (None, ""):
        return list(default)
    try:
        return [float(s) for s in (v.split(",") if isinstance(v, str) else v)]
    except ValueError:
        raise OESError(f"{label} must be a comma-separated list of numbers.")


# ---------------------------------------------------------------------------------------------------------------
# ARIMA / SARIMA
# ---------------------------------------------------------------------------------------------------------------
def arima(df, spec):
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    y = need(spec, "y", "series")
    d, labels = _series_frame(df, spec, [y])
    x = d[y].to_numpy(float); n = len(x)
    s = int_option(spec, "season", 0, 0, 52, "Seasonal period")
    maxp = int_option(spec, "max_p", 3, 0, 5, "Maximum p"); maxq = int_option(spec, "max_q", 3, 0, 5, "Maximum q")
    h = int_option(spec, "holdout", 12, 4, 60, "Hold-out periods")
    dsel = spec.get("d") or "auto"
    if n < max(60, 3 * h):
        raise OESError("The series is too short for this hold-out. Use at least 60 observations.")
    # unit-root tests
    rows = []
    def ur(z, lab):
        a = adfuller(z, autolag="AIC"); k = kpss(z, regression="c", nlags="auto")
        rows.append([lab, f(a[0], 3), f(a[1], 4), f(k[0], 3), f(k[1], 4)]); return a[1], k[1]
    pa0, pk0 = ur(x, "level"); pa1, pk1 = ur(np.diff(x), "first difference")
    dd = int(dsel) if dsel != "auto" else (0 if pa0 < .10 else 1)
    train, test = x[:-h], x[-h:]
    seas = (1, 1 if spec.get("sdiff") == "yes" else 0, 0, s) if s > 1 else (0, 0, 0, 0)
    trend = "c" if dd == 0 and seas[1] == 0 else "n"
    def fit(p, q, data):
        return SARIMAX(data, order=(p, dd, q), seasonal_order=seas, trend=trend).fit(disp=False, maxiter=200)
    grid = []
    for p in range(maxp + 1):
        for q in range(maxq + 1):
            try:
                r = fit(p, q, train); grid.append((p, q, r.aic, r.bic))
            except Exception:
                continue
    if not grid:
        raise OESError("No ARIMA model could be estimated.")
    g = pd.DataFrame(grid, columns=["p", "q", "AIC", "BIC"]).sort_values("BIC")
    p, q = int(g.iloc[0].p), int(g.iloc[0].q)
    m = fit(p, q, train)
    names = list(m.param_names)
    se = np.asarray(m.bse); pr = np.asarray(m.params)
    lb = acorr_ljungbox(m.resid[max(dd + p, 1):], lags=[6, 12, 24], return_df=True)
    res_ = m.resid[max(dd + p, 1):]
    tests = [test_row(f"Ljung-Box serial correlation ({L} lags)", lb.lb_stat.iloc[i], lb.lb_pvalue.iloc[i], "H0: no serial correlation") for i, L in enumerate((6, 12, 24))]
    ar = het_arch(res_, nlags=6); tests.append(test_row("ARCH LM (6 lags)", ar[0], ar[1], "H0: no ARCH effects"))
    jb = stats.jarque_bera(res_); tests.append(test_row("Jarque-Bera normality", jb[0], jb[1], "H0: normal residuals"))
    # rolling one-step forecasts with the selected order against a random-walk / mean benchmark
    fc, bench = [], []
    for i in range(h):
        data = x[:n - h + i]
        try:
            r = fit(p, q, data)
            fc.append(float(r.forecast(1)[0]))
        except Exception:
            fc.append(np.nan)
        bench.append(float(data[-1]))
    fc, bench = np.array(fc), np.array(bench)
    e1, e0 = test - fc, test - bench
    rmse1, rmse0 = float(np.sqrt(np.nanmean(e1 ** 2))), float(np.sqrt(np.mean(e0 ** 2)))
    dm = e1 ** 2 - e0 ** 2; ok = np.isfinite(dm)
    dmstat = dmp = np.nan
    if ok.sum() > 4:
        v = dm[ok]; bw = max(int(round(ok.sum() ** (1 / 3))), 1)
        u = v - v.mean(); lrv = u @ u / len(v) + 2 * sum((1 - l / (bw + 1)) * (u[l:] @ u[:-l]) / len(v) for l in range(1, bw + 1))
        dmstat = v.mean() / np.sqrt(max(lrv, 1e-12) / len(v)); dmp = 2 * stats.norm.sf(abs(dmstat))
    # multi-step forecast with intervals from the full-sample fit
    full = fit(p, q, x); fo = full.get_forecast(h); fm = np.asarray(fo.predicted_mean); ci = np.asarray(fo.conf_int(alpha=.05))
    tabs = [table("Unit-root tests", ["Series", "ADF stat", "ADF p", "KPSS stat", "KPSS p"], rows,
                  "ADF: H0 unit root. KPSS: H0 stationary. A disagreement usually signals a shifting mean; differencing is then judged by forecasts."),
            table("Order selection (training sample)", ["p", "q", "AIC", "BIC"], [[int(a), int(b), f(c, 2), f(e, 2)] for a, b, c, e in g.head(6).values],
                  f"Best six by BIC out of {len(g)} models, d = {dd}" + (f", seasonal period {s}" if s > 1 else "") + "."),
            table(f"ARIMA({p},{dd},{q}) estimates", ["Parameter", "Coef.", "Std. err.", "z", "p-value"],
                  [[nm, f(b), f(a), f(b / a, 3), f(2 * stats.norm.sf(abs(b / a)), 4)] for nm, b, a in zip(names, pr, se)]),
            tests_table(tests, "Residual diagnostics"),
            table("Hold-out forecast accuracy (rolling one-step)", ["Forecast", "RMSE"], [[f"ARIMA({p},{dd},{q})", f(rmse1, 4)], ["Random walk (last value)", f(rmse0, 4)]],
                  f"Diebold-Mariano test of equal squared errors: statistic {f(dmstat, 3)}, p = {f(dmp, 4)} ({int(ok.sum())} forecasts).")]
    xs = list(range(n)); lab = labels
    ch = [chart(f"{y}: observed and {h}-step forecast", lab + [f"+{i + 1}" for i in range(h)], [("Observed", list(x) + [None] * h), ("Forecast", [None] * (n - 1) + [x[-1]] + list(fm))], "", y,
                band=([None] * n + list(ci[:, 0]), [None] * n + list(ci[:, 1])))]
    summary = [stat("Observations", n), stat("Selected p", p), stat("Selected d", dd), stat("Selected q", q), stat("Hold-out RMSE ratio vs random walk", rmse1 / rmse0, 3)]
    notes = ["Order chosen by BIC on the training sample (all but the last hold-out periods); the hold-out is forecast one step ahead with re-estimation at each origin. Constant included when no differencing is used; the reported intercept is the constant of the equation, so the process mean is intercept / (1 - sum of AR coefficients)."]
    return result("arima", f"ARIMA / SARIMA: {y}", n, summary, tabs, ch, notes)


# ---------------------------------------------------------------------------------------------------------------
# GARCH family with normal, Student-t and Hansen skewed-t errors, estimated by maximum likelihood
# ---------------------------------------------------------------------------------------------------------------
def _skewt_logpdf(z, nu, lam):
    c = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - .5 * np.log(np.pi * (nu - 2))
    a = 4 * lam * np.exp(c) * (nu - 2) / (nu - 1); b = np.sqrt(1 + 3 * lam ** 2 - a ** 2)
    w = b * z + a
    return np.log(b) + c - (nu + 1) / 2 * np.log1p((w / (1 + lam * np.sign(w))) ** 2 / (nu - 2))


def _logdens(z, dist, sh):
    if dist == "normal":
        return -.5 * np.log(2 * np.pi) - .5 * z ** 2
    nu = sh[0]
    if dist == "t":
        return (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - .5 * np.log(np.pi * (nu - 2)) - (nu + 1) / 2 * np.log1p(z ** 2 / (nu - 2)))
    return _skewt_logpdf(z, nu, sh[1])


def _recursion(e, vol, par, var0):
    n = len(e); s2 = np.empty(n)
    if vol == "const":
        s2[:] = par[0]; return s2
    if vol == "egarch":
        om, al, ga, be = par; lv = np.log(var0); s2[0] = var0
        for t in range(1, n):
            z = e[t - 1] / np.sqrt(s2[t - 1])
            lv = om + al * abs(z) + ga * z + be * lv
            lv = min(max(lv, -50), 50); s2[t] = np.exp(lv)
        return s2
    om, al, ga, be = par; s2[0] = var0
    for t in range(1, n):
        s2[t] = om + (al + (ga if e[t - 1] < 0 else 0.0) * (vol == "gjr")) * e[t - 1] ** 2 + be * s2[t - 1]
    return s2


def _unpack(theta, vol, dist):
    mu = theta[0]
    k = {"const": 1, "garch": 4, "gjr": 4, "egarch": 4}[vol]
    vp = theta[1:1 + k]; sh = theta[1 + k:]
    if dist != "normal":
        sh = np.array([2.05 + np.exp(sh[0])] + ([np.tanh(sh[1])] if dist == "skewt" else []))
    return mu, vp, sh


def _negll(theta, r, vol, dist, var0):
    mu, vp, sh = _unpack(theta, vol, dist)
    if vol in ("garch", "gjr"):
        if vp[0] <= 0 or vp[1] < 0 or vp[3] < 0 or vp[1] + (vp[2] / 2 if vol == "gjr" else 0) + vp[3] >= 0.9999 or (vol == "garch" and vp[2] != 0):
            return 1e10
        if vol == "gjr" and vp[1] + vp[2] < 0:
            return 1e10
    if vol == "egarch" and abs(vp[3]) >= .9999:
        return 1e10
    e = r - mu
    s2 = _recursion(e, vol, vp, var0)
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return 1e10
    z = e / np.sqrt(s2)
    ll = _logdens(z, dist, sh) - .5 * np.log(s2)
    v = -ll.sum()
    return v if np.isfinite(v) else 1e10


def _hessian(fn, x, eps=1e-4):
    k = len(x); H = np.zeros((k, k)); h = eps * np.maximum(np.abs(x), 1e-2)
    f0 = fn(x)
    for i in range(k):
        for j in range(i, k):
            xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
            xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
            xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
            xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
            H[i, j] = H[j, i] = (fn(xpp) - fn(xpm) - fn(xmp) + fn(xmm)) / (4 * h[i] * h[j])
    return H


def _fit_garch(r, vol, dist):
    n = len(r); var0 = float(r.var()); mu0 = float(r.mean())
    if vol == "const":
        v0 = [mu0, var0]
    elif vol == "egarch":
        v0 = [mu0, (1 - .9) * np.log(var0), .1, -.05, .9]
    elif vol == "gjr":
        v0 = [mu0, var0 * .05, .03, .08, .88]
    else:
        v0 = [mu0, var0 * .05, .08, 0.0, .88]
    s0 = [] if dist == "normal" else ([np.log(6.0)] if dist == "t" else [np.log(6.0), 0.0])
    x0 = np.array(v0 + s0)
    fixed = None
    if vol == "garch":
        # the leverage slot is held at zero in the plain GARCH
        free = [i for i in range(len(x0)) if i != 3]
    else:
        free = list(range(len(x0)))
    def obj(z):
        th = x0.copy(); th[free] = z; return _negll(th, r, vol, dist, var0)
    best = None
    starts = [x0[free]]
    if vol != "const":
        alt = x0.copy(); alt[2] *= 2; alt[-len(s0) - 1 if s0 else -1] = .8 if vol != "egarch" else .7; starts.append(alt[free])
    for st in starts:
        try:
            o = optimize.minimize(obj, st, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8, adaptive=True))
            o = optimize.minimize(obj, o.x, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-7, fatol=1e-9, adaptive=True))
        except Exception:
            continue
        if best is None or o.fun < best.fun:
            best = o
    if best is None or best.fun >= 1e9:
        raise OESError("The likelihood could not be maximised for this specification.")
    th = x0.copy(); th[free] = best.x
    try:
        H = _hessian(obj, best.x); V = np.linalg.inv(H); se = np.sqrt(np.clip(np.diag(V), 0, None)); se[~np.isfinite(se)] = np.nan
    except Exception:
        se = np.full(len(free), np.nan)
    return dict(theta=th, free=free, se=se, ll=-best.fun, k=len(free), var0=var0, vol=vol, dist=dist)


_PNAMES = {"const": ["mean", "variance"], "garch": ["mean", "omega", "alpha", "gamma (leverage)", "beta"], "gjr": ["mean", "omega", "alpha", "gamma (leverage)", "beta"],
           "egarch": ["mean", "omega", "alpha (size)", "gamma (sign)", "beta"]}


def _garch_view(fit, r):
    th = fit["theta"]; vol, dist = fit["vol"], fit["dist"]
    mu, vp, sh = _unpack(th, vol, dist)
    e = r - mu; s2 = _recursion(e, vol, vp, fit["var0"])
    return mu, vp, sh, e, s2


def garch(df, spec):
    from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
    y = need(spec, "y", "return series")
    d, labels = _series_frame(df, spec, [y])
    r = d[y].to_numpy(float); n = len(r)
    if n < 150:
        raise OESError("GARCH models need at least 150 observations.")
    scale = float(r.std()); rs = r / scale if scale > 0 else r      # fit on a standardised scale for stable optimisation
    # tests for ARCH effects
    e0 = rs - rs.mean()
    arch_rows = [test_row(f"ARCH LM ({L} lags)", *het_arch(e0, nlags=L)[:2], "H0: no ARCH effects") for L in (1, 3, 6, 12)]
    lbq = acorr_ljungbox(e0 ** 2, lags=[12], return_df=True)
    arch_rows.append(test_row("Ljung-Box on squared deviations (12)", lbq.lb_stat.iloc[0], lbq.lb_pvalue.iloc[0], "H0: no clustering"))
    specs = [("Constant variance, t", "const", "t"), ("GARCH, normal", "garch", "normal"), ("GARCH, t", "garch", "t"), ("GJR, t", "gjr", "t"),
             ("EGARCH, t", "egarch", "t"), ("GJR, skewed t", "gjr", "skewt"), ("EGARCH, skewed t", "egarch", "skewt")]
    fits = {}
    for name, v, dist in specs:
        try:
            fits[name] = _fit_garch(rs, v, dist)
        except OESError:
            continue
    if "GARCH, t" not in fits:
        raise OESError("The GARCH(1,1) model with t errors could not be estimated on these data.")
    rows = []
    for name, ft in fits.items():
        k = ft["k"]; ll = ft["ll"] - n * np.log(scale)                  # log-likelihood on the original scale
        rows.append([name, k, f(ll, 2), f(-2 * ll + 2 * k, 2), f(-2 * ll + k * np.log(n), 2)])
    tab = pd.DataFrame(rows, columns=["Model", "k", "ll", "AIC", "BIC"])
    pick = spec.get("model") or "best"
    bname = tab.loc[tab.BIC.astype(float).idxmin(), "Model"] if pick in ("best", "") else pick
    if bname not in fits:
        bname = tab.loc[tab.BIC.astype(float).idxmin(), "Model"]
    ft = fits[bname]; mu, vp, sh, e, s2 = _garch_view(ft, rs)
    z = e / np.sqrt(s2)
    # parameter table in original units
    names = _PNAMES[ft["vol"]] + (["df (nu)"] if ft["dist"] != "normal" else []) + (["skew (lambda)"] if ft["dist"] == "skewt" else [])
    val = np.concatenate([[mu * scale], vp, sh]).astype(float)
    if ft["vol"] in ("garch", "gjr"):
        val[1] *= scale ** 2
        if ft["vol"] == "gjr":
            pass
    elif ft["vol"] == "egarch":
        val[1] += (1 - vp[3]) * 2 * np.log(scale)
    elif ft["vol"] == "const":
        val[1] *= scale ** 2
    ses = np.full(len(val), np.nan)
    # standard errors are on the internal parametrisation: report them only for mean and the volatility parameters (delta method skipped for shape)
    idx = ft["free"]; se_int = dict(zip(idx, ft["se"]))
    raw_to_pos = {0: 0}
    kv = {"const": 1, "garch": 4, "gjr": 4, "egarch": 4}[ft["vol"]]
    for pos in range(1 + kv):
        if pos in se_int:
            ses[pos] = se_int[pos] * (scale if pos == 0 else (scale ** 2 if (pos == 1 and ft["vol"] != "egarch") else 1))
    prow = []
    for i, nm in enumerate(names):
        if ft["vol"] == "garch" and nm == "gamma (leverage)":
            continue
        b = val[i]; s = ses[i] if i < len(ses) else np.nan
        prow.append([nm, f(b, 5), f(s, 5), f(b / s, 2) if np.isfinite(s) and s > 0 else None])
    pers = None
    if ft["vol"] == "garch":
        pers = vp[1] + vp[3]
    elif ft["vol"] == "gjr":
        pers = vp[1] + vp[2] / 2 + vp[3]
    elif ft["vol"] == "egarch":
        pers = vp[3]
    # standardised residual diagnostics
    zd = [test_row("Ljung-Box on z (12 lags)", *[acorr_ljungbox(z, lags=[12], return_df=True).iloc[0, i] for i in (0, 1)], "H0: no serial correlation"),
          test_row("Ljung-Box on z^2 (12 lags)", *[acorr_ljungbox(z ** 2, lags=[12], return_df=True).iloc[0, i] for i in (0, 1)], "H0: variance fully modelled")]
    neg = (z[:-1] < 0).astype(float); X = np.column_stack([np.ones(n - 1), neg, neg * z[:-1], (1 - neg) * z[:-1]])
    yy = z[1:] ** 2; bb = np.linalg.lstsq(X, yy, rcond=None)[0]; resid = yy - X @ bb
    s2e = resid @ resid / (n - 1 - 4); Vb = s2e * np.linalg.inv(X.T @ X); R = np.zeros((3, 4)); R[0, 1] = R[1, 2] = R[2, 3] = 1
    wald = float((R @ bb) @ np.linalg.solve(R @ Vb @ R.T, R @ bb))
    zd.append(test_row("Engle-Ng sign-bias (joint)", wald, stats.chi2.sf(wald, 3), "H0: no leverage left in z^2"))
    # value at risk, in-sample 1-step with the fitted model
    qfun = (lambda p: stats.norm.ppf(p)) if ft["dist"] == "normal" else (lambda p: float(np.quantile(z, p)))
    var_rows = []
    for a in (.05, .01):
        q = qfun(a) if ft["dist"] == "normal" else float(np.quantile(z, a))
        thr = mu + q * np.sqrt(s2); viol = (rs < thr).sum(); p0 = a
        phat = viol / n
        lr = -2 * ((n - viol) * np.log(1 - p0) + viol * np.log(p0)) + 2 * ((n - viol) * np.log(1 - phat) + viol * np.log(phat)) if 0 < viol < n else np.nan
        var_rows.append([f"{int(a * 100)}% VaR", int(viol), f(phat, 4), f(lr, 3), f(stats.chi2.sf(lr, 1), 4) if np.isfinite(lr) else None])
    vol = np.sqrt(s2) * scale
    tabs = [tests_table(arch_rows, "Tests for time-varying volatility", "Constant-variance residuals from the sample mean."),
            table("Specification comparison (original scale)", ["Model", "Parameters", "Log-likelihood", "AIC", "BIC"], rows,
                  "Maximum likelihood on the standardised series; log-likelihoods are converted back to the original units."),
            table(f"Estimates: {bname}", ["Parameter", "Estimate", "Std. err.", "z"], prow,
                  ("Persistence: " + str(f(pers, 4)) + ". " if pers is not None else "") + "Standard errors are numerical and refer to the internal parametrisation for shape parameters (not shown)."),
            tests_table(zd, "Standardised-residual diagnostics"),
            table("In-sample value at risk and Kupiec test", ["Level", "Violations", "Share", "LR (Kupiec)", "p-value"], var_rows,
                  "Violation share should match the nominal level; the quantile uses the empirical quantile of standardised residuals for t-type errors.")]
    ch = [chart(f"Conditional volatility ({bname})", labels, [("Conditional s.d.", vol), ("|deviation from mean|", np.abs(r - mu * scale))], "", "Percent / units of the series")]
    summary = [stat("Observations", n), stat("Selected model (BIC)", bname), stat("Persistence", pers if pers is not None else float("nan"), 3), stat("Log-likelihood", float(tab.set_index("Model").loc[bname, "ll"]), 1)]
    notes = ["Models are estimated by Nelder-Mead maximum likelihood in NumPy (no external GARCH package), on the series divided by its standard deviation. The OES version omits the multivariate DCC step shown in the guide."]
    return result("garch", f"GARCH family: {y}", n, summary, tabs, ch, notes)


# ---------------------------------------------------------------------------------------------------------------
# Quantile regression with a block bootstrap
# ---------------------------------------------------------------------------------------------------------------
def quantile(df, spec):
    import statsmodels.api as sm
    from statsmodels.regression.quantile_regression import QuantReg
    y = need(spec, "y", "dependent variable"); xs = as_list(need(spec, "x", "explanatory variables")); distinct([y], xs)
    taus = _nums(spec.get("taus"), [.05, .25, .5, .75, .95], "Quantiles")
    if not taus or any(not 0 < t < 1 for t in taus):
        raise OESError("Quantiles must lie strictly between 0 and 1.")
    taus = sorted(set(taus))
    d, labels = _series_frame(df, spec, [y] + xs)
    yv = d[y].to_numpy(float); n = len(d)
    X = np.column_stack([np.ones(n), d[xs].to_numpy(float)]); names = ["const"] + xs
    L = int_option(spec, "block", 1, 1, 40, "Block length"); R = int_option(spec, "boot", 200, 50, 1000, "Bootstrap replications")
    if n < 8 * X.shape[1]:
        raise OESError("Too few observations for this number of regressors.")
    est = {t: QuantReg(yv, X).fit(q=t, max_iter=5000).params for t in taus}
    ols = sm.OLS(yv, X).fit()
    rng = np.random.default_rng(int(spec.get("seed") or 0)); nb = int(np.ceil(n / L)); draws = {t: [] for t in taus}
    for _ in range(R):
        st = rng.integers(0, n - L + 1, nb); idx = np.concatenate([np.arange(s, s + L) for s in st])[:n]
        for t in taus:
            try:
                draws[t].append(QuantReg(yv[idx], X[idx]).fit(q=t, max_iter=5000).params)
            except Exception:
                draws[t].append(np.full(X.shape[1], np.nan))
    draws = {t: np.array(v) for t, v in draws.items()}
    coef_rows = []
    for k, nm in enumerate(names):
        for t in taus:
            b = est[t][k]; sd = np.nanstd(draws[t][:, k], ddof=1); lo, hi = np.nanpercentile(draws[t][:, k], [5, 95])
            coef_rows.append([nm, t, f(b, 4), f(sd, 4), f(b / sd, 2) if sd > 0 else None, f(lo, 4), f(hi, 4), f(ols.params[k], 4)])
    tabs = [table("Quantile regression coefficients", ["Variable", "Quantile", "Coef.", "Bootstrap s.e.", "t", "90% low", "90% high", "OLS coef."], coef_rows,
                  f"{R} bootstrap replications" + (f" (moving blocks of length {L})" if L > 1 else " (pairs bootstrap)") + "; 90% intervals are bootstrap percentiles.")]
    k1 = 1
    pairs = [(taus[0], taus[len(taus) // 2]), (taus[-1], taus[len(taus) // 2]), (taus[0], taus[-1])]
    drows = []
    for kk, nm in enumerate(names[1:], start=1):
        for a, b in pairs:
            if a == b: continue
            e_ = est[a][kk] - est[b][kk]; dd = draws[a][:, kk] - draws[b][:, kk]
            cen = dd - np.nanmean(dd); p = np.nanmean(np.abs(cen) >= abs(e_)); lo, hi = np.nanpercentile(dd, [2.5, 97.5])
            drows.append([nm, f"q{a:g} - q{b:g}", f(e_, 4), f(lo, 4), f(hi, 4), f(p, 4)])
    tabs.append(table("Does the effect differ across quantiles?", ["Variable", "Difference", "Estimate", "95% low", "95% high", "p-value (bootstrap)"], drows,
                      "Bootstrap distribution recentred at zero for the p-value."))
    fit_q = np.column_stack([X @ est[t] for t in taus]); cross = (np.diff(fit_q, axis=1) < 0).any(axis=1)
    cov = [[t, f((yv < fit_q[:, i]).mean(), 3)] for i, t in enumerate(taus)]
    tabs.append(table("In-sample coverage of the fitted quantiles", ["Quantile", "Share of outcomes below the fitted quantile"], cov, f"Quantile curves cross in {int(cross.sum())} of {n} observations."))
    ch = []
    for kk, nm in enumerate(names[1:], start=1):
        lo, hi = zip(*[np.nanpercentile(draws[t][:, kk], [5, 95]) for t in taus])
        ch.append(chart(f"Coefficient on {nm} by quantile", taus, [("Quantile regression", [est[t][kk] for t in taus]), ("OLS", [ols.params[kk]] * len(taus))], "Quantile", "Coefficient", band=(lo, hi)))
    lo5, hi5 = taus[0], taus[-1]
    fq = np.sort(fit_q, axis=1)
    ch.append(chart("Observed outcome and fitted quantiles (rearranged)", labels, [("Outcome", yv), (f"q{lo5:g}", fq[:, 0]), (f"q{taus[len(taus) // 2]:g}", fq[:, len(taus) // 2]), (f"q{hi5:g}", fq[:, -1])]))
    summary = [stat("Observations", n), stat("Quantiles", len(taus)), stat("Bootstrap replications", R), stat("Crossing observations", int(cross.sum()))]
    return result("quantile", f"Quantile regression: {y}", n, summary, tabs, ch, ["Fitted quantiles can cross; the final chart rearranges them by sorting (Chernozhukov, Fernandez-Val and Galichon)."])


# ---------------------------------------------------------------------------------------------------------------
# Cointegration: unit roots, Engle-Granger, Johansen, Gregory-Hansen, DOLS
# ---------------------------------------------------------------------------------------------------------------
def _adf_stat(u, maxlag=8):
    import statsmodels.api as sm
    best = None; du = np.diff(u); n = len(du)
    for k in range(0, maxlag + 1):
        Y = du[maxlag:]
        cols = [u[maxlag:-1]] + [du[maxlag - j:n - j] for j in range(1, k + 1)]
        res = sm.OLS(Y, np.column_stack(cols)).fit()
        ic = len(Y) * np.log(res.ssr / len(Y)) + 2 * (k + 1)
        if best is None or ic < best[0]:
            best = (ic, res.tvalues[0], k)
    return best[1], best[2]


def _gh_design(x, tb, model):
    T = x.shape[0]; tt = np.arange(T); D = (tt >= tb).astype(float)
    cols = [np.ones(T), D]
    if model == "C/T":
        cols.append(tt.astype(float))
    cols += [x[:, j] for j in range(x.shape[1])]
    if model == "C/S":
        cols += [x[:, j] * D for j in range(x.shape[1])]
    return np.column_stack(cols)


def _adf0(u):
    du = np.diff(u); xx = u[:-1]; b = (xx @ du) / (xx @ xx); e = du - b * xx
    return b / np.sqrt(e @ e / (len(du) - 1) / (xx @ xx))


def cointegration(df, spec):
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, kpss, coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    y = need(spec, "y", "dependent variable"); xs = as_list(need(spec, "x", "regressors")); distinct([y], xs)
    d, labels = _series_frame(df, spec, [y] + xs)
    n = len(d)
    if n < 60:
        raise OESError("Cointegration tests need at least 60 observations.")
    yv = d[y].to_numpy(float); xm = d[xs].to_numpy(float)
    urows = []
    for nm in [y] + xs:
        s = d[nm].to_numpy(float)
        for lab, z in (("level", s), ("first difference", np.diff(s))):
            a = adfuller(z, regression="c", autolag="AIC"); k = kpss(z, regression="c", nlags="auto")
            urows.append([nm, lab, f(a[0], 3), f(a[1], 4), f(k[0], 3), f(k[1], 4)])
    tabs = [table("Unit-root tests", ["Series", "Form", "ADF stat", "ADF p", "KPSS stat", "KPSS p"], urows, "The tests are applied to levels and first differences; cointegration needs I(1) series.")]
    eg_rows = []
    jobs = [(y, yv, xm, "the regressors")]
    if len(xs) == 1:
        jobs.append((xs[0], xm[:, 0], yv[:, None], y))
    for dep, dv, regs, other in jobs:
        r = coint(dv, regs, trend="c", maxlag=8, autolag="aic")
        ols = sm.OLS(dv, sm.add_constant(regs)).fit()
        eg_rows.append([f"{dep} on {other}", f(r[0], 3), f(r[1], 4), f(r[2][1], 3), "; ".join(f"{b:.3f}" for b in ols.params[1:])])
    tabs.append(table("Engle-Granger test", ["Regression", "Statistic", "p-value", "5% critical value", "Long-run slope(s)"], eg_rows, "H0: no cointegration. The test is not symmetric in the choice of dependent variable."))
    if len(xs) >= 1:
        jrows = []
        data = d[[y] + xs].to_numpy(float)
        for k in (1, 2, 3, 4):
            j = coint_johansen(data, det_order=0, k_ar_diff=k)
            for r in range(min(len(xs) + 1, 3)):
                jrows.append([k, f"r <= {r}" if r else "r = 0", f(j.lr1[r], 2), f(j.cvt[r, 1], 2), f(j.lr2[r], 2), f(j.cvm[r, 1], 2)])
        tabs.append(table("Johansen test (constant in the cointegrating space)", ["VAR lag differences", "H0 rank", "Trace", "Trace 5% cv", "Max-eigenvalue", "Max-eig 5% cv"], jrows,
                          "Reject when the statistic exceeds the critical value."))
    # Gregory-Hansen with simulated critical values
    trim = .15; res = {}
    for m in ("C", "C/T", "C/S"):
        out = []
        for tb in range(int(trim * n), int((1 - trim) * n)):
            u = sm.OLS(yv, _gh_design(xm, tb, m)).fit().resid
            st, k = _adf_stat(u, 8); out.append((st, tb, k))
        res[m] = (min(out), out)
    Rsim = int_option(spec, "sims", 100, 20, 500, "Simulation replications"); rng = np.random.default_rng(int(spec.get("seed") or 0))
    sim = {m: [] for m in res}
    for _ in range(Rsim):
        ys = np.cumsum(rng.standard_normal(n)); xsim = np.cumsum(rng.standard_normal((n, len(xs))), axis=0)
        for m in sim:
            best = 1e9
            for tb in range(int(trim * n), int((1 - trim) * n), 2):
                Dm = _gh_design(xsim, tb, m); u = ys - Dm @ np.linalg.lstsq(Dm, ys, rcond=None)[0]
                best = min(best, _adf0(u))
            sim[m].append(best)
    cv = {m: np.percentile(v, [1, 5, 10]) for m, v in sim.items()}
    grows = []
    for m in res:
        (st, tb, k), _ = res[m]
        grows.append([m, f(st, 3), labels[tb], int(k), f(cv[m][0], 2), f(cv[m][1], 2), f(cv[m][2], 2), "yes" if st < cv[m][1] else "no"])
    tabs.append(table("Gregory-Hansen test with one structural break", ["Model", "Statistic", "Break date", "ADF lags", "1% cv", "5% cv", "10% cv", "Rejects at 5%"], grows,
                      f"Critical values simulated for independent random walks without break ({Rsim} replications, T = {n}, break grid every second date). Models: C = level shift, C/T = level shift with trend, C/S = regime shift (intercept and slopes)."))
    # DOLS and slopes before/after the C/S break
    tb = res["C/S"][0][1]
    Dm = _gh_design(xm, tb, "C/S"); ols = sm.OLS(yv, Dm).fit(cov_type="HAC", cov_kwds={"maxlags": 8})
    nx = len(xs); b = ols.params; V = ols.cov_params(); srows = []
    for j in range(nx):
        i0 = 2 + j; i1 = 2 + nx + j
        w = np.zeros(len(b)); w[[i0, i1]] = 1; sea = float(np.sqrt(w @ V @ w))
        srows.append([xs[j], f(b[i0], 3), f(np.sqrt(V[i0, i0]), 3), f(b[i0] + b[i1], 3), f(sea, 3)])
    tabs.append(table(f"Long-run slopes before and after the break ({labels[tb]})", ["Regressor", "Before", "HAC s.e.", "After", "HAC s.e."], srows))
    def dols(yy, xx, lags=2):
        T = len(yy); dx = np.diff(xx, axis=0); dx = np.vstack([np.zeros((1, xx.shape[1])), dx])
        idx = np.arange(lags, T - lags)
        cols = [np.ones(len(idx))] + [xx[idx, j] for j in range(xx.shape[1])]
        for l in range(-lags, lags + 1):
            for j in range(xx.shape[1]):
                cols.append(dx[idx - l, j])
        r = sm.OLS(yy[idx], np.column_stack(cols)).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        return r.params[1:1 + xx.shape[1]], r.bse[1:1 + xx.shape[1]], len(idx)
    drows = []
    for lab, sl in (("whole sample", slice(None)), ("before break", slice(0, tb)), ("after break", slice(tb, None))):
        if len(yv[sl]) < 25:
            continue
        pb, ps, nn = dols(yv[sl], xm[sl])
        for j in range(nx):
            drows.append([lab, xs[j], f(pb[j], 3), f(ps[j], 3), nn])
    tabs.append(table("Dynamic OLS (2 leads and 2 lags)", ["Sample", "Regressor", "Slope", "s.e. (HAC)", "n"], drows, "DOLS corrects for endogeneity of the regressors in the long-run relation."))
    ch = []
    for m in ("C", "C/T", "C/S"):
        path = res[m][1]
        ch.append(chart(f"Gregory-Hansen: ADF statistic by break date ({m})", [labels[b] for _, b, _ in path], [("ADF statistic", [s for s, _, _ in path]), ("5% critical value", [cv[m][1]] * len(path))], "Break date", "ADF statistic of the residual"))
    summary = [stat("Observations", n), stat("Engle-Granger p", float(eg_rows[0][2]), 3), stat("Break date (C/S)", labels[tb]), stat("Rejects at 5% (C/S)", grows[2][-1])]
    return result("cointegration", f"Cointegration: {y} and {', '.join(xs)}", n, summary, tabs, ch,
                  ["Phillips-Ouliaris tests, available in the guide through the `arch` package, are not part of the OES version."])


# ---------------------------------------------------------------------------------------------------------------
# Wavelet coherence
# ---------------------------------------------------------------------------------------------------------------
def _morlet_cwt(x, scales, w0=6.0):
    m = len(x); N = 1 << int(np.ceil(np.log2(2 * m)))
    xf = np.fft.fft(x - x.mean(), N); om = 2 * np.pi * np.fft.fftfreq(N)
    W = np.empty((len(scales), m), complex)
    for k, s in enumerate(scales):
        psi = np.pi ** -.25 * np.exp(-.5 * (s * om - w0) ** 2) * (om > 0) * np.sqrt(2 * np.pi * s)
        W[k] = np.fft.ifft(xf * psi)[:m]
    return W


def _smooth(A, scales):
    from scipy.ndimage import uniform_filter1d
    out = np.empty_like(A)
    for k, s in enumerate(scales):
        w = max(int(round(2 * s)), 1)
        out[k] = uniform_filter1d(A[k].real, w, mode="nearest") + 1j * uniform_filter1d(A[k].imag, w, mode="nearest")
    return uniform_filter1d(out.real, 3, axis=0, mode="nearest") + 1j * uniform_filter1d(out.imag, 3, axis=0, mode="nearest")


def _wtc(x, y, scales):
    Wx, Wy = _morlet_cwt(x, scales), _morlet_cwt(y, scales)
    Sxy = _smooth(Wx * np.conj(Wy) / scales[:, None], scales)
    Sx = _smooth(np.abs(Wx) ** 2 / scales[:, None], scales).real
    Sy = _smooth(np.abs(Wy) ** 2 / scales[:, None], scales).real
    return np.abs(Sxy) ** 2 / (Sx * Sy), np.angle(Sxy)


def wavelet(df, spec):
    xn = need(spec, "x", "first series"); yn = need(spec, "y", "second series"); distinct([xn], [yn])
    d, labels = _series_frame(df, spec, [xn, yn])
    n = len(d)
    if n < 100:
        raise OESError("Wavelet coherence needs at least 100 observations.")
    x = d[xn].to_numpy(float); y = d[yn].to_numpy(float)
    pmin = float_option(spec, "pmin", 16, "Shortest period"); pmax = float_option(spec, "pmax", min(512, n // 2), "Longest period")
    if pmin < 4 or pmax <= pmin or pmax > n:
        raise OESError("Choose periods with 4 <= shortest < longest <= number of observations.")
    nj = int(np.floor(8 * np.log2(pmax / pmin))) + 1
    period = pmin * 2 ** (np.arange(nj) / 8)
    scales = period / (4 * np.pi / (6 + np.sqrt(2 + 36)))
    R, ph = _wtc(x, y, scales)
    coi = np.minimum(np.arange(n), n - 1 - np.arange(n)); mask = coi[None, :] >= (np.sqrt(2) * scales)[:, None]
    if mask.sum() == 0:
        raise OESError("No part of the time-period plane lies inside the cone of influence; reduce the longest period.")
    rng = np.random.default_rng(int(spec.get("seed") or 0)); Rs = int_option(spec, "sims", 100, 20, 500, "Simulation replications")
    def ar1(v):
        v = v - v.mean(); a = (v[1:] @ v[:-1]) / (v[:-1] @ v[:-1]); return a, np.std(v[1:] - a * v[:-1])
    (a1, s1), (a2, s2) = ar1(x), ar1(y)
    def sim(a, s):
        e = rng.standard_normal(n + 100) * s; u = np.zeros(n + 100)
        for j in range(1, n + 100): u[j] = a * u[j - 1] + e[j]
        return u[100:]
    sims = np.array([_wtc(sim(a1, s1), sim(a2, s2), scales)[0] for _ in range(Rs)])
    c95 = np.percentile(sims, 95, axis=0); sig = (R > c95) & mask
    fa = ((sims > c95[None]) & mask[None]).sum((1, 2)) / mask.sum()
    # by period
    pick = [p for p in (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512) if pmin <= p <= pmax]
    prow = []
    for p_ in pick:
        k = int(np.argmin(abs(period - p_))); m_ = mask[k]
        if m_.sum() == 0: continue
        prow.append([p_, f(R[k, m_].mean(), 3), f(c95[k, m_].mean(), 3), f((R[k, m_] > c95[k, m_]).mean(), 3), f(np.angle(np.exp(1j * ph[k, m_]).mean()), 2)])
    band = (period >= min(32, pmax)) & (period <= min(128, pmax))
    seg = np.array_split(np.arange(n), 6); trow = []; tmean = []
    for ix in seg:
        vals = R[np.ix_(band, ix)]; ok = mask[np.ix_(band, ix)]
        if ok.sum() == 0:
            trow.append([f"{labels[ix[0]]} to {labels[ix[-1]]}", None, None]); tmean.append(np.nan); continue
        trow.append([f"{labels[ix[0]]} to {labels[ix[-1]]}", f(vals[ok].mean(), 3), f(sig[np.ix_(band, ix)].sum() / ok.sum(), 3)]); tmean.append(vals[ok].mean())
    ss = sig[band]; pp = ph[band][ss]
    inph = float((np.abs(pp) < np.pi / 2).mean()) if len(pp) else float("nan")
    cm = float(np.angle(np.exp(1j * pp).mean())) if len(pp) else float("nan")
    tabs = [table("Coherence by period", ["Period", "Mean coherence", "Mean 95% critical value", "Share significant", "Mean phase (rad)"], prow,
                  "Averaged over dates inside the cone of influence. Phase near 0 = in phase, near +/-pi = opposite directions; a positive phase means the second series lags the first."),
            table("Coherence over time (periods 32 to 128, or the available range)", ["Segment", "Mean coherence", "Share significant"], trow),
            table("Overall", ["Item", "Value"], [["Share of the plane (inside the cone) above the 95% critical value", f(sig.sum() / mask.sum(), 4)], ["Same share in simulated null series, mean", f(fa.mean(), 4)],
                                                 ["Same share in simulated null series, 95th percentile", f(np.percentile(fa, 95), 4)], ["Significant points in the band: share in phase (|phase| < 90 degrees)", f(inph, 3)],
                                                 ["Circular mean phase in the band (radians)", f(cm, 3)], ["AR(1) coefficients (first, second series)", f"{a1:.3f}, {a2:.3f}"]],
                  "The null is two independent AR(1) processes fitted to the series. The share above the critical value should be compared with the simulated null share, because the 5% pointwise level overstates joint significance.")]
    ch = [chart("Mean squared coherence by period", [f(p, 1) for p in period], [("Coherence", [R[k, mask[k]].mean() if mask[k].any() else np.nan for k in range(nj)]),
                                                                                  ("95% critical value", [c95[k, mask[k]].mean() if mask[k].any() else np.nan for k in range(nj)])], "Period", "Squared coherence"),
          chart("Mean coherence over time (band 32 to 128 periods)", [f"{labels[ix[0]]}" for ix in seg], [("Coherence", tmean)], "", "Squared coherence", kind="bar")]
    summary = [stat("Observations", n), stat("Periods", f"{period[0]:.0f} to {period[-1]:.0f}"), stat("Share significant", float(sig.sum() / mask.sum()), 3), stat("Null share (mean)", float(fa.mean()), 3)]
    return result("wavelet", f"Wavelet coherence: {xn} and {yn}", n, summary, tabs, ch,
                  ["Morlet wavelet (w0 = 6), smoothing over time and three neighbouring scales (Torrence and Webster 1999); significance by Monte Carlo from AR(1) nulls. The coherence heat map in the guide is shown here as averages by period and by sub-period."])


# ---------------------------------------------------------------------------------------------------------------
# Unit-root and stationarity tests: ADF, DF-GLS, Phillips-Perron, KPSS, Zivot-Andrews
# ---------------------------------------------------------------------------------------------------------------
def _dfgls_stat(y, trend, k):
    """Elliott-Rothenberg-Stock DF-GLS t statistic with k lagged differences (GLS-detrended series)."""
    T = len(y)
    z = np.ones((T, 1)) if trend == "c" else np.column_stack([np.ones(T), np.arange(1, T + 1)])
    ab = 1 - (7.0 if trend == "c" else 13.5) / T
    ya = np.r_[y[0], y[1:] - ab * y[:-1]]; za = np.vstack([z[:1], z[1:] - ab * z[:-1]])
    beta = np.linalg.lstsq(za, ya, rcond=None)[0]; yd = y - z @ beta
    return _adf_nodet(yd, k)


def _adf_nodet(yd, k, return_ic=False):
    dy = np.diff(yd); T = len(dy)
    idx = np.arange(k, T)
    X = [yd[idx]]                                   # y_{t-1}
    for j in range(1, k + 1):
        X.append(dy[idx - j])
    X = np.column_stack(X); yy = dy[idx]
    b = np.linalg.lstsq(X, yy, rcond=None)[0]; e = yy - X @ b
    s2 = e @ e / (len(yy) - X.shape[1]); se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
    if return_ic:
        return b[0] / se, np.log(e @ e / len(yy)), X.shape[1], len(yy)
    return b[0] / se


def _pp(y, regression, q):
    """Phillips-Perron Z_t statistic (Newey-West long-run variance with q lags)."""
    T = len(y) - 1
    X = [y[:-1]]
    if regression in ("c", "ct"):
        X.append(np.ones(T))
    if regression == "ct":
        X.append(np.arange(1, T + 1))
    X = np.column_stack(X); yy = y[1:]
    b = np.linalg.lstsq(X, yy, rcond=None)[0]; u = yy - X @ b
    k = X.shape[1]; s2 = u @ u / (T - k); se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
    tstat = (b[0] - 1) / se
    g0 = u @ u / T; lam2 = g0 + 2 * sum((1 - j / (q + 1)) * (u[j:] @ u[:-j]) / T for j in range(1, q + 1))
    z = np.sqrt(g0 / lam2) * tstat - 0.5 * (lam2 - g0) / np.sqrt(lam2) * (T * se / np.sqrt(s2))
    return z, tstat


def unitroot(df, spec):
    from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
    from statsmodels.tsa.adfvalues import mackinnonp
    y = need(spec, "y", "series")
    d, labels = _series_frame(df, spec, [y])
    x = d[y].to_numpy(float); n = len(x)
    if n < 40:
        raise OESError("Unit-root tests need at least 40 observations.")
    maxlag = int_option(spec, "maxlag", 8, 0, 24, "Maximum lag"); ic = spec.get("ic") or "bic"
    if ic not in ("aic", "bic"):
        raise OESError("Choose AIC or BIC for the lag length.")
    sims = int_option(spec, "sims", 200, 50, 1000, "Simulation replications (DF-GLS critical values)")
    rng = np.random.default_rng(int(spec.get("seed") or 0))
    forms = [("level", x), ("first difference", np.diff(x))]
    q = int(4 * (n / 100) ** (2 / 9))
    rows = []; dfg = []
    for lab, z in forms:
        for reg, rl in (("c", "constant"), ("ct", "constant + trend")):
            a = adfuller(z, maxlag=maxlag, regression=reg, autolag=ic.upper())
            rows.append(["ADF", lab, rl, f(a[0], 3), int(a[2]), f(a[1], 4), f(a[4]["5%"], 3), "yes" if a[1] < .05 else "no"])
    for lab, z in forms:
        for reg, rl in (("c", "constant"), ("ct", "constant + trend")):
            zt, _ = _pp(z, reg, q); p = float(mackinnonp(zt, regression=reg, N=1))
            rows.append(["Phillips-Perron", lab, rl, f(zt, 3), q, f(p, 4), None, "yes" if p < .05 else "no"])
    # DF-GLS with simulated critical values under the unit-root null
    for lab, z in forms:
        T = len(z)
        for reg, rl in (("c", "constant"), ("ct", "constant + trend")):
            ics = []
            T1 = T
            zz = z
            ab = 1 - (7.0 if reg == "c" else 13.5) / T1
            Z = np.ones((T1, 1)) if reg == "c" else np.column_stack([np.ones(T1), np.arange(1, T1 + 1)])
            ya = np.r_[zz[0], zz[1:] - ab * zz[:-1]]; za = np.vstack([Z[:1], Z[1:] - ab * Z[:-1]])
            yd = zz - Z @ np.linalg.lstsq(za, ya, rcond=None)[0]
            dy = np.diff(yd)
            for k in range(maxlag + 1):
                idx = np.arange(maxlag, len(dy)); Xk = [yd[idx]] + [dy[idx - j] for j in range(1, k + 1)]
                Xk = np.column_stack(Xk); yk = dy[idx]; e = yk - Xk @ np.linalg.lstsq(Xk, yk, rcond=None)[0]
                m = len(yk); pen = (2 if ic == "aic" else np.log(m)) * (k + 1) / m
                ics.append(np.log(e @ e / m) + pen)
            kopt = int(np.argmin(ics)); st = _adf_nodet(yd, kopt)
            sim = np.array([_dfgls_stat(np.cumsum(rng.standard_normal(T1)), reg, kopt) for _ in range(sims)])
            cv5 = float(np.percentile(sim, 5)); p = float((sim <= st).mean())
            rows.append(["DF-GLS", lab, rl, f(st, 3), kopt, f(max(p, 1 / sims), 3), f(cv5, 3), "yes" if st < cv5 else "no"])
    ur = table("Unit-root tests (H0: unit root)", ["Test", "Series form", "Deterministics", "Statistic", "Lags", "p-value", "5% critical value", "Rejects unit root at 5%"], rows,
               "Reject the unit root when the statistic is more negative than the critical value. ADF lags chosen by " + ic.upper() + f"; Phillips-Perron uses {q} Newey-West lags and MacKinnon p-values; DF-GLS p-values and critical values are simulated ({sims} random walks of the same length).")
    krows = []
    for lab, z in forms:
        for reg, rl in (("c", "level"), ("ct", "trend")):
            k_ = kpss(z, regression=reg, nlags=int(4 * (len(z) / 100) ** 0.25))
            krows.append(["KPSS", lab, f"stationary around a {rl}", f(k_[0], 3), int(k_[2]), f(k_[1], 4), f(k_[3]["5%"], 3), "yes" if k_[0] > k_[3]["5%"] else "no"])
    kt = table("Stationarity test (H0: stationary)", ["Test", "Series form", "H0", "Statistic", "Lags", "p-value", "5% critical value", "Rejects stationarity at 5%"], krows,
               "KPSS reverses the null. Rejecting stationarity in levels while not rejecting it in first differences points to I(1). The p-value is interpolated and truncated to the range 0.01 to 0.10.")
    za = zivot_andrews(x, maxlag=maxlag, regression="c", autolag=ic.upper()); zt_ = zivot_andrews(x, maxlag=maxlag, regression="ct", autolag=ic.upper())
    zrows = []
    for lab, r in (("break in the level", za), ("break in level and trend", zt_)):
        zrows.append([lab, f(r[0], 3), f(r[1], 4), f(r[2]["5%"], 3), labels[int(r[4])] if 0 <= int(r[4]) < n else str(r[4]), int(r[3]), "yes" if r[0] < r[2]["5%"] else "no"])
    zat = table("Zivot-Andrews test (one structural break, H0: unit root without break)", ["Break allowed in", "Statistic", "p-value", "5% critical value", "Break date", "Lags", "Rejects unit root at 5%"], zrows,
                "Applied to the level of the series. The break date is the one that gives the strongest evidence against the unit root.")
    # verdict
    def rej(rowlist, lab):
        return [r[-1] == "yes" for r in rowlist if r[1] == lab]
    adf_lev = rej([r for r in rows if r[0] == "ADF"], "level"); adf_dif = rej([r for r in rows if r[0] == "ADF"], "first difference")
    kp_lev = rej(krows, "level"); kp_dif = rej(krows, "first difference")
    n_unit_lev = sum(1 for r in rows if r[1] == "level" and r[-1] == "no") + sum(kp_lev)
    n_stat_dif = sum(1 for r in rows if r[1] == "first difference" and r[-1] == "yes") + sum(1 for v in kp_dif if not v)
    tot_lev = sum(1 for r in rows if r[1] == "level") + len(kp_lev); tot_dif = sum(1 for r in rows if r[1] == "first difference") + len(kp_dif)
    if n_unit_lev >= 0.75 * tot_lev and n_stat_dif >= 0.75 * tot_dif:
        verdict = "I(1): a unit root in levels, stationary in first differences"
    elif n_unit_lev <= 0.25 * tot_lev:
        verdict = "I(0): stationary in levels"
    elif n_unit_lev >= 0.75 * tot_lev:
        verdict = "Not stationary in first differences: check for I(2) or a break"
    else:
        verdict = "Mixed evidence: the tests disagree (a break or a near-unit root is likely)"
    vt = table("Summary of the evidence", ["Statement", "Tests in favour", "Out of"],
               [["Unit root in levels", n_unit_lev, tot_lev], ["Stationary in first differences", n_stat_dif, tot_dif]], "Counts ADF, Phillips-Perron, DF-GLS (reject = stationary) and KPSS (not reject = stationary) in both deterministic specifications.")
    xs_ = np.arange(n)
    ch = [chart(f"{y}: level", labels, [(y, x)]), chart(f"{y}: first difference", labels[1:], [("first difference", np.diff(x))])]
    summary = [stat("Observations", n), stat("Conclusion", verdict), stat("ADF p-value (level, constant)", float(rows[0][5]), 3), stat("KPSS p-value (level)", float(krows[0][5]), 3)]
    return result("unitroot", f"Unit-root and stationarity tests: {y}", n, summary, [ur, kt, zat, vt], ch,
                  ["Run the tests on levels and first differences. A conclusion of I(1) means you should difference the series or model it with cointegration.", "Seasonal unit roots and the Lee-Strazicich LM test are not part of the OES version."])


# ---------------------------------------------------------------------------------------------------------------
# Unobserved-components trend-cycle decomposition, compared with the HP and Hamilton filters
# ---------------------------------------------------------------------------------------------------------------
_UC_TRENDS = {"smooth trend": "smooth trend", "local linear trend": "lltrend", "random walk with drift": "rwdrift"}


def _hamilton(x, h, p=4):
    """Hamilton (2018) regression filter: the cycle is the residual of x_{t+h} on a constant and x_t ... x_{t-p+1}."""
    T = len(x); idx = np.arange(p - 1, T - h)
    X = np.column_stack([np.ones(len(idx))] + [x[idx - j] for j in range(p)]); yy = x[idx + h]
    b = np.linalg.lstsq(X, yy, rcond=None)[0]; e = np.full(T, np.nan); e[idx + h] = yy - X @ b
    return e


def _uc_fit(x, trend, cycle, ppy, bounds):
    import statsmodels.api as sm
    kw = dict(level=_UC_TRENDS[trend])
    if cycle:
        kw.update(cycle=True, stochastic_cycle=True, damped_cycle=True, cycle_period_bounds=bounds)
    m = sm.tsa.UnobservedComponents(x, **kw)
    r = m.fit(method="powell", disp=0, maxiter=3000)
    return m, m.fit(r.params, disp=0, maxiter=300)


def unobserved(df, spec):
    from statsmodels.tsa.filters.hp_filter import hpfilter
    y = need(spec, "y", "series")
    d, labels = _series_frame(df, spec, [y])
    x = d[y].to_numpy(float); n = len(x)
    if (spec.get("log") or "no") == "yes":
        if (x <= 0).any():
            raise OESError("The log transformation needs a series that is positive in every period.")
        x = 100 * np.log(x)
    ppy = int_option(spec, "ppy", 4, 1, 12, "Periods per year")
    if n < 8 * ppy or n < 40:
        raise OESError("The decomposition needs at least 40 observations and eight years of data.")
    trend = spec.get("trend") or "smooth trend"
    if trend not in _UC_TRENDS:
        raise OESError("Choose smooth trend, local linear trend or random walk with drift.")
    cyc = (spec.get("cycle") or "yes") == "yes"
    bounds = (int_option(spec, "pmin", 6, 2, 60, "Shortest cycle"), int_option(spec, "pmax", 40, 8, 200, "Longest cycle"))
    if bounds[0] >= bounds[1]:
        raise OESError("The shortest cycle must be shorter than the longest.")
    m, r = _uc_fit(x, trend, cyc, ppy, bounds)
    se = r.bse
    prow = []
    for nm, v, s_ in zip(m.param_names, r.params, se):
        prow.append([nm, f(v, 5), f(s_, 5)])
    tabs = [table(f"Unobserved-components model: {trend}" + (" + damped stochastic cycle" if cyc else ""), ["Parameter", "Estimate", "s.e."], prow,
                  "Variances of the shocks to each component; a variance of zero means that component is deterministic (e.g. a smooth trend has no level shock).")]
    summ = [stat("Observations", n), stat("Log-likelihood", float(r.llf), 2), stat("AIC", float(r.aic), 1)]
    tr_est = r.smoothed_state[0]
    cycle_est = np.asarray(r.cycle.smoothed) if cyc else x - tr_est
    if cyc:
        fr = float(r.params[m.param_names.index("frequency.cycle")]); damp = float(r.params[m.param_names.index("damping.cycle")])
        per = 2 * np.pi / fr
        summ += [stat("Cycle period (years)", per / ppy, 2), stat("Cycle damping", damp, 3)]
    # comparison of specifications by information criteria
    crow = []
    for tn in _UC_TRENDS:
        for cy in (True, False):
            try:
                m2, r2 = _uc_fit(x, tn, cy, ppy, bounds)
                crow.append([tn, "yes" if cy else "no", f(float(r2.llf), 2), f(float(r2.aic), 1), f(float(r2.bic), 1)])
            except Exception:
                crow.append([tn, "yes" if cy else "no", None, None, None])
    tabs.append(table("Model comparison", ["Trend", "Cycle", "Log-likelihood", "AIC", "BIC"], crow, "Lower AIC and BIC are better."))
    # filters
    lam = {1: 100, 4: 1600, 12: 129600}.get(ppy, 1600 * (ppy / 4) ** 4)
    hp_c, hp_t = hpfilter(x, lamb=lam)
    ham = _hamilton(x, h=2 * ppy, p=4); ok = ~np.isnan(ham)
    cycles = {"Unobserved components": cycle_est, "Hodrick-Prescott": np.asarray(hp_c), "Hamilton": ham}
    srows = []; names = list(cycles)
    for nm in names:
        c_ = cycles[nm]; srows.append([nm, f(np.nanstd(c_), 3), f(np.nanstd(c_[ok]), 3)] + [f(np.corrcoef(c_[ok], cycles[o_][ok])[0, 1], 3) for o_ in names])
    tabs.append(table("Cycle estimates compared", ["Filter", "Std. dev. (own sample)", "Std. dev. (common sample)"] + [f"Corr. with {o_}" for o_ in names], srows,
                      f"HP smoothing parameter {lam:g}; Hamilton regression uses h = {2 * ppy} periods and 4 lags. Correlations use the sample in which all three are defined."))
    gr = np.diff(tr_est) * ppy
    tabs.append(table("Trend growth (annualised, in the units of the series per year)", ["Period", "Trend growth"],
                      [[labels[i], f(gr[i - 1], 3)] for i in sorted(set([1, n // 4, n // 2, 3 * n // 4, n - 1]))], "Slope of the estimated trend."))
    ch = [chart(f"{y}: series and estimated trend", labels, [(y, x), ("UC trend", tr_est), ("HP trend", np.asarray(hp_t))]),
          chart(f"{y}: estimated cycles", labels, [(k, v) for k, v in cycles.items()])]
    return result("unobserved", f"Trend-cycle decomposition: {y}", n, summ, tabs, ch,
                  ["The cycle is identified by the assumption that it is a stationary, damped, stochastic cycle with a period between the bounds; the estimate depends on it.",
                   "End-of-sample trend estimates from all the filters are revised when data are added."])


# ---------------------------------------------------------------------------------------------------------------
# Forecast evaluation: rolling-origin forecasts and Diebold-Mariano tests
# ---------------------------------------------------------------------------------------------------------------
def _ar_fc(z, h, maxp):
    """Direct-iterated AR(p) forecast with p chosen by BIC."""
    best = None
    for p in range(1, maxp + 1):
        if len(z) <= p + 5:
            break
        Y = z[p:]; X = np.column_stack([np.ones(len(Y))] + [z[p - j:len(z) - j] for j in range(1, p + 1)])
        b = np.linalg.lstsq(X, Y, rcond=None)[0]; e = Y - X @ b; bic = np.log(e @ e / len(Y)) + (p + 1) * np.log(len(Y)) / len(Y)
        if best is None or bic < best[0]:
            best = (bic, p, b)
    _, p, b = best; hist = list(z[-p:][::-1])
    for _ in range(h):
        nxt = b[0] + sum(b[j + 1] * hist[j] for j in range(p)); hist.insert(0, nxt)
    return hist[0]


def _dm(d, h):
    """Diebold-Mariano statistic with the Harvey-Leybourne-Newbold small-sample correction (loss differential d, horizon h)."""
    T = len(d); dm = d.mean(); lam = ((d - dm) @ (d - dm)) / T
    for j in range(1, h):
        lam += 2 * ((d[j:] - dm) @ (d[:-j] - dm)) / T
    lam = max(lam, 1e-12); stat_ = dm / np.sqrt(lam / T)
    hln = stat_ * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return hln, 2 * stats.t.sf(abs(hln), T - 1)


def forecast_eval(df, spec):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y = need(spec, "y", "series")
    d, labels = _series_frame(df, spec, [y])
    x = d[y].to_numpy(float); n = len(x)
    h = int_option(spec, "h", 1, 1, 12, "Forecast horizon"); w = int_option(spec, "window", max(40, n // 2), 30, n - 15, "Initial estimation window")
    maxp = int_option(spec, "maxp", 4, 1, 8, "Maximum AR lags")
    if n - w - h + 1 < 15:
        raise OESError("Too few forecast origins: shorten the initial window or the horizon (at least 15 forecasts are needed).")
    loss = spec.get("loss") or "squared"
    if loss not in ("squared", "absolute"):
        raise OESError("Choose squared or absolute loss.")
    base = spec.get("benchmark") or "random walk"
    names = ["random walk", "historical mean", "AR (BIC)", "exponential smoothing"]
    if base not in names:
        raise OESError("Choose a benchmark from: " + ", ".join(names) + ".")
    fc = {k: [] for k in names}; act = []; per = []
    for t in range(w, n - h + 1):
        z = x[:t]; act.append(x[t + h - 1]); per.append(labels[t + h - 1])
        fc["random walk"].append(z[-1]); fc["historical mean"].append(z.mean()); fc["AR (BIC)"].append(_ar_fc(z, h, maxp))
        try:
            fc["exponential smoothing"].append(float(ExponentialSmoothing(z, trend="add", damped_trend=True).fit().forecast(h)[-1]))
        except Exception:
            fc["exponential smoothing"].append(z[-1])
    act = np.array(act); fc = {k: np.array(v) for k, v in fc.items()}; err = {k: act - v for k, v in fc.items()}
    L = (lambda e: e ** 2) if loss == "squared" else np.abs
    rows = []; T = len(act)
    for k in names:
        rmse = float(np.sqrt((err[k] ** 2).mean())); mae = float(np.abs(err[k]).mean())
        rel = float(np.sqrt((err[k] ** 2).mean() / (err[base] ** 2).mean()))
        if k == base:
            rows.append([k, f(rmse, 4), f(mae, 4), 1.0, None, None, "benchmark"])
        else:
            st, p = _dm(L(err[base]) - L(err[k]), h)
            rows.append([k, f(rmse, 4), f(mae, 4), f(rel, 3), f(st, 3), f(p, 4), "better than benchmark" if (p < .05 and st > 0) else ("worse than benchmark" if (p < .05 and st < 0) else "no significant difference")])
    tabs = [table(f"Out-of-sample accuracy ({T} {h}-step-ahead forecasts)", ["Model", "RMSE", "MAE", "RMSE relative to benchmark", "DM statistic", "DM p-value", "Result"], rows,
                  f"Forecasts are re-estimated on an expanding window starting with {w} observations. Benchmark: {base}. DM tests the equality of {loss} loss with the Harvey-Leybourne-Newbold small-sample correction; a positive statistic favours the model over the benchmark.")]
    pair = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            st, p = _dm(L(err[a]) - L(err[b]), h); pair.append([f"{a} vs {b}", f(st, 3), f(p, 4)])
    tabs.append(table("Pairwise Diebold-Mariano tests", ["Pair", "DM statistic", "p-value"], pair, "H0: equal expected loss. A positive statistic means the first model has the larger loss, so the second is better."))
    best = min(names, key=lambda k: float(np.sqrt((err[k] ** 2).mean())))
    ch = [chart(f"{y}: actual and forecasts", per, [("actual", act)] + [(k, fc[k]) for k in names]),
          chart("Cumulative squared forecast error", per, [(k, np.cumsum(err[k] ** 2)) for k in names])]
    return result("forecast_eval", f"Forecast comparison: {y}", T, [stat("Forecasts", T), stat("Horizon", h), stat("Lowest RMSE", best), stat("Benchmark", base)], tabs, ch,
                  ["Forecasts use only the data available at each origin (no look-ahead), but the data are the final revised values, not real-time vintages.",
                   "The Diebold-Mariano test is for forecasts that do not come from nested models estimated on a short window; for nested models use the Clark-West test."])


# ---------------------------------------------------------------------------------------------------------------
# Diebold-Yilmaz connectedness (spillover) table from a VAR with generalised forecast-error variance decompositions
# ---------------------------------------------------------------------------------------------------------------
def _gfevd(res, H):
    """Row-normalised generalised FEVD (Pesaran-Shin): element [i, j] is the share of variable i's H-step forecast-error variance due to shocks to j."""
    A = res.ma_rep(H - 1); S = np.asarray(res.sigma_u); k = S.shape[0]
    th = np.zeros((k, k))
    for i in range(k):
        den = sum(A[h][i] @ S @ A[h][i] for h in range(H))
        for j in range(k):
            th[i, j] = sum((A[h][i] @ S[:, j]) ** 2 for h in range(H)) / (S[j, j] * den)
    return th / th.sum(1, keepdims=True)


def connectedness(df, spec):
    from statsmodels.tsa.api import VAR
    xs = as_list(need(spec, "x", "variables"))
    if len(xs) < 2:
        raise OESError("Connectedness needs at least two variables.")
    d, labels = _series_frame(df, spec, xs); n = len(d)
    if n < 60:
        raise OESError("Connectedness analysis needs at least 60 observations.")
    H = int_option(spec, "horizon", 10, 2, 40, "Forecast horizon"); w = int_option(spec, "window", min(120, n // 2), 30, n, "Rolling window")
    maxlag = int_option(spec, "maxlag", 6, 1, 12, "Maximum lag")
    data = d[xs].to_numpy(float)
    ic = VAR(data).select_order(maxlag); p = max(1, int(ic.selected_orders["bic"])); res = VAR(data).fit(p)
    th = _gfevd(res, H); k = len(xs)
    rows = [[xs[i]] + [f(100 * th[i, j], 1) for j in range(k)] + [f(100 * (th[i].sum() - th[i, i]), 1)] for i in range(k)]
    frm = 100 * (th.sum(1) - np.diag(th)); to = 100 * (th.sum(0) - np.diag(th)); net = to - frm
    rows.append(["To others"] + [f(to[j], 1) for j in range(k)] + [f(to.sum() / k, 1)])
    total = float(to.sum() / k)
    tabs = [table(f"Connectedness table (VAR({p}), horizon {H}, % of forecast-error variance)", ["Variable i"] + [f"from {x}" for x in xs] + ["From others"], rows,
                  "Row i, column j: the share of the H-step forecast-error variance of i that comes from shocks to j (generalised decomposition, rows normalised to 100). The bottom-right entry is the total connectedness.")]
    tabs.append(table("Directional spillovers", ["Variable", "To others", "From others", "Net (to - from)", "Role"],
                      [[xs[i], f(to[i], 1), f(frm[i], 1), f(net[i], 1), "net transmitter" if net[i] > 0 else "net receiver"] for i in range(k)]))
    # rolling total connectedness
    roll = []; rl = []
    for s0 in range(0, n - w + 1, max(1, (n - w) // 80 or 1)):
        try:
            rr = VAR(data[s0:s0 + w]).fit(p); t_ = _gfevd(rr, H); roll.append(100 * (t_.sum() - np.trace(t_)) / k); rl.append(labels[s0 + w - 1])
        except Exception:
            continue
    ch = [chart(f"Total connectedness, {w}-period rolling window", rl, [("total connectedness (%)", roll)])]
    tabs.append(table("Total connectedness over time", ["Statistic", "Value"], [["Full sample", f(total, 1)], ["Minimum over rolling windows", f(min(roll), 1)], ["Maximum over rolling windows", f(max(roll), 1)],
                      ["Date of maximum", rl[int(np.argmax(roll))]]]))
    return result("connectedness", f"Connectedness (Diebold-Yilmaz): {', '.join(xs)}", n, [stat("Observations", n), stat("VAR lags (BIC)", p), stat("Total connectedness (%)", total, 1),
                  stat("Largest net transmitter", xs[int(np.argmax(net))])], tabs, ch,
                  ["Generalised decompositions do not depend on the ordering of the variables, but the shocks are not orthogonal, so the shares are normalised to sum to one.",
                   "Connectedness is a statement about forecast-error variance, not about causality."])


# ---------------------------------------------------------------------------------------------------------------
# Toda-Yamamoto Granger non-causality test in levels (valid whether the series are I(0), I(1) or cointegrated)
# ---------------------------------------------------------------------------------------------------------------
def granger_ty(df, spec):
    from statsmodels.tsa.api import VAR
    xs = as_list(need(spec, "x", "variables"))
    if len(xs) < 2:
        raise OESError("The test needs at least two variables.")
    d, labels = _series_frame(df, spec, xs); n = len(d); k = len(xs)
    dmax = int_option(spec, "dmax", 1, 0, 3, "Maximum order of integration (dmax)"); maxlag = int_option(spec, "maxlag", 8, 1, 12, "Maximum lag")
    if n < 8 * k + 20:
        raise OESError("Too few observations for this number of variables.")
    data = d[xs].to_numpy(float)
    ic = VAR(data).select_order(maxlag); crit = spec.get("ic") or "aic"
    if crit not in ("aic", "bic", "hqic"):
        raise OESError("Choose AIC, BIC or HQIC.")
    p = int(spec.get("lags") or 0) or max(1, int(ic.selected_orders[crit]))
    rows = []
    for i, dep in enumerate(xs):
        for j, cause in enumerate(xs):
            if i == j:
                continue
            # H0: the first p lags of `cause` do not enter the equation of `dep` (the extra dmax lags are left unrestricted)
            idx = [1 + (l * k + j) for l in range(p)]
            X = _ty_design(data, p + dmax); yv = data[p + dmax:, i]
            beta = np.linalg.lstsq(X, yv, rcond=None)[0]; e = yv - X @ beta; s2 = e @ e / (len(yv) - X.shape[1])
            Vb = s2 * np.linalg.inv(X.T @ X); R = beta[idx]; W = float(R @ np.linalg.solve(Vb[np.ix_(idx, idx)], R))
            rows.append([f"{cause} -> {dep}", f(W, 3), p, f(stats.chi2.sf(W, p), 4), "rejects non-causality" if stats.chi2.sf(W, p) < .05 else "does not reject"])
    tabs = [table(f"Toda-Yamamoto Granger causality (VAR({p}) plus {dmax} extra lag(s) in levels)", ["Direction", "Wald chi-squared", "df", "p-value", "5% decision"], rows,
                  "H0: the first p lags of the causing variable are jointly zero in the equation of the dependent variable. The dmax extra lags are included but not tested, which keeps the Wald statistic chi-squared even with unit roots.")]
    tabs.append(table("Lag selection", ["Criterion", "Selected lag"], [[c.upper(), int(ic.selected_orders[c])] for c in ("aic", "bic", "hqic", "fpe")]))
    # robustness: p-values over alternative lag lengths
    rob = []
    for pp in range(1, min(maxlag, 6) + 1):
        Xp = _ty_design(data, pp + dmax); row = [pp]
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                yv = data[pp + dmax:, i]; be = np.linalg.lstsq(Xp, yv, rcond=None)[0]; e = yv - Xp @ be; s2 = e @ e / (len(yv) - Xp.shape[1])
                Vb = s2 * np.linalg.inv(Xp.T @ Xp); idx = [1 + (l * k + j) for l in range(pp)]; Rr = be[idx]
                row.append(f(stats.chi2.sf(float(Rr @ np.linalg.solve(Vb[np.ix_(idx, idx)], Rr)), pp), 3))
        rob.append(row)
    cols_ = ["Lags p"] + [f"{xs[j]} -> {xs[i]}" for i in range(k) for j in range(k) if i != j]
    tabs.append(table("p-values for other lag lengths", cols_, rob, "A causal conclusion that holds only for one lag length is fragile."))
    sig = [r[0] for r in rows if r[-1].startswith("rejects")]
    return result("granger_ty", f"Toda-Yamamoto Granger causality: {', '.join(xs)}", n - p - dmax, [stat("Observations used", n - p - dmax), stat("Lags tested (p)", p), stat("Extra lags (dmax)", dmax),
                  stat("Directions rejecting non-causality", ", ".join(sig) if sig else "none")], tabs, [],
                  ["Granger causality is predictive content, not structural causality. Set dmax to the highest order of integration among the variables (from the unit-root tests)."])


def _ty_design(data, L):
    T, k = data.shape
    return np.column_stack([np.ones(T - L)] + [data[L - l:T - l, j] for l in range(1, L + 1) for j in range(k)])


# ---------------------------------------------------------------------------------------------------------------
# Seasonal decomposition (STL), seasonality tests and seasonal adjustment
# ---------------------------------------------------------------------------------------------------------------
def seasonal(df, spec):
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import STL
    y = need(spec, "y", "series")
    d, labels = _series_frame(df, spec, [y])
    x = d[y].to_numpy(float); n = len(x)
    m = int_option(spec, "period", 4, 2, 52, "Seasonal period")
    if n < 3 * m + 4:
        raise OESError(f"The series is too short for a seasonal period of {m}: at least {3 * m + 4} observations are needed.")
    mode = spec.get("mode") or "multiplicative"
    if mode not in ("additive", "multiplicative"):
        raise OESError("Choose additive or multiplicative.")
    if mode == "multiplicative" and (x <= 0).any():
        raise OESError("A multiplicative decomposition needs a positive series; choose additive.")
    robust = (spec.get("robust") or "yes") == "yes"
    z = np.log(x) if mode == "multiplicative" else x
    res = STL(z, period=m, robust=robust, seasonal=int_option(spec, "swindow", 7, 3, 101, "Seasonal window") | 1).fit()
    S, T_, R = np.asarray(res.seasonal), np.asarray(res.trend), np.asarray(res.resid)
    Fs = max(0.0, 1 - R.var() / (S + R).var()); Ft = max(0.0, 1 - R.var() / (T_ + R).var())
    # test for seasonality: F test of period dummies in a regression of the detrended series
    det = z - T_; per = np.arange(n) % m
    D = np.column_stack([(per == j).astype(float) for j in range(m)])
    ols = sm.OLS(det, D).fit(); Fst = ((det - det.mean()) @ (det - det.mean()) - ols.ssr) / (m - 1) / (ols.ssr / (n - m)); pF = stats.f.sf(Fst, m - 1, n - m)
    kw = stats.kruskal(*[det[per == j] for j in range(m)])
    adj = np.exp(z - S) if mode == "multiplicative" else z - S
    factors = np.exp(S[:m]) if mode == "multiplicative" else S[:m]
    # seasonal pattern by period (average over the sample) and stability of its amplitude
    srow = []
    thirds = np.array_split(np.arange(n), 3)
    for j in range(m):
        vals = [(np.exp(S[i]) if mode == "multiplicative" else S[i]) for i in range(n) if per[i] == j]
        srow.append([f"period {j + 1}", f(np.mean(vals), 4), f(np.min(vals), 4), f(np.max(vals), 4)])
    amp = [float(np.ptp([np.mean([S[i] for i in th if per[i] == j]) for j in range(m)])) for th in thirds]
    tabs = [table(f"Seasonal factors ({mode})", ["Position in the cycle", "Mean factor", "Minimum", "Maximum"], srow,
                  "Multiplicative factors are ratios to the trend (1.00 = no effect); additive factors are in the units of the series. The first observation is position 1.")]
    tabs.append(table("Strength and tests", ["Statistic", "Value", "p-value"],
                      [["Seasonal strength (0 to 1)", f(Fs, 3), None], ["Trend strength (0 to 1)", f(Ft, 3), None], ["F test of seasonal dummies (detrended)", f(Fst, 2), f(pF, 4)],
                       ["Kruskal-Wallis test across periods", f(kw[0], 2), f(kw[1], 4)]],
                      "Strength = 1 - Var(remainder) / Var(component + remainder) (Wang, Smith and Hyndman). The tests are for equal means across the positions of the cycle."))
    tabs.append(table("Is the seasonal pattern stable?", ["Part of the sample", "Seasonal amplitude (max - min of the mean factor, log units)" if mode == "multiplicative" else "Seasonal amplitude (max - min)"],
                      [[f"Part {i + 1} (to {labels[th[-1]]})", f(a, 4)] for i, (th, a) in enumerate(zip(thirds, amp))], "A changing amplitude means the pattern evolves; STL allows this, fixed dummies do not."))
    ch = [chart(f"{y}: original, trend and seasonally adjusted", labels, [(y, x), ("trend", np.exp(T_) if mode == "multiplicative" else T_), ("seasonally adjusted", adj)]),
          chart("Seasonal component", labels, [("seasonal", np.exp(S) if mode == "multiplicative" else S)])]
    return result("seasonal", f"Seasonal decomposition (STL): {y}", n, [stat("Observations", n), stat("Seasonal strength", Fs, 3), stat("Trend strength", Ft, 3),
                  stat("Seasonality test p-value", float(pF), 4)], tabs, ch,
                  ["STL is a descriptive filter: it is not the X-13 ARIMA-SEATS procedure used by statistical agencies, and it does not adjust for trading days or holidays.",
                   "Seasonally adjust before estimating only if the seasonal pattern is not part of the question; otherwise include seasonal terms in the model."])


# ---------------------------------------------------------------------------------------------------------------
# Long memory: GPH and local Whittle estimates of the fractional order of integration d
# ---------------------------------------------------------------------------------------------------------------
def _periodogram(x):
    n = len(x); xc = x - x.mean(); fft = np.fft.fft(xc); I = (np.abs(fft) ** 2) / (2 * np.pi * n); lam = 2 * np.pi * np.arange(n) / n
    return lam[1:(n - 1) // 2 + 1], I[1:(n - 1) // 2 + 1]


def _gph(x, m):
    lam, I = _periodogram(x); lam, I = lam[:m], I[:m]
    X = -2 * np.log(2 * np.sin(lam / 2)); Xc = X - X.mean(); b = (Xc @ np.log(I)) / (Xc @ Xc)
    return float(b), float(np.sqrt(np.pi ** 2 / 6 / (Xc @ Xc)))


def _whittle(x, m):
    lam, I = _periodogram(x); lam, I = lam[:m], I[:m]
    def obj(d):
        return np.log(np.mean(lam ** (2 * d) * I)) - 2 * d * np.mean(np.log(lam))
    r = optimize.minimize_scalar(obj, bounds=(-0.49, 1.49), method="bounded")
    return float(r.x), float(1 / (2 * np.sqrt(m)))


def _fracdiff(x, d, thresh=1e-5):
    n = len(x); w = [1.0]
    for k in range(1, n):
        w.append(-w[-1] * (d - k + 1) / k)
    w = np.array(w); return np.array([w[:t + 1] @ x[t::-1][:t + 1] for t in range(n)])


def longmemory(df, spec):
    from statsmodels.tsa.stattools import acf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    y = need(spec, "y", "series")
    d_, labels = _series_frame(df, spec, [y])
    x0 = d_[y].to_numpy(float); n = len(x0)
    if n < 100:
        raise OESError("Long-memory estimates need at least 100 observations.")
    tr = spec.get("transform") or "level"
    if tr not in ("level", "absolute", "squared", "log absolute"):
        raise OESError("Choose level, absolute, squared or log absolute.")
    xc = x0 - x0.mean()
    x = {"level": x0, "absolute": np.abs(xc), "squared": xc ** 2, "log absolute": np.log(np.abs(xc) + 1e-8)}[tr]
    pw = float_option(spec, "power", 0.65, "Bandwidth power")
    if not 0.4 <= pw <= 0.85:
        raise OESError("Bandwidth power must be between 0.4 and 0.85.")
    ms = sorted(set([int(n ** p_) for p_ in (0.5, 0.6, pw, 0.75, 0.8)]))
    rows = []
    for m in ms:
        g, sg = _gph(x, m); w_, sw = _whittle(x, m)
        rows.append([m, f"n^{np.log(m) / np.log(n):.2f}", f(g, 3), f(sg, 3), f(w_, 3), f(sw, 3), f(2 * stats.norm.sf(abs(w_) / sw), 4)])
    m0 = int(n ** pw); g0, sg0 = _gph(x, m0); w0, sw0 = _whittle(x, m0)
    tabs = [table("Estimates of the fractional order d", ["Frequencies m", "Bandwidth", "GPH d", "s.e.", "Local Whittle d", "s.e.", "p (Whittle, d = 0)"], rows,
                  "d = 0: short memory; 0 < d < 0.5: stationary long memory (autocorrelations decay hyperbolically); 0.5 <= d < 1: non-stationary but mean-reverting; d = 1: unit root. Standard errors are asymptotic and ignore the bandwidth choice.")]
    ac = acf(x, nlags=60, fft=True); lags = [1, 5, 10, 20, 40, 60]
    lb0 = acorr_ljungbox(x, lags=[20], return_df=True)
    xf = _fracdiff(x - x.mean(), w0); lb1 = acorr_ljungbox(xf[20:], lags=[20], return_df=True)
    tabs.append(table("Autocorrelations", ["Lag"] + [str(l) for l in lags], [["Series"] + [f(ac[l], 3) for l in lags]], "Short memory: autocorrelations decay geometrically; long memory: slowly."))
    tabs.append(table("Does fractional differencing remove the dependence?", ["Series", "Ljung-Box (20)", "p-value"],
                      [[f"{tr}, original", f(lb0.lb_stat.iloc[0], 2), f(lb0.lb_pvalue.iloc[0], 4)], [f"after (1 - L)^{w0:.2f}", f(lb1.lb_stat.iloc[0], 2), f(lb1.lb_pvalue.iloc[0], 4)]],
                      "If the fractional difference is white noise the long-memory model captures the dependence."))
    ch = [chart("Autocorrelation function", list(range(1, 61)), [("ACF", ac[1:61])]), chart("Log periodogram against log frequency", list(range(1, m0 + 1)), [("log I(lambda)", np.log(_periodogram(x)[1][:m0]))])]
    return result("longmemory", f"Long memory: {y} ({tr})", n, [stat("Observations", n), stat("GPH d", g0, 3), stat("Local Whittle d", w0, 3), stat("Bandwidth m", m0)], tabs, ch,
                  ["Structural breaks and regime changes produce spurious long memory: check for breaks before you conclude that d > 0.",
                   "The estimates are semiparametric and depend on the bandwidth; report a range."])
