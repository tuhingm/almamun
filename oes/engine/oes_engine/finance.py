"""Finance models: factor-pricing regressions with the GRS test, value at risk and expected shortfall with backtests, DCC-GARCH.
Pure NumPy and SciPy. Same result format as the other models (core.py)."""
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.signal import lfilter

from .core import (OESError, as_list, chart, distinct, f, float_option, int_option, need, numeric, result, stat, table, test_row, tests_table)
from .timeseries import _series_frame


def _nw_cov(X, e, lags):
    """Newey-West covariance of the OLS coefficients."""
    n = len(e)
    XtXi = np.linalg.inv(X.T @ X)
    U = X * e[:, None]
    S = U.T @ U
    for L in range(1, lags + 1):
        G = U[L:].T @ U[:-L]
        S += (1 - L / (lags + 1)) * (G + G.T)
    return XtXi @ S @ XtXi


# ================================================================================================ factor pricing and the GRS test
def capm(df, spec):
    assets = as_list(need(spec, "assets", "asset or portfolio returns"))
    facs = as_list(need(spec, "factors", "factor(s), e.g. the market excess return"))
    rf = spec.get("rf") or None
    distinct(assets, facs)
    cols_ = assets + facs + ([rf] if rf else [])
    d, labels = _series_frame(df, spec, cols_)
    T = len(d)
    N, K = len(assets), len(facs)
    if T < 3 * (N + K) + 20:
        raise OESError(f"Too few periods ({T}) for {N} assets and {K} factors.")
    lags = int_option(spec, "lags", int(np.floor(4 * (T / 100) ** (2 / 9))), 0, 24, "Newey-West lags")
    Y = d[assets].to_numpy(float)
    if rf:
        Y = Y - d[[rf]].to_numpy(float)
    F = d[facs].to_numpy(float)
    X = np.column_stack([np.ones(T), F])
    B = np.linalg.solve(X.T @ X, X.T @ Y)                       # (1+K) x N
    E = Y - X @ B
    rows, alphas, se_a, fits = [], [], [], []
    for j in range(N):
        V = _nw_cov(X, E[:, j], lags)
        se = np.sqrt(np.diag(V))
        r2 = 1 - E[:, j].var() / Y[:, j].var()
        alphas.append(B[0, j]); se_a.append(se[0])
        rows.append([assets[j], B[0, j], se[0], B[0, j] / se[0], 2 * stats.norm.sf(abs(B[0, j] / se[0]))] + [B[1 + k, j] for k in range(K)] + [r2])
    ctab = table("Alpha and factor loadings (Newey-West standard errors for alpha)",
                 ["Asset", "Alpha", "s.e.", "t", "p"] + [f"Beta: {x}" for x in facs] + ["R-squared"], rows,
                 f"Excess returns{' (risk-free column subtracted)' if rf else ''}; Newey-West with {lags} lags.")
    betas = []
    for j in range(N):
        V = _nw_cov(X, E[:, j], lags)
        se = np.sqrt(np.diag(V))
        betas.append([assets[j]] + [f"{B[1 + k, j]:.3f} ({se[1 + k]:.3f})" for k in range(K)])
    btab = table("Factor loadings with standard errors", ["Asset"] + [f"Beta: {x}" for x in facs], betas)
    Sig = E.T @ E / T
    a = B[0]
    mu = F.mean(0)
    Om = np.atleast_2d(np.cov(F.T, bias=True))
    SR2 = float(mu @ np.linalg.solve(Om, mu))
    q = float(a @ np.linalg.solve(Sig, a))
    grs = (T - N - K) / N * q / (1 + SR2)
    dfd = T - N - K
    p = stats.f.sf(grs, N, dfd)
    # Wald with the HAC covariance of the stacked alphas is not used: the GRS test is exact under normality
    tests = [test_row("GRS: all alphas jointly zero", grs, p, f"F({N}, {dfd}); H0: the factors price the assets"),
             test_row("Wald (chi-square form of the GRS)", T * q / (1 + SR2), stats.chi2.sf(T * q / (1 + SR2), N), f"{N} df")]
    sr_f = float(np.sqrt(SR2))
    sr_all = float(np.sqrt(SR2 + q))
    summ = [stat("Periods", T), stat("Assets", N), stat("GRS F", grs, 3), stat("GRS p", p, 4), stat("Factor Sharpe ratio", sr_f, 3), stat("Sharpe ratio with the alphas", sr_all, 3)]
    ci = 1.96 * np.array(se_a)
    charts = [chart("Alphas with 95% intervals", assets, [("Alpha", alphas)], "Asset", "Alpha per period", "bar", band=(np.array(alphas) - ci, np.array(alphas) + ci))]
    w = min(60, T // 3)
    if T >= 90:
        rb = [np.linalg.lstsq(X[t - w:t], Y[t - w:t, 0], rcond=None)[0][1] for t in range(w, T + 1)]
        charts.append(chart(f"Rolling {w}-period beta of {assets[0]} on {facs[0]}", labels[w - 1:], [("Beta", rb)], "", "Beta"))
    return result("capm", "Factor-pricing regressions and the GRS test", T, summ, [ctab, btab, tests_table(tests, "Joint test of the alphas")], charts,
                  ["The GRS statistic is exact in finite samples when the errors are normal and homoskedastic; the Newey-West alpha standard errors are robust to heteroskedasticity and autocorrelation.",
                   "A significant joint test means that the factors leave pricing errors; the Sharpe ratio with the alphas shows how much the pricing errors would add."])


# ================================================================================================ value at risk and expected shortfall
def _garch11_fit(r):
    n = len(r)
    sc = r.std()
    x = (r - r.mean()) / sc
    v0 = 1.0

    def filt(p, e):
        om, al, be = p
        s2 = lfilter([1.0], [1.0, -be], om + al * np.concatenate([[v0], e[:-1] ** 2]))
        return s2

    def nll(p):
        mu, lo, la, lb = p
        om, al, be = np.exp(lo), 1 / (1 + np.exp(-la)), 1 / (1 + np.exp(-lb))
        if al + be >= 0.9995:
            return 1e10
        e = x - mu
        s2 = filt((om, al, be), e)
        if np.any(s2 <= 0):
            return 1e10
        return 0.5 * np.sum(np.log(s2) + e ** 2 / s2)
    best = None
    for a0, b0 in [(0.05, 0.9), (0.1, 0.8), (0.03, 0.95)]:
        st = [0.0, np.log(0.05), np.log(a0 / (1 - a0)), np.log(b0 / (1 - b0))]
        o = optimize.minimize(nll, st, method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-7, fatol=1e-9))
        if best is None or o.fun < best.fun:
            best = o
    mu, lo, la, lb = best.x
    om, al, be = np.exp(lo), 1 / (1 + np.exp(-la)), 1 / (1 + np.exp(-lb))
    return dict(mu=mu * sc + r.mean(), om=om * sc ** 2, al=al, be=be, sc=sc, m0=r.mean(), ll=-best.fun - n * np.log(sc) - 0.5 * n * np.log(2 * np.pi))


def _garch_path(r, p):
    e = r - p["mu"]
    s2 = lfilter([1.0], [1.0, -p["be"]], p["om"] + p["al"] * np.concatenate([[p["om"] / (1 - p["al"] - p["be"])], e[:-1] ** 2]))
    return s2


def _t_fit_nu(z):
    def nll(lnu):
        nu = 2.05 + np.exp(lnu)
        return -np.sum(stats.t.logpdf(z * np.sqrt(nu / (nu - 2)), nu) + 0.5 * np.log(nu / (nu - 2)))
    o = optimize.minimize_scalar(nll, bounds=(-3, 5), method="bounded")
    return 2.05 + np.exp(o.x)


def _kupiec(x, n, p):
    if x == 0:
        return -2 * n * np.log(1 - p)
    ph = x / n
    if ph >= 1:
        return np.inf
    return -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - ph) - x * np.log(ph))


def _christ(v):
    v = v.astype(int)
    n00 = np.sum((v[:-1] == 0) & (v[1:] == 0)); n01 = np.sum((v[:-1] == 0) & (v[1:] == 1))
    n10 = np.sum((v[:-1] == 1) & (v[1:] == 0)); n11 = np.sum((v[:-1] == 1) & (v[1:] == 1))
    if n01 + n11 == 0 or n01 == 0 or n00 + n01 == 0:
        return 0.0
    p01 = n01 / (n00 + n01); p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = (n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p)
    ll1 = n00 * np.log(1 - p01) + n01 * np.log(p01) + (n10 * np.log(1 - p11) if n10 else 0) + (n11 * np.log(p11) if n11 else 0)
    return max(-2 * (ll0 - ll1), 0.0)


def var_es(df, spec):
    y = need(spec, "y", "return series")
    d, labels = _series_frame(df, spec, [y])
    r = d[y].to_numpy(float)
    n = len(r)
    p = float_option(spec, "alpha", 0.01, "Tail probability")
    if not 0.001 <= p <= 0.2:
        raise OESError("The tail probability must be between 0.001 and 0.2.")
    w = int_option(spec, "window", 500, 100, 5000, "Estimation window")
    lam = float_option(spec, "lambda", 0.94, "EWMA decay")
    if not 0.5 <= lam < 1:
        raise OESError("The EWMA decay must be between 0.5 and 1.")
    if n < w + 100:
        raise OESError(f"Too few observations ({n}): the window of {w} leaves {n - w} out-of-sample days; at least 100 are needed.")
    zq = stats.norm.ppf(p)
    es_n = -stats.norm.pdf(zq) / p
    m = n - w
    out = {}
    # rolling historical simulation and rolling normal
    H = np.lib.stride_tricks.sliding_window_view(r, w)[:-1][:m]            # window ending the day before each forecast day
    q_h = np.quantile(H, p, axis=1)
    es_h = np.array([h[h <= qq].mean() if np.any(h <= qq) else qq for h, qq in zip(H, q_h)])
    out["Historical simulation"] = (q_h, es_h)
    mu_w = H.mean(1); sd_w = H.std(1, ddof=1)
    out["Normal (rolling window)"] = (mu_w + zq * sd_w, mu_w + es_n * sd_w)
    # EWMA
    e = r - r[:w].mean()
    s2 = lfilter([1 - lam], [1, -lam], np.concatenate([[r[:w].var()], e[:-1] ** 2]))
    s2 = s2[w:]
    mu0 = r[:w].mean()
    out["EWMA normal"] = (mu0 + zq * np.sqrt(s2), mu0 + es_n * np.sqrt(s2))
    # GARCH(1,1): fitted on the first window only, then filtered through the out-of-sample days
    g = _garch11_fit(r[:w])
    sg = np.sqrt(_garch_path(r, g))[w:]
    zres = ((r[:w] - g["mu"]) / np.sqrt(_garch_path(r[:w], g)))
    nu = _t_fit_nu(zres)
    sc = np.sqrt((nu - 2) / nu)
    tq = stats.t.ppf(p, nu) * sc
    t_es = -(stats.t.pdf(stats.t.ppf(p, nu), nu) * (nu + stats.t.ppf(p, nu) ** 2) / ((nu - 1) * p)) * sc
    out["GARCH(1,1), t errors"] = (g["mu"] + tq * sg, g["mu"] + t_es * sg)
    qz = np.quantile(zres, p)
    esz = zres[zres <= qz].mean()
    out["GARCH filtered historical simulation"] = (g["mu"] + qz * sg, g["mu"] + esz * sg)
    ro = r[w:]
    rows, tests, series = [], [], []
    for name, (q, es) in out.items():
        v = ro < q
        x = int(v.sum())
        lr_u = _kupiec(x, m, p)
        lr_i = _christ(v)
        sev = float(np.mean(ro[v] / es[v])) if x else np.nan
        tick = float(np.mean((p - (ro < q)) * (ro - q)))
        rows.append([name, x, 100 * x / m, 100 * p, float(np.mean(q)), float(np.mean(es)), float(np.mean(ro[v])) if x else None, sev, tick * 100])
        tests.append([name, f(lr_u, 3), f(stats.chi2.sf(lr_u, 1), 4), f(lr_i, 3), f(stats.chi2.sf(lr_i, 1), 4), f(lr_u + lr_i, 3), f(stats.chi2.sf(lr_u + lr_i, 2), 4)])
        series.append((name, q))
    tab = table(f"Out-of-sample value at risk at the {100 * (1 - p):g}% level ({m} days)", ["Method", "Violations", "Violation rate (%)", "Expected rate (%)", "Mean VaR", "Mean ES",
                                                                                         "Mean return on violation days", "Realised / forecast ES", "Quantile loss x100"], rows,
                f"VaR is the {100 * p:g}% quantile of the return; a violation is a return below the forecast. Estimation window {w} days.")
    ttab = table("Backtests (p-values)", ["Method", "Kupiec LR", "p", "Independence LR", "p", "Conditional coverage LR", "p"], tests,
                 "Kupiec: the violation rate equals the tail probability. Independence: violations do not cluster. Conditional coverage: both together (2 df).")
    lab = labels[w:]
    k = max(1, m // 400)
    ch = [chart("Returns and the value at risk forecasts", lab[::k], [("Return", ro[::k])] + [(nm, q[::k]) for nm, q in series[1:2] + series[3:4]], "", "Return"),
          chart("Violation rate by method (%)", [r_[0] for r_ in rows], [("Violation rate", [r_[2] for r_ in rows]), ("Expected", [100 * p] * len(rows))], "", "% of days", "bar")]
    return result("var_es", "Value at risk and expected shortfall with backtests", m, [stat("Out-of-sample days", m), stat("Tail probability", p, 4), stat("Window", w), stat("GARCH t degrees of freedom", float(nu), 2)],
                  [tab, ttab], ch,
                  ["The GARCH parameters are estimated once on the first window and filtered through the out-of-sample days, so no forecast uses future data.",
                   "Realised / forecast ES is the mean ratio of the loss on a violation day to the expected shortfall forecast: about 1 for a well-calibrated model."])


# ================================================================================================ DCC-GARCH
def _dcc_ll(ab, Z, S):
    a, b = ab
    n, k = Z.shape
    if a < 0 or b < 0 or a + b >= 0.9999:
        return 1e10
    Q = np.empty((n, k, k))
    iu = np.triu_indices(k)
    zz = Z[:, :, None] * Z[:, None, :]
    c = (1 - a - b) * S
    for i, j in zip(*iu):
        drive = c[i, j] + a * np.concatenate([[S[i, j]], zz[:-1, i, j]])
        Q[:, i, j] = lfilter([1.0], [1.0, -b], drive + b * 0)       # the recursion Q_t = c + a z z' + b Q_{t-1}, started at S
        Q[:, j, i] = Q[:, i, j]
    d = 1 / np.sqrt(np.einsum("tii->ti", Q))
    R = Q * d[:, :, None] * d[:, None, :]
    sign, logdet = np.linalg.slogdet(R)
    if np.any(sign <= 0):
        return 1e10
    sol = np.linalg.solve(R, Z[:, :, None])[:, :, 0]
    return 0.5 * float(np.sum(logdet + np.einsum("ti,ti->t", Z, sol) - np.einsum("ti,ti->t", Z, Z)))


def _dcc_path(ab, Z, S):
    a, b = ab
    n, k = Z.shape
    Q = np.empty((n, k, k))
    zz = Z[:, :, None] * Z[:, None, :]
    c = (1 - a - b) * S
    for i in range(k):
        for j in range(i, k):
            drive = c[i, j] + a * np.concatenate([[S[i, j]], zz[:-1, i, j]])
            Q[:, i, j] = lfilter([1.0], [1.0, -b], drive)
            Q[:, j, i] = Q[:, i, j]
    d = 1 / np.sqrt(np.einsum("tii->ti", Q))
    return Q * d[:, :, None] * d[:, None, :]


def dcc(df, spec):
    from statsmodels.stats.diagnostic import acorr_ljungbox
    ys = as_list(need(spec, "y", "return series (two to five)"))
    if not 2 <= len(ys) <= 5:
        raise OESError("Choose two to five return series.")
    d, labels = _series_frame(df, spec, ys)
    R_ = d[ys].to_numpy(float)
    n, k = R_.shape
    if n < 300:
        raise OESError(f"DCC-GARCH needs at least 300 observations; there are {n}.")
    Z = np.empty_like(R_)
    urows, vols = [], []
    for j in range(k):
        g = _garch11_fit(R_[:, j])
        s2 = _garch_path(R_[:, j], g)
        Z[:, j] = (R_[:, j] - g["mu"]) / np.sqrt(s2)
        vols.append(np.sqrt(s2))
        lb = acorr_ljungbox(Z[:, j] ** 2, lags=[10], return_df=True)
        urows.append([ys[j], g["mu"], g["om"], g["al"], g["be"], g["al"] + g["be"], float(np.sqrt(g["om"] / max(1 - g["al"] - g["be"], 1e-9))) if g["al"] + g["be"] < 1 else None, float(lb.lb_pvalue.iloc[0])])
    S = np.cov(Z.T, bias=True)
    S = S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    best = None
    for a0, b0 in [(0.02, 0.95), (0.05, 0.9), (0.1, 0.8), (0.01, 0.9)]:
        o = optimize.minimize(lambda v: _dcc_ll(v, Z, S), [a0, b0], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8, maxiter=600))
        if best is None or o.fun < best.fun:
            best = o
    a, b = best.x
    ll_dcc = -best.fun
    ll_ccc = -_dcc_ll([0.0, 0.0], Z, S)
    lr = 2 * (ll_dcc - ll_ccc)
    # numerical standard errors of (a, b), ignoring the first-stage estimation
    try:
        h = 1e-4
        H = np.zeros((2, 2))
        f0 = _dcc_ll([a, b], Z, S)
        for i in range(2):
            for j in range(2):
                ei = np.eye(2)[i] * h; ej = np.eye(2)[j] * h
                H[i, j] = (_dcc_ll([a, b] + ei + ej, Z, S) - _dcc_ll([a, b] + ei - ej, Z, S) - _dcc_ll([a, b] - ei + ej, Z, S) + _dcc_ll([a, b] - ei - ej, Z, S)) / (4 * h * h)
        se = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))
    except Exception:
        se = np.array([np.nan, np.nan])
    Rt = _dcc_path([a, b], Z, S)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    prow = []
    for i, j in pairs:
        c_ = Rt[:, i, j]
        prow.append([f"{ys[i]} / {ys[j]}", float(S[i, j]), float(c_.mean()), float(c_.min()), float(c_.max()), float(np.percentile(c_, 5)), float(np.percentile(c_, 95))])
    tabs = [table("Step 1: univariate GARCH(1,1) for each series", ["Series", "Mean", "omega", "alpha", "beta", "alpha + beta", "Long-run volatility", "Ljung-Box p (squared residuals, 10 lags)"], urows,
                  "Gaussian quasi-maximum likelihood; the last column tests for remaining ARCH in the standardised residuals."),
            table("Step 2: DCC parameters", ["Parameter", "Estimate", "s.e.", "t"], [["a (news)", a, se[0], a / se[0] if se[0] else None], ["b (persistence)", b, se[1], b / se[1] if se[1] else None], ["a + b", a + b, None, None]],
                  "Standard errors treat the first-stage estimates as given."),
            table("Conditional correlations", ["Pair", "Unconditional", "Mean", "Min", "Max", "5th percentile", "95th percentile"], prow),
            tests_table([test_row("Dynamic against constant correlation (LR)", lr, stats.chi2.sf(lr, 2), "2 df; a = b = 0 is on the boundary, so the p-value is conservative")], "Test of time-varying correlation")]
    kk = max(1, n // 500)
    ch = []
    for i, j in pairs[:3]:
        ch.append(chart(f"Conditional correlation: {ys[i]} and {ys[j]}", labels[::kk], [("DCC", Rt[::kk, i, j]), ("Constant", [S[i, j]] * len(labels[::kk]))], "", "Correlation"))
    return result("dcc", "Dynamic conditional correlation (DCC-GARCH)", n, [stat("Observations", n), stat("DCC a", float(a), 4), stat("DCC b", float(b), 4), stat("LR against constant correlation", float(lr), 2)], tabs, ch,
                  ["Engle (2002): standardised residuals from univariate GARCH(1,1) models drive the correlation recursion Q_t = (1 - a - b) S + a z z' + b Q_{t-1}.",
                   "Correlations that rise in turbulent periods are the reason to use a dynamic model for risk and hedging."])


# ================================================================================================ portfolio optimisation
def _shrink_cov(R, method):
    if method == "ledoit":
        from sklearn.covariance import ledoit_wolf
        return ledoit_wolf(R)[0]
    return np.cov(R.T, bias=False) if R.shape[1] > 1 else np.atleast_2d(R.var(ddof=1))


def _weights(kind, mu, S, long_only, rf=0.0):
    n = len(mu)
    x0 = np.full(n, 1.0 / n)
    cons = [dict(type="eq", fun=lambda w: w.sum() - 1.0)]
    bnds = [(0.0, 1.0)] * n if long_only else [(-1.0, 2.0)] * n
    if kind == "equal":
        return x0
    if kind == "minvar":
        if not long_only:
            w = np.linalg.solve(S, np.ones(n))
            return w / w.sum()
        o = optimize.minimize(lambda w: w @ S @ w, x0, jac=lambda w: 2 * S @ w, bounds=bnds, constraints=cons, method="SLSQP", options=dict(maxiter=500, ftol=1e-14))
        return o.x
    if kind == "maxsharpe":
        ex = mu - rf
        if not long_only:
            w = np.linalg.solve(S, ex)
            return w / w.sum() if abs(w.sum()) > 1e-12 else x0
        o = optimize.minimize(lambda w: -(w @ ex) / np.sqrt(w @ S @ w), x0, bounds=bnds, constraints=cons, method="SLSQP", options=dict(maxiter=500, ftol=1e-14))
        return o.x
    # risk parity: equal risk contributions (long only)
    def obj(w):
        rc = w * (S @ w)
        return np.sum((rc / rc.sum() - 1.0 / n) ** 2)
    o = optimize.minimize(obj, x0, bounds=[(1e-6, 1.0)] * n, constraints=cons, method="SLSQP", options=dict(maxiter=1000, ftol=1e-16))
    return o.x


def portfolio(df, spec):
    assets = as_list(need(spec, "assets", "assets (returns)"))
    if len(assets) < 3:
        raise OESError("Choose at least three assets.")
    d, labels = _series_frame(df, spec, assets)
    R = d[assets].to_numpy(float)
    T, n = R.shape
    ppy = int_option(spec, "periods", 252, 1, 366, "Periods per year")
    window = int_option(spec, "window", 250, 60, 5000, "Estimation window for the out-of-sample test")
    reb = int_option(spec, "rebalance", 21, 1, 250, "Rebalance every")
    boot = int_option(spec, "boot", 300, 0, 3000, "Bootstrap replications")
    seed = int_option(spec, "seed", 0, 0, 10 ** 9, "Random seed")
    long_only = str(spec.get("long_only", "yes")).lower() in ("yes", "true", "1")
    shrink = str(spec.get("shrink", "ledoit"))
    rf = float_option(spec, "rf", 0.0, "Risk-free rate per period")
    if T < window + 3 * reb:
        raise OESError(f"Too few observations ({T}) for a window of {window} and rebalancing every {reb} periods.")
    mu = R.mean(0)
    S = _shrink_cov(R, shrink)
    kinds = [("equal", "Equal weights"), ("minvar", "Minimum variance"), ("maxsharpe", "Maximum Sharpe ratio"), ("riskparity", "Risk parity")]
    W = {k: _weights(k, mu, S, long_only, rf) for k, _ in kinds}
    rows = []
    for k, nm in kinds:
        w = W[k]
        m_, v_ = float(w @ mu), float(np.sqrt(w @ S @ w))
        rc = w * (S @ w) / (w @ S @ w)
        rows.append([nm] + [float(x) for x in w] + [m_ * ppy, v_ * np.sqrt(ppy), (m_ - rf) / v_ * np.sqrt(ppy), float(1 / np.sum(w ** 2)), float(rc.max())])
    wt = table("Portfolio weights and in-sample performance (annualised)", ["Strategy"] + assets + ["Return", "Volatility", "Sharpe ratio", "Effective number of assets", "Largest risk share"], rows,
               f"Covariance: {'Ledoit-Wolf shrinkage' if shrink == 'ledoit' else 'sample'}; {'long only' if long_only else 'short positions allowed'}; {T} periods, {ppy} per year.")
    tabs = [wt]
    rng = np.random.default_rng(seed)
    if boot:
        key = str(spec.get("objective", "maxsharpe"))
        if key not in W:
            key = "maxsharpe"
        BW = np.empty((boot, n))
        for b in range(boot):
            Rb = R[rng.integers(0, T, T)]
            BW[b] = _weights(key, Rb.mean(0), _shrink_cov(Rb, shrink), long_only, rf)
        lo, hi = np.percentile(BW, [2.5, 97.5], axis=0)
        nm = dict(kinds)[key]
        tabs.append(table(f"Sampling uncertainty of the {nm.lower()} weights (bootstrap, {boot} resamples)", ["Asset", "Weight", "2.5%", "97.5%", "Share of resamples with weight < 1%"],
                          [[assets[i], float(W[key][i]), float(lo[i]), float(hi[i]), float(np.mean(BW[:, i] < 0.01))] for i in range(n)],
                          "Resampling the periods with replacement and re-solving the problem each time."))
    # out of sample
    perf = {k: [] for k, _ in kinds}
    turn = {k: [] for k, _ in kinds}
    prev = {k: None for k, _ in kinds}
    t = window
    while t < T:
        te = min(t + reb, T)
        tr = R[t - window:t]
        mu_t, S_t = tr.mean(0), _shrink_cov(tr, shrink)
        for k, _ in kinds:
            w = _weights(k, mu_t, S_t, long_only, rf)
            if prev[k] is not None:
                turn[k].append(float(np.abs(w - prev[k]).sum()))
            prev[k] = w
            perf[k].extend((R[t:te] @ w).tolist())
        t = te
    oos = {k: np.array(v) for k, v in perf.items()}
    m = len(oos["equal"])
    orows = []
    for k, nm in kinds:
        x = oos[k]
        sr = (x.mean() - rf) / x.std(ddof=1)
        # Jobson-Korkie / Memmel test of the Sharpe-ratio difference against equal weights
        y_ = oos["equal"]
        sr0 = (y_.mean() - rf) / y_.std(ddof=1)
        rho = np.corrcoef(x, y_)[0, 1]
        var = (2 - 2 * rho + 0.5 * (sr ** 2 + sr0 ** 2 - sr * sr0 * (1 + rho ** 2))) / m
        z = (sr - sr0) / np.sqrt(var) if var > 0 and k != "equal" else None
        orows.append([nm, float(x.mean() * ppy), float(x.std(ddof=1) * np.sqrt(ppy)), float(sr * np.sqrt(ppy)), float(np.mean(turn[k])) if turn[k] else None, float(x.min()), None if z is None else float(z),
                      None if z is None else float(2 * stats.norm.sf(abs(z)))])
    tabs.append(table(f"Out-of-sample performance ({m} periods, window {window}, rebalancing every {reb})", ["Strategy", "Return", "Volatility", "Sharpe ratio", "Turnover per rebalance", "Worst period", "z vs equal weights", "p"], orows,
                      "Weights are estimated on the previous window only. The z statistic is the Jobson-Korkie test of the Sharpe-ratio difference with Memmel's correction."))
    # frontier
    targets = np.linspace(mu.min(), mu.max(), 25)
    fr = []
    for tg in targets:
        if long_only:
            o = optimize.minimize(lambda w: w @ S @ w, np.full(n, 1 / n), bounds=[(0, 1)] * n,
                                  constraints=[dict(type="eq", fun=lambda w: w.sum() - 1), dict(type="eq", fun=lambda w, tg=tg: w @ mu - tg)], method="SLSQP", options=dict(ftol=1e-14, maxiter=500))
            if o.success:
                fr.append((float(np.sqrt(o.fun) * np.sqrt(ppy)), float(tg * ppy)))
        else:
            A = np.column_stack([np.ones(n), mu])
            Si = np.linalg.inv(S)
            M = A.T @ Si @ A
            lam = np.linalg.solve(M, [1.0, tg])
            w = Si @ A @ lam
            fr.append((float(np.sqrt(w @ S @ w) * np.sqrt(ppy)), float(tg * ppy)))
    fr = sorted(set(fr))
    ch = [chart("Efficient frontier (annualised)", [round(a, 4) for a, _ in fr], [("Frontier", [b for _, b in fr])], "Volatility", "Return", "line"),
          chart("Weights by strategy", assets, [(nm, W[k]) for k, nm in kinds], "Asset", "Weight", "bar")]
    cum = np.cumsum(np.column_stack([oos[k] for k, _ in kinds]), axis=0)
    kk = max(1, m // 400)
    ch.append(chart("Cumulative out-of-sample return", list(range(0, m, kk)), [(nm, cum[::kk, i]) for i, (_, nm) in enumerate(kinds)], "Out-of-sample period", "Cumulative return"))
    return result("portfolio", "Portfolio optimisation and out-of-sample comparison", T, [stat("Periods", T), stat("Assets", n), stat("Max-Sharpe in-sample Sharpe", rows[2][-3], 3), stat("Max-Sharpe out-of-sample Sharpe", orows[2][3], 3)], tabs, ch,
                  ["In-sample optimised portfolios overstate their performance: estimation error in the means is large, and the out-of-sample table is the honest comparison.",
                   "Shrinking the covariance matrix (Ledoit and Wolf 2004) and holding the weights non-negative both reduce the damage from estimation error."])


# ================================================================================================ volatility forecast evaluation
def _dm(l1, l2, lags=10):
    dd = l1 - l2
    n = len(dd)
    dm = dd - dd.mean()
    v = dm @ dm / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (dm[L:] @ dm[:-L]) / n
    stat_ = dd.mean() / np.sqrt(v / n)
    return float(stat_), float(2 * stats.norm.sf(abs(stat_)))


def vol_eval(df, spec):
    y = need(spec, "y", "return series")
    d, labels = _series_frame(df, spec, [y])
    r = d[y].to_numpy(float)
    n = len(r)
    w = int_option(spec, "window", 500, 100, 5000, "Estimation window")
    lam = float_option(spec, "lambda", 0.94, "EWMA decay")
    bench = str(spec.get("benchmark", "EWMA"))
    if n < w + 150:
        raise OESError(f"Too few observations ({n}): a window of {w} leaves {n - w} forecasts; at least 150 are needed.")
    m = n - w
    fc = {}
    g = _garch11_fit(r[:w])
    fc["GARCH(1,1)"] = _garch_path(r, g)[w:]
    e = r - r[:w].mean()
    fc["EWMA"] = lfilter([1 - lam], [1, -lam], np.concatenate([[r[:w].var()], e[:-1] ** 2]))[w:]
    Hs = np.lib.stride_tricks.sliding_window_view(r, w)[:-1][:m]
    fc[f"Rolling {w}-day variance"] = Hs.var(1, ddof=1)
    H25 = np.lib.stride_tricks.sliding_window_view(r, 25)[w - 25:-1][:m]
    fc["Rolling 25-day variance"] = H25.var(1, ddof=1)
    fc["Constant (first window)"] = np.full(m, r[:w].var(ddof=1))
    proxy = (r[w:] - r[:w].mean()) ** 2
    if bench not in fc:
        bench = "EWMA"
    rows, mz, losses = [], [], {}
    for nm, h in fc.items():
        mse = float(np.mean((proxy - h) ** 2))
        qlike = float(np.mean(np.log(h) + proxy / h))
        losses[nm] = (proxy - h) ** 2, np.log(h) + proxy / h
        if np.ptp(h) < 1e-14:
            rows.append([nm, mse * 1e8, qlike, 0.0])
            mz.append([nm, None, None, None, None, None, None])
            continue
        X = np.column_stack([np.ones(m), h])
        b_ = np.linalg.lstsq(X, proxy, rcond=None)[0]
        e_ = proxy - X @ b_
        from .finance import _nw_cov
        V = _nw_cov(X, e_, 10)
        se = np.sqrt(np.diag(V))
        wald = (b_ - [0, 1]) @ np.linalg.solve(V, b_ - [0, 1])
        r2 = 1 - e_.var() / proxy.var()
        rows.append([nm, mse * 1e8, qlike, float(r2)])
        mz.append([nm, float(b_[0]), float(se[0]), float(b_[1]), float(se[1]), float(wald), float(stats.chi2.sf(wald, 2))])
    ltab = table(f"Out-of-sample loss ({m} days, squared return as the volatility proxy)", ["Forecast", "MSE x 1e8", "QLIKE", "R-squared of the Mincer-Zarnowitz regression"], rows,
                 "Lower loss is better. QLIKE is the loss of Patton (2011) and is robust to the noise in the proxy.")
    mtab = table("Mincer-Zarnowitz regressions: squared return on the forecast", ["Forecast", "Intercept", "s.e.", "Slope", "s.e.", "Wald: intercept 0 and slope 1", "p"], mz,
                 "Newey-West standard errors, 10 lags. An unbiased forecast has intercept 0 and slope 1.")
    drows = []
    for nm in fc:
        if nm == bench:
            continue
        a = _dm(losses[nm][1], losses[bench][1])
        b = _dm(losses[nm][0], losses[bench][0])
        drows.append([nm, a[0], a[1], b[0], b[1]])
    dtab = table(f"Diebold-Mariano tests against {bench}", ["Forecast", "QLIKE: DM", "p", "MSE: DM", "p"], drows,
                 "A negative statistic means the forecast has a smaller loss than the benchmark. Newey-West with 10 lags.")
    k = max(1, m // 400)
    ch = [chart("Volatility forecasts (annualised, %)", labels[w::][::k], [(nm, 100 * np.sqrt(252 * h[::k])) for nm, h in list(fc.items())[:3]] + [("Absolute return x sqrt(252)", 100 * np.sqrt(252) * np.abs(r[w:][::k]))], "", "%")]
    best = rows[int(np.argmin([x[2] for x in rows]))][0]
    return result("vol_eval", "Volatility forecast evaluation", m, [stat("Out-of-sample days", m), stat("Best by QLIKE", best), stat("Benchmark", bench)], [ltab, mtab, dtab], ch,
                  ["The squared return is an unbiased but very noisy proxy for the variance; with it, the MSE loss is dominated by a few extreme days, and QLIKE is the more reliable ranking.",
                   "The GARCH parameters are estimated once on the first window and filtered forward, so all forecasts use only past data."])


# ================================================================================================ explosive bubble tests (SADF and GSADF)
def _adf_grid(y, r0):
    """ADF t statistic (lag 0, constant) for every window [i, j] with j - i >= r0, by cumulative sums. Returns an array T x T (nan elsewhere)."""
    T = len(y)
    dy = y[1:] - y[:-1]
    x = y[:-1]
    cx = np.concatenate([[0], np.cumsum(x)]); cy = np.concatenate([[0], np.cumsum(dy)])
    cxx = np.concatenate([[0], np.cumsum(x * x)]); cxy = np.concatenate([[0], np.cumsum(x * dy)]); cyy = np.concatenate([[0], np.cumsum(dy * dy)])
    out = np.full((T, T), np.nan)
    for i in range(T - r0):
        j = np.arange(i + r0, T)                       # window uses observations i..j, i.e. regression rows i..j-1
        k = j - i
        sx = cx[j] - cx[i]; sy = cy[j] - cy[i]; sxx = cxx[j] - cxx[i]; sxy = cxy[j] - cxy[i]; syy = cyy[j] - cyy[i]
        Sxx = sxx - sx * sx / k
        Sxy = sxy - sx * sy / k
        Syy = syy - sy * sy / k
        rho = Sxy / Sxx
        rss = Syy - rho * Sxy
        s2 = rss / (k - 2)
        out[i, j] = rho / np.sqrt(s2 / Sxx)
    return out


def _bsadf_paths(y, r0):
    G = _adf_grid(y, r0)
    bs = np.nanmax(G, axis=0)                          # sup over the start for each end point
    return bs


def bubble(df, spec):
    yc = need(spec, "y", "price (or price ratio) series")
    d, labels = _series_frame(df, spec, [yc])
    y0 = d[yc].to_numpy(float)
    if np.any(y0 <= 0) and str(spec.get("log", "no")) == "yes":
        raise OESError("The series has values at or below zero; choose log = no or another series.")
    y = np.log(y0) if str(spec.get("log", "no")) == "yes" else y0
    T = len(y)
    reps = int_option(spec, "reps", 200, 20, 2000, "Monte Carlo replications")
    seed = int_option(spec, "seed", 0, 0, 10 ** 9, "Random seed")
    r0 = int(np.floor(T * (0.01 + 1.8 / np.sqrt(T))))
    r0 = max(int_option(spec, "min_window", r0, 10, T // 2, "Minimum window"), 10)
    if T < 60:
        raise OESError(f"Too few observations ({T}) for a bubble test: at least 60 are needed.")
    G = _adf_grid(y, r0)
    sadf_path = G[0, :]                                       # forward expanding window from the start
    sadf = float(np.nanmax(sadf_path))
    bs = np.nanmax(G, axis=0)
    gsadf = float(np.nanmax(bs))
    # critical values under a random walk with a small drift (Phillips, Shi and Yu 2015: T^(-1) drift)
    rng = np.random.default_rng(seed)
    sims_sadf = np.empty(reps); sims_g = np.empty(reps); paths = np.empty((reps, T))
    for b in range(reps):
        e = rng.normal(size=T)
        w = np.cumsum(e) + np.arange(T) / (T ** 1.0)
        Gb = _adf_grid(w, r0)
        sims_sadf[b] = np.nanmax(Gb[0, :])
        pb = np.nanmax(Gb, axis=0)
        paths[b] = pb
        sims_g[b] = np.nanmax(pb)
    cv = lambda a, p: float(np.percentile(a, 100 * p))
    tests = [["SADF (right-tailed ADF, expanding window)", sadf, cv(sims_sadf, .9), cv(sims_sadf, .95), cv(sims_sadf, .99), float(np.mean(sims_sadf >= sadf))],
             ["GSADF (generalised: flexible start and end)", gsadf, cv(sims_g, .9), cv(sims_g, .95), cv(sims_g, .99), float(np.mean(sims_g >= gsadf))]]
    ttab = table("Tests for explosive behaviour", ["Test", "Statistic", "10% critical value", "5% critical value", "1% critical value", "Monte Carlo p-value"], tests,
                 f"Critical values from {reps} random walks of length {T} with a vanishing drift; minimum window {r0} periods; ADF regression without lags.")
    cv_path = np.percentile(paths, 95, axis=0)
    flag = np.where(np.isfinite(bs) & np.isfinite(cv_path) & (bs > cv_path))[0]
    eps = []
    if len(flag):
        start = flag[0]; prev = flag[0]
        for i in flag[1:]:
            if i - prev > 1:
                eps.append((start, prev)); start = i
            prev = i
        eps.append((start, prev))
    minlen = max(int(np.log(T)), 2)
    eps = [(a, b_) for a, b_ in eps if b_ - a + 1 >= minlen]
    erows = [[labels[a], labels[b_], int(b_ - a + 1), float(np.nanmax(bs[a:b_ + 1]))] for a, b_ in eps]
    etab = table("Date-stamped episodes of explosive behaviour (BSADF above its 95% critical value)", ["Start", "End", "Periods", "Peak BSADF"], erows or [["none", "", 0, None]],
                 f"Episodes shorter than log(T) = {minlen} periods are discarded (Phillips, Shi and Yu 2015).")
    k = max(1, T // 500)
    xs = np.arange(T)[~np.isnan(bs)][::k]
    ch = [chart("BSADF statistic and its 95% critical value", [labels[i] for i in xs], [("BSADF", bs[xs]), ("95% critical value", cv_path[xs])], "", "ADF t statistic"),
          chart("The series", labels[::k], [(yc, y[::k])], "", yc)]
    return result("bubble", "Explosive bubble tests (SADF and GSADF) and date-stamping", T,
                  [stat("Observations", T), stat("SADF", sadf, 3), stat("GSADF", gsadf, 3), stat("GSADF 95% critical value", cv(sims_g, .95), 3), stat("Episodes", len(eps))], [ttab, etab], ch,
                  ["A statistic above the critical value rejects a unit root in favour of a mildly explosive root: prices that grow faster than a random walk. The test does not say whether the cause is a bubble or a change in fundamentals; use a ratio such as price to dividends to hold the fundamentals fixed.",
                   "The critical values depend on the sample size and the minimum window, so they are simulated for each run."])


# ================================================================================================ Nelson-Siegel yield curve
def _ns_loadings(tau, lam):
    x = lam * tau
    l1 = (1 - np.exp(-x)) / x
    return np.column_stack([np.ones_like(tau), l1, l1 - np.exp(-x)])


def nelson_siegel(df, spec):
    ycols = as_list(need(spec, "yields", "yield columns (one per maturity)"))
    mats = need(spec, "maturities", "maturities in months, one for each yield column (comma-separated)")
    mats = [x for x in str(mats).replace("[", "").replace("]", "").replace(";", ",").split(",") if x.strip()] if not isinstance(mats, (list, tuple)) else mats
    try:
        tau = np.array([float(m_) for m_ in mats])
    except ValueError:
        raise OESError("The maturities must be numbers (months), for example 3, 12, 60.")
    if len(tau) != len(ycols):
        raise OESError(f"There are {len(ycols)} yield columns but {len(tau)} maturities.")
    if len(ycols) < 4:
        raise OESError("Use at least four maturities.")
    order = np.argsort(tau)
    ycols = [ycols[i] for i in order]; tau = tau[order]
    d, labels = _series_frame(df, spec, ycols)
    Y = d[ycols].to_numpy(float)
    T = len(Y)
    if T < 60:
        raise OESError(f"Too few dates ({T}): at least 60 are needed.")
    lam_in = spec.get("lambda")
    if lam_in in (None, "", "estimate"):
        grid = np.linspace(0.005, 0.30, 60)
        sse = []
        for lam in grid:
            L = _ns_loadings(tau, lam)
            B = np.linalg.lstsq(L, Y.T, rcond=None)[0]
            sse.append(np.sum((Y.T - L @ B) ** 2))
        lam = float(grid[int(np.argmin(sse))])
        o = optimize.minimize_scalar(lambda l_: np.sum((Y.T - _ns_loadings(tau, l_) @ np.linalg.lstsq(_ns_loadings(tau, l_), Y.T, rcond=None)[0]) ** 2), bounds=(max(lam - 0.01, 0.002), lam + 0.01), method="bounded")
        lam = float(o.x)
    else:
        lam = float(lam_in)
        if lam <= 0:
            raise OESError("lambda must be positive.")
    L = _ns_loadings(tau, lam)
    B = np.linalg.lstsq(L, Y.T, rcond=None)[0].T                 # T x 3: level, slope (negative), curvature
    fit = B @ L.T
    res = Y - fit
    names = ["Level (beta 0)", "Slope (beta 1)", "Curvature (beta 2)"]
    rmse = np.sqrt((res ** 2).mean(0))
    maxlam = tau[int(np.argmax(_ns_loadings(tau, lam)[:, 2]))]
    frows = []
    for j in range(3):
        b = B[:, j]
        ac = [float(np.corrcoef(b[h:], b[:-h])[0, 1]) for h in (1, 12, 24)] if T > 40 else [None] * 3
        frows.append([names[j], float(b.mean()), float(b.std(ddof=1)), float(b.min()), float(b.max())] + ac)
    ftab = table("Factors (estimated date by date)", ["Factor", "Mean", "S.d.", "Min", "Max", "Autocorrelation lag 1", "lag 12", "lag 24"], frows,
                 f"lambda = {lam:.4f} per month: the curvature loading peaks at about {1.79 / lam:.0f} months.")
    ptab = table("Fit by maturity", ["Maturity (months)", "Mean yield", "RMSE of the fit", "Mean error"], [[float(tau[i]), float(Y[:, i].mean()), float(rmse[i]), float(res[:, i].mean())] for i in range(len(tau))],
                 "Errors in the same units as the yields (percentage points if the yields are in percent).")
    # forecasting: AR(1) on the factors against the random walk (Diebold and Li 2006)
    h = int_option(spec, "horizon", 12, 1, 36, "Forecast horizon (months)")
    start = max(int(T * 0.6), 60)
    errs = {"Random walk": [], "AR(1) factors": [], "AR(1) on the yield": []}
    for t in range(start, T - h):
        Bt = B[:t + 1]
        fb = np.empty(3)
        for j in range(3):
            x_, z = Bt[:-1, j], Bt[1:, j]
            X = np.column_stack([np.ones(len(x_)), x_])
            c = np.linalg.lstsq(X, z, rcond=None)[0]
            v = Bt[-1, j]
            for _ in range(h):
                v = c[0] + c[1] * v
            fb[j] = v
        yf = L @ fb
        errs["AR(1) factors"].append(Y[t + h] - yf)
        errs["Random walk"].append(Y[t + h] - Y[t])
        ar = []
        for i in range(len(tau)):
            x_, z = Y[:t, i], Y[1:t + 1, i]
            X = np.column_stack([np.ones(len(x_)), x_])
            c = np.linalg.lstsq(X, z, rcond=None)[0]
            v = Y[t, i]
            for _ in range(h):
                v = c[0] + c[1] * v
            ar.append(v)
        errs["AR(1) on the yield"].append(Y[t + h] - np.array(ar))
    sel_i = [0, len(tau) // 2, len(tau) - 1]
    frc = []
    for k_, e_ in errs.items():
        E = np.array(e_)
        frc.append([k_] + [float(np.sqrt((E[:, i] ** 2).mean())) for i in sel_i])
    ftab2 = table(f"{h}-month-ahead forecast RMSE (out of sample, {T - h - start} forecasts)", ["Forecast"] + [f"{int(tau[i])}-month yield" for i in sel_i], frc,
                  "Expanding-window estimation; the factors are re-estimated at every date with lambda fixed.")
    k = max(1, T // 400)
    ch = [chart("Factors over time", labels[::k], [(names[j], B[::k, j]) for j in range(3)], "", "Factor"),
          chart("Average and last yield curve with fit", [f"{int(t_)}" for t_ in tau], [("Mean yield", Y.mean(0)), ("Mean fitted", fit.mean(0)), ("Last observed", Y[-1]), ("Last fitted", fit[-1])], "Maturity (months)", "Yield", "line")]
    return result("nelson_siegel", "Nelson-Siegel yield curve: level, slope and curvature", T,
                  [stat("Dates", T), stat("Maturities", len(tau)), stat("lambda", lam, 4), stat("Mean RMSE of the fit", float(rmse.mean()), 4)], [ftab, ptab, ftab2], ch,
                  ["Nelson and Siegel (1987): the curve is level + slope x (1 - e^(-lambda tau))/(lambda tau) + curvature x [(1 - e^(-lambda tau))/(lambda tau) - e^(-lambda tau)]. The factors are estimated by least squares at every date.",
                   "The slope factor beta 1 is the short rate minus the long rate with a negative sign: a positive beta 1 is an inverted curve."])
