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
