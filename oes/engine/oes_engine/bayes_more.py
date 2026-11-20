"""More Bayesian models: Bayesian model averaging (g-prior, exact enumeration or MC3). Same result format as the other models (core.py)."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

from .core import (OESError, as_list, check_columns, chart, distinct, f, int_option, float_option, need, result, stat, table, numeric)


# ================================================================================================ Bayesian model averaging
def _g_value(kind, n, k):
    if kind == "uip":
        return float(n)
    if kind == "ric":
        return float(k * k)
    return float(max(n, k * k))                               # BRIC (Fernandez, Ley and Steel 2001)


def _log_prior(kind, p, k, m):
    p = np.asarray(p, float)
    if kind == "uniform":
        return np.zeros_like(p)
    if kind == "binomial":
        th = m / k
        return p * np.log(th) + (k - p) * np.log(1 - th)
    return gammaln(p + 1) + gammaln(k - p + 1) - gammaln(k + 2)      # random theta: beta-binomial(1, 1)


def _log_bf(R2, p, n, g):
    """log Bayes factor of a model against the null (Liang et al. 2008): g-prior on the slopes, flat on the constant and the variance."""
    return (n - 1 - p) / 2 * np.log1p(g) - (n - 1) / 2 * np.log1p(g * (1 - R2))


def bma(df, spec):
    y = need(spec, "y", "dependent variable")
    x = as_list(need(spec, "x", "candidate regressors"))
    distinct([y], x)
    d = numeric(df, [y] + x)
    Y = d[y].to_numpy(float)
    X = d[x].to_numpy(float)
    n, k = X.shape
    if k < 2:
        raise OESError("Choose at least two candidate regressors; with one there is nothing to average over.")
    if k > 30:
        raise OESError("At most 30 candidate regressors are supported.")
    if n < k + 5:
        raise OESError(f"Too few observations ({n}) for {k} candidate regressors.")
    if (X.std(0) == 0).any():
        raise OESError("A candidate regressor is constant: drop it (the constant is always in the model).")
    gk = spec.get("g", "bric")
    if gk not in ("bric", "uip", "ric"):
        raise OESError("Choose the g-prior: bric, uip or ric.")
    mp = spec.get("model_prior", "random")
    if mp not in ("random", "uniform", "binomial"):
        raise OESError("Choose the model prior: random, uniform or binomial.")
    msize = float_option(spec, "expected_size", k / 2, "Expected model size")
    if not 0 < msize < k:
        raise OESError(f"The expected model size must be between 0 and {k}.")
    ntop = int_option(spec, "top", 10, 3, 30, "Number of top models")
    g = _g_value(gk, n, k)
    Xc, yc = X - X.mean(0), Y - Y.mean()
    C, cv, sst = Xc.T @ Xc, Xc.T @ yc, float(yc @ yc)
    method = spec.get("method", "auto")
    if method not in ("auto", "exact", "mc3"):
        raise OESError("Choose the method: auto, exact or mc3.")
    if method == "exact" and k > 18:
        raise OESError("Exact enumeration is limited to 18 regressors; use MC3.")
    exact = method == "exact" or (method == "auto" and k <= 14)
    ndraw = int_option(spec, "draws", 20000, 1000, 500000, "MC3 iterations")
    burn = int_option(spec, "burn", 5000, 0, 500000, "MC3 burn-in")
    rng = np.random.default_rng(int_option(spec, "seed", 0, 0, 10 ** 9, "Random seed"))
    cache = {}

    def fit(mask):
        if mask in cache:
            return cache[mask]
        idx = [j for j in range(k) if mask >> j & 1]
        if not idx:
            out = (0.0, 0, None, None)
        else:
            Cs = C[np.ix_(idx, idx)]
            try:
                inv = np.linalg.inv(Cs)
            except np.linalg.LinAlgError:
                raise OESError("The candidate regressors are perfectly collinear; remove the redundant ones.")
            b = inv @ cv[idx]
            out = (min(float(b @ cv[idx]) / sst, 1 - 1e-12), len(idx), b, np.diag(inv).copy())
        cache[mask] = out
        return out
    if exact:
        masks = np.arange(2 ** k)
        for m in masks:
            fit(int(m))
        R2 = np.array([cache[int(m)][0] for m in masks]); P = np.array([cache[int(m)][1] for m in masks])
        w_all = None
    else:
        cur, lcur = 0, 0.0
        counts = {}
        lp_of = lambda m: _log_bf(fit(m)[0], fit(m)[1], n, g) + _log_prior(mp, [fit(m)[1]], k, msize)[0]
        lcur = lp_of(cur)
        acc = 0
        for it in range(burn + ndraw):
            j = int(rng.integers(k))
            if rng.random() < 0.5 or bin(cur).count("1") in (0, k):
                prop = cur ^ (1 << j)
            else:                                          # swap an included with an excluded regressor
                inc = [i for i in range(k) if cur >> i & 1]; exc = [i for i in range(k) if not cur >> i & 1]
                prop = cur ^ (1 << int(rng.choice(inc))) ^ (1 << int(rng.choice(exc)))
            lprop = lp_of(prop)
            if np.log(rng.random()) < lprop - lcur:
                cur, lcur = prop, lprop; acc += it >= burn
            if it >= burn:
                counts[cur] = counts.get(cur, 0) + 1
        masks = np.array(sorted(counts), dtype=np.int64)
        R2 = np.array([cache[int(m)][0] for m in masks]); P = np.array([cache[int(m)][1] for m in masks])
        w_all = np.array([counts[int(m)] for m in masks], float); w_all /= w_all.sum()
    inc_mat = ((masks[:, None] >> np.arange(k)[None, :]) & 1).astype(bool)

    def weights(gv, prior, size=None):
        if w_all is not None:
            return w_all
        lw = _log_bf(R2, P, n, gv) + _log_prior(prior, P, k, msize if size is None else size)
        lw -= lw.max(); w = np.exp(lw)
        return w / w.sum()
    w = weights(g, mp)
    pip = w @ inc_mat
    # coefficients: model-conditional Student-t posteriors (n - 1 degrees of freedom), averaged over models
    mu = np.zeros((len(masks), k)); var = np.zeros((len(masks), k)); sc = np.ones((len(masks), k))
    sh = g / (1 + g)
    for i, m in enumerate(masks):
        r2, p, b, dg = cache[int(m)]
        if p == 0:
            continue
        idx = [j for j in range(k) if int(m) >> j & 1]
        s2 = sst * (1 - sh * r2) / (n - 3) * sh
        mu[i, idx] = sh * b; var[i, idx] = s2 * dg
        sc[i, idx] = np.sqrt(s2 * dg * (n - 3) / (n - 1))
    bm = w @ mu
    bv = w @ (var + mu ** 2) - bm ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        cm = np.where(pip > 1e-12, bm / pip, 0.0)
        cvv = np.where(pip > 1e-12, (w @ ((var + mu ** 2) * inc_mat)) / pip - cm ** 2, 0.0)
        pos = (w[:, None] * inc_mat * stats.t.sf(-mu / sc, n - 1)).sum(0) / pip
    sign_p = np.where(cm >= 0, pos, 1 - pos)
    order = np.argsort(-pip)
    rows = [[x[j], f(pip[j], 3), f(bm[j]), f(np.sqrt(max(bv[j], 0))), f(cm[j]), f(np.sqrt(max(cvv[j], 0))), f(sign_p[j], 3)] for j in order]
    tabs = [table("Posterior inclusion probabilities and coefficients (averaged over models)",
                  ["Variable", "PIP", "Post. mean", "Post. s.d.", "Mean if included", "S.d. if included", "P(sign | incl.)"], rows,
                  "PIP = posterior probability that the variable is in the model. Post. mean averages over all models (zero when excluded); P(sign | incl.) is the probability that the coefficient has the sign of its mean, given inclusion.")]
    top = np.argsort(-w)[:ntop]
    trows = []
    for r_, i in enumerate(top, 1):
        names = [x[j] for j in range(k) if inc_mat[i, j]]
        trows.append([r_, f(w[i], 4), int(P[i]), f(R2[i], 3), ", ".join(names) if names else "(no regressors)"])
    tabs.append(table(f"Best {len(top)} models", ["Rank", "Posterior prob.", "Size", "R-squared", "Variables"], trows,
                      "In MC3 the probabilities are visit frequencies of the chain." if not exact else "Exact posterior probabilities from all 2^k models."))
    ps = np.array([w[P == s].sum() for s in range(k + 1)])
    sizes = np.arange(k + 1)
    pr_s = np.exp(gammaln(k + 1) - gammaln(sizes + 1) - gammaln(k - sizes + 1) + _log_prior(mp, sizes, k, msize))
    pr_s /= pr_s.sum()
    tabs.append(table("Posterior distribution of model size", ["Size", "Posterior prob.", "Prior prob."],
                      [[int(s_), f(ps[s_], 4), f(pr_s[s_], 4)] for s_ in sizes]))
    esize = float(w @ P)
    if exact:
        sens = []
        for gname in ("uip", "bric", "ric"):
            for pr in ("uniform", "binomial", "random"):
                wv = weights(_g_value(gname, n, k), pr, k / 4)
                sens.append([gname.upper(), pr if pr != "binomial" else f"binomial (size {k / 4:g})"] + [f(v, 3) for v in (wv @ inc_mat)[order[:6]]] + [f(float(wv @ P), 2)])
        tabs.append(table("Prior sensitivity: PIPs of the six variables with the highest PIP",
                          ["g-prior", "Model prior"] + [x[j] for j in order[:6]] + ["Expected size"], sens,
                          "Each row re-computes the posterior over all models with a different prior."))
    charts = [chart("Posterior inclusion probabilities", [x[j] for j in order], [("PIP", pip[order])], "", "probability", kind="bar")]
    summary = [stat("Observations", n, 0), stat("Candidate regressors", k, 0), stat("Models " + ("evaluated" if exact else "visited"), len(masks), 0),
               stat("Expected model size", esize, 2), stat("Posterior probability of the best model", w[top[0]], 4),
               stat("Variables with PIP above 0.5", int((pip > 0.5).sum()), 0)]
    notes = [f"g-prior with g = {g:g} ({gk.upper()}); model prior: {mp}" + (f" with expected size {msize:g}" if mp == "binomial" else " (the inclusion probability has a uniform prior)") + ".",
             "Marginal likelihoods are exact under the g-prior (Zellner 1986; Liang et al. 2008)." + ("" if exact else f" With {k} regressors the 2^{k} models are sampled by MC3 ({ndraw} iterations after {burn} burn-in; acceptance {acc / ndraw:.2f}).")]
    return result("bma", "Bayesian model averaging", n, summary, tabs, charts, notes)


# ================================================================================================ Bayesian logit and probit
def _binary_setup(df, spec):
    y = need(spec, "y", "binary outcome")
    x = as_list(need(spec, "x", "explanatory variables"))
    distinct([y], x)
    d = numeric(df, [y] + x)
    Y = d[y].to_numpy(float)
    if not set(np.unique(Y)) <= {0.0, 1.0} or len(np.unique(Y)) < 2:
        raise OESError(f"{y} must be a 0/1 variable with both values present.")
    X = d[x].to_numpy(float)
    if (X.std(0) == 0).any():
        raise OESError("An explanatory variable is constant: drop it (the model has a constant).")
    if len(d) < 5 * (len(x) + 1):
        raise OESError(f"Too few observations ({len(d)}) for {len(x)} explanatory variables.")
    return y, x, Y, X


def _sample_binary(Y, Xc, link, s_slope, s_const, ndraw, burn, rng, init=None):
    """Posterior draws of (alpha_c, beta) for centred regressors. Probit: Albert-Chib Gibbs sampler. Logit: independence
    Metropolis-Hastings with a Student-t proposal centred at the posterior mode (Laplace)."""
    from scipy.special import ndtr, ndtri, expit
    n, k = Xc.shape
    Z = np.column_stack([np.ones(n), Xc])
    V0i = np.diag([1 / s_const ** 2] + list(1 / s_slope ** 2))
    out = np.empty((ndraw, k + 1))
    if link == "probit":
        P = V0i + Z.T @ Z
        L = np.linalg.cholesky(P); Pinv = np.linalg.inv(P)
        b = np.zeros(k + 1) if init is None else init.copy()
        for it in range(burn + ndraw):
            mu = Z @ b
            lo = ndtr(-mu); u = rng.random(n)
            q = np.where(Y == 1, lo + u * (1 - lo), u * lo)
            zz = mu + ndtri(np.clip(q, 1e-300, 1 - 1e-16))
            m = Pinv @ (Z.T @ zz)
            b = m + np.linalg.solve(L.T, rng.standard_normal(k + 1))
            if it >= burn:
                out[it - burn] = b
        return out, 1.0
    def lpost(b):
        e = Z @ b
        return float(Y @ e - np.logaddexp(0, e).sum() - 0.5 * b @ V0i @ b)
    b = np.zeros(k + 1)
    for _ in range(50):                                      # Newton iterations to the posterior mode
        p = expit(Z @ b); g = Z.T @ (Y - p) - V0i @ b; H = Z.T @ (Z * (p * (1 - p))[:, None]) + V0i
        step = np.linalg.solve(H, g); b = b + step
        if np.abs(step).max() < 1e-9:
            break
    p = expit(Z @ b); H = Z.T @ (Z * (p * (1 - p))[:, None]) + V0i
    Sig = np.linalg.inv(H); Lc = np.linalg.cholesky(Sig); nu = 6.0
    def lprop(b_):
        r = np.linalg.solve(Lc, b_ - b)
        return -(nu + k + 1) / 2 * np.log1p(r @ r / nu)
    cur = b.copy() if init is None else init.copy(); lcur = lpost(cur) - lprop(cur); acc = 0
    for it in range(burn + ndraw):
        w = rng.chisquare(nu) / nu
        prop = b + Lc @ rng.standard_normal(k + 1) / np.sqrt(w)
        lp = lpost(prop) - lprop(prop)
        if np.log(rng.random()) < lp - lcur:
            cur, lcur = prop, lp; acc += it >= burn
        if it >= burn:
            out[it - burn] = cur
    return out, acc / ndraw


def bayes_binary(df, spec):
    from scipy.special import ndtr, expit
    from scipy.stats import rankdata
    import statsmodels.api as sm
    from .bayes_extra import _rhat_ess
    from .bayes import _geweke
    y, x, Y, X = _binary_setup(df, spec)
    n, k = X.shape
    link = spec.get("link", "logit")
    if link not in ("logit", "probit"):
        raise OESError("Choose the link: logit or probit.")
    s = float_option(spec, "prior_sd", 2.5, "Prior standard deviation")
    if s <= 0:
        raise OESError("The prior standard deviation must be positive.")
    ndraw = int_option(spec, "draws", 2000, 200, 50000, "Draws per chain")
    burn = int_option(spec, "burn", 1000, 0, 50000, "Burn-in")
    nch = int_option(spec, "chains", 2, 2, 8, "Chains")
    seed = int_option(spec, "seed", 0, 0, 10 ** 9, "Random seed")
    xm, xs = X.mean(0), X.std(0)
    Xc = X - xm
    s_slope, s_const = s / xs, 10.0                          # the prior is N(0, s^2) per one-standard-deviation change in x
    F = ndtr if link == "probit" else expit
    chains, accs = [], []
    for c in range(nch):
        rng = np.random.default_rng(seed + 1000 * c)
        init = rng.normal(scale=0.5, size=k + 1) / np.r_[1.0, xs] if c else None
        dr, a = _sample_binary(Y, Xc, link, s_slope, s_const, ndraw, burn, rng, init)
        chains.append(dr); accs.append(a)
    allc = np.vstack(chains)
    draws = allc.copy(); draws[:, 0] = allc[:, 0] - allc[:, 1:] @ xm     # back to the original (uncentred) constant
    names = ["const"] + x
    from .bayes import post_table
    tabs = [post_table(f"Posterior of the coefficients ({link}, normal prior with s.d. {s:g} per one-s.d. change in x)", names, draws,
                       f"{nch} chains of {ndraw} draws after {burn} burn-in.")]
    ml = (sm.Logit if link == "logit" else sm.Probit)(Y, sm.add_constant(X)).fit(disp=0)
    pm = draws.mean(0)
    tabs.append(table("Maximum likelihood for comparison", ["Parameter", "Coef.", "Std. err.", "Posterior mean", "Posterior s.d."],
                      [[v, f(ml.params[j]), f(ml.bse[j]), f(pm[j]), f(draws[:, j].std(ddof=1))] for j, v in enumerate(names)]))
    if link == "logit":
        OR = np.exp(draws[:, 1:])
        q = np.percentile(OR, [2.5, 50, 97.5], axis=0)
        tabs.append(table("Posterior of the odds ratios (per unit change)", ["Variable", "Mean", "2.5%", "Median", "97.5%", "P(> 1)"],
                          [[v, f(OR[:, j].mean()), f(q[0, j]), f(q[1, j]), f(q[2, j]), f((OR[:, j] > 1).mean(), 3)] for j, v in enumerate(x)]))
    sub = draws[np.linspace(0, len(draws) - 1, min(400, len(draws))).astype(int)]
    isbin = [set(np.unique(X[:, j])) <= {0.0, 1.0} for j in range(k)]
    AME = np.empty((len(sub), k))
    for i, b in enumerate(sub):
        eta = b[0] + X @ b[1:]
        for j in range(k):
            if isbin[j]:
                e1, e0 = eta + (1 - X[:, j]) * b[j + 1], eta - X[:, j] * b[j + 1]
                AME[i, j] = (F(e1) - F(e0)).mean()
            else:
                dens = F(eta) * (1 - F(eta)) if link == "logit" else np.exp(-eta ** 2 / 2) / np.sqrt(2 * np.pi)
                AME[i, j] = dens.mean() * b[j + 1]
    qa = np.percentile(AME, [2.5, 97.5], axis=0)
    tabs.append(table("Posterior of the average marginal effects (change in the probability)", ["Variable", "Mean", "Std. dev.", "2.5%", "97.5%", "P(> 0)", "Type"],
                      [[v, f(AME[:, j].mean()), f(AME[:, j].std(ddof=1)), f(qa[0, j]), f(qa[1, j]), f((AME[:, j] > 0).mean(), 3), "0 to 1" if isbin[j] else "per unit"] for j, v in enumerate(x)],
                      "Computed draw by draw; a dummy variable's effect is the change from 0 to 1."))
    # R-hat and ESS are computed in the original parametrisation (the constant is not centred)
    chains_o = [np.column_stack([c[:, 0] - c[:, 1:] @ xm, c[:, 1:]]) for c in chains]
    rows = []
    for j, v in enumerate(names):
        rh, ess = _rhat_ess([c[:, j] for c in chains_o]); z = _geweke(chains_o[0][:, j])
        rows.append([v, f(rh, 3), f(ess, 0), f(z, 2)])
    tabs.append(table("MCMC convergence diagnostics", ["Parameter", "R-hat", "Effective sample size", "Geweke z (chain 1)"], rows,
                      "R-hat above 1.05 or an effective sample size below a few hundred means the chains have not mixed. " + (
                          f"Metropolis-Hastings acceptance rate {np.mean(accs):.2f}." if link == "logit" else "The probit Gibbs sampler draws from exact conditional distributions.")))
    pr = F(sub[:, [0]] + sub[:, 1:] @ X.T)
    ph = pr.mean(0)
    base = max(Y.mean(), 1 - Y.mean())
    acc50 = ((ph > 0.5) == (Y == 1)).mean()
    r_ = rankdata(ph); n1 = Y.sum(); auc = (r_[Y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (n - n1))
    brier = float(((ph - Y) ** 2).mean())
    ppc = (pr > np.random.default_rng(seed).random(pr.shape)).mean(1)
    tabs.append(table("Fit and posterior predictive check", ["Measure", "Value"],
                      [["Share of ones in the data", f(Y.mean(), 3)], ["Posterior predictive share of ones: mean", f(ppc.mean(), 3)],
                       ["Posterior predictive share of ones: 2.5% to 97.5%", f"{np.percentile(ppc, 2.5):.3f} to {np.percentile(ppc, 97.5):.3f}"],
                       ["Accuracy at 0.5", f(acc50, 3)], ["Accuracy of always predicting the majority", f(base, 3)], ["Area under the ROC curve", f(auc, 3)], ["Brier score", f(brier, 4)]],
                      "In-sample measures at the posterior mean probabilities."))
    sens = []
    for sd_ in (0.5, 1.0, 2.5, 10.0, 100.0):
        rg = np.random.default_rng(seed + 7)
        dr, _ = _sample_binary(Y, Xc, link, np.full(k, sd_) / xs, s_const, 800, 300, rg)
        sens.append([f"s.d. {sd_:g}"] + [f(v) for v in dr[:, 1:].mean(0)])
    sens.append(["Maximum likelihood"] + [f(v) for v in ml.params[1:]])
    tabs.append(table("Prior sensitivity: posterior means of the slopes", ["Prior"] + x, sens,
                      "Each row is a separate run of 800 draws. A prior that is tight (s.d. 0.5 per one-s.d. change in x) pulls the slopes towards zero; the ML row is the limit of a very diffuse prior."))
    j0 = int(np.argmax(np.abs(pm[1:] / draws[:, 1:].std(0)))) + 1
    ds = max(1, len(allc) // 500)
    charts = [chart(f"Trace of {names[j0]} (all chains)", list(range(1, len(draws) + 1, ds)), [(names[j0], draws[::ds, j0])], "draw", names[j0])]
    hist, edges = np.histogram(draws[:, j0], bins=30)
    charts.append(chart(f"Posterior density of {names[j0]}", [f((a + b) / 2, 4) for a, b in zip(edges[:-1], edges[1:])], [("draws", hist)], names[j0], "count", kind="bar"))
    summary = [stat("Observations", n), stat("Link", link), stat("Share of ones", Y.mean(), 3), stat("Chains x draws", f"{nch} x {ndraw}"),
               stat("Largest R-hat", max(float(r_[1]) for r_ in rows), 3), stat("Smallest effective sample size", min(float(r_[2]) for r_ in rows), 0),
               stat("Area under the ROC curve", auc, 3)]
    return result("bayes_binary", f"Bayesian {link}: {y}", n, summary, tabs, charts,
                  [f"Independent N(0, s^2) prior on each slope per one-standard-deviation change in x with s = {s:g} (Gelman et al. 2008); N(0, 10^2) on the centred constant."])


# ================================================================================================ Bayesian shrinkage regression
def _ig(rng, shape, scale):
    return scale / rng.gamma(shape, size=np.shape(scale) or None)


def _shrink_chain(Xs, yc, prior, ndraw, burn, rng, tau_fixed, lam_fixed, init_scale=1.0):
    """Gibbs sampler for y = X b + e on standardised regressors. prior: horseshoe (Makalic and Schmidt 2016), lasso
    (Park and Casella 2008) or ridge (normal prior with a half-Cauchy global scale)."""
    n, p = Xs.shape
    XtX, Xty = Xs.T @ Xs, Xs.T @ yc
    s2 = float(yc @ yc / n) * 0.5
    lam2 = np.ones(p) * init_scale; nu = np.ones(p); tau2 = 1.0 if tau_fixed is None else tau_fixed ** 2; xi = 1.0
    tau_j2 = np.ones(p)
    lam_l = 1.0 if lam_fixed is None else lam_fixed
    B, S2, T2, KP, LAM = (np.empty((ndraw, p)), np.empty(ndraw), np.empty(ndraw), np.empty((ndraw, p)), np.empty(ndraw))
    for it in range(burn + ndraw):
        if prior == "lasso":
            dinv = 1 / tau_j2
        else:
            dinv = 1 / (tau2 * (lam2 if prior == "horseshoe" else 1.0))
            dinv = np.broadcast_to(dinv, (p,)) if np.ndim(dinv) == 0 else dinv
        A = XtX + np.diag(dinv)
        L = np.linalg.cholesky(A)
        mean = np.linalg.solve(A, Xty)
        b = mean + np.sqrt(s2) * np.linalg.solve(L.T, rng.standard_normal(p))
        e = yc - Xs @ b
        if prior == "lasso":
            s2 = 1 / rng.gamma((n - 1 + p) / 2, 2 / (e @ e + b @ (dinv * b)))
            mu = np.sqrt(lam_l ** 2 * s2 / np.maximum(b ** 2, 1e-300))
            tau_j2 = 1 / np.maximum(rng.wald(mu, lam_l ** 2), 1e-12)
            if lam_fixed is None:
                lam_l = np.sqrt(rng.gamma(1.0 + p, 1 / (0.1 + tau_j2.sum() / 2)))
        else:
            s2 = 1 / rng.gamma((n - 1 + p) / 2, 2 / (e @ e + b @ (dinv * b)))
            if prior == "horseshoe":
                lam2 = 1 / rng.gamma(1.0, 1 / (1 / nu + b ** 2 / (2 * tau2 * s2)))
                nu = 1 / rng.gamma(1.0, 1 / (1 + 1 / lam2))
            if tau_fixed is None:
                lam_eff = lam2 if prior == "horseshoe" else np.ones(p)
                tau2 = 1 / rng.gamma((p + 1) / 2, 1 / (1 / xi + np.sum(b ** 2 / lam_eff) / (2 * s2)))
                xi = 1 / rng.gamma(1.0, 1 / (1 + 1 / tau2))
        if it >= burn:
            i = it - burn
            B[i], S2[i], T2[i] = b, s2, tau2
            KP[i] = 1 / (1 + n * tau2 * (lam2 if prior == "horseshoe" else 1.0)) if prior != "lasso" else np.nan
            LAM[i] = lam_l
    return B, S2, T2, KP, LAM


def bayes_shrink(df, spec):
    from .bayes import post_table, _geweke
    from .bayes_extra import _rhat_ess
    import statsmodels.api as sm
    y = need(spec, "y", "dependent variable")
    x = as_list(need(spec, "x", "regressors"))
    distinct([y], x)
    d = numeric(df, [y] + x)
    Y = d[y].to_numpy(float); X = d[x].to_numpy(float)
    n, p = X.shape
    prior = spec.get("prior", "horseshoe")
    if prior not in ("horseshoe", "lasso", "ridge"):
        raise OESError("Choose the prior: horseshoe, lasso or ridge.")
    if (X.std(0) == 0).any():
        raise OESError("A regressor is constant: drop it (the model has a constant).")
    if n < p + 5:
        raise OESError(f"Too few observations ({n}) for {p} regressors.")
    if p < 2:
        raise OESError("Choose at least two regressors.")
    ndraw = int_option(spec, "draws", 2000, 200, 50000, "Draws per chain")
    burn = int_option(spec, "burn", 1000, 0, 50000, "Burn-in")
    nch = int_option(spec, "chains", 2, 2, 8, "Chains")
    seed = int_option(spec, "seed", 0, 0, 10 ** 9, "Random seed")
    thr = float_option(spec, "threshold", 0.05, "Negligible effect")
    tau_f = spec.get("tau"); lam_f = spec.get("lasso_lambda")
    tau_f = None if tau_f in (None, "") else float_option(spec, "tau", 1.0, "Global scale")
    lam_f = None if lam_f in (None, "") else float_option(spec, "lasso_lambda", 1.0, "Lasso penalty")
    if thr <= 0:
        raise OESError("The negligible-effect threshold must be positive.")
    xm, xs = X.mean(0), X.std(0, ddof=0)
    Xs = (X - xm) / xs
    ym, ysd = Y.mean(), Y.std()
    yc = (Y - ym)
    chains = []
    for c in range(nch):
        rng = np.random.default_rng(seed + 1000 * c)
        chains.append(_shrink_chain(Xs, yc, prior, ndraw, burn, rng, tau_f, lam_f, init_scale=1.0 if c == 0 else 5.0))
    Bs = np.vstack([c[0] for c in chains]); S2 = np.concatenate([c[1] for c in chains]); KP = np.vstack([c[3] for c in chains])
    Bo = Bs / xs                                            # coefficients in the original units
    const = ym - Bo @ xm
    names = ["const"] + x
    notneg = (np.abs(Bs) > thr * ysd).mean(0)
    q = np.percentile(Bo, [2.5, 97.5], axis=0)
    shr = KP.mean(0) if prior != "lasso" else None
    ols = sm.OLS(Y, sm.add_constant(X)).fit()
    rows = []
    for j in np.argsort(-np.abs(Bs.mean(0))):
        v = x[j]
        rows.append([v, f(Bo[:, j].mean()), f(Bo[:, j].std(ddof=1)), f(q[0, j]), f(q[1, j]), f((Bo[:, j] > 0).mean(), 3), f(notneg[j], 3),
                     "yes" if (q[0, j] > 0 or q[1, j] < 0) else "no"] + ([f(shr[j], 3)] if shr is not None else []))
    heads = ["Variable", "Mean", "Std. dev.", "2.5%", "97.5%", "P(> 0)", "P(non-negligible)", "Interval excludes 0"] + (["Shrinkage"] if shr is not None else [])
    tabs = [table(f"Posterior of the coefficients ({prior} prior)", heads, rows,
                  "Sorted by the size of the standardised posterior mean. P(non-negligible) = posterior probability that a one-s.d. change in the regressor moves y by more than " + f"{thr:g} of its s.d." +
                  (" Shrinkage = posterior mean of 1/(1 + n tau^2 lambda^2) (Piironen and Vehtari 2017): near 1 means the coefficient is pulled to zero, near 0 left alone." if shr is not None else ""))]
    tabs.append(table("Ordinary least squares for comparison", ["Variable", "OLS coef.", "OLS std. err.", "Posterior mean", "Ratio (posterior / OLS)"],
                      [[v, f(ols.params[j + 1]), f(ols.bse[j + 1]), f(Bo[:, j].mean()), f(Bo[:, j].mean() / ols.params[j + 1] if ols.params[j + 1] != 0 else np.nan, 3)] for j, v in enumerate(x)],
                      "The ratio is the share of the OLS coefficient that survives the shrinkage."))
    eff = float(notneg.sum())
    ch_o = [c[0] / xs for c in chains]
    diag = []
    for j, v in enumerate(x):
        rh, ess = _rhat_ess([c[:, j] for c in ch_o]); diag.append([v, rh, ess, _geweke(ch_o[0][:, j])])
    rh_s, ess_s = _rhat_ess([c[1] for c in chains]); diag.append(["sigma^2", rh_s, ess_s, _geweke(chains[0][1])])
    worst = sorted(diag, key=lambda r_: r_[2])[:10]
    rows = [[r_[0], f(r_[1], 3), f(r_[2], 0), f(r_[3], 2)] for r_ in worst]
    tabs.append(table("MCMC convergence diagnostics", ["Parameter", "R-hat", "Effective sample size", "Geweke z (chain 1)"], rows,
                      f"The {len(worst)} parameters with the smallest effective sample size (of {len(diag)}). R-hat above 1.05 or an effective sample size of a few dozen means the chains have not mixed; the horseshoe needs more draws than a normal prior."))
    top = np.argsort(-np.abs(Bs.mean(0)))[:8]
    sens = []
    for pr_ in ("horseshoe", "lasso", "ridge"):
        c_ = _shrink_chain(Xs, yc, pr_, 800, 400, np.random.default_rng(seed + 11), None, None)
        sens.append([pr_] + [f(v) for v in (c_[0].mean(0) / xs)[top]])
    sens.append(["OLS"] + [f(v) for v in ols.params[1:][top]])
    tabs.append(table(f"Prior sensitivity: posterior means of the {len(top)} largest coefficients", ["Prior"] + [x[j] for j in top], sens,
                      "Each row is a separate run of 800 draws with the default hyperparameters. The ridge prior shrinks all coefficients by the same factor; the horseshoe and lasso shrink small ones more."))
    order = np.argsort(-np.abs(Bs.mean(0)))[:20]
    charts = [chart("Standardised coefficients: posterior mean against OLS (20 largest)", [x[j] for j in order],
                    [("Posterior mean", Bs.mean(0)[order]), ("OLS", (ols.params[1:] * xs)[order])], "", "effect of a one-s.d. change in x", kind="bar")]
    if shr is not None:
        charts.append(chart("Shrinkage of the 20 largest coefficients (1 = pulled to zero)", [x[j] for j in order], [("shrinkage", shr[order])], "", "shrinkage", kind="bar"))
    else:
        charts.append(chart("Probability that the effect is non-negligible", [x[j] for j in order], [("P(non-negligible)", notneg[order])], "", "probability", kind="bar"))
    fit = ym + Xs @ Bs.mean(0)
    r2 = 1 - ((Y - fit) ** 2).sum() / ((Y - ym) ** 2).sum()
    summary = [stat("Observations", n), stat("Regressors", p), stat("Prior", prior), stat("In-sample R-squared", r2, 4),
               stat("Posterior mean of sigma", float(np.sqrt(S2).mean())), stat("Effective number of non-negligible coefficients", eff, 2),
               stat("Largest R-hat", max(float(r_[1]) for r_ in diag), 3), stat("Smallest effective sample size", min(float(r_[2]) for r_ in diag), 0)]
    notes = [f"Regressors are standardised for the prior and the coefficients reported in the original units. {nch} chains of {ndraw} draws after {burn} burn-in."
             + (f" Global scale tau fixed at {tau_f:g}." if tau_f is not None and prior != "lasso" else "") + (f" Lasso penalty fixed at {lam_f:g}." if lam_f is not None and prior == "lasso" else "")]
    return result("bayes_shrink", f"Bayesian shrinkage regression ({prior}): {y}", n, summary, tabs, charts, notes)


# ================================================================================================ Bayesian change points
def _lse(a, axis=None):
    a = np.asarray(a, float)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return out.squeeze(axis) if axis is not None else float(out.squeeze())


def _seg_logml(y, Z, minseg, c, a0, b0):
    """log marginal likelihood of every segment y[t:s] (0 <= t < s <= n) under y = Z b + e, b | s2 ~ N(0, s2 c (Z'Z)^-1),
    s2 ~ IG(a0, b0). Returns an (n+1) x (n+1) array with -inf where the segment is shorter than minseg."""
    from scipy.special import gammaln
    n, k1 = Z.shape
    cy2 = np.concatenate([[0.0], np.cumsum(y * y)]); cZy = np.vstack([np.zeros(k1), np.cumsum(Z * y[:, None], axis=0)])
    cZZ = np.concatenate([np.zeros((1, k1, k1)), np.cumsum(Z[:, :, None] * Z[:, None, :], axis=0)])
    T, S = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    ok = (S - T) >= minseg
    t_, s_ = T[ok], S[ok]; L = (s_ - t_).astype(float)
    A = cZZ[s_] - cZZ[t_] + 1e-9 * np.eye(k1); r = cZy[s_] - cZy[t_]; yy = cy2[s_] - cy2[t_]
    bh = np.linalg.solve(A, r[:, :, None])[:, :, 0]
    bn = b0 + 0.5 * (yy - c / (1 + c) * (r * bh).sum(1)); an = a0 + L / 2
    out = np.full((n + 1, n + 1), -np.inf)
    out[t_, s_] = -L / 2 * np.log(2 * np.pi) + a0 * np.log(b0) - an * np.log(bn) + gammaln(an) - gammaln(a0) - k1 / 2 * np.log1p(c)
    return out


def _cp_posterior(logm, n, p, kmax):
    """Exact posterior over partitions with independent Bernoulli(p) breaks between observations (Fearnhead 2006; Barry and Hartigan 1993)."""
    lp, lq = np.log(p), np.log1p(-p)
    W = lambda t, s: logm[t, s] + (s - t - 1) * lq
    F = np.full((kmax + 1, n + 1), -np.inf); F[0, 0] = 0.0
    for s in range(1, n):
        t = np.arange(0, s)
        w = logm[t, s] + (s - t - 1) * lq + lp
        for k in range(1, kmax + 1):
            F[k, s] = _lse(F[k - 1, t] + w)
    tt = np.arange(0, n)
    last = logm[tt, n] + (n - tt - 1) * lq
    Z = np.array([_lse(F[k, tt] + last) for k in range(kmax + 1)])
    B = np.full((kmax + 1, n + 1), -np.inf)
    B[0, tt] = last
    for t in range(n - 1, -1, -1):
        s = np.arange(t + 1, n)
        if len(s) == 0:
            continue
        w = logm[t, s] + (s - t - 1) * lq + lp
        for k in range(1, kmax + 1):
            B[k, t] = _lse(B[k - 1, s] + w)
    return F, B, Z


def changepoint(df, spec):
    from scipy.special import gammaln
    y_ = need(spec, "y", "series")
    xs = as_list(spec.get("x")) if spec.get("x") not in (None, "", []) else []
    distinct([y_], xs)
    tcol = spec.get("time")
    cols = [y_] + xs
    if tcol:
        from .core import check_columns, time_order
        check_columns(df, [tcol]); df = time_order(df, tcol)
    d = numeric(df, cols, keep=[tcol] if tcol else []).reset_index(drop=True)
    labels = d[tcol].astype(str).tolist() if tcol else [str(i + 1) for i in range(len(d))]
    n = len(d); k1 = 1 + len(xs)
    if n < 10:
        raise OESError("At least 10 observations are needed to look for change points.")
    if n > 700:
        raise OESError("At most 700 observations are supported (the exact algorithm is quadratic in n).")
    minseg = int_option(spec, "min_segment", max(5, 3 * k1), 2, n // 2, "Minimum segment length")
    if minseg < k1 + 1:
        raise OESError(f"The minimum segment length must be at least {k1 + 1} (the number of coefficients plus one).")
    eb = float_option(spec, "expected_breaks", 1.0, "Expected number of breaks")
    if eb <= 0 or eb >= n - 1:
        raise OESError("The expected number of breaks must be positive and below n - 1.")
    kmax = int_option(spec, "max_breaks", min(10, n // minseg), 1, 15, "Maximum number of breaks")
    cg = float_option(spec, "prior_scale", float(n), "Prior scale c")
    if cg <= 0:
        raise OESError("The prior scale c must be positive.")
    y = d[y_].to_numpy(float); ym, ysd = y.mean(), y.std()
    if ysd == 0:
        raise OESError("The series is constant.")
    ys = (y - ym) / ysd
    Z = np.column_stack([np.ones(n)] + [(d[c].to_numpy(float) - d[c].mean()) / (d[c].std() if d[c].std() > 0 else 1.0) for c in xs])
    a0, b0 = 1.0, 0.1
    p = eb / (n - 1)
    logm = _seg_logml(ys, Z, minseg, cg, a0, b0)
    F, B, Zk = _cp_posterior(logm, n, p, kmax)
    lpy = _lse(Zk)
    pk = np.exp(Zk - lpy)
    # exact marginal probability of a break after observation t (t = 1 .. n-1)
    pb = np.zeros(n + 1)
    for t in range(1, n):
        v = []
        for k in range(1, kmax + 1):
            for j in range(0, kmax - k + 1):
                v.append(F[k, t] + B[j, t])
        pb[t] = np.exp(_lse(np.array(v)) - lpy) if v else 0.0
    # null model (no break) and Bayes factors against it
    lq = np.log1p(-p)
    null = logm[0, n] + (n - 1) * lq
    # most probable partition (at most kmax breaks) by dynamic programming over the number of breaks
    lp_, lq_ = np.log(p), lq
    V = np.full((kmax + 1, n + 1), -np.inf); V[0, 0] = 0.0; bp = np.zeros((kmax + 1, n + 1), int)
    for s in range(1, n):
        t = np.arange(0, s)
        for k in range(1, kmax + 1):
            sc = V[k - 1, t] + logm[t, s] + (s - t - 1) * lq_ + lp_
            i = int(np.argmax(sc)); V[k, s], bp[k, s] = sc[i], t[i]
    tt = np.arange(0, n)
    fin = [np.max(V[k, tt] + logm[tt, n] + (n - tt - 1) * lq_) for k in range(kmax + 1)]
    kbest = int(np.argmax(fin)); vbest = fin[kbest]
    cps, s, k = [], int(tt[np.argmax(V[kbest, tt] + logm[tt, n] + (n - tt - 1) * lq_)]), kbest
    while k > 0:
        cps.append(s); s = int(bp[k, s]); k -= 1
    cps = sorted(cps)
    bounds = [0] + cps + [n]
    nm = [x_ for x_ in ["const"] + xs]
    segrows, fitted = [], np.zeros(n)
    for i in range(len(bounds) - 1):
        t, s = bounds[i], bounds[i + 1]
        Zs, yv = Z[t:s], ys[t:s]; L = s - t
        A = Zs.T @ Zs + 1e-9 * np.eye(k1); bh = np.linalg.solve(A, Zs.T @ yv); sh = cg / (1 + cg)
        bn = b0 + 0.5 * (yv @ yv - sh * (Zs.T @ yv) @ bh); an = a0 + L / 2
        cov = np.linalg.inv(A) * sh * bn / an
        scale = np.sqrt(np.diag(cov)); q = stats.t.ppf([0.025, 0.975], 2 * an)
        mu = sh * bh
        sig = np.sqrt(bn / (an - 1)) * ysd
        for j in range(k1):
            # coefficients on the original scale: slopes divide by the regressor's scale; the constant is the level of y at the mean of x
            m_, sd_ = mu[j] * ysd + (ym if j == 0 else 0.0), scale[j] * ysd * np.sqrt(2 * an / (2 * an - 2))
            sc_ = 1.0 if j == 0 else (d[xs[j - 1]].std() if d[xs[j - 1]].std() > 0 else 1.0)
            if j > 0:
                m_, sd_ = m_ / sc_, sd_ / sc_
            lo, hi = m_ + q[0] * (scale[j] * ysd / sc_), m_ + q[1] * (scale[j] * ysd / sc_)
            segrows.append([f"{i + 1}", f"{labels[t]} to {labels[s - 1]}", int(L), nm[j], f(m_), f(sd_), f(lo), f(hi), f(sig) if j == 0 else ""])
        fitted[t:s] = ym + ysd * (Zs @ mu) if k1 == 1 else np.nan
    tabs = [table("Posterior probability of the number of breaks", ["Breaks", "Posterior prob.", "Prior prob.", "Log Bayes factor against no break"],
                  [[kk, f(pk[kk], 4), f(float(np.exp(stats.binom.logpmf(kk, n - 1, p))), 4), (f(float(Zk[kk] - Zk[0]), 2) if np.isfinite(Zk[kk]) else "not possible") if kk else "0"] for kk in range(kmax + 1)],
                  f"Exact (no simulation): all segmentations with at most {kmax} breaks and segments of at least {minseg} observations are summed. The prior puts an independent break probability of {p:.4f} between consecutive observations (expected number of breaks {eb:g}).")]
    top = np.argsort(-pb)[:10]
    tabs.append(table("Posterior probability of a break after each date (10 highest)", ["After", "Before", "Posterior prob."],
                      [[labels[t - 1], labels[t], f(pb[t], 4)] for t in top if pb[t] > 1e-4]))
    tabs.append(table("Most probable segmentation: coefficients in each regime", ["Regime", "Period", "Observations", "Parameter", "Mean", "Std. dev.", "2.5%", "97.5%", "Residual s.d."], segrows,
                      f"Posterior of each regime given the segmentation with the highest posterior probability ({len(cps)} break{"" if len(cps) == 1 else "s"}; log posterior of the segmentation relative to no break: {vbest - null:.2f})."))
    sens = []
    for e_ in (0.5, 1.0, 2.0, 4.0):
        for c_ in (n / 10, float(n), 10.0 * n):
            lm2 = _seg_logml(ys, Z, minseg, c_, a0, b0)
            F2, B2, Z2 = _cp_posterior(lm2, n, e_ / (n - 1), kmax)
            pk2 = np.exp(Z2 - _lse(Z2)); mp = int(np.argmax(pk2))
            sens.append([f"{e_:g}", f"{c_:g}", f(1 - pk2[0], 3), mp, f(float(np.dot(np.arange(kmax + 1), pk2)), 2)])
    tabs.append(table("Prior sensitivity", ["Expected breaks (prior)", "Prior scale c", "P(at least one break)", "Most probable number", "Posterior mean number"], sens,
                      "Rows re-run the exact calculation with other priors on the number of breaks and on the size of the coefficients."))
    charts = [chart("Posterior probability of a break after each date", labels[:-1], [("P(break)", pb[1:n])], tcol or "observation", "probability", kind="bar")]
    if k1 == 1:
        charts.append(chart("Series and most probable regime means", labels, [(y_, y), ("regime mean", fitted)], tcol or "observation", y_))
    summary = [stat("Observations", n), stat("Posterior mean number of breaks", float(np.dot(np.arange(kmax + 1), pk)), 3),
               stat("Most probable number of breaks", int(np.argmax(pk))), stat("P(at least one break)", 1 - pk[0], 4),
               stat("Breaks in the most probable segmentation", len(cps)),
               stat("Dates of the most probable breaks (last date of each regime)", ", ".join(labels[t - 1] for t in cps) if cps else "none")]
    notes = []
    if pk[kmax] > 0.05:
        notes.append(f"The posterior puts {pk[kmax]:.2f} of its probability on the largest number of breaks considered ({kmax}); raise 'largest number of breaks' or the minimum regime length.")
    return result("changepoint", f"Bayesian change points: {y_}", n, summary, tabs, charts,
                  notes + ["Conjugate normal-inverse-gamma regime models (g-prior with c = " + f"{cg:g}" + " on the standardised data; IG(1, 0.1) on the standardised variance); every regime has its own coefficients and variance. Breaks are independent Bernoulli events between consecutive observations."])


# ================================================================================================ Bayesian quantile regression
def _qr_chain(Xs, ys, tau, ndraw, burn, rng, prior_sd, init=None):
    """Gibbs sampler for the asymmetric-Laplace quantile regression (Kozumi and Kobayashi 2011), standardised data."""
    n, k = Xs.shape
    th = (1 - 2 * tau) / (tau * (1 - tau)); t2 = 2.0 / (tau * (1 - tau))
    B0inv = np.eye(k) / prior_sd ** 2
    beta = np.zeros(k) if init is None else init.copy()
    sig = 1.0
    a0, b0 = 1e-3, 1e-3
    out_b = np.empty((ndraw, k)); out_s = np.empty(ndraw)
    for it in range(ndraw + burn):
        r = ys - Xs @ beta
        a = np.maximum(r ** 2 / (t2 * sig), 1e-12)
        b = th ** 2 / (t2 * sig) + 2.0 / sig
        w = rng.wald(np.sqrt(b / a), b)                     # w = 1 / v
        v = 1.0 / w
        prec = (Xs * (w / (t2 * sig))[:, None]).T @ Xs + B0inv
        cov = np.linalg.inv(prec); cov = (cov + cov.T) / 2
        mean = cov @ (Xs.T @ (w * (ys - th * v)) / (t2 * sig))
        beta = rng.multivariate_normal(mean, cov)
        r = ys - Xs @ beta - th * v
        sig = _ig(rng, 1.5 * n + a0, float((r ** 2 * w).sum() / (2 * t2) + v.sum() + b0))
        if it >= burn:
            out_b[it - burn] = beta; out_s[it - burn] = sig
    return out_b, out_s


def _qr_adjust(Bs, Xs, ys, tau):
    """Rescale the draws so their covariance is the sandwich covariance of the quantile-regression estimator (Yang, Wang and He 2016):
    tau (1 - tau) D1^-1 D0 D1^-1 / n, with D1 estimated by a kernel (Powell 1991) at the posterior mean."""
    n, k = Xs.shape
    m = Bs.mean(0); r = ys - Xs @ m
    z = stats.norm.ppf(tau)
    hq = n ** (-1 / 3) * stats.norm.ppf(0.975) ** (2 / 3) * (1.5 * stats.norm.pdf(z) ** 2 / (2 * z ** 2 + 1)) ** (1 / 3)
    hq = min(hq, 0.45 * min(tau, 1 - tau) * 2) if min(tau, 1 - tau) < hq else hq
    b = float(np.quantile(r, min(tau + hq, 0.999)) - np.quantile(r, max(tau - hq, 0.001)))
    if not np.isfinite(b) or b <= 0:
        return Bs
    K = (np.abs(r) < b)
    D1 = (Xs * K[:, None]).T @ Xs / (2 * b * n); D0 = Xs.T @ Xs / n
    try:
        D1i = np.linalg.inv(D1)
    except np.linalg.LinAlgError:
        return Bs
    Sa = tau * (1 - tau) * D1i @ D0 @ D1i / n
    Sp = np.cov(Bs.T).reshape(k, k)
    try:
        La = np.linalg.cholesky((Sa + Sa.T) / 2); Lp = np.linalg.cholesky((Sp + Sp.T) / 2)
    except np.linalg.LinAlgError:
        return Bs
    return m + (Bs - m) @ (La @ np.linalg.inv(Lp)).T


def bayes_qr(df, spec):
    from .bayes import _geweke
    from .bayes_extra import _rhat_ess
    import statsmodels.api as sm
    y = need(spec, "y", "dependent variable")
    x = as_list(need(spec, "x", "regressors"))
    distinct([y], x)
    d = numeric(df, [y] + x)
    Y = d[y].to_numpy(float); X = d[x].to_numpy(float)
    n, p = X.shape
    raw = spec.get("quantiles", "0.1, 0.25, 0.5, 0.75, 0.9")
    try:
        qs = sorted({float(t) for t in (raw if isinstance(raw, (list, tuple)) else str(raw).replace(";", ",").split(",")) if str(t).strip() != ""})
    except ValueError:
        raise OESError("Quantiles must be numbers between 0 and 1, separated by commas.")
    if not qs or len(qs) > 9 or any(q_ <= 0 or q_ >= 1 for q_ in qs):
        raise OESError("Give between 1 and 9 quantiles strictly between 0 and 1, for example 0.1, 0.5, 0.9.")
    if (X.std(0) == 0).any():
        raise OESError("A regressor is constant: drop it (the model has a constant).")
    if n < 5 * (p + 1):
        raise OESError(f"Too few observations ({n}) for {p} regressors.")
    ndraw = int_option(spec, "draws", 2000, 200, 50000, "Draws per chain")
    burn = int_option(spec, "burn", 1000, 0, 50000, "Burn-in")
    nch = int_option(spec, "chains", 2, 2, 8, "Chains")
    seed = int_option(spec, "seed", 0, 0, 10 ** 9, "Random seed")
    psd = float_option(spec, "prior_sd", 10.0, "Prior standard deviation")
    adj = str(spec.get("adjust", "yes")).lower() not in ("no", "false", "0")
    if psd <= 0:
        raise OESError("The prior standard deviation must be positive.")
    xm, xs = X.mean(0), X.std(0)
    ym, ysd = Y.mean(), Y.std()
    Xs = np.column_stack([np.ones(n), (X - xm) / xs]); ys = (Y - ym) / ysd
    names = ["const"] + x

    def to_orig(Bs):
        B = Bs.copy()
        B[:, 1:] = Bs[:, 1:] * ysd / xs
        B[:, 0] = ym + ysd * Bs[:, 0] - B[:, 1:] @ xm
        return B

    post = {}; rows = []; diag = []; calib = []
    for qi, tau in enumerate(qs):
        ch = [_qr_chain(Xs, ys, tau, ndraw, burn, np.random.default_rng(seed + 1000 * c + qi), psd, init=None if c == 0 else np.random.default_rng(seed + 7 * c).normal(0, 1, p + 1)) for c in range(nch)]
        Braw = np.vstack([c[0] for c in ch])
        B = to_orig(_qr_adjust(Braw, Xs, ys, tau) if adj else Braw); post[tau] = B
        fq = sm.QuantReg(Y, sm.add_constant(X)).fit(q=tau, max_iter=5000)
        pe = B.mean(0); lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
        for j, v in enumerate(names):
            rows.append([f"{tau:g}", v, f(pe[j]), f(B[:, j].std(ddof=1)), f(lo[j]), f(hi[j]), f((B[:, j] > 0).mean(), 3), f(fq.params[j])])
        for j, v in enumerate(names):
            rh, ess = _rhat_ess([to_orig(c[0])[:, j] for c in ch]); diag.append([f"{tau:g}", v, rh, ess])
        fit = np.column_stack([np.ones(n), X]) @ pe
        calib.append([f"{tau:g}", f((Y <= fit).mean(), 3), f(abs((Y <= fit).mean() - tau), 3)])
    tabs = [table("Posterior of the coefficients at each quantile",
                  ["Quantile", "Variable", "Mean", "Std. dev.", "2.5%", "97.5%", "P(> 0)", "Frequentist quantile regression"], rows,
                  f"Asymmetric-Laplace likelihood, normal prior with s.d. {psd:g} on the standardised coefficients, {nch} chains of {ndraw} draws after {burn} burn-in. " + ("Draws rescaled to the sandwich covariance of the quantile-regression estimator (Yang, Wang and He 2016), so intervals have the right width. " if adj else "Draws NOT rescaled: asymmetric-Laplace intervals are usually too narrow. ") + "The last column is the ordinary (check-function) quantile-regression estimate, which the posterior mean should be close to.")]
    if len(qs) >= 2:
        lo_q, hi_q = qs[0], qs[-1]
        rows = []
        for j in range(1, p + 1):
            dif = post[hi_q][:, j] - post[lo_q][:, j]
            a_, b_ = np.percentile(dif, [2.5, 97.5])
            rows.append([names[j], f(post[lo_q][:, j].mean()), f(post[hi_q][:, j].mean()), f(dif.mean()), f(a_), f(b_), f((dif > 0).mean(), 3)])
        tabs.append(table(f"Does the effect differ between the {lo_q:g} and the {hi_q:g} quantile?",
                          ["Variable", f"Slope at {lo_q:g}", f"Slope at {hi_q:g}", "Difference", "2.5%", "97.5%", "P(difference > 0)"], rows,
                          "A difference whose interval excludes zero means the regressor changes the spread or the skew of the outcome, not only its centre."))
    tabs.append(table("Calibration: share of observations below the fitted quantile", ["Quantile", "Share below", "Gap"], calib,
                      "A well-fitted quantile line has the share below it close to the quantile itself."))
    worst = sorted(diag, key=lambda r_: r_[3])[:10]
    tabs.append(table("MCMC convergence diagnostics", ["Quantile", "Parameter", "R-hat", "Effective sample size"],
                      [[r_[0], r_[1], f(r_[2], 3), f(r_[3], 0)] for r_ in worst],
                      f"The {len(worst)} parameters with the smallest effective sample size (of {len(diag)}). Extreme quantiles mix more slowly; add draws if the effective sample size is below a few hundred."))
    ols = sm.OLS(Y, sm.add_constant(X)).fit()
    charts = []
    for j in range(1, min(p, 3) + 1):
        m_ = [post[t_][:, j].mean() for t_ in qs]
        band = ([np.percentile(post[t_][:, j], 2.5) for t_ in qs], [np.percentile(post[t_][:, j], 97.5) for t_ in qs])
        charts.append(chart(f"Effect of {names[j]} across the distribution of {y}", [f"{t_:g}" for t_ in qs],
                            [("Posterior mean", m_), ("OLS", [ols.params[j]] * len(qs))], "quantile", "coefficient", band=band))
    worst_r = max(float(r_[2]) for r_ in diag)
    summary = [stat("Observations", n), stat("Regressors", p), stat("Quantiles", ", ".join(f"{t_:g}" for t_ in qs)),
               stat("Largest R-hat", worst_r, 3), stat("Smallest effective sample size", min(float(r_[3]) for r_ in diag), 0)]
    return result("bayes_qr", f"Bayesian quantile regression: {y}", n, summary, tabs, charts,
                  ["Kozumi-Kobayashi (2011) location-scale mixture of the asymmetric Laplace distribution. The working likelihood gives the right centre for the quantile but not the right spread, so by default the draws are rescaled to the sandwich covariance (Yang, Wang and He 2016); set the adjustment to no to see the raw posterior. Each quantile is a separate model, so differences between quantiles treat their estimates as independent, which slightly overstates the uncertainty of the difference."])


# ================================================================================================ Bayesian structural time series
def _bsts_layout(T, trend, s):
    """Index of each state: interleaved by time (level, slope, seasonal) so that the precision matrix is banded."""
    m = 1 + (1 if trend else 0) + (1 if s > 1 else 0)
    return m, (lambda t, c: t * m + c)


def _bsts_gibbs(ys, X, obs, trend, s, T, ndraw, burn, rng, prior_b, prior_a, init_scale=1.0, b_irr=0.001):
    """Gibbs sampler for y_t = level_t + seasonal_t + x_t'b + e_t with random-walk level, optional random-walk slope and seasonal.
    The states are drawn jointly from their banded Gaussian conditional (perturbation sampling), the variances from inverse gammas."""
    import scipy.sparse as sp
    from scipy.linalg import cholesky_banded, cho_solve_banded
    n = len(ys)
    m, ix = _bsts_layout(T, trend, s)
    N = T * m
    lv = lambda t: ix(t, 0)
    sl = (lambda t: ix(t, 1)) if trend else None
    se = (lambda t: ix(t, 1 + (1 if trend else 0))) if s > 1 else None
    big = 1e4
    rows_i, rows_j, rows_v, kind = [], [], [], []             # sparse design A: one row per constraint
    r = 0
    def add(cols, vals, k):
        nonlocal r
        for c_, v_ in zip(cols, vals):
            rows_i.append(r); rows_j.append(c_); rows_v.append(v_)
        kind.append(k); r += 1
    for t in range(n):
        if obs[t]:
            add([lv(t)] + ([se(t)] if se else []), [1.0] + ([1.0] if se else []), 0)          # observation
    for t in range(T - 1):
        add([lv(t + 1), lv(t)] + ([sl(t)] if trend else []), [1.0, -1.0] + ([-1.0] if trend else []), 1)          # level
    if trend:
        for t in range(T - 1):
            add([sl(t + 1), sl(t)], [1.0, -1.0], 2)                                                                   # slope
    if s > 1:
        for t in range(s - 1, T):
            add([se(t - j) for j in range(s)], [1.0] * s, 3)                                                           # seasonal sum
    first = r
    add([lv(0)], [1.0], 4)                                                                                            # diffuse initial level
    if trend:
        add([sl(0)], [1.0], 5)
    if s > 1:
        for j in range(s - 1):
            add([se(j)], [1.0], 6)
    A = sp.csr_matrix((rows_v, (rows_i, rows_j)), shape=(r, N))
    kind = np.array(kind)
    nobs_rows = int((kind == 0).sum())
    obs_idx = np.where(obs)[0]
    prec_diag = np.ones(r)
    Q0 = None
    nb = X.shape[1] if X is not None else 0
    beta = np.zeros(nb)
    v_e, v_l, v_s, v_g = 0.1, 0.01, 0.001, 0.001
    out = dict(lvl=[], slp=[], sea=[], sig=[], beta=[])
    # state prior means: initial level at y_1
    y1 = ys[obs_idx[0]]
    for it in range(ndraw + burn):
        w = np.empty(r)
        w[kind == 0] = 1.0 / v_e; w[kind == 1] = 1.0 / v_l; w[kind == 2] = 1.0 / v_s; w[kind == 3] = 1.0 / v_g
        w[kind >= 4] = 1.0 / big
        yy = ys[obs_idx] - (X[obs_idx] @ beta if nb else 0.0)
        c = np.zeros(r); c[kind == 0] = yy; c[kind == 4] = y1
        sw = np.sqrt(w)
        Q = (A.T @ sp.diags(w) @ A).tocsc()
        rhs = A.T @ (w * c + sw * rng.standard_normal(r))
        D = Q.todia(); offs = D.offsets; bw = int(max(-offs.min(), 0))
        ab = np.zeros((bw + 1, N))
        for k_, o in enumerate(offs):
            if o <= 0:
                d_ = -o; ab[d_, :N - d_] = D.data[k_, :N - d_]
        cb = cholesky_banded(ab, lower=True)
        z = cho_solve_banded((cb, True), rhs)
        mu = z[0::m]; col = 1
        slp = z[col::m] if trend else None; col += 1 if trend else 0
        sea = z[col::m] if s > 1 else None
        fit = mu + (sea if s > 1 else 0.0)
        if nb:
            res_b = ys[obs_idx] - fit[obs_idx]
            Xo = X[obs_idx]
            P = Xo.T @ Xo / v_e + np.eye(nb) / 100.0
            Pc = np.linalg.inv(P); Pc = (Pc + Pc.T) / 2
            beta = rng.multivariate_normal(Pc @ (Xo.T @ res_b / v_e), Pc)
            e = res_b - Xo @ beta
        else:
            e = ys[obs_idx] - fit[obs_idx]
        v_e = _ig(rng, prior_a + 0.5 * len(e), b_irr + 0.5 * float((e ** 2).sum()))
        dl = mu[1:] - mu[:-1] - (slp[:-1] if trend else 0.0)
        v_l = _ig(rng, prior_a + 0.5 * len(dl), prior_b + 0.5 * float((dl ** 2).sum()))
        if trend:
            ds = slp[1:] - slp[:-1]
            v_s = _ig(rng, prior_a + 0.5 * len(ds), prior_b + 0.5 * float((ds ** 2).sum()))
        if s > 1:
            cs = np.array([sea[t - s + 1:t + 1].sum() for t in range(s - 1, T)])
            v_g = _ig(rng, prior_a + 0.5 * len(cs), prior_b + 0.5 * float((cs ** 2).sum()))
        if it >= burn:
            out["lvl"].append(mu); out["sig"].append([v_e, v_l, v_s if trend else 0.0, v_g if s > 1 else 0.0])
            if trend: out["slp"].append(slp)
            if s > 1: out["sea"].append(sea)
            out["beta"].append(beta.copy())
    return {k: (np.array(v) if len(v) else None) for k, v in out.items()}


def bsts(df, spec):
    import statsmodels.api as sm
    from .bayes import _geweke
    from .bayes_extra import _rhat_ess
    from .core import check_columns, time_order
    y_ = need(spec, "y", "series")
    xs = as_list(spec.get("x"))
    distinct([y_], xs)
    tcol = spec.get("time")
    if tcol:
        check_columns(df, [tcol] + [y_] + xs); df = time_order(df, tcol)
    else:
        check_columns(df, [y_] + xs)
    d = df.reset_index(drop=True)
    y = pd.to_numeric(d[y_], errors="coerce").to_numpy(float)
    n0 = len(d)
    labels = d[tcol].astype(str).tolist() if tcol else [str(i + 1) for i in range(n0)]
    trend = str(spec.get("trend", "level")) == "trend"
    if str(spec.get("trend", "level")) not in ("level", "trend"):
        raise OESError("Choose the trend: level or trend.")
    s = int_option(spec, "seasonal", 0, 0, 52, "Seasonal period")
    if s == 1:
        raise OESError("The seasonal period must be 0 (none) or at least 2.")
    h = int_option(spec, "horizon", 0, 0, 120, "Forecast horizon")
    if xs and h > 0:
        raise OESError("Forecasts need future values of the regressors: set the horizon to 0 or drop the regressors.")
    obs = np.isfinite(y)
    if xs:
        Xd = d[xs].apply(pd.to_numeric, errors="coerce")
        if Xd.isna().any().any():
            raise OESError("The regressors have missing values: drop those rows or fill them in.")
        Xr = Xd.to_numpy(float); Xr = (Xr - Xr.mean(0)) / np.where(Xr.std(0) > 0, Xr.std(0), 1.0)
        xm, xsd = Xd.to_numpy(float).mean(0), Xd.to_numpy(float).std(0)
        if (xsd == 0).any():
            raise OESError("A regressor is constant: drop it.")
    else:
        Xr = None
    if obs.sum() < max(15, 2 * s + 5):
        raise OESError(f"Too few observations ({int(obs.sum())}) for this model: at least {max(15, 2 * s + 5)} are needed.")
    if n0 + h > 400:
        raise OESError("At most 400 time points (observations plus forecast horizon) are supported.")
    ym, ysd = np.nanmean(y), np.nanstd(y)
    if ysd == 0:
        raise OESError("The series is constant.")
    ys = np.full(n0 + h, np.nan); ys[:n0] = (y - ym) / ysd
    obs_f = np.isfinite(ys)
    ys0 = np.where(obs_f, ys, 0.0)
    Xf = None if Xr is None else Xr
    ndraw = int_option(spec, "draws", 1500, 200, 20000, "Draws per chain")
    burn = int_option(spec, "burn", 500, 0, 20000, "Burn-in")
    nch = int_option(spec, "chains", 2, 2, 6, "Chains")
    seed = int_option(spec, "seed", 0, 0, 10 ** 9, "Random seed")
    pb = float_option(spec, "prior_scale", 0.01, "Prior scale of the components")
    if pb <= 0:
        raise OESError("The prior scale of the variances must be positive.")
    pb = pb ** 2                                          # prior mean variance of each component, in standardised units
    T = n0 + h
    pa = 1.0
    if xs:
        Xf = np.vstack([Xr, np.zeros((h, Xr.shape[1]))]) if h else Xr
    chains = [_bsts_gibbs(ys0, Xf, obs_f, trend, s, T, ndraw, burn, np.random.default_rng(seed + 1000 * c), pb, pa) for c in range(nch)]  # the irregular term has a weak prior (b = 0.001)
    cat = lambda k: None if chains[0][k] is None else np.vstack([c[k] for c in chains])
    LV, SG = cat("lvl"), cat("sig"); SL, SE, BE = cat("slp"), cat("sea"), cat("beta")
    SD = np.sqrt(SG) * ysd
    rng = np.random.default_rng(seed + 77)
    names = ["Irregular (observation noise)", "Level", "Slope", "Seasonal"]
    use = [0, 1] + ([2] if trend else []) + ([3] if s > 1 else [])
    ml = sm.tsa.UnobservedComponents(pd.Series(y[:n0]), "local linear trend" if trend else "local level",
                                     seasonal=s if s > 1 else None).fit(disp=0) if not xs else None
    mle = {}
    if ml is not None:
        pn = list(ml.params.index)
        for lab, key in ((0, "irregular"), (1, "level"), (2, "trend"), (3, "seasonal")):
            for nm in pn:
                if nm.endswith(key) and nm.startswith("sigma2"):
                    mle[lab] = float(np.sqrt(ml.params[nm]))
    rows = []; diag = []
    for j in use:
        v = SD[:, j]
        per = [np.sqrt(c["sig"][:, j]) * ysd for c in chains]
        rh, ess = _rhat_ess(per)
        diag.append([names[j], rh, ess, _geweke(per[0])])
        rows.append([names[j], f(v.mean()), f(np.median(v)), f(np.percentile(v, 2.5)), f(np.percentile(v, 97.5)), f(mle[j]) if j in mle else "n/a"])
    tabs = [table("Posterior of the standard deviations of the components", ["Component", "Mean", "Median", "2.5%", "97.5%", "Maximum likelihood (Kalman filter)"], rows,
                  "Standard deviations in the units of the series. The maximum-likelihood column is the classical estimate from the same model; it is a point estimate and can be exactly zero for a component, where the posterior still gives a range.")]
    lvl_o = LV[:, :T] * ysd + ym
    tr_m = lvl_o.mean(0); tr_lo, tr_hi = np.percentile(lvl_o, [2.5, 97.5], axis=0)
    ratio = SD[:, 1] / SD[:, 0]
    lastrows = [["Level at the last observation", f(lvl_o[:, n0 - 1].mean()), f(lvl_o[:, n0 - 1].std(ddof=1)), f(np.percentile(lvl_o[:, n0 - 1], 2.5)), f(np.percentile(lvl_o[:, n0 - 1], 97.5))]]
    if trend:
        sl_o = SL[:, :T] * ysd
        lastrows.append(["Slope at the last observation (per period)", f(sl_o[:, n0 - 1].mean()), f(sl_o[:, n0 - 1].std(ddof=1)), f(np.percentile(sl_o[:, n0 - 1], 2.5)), f(np.percentile(sl_o[:, n0 - 1], 97.5))])
        lastrows.append(["Probability that the slope is positive", f((sl_o[:, n0 - 1] > 0).mean(), 3), "", "", ""])
    if xs:
        bo = BE * ysd / xsd
        for j, v in enumerate(xs):
            lastrows.append([f"Coefficient of {v}", f(bo[:, j].mean()), f(bo[:, j].std(ddof=1)), f(np.percentile(bo[:, j], 2.5)), f(np.percentile(bo[:, j], 97.5))])
    tabs.append(table("States and coefficients", ["Quantity", "Mean", "Std. dev.", "2.5%", "97.5%"], lastrows))
    sea_o = None
    if s > 1:
        sea_o = SE[:, :T] * ysd
        sm_ = sea_o[:, n0 - s:n0]
        tabs.append(table(f"Seasonal effects over the last {s} periods", ["Period", "Mean", "2.5%", "97.5%"],
                          [[labels[n0 - s + i], f(sm_[:, i].mean()), f(np.percentile(sm_[:, i], 2.5)), f(np.percentile(sm_[:, i], 97.5))] for i in range(s)],
                          "Effects add to the level; over a full cycle they sum to about zero."))
    # fitted, residuals
    fit = lvl_o[:, :n0] + (sea_o[:, :n0] if sea_o is not None else 0.0)
    if xs:
        fit = fit + (Xr @ (BE * ysd).T).T
    sd_e = SD[:, 0]
    oi = np.where(obs)[0]
    zres = ((y[oi] - fit[:, oi].mean(0)) / sd_e.mean())
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lag = min(10, max(2, len(oi) // 5))
    lb = acorr_ljungbox(zres, lags=[lag], return_df=True)
    ac1 = float(np.corrcoef(zres[1:], zres[:-1])[0, 1])
    tabs.append(table("Residual check", ["Check", "Value", "Detail"],
                      [["Standardised residuals: share beyond +-2", f((np.abs(zres) > 2).mean(), 3), "about 0.05 if the model fits"],
                       ["First-order autocorrelation", f(ac1, 3), "near 0 if the components capture the dynamics"],
                       [f"Ljung-Box p-value ({lag} lags)", f(float(lb['lb_pvalue'].iloc[0]), 4), "a small value means remaining dynamics"],
                       ["Signal-to-noise ratio (level s.d. / irregular s.d.), posterior mean", f(ratio.mean(), 3), "how fast the level moves relative to the noise"]],
                      "These residuals come from the smoothed states, which have already adapted to the data: they are somewhat too small and negatively autocorrelated by construction. Use them to spot gross problems; for an honest test of forecasts, hold out the last observations and compare (see the guide)."))
    charts = []
    xlab = labels + [f"+{i + 1}" for i in range(h)]
    charts.append(chart(f"{y_} and the posterior level (95% band)", labels, [("Series", np.where(obs, y, np.nan)), ("Level", tr_m[:n0])], "", y_, band=(tr_lo[:n0], tr_hi[:n0])))
    if h:
        ye = lvl_o[:, n0:] + (sea_o[:, n0:] if sea_o is not None else 0.0) + rng.standard_normal((len(LV), h)) * sd_e[:, None]
        pm = ye.mean(0); pl, ph = np.percentile(ye, [2.5, 97.5], axis=0)
        k0 = max(0, n0 - 3 * max(s, 4))
        tabs.append(table(f"Forecast for the next {h} periods", ["Step", "Mean", "Std. dev.", "2.5%", "97.5%"],
                          [[f"+{i + 1}", f(pm[i]), f(ye[:, i].std(ddof=1)), f(pl[i]), f(ph[i])] for i in range(h)],
                          "Predictive distribution of the observation: includes the uncertainty of the level, the slope, the seasonal and the noise."))
        charts.append(chart("Forecast and 95% predictive band", labels[k0:] + [f"+{i + 1}" for i in range(h)],
                            [("Series", list(y[k0:n0]) + [np.nan] * h), ("Forecast", [np.nan] * (n0 - 1 - k0) + [fit[:, n0 - 1].mean()] + list(pm))],
                            "", y_, band=([np.nan] * (n0 - 1 - k0) + [fit[:, n0 - 1].mean()] + list(pl), [np.nan] * (n0 - 1 - k0) + [fit[:, n0 - 1].mean()] + list(ph))))
    if s > 1:
        charts.append(chart("Seasonal effect over the last cycle (95% band)", [labels[n0 - s + i] for i in range(s)], [("Seasonal", sm_.mean(0))], "", "effect",
                            band=(np.percentile(sm_, 2.5, axis=0), np.percentile(sm_, 97.5, axis=0))))
    miss = int((~obs).sum())
    summary = [stat("Observations", int(obs.sum())), stat("Trend", "local linear trend" if trend else "local level"), stat("Seasonal period", s if s > 1 else "none")]
    if miss: summary.append(stat("Missing values filled in", miss))
    summary += [stat("Posterior mean of the signal-to-noise ratio", float(ratio.mean()), 3), stat("Largest R-hat", max(float(r_[1]) for r_ in diag), 3), stat("Smallest effective sample size", min(float(r_[2]) for r_ in diag), 0)]
    tabs.append(table("MCMC convergence diagnostics (variance components)", ["Component", "R-hat", "Effective sample size", "Geweke z (chain 1)"],
                      [[r_[0], f(r_[1], 3), f(r_[2], 0), f(r_[3], 2)] for r_ in diag],
                      f"{nch} chains of {ndraw} draws after {burn} burn-in. The states are drawn jointly in one block, so mixing is usually good; the variances of weakly identified components mix slowest."))
    notes = [f"Series standardised for the prior; results in the original units. Inverse-gamma({pa:g}, {pb:g}) prior on the variance of each state component in standardised units (prior guess for its s.d.: {np.sqrt(pb):g} of the s.d. of the series) and a weak inverse-gamma({pa:g}, 0.001) prior on the irregular variance, diffuse prior on the initial states. "
             "States are sampled jointly from their banded Gaussian conditional (perturbation sampling), so no Kalman filter loop is needed. Missing values are treated as unobserved and the model fills them in."]
    return result("bsts", f"Bayesian structural time series: {y_}", int(obs.sum()), summary, tabs, charts, notes)


# ================================================================================================ Bayesian count models
def _count_ll(y, eta, family, r=None):
    """Pointwise log-likelihood of a Poisson (family 'poisson') or negative binomial (NB2, dispersion alpha = 1 / r) model."""
    from scipy.special import gammaln
    mu = np.exp(np.clip(eta, -30, 30))
    if family == "poisson":
        return y * eta - mu - gammaln(y + 1)
    return gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * (np.log(r) - np.log(r + mu)) + y * (eta - np.log(r + mu))


def _count_post(Z, y, off, family, s_b, b0):
    """Return log posterior and its gradient in (beta, log r) (log r only for the negative binomial)."""
    from scipy.special import digamma
    k = Z.shape[1]
    ps = np.r_[np.full(k, s_b), 2.0]
    def lp(th):
        b = th[:k]; eta = Z @ b + off
        r = np.exp(th[k]) if family == "nbinom" else None
        pr = -0.5 * (((b - b0) / s_b) ** 2).sum()
        if family == "nbinom":
            pr += -0.5 * ((th[k] - 1.0) / 2.0) ** 2
        return float(_count_ll(y, eta, family, r).sum() + pr)
    def grad(th):
        b = th[:k]; eta = Z @ b + off; mu = np.exp(np.clip(eta, -30, 30))
        if family == "poisson":
            return Z.T @ (y - mu) - (b - b0) / s_b ** 2
        r = np.exp(th[k])
        gb = Z.T @ ((y - mu) * r / (r + mu)) - (b - b0) / s_b ** 2
        gr = r * (digamma(y + r) - digamma(r) + np.log(r) - np.log(r + mu) + 1 - (r + y) / (r + mu)).sum() - (th[k] - 1.0) / 4.0
        return np.r_[gb, gr]
    return lp, grad


def _count_sample(Z, y, off, family, s_b, b0, ndraw, burn, rng, init_jitter=0.0):
    from scipy.optimize import minimize
    k = Z.shape[1]; d = k + (1 if family == "nbinom" else 0)
    lp, grad = _count_post(Z, y, off, family, s_b, b0)
    th0 = np.r_[b0 * 0 + np.r_[np.log(max(y.mean(), 0.1)), np.zeros(k - 1)], ([0.0] if family == "nbinom" else [])]
    res = minimize(lambda t: -lp(t), th0, jac=lambda t: -grad(t), method="BFGS", options=dict(maxiter=500, gtol=1e-6))
    m = res.x
    H = np.zeros((d, d)); e = 1e-4
    for i in range(d):
        a = m.copy(); a[i] += e; b_ = m.copy(); b_[i] -= e
        H[:, i] = -(grad(a) - grad(b_)) / (2 * e)
    H = (H + H.T) / 2
    try:
        Sig = np.linalg.inv(H + 1e-9 * np.eye(d)); Lc = np.linalg.cholesky(Sig)
    except np.linalg.LinAlgError:
        raise OESError("The posterior is too flat to build a proposal: drop collinear or near-constant regressors.")
    nu = 6.0
    def lprop(t):
        r_ = np.linalg.solve(Lc, t - m); return -(nu + d) / 2 * np.log1p(r_ @ r_ / nu)
    cur = m + init_jitter * (Lc @ rng.standard_normal(d)); lcur = lp(cur) - lprop(cur)
    out = np.empty((ndraw, d)); acc = 0
    for it in range(burn + ndraw):
        w = rng.chisquare(nu) / nu
        prop = m + Lc @ rng.standard_normal(d) / np.sqrt(w)
        lq = lp(prop) - lprop(prop)
        if np.log(rng.random()) < lq - lcur:
            cur, lcur = prop, lq; acc += it >= burn
        if it >= burn:
            out[it - burn] = cur
    return out, acc / ndraw, m


def bayes_count(df, spec):
    import statsmodels.api as sm
    from .bayes_extra import _rhat_ess
    from scipy.special import logsumexp
    y_ = need(spec, "y", "count outcome")
    x = as_list(need(spec, "x", "explanatory variables"))
    ex = spec.get("exposure")
    distinct([y_], x + ([ex] if ex else []))
    d = numeric(df, [y_] + x + ([ex] if ex else []))
    Y = d[y_].to_numpy(float)
    if (Y < 0).any() or (Y != np.round(Y)).any():
        raise OESError(f"{y_} must be a count: non-negative whole numbers.")
    if Y.max() == 0:
        raise OESError(f"{y_} is zero everywhere.")
    X = d[x].to_numpy(float); n, k = X.shape
    if (X.std(0) == 0).any():
        raise OESError("An explanatory variable is constant: drop it (the model has a constant).")
    if n < 10 * (k + 1):
        raise OESError(f"Too few observations ({n}) for {k} explanatory variables.")
    off = np.zeros(n)
    if ex:
        e_ = d[ex].to_numpy(float)
        if (e_ <= 0).any():
            raise OESError("The exposure must be positive.")
        off = np.log(e_)
    fam = str(spec.get("family", "nbinom"))
    if fam not in ("poisson", "nbinom"):
        raise OESError("Choose the family: poisson or nbinom.")
    ndraw = int_option(spec, "draws", 2000, 200, 50000, "Draws per chain")
    burn = int_option(spec, "burn", 500, 0, 50000, "Burn-in")
    nch = int_option(spec, "chains", 2, 2, 8, "Chains")
    seed = int_option(spec, "seed", 0, 0, 10 ** 9, "Random seed")
    sb = float_option(spec, "prior_sd", 5.0, "Prior standard deviation")
    if sb <= 0:
        raise OESError("The prior standard deviation must be positive.")
    xm, xs = X.mean(0), X.std(0)
    Z = np.column_stack([np.ones(n), (X - xm) / xs])
    b0 = np.r_[np.log(max((Y / np.exp(off)).mean(), 1e-3)), np.zeros(k)]
    fits = {}
    for fm in ("poisson", "nbinom"):
        ch = [_count_sample(Z, Y, off, fm, sb, b0, ndraw, burn, np.random.default_rng(seed + 1000 * c + (0 if fm == "poisson" else 7)), init_jitter=0.0 if c == 0 else 1.0) for c in range(nch)]
        fits[fm] = ch
    def to_orig(Th):
        B = Th[:, :k + 1].copy(); B[:, 1:] = Th[:, 1:k + 1] / xs; B[:, 0] = Th[:, 0] - B[:, 1:] @ xm
        return B
    ch = fits[fam]; Th = np.vstack([c[0] for c in ch]); acc = float(np.mean([c[1] for c in ch]))
    B = to_orig(Th); names = ["const"] + x
    mle = None
    try:
        Xc = sm.add_constant(X)
        if fam == "poisson":
            mle = sm.GLM(Y, Xc, family=sm.families.Poisson(), offset=off).fit()
        else:
            mle = sm.NegativeBinomial(Y, Xc, offset=off).fit(disp=0, maxiter=200)
    except Exception:
        mle = None
    rows = []
    for j, v in enumerate(names):
        irr = np.exp(B[:, j]); lo, hi = np.percentile(irr, [2.5, 97.5])
        rows.append([v, f(B[:, j].mean()), f(B[:, j].std(ddof=1)), f(irr.mean()) if j else "", f(lo) if j else "", f(hi) if j else "", f((B[:, j] > 0).mean(), 3) if j else "",
                     f(float(np.exp(mle.params[j]))) if (mle is not None and j) else ("" if j else f(float(mle.params[0]) if mle is not None else np.nan))])
    tabs = [table(f"Posterior of the coefficients ({'Poisson' if fam == 'poisson' else 'negative binomial'})",
                  ["Variable", "Coefficient (mean)", "Std. dev.", "Rate ratio (mean)", "2.5%", "97.5%", "P(> 0)", "Maximum likelihood rate ratio"], rows,
                  "Rate ratio = exp(coefficient): the multiplicative change in the expected count for a one-unit increase in the variable. The 2.5% and 97.5% columns are the interval of the rate ratio. The last column is the classical estimate from statsmodels.")]
    if fam == "nbinom":
        al = np.exp(-Th[:, k + 1]); a_lo, a_hi = np.percentile(al, [2.5, 97.5])
        mal = float(mle.params[-1]) if mle is not None else np.nan
        tabs.append(table("Overdispersion", ["Parameter", "Mean", "Median", "2.5%", "97.5%", "Maximum likelihood"],
                          [["alpha (variance = mean + alpha * mean^2)", f(al.mean()), f(np.median(al)), f(a_lo), f(a_hi), f(mal)]],
                          "alpha = 0 is the Poisson model. A posterior far from zero means that counts vary more than a Poisson allows."))
    # posterior predictive checks and WAIC for both families
    rng = np.random.default_rng(seed + 99)
    sel = rng.choice(len(Th), size=min(400, len(Th)), replace=False)
    pp = {}; waic = {}; lppd_i = {}
    for fm in ("poisson", "nbinom"):
        T_ = np.vstack([c[0] for c in fits[fm]]); S_ = T_[rng.choice(len(T_), size=min(600, len(T_)), replace=False)]
        LL = np.empty((len(S_), n)); rep = np.empty((len(S_), n))
        for s_i, t in enumerate(S_):
            eta = Z @ t[:k + 1] + off
            r_ = np.exp(t[k + 1]) if fm == "nbinom" else None
            LL[s_i] = _count_ll(Y, eta, fm, r_)
            mu = np.exp(np.clip(eta, -30, 30))
            rep[s_i] = rng.poisson(mu) if fm == "poisson" else rng.poisson(rng.gamma(r_, mu / r_))
        lp_i = logsumexp(LL, axis=0) - np.log(len(S_)); pw = LL.var(axis=0, ddof=1)
        el = lp_i - pw; waic[fm] = (el.sum(), pw.sum()); lppd_i[fm] = el; pp[fm] = rep
    zero_o = (Y == 0).mean(); var_o = Y.var(ddof=1); mx_o = Y.max()
    rows = []
    for fm, lab in (("poisson", "Poisson"), ("nbinom", "Negative binomial")):
        rep = pp[fm]; z = (rep == 0).mean(1); v = rep.var(1, ddof=1); mxx = rep.max(1)
        rows.append([lab, f(zero_o, 4), f(z.mean(), 4), f(min((z >= zero_o).mean(), (z <= zero_o).mean()) * 2, 3),
                     f(var_o, 3), f(v.mean(), 3), f(min((v >= var_o).mean(), (v <= var_o).mean()) * 2, 3),
                     f(mx_o, 0), f(mxx.mean(), 1)])
    tabs.append(table("Posterior predictive check", ["Model", "Zeros (observed)", "Zeros (predicted)", "p-value", "Variance (observed)", "Variance (predicted)", "p-value", "Max (observed)", "Max (predicted)"], rows,
                      "Counts are simulated from the posterior and compared with the data. A p-value near 0 (two-sided) means the model cannot reproduce that feature of the data."))
    dd = lppd_i["nbinom"] - lppd_i["poisson"]
    tabs.append(table("Model comparison (WAIC, higher elpd is better)", ["Model", "elpd (WAIC)", "Effective parameters", "Difference from the best", "Std. error of the difference"],
                      [[lab, f(waic[fm][0], 1), f(waic[fm][1], 1), f(waic[fm][0] - max(waic["poisson"][0], waic["nbinom"][0]), 1), (f(float(np.sqrt(n * dd.var(ddof=1))), 1) if waic[fm][0] < max(waic["poisson"][0], waic["nbinom"][0]) else "")]
                       for fm, lab in (("poisson", "Poisson"), ("nbinom", "Negative binomial"))],
                      "WAIC (Watanabe 2010) estimates the out-of-sample log predictive density. A difference of more than two or three standard errors is clear."))
    # average marginal effects on the expected count
    Bs = B[sel]; Xo = np.column_stack([np.ones(n), X]); mus = np.exp(np.clip(Xo @ Bs.T + off[:, None], -30, 30))
    ame = np.array([(Bs[:, j] * mus).mean(0) for j in range(1, k + 1)])
    tabs.append(table("Average marginal effects on the expected count", ["Variable", "Mean", "Std. dev.", "2.5%", "97.5%"],
                      [[x[j], f(ame[j].mean()), f(ame[j].std(ddof=1)), f(np.percentile(ame[j], 2.5)), f(np.percentile(ame[j], 97.5))] for j in range(k)],
                      "Change in the expected count per one-unit increase, averaged over the sample (exposure included): the rate ratio translated to counts."))
    diag = []
    for j, v in enumerate(names):
        rh, ess = _rhat_ess([to_orig(c[0])[:, j] for c in ch]); diag.append([v, rh, ess])
    if fam == "nbinom":
        rh, ess = _rhat_ess([np.exp(-c[0][:, k + 1]) for c in ch]); diag.append(["alpha", rh, ess])
    wst = sorted(diag, key=lambda r_: r_[2])[:8]
    tabs.append(table("MCMC convergence diagnostics", ["Parameter", "R-hat", "Effective sample size"], [[r_[0], f(r_[1], 3), f(r_[2], 0)] for r_ in wst],
                      f"The {len(wst)} parameters with the smallest effective sample size (of {len(diag)}). Independence Metropolis-Hastings with a Student-t proposal at the posterior mode; acceptance rate {acc:.2f}."))
    kmax = int(min(np.percentile(Y, 99.5), 15)) + 1
    ks = np.arange(0, kmax + 1)
    obs_p = [(Y == i).mean() for i in ks]
    charts = [chart(f"Distribution of {y_}: data and posterior predictive", [str(i) for i in ks],
                    [("Data", obs_p), ("Poisson", [(pp["poisson"] == i).mean() for i in ks]), ("Negative binomial", [(pp["nbinom"] == i).mean() for i in ks])], "count", "share of observations", kind="bar")]
    order = np.argsort(-np.abs(Th[:, 1:k + 1].mean(0)))
    charts.append(chart("Rate ratio of a one-s.d. increase in each regressor (posterior mean)", [x[j] for j in order], [("Rate ratio", [float(np.exp(Th[:, 1 + j]).mean()) for j in order])], "", "rate ratio", kind="bar"))
    summary = [stat("Observations", n), stat("Family", "Poisson" if fam == "poisson" else "negative binomial"), stat("Mean of the count", float(Y.mean()), 3), stat("Variance of the count", float(var_o), 3),
               stat("Share of zeros", float(zero_o), 3), stat("Acceptance rate", acc, 2), stat("Largest R-hat", max(float(r_[1]) for r_ in diag), 3), stat("Smallest effective sample size", min(float(r_[2]) for r_ in diag), 0)]
    return result("bayes_count", f"Bayesian count regression: {y_}", n, summary, tabs, charts,
                  [f"Regressors are standardised for the prior (N(0, {sb:g}^2) on each slope; the constant is centred at the log mean count" + ("; the log of the overdispersion parameter has a N(1, 2^2) prior on log(1/alpha)" if fam == "nbinom" else "") +
                   f"). {nch} chains of {ndraw} draws after {burn} burn-in. Both families are fitted for the predictive check and the WAIC comparison, whichever one is shown above."])


# ================================================================================================ Bayesian Markov-switching regression
def _ms_ffbs(lik, P, rng):
    """Hamilton filter and backward sampling of the regime path (Chib 1996). lik is T x K (densities, not logs)."""
    T, K = lik.shape
    pi = np.linalg.solve(np.vstack([(np.eye(K) - P).T, np.ones(K)]).T @ np.vstack([(np.eye(K) - P).T, np.ones(K)]) + 1e-12 * np.eye(K), np.vstack([(np.eye(K) - P).T, np.ones(K)]).T @ np.r_[np.zeros(K), 1.0])
    pi = np.clip(pi, 1e-8, None); pi /= pi.sum()
    al = np.empty((T, K)); pred = pi
    for t in range(T):
        a = pred * lik[t]; s = a.sum()
        a = a / s if s > 0 else np.full(K, 1.0 / K)
        al[t] = a; pred = a @ P
    s = np.empty(T, dtype=int)
    s[-1] = rng.choice(K, p=al[-1])
    for t in range(T - 2, -1, -1):
        w = al[t] * P[:, s[t + 1]]; tot = w.sum()
        s[t] = rng.choice(K, p=w / tot) if tot > 0 else rng.choice(K)
    return s


def _ms_chain(ys, W, K, sw_slopes, sw_var, order, ndraw, burn, rng, prior_sd, b0v, a0v, stay, init_s):
    """Gibbs sampler: regime path by forward filtering backward sampling, regression coefficients by (weighted) normal updates,
    variances by inverse gammas, transition rows by Dirichlet draws. Labels are fixed by ordering the regime intercepts (or variances)."""
    T = len(ys); q = W.shape[1]
    nb = K * (1 + q) if sw_slopes else K + q
    s = init_s.copy()
    sig2 = np.ones(K)
    P = np.full((K, K), 1.0 / K) + stay * np.eye(K); P /= P.sum(1, keepdims=True)
    beta = np.zeros(nb)
    out = dict(mu=[], slp=[], sig=[], P=[], S=np.zeros((T, K)), relabel=0)
    def design(s):
        D = np.zeros((T, K)); D[np.arange(T), s] = 1.0
        if sw_slopes:
            return np.hstack([D] + [D * W[:, [j]] for j in range(q)])
        return np.hstack([D, W])
    def unpack(b):
        if sw_slopes:
            return b[:K], np.vstack([b[K * (1 + j):K * (2 + j)] for j in range(q)]).T if q else np.zeros((K, 0))
        return b[:K], (np.tile(b[K:], (K, 1)) if q else np.zeros((K, 0)))
    for it in range(burn + ndraw):
        mu, sl = unpack(beta)
        mean_t = mu[None, :] + W @ sl.T if q else np.tile(mu, (T, 1))                 # T x K
        lik = np.exp(-0.5 * (ys[:, None] - mean_t) ** 2 / sig2[None, :]) / np.sqrt(sig2)[None, :]
        s = _ms_ffbs(lik, P, rng)
        Z = design(s)
        wv = 1.0 / sig2[s]
        prec = (Z * wv[:, None]).T @ Z + np.eye(nb) / prior_sd ** 2
        cov = np.linalg.inv(prec); cov = (cov + cov.T) / 2
        beta = rng.multivariate_normal(cov @ ((Z * wv[:, None]).T @ ys), cov)
        mu, sl = unpack(beta)
        res = ys - mu[s] - ((W * sl[s]).sum(1) if q else 0.0)
        if sw_var:
            for j in range(K):
                m = s == j
                sig2[j] = _ig(rng, a0v + 0.5 * m.sum(), b0v + 0.5 * float((res[m] ** 2).sum()))
        else:
            sig2[:] = _ig(rng, a0v + 0.5 * T, b0v + 0.5 * float((res ** 2).sum()))
        cnt = np.zeros((K, K))
        np.add.at(cnt, (s[:-1], s[1:]), 1.0)
        for i in range(K):
            P[i] = rng.dirichlet(1.0 + stay * (np.arange(K) == i) + cnt[i])
        key = mu if order == "mean" else sig2
        perm = np.argsort(key)
        if (perm != np.arange(K)).any():
            inv = np.empty(K, dtype=int); inv[perm] = np.arange(K)
            s = inv[s]; sig2 = sig2[perm]; P = P[np.ix_(perm, perm)]
            if sw_slopes:
                beta = np.concatenate([beta[K * b_:K * (b_ + 1)][perm] for b_ in range(1 + q)])
            else:
                beta = np.r_[beta[:K][perm], beta[K:]]
            if it >= burn:
                out["relabel"] += 1
        if it >= burn:
            mu, sl = unpack(beta)
            out["mu"].append(mu.copy()); out["slp"].append(sl.copy()); out["sig"].append(np.sqrt(sig2).copy()); out["P"].append(P.copy())
            out["S"][np.arange(T), s] += 1.0
    out["S"] /= ndraw
    return {k_: (np.array(v) if k_ in ("mu", "slp", "sig", "P") else v) for k_, v in out.items()}


def bayes_ms(df, spec):
    from .bayes_extra import _rhat_ess
    from .timeseries import _series_frame
    import statsmodels.api as sm
    y_ = need(spec, "y", "dependent variable")
    xs = as_list(spec.get("x"))
    distinct([y_], xs)
    ar = int_option(spec, "ar", 0, 0, 8, "Autoregressive lags")
    K = int_option(spec, "regimes", 2, 2, 4, "Number of regimes")
    sw_s = str(spec.get("switching_slopes", "no")).lower() in ("yes", "true", "1")
    sw_v = str(spec.get("switching_variance", "yes")).lower() in ("yes", "true", "1")
    ref = spec.get("reference")
    cols_ = [y_] + xs + ([ref] if ref else [])
    d, labels = _series_frame(df, spec, cols_)
    if ref and ref not in d.columns:
        raise OESError(f"Reference column {ref} not found.")
    yv = d[y_].to_numpy(float)
    n0 = len(yv)
    if n0 < 40 + 10 * (ar + len(xs)):
        raise OESError(f"Too few observations ({n0}) for {K} regimes with {ar} lags and {len(xs)} regressors: at least {40 + 10 * (ar + len(xs))} are needed.")
    if yv.std() == 0:
        raise OESError("The series is constant.")
    cols_w, names_w = [], []
    for j in range(1, ar + 1):
        cols_w.append(np.r_[np.full(j, np.nan), yv[:-j]]); names_w.append(f"{y_}(t-{j})")
    for v in xs:
        cols_w.append(d[v].to_numpy(float)); names_w.append(v)
    Wm = np.column_stack(cols_w) if cols_w else np.zeros((n0, 0))
    keep = ~np.isnan(Wm).any(1) if cols_w else np.ones(n0, bool)
    yv, Wm = yv[keep], Wm[keep]; lab = [l for l, k_ in zip(labels, keep) if k_]
    refv = d[ref].to_numpy(float)[keep] if ref else None
    n = len(yv)
    if sw_s and not names_w:
        raise OESError("There are no slopes to switch: add regressors or autoregressive lags, or set switching slopes to no.")
    order = str(spec.get("order", "mean" if True else "variance"))
    if order not in ("mean", "variance"):
        raise OESError("Order the regimes by mean or variance.")
    if order == "variance" and not sw_v:
        raise OESError("Ordering by variance needs a switching variance.")
    ndraw = int_option(spec, "draws", 3000, 200, 30000, "Draws per chain")
    burn = int_option(spec, "burn", 1000, 0, 30000, "Burn-in")
    nch = int_option(spec, "chains", 2, 2, 6, "Chains")
    seed = int_option(spec, "seed", 0, 0, 10 ** 9, "Random seed")
    psd = float_option(spec, "prior_sd", 5.0, "Prior standard deviation")
    stay = float_option(spec, "prior_stay", 0.0, "Extra prior weight on staying in a regime")
    if psd <= 0 or stay < 0:
        raise OESError("The prior standard deviation must be positive and the prior weight on staying cannot be negative.")
    ym, ysd = yv.mean(), yv.std()
    wm, wsd = (Wm.mean(0), np.where(Wm.std(0) > 0, Wm.std(0), 1.0)) if Wm.shape[1] else (np.zeros(0), np.ones(0))
    ys = (yv - ym) / ysd; Ws = (Wm - wm) / wsd
    qn = np.quantile(ys, np.linspace(0, 1, K + 1)[1:-1]) if K > 1 else []
    chains = []
    for c in range(nch):
        rng = np.random.default_rng(seed + 1000 * c)
        init = np.digitize(ys, qn) if c == 0 else rng.integers(0, K, n)
        chains.append(_ms_chain(ys, Ws, K, sw_s, sw_v, order, ndraw, burn, rng, psd, 1.0, 2.0, stay, init))
    MU = np.vstack([c["mu"] for c in chains]); SL = np.vstack([c["slp"] for c in chains]); SG = np.vstack([c["sig"] for c in chains]); PP = np.vstack([c["P"] for c in chains])
    S = np.mean([c["S"] for c in chains], axis=0)
    q = Ws.shape[1]
    def orig(MUc, SLc, SGc):
        slo = SLc * ysd / wsd[None, None, :] if q else SLc
        mu = ym + ysd * MUc                                      # intercept at the mean of the regressors
        return mu, slo, SGc * ysd
    mu_o, sl_o, sg_o = orig(MU, SL, SG)
    # maximum-likelihood comparison (statsmodels), regimes matched by the ordering variable
    ml = {}
    try:
        endog = pd.Series(yv); exo = pd.DataFrame(Wm) if Wm.shape[1] else None
        mr = sm.tsa.MarkovRegression(endog, k_regimes=K, trend="c", exog=exo, switching_exog=sw_s, switching_variance=sw_v)
        best = None
        for sd_ in (0, 1):
            try:
                r_ = mr.fit(disp=False, maxiter=400, search_reps=20, search_scale=1.0) if sd_ == 0 else mr.fit(disp=False, maxiter=400)
                best = r_ if best is None or r_.llf > best.llf else best
            except Exception:
                pass
        if best is not None:
            pr = best.params
            cst = np.array([pr[f"const[{j}]"] for j in range(K)])
            sg = np.sqrt(np.array([pr[f"sigma2[{j}]"] for j in range(K)])) if sw_v else np.full(K, np.sqrt(pr["sigma2"]))
            key = cst if order == "mean" else sg
            o_ = np.argsort(key); ml = dict(mu=cst[o_], sg=sg[o_], llf=float(best.llf))
            pm = np.array(best.regime_transition)[..., 0] if False else None
    except Exception:
        ml = {}
    rows = []
    for j in range(K):
        for nm, arr, mlv in (("Intercept" if not q else "Level at the mean of the regressors", mu_o[:, j], ml.get("mu", [None] * K)[j] if (ml and not q) else None), ("Std. deviation", sg_o[:, j] if sw_v else sg_o[:, 0], ml.get("sg", [None] * K)[j] if (ml and not q) else None)):
            if nm == "Std. deviation" and not sw_v and j > 0:
                continue
            rows.append([f"Regime {j + 1}" if not (nm == "Std. deviation" and not sw_v) else "All regimes", nm, f(arr.mean()), f(arr.std(ddof=1)), f(np.percentile(arr, 2.5)), f(np.percentile(arr, 97.5)), f(mlv) if mlv is not None else "n/a"])
        if q:
            for i, nm in enumerate(names_w):
                if sw_s or j == 0:
                    a_ = sl_o[:, j, i]; rows.append([f"Regime {j + 1}" if sw_s else "All regimes", nm, f(a_.mean()), f(a_.std(ddof=1)), f(np.percentile(a_, 2.5)), f(np.percentile(a_, 97.5)), "n/a"])
    tabs = [table("Posterior of the regime parameters", ["Regime", "Parameter", "Mean", "Std. dev.", "2.5%", "97.5%", "Maximum likelihood"], rows,
                  f"Regimes ordered by {'their intercepts' if order == 'mean' else 'their variances'}, which fixes the labelling. The maximum-likelihood column is the classical estimate from statsmodels, matched by the same ordering.")]
    prow = []
    for i in range(K):
        for j in range(K):
            v = PP[:, i, j]; prow.append([f"{i + 1} -> {j + 1}", f(v.mean(), 4), f(v.std(ddof=1), 4), f(np.percentile(v, 2.5), 4), f(np.percentile(v, 97.5), 4)])
    tabs.append(table("Transition probabilities", ["From -> to", "Mean", "Std. dev.", "2.5%", "97.5%"], prow, "Probability of moving from the regime in one period to the regime in the next."))
    dur = 1.0 / np.clip(1 - PP[:, np.arange(K), np.arange(K)], 1e-9, None)
    erg = np.empty((len(PP), K))
    for i, P_ in enumerate(PP):
        A = np.vstack([(np.eye(K) - P_).T, np.ones(K)]); erg[i] = np.linalg.lstsq(A, np.r_[np.zeros(K), 1.0], rcond=None)[0]
    tabs.append(table("Expected duration and long-run share of each regime", ["Regime", "Duration (periods), mean", "median", "2.5%", "97.5%", "Long-run share, mean", "2.5%", "97.5%"],
                      [[f"Regime {j + 1}", f(dur[:, j].mean(), 2), f(np.median(dur[:, j]), 2), f(np.percentile(dur[:, j], 2.5), 2), f(np.percentile(dur[:, j], 97.5), 2),
                        f(erg[:, j].mean(), 3), f(np.percentile(erg[:, j], 2.5), 3), f(np.percentile(erg[:, j], 97.5), 3)] for j in range(K)],
                      "Duration = 1 / (1 - p_jj). The long-run share is the stationary distribution of the chain; its interval comes from the posterior of the transition matrix."))
    # episodes of regime 1
    ep = []; low = S[:, 0] > 0.5; t = 0
    while t < n:
        if low[t]:
            u = t
            while u + 1 < n and low[u + 1]:
                u += 1
            ep.append([lab[t], lab[u], u - t + 1, f(S[t:u + 1, 0].mean(), 3)]); t = u + 1
        else:
            t += 1
    if ep:
        tabs.append(table("Episodes of regime 1 (smoothed probability above 0.5)", ["From", "To", "Length", "Mean probability"], ep[:40],
                          "Dated from the full-sample posterior probabilities; an episode is a run of periods in which regime 1 is more likely than not."))
    if ref:
        rv = refv.astype(float); dec = (S[:, 0] > 0.5).astype(float)
        agree = float((dec == rv).mean()); tp = float(((dec == 1) & (rv == 1)).sum() / max(rv.sum(), 1)); fp = float(((dec == 1) & (rv == 0)).sum() / max((rv == 0).sum(), 1))
        tabs.append(table(f"Agreement with the reference series {ref}", ["Measure", "Value"],
                          [["Share of periods classified the same", f(agree, 3)], ["Reference periods that fall in regime 1", f(tp, 3)], ["Other periods that fall in regime 1", f(fp, 3)],
                           ["Correlation of the regime-1 probability with the reference", f(float(np.corrcoef(S[:, 0], rv)[0, 1]), 3)]]))
    diag = []
    for j in range(K):
        for nm, a_ in (("Intercept", [orig(c["mu"], c["slp"], c["sig"])[0][:, j] for c in chains]), ("Std. deviation", [orig(c["mu"], c["slp"], c["sig"])[2][:, j] if sw_v else orig(c["mu"], c["slp"], c["sig"])[2][:, 0] for c in chains])):
            rh, ess = _rhat_ess(a_); diag.append([f"Regime {j + 1}", nm, rh, ess])
        rh, ess = _rhat_ess([c["P"][:, j, j] for c in chains]); diag.append([f"Regime {j + 1}", "Staying probability", rh, ess])
    relab = np.mean([c["relabel"] / ndraw for c in chains])
    tabs.append(table("MCMC convergence diagnostics", ["Regime", "Parameter", "R-hat", "Effective sample size"], [[r_[0], r_[1], f(r_[2], 3), f(r_[3], 0)] for r_ in diag],
                      f"{nch} chains of {ndraw} draws after {burn} burn-in. In {100 * relab:.1f}% of the draws the labels had to be reordered to keep the regimes ordered; a high share means that the regimes are weakly separated."))
    charts = [chart("Smoothed probability of each regime", lab, [(f"Regime {j + 1}", S[:, j]) for j in range(K)], "", "probability")]
    mean_path = (S * (ym + ysd * MU.mean(0))[None, :]).sum(1) if not q else None
    if q == 0:
        charts.append(chart(f"{y_} and the regime-weighted mean", lab, [("Series", yv), ("Posterior mean of the regime mean", mean_path)], "", y_))
    else:
        fit = np.array([(S[t_] * (mu_o.mean(0) + (sl_o.mean(0) * (Wm[t_] - wm)[None, :]).sum(1))).sum() for t_ in range(n)])
        charts.append(chart(f"{y_} and the regime-weighted fit", lab, [("Series", yv), ("Fit", fit)], "", y_))
    summary = [stat("Observations", n), stat("Regimes", K), stat("Lags and regressors", len(names_w)), stat("Largest R-hat", max(float(r_[2]) for r_ in diag), 3),
               stat("Smallest effective sample size", min(float(r_[3]) for r_ in diag), 0), stat("Draws relabelled (%)", 100 * float(relab), 1)]
    notes = [f"Bayesian Markov-switching regression (Albert and Chib 1993; Chib 1996; Kim and Nelson 1999): the regime path is sampled by forward filtering and backward sampling, "
             f"the coefficients by a normal update, the variances by inverse gammas and each row of the transition matrix by a Dirichlet draw. Prior: N(0, {psd:g}^2) on standardised coefficients, "
             f"inverse-gamma(2, 1) on standardised variances, Dirichlet(1" + (f" + {stay:g} on the diagonal" if stay else "") + ") on each row. Intercepts (levels at the mean of the regressors)" + (", slopes" if sw_s else "") + (" and variances" if sw_v else "") + " switch."]
    return result("bayes_ms", f"Bayesian Markov-switching regression: {y_}", n, summary, tabs, charts, notes)


# ================================================================================================ Markov chain transition model
def _stationary(P):
    K = P.shape[-1]
    A = np.vstack([np.swapaxes(P, -1, -2) - np.eye(K), np.ones(K)]) if P.ndim == 2 else None
    b = np.zeros(K + 1)
    b[-1] = 1.0
    return np.linalg.lstsq(A, b, rcond=None)[0]


def _first_passage(P):
    K = P.shape[0]
    M = np.zeros((K, K))
    for j in range(K):
        idx = [i for i in range(K) if i != j]
        Q = P[np.ix_(idx, idx)]
        m = np.linalg.solve(np.eye(K - 1) - Q, np.ones(K - 1))
        M[idx, j] = m
    for j in range(K):                                          # mean recurrence time 1/pi_j on the diagonal
        M[j, j] = 1 + P[j, [i for i in range(K) if i != j]] @ M[[i for i in range(K) if i != j], j]
    return M


def bayes_mc(df, spec):
    st = need(spec, "state", "state variable")
    idc = spec.get("id") or None
    tm = spec.get("time") or None
    bins = int_option(spec, "bins", 4, 2, 10, "Number of states for a continuous variable")
    draws = int_option(spec, "draws", 4000, 500, 50000, "Posterior draws")
    seed = int_option(spec, "seed", 0, 0, 10 ** 9, "Random seed")
    a0 = float_option(spec, "prior", 1.0, "Dirichlet prior weight per cell")
    cols_ = [st] + ([idc] if idc else []) + ([tm] if tm else [])
    check_columns(df, cols_)
    d = df[cols_].dropna().copy()
    if tm:
        d = d.sort_values(([idc] if idc else []) + [tm], kind="stable")
    s = d[st]
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().all() and sn.nunique() > 12:
        codes = pd.qcut(sn.rank(method="first"), bins, labels=False)
        lo = sn.groupby(codes).min()
        hi = sn.groupby(codes).max()
        names = [f"S{i + 1} [{lo[i]:.3g}, {hi[i]:.3g}]" for i in range(bins)]
        d["_s"] = codes.astype(int)
    else:
        cats = sorted(s.unique(), key=lambda v: (str(type(v)), v))
        names = [str(c) for c in cats]
        d["_s"] = s.map({c: i for i, c in enumerate(cats)}).astype(int)
    K = len(names)
    if K < 2:
        raise OESError("The state variable has only one value.")
    g = d.groupby(idc) if idc else [(None, d)]
    N = np.zeros((K, K))
    Nt = {}
    for _, gg in g:
        v = gg["_s"].to_numpy()
        tv = gg[tm].to_numpy() if tm else np.arange(len(v))
        for a, b, t in zip(v[:-1], v[1:], tv[1:]):
            N[a, b] += 1
            Nt.setdefault(t, np.zeros((K, K)))[a, b] += 1
    n = int(N.sum())
    if n < 10 * K:
        raise OESError(f"Only {n} transitions for {K} states: at least {10 * K} are needed.")
    if (N.sum(1) == 0).any() and a0 == 0:
        raise OESError("A state is never left in the data; use a positive prior weight.")
    rng = np.random.default_rng(seed)
    A = N + a0
    Pm = A / A.sum(1, keepdims=True)
    P_ml = N / np.maximum(N.sum(1, keepdims=True), 1)
    Pd = np.stack([rng.dirichlet(A[i], size=draws) for i in range(K)], axis=1)       # draws x K x K
    PI = np.array([_stationary(p) for p in Pd])
    FP = np.array([_first_passage(p) for p in Pd])
    sh = (K - np.trace(Pd, axis1=1, axis2=2)) / (K - 1)
    dur = 1 / np.maximum(1 - np.diagonal(Pd, axis1=1, axis2=2), 1e-9)
    q = lambda x: np.percentile(x, [2.5, 97.5], axis=0)
    # tests
    rowN = N.sum(1)
    colp = N.sum(0) / N.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        ll1 = np.nansum(np.where(N > 0, N * np.log(P_ml), 0.0))
        ll0 = np.nansum(np.where(N > 0, N * np.log(colp)[None, :], 0.0))
    lr0 = 2 * (ll1 - ll0)
    df0 = (K - 1) ** 2
    tests = [["Independence (zero-order) against first-order Markov", lr0, df0, stats.chi2.sf(lr0, df0)]]
    ts = sorted(Nt)
    if len(ts) >= 4:
        h = len(ts) // 2
        N1 = sum(Nt[t] for t in ts[:h])
        N2 = sum(Nt[t] for t in ts[h:])
        def _ll(M):
            P = M / np.maximum(M.sum(1, keepdims=True), 1)
            return float(np.sum(np.where(M > 0, M * np.log(np.where(P > 0, P, 1)), 0.0)))
        lrh = 2 * (_ll(N1) + _ll(N2) - _ll(N1 + N2))
        dfh = int(((N1.sum(1) > 0) & (N2.sum(1) > 0)).sum() * (K - 1))
        tests.append([f"Time homogeneity: first against second half of the periods", lrh, dfh, stats.chi2.sf(lrh, dfh)])
    rows = []
    for i in range(K):
        for j in range(K):
            lo_, hi_ = np.percentile(Pd[:, i, j], [2.5, 97.5])
            rows.append([names[i], names[j], int(N[i, j]), P_ml[i, j], Pd[:, i, j].mean(), Pd[:, i, j].std(), lo_, hi_])
    tabs = [table("Transition probabilities (row = from, column = to)", ["From", "To", "Count", "ML", "Posterior mean", "Posterior s.d.", "2.5%", "97.5%"], rows,
                  f"Dirichlet({a0:g}) prior on each row; {draws} posterior draws; {n} transitions."),
            table("Stationary distribution (long-run shares) and persistence",
                  ["State", "Observed share", "Long-run share", "2.5%", "97.5%", "Stay probability", "Expected duration", "Dur. 2.5%", "Dur. 97.5%"],
                  [[names[i], (d["_s"] == i).mean(), PI[:, i].mean(), q(PI)[0][i], q(PI)[1][i], Pd[:, i, i].mean(), dur[:, i].mean(), q(dur)[0][i], q(dur)[1][i]] for i in range(K)],
                  "Duration = 1 / (1 - stay probability), in periods."),
            table("Mean first-passage times (periods to reach the column state, from the row state)", ["From \\ To"] + names,
                  [[names[i]] + [f"{np.median(FP[:, i, j]):.2f}" + f" ({np.percentile(FP[:, i, j], 2.5):.1f}-{np.percentile(FP[:, i, j], 97.5):.1f})" for j in range(K)] for i in range(K)],
                  "Posterior median and 95% interval. The diagonal is the mean recurrence time."),
            table("Mobility", ["Index", "ML", "Posterior mean", "2.5%", "97.5%"],
                  [["Shorrocks (K - trace P)/(K - 1)", (K - np.trace(P_ml)) / (K - 1), sh.mean(), *np.percentile(sh, [2.5, 97.5])]]),
            table("Specification tests", ["Test", "LR statistic", "df", "p-value"], tests)]
    ch = [chart("Long-run shares and observed shares", names, [("Observed", [(d["_s"] == i).mean() for i in range(K)]), ("Long-run (posterior mean)", PI.mean(0))], "State", "Share", "bar"),
          chart("Probability of staying in each state", names, [("Stay probability", [Pd[:, i, i].mean() for i in range(K)])], "State", "Probability", "bar",
                band=([q(np.stack([Pd[:, i, i] for i in range(K)], 1))[0][i] for i in range(K)], [q(np.stack([Pd[:, i, i] for i in range(K)], 1))[1][i] for i in range(K)]))]
    summ = [stat("Transitions", n), stat("States", K), stat("Shorrocks mobility", float(sh.mean()), 3), stat("Mixing time (1/(1-2nd eigenvalue))", float(1 / max(1 - np.sort(np.abs(np.linalg.eigvals(Pm)))[-2], 1e-9)), 2)]
    return result("bayes_mc", "Markov chain transition model (Bayesian Dirichlet posterior)", n, summ, tabs, ch,
                  [f"States: {', '.join(names)}.", "Transitions are counted within each unit in time order. Long-run shares and first-passage times are computed for every posterior draw of the matrix."])
