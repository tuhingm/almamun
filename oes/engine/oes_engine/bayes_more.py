"""More Bayesian models: Bayesian model averaging (g-prior, exact enumeration or MC3). Same result format as the other models (core.py)."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

from .core import (OESError, as_list, chart, distinct, f, int_option, float_option, need, result, stat, table, numeric)


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
