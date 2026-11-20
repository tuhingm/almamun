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
