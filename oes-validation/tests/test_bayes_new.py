"""Validation of Bayesian models added after the first release."""
import itertools, os, sys, warnings
import numpy as np, pandas as pd, pytest
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "oes", "engine")); sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import oes_engine as o
from oes_engine import bayes_more
from oes_engine.core import OESError

BX = os.path.join(ROOT, "bayesian-guide", "examples")
def tab(r, title): return next(t for t in r["tables"] if t["title"].startswith(title))
def S(r): return {x["label"]: x["value"] for x in r["summary"]}


def _reg_data(n=300, k=6, beta=(1.0, -0.8, 0.6), seed=0, sd=1.0):
    rng = np.random.default_rng(seed); X = rng.normal(size=(n, k))
    y = X[:, :len(beta)] @ np.array(beta) + rng.normal(scale=sd, size=n)
    return pd.DataFrame(np.column_stack([y, X]), columns=["y"] + [f"x{i}" for i in range(k)])


def test_bma_registered():
    m = o.BY_ID["bma"]; assert m["group"] == "Bayesian" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_bma_matches_brute_force_posterior():
    """PIPs, model probabilities and shrunk coefficients against a plain re-implementation of Liang et al. (2008)."""
    d = _reg_data(n=120, k=4, seed=3); x = ["x0", "x1", "x2", "x3"]; n, k = 120, 4; g = max(n, k * k)
    r = bayes_more.bma(d, dict(y="y", x=x, g="bric", model_prior="uniform"))
    Y = d["y"].to_numpy(); X = d[x].to_numpy(); sst = ((Y - Y.mean()) ** 2).sum()
    lw, b_of = {}, {}
    for sub in itertools.product([0, 1], repeat=k):
        idx = [j for j in range(k) if sub[j]]; p = len(idx)
        if p:
            Z = np.column_stack([np.ones(n), X[:, idx]]); bhat = np.linalg.lstsq(Z, Y, rcond=None)[0]; e = Y - Z @ bhat; R2 = 1 - e @ e / sst
        else:
            R2, bhat = 0.0, None
        lw[sub] = (n - 1 - p) / 2 * np.log(1 + g) - (n - 1) / 2 * np.log(1 + g * (1 - R2)); b_of[sub] = (idx, bhat)
    mx = max(lw.values()); w = {s: np.exp(v - mx) for s, v in lw.items()}; tot = sum(w.values()); w = {s: v / tot for s, v in w.items()}
    pip = [sum(v for s, v in w.items() if s[j]) for j in range(k)]
    got = {row[0]: row[1] for row in tab(r, "Posterior inclusion")["rows"]}
    for j in range(k):
        assert got[x[j]] == pytest.approx(pip[j], abs=1e-3)
    bm = np.zeros(k)
    for s, v in w.items():
        idx, bhat = b_of[s]
        if idx: bm[idx] += v * g / (1 + g) * bhat[1:]
    gotm = {row[0]: row[2] for row in tab(r, "Posterior inclusion")["rows"]}
    for j in range(k):
        assert gotm[x[j]] == pytest.approx(bm[j], abs=2e-6)
    best = max(w, key=w.get); assert tab(r, "Best")["rows"][0][1] == pytest.approx(w[best], abs=1e-4)


def test_bma_bayes_factor_matches_numerical_integration():
    """The closed-form Bayes factor against numerical integration of the marginal likelihood over the error variance
    (g-prior on the slopes, flat prior on the constant, p(sigma^2) proportional to 1/sigma^2)."""
    from scipy.integrate import quad
    n, sst = 80, 37.0
    for R2, p, g in [(0.3, 2, 80.0), (0.55, 3, 400.0), (0.05, 1, 80.0)]:
        def ld(t, p_, r2):                                   # log density of the centred data given sigma^2 = exp(t), times the 1/sigma^2 prior (Jacobian 1)
            s2 = np.exp(t)
            return -(n - 1) / 2 * np.log(2 * np.pi * s2) - p_ / 2 * np.log(1 + g) - ((1 - r2) * sst + r2 * sst / (1 + g)) / (2 * s2)
        def lml(p_, r2):
            c = ld(np.log(sst / n), p_, r2)
            return c + np.log(quad(lambda t: np.exp(ld(t, p_, r2) - c), -15, 15, points=[np.log(sst / n)], limit=200)[0])
        assert lml(p, R2) - lml(0, 0.0) == pytest.approx(float(bayes_more._log_bf(R2, p, n, g)), abs=1e-6)


def test_bma_monte_carlo_recovers_true_regressors():
    hi, lo = [], []
    for s in range(25):
        d = _reg_data(n=200, k=8, beta=(0.5, -0.5, 0.4), seed=100 + s)
        r = bayes_more.bma(d, dict(y="y", x=[f"x{i}" for i in range(8)], g="uip"))
        pip = {row[0]: row[1] for row in tab(r, "Posterior inclusion")["rows"]}
        hi.append(np.mean([pip[f"x{i}"] for i in range(3)])); lo.append(np.mean([pip[f"x{i}"] for i in range(3, 8)]))
    assert np.mean(hi) > 0.93 and np.mean(lo) < 0.15


def test_bma_mc3_agrees_with_exact():
    d = _reg_data(n=250, k=10, beta=(0.6, -0.5, 0.4, 0.25), seed=11); x = [f"x{i}" for i in range(10)]
    ex = bayes_more.bma(d, dict(y="y", x=x, method="exact")); mc = bayes_more.bma(d, dict(y="y", x=x, method="mc3", draws=60000, burn=5000))
    pe = {r_[0]: r_[1] for r_ in tab(ex, "Posterior inclusion")["rows"]}; pm = {r_[0]: r_[1] for r_ in tab(mc, "Posterior inclusion")["rows"]}
    assert max(abs(pe[v] - pm[v]) for v in x) < 0.04


def test_bma_sizes_sum_to_one_and_default_is_mc3_above_14():
    d = _reg_data(n=300, k=16, beta=(1.0, -1.0), seed=2); x = [f"x{i}" for i in range(16)]
    r = bayes_more.bma(d, dict(y="y", x=x, draws=4000, burn=1000))
    assert sum(row[1] for row in tab(r, "Posterior distribution of model size")["rows"]) == pytest.approx(1, abs=1e-3)
    assert "MC3" in " ".join(r["notes"])
    assert not any(t["title"].startswith("Prior sensitivity") for t in r["tables"])


def test_bma_errors():
    d = _reg_data(n=60, k=4)
    for spec, msg in [(dict(y="y", x=["x0"]), "at least two"), (dict(y="y", x=["x0", "x1"], g="zz"), "g-prior"),
                      (dict(y="y", x=["x0", "x1"], model_prior="zz"), "model prior"), (dict(y="y", x=["x0", "x1"], method="zz"), "method"),
                      (dict(y="y", x=["x0", "x1"], model_prior="binomial", expected_size=5), "between 0 and 2"),
                      (dict(y="y", x=["x0", "y"]), "")]:
        with pytest.raises(OESError, match=msg): bayes_more.bma(d, spec)
    d2 = d.assign(c=1.0)
    with pytest.raises(OESError, match="constant"): bayes_more.bma(d2, dict(y="y", x=["x0", "c"]))
    with pytest.raises(OESError, match="collinear"): bayes_more.bma(d.assign(z=d["x0"] * 2), dict(y="y", x=["x0", "z"]))
    with pytest.raises(OESError, match="Too few"): bayes_more.bma(d.head(8), dict(y="y", x=["x0", "x1", "x2", "x3"]))


# ------------------------------------------------------------------------------------------------ Bayesian logit and probit
MX = os.path.join(ROOT, "microeconometrics-guide", "examples")


def _bin_data(n=400, beta=(-0.3, 0.9), link="logit", seed=0):
    rng = np.random.default_rng(seed); x = rng.normal(size=n)
    eta = beta[0] + beta[1] * x
    p = 1 / (1 + np.exp(-eta)) if link == "logit" else stats.norm.cdf(eta)
    return pd.DataFrame(dict(y=(rng.random(n) < p).astype(float), x=x))


def test_binary_registered():
    m = o.BY_ID["bayes_binary"]; assert m["group"] == "Bayesian" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_binary_matches_grid_posterior(link):
    """Sampler against brute-force numerical integration of the exact posterior over (constant, slope) on a grid."""
    d = _bin_data(n=150, seed=4, link=link); Y = d["y"].to_numpy(); x = d["x"].to_numpy(); xs = x.std(); xc = x - x.mean()
    s = 2.5
    r = bayes_more.bayes_binary(d, dict(y="y", x=["x"], link=link, prior_sd=s, draws=6000, burn=1000, chains=2))
    a = np.linspace(-2.5, 2.5, 400); b = np.linspace(-1, 3.5, 400); A, B = np.meshgrid(a, b, indexing="ij")
    eta = A[..., None] + B[..., None] * xc
    ll = (Y * (-np.logaddexp(0, -eta)) + (1 - Y) * (-np.logaddexp(0, eta))).sum(-1) if link == "logit" else \
         (Y * stats.norm.logcdf(eta) + (1 - Y) * stats.norm.logcdf(-eta)).sum(-1)
    lp = ll - A ** 2 / (2 * 10 ** 2) - B ** 2 / (2 * (s / xs) ** 2)
    w = np.exp(lp - lp.max()); w /= w.sum()
    mb = (w * B).sum(); sb = np.sqrt((w * B ** 2).sum() - mb ** 2)
    row = next(x_ for x_ in next(t for t in r["tables"] if t["title"].startswith("Posterior of the coefficients"))["rows"] if x_[0] == "x")
    assert row[1] == pytest.approx(mb, abs=0.02) and row[2] == pytest.approx(sb, rel=0.06)


def test_binary_close_to_ml_with_diffuse_prior_and_large_sample():
    d = pd.read_csv(os.path.join(MX, "mroz.csv")); x = ["nwifeinc", "educ", "exper", "age", "kidslt6", "kidsge6"]
    import statsmodels.api as sm
    for link, M in (("logit", sm.Logit), ("probit", sm.Probit)):
        ml = M(d["inlf"], sm.add_constant(d[x])).fit(disp=0)
        r = bayes_more.bayes_binary(d, dict(y="inlf", x=x, link=link, prior_sd=100, draws=4000))
        rows = {x_[0]: x_ for x_ in next(t for t in r["tables"] if t["title"].startswith("Posterior of the coefficients"))["rows"]}
        for v in x:
            assert rows[v][1] == pytest.approx(ml.params[v], abs=0.1 * ml.bse[v] + 0.003)
            assert rows[v][2] == pytest.approx(ml.bse[v], rel=0.08)
        assert float(S(r)["Largest R-hat"]) < 1.02


def test_binary_probit_logit_scale_ratio():
    d = pd.read_csv(os.path.join(MX, "mroz.csv")); x = ["educ", "exper", "age", "kidslt6"]
    g = lambda link: {r_[0]: r_[1] for r_ in next(t for t in bayes_more.bayes_binary(d, dict(y="inlf", x=x, link=link, draws=2000))["tables"]
                                                   if t["title"].startswith("Posterior of the coefficients"))["rows"]}
    gl, gp = g("logit"), g("probit")
    assert all(1.5 < gl[v] / gp[v] < 1.9 for v in x)


def test_binary_monte_carlo_coverage():
    hit = 0; R = 80
    for s in range(R):
        d = _bin_data(n=250, beta=(-0.2, 0.8), link="logit", seed=300 + s)
        r = bayes_more.bayes_binary(d, dict(y="y", x=["x"], draws=500, burn=300, chains=2, seed=s))
        row = next(x_ for x_ in next(t for t in r["tables"] if t["title"].startswith("Posterior of the coefficients"))["rows"] if x_[0] == "x")
        hit += row[3] <= 0.8 <= row[5]
    assert hit / R >= 0.88


def test_binary_marginal_effect_dummy_and_errors():
    d = _bin_data(n=300, seed=8); d["z"] = (np.random.default_rng(1).random(300) < 0.4).astype(float)
    r = bayes_more.bayes_binary(d, dict(y="y", x=["x", "z"], draws=300, burn=200))
    ame = {x_[0]: x_ for x_ in next(t for t in r["tables"] if t["title"].startswith("Posterior of the average marginal"))["rows"]}
    assert ame["z"][6] == "0 to 1" and ame["x"][6] == "per unit"
    for spec, msg in [(dict(y="x", x=["z"]), "0/1"), (dict(y="y", x=["x"], link="zz"), "link"), (dict(y="y", x=["x"], prior_sd=-1), "positive"),
                      (dict(y="y", x=["x", "y"]), "")]:
        with pytest.raises(OESError, match=msg or None): bayes_more.bayes_binary(d, spec)
    with pytest.raises(OESError, match="constant"): bayes_more.bayes_binary(d.assign(c=2.0), dict(y="y", x=["x", "c"]))
    with pytest.raises(OESError, match="Too few"): bayes_more.bayes_binary(d.head(8), dict(y="y", x=["x"]))
    with pytest.raises(OESError, match="between 2 and 8"): bayes_more.bayes_binary(d, dict(y="y", x=["x"], chains=1))


# ------------------------------------------------------------------------------------------------ Bayesian shrinkage regression
def _sparse_data(n=100, p=30, k=3, beta=1.0, sd=1.0, seed=0):
    rng = np.random.default_rng(seed); X = rng.normal(size=(n, p)); b = np.zeros(p); b[:k] = beta
    return pd.DataFrame(np.column_stack([X @ b + rng.normal(scale=sd, size=n), X]), columns=["y"] + [f"x{i}" for i in range(p)]), b


def _shr_rows(r, title="Posterior of the coefficients"):
    return {x_[0]: x_ for x_ in next(t for t in r["tables"] if t["title"].startswith(title))["rows"]}


def test_shrink_registered():
    m = o.BY_ID["bayes_shrink"]; assert m["group"] == "Bayesian" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))
    assert len(m["example"]["spec"]["x"]) == 44


def test_ridge_matches_exact_conjugate_posterior():
    """Ridge with a fixed global scale and p(sigma^2) proportional to 1/sigma^2 has an exact Student-t posterior."""
    d, _ = _sparse_data(n=80, p=4, k=2, seed=2); x = [f"x{i}" for i in range(4)]; tau = 0.7
    r = bayes_more.bayes_shrink(d, dict(y="y", x=x, prior="ridge", tau=tau, draws=6000, burn=500, chains=2))
    X = d[x].to_numpy(); xs = X.std(0); Xs = (X - X.mean(0)) / xs; yc = d["y"].to_numpy() - d["y"].mean(); n = 80
    A = Xs.T @ Xs + np.eye(4) / tau ** 2; m = np.linalg.solve(A, Xs.T @ yc); sc2 = (yc @ yc - m @ A @ m) / (n - 1) * np.linalg.inv(A).diagonal()
    sd = np.sqrt(sc2 * (n - 1) / (n - 3))                                     # Student-t with n - 1 degrees of freedom (flat intercept integrated out)
    rows = _shr_rows(r)
    for j, v in enumerate(x):
        assert rows[v][1] == pytest.approx(m[j] / xs[j], abs=0.03 * sd[j] / xs[j] * 3 + 1e-3)
        assert rows[v][2] == pytest.approx(sd[j] / xs[j], rel=0.06)


def test_lasso_matches_grid_posterior():
    """Bayesian lasso with a fixed penalty (p = 2): Gibbs sampler against numerical integration over (b1, b2, sigma)."""
    d, _ = _sparse_data(n=40, p=2, k=1, beta=0.8, seed=5); lam = 6.0; n = 40
    r = bayes_more.bayes_shrink(d, dict(y="y", x=["x0", "x1"], prior="lasso", lasso_lambda=lam, draws=8000, burn=1000))
    X = d[["x0", "x1"]].to_numpy(); xs = X.std(0); Xs = (X - X.mean(0)) / xs; yc = d["y"].to_numpy() - d["y"].mean()
    g = np.linspace(-1.6, 2.0, 361); B1, B2 = np.meshgrid(g, g, indexing="ij"); sig = np.exp(np.linspace(-1.5, 1.2, 160))
    rss = ((yc[:, None, None] - Xs[:, [0]][:, :, None] * B1[None] - Xs[:, [1]][:, :, None] * B2[None]) ** 2).sum(0)
    l1 = np.abs(B1) + np.abs(B2); post = np.zeros_like(B1); ds = np.gradient(np.log(sig))
    for sg, w in zip(sig, ds):
        post += w * sg ** 2 * sg ** (-(n - 1)) * sg ** (-2) * (lam / (2 * sg)) ** 2 * np.exp(-rss / (2 * sg ** 2) - lam * l1 / sg)
    post /= post.sum(); m1 = (post * B1).sum(); s1 = np.sqrt((post * B1 ** 2).sum() - m1 ** 2); m2 = (post * B2).sum()
    rows = _shr_rows(r)
    assert rows["x0"][1] * xs[0] == pytest.approx(m1, abs=0.02) and rows["x0"][2] * xs[0] == pytest.approx(s1, rel=0.08)
    assert rows["x1"][1] * xs[1] == pytest.approx(m2, abs=0.02)


def test_shrink_limits_ols_and_zero():
    d, _ = _sparse_data(n=120, p=5, k=2, seed=7); x = [f"x{i}" for i in range(5)]
    import statsmodels.api as sm
    ols = sm.OLS(d["y"], sm.add_constant(d[x])).fit()
    r = bayes_more.bayes_shrink(d, dict(y="y", x=x, prior="horseshoe", tau=50, draws=3000))
    rows = _shr_rows(r)
    assert all(rows[v][1] == pytest.approx(ols.params[v], abs=0.12 * ols.bse[v] + 0.005) for v in x)       # diffuse global scale -> OLS
    r0 = bayes_more.bayes_shrink(d, dict(y="y", x=x, prior="lasso", lasso_lambda=4000, draws=2000))
    assert all(abs(_shr_rows(r0)[v][1]) < 0.02 for v in x)                                               # enormous penalty -> zero


@pytest.mark.parametrize("prior", ["horseshoe", "lasso"])
def test_shrink_monte_carlo_sparse_recovery(prior):
    err_s, err_o, tp, fp = [], [], [], []
    for s in range(12):
        d, b = _sparse_data(n=80, p=30, k=3, beta=1.0, sd=1.0, seed=50 + s)
        r = bayes_more.bayes_shrink(d, dict(y="y", x=[f"x{i}" for i in range(30)], prior=prior, draws=700, burn=400, seed=s))
        rows = _shr_rows(r); est = np.array([rows[f"x{i}"][1] for i in range(30)])
        import statsmodels.api as sm
        ols = sm.OLS(d["y"], sm.add_constant(d[[f"x{i}" for i in range(30)]])).fit().params.to_numpy()[1:]
        err_s.append(np.sum((est - b) ** 2)); err_o.append(np.sum((ols - b) ** 2))
        tp.append(np.mean([rows[f"x{i}"][7] == "yes" for i in range(3)])); fp.append(np.mean([rows[f"x{i}"][7] == "yes" for i in range(3, 30)]))
    # the horseshoe is the better sparse-signal prior; the Bayesian lasso shrinks the true coefficients too (Carvalho et al. 2010)
    assert np.mean(err_s) < {"horseshoe": 0.35, "lasso": 0.6}[prior] * np.mean(err_o) and np.mean(tp) > 0.95 and np.mean(fp) < 0.06


def test_shrink_errors():
    d, _ = _sparse_data(n=60, p=4)
    for spec, msg in [(dict(y="y", x=["x0", "x1"], prior="zz"), "prior"), (dict(y="y", x=["x0"]), "at least two"), (dict(y="y", x=["x0", "x1"], threshold=-1), "positive"),
                      (dict(y="y", x=["x0", "x1"], chains=1), "between 2 and 8")]:
        with pytest.raises(OESError, match=msg): bayes_more.bayes_shrink(d, spec)
    with pytest.raises(OESError, match="constant"): bayes_more.bayes_shrink(d.assign(c=1.0), dict(y="y", x=["x0", "c"]))
    with pytest.raises(OESError, match="Too few"): bayes_more.bayes_shrink(d.head(6), dict(y="y", x=["x0", "x1", "x2", "x3"]))


# ------------------------------------------------------------------------------------------------ Bayesian change points
def _brute_changepoint(y, Z, minseg, c, a0, b0, p, kmax):
    """All partitions by enumeration, with segment marginal likelihoods from the multivariate Student-t form of the conjugate model."""
    n = len(y); k1 = Z.shape[1]; cache = {}
    def lm(t, s):
        if (t, s) not in cache:
            Zs = Z[t:s]; V = c * np.linalg.inv(Zs.T @ Zs + 1e-9 * np.eye(k1)); S = (b0 / a0) * (np.eye(s - t) + Zs @ V @ Zs.T)
            cache[(t, s)] = stats.multivariate_t.logpdf(y[t:s], loc=np.zeros(s - t), shape=S, df=2 * a0)
        return cache[(t, s)]
    res = {}
    for mask in itertools.product([0, 1], repeat=n - 1):
        cps = [i + 1 for i, m in enumerate(mask) if m]; k = len(cps)
        if k > kmax: continue
        b = [0] + cps + [n]
        if any(b[i + 1] - b[i] < minseg for i in range(len(b) - 1)): continue
        res[tuple(cps)] = sum(lm(b[i], b[i + 1]) for i in range(len(b) - 1)) + k * np.log(p) + (n - 1 - k) * np.log1p(-p)
    mx = max(res.values()); w = {k_: np.exp(v - mx) for k_, v in res.items()}; tot = sum(w.values())
    return {k_: v / tot for k_, v in w.items()}


def test_changepoint_registered():
    m = o.BY_ID["changepoint"]; assert m["group"] == "Bayesian" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


@pytest.mark.parametrize("with_x", [False, True])
def test_changepoint_matches_brute_force_enumeration(with_x):
    rng = np.random.default_rng(3); n = 13
    x = rng.normal(size=n); y = np.r_[rng.normal(0, 1, 6), rng.normal(2.5, 1, 7)] + (0.8 * x if with_x else 0)
    d = pd.DataFrame(dict(y=y, x=x)); eb, minseg, kmax = 1.5, 3 if with_x else 2, 4
    spec = dict(y="y", min_segment=minseg, expected_breaks=eb, max_breaks=kmax); spec.update(dict(x=["x"]) if with_x else {})
    r = bayes_more.changepoint(d, spec)
    ys = (y - y.mean()) / y.std(); Z = np.column_stack([np.ones(n)] + ([(x - x.mean()) / pd.Series(x).std()] if with_x else []))
    post = _brute_changepoint(ys, Z, minseg, float(n), 1.0, 0.1, eb / (n - 1), kmax)
    pk = np.zeros(kmax + 1); pb = np.zeros(n + 1)
    for cps, w in post.items():
        pk[len(cps)] += w
        for t in cps: pb[t] += w
    got = {row[0]: row[1] for row in tab(r, "Posterior probability of the number")["rows"]}
    for k in range(kmax + 1): assert got[k] == pytest.approx(pk[k], abs=2e-4)
    top = {row[1]: row[2] for row in tab(r, "Posterior probability of a break")["rows"]}
    for t in np.argsort(-pb)[:3]:
        assert top[str(t + 1)] == pytest.approx(pb[t], abs=2e-4)
    best = max(post, key=post.get); assert S(r)["Breaks in the most probable segmentation"] == len(best)


def test_changepoint_nile_break_in_1898():
    """The Nile flow (Cobb 1978; Balke 1993): a fall in the mean flow around 1898."""
    d = pd.read_csv(os.path.join(BX, "nile.csv")); r = bayes_more.changepoint(d, dict(y="flow", time="year"))
    s = S(r); assert s["Dates of the most probable breaks (last date of each regime)"] == "1898" and s["Most probable number of breaks"] == 1
    seg = {row[0]: row for row in tab(r, "Most probable segmentation")["rows"]}
    assert seg["1"][4] == pytest.approx(d[d.year <= 1898].flow.mean(), abs=6) and seg["2"][4] == pytest.approx(d[d.year > 1898].flow.mean(), abs=4)
    assert tab(r, "Posterior probability of a break")["rows"][0][:2] == ["1898", "1899"]


def test_changepoint_monte_carlo():
    near, null_ok = 0, 0; R = 30
    for s in range(R):
        rng = np.random.default_rng(900 + s); n = 120
        y = np.r_[rng.normal(0, 1, 60), rng.normal(1.3, 1, 60)]
        r = bayes_more.changepoint(pd.DataFrame(dict(y=y)), dict(y="y"))
        top = tab(r, "Posterior probability of a break")["rows"][0]; near += abs(int(top[1]) - 61) <= 8
        y0 = rng.normal(size=n); r0 = bayes_more.changepoint(pd.DataFrame(dict(y=y0)), dict(y="y"))
        null_ok += S(r0)["P(at least one break)"] < 0.5
    assert near / R >= 0.85 and null_ok / R >= 0.8


def test_changepoint_slope_change_with_regressor():
    rng = np.random.default_rng(5); n = 150; x = rng.normal(size=n); b = np.r_[np.full(80, 0.2), np.full(70, 1.5)]
    d = pd.DataFrame(dict(y=1 + b * x + rng.normal(scale=0.5, size=n), x=x)); r = bayes_more.changepoint(d, dict(y="y", x=["x"]))
    assert abs(int(tab(r, "Posterior probability of a break")["rows"][0][1]) - 81) <= 6
    seg = [row for row in tab(r, "Most probable segmentation")["rows"] if row[3] == "x"]
    assert seg[0][4] == pytest.approx(0.2, abs=0.15) and seg[1][4] == pytest.approx(1.5, abs=0.25)


def test_changepoint_errors():
    d = pd.DataFrame(dict(y=np.random.default_rng(0).normal(size=60), x=np.arange(60.0)))
    for spec, msg in [(dict(y="y", expected_breaks=0), "positive"), (dict(y="y", prior_scale=-1), "positive"), (dict(y="y", x=["x"], min_segment=2), "at least"), (dict(y="y", min_segment=1), "between 2"),
                      (dict(y="y", max_breaks=0), "between")]:
        with pytest.raises(OESError, match=msg): bayes_more.changepoint(d, spec)
    with pytest.raises(OESError, match="At least 10"): bayes_more.changepoint(d.head(8), dict(y="y"))
    with pytest.raises(OESError, match="constant"): bayes_more.changepoint(d.assign(y=1.0), dict(y="y"))
    with pytest.raises(OESError, match="Column"): bayes_more.changepoint(d, dict(y="y", time="zz"))


# ---------------------------------------------------------------------------------------- Bayesian quantile regression
def _qr_data(n=400, seed=1):
    rng = np.random.default_rng(seed); x = rng.uniform(0, 2, n)
    return pd.DataFrame(dict(y=1 + 2 * x + (0.5 + x) * rng.normal(size=n), x=x))


def test_bayes_qr_registered():
    m = o.BY_ID["bayes_qr"]; assert m["group"] == "Bayesian" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_bayes_qr_matches_frequentist_quantreg():
    import statsmodels.api as sm
    d = pd.read_csv(os.path.join(BX, "engel.csv")); r = bayes_more.bayes_qr(d, dict(y="foodexp", x=["income"], draws=1500, burn=500))
    for row in tab(r, "Posterior of the coefficients")["rows"]:
        if row[1] == "income":
            fq = sm.QuantReg(d.foodexp, sm.add_constant(d.income)).fit(q=float(row[0]), max_iter=5000).params["income"]
            assert row[2] == pytest.approx(fq, abs=0.03)


def test_bayes_qr_recovers_location_scale_truth():
    """y = 1 + 2x + (0.5 + x) e: the quantile-tau slope is 2 + z_tau and the intercept 1 + 0.5 z_tau."""
    r = bayes_more.bayes_qr(_qr_data(n=800), dict(y="y", x=["x"], quantiles="0.1, 0.5, 0.9", draws=1500, burn=500))
    for row in tab(r, "Posterior of the coefficients")["rows"]:
        z = stats.norm.ppf(float(row[0])); truth = 2 + z if row[1] == "x" else 1 + 0.5 * z
        assert abs(row[2] - truth) < 3 * row[3] + 0.1


def test_bayes_qr_heterogeneity_detected_and_absent():
    r = bayes_more.bayes_qr(_qr_data(n=500), dict(y="y", x=["x"], quantiles="0.1, 0.9", draws=800, burn=400))
    row = tab(r, "Does the effect differ")["rows"][0]; assert row[4] > 0 and row[6] > 0.99                        # lower 2.5% bound above zero
    rng = np.random.default_rng(8); x = rng.normal(size=500); d = pd.DataFrame(dict(y=1 + x + rng.normal(size=500), x=x))
    r0 = bayes_more.bayes_qr(d, dict(y="y", x=["x"], quantiles="0.1, 0.9", draws=800, burn=400))
    row0 = tab(r0, "Does the effect differ")["rows"][0]; assert row0[4] < 0 < row0[5]


def test_bayes_qr_calibration_and_convergence():
    r = bayes_more.bayes_qr(_qr_data(), dict(y="y", x=["x"], draws=800, burn=400))
    for row in tab(r, "Calibration")["rows"]: assert row[2] < 0.04
    assert S(r)["Largest R-hat"] < 1.1


def test_bayes_qr_monte_carlo_coverage():
    """95% intervals for the median slope cover the truth in most replications once the draws are rescaled to the sandwich covariance."""
    cover = 0; R = 24
    for s in range(R):
        rng = np.random.default_rng(300 + s); x = rng.normal(size=150); y = 1 + 1.5 * x + rng.standard_t(5, 150)
        r = bayes_more.bayes_qr(pd.DataFrame(dict(y=y, x=x)), dict(y="y", x=["x"], quantiles="0.5", draws=400, burn=200, seed=s))
        row = [q for q in tab(r, "Posterior of the coefficients")["rows"] if q[1] == "x"][0]; cover += row[4] <= 1.5 <= row[5]
    assert cover / R >= 0.85


def test_bayes_qr_adjustment_widens_and_matches_bootstrap():
    import statsmodels.api as sm
    d = pd.read_csv(os.path.join(BX, "engel.csv")); sp = dict(y="foodexp", x=["income"], quantiles="0.5", draws=1500, burn=500)
    sd = lambda r: next(q[3] for q in tab(r, "Posterior of the coefficients")["rows"] if q[1] == "income")
    a, raw = sd(bayes_more.bayes_qr(d, sp)), sd(bayes_more.bayes_qr(d, dict(sp, adjust="no")))
    assert a > 1.5 * raw
    X = sm.add_constant(d.income.values); y = d.foodexp.values; rng = np.random.default_rng(0)
    bs = [sm.QuantReg(y[i], X[i]).fit(q=0.5, max_iter=5000).params[1] for i in (rng.integers(0, len(d), len(d)) for _ in range(200))]
    assert a == pytest.approx(np.std(bs), rel=0.4)


def test_bayes_qr_errors():
    d = _qr_data(60)
    for spec, msg in [(dict(y="y", x=["x"], quantiles="0, 0.5"), "strictly"), (dict(y="y", x=["x"], quantiles="a"), "numbers"), (dict(y="y", x=["x"], prior_sd=0), "positive"),
                      (dict(y="y", x=["y"]), "more than one role")]:
        with pytest.raises(OESError, match=msg): bayes_more.bayes_qr(d, spec)
    with pytest.raises(OESError, match="constant"): bayes_more.bayes_qr(d.assign(x=1.0), dict(y="y", x=["x"]))
    with pytest.raises(OESError, match="Too few"): bayes_more.bayes_qr(d.head(8), dict(y="y", x=["x"]))
