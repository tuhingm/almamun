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
