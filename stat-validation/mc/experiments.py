"""Data-generating processes and extractors for the 31 methods. Each EXPERIMENTS[model_id] is a list of Scenario objects.
Truths are derived analytically in the comments next to each DGP. H0 scenarios report the rejection rate (Type I error);
alternatives report power; estimators report bias, empirical SD, RMSE, SE ratio and CI coverage."""
import numpy as np, pandas as pd
from scipy import stats
from mc_lib import Scenario, S, cell, NGRID, o

def tdraw(rng, size, df=None):
    """Standardised draws: normal, or Student t with df degrees of freedom scaled to unit variance."""
    if df is None: return rng.standard_normal(size)
    return rng.standard_t(df, size) / np.sqrt(df / (df - 2))

# ----------------------------------------------------------------- A. ANOVA
def g_anova1(delta=0.0, het=False, unbal=False, tdf=None, k=3):
    def gen(n, rng):
        if unbal:
            w = np.array([0.5, 0.3, 0.2]); g = rng.choice(k, n, p=w[:k] / w[:k].sum())
        else: g = np.arange(n) % k
        sd = np.array([1.0, 2.0, 3.0])[g] if het else 1.0
        if unbal and het: sd = np.array([3.0, 2.0, 1.0])[g]   # largest variance in the smallest group: Welch vs F contrast
        y = delta * (g == 0) + sd * tdraw(rng, n, tdf)
        return pd.DataFrame({"y": y, "g": g}), dict(y="y", group="g", posthoc="none")
    return gen
ex1 = lambda r: dict(p_F=S(r, "p-value"), p_Welch=S(r, "Welch p"))
def g_anova2(effA=0.0, inter=0.0, tdf=None):
    def gen(n, rng):
        a = np.arange(n) % 2; b = (np.arange(n) // 2) % 3
        y = effA * a + inter * ((a == 1) & (b == 2)) + tdraw(rng, n, tdf)
        return pd.DataFrame({"y": y, "a": a, "b": b}), dict(y="y", factor_a="a", factor_b="b", ss_type="III")
    return gen
def ex2(r): return dict(p_A=cell(r, "Two-way", "a", "p-value"), p_B=cell(r, "Two-way", "b", "p-value"), p_AB=cell(r, "Two-way", "a x b", "p-value"))
def g_rm(delta=0.0, rho=0.5, nonsph=False):
    def gen(n, rng):
        k = 4; s = rng.standard_normal((n, 1)) * np.sqrt(rho / (1 - rho))
        if nonsph:  # AR(1)-like correlation with strong decay violates sphericity
            idx = np.arange(k); C = 0.85 ** np.abs(idx[:, None] - idx[None]); e = rng.standard_normal((n, k)) @ np.linalg.cholesky(C).T
            Y = e
        else: Y = s + rng.standard_normal((n, k))
        Y = Y + delta * np.array([0, 0, 0, 1.0])
        d = pd.DataFrame({"s": np.repeat(np.arange(n), k), "t": np.tile(np.arange(k), n), "y": Y.ravel()})
        return d, dict(subject="s", within="t", y="y")
    return gen
ex_rm = lambda r: dict(p=S(r, "p-value (used)"), p_uncorr=cell(r, "Repeated-measures ANOVA (sphericity", "Within factor: t", "p-value"))
def g_ancova(eff=0.5, tdf=None, slope=0.8):
    def gen(n, rng):
        g = np.arange(n) % 2; x = rng.standard_normal(n) + 0.5 * g   # covariate imbalance across groups
        y = slope * x + eff * g + tdraw(rng, n, tdf)
        return pd.DataFrame({"y": y, "g": g, "x": x}), dict(y="y", group="g", covariates=["x"])
    return gen
def ex_ancova(r):
    t = next(t for t in r["tables"] if t["title"].startswith("Pairwise differences")); row = t["rows"][0]; c = t["columns"]
    return dict(est=row[c.index("Difference")], se=row[c.index("Std. err.")], lo=row[c.index("95% CI low")], hi=row[c.index("CI high")], p=row[c.index("p-value")])
def g_manova(delta=0.0, tdf=None, k=3, p=3, rho=0.4, cov=False):
    def gen(n, rng):
        g = np.arange(n) % k; C = rho * np.ones((p, p)) + (1 - rho) * np.eye(p); L = np.linalg.cholesky(C)
        E = tdraw(rng, (n, p), tdf) @ L.T
        x = rng.standard_normal(n); Y = E + (delta * (g == 0))[:, None] * np.array([1.0, 0.5, 0.0]) + (0.7 * x)[:, None]
        d = pd.DataFrame(Y, columns=["y1", "y2", "y3"]); d["g"] = g; d["x"] = x
        sp = dict(y_vars=["y1", "y2", "y3"], group="g");
        if cov: sp["covariates"] = ["x"]
        return d, sp
    return gen
def ex_manova(r):
    t = next(t for t in r["tables"] if t["title"].startswith("Multivariate tests for g")); c = t["columns"]
    w = next(x for x in t["rows"] if x[0].startswith("Wilks")); pl = next(x for x in t["rows"] if x[0].startswith("Pillai"))
    return dict(p_wilks=w[c.index("p-value")], p_pillai=pl[c.index("p-value")])

# ----------------------------------------------------------------- B. Nonparametric
def g_two_sample(shift=0.0, tdf=None, dist="normal"):
    def gen(n, rng):
        g = np.arange(n) % 2
        e = rng.exponential(1, n) - 1 if dist == "exp" else tdraw(rng, n, tdf)
        return pd.DataFrame({"y": e + shift * g, "g": g}), dict(y="y", group="g")
    return gen
def g_paired(shift=0.0, tdf=None, dist="normal"):
    def gen(n, rng):
        d = rng.exponential(1, n) - np.log(2) if dist == "exp" else tdraw(rng, n, tdf)   # exponential shifted to median 0
        return pd.DataFrame({"a": rng.standard_normal(n), "b": 0.0}).assign(b=lambda q: q.a + d + shift), dict(x1="a", x2="b")
    return gen
def g_friedman(shift=0.0, tdf=None):
    def gen(n, rng):
        k = 4; Y = rng.standard_normal((n, 1)) + tdraw(rng, (n, k), tdf) + shift * np.array([0, 0, 0, 1.0])
        return pd.DataFrame({"s": np.repeat(np.arange(n), k), "t": np.tile(np.arange(k), n), "y": Y.ravel()}), dict(subject="s", treatment="t", y="y")
    return gen
def g_kendall(rho=0.0, tdf=None):
    def gen(n, rng):
        x = tdraw(rng, n, tdf); y = rho * x + np.sqrt(1 - rho ** 2) * tdraw(rng, n, tdf)
        return pd.DataFrame({"x": x, "y": y}), dict(x="x", y="y")
    return gen
ex_p = lambda r: dict(p=S(r, "p-value"))
def ex_kendall(r):
    t = next(t for t in r["tables"] if t["title"].startswith("Kendall's tau")); row = t["rows"][0]
    return dict(est=row[1], lo=row[2], hi=row[3], p=S(r, "p-value"))

# ----------------------------------------------------------------- C. Multivariate
def g_pca(p=6, r=0.5):
    def gen(n, rng):
        C = r * np.ones((p, p)) + (1 - r) * np.eye(p); X = rng.standard_normal((n, p)) @ np.linalg.cholesky(C).T
        return pd.DataFrame(X, columns=[f"v{i}" for i in range(p)]), dict(vars=[f"v{i}" for i in range(p)], basis="correlation", n_components=2)
    return gen
ex_pca = lambda r: dict(est=cell(r, "Eigenvalues", "PC1", "Eigenvalue"))
def g_efa(p=6, lam=0.7, tdf=None):
    def gen(n, rng):
        f = rng.standard_normal((n, 1)); X = f * lam + np.sqrt(1 - lam ** 2) * tdraw(rng, (n, p), tdf)
        return pd.DataFrame(X, columns=[f"v{i}" for i in range(p)]), dict(vars=[f"v{i}" for i in range(p)], n_factors=1, rotation="none")
    return gen
ex_efa = lambda r: dict(est=abs(cell(r, "Factor loadings", "v0", "Factor 1")))
def g_cfa(lam=(1.0, 0.8, 0.7, 0.6, 0.9), psi=0.5, tdf=None):
    lam = np.array(lam); th = 0.5
    def gen(n, rng):
        f = np.sqrt(psi) * rng.standard_normal((n, 1)); X = f * lam + np.sqrt(th) * tdraw(rng, (n, len(lam)), tdf)
        c = [f"x{i+1}" for i in range(len(lam))]
        return pd.DataFrame(X, columns=c), dict(model="f =~ " + " + ".join(c))
    return gen
def ex_cfa(r):
    t = next(t for t in r["tables"] if t["title"].startswith("Factor loadings")); c = t["columns"]
    row = next(x for x in t["rows"] if x[0] == "f =~ x2")
    return dict(est=row[c.index("Estimate")], se=row[c.index("Std. err.")], lo=row[c.index("95% CI low")], hi=row[c.index("CI high")])
def g_lda(delta=2.0, p=3):
    def gen(n, rng):
        g = np.arange(n) % 2; X = rng.standard_normal((n, p)); X[:, 0] += delta * g
        d = pd.DataFrame(X, columns=["a", "b", "c"]); d["g"] = g; return d, dict(vars=["a", "b", "c"], group="g", priors="equal")
    return gen
ex_lda = lambda r: dict(est=S(r, "Cross-validated accuracy"), est_resub=S(r, "Resubstitution accuracy"))
def g_cca(rho=0.0, p=2, q=2):
    def gen(n, rng):
        X = rng.standard_normal((n, p)); Y = rng.standard_normal((n, q)); Y[:, 0] = rho * X[:, 0] + np.sqrt(1 - rho ** 2) * Y[:, 0]
        d = pd.DataFrame(np.c_[X, Y], columns=["x1", "x2", "y1", "y2"]); return d, dict(x_vars=["x1", "x2"], y_vars=["y1", "y2"], perms=0)
    return gen
ex_cca = lambda r: dict(est=S(r, "First canonical correlation"), p=S(r, "Bartlett p-value (all pairs)"))
def g_ca(null=True, r_=4, c_=4):
    def gen(n, rng):
        if null: P = np.outer(np.ones(r_) / r_, np.arange(1, c_ + 1) / np.arange(1, c_ + 1).sum())
        else:
            P = np.outer(np.ones(r_) / r_, np.ones(c_) / c_) * (1 + 0.5 * (np.eye(r_, c_) - 1 / c_)); P /= P.sum()
        cnt = rng.multinomial(n * 5, P.ravel()).reshape(r_, c_)
        d = pd.DataFrame(cnt, columns=[f"c{j}" for j in range(c_)]); d.insert(0, "r", [f"r{i}" for i in range(r_)])
        return d, dict(row_var="r", counts=[f"c{j}" for j in range(c_)])
    return gen
ex_ca = lambda r: dict(p=S(r, "p-value"))

# ----------------------------------------------------------------- D. Survival
BETA_COX = 0.5
def g_cox(beta=BETA_COX, cens=1.2, ties=False, tdf=None, ties_opt="efron"):
    def gen(n, rng):
        x = rng.standard_normal(n); t = rng.exponential(1, n) / np.exp(beta * x); c = rng.exponential(1 / cens, n) if cens else np.full(n, np.inf)
        T = np.minimum(t, c); e = (t <= c).astype(int)
        if ties: T = np.ceil(T * 8) / 8
        return pd.DataFrame({"t": T, "e": e, "x": x}), dict(time="t", event="e", covariates=["x"], ties=ties_opt)
    return gen
def ex_cox(r):
    t = next(t for t in r["tables"] if t["title"].startswith("Cox")); c = t["columns"]; row = t["rows"][0]
    return dict(est=row[c.index("coef (log HR)")], se=row[c.index("SE")], lo=np.log(row[c.index("HR 95% CI low")]), hi=np.log(row[c.index("HR CI high")]), p=row[c.index("p-value")])
def g_aft(beta=0.5, shape=1.5, cens=0.15, dist="weibull"):
    def gen(n, rng):
        x = rng.standard_normal(n)
        if dist == "weibull": u = rng.weibull(shape, n); t = np.exp(1.0 + beta * x) * u      # log T = 1 + beta x + (1/shape) log E, E ~ Exp(1)
        else: t = np.exp(1.0 + beta * x + 0.7 * rng.standard_normal(n))
        c = rng.exponential(1 / cens, n) * np.exp(1.0); T = np.minimum(t, c)
        return pd.DataFrame({"t": T, "e": (t <= c).astype(int), "x": x}), dict(time="t", event="e", covariates=["x"], distribution=dist)
    return gen
def ex_aft(r):
    t = next(t for t in r["tables"] if "accelerated failure time" in t["title"]); c = t["columns"]; row = next(x for x in t["rows"] if x[0] == "x")
    return dict(est=row[c.index("Coefficient (log time)")], se=row[c.index("SE")], lo=row[c.index("95% CI low")], hi=row[c.index("CI high")], p=row[c.index("p-value")])
def g_fg(beta=0.7, pcause=0.6, cens=0.3):
    # Fine-Gray DGP: F1(t|x) = 1 - (1 - p(1 - e^-t))^exp(beta x); competing event with the remaining mass
    def gen(n, rng):
        x = rng.standard_normal(n); th = np.exp(beta * x); u = rng.uniform(size=n); pc1 = 1 - (1 - pcause) ** th
        cause1 = u <= pc1; t = np.empty(n)
        uu = np.where(cause1, u, 0.5); # inverse of F1 for cause-1 subjects: 1 - (1 - p(1 - e^-t))^th = u
        t1 = -np.log(1 - (1 - (1 - np.clip(uu, 0, pc1 - 1e-12)) ** (1 / th)) / pcause)
        t2 = rng.exponential(1, n); t = np.where(cause1, t1, t2)
        c = rng.exponential(1 / cens, n); T = np.minimum(t, c); st = np.where(t <= c, np.where(cause1, 1, 2), 0)
        return pd.DataFrame({"t": T, "s": st, "x": x}), dict(time="t", event="s", cause=1, covariates=["x"])
    return gen
def ex_fg(r):
    t = next(t for t in r["tables"] if t["title"].startswith("Fine-Gray")); c = t["columns"]; row = t["rows"][0]
    return dict(est=row[c.index("coef")], se=row[c.index("SE")], lo=np.log(row[c.index("sHR 95% CI low")]), hi=np.log(row[c.index("sHR CI high")]), p=row[c.index("p-value")])
def g_km(shift=0.0, cens=0.5):
    def gen(n, rng):
        g = np.arange(n) % 2; t = rng.exponential(1, n) / np.exp(shift * g); c = rng.exponential(1 / cens, n)
        return pd.DataFrame({"t": np.minimum(t, c), "e": (t <= c).astype(int), "g": g}), dict(time="t", event="e", group="g")
    return gen
def g_km1(cens=0.5):
    def gen(n, rng):
        t = rng.exponential(1, n); c = rng.exponential(1 / cens, n)
        return pd.DataFrame({"t": np.minimum(t, c), "e": (t <= c).astype(int)}), dict(time="t", event="e")
    return gen
def ex_km_cov(r):   # S(1) from the life table; truth exp(-1)
    t = next(t for t in r["tables"] if t["title"].startswith("Kaplan-Meier life table")); c = t["columns"]
    rows = [x for x in t["rows"] if x[c.index("Time")] <= 1.0]; row = rows[-1]
    return dict(est=row[c.index("S(t)")], se=row[c.index("SE (Greenwood)")], lo=row[c.index("CI low")], hi=row[c.index("CI high")])
def g_parametric(true="weibull"):
    def gen(n, rng):
        x = rng.standard_normal(n)
        if true == "weibull": t = np.exp(1.0 + 0.3 * x) * rng.weibull(1.5, n)
        elif true == "lognormal": t = np.exp(1.0 + 0.3 * x + 0.9 * rng.standard_normal(n))
        else: t = np.exp(1.0 + 0.3 * x) * rng.exponential(1, n)
        c = rng.exponential(4.0, n); return pd.DataFrame({"t": np.minimum(t, c), "e": (t <= c).astype(int), "x": x}), dict(time="t", event="e", covariates=["x"])
    return gen
def ex_param(true):
    def f(r):
        best = S(r, "Best by AIC"); return dict(flag_correct=float(true.lower() in str(best).lower()))
    return f

# ----------------------------------------------------------------- E. Mixed / longitudinal
def g_lmm(beta=0.5, size=5, sdu=1.0, unbal=False, tdf=None, slope=False):
    def gen(n, rng):
        m = max(2, n // size); sizes = rng.integers(2, 2 * size, m) if unbal else np.full(m, size)
        g = np.repeat(np.arange(m), sizes); N = len(g); x = rng.standard_normal(N)
        u = sdu * tdraw(rng, m, tdf); y = beta * x + u[g] + rng.standard_normal(N)
        if slope: y = y + (0.5 * rng.standard_normal(m))[g] * x
        return pd.DataFrame({"y": y, "g": g, "x": x}), dict(y="y", group="g", covariates=["x"], **(dict(slope="x") if slope else {}))
    return gen
def ex_lmm(r):
    t = next(t for t in r["tables"] if t["title"].startswith("Fixed effects")); c = t["columns"]; row = next(x for x in t["rows"] if x[0] == "x")
    return dict(est=row[c.index("Estimate")], se=row[c.index("SE")], lo=row[c.index("95% CI low")], hi=row[c.index("CI high")], p=row[c.index("p-value")])
def g_glmm(beta=0.5, size=10, sd=1.0, fam="binomial"):
    def gen(n, rng):
        m = max(3, n // size); g = np.repeat(np.arange(m), size); N = len(g); x = rng.standard_normal(N); u = sd * rng.standard_normal(m)
        eta = -0.5 + beta * x + u[g]
        y = rng.binomial(1, 1 / (1 + np.exp(-eta))) if fam == "binomial" else rng.poisson(np.exp(0.2 + beta * x + u[g] * 0.5))
        return pd.DataFrame({"y": y, "g": g, "x": x}), dict(y="y", group="g", covariates=["x"], family=fam)
    return gen
def ex_glmm(r):
    t = next(t for t in r["tables"] if t["title"].startswith("Fixed effects")); c = t["columns"]; row = next(x for x in t["rows"] if x[0] == "x")
    return dict(est=row[c.index("Estimate")], se=row[c.index("SE")], lo=row[c.index("95% CI low")] if "95% CI low" in c else row[c.index("Estimate")] - 1.96 * row[c.index("SE")],
                hi=row[c.index("CI high")] if "CI high" in c else row[c.index("Estimate")] + 1.96 * row[c.index("SE")], p=row[c.index("p-value")])
def g_gee(beta=0.5, size=5, fam="gaussian", corr="exchangeable", misspec=False):
    def gen(n, rng):
        m = max(3, n // size); g = np.repeat(np.arange(m), size); N = len(g); x = rng.standard_normal(N); u = rng.standard_normal(m)
        if fam == "gaussian": y = beta * x + u[g] + rng.standard_normal(N)
        else: y = rng.binomial(1, 1 / (1 + np.exp(-(beta * x + 0.7 * u[g]))))
        return pd.DataFrame({"y": y, "g": g, "t": np.tile(np.arange(size), m), "x": x}), dict(y="y", group="g", covariates=["x"], family=fam, corr="independence" if misspec else corr)
    return gen
def ex_gee(r):
    t = next(t for t in r["tables"] if t["title"].startswith("GEE coefficients")); c = t["columns"]; row = next(x for x in t["rows"] if x[0] == "x")
    return dict(est=row[c.index("Estimate")], se=row[c.index("Robust SE")], lo=row[c.index("95% CI low")], hi=row[c.index("CI high")], p=row[c.index("p-value")])
def g_icc(sr=0.2, k=4):
    def gen(n, rng):
        s = rng.standard_normal((n, 1)); rr = np.sqrt(sr) * rng.standard_normal((1, k)); Y = s + rr + rng.standard_normal((n, k))
        return pd.DataFrame(Y, columns=[f"r{j}" for j in range(k)]), dict(raters=[f"r{j}" for j in range(k)])
    return gen
def ex_icc(form):
    def f(r):
        t = next(t for t in r["tables"] if t["title"].startswith("Intraclass")); c = t["columns"]; row = next(x for x in t["rows"] if x[0] == form)
        return dict(est=row[c.index("ICC")], lo=row[c.index("95% CI low")], hi=row[c.index("CI high")])
    return f

# ----------------------------------------------------------------- F. Resampling
def g_boot(stat="mean", dist="normal"):
    def gen(n, rng):
        y = rng.standard_normal(n) if dist == "normal" else rng.exponential(1, n)
        return pd.DataFrame({"y": y, "w": 0.0}), dict(y="y", statistic=stat, n_boot=1000, seed=int(rng.integers(1, 10 ** 6)))
    return gen
def ex_boot(method):
    def f(r):
        t = next(t for t in r["tables"] if t["title"].startswith("95% bootstrap")); row = next(x for x in t["rows"] if x[0].startswith(method))
        return dict(lo=row[1], hi=row[2])
    return f
def g_perm(test="mean_diff", shift=0.0, dist="normal", n_perm=999):
    def gen(n, rng):
        g = np.arange(n) % 2; y = (rng.exponential(1, n) if dist == "exp" else rng.standard_normal(n)) + shift * g
        return pd.DataFrame({"y": y, "g": g}), dict(y="y", group="g", test=test, n_perm=n_perm, seed=int(rng.integers(1, 10 ** 6)), exact_max=2000)
    return gen
ex_perm = lambda r: dict(p=S(r, "Permutation p-value"), p_welch=cell(r, "Reference parametric", "Welch t-test", "p-value"))

# ----------------------------------------------------------------- G. Meta-analysis
def g_meta(theta=0.3, tau2=0.0, tau_method="reml", ci="wald", model="random"):
    def gen(k, rng):
        vi = rng.uniform(0.01, 0.15, k); yi = theta + np.sqrt(tau2) * rng.standard_normal(k) + np.sqrt(vi) * rng.standard_normal(k)
        sp = dict(input="generic", study="s", yi="yi", sei="sei")
        if model == "random": sp.update(tau_method=tau_method, ci_method=ci)
        return pd.DataFrame({"s": [f"s{i}" for i in range(k)], "yi": yi, "sei": np.sqrt(vi)}), sp
    return gen
def ex_meta(r):
    t = next(t for t in r["tables"] if t["title"].startswith("Pooled effect")); c = t["columns"]; row = t["rows"][0]
    return dict(est=row[c.index("Estimate")], se=row[c.index("SE")], lo=row[c.index("95% CI low")], hi=row[c.index("CI high")], p=row[c.index("p-value")])

KG = (10, 20, 50, 100, 200)  # numbers of studies for the meta-analysis grid
EXPERIMENTS = {
 "stat_anova1": [Scenario("H0, normal, balanced", "stat_anova1", g_anova1(), ex1, h0=True),
                 Scenario("H0, heteroskedastic + unbalanced (robustness)", "stat_anova1", g_anova1(het=True, unbal=True), ex1, h0=True, note="largest variance in the smallest group"),
                 Scenario("H0, t(3) errors (robustness)", "stat_anova1", g_anova1(tdf=3), ex1, h0=True),
                 Scenario("Power, delta=0.5 SD", "stat_anova1", g_anova1(delta=0.5), ex1, h0=False)],
 "stat_anova2": [Scenario("H0 (no effects), normal", "stat_anova2", g_anova2(), ex2, h0=True),
                 Scenario("Power: main effect A = 0.5", "stat_anova2", g_anova2(effA=0.5), ex2, h0=False),
                 Scenario("Power: interaction = 0.8", "stat_anova2", g_anova2(inter=0.8), ex2, h0=False),
                 Scenario("H0, t(3) errors (robustness)", "stat_anova2", g_anova2(tdf=3), ex2, h0=True)],
 "stat_rm_anova": [Scenario("H0, compound symmetry", "stat_rm_anova", g_rm(), ex_rm, h0=True, ns=(20, 50, 100, 250, 500)),
                   Scenario("H0, non-spherical AR(1) 0.85 (robustness)", "stat_rm_anova", g_rm(nonsph=True), ex_rm, h0=True, ns=(20, 50, 100, 250, 500)),
                   Scenario("Power, last condition +0.5", "stat_rm_anova", g_rm(delta=0.5), ex_rm, h0=False, ns=(20, 50, 100, 250, 500))],
 "stat_ancova": [Scenario("Adjusted difference, normal (truth -0.5)", "stat_ancova", g_ancova(), ex_ancova, truth=-0.5),
                 Scenario("Adjusted difference, t(3) errors", "stat_ancova", g_ancova(tdf=3), ex_ancova, truth=-0.5),
                 Scenario("H0 (no group effect)", "stat_ancova", g_ancova(eff=0.0), ex_ancova, truth=0.0, h0=True)],
 "stat_manova": [Scenario("H0, normal", "stat_manova", g_manova(), ex_manova, h0=True, ns=(60, 100, 250, 500, 1000)),
                 Scenario("H0, t(3) errors (robustness)", "stat_manova", g_manova(tdf=3), ex_manova, h0=True, ns=(60, 100, 250, 500, 1000)),
                 Scenario("Power, delta=0.5", "stat_manova", g_manova(delta=0.5), ex_manova, h0=False, ns=(60, 100, 250, 500, 1000))],
 "stat_mancova": [Scenario("H0, normal", "stat_mancova", g_manova(cov=True), ex_manova, h0=True, ns=(60, 100, 250, 500, 1000)),
                  Scenario("Power, delta=0.5", "stat_mancova", g_manova(delta=0.5, cov=True), ex_manova, h0=False, ns=(60, 100, 250, 500, 1000))],
 "stat_mann_whitney": [Scenario("H0, normal", "stat_mann_whitney", g_two_sample(), ex_p, h0=True),
                       Scenario("H0, exponential (skewed)", "stat_mann_whitney", g_two_sample(dist="exp"), ex_p, h0=True),
                       Scenario("H0, t(3)", "stat_mann_whitney", g_two_sample(tdf=3), ex_p, h0=True),
                       Scenario("Power, shift 0.5", "stat_mann_whitney", g_two_sample(shift=0.5), ex_p, h0=False)],
 "stat_wilcoxon": [Scenario("H0, normal differences", "stat_wilcoxon", g_paired(), ex_p, h0=True),
                   Scenario("H0, skewed-then-centred differences (median 0)", "stat_wilcoxon", g_paired(dist="exp"), ex_p, h0=True, note="exponential minus its median: symmetry violated, so the signed-rank null is not exactly true"),
                   Scenario("Power, shift 0.3", "stat_wilcoxon", g_paired(shift=0.3), ex_p, h0=False)],
 "stat_kruskal": [Scenario("H0, normal", "stat_kruskal", g_anova1(), ex_p, h0=True), Scenario("H0, t(3)", "stat_kruskal", g_anova1(tdf=3), ex_p, h0=True),
                  Scenario("Power, delta=0.7", "stat_kruskal", g_anova1(delta=0.7), ex_p, h0=False)],
 "stat_friedman": [Scenario("H0", "stat_friedman", g_friedman(), ex_p, h0=True, ns=(20, 50, 100, 250, 500)), Scenario("H0, t(3)", "stat_friedman", g_friedman(tdf=3), ex_p, h0=True, ns=(20, 50, 100, 250, 500)),
                   Scenario("Power, last condition +0.5", "stat_friedman", g_friedman(shift=0.5), ex_p, h0=False, ns=(20, 50, 100, 250, 500))],
 "stat_kendall": [Scenario("H0 (rho=0)", "stat_kendall", g_kendall(), ex_kendall, h0=True), Scenario("H0, t(3)", "stat_kendall", g_kendall(tdf=3), ex_kendall, h0=True),
                  Scenario("tau-b recovery, bivariate normal rho=0.5 (truth 2/pi*asin(0.5))", "stat_kendall", g_kendall(rho=0.5), ex_kendall, truth=2 / np.pi * np.arcsin(0.5), h0=False)],
 "stat_pca": [Scenario("First eigenvalue, equicorrelation 0.5, p=6 (truth 1+5*0.5=3.5)", "stat_pca", g_pca(), ex_pca, truth=3.5)],
 "stat_efa": [Scenario("Loading of v0, one factor lambda=0.7", "stat_efa", g_efa(), ex_efa, truth=0.7), Scenario("Loading, t(5) noise (robustness)", "stat_efa", g_efa(tdf=5), ex_efa, truth=0.7)],
 "stat_cfa": [Scenario("Loading of x2 (truth 0.8), normal", "stat_cfa", g_cfa(), ex_cfa, truth=0.8, ns=(100, 250, 500, 1000)),
              Scenario("Loading of x2, t(5) errors (robustness)", "stat_cfa", g_cfa(tdf=5), ex_cfa, truth=0.8, ns=(100, 250, 500, 1000))],
 "stat_lda": [Scenario("CV accuracy vs Bayes accuracy Phi(1)=0.8413 (delta=2, equal priors)", "stat_lda", g_lda(), ex_lda, truth=float(stats.norm.cdf(1.0)))],
 "stat_cca": [Scenario("H0 (rho=0), Bartlett test", "stat_cca", g_cca(), ex_cca, h0=True), Scenario("First canonical correlation, rho=0.5", "stat_cca", g_cca(rho=0.5), ex_cca, truth=0.5, h0=False)],
 "stat_ca": [Scenario("H0 independence, chi-square test", "stat_ca", g_ca(True), ex_ca, h0=True), Scenario("Power (dependence)", "stat_ca", g_ca(False), ex_ca, h0=False)],
 "stat_km": [Scenario("S(1) coverage, exponential, ~33% censoring (truth e^-1)", "stat_km", g_km1(), ex_km_cov, truth=float(np.exp(-1))),
             Scenario("Log-rank H0", "stat_km", g_km(), ex_p, h0=True), Scenario("Log-rank power, HR=e^0.5", "stat_km", g_km(shift=0.5), ex_p, h0=False)],
 "stat_cox": [Scenario("Baseline: beta=0.5, ~55% events (Efron)", "stat_cox", g_cox(), ex_cox, truth=BETA_COX),
              Scenario("Heavy ties (times rounded to 1/8), Efron", "stat_cox", g_cox(ties=True), ex_cox, truth=BETA_COX),
              Scenario("Heavy ties, Breslow", "stat_cox", g_cox(ties=True, ties_opt="breslow"), ex_cox, truth=BETA_COX),
              Scenario("No censoring", "stat_cox", g_cox(cens=0), ex_cox, truth=BETA_COX),
              Scenario("H0 (beta=0)", "stat_cox", g_cox(beta=0.0), ex_cox, truth=0.0, h0=True)],
 "stat_aft": [Scenario("Weibull AFT: beta=0.5, shape 1.5 (log-time coefficient)", "stat_aft", g_aft(), ex_aft, truth=0.5),
              Scenario("Log-normal AFT fitted to log-normal data", "stat_aft", g_aft(dist="lognormal"), ex_aft, truth=0.5),
              Scenario("Misspecified: Weibull fitted to log-normal data", "stat_aft", lambda n, rng, g=g_aft(dist="lognormal"): (lambda d, s: (d, {**s, "distribution": "weibull"}))(*g(n, rng)), ex_aft, truth=0.5, note="coefficient is not the same estimand under a different error law; reported for coverage behaviour")],
 "stat_finegray": [Scenario("Subdistribution log-HR beta=0.7 (Fine-Gray DGP)", "stat_finegray", g_fg(), ex_fg, truth=0.7, ns=(100, 250, 500, 1000), note="the SE omits the censoring-weight estimation variance (documented)")],
 "stat_parametric": [Scenario("True Weibull: AIC picks Weibull", "stat_parametric", g_parametric("weibull"), ex_param("weibull")),
                     Scenario("True log-normal: AIC picks log-normal", "stat_parametric", g_parametric("lognormal"), ex_param("lognormal")),
                     Scenario("True exponential: AIC picks exponential", "stat_parametric", g_parametric("exponential"), ex_param("exponential"))],
 "stat_lmm": [Scenario("Random intercept, slope 0.5 (REML, Satterthwaite), 5 per cluster", "stat_lmm", g_lmm(), ex_lmm, truth=0.5),
              Scenario("Unbalanced cluster sizes", "stat_lmm", g_lmm(unbal=True), ex_lmm, truth=0.5),
              Scenario("t(3) random intercepts (robustness)", "stat_lmm", g_lmm(tdf=3), ex_lmm, truth=0.5),
              Scenario("Random slope present and fitted", "stat_lmm", g_lmm(slope=True), ex_lmm, truth=0.5, ns=(100, 250, 500, 1000))],
 "stat_glmm": [Scenario("Binomial random intercept, beta=0.5, sd=1, 10 per cluster", "stat_glmm", g_glmm(), ex_glmm, truth=0.5, ns=(100, 250, 500, 1000)),
               Scenario("Poisson random intercept", "stat_glmm", g_glmm(fam="poisson"), ex_glmm, truth=0.5, ns=(100, 250, 500, 1000), note="the log-mean slope; random effect enters with coefficient 0.5 (truth 0.5)")],
 "stat_gee": [Scenario("Gaussian, exchangeable (correct), robust-SE coverage", "stat_gee", g_gee(), ex_gee, truth=0.5),
              Scenario("Gaussian, independence working correlation (misspecified)", "stat_gee", g_gee(misspec=True), ex_gee, truth=0.5),
              Scenario("Binomial, exchangeable (marginal log-odds attenuated)", "stat_gee", g_gee(fam="binomial"), ex_gee, truth=None, note="marginal effect differs from conditional; used for coverage under truth computed by simulation is out of scope: only convergence is reported")],
 "stat_icc": [Scenario("ICC(2,1): truth 1/2.2", "stat_icc", g_icc(), ex_icc("ICC(2,1)"), truth=1 / 2.2, ns=(20, 50, 100, 250, 500)),
              Scenario("ICC(3,1): truth 1/2", "stat_icc", g_icc(), ex_icc("ICC(3,1)"), truth=0.5, ns=(20, 50, 100, 250, 500))],
 "stat_bootstrap": [Scenario("Mean, normal: percentile vs BCa coverage", "stat_bootstrap", g_boot(), ex_boot("BCa"), truth=0.0, reps=2000),
                    Scenario("Mean, exponential (skewed): BCa", "stat_bootstrap", g_boot(dist="exp"), lambda r: {**ex_boot("BCa")(r)}, truth=1.0, reps=2000),
                    Scenario("Mean, exponential: percentile", "stat_bootstrap", g_boot(dist="exp"), ex_boot("Percentile"), truth=1.0, reps=2000),
                    Scenario("Median, normal: percentile", "stat_bootstrap", g_boot(stat="median"), ex_boot("Percentile"), truth=0.0, reps=2000)],
 "stat_permutation": [Scenario("H0, normal (n_perm=999)", "stat_permutation", g_perm(), ex_perm, h0=True), Scenario("H0, exponential", "stat_permutation", g_perm(dist="exp"), ex_perm, h0=True),
                      Scenario("Power, shift 0.5", "stat_permutation", g_perm(shift=0.5), ex_perm, h0=False)],
 "stat_meta_fixed": [Scenario("Homogeneous, theta=0.3 (fixed-effect, correct model)", "stat_meta_fixed", g_meta(model="fixed"), ex_meta, truth=0.3, ns=KG),
                     Scenario("Heterogeneous tau2=0.1 (fixed-effect misspecified)", "stat_meta_fixed", g_meta(tau2=0.1, model="fixed"), ex_meta, truth=0.3, ns=KG)],
 "stat_meta_random": [Scenario("tau2=0.1, REML + Wald", "stat_meta_random", g_meta(tau2=0.1), ex_meta, truth=0.3, ns=KG),
                      Scenario("tau2=0.1, REML + HKSJ", "stat_meta_random", g_meta(tau2=0.1, ci="hksj"), ex_meta, truth=0.3, ns=KG),
                      Scenario("tau2=0.1, DL + Wald", "stat_meta_random", g_meta(tau2=0.1, tau_method="dl"), ex_meta, truth=0.3, ns=KG),
                      Scenario("tau2=0 (homogeneous), REML + HKSJ", "stat_meta_random", g_meta(tau2=0.0, ci="hksj"), ex_meta, truth=0.3, ns=KG)],
}
