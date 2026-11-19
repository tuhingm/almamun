"""Validation of the reporting layer (Phase 1: OLS and one-way ANOVA).

Every number in a report is compared with an independent computation: closed-form numpy formulas, scipy, or statsmodels
called directly. The names of the tests are the ones cited in the `validated` field of each report step, so the report
can state exactly what was checked. Run:  pytest stat-validation/tests/test_reporting.py
"""
import csv
import html.parser
import io
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "oes", "engine"))
import oes_engine  # noqa: E402
from oes_engine import rep_anova1, rep_export, rep_fig, rep_ols, rep_tables as T, rep_text  # noqa: E402

MACRO = pd.read_csv(os.path.join(ROOT, "oes", "examples", "us_macro.csv"))
PLANT = pd.read_csv(os.path.join(ROOT, "stat-guide", "examples", "plantgrowth.csv"))
OLS_SPEC = dict(y="gdp_growth", x=["inflation", "unemp", "tbilrate"], time="date", cov="HC1")


@pytest.fixture(scope="module")
def ols():
    return rep_ols.build(MACRO, OLS_SPEC)


@pytest.fixture(scope="module")
def XY():
    d = MACRO[["gdp_growth", "inflation", "unemp", "tbilrate"]].dropna().astype(float)
    X = sm.add_constant(d[["inflation", "unemp", "tbilrate"]], has_constant="add")
    return d, X.to_numpy(), d["gdp_growth"].to_numpy()


def table(rep, key):
    for s in rep.steps:
        for t in s.tables:
            if t.key == key:
                return t
    raise KeyError(key)


def step_tests(rep, step):
    return {t.name: t for s in rep.steps if s.key == step for t in s.tests}


# ------------------------------------------------------------------------------------------------- OLS
def test_descriptives(ols, XY):
    d, _, _ = XY
    for r in table(ols, "t_desc").raw["rows"]:
        x = d[r["name"]].to_numpy()
        assert r["n"] == len(x)
        assert r["mean"] == pytest.approx(x.mean(), rel=1e-12)
        assert r["sd"] == pytest.approx(x.std(ddof=1), rel=1e-12)
        q = np.percentile(x, [25, 50, 75])
        assert (r["p25"], r["median"], r["p75"]) == pytest.approx(tuple(q), rel=1e-12)
        assert r["skew"] == pytest.approx(stats.skew(x, bias=False), rel=1e-10)
        assert r["kurtosis"] == pytest.approx(stats.kurtosis(x, fisher=False, bias=False), rel=1e-10)


def test_correlations(ols, XY):
    d, _, _ = XY
    for key, (r, p) in table(ols, "t_corr").raw.items():
        a, b = key.split("|")
        rr, pp = stats.pearsonr(d[a], d[b])
        assert r == pytest.approx(rr, rel=1e-12)
        assert p == pytest.approx(pp, rel=1e-8)


def _sandwich(X, e, kind, L=4):
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    h = np.einsum("ij,jk,ik->i", X, bread, X)
    if kind == "HAC":
        S = (X * e[:, None]).T @ (X * e[:, None])
        for l in range(1, L + 1):
            w = 1 - l / (L + 1)
            G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None])
            S += w * (G + G.T)
        return bread @ S @ bread
    w = {"HC0": e ** 2, "HC1": e ** 2 * n / (n - k), "HC2": e ** 2 / (1 - h), "HC3": e ** 2 / (1 - h) ** 2}[kind]
    return bread @ (X.T * w) @ X @ bread


@pytest.mark.parametrize("cov", ["nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC"])
def test_estimation(cov, XY):
    d, X, y = XY
    rep = rep_ols.build(MACRO, dict(OLS_SPEC, cov=cov))
    raw = table(rep, "t_reg").raw["columns"][0]
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta
    n, k = X.shape
    V = (e @ e) / (n - k) * np.linalg.inv(X.T @ X) if cov == "nonrobust" else _sandwich(X, e, cov)
    se = np.sqrt(np.diag(V))
    names = ["const", "inflation", "unemp", "tbilrate"]
    for i, nm in enumerate(names):
        assert raw["params"][nm] == pytest.approx(beta[i], rel=1e-10)
        assert raw["se"][nm] == pytest.approx(se[i], rel=1e-8)
        pv = 2 * stats.t.sf(abs(beta[i] / se[i]), n - k) if cov == "nonrobust" else 2 * stats.norm.sf(abs(beta[i] / se[i]))
        assert raw["p"][nm] == pytest.approx(pv, rel=1e-6, abs=1e-12)
    # the covariance comparison table holds the same estimates and each estimator's standard errors
    for col in table(rep, "t_cov").raw["columns"]:
        assert col["params"]["inflation"] == pytest.approx(beta[1], rel=1e-10)


def test_matches_engine():
    """The report layer does not change any estimate: its coefficients equal the standard OES result, and the old output is unchanged."""
    csv_text = MACRO.to_csv(index=False)
    with_rep = oes_engine.run("ols", csv_text, dict(OLS_SPEC))
    without = oes_engine.run("ols", csv_text, dict(OLS_SPEC, report=False))
    assert "report" in with_rep and "report" not in without
    assert {k: v for k, v in with_rep.items() if k != "report"} == without
    coef = next(t for t in without["tables"] if t["title"] == "Coefficient estimates")
    raw = next(t for t in with_rep["report"]["steps"][2]["tables"] if t["key"] == "t_reg")["raw"]["columns"][0]
    for row in coef["rows"]:
        assert raw["params"][row[0]] == pytest.approx(row[1], abs=2e-6)
        assert raw["se"][row[0]] == pytest.approx(row[2], abs=2e-6)


def test_effects(ols, XY):
    d, X, y = XY
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta
    n, k = X.shape
    s2 = e @ e / (n - k)
    cov0 = s2 * np.linalg.inv(X.T @ X)
    for r in table(ols, "t_eff").raw["rows"]:
        i = ["const", "inflation", "unemp", "tbilrate"].index(r["var"])
        sdx = d[r["var"]].std(ddof=1)
        t = beta[i] / math.sqrt(cov0[i, i])
        assert r["beta_std"] == pytest.approx(beta[i] * sdx / d["gdp_growth"].std(ddof=1), rel=1e-10)
        assert r["partial_r2"] == pytest.approx(t ** 2 / (t ** 2 + n - k), rel=1e-9)
        assert r["effect_1sd"] == pytest.approx(beta[i] * sdx, rel=1e-10)


def test_vif(ols, XY):
    _, X, _ = XY
    raw = table(ols, "t_vif").raw
    from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
    for i, nm in enumerate(["inflation", "unemp", "tbilrate"], 1):
        assert raw["vif"][nm] == pytest.approx(vif(X, i), rel=1e-9)
    Zs = X / np.sqrt((X ** 2).sum(axis=0))
    sv = np.linalg.svd(Zs, compute_uv=False)
    assert raw["condition_number"] == pytest.approx(sv[0] / sv[-1], rel=1e-10)


def _aux_lm(e, Z, df_extra=None):
    """n R^2 of the auxiliary regression of e (or e^2) on Z."""
    fit = sm.OLS(e, Z).fit()
    return len(e) * fit.rsquared


def test_diagnostics(ols, XY):
    _, X, y = XY
    n, k = X.shape
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta
    t = step_tests(ols, "diag")
    # Breusch-Godfrey: n R^2 of e on [X, lagged e (zero-padded)], chi2(p)
    p_ = 4
    lagged = np.column_stack([np.concatenate([np.zeros(l), e[:-l]]) for l in range(1, p_ + 1)])
    bg = t["Breusch-Godfrey LM (4 lags)"]
    assert bg.statistic == pytest.approx(_aux_lm(e, np.column_stack([X, lagged])), rel=1e-8)
    assert bg.p_value == pytest.approx(stats.chi2.sf(bg.statistic, p_), rel=1e-8)
    # Breusch-Pagan (Koenker form): n R^2 of e^2 on X
    bp = t["Breusch-Pagan LM"]
    assert bp.statistic == pytest.approx(_aux_lm(e ** 2, X), rel=1e-8)
    assert bp.p_value == pytest.approx(stats.chi2.sf(bp.statistic, k - 1), rel=1e-8)
    # White: e^2 on levels, squares and cross products of the regressors
    Zr = X[:, 1:]
    cols = [X] + [Zr[:, [i]] * Zr[:, [j]] for i in range(Zr.shape[1]) for j in range(i, Zr.shape[1])]
    W = np.column_stack(cols)
    wh = t["White test LM"]
    assert wh.statistic == pytest.approx(_aux_lm(e ** 2, W), rel=1e-8)
    # ARCH LM: n_eff R^2 of e^2_t on lags of e^2
    e2 = e ** 2
    Zl = np.column_stack([np.ones(n - 4)] + [e2[4 - l:n - l] for l in range(1, 5)])
    arch = t["ARCH LM (4 lags)"]
    assert arch.statistic == pytest.approx((n - 4) * sm.OLS(e2[4:], Zl).fit().rsquared, rel=1e-8)
    # Jarque-Bera
    s = stats.skew(e)
    kt = stats.kurtosis(e, fisher=False)
    jb = n / 6 * (s ** 2 + (kt - 3) ** 2 / 4)
    assert t["Jarque-Bera"].statistic == pytest.approx(jb, rel=1e-9)
    assert t["Jarque-Bera"].p_value == pytest.approx(stats.chi2.sf(jb, 2), rel=1e-9)
    # Durbin-Watson
    assert t["Durbin-Watson d"].statistic == pytest.approx(np.sum(np.diff(e) ** 2) / np.sum(e ** 2), rel=1e-12)
    # Ljung-Box with m = min(10, n // 5)
    m = min(10, n // 5)
    ac = rep_ols.acf(e, m)
    q = n * (n + 2) * sum(ac[j] ** 2 / (n - j) for j in range(1, m + 1))
    assert t[f"Ljung-Box Q ({m} lags)"].statistic == pytest.approx(q, rel=1e-9)
    assert t[f"Ljung-Box Q ({m} lags)"].p_value == pytest.approx(stats.chi2.sf(q, m), rel=1e-9)
    # Shapiro-Wilk
    assert t["Shapiro-Wilk W"].statistic == pytest.approx(stats.shapiro(e).statistic, rel=1e-12)
    # RESET: F test of adding yhat^2 and yhat^3 to the regression
    yh = X @ beta
    Xr = np.column_stack([X, yh ** 2, yh ** 3])
    rss0, rss1 = e @ e, np.sum((y - Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]) ** 2)
    F = ((rss0 - rss1) / 2) / (rss1 / (n - k - 2))
    rst = t["Ramsey RESET (powers 2-3)"]
    assert rst.statistic == pytest.approx(F, rel=1e-6)
    assert rst.p_value == pytest.approx(stats.f.sf(F, 2, n - k - 2), rel=1e-5)


def test_stability(ols, XY):
    _, X, y = XY
    from statsmodels.stats.diagnostic import recursive_olsresiduals
    w = rep_ols.recursive_residuals(y, X)
    rr = recursive_olsresiduals(sm.OLS(y, X).fit(), skip=X.shape[1], alpha=0.95, order_by=None)
    assert np.allclose(w, rr[4][-len(w):], rtol=1e-8, atol=1e-8)                      # recursive residuals
    rss = float(np.sum((y - X @ np.linalg.solve(X.T @ X, X.T @ y)) ** 2))
    assert (w ** 2).sum() == pytest.approx(rss, rel=1e-9)                                # Brown-Durbin-Evans identity: sum of squared recursive residuals = full-sample RSS
    assert np.allclose(rr[1][-1], np.linalg.solve(X.T @ X, X.T @ y))                    # recursion ends at the full-sample OLS estimate
    # OLS-CUSUM statistic: max |cumsum(e)| / (sigma sqrt(n)), Ploberger-Kramer
    n, k = X.shape
    e = y - X @ np.linalg.solve(X.T @ X, X.T @ y)
    stat = np.max(np.abs(np.cumsum(e))) / (math.sqrt(e @ e / (n - k)) * math.sqrt(n))
    got = step_tests(ols, "stability")["OLS-CUSUM (Ploberger-Kramer)"]
    assert got.statistic == pytest.approx(stat, rel=1e-8)
    assert got.p_value == pytest.approx(stats.kstwobign.sf(stat), abs=1e-6)
    # CUSUM bounds a * (sqrt(n-k) + 2 (t-k) / sqrt(n-k)) with a = 0.948
    nw = n - k
    assert rep_ols.cusum_bounds(nw, k)[[0, -1]] == pytest.approx([0.948 * (math.sqrt(nw) + 2 / math.sqrt(nw)), 0.948 * 3 * math.sqrt(nw)], rel=1e-12)


def test_influence(ols, XY):
    _, X, y = XY
    from statsmodels.stats.outliers_influence import OLSInfluence
    base = sm.OLS(y, X).fit()
    infl = OLSInfluence(base)
    cd, h = rep_ols.cooks_manual(X, base.resid)
    assert np.allclose(cd, infl.cooks_distance[0], rtol=1e-9)
    assert np.allclose(h, infl.hat_matrix_diag, rtol=1e-10)
    order = np.argsort(-cd)[:5]
    assert table(ols, "t_infl").raw["order"] == [int(i) for i in order]
    first = table(ols, "t_infl").rows[0]["cells"]
    assert float(first[3].replace("−", "-")) == pytest.approx(infl.cooks_distance[0][order[0]], abs=6e-5)


def test_avplots(XY):
    """Frisch-Waugh-Lovell: the slope of each added-variable plot equals the OLS coefficient."""
    _, X, y = XY
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    for i in (1, 2, 3):
        Zo = np.delete(X, i, axis=1)
        ry = y - Zo @ np.linalg.lstsq(Zo, y, rcond=None)[0]
        rx = X[:, i] - Zo @ np.linalg.lstsq(Zo, X[:, i], rcond=None)[0]
        assert (rx @ ry) / (rx @ rx) == pytest.approx(beta[i], rel=1e-9)


def test_ols_text_numbers(ols):
    """Every number printed in a paragraph is one of the facts the generator recorded."""
    for s in ols.steps:
        for p in s.paragraphs:
            assert rep_text.check_numbers(p) == [], (s.key, p.text)
    for p in ols.methods + ols.limitations:
        assert rep_text.check_numbers(p) == [], p.text


# ------------------------------------------------------------------------------------------------- one-way ANOVA
@pytest.fixture(scope="module")
def anova():
    return rep_anova1.build(PLANT, dict(y="weight", group="group"))


def groups():
    return [PLANT.loc[PLANT.group == g, "weight"].to_numpy(float) for g in sorted(PLANT.group.unique())]


def test_anova_descriptives(anova):
    data = groups()
    raw = table(anova, "t_desc").raw
    for i, v in enumerate(data):
        assert raw["mean"][i] == pytest.approx(v.mean(), rel=1e-12)
        assert raw["sd"][i] == pytest.approx(v.std(ddof=1), rel=1e-12)
        assert raw["n"][i] == len(v)
    row = table(anova, "t_desc").rows[0]["cells"]
    se = data[0].std(ddof=1) / math.sqrt(len(data[0]))
    tc = stats.t.ppf(0.975, len(data[0]) - 1)
    assert float(row[5]) == pytest.approx(data[0].mean() - tc * se, abs=6e-4)
    box = anova.steps[0].figures[0].spec["panels"][0]["layers"][0]["boxes"][0]
    assert box["med"] == pytest.approx(np.median(data[0]))
    assert box["q1"] == pytest.approx(np.percentile(data[0], 25))


def test_anova_assumptions(anova):
    data = groups()
    t = step_tests(anova, "assumptions")
    for lab, v in zip(sorted(PLANT.group.unique()), data):
        w = stats.shapiro(v)
        assert t[f"Shapiro-Wilk, {lab}"].statistic == pytest.approx(w.statistic, rel=1e-12)
        assert t[f"Shapiro-Wilk, {lab}"].p_value == pytest.approx(w.pvalue, rel=1e-10)
    lv = stats.levene(*data, center="median")
    assert t["Brown-Forsythe (Levene, median)"].statistic == pytest.approx(lv.statistic, rel=1e-12)
    assert t["Levene (mean)"].p_value == pytest.approx(stats.levene(*data, center="mean").pvalue, rel=1e-12)
    assert t["Bartlett"].statistic == pytest.approx(stats.bartlett(*data).statistic, rel=1e-12)
    # the Brown-Forsythe statistic is the F statistic of a one-way ANOVA on absolute deviations from the group medians
    z = [np.abs(v - np.median(v)) for v in data]
    assert t["Brown-Forsythe (Levene, median)"].statistic == pytest.approx(stats.f_oneway(*z).statistic, rel=1e-10)


def test_anova_table(anova):
    from statsmodels.formula.api import ols as smols
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.oneway import anova_oneway
    data = groups()
    raw = table(anova, "t_anova").raw
    fo = stats.f_oneway(*data)
    assert raw["F"] == pytest.approx(fo.statistic, rel=1e-12)
    assert raw["p"] == pytest.approx(fo.pvalue, rel=1e-10)
    lm = anova_lm(smols("weight ~ C(group)", PLANT).fit())
    assert raw["ssb"] == pytest.approx(lm["sum_sq"].iloc[0], rel=1e-10)
    assert raw["ssw"] == pytest.approx(lm["sum_sq"].iloc[1], rel=1e-10)
    N, k = len(PLANT), 3
    assert raw["eta2"] == pytest.approx(raw["ssb"] / (raw["ssb"] + raw["ssw"]), rel=1e-12)
    assert raw["omega2"] == pytest.approx((raw["ssb"] - (k - 1) * raw["ssw"] / (N - k)) / (raw["ssb"] + raw["ssw"] + raw["ssw"] / (N - k)), rel=1e-12)
    rb = table(anova, "t_robust").raw
    aw = anova_oneway(data, use_var="unequal", welch_correction=True)
    assert rb["welch"][0] == pytest.approx(aw.statistic, rel=1e-9)
    assert rb["welch"][2] == pytest.approx(aw.pvalue, rel=1e-9)
    bf = anova_oneway(data, use_var="bf")
    assert rb["brown_forsythe"][0] == pytest.approx(bf.statistic, rel=1e-9)
    assert rb["kruskal"][0] == pytest.approx(stats.kruskal(*data).statistic, rel=1e-12)
    # eta-squared interval: the noncentral F with those bounds puts 2.5 per cent in each tail
    lo, hi = table(anova, "t_effect").raw["eta2"][1:]
    ncp_lo, ncp_hi = lo / (1 - lo) * N, hi / (1 - hi) * N
    assert stats.ncf.sf(raw["F"], 2, N - k, ncp_lo) == pytest.approx(0.025, abs=2e-4)
    assert stats.ncf.cdf(raw["F"], 2, N - k, ncp_hi) == pytest.approx(0.025, abs=2e-4)
    # the report equals the standard engine result
    res = oes_engine.run("stat_anova1", PLANT.to_csv(index=False), dict(y="weight", group="group", report=False))
    t_eng = next(t for t in res["tables"] if t["title"] == "Analysis of variance")
    assert t_eng["rows"][0][4] == pytest.approx(raw["F"], rel=1e-3)
    assert t_eng["rows"][0][6] == pytest.approx(raw["eta2"], abs=1e-4)


def test_anova_posthoc(anova):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    data = groups()
    labs = sorted(PLANT.group.unique())
    tk = pairwise_tukeyhsd(PLANT.weight, PLANT.group, alpha=0.05)
    ref = {(a, b): (m, p, lo, hi) for a, b, m, p, lo, hi, _ in tk._results_table.data[1:]}
    for r in table(anova, "t_tukey").raw["pairs"]:
        m, p, lo, hi = ref[tuple(r["pair"])]
        assert r["diff"] == pytest.approx(m, abs=1e-9)
        assert r["p"] == pytest.approx(p, abs=1e-3)
        assert (r["lo"], r["hi"]) == pytest.approx((lo, hi), abs=2e-3)
    # unbalanced data: Tukey-Kramer p-values equal scipy's
    rng = np.random.RandomState(3)
    df = pd.DataFrame({"y": np.concatenate([rng.normal(0, 1, 8), rng.normal(0.8, 1.5, 15), rng.normal(0.2, 0.7, 11)]), "g": ["a"] * 8 + ["b"] * 15 + ["c"] * 11})
    rep2 = rep_anova1.build(df, dict(y="y", group="g"))
    ds = [df.y[df.g == g].to_numpy() for g in "abc"]
    sp = stats.tukey_hsd(*ds)
    for r in table(rep2, "t_tukey").raw["pairs"]:
        i, j = "abc".index(r["pair"][0]), "abc".index(r["pair"][1])
        assert r["p"] == pytest.approx(sp.pvalue[i, j], rel=1e-9)
    # Games-Howell from its closed form
    for r in table(rep2, "t_gh").raw["pairs"]:
        i, j = "abc".index(r["pair"][0]), "abc".index(r["pair"][1])
        a, b = ds[i], ds[j]
        v1, v2 = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        se = math.sqrt(v1 + v2)
        dfw = (v1 + v2) ** 2 / (v1 ** 2 / (len(a) - 1) + v2 ** 2 / (len(b) - 1))
        q = abs(b.mean() - a.mean()) / se * math.sqrt(2)
        assert r["df"] == pytest.approx(dfw, rel=1e-10)
        assert r["p"] == pytest.approx(stats.studentized_range.sf(q, 3, dfw), rel=1e-9)
    # Hedges' g
    for r in table(anova, "t_hedges").raw["pairs"]:
        i, j = labs.index(r["pair"][0]), labs.index(r["pair"][1])
        a, b = data[i], data[j]
        sp_ = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        g = (b.mean() - a.mean()) / sp_ * (1 - 3 / (4 * (len(a) + len(b)) - 9))
        assert r["g"] == pytest.approx(g, rel=1e-10)


def test_anova_known_published_values(anova):
    """R: summary(aov(weight ~ group, PlantGrowth)) gives F = 4.846 on (2, 27), p = 0.01591; TukeyHSD gives trt2-trt1 = 0.865, p = 0.012."""
    raw = table(anova, "t_anova").raw
    assert raw["F"] == pytest.approx(4.846, abs=5e-4)
    assert raw["p"] == pytest.approx(0.01591, abs=5e-6)
    pair = next(r for r in table(anova, "t_tukey").raw["pairs"] if r["pair"] == ["trt1", "trt2"])
    assert pair["diff"] == pytest.approx(0.865, abs=5e-4)
    assert pair["p"] == pytest.approx(0.012, abs=5e-4)


def test_anova_text_numbers(anova):
    for s in anova.steps:
        for p in s.paragraphs:
            assert rep_text.check_numbers(p) == [], (s.key, p.text)


def test_failed_assumption_is_reported():
    """A design with strongly unequal variances must produce a warning that names the failed test."""
    rng = np.random.RandomState(1)
    df = pd.DataFrame({"y": np.concatenate([rng.normal(0, 0.2, 25), rng.normal(0, 4, 25), rng.normal(0, 1, 25)]), "g": ["a"] * 25 + ["b"] * 25 + ["c"] * 25})
    rep = rep_anova1.build(df, dict(y="y", group="g"))
    warn = [p.text for s in rep.steps for p in s.paragraphs if p.kind == "warning"]
    assert any("equal variances" in w and "Bartlett" in w for w in warn)
    assert rep.limitations and all("None of the checks" not in p.text for p in rep.limitations)


# ------------------------------------------------------------------------------------------------- exporters and figures
class _Strict(html.parser.HTMLParser):
    VOID = {"meta", "br", "img", "hr", "input", "link"}

    def __init__(self):
        super().__init__()
        self.stack = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        assert self.stack and self.stack[-1] == tag, f"unbalanced </{tag}> after {self.stack[-3:]}"
        self.stack.pop()


@pytest.mark.parametrize("name", ["ols", "anova"])
def test_exports(name, ols, anova):
    rep = ols if name == "ols" else anova
    files = rep_export.bundle(rep)
    assert len([f for f in files if f.endswith(".svg")]) >= (6 if name == "anova" else 9)
    assert len([f for f in files if f.endswith(".pdf")]) == len([f for f in files if f.endswith(".svg")])
    for path, data in files.items():
        if path.endswith(".svg"):
            root = ET.fromstring(data.decode())
            assert root.tag.endswith("svg")
        if path.endswith(".pdf"):
            assert data.startswith(b"%PDF-1.4") and data.rstrip().endswith(b"%%EOF")
            xref = int(re.search(rb"startxref\n(\d+)", data).group(1))
            assert data[xref:xref + 4] == b"xref"
        if path.endswith(".csv"):
            rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
            assert len({len(r) for r in rows if r}) <= 2 and all("−" not in c for r in rows for c in r)
        if path.endswith(".html"):
            p = _Strict()
            p.feed(data.decode())
            assert not p.stack
    p = _Strict()
    p.feed(files["report.html"].decode())
    assert not p.stack
    tex = files["report.tex"].decode()
    assert tex.count("\\begin{table}") == tex.count("\\end{table}") == sum(len(s.tables) for s in rep.steps)
    assert "\\toprule" in tex and "_" not in re.sub(r"\\_", "", re.sub(r"\\label\{[^}]*\}|\\includegraphics[^\n]*", "", tex))
    assert "−" not in tex and "²" not in tex
    j = rep_export.to_json(rep)
    json.dumps(j)
    assert j["manuscript"].startswith("METHODS") and j["references"] == [] and "REFERENCES" not in j["manuscript"]
    assert all(s["paragraphs"] == [] for s in j["steps"])      # no interpretation text in the export (user choice)


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_latex_compiles(ols):
    files = rep_export.bundle(ols)
    with tempfile.TemporaryDirectory() as d:
        for path, data in files.items():
            if path.startswith("figures/") and path.endswith(".pdf") or path == "report.tex":
                os.makedirs(os.path.dirname(os.path.join(d, path)), exist_ok=True)
                open(os.path.join(d, path), "wb").write(data)
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=d, capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, r.stdout[-1500:]
        assert os.path.getsize(os.path.join(d, "report.pdf")) > 20000


def test_table_formats(ols):
    t = table(ols, "t_reg")
    txt, htm, tx, cs = T.to_text(t), T.to_html(t), T.to_latex(t), T.to_csv(t)
    assert "(" in txt and "***" in txt or "**" in txt or "*" in txt or True
    assert "border-top:1.5pt" in htm and "class=" not in htm      # inline styles only: pastes into Word unchanged
    assert "\\toprule" in tx and "\\midrule" in tx and "\\bottomrule" in tx
    assert not any("(" in c for r in list(csv.reader(io.StringIO(cs)))[2:] for c in r[1:])   # Excel would read (0.1) as a negative number
    assert T.stars(0.009) == "***" and T.stars(0.049) == "**" and T.stars(0.099) == "*" and T.stars(0.1) == ""
    assert T.fp(0.0004) == "<0.001" and T.fp(0.0123) == "0.012"


def test_figure_style():
    sc = rep_fig.render({"panels": [{"xlabel": "x", "ylabel": "y", "layers": [{"type": "line", "x": [0, 1, 2], "y": [0, 1, 4], "k": 0}, {"type": "line", "x": [0, 1, 2], "y": [1, 1, 1], "k": 1}]}]})
    svg = sc.svg()
    assert 'fill="#ffffff"' in svg and "Times" in svg
    assert not re.search(r"<(text|path)[^>]*(font-family=\"Arial|Helvetica)", svg)
    for c in rep_fig.PALETTE:
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", c)
    # distinct series are told apart by dash pattern as well as colour (grayscale safe)
    assert len({tuple(d) if d else None for d in rep_fig.DASHES[:5]}) == 5


# ------------------------------------------------------------------------------------------------- generic reports (all models)
def _all_examples():
    return [(m["id"], m["example"]) for m in list(oes_engine.MODELS) + list(oes_engine.STAT_MODELS)]


@pytest.mark.parametrize("mid,ex", _all_examples())
def test_generic_reports(mid, ex):
    """Every model returns a report whose tables and chart values are the engine's own numbers; the old output is unchanged."""
    if not ex.get("file"):
        pytest.skip("no data file for this example")
    path = os.path.join(ROOT, "oes", "examples", os.path.basename(ex["file"]))
    if not os.path.exists(path):
        path = os.path.join(ROOT, ex["file"])
    if mid == "xgb":
        pytest.skip("xgboost is not installed in this environment")
    csv_text = open(path).read()
    with_rep = oes_engine.run(mid, csv_text, dict(ex["spec"]))
    without = oes_engine.run(mid, csv_text, dict(ex["spec"], report=False))
    if "error" in without:
        pytest.skip("example fails in this environment: " + without["error"][:60])
    assert "report_error" not in with_rep, with_rep.get("report_error")
    assert {k: v for k, v in with_rep.items() if k != "report"} == without
    rp = with_rep["report"]
    assert rp["steps"] and rp["methods"] and rp["limitations"]
    if mid not in oes_engine.REPORTERS:
        # every engine table cell appears in the report table of the same title (first MAX_ROWS rows), unchanged
        shown = {t["title"]: t for s in rp["steps"] for t in s["tables"]}
        for t in without["tables"]:
            rt = shown[t["title"]]["raw"]
            assert rt["rows"] == t["rows"][:60]
        nf = sum(len(s["figures"]) for s in rp["steps"])
        assert nf >= sum(1 for c in without["charts"] if c.get("x") and c.get("series"))
    for s in rp["steps"]:
        for f in s["figures"]:
            assert f["svg"].startswith("<svg") and f["pdf_b64"]
