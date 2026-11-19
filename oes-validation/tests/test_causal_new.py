"""Validation of causal-inference models added after the first release: weak-instrument-robust IV inference."""
import os, re, sys, warnings
import numpy as np, pandas as pd, pytest
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "oes", "engine")); sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import oes_engine as o
from oes_engine import causal_extra

CX = os.path.join(ROOT, "causal-inference-guide", "examples")
def card(): return pd.read_csv(os.path.join(CX, "card_full.csv"))
CTRL = ["exper", "expersq", "black", "smsa", "south"]
def wspec(zs, **kw): return dict(outcome="lwage", treat="educ", instruments=zs, controls=CTRL, **kw)
def est(r, name): return next(x for x in r["tables"][0]["rows"] if x[0].startswith(name))
def interval(txt): return [tuple(float(v) for v in m) for m in re.findall(r"\[(-?[\d.]+|-inf), (-?[\d.]+|inf)\]", txt)]


def test_weakiv_registered():
    m = o.BY_ID["weakiv"]; assert m["group"] == "Causal inference" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_weakiv_2sls_matches_iv_model():
    d = card(); r = causal_extra.weakiv(d, wspec(["nearc4", "nearc2"], sims=200))
    iv = o.run("iv", d.to_csv(index=False), dict(outcome="lwage", treat="educ", instruments=["nearc4", "nearc2"], controls=CTRL))
    assert float(est(r, "2SLS")[1]) == pytest.approx(float(iv["tables"][0]["rows"][0][1]), abs=2e-4)


def test_weakiv_ar_set_endpoints_have_pvalue_alpha():
    import statsmodels.api as sm
    d = card(); r = causal_extra.weakiv(d, wspec(["nearc4"], sims=200)); lo, hi = interval(est(r, "Anderson-Rubin")[3])[0]
    X = sm.add_constant(d[CTRL + ["nearc4"]])
    for b0 in (lo, hi):
        F = sm.OLS(d.lwage - b0 * d.educ, X).fit().f_test(np.eye(X.shape[1])[-1:]).fvalue
        assert stats.chi2.sf(float(F), 1) == pytest.approx(0.05, abs=0.006)


def test_weakiv_single_instrument_all_sets_agree_and_liml_equals_2sls():
    r = causal_extra.weakiv(card(), wspec(["nearc4"], sims=200))
    assert est(r, "Anderson-Rubin")[3] == est(r, "Lagrange")[3] == est(r, "Conditional")[3]
    assert float(est(r, "LIML")[1]) == pytest.approx(float(est(r, "2SLS")[1]), abs=1e-6)


def test_weakiv_strong_instruments_sets_close_to_wald():
    r = causal_extra.weakiv(card(), wspec(["fatheduc", "motheduc"], sims=400)); w = interval(est(r, "2SLS")[3])[0]; c = interval(est(r, "Conditional")[3])[0]
    assert abs(c[0] - w[0]) < 0.01 and abs(c[1] - w[1]) < 0.01


def test_weakiv_weak_instruments_give_wide_or_unbounded_sets():
    r = causal_extra.weakiv(card(), wspec(["nearc4", "nearc2"], sims=400))
    assert "unbounded" in est(r, "Lagrange")[3] or interval(est(r, "Lagrange")[3])[0][1] - interval(est(r, "Lagrange")[3])[0][0] > 1
    w = interval(est(r, "2SLS")[3])[0]; a = interval(est(r, "Anderson-Rubin")[3])[0]; assert a[1] - a[0] > 2 * (w[1] - w[0])


def weak_sim(seed, pi=0.08, n=250, k=2, beta=1.0, rho=0.8, bad=0.0):
    rng = np.random.default_rng(seed); Z = rng.normal(size=(n, k)); v = rng.normal(size=n); u = rho * v + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    x = Z @ np.full(k, pi) + v; y = beta * x + u + bad * Z[:, 0]
    return pd.DataFrame(np.column_stack([y, x, Z]), columns=["y", "x"] + [f"z{i}" for i in range(k)])


def test_weakiv_coverage_robust_sets_vs_wald_under_weak_instruments():
    cov = dict(w=0, ar=0, clr=0); R = 150
    for s in range(R):
        r = causal_extra.weakiv(weak_sim(s), dict(outcome="y", treat="x", instruments=["z0", "z1"], sims=400))
        for key, name in (("w", "2SLS"), ("ar", "Anderson-Rubin"), ("clr", "Conditional")):
            txt = est(r, name)[3]; cov[key] += ("unbounded" in txt) or any(a <= 1.0 <= b for a, b in interval(txt))
    assert cov["ar"] / R >= 0.90 and cov["clr"] / R >= 0.90 and cov["w"] / R < cov["clr"] / R


def test_weakiv_overidentification_test_flags_invalid_instrument():
    ok = causal_extra.weakiv(weak_sim(1, pi=0.5, n=800), dict(outcome="y", treat="x", instruments=["z0", "z1"], sims=200))
    bad = causal_extra.weakiv(weak_sim(1, pi=0.5, n=800, bad=0.5), dict(outcome="y", treat="x", instruments=["z0", "z1"], sims=200))
    assert float(ok["tables"][2]["rows"][1][2]) > 0.01 and float(bad["tables"][2]["rows"][1][2]) < 0.001


def test_weakiv_errors():
    d = card()
    assert "Column(s) not found" in o.run("weakiv", d.to_csv(index=False), wspec(["nope"]))["error"]
    assert "collinear" in o.run("weakiv", d.assign(z2=d.nearc4).to_csv(index=False), wspec(["nearc4", "z2"]))["error"]
    assert "confidence level" in o.run("weakiv", d.to_csv(index=False), wspec(["nearc4"], level=0.2))["error"]


# ---- randomised experiments ------------------------------------------------------------------------------------
def nsw(): return pd.read_csv(os.path.join(CX, "nsw_experiment.csv"))
COV = ["age", "educ", "black", "hisp", "married", "nodegree", "re74", "re75"]
ES = dict(outcome="re78", treat="train", covariates=COV, draws=1000)
def erow(r, name): return next(x for x in r["tables"][0]["rows"] if x[0].startswith(name))


def test_experiment_registered():
    m = o.BY_ID["experiment"]; assert m["group"] == "Causal inference" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_experiment_difference_in_means_matches_published_nsw():
    # Dehejia and Wahba (1999) experimental sample: 1.794 (0.671) in thousands of 1978 dollars
    r = causal_extra.experiment(nsw(), ES); row = erow(r, "Difference")
    assert float(row[1]) == pytest.approx(1.794, abs=1e-3) and float(row[2]) == pytest.approx(0.671, abs=1e-3)


def test_experiment_regression_estimators_match_statsmodels():
    import statsmodels.api as sm
    d = nsw(); r = causal_extra.experiment(d, ES); Xc = d[COV] - d[COV].mean()
    a = sm.OLS(d.re78, sm.add_constant(pd.concat([d.train, Xc], axis=1))).fit(cov_type="HC2")
    assert float(erow(r, "Regression adjustment")[1]) == pytest.approx(a.params["train"], abs=1e-3) and float(erow(r, "Regression adjustment")[2]) == pytest.approx(a.bse["train"], abs=1e-3)
    inter = pd.concat([d.train, Xc, Xc.mul(d.train, axis=0).add_suffix("_x")], axis=1)
    l = sm.OLS(d.re78, sm.add_constant(inter)).fit(cov_type="HC2")
    assert float(erow(r, "Lin")[1]) == pytest.approx(l.params["train"], abs=1e-3) and float(erow(r, "Lin")[2]) == pytest.approx(l.bse["train"], abs=1e-3)


def test_experiment_randomisation_inference_matches_scipy_permutation():
    d = nsw(); r = causal_extra.experiment(d, dict(outcome="re78", treat="train", draws=4000))
    res = stats.permutation_test((d.re78[d.train == 1].to_numpy(), d.re78[d.train == 0].to_numpy()), lambda a, b: a.mean() - b.mean(), n_resamples=4000, random_state=1)
    assert float(r["tables"][1]["rows"][0][2]) == pytest.approx(res.pvalue, abs=0.012)


def test_experiment_balance_joint_test_matches_statsmodels():
    import statsmodels.api as sm
    d = nsw(); r = causal_extra.experiment(d, ES); m = sm.OLS(d.train, sm.add_constant(d[COV])).fit()
    note = next(t for t in r["tables"] if t["title"] == "Covariate balance")["note"]; F = float(re.search(r"F = ([\d.]+)", note).group(1))
    assert F == pytest.approx(m.fvalue, abs=0.01)


def exp_sim(seed, n=200, tau=0.0, prog=2.0):
    rng = np.random.default_rng(seed); x = rng.normal(size=n); T = np.zeros(n); T[rng.permutation(n)[: n // 2]] = 1
    y = prog * x + tau * T + rng.normal(size=n); return pd.DataFrame(dict(y=y, t=T, x=x))


def test_experiment_neyman_coverage_ri_size_and_lin_precision():
    cov = rej = 0; se_d = se_l = 0.0; R = 300
    for s in range(R):
        r = causal_extra.experiment(exp_sim(s, tau=1.0), dict(outcome="y", treat="t", covariates=["x"], draws=200))
        e, se = float(erow(r, "Difference")[1]), float(erow(r, "Difference")[2]); cov += abs(e - 1.0) <= 1.96 * se; se_d += se; se_l += float(erow(r, "Lin")[2])
        rej += float(causal_extra.experiment(exp_sim(s, tau=0.0), dict(outcome="y", treat="t", draws=200))["tables"][1]["rows"][0][2]) < 0.05
    assert 0.92 <= cov / R <= 0.985 and 0.02 <= rej / R <= 0.09 and se_l < 0.6 * se_d


def test_experiment_errors():
    d = nsw()
    assert "0/1" in o.run("experiment", d.assign(train=d.train * 2).to_csv(index=False), dict(ES))["error"]
    assert "Column(s) not found" in o.run("experiment", d.to_csv(index=False), dict(ES, covariates=["nope"]))["error"]
    few = d.iloc[:60].assign(train=[1.0] * 3 + [0.0] * 57)
    assert "at least 5" in o.run("experiment", few.to_csv(index=False), dict(outcome="re78", treat="train"))["error"]


# ---- causal mediation -------------------------------------------------------------------------------------------
def nswm(): return pd.read_csv(os.path.join(CX, "nsw_mediation.csv"))
MS = dict(outcome="re78", treat="train", mediator="employed78", covariates=COV, boot=99)
def mrow(r, name): return next(x for x in r["tables"][0]["rows"] if x[0].startswith(name))


def test_mediation_registered():
    m = o.BY_ID["mediation"]; assert m["group"] == "Causal inference" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_mediation_product_of_coefficients_and_total_effect_identity():
    import statsmodels.api as sm
    d = nswm(); r = causal_extra.mediation(d, MS); X = d[COV]
    a = sm.OLS(d.employed78, sm.add_constant(pd.concat([d.train, X], axis=1))).fit().params["train"]
    yfit = sm.OLS(d.re78, sm.add_constant(pd.concat([d.train, d.employed78, X], axis=1))).fit(); b, c = yfit.params["employed78"], yfit.params["train"]
    tot = sm.OLS(d.re78, sm.add_constant(pd.concat([d.train, X], axis=1))).fit().params["train"]
    assert float(mrow(r, "ACME")[1]) == pytest.approx(a * b, abs=1e-3) and float(mrow(r, "ADE")[1]) == pytest.approx(c, abs=1e-3)
    assert float(mrow(r, "Total")[1]) == pytest.approx(tot, abs=1e-3)


def med_sim(seed, n=800, rho=0.0, a=0.6, b=1.0, c=0.5, kap=0.0):
    rng = np.random.default_rng(seed); T = (rng.uniform(size=n) < 0.5).astype(float); e1 = rng.normal(size=n); e2 = rho * e1 + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    M = a * T + e1; Y = c * T + b * M + kap * T * M + e2; return pd.DataFrame(dict(y=Y, t=T, m=M))
SM = dict(outcome="y", treat="t", mediator="m", boot=99)


def test_mediation_recovers_effects_and_covers_truth_under_ignorability():
    cov = 0; R = 120
    for s in range(R):
        r = causal_extra.mediation(med_sim(s, n=400), SM); row = mrow(r, "ACME")
        cov += float(row[3]) <= 0.6 * 1.0 <= float(row[4])
    assert 0.88 <= cov / R <= 0.995
    r = causal_extra.mediation(med_sim(1, n=4000), SM); assert float(mrow(r, "ACME")[1]) == pytest.approx(0.6, abs=0.06) and float(mrow(r, "ADE")[1]) == pytest.approx(0.5, abs=0.06)


def test_mediation_interaction_recovers_effects_by_treatment_arm():
    r = causal_extra.mediation(med_sim(2, n=6000, kap=0.5), dict(SM, interaction="yes"))
    assert float(mrow(r, "ACME, control")[1]) == pytest.approx(0.6 * 1.0, abs=0.07) and float(mrow(r, "ACME, treated")[1]) == pytest.approx(0.6 * 1.5, abs=0.09)


def test_mediation_sensitivity_curve_passes_through_truth_and_zero_at_rho_star():
    rho0 = 0.5; r = causal_extra.mediation(med_sim(3, n=20000, rho=rho0), SM)
    srows = next(t for t in r["tables"] if t["title"].startswith("Sensitivity"))["rows"]; at = {float(x[0]): float(x[1]) for x in srows}
    assert float(mrow(r, "ACME")[1]) > 0.6 + 0.2                       # naive estimate is biased upward by the confounding
    assert at[0.5] == pytest.approx(0.6, abs=0.05)                      # the curve at the true rho gives the true ACME
    rstar = next(x["value"] for x in r["summary"] if x["label"].startswith("rho at which")); sign = np.sign(at[0.9]); assert np.sign(at[-0.9]) != sign and 0 < abs(rstar) < 0.9


def test_mediation_errors():
    d = nswm()
    assert "0/1" in o.run("mediation", d.assign(train=d.train * 2).to_csv(index=False), MS)["error"]
    assert "does not vary" in o.run("mediation", d.assign(employed78=1.0).to_csv(index=False), MS)["error"]
    assert "Column(s) not found" in o.run("mediation", d.to_csv(index=False), dict(MS, mediator="nope"))["error"]


# ------------------------------------------------------------------ Lee bounds
def _lee_sim(n=4000, effect=1.0, sel_shift=0.8, seed=0):
    rng = np.random.default_rng(seed); T = rng.integers(0, 2, n).astype(float); u = rng.normal(size=n); v = rng.normal(size=n)
    S = (0.2 + sel_shift * T * 0 + v + sel_shift * T > 0).astype(float)
    Y = effect * T + 0.7 * v + u; return pd.DataFrame(dict(y=np.where(S == 1, Y, np.nan), t=T, s=S))


def _lee_run(d, **kw): return causal_extra.leebounds(d, dict(outcome="y", treat="t", selected="s", boot=150, **kw))


def test_lee_registered():
    m = o.BY_ID["leebounds"]; assert m["group"] == "Causal inference" and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_lee_no_selection_difference_is_point_identified():
    rng = np.random.default_rng(1); n = 3000; T = rng.integers(0, 2, n).astype(float); S = (rng.random(n) < 0.7).astype(float)
    d = pd.DataFrame(dict(y=np.where(S == 1, 2 * T + rng.normal(size=n), np.nan), t=T, s=S)); r = _lee_run(d)
    s = {x["label"]: x["value"] for x in r["summary"]}
    # selection rates differ only by chance, so trimming is tiny and the bounds hug the plain difference
    assert s["Share trimmed"] < 0.05 and s["Lee upper bound"] - s["Lee lower bound"] < 0.25 and abs(s["Naive effect"] - 2) < 0.15


def test_lee_bounds_cover_true_effect_for_always_selected():
    # monotone selection: always-selected effect is exactly 'effect' for the stayers
    d = _lee_sim(n=20000, effect=1.0); r = _lee_run(d); s = {x["label"]: x["value"] for x in r["summary"]}
    assert s["Lee lower bound"] <= 1.0 + 0.1 and s["Lee upper bound"] >= 1.0 - 0.1 and s["Lee lower bound"] <= s["Lee upper bound"]


def test_lee_matches_hand_trimming():
    d = _lee_sim(n=2000, seed=3); Y = d.y.to_numpy(); T = d.t.to_numpy(); S = d.s.to_numpy()
    q1, q0 = S[T == 1].mean(), S[T == 0].mean(); p = (q1 - q0) / q1; y1 = np.sort(Y[(T == 1) & (S == 1)]); y0 = Y[(T == 0) & (S == 1)]
    k = int(round(p * len(y1))); lo = y1[:len(y1) - k].mean() - y0.mean(); hi = y1[k:].mean() - y0.mean()
    s = {x["label"]: x["value"] for x in _lee_run(d)["summary"]}
    assert abs(s["Lee lower bound"] - lo) < 0.01 and abs(s["Lee upper bound"] - hi) < 0.01


def test_lee_control_arm_trimmed_when_control_selected_more():
    d = _lee_sim(n=3000, seed=4); d["t"] = 1 - d["t"]; r = _lee_run(d)
    assert any("control" in row[0] for row in r["tables"][1]["rows"] if row[0].startswith("Share trimmed"))


def test_lee_narrower_than_manski():
    r = _lee_run(_lee_sim(n=3000, seed=5))
    lee = [float(v) for v in re.findall(r"-?\d+\.\d+", next(x[1] for x in r["tables"][1]["rows"] if x[0].startswith("Lee bounds:")))]
    man = [float(v) for v in re.findall(r"-?\d+\.\d+", next(x[1] for x in r["tables"][1]["rows"] if x[0].startswith("Manski")))]
    assert lee[1] - lee[0] < man[1] - man[0]


def test_lee_errors():
    d = _lee_sim(n=500)
    for spec, msg in [(dict(outcome="y", treat="s", selected="s"), "more than one role"), (dict(outcome="y", treat="y", selected="s"), "more than one role"), (dict(outcome="nope", treat="t", selected="s"), "not found")]:
        with pytest.raises(causal_extra.OESError, match=msg): causal_extra.leebounds(d, spec)
    bad = d.copy(); bad["t"] = bad["t"] * 2
    with pytest.raises(causal_extra.OESError, match="0/1"): causal_extra.leebounds(bad, dict(outcome="y", treat="t", selected="s"))
    bad = d.copy(); bad.loc[bad.s == 1, "y"] = np.nan
    with pytest.raises(causal_extra.OESError, match="missing"): causal_extra.leebounds(bad, dict(outcome="y", treat="t", selected="s"))
