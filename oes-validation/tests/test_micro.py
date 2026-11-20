"""Validation of the OES microeconometrics models.
Run:  pip install numpy pandas scipy statsmodels scikit-learn pytest  &&  pytest oes-validation/tests -q
Each model is checked against (1) published results, (2) an independent implementation, (3) recovery of known parameters
from simulated data and (4) the error messages for invalid input."""
import json, os, sys, warnings
import numpy as np, pandas as pd, pytest
import statsmodels.api as sm
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "oes", "engine")); sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import oes_engine as o

ME = os.path.join(ROOT, "microeconometrics-guide", "examples")
EXPECTED = json.load(open(os.path.join(ROOT, "oes", "engine", "expected.json")))


def table(r, prefix): return next(t for t in r["tables"] if t["title"].startswith(prefix))
def row(r, tbl, name): return next(x for x in table(r, tbl)["rows"] if x[0] == name)
def summ(r, label): return next(s["value"] for s in r["summary"] if s["label"] == label)
def spector(**kw):
    spec = dict(y="GRADE", x=["GPA", "TUCE", "PSI"]); spec.update(kw)
    return o.run("logit_probit", open(os.path.join(ME, "spector.csv")).read(), spec)


# ------------------------------------------------------------------ logit and probit
def test_registered_with_guide_and_example():
    m = o.BY_ID["logit_probit"]
    assert m["group"] == "Microeconometrics" and os.path.exists(os.path.join(ROOT, m["guide"]))
    assert os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_logit_matches_greene_published_results():
    # Greene, Econometric Analysis, Spector and Mazzeo (1980) example: logit coefficients, standard errors, log-likelihood
    r = spector()
    for name, b, se in [("const", -13.0213, 4.9313), ("GPA", 2.8261, 1.2629), ("TUCE", 0.0952, 0.1416), ("PSI", 2.3787, 1.0646)]:
        x = row(r, "Coefficient", name)
        assert x[1] == pytest.approx(b, abs=5e-4) and x[2] == pytest.approx(se, abs=5e-4)
    assert summ(r, "Log-likelihood") == pytest.approx(-12.8896, abs=5e-4)
    assert summ(r, "McFadden pseudo R-squared") == pytest.approx(0.3740, abs=5e-4)


def test_probit_matches_greene_published_results():
    r = spector(link="probit")
    for name, b in [("const", -7.4523), ("GPA", 1.6258), ("TUCE", 0.0517), ("PSI", 1.4263)]:
        assert row(r, "Coefficient", name)[1] == pytest.approx(b, abs=5e-4)
    assert summ(r, "Log-likelihood") == pytest.approx(-12.8188, abs=5e-4)


def test_average_marginal_effects_match_published_values():
    # statsmodels documentation / Stata margins, dydx(*), PSI as a discrete change
    r = spector()
    for name, v in [("GPA", 0.3626), ("TUCE", 0.0122), ("PSI", 0.3575)]:
        assert row(r, "Average marginal effects", name)[1] == pytest.approx(v, abs=5e-4)
    assert row(r, "Average marginal effects", "PSI")[7].startswith("discrete")


def test_odds_ratios_are_exponentiated_coefficients_and_logit_only():
    r = spector()
    for name in ("GPA", "TUCE", "PSI"):
        assert row(r, "Odds ratios", name)[1] == pytest.approx(np.exp(row(r, "Coefficient", name)[1]), rel=1e-5)
    assert not any(t["title"] == "Odds ratios" for t in spector(link="probit")["tables"])


def test_auc_equals_scikit_learn_and_classification_counts_add_up():
    from sklearn.metrics import roc_auc_score
    d = pd.read_csv(os.path.join(ME, "spector.csv"))
    fit = sm.Logit(d.GRADE, sm.add_constant(d[["GPA", "TUCE", "PSI"]])).fit(disp=0)
    r = spector()
    assert summ(r, "Area under the ROC curve") == pytest.approx(roc_auc_score(d.GRADE, fit.predict()), abs=5e-4)
    cl = table(r, "Classification")["rows"]
    assert sum(x[1] + x[2] for x in cl) == 32 and cl[1][3] == 11


def test_roc_chart_ends_at_one_and_lies_above_chance():
    c = spector()["charts"][0]
    model, chance = c["series"][0]["y"], c["series"][1]["y"]
    assert model[0] >= 0 and model[-1] == 1 and all(m >= q - 1e-9 for m, q in zip(model, chance))


def test_hosmer_lemeshow_and_link_test_computed_independently():
    d = pd.read_csv(os.path.join(ME, "spector.csv"))
    X = sm.add_constant(d[["GPA", "TUCE", "PSI"]]); fit = sm.Logit(d.GRADE, X).fit(disp=0); p = fit.predict()
    grp = pd.qcut(pd.Series(p).rank(method="first"), 6, labels=False)
    chi = sum((d.GRADE[grp == g].sum() - (grp == g).sum() * p[grp == g].mean()) ** 2 /
              ((grp == g).sum() * p[grp == g].mean() * (1 - p[grp == g].mean())) for g in range(6))
    t = next(x for x in table(spector(), "Specification")["rows"] if x[0].startswith("Hosmer"))
    assert t[1] == pytest.approx(chi, abs=5e-4) and t[2] == pytest.approx(stats.chi2.sf(chi, 4), abs=5e-4)


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_recovers_known_parameters_from_simulated_data(link):
    rng = np.random.default_rng(7); n = 40000
    x1, x2 = rng.normal(size=n), rng.binomial(1, .4, n)
    idx = -0.5 + 1.0 * x1 + 0.8 * x2
    py = 1 / (1 + np.exp(-idx)) if link == "logit" else stats.norm.cdf(idx)
    df = pd.DataFrame({"y": rng.binomial(1, py), "x1": x1, "x2": x2})
    r = o.run("logit_probit", df.to_csv(index=False), dict(y="y", x=["x1", "x2"], link=link))
    for name, b in [("const", -0.5), ("x1", 1.0), ("x2", 0.8)]:
        x = row(r, "Coefficient", name)
        assert abs(x[1] - b) < 4 * x[2]            # within four standard errors of the truth
    # the AME of x1 equals the sample mean of the true marginal effect
    true_me = (np.exp(idx) / (1 + np.exp(idx)) ** 2 if link == "logit" else stats.norm.pdf(idx)).mean() * 1.0
    assert row(r, "Average marginal effects", "x1")[1] == pytest.approx(true_me, abs=0.01)


def test_robust_and_clustered_errors_change_only_the_standard_errors():
    base, rob = spector(), spector(cov="robust")
    d = pd.read_csv(os.path.join(ME, "spector.csv")); d["g"] = np.arange(len(d)) // 4
    clu = o.run("logit_probit", d.to_csv(index=False), dict(y="GRADE", x=["GPA", "TUCE", "PSI"], cov="cluster", cluster="g"))
    for name in ("GPA", "TUCE", "PSI"):
        assert row(base, "Coefficient", name)[1] == row(rob, "Coefficient", name)[1] == row(clu, "Coefficient", name)[1]
    assert row(base, "Coefficient", "TUCE")[2] != row(rob, "Coefficient", "TUCE")[2]


def test_cutoff_changes_classification_not_the_model():
    lo, hi = spector(cutoff=0.2), spector(cutoff=0.8)
    assert summ(lo, "Log-likelihood") == summ(hi, "Log-likelihood")
    assert table(lo, "Classification")["rows"][1][2] >= table(hi, "Classification")["rows"][1][2]


def test_example_reproduces_the_stored_reference():
    e = EXPECTED["logit_probit"]; m = o.BY_ID["logit_probit"]
    r = o.run("logit_probit", open(os.path.join(ROOT, m["example"]["file"])).read(), dict(m["example"]["spec"]))
    assert r["tables"][e["table_index"]]["rows"][e["row_index"]][e["col_index"]] == pytest.approx(e["value"], rel=e["rtol"])


def test_errors_are_reported_not_raised():
    d = pd.read_csv(os.path.join(ME, "spector.csv"))
    run = lambda df, **kw: o.run("logit_probit", df.to_csv(index=False), dict(y="GRADE", x=["GPA", "TUCE", "PSI"], **kw))
    assert "coded 0 and 1" in run(d.assign(GRADE=d.GRADE * 2))["error"]
    assert "only one value" in run(d.assign(GRADE=1))["error"]
    assert "cluster variable" in run(d, cov="cluster")["error"]
    assert "cut-off" in run(d, cutoff=1.5)["error"]
    sep = d.assign(SEP=d.GRADE.astype(float))
    r = o.run("logit_probit", sep.to_csv(index=False), dict(y="GRADE", x=["GPA", "SEP"]))
    assert "error" in r and "separation" in r["error"]
    assert "at least one" in o.run("logit_probit", d.to_csv(index=False), dict(y="GRADE", x=[]))["error"]
    assert "too few" in o.run("logit_probit", d.head(8).to_csv(index=False), dict(y="GRADE", x=["GPA", "TUCE", "PSI"]))["error"]


# ------------------------------------------------------------------ ordered logit and probit
HX = ["InflMedium", "InflHigh", "TypeApartment", "TypeAtrium", "TypeTerrace", "ContHigh"]
def housing(**kw):
    spec = dict(y="Sat", x=HX); spec.update(kw)
    return o.run("ordered", open(os.path.join(ME, "housing.csv")).read(), spec)


def test_ordered_matches_published_MASS_housing_results():
    # Venables and Ripley, MASS, polr(Sat ~ Infl + Type + Cont, weights = Freq): coefficients, standard errors, cut-points, deviance
    r = housing()
    pub = dict(InflMedium=(0.5664, 0.1047), InflHigh=(1.2888, 0.1272), TypeApartment=(-0.5724, 0.1192),
               TypeAtrium=(-0.3662, 0.1552), TypeTerrace=(-1.0910, 0.1515), ContHigh=(0.3603, 0.0955))
    for name, (b, se) in pub.items():
        x = row(r, "Coefficient", name)
        assert x[1] == pytest.approx(b, abs=5e-4) and x[2] == pytest.approx(se, abs=5e-4)
    cp = table(r, "Cut-points")["rows"]
    assert cp[0][1] == pytest.approx(-0.4961, abs=5e-4) and cp[1][1] == pytest.approx(0.6907, abs=5e-4)
    assert -2 * summ(r, "Log-likelihood") == pytest.approx(3479.149, abs=5e-3)


def test_ordered_logit_equals_an_independent_likelihood_maximisation():
    from scipy.optimize import minimize
    d = pd.read_csv(os.path.join(ME, "housing.csv")); X = d[HX].to_numpy(float); yy = d.Sat.to_numpy() - 1
    def nll(t):
        b, c1, g = t[:6], t[6], np.exp(t[7]); cuts = [c1, c1 + g]; xb = X @ b
        F = lambda z: 1 / (1 + np.exp(-z))
        cum = np.column_stack([np.zeros(len(xb)), F(cuts[0] - xb), F(cuts[1] - xb), np.ones(len(xb))])
        return -np.log(np.diff(cum, axis=1)[np.arange(len(yy)), yy]).sum()
    best = minimize(nll, np.zeros(8), method="BFGS", options={"gtol": 1e-8})
    r = housing()
    assert -best.fun == pytest.approx(summ(r, "Log-likelihood"), abs=1e-3)
    assert best.x[:6] == pytest.approx([row(r, "Coefficient", c)[1] for c in HX], abs=2e-3)


def test_ordered_marginal_effects_sum_to_zero_across_categories_and_match_by_hand():
    r = housing()
    for c in HX:
        assert sum(x[2] for x in table(r, "Average marginal effects")["rows"] if x[0] == c) == pytest.approx(0, abs=1e-5)
    d = pd.read_csv(os.path.join(ME, "housing.csv")); X = d[HX].to_numpy(float)
    b = np.array([row(r, "Coefficient", c)[1] for c in HX]); cp = [x[1] for x in table(r, "Cut-points")["rows"]]
    F = lambda z: 1 / (1 + np.exp(-z))
    def p3(M): xb = M @ b; return 1 - F(cp[1] - xb)
    X1, X0 = X.copy(), X.copy(); X1[:, 1], X0[:, 1] = 1, 0
    by_hand = (p3(X1) - p3(X0)).mean()
    got = next(x for x in table(r, "Average marginal effects")["rows"] if x[0] == "InflHigh" and x[1] == "y = 3.0")[2]
    assert got == pytest.approx(by_hand, abs=2e-4)


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_ordered_recovers_known_parameters_from_simulated_data(link):
    rng = np.random.default_rng(11); n = 30000
    x1, x2 = rng.normal(size=n), rng.binomial(1, .5, n); xb = 0.9 * x1 - 0.6 * x2
    e = rng.logistic(size=n) if link == "logit" else rng.normal(size=n)
    ystar = xb + e; y = 1 + (ystar > -0.7) + (ystar > 0.4) + (ystar > 1.5)
    r = o.run("ordered", pd.DataFrame({"y": y, "x1": x1, "x2": x2}).to_csv(index=False), dict(y="y", x=["x1", "x2"], link=link))
    for name, b in [("x1", 0.9), ("x2", -0.6)]:
        x = row(r, "Coefficient", name); assert abs(x[1] - b) < 4 * x[2]
    cp = [x[1] for x in table(r, "Cut-points")["rows"]]
    assert cp == pytest.approx([-0.7, 0.4, 1.5], abs=0.08)


def test_brant_test_accepts_parallel_lines_and_rejects_non_parallel_data():
    rng = np.random.default_rng(5); n = 6000; x = rng.normal(size=n)
    ystar = x + rng.logistic(size=n); good = 1 + (ystar > -.5) + (ystar > .8)
    u = rng.logistic(size=n)
    bad = 1 + (1.8 * x + u > -.5) + (0.2 * x + u > .8)                                  # the slope differs between the two cut-points
    pv = lambda y: next(t for t in o.run("ordered", pd.DataFrame({"y": y, "x": x}).to_csv(index=False), dict(y="y", x=["x"]))["tables"] if t["title"].startswith("Specification"))["rows"][1][2]
    assert pv(good) > 0.01 and pv(bad) < 0.001


def test_ordered_errors_and_example_reference():
    d = pd.read_csv(os.path.join(ME, "housing.csv"))
    run = lambda df, **kw: o.run("ordered", df.to_csv(index=False), dict(y="Sat", x=["InflHigh", "ContHigh"], **kw))
    assert "binary" in run(d.assign(Sat=(d.Sat > 1) * 1))["error"]
    assert "3 to 10" in run(d.assign(Sat=np.arange(len(d)) % 15))["error"]
    assert "at least 5" in run(d.assign(Sat=np.where(np.arange(len(d)) < 3, 4, d.Sat)))["error"]
    assert "collinear" in run(d.assign(InflHigh=1))["error"]
    e = EXPECTED["ordered"]; m = o.BY_ID["ordered"]
    r = o.run("ordered", open(os.path.join(ROOT, m["example"]["file"])).read(), dict(m["example"]["spec"]))
    assert r["tables"][e["table_index"]]["rows"][e["row_index"]][e["col_index"]] == pytest.approx(e["value"], rel=e["rtol"])
