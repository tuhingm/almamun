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


# ------------------------------------------------------------------ multinomial logit
BX = ["age", "econ_national", "econ_household", "europe", "political_knowledge", "female"]
def beps(**kw):
    spec = dict(y="vote", x=BX, base=0); spec.update(kw)
    return o.run("multinomial", open(os.path.join(ME, "beps.csv")).read(), spec)


def test_multinomial_equals_an_independent_likelihood_maximisation():
    from scipy.optimize import minimize
    d = pd.read_csv(os.path.join(ME, "beps.csv")); X = sm.add_constant(d[BX]).to_numpy(float); yy = d.vote.to_numpy().astype(int)
    def nll(t):
        B = t.reshape(7, 2); z = np.column_stack([np.zeros(len(X)), X @ B]); z -= z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True)); return -lp[np.arange(len(yy)), yy].sum()
    best = minimize(nll, np.zeros(14), method="BFGS", options={"gtol": 1e-7, "maxiter": 2000})
    r = beps()
    assert -best.fun == pytest.approx(summ(r, "Log-likelihood"), abs=1e-3)
    assert best.x.reshape(7, 2)[:, 0] == pytest.approx([x[1] for x in table(r, "Equation: vote = 1.0")["rows"]], abs=5e-3)


def test_multinomial_changing_the_base_changes_coefficients_but_not_probabilities():
    a, b = beps(base=0), beps(base=2)
    assert summ(a, "Log-likelihood") == pytest.approx(summ(b, "Log-likelihood"), abs=1e-4)
    ame = lambda r, v, c: next(x for x in table(r, "Average marginal effects")["rows"] if x[0] == v and x[1] == f"vote = {c}.0")[2]
    for c in (0, 1, 2):
        assert ame(a, "econ_national", c) == pytest.approx(ame(b, "econ_national", c), abs=1e-4)
    assert row(a, "Equation: vote = 1.0", "age")[1] != row(b, "Equation: vote = 1.0", "age")[1]


def test_multinomial_marginal_effects_sum_to_zero_and_match_by_hand():
    r = beps()
    for v in BX:
        assert sum(x[2] for x in table(r, "Average marginal effects")["rows"] if x[0] == v) == pytest.approx(0, abs=1e-5)
    d = pd.read_csv(os.path.join(ME, "beps.csv")); X = sm.add_constant(d[BX]).to_numpy(float)
    B = np.column_stack([[x[1] for x in table(r, f"Equation: vote = {c}.0")["rows"]] for c in (1, 2)])
    def P(M): z = np.column_stack([np.zeros(len(M)), M @ B]); e = np.exp(z); return e / e.sum(1, keepdims=True)
    X1, X0 = X.copy(), X.copy(); X1[:, 6], X0[:, 6] = 1, 0
    by_hand = (P(X1) - P(X0)).mean(0)
    got = [next(x for x in table(r, "Average marginal effects")["rows"] if x[0] == "female" and x[1] == f"vote = {c}.0")[2] for c in (0, 1, 2)]
    assert got == pytest.approx(list(by_hand), abs=2e-4)


def test_multinomial_recovers_known_parameters_from_simulated_data():
    rng = np.random.default_rng(3); n = 40000; x1, x2 = rng.normal(size=n), rng.binomial(1, .5, n)
    B = np.array([[0.2, -0.4], [0.8, -0.5], [-0.6, 0.7]])           # rows: const, x1, x2; columns: categories 1 and 2
    z = np.column_stack([np.zeros(n), np.column_stack([np.ones(n), x1, x2]) @ B]); pr = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    yy = np.array([rng.choice(3, p=p) for p in pr])
    r = o.run("multinomial", pd.DataFrame({"y": yy, "x1": x1, "x2": x2}).to_csv(index=False), dict(y="y", x=["x1", "x2"], base=0))
    for c, col_ in ((1, 0), (2, 1)):
        for i, name in enumerate(["const", "x1", "x2"]):
            x = row(r, f"Equation: y = {c}.0", name); assert abs(x[1] - B[i, col_]) < 4 * x[2]


def test_multinomial_errors_and_example_reference():
    d = pd.read_csv(os.path.join(ME, "beps.csv"))
    run = lambda df, **kw: o.run("multinomial", df.to_csv(index=False), dict(y="vote", x=["age", "europe"], **kw))
    assert "binary" in run(d.assign(vote=(d.vote > 0) * 1))["error"]
    assert "not a value" in run(d, base=7)["error"]
    assert "at least 10" in run(d.assign(vote=np.where(np.arange(len(d)) < 4, 3, d.vote)))["error"]
    e = EXPECTED["multinomial"]; m = o.BY_ID["multinomial"]
    r = o.run("multinomial", open(os.path.join(ROOT, m["example"]["file"])).read(), dict(m["example"]["spec"]))
    assert r["tables"][e["table_index"]]["rows"][e["row_index"]][e["col_index"]] == pytest.approx(e["value"], rel=e["rtol"])


# ------------------------------------------------------------------ Tobit
TX = ["age", "yearsmarried", "religiousness", "occupation", "rating"]
def affairs(**kw):
    spec = dict(y="affairs", x=TX, lower=0); spec.update(kw)
    return o.run("tobit", open(os.path.join(ME, "affairs.csv")).read(), spec)


def test_tobit_matches_published_AER_affairs_results():
    # Kleiber and Zeileis, Applied Econometrics with R: tobit(affairs ~ age + yearsmarried + religiousness + occupation + rating)
    r = affairs()
    pub = dict(const=(8.1742, 2.7414), age=(-0.1793, 0.0791), yearsmarried=(0.5541, 0.1345), religiousness=(-1.6862, 0.4038), occupation=(0.3261, 0.2544), rating=(-2.2850, 0.4078))
    for name, (b, se) in pub.items():
        x = row(r, "Coefficient", name); assert x[1] == pytest.approx(b, abs=5e-4) and x[2] == pytest.approx(se, abs=1e-3)
    assert row(r, "Coefficient", "sigma")[1] == pytest.approx(np.exp(2.1099), abs=2e-3)
    assert summ(r, "Log-likelihood") == pytest.approx(-705.576, abs=5e-3)


def test_tobit_equals_independent_likelihood_and_mcdonald_moffitt_identity():
    from scipy.optimize import minimize
    d = pd.read_csv(os.path.join(ME, "affairs.csv")); X = np.column_stack([np.ones(len(d)), d[TX].to_numpy(float)]); y = d.affairs.to_numpy(float)
    def nll(t):
        xb, s = X @ t[:-1], np.exp(t[-1]); c = y <= 0
        return -(stats.norm.logcdf(-xb[c] / s).sum() + (stats.norm.logpdf((y[~c] - xb[~c]) / s) - t[-1]).sum())
    best = minimize(nll, np.r_[np.linalg.lstsq(X, y, rcond=None)[0], 2.0], method="BFGS", options={"gtol": 1e-8})
    r = affairs()
    assert -best.fun == pytest.approx(summ(r, "Log-likelihood"), abs=1e-3)
    # observed-outcome marginal effect = latent coefficient x average probability of being uncensored (left censoring only)
    b, s = best.x[:-1], np.exp(best.x[-1]); pu = stats.norm.cdf(X @ b / s).mean()
    obs = next(x for x in table(r, "Average marginal effects")["rows"] if x[0] == "rating" and x[1].startswith("Observed"))[2]
    assert obs == pytest.approx(b[5] * pu, abs=2e-3)


def test_tobit_coefficients_are_larger_than_ols_on_censored_data():
    for x in table(affairs(), "Comparison with OLS")["rows"]:
        assert abs(x[1]) > abs(x[2]) and x[3] > 2


@pytest.mark.parametrize("lower,upper", [(0.0, None), (0.0, 6.0)])
def test_tobit_recovers_known_parameters_from_simulated_data(lower, upper):
    rng = np.random.default_rng(2); n = 8000; x1, x2 = rng.normal(size=n), rng.binomial(1, .5, n)
    ystar = 1.0 + 2.0 * x1 - 1.0 * x2 + rng.normal(scale=2.0, size=n); y = np.maximum(ystar, lower)
    if upper is not None: y = np.minimum(y, upper)
    r = o.run("tobit", pd.DataFrame({"y": y, "x1": x1, "x2": x2}).to_csv(index=False), dict(y="y", x=["x1", "x2"], lower=lower, upper=upper if upper is not None else ""))
    for name, b in [("const", 1.0), ("x1", 2.0), ("x2", -1.0), ("sigma", 2.0)]:
        x = row(r, "Coefficient", name); assert abs(x[1] - b) < 4 * x[2]


def test_tobit_errors_and_example_reference():
    d = pd.read_csv(os.path.join(ME, "affairs.csv"))
    run = lambda df, **kw: o.run("tobit", df.to_csv(index=False), dict(y="affairs", x=["age", "rating"], **kw))
    assert "needs censoring" in run(d, lower="", upper="")["error"]
    assert "not censored" in run(d.assign(affairs=d.affairs + 1), lower=0)["error"]
    assert "below the lower limit" in run(d.assign(affairs=d.affairs - 1), lower=0)["error"]
    assert "must be below" in run(d, lower=5, upper=2)["error"]
    e = EXPECTED["tobit"]; m = o.BY_ID["tobit"]
    r = o.run("tobit", open(os.path.join(ROOT, m["example"]["file"])).read(), dict(m["example"]["spec"]))
    assert r["tables"][e["table_index"]]["rows"][e["row_index"]][e["col_index"]] == pytest.approx(e["value"], rel=e["rtol"])


# ------------------------------------------------------------------ Heckman selection
def mroz(**kw):
    spec = dict(o.BY_ID["heckman"]["example"]["spec"]); spec.update(kw)
    return o.run("heckman", open(os.path.join(ME, "mroz.csv")).read(), spec)


def test_heckman_two_step_matches_published_mroz():
    # Wooldridge/Stata two-step on the Mroz (1987) data: educ .1091, exper .0439, lambda .0323
    t = table(mroz(), "Two-step")
    assert next(x for x in t["rows"] if x[0] == "educ")[1] == pytest.approx(0.109066, abs=5e-4)
    assert next(x for x in t["rows"] if x[0] == "exper")[1] == pytest.approx(0.043887, abs=5e-4)
    assert next(x for x in t["rows"] if x[0].startswith("lambda"))[1] == pytest.approx(0.032261, abs=5e-4)


def test_heckman_ml_matches_independent_likelihood():
    from scipy import optimize
    d = pd.read_csv(os.path.join(ME, "mroz.csv")); s = d.inlf.values == 1
    Z = sm.add_constant(d[["educ", "exper", "expersq", "age", "kidslt6", "kidsge6", "nwifeinc"]]).values
    X = sm.add_constant(d[["educ", "exper", "expersq"]]).values; y = d.lwage.fillna(0).values

    def nll(t):
        g, b, sg, rh = t[:8], t[8:12], np.exp(t[12]), np.tanh(t[13]); zg = Z @ g; u = (y[s] - X[s] @ b) / sg
        return -(stats.norm.logcdf(-zg[~s]).sum() + (stats.norm.logpdf(u) - np.log(sg) + stats.norm.logcdf((zg[s] + rh * u) / np.sqrt(1 - rh ** 2))).sum())
    r0 = optimize.minimize(nll, np.r_[np.zeros(8), np.zeros(4), 0, 0], method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10})
    r1 = optimize.minimize(nll, r0.x, method="BFGS")
    assert summ(mroz(), "Log-likelihood") == pytest.approx(-r1.fun, abs=1e-3)


def test_heckman_recovers_selection_bias_in_simulation():
    rng = np.random.default_rng(3); n = 6000
    x, w = rng.normal(size=n), rng.normal(size=n)
    e = rng.multivariate_normal([0, 0], [[1, .7], [.7, 1]], n)
    sel = (0.3 + 0.8 * x + 1.0 * w + e[:, 0] > 0).astype(int)
    y = 1.0 + 2.0 * x + e[:, 1]
    d = pd.DataFrame(dict(y=np.where(sel == 1, y, np.nan), x=x, w=w, s=sel))
    r = o.run("heckman", d.to_csv(index=False), dict(y="y", x=["x"], sel="s", z=["x", "w"]))
    assert row(r, "Outcome equation", "x")[1] == pytest.approx(2.0, abs=0.1)
    assert row(r, "Outcome equation", "rho")[1] == pytest.approx(0.7, abs=0.1)
    ols = sm.OLS(y[sel == 1], sm.add_constant(x[sel == 1])).fit().params[0]
    assert abs(ols - 1.0) > abs(row(r, "Outcome equation", "const")[1] - 1.0)       # OLS intercept is biased, Heckman is not


def test_heckman_errors():
    for kw, msg in [(dict(sel="kidsge6", z=["educ", "age"]), "0 and 1"), (dict(z=[]), "selection equation"), (dict(y="lwage", x=["lwage"]), "more than one role")]:
        assert msg in mroz(**kw)["error"]


# ------------------------------------------------------------------ count models
def visits(**kw):
    spec = dict(o.BY_ID["count"]["example"]["spec"]); spec.update(kw)
    return o.run("count", open(os.path.join(ME, "doctor_visits.csv")).read(), spec)


def test_poisson_matches_independent_irls():
    d = pd.read_csv(os.path.join(ME, "doctor_visits.csv")); xs = o.BY_ID["count"]["example"]["spec"]["x"]
    X = sm.add_constant(d[xs]).values; y = d.mdvis.values.astype(float); b = np.zeros(X.shape[1]); b[0] = np.log(y.mean())
    for _ in range(100):                                    # Newton-Raphson for the Poisson likelihood, written out by hand
        mu = np.exp(X @ b); b = b + np.linalg.solve(X.T @ (X * mu[:, None]), X.T @ (y - mu))
    r = visits(family="poisson")
    for i, nm in enumerate(["const"] + xs):
        assert row(r, "Coefficient", nm)[1] == pytest.approx(b[i], abs=1e-5)
    mu = np.exp(X @ b); meat = X.T @ (X * ((y - mu) ** 2)[:, None]); bread = np.linalg.inv(X.T @ (X * mu[:, None]))
    V = bread @ meat @ bread
    assert row(r, "Coefficient", "disea")[2] == pytest.approx(np.sqrt(V[6, 6]), rel=1e-4)


def test_negbin_matches_scipy_likelihood():
    from scipy import optimize
    d = pd.read_csv(os.path.join(ME, "doctor_visits.csv")); xs = o.BY_ID["count"]["example"]["spec"]["x"]
    X = sm.add_constant(d[xs]).values; y = d.mdvis.values.astype(float)

    def nll(t):
        mu = np.exp(X @ t[:-1]); a = np.exp(t[-1]); r = 1 / a
        return -stats.nbinom.logpmf(y, r, r / (r + mu)).sum()
    r0 = optimize.minimize(nll, np.r_[np.zeros(X.shape[1]), 0], method="BFGS", options={"gtol": 1e-7, "maxiter": 2000})
    r = visits(family="negbin")
    assert summ(r, "Log-likelihood") == pytest.approx(-r0.fun, abs=1e-2)
    assert summ(r, "alpha") == pytest.approx(np.exp(r0.x[-1]), rel=2e-3)


def test_count_recovers_parameters_and_detects_overdispersion():
    rng = np.random.default_rng(5); n = 5000
    x = rng.normal(size=n); mu = np.exp(0.5 + 0.4 * x)
    y_nb = rng.negative_binomial(2.0, 2.0 / (2.0 + mu))
    y_p = rng.poisson(mu)
    for y, over in [(y_nb, True), (y_p, False)]:
        r = o.run("count", pd.DataFrame(dict(y=y, x=x)).to_csv(index=False), dict(y="y", x=["x"], family="negbin"))
        assert row(r, "Coefficient", "x")[1] == pytest.approx(0.4, abs=0.05)
        p = next(t for t in table(r, "Specification")["rows"] if t[0].startswith("Cameron"))[2]
        assert (p < 0.01) == over
    assert summ(r, "alpha") < 0.1                          # NB collapses to Poisson when the data are Poisson


def test_irr_and_marginal_effect_by_hand():
    r = visits(family="poisson"); b = row(r, "Coefficient", "disea")[1]
    assert row(r, "Incidence", "disea")[1] == pytest.approx(np.exp(b), abs=1e-5)
    d = pd.read_csv(os.path.join(ME, "doctor_visits.csv")); xs = o.BY_ID["count"]["example"]["spec"]["x"]
    X = sm.add_constant(d[xs]).values; bb = np.array([row(r, "Coefficient", nm)[1] for nm in ["const"] + xs])
    assert row(r, "Average marginal", "disea")[1] == pytest.approx((bb[6] * np.exp(X @ bb)).mean(), rel=1e-4)


def test_count_errors():
    for kw, msg in [(dict(y="fmde", x=["disea"]), "whole numbers"), (dict(x=[]), "at least one"), (dict(family="zip"), "must be one of")]:
        assert msg in visits(**kw)["error"]


# ------------------------------------------------------------------ zero-inflated and hurdle models
def arrests(**kw):
    spec = dict(o.BY_ID["zero_count"]["example"]["spec"]); spec.update(kw)
    return o.run("zero_count", open(os.path.join(ME, "arrests.csv")).read(), spec)


def test_zip_matches_statsmodels_at_the_same_optimum():
    from statsmodels.discrete.count_model import ZeroInflatedPoisson
    d = pd.read_csv(os.path.join(ME, "arrests.csv")); xs = o.BY_ID["zero_count"]["example"]["spec"]["x"]
    X = sm.add_constant(d[xs])
    r = arrests(model="zip")
    sm_fit = ZeroInflatedPoisson(d.narr86, X, exog_infl=X, inflation="logit").fit(method="bfgs", maxiter=2000, disp=0)
    assert summ(r, "Log-likelihood") >= sm_fit.llf - 1e-3                      # engine reaches at least the statsmodels optimum
    assert summ(r, "Log-likelihood") == pytest.approx(sm_fit.llf, abs=0.05)


def test_zinb_and_hurdle_likelihoods_by_hand():
    from scipy import optimize
    d = pd.read_csv(os.path.join(ME, "arrests.csv")); xs = o.BY_ID["zero_count"]["example"]["spec"]["x"]
    X = sm.add_constant(d[xs]).values; y = d.narr86.values.astype(float)
    # Poisson hurdle: the logit part and the zero-truncated Poisson part separate, so the log-likelihood is the sum of the two maximised parts
    lg = sm.Logit((y > 0).astype(float), X).fit(disp=0)
    pos = y > 0

    def nll(b):
        mu = np.exp(X[pos] @ b); return -(stats.poisson.logpmf(y[pos], mu) - np.log1p(-np.exp(-mu))).sum()
    t = optimize.minimize(nll, np.zeros(X.shape[1]), method="BFGS", options={"gtol": 1e-8})
    assert summ(arrests(model="hurdle_poisson"), "Log-likelihood") == pytest.approx(lg.llf - t.fun, abs=1e-2)
    r = arrests(model="zinb")
    assert row(r, "Count equation", "alpha (dispersion)")[1] > 0 and summ(r, "Log-likelihood") > summ(arrests(model="zip"), "Log-likelihood")


def test_zero_inflation_recovered_in_simulation():
    rng = np.random.default_rng(11); n = 6000
    x = rng.normal(size=n); w = rng.normal(size=n)
    struct = rng.random(n) < 1 / (1 + np.exp(-(-0.5 + 1.0 * w)))
    y = np.where(struct, 0, rng.poisson(np.exp(0.3 + 0.5 * x)))
    dd = pd.DataFrame(dict(y=y, x=x, w=w))
    r = o.run("zero_count", dd.to_csv(index=False), dict(y="y", x=["x"], z=["w"], model="zip"))
    assert row(r, "Count equation", "x")[1] == pytest.approx(0.5, abs=0.06)
    assert row(r, "Zero (inflation)", "w")[1] == pytest.approx(1.0, abs=0.2)
    assert next(t for t in table(r, "Specification")["rows"] if t[0].startswith("Vuong"))[1] > 1.96


def test_zero_count_errors():
    for kw, msg in [(dict(y="pcnv", x=["black"], z=["hispan"]), "whole numbers"), (dict(model="zz"), "must be one of"), (dict(x=[]), "at least one")]:
        assert msg in arrests(**kw)["error"]
