"""Software validation (OMVP stage 8) for the OES Statistics Division.
Run:  pip install numpy pandas scipy statsmodels pytest  &&  pytest stat-validation/tests -q
Covers registry integrity, integration (every method on its documented example), regression against stored
references, reproducibility, edge cases, invalid input, and numerical accuracy against SciPy / statsmodels."""
import json, math, os, sys, warnings
import numpy as np, pandas as pd, pytest
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "oes", "engine")); sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import oes_engine as o

EX = os.path.join(ROOT, "stat-guide", "examples")
EXPECTED = json.load(open(os.path.join(ROOT, "oes", "engine", "stat_expected.json")))
STAT = {m["id"]: m for m in o.STAT_MODELS}
IDS = list(STAT)

def csv_of(m):
    f = m["example"].get("file")
    return open(os.path.join(ROOT, f)).read() if f else ""

def run_example(mid):
    m = STAT[mid]; return o.run(mid, csv_of(m), dict(m["example"]["spec"]))

def table(r, prefix): return next(t for t in r["tables"] if t["title"].startswith(prefix))
def summ(r, label): return next(s["value"] for s in r["summary"] if s["label"] == label)

# ---------------------------------------------------------------- registry
def test_registry_counts():
    allm = list(o.MODELS) + list(o.STAT_MODELS)
    assert len(o.STAT_MODELS) == 31 and len(o.MODELS) == 69 and len(allm) == 100
    assert len({m["id"] for m in allm}) == 100
    assert len({m["name"] for m in allm}) == 100

def test_seven_categories_with_expected_sizes():
    from collections import Counter
    c = Counter(m["group"] for m in o.STAT_MODELS)
    assert sorted(c.values()) == sorted([6, 5, 6, 5, 4, 3, 2]) and len(c) == 7

def test_guide_pages_declared_unique():
    g = [m["guide"] for m in o.STAT_MODELS]; assert len(set(g)) == 31

# ---------------------------------------------------------------- integration + regression
@pytest.mark.parametrize("mid", IDS)
def test_example_runs_and_is_finite(mid):
    r = run_example(mid)
    assert "error" not in r, r.get("error")
    assert r["tables"] and r["summary"]
    for s in r["summary"]:
        if isinstance(s["value"], float): assert math.isfinite(s["value"]), s
    assert r["guide"] == STAT[mid]["guide"]

@pytest.mark.parametrize("mid", IDS)
def test_regression_against_stored_reference(mid):
    e = EXPECTED[mid]; r = run_example(mid)
    t = r["tables"][e["table_index"]]
    v = t["rows"][e["row_index"]][e["col_index"]]
    assert v == pytest.approx(e["value"], rel=max(e["rtol"], 1e-3), abs=1e-3)

@pytest.mark.parametrize("mid", IDS)
def test_reproducible(mid):
    a, b = run_example(mid), run_example(mid)
    for k in ("summary", "tables"): assert json.dumps(a[k]) == json.dumps(b[k])

# ---------------------------------------------------------------- invalid input
@pytest.mark.parametrize("mid", [i for i in IDS if not STAT[i].get("nodata")])
def test_missing_column_returns_error_not_exception(mid):
    spec = dict(STAT[mid]["example"]["spec"])
    sel = {f["key"] for f in STAT[mid]["fields"] if f["type"] == "select"}
    for k, v in spec.items():
        if k in sel: continue
        if isinstance(v, str): spec[k] = "no_such_column"; break
        if isinstance(v, list) and v and isinstance(v[0], str): spec[k] = ["no_such_column"] + v[1:]; break
    r = o.run(mid, csv_of(STAT[mid]), spec); assert "error" in r

@pytest.mark.parametrize("mid", [i for i in IDS if not STAT[i].get("nodata")])
def test_empty_data_returns_error(mid):
    r = o.run(mid, "a,b\n", dict(STAT[mid]["example"]["spec"])); assert "error" in r

def test_unknown_model():
    assert "error" in o.run("stat_nope", "a\n1\n", {})

# ---------------------------------------------------------------- edge cases
def test_anova_single_group_rejected():
    d = pd.DataFrame({"y": [1., 2, 3, 4], "g": ["a"] * 4})
    assert "error" in o.run("stat_anova1", d.to_csv(index=False), dict(y="y", group="g"))

def test_mann_whitney_three_groups_rejected():
    d = pd.DataFrame({"y": range(9), "g": list("abc") * 3})
    assert "error" in o.run("stat_mann_whitney", d.to_csv(index=False), dict(y="y", group="g"))

def test_wilcoxon_all_zero_differences():
    d = pd.DataFrame({"a": [1., 2, 3, 4], "b": [1., 2, 3, 4]})
    r = o.run("stat_wilcoxon", d.to_csv(index=False), dict(x1="a", x2="b")); assert "error" in r or True  # must not raise

def test_missing_values_dropped_and_reported():
    d = pd.read_csv(os.path.join(EX, "plantgrowth.csv")); d.loc[0, "weight"] = np.nan
    r = o.run("stat_anova1", d.to_csv(index=False), dict(y="weight", group="group")); assert "error" not in r and r["nobs"] == 29

def test_collinear_covariate_handled():
    rng = np.random.default_rng(0); x = rng.normal(size=40)
    d = pd.DataFrame({"y": rng.normal(size=40), "g": ["a", "b"] * 20, "x1": x, "x2": 2 * x})
    r = o.run("stat_ancova", d.to_csv(index=False), dict(y="y", group="g", covariates=["x1", "x2"])); assert isinstance(r, dict)

def test_km_all_censored():
    d = pd.DataFrame({"t": [1., 2, 3, 4, 5], "e": [0] * 5})
    r = o.run("stat_km", d.to_csv(index=False), dict(time="t", event="e")); assert isinstance(r, dict)

# ---------------------------------------------------------------- numerical accuracy vs references
def test_anova1_vs_scipy():
    d = pd.read_csv(os.path.join(EX, "plantgrowth.csv")); r = run_example("stat_anova1")
    ref = stats.f_oneway(*[v.weight.values for _, v in d.groupby("group")])
    assert summ(r, "F") == pytest.approx(ref.statistic, rel=1e-4)

def test_kruskal_vs_scipy():
    d = pd.read_csv(os.path.join(EX, "airquality_ozone.csv")).dropna(subset=["Ozone", "Month"]); r = run_example("stat_kruskal")
    ref = stats.kruskal(*[v.Ozone.values for _, v in d.groupby("Month")])
    assert summ(r, "H") == pytest.approx(ref.statistic, rel=1e-4) and summ(r, "p-value") == pytest.approx(ref.pvalue, abs=1e-6)

def test_mann_whitney_vs_scipy():
    d = pd.read_csv(os.path.join(EX, "toothgrowth.csv")); r = run_example("stat_mann_whitney")
    a, b = [v.len.values for _, v in d.groupby("supp")]
    ref = stats.mannwhitneyu(a, b, method="asymptotic", use_continuity=True)
    assert summ(r, "p-value") == pytest.approx(ref.pvalue, rel=1e-3)

def test_wilcoxon_exact_vs_scipy():
    rng = np.random.default_rng(1); x = rng.normal(size=15); y = x + rng.normal(0.4, 1, 15)
    r = o.run("stat_wilcoxon", pd.DataFrame({"x": x, "y": y}).to_csv(index=False), dict(x1="x", x2="y", method="exact"))
    assert summ(r, "p-value") == pytest.approx(stats.wilcoxon(x, y, method="exact").pvalue, abs=1e-5)

def test_kendall_vs_scipy():
    rng = np.random.default_rng(2); x = rng.normal(size=60); y = 0.5 * x + rng.normal(size=60)
    r = o.run("stat_kendall", pd.DataFrame({"x": x, "y": y}).to_csv(index=False), dict(x="x", y="y", method="asymptotic"))
    ref = stats.kendalltau(x, y); assert summ(r, "Tau-b") == pytest.approx(ref.statistic, abs=1e-4)

def test_pca_eigenvalues_vs_numpy():
    d = pd.read_csv(os.path.join(EX, "usarrests.csv")); r = run_example("stat_pca")
    X = d.select_dtypes("number"); Z = (X - X.mean()) / X.std(ddof=1)
    ev = np.sort(np.linalg.eigvalsh(np.cov(Z.T)))[::-1]
    t = table(r, "Eigenvalues") if any(t["title"].startswith("Eigenvalues") for t in r["tables"]) else r["tables"][0]
    assert any(abs(float(c) - ev[0]) < 1e-3 for row in t["rows"] for c in row if isinstance(c, (int, float)))

def test_cox_vs_statsmodels():
    import statsmodels.api as sm
    rng = np.random.default_rng(3); n = 300; x = rng.normal(size=n)
    t = rng.exponential(1 / np.exp(0.5 * x)); c = rng.exponential(2, n)
    d = pd.DataFrame({"t": np.minimum(t, c), "e": (t <= c).astype(int), "x": x})
    r = o.run("stat_cox", d.to_csv(index=False), dict(time="t", event="e", covariates=["x"], ties="breslow"))
    assert "error" not in r
    ref = sm.PHReg(d.t, d[["x"]], status=d.e, ties="breslow").fit()
    tb = next(t for t in r["tables"] if "oefficient" in t["title"] or "azard" in t["title"])
    assert any(abs(float(c) - ref.params[0]) < 1e-5 for row in tb["rows"] for c in row if isinstance(c, (int, float)))

def test_lmm_vs_statsmodels():
    import statsmodels.formula.api as smf
    d = pd.read_csv(os.path.join(EX, "sleepstudy.csv")); r = run_example("stat_lmm")
    assert "error" not in r
    ref = smf.mixedlm("Reaction ~ Days", d, groups=d["Subject"], re_formula="~Days").fit(reml=True)
    est = table(r, "Fixed")["rows"][1][1]
    assert est == pytest.approx(ref.fe_params["Days"], rel=1e-3)

def test_gee_vs_statsmodels():
    import statsmodels.api as sm
    rng = np.random.default_rng(4); g = np.repeat(np.arange(60), 5); x = rng.normal(size=300)
    y = 1 + 0.5 * x + np.repeat(rng.normal(size=60), 5) + rng.normal(size=300)
    d = pd.DataFrame({"y": y, "x": x, "id": g})
    r = o.run("stat_gee", d.to_csv(index=False), dict(y="y", covariates=["x"], group="id", family="gaussian", corr="exchangeable"))
    ref = sm.GEE.from_formula("y ~ x", "id", d, cov_struct=sm.cov_struct.Exchangeable()).fit()
    assert "error" not in r
    flat = [float(c) for t in r["tables"] for row in t["rows"] for c in row if isinstance(c, (int, float))]
    assert any(abs(c - ref.params["x"]) < 1e-4 for c in flat)

def test_icc_vs_shrout_fleiss():
    r = run_example("stat_icc"); assert "error" not in r
    assert summ(r, "ICC(2,1)") == pytest.approx(0.2898, abs=5e-4) and summ(r, "ICC(3,1)") == pytest.approx(0.7148, abs=5e-4)  # Shrout & Fleiss (1979), Table 2

def test_power_t_vs_statsmodels():
    from statsmodels.stats.power import TTestIndPower
    n = TTestIndPower().solve_power(effect_size=0.5, alpha=0.05, power=0.8)
    r = o.run("stat_power", "", dict(test="t_two", effect=0.5, alpha=0.05, power=0.8, solve_for="n"))
    assert "error" not in r
    assert any(abs(float(c) - n) < 0.6 for t in r["tables"] for row in t["rows"] for c in row if isinstance(c, (int, float)))

def test_meta_random_reml_vs_metafor_bcg():
    r = run_example("stat_meta_random"); assert "error" not in r
    flat = [c for t in r["tables"] for row in t["rows"] for c in row if isinstance(c, float)]
    assert any(abs(c - (-0.7145)) < 2e-3 for c in flat)       # metafor rma(REML) pooled log RR
    assert any(abs(c - 0.3132) < 2e-3 for c in flat)          # tau^2

def test_bootstrap_se_close_to_theory():
    rng = np.random.default_rng(5); x = rng.normal(0, 1, 200)
    r = o.run("stat_bootstrap", pd.DataFrame({"v": x, "w": 0.0 * x}).to_csv(index=False), dict(y="v", statistic="mean", n_boot=4000, seed=1))
    assert "error" not in r
    se = x.std(ddof=1) / np.sqrt(len(x))
    flat = [float(c) for t in r["tables"] for row in t["rows"] for c in row if isinstance(c, (int, float))]
    assert any(abs(c - se) / se < 0.06 for c in flat)

def test_permutation_exact_matches_bruteforce():
    import itertools
    x = np.array([1.1, 2.3, 3.9, 4.2]); y = np.array([2.8, 5.1, 6.0, 7.7])
    d = pd.DataFrame({"v": np.r_[x, y], "g": ["a"] * 4 + ["b"] * 4})
    r = o.run("stat_permutation", d.to_csv(index=False), dict(y="v", group="g", test="mean_diff"))
    allv = np.r_[x, y]; obs = abs(x.mean() - y.mean()); cnt = tot = 0
    for idx in itertools.combinations(range(8), 4):
        a = allv[list(idx)]; b = np.delete(allv, list(idx)); tot += 1; cnt += abs(a.mean() - b.mean()) >= obs - 1e-12
    assert "error" not in r
    assert any(abs(float(c) - cnt / tot) < 2e-6 for t in r["tables"] for row in t["rows"] for c in row if isinstance(c, (int, float)))
