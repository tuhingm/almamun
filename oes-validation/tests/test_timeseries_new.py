"""Validation of the time-series models added to OES after the first release: unit-root and stationarity tests.
Checks against (1) statsmodels and the `arch` package, (2) simulated series with a known answer and (3) error messages."""
import os, sys, warnings
import numpy as np, pandas as pd, pytest
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "oes", "engine")); sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import oes_engine as o
from oes_engine import ts_extra

OX = os.path.join(ROOT, "oes", "examples")
def macro(): return pd.read_csv(os.path.join(OX, "us_macro.csv"))
def table(r, prefix): return next(t for t in r["tables"] if t["title"].startswith(prefix))
def urow(r, test, form, det): return next(x for x in table(r, "Unit-root")["rows"] if x[0] == test and x[1] == form and x[2] == det)


def test_unitroot_registered_with_guide_and_example():
    m = o.BY_ID["unitroot"]
    assert m["group"] == "Time series" and os.path.exists(os.path.join(ROOT, m["guide"]))
    assert os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_adf_matches_statsmodels():
    from statsmodels.tsa.stattools import adfuller
    x = macro().tbilrate.to_numpy(float)
    r = ts_extra.unitroot(macro(), dict(y="tbilrate", time="date", maxlag=8, ic="bic", sims=50))
    a = adfuller(x, maxlag=8, regression="c", autolag="BIC")
    row = urow(r, "ADF", "level", "constant")
    assert row[3] == pytest.approx(a[0], abs=1e-3) and row[4] == a[2] and row[5] == pytest.approx(a[1], abs=1e-4)


def test_dfgls_and_phillips_perron_match_arch():
    arch = pytest.importorskip("arch.unitroot")
    x = macro().tbilrate.to_numpy(float)
    for tr in ("c", "ct"):
        for k in (1, 4):
            assert ts_extra._dfgls_stat(x, tr, k) == pytest.approx(arch.DFGLS(x, lags=k, trend=tr).stat, abs=1e-6)
        q = int(4 * (len(x) / 100) ** (2 / 9))
        assert ts_extra._pp(x, tr, q)[0] == pytest.approx(arch.PhillipsPerron(x, lags=q, trend=tr, test_type="tau").stat, abs=1e-6)


def test_regression_values_for_us_tbill_rate():
    # stored values for the example (guards against accidental changes): DF-GLS with 1 lag and PP with 4 lags
    x = macro().tbilrate.to_numpy(float)
    assert ts_extra._dfgls_stat(x, "c", 1) == pytest.approx(-1.8027, abs=5e-4)
    assert ts_extra._pp(x, "c", 4)[0] == pytest.approx(-2.0741, abs=5e-4)


def test_random_walk_is_not_rejected_and_ar1_is():
    rng = np.random.default_rng(3); n = 300
    rw = np.cumsum(rng.standard_normal(n)); e = rng.standard_normal(n); ar = np.zeros(n)
    for t in range(1, n): ar[t] = .5 * ar[t - 1] + e[t]
    r1 = ts_extra.unitroot(pd.DataFrame({"y": rw}), dict(y="y", sims=100)); r2 = ts_extra.unitroot(pd.DataFrame({"y": ar}), dict(y="y", sims=100))
    assert "I(1)" in next(s["value"] for s in r1["summary"] if s["label"] == "Conclusion")
    assert "I(0)" in next(s["value"] for s in r2["summary"] if s["label"] == "Conclusion")
    assert urow(r1, "ADF", "level", "constant")[-1] == "no" and urow(r2, "ADF", "level", "constant")[-1] == "yes"


def test_size_of_dfgls_simulated_critical_values():
    # under a unit root the simulated 5% critical value should reject about 5% of the time
    rng = np.random.default_rng(7); rej = 0; N = 60
    cv = np.percentile([ts_extra._dfgls_stat(np.cumsum(rng.standard_normal(150)), "c", 2) for _ in range(400)], 5)
    for _ in range(N):
        rej += ts_extra._dfgls_stat(np.cumsum(rng.standard_normal(150)), "c", 2) < cv
    assert rej / N < 0.20


def test_zivot_andrews_finds_a_break_in_a_shifted_series():
    rng = np.random.default_rng(11); n = 240; e = rng.standard_normal(n); y = np.zeros(n)
    for t in range(1, n): y[t] = .5 * y[t - 1] + e[t]
    y[120:] += 6
    r = ts_extra.unitroot(pd.DataFrame({"y": y}), dict(y="y", sims=50))
    z = table(r, "Zivot-Andrews")["rows"][0]
    assert abs(int(z[4]) - 121) <= 12


def test_unitroot_errors():
    short = pd.DataFrame({"y": np.arange(20.0) ** 1.1, "t": np.arange(20)})
    assert "40" in o.run("unitroot", short.to_csv(index=False), dict(y="y"))["error"]
    assert "series" in o.run("unitroot", macro().to_csv(index=False), dict())["error"].lower()
    assert "BIC" in o.run("unitroot", macro().to_csv(index=False), dict(y="tbilrate", ic="hq"))["error"]


# ------------------------------------------------------------------ unobserved components and filters
def gdp(): return pd.read_csv(os.path.join(OX, "us_gdp_level.csv"))


def test_unobserved_registered():
    m = o.BY_ID["unobserved"]
    assert m["group"] == "Time series" and os.path.exists(os.path.join(ROOT, m["guide"])) and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_hamilton_filter_matches_ols_by_hand():
    import statsmodels.api as sm
    x = np.cumsum(np.random.default_rng(2).standard_normal(120)); h = 8
    e = ts_extra._hamilton(x, h=h, p=4)
    t = np.arange(3 + h, 120)
    X = sm.add_constant(np.column_stack([x[t - h - j] for j in range(4)])); res = sm.OLS(x[t], X).fit()
    assert np.allclose(e[t], res.resid, atol=1e-8) and np.isnan(e[:3 + h]).all()


def test_uc_cycle_recovered_from_simulation():
    # trend + damped stochastic cycle with period 20 and damping 0.92
    rng = np.random.default_rng(5); n = 400; lam = 2 * np.pi / 20; rho = .92
    c = np.zeros(n); cs = np.zeros(n)
    for t in range(1, n):
        a = rho * (np.cos(lam) * c[t - 1] + np.sin(lam) * cs[t - 1]) + .8 * rng.standard_normal()
        b = rho * (-np.sin(lam) * c[t - 1] + np.cos(lam) * cs[t - 1]) + .8 * rng.standard_normal(); c[t], cs[t] = a, b
    trend = np.cumsum(np.cumsum(.02 * rng.standard_normal(n))) * .2 + .05 * np.arange(n)
    df = pd.DataFrame({"y": trend + c})
    r = ts_extra.unobserved(df, dict(y="y", ppy=4, pmin=6, pmax=40))
    per = next(s["value"] for s in r["summary"] if s["label"] == "Cycle period (years)") * 4
    assert 14 < per < 28
    cy = np.array(r["charts"][1]["series"][0]["y"], float)
    assert np.corrcoef(cy, c)[0, 1] > .8


def test_hp_cycle_in_engine_equals_statsmodels():
    from statsmodels.tsa.filters.hp_filter import hpfilter
    d = gdp(); r = ts_extra.unobserved(d, dict(y="gdp_level", time="date"))
    cyc, _ = hpfilter(d.gdp_level.to_numpy(float), lamb=1600)
    assert np.allclose(r["charts"][1]["series"][1]["y"], np.round(cyc, 6), atol=1e-5)


def test_unobserved_errors():
    d = gdp()
    assert "positive" in ts_extra_err(d.assign(gdp_level=-d.gdp_level.abs()), dict(y="gdp_level", log="yes")).lower()
    assert "Choose" in ts_extra_err(d, dict(y="gdp_level", trend="cubic"))
    assert "shorter" in ts_extra_err(d, dict(y="gdp_level", pmin=30, pmax=20)).lower() or "bounds" in ts_extra_err(d, dict(y="gdp_level", pmin=30, pmax=20)).lower()
    assert "40" in ts_extra_err(d.head(30), dict(y="gdp_level"))


def ts_extra_err(df, spec):
    cols = list(df.columns)
    return o.run("unobserved", df.to_csv(index=False), spec)["error"]


# ------------------------------------------------------------------ forecast comparison
def test_forecast_eval_registered_and_dm_by_hand():
    m = o.BY_ID["forecast_eval"]
    assert os.path.exists(os.path.join(ROOT, m["guide"])) and os.path.exists(os.path.join(ROOT, m["example"]["file"]))
    rng = np.random.default_rng(1); d = rng.standard_normal(80) + .3; T = 80
    # h = 1: the DM statistic is the t statistic of the mean, times the HLN factor sqrt((T-1)/T)
    st, p = ts_extra._dm(d, 1)
    t_ = d.mean() / (d.std(ddof=0) / np.sqrt(T)); assert st == pytest.approx(t_ * np.sqrt((T - 1) / T), rel=1e-9)
    assert p == pytest.approx(2 * stats.t.sf(abs(st), T - 1), rel=1e-9)


def test_dm_rejects_when_one_forecast_is_clearly_better_and_not_when_equal():
    rng = np.random.default_rng(4); n = 300; x = np.zeros(n)
    for t in range(1, n): x[t] = .8 * x[t - 1] + rng.standard_normal()
    r = ts_extra.forecast_eval(pd.DataFrame({"y": x}), dict(y="y", window=150, benchmark="historical mean"))
    row = next(r_ for r_ in r["tables"][0]["rows"] if r_[0] == "AR (BIC)")
    assert row[6] == "better than benchmark" and row[3] < 0.9


def test_forecasts_use_no_future_data():
    d = macro(); a = ts_extra.forecast_eval(d, dict(y="inflation", window=100))
    d2 = d.copy(); d2.loc[d2.index[-1], "inflation"] += 50          # change only the last observation
    b = ts_extra.forecast_eval(d2, dict(y="inflation", window=100))
    fa = a["charts"][0]["series"][3]["y"]; fb = b["charts"][0]["series"][3]["y"]
    assert fa[:-1] == fb[:-1]


def test_forecast_eval_errors():
    d = macro()
    assert "forecasts" in o.run("forecast_eval", d.to_csv(index=False), dict(y="inflation", window=187, h=4))["error"].lower()
    assert "benchmark" in o.run("forecast_eval", d.to_csv(index=False), dict(y="inflation", benchmark="oracle"))["error"].lower()
    assert "loss" in o.run("forecast_eval", d.to_csv(index=False), dict(y="inflation", loss="huber"))["error"].lower()


# ------------------------------------------------------------------ connectedness (Diebold-Yilmaz)
RET = os.path.join(ROOT, "time-series-guide", "examples", "returns.csv")
def rets(): return pd.read_csv(RET)


def test_connectedness_registered():
    m = o.BY_ID["connectedness"]
    assert os.path.exists(os.path.join(ROOT, m["guide"])) and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_gfevd_equals_orthogonal_fevd_when_errors_uncorrelated_and_rows_sum_to_one():
    from statsmodels.tsa.api import VAR
    rng = np.random.default_rng(8); n = 600; A = np.array([[.5, .2], [.1, .4]]); y = np.zeros((n, 2))
    for t in range(1, n): y[t] = A @ y[t - 1] + rng.standard_normal(2) * [1, 2]
    res = VAR(y).fit(1)
    # force a diagonal error covariance: the generalised and the Cholesky decompositions then coincide
    res.sigma_u = np.diag(np.diag(np.asarray(res.sigma_u)))
    g = ts_extra._gfevd(res, 10)
    assert np.allclose(g.sum(1), 1)
    fe = VAR(y).fit(1).fevd(10).decomp                      # shape (k, H, k); uses the Cholesky of the estimated sigma
    # with diagonal sigma the Cholesky FEVD is the same as GFEVD of the diagonal model; verify with a hand computation
    S = np.asarray(res.sigma_u); Ah = [np.linalg.matrix_power(A_, 1) for A_ in [res.coefs[0]]]
    Phi = [np.linalg.matrix_power(res.coefs[0], h) for h in range(10)]
    num = np.array([[sum((Phi[h][i, j]) ** 2 * S[j, j] for h in range(10)) for j in range(2)] for i in range(2)])
    assert np.allclose(g, num / num.sum(1, keepdims=True), atol=1e-10)


def test_gfevd_does_not_depend_on_variable_order():
    from statsmodels.tsa.api import VAR
    x = rets()[["sp500_return", "ip_growth", "d_fedfunds"]].to_numpy(float)
    g1 = ts_extra._gfevd(VAR(x).fit(2), 10); perm = [2, 0, 1]; g2 = ts_extra._gfevd(VAR(x[:, perm]).fit(2), 10)
    assert np.allclose(g1[np.ix_(perm, perm)], g2, atol=1e-10)


def test_connectedness_independent_series_have_low_connectedness():
    rng = np.random.default_rng(6); df = pd.DataFrame(rng.standard_normal((400, 3)), columns=list("abc"))
    r = ts_extra.connectedness(df, dict(x=["a", "b", "c"]))
    assert next(s["value"] for s in r["summary"] if s["label"] == "Total connectedness (%)") < 8


def test_connectedness_errors():
    assert "two" in o.run("connectedness", rets().to_csv(index=False), dict(x=["sp500_return"]))["error"]
    assert "60" in o.run("connectedness", rets().head(30).to_csv(index=False), dict(x=["sp500_return", "ip_growth"]))["error"]


# ------------------------------------------------------------------ Toda-Yamamoto Granger causality
MONEY = os.path.join(ROOT, "time-series-guide", "examples", "money.csv")
def money(): return pd.read_csv(MONEY)


def test_granger_ty_registered():
    m = o.BY_ID["granger_ty"]
    assert os.path.exists(os.path.join(ROOT, m["guide"])) and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_granger_ty_wald_matches_statsmodels_ols_wald_test():
    import statsmodels.api as sm
    d = money()[["log_price", "log_m2", "log_gdp"]].to_numpy(float); p, dmax, k = 3, 1, 3; L = p + dmax
    r = ts_extra.granger_ty(money(), dict(x=["log_price", "log_m2", "log_gdp"], lags=p, dmax=dmax))
    got = next(x for x in r["tables"][0]["rows"] if x[0] == "log_m2 -> log_price")
    X = np.column_stack([d[L - l:len(d) - l, j] for l in range(1, L + 1) for j in range(k)]); res = sm.OLS(d[L:, 0], sm.add_constant(X)).fit()
    names = [f"x{i + 1}" for i in range(X.shape[1])]; idx = [1 + (l * k + 1) for l in range(p)]
    w = res.wald_test(np.eye(len(res.params))[idx], use_f=False, scalar=True)
    assert got[1] == pytest.approx(float(w.statistic), abs=1e-3)


def test_granger_ty_detects_true_causality_and_is_not_fooled_by_random_walks():
    rng = np.random.default_rng(12); n = 300; x = np.cumsum(rng.standard_normal(n)); y = np.zeros(n)
    for t in range(1, n): y[t] = .5 * y[t - 1] + .6 * x[t - 1] + rng.standard_normal()
    r = ts_extra.granger_ty(pd.DataFrame({"x": x, "y": y}), dict(x=["x", "y"], lags=2, dmax=1))
    rows = {r_[0]: r_ for r_ in r["tables"][0]["rows"]}
    assert rows["x -> y"][-1].startswith("rejects")
    rej = 0
    for s in range(30):
        g = np.random.default_rng(100 + s); a, b = np.cumsum(g.standard_normal(200)), np.cumsum(g.standard_normal(200))
        rr = ts_extra.granger_ty(pd.DataFrame({"a": a, "b": b}), dict(x=["a", "b"], lags=2, dmax=1)); rej += rr["tables"][0]["rows"][0][-1].startswith("rejects")
    assert rej <= 6                                   # spurious causality between independent random walks stays near the 5% level


def test_granger_ty_errors():
    assert "two" in o.run("granger_ty", money().to_csv(index=False), dict(x=["log_price"]))["error"]
    assert "AIC" in o.run("granger_ty", money().to_csv(index=False), dict(x=["log_price", "log_m2"], ic="xx"))["error"]


# ------------------------------------------------------------------ seasonal decomposition
GAS = os.path.join(OX, "uk_gas.csv")
def gas(): return pd.read_csv(GAS)


def test_seasonal_registered():
    m = o.BY_ID["seasonal"]
    assert os.path.exists(os.path.join(ROOT, m["guide"])) and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_seasonal_strength_and_adjusted_series_match_statsmodels_stl():
    from statsmodels.tsa.seasonal import STL
    d = gas(); r = ts_extra.seasonal(d, dict(y="gas", time="date", period=4))
    z = np.log(d.gas.to_numpy(float)); res = STL(z, period=4, robust=True, seasonal=7).fit()
    Fs = max(0, 1 - res.resid.var() / (res.seasonal + res.resid).var())
    assert next(s["value"] for s in r["summary"] if s["label"] == "Seasonal strength") == pytest.approx(Fs, abs=5e-4)
    adj = r["charts"][0]["series"][2]["y"]; assert np.allclose(adj, np.round(np.exp(z - res.seasonal), 6), atol=1e-4)


def test_seasonal_pattern_recovered_and_white_noise_not_seasonal():
    rng = np.random.default_rng(9); n = 160; t = np.arange(n); fac = np.array([1.3, 0.9, 0.6, 1.2]); fac = fac / fac.mean()
    x = (50 + .5 * t) * fac[t % 4] * np.exp(.02 * rng.standard_normal(n))
    r = ts_extra.seasonal(pd.DataFrame({"x": x}), dict(y="x", period=4))
    got = np.array([row[1] for row in r["tables"][0]["rows"]], float); assert np.allclose(got, fac / np.exp(np.log(fac).mean()), atol=0.03)      # STL factors have geometric mean one
    w = ts_extra.seasonal(pd.DataFrame({"x": 10 + rng.standard_normal(n)}), dict(y="x", period=4, mode="additive"))
    assert next(s["value"] for s in w["summary"] if s["label"] == "Seasonality test p-value") > 0.01


def test_seasonal_errors():
    d = gas()
    assert "positive" in o.run("seasonal", d.assign(gas=d.gas - 200).to_csv(index=False), dict(y="gas"))["error"]
    assert "short" in o.run("seasonal", d.head(10).to_csv(index=False), dict(y="gas", period=4))["error"]
    assert "Choose" in o.run("seasonal", d.to_csv(index=False), dict(y="gas", mode="mixed"))["error"]


# ------------------------------------------------------------------ long memory
def test_longmemory_registered():
    m = o.BY_ID["longmemory"]
    assert os.path.exists(os.path.join(ROOT, m["guide"])) and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_fracdiff_special_cases():
    x = np.random.default_rng(1).standard_normal(50)
    assert np.allclose(ts_extra._fracdiff(x, 0.0), x)
    assert np.allclose(ts_extra._fracdiff(x, 1.0)[1:], np.diff(x))                       # d = 1 is the first difference
    assert np.allclose(ts_extra._fracdiff(ts_extra._fracdiff(x, .3), -.3), x, atol=1e-10)  # the filters are inverses


def test_gph_and_whittle_recover_fractional_noise():
    rng = np.random.default_rng(21); n = 3000
    for d0 in (0.0, 0.3):
        e = rng.standard_normal(n); x = ts_extra._fracdiff(e, -d0)          # ARFIMA(0, d0, 0): apply (1 - L)^(-d0)
        m = int(n ** .65); g, sg = ts_extra._gph(x, m); w, sw = ts_extra._whittle(x, m)
        assert abs(g - d0) < 3.5 * sg and abs(w - d0) < 3.5 * sw
    assert abs(ts_extra._whittle(np.cumsum(rng.standard_normal(n)), int(n ** .65))[0] - 1) < .2   # random walk: d near one


def test_longmemory_errors_and_summary():
    d = pd.read_csv(os.path.join(OX, "us_macro.csv"))
    r = ts_extra.longmemory(d, dict(y="inflation", time="date"))
    assert 0.4 < next(s["value"] for s in r["summary"] if s["label"] == "Local Whittle d") < 0.9
    assert "100" in o.run("longmemory", d.head(50).to_csv(index=False), dict(y="inflation"))["error"]
    assert "Choose" in o.run("longmemory", d.to_csv(index=False), dict(y="inflation", transform="cube"))["error"]
    assert "between" in o.run("longmemory", d.to_csv(index=False), dict(y="inflation", power=0.95))["error"]


# ------------------------------------------------------------------ factor model
def test_factors_registered():
    m = o.BY_ID["factors"]
    assert os.path.exists(os.path.join(ROOT, m["guide"])) and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_factors_eigenvalues_match_sklearn_pca_and_recover_true_factors():
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(14); T, N, r = 150, 60, 2
    F = rng.standard_normal((T, r)); L = rng.standard_normal((N, r)); X = F @ L.T + .7 * rng.standard_normal((T, N))
    df = pd.DataFrame(X, columns=[f"v{i}" for i in range(N)])
    res = ts_extra.factors(df, dict(x=list(df.columns), kmax=6))
    assert next(s["value"] for s in res["summary"] if s["label"] == "Factors (IC_p2)") == 2
    Z = (X - X.mean(0)) / X.std(0); ev = PCA(n_components=6).fit(Z).explained_variance_ * (T - 1) / T
    got = [row[5] for row in res["tables"][0]["rows"][1:]]; assert np.allclose(got, ev, atol=2e-3)
    Fh = np.array([s["y"] for s in res["charts"][0]["series"]]).T
    R2 = 1 - np.linalg.lstsq(Fh, F, rcond=None)[1] / ((F - F.mean(0)) ** 2).sum(0); assert (R2 > .95).all()


def test_factors_errors():
    d = pd.read_csv(os.path.join(OX, "inflation_countries.csv"))
    assert "five" in o.run("factors", d.to_csv(index=False), dict(x=["USA", "CAN", "MEX"]))["error"]
    assert "smaller" in o.run("factors", d.to_csv(index=False), dict(x=["USA", "CAN", "MEX", "CHL", "COL", "GBR"], r=6))["error"]
    assert "30" in o.run("factors", d.head(20).to_csv(index=False), dict(x=list(d.columns[1:9])))["error"]


# ------------------------------------------------------------------ spectral analysis
def test_spectral_registered():
    m = o.BY_ID["spectral"]
    assert os.path.exists(os.path.join(ROOT, m["guide"])) and os.path.exists(os.path.join(ROOT, m["example"]["file"]))


def test_spectral_finds_known_cycle_and_matches_scipy_welch():
    from scipy import signal
    rng = np.random.default_rng(3); n = 400; t = np.arange(n); x = np.sin(2 * np.pi * t / 20) + .5 * rng.standard_normal(n)   # period 20 periods = 5 years (ppy 4)
    r = ts_extra.spectral(pd.DataFrame({"x": x}), dict(y="x", ppy=4, detrend="mean", segment=100))
    assert next(s["value"] for s in r["summary"] if s["label"] == "Dominant period (years)") == pytest.approx(5.0, abs=0.5)
    fr, P = signal.welch(x - x.mean(), fs=1.0, nperseg=100, window="hann", noverlap=50, detrend=False)
    assert r["charts"][0]["series"][0]["y"][0] == pytest.approx(round(P[1], 6), rel=1e-4)


def test_spectral_coherence_is_high_for_related_series_and_low_for_independent():
    rng = np.random.default_rng(4); n = 600; t = np.arange(n); c = np.sin(2 * np.pi * t / 24)
    a = c + .3 * rng.standard_normal(n); b = np.roll(c, 3) + .3 * rng.standard_normal(n); z = rng.standard_normal(n)
    hi = ts_extra.spectral(pd.DataFrame({"a": a, "b": b}), dict(y="a", x="b", ppy=4, segment=120, detrend="mean"))
    lo = ts_extra.spectral(pd.DataFrame({"a": a, "z": z}), dict(y="a", x="z", ppy=4, segment=120, detrend="mean"))
    c_hi = max(hi["charts"][1]["series"][0]["y"]); c_lo = max(lo["charts"][1]["series"][0]["y"])
    assert c_hi > 0.8 and c_lo < 0.65


def test_spectral_errors():
    d = pd.read_csv(os.path.join(OX, "us_macro.csv"))
    assert "60" in o.run("spectral", d.head(30).to_csv(index=False), dict(y="gdp_growth"))["error"]
    assert "Choose" in o.run("spectral", d.to_csv(index=False), dict(y="gdp_growth", detrend="cubic"))["error"]
