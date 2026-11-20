"""Validation of the finance models: published/independent values, Monte Carlo and error messages."""
import os, sys, warnings
import numpy as np, pandas as pd, pytest
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "oes", "engine")); sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import oes_engine as o
from oes_engine import finance

EX = os.path.join(ROOT, "finance-guide", "examples")


def _tab(r, title):
    return next(t for t in r["tables"] if t["title"].startswith(title))


def _csv(df):
    return df.to_csv(index=False)


# ------------------------------------------------------------------ CAPM and GRS
def test_capm_matches_statsmodels_and_manual_grs():
    import statsmodels.api as sm
    d = pd.read_csv(os.path.join(EX, "capm.csv"))
    r = o.run("capm", _csv(d), dict(assets=["rfood", "rdur", "rcon"], factors=["rmrf"], time="month"))
    t = _tab(r, "Alpha")["rows"]
    for row in t:
        m = sm.OLS(d[row[0]], sm.add_constant(d[["rmrf"]])).fit(cov_type="HAC", cov_kwds=dict(maxlags=4, use_correction=False))
        assert abs(row[1] - m.params["const"]) < 1e-5 and abs(row[5] - m.params["rmrf"]) < 1e-5
        assert abs(row[2] - m.bse["const"]) < 2e-3
    Y = d[["rfood", "rdur", "rcon"]].to_numpy(); X = np.column_stack([np.ones(len(d)), d.rmrf])
    B = np.linalg.lstsq(X, Y, rcond=None)[0]; E = Y - X @ B; T = len(d); S = E.T @ E / T
    mu = d.rmrf.mean(); om = d.rmrf.var(ddof=0)
    grs = (T - 3 - 1) / 3 * (B[0] @ np.linalg.solve(S, B[0])) / (1 + mu ** 2 / om)
    got = _tab(r, "Joint")["rows"][0]
    assert abs(got[1] - grs) < 1e-3 and abs(got[2] - stats.f.sf(grs, 3, T - 4)) < 1e-4


def test_capm_grs_size_and_power_monte_carlo():
    rng = np.random.default_rng(1)
    rej0 = rej1 = 0
    for k in range(200):
        T = 240
        f = rng.normal(0.5, 4, T)
        e = rng.normal(0, 2, (T, 5))
        beta = np.linspace(0.6, 1.4, 5)
        d0 = pd.DataFrame(np.outer(f, beta) + e, columns=list("abcde")); d0["m"] = f
        d1 = d0.copy(); d1[list("abcde")] += 0.35
        for d, store in ((d0, 0), (d1, 1)):
            r = o.run("capm", _csv(d), dict(assets=list("abcde"), factors=["m"]))
            p = _tab(r, "Joint")["rows"][0][2]
            if store == 0:
                rej0 += p < 0.05
            else:
                rej1 += p < 0.05
    assert 3 <= rej0 <= 20 and rej1 > 120            # size about 5% of 200; power well above 50%


def test_capm_multi_factor_and_rf_subtraction():
    rng = np.random.default_rng(2)
    T = 400
    f = rng.normal(0, 3, (T, 2)); rf = np.full(T, 0.3)
    y = 0.2 + f @ [1.0, -0.5] + rng.normal(0, 1, T) + 0.3
    d = pd.DataFrame(dict(y=y, f1=f[:, 0], f2=f[:, 1], rf=rf))
    r = o.run("capm", _csv(d), dict(assets=["y"], factors=["f1", "f2"], rf="rf"))
    row = _tab(r, "Alpha")["rows"][0]
    assert abs(row[1] - 0.2) < 0.15 and abs(row[5] - 1.0) < 0.1 and abs(row[6] + 0.5) < 0.1


def test_capm_errors():
    d = pd.DataFrame(dict(a=np.arange(10.0), m=np.arange(10.0) ** 2))
    assert "Too few" in o.run("capm", _csv(d), dict(assets=["a"], factors=["m"]))["error"]
    assert "not found" in o.run("capm", _csv(d), dict(assets=["zz"], factors=["m"]))["error"]
    assert "Choose" in o.run("capm", _csv(d), dict(factors=["m"]))["error"]


# ------------------------------------------------------------------ GARCH core against the arch package
def test_garch11_fit_matches_arch_package():
    arch = pytest.importorskip("arch")
    rng = np.random.default_rng(3)
    n = 3000
    r = np.empty(n); s2 = 0.04
    for t in range(n):
        r[t] = 0.05 + np.sqrt(s2) * rng.normal()
        s2 = 0.02 + 0.08 * (r[t] - 0.05) ** 2 + 0.9 * s2
    g = finance._garch11_fit(r)
    m = arch.arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
    assert abs(g["al"] - m.params["alpha[1]"]) < 0.01 and abs(g["be"] - m.params["beta[1]"]) < 0.01
    assert abs(g["om"] - m.params["omega"]) < 0.01 and abs(g["mu"] - m.params["mu"]) < 0.01
    assert abs(g["ll"] - m.loglikelihood) < 1.0


# ------------------------------------------------------------------ value at risk
def _garch_returns(n, seed):
    rng = np.random.default_rng(seed)
    r = np.empty(n); s2 = 1.0
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_t(6) * np.sqrt(4 / 6)
        s2 = 0.05 + 0.08 * r[t] ** 2 + 0.9 * s2
    return r


def test_var_backtest_matches_manual_kupiec():
    r = _garch_returns(1800, 4)
    d = pd.DataFrame(dict(t=range(len(r)), r=r))
    out = o.run("var_es", _csv(d), dict(y="r", time="t", window=500, alpha=0.05))
    row = _tab(out, "Out-of-sample")["rows"][0]
    m = len(r) - 500
    H = np.lib.stride_tricks.sliding_window_view(r, 500)[:-1][:m]
    v = int((r[500:] < np.quantile(H, 0.05, axis=1)).sum())
    assert row[1] == v
    x = v; p = 0.05
    lr = -2 * ((m - x) * np.log(1 - p) + x * np.log(p) - (m - x) * np.log(1 - x / m) - x * np.log(x / m))
    assert abs(_tab(out, "Backtests")["rows"][0][1] - lr) < 5e-3


def test_var_garch_calibrated_on_garch_data_and_normal_fails_in_tails():
    ok, bad = 0, 0
    for s in range(6):
        r = _garch_returns(2500, 10 + s)
        out = o.run("var_es", _csv(pd.DataFrame(dict(t=range(len(r)), r=r))), dict(y="r", time="t", window=600, alpha=0.01))
        rows = {x[0]: x for x in _tab(out, "Out-of-sample")["rows"]}
        ok += abs(rows["GARCH(1,1), t errors"][2] - 1.0) < 0.9
        bad += rows["Normal (rolling window)"][2] > rows["GARCH(1,1), t errors"][2]
    assert ok >= 5 and bad >= 5


def test_var_iid_returns_historical_close_to_nominal():
    rng = np.random.default_rng(5)
    r = rng.standard_t(5, 4000)
    out = o.run("var_es", _csv(pd.DataFrame(dict(t=range(4000), r=r))), dict(y="r", time="t", window=1000, alpha=0.05))
    rows = {x[0]: x for x in _tab(out, "Out-of-sample")["rows"]}
    assert abs(rows["Historical simulation"][2] - 5.0) < 1.0
    assert 0.8 < rows["Historical simulation"][7] < 1.3                   # realised / forecast ES near one


def test_var_errors():
    d = pd.DataFrame(dict(t=range(300), r=np.random.default_rng(6).normal(size=300)))
    assert "Too few" in o.run("var_es", _csv(d), dict(y="r", time="t", window=500))["error"]
    assert "tail probability" in o.run("var_es", _csv(d), dict(y="r", time="t", window=100, alpha=0.5))["error"]


# ------------------------------------------------------------------ DCC
def _sim_dcc(n, a, b, seed):
    rng = np.random.default_rng(seed)
    S = np.array([[1, 0.4], [0.4, 1.0]]); Q = S.copy(); r = np.empty((n, 2)); h = np.ones(2)
    z_prev = np.zeros(2)
    for t in range(n):
        Q = (1 - a - b) * S + a * np.outer(z_prev, z_prev) + b * Q
        d = 1 / np.sqrt(np.diag(Q)); R = Q * np.outer(d, d)
        z = np.linalg.cholesky(R) @ rng.normal(size=2)
        r[t] = np.sqrt(h) * z
        h = 0.05 + 0.08 * r[t] ** 2 + 0.88 * h
        z_prev = z
    return pd.DataFrame(dict(t=range(n), x=r[:, 0], y=r[:, 1]))


def test_dcc_recovers_parameters():
    est = []
    for s in range(5):
        r = o.run("dcc", _csv(_sim_dcc(3000, 0.05, 0.9, 20 + s)), dict(y=["x", "y"], time="t"))
        est.append([_tab(r, "Step 2")["rows"][0][1], _tab(r, "Step 2")["rows"][1][1]])
    est = np.mean(est, axis=0)
    assert abs(est[0] - 0.05) < 0.03 and abs(est[1] - 0.9) < 0.06


def test_dcc_constant_correlation_has_small_persistence_signal_and_lr_not_extreme():
    rng = np.random.default_rng(7)
    C = np.linalg.cholesky([[1, 0.5], [0.5, 1]])
    z = rng.normal(size=(2500, 2)) @ C.T
    d = pd.DataFrame(dict(t=range(2500), x=z[:, 0], y=z[:, 1]))
    r = o.run("dcc", _csv(d), dict(y=["x", "y"], time="t"))
    pair = _tab(r, "Conditional")["rows"][0]
    assert abs(pair[2] - 0.5) < 0.06 and pair[4] - pair[3] < 0.35


def test_dcc_real_data_and_univariate_ok():
    d = pd.read_csv(os.path.join(EX, "crspday.csv"))
    r = o.run("dcc", _csv(d), dict(y=["ibm", "ge", "mobil"], time="date"))
    u = _tab(r, "Step 1")["rows"]
    assert all(0.9 < x[5] < 1.0 for x in u)
    b = _tab(r, "Step 2")["rows"]
    assert 0.0 < b[0][1] < 0.05 and 0.95 < b[2][1] < 1.0
    pr = _tab(r, "Conditional")["rows"]
    assert all(-1 < x[3] and x[4] < 1 for x in pr)


def test_dcc_errors():
    d = pd.DataFrame(dict(t=range(100), x=np.random.default_rng(8).normal(size=100), y=np.random.default_rng(9).normal(size=100)))
    assert "at least 300" in o.run("dcc", _csv(d), dict(y=["x", "y"], time="t"))["error"]
    assert "two to five" in o.run("dcc", _csv(d), dict(y=["x"], time="t"))["error"]


# ================================================================== portfolio optimisation
EUS = os.path.join(ROOT, "oes", "examples", "eustock.csv")
SHI = os.path.join(ROOT, "oes", "examples", "shiller.csv")
IRA = os.path.join(ROOT, "oes", "examples", "irates.csv")


def _rets(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    C = np.array([[1, .6, .5, .3], [.6, 1, .4, .2], [.5, .4, 1, .3], [.3, .2, .3, 1.0]]) * 1e-4
    mu = np.array([.0003, .0004, .0002, .0005])
    return pd.DataFrame(rng.multivariate_normal(mu, C, n), columns=list("abcd")).assign(t=range(n))


def test_portfolio_closed_forms_unconstrained():
    d = _rets()
    r = o.run("portfolio", _csv(d), dict(assets=list("abcd"), time="t", long_only="no", shrink="none", boot=0, window=250))
    rows = {x[0]: x for x in _tab(r, "Portfolio weights")["rows"]}
    R = d[list("abcd")].to_numpy(); S = np.cov(R.T); mu = R.mean(0); one = np.ones(4)
    wmv = np.linalg.solve(S, one); wmv /= wmv.sum()
    wms = np.linalg.solve(S, mu); wms /= wms.sum()
    assert np.allclose(rows["Minimum variance"][1:5], wmv, atol=1e-5)
    assert np.allclose(rows["Maximum Sharpe ratio"][1:5], wms, atol=1e-5)
    assert abs(rows["Minimum variance"][6] - np.sqrt(wmv @ S @ wmv) * np.sqrt(252)) < 1e-5


def test_portfolio_risk_parity_equalises_risk_contributions_and_long_only():
    d = _rets()
    r = o.run("portfolio", _csv(d), dict(assets=list("abcd"), time="t", boot=0, shrink="none"))
    rows = {x[0]: x for x in _tab(r, "Portfolio weights")["rows"]}
    assert abs(rows["Risk parity"][-1] - 0.25) < 0.002                      # largest risk share equals the equal share
    for x in rows.values():
        assert min(x[1:5]) > -1e-6 and abs(sum(x[1:5]) - 1) < 1e-6


def test_portfolio_out_of_sample_equal_weight_matches_manual():
    d = _rets(900, 2)
    r = o.run("portfolio", _csv(d), dict(assets=list("abcd"), time="t", boot=0, window=250, rebalance=21))
    oos = {x[0]: x for x in _tab(r, "Out-of-sample")["rows"]}
    ew = d[list("abcd")].to_numpy()[250:].mean(1)
    assert abs(oos["Equal weights"][1] - ew.mean() * 252) < 1e-6 and abs(oos["Equal weights"][2] - ew.std(ddof=1) * np.sqrt(252)) < 1e-6
    assert oos["Equal weights"][4] == 0.0


def test_portfolio_bootstrap_interval_contains_weight_and_estimation_error_visible():
    d = _rets(1000, 3)
    r = o.run("portfolio", _csv(d), dict(assets=list("abcd"), time="t", boot=200, objective="minvar", shrink="none", long_only="no"))
    for x in _tab(r, "Sampling")["rows"]:
        assert x[2] <= x[1] <= x[3]
    rows = {x[0]: x for x in _tab(r, "Portfolio weights")["rows"]}
    oos = {x[0]: x for x in _tab(r, "Out-of-sample")["rows"]}
    assert rows["Maximum Sharpe ratio"][-3] > oos["Maximum Sharpe ratio"][3] - 1.0


def test_portfolio_real_data_and_errors():
    e = pd.read_csv(EUS)
    r = o.run("portfolio", _csv(e), dict(assets=["r_dax", "r_smi", "r_cac", "r_ftse"], time="t", boot=50))
    assert len(_tab(r, "Out-of-sample")["rows"]) == 4
    assert "at least three" in o.run("portfolio", _csv(e), dict(assets=["r_dax", "r_smi"], time="t"))["error"]
    assert "Too few" in o.run("portfolio", _csv(e.iloc[:200]), dict(assets=["r_dax", "r_smi", "r_cac"], time="t"))["error"]


# ================================================================== volatility forecast evaluation
def test_vol_eval_garch_calibrated_on_garch_data():
    slopes, qbest = [], 0
    for s in range(6):
        r_ = _garch_returns(3000, 30 + s)
        out = o.run("vol_eval", _csv(pd.DataFrame(dict(t=range(len(r_)), r=r_))), dict(y="r", time="t", window=600))
        mz = {x[0]: x for x in _tab(out, "Mincer")["rows"]}
        slopes.append(mz["GARCH(1,1)"][3])
        loss = {x[0]: x[2] for x in _tab(out, "Out-of-sample loss")["rows"]}
        qbest += loss["GARCH(1,1)"] < loss["Constant (first window)"]
    assert 0.5 < np.median(slopes) < 1.5 and qbest == 6          # the slope on a noisy squared-return proxy is variable: use the median


def test_vol_eval_qlike_and_mse_match_manual():
    r_ = _garch_returns(1500, 40)
    d = pd.DataFrame(dict(t=range(len(r_)), r=r_))
    out = o.run("vol_eval", _csv(d), dict(y="r", time="t", window=500, **{"lambda": 0.94}))
    w = 500
    h = np.full(len(r_) - w, r_[:w].var(ddof=1))
    proxy = (r_[w:] - r_[:w].mean()) ** 2
    row = {x[0]: x for x in _tab(out, "Out-of-sample loss")["rows"]}["Constant (first window)"]
    assert abs(row[1] / (np.mean((proxy - h) ** 2) * 1e8) - 1) < 1e-6 and abs(row[2] - np.mean(np.log(h) + proxy / h)) < 1e-6


def test_vol_eval_dm_prefers_better_forecast_and_symmetry():
    r_ = _garch_returns(3000, 41)
    d = pd.DataFrame(dict(t=range(len(r_)), r=r_))
    a = o.run("vol_eval", _csv(d), dict(y="r", time="t", window=600, benchmark="EWMA"))
    row = {x[0]: x for x in _tab(a, "Diebold")["rows"]}["Constant (first window)"]
    assert row[1] > 0 and row[2] < 0.05                                      # the constant forecast loses to the EWMA on QLIKE
    b = o.run("vol_eval", _csv(d), dict(y="r", time="t", window=600, benchmark="GARCH(1,1)"))
    assert "Diebold-Mariano tests against GARCH(1,1)" in _tab(b, "Diebold")["title"]


def test_vol_eval_errors():
    d = pd.DataFrame(dict(t=range(300), r=np.random.default_rng(3).normal(size=300)))
    assert "Too few" in o.run("vol_eval", _csv(d), dict(y="r", time="t", window=500))["error"]


# ================================================================== bubble tests
def test_bubble_sadf_window_statistic_matches_statsmodels_adf():
    from statsmodels.tsa.stattools import adfuller
    rng = np.random.default_rng(5)
    y = np.cumsum(rng.normal(size=200))
    G = finance._adf_grid(y, 100)
    for j in (150, 199):
        ref = adfuller(y[:j + 1], maxlag=0, regression="c", autolag=None)[0]
        assert abs(G[0, j] - ref) < 1e-8
    # a window that does not start at zero
    ref = adfuller(y[40:171], maxlag=0, regression="c", autolag=None)[0]
    assert abs(G[40, 170] - ref) < 1e-8


def test_bubble_detects_simulated_explosive_episode_and_stamps_it():
    rng = np.random.default_rng(6)
    n = 400
    y = np.empty(n); y[0] = 100
    for t in range(1, n):
        y[t] = y[t - 1] * (1.04 if 200 <= t < 250 else 1.0) + rng.normal(0, 1.0) if t < 250 else (y[t - 1] - 0.35 * (y[t - 1] - 100) + rng.normal(0, 1.0) if t < 270 else y[t - 1] + rng.normal(0, 1.0))
    out = o.run("bubble", _csv(pd.DataFrame(dict(t=range(n), y=y))), dict(y="y", time="t", reps=60, seed=1))
    g = _tab(out, "Tests")["rows"][1]
    assert g[1] > g[3] and g[5] < 0.05
    eps = _tab(out, "Date-stamped")["rows"]
    assert any(abs(int(e[0]) - 200) <= 25 for e in eps if e[0] != "none")


def test_bubble_random_walk_rejection_rate_not_inflated():
    rej = 0
    for s in range(30):
        rng = np.random.default_rng(100 + s)
        y = 100 + np.cumsum(rng.normal(size=150))
        out = o.run("bubble", _csv(pd.DataFrame(dict(t=range(150), y=y))), dict(y="y", time="t", reps=60, seed=s))
        rej += _tab(out, "Tests")["rows"][1][5] < 0.05
    assert rej <= 7                                                           # 5% nominal, 30 samples, 60 replications: generous bound


def test_bubble_real_data_finds_1929_and_1990s():
    d = pd.read_csv(SHI)
    out = o.run("bubble", _csv(d), dict(y="pd_ratio", time="date", reps=40, seed=0))
    eps = _tab(out, "Date-stamped")["rows"]
    starts = [e[0][:4] for e in eps]
    assert any(s in ("1928", "1929") for s in starts) and any(1994 <= int(s) <= 1998 for s in starts)


def test_bubble_errors():
    d = pd.DataFrame(dict(t=range(30), y=np.arange(30.0)))
    assert "at least 60" in o.run("bubble", _csv(d), dict(y="y", time="t"))["error"]


# ================================================================== Nelson-Siegel
def test_nelson_siegel_recovers_exact_curve_and_lambda():
    tau = np.array([3, 6, 12, 24, 36, 60, 84, 120.0]); lam = 0.0609
    rng = np.random.default_rng(7)
    T = 120
    B = np.column_stack([5 + np.cumsum(rng.normal(0, .05, T)), -1 + np.cumsum(rng.normal(0, .05, T)), np.cumsum(rng.normal(0, .05, T))])
    L = finance._ns_loadings(tau, lam)
    Y = B @ L.T
    d = pd.DataFrame(Y, columns=[f"y{int(t)}" for t in tau]); d["t"] = range(T)
    out = o.run("nelson_siegel", _csv(d), dict(yields=list(d.columns[:-1]), maturities=",".join(str(int(t)) for t in tau), time="t"))
    fac = {x[0]: x for x in _tab(out, "Factors")["rows"]}
    assert abs(fac["Level (beta 0)"][1] - B[:, 0].mean()) < 5e-3 and abs(fac["Curvature (beta 2)"][1] - B[:, 2].mean()) < 5e-3
    assert abs(next(s for s in out["summary"] if s["label"] == "lambda")["value"] - lam) < 0.003
    assert max(x[2] for x in _tab(out, "Fit by")["rows"]) < 1e-3


def test_nelson_siegel_matches_scipy_least_squares_for_fixed_lambda():
    d = pd.read_csv(IRA)
    cols = ["r1", "r3", "r6", "r12", "r36", "r60", "r120"]; tau = np.array([1, 3, 6, 12, 36, 60, 120.0])
    out = o.run("nelson_siegel", _csv(d), dict(yields=cols, maturities="1,3,6,12,36,60,120", time="month", **{"lambda": 0.0609}))
    L = finance._ns_loadings(tau, 0.0609)
    i = 200
    ref = optimize_lsq(L, d[cols].to_numpy()[i])
    assert np.allclose(ref, np.linalg.lstsq(L, d[cols].to_numpy()[i], rcond=None)[0])
    assert out["nobs"] == len(d) and _tab(out, "Factors")["rows"][0][1] > 5


def optimize_lsq(L, y):
    from scipy.optimize import least_squares
    return least_squares(lambda b: L @ b - y, np.zeros(3), xtol=1e-14, ftol=1e-14).x


def test_nelson_siegel_forecast_table_and_errors():
    d = pd.read_csv(IRA)
    out = o.run("nelson_siegel", _csv(d), dict(yields=["r1", "r6", "r12", "r36", "r120"], maturities="1,6,12,36,120", time="month", horizon=6))
    assert len(_tab(out, "6-month-ahead")["rows"]) == 3
    assert "maturities" in o.run("nelson_siegel", _csv(d), dict(yields=["r1", "r6", "r12", "r36"], maturities="1,6,12", time="month"))["error"]
    assert "four maturities" in o.run("nelson_siegel", _csv(d), dict(yields=["r1", "r6", "r12"], maturities="1,6,12", time="month"))["error"]
    assert "lambda must" in o.run("nelson_siegel", _csv(d), dict(yields=["r1", "r6", "r12", "r36"], maturities="1,6,12,36", time="month", **{"lambda": -1}))["error"]
