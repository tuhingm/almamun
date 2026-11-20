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
