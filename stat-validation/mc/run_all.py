"""Run the Monte Carlo experiments.
  python stat-validation/mc/run_all.py [--ids stat_cox,stat_lmm] [--reps 2000] [--budget 25] [--out stat-validation/results]
--budget caps the wall-clock seconds spent on one (scenario, n) cell once at least 100 replications are done; the number of
replications actually run is stored with every result, so a capped cell is visible as such. Results: results/mc_<id>.json."""
import argparse, json, os, sys, time, platform
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import mc_lib as M, experiments as X
from scipy import stats

def power_check(reps=20000, out=None):
    """F3: the analytic power returned by the engine against the empirical rejection rate of the actual test in simulation."""
    cases = [("t_two", dict(effect=0.5), lambda n, rng: stats.ttest_ind(rng.normal(0.5, 1, n), rng.normal(0, 1, n)).pvalue),
             ("t_one", dict(effect=0.4), lambda n, rng: stats.ttest_1samp(rng.normal(0.4, 1, n), 0).pvalue),
             ("corr", dict(effect=0.3), None), ("anova", dict(effect=0.25, k=3), None), ("prop", dict(p1=0.5, p2=0.7), None)]
    res = []
    for test, ex, f in cases:
        for n in (20, 50, 100, 250):
            rng = np.random.default_rng([99, n, abs(hash(test)) % 1000])
            if test == "corr":
                r_ = 0.3; x = rng.standard_normal((reps, n)); y = r_ * x + np.sqrt(1 - r_ ** 2) * rng.standard_normal((reps, n))
                xc = x - x.mean(1, keepdims=True); yc = y - y.mean(1, keepdims=True)
                r = (xc * yc).sum(1) / np.sqrt((xc ** 2).sum(1) * (yc ** 2).sum(1)); t = r * np.sqrt((n - 2) / (1 - r ** 2)); emp = np.mean(2 * stats.t.sf(abs(t), n - 2) < 0.05)
            elif test == "anova":
                g = 3; mu = np.array([-0.25 * np.sqrt(1.5), 0, 0.25 * np.sqrt(1.5)])   # Cohen's f = 0.25 with unit error SD
                Y = rng.standard_normal((reps, g, n)) + mu[None, :, None]
                gm = Y.mean(2); grand = Y.mean((1, 2)); ssb = n * ((gm - grand[:, None]) ** 2).sum(1); ssw = ((Y - gm[:, :, None]) ** 2).sum((1, 2))
                F = (ssb / (g - 1)) / (ssw / (g * (n - 1))); emp = np.mean(stats.f.sf(F, g - 1, g * (n - 1)) < 0.05)
            elif test == "prop":
                a = rng.binomial(n, 0.5, reps); b = rng.binomial(n, 0.7, reps)
                z = (2 * np.arcsin(np.sqrt(a / n)) - 2 * np.arcsin(np.sqrt(b / n))) / np.sqrt(2 / n); emp = np.mean(2 * stats.norm.sf(abs(z)) < 0.05)
            elif test == "t_two":
                a = rng.normal(0.5, 1, (reps, n)); b = rng.normal(0, 1, (reps, n))
                sp = np.sqrt((a.var(1, ddof=1) + b.var(1, ddof=1)) / 2); t = (a.mean(1) - b.mean(1)) / (sp * np.sqrt(2 / n)); emp = np.mean(2 * stats.t.sf(abs(t), 2 * n - 2) < 0.05)
            else:
                a = rng.normal(0.4, 1, (reps, n)); t = a.mean(1) / (a.std(1, ddof=1) / np.sqrt(n)); emp = np.mean(2 * stats.t.sf(abs(t), n - 1) < 0.05)
            r = M.o.run("stat_power", "", dict(test=test, solve_for="power", n=n, alpha=0.05, **ex)); eng = M.S(r, "Power")
            se = float(np.sqrt(emp * (1 - emp) / reps))
            res.append(dict(test=test, n=n, engine_power=eng, empirical_power=float(emp), diff=float(eng - emp), mc_se=se))
            print("  power", test, n, round(eng, 4), round(emp, 4), flush=True)
    return res

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ids", default=""); ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--budget", type=float, default=25.0); ap.add_argument("--out", default=os.path.join(M.ROOT, "stat-validation", "results"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ids = [i for i in a.ids.split(",") if i] or list(X.EXPERIMENTS) + ["stat_power"]
    import numpy, scipy, pandas
    env = dict(python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__, pandas=pandas.__version__, engine=M.o.__version__)
    for mid in ids:
        t = time.time(); print(mid, flush=True)
        if mid == "stat_power": res = dict(model=mid, power_vs_simulation=power_check())
        else:
            code = list(X.EXPERIMENTS).index(mid) + 1
            res = dict(model=mid, scenarios=M.run_method(code, X.EXPERIMENTS[mid], a.reps, a.budget, log=lambda s: print(s, flush=True)))
        res.update(env=env, wall_seconds=round(time.time() - t, 1), budget_seconds_per_cell=a.budget)
        json.dump(res, open(os.path.join(a.out, f"mc_{mid}.json"), "w"), indent=1)
main()
