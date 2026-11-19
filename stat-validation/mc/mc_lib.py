"""Monte Carlo harness for the OES Statistics Division (OMVP stages 3-5).
Every experiment is a Scenario: a reproducible data-generating process, an extractor that reads numbers from the
result dict returned by oes_engine.run, and the truth being estimated (or the null/alternative status of a test).
Replication r of scenario s at size n uses numpy.random.default_rng([method_code, scenario_index, n, r]), so any single
replication can be re-run in isolation. Failures (an 'error' result or an exception) are counted, not hidden."""
import json, math, os, sys, time, warnings
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "oes", "engine")); sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import oes_engine as o

NGRID = (50, 100, 250, 500, 1000)

def S(r, label):
    return next(s["value"] for s in r["summary"] if s["label"] == label)

def cell(r, title, row, col):
    """Value in the table whose title starts with `title`, in the row whose first cell equals `row`, column `col`."""
    t = next(t for t in r["tables"] if t["title"].startswith(title))
    j = t["columns"].index(col)
    for rw in t["rows"]:
        if str(rw[0]) == str(row): return rw[j]
    raise KeyError(row)

class Scenario:
    def __init__(self, name, model, gen, extract, truth=None, h0=None, note="", ns=NGRID, reps=2000, alpha=0.05, kind=None):
        self.name, self.model, self.gen, self.extract = name, model, gen, extract
        self.truth, self.h0, self.note, self.ns, self.reps, self.alpha = truth, h0, note, ns, reps, alpha
        self.kind = kind  # 'estimator' | 'test' | 'custom'

def one(model, df, spec):
    t = time.perf_counter()
    try:
        r = o.run(model, df, dict(spec))
    except Exception as e:  # engine promises not to raise; record it if it does
        return None, time.perf_counter() - t, f"exception {type(e).__name__}"
    dt = time.perf_counter() - t
    if "error" in r: return None, dt, r["error"][:80]
    return r, dt, None

def run_scenario(sc, sidx, mcode, n, reps, budget=None):
    rows, fails, times, why = [], 0, [], {}
    t0 = time.perf_counter()
    done = 0
    for rep in range(reps):
        rng = np.random.default_rng([mcode, sidx, n, rep])
        df, spec = sc.gen(n, rng)
        r, dt, err = one(sc.model, df, spec); times.append(dt); done += 1
        if r is None:
            fails += 1; why[err] = why.get(err, 0) + 1
        else:
            try:
                v = sc.extract(r)
                if any(x is None or (isinstance(x, float) and not math.isfinite(x)) for x in v.values() if not isinstance(x, str)):
                    fails += 1; why["non-finite extract"] = why.get("non-finite extract", 0) + 1
                else: rows.append(v)
            except Exception as e:
                fails += 1; why[f"extract {type(e).__name__}"] = why.get(f"extract {type(e).__name__}", 0) + 1
        if budget and time.perf_counter() - t0 > budget and done >= 100: break
    return summarise(sc, rows, fails, done, times, why, n)

def summarise(sc, rows, fails, done, times, why, n):
    out = dict(n=n, reps_requested=sc.reps, reps_run=done, failures=fails, convergence=round(1 - fails / max(done, 1), 4),
               sec_per_rep=round(float(np.mean(times)), 5) if times else None, failure_reasons=why)
    if not rows: return out
    df = pd.DataFrame(rows); m = len(df); out["reps_valid"] = m
    if "est" in df and sc.truth is not None:
        e = df["est"].to_numpy(float); tr = sc.truth(n) if callable(sc.truth) else sc.truth
        out.update(truth=float(tr), mean_est=float(e.mean()), bias=float(e.mean() - tr), emp_sd=float(e.std(ddof=1)),
                   rmse=float(np.sqrt(np.mean((e - tr) ** 2))), mae=float(np.mean(np.abs(e - tr))),
                   mc_se_bias=float(e.std(ddof=1) / np.sqrt(m)))
        if "se" in df: out["mean_se"] = float(df["se"].mean()); out["se_ratio"] = float(df["se"].mean() / e.std(ddof=1))
    if "lo" in df and "hi" in df and sc.truth is not None:
        tr = sc.truth(n) if callable(sc.truth) else sc.truth; m = len(df)
        if True:
            c = float(np.mean((df["lo"] <= tr) & (tr <= df["hi"]))); out["coverage"] = c
            out["mc_se_coverage"] = float(math.sqrt(c * (1 - c) / m)); out["mean_ci_width"] = float((df["hi"] - df["lo"]).mean())
    for c in [c for c in df if c.startswith("p")]:
        rej = float(np.mean(df[c] < sc.alpha)); out[f"reject_{c}"] = rej; out[f"mc_se_reject_{c}"] = float(math.sqrt(rej * (1 - rej) / m))
    for c in [c for c in df if c.startswith("flag")]:
        out[f"rate_{c}"] = float(df[c].mean())
    return out

def run_method(mcode, scenarios, reps=None, budget=None, ns=None, log=print):
    res = []
    for si, sc in enumerate(scenarios):
        entry = dict(scenario=sc.name, model=sc.model, note=sc.note, h0_true=sc.h0, results=[])
        for n in (ns or sc.ns):
            r = run_scenario(sc, si, mcode, n, reps or sc.reps, budget)
            entry["results"].append(r); log(f"  {sc.name} n={n}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("reps_run", "failures", "bias", "emp_sd", "rmse", "coverage") or k.startswith("reject")}))
        res.append(entry)
    return res
