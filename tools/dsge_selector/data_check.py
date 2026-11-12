"""Data preparation and checks for DSGE estimation in Dynare (runs in the browser with Pyodide, and offline).

check(csv_text, options_json) takes a CSV of time series and options:
    observables : list of column names to use as observables
    shocks      : number of structural shocks in the model (for the stochastic-singularity check)
    transforms  : optional dict column -> 'growth' (100 dlog) | 'diff' (100 d, for logs) | 'level'; when absent, the tool proposes one
and returns JSON with, for each observable: the proposed transformation, summary statistics, ADF and KPSS tests on
the transformed series, and flags; overall checks (sample length, missing values, observables vs shocks); and
ready-to-use Dynare pieces: a demeaned data file (CSV text), the varobs line and measurement-equation templates.
"""
import io, json, math, warnings
import numpy as np
import pandas as pd


def _f(x, nd=4):
    try:
        x = float(x)
    except Exception:
        return None
    return None if not np.isfinite(x) else round(x, nd)


def _propose(s, name=""):
    """'growth' (100*dlog) for positive trending levels, 'diff' (100*d) for trending log levels, 'level' otherwise."""
    x = s.dropna().to_numpy(float)
    if len(x) < 10:
        return "level"
    t = np.arange(len(x))
    r2 = np.corrcoef(t, x)[0, 1] ** 2
    if r2 < 0.8:
        return "level"
    looks_log = name.lower().startswith(("log", "ln", "l_")) or (np.abs(x).max() < 30 and x.max() - x.min() < 5)
    if looks_log:
        return "diff"
    return "growth" if (x > 0).all() else "diff"


def _transform(s, how):
    if how == "growth":
        return 100 * np.log(s).diff()
    if how == "diff":
        return 100 * s.diff()
    return s.astype(float)


def _tests(x):
    from statsmodels.tsa.stattools import adfuller, kpss
    x = x.dropna().to_numpy(float)
    out = {}
    warnings.simplefilter("ignore")
    try:
        out["adf_p"] = _f(adfuller(x, autolag="AIC", regression="c")[1])
    except Exception:
        out["adf_p"] = None
    try:
        out["kpss_p"] = _f(kpss(x, regression="c", nlags="auto")[1])
    except Exception:
        out["kpss_p"] = None
    return out


def check(csv_text, options_json):
    warnings.simplefilter("ignore")
    o = json.loads(options_json) if isinstance(options_json, str) else dict(options_json)
    try:
        df = pd.read_csv(io.StringIO(csv_text), sep=None, engine="python")
    except Exception as e:
        return json.dumps({"error": f"Could not read the file as CSV ({e})."})
    df.columns = [str(c).strip() for c in df.columns]
    obs = [c for c in (o.get("observables") or []) if c in df.columns]
    if not obs:
        return json.dumps({"error": "Choose at least one observable."})
    tr = o.get("transforms") or {}
    out = {"n_rows": int(len(df)), "series": [], "flags": []}
    data = pd.DataFrame(index=df.index)
    for c in obs:
        s = pd.to_numeric(df[c], errors="coerce")
        how = tr.get(c) or _propose(s, c)
        x = _transform(s, how)
        miss = int(s.isna().sum())
        t = _tests(x)
        st = dict(name=c, how=how, missing=miss, mean=_f(x.mean()), sd=_f(x.std()), min=_f(x.min()), max=_f(x.max()),
                  raw=[_f(v) for v in s.values], transformed=[_f(v) for v in x.values], **t)
        stat = (t["adf_p"] is not None and t["adf_p"] < 0.05) and (t["kpss_p"] is None or t["kpss_p"] >= 0.05)
        nonstat = (t["adf_p"] is not None and t["adf_p"] >= 0.05) and (t["kpss_p"] is not None and t["kpss_p"] < 0.05)
        st["verdict"] = "stationary" if stat else ("non-stationary" if nonstat else "unclear")
        out["series"].append(st)
        data[c + "_obs"] = x
    data = data.dropna()
    out["n_obs"] = int(len(data))
    k = int(o.get("shocks") or 0)
    out["shocks"] = k
    out["n_observables"] = len(obs)
    if k and len(obs) > k:
        out["flags"].append("singular")
    if len(data) < 60:
        out["flags"].append("short")
    if any(s["verdict"] == "non-stationary" for s in out["series"]):
        out["flags"].append("nonstationary")
    if any(s["missing"] > 1 for s in out["series"]):
        out["flags"].append("missing")
    dm = data - data.mean()
    out["means"] = {c: _f(v) for c, v in data.mean().items()}
    out["csv"] = dm.round(6).to_csv(index=False)
    out["varobs"] = "varobs " + " ".join(c + "_obs" for c in obs) + ";"
    meas = []
    for s in out["series"]:
        n = s["name"]
        m = n.lower()
        if s["how"] in ("growth", "diff"):
            meas.append(f"  {n}_obs = {m} - {m}(-1);    // growth: first difference of the model variable (log deviation, in percent)")
        else:
            meas.append(f"  {n}_obs = {m};             // series in levels, demeaned (use 4*{m} if the data are annualised and the model quarterly)")
    out["measurement"] = "\n".join(meas)
    out["estimation"] = ("estimation(datafile = 'dsge_observables.csv', mode_compute = 5, mh_replic = 20000, mh_nblocks = 2,\n"
                         "           mh_jscale = 0.3, mh_drop = 0.5, bayesian_irf);")
    out["model"] = "MED" if len(obs) >= 5 else "EST"
    return json.dumps(out)
