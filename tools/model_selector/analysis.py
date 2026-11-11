"""Model-selection analysis for the Empirical Research Guide.

analyze(csv_text) reads a CSV (one column per variable, optional date column first), then reports summary statistics,
correlations, unit-root tests (ADF on levels and first differences, KPSS on levels), and, when every series is I(1),
the Johansen trace test. The first numeric column is treated as the dependent variable. Returns a JSON string.
The same file runs in the browser (Pyodide) for uploaded data and offline to prepare the built-in examples.
"""
import io, json, warnings
import numpy as np
import pandas as pd


def _f(x, nd=4):
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


def analyze(csv_text, max_vars=8):
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    warnings.simplefilter("ignore")
    try:
        raw = pd.read_csv(io.StringIO(csv_text), sep=None, engine="python")
    except Exception as e:
        return json.dumps({"error": f"Could not read the file as CSV ({e})."})
    num = raw.apply(pd.to_numeric, errors="coerce")
    keep = [c for c in num.columns if num[c].notna().mean() > 0.9 and num[c].nunique() > 2]
    # drop an index-like first column (year, t, obs) if it increases by a constant step
    if keep and len(keep) > 2:
        d0 = num[keep[0]].dropna().diff().dropna()
        if len(d0) and d0.nunique() <= 2 and str(keep[0]).strip().lower() in ("year", "t", "time", "obs", "date", "month", "quarter", "index", "unnamed: 0"):
            keep = keep[1:]
    skipped = [str(c) for c in raw.columns if c not in keep]
    df = num[keep[:max_vars]].dropna().reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    n, k = df.shape
    if k < 2:
        return json.dumps({"error": "The file needs at least two numeric columns (one per variable)."})
    if n < 30:
        return json.dumps({"error": f"Only {n} complete rows were found. These tests need at least 30 observations."})

    out = {"n": int(n), "vars": list(df.columns), "skipped": skipped, "dep": df.columns[0]}
    step = max(1, n // 120)
    out["series"] = {c: [_f(v, 5) for v in df[c].values[::step]] for c in df}
    out["dseries"] = {c: [_f(v, 5) for v in df[c].diff().dropna().values[::step]] for c in df}
    out["summary"] = [dict(var=c, mean=_f(df[c].mean()), sd=_f(df[c].std()), min=_f(df[c].min()), max=_f(df[c].max()),
                           skew=_f(df[c].skew(), 3), kurt=_f(df[c].kurt() + 3, 3)) for c in df]
    out["corr"] = [[_f(v, 3) for v in row] for row in df.corr().values]

    ur, orders = [], []
    for c in df:
        x = df[c]
        a0 = adfuller(x, autolag="AIC")
        a1 = adfuller(x.diff().dropna(), autolag="AIC")
        try:
            ks, kp = kpss(x, regression="c", nlags="auto")[:2]
        except Exception:
            ks, kp = np.nan, np.nan
        d2p = d2s = None
        if a0[1] >= 0.05 and a1[1] >= 0.05:          # still a unit root after one difference: test the second difference
            a2 = adfuller(x.diff().diff().dropna(), autolag="AIC")
            d2s, d2p = a2[0], a2[1]
        order = 0 if a0[1] < 0.05 else (1 if a1[1] < 0.05 else (2 if d2p is not None and d2p < 0.05 else 3))
        conflict = bool(order == 0 and np.isfinite(kp) and kp < 0.05)
        orders.append(order)
        ur.append(dict(var=c, adf_stat=_f(a0[0], 3), adf_p=_f(a0[1], 3), adf_lags=int(a0[2]), dadf_stat=_f(a1[0], 3),
                       dadf_p=_f(a1[1], 3), d2adf_stat=None if d2s is None else _f(d2s, 3), d2adf_p=None if d2p is None else _f(d2p, 3), kpss_stat=_f(ks, 3), kpss_p=_f(kp, 3), order=order, conflict=conflict))
    out["unitroot"] = ur
    out["conflicts"] = [u["var"] for u in ur if u["conflict"]]

    if max(orders) >= 2:
        out["model"] = "I2"
    elif all(o == 0 for o in orders):
        out["model"] = "OLS"
    elif all(o == 1 for o in orders):
        try:
            maxl = max(1, min(8, (n // (k + 1)) // 4))
            p = max(1, int(VAR(df).select_order(maxl).aic))
        except Exception:
            p = 2
        j = coint_johansen(df, 0, p - 1)
        rows, rank, stop = [], 0, False
        for i in range(k):
            rej = bool(j.lr1[i] > j.cvt[i, 1])
            rows.append(dict(h0=i, trace=_f(j.lr1[i], 2), cv=_f(j.cvt[i, 1], 2), reject=rej))
            if rej and not stop:
                rank = i + 1
            else:
                stop = True
        out["johansen"] = dict(lags=p, rows=rows, rank=rank)
        out["model"] = "VAR" if rank == 0 else "VECM"
        out["fullrank"] = bool(rank == k)
    else:
        out["model"] = "ARDL"
    out["orders"] = orders
    return json.dumps(out)
