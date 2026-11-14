#!/usr/bin/env python3
"""Build time-series-guide/index.html: prepares the real example datasets, runs the analysis on each, and fills the template."""
import json, pathlib, sys
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from analysis import analyze

OUT = ROOT / "time-series-guide"
(OUT / "examples").mkdir(parents=True, exist_ok=True)
m = pd.read_csv(ROOT / "teaching/monetary-economics/data/us_monthly.csv")
m["date"] = m.year.astype(str) + "-" + m.month.astype(str).str.zfill(2)
q = pd.read_csv(ROOT / "teaching/monetary-economics/data/us_quarterly.csv")
q["date"] = q.t.apply(lambda t: f"{int(t)}Q{int(round((t % 1) * 4)) + 1}")
M = m[(m.year >= 1984) & (m.year <= 2019)].copy()
Q = q[(q.t >= 1959.75) & (q.t < 2020)].copy()

ex = {}
a = pd.DataFrame({"date": M.date, "sp500_return": 100 * np.log(M.sp500).diff(), "ip_growth": 100 * np.log(M.indpro).diff(),
                  "d_fedfunds": M.ffr.diff(), "d_credit_spread": (M.baa - M.gs10).diff()})[M.year >= 1985]
ex["returns"] = (a, "Stock returns and the economy", "Monthly, 1985–2019 · U.S.",
                 "S&P 500 return, industrial production growth, change in the federal funds rate, change in the Baa–Treasury credit spread.")
b = pd.DataFrame({"date": M.date, "log_sp500": np.log(M.sp500), "log_indpro": np.log(M.indpro)})[M.year >= 1985]
ex["prices"] = (b, "Stock prices and industrial production", "Monthly, 1985–2019 · U.S.",
                "Log S&P 500 index and log industrial production.")
c = pd.DataFrame({"date": Q.date, "yield_10y": Q.gs10, "yield_1y": Q.gs1})[Q.t >= 1960]
ex["rates"] = (c, "Short and long interest rates", "Quarterly, 1960–2019 · U.S.", "10-year and 1-year Treasury yields.")
d = pd.DataFrame({"date": Q.date, "yield_10y": Q.gs10, "inflation": 400 * np.log(Q.gdpdef).diff(),
                  "gdp_growth": 400 * np.log(Q.rgdp).diff(), "unemployment": Q.unrate})[Q.t >= 1960]
ex["yield"] = (d, "What moves the 10-year yield?", "Quarterly, 1960–2019 · U.S.",
               "10-year Treasury yield, GDP-deflator inflation, real GDP growth and the unemployment rate.")

e = pd.DataFrame({"date": Q.date, "log_price": np.log(Q.gdpdef), "log_m2": np.log(Q.m2), "log_gdp": np.log(Q.rgdp)})[Q.t >= 1960]
ex["money"] = (e, "Money, prices and output", "Quarterly, 1960–2019 · U.S.",
               "Log GDP deflator, log M2 money stock and log real GDP.")

# extra data file used by the NARDL guide (not one of the selector examples)
M2 = m[(m.year >= 1970) & (m.year <= 2019)]
pd.DataFrame({"date": M2.date, "baa_yield": M2.baa, "fedfunds": M2.ffr}).to_csv(OUT / "examples" / "passthrough.csv", index=False)

# data for the proxy-SVAR section of the SVAR guide: policy rates, output, prices, credit spread and a high-frequency monetary shock (Jarocinski-Karadi)
_s = pd.read_csv(ROOT / "teaching/monetary-economics/data/mp_shocks_monthly.csv")
_p = m.merge(_s, on=["year", "month"])
_p = _p[(_p.year >= 1990) & (_p.year <= 2019)]
pd.DataFrame({"date": _p.date, "ffr": _p.ffr, "gs1": _p.gs1, "log_ip": 100 * np.log(_p.indpro), "log_cpi": 100 * np.log(_p.cpi),
              "credit_spread": _p.baa - _p.gs10, "mp_shock": _p.jk_mp}).round(5).to_csv(OUT / "examples" / "monetary_proxy.csv", index=False)

# data for the 2SLS / GMM guide: federal funds rate, 4-quarter GDP-deflator inflation and the HP-filtered output gap
from statsmodels.tsa.filters.hp_filter import hpfilter
T = q[(q.t >= 1959) & (q.t < 2020)].copy()
T["inflation"] = 100 * (np.log(T.gdpdef) - np.log(T.gdpdef.shift(4)))
T["output_gap"] = hpfilter(100 * np.log(T.rgdp), 1600)[0]
T[T.t >= 1960][["date", "ffr", "inflation", "output_gap"]].rename(columns={"ffr": "fedfunds"}).round(4).to_csv(OUT / "examples" / "taylor.csv", index=False)

examples = []
for key, (df, title, meta, desc) in ex.items():
    df = df.round(5); path = OUT / "examples" / f"{key}.csv"; df.to_csv(path, index=False)
    r = json.loads(analyze(path.read_text()))
    print(f"{key}: n={r['n']} orders={r['orders']} model={r['model']}", r.get("johansen", ""))
    examples.append(dict(key=key, title=title, meta=meta, desc=desc, file=f"examples/{key}.csv", result=r))

tpl = (HERE / "template.html").read_text()
guides = [p.stem for p in (HERE / "guides").glob("*.md")] + ["lp", "svar", "bp", "tar", "str", "arima", "garch"]
page = tpl.replace("/*GUIDES*/[]", json.dumps(guides)).replace("/*EXAMPLES*/", json.dumps(examples, ensure_ascii=False)).replace("#PYSRC#", (HERE / "analysis.py").read_text())
(OUT / "index.html").write_text(page)
print("wrote time-series-guide/index.html", len(page) // 1024, "KB")

import subprocess as _sp, sys as _sys, pathlib as _pl
_sp.run([_sys.executable, str(_pl.Path(__file__).resolve().parents[2] / "tools" / "og_tags.py")], check=True)  # complete preview tags
