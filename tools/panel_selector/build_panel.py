#!/usr/bin/env python3
"""Build panel-data-guide/index.html: prepares the real example panels, runs the analysis on each, and fills the template.

The page shares its stylesheet with the Time-Series Guide (tools/model_selector/template.html), so the two look the same.
Needs: pip install numpy pandas scipy statsmodels wooldridge pydataset
Run simulate_cips.py first if cips_quantiles.json is missing (it takes about a minute).
"""
import json, pathlib, re, sys, warnings
import numpy as np, pandas as pd

warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "panel-data-guide"
sys.path.insert(0, str(HERE))
import panel_analysis as pa

Q = json.loads((HERE / "cips_quantiles.json").read_text())
pa.CIPS_Q = Q
src = (HERE / "panel_analysis.py").read_text().replace("CIPS_Q = {}", "CIPS_Q = " + json.dumps(Q, separators=(",", ":")), 1)

import wooldridge as wd
from pydataset import data as rdata

ex = {}
w = wd.data("wagepan")
ex["wagepan"] = (pd.DataFrame({"nr": w.nr, "year": w.year, "lwage": w.lwage, "union": w.union, "married": w.married,
                               "expersq": w.expersq, "lhours": np.log(w.hours)}),
                 "Union wage premium", "Workers × years · 545 men, 1980–1987",
                 "Log wage, union membership, marriage, experience squared and log hours worked (Wooldridge wagepan).")
c = wd.data("crime4")
ex["crime4"] = (c[["county", "year", "lcrmrte", "lprbarr", "lprbconv", "lprbpris", "lavgsen", "lpolpc"]],
                "Crime and deterrence", "Counties × years · North Carolina, 1981–1987",
                "Log crime rate, probabilities of arrest, conviction and prison, average sentence and police per capita.")
e = rdata("EmplUK")
ex["employment"] = (pd.DataFrame({"firm": e.firm, "year": e.year, "log_emp": np.log(e.emp), "log_wage": np.log(e.wage),
                                  "log_capital": np.log(e.capital), "log_output": np.log(e.output)}),
                    "Employment in UK firms", "Firms × years · 140 firms, 1976–1984",
                    "Log employment, real wage, capital and industry output; the Arellano–Bond data, unbalanced.")
g = rdata("Cigar")
ex["cigar"] = (pd.DataFrame({"state": g.state, "year": g.year, "log_sales": np.log(g.sales), "log_price": np.log(g.price / g.cpi),
                             "log_income": np.log(g.ndi / g.cpi), "log_pimin": np.log(g.pimin / g.cpi)}),
               "Cigarette demand", "States × years · 46 U.S. states, 1963–1992",
               "Log packs per capita, real price, real income per capita and the real minimum price in neighbouring states.")
ga = rdata("Gasoline")
ex["gasoline"] = (ga[["country", "year", "lgaspcar", "lincomep", "lrpmg", "lcarpcap"]],
                  "Gasoline demand in the OECD", "Countries × years · 18 countries, 1960–1978",
                  "Log gasoline use per car, real income per capita, real gasoline price and cars per capita (Baltagi–Griffin).")
pr = rdata("Produc")
ex["produc"] = (pd.DataFrame({"state": pr.state, "year": pr.year, "log_gsp": np.log(pr.gsp), "log_pcap": np.log(pr.pcap),
                              "log_pc": np.log(pr.pc), "log_emp": np.log(pr.emp), "unemp": pr.unemp}),
                "Public capital and state output", "States × years · 48 U.S. states, 1970–1986",
                "Log gross state product, public capital, private capital and employment, and the unemployment rate (Munnell).")

(OUT / "examples").mkdir(parents=True, exist_ok=True)

# extra data and helper files used by the step-by-step estimator guides (build_panel_guides.py)
import shutil
L = rdata("LaborSupply")
pd.DataFrame({"id": L.id, "year": L.year, "lnhr": L.lnhr, "lnwg": L.lnwg, "kids": L.kids, "age": L.age, "disab": L.disab}
             ).to_csv(OUT / "examples" / "laborsupply.csv", index=False)
M = rdata("macro")
M[["country", "year", "gdp", "unem", "capmob", "trade"]].rename(columns={"gdp": "gdp_growth"}).round(6).to_csv(OUT / "examples" / "garrett.csv", index=False)
S = rdata("SumHes")
S[["country", "year", "pop", "gdp", "sr"]].to_csv(OUT / "examples" / "growth_saving.csv", index=False)
g[["state", "year", "price", "pop", "pop16", "cpi", "ndi", "sales", "pimin"]].to_csv(OUT / "examples" / "cigar_raw.csv", index=False)
w[["nr", "year", "lwage", "union", "married", "exper", "expersq", "hours", "educ", "black", "hisp"]].to_csv(OUT / "examples" / "wagepan_full.csv", index=False)
shutil.copy(HERE / "panel_estimators.py", OUT / "examples" / "panel_estimators.py")
shutil.copy(ROOT / "time-series-guide" / "examples" / "diff_gmm.py", OUT / "examples" / "diff_gmm.py")
examples = []
for key, (df, title, meta, desc) in ex.items():
    path = OUT / "examples" / f"{key}.csv"
    df.round(6).to_csv(path, index=False)
    r = json.loads(pa.analyze(path.read_text()))
    assert "error" not in r, (key, r)
    print(f"{key}: N={r['N']} T={r['T']} {r['branch']} -> {r['model']}")
    examples.append(dict(key=key, title=title, meta=meta, desc=desc, file=f"examples/{key}.csv", result=r))

ts = (ROOT / "tools/model_selector/template.html").read_text()
css = ts[ts.index("<style>") + 7: ts.index("</style>")].strip()
tpl = (HERE / "template.html").read_text()
page = (tpl.replace("/*TS_CSS*/", css)
        .replace("/*EXAMPLES*/", json.dumps(examples, ensure_ascii=False, separators=(",", ":")))
        .replace("#LONGT#", str(pa.LONG_T)).replace("#GMMN#", str(pa.GMM_N)).replace("#FEWN#", str(pa.FEW_N)).replace("#PYSRC#", src.replace("</script", "<\\/script")))
(OUT / "index.html").write_text(page)
(OUT / "implementations" / "panel_tests.py").write_text(src)
(OUT / "examples" / "panel_tests.py").write_text(src)
print("wrote panel-data-guide/index.html", len(page) // 1024, "KB")
