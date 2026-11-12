#!/usr/bin/env python3
"""Build causal-inference-guide/index.html: prepares the real example datasets, runs the analysis on each, fills the template.

The page shares its stylesheet with the Time-Series Guide (tools/model_selector/template.html).
Needs: pip install numpy pandas scipy statsmodels causaldata pydataset
"""
import json, pathlib, sys, warnings
import numpy as np, pandas as pd

warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "causal-inference-guide"
sys.path.insert(0, str(HERE))
import causal_analysis as ca
import causaldata as cd
from pydataset import data as rdata

(OUT / "examples").mkdir(parents=True, exist_ok=True)
(OUT / "implementations").mkdir(parents=True, exist_ok=True)

c = cd.castle.load_pandas().data
castle = pd.DataFrame({"state": c["sid"], "year": c["year"], "l_homicide": c["l_homicide"], "l_police": c["l_police"], "l_murder": np.log(c["murder"]),
                       "castle_law": (c["post"] > 0).astype(int)})
first = castle[castle.castle_law == 1].groupby("state")["year"].min()
castle2007 = castle[castle.state.isin(first[first == 2007].index) | ~castle.state.isin(first.index)]
t = cd.texas.load_pandas().data
texas = pd.DataFrame({"state": t["state"], "year": t["year"], "bmprison": t["bmprison"],
                      "expansion": ((t["state"] == "Texas") & (t["year"] >= 1993)).astype(int)})
u = rdata("UKDriverDeaths")
seat = pd.DataFrame({"time": u["time"].round(4), "deaths": u["UKDriverDeaths"], "law": (u["time"] >= 1983.08).astype(int)})
l = cd.close_elections_lmb.load_pandas().data
lmb = l[["demvoteshare", "score", "democrat", "lagdemvoteshare"]].dropna()
card = cd.close_college.load_pandas().data

ex = [
    ("castle_2007", castle2007[["state", "year", "l_homicide", "castle_law"]], dict(outcome="l_homicide", treat="castle_law", unit="state", time="year"),
     "Castle Doctrine laws of 2007", "States × years · 2000–2010",
     "Log homicide rate in the 13 states that passed a stand-your-ground law in 2007 and the states that never did (Cheng and Hoekstra 2013)."),
    ("castle", castle[["state", "year", "l_homicide", "castle_law"]], dict(outcome="l_homicide", treat="castle_law", unit="state", time="year"),
     "Castle Doctrine laws, all adopters", "States × years · 2000–2010",
     "Log homicide rate; 21 states passed stand-your-ground laws between 2005 and 2009, at different dates."),
    ("castle_murder", castle[["state", "year", "l_murder", "castle_law"]], dict(outcome="l_murder", treat="castle_law", unit="state", time="year"),
     "Castle Doctrine laws and murder", "States × years · 2000–2010",
     "Log murder rate from police reports, the same staggered laws; a case where the pre-trends are not parallel."),
    ("texas", texas, dict(outcome="bmprison", treat="expansion", unit="state", time="year"),
     "Texas prison expansion", "States × years · 1985–2000",
     "Black male prisoners; Texas greatly expanded its prison capacity in 1993, and no other state did (Cunningham)."),
    ("seatbelt", seat, dict(outcome="deaths", treat="law", time="time"),
     "UK seatbelt law", "Monthly · 1969–1984 · one series",
     "Car drivers killed or seriously injured; wearing front seatbelts became compulsory in February 1983."),
    ("close_elections", lmb, dict(outcome="score", treat="democrat", running="demvoteshare", cutoff=0.5),
     "Close U.S. House elections", "Elections · 1946–1995",
     "ADA liberal-voting score of the winner, by the Democratic vote share; a Democrat wins above 0.5 (Lee, Moretti and Butler 2004)."),
    ("card", card, dict(outcome="lwage", treat="educ", instruments=["nearc4"], controls=["black", "smsa", "south", "married", "exper"]),
     "Returns to schooling", "Cross-section · 3,010 men, 1976",
     "Log wage and years of education, instrumented by growing up near a four-year college (Card 1995)."),
]
examples = []
for key, df, roles, title, meta, desc in ex:
    path = OUT / "examples" / f"{key}.csv"
    df.round(6).to_csv(path, index=False)
    r = json.loads(ca.analyze(path.read_text(), json.dumps(roles)))
    assert "error" not in r, (key, r)
    print(f"{key}: {r['model']}")
    examples.append(dict(key=key, title=title, meta=meta, desc=desc, file=f"examples/{key}.csv", result=r))

src = (HERE / "causal_analysis.py").read_text()
ts = (ROOT / "tools/model_selector/template.html").read_text()
css = ts[ts.index("<style>") + 7: ts.index("</style>")].strip()
page = ((HERE / "template.html").read_text().replace("/*TS_CSS*/", css)
        .replace("/*EXAMPLES*/", json.dumps(examples, ensure_ascii=False, separators=(",", ":")))
        .replace("#FEW#", str(ca.FEW_TREATED)).replace("#DONORS#", str(ca.MIN_DONORS))
        .replace("#PYSRC#", src.replace("</script", "<\\/script")))
(OUT / "index.html").write_text(page)
(OUT / "implementations" / "causal_tests.py").write_text(src)
(OUT / "examples" / "causal_tests.py").write_text(src)
print("wrote causal-inference-guide/index.html", len(page) // 1024, "KB")
