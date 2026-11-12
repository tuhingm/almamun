#!/usr/bin/env python3
"""Build ml-guide/index.html: prepare the real example datasets, run the decision engine on each, and fill the template.

The page shares its stylesheet with the Time-Series Guide (tools/model_selector/template.html), like the Panel Data Guide.
Needs: pip install numpy pandas scipy statsmodels scikit-learn wooldridge pyreadr
Data downloaded once into tools/ml_selector/data/ (all public):
  sipp1991.dta  401(k) data, Chernozhukov et al. (2018)  https://raw.githubusercontent.com/VC2015/DMLonGitHub/master/sipp1991.dta
  GrowthData.rda  Barro-Lee growth data (R package hdm)  https://raw.githubusercontent.com/cran/hdm/master/data/GrowthData.rda
Then run tools/site_polish.py, tools/og_tags.py and tools/make_sitemap.py.
"""
import json, pathlib, sys, time, urllib.request
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "ml-guide"
sys.path.insert(0, str(HERE))
import ml_analysis as A

DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
SRC = {"sipp1991.dta": "https://raw.githubusercontent.com/VC2015/DMLonGitHub/master/sipp1991.dta",
       "GrowthData.rda": "https://raw.githubusercontent.com/cran/hdm/master/data/GrowthData.rda"}
for f, u in SRC.items():
    if not (DATA / f).exists():
        urllib.request.urlretrieve(u, DATA / f)


def datasets():
    import wooldridge as wd, pyreadr, statsmodels.api as sm
    X = {}
    e = wd.data("econmath")
    X["econmath"] = e[["score", "age", "work", "study", "econhs", "colgpa", "hsgpa", "acteng", "actmth", "male",
                       "calculus", "attexc", "attgood", "fathcoll", "mothcoll"]].dropna()
    w = wd.data("wage2")
    X["wage2"] = w[["lwage", "hours", "IQ", "KWW", "educ", "exper", "tenure", "age", "married", "black", "south", "urban",
                    "sibs", "brthord", "meduc", "feduc"]].dropna()
    m = sm.datasets.macrodata.load_pandas().data
    q = pd.PeriodIndex.from_fields(year=m.year.astype(int), quarter=m.quarter.astype(int), freq="Q").astype(str)
    mm = pd.DataFrame({"quarter": q, "infl": m.infl, "unemp": m.unemp, "tbilrate": m.tbilrate,
                       "gdp_growth": 400 * np.log(m.realgdp).diff(), "m1_growth": 400 * np.log(m.m1).diff()})
    for v in ["infl", "unemp", "tbilrate", "gdp_growth", "m1_growth"]:
        for L in (1, 2, 3, 4):
            mm[f"{v}_l{L}"] = mm[v].shift(L)
    X["macro"] = mm[["quarter", "infl"] + [c for c in mm if "_l" in c]].dropna().round(4)
    b = wd.data("mlb1")
    X["mlb1"] = b[["lsalary", "years", "games", "atbats", "runs", "hits", "hruns", "rbis", "bavg", "bb", "so", "sbases",
                   "fldperc", "yrsallst", "black", "hispan"]].dropna()
    inj = wd.data("injury")
    X["injury"] = inj[["ldurat", "afchnge", "highearn", "male", "married", "hosp", "age", "lprewage", "ltotmed", "head", "neck",
                       "upextr", "trunk", "lowback", "lowextr", "occdis", "manuf", "construc", "ky"]].dropna()
    j = wd.data("jtrain2")
    X["jtrain2"] = j[["re78", "train", "age", "educ", "black", "hisp", "married", "nodegree", "re74", "re75"]]
    g = pyreadr.read_r(str(DATA / "GrowthData.rda"))["GrowthData"].drop(columns=["intercept"]).rename(columns={"Outcome": "growth"})
    X["growth"] = g
    s = pd.read_stata(DATA / "sipp1991.dta")
    f = pd.DataFrame({"net_tfa": s.net_tfa, "e401": s.e401, "p401": s.p401, "age": s.age, "age_sq": s.age ** 2, "inc": s.inc,
                      "inc_sq": s.inc ** 2, "educ": s.educ, "educ_sq": s.educ ** 2, "fsize": s.fsize, "marr": s.marr,
                      "twoearn": s.twoearn, "db": s.db, "pira": s.pira, "hown": s.hown}).astype(float)
    X["k401_irm"] = f.drop(columns=["p401"])
    X["k401_iv"] = f[["net_tfa", "p401", "e401"] + [c for c in f if c not in ("net_tfa", "p401", "e401")]]
    return X


EX = [  # key, group, title, meta, description, goal, treatment, instrument, B1 justification
    ("econmath", "Prediction", "Exam scores in economics", "Students · 814 · cross-section",
     "Score in an introductory economics course from study habits, grades, ACT scores and background (Wooldridge econmath).", "predict", None, None, None),
    ("wage2", "Prediction", "Wages and ability", "Men · 663 · cross-section",
     "Log wage from IQ, a knowledge test, schooling, experience, family background and region (Blackburn–Neumark).", "predict", None, None, None),
    ("macro", "Prediction", "Forecasting U.S. inflation", "Quarters · 1960–2009 · time series",
     "Next-quarter inflation from four lags of inflation, unemployment, the T-bill rate, GDP growth and money growth.", "predict", None, None, None),
    ("mlb1", "Prediction", "Baseball salaries", "Players · 353 · cross-section",
     "Log salary of Major League Baseball players from career statistics (Wooldridge mlb1).", "predict", None, None, None),
    ("injury", "Prediction", "Time out of work after an injury", "Workers · 6,822 · cross-section",
     "Log weeks of workers’ compensation from earnings, age, medical costs, injury type and a benefit increase (Meyer, Viscusi and Durbin).", "predict", None, None, None),
    ("jtrain2", "Causal effect", "Job training and earnings", "Men · 445 · randomised experiment",
     "Effect of the National Supported Work programme on 1978 earnings, with eight controls (LaLonde; Wooldridge jtrain2).", "causal", "train", None,
     "training was assigned at random"),
    ("growth", "Causal effect", "Growth convergence", "Countries · 90 · 60 controls",
     "Effect of initial GDP per capita on growth, 1965–1975, with 60 country characteristics (Barro–Lee; Belloni et al.).", "causal", "gdpsh465", None,
     "convergence is conditional on the country characteristics in the controls"),
    ("k401_irm", "Causal effect", "401(k) eligibility and saving", "Households · 9,915 · binary treatment",
     "Effect of 401(k) eligibility on net financial assets, with flexible income, age and education controls (SIPP 1991).", "causal", "e401", None,
     "eligibility is as good as random once income, age, education and family are held fixed (Poterba, Venti and Wise)"),
    ("k401_iv", "Causal effect", "401(k) participation, instrumented", "Households · 9,915 · instrument",
     "Effect of participating in a 401(k) on net financial assets, with eligibility as the instrument (SIPP 1991).", "causal", "p401", "e401", None),
]


def main():
    (OUT / "examples").mkdir(parents=True, exist_ok=True)
    X = datasets()
    examples = []
    for key, group, title, meta, desc, goal, treat, instr, b1 in EX:
        df = X[key]
        path = OUT / "examples" / f"{key}.csv"
        df.to_csv(path, index=False)
        t = time.time()
        cache = HERE / "data" / f"{key}.json"
        if cache.exists() and "--rerun" not in sys.argv and cache.stat().st_mtime > (HERE / "ml_analysis.py").stat().st_mtime:
            r = json.loads(cache.read_text())
        else:
            r = json.loads(A.analyze(path.read_text(), goal, treat, instr))
            cache.write_text(json.dumps(r))
        assert "error" not in r, (key, r)
        if b1:
            r["b1"] = b1
        print(f"{key}: {r['model']}  ({time.time() - t:.0f}s)", flush=True)
        examples.append(dict(key=key, group=group, title=title, meta=meta, desc=desc, file=f"examples/{key}.csv", result=r))
    src = (HERE / "ml_analysis.py").read_text()
    ts = (ROOT / "tools/model_selector/template.html").read_text()
    css = ts[ts.index("<style>") + 7: ts.index("</style>")].strip()
    page = (HERE / "template.html").read_text().replace("/*TS_CSS*/", css).replace("/*APP_JS*/", (HERE / "app.js").read_text())
    page = page.replace("/*EXAMPLES*/", json.dumps(examples, ensure_ascii=False, separators=(",", ":")))
    for k, v in dict(MINTS=A.MIN_TS, SMALLN=A.SMALL_N, GAIN=int(A.GAIN * 100), HIGHR=A.HIGH_R, BIGN=A.BIG_N, FEWCTRL=A.FEW_CTRL,
                     WEAKF=A.WEAK_F, OVLO=A.OVERLAP[0], OVHI=A.OVERLAP[1], CFN=A.CF_N, SPARSE=A.SPARSE).items():
        page = page.replace(f"#{k}#", str(v))
    ready = [p.stem for p in OUT.glob("*.html") if p.name != "index.html"]
    page = page.replace("/*READY*/", json.dumps(sorted(ready)))
    page = page.replace("#PYSRC#", src.replace("</script", "<\\/script"))
    (OUT / "index.html").write_text(page)
    (OUT / "examples" / "ml_analysis.py").write_text(src)
    print("wrote ml-guide/index.html", len(page) // 1024, "KB")


if __name__ == "__main__":
    main()
