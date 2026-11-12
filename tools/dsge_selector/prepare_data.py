#!/usr/bin/env python3
"""Prepare the observables used by the Dynare guides from teaching/monetary-economics/data/us_quarterly.csv.

All series are quarterly, 1966Q1-2007Q4 (the Great Moderation sample ends before the zero lower bound years):
    YGR  = 100 * quarterly log change of real GDP           (output growth, percent per quarter)
    CGR  = 100 * quarterly log change of real consumption
    IGR  = 100 * quarterly log change of real investment
    INFL = 400 * quarterly log change of the GDP deflator   (annualised inflation, percent)
    INT  = effective federal funds rate                     (annualised, percent)
us_levels.csv keeps log levels of output, consumption and investment for the RBC moments.
The *_demeaned files subtract the sample mean of each observable (estimation in deviations from the mean).
"""
import pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
q = pd.read_csv(ROOT / "teaching/monetary-economics/data/us_quarterly.csv")
q["date"] = q["t"].apply(lambda t: f"{int(t)}Q{int(round((t % 1) * 4)) + 1}")
d = pd.DataFrame({"date": q["date"], "t": q["t"],
                  "YGR": 100 * np.log(q["rgdp"]).diff(), "CGR": 100 * np.log(q["rcons"]).diff(),
                  "IGR": 100 * np.log(q["rinv"]).diff(), "INFL": 400 * np.log(q["gdpdef"]).diff(), "INT": q["ffr"]})
s = d[(d["t"] >= 1966) & (d["t"] < 2008)]
out = ROOT / "dsge-guide" / "data"
out.mkdir(parents=True, exist_ok=True)
s[["YGR", "INFL", "INT"]].round(6).to_csv(out / "us_small.csv", index=False)
(s[["YGR", "INFL", "INT"]] - s[["YGR", "INFL", "INT"]].mean()).round(6).to_csv(out / "us_small_demeaned.csv", index=False)
(s[["YGR", "CGR", "IGR", "INFL", "INT"]] - s[["YGR", "CGR", "IGR", "INFL", "INT"]].mean()).round(6).to_csv(out / "us_medium_demeaned.csv", index=False)
s[["YGR", "CGR", "IGR", "INFL", "INT"]].round(6).to_csv(out / "us_medium.csv", index=False)
s[["date", "YGR", "CGR", "IGR", "INFL", "INT"]].round(6).to_csv(out / "us_observables_dated.csv", index=False)
lv = q[(q["t"] >= 1966) & (q["t"] < 2008)]
pd.DataFrame({"date": lv["date"], "log_y": np.log(lv["rgdp"]), "log_c": np.log(lv["rcons"]), "log_i": np.log(lv["rinv"])}).round(6).to_csv(out / "us_levels.csv", index=False)
print(len(s), "quarters;", s[["YGR", "INFL", "INT"]].mean().round(2).to_dict())
