#!/usr/bin/env python3
"""Build dsge-guide/index.html from the Dynare results exported by the guide builds (tools/dsge_selector/exports/*.json).

Run build_dsge_guides.py first: each guide's hidden Octave cell writes the numbers the page shows.
The page shares its stylesheet with the Time-Series Guide (tools/model_selector/template.html).
The upload mode runs data_check.py in the browser with Pyodide; it prepares data, the models run in Dynare.
"""
import json, pathlib, shutil

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "dsge-guide"
SLUGS = ["rbc", "nk", "soe", "estimation", "medium", "optimal", "zlb", "higher"]

exports = {}
for s in SLUGS:
    p = HERE / "exports" / f"{s}.json"
    if p.exists():
        exports[s] = json.loads(p.read_text())
    else:
        print(f"warning: no export for {s}; its example shows the setup only")

src = (HERE / "data_check.py").read_text()
ts = (ROOT / "tools/model_selector/template.html").read_text()
css = ts[ts.index("<style>") + 7: ts.index("</style>")].strip()
page = ((HERE / "template.html").read_text().replace("/*TS_CSS*/", css)
        .replace("/*EXPORTS*/", json.dumps(exports, separators=(",", ":")).replace("NaN", "null"))
        .replace("#PYSRC#", src.replace("</script", "<\\/script")))
(OUT / "index.html").write_text(page)
(OUT / "data" / "data_check.py").write_text(src)
print("wrote dsge-guide/index.html", len(page) // 1024, "KB;", len(exports), "exports")
