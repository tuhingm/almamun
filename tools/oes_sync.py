#!/usr/bin/env python3
"""Copy the Open Econometric Systems engine into the website and prepare the files the OES page needs.

    python3 tools/oes_sync.py /path/to/open-econometric-systems

Writes
    oes/engine/oes_engine/*.py   the engine, loaded by oes/worker.js into Pyodide
    oes/engine/catalog.json      models, input fields, guide examples and the Pyodide packages each model needs
    oes/engine/expected.json     reference results of every guide example, used by the in-browser test suite

Run it with the same package versions as Pyodide 0.27.2 (numpy 2.0.2, pandas 2.2.3, scipy 1.14.1, statsmodels 0.14.4,
scikit-learn 1.6.1, xgboost 2.1.2) so the browser test suite compares like with like.
"""
import json
import pathlib
import shutil
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ROOT.parent / "open-econometric-systems") / "oes-engine" / "oes_engine"
dst = ROOT / "oes" / "engine" / "oes_engine"
dst.mkdir(parents=True, exist_ok=True)
for f in dst.glob("*.py"):
    f.unlink()
for f in sorted(src.glob("*.py")):
    shutil.copy(f, dst / f.name)
print("engine files:", ", ".join(sorted(f.name for f in dst.glob("*.py"))))

sys.dont_write_bytecode = True
sys.path.insert(0, str(dst.parent))
import oes_engine as E  # noqa: E402

(ROOT / "oes/engine/catalog.json").write_text(json.dumps({"version": E.__version__, "models": E.catalog()}, separators=(",", ":")))

expected = {}
for m in E.MODELS:
    ex = m["example"]
    spec = dict(ex["spec"], ops=ex.get("ops", []))
    t = time.time()
    r = E.run(m["id"], (ROOT / ex["file"]).read_text(), spec)
    if "error" in r:
        raise SystemExit(f"{m['id']}: {r['error']}")
    # the first estimate in the output: column 2 of the first table whose first row holds a number there
    ti = next(i for i, tb in enumerate(r["tables"]) if tb["rows"] and isinstance(tb["rows"][0][1], (int, float)))
    tb = r["tables"][ti]
    row = tb["rows"][0]
    ml = m["group"] == "Machine learning"
    expected[m["id"]] = {"nobs": r["nobs"], "table_index": ti, "table": tb["title"], "row": str(row[0]), "col": tb["columns"][1],
                         "value": row[1], "rtol": 0.05 if ml else 1e-3, "seconds": round(time.time() - t, 1)}
    # and the first non-integer statistic of the summary tiles (bounds F, R-squared, ATT, test RMSE, ...)
    sm = next((x for x in r["summary"] if isinstance(x["value"], float)), None)
    if sm:
        expected[m["id"]]["summary"] = {"label": sm["label"], "value": sm["value"]}
    print(f"{m['id']:14s} {expected[m['id']]['seconds']:5.1f}s  {row[0]} = {row[1]};  {sm and sm['label']} = {sm and sm['value']}")
(ROOT / "oes/engine/expected.json").write_text(json.dumps(expected, indent=1))
