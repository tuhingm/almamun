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
LOCAL = "--local" in sys.argv      # engine already edited in place in oes/engine: only rebuild catalog.json and expected.json
ONLY = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)   # run the examples of these models only and merge into expected.json
args = [a for a in sys.argv[1:] if a != "--local" and not a.startswith("--only=")]
engine_root = pathlib.Path(args[0] if args else ROOT.parent / "open-econometric-systems")
src = engine_root / "oes-engine" / "oes_engine"
dst = ROOT / "oes" / "engine" / "oes_engine"
dst.mkdir(parents=True, exist_ok=True)
if not LOCAL:
    for f in dst.glob("*.py"):
        f.unlink()
    for f in sorted(src.glob("*.py")):
        shutil.copy(f, dst / f.name)
print("engine files:", ", ".join(sorted(f.name for f in dst.glob("*.py"))))

sys.dont_write_bytecode = True
sys.path.insert(0, str(dst.parent))
import oes_engine as E  # noqa: E402

(ROOT / "oes/engine/catalog.json").write_text(json.dumps({"version": E.__version__, "models": E.catalog()}, separators=(",", ":")))
(ROOT / "oes/engine/stat_catalog.json").write_text(json.dumps({"version": E.__version__, "models": E.catalog("statistics")}, separators=(",", ":")))


def build_expected(models, path):
    expected = json.loads(path.read_text()) if ONLY and path.exists() else {}
    for m in models:
        if ONLY and m["id"] not in ONLY:
            continue
        ex = m["example"]
        spec = dict(ex["spec"], ops=ex.get("ops", []))
        t = time.time()
        r = E.run(m["id"], (ROOT / ex["file"]).read_text() if ex.get("file") else "", spec)
        if "error" in r:
            raise SystemExit(f"{m['id']}: {r['error']}")
        # the cell to check: the registry's own choice (check=dict(table, row, col)), else the first estimate in the output:
        # the first non-integer number in the table rows (column 2 onwards), falling back to column 2 of the first numeric row
        ck = m.get("check")
        if ck:
            ti, ri, ci = ck["table"], ck["row"], ck["col"]
        else:
            hit = None
            for i, tb in enumerate(r["tables"]):
                for rj, rw in enumerate(tb["rows"]):
                    for cj in range(1, len(rw)):
                        if isinstance(rw[cj], float):
                            hit = (i, rj, cj)
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit is None:
                hit = next((i, 0, 1) for i, tb in enumerate(r["tables"]) if tb["rows"] and isinstance(tb["rows"][0][1], (int, float)))
            ti, ri, ci = hit
        tb = r["tables"][ti]
        ri = ri if ri >= 0 else len(tb["rows"]) + ri
        row = tb["rows"][ri]
        ml = m["group"] in ("Machine learning", "Bayesian")   # simulation-based: allow small cross-platform differences
        expected[m["id"]] = {"nobs": r["nobs"], "table_index": ti, "table": tb["title"], "row": str(row[0]), "row_index": ri, "col": tb["columns"][ci], "col_index": ci,
                             "value": row[ci], "rtol": 0.05 if ml else 1e-3, "seconds": round(time.time() - t, 1)}
        # and the first non-integer statistic of the summary tiles (bounds F, R-squared, ATT, test RMSE, ...)
        sm = next((x for x in r["summary"] if isinstance(x["value"], float)), None)
        if sm:
            expected[m["id"]]["summary"] = {"label": sm["label"], "value": sm["value"]}
        print(f"{m['id']:18s} {expected[m['id']]['seconds']:5.1f}s  {row[0]} [{tb['columns'][ci]}] = {row[ci]};  {sm and sm['label']} = {sm and sm['value']}")
    path.write_text(json.dumps(expected, indent=1))


build_expected(E.MODELS, ROOT / "oes/engine/expected.json")
build_expected(E.STAT_MODELS, ROOT / "oes/engine/stat_expected.json")

# the Web Worker loads the engine file by file: its list must match the engine
import re
listed = set(re.findall(r'"(\w+\.py)"', (ROOT / "oes/worker.js").read_text()))
on_disk = {f.name for f in dst.glob("*.py")}
if listed != on_disk:
    raise SystemExit(f"oes/worker.js FILES is out of date: missing {sorted(on_disk - listed)}, extra {sorted(listed - on_disk)}")
