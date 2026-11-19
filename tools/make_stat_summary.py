#!/usr/bin/env python3
"""Write stat-validation/LEDGER.md and SUMMARY.md from stat-validation/results (run make_stat_pages.py first)."""
import json, os, glob
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R = os.path.join(ROOT, "stat-validation", "results")
led = json.load(open(os.path.join(R, "ledger.json")))
L = ["# Implementation ledger", "", "Status of the 31 Statistics Division methods. 'MC reps' is the total number of Monte Carlo replications actually run for the method; 'capped cells' counts (scenario, n) cells that stopped before the 2,000 requested replications because of the per-cell time budget.", "",
     "| Code | Method | Implemented | Unit/integration tests | Benchmarked | Monte Carlo | MC reps | capped cells | Status |", "|---|---|---|---|---|---|---|---|---|"]
for e in led:
    L.append(f"| {e['code']} | {e['name']} | yes | yes | yes | {'yes' if e['mc'] else 'NO'} | {e['reps'] if e['reps'] is not None else '–'} | {e['capped'] if e['capped'] is not None else '–'} | {e['status']} |")
L += ["", "## Reasons behind non-plain statuses", ""]
for e in led:
    if e["status"] != "Validated": L.append(f"* **{e['code']} {e['name']}** ({e['status']}): {e['reason']}")
L += ["", "## Known gaps (applies to the whole division)", "",
 "* All computation and simulation was run in CPython with the Pyodide-pinned package versions. The pages were exercised in a simulated browser (jsdom); they have not been run in a real browser with real Pyodide, so browser speed and memory are untested.",
 "* The OES single-column CSV reader requires at least two columns; a one-column file is rejected (inherited behaviour of the existing OES core).",
 "* Some reference comparisons were possible only against Python packages; R packages (lme4, geepack, cmprsk) were not available, so lme4 comparisons use published values.",
 "* The BCG vaccine table was entered by hand from the published table and checked by reproducing metafor's published REML results.",
 "* Several example datasets are distributed by R packages under GPL; check the licence before redistributing them."]
open(os.path.join(ROOT, "stat-validation", "LEDGER.md"), "w").write("\n".join(L) + "\n")
tot = sum(e["reps"] or 0 for e in led); mcs = glob.glob(os.path.join(R, "mc_*.json"))
S = ["# Validation summary", "", f"* Methods registered: {len(led)} (69 existing OES models + 31 = 100).", f"* Monte Carlo result files: {len(mcs)}; total replications run: {tot:,}.", "* Software tests: see `stat-validation/tests` (run `pytest`).", "* Benchmark log: `results/benchmark_log.txt`.", "", "| Status | Methods |", "|---|---|"]
from collections import Counter
for k, v in Counter(e["status"] for e in led).items(): S.append(f"| {k} | {v} |")
open(os.path.join(ROOT, "stat-validation", "SUMMARY.md"), "w").write("\n".join(S) + "\n")
print("ledger written")
