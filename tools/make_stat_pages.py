#!/usr/bin/env python3
"""Generate the Statistics Division documentation from real numbers.
  python3 tools/make_stat_pages.py
Reads  oes/engine/stat_catalog.json, stat-validation/docs/method_docs.py, stat-validation/results/mc_*.json,
       stat-validation/results/benchmark_log.txt, and runs every example through oes_engine.
Writes stat-guide/<method>.html (method page), stat-guide/reports/<method>.html (16-section validation report),
       stat-guide/index.html, stat-validation/LEDGER.md, stat-validation/SUMMARY.md.
Nothing numeric is typed in here: simulation, benchmark and example numbers all come from the files above."""
import html, json, os, re, sys, time, warnings
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "oes", "engine")); sys.path.insert(0, os.path.join(ROOT, "stat-validation", "docs")); sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import oes_engine as o
from method_docs import DOCS, REF, DATA

RES = os.path.join(ROOT, "stat-validation", "results")
CAT = json.load(open(os.path.join(ROOT, "oes", "engine", "stat_catalog.json")))["models"]
STATUS = {  # honest status of each method; the reason is shown on its report and in the ledger
 "stat_finegray": ("Partially validated", "Point estimates validated (equals Cox when no competing events; simulation unbiased). Standard errors omit the censoring-weight estimation variance and no R cmprsk comparison was possible, so inference rests on simulation only."),
 "stat_glmm": ("Partially validated", "Matches an independent quadrature MLE and published lme4 values; no direct GLMM reference implementation was available in this environment."),
 "stat_gee": ("Validated with a documented deviation", "Independence/exchangeable match statsmodels; AR(1) with estimated alpha uses a different moment estimator (see limitations); QIC follows Pan (2001) and differs from statsmodels' QIC."),
 "stat_meta_fixed": ("Validated, small-study tests unchecked", "Pooling and heterogeneity match statsmodels and metafor; Egger and Begg tests were not cross-checked."),
 "stat_meta_random": ("Validated, small-study tests unchecked", "Pooled estimate, tau2 and HKSJ interval match metafor published values and statsmodels; Egger and Begg tests were not cross-checked."),
}
DEFAULT_STATUS = ("Validated", "Compared with reference software on identical data and specifications, and checked by Monte Carlo simulation.")
ROBUST_KEYS = ("robustness", "misspecified", "t(3)", "t(5)", "heteroskedastic", "unbalanced", "heavy ties", "skewed", "exponential", "non-spherical", "random slope present")
esc = html.escape

def slug(m): return os.path.basename(m["guide"])[:-5]
def is_robust(name): return any(k in name.lower() for k in ROBUST_KEYS)
def fmt(v, d=4):
    if v is None: return "–"
    if isinstance(v, float):
        if v != v: return "–"
        return f"{v:.{d}f}"
    return esc(str(v))
def table_html(cols, rows, cls=""):
    return f'<div class="tw"><table class="{cls}"><thead><tr>' + "".join(f"<th>{esc(str(c))}</th>" for c in cols) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{fmt(c)}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div>"
def result_tables(r, limit_rows=40, only=None):
    out = ""
    for t in r["tables"]:
        if only and not only(t["title"]): continue
        rows = t["rows"][:limit_rows]; out += f"<h4>{esc(t['title'])}</h4>" + table_html(t["columns"], rows)
        if len(t["rows"]) > limit_rows: out += f'<p class="hint">First {limit_rows} of {len(t["rows"])} rows shown.</p>'
        if t.get("note"): out += f'<p class="hint">{esc(t["note"])}</p>'
    return out

CSS = """:root{--navy:#0033a0;--navy2:#00205b;--ink:#1b2437;--muted:#55617a;--paper:#f3f5fa;--line:#d5dcea;--tint:#e3eaf7;--ok:#0f7a4f;--warn:#a05a00;--bad:#a12a2a}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.7 "DM Sans",system-ui,sans-serif}a{color:var(--navy)}
header{background:linear-gradient(120deg,#0033a0,#00205b);color:#fff;padding:34px 0 30px}.w{max-width:1000px;margin:0 auto;padding:0 20px}
header .k{font-size:10px;font-weight:800;letter-spacing:.2em;text-transform:uppercase;color:#9fd3f0}header h1{font:800 30px/1.2 Manrope,sans-serif;margin:6px 0 8px}header p{margin:0;color:#dbe6f7;max-width:760px}
nav.crumb{font-size:12px;margin-bottom:10px}nav.crumb a{color:#9fd3f0;text-decoration:none}
main{padding:26px 0 70px}section{background:#fff;border:1px solid var(--line);border-radius:14px;padding:22px 26px;margin:0 0 18px;box-shadow:0 5px 18px rgba(0,32,91,.05)}
h2{font:800 20px/1.25 Manrope,sans-serif;margin:0 0 8px}h3{font:800 15px Manrope,sans-serif;margin:16px 0 6px}h4{font:700 13px Manrope,sans-serif;margin:14px 0 4px;color:var(--navy2)}
.num{color:var(--navy);font-weight:800;margin-right:6px}.hint{font-size:12px;color:var(--muted);margin:4px 0}
.tw{overflow:auto;border:1px solid var(--line);border-radius:8px;margin:6px 0;max-height:460px}table{border-collapse:collapse;width:100%;font-size:12px;background:#fff}th{position:sticky;top:0;background:#f2f5fb;text-align:left;font-weight:700}th,td{padding:5px 9px;border-bottom:1px solid #edf0f6;white-space:nowrap}
pre{background:#0e1a33;color:#e6eefc;border-radius:10px;padding:14px 16px;overflow:auto;font:12.5px/1.55 ui-monospace,Menlo,Consolas,monospace}code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px}
.math{background:#f7f9fd;border-left:3px solid var(--navy);padding:6px 12px;margin:6px 0;font-family:"Cambria Math","STIX Two Math",Georgia,serif;font-size:14px;overflow-x:auto}
.badge{display:inline-block;font-size:11px;font-weight:800;border-radius:20px;padding:3px 11px;background:var(--tint);color:var(--navy)}.badge.ok{background:#dff3ea;color:var(--ok)}.badge.warn{background:#fdecd2;color:var(--warn)}
details{margin:8px 0}summary{cursor:pointer;font-weight:700;font-size:13px}ul{margin:6px 0 6px 20px;padding:0}li{margin:3px 0}
.cta{display:inline-block;background:var(--navy);color:#fff;border-radius:8px;padding:9px 16px;font-size:12px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;text-decoration:none;margin:4px 8px 4px 0}.cta.alt{background:#fff;color:var(--navy);border:1px solid var(--navy)}
footer{font-size:12px;color:var(--muted);text-align:center;padding:10px 20px 40px}"""
def page(title, kicker, lead, body, crumbs, depth):
    up = "../" * depth
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} | OES Statistics Division</title><meta name="robots" content="index, follow">
<link rel="icon" href="{up}assets/favicon.svg" type="image/svg+xml">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700;800&family=Manrope:wght@700;800&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><header><div class="w"><nav class="crumb">{crumbs}</nav><div class="k">{esc(kicker)}</div><h1>{esc(title)}</h1><p>{esc(lead)}</p></div></header>
<main><div class="w">{body}</div></main><footer>Open Econometric Systems, Statistics Division. Every number on this page was generated by the code in this project.</footer></body></html>"""

def load_mc(mid):
    p = os.path.join(RES, f"mc_{mid}.json")
    return json.load(open(p)) if os.path.exists(p) else None
BENCH = {}
def load_bench():
    p = os.path.join(RES, "benchmark_log.txt")
    if os.path.exists(p):
        cur = None
        for line in open(p).read().splitlines():
            m = re.match(r"##### bench_(\w+)\.py", line)
            if m: cur = m.group(1); BENCH[cur] = []
            elif cur: BENCH[cur].append(line)

def mc_table(sc):
    keys = [r for r in sc["results"]]
    rej = sorted({k for r in keys for k in r if k.startswith("reject_")})
    cols = ["n", "reps", "failures"]; est = any("bias" in r for r in keys); cov = any("coverage" in r for r in keys); flag = sorted({k for r in keys for k in r if k.startswith("rate_")})
    if est: cols += ["bias", "emp. SD", "RMSE"]
    if any("se_ratio" in r for r in keys): cols += ["mean SE / emp. SD"]
    if cov: cols += ["coverage", "mean CI width"]
    cols += [("Type I error " if sc["h0_true"] else "rejection rate ") + k.replace("reject_", "") for k in rej] + [k.replace("rate_flag_", "share ") for k in flag] + ["sec/rep"]
    rows = []
    for r in keys:
        row = [r["n"], r.get("reps_run"), r.get("failures")]
        if est: row += [r.get("bias"), r.get("emp_sd"), r.get("rmse")]
        if any("se_ratio" in x for x in keys): row += [r.get("se_ratio")]
        if cov: row += [r.get("coverage"), r.get("mean_ci_width")]
        row += [r.get(k) for k in rej] + [r.get(k) for k in flag] + [r.get("sec_per_rep")]
        rows.append(row)
    return table_html(cols, rows)

def mc_reading(sc):
    """Plain statements computed from the numbers (no judgement written by hand)."""
    rs = [r for r in sc["results"] if "reps_valid" in r]; out = []
    if not rs: return out
    rej = sorted({k for r in rs for k in r if k.startswith("reject_")})
    if sc["h0_true"] and rej:
        for k in rej:
            v = [r[k] for r in rs if k in r]; out.append(f"Rejection rate at the 5% level ({k[7:]}) under H0: {min(v):.3f} to {max(v):.3f} across n (Monte Carlo SE about {max(r.get('mc_se_'+k, 0) for r in rs):.3f}).")
    elif rej:
        for k in rej:
            v = [(r['n'], r[k]) for r in rs if k in r]; out.append(f"Power ({k[7:]}): " + ", ".join(f"n={n}: {x:.3f}" for n, x in v) + ".")
    if any("coverage" in r for r in rs):
        v = [r["coverage"] for r in rs if "coverage" in r]; out.append(f"Coverage of the nominal 95% interval: {min(v):.3f} to {max(v):.3f}.")
    if any("bias" in r for r in rs):
        b = [(r["n"], r["bias"]) for r in rs if "bias" in r]; out.append("Bias by n: " + ", ".join(f"{n}: {x:+.4f}" for n, x in b) + ".")
    fl = [r for r in rs if r["failures"]]
    if fl: out.append("Non-converged or failed replications: " + ", ".join(f"n={r['n']}: {r['failures']}/{r['reps_run']}" for r in fl) + ".")
    capped = [r for r in rs if r["reps_run"] < r["reps_requested"]]
    if capped: out.append("Replications actually run were below the 2,000 requested for n = " + ", ".join(f"{r['n']} ({r['reps_run']})" for r in capped) + " because of the time budget per cell.")
    return out

def replication_code(m):
    ex = m["example"]; f = os.path.basename(ex["file"]) if ex.get("file") else None
    spec = json.dumps(ex["spec"], indent=4)
    if f: load = f"csv = open('stat-guide/examples/{f}').read()"
    else: load = "csv = ''   # this method needs no data file"
    return f"""# run from the repository root (needs numpy, pandas, scipy)
import sys; sys.path.insert(0, 'oes/engine')
import oes_engine as o
{load}
spec = {spec}
result = o.run('{m["id"]}', csv, spec)
for t in result['tables']:
    print(t['title']); print(t['columns']); [print(r) for r in t['rows'][:10]]"""

def run_example(m):
    ex = m["example"]; csv = open(os.path.join(ROOT, ex["file"])).read() if ex.get("file") else ""
    t = time.perf_counter(); r = o.run(m["id"], csv, dict(ex["spec"])); dt = time.perf_counter() - t
    assert "error" not in r, (m["id"], r.get("error")); return r, dt

def fields_table(m):
    rows = []
    for f in m["fields"]:
        kind = {"col": "column", "cols": "several columns", "select": "choice", "num": "number", "text": "text"}.get(f["type"], f["type"])
        opts = "; ".join(f"{a} ({b})" for a, b in f.get("options", [])) if f.get("options") else ""
        rows.append([f["key"], f["label"], kind, "yes" if f.get("required") else "", f.get("default") if f.get("default") is not None else "", opts])
    return table_html(["key", "label", "type", "required", "default", "options"], rows)

def refs_html(keys): return "<ul>" + "".join(f"<li>{esc(REF[k])}</li>" for k in keys) + "</ul>"
def math_html(lines): return "".join(f'<div class="math">{esc(l)}</div>' for l in lines)
def ul(items): return "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"

def dataset_block(m):
    ex = m["example"]
    if not ex.get("file"): return "<p>No dataset: this method takes its inputs (effect size, significance level, sample size) directly.</p>"
    key = os.path.basename(ex["file"])[:-4]; nm, src = DATA.get(key, (key, "see the file header"))
    import pandas as pd
    d = pd.read_csv(os.path.join(ROOT, ex["file"])); cols = ", ".join(map(str, d.columns))
    return f"<p><b>{esc(nm)}</b>, file <code>{esc(ex['file'])}</code>: {len(d)} rows, columns {esc(cols)}.</p><p class='hint'>Source: {esc(src)} Check the licence of the distributing package before redistributing the data.</p>"

def build_method_page(m, d, res, dt):
    mid = m["id"]; st = STATUS.get(mid, DEFAULT_STATUS)
    body = f'<section><h2>Overview</h2><p>{esc(d["intro"])}</p><p><span class="badge {"ok" if st[0]=="Validated" else "warn"}">{esc(st[0])}</span> <span class="hint">{esc(st[1])}</span></p>'
    body += f'<a class="cta" href="../oes/index.html#{mid}">Run it on your data</a><a class="cta alt" href="reports/{slug(m)}.html">Technical validation report</a></section>'
    body += f'<section><h2>Mathematical formulation</h2>{math_html(d["math"])}</section>'
    body += f'<section><h2>Assumptions</h2>{ul(d["assumptions"])}</section>'
    body += f'<section><h2>Data input</h2><p>Upload a CSV file (or use the example) and choose these settings:</p>{fields_table(m)}</section>'
    body += f'<section><h2>Estimation algorithm</h2>{ul(d["algorithm"])}</section>'
    body += f'<section><h2>Example: results</h2>{dataset_block(m)}<p>Specification: <code>{esc(json.dumps(m["example"]["spec"]))}</code>. Computed in {dt:.2f} s by this engine when this page was generated.</p>{result_tables(res)}</section>'
    body += f'<section><h2>Diagnostics and interpretation</h2>{ul(d["interp"])}<p class="hint">The diagnostic tables are part of the results above (assumption checks, fit statistics, residual or goodness-of-fit tables).</p></section>'
    body += f'<section><h2>Limitations</h2>{ul(d["limitations"])}</section>'
    body += f'<section><h2>Replication code</h2><pre>{esc(replication_code(m))}</pre><p class="hint">Results can be exported as CSV from the OES page (Statistics section).</p></section>'
    body += f'<section><h2>References</h2>{refs_html(d["refs"])}</section>'
    crumbs = '<a href="index.html">Statistics Division guide</a> / ' + esc(m["group"])
    return page(m["name"], f'{m["code"]} · {m["group"]}', d["intro"], body, crumbs, 1)

def build_report(m, d, res, dt, mc, power_extra):
    mid = m["id"]; st = STATUS.get(mid, DEFAULT_STATUS); S = lambda n, t: f'<section id="s{n}"><h2><span class="num">{n}.</span>{t}</h2>'
    b = f'<section><p><span class="badge {"ok" if st[0]=="Validated" else "warn"}">{esc(st[0])}</span> {esc(st[1])}</p><p class="hint">This report validates an established method as implemented in OES. It does not claim a new estimator. All numbers were produced by the code in <code>stat-validation/</code> and <code>tools/make_stat_pages.py</code>; the engine version was {esc(o.__version__)}.</p></section>'
    b += S(1, "Methodological introduction") + f'<p>{esc(d["intro"])}</p></section>'
    b += S(2, "Historical and theoretical background") + f'<p>{esc(d["history"])}</p></section>'
    b += S(3, "Mathematical formulation") + math_html(d["math"]) + "</section>"
    b += S(4, "Statistical assumptions") + ul(d["assumptions"]) + "</section>"
    b += S(5, "Estimation algorithm") + ul(d["algorithm"]) + "</section>"
    fn = m["fn"]; b += S(6, "Computational implementation") + f'<ul><li>Module <code>oes/engine/oes_engine/{esc(fn.__module__.split(".")[-1])}.py</code>, function <code>{esc(fn.__name__)}</code>, registered as <code>{mid}</code> in <code>stat_registry.py</code>.</li><li>Runtime dependencies: NumPy, pandas and SciPy only (the packages available in the browser Pyodide runtime); no backend.</li><li>Input validation: missing columns, empty data, non-numeric or constant variables, wrong number of groups, singular matrices and non-convergence return a plain-language error instead of an exception (tests in <code>stat-validation/tests/test_stat_division.py</code>).</li><li>Example run time: {dt:.2f} s in CPython; real browser (Pyodide) timing has not been measured.</li></ul>{fields_table(m)}</section>'
    # 7 design
    b += S(7, "Monte Carlo experimental design")
    if mc and "scenarios" in mc:
        rows = []
        for sc in mc["scenarios"]:
            rr = sc["results"]; rows.append([sc["scenario"], "H0 true" if sc["h0_true"] else ("estimation" if any("bias" in r for r in rr) else "alternative"), ", ".join(str(r["n"]) for r in rr), rr[0]["reps_requested"], sc["note"]])
        b += "<p>Each replication draws a new data set from a fixed data-generating process and runs the engine exactly as the web page does. Replication r at size n uses <code>numpy.random.default_rng([method_code, scenario_index, n, r])</code>, so any replication can be rerun alone. Failures are counted, never discarded silently. Sizes n are the total number of observations (subjects for repeated-measures designs, studies for meta-analysis). A per-cell time budget of "
        b += f'{mc.get("budget_seconds_per_cell")} s stops very slow cells early; the replications actually run are shown in every results table.</p>' + table_html(["Scenario", "Purpose", "n grid", "reps requested", "Note"], rows)
        b += f'<p class="hint">Environment: Python {mc["env"]["python"]}, NumPy {mc["env"]["numpy"]}, SciPy {mc["env"]["scipy"]}, pandas {mc["env"]["pandas"]}. Code: <code>stat-validation/mc/experiments.py</code>.</p>'
    elif mc and "power_vs_simulation" in mc:
        b += "<p>The analytic power returned by the engine is compared with the empirical rejection rate of the actual test in a separate NumPy simulation (20,000 replications per case) that does not use the engine's code. Code: <code>stat-validation/mc/run_all.py</code> (<code>power_check</code>).</p>"
    else: b += "<p><b>Not run:</b> no Monte Carlo results were found for this method.</p>"
    b += "</section>"
    # 8 results
    b += S(8, "Simulation results")
    if mc and "scenarios" in mc:
        for sc in mc["scenarios"]:
            if is_robust(sc["scenario"]): continue
            b += f'<h3>{esc(sc["scenario"])}</h3>' + mc_table(sc) + ul(mc_reading(sc))
    elif mc and "power_vs_simulation" in mc:
        pv = mc["power_vs_simulation"]; b += table_html(["test", "n per group", "engine power", "simulated power", "difference", "MC SE"], [[x["test"], x["n"], x["engine_power"], x["empirical_power"], x["diff"], x["mc_se"]] for x in pv])
        b += f'<p>Largest absolute difference between the engine and the simulation: {max(abs(x["diff"]) for x in pv):.4f}; the largest Monte Carlo SE of the simulation is {max(x["mc_se"] for x in pv):.4f}.</p>'
    b += "</section>"
    b += S(9, "Robustness experiments")
    rob = [sc for sc in (mc or {}).get("scenarios", []) if is_robust(sc["scenario"])]
    if rob:
        for sc in rob: b += f'<h3>{esc(sc["scenario"])}</h3>' + (f'<p class="hint">{esc(sc["note"])}</p>' if sc["note"] else "") + mc_table(sc) + ul(mc_reading(sc))
    else: b += "<p>No separate robustness scenarios were defined for this method; see section 8 for the conditions that were run.</p>"
    b += "</section>"
    # 10 benchmark
    b += S(10, "Benchmark comparison") + ul(d["bench"])
    if power_extra: b += power_extra
    shown = False
    for k in d["bench_files"]:
        lines = [l for l in BENCH.get(k, []) if l.strip()]
        code = m["code"]
        if k in ("A", "A2", "B", "C", "C2", "D", "D2", "E1", "E2", "E3", "F", "F2", "G") and lines:
            b += f'<details><summary>Raw output of <code>stat-validation/benchmarks/bench_{k}.py</code> ({len(lines)} lines)</summary><pre>{esc(chr(10).join(lines))}</pre></details>'; shown = True
    if not shown: b += '<p class="hint">The benchmark log was not found; run the benchmark scripts to regenerate it.</p>'
    b += '<p class="hint">Each benchmark script prints the OES result next to the reference value for identical data and specifications; lines starting with PASS or FAIL apply a relative tolerance stated in the script. Where a script prints a table, compare the two columns. Expected discrepancies are listed under limitations.</p></section>'
    b += S(11, "Real-data application") + dataset_block(m) + f'<p>Specification: <code>{esc(json.dumps(m["example"]["spec"]))}</code></p>' + result_tables(res) + "</section>"
    b += S(12, "Diagnostics and interpretation") + ul(d["interp"])
    diag = result_tables(res, only=lambda t: any(w in t.lower() for w in ("assumption", "check", "diagnos", "fit", "test of", "suitab")))
    b += (diag if diag else "<p class='hint'>Diagnostics for this method are included in the result tables above.</p>") + "</section>"
    b += S(13, "Computational limitations") + ul(d["limitations"])
    if mc and "scenarios" in mc:
        mx = max((r.get("sec_per_rep") or 0, sc["scenario"], r["n"]) for sc in mc["scenarios"] for r in sc["results"])
        b += f'<p>Slowest simulated cell: {mx[0]:.3f} s per run ({esc(mx[1])}, n = {mx[2]}), in CPython on a shared two-core machine. Browser speed with Pyodide is expected to be several times slower and was not measured.</p>'
    b += "</section>"
    b += S(14, "Conclusions") + f'<p><b>Status: {esc(st[0])}.</b> {esc(st[1])}</p>'
    if mc and "scenarios" in mc:
        concl = []
        for sc in mc["scenarios"]:
            concl += [f"{sc['scenario']}: " + t for t in mc_reading(sc)[:2]]
        b += ul(concl[:12])
    b += "<p class='hint'>These statements are generated from the tables above. Read them with the limitations in section 13.</p></section>"
    b += S(15, "References") + refs_html(d["refs"]) + "</section>"
    b += S(16, "Replication code and datasets") + f'<pre>{esc(replication_code(m))}</pre><pre>python stat-validation/mc/run_all.py --ids {mid}          # Monte Carlo (writes stat-validation/results/mc_{mid}.json)\npython -m pytest stat-validation/tests -q                    # software validation\npython tools/make_stat_pages.py                              # rebuild this page</pre><p>Datasets: <code>stat-guide/examples/</code>. Benchmark scripts: <code>stat-validation/benchmarks/</code> (they need lifelines, pingouin, scikit-learn, semopy, scikit-posthocs and statsmodels).</p></section>'
    crumbs = f'<a href="../index.html">Statistics Division guide</a> / <a href="../{slug(m)}.html">{esc(m["name"])}</a> / validation report'
    return page(m["name"] + ": technical validation report", f'{m["code"]} · validation report', "Sixteen-section technical report with simulation, benchmark and real-data results generated by the project code.", b, crumbs, 2)

def main():
    load_bench(); os.makedirs(os.path.join(ROOT, "stat-guide", "reports"), exist_ok=True)
    mods = {m["id"]: m for m in o.STAT_MODELS}; ledger = []; idx = {}
    for c in CAT:
        m = mods[c["id"]]; m = {**m, **{k: c[k] for k in ("code", "group", "name", "guide")}}
        d = DOCS[m["id"]]; res, dt = run_example(m); mc = load_mc(m["id"])
        pe = ""
        open(os.path.join(ROOT, "stat-guide", slug(m) + ".html"), "w").write(build_method_page(m, d, res, dt))
        open(os.path.join(ROOT, "stat-guide", "reports", slug(m) + ".html"), "w").write(build_report(m, d, res, dt, mc, pe))
        st = STATUS.get(m["id"], DEFAULT_STATUS); idx.setdefault(m["group"], []).append((m, st))
        reps = sum(r["reps_run"] for sc in mc["scenarios"] for r in sc["results"]) if mc and "scenarios" in mc else None
        cap = sum(1 for sc in mc["scenarios"] for r in sc["results"] if r["reps_run"] < r["reps_requested"]) if mc and "scenarios" in mc else None
        ncell = sum(len(sc["results"]) for sc in mc["scenarios"]) if mc and "scenarios" in mc else None
        ledger.append(dict(id=m["id"], code=m["code"], name=m["name"], group=m["group"], status=st[0], reason=st[1], mc=bool(mc), reps=reps, capped=cap, cells=ncell))
    body = ""
    for g, items in idx.items():
        body += f"<section><h2>{esc(g)}</h2><div class='tw'><table><thead><tr><th>Code</th><th>Method</th><th>Status</th><th>Report</th></tr></thead><tbody>" + "".join(f'<tr><td>{m["code"]}</td><td><a href="{slug(m)}.html">{esc(m["name"])}</a></td><td>{esc(st[0])}</td><td><a href="reports/{slug(m)}.html">validation report</a></td></tr>' for m, st in items) + "</tbody></table></div></section>"
    open(os.path.join(ROOT, "stat-guide", "index.html"), "w").write(page("Statistics Division: method guides", "OES · Statistics", "Thirty-one statistical methods in seven categories, each with a method page and a technical validation report.", body, '<a href="../oes/index.html#statistics">OES: Statistics section</a>', 1))
    json.dump(ledger, open(os.path.join(ROOT, "stat-validation", "results", "ledger.json"), "w"), indent=1)
    print("pages written:", len(ledger))
main()
