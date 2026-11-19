#!/usr/bin/env python3
"""Build oes/statistics.html, the page of the OES Statistics Division, from oes/index.html.

The Statistics Division is a separate page with its own model catalogue (oes/engine/stat_catalog.json) and test references
(oes/engine/stat_expected.json); it shares the estimation worker, the engine and the look of the OES page. Run

    python3 tools/make_statistics_page.py

after changing oes/index.html or the Statistics Division registry. Every replacement is checked, so the script fails loudly
instead of producing a half-patched page."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
src = (ROOT / "oes/index.html").read_text()
html = src


def sub(old, new, count=1):
    global html
    n = html.count(old)
    if n != count:
        raise SystemExit(f"expected {count} occurrence(s) of {old[:70]!r}, found {n}")
    html = html.replace(old, new)


def regex(pattern, new, flags=re.S, literal=True):
    global html
    html, n = re.subn(pattern, (lambda m: new) if literal else new, html, count=1, flags=flags)
    if n != 1:
        raise SystemExit(f"pattern not found: {pattern[:70]!r}")


TITLE = "OES Statistics Division | Tuhin Al Mamun"
DESC = "OES Statistics Division: 31 validated statistical methods (ANOVA, nonparametric, multivariate, survival, mixed models, resampling, meta-analysis) that run on your own data, free, in your browser."
URL = "https://tuhingm.github.io/almamun/oes/statistics.html"

# ---- head and structured data
regex(r'<meta name="description" content="[^"]*">', f'<meta name="description" content="{DESC}">')
regex(r"<title>.*?</title>", f"<title>{TITLE}</title>")
sub('<link rel="canonical" href="https://tuhingm.github.io/almamun/oes/">', f'<link rel="canonical" href="{URL}">')
regex(r'<meta property="og:title" content="[^"]*"><meta property="og:description" content="[^"]*">', f'<meta property="og:title" content="{TITLE}"><meta property="og:description" content="{DESC}">')
sub('<meta property="og:url" content="https://tuhingm.github.io/almamun/oes/">', f'<meta property="og:url" content="{URL}">')
regex(r'<meta name="twitter:title" content="[^"]*"><meta name="twitter:description" content="[^"]*">', f'<meta name="twitter:title" content="{TITLE}"><meta name="twitter:description" content="{DESC}">')
sub('content="Preview card: Open Econometric Systems | Tuhin Al Mamun"', f'content="Preview card: {TITLE}"', 2)
ld = re.search(r'<script type="application/ld\+json" id="site-ld">(.*?)</script>', html, re.S).group(1)
ld2 = (ld.replace("https://tuhingm.github.io/almamun/oes/#webpage", URL + "#webpage").replace('"url":"https://tuhingm.github.io/almamun/oes/"', f'"url":"{URL}"')
         .replace("https://tuhingm.github.io/almamun/oes/#breadcrumb", URL + "#breadcrumb").replace('"name":"Open Econometric Systems | Tuhin Al Mamun"', f'"name":"{TITLE}"')
         .replace('"name":"Open Econometric Systems","item":"https://tuhingm.github.io/almamun/oes/"', '"name":"OES Statistics Division","item":"' + URL + '"'))
ld2 = re.sub(r'"description":"Open Econometric Systems: run 69[^"]*"', f'"description":"{DESC}"', ld2)
html = html.replace(ld, ld2)

# ---- navigation and hero
sub('<a href="../oes/index.html" class="current" aria-current="page">Open Econometric Systems</a>',
    '<a href="../oes/index.html">Open Econometric Systems</a><a href="statistics.html" class="current" aria-current="page">Statistics Division</a>')
regex(r'<section class="hero">.*?</section>', '''<section class="hero"><div class="wrap"><div class="kicker">Open-source statistics workspace</div><h1>OES Statistics Division</h1>
<p class="lead">Thirty-one statistical methods in seven families: analysis of variance, nonparametric tests, multivariate analysis, survival analysis, mixed-effects and longitudinal models, resampling, and meta-analysis. Choose a method, load its example dataset or upload your own CSV file, and get estimates, tests, diagnostics, charts and the exact specification to replicate them. Python runs in your browser: no installation, and your data never leaves your device.</p>
<div class="meta"><span><strong id="n-models">31</strong> methods</span><span><strong>7</strong> families</span><span>Runs in your browser: <strong>your data stays on your device</strong></span><span><strong>Free</strong> to use</span></div>
<aside class="author-note" aria-label="Author's note"><span class="an-label">Author's note</span><p>The method menu, the default specifications and the validation protocol in the Statistics Division are my own design. Each method is implemented in the OES engine and checked against reference software and published results; the validation report of each method states exactly what was compared and what was not. The estimators are standard methods from the literature.</p><span class="an-sign">— Tuhin G. M. Al Mamun, Ph.D.</span></aside>
<p class="hint" style="margin-top:14px;color:#cfe3f7">Looking for regression and time-series models? See <a href="index.html" style="text-decoration:underline;color:#fff">Open Econometric Systems (69 econometric models)</a>.</p></div></section>''')

# ---- steps
sub('<span class="stepno">01 · Model</span><h2>Choose a model</h2>', '<span class="stepno">01 · Method</span><h2>Choose a method</h2>')
regex(r'<p class="sub">Each model links to its guide.*?</p>\n<div class="tabs" id="groups"', '<p class="sub">Each method has a guide with the mathematics, assumptions, how to read the output and a replication example, and a validation report that lists what it was checked against.</p>\n<div class="tabs" id="groups"')
sub('<section class="step" id="s-ops">', '<section class="step" id="s-ops" hidden>')
sub('<span class="stepno">04 · Specification</span><h2 id="spec-title">Specify the model</h2><p class="sub" id="spec-sub">Choose a model and load data first.</p>',
    '<span class="stepno">03 · Specification</span><h2 id="spec-title">Specify the analysis</h2><p class="sub" id="spec-sub">Choose a method and load data first.</p>')
sub('<span class="stepno">05 · Results</span>', '<span class="stepno">04 · Results</span>')
sub('<span class="stepno">06 · Test suite</span><h2>Test every model in your browser</h2>', '<span class="stepno">05 · Test suite</span><h2>Test every method in your browser</h2>')
regex(r'(<span class="stepno">05 · Test suite</span><h2>.*?</h2>\n)<p class="sub">.*?</p>\n<div class="row"><button class="btn" id="test-run" type="button">Run the test suite</button><label class="hint">.*?</label></div>',
      r'\1<p class="sub">Runs each method on its guide example here, in your browser, and compares the first reported estimate with the reference value computed in the validation run. The validation reports compare the same code with scipy, statsmodels, scikit-learn, pingouin, lifelines and published results.</p>\n<div class="row"><button class="btn" id="test-run" type="button">Run the test suite</button><span class="hint">31 methods; the first run also downloads Python (about 30 MB).</span></div>', literal=False)
regex(r'<section class="step" id="s-about">.*?</section>\n\n<section class="step" id="s-cite".*?</section>', '''<section class="step" id="s-about"><span class="stepno">06 · About</span><h2>How it works</h2>
<div class="info">
<div><b>Python in the browser</b>The methods are Python (NumPy, pandas, SciPy) running in <a href="https://pyodide.org" style="text-decoration:underline" target="_blank" rel="noopener">Pyodide</a>. The first run downloads Python once (about 30 MB); later runs start at once. Your file is read on your device and never uploaded.</div>
<div><b>Validated against references</b>Every method is compared with at least one independent implementation or a published result, with the same data and options. Where no independent reference exists, the method is checked by simulation and the report says so.</div>
<div><b>Reproducible</b>Each result lists the engine version and the exact specification. Randomised methods (bootstrap, permutation tests, canonical-correlation permutation test) take a seed, so a run can be repeated.</div>
<div><b>Not a substitute for judgement</b>The output reports assumptions and diagnostics next to every estimate, but choosing a suitable method, and checking that the data meet its assumptions, remains the analyst's responsibility.</div>
<div><b>Large files</b>Everything runs on your computer, so very large datasets take longer. Mixed models, generalised mixed models and Fine-Gray regression are the most demanding.</div>
</div>
</section>''')
sub('<span><a href="https://github.com/tuhingm/open-econometric-systems" target="_blank" rel="noopener">Source code</a> · © <span id="year"></span></span>', '<span>© <span id="year"></span></span>')
sub('<span><b>Tuhin G. M. Al Mamun, Ph.D.</b> · Open Econometric Systems</span>', '<span><b>Tuhin G. M. Al Mamun, Ph.D.</b> · OES Statistics Division</span>')
sub("<h2>Load data</h2>", "<h2>Load data</h2>")
sub('<p class="sub">Use the dataset from the model\'s guide, or your own CSV file with one column per variable (and unit and time columns for panel data).</p>',
    '<p class="sub">Use the dataset from the method\'s guide, or your own CSV file with one column per variable. Long format (one row per observation) is expected, except where the method says one row per subject.</p>')

# ---- script: catalogue, no-data methods, text fields, categorical columns, forest plot, tests
sub('worker = new Worker("worker.js?v=" + (CAT ? CAT.version : "1"));', 'worker = new Worker("worker.js?v=" + (CAT ? CAT.version : "1"));')
sub('fetch("engine/catalog.json", {cache: "no-cache"})', 'fetch("engine/stat_catalog.json", {cache: "no-cache"})')
sub('fetch("engine/expected.json?v="', 'fetch("engine/stat_expected.json?v="')
sub("const pending = {};", "const pending = {};\nconst exName = m => m.example.file ? m.example.file.split(\"/\").pop() : \"no data needed\";")
sub('<small>${esc(m.example.file.split("/").pop())}${m.slow ? " · slower" : ""}</small>', '<small>${m.code} · ${esc(exName(m))}</small>')
sub('''  $("load-ex").disabled = false;
  $("load-ex").textContent = "Load the guide example (" + MODEL.example.file.split("/").pop() + ")";''', '''  $("s-data").hidden = !!MODEL.nodata;
  $("load-ex").disabled = !MODEL.example.file;
  $("load-ex").textContent = MODEL.example.file ? "Load the guide example (" + exName(MODEL) + ")" : "This method needs no data";''')
sub('''  renderFields();
  if (scroll) $("s-data").scrollIntoView({behavior: "smooth"});''', '''  renderFields(MODEL.nodata ? MODEL.example.spec : undefined);
  if (scroll) (MODEL.nodata ? $("s-spec") : $("s-data")).scrollIntoView({behavior: "smooth"});''')
sub('if (!MODEL || !DATA) { box.innerHTML = ""; $("run").disabled = true; $("spec-sub").textContent = !MODEL ? "Choose a model first." : "Load data first."; return; }',
    'if (!MODEL || (!DATA && !MODEL.nodata)) { box.innerHTML = ""; $("run").disabled = true; $("spec-sub").textContent = !MODEL ? "Choose a method first." : "Load data first."; return; }')
sub('When to use this model</a>', 'Guide and validation report</a>')
sub('''${cols.filter(c => c.numeric).map(c => `<label><input type="checkbox" value="${esc(c.name)}">${esc(c.name)}</label>`).join("")}</div></div>`;
    if (f.type === "select")''', '''${cols.filter(c => c.numeric || f.key === "factors").map(c => `<label><input type="checkbox" value="${esc(c.name)}">${esc(c.name)}</label>`).join("")}</div></div>`;
    if (f.type === "text") return `<div class="field" style="grid-column:1/-1"><label for="${id}">${esc(f.label)}${req}</label><textarea id="${id}" data-k="${f.key}" data-t="text" rows="4" spellcheck="false">${esc(f.default || "")}</textarea></div>`;
    if (f.type === "select")''')
sub('if (!MODEL || !DATA) return;\n  hideErr();', 'if (!MODEL || (!DATA && !MODEL.nodata)) return;\n  hideErr();')
sub('const m = await estimate(MODEL, DATA.text, spec);', 'const m = await estimate(MODEL, DATA ? DATA.text : "", spec);')
sub('RESULT = {model: MODEL, spec, data: DATA.name, out: m.result, seconds: m.seconds};', 'RESULT = {model: MODEL, spec, data: DATA ? DATA.name : "(no data)", out: m.result, seconds: m.seconds};')
sub('''$("charts").innerHTML = out.charts.map(svgChart).join("");''', '''$("charts").innerHTML = (out.forest ? forestPlot(out.forest) : "") + out.charts.map(svgChart).join("");''')
regex(r'  const s = JSON.stringify\(spec, null, 1\)\.replace.*?\n  \$\("code"\)\.textContent = `[^`]*`;',
      '''  const s = JSON.stringify(spec, null, 1).replace(/\\n\\s*/g, " ");
  $("code").textContent = `# Replicate this result in Python. The engine is the folder oes/engine/oes_engine of this website\\n# (engine version ${out.engine}); it needs numpy, pandas and scipy.\\nimport sys; sys.path.insert(0, "oes/engine")\\nimport oes_engine\\n\\ncsv = ${model.nodata ? '""' : `open("${RESULT.data}").read()`}\\nresult = oes_engine.run("${model.id}", csv, ${s})\\nfor table in result["tables"]:\\n    print(table["title"])\\n    for row in table["rows"]:\\n        print(row)`;''')
sub('download(`oes_${RESULT.model.id}.csv`', 'download(`oes_stat_${RESULT.model.id.replace("stat_", "")}.csv`')
sub('download(`oes_${RESULT.model.id}.json`', 'download(`oes_stat_${RESULT.model.id.replace("stat_", "")}.json`')
sub('''const text = await (await fetch("../" + m.example.file)).text();''', '''const text = m.example.file ? await (await fetch("../" + m.example.file)).text() : "";''')
sub('const list = CAT.models.filter(m => $("test-slow").checked || !m.slow);', 'const list = CAT.models;')
sub("const v = r.result.tables[e.table_index].rows[0][1];", "const v = r.result.tables[e.table_index].rows[e.row_index || 0][e.col_index || 1];")
sub('''`${ok} passed, ${bad} failed of ${list.length}` + (list.length < CAT.models.length ? ` (${CAT.models.length - list.length} slow models not run)` : "");''', '`${ok} passed, ${bad} failed of ${list.length}`;')
sub('<td>${esc(m.name)}</td><td>${esc(m.group)}</td><td class="test-run">waiting</td>', '<td>${esc(m.code + " " + m.name)}</td><td>${esc(m.group)}</td><td class="test-run">waiting</td>')

forest_js = '''
function forestPlot(F) {
  const rows = F.labels.length, rowH = 20, L = 190, R = 150, T = 34, W = 640, plotW = W - L - R, H = T + (rows + 3) * rowH + 30;
  const lo = Math.min(...F.lo, F.pooled.lo, ...(F.pred || [])), hi = Math.max(...F.hi, F.pooled.hi, ...(F.pred || []));
  const a = Math.min(lo, F.null), b = Math.max(hi, F.null), pad = (b - a) * 0.05 || 1;
  const X = v => L + (v - (a - pad)) / ((b + pad) - (a - pad)) * plotW;
  const fmt = v => NUM(+(F.log ? Math.exp(v) : v).toPrecision(3));
  let g = `<text x="${L - 8}" y="${T - 14}" text-anchor="end" font-size="10" font-weight="700" fill="#41506a">Study</text><text x="${W - R + 10}" y="${T - 14}" font-size="10" font-weight="700" fill="#41506a">${F.log ? "Ratio" : "Effect"} [CI]    Weight</text>`;
  g += `<line x1="${X(F.null)}" x2="${X(F.null)}" y1="${T - 6}" y2="${T + rows * rowH + 6}" stroke="#9aa6bd" stroke-dasharray="3 3"/>`;
  const wmax = Math.max(...F.weight);
  F.labels.forEach((lb, i) => {
    const y = T + i * rowH + rowH / 2, s = 3 + 5 * F.weight[i] / wmax;
    g += `<text x="${L - 8}" y="${y + 3}" text-anchor="end" font-size="10.5" fill="#1b2437">${esc(String(lb).slice(0, 28))}</text><line x1="${X(F.lo[i])}" x2="${X(F.hi[i])}" y1="${y}" y2="${y}" stroke="#41506a"/>` +
         `<rect x="${X(F.est[i]) - s / 2}" y="${y - s / 2}" width="${s}" height="${s}" fill="#0033a0"/><text x="${W - R + 10}" y="${y + 3}" font-size="10" fill="#41506a">${fmt(F.est[i])} [${fmt(F.lo[i])}, ${fmt(F.hi[i])}]    ${F.weight[i]}%</text>`;
  });
  const y = T + rows * rowH + rowH, d = 7;
  g += `<polygon points="${X(F.pooled.lo)},${y} ${X(F.pooled.est)},${y - d} ${X(F.pooled.hi)},${y} ${X(F.pooled.est)},${y + d}" fill="#d9822b"/>` +
       `<text x="${L - 8}" y="${y + 3}" text-anchor="end" font-size="10.5" font-weight="700" fill="#1b2437">${esc(F.pooled.label.slice(0, 26))}</text><text x="${W - R + 10}" y="${y + 3}" font-size="10" font-weight="700" fill="#1b2437">${fmt(F.pooled.est)} [${fmt(F.pooled.lo)}, ${fmt(F.pooled.hi)}]</text>`;
  if (F.pred) g += `<line x1="${X(F.pred[0])}" x2="${X(F.pred[1])}" y1="${y + rowH}" y2="${y + rowH}" stroke="#d9822b" stroke-width="2"/><text x="${L - 8}" y="${y + rowH + 3}" text-anchor="end" font-size="10.5" fill="#1b2437">Prediction interval</text><text x="${W - R + 10}" y="${y + rowH + 3}" font-size="10" fill="#41506a">[${fmt(F.pred[0])}, ${fmt(F.pred[1])}]</text>`;
  const ticks = 5; let ax = "";
  for (let k = 0; k <= ticks; k++) { const v = (a - pad) + ((b + pad) - (a - pad)) * k / ticks; ax += `<text x="${X(v)}" y="${H - 8}" text-anchor="middle" font-size="9.5" fill="#77849a">${fmt(v)}</text>`; }
  return `<div class="chart" style="grid-column:1/-1;overflow:auto"><h4>Forest plot (${esc(F.measure)})</h4><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Forest plot" style="min-width:560px">${g}${ax}</svg></div>`;
}
'''
sub("const COLORS = [", forest_js + "const COLORS = [")
sub("\n.opform{", "\ntextarea{width:100%;font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;border:1px solid #ccd6e5;border-radius:7px;padding:8px 10px;background:#fff;color:var(--ink)}\n.opform{")
# the guide links and example paths are relative to the site root, as on the OES page
(ROOT / "oes/statistics.html").write_text(html)
print("wrote oes/statistics.html", len(html), "bytes")
