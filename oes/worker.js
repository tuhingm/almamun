/* Open Econometric Systems: estimation worker.
   Runs Python (Pyodide) in a background thread so the page stays responsive while a model is estimated.
   Messages in:  {cmd: "init"} | {cmd: "run", job, model, csv, spec, packages}
   Messages out: {type: "status", text} | {type: "ready", version} | {type: "result", job, result, seconds} | {type: "error", job, error} */
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/";
const FILES = ["__init__.py", "core.py", "lp.py", "svar.py", "breaks.py", "pthreshold.py", "tar.py", "smooth.py", "_bp_cv.py", "timeseries.py", "panel.py", "causal.py", "causal_extra.py", "ml.py", "bayes.py", "bayes_extra.py", "ts_extra.py", "panel_extra.py", "statespace_var.py",
               "pvar.py", "markov.py", "diagnostics.py", "report.py", "_causal_core.py", "_panel_core.py", "_panel_tests.py", "_diff_gmm.py", "_ml_core.py"];
const BASE = ["numpy", "pandas", "scipy", "statsmodels"];
let py = null, booting = null;
const loaded = new Set();
const say = text => postMessage({type: "status", text});

async function boot(version) {
  importScripts(PYODIDE + "pyodide.js");
  say("Starting Python in your browser…");
  py = await loadPyodide({indexURL: PYODIDE});
  say("Loading NumPy, pandas, SciPy and statsmodels…");
  await py.loadPackage(BASE);
  BASE.forEach(p => loaded.add(p));
  say("Loading the OES engine…");
  py.FS.mkdirTree("/home/pyodide/oes_engine");
  for (const f of FILES) {
    const r = await fetch(`engine/oes_engine/${f}?v=${version}`);
    if (!r.ok) throw new Error(`Could not load engine file ${f} (${r.status}).`);
    py.FS.writeFile(`/home/pyodide/oes_engine/${f}`, await r.text());
  }
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide'); import oes_engine");
  return py.runPython("oes_engine.__version__");
}

async function ensure(packages) {
  const need = (packages || []).filter(p => !loaded.has(p));
  if (need.length) {
    say(`Loading ${need.join(", ")}…`);
    await py.loadPackage(need);
    need.forEach(p => loaded.add(p));
  }
}

onmessage = async e => {
  const m = e.data;
  try {
    if (!booting) booting = boot(m.version || "1");
    const version = await booting;
    if (m.cmd === "init") { postMessage({type: "ready", version}); return; }
    if (m.cmd === "run") {
      await ensure(m.packages);
      say("Estimating…");
      py.globals.set("oes_model", m.model);
      py.globals.set("oes_csv", m.csv);
      py.globals.set("oes_spec", JSON.stringify(m.spec || {}));
      const t0 = performance.now();
      const out = py.runPython("oes_engine.run_json(oes_model, oes_csv, oes_spec)");
      postMessage({type: "result", job: m.job, result: JSON.parse(out), seconds: (performance.now() - t0) / 1000});
    }
  } catch (err) {
    postMessage({type: "error", job: m.job, error: String(err && err.message || err)});
  }
};
