/* Open Econometric Systems — Statistics Division Worker
   Isolated Pyodide v0.27.2 Web Worker for the 31-method Statistics Division.
   Does NOT touch /home/pyodide/oes_engine or oes/worker.js.
   Messages in:  {cmd: "init", version} | {cmd: "run", job, model, csv, spec, packages, version} | {cmd: "validate", job, model, n_reps, version}
   Messages out: {type: "status", text} | {type: "ready", version} | {type: "result", job, result, seconds} | {type: "error", job, error} */

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/";
const BASE = ["numpy", "pandas", "scipy", "statsmodels"];

let py = null, booting = null;
const loaded = new Set();

const say = text => postMessage({type: "status", text});

async function boot(version) {
  importScripts(PYODIDE + "pyodide.js");

  say("Starting Python (Pyodide v0.27.2) in background thread…");
  py = await loadPyodide({indexURL: PYODIDE});

  say("Loading NumPy, pandas, SciPy and statsmodels…");
  await py.loadPackage(BASE);
  BASE.forEach(p => loaded.add(p));

  say("Loading OES Statistics Division engine…");
  const r = await fetch(`engine/statistics_division.py?v=${version}`);
  if (!r.ok) {
    throw new Error(`Could not load engine/statistics_division.py (${r.status}).`);
  }
  py.FS.writeFile("/home/pyodide/statistics_division.py", await r.text());

  py.runPython(
    "import sys; sys.path.insert(0, '/home/pyodide'); import statistics_division"
  );

  return py.runPython("statistics_division.__version__");
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
    if (!booting) booting = boot(m.version || "1.0.0");
    const version = await booting;

    if (m.cmd === "init") {
      postMessage({type: "ready", version});
      return;
    }

    if (m.cmd === "run") {
      await ensure(m.packages);
      say("Running statistical estimation & diagnostics…");

      py.globals.set("stat_model", m.model);
      py.globals.set("stat_csv", m.csv);
      py.globals.set("stat_spec", JSON.stringify(m.spec || {}));

      const t0 = performance.now();
      const out = py.runPython(
        "statistics_division.run_json(stat_model, stat_csv, stat_spec)"
      );

      postMessage({
        type: "result",
        job: m.job,
        result: JSON.parse(out),
        seconds: (performance.now() - t0) / 1000
      });
      return;
    }

    if (m.cmd === "validate") {
      await ensure(m.packages);
      say(`Executing OMVP Validation Protocol for ${m.model}…`);

      py.globals.set("val_model", m.model);
      py.globals.set("val_reps", m.n_reps || 200);

      const t0 = performance.now();
      const out = py.runPython(
        "statistics_division.run_omvp_json(val_model, int(val_reps))"
      );

      postMessage({
        type: "result",
        job: m.job,
        result: JSON.parse(out),
        seconds: (performance.now() - t0) / 1000
      });
    }
  } catch (err) {
    postMessage({
      type: "error",
      job: m.job,
      error: String(err && err.message || err)
    });
  }
};
