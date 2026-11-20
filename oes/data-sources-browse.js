/* OES Data Sources — browse every dataset and series of ECB, Eurostat, OECD and BIS (SDMX 2.1, browser-only) */
(function () {
  "use strict";
  const $ = (r, s) => r.querySelector(s);
  const esc = v => String(v == null ? "" : v).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const NS = "*";
  const tags = (n, name) => Array.from(n.getElementsByTagNameNS(NS, name));
  const first = (n, name) => n.getElementsByTagNameNS(NS, name)[0] || null;
  const nameOf = n => { const ns = Array.from(n.children).filter(c => c.localName === "Name"); const en = ns.find(c => c.getAttribute("xml:lang") === "en") || ns[0]; return en ? en.textContent.trim() : ""; };

  const PROV = {
    ecb: {
      name: "ECB", site: "https://data.ecb.europa.eu/",
      list: "https://data-api.ecb.europa.eu/service/dataflow/ECB",
      struct: f => "https://data-api.ecb.europa.eu/service/dataflow/ECB/" + f.id + "/latest?references=descendants",
      data: (f, key, s) => "https://data-api.ecb.europa.eu/service/data/" + f.id + "/" + key + "?startPeriod=" + s + "&format=csvdata"
    },
    eurostat: {
      name: "Eurostat", site: "https://ec.europa.eu/eurostat/",
      list: "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/dataflow/ESTAT/all/latest?detail=allstubs",
      struct: f => "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/dataflow/ESTAT/" + f.id + "/latest?references=descendants",
      data: (f, key, s) => "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/" + f.id + "/" + key + "?startPeriod=" + s + "&format=SDMX-CSV",
      big: true
    },
    oecd: {
      name: "OECD", site: "https://data-explorer.oecd.org/",
      list: "https://sdmx.oecd.org/public/rest/dataflow/all/all/latest?detail=allstubs",
      struct: f => "https://sdmx.oecd.org/public/rest/dataflow/" + f.agency + "/" + f.id + "/" + f.version + "?references=descendants",
      data: (f, key, s) => "https://sdmx.oecd.org/public/rest/data/" + f.agency + "," + f.id + "," + f.version + "/" + key + "?startPeriod=" + s + "&format=csvfilewithlabels"
    },
    imf: {
      name: "IMF", site: "https://data.imf.org/", xml: true,
      list: "https://api.imf.org/external/sdmx/2.1/dataflow/all/all/latest?detail=allstubs",
      struct: f => "https://api.imf.org/external/sdmx/2.1/dataflow/" + f.agency + "/" + f.id + "/" + f.version + "?references=descendants",
      data: (f, key, s) => "https://api.imf.org/external/sdmx/2.1/data/" + f.agency + "," + f.id + "," + f.version + "/" + key + "?startPeriod=" + s
    },
    bis: {
      name: "BIS", site: "https://data.bis.org/",
      list: "https://stats.bis.org/api/v2/structure/dataflow/BIS/*/*?format=sdmx-2.1",
      struct: f => "https://stats.bis.org/api/v2/structure/dataflow/BIS/" + f.id + "/" + f.version + "?references=descendants&format=sdmx-2.1",
      data: (f, key, s) => "https://stats.bis.org/api/v2/data/dataflow/BIS/" + f.id + "/" + f.version + "/" + key + "?startPeriod=" + s + "&format=csv"
    }
  };

  async function getText(url, ms) {
    const ctl = new AbortController(), to = setTimeout(() => ctl.abort(), ms || 120000);
    try {
      const r = await fetch(url, { mode: "cors", credentials: "omit", signal: ctl.signal });
      const t = await r.text();
      if (r.status === 404 || r.status === 204 || (r.ok && !t.trim())) return "";
      if (!r.ok) throw new Error("The provider answered HTTP " + r.status + ". Narrow the selection (pick fewer values) and try again.");
      return t;
    } catch (e) {
      if (e.name === "AbortError" || e instanceof TypeError) throw new Error("The request did not finish (timeout, size or provider restriction). Pick fewer values, a later start year, or try again.");
      throw e;
    } finally { clearTimeout(to); }
  }
  async function cached(key, make) {
    const c = window.OESDataSources && window.OESDataSources.cache;
    if (c) { const hit = await c.get(key); if (hit) return hit; }
    const v = await make();
    if (c) { try { await c.set(key, v); } catch (e) { /* cache optional */ } }
    return v;
  }
  async function listFlows(k) {
    return cached("browse-list-" + k + "-v1", async () => {
      const doc = new DOMParser().parseFromString(await getText(PROV[k].list, 180000), "application/xml");
      return tags(doc, "Dataflow").map(d => ({ id: d.getAttribute("id"), agency: d.getAttribute("agencyID"), version: d.getAttribute("version"), name: nameOf(d) })).filter(f => f.id);
    });
  }
  async function loadStructure(k, f) {
    const doc = new DOMParser().parseFromString(await getText(PROV[k].struct(f), 180000), "application/xml");
    const lists = new Map();
    for (const cl of tags(doc, "Codelist")) lists.set(cl.getAttribute("id"), tags(cl, "Code").map(c => ({ id: c.getAttribute("id"), name: nameOf(c) })));
    const concepts = new Map();
    for (const c of tags(doc, "Concept")) { const en = first(c, "Enumeration"), r = en && first(en, "Ref"); if (r) concepts.set(c.getAttribute("id"), r.getAttribute("id")); }
    const dsd = first(doc, "DataStructure");
    if (!dsd) throw new Error("This dataset has no structure information.");
    const dl = first(dsd, "DimensionList");
    const dims = tags(dl, "Dimension").filter(d => d.parentNode === dl).map(d => {
      const en = first(d, "Enumeration"), ref = en && first(en, "Ref");
      let codes = ref ? (lists.get(ref.getAttribute("id")) || []) : [];
      if (!codes.length) {                                   // IMF: the code list is declared on the concept, not on the dimension
        const ci = first(d, "ConceptIdentity"), cr = ci && first(ci, "Ref"), cid = cr && cr.getAttribute("id");
        if (cid && concepts.has(cid)) codes = lists.get(concepts.get(cid)) || [];
        if (!codes.length) {                                 // last resort: match the code list by name (CL_<DIMENSION>)
          const did = d.getAttribute("id"), alt = did === "FREQUENCY" ? "CL_FREQ" : "CL_" + did;
          const hit = lists.has(alt) ? alt : Array.from(lists.keys()).find(k => k.endsWith("_" + did));
          if (hit) codes = lists.get(hit);
        }
      }
      return { id: d.getAttribute("id"), pos: +d.getAttribute("position") || 0, codes };
    }).sort((a, b) => a.pos - b.pos);
    if (!dims.length) throw new Error("This dataset has no selectable dimensions.");
    return dims;
  }
  function parseCsv(t) {
    const rows = []; let r = [], f = "", q = false;
    for (let i = 0; i < t.length; i++) {
      const c = t[i];
      if (q) { if (c === '"') { if (t[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
      else if (c === '"') q = true; else if (c === ",") { r.push(f); f = ""; }
      else if (c === "\n") { r.push(f); rows.push(r); r = []; f = ""; } else if (c !== "\r") f += c;
    }
    if (f || r.length) { r.push(f); rows.push(r); }
    return rows;
  }
  function xmlRows(text, dims) {
    const doc = new DOMParser().parseFromString(text, "application/xml"), ids = dims.map(d => d.id), out = [[...ids, "TIME_PERIOD", "OBS_VALUE"]];
    for (const se of tags(doc, "Series")) {
      const base = ids.map(i => se.getAttribute(i) || "");
      for (const o of Array.from(se.children).filter(c => c.localName === "Obs")) out.push([...base, o.getAttribute("TIME_PERIOD") || "", o.getAttribute("OBS_VALUE") || ""]);
    }
    return out;
  }
  const csvOut = (h, rows) => { const q = v => '"' + String(v == null ? "" : v).replace(/"/g, '""') + '"'; return [h.map(q).join(","), ...rows.map(r => r.map(q).join(","))].join("\r\n"); };
  function save(name, content, type) { const u = URL.createObjectURL(new Blob([content], { type })); const a = document.createElement("a"); a.href = u; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(u), 1000); }

  function mount(root, k) {
    const P = PROV[k]; let flows = null, flow = null, dims = null, last = null;
    root.innerHTML =
      '<details class="ds-browse" open style="border:2px solid #0f9d6e;border-radius:10px;padding:12px;background:#f6fdf9"><summary style="cursor:pointer;font-size:15px"><b>All ' + esc(P.name) + ' datasets and series</b> <span class="sub">(click "Load dataset list" to search everything the provider publishes)</span></summary>' +
      '<div class="ds-controls"><div class="field"><label>Find a dataset</label><input type="search" class="b-find" placeholder="e.g. inflation, GDP, bank, trade…" autocomplete="off"></div></div>' +
      '<div class="row ds-actions"><button type="button" class="btn secondary b-load">Load dataset list</button><span class="sub b-count"></span></div>' +
      '<div class="ds-catalog b-flows" style="max-height:220px"><p class="sub">Load the list to search all datasets.</p></div>' +
      '<div class="b-dims"></div><div class="status b-status" role="status" aria-live="polite"></div>' +
      '<div class="b-result ds-result hidden"><h3>Retrieved observations</h3><p class="sub b-meta"></p><div class="b-prev"></div>' +
      '<div class="row ds-actions"><button type="button" class="btn secondary b-csv">Download CSV</button><button type="button" class="btn secondary b-xlsx">Download Excel</button><button type="button" class="btn b-send">Send to Cleaning Lab</button></div></div></details>';
    const q = s => $(root, s);
    const status = (m, err) => { const el = q(".b-status"); el.textContent = m; el.className = "status b-status show" + (err ? " error" : ""); };
    function showFlows() {
      if (!flows) return;
      const s = q(".b-find").value.trim().toLowerCase();
      const m = flows.filter(f => !s || (f.id + " " + f.name).toLowerCase().includes(s));
      q(".b-count").textContent = m.length.toLocaleString() + " of " + flows.length.toLocaleString() + " datasets";
      q(".b-flows").innerHTML = m.slice(0, 80).map(f => '<div class="ds-catalog-item"><span><b>' + esc(f.name || f.id) + '</b><small>' + esc(f.id) + '</small></span><button type="button" class="btn secondary ds-add" data-id="' + esc(f.id) + '">Open</button></div>').join("") || '<p class="sub">No dataset matches.</p>';
    }
    q(".b-load").onclick = async () => {
      q(".b-load").disabled = true; status("Loading the " + P.name + " dataset list" + (P.big ? " (large, may take up to a minute the first time)" : "") + "…");
      try { flows = await listFlows(k); showFlows(); status(flows.length.toLocaleString() + " datasets available."); }
      catch (e) { status(e.message, true); } finally { q(".b-load").disabled = false; }
    };
    q(".b-find").oninput = showFlows;
    q(".b-flows").onclick = async e => {
      const b = e.target.closest("[data-id]"); if (!b) return;
      flow = flows.find(f => f.id === b.dataset.id); dims = null; q(".b-result").classList.add("hidden");
      q(".b-dims").innerHTML = ""; status("Loading structure of " + flow.id + "…");
      try { dims = await loadStructure(k, flow); renderDims(); status("Choose values (leave a box empty to include everything), then fetch."); }
      catch (err) { status(err.message, true); }
    };
    function renderDims() {
      q(".b-dims").innerHTML = '<h3>' + esc(flow.name || flow.id) + '</h3><div class="ds-controls">' + dims.map((d, i) =>
        '<div class="field"><label>' + esc(d.id) + ' <small>(' + d.codes.length.toLocaleString() + ' values)</small></label>' +
        (d.codes.length > 8 ? '<input type="search" data-f="' + i + '" placeholder="filter…" aria-label="Filter ' + esc(d.id) + '">' : "") +
        '<select multiple size="' + Math.min(6, Math.max(3, d.codes.length)) + '" data-d="' + i + '" aria-label="' + esc(d.id) + '"></select></div>').join("") + '</div>' +
        '<div class="ds-controls"><div class="field"><label>Start year</label><input type="number" class="b-start" value="2000" min="1900" max="2100"></div><div class="field"><label>End year</label><input type="number" class="b-end" value="' + new Date().getFullYear() + '" min="1900" max="2100"></div></div>' +
        '<div class="row ds-actions"><button type="button" class="btn b-fetch">Fetch data</button></div>';
      dims.forEach((d, i) => fillSel(i, ""));
      q(".b-dims").oninput = e => { const f = e.target.dataset.f; if (f != null) fillSel(+f, e.target.value); };
      q(".b-fetch").onclick = run;
    }
    function fillSel(i, text) {
      const sel = q('select[data-d="' + i + '"]'), keep = new Set(Array.from(sel.selectedOptions).map(o => o.value)), t = text.toLowerCase();
      const m = dims[i].codes.filter(c => keep.has(c.id) || !t || (c.id + " " + c.name).toLowerCase().includes(t));
      sel.innerHTML = m.slice(0, 300).map(c => '<option value="' + esc(c.id) + '"' + (keep.has(c.id) ? " selected" : "") + '>' + esc(c.id) + (c.name ? " — " + esc(c.name) : "") + '</option>').join("");
    }
    async function run() {
      const key = dims.map((d, i) => Array.from(q('select[data-d="' + i + '"]').selectedOptions).map(o => o.value).join("+")).join(".");
      const s = +q(".b-start").value, e = +q(".b-end").value;
      if (!(s <= e)) return status("Start year must not be after end year.", true);
      q(".b-fetch").disabled = true; q(".b-result").classList.add("hidden"); status("Fetching…");
      try {
        const url = P.data(flow, key, s), text = await getText(url, 180000);
        if (!text) return status("The provider returned no data for this selection.", true);
        const rows = P.xml ? xmlRows(text, dims) : parseCsv(text), h = rows[0] || [], ti = h.indexOf("TIME_PERIOD"), vi = h.indexOf("OBS_VALUE");
        if (ti < 0 || vi < 0) throw new Error("Unexpected response format.");
        const dimCols = dims.map(d => h.indexOf(d.id)).filter(i => i >= 0);
        const headers = [...dimCols.map(i => h[i]), "time", "value"];
        const out = [];
        for (const r of rows.slice(1)) {
          if (r.length < h.length - 3 || !r[ti]) continue;
          if (parseInt(r[ti].slice(0, 4), 10) > e) continue;
          const v = r[vi] === "" ? null : Number(r[vi]);
          out.push([...dimCols.map(i => r[i]), String(r[ti]).replace(/^(\d{4})-M(\d{2})$/, "$1-$2"), Number.isFinite(v) ? v : null]);
          if (out.length > 500000) throw new Error("More than 500,000 observations. Pick fewer values or a later start year.");
        }
        if (!out.length) return status("No observations in " + s + "–" + e + ".", true);
        last = { headers, rows: out, meta: { provider: P.name, provider_id: k, dataset: flow.id, fetched_at: new Date().toISOString(), start_year: s, end_year: e, series: [{ id: flow.id, title: flow.name }], entities: [], query_urls: [url], warnings: [], records: out.length, license: "See " + P.site, citation: P.name + ". " + (flow.name || flow.id) + " (" + flow.id + "). Retrieved " + new Date().toISOString().slice(0, 10) + " from " + P.site } };
        q(".b-meta").textContent = out.length.toLocaleString() + " observations · one row per series and period (all dimension codes kept)";
        q(".b-prev").innerHTML = '<div class="table-wrap"><table class="diagnostic-table"><thead><tr>' + headers.map(x => "<th>" + esc(x) + "</th>").join("") + "</tr></thead><tbody>" + out.slice(0, 25).map(r => "<tr>" + r.map(v => "<td>" + (v == null ? "<em>Missing</em>" : esc(v)) + "</td>").join("") + "</tr>").join("") + "</tbody></table></div>";
        q(".b-result").classList.remove("hidden"); status("Done: " + out.length.toLocaleString() + " observations.");
      } catch (err) { status(err.message, true); } finally { q(".b-fetch").disabled = false; }
    }
    q(".b-csv").onclick = () => last && save("oes_" + k + "_" + flow.id + ".csv", "﻿" + csvOut(last.headers, last.rows), "text/csv;charset=utf-8");
    q(".b-xlsx").onclick = () => { if (!last) return; if (!window.XLSX) return status("Excel export library unavailable. Use CSV.", true); if (last.rows.length > 1000000) return; const b = XLSX.utils.book_new(); XLSX.utils.book_append_sheet(b, XLSX.utils.aoa_to_sheet([last.headers, ...last.rows]), flow.id.slice(0, 28)); XLSX.writeFile(b, "oes_" + k + "_" + flow.id + ".xlsx"); };
    q(".b-send").onclick = () => { if (!last || typeof window.OESDataCleaningLoad !== "function") return; window.OESDataCleaningLoad([last.headers, ...last.rows], k + "_" + flow.id + ".csv", last.meta); document.getElementById("uploadPanel").scrollIntoView({ behavior: "smooth", block: "start" }); };
  }
  window.OESBrowse = { mount, providers: PROV, listFlows, loadStructure };
})();
