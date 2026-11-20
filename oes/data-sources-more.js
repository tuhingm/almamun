/* OES Data Sources — ECB, Eurostat, OECD, BIS (browser-only, curated indicators, SDMX CSV) */
(function () {
  "use strict";
  const $ = id => document.getElementById(id);
  const esc = v => String(v == null ? "" : v).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  /* ---------- country lists ---------- */
  const C = [
    ["AU", "AUS", "Australia"], ["AT", "AUT", "Austria"], ["BE", "BEL", "Belgium"], ["BR", "BRA", "Brazil"], ["CA", "CAN", "Canada"],
    ["CL", "CHL", "Chile"], ["CN", "CHN", "China"], ["CZ", "CZE", "Czechia"], ["DK", "DNK", "Denmark"], ["EE", "EST", "Estonia"],
    ["FI", "FIN", "Finland"], ["FR", "FRA", "France"], ["DE", "DEU", "Germany"], ["GR", "GRC", "Greece"], ["HU", "HUN", "Hungary"],
    ["IN", "IND", "India"], ["ID", "IDN", "Indonesia"], ["IE", "IRL", "Ireland"], ["IL", "ISR", "Israel"], ["IT", "ITA", "Italy"],
    ["JP", "JPN", "Japan"], ["KR", "KOR", "Korea"], ["LV", "LVA", "Latvia"], ["LT", "LTU", "Lithuania"], ["MX", "MEX", "Mexico"],
    ["NL", "NLD", "Netherlands"], ["NZ", "NZL", "New Zealand"], ["NO", "NOR", "Norway"], ["PL", "POL", "Poland"], ["PT", "PRT", "Portugal"],
    ["RU", "RUS", "Russia"], ["SK", "SVK", "Slovakia"], ["SI", "SVN", "Slovenia"], ["ZA", "ZAF", "South Africa"], ["ES", "ESP", "Spain"],
    ["SE", "SWE", "Sweden"], ["CH", "CHE", "Switzerland"], ["TR", "TUR", "Türkiye"], ["GB", "GBR", "United Kingdom"], ["US", "USA", "United States"]
  ];
  const EU = ["AT", "BE", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV", "LT", "NL", "PL", "PT", "SK", "SI", "ES", "SE"];
  const list = (idx, only) => C.filter(c => !only || only.includes(c[0])).map(c => ({
    code: only && idx === 0 && c[0] === "GR" ? "EL" : c[idx], name: c[2]
  }));
  const CUR = [["USD", "US dollar"], ["GBP", "Pound sterling"], ["JPY", "Japanese yen"], ["CNY", "Chinese yuan"], ["KRW", "Korean won"], ["CHF", "Swiss franc"], ["INR", "Indian rupee"], ["CAD", "Canadian dollar"], ["AUD", "Australian dollar"], ["BRL", "Brazilian real"], ["TRY", "Turkish lira"], ["PLN", "Polish zloty"], ["SEK", "Swedish krona"], ["NOK", "Norwegian krone"], ["IDR", "Indonesian rupiah"]].map(x => ({ code: x[0], name: x[1] }));

  /* ---------- providers ---------- */
  const SDMX = "https://sdmx.oecd.org/public/rest/data/";
  const BIS = "https://stats.bis.org/api/v2/data/dataflow/BIS/";
  const ECB = "https://data-api.ecb.europa.eu/service/data/";
  const EUS = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/";
  const P = {
    ecb: {
      name: "European Central Bank", site: "https://data.ecb.europa.eu/", licence: "ECB statistics: reuse permitted with source acknowledgement.",
      entities: CUR, entityLabel: "Currencies (for exchange rates)",
      indicators: [
        { id: "ECB_EXR", label: "Euro exchange rate (monthly average, units per EUR)", freq: "M", needs: true, url: (e, s) => ECB + "EXR/M." + e.join("+") + ".EUR.SP00.A?startPeriod=" + s + "&format=csvdata", ent: "CURRENCY" },
        { id: "ECB_POLICY_RATE", label: "Euro area main refinancing rate (daily, %)", freq: "D", fixed: { code: "U2", name: "Euro area" }, url: (e, s) => ECB + "FM/B.U2.EUR.4F.KR.MRR_FR.LEV?startPeriod=" + s + "&format=csvdata", ent: "REF_AREA" },
        { id: "ECB_HICP_INFL", label: "Euro area inflation, HICP (monthly, % y/y)", freq: "M", fixed: { code: "U2", name: "Euro area" }, url: (e, s) => ECB + "ICP/M.U2.N.000000.4.ANR?startPeriod=" + s + "&format=csvdata", ent: "REF_AREA" },
        { id: "ECB_UNEMP", label: "Euro area unemployment rate (monthly, %)", freq: "M", fixed: { code: "I9", name: "Euro area" }, url: (e, s) => ECB + "LFSI/M.I9.S.UNEHRT.TOTAL0.15_74.T?startPeriod=" + s + "&format=csvdata", ent: "REF_AREA" },
        { id: "ECB_GDP_GROWTH", label: "Euro area real GDP growth (quarterly, % y/y)", freq: "Q", fixed: { code: "I9", name: "Euro area" }, url: (e, s) => ECB + "MNA/Q.Y.I9.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.GY?startPeriod=" + s + "&format=csvdata", ent: "REF_AREA" },
        { id: "ECB_CORE_INFL", label: "Euro area core inflation, HICP excl. energy and food (monthly, % y/y)", freq: "M", fixed: { code: "U2", name: "Euro area" }, url: (e, s) => ECB + "ICP/M.U2.N.XEF000.4.ANR?startPeriod=" + s + "&format=csvdata", ent: "REF_AREA" },
        { id: "ECB_EURIBOR_3M", label: "3-month Euribor (monthly, %)", freq: "M", fixed: { code: "U2", name: "Euro area" }, url: (e, s) => ECB + "FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA?startPeriod=" + s + "&format=csvdata", ent: "REF_AREA" },
        { id: "ECB_M3_GROWTH", label: "Euro area M3 money supply growth (monthly, % y/y)", freq: "M", fixed: { code: "U2", name: "Euro area" }, url: (e, s) => ECB + "BSI/M.U2.Y.V.M30.X.I.U2.2300.Z01.A?startPeriod=" + s + "&format=csvdata", ent: "REF_AREA" },
        { id: "ECB_LOANS_NFC", label: "Euro area bank loans to non-financial corporations (monthly, % y/y)", freq: "M", fixed: { code: "U2", name: "Euro area" }, url: (e, s) => ECB + "BSI/M.U2.Y.U.A20T.A.I.U2.2240.Z01.A?startPeriod=" + s + "&format=csvdata", ent: "REF_AREA" },
      ]
    },
    eurostat: {
      name: "Eurostat", site: "https://ec.europa.eu/eurostat/", licence: "Eurostat: reuse authorised with acknowledgement of the source (CC BY 4.0).",
      entities: list(0, EU), entityLabel: "Countries (EU members)",
      indicators: [
        { id: "ES_REAL_GDP", label: "Real GDP (annual, chain-linked 2010 million euro)", freq: "A", needs: true, url: (e, s) => EUS + "nama_10_gdp/A.CLV10_MEUR.B1GQ." + e.join("+") + "?startPeriod=" + s + "&format=SDMX-CSV", ent: "geo" },
        { id: "ES_UNEMP", label: "Unemployment rate (annual, % of active population, 15–74)", freq: "A", needs: true, url: (e, s) => EUS + "une_rt_a/A.Y15-74.PC_ACT.T." + e.join("+") + "?startPeriod=" + s + "&format=SDMX-CSV", ent: "geo" },
        { id: "ES_HICP_INFL", label: "Inflation, HICP all items (monthly, % y/y)", freq: "M", needs: true, url: (e, s) => EUS + "prc_hicp_manr/M.RCH_A.CP00." + e.join("+") + "?startPeriod=" + s + "&format=SDMX-CSV", ent: "geo" },
        { id: "ES_GDP_CURRENT", label: "GDP at current prices (annual, million euro)", freq: "A", needs: true, url: (e, s) => EUS + "nama_10_gdp/A.CP_MEUR.B1GQ." + e.join("+") + "?startPeriod=" + s + "&format=SDMX-CSV", ent: "geo" },
        { id: "ES_GDP_GROWTH", label: "Real GDP growth (annual, % change on previous year)", freq: "A", needs: true, url: (e, s) => EUS + "nama_10_gdp/A.CLV_PCH_PRE.B1GQ." + e.join("+") + "?startPeriod=" + s + "&format=SDMX-CSV", ent: "geo" },
        { id: "ES_GDP_PC", label: "Real GDP per capita (annual, chain-linked 2010 euro)", freq: "A", needs: true, url: (e, s) => EUS + "nama_10_pc/A.CLV10_EUR_HAB.B1GQ." + e.join("+") + "?startPeriod=" + s + "&format=SDMX-CSV", ent: "geo" },
        { id: "ES_GOV_DEBT", label: "Government gross debt (annual, % of GDP)", freq: "A", needs: true, url: (e, s) => EUS + "gov_10dd_edpt1/A.PC_GDP.S13.GD." + e.join("+") + "?startPeriod=" + s + "&format=SDMX-CSV", ent: "geo" },
        { id: "ES_GOV_BALANCE", label: "Government deficit/surplus (annual, % of GDP)", freq: "A", needs: true, url: (e, s) => EUS + "gov_10dd_edpt1/A.PC_GDP.S13.B9." + e.join("+") + "?startPeriod=" + s + "&format=SDMX-CSV", ent: "geo" },
        { id: "ES_IND_PROD", label: "Industrial production index (monthly, 2021=100, calendar adjusted)", freq: "M", needs: true, url: (e, s) => EUS + "sts_inpr_m/M.PRD.B-D.CA.I21." + e.join("+") + "?startPeriod=" + s + "&format=SDMX-CSV", ent: "geo" },
      ]
    },
    oecd: {
      name: "OECD", site: "https://data-explorer.oecd.org/", licence: "OECD data: free use with attribution under the OECD terms and conditions.",
      entities: list(1), entityLabel: "Countries",
      indicators: [
        { id: "OECD_UNEMP", label: "Unemployment rate (monthly, % of labour force)", freq: "M", needs: true, url: (e, s) => SDMX + "OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,1.0/" + e.join("+") + "..._Z.Y._T.Y_GE15..M?startPeriod=" + s + "&format=csvfilewithlabels", ent: "REF_AREA" },
        { id: "OECD_CPI_INFL", label: "Consumer price inflation (monthly, % y/y)", freq: "M", needs: true, url: (e, s) => SDMX + "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0/" + e.join("+") + ".M.N.CPI.PA._T.N.GY?startPeriod=" + s + "&format=csvfilewithlabels", ent: "REF_AREA" },
        { id: "OECD_CLI", label: "Composite leading indicator (monthly, amplitude adjusted)", freq: "M", needs: true, url: (e, s) => SDMX + "OECD.SDD.STES,DSD_STES@DF_CLI,4.1/" + e.join("+") + ".M.LI...AA...H?startPeriod=" + s + "&format=csvfilewithlabels", ent: "REF_AREA" }
      ]
    },
    bis: {
      name: "Bank for International Settlements", site: "https://data.bis.org/", licence: "BIS statistics: free use with acknowledgement of the BIS as source.",
      entities: list(0), entityLabel: "Countries",
      indicators: [
        { id: "BIS_POLICY_RATE", label: "Central bank policy rate (monthly, %)", freq: "M", needs: true, url: (e, s) => BIS + "WS_CBPOL/1.0/M." + e.join("+") + "?startPeriod=" + s + "&format=csv", ent: "REF_AREA" },
        { id: "BIS_REER", label: "Real effective exchange rate, broad basket (monthly, index)", freq: "M", needs: true, url: (e, s) => BIS + "WS_EER/1.0/M.R.B." + e.join("+") + "?startPeriod=" + s + "&format=csv", ent: "REF_AREA" },
        { id: "BIS_CPI", label: "Consumer prices (monthly, index)", freq: "M", needs: true, url: (e, s) => BIS + "WS_LONG_CPI/1.0/M." + e.join("+") + ".628?startPeriod=" + s + "&format=csv", ent: "REF_AREA" },
        { id: "BIS_NEER", label: "Nominal effective exchange rate, broad basket (monthly, index)", freq: "M", needs: true, url: (e, s) => BIS + "WS_EER/1.0/M.N.B." + e.join("+") + "?startPeriod=" + s + "&format=csv", ent: "REF_AREA" },
        { id: "BIS_PROP_PRICES", label: "Residential property prices (quarterly, index)", freq: "Q", needs: true, url: (e, s) => BIS + "WS_SPP/1.0/Q." + e.join("+") + ".N.628?startPeriod=" + s + "&format=csv", ent: "REF_AREA" },
        { id: "BIS_USD_RATE", label: "Exchange rate against the US dollar (monthly average, local currency per USD)", freq: "M", needs: true, url: (e, s) => BIS + "WS_XRU/1.0/M." + e.join("+") + "..A?startPeriod=" + s + "&format=csv", ent: "REF_AREA" },
        { id: "BIS_DEBT_SERVICE", label: "Private non-financial debt service ratio (quarterly, %)", freq: "Q", needs: true, url: (e, s) => BIS + "WS_DSR/1.0/Q." + e.join("+") + ".P?startPeriod=" + s + "&format=csv", ent: "BORROWERS_CTY" },
        { id: "BIS_CREDIT_GAP", label: "Credit-to-GDP gap (quarterly, percentage points)", freq: "Q", needs: true, url: (e, s) => BIS + "WS_CREDIT_GAP/1.0/Q." + e.join("+") + ".P.A.C?startPeriod=" + s + "&format=csv", ent: "BORROWERS_CTY" },
      ]
    }
  };

  /* ---------- fetch + parse ---------- */
  function parseCsv(t) {
    const rows = []; let r = [], f = "", q = false;
    for (let i = 0; i < t.length; i++) {
      const c = t[i];
      if (q) { if (c === '"') { if (t[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
      else if (c === '"') q = true;
      else if (c === ",") { r.push(f); f = ""; }
      else if (c === "\n") { r.push(f); rows.push(r); r = []; f = ""; }
      else if (c !== "\r") f += c;
    }
    if (f || r.length) { r.push(f); rows.push(r); }
    return rows;
  }
  async function get(url) {
    const ctl = new AbortController(); const to = setTimeout(() => ctl.abort(), 45000);
    try {
      const res = await fetch(url, { mode: "cors", credentials: "omit", signal: ctl.signal });
      const text = await res.text();
      if (res.status === 404 || res.status === 204 || (res.ok && !text.trim())) return "";
      if (!res.ok) throw new Error("The provider answered HTTP " + res.status + ". Try fewer countries or a later start year.");
      return text;
    } catch (e) {
      if (e.name === "AbortError" || e instanceof TypeError) throw new Error("The request did not complete (network, timeout or provider restriction). Try again, or narrow the selection.");
      throw e;
    } finally { clearTimeout(to); }
  }
  const year = t => parseInt(String(t).slice(0, 4), 10);
  async function fetchProvider(key, inds, ents, start, end, onStatus) {
    const prov = P[key], records = [], urls = [], warnings = [];
    for (const ind of inds) {
      const codes = ind.fixed ? [ind.fixed.code] : ents.map(e => e.code);
      onStatus("Fetching " + ind.label + "…");
      const url = ind.url(codes, start);
      urls.push(url);
      const text = await get(url);
      if (!text) { warnings.push(ind.label + ": the provider returned no data for this selection."); continue; }
      const rows = parseCsv(text), h = rows[0] || [];
      const ci = h.indexOf(ind.ent), ti = h.indexOf("TIME_PERIOD"), vi = h.indexOf("OBS_VALUE");
      if (ci < 0 || ti < 0 || vi < 0) throw new Error("Unexpected response format from " + prov.name + ".");
      const names = new Map((ind.fixed ? [ind.fixed] : prov.entities).map(e => [e.code, e.name]));
      const seen = new Set(); let n = 0;
      for (const r of rows.slice(1)) {
        if (r.length < h.length - 2 || !r[ti]) continue;
        const y = year(r[ti]); if (y > end) continue;
        const v = r[vi] === "" ? null : Number(r[vi]);
        const code = r[ci]; seen.add(code); n++;
        records.push({ entity: names.get(code) || code, entity_code: code, time: r[ti], series_id: ind.id, value: Number.isFinite(v) ? v : null });
      }
      if (!ind.fixed) {
        const miss = codes.filter(c => !seen.has(c));
        if (miss.length) warnings.push(ind.label + ": no data returned for " + miss.join(", ") + ".");
      }
      if (!n) warnings.push(ind.label + ": no observations in " + start + "–" + end + ".");
    }
    if (inds.some(i => i.fixed)) warnings.push("Euro-area series describe the euro area as a whole, not individual countries.");
    return { records, urls, warnings };
  }
  function shape(records, inds) {
    const freqs = new Set(inds.map(i => i.freq));
    if (freqs.size !== 1) return { headers: ["entity", "entity_code", "time", "series_id", "value"], rows: records.map(r => [r.entity, r.entity_code, r.time, r.series_id, r.value]), layout: "long" };
    const ids = inds.map(i => i.id), map = new Map();
    for (const r of records) {
      const k = r.entity_code + "\u0001" + r.time;
      if (!map.has(k)) map.set(k, { entity: r.entity, entity_code: r.entity_code, time: r.time, v: {} });
      map.get(k).v[r.series_id] = r.value;
    }
    const rows = [...map.values()].sort((a, b) => a.entity_code.localeCompare(b.entity_code) || String(a.time).localeCompare(String(b.time)))
      .map(o => [o.entity, o.entity_code, o.time, ...ids.map(i => o.v[i] == null ? null : o.v[i])]);
    return { headers: ["entity", "entity_code", "time", ...ids], rows, layout: "wide" };
  }

  /* ---------- UI ---------- */
  const pane = $("morePane"), wb = $("wbPane");
  if (!pane || !wb) return;
  let current = null, last = null;
  const csvText = (h, rows) => { const q = v => '"' + String(v == null ? "" : v).replace(/"/g, '""') + '"'; return [h.map(q).join(","), ...rows.map(r => r.map(q).join(","))].join("\r\n"); };
  function save(name, content, type) { const u = URL.createObjectURL(new Blob([content], { type })); const a = document.createElement("a"); a.href = u; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(u), 1000); }
  const status = (m, err) => { const el = $("mpStatus"); if (el) { el.textContent = m; el.className = "status show" + (err ? " error" : ""); } };

  function render(key) {
    const p = P[key]; current = key; last = null;
    pane.innerHTML =
      '<div class="ds-provider"><div class="ds-provider-head"><div><b>' + esc(p.name) + '</b><small>No API key needed · data go straight from the provider to your browser</small></div></div>' +
      '<div class="ds-columns"><div><h3>1 · Indicators</h3><input id="mpSearch" type="search" placeholder="Filter indicators…" aria-label="Filter indicators" style="width:100%;margin-bottom:6px"><div class="ds-catalog" id="mpInds">' +
      p.indicators.map((x, i) => '<label class="ds-ind"><input type="checkbox" data-i="' + i + '"' + (i === 0 ? " checked" : "") + '><span><b>' + esc(x.label) + '</b><small>' + esc(x.id) + '</small></span></label>').join("") +
      '</div></div><div id="mpEntBox"><h3>2 · ' + esc(p.entityLabel) + '</h3><select id="mpEnts" multiple size="9" aria-label="' + esc(p.entityLabel) + '">' +
      p.entities.map((e, i) => '<option value="' + esc(e.code) + '"' + (i < 2 ? " selected" : "") + '>' + esc(e.name) + ' (' + esc(e.code) + ')</option>').join("") +
      '</select><p class="sub">Ctrl/Command-click to select several.</p></div></div>' +
      '<div class="ds-controls"><div class="field"><label for="mpStart">3 · Start year</label><input id="mpStart" type="number" min="1900" max="2100" value="2000"></div><div class="field"><label for="mpEnd">End year</label><input id="mpEnd" type="number" min="1900" max="2100" value="' + new Date().getFullYear() + '"></div></div>' +
      '<div class="row ds-actions"><button type="button" class="btn" id="mpFetch">Fetch data</button></div><div id="mpStatus" class="status" role="status" aria-live="polite"></div>' +
      '<div id="mpResult" class="ds-result hidden"><h3>Retrieved observations</h3><p class="sub" id="mpMeta"></p><div id="mpWarn"></div><div id="mpPrev"></div>' +
      '<div class="row ds-actions"><button type="button" class="btn secondary" id="mpCsv">Download CSV</button><button type="button" class="btn secondary" id="mpXlsx">Download Excel</button><button type="button" class="btn" id="mpSend">Send to Cleaning Lab</button></div>' +
      '<p class="sub">Source: ' + esc(p.name) + ' (<a href="' + esc(p.site) + '" target="_blank" rel="noopener">' + esc(p.site) + '</a>). ' + esc(p.licence) + ' Observations may be revised; check the provider metadata before publishing.</p></div></div>';
    $("mpFetch").onclick = run;
    updateEnts();
    $("mpInds").onchange = updateEnts;
    $("mpSearch").oninput = e => { const q = e.target.value.trim().toLowerCase(); pane.querySelectorAll("#mpInds .ds-ind").forEach(l => { l.style.display = !q || l.textContent.toLowerCase().includes(q) ? "" : "none"; }); };
    $("mpCsv").onclick = () => last && save("oes_" + key + "_data.csv", "﻿" + csvText(last.s.headers, last.s.rows), "text/csv;charset=utf-8");
    $("mpXlsx").onclick = () => { if (!last) return; if (!window.XLSX) { status("Excel export library is unavailable. Use CSV.", true); return; } const ws = XLSX.utils.aoa_to_sheet([last.s.headers, ...last.s.rows]), b = XLSX.utils.book_new(); XLSX.utils.book_append_sheet(b, ws, p.name.slice(0, 28)); XLSX.writeFile(b, "oes_" + key + "_data.xlsx"); };
    $("mpSend").onclick = () => {
      if (!last || typeof window.OESDataCleaningLoad !== "function") return;
      window.OESDataCleaningLoad([last.s.headers, ...last.s.rows], key + "_data.csv", last.meta);
      document.getElementById("uploadPanel").scrollIntoView({ behavior: "smooth", block: "start" });
    };
  }
  const chosen = () => [...pane.querySelectorAll("#mpInds input:checked")].map(i => P[current].indicators[+i.dataset.i]);
  function updateEnts() { $("mpEntBox").style.opacity = chosen().some(i => i.needs) ? "1" : ".45"; }
  async function run() {
    const inds = chosen(), p = P[current];
    const ents = [...$("mpEnts").selectedOptions].map(o => ({ code: o.value, name: o.textContent.replace(/\s*\(.*$/, "") }));
    const s = +$("mpStart").value, e = +$("mpEnd").value;
    if (!inds.length) return status("Tick at least one indicator.", true);
    if (inds.some(i => i.needs) && !ents.length) return status("Select at least one country or currency.", true);
    if (!(s <= e)) return status("Start year must not be after end year.", true);
    $("mpFetch").disabled = true; $("mpResult").classList.add("hidden");
    try {
      const r = await fetchProvider(current, inds, ents, s, e, status);
      if (!r.records.length) { status("The provider returned no observations for this selection. " + r.warnings.join(" "), true); return; }
      const sh = shape(r.records, inds);
      last = {
        s: sh, meta: {
          provider: p.name, provider_id: current, fetched_at: new Date().toISOString(), start_year: s, end_year: e,
          series: inds.map(i => ({ id: i.id, title: i.label })), entities: ents.map(x => ({ code: x.code, name: x.name })),
          query_urls: r.urls, warnings: r.warnings, records: r.records.length, license: p.licence,
          citation: p.name + ". " + inds.map(i => i.label).join("; ") + ". Retrieved " + new Date().toISOString().slice(0, 10) + " from " + p.site
        }
      };
      $("mpMeta").textContent = r.records.length.toLocaleString() + " observations · " + inds.length + " indicator(s) · layout: " + (sh.layout === "wide" ? "one column per indicator" : "long (mixed frequencies kept separate)");
      $("mpWarn").innerHTML = r.warnings.map(w => '<p class="ds-warning">' + esc(w) + '</p>').join("");
      $("mpPrev").innerHTML = '<div class="table-wrap"><table class="diagnostic-table"><thead><tr>' + sh.headers.map(h => "<th>" + esc(h) + "</th>").join("") + "</tr></thead><tbody>" +
        sh.rows.slice(0, 25).map(row => "<tr>" + row.map(v => "<td>" + (v == null ? "<em>Missing</em>" : esc(v)) + "</td>").join("") + "</tr>").join("") + "</tbody></table></div>";
      $("mpResult").classList.remove("hidden");
      status("Done: " + r.records.length.toLocaleString() + " observations. Download them or send them to the Cleaning Lab.");
    } catch (err) { status(err.message || "Request failed.", true); }
    finally { $("mpFetch").disabled = false; }
  }
  function show(key) {
    document.querySelectorAll(".ds-tab[data-pane]").forEach(b => { const on = b.dataset.pane === key; b.classList.toggle("active", on); b.setAttribute("aria-selected", on); });
    const isWb = key === "worldbank";
    wb.hidden = !isWb; pane.hidden = isWb;
    if (!isWb) render(key);
  }
  document.querySelectorAll(".ds-tab[data-pane]").forEach(b => b.addEventListener("click", () => show(b.dataset.pane)));
  window.OESMoreSources = { providers: P, fetchProvider, shape, parseCsv };
})();
