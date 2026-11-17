/* OES Data Sources UI — World Bank phase */
(function () {
  "use strict";
  const $ = id => document.getElementById(id);
  const state = { catalog: [], entities: [], basket: [], lastResult: null, filtered: [] };
  const esc = v => String(v == null ? "" : v).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const status = (message, error) => {
    const el = $("dsStatus");
    if (!el) return;
    el.textContent = message;
    el.className = "status show" + (error ? " error" : "");
  };
  const progress = (message, detail) => {
    status(message + (detail && detail.count != null ? " · " + Number(detail.count).toLocaleString() + " records" : ""));
    const bar = $("dsProgressBar");
    if (bar && detail && detail.pages) bar.style.width = Math.max(3, Math.min(100, 100 * detail.page / detail.pages)) + "%";
  };
  function renderCatalog(items) {
    state.filtered = items;
    $("dsCatalogCount").textContent = items.length.toLocaleString() + " indicators shown";
    $("dsCatalog").innerHTML = items.slice(0, 120).map((x, i) =>
      '<label class="ds-catalog-item"><input type="checkbox" data-catalog-index="' + i + '" aria-label="Select ' + esc(x.title) + '"><span><b>' + esc(x.title) + '</b><small>' + esc(x.id) + ' · ' + esc(x.topic) + ' · ' + esc(x.unit) + ' · ' + esc(x.source || "World Bank") + '</small></span><button type="button" class="btn secondary ds-add" data-add-index="' + i + '">Add</button></label>'
    ).join("") || '<p class="sub">No indicators match this search.</p>';
  }
  function applyCatalogFilters() {
    const q = $("dsSearch").value.trim().toLowerCase();
    const topic = $("dsTopic").value;
    const list = state.catalog.filter(x => {
      const match = !q || (x.id + " " + x.title + " " + x.topics.join(" ") + " " + x.source).toLowerCase().includes(q);
      return match && (topic === "all" || x.topics.includes(topic));
    });
    renderCatalog(list);
  }
  function renderBasket() {
    $("dsBasketCount").textContent = state.basket.length + " selected";
    $("dsBasket").innerHTML = state.basket.map((x, i) =>
      '<div class="ds-basket-item"><span><b>' + esc(x.title) + '</b><small>' + esc(x.id) + ' · ' + esc(x.unit) + '</small></span><button type="button" class="btn outline" data-remove-index="' + i + '" aria-label="Remove ' + esc(x.id) + '">Remove</button></div>'
    ).join("") || '<p class="sub">Add indicators from the catalog to create your basket.</p>';
  }
  async function loadCatalog(force) {
    status("Loading the World Bank indicator catalog. The first load may take several seconds.");
    $("dsLoadCatalog").disabled = true;
    try {
      const result = await window.OESDataSources.worldBank.listCatalog({
        forceRefresh: !!force,
        limit: 25000,
        onProgress: progress
      });
      state.catalog = result.items;
      const topics = Array.from(new Set(state.catalog.flatMap(x => x.topics))).sort();
      $("dsTopic").innerHTML = '<option value="all">All topics</option>' + topics.map(x => '<option value="' + esc(x) + '">' + esc(x) + '</option>').join("");
      applyCatalogFilters();
      status("Catalog ready: " + result.total.toLocaleString() + " indicators available for search" + (result.cachedCatalogSize ? " · " + result.cachedCatalogSize.toLocaleString() + " cached" : "") + ".");
    } catch (e) {
      status(e.message || "Could not load the World Bank catalog.", true);
    } finally {
      $("dsLoadCatalog").disabled = false;
    }
  }
  async function loadEntities() {
    status("Loading countries and regional aggregates…");
    $("dsLoadEntities").disabled = true;
    try {
      state.entities = await window.OESDataSources.worldBank.listEntities({
        includeAggregates: $("dsIncludeAggregates").checked,
        onProgress: progress
      });
      const selected = new Set(Array.from($("dsEntities").selectedOptions).map(x => x.value));
      $("dsEntities").innerHTML = state.entities.map(x =>
        '<option value="' + esc(x.code) + '" data-aggregate="' + (x.aggregate ? "1" : "0") + '">' +
        esc(x.name) + ' (' + esc(x.iso3 || x.code) + ')' + (x.aggregate ? " · aggregate" : "") + '</option>'
      ).join("");
      Array.from($("dsEntities").options).forEach(o => { o.selected = selected.has(o.value); });
      status("Loaded " + state.entities.length.toLocaleString() + " " + ($("dsIncludeAggregates").checked ? "countries and aggregates" : "country/economy entities") + ".");
    } catch (e) {
      status(e.message || "Could not load World Bank entities.", true);
    } finally {
      $("dsLoadEntities").disabled = false;
    }
  }
  function addToBasket(item) {
    if (state.basket.some(x => x.id === item.id && x.source_id === item.source_id)) return;
    state.basket.push(item);
    renderBasket();
  }
  function selectedEntities() {
    const selected = new Set(Array.from($("dsEntities").selectedOptions).map(x => x.value));
    return state.entities.filter(x => selected.has(x.code));
  }
  function basketRequest(preview) {
    return {
      series: state.basket,
      entities: selectedEntities(),
      startYear: $("dsStartYear").value,
      endYear: $("dsEndYear").value,
      preview: !!preview,
      maxRows: 30,
      onProgress: progress
    };
  }
  function renderResult(result, preview) {
    const rows = result.records || [];
    const shown = rows.slice(0, 30);
    $("dsPreview").innerHTML = shown.length ?
      '<div class="table-wrap"><table class="diagnostic-table"><thead><tr><th>Entity</th><th>Code</th><th>Time</th><th>Series ID</th><th>Value</th></tr></thead><tbody>' +
      shown.map(r => '<tr><td>' + esc(r.entity) + '</td><td>' + esc(r.entity_code) + '</td><td>' + esc(r.time) + '</td><td>' + esc(r.series_id) + '</td><td>' + (r.value == null ? '<em>Missing</em>' : esc(r.value)) + '</td></tr>').join("") +
      '</tbody></table></div>' : '<p class="sub">No observations returned for this selection and period.</p>';
    $("dsResultMeta").textContent = (preview ? "Preview sample" : "Fetched dataset") + " · " + rows.length.toLocaleString() + " records · " + result.metadata.series.length + " series · " + result.metadata.entities.length + " selected entities";
    $("dsWarnings").innerHTML = (result.metadata.warnings || []).map(x => '<p class="ds-warning">' + esc(x) + '</p>').join("");
    $("dsOutputs").classList.toggle("hidden", preview || !rows.length);
    $("dsResult").classList.remove("hidden");
  }
  async function fetchBasket(preview) {
    if (!state.basket.length) { status("Add at least one indicator to the basket.", true); return; }
    if (!selectedEntities().length) { status("Select at least one country or aggregate.", true); return; }
    $("dsFetch").disabled = true;
    $("dsPreviewButton").disabled = true;
    $("dsResult").classList.add("hidden");
    $("dsProgressBar").style.width = "0%";
    try {
      const result = await window.OESDataSources.worldBank.fetchData(basketRequest(preview));
      state.lastResult = result;
      renderResult(result, preview);
      status((preview ? "Preview completed" : "World Bank data fetched") + ": " + result.records.length.toLocaleString() + " observations.");
    } catch (e) {
      status(e.message || "World Bank data request failed.", true);
    } finally {
      $("dsFetch").disabled = false;
      $("dsPreviewButton").disabled = false;
    }
  }
  function csvText(records) {
    const headers = ["entity", "entity_code", "time", "series_id", "value"];
    const quote = v => '"' + String(v == null ? "" : v).replace(/"/g, '""') + '"';
    return [headers.join(","), ...records.map(r => headers.map(k => quote(r[k])).join(","))].join("\r\n");
  }
  function download(name, content, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function downloadCsv() {
    if (!state.lastResult) return;
    download("oes_worldbank_data.csv", "\ufeff" + csvText(state.lastResult.records), "text/csv;charset=utf-8");
  }
  function downloadXlsx() {
    if (!state.lastResult || !window.XLSX) { status("Excel export library is unavailable.", true); return; }
    const ws = XLSX.utils.json_to_sheet(state.lastResult.records);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "World Bank data");
    XLSX.writeFile(wb, "oes_worldbank_data.xlsx");
  }
  function sendToCleaning() {
    if (!state.lastResult || !state.lastResult.records.length) return;
    if (state.lastResult.metadata.preview) { status("Fetch the complete dataset before sending it to the Cleaning Lab.", true); return; }
    const headers = ["entity", "entity_code", "time", "series_id", "value"];
    const rows = [headers, ...state.lastResult.records.map(r => headers.map(k => r[k] == null ? null : r[k]))];
    if (typeof window.OESDataCleaningLoad !== "function") {
      status("Cleaning Lab hand-off is not ready. Reload the page and try again.", true);
      return;
    }
    window.OESDataCleaningLoad(rows, "worldbank_indicators.csv", state.lastResult.metadata);
    status("World Bank data sent directly to the Cleaning Lab. Review its diagnostics before cleaning.");
    document.getElementById("uploadPanel").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  async function testConnection() {
    $("dsTestConnection").disabled = true;
    status("Testing a real browser fetch from this GitHub Pages origin…");
    try {
      const result = await window.OESDataSources.testWorldBankConnection();
      status("Browser connection successful. " + result.countriesReturned + " country record received; pagination reports " + result.pages + " page(s).");
    } catch (e) {
      status(e.message || "Browser connection failed. This may be a CORS restriction.", true);
    } finally {
      $("dsTestConnection").disabled = false;
    }
  }
  function init() {
    if (!$("dsCatalog")) return;
    $("dsLoadCatalog").addEventListener("click", () => loadCatalog(false));
    $("dsRefreshCatalog").addEventListener("click", () => loadCatalog(true));
    $("dsSearch").addEventListener("input", applyCatalogFilters);
    $("dsTopic").addEventListener("change", applyCatalogFilters);
    $("dsLoadEntities").addEventListener("click", loadEntities);
    $("dsIncludeAggregates").addEventListener("change", loadEntities);
    $("dsCatalog").addEventListener("click", e => {
      const add = e.target.closest("[data-add-index]");
      if (add) addToBasket(state.filtered[Number(add.dataset.addIndex)]);
    });
    $("dsCatalog").addEventListener("change", e => {
      const check = e.target.closest("[data-catalog-index]");
      if (check && check.checked) addToBasket(state.filtered[Number(check.dataset.catalogIndex)]);
    });
    $("dsBasket").addEventListener("click", e => {
      const remove = e.target.closest("[data-remove-index]");
      if (remove) { state.basket.splice(Number(remove.dataset.removeIndex), 1); renderBasket(); }
    });
    $("dsPreviewButton").addEventListener("click", () => fetchBasket(true));
    $("dsFetch").addEventListener("click", () => fetchBasket(false));
    $("dsDownloadCsv").addEventListener("click", downloadCsv);
    $("dsDownloadXlsx").addEventListener("click", downloadXlsx);
    $("dsSendToCleaning").addEventListener("click", sendToCleaning);
    $("dsTestConnection").addEventListener("click", testConnection);
    renderBasket();
    loadEntities();
  }
  init();
})();
