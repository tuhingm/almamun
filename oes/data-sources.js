/* OES Data Sources — provider adapter layer (browser-only) */
(function (global) {
  "use strict";
  const WB_BASE = "https://api.worldbank.org/v2";
  const CATALOG_TTL = 7 * 24 * 60 * 60 * 1000;
  const MAX_RETRIES = 5;
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  const notify = (fn, message, detail) => { if (typeof fn === "function") fn(message, detail || {}); };
  const cache = {
    db: null,
    async open() {
      if (!("indexedDB" in global)) return null;
      if (this.db) return this.db;
      this.db = await new Promise(resolve => {
        const req = indexedDB.open("OESDataSourceCache", 1);
        req.onupgradeneeded = () => {
          const db = req.result;
          if (!db.objectStoreNames.contains("catalogs")) db.createObjectStore("catalogs");
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => resolve(null);
      });
      return this.db;
    },
    async get(key) {
      const db = await this.open();
      if (!db) return null;
      return new Promise(resolve => {
        const req = db.transaction("catalogs", "readonly").objectStore("catalogs").get(key);
        req.onsuccess = () => {
          const value = req.result;
          resolve(value && Date.now() - value.savedAt < CATALOG_TTL ? value.data : null);
        };
        req.onerror = () => resolve(null);
      });
    },
    async set(key, data) {
      const db = await this.open();
      if (!db) return;
      await new Promise(resolve => {
        const tx = db.transaction("catalogs", "readwrite");
        tx.objectStore("catalogs").put({ savedAt: Date.now(), data }, key);
        tx.oncomplete = resolve;
        tx.onerror = resolve;
      });
    }
  };
  async function fetchJson(url, options) {
    const opts = options || {};
    let lastError;
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), opts.timeout || 60000);
      try {
        const response = await fetch(url, {
          method: "GET",
          mode: "cors",
          credentials: "omit",
          cache: "no-store",
          headers: { Accept: "application/json" },
          signal: controller.signal
        });
        clearTimeout(timeout);
        if (!response.ok) {
          const error = new Error("HTTP " + response.status + " from World Bank API.");
          error.status = response.status;
          if (response.status === 429 || response.status >= 500) throw error;
          throw error;
        }
        const body = await response.json();
        if (body && !Array.isArray(body) && body.message) throw new Error(body.message);
        if (Array.isArray(body) && body.length === 1 && body[0] && body[0].message) {
          throw new Error(body[0].message.map(x => x.value || x.key || "").join("; "));
        }
        return body;
      } catch (error) {
        clearTimeout(timeout);
        lastError = error;
        const retryable = error.name === "AbortError" || error instanceof TypeError || error.status === 429 || (error.status >= 500);
        if (!retryable || attempt === MAX_RETRIES) {
          if (error instanceof TypeError || error.name === "AbortError") {
            throw new Error("The World Bank server did not answer after several tries. Its data service is sometimes slow or overloaded, so this is usually temporary: wait a few minutes and try again, or fetch fewer countries and a shorter year range.");
          }
          throw error;
        }
        await sleep(Math.min(15000, 1500 * Math.pow(2, attempt)));
      }
    }
    throw lastError || new Error("World Bank request failed.");
  }
  function unpackPage(body) {
    if (!Array.isArray(body)) return { meta: null, rows: [] };
    return { meta: body[0] || null, rows: Array.isArray(body[1]) ? body[1] : [] };
  }
  async function fetchAllPages(urlForPage, onProgress, label, options) {
    const opts = options || {};
    const firstUrl = urlForPage(1);
    const first = unpackPage(await fetchJson(firstUrl));
    const pages = Math.max(1, Number(first.meta && first.meta.pages) || 1);
    const rows = first.rows.slice();
    const urls = [firstUrl];
    notify(onProgress, label + ": page 1 of " + pages, { page: 1, pages, count: rows.length });
    const lastPage = opts.preview ? 1 : pages;
    for (let page = 2; page <= lastPage; page++) {
      await sleep(120);
      const pageUrl = urlForPage(page);
      const next = unpackPage(await fetchJson(pageUrl));
      urls.push(pageUrl);
      rows.push(...next.rows);
      notify(onProgress, label + ": page " + page + " of " + pages, { page, pages, count: rows.length });
    }
    return { rows, meta: first.meta, pages, url: firstUrl, urls };
  }
  function isAggregate(country) {
    return !!(country && country.region && country.region.id === "NA");
  }
  function toCatalogItem(item) {
    const topics = Array.isArray(item.topics) ? item.topics.map(x => x.value || x.name || "").filter(Boolean) : [];
    return {
      id: String(item.id || ""),
      title: item.name || item.id || "Untitled indicator",
      topic: topics[0] || "Unclassified",
      topics,
      unit: item.unit || "Not stated",
      frequency: "Not supplied by indicator catalog",
      source_id: item.source && item.source.id ? String(item.source.id) : "",
      source: item.source && item.source.value ? item.source.value : "",
      source_note: item.sourceNote || "",
      source_organization: item.sourceOrganization || "",
      provider: "World Bank",
      catalog_url: WB_BASE + "/indicator/" + encodeURIComponent(item.id || "") + "?format=json"
    };
  }
  const worldBank = {
    id: "worldbank",
    name: "World Bank Indicators",
    baseUrl: WB_BASE,
    async listCatalog(options) {
      const opts = options || {};
      const key = "worldbank-indicators-v2";
      let all = opts.forceRefresh ? null : await cache.get(key);
      if (!all) {
        const result = await fetchAllPages(
          page => WB_BASE + "/indicator?format=json&per_page=1000&page=" + page,
          opts.onProgress,
          "Loading World Bank indicator catalog"
        );
        all = result.rows.map(toCatalogItem).filter(x => x.id);
        await cache.set(key, all);
      } else {
        notify(opts.onProgress, "Loaded cached World Bank indicator catalog", { cached: true, count: all.length });
      }
      let filtered = all;
      if (opts.search) {
        const q = String(opts.search).trim().toLowerCase();
        filtered = filtered.filter(x => (x.id + " " + x.title + " " + x.topics.join(" ") + " " + x.source).toLowerCase().includes(q));
      }
      if (opts.topic && opts.topic !== "all") filtered = filtered.filter(x => x.topics.includes(opts.topic));
      return { items: filtered.slice(0, opts.limit || 300), total: filtered.length, cachedCatalogSize: all.length };
    },
    async listEntities(options) {
      const opts = options || {};
      const key = "worldbank-countries-v2";
      let all = opts.forceRefresh ? null : await cache.get(key);
      if (!all) {
        const result = await fetchAllPages(
          page => WB_BASE + "/country?format=json&per_page=400&page=" + page,
          opts.onProgress,
          "Loading World Bank country and aggregate catalog"
        );
        all = result.rows.map(c => ({
          code: c.id || "",
          iso3: c.iso3Code || (c.id && /^[A-Z]{3}$/.test(c.id) ? c.id : ""),
          iso2: c.iso2Code || "",
          name: c.name || c.id || "",
          region: c.region && c.region.value ? c.region.value : "",
          region_id: c.region && c.region.id ? c.region.id : "",
          income_level: c.incomeLevel && c.incomeLevel.value ? c.incomeLevel.value : "",
          aggregate: isAggregate(c)
        })).filter(x => x.code);
        await cache.set(key, all);
      } else {
        notify(opts.onProgress, "Loaded cached World Bank country catalog", { cached: true, count: all.length });
      }
      return all.filter(x => opts.includeAggregates || !x.aggregate);
    },
    async fetchData(args) {
      const opts = args || {};
      const series = Array.isArray(opts.series) ? opts.series : [opts.series].filter(Boolean);
      const entities = Array.isArray(opts.entities) ? opts.entities : [];
      const start = Number(opts.startYear), end = Number(opts.endYear);
      if (!series.length) throw new Error("Add at least one World Bank indicator to the basket.");
      if (!entities.length) throw new Error("Select at least one country or aggregate.");
      if (!Number.isInteger(start) || !Number.isInteger(end) || start > end) throw new Error("Enter a valid start and end year.");
      const records = [], requests = [], warnings = [], observationStatus = {};
      const entityMap = new Map(entities.map(e => [String(e.code || e), e]));
      const batchSize = 25;
      const batches = [];
      for (let i = 0; i < entities.length; i += batchSize) batches.push(entities.slice(i, i + batchSize));
      let completed = 0;
      for (const item of series) {
        for (const batch of batches) {
          const codes = batch.map(e => String(e.code || e).trim()).filter(Boolean);
          if (codes.length !== batch.length) throw new Error("One or more selected World Bank entity codes are invalid.");
          const countryPath = codes.join(";");
          const urlForPage = page => {
            const u = new URL(WB_BASE + "/country/" + countryPath + "/indicator/" + encodeURIComponent(item.id));
            u.searchParams.set("format", "json");
            u.searchParams.set("date", start + ":" + end);
            u.searchParams.set("per_page", opts.preview ? "30" : "1000");
            u.searchParams.set("page", String(page));
            if (item.source_id) u.searchParams.set("source", item.source_id);
            return u.toString();
          };
          const result = await fetchAllPages(urlForPage, opts.onProgress, "Fetching " + item.id, { preview: !!opts.preview });
          requests.push({ series_id: item.id, url: result.url, urls: result.urls, pages: result.pages, preview: !!opts.preview });
          for (const row of result.rows) {
            const entityCode = row.countryiso3code || (row.country && row.country.id) || "";
            const obsStatus = row.obs_status || "";
            if (obsStatus) observationStatus[obsStatus] = (observationStatus[obsStatus] || 0) + 1;
            const selectedEntity = entityMap.get(entityCode) || entityMap.get(row.country && row.country.id) || {};
            records.push({
              entity: row.country && row.country.value ? row.country.value : (selectedEntity.name || entityCode),
              entity_code: entityCode,
              time: String(row.date || ""),
              series_id: row.indicator && row.indicator.id ? row.indicator.id : item.id,
              value: row.value === undefined ? null : row.value
            });
            if (records.length > 500000) throw new Error("This request exceeds the 500,000-observation browser safety limit. Narrow the countries, indicators, or year range and retry.");
          }
          completed++;
          notify(opts.onProgress, "Completed " + completed + " of " + (series.length * batches.length) + " World Bank series/entity batches", { completed, total: series.length * batches.length, records: records.length });
          if (opts.preview && records.length >= (opts.maxRows || 25)) break;
        }
        if (opts.preview && records.length >= (opts.maxRows || 25)) break;
      }
      if (records.length > 500000) throw new Error("This request exceeds the 500,000-observation browser safety limit. Narrow the countries, indicators, or year range and retry.");
      const selectedAggregates = entities.filter(e => e.aggregate);
      if (selectedAggregates.length) warnings.push("The selection includes World Bank regional or income-group aggregates. Do not mix them with individual countries without an explicit analytical reason.");
      if (records.some(r => r.value === null)) warnings.push("Some observations are null. Latest-year values may be incomplete or not yet reported.");
      if (observationStatus.F) warnings.push(observationStatus.F + " observations carry World Bank status F (forecast). Check observation status before treating these as observed values.");
      const metadata = {
        provider: "World Bank",
        provider_id: "worldbank",
        fetched_at: new Date().toISOString(),
        start_year: start,
        end_year: end,
        series: series.map(s => ({ id: s.id, title: s.title, unit: s.unit, topic: s.topic, source: s.source, source_id: s.source_id })),
        entities: entities.map(e => ({ code: e.code || e, name: e.name || e, aggregate: !!e.aggregate })),
        requests,
        query_urls: requests.flatMap(x => x.urls),
        warnings,
        observation_status_counts: observationStatus,
        preview: !!opts.preview,
        frequency_note: "Frequency is not supplied in the World Bank indicator catalog response; inspect observation periods before combining with other providers.",
        license: "Default World Bank dataset terms are CC BY 4.0 unless the relevant indicator or source metadata specifies otherwise.",
        attribution: "The World Bank: " + series.map(s => s.title + " (" + s.id + ")").join("; ") + ". Data retrieved through the World Bank Indicators API.",
        records: records.length
      };
      metadata.citation = this.citation(metadata);
      return { records, metadata };
    },
    citation(meta) {
      const titles = (meta && meta.series || []).map(s => s.title + " (" + s.id + ")").join("; ");
      return "The World Bank. " + (titles || "World Development Indicators") + ". Data retrieved " + new Date(meta && meta.fetched_at || Date.now()).toISOString().slice(0, 10) + " through the World Bank Indicators API. https://api.worldbank.org/";
    }
  };
  function validateAdapter(adapter) {
    ["listCatalog", "listEntities", "fetchData", "citation"].forEach(method => {
      if (!adapter || typeof adapter[method] !== "function") throw new Error("Provider adapter must implement " + method + "().");
    });
    return adapter;
  }
  const registry = new Map();
  function registerProvider(adapter) {
    validateAdapter(adapter);
    if (registry.has(adapter.id)) throw new Error("Provider already registered: " + adapter.id);
    registry.set(adapter.id, adapter);
    return adapter;
  }
  registerProvider(worldBank);
  global.OESDataSources = {
    version: "0.1.0",
    adapterContract: ["listCatalog(options)", "listEntities(options)", "fetchData({series, entities, startYear, endYear, onProgress, preview})", "citation(metadata)"],
    registerProvider,
    getProvider: id => registry.get(id),
    listProviders: () => Array.from(registry.values()).map(x => ({ id: x.id, name: x.name })),
    cache,
    worldBank,
    async testWorldBankConnection() {
      const url = WB_BASE + "/country?format=json&per_page=1";
      const result = unpackPage(await fetchJson(url));
      return { ok: true, url, countriesReturned: result.rows.length, pages: result.meta && result.meta.pages };
    }
  };
})(window);
