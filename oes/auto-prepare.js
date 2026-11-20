/* OES auto-prepare: turn an uploaded dataset into an estimation-ready CSV, entirely in the browser.
 *
 * OESPrepare.prepare(text) -> {csv, rows, cols, changes[], warnings[], columns[], structure, ready}
 *
 * Rules (only safe, reversible, reported changes; nothing is imputed and no observation is dropped except
 * blank rows and exact duplicate rows):
 *   file    BOM, delimiter (, ; tab |), ragged rows, blank rows/columns, leftover row-index column
 *   names   kept when already plain; otherwise lower-case with spaces/symbols -> "_"; unique (any script: Korean, Bengali ... is kept)
 *   values  numbers written as text (1,234.5  1.234,5  $12  12%  (5)  -3), missing codes (NA, ., -, #N/A) in
 *           numeric columns -> blank, yes/no and true/false -> 1/0
 *   dates   ISO, 2020/01/31, 31/12/2020 (only when day/month order is certain), 2020Q1, Q1 2020, Jan 2020, 2020M01
 *   layout  panel (unit + time) or time series detected; rows sorted; duplicate keys, gaps and unbalanced panels reported
 *   checks  missing values per column, constant columns, mixed columns, missing-value codes such as -999
 */
(function (root) {
"use strict";

const MISSING = new Set(["na", "n/a", "n.a.", "n.a", "nan", "null", "none", "nil", ".", "..", "-", "--", "—", "–",
  "#n/a", "#na", "#value!", "#div/0!", "#ref!", "missing", "?", "nd", "n.d."]);
const MONTHS = {jan: 1, feb: 2, mar: 3, apr: 4, may: 5, jun: 6, jul: 7, aug: 8, sep: 9, sept: 9, oct: 10, nov: 11, dec: 12};
const TIME_NAMES = /^(year|yr|date|time|period|quarter|month|obs|fyear|wave|t|dt|datetime|yyyy|time_period)$/i;
const ID_NAMES = /(^|_)(id|unit|entity|country|iso|iso3|code|firm|state|region|county|city|company|panel|ticker|cusip|gvkey|name|household|hhid|individual|person|district)(_|$)|id$/i;
const SENTINELS = [-99999, -9999, -999, -99, 99999, 9999999, -9999999];
const NO_INDEX = new Set(["", "unnamed: 0", "unnamed:0", "index", "row.names", "rownames", "row_names", "x", "level_0"]);

// ------------------------------------------------------------------ parsing
function countOutside(line, d) {
  let q = false, n = 0;
  for (const c of line) { if (c === '"') q = !q; else if (c === d && !q) n++; }
  return n;
}
function detectDelimiter(text) {
  const lines = text.split(/\r?\n/).filter(l => l.trim() !== "").slice(0, 8);
  let best = ",", score = -1;
  for (const d of [",", ";", "\t", "|"]) {
    const counts = lines.map(l => countOutside(l, d));
    if (!counts[0]) continue;
    const s = counts.filter(c => c === counts[0]).length * 1000 + counts[0];
    if (s > score) { score = s; best = d; }
  }
  return best;
}
function tokenize(text, delim) {
  const rows = []; let row = [], cell = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; } else cell += c; }
    else if (c === '"') q = true;
    else if (c === delim) { row.push(cell); cell = ""; }
    else if (c === "\n" || c === "\r") { if (c === "\r" && text[i + 1] === "\n") i++; row.push(cell); rows.push(row); row = []; cell = ""; }
    else cell += c;
  }
  if (cell !== "" || row.length) { row.push(cell); rows.push(row); }
  return rows;
}
const clean = s => String(s == null ? "" : s).replace(/[   ​]/g, " ").trim();
function csvCell(v) { return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }

function cleanName(raw, i) {
  const keep = clean(raw);
  if (/^[\p{L}_][\p{L}\p{N}_]*$/u.test(keep)) return keep;          // already a plain name: leave it alone
  let s = keep.normalize("NFC").toLowerCase().replace(/%/g, " pct ").replace(/&/g, " and ")
    .replace(/[^\p{L}\p{N}]+/gu, "_").replace(/^_+|_+$/g, "");
  if (!s) s = "col_" + (i + 1);
  if (/^\d/.test(s)) s = "v_" + s;
  return s;
}

// ------------------------------------------------------------------ numbers
const PLAIN = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;
function commaStyle(values) {     // true when the column writes 1.234,56 (decimal comma)
  let euro = 0, anglo = 0;
  for (const raw of values) {
    const s = raw.replace(/[^\d.,-]/g, "");
    if (/^-?\d{1,3}(\.\d{3})+,\d+$/.test(s)) euro += 2;
    else if (/^-?\d+,\d+$/.test(s) && !/^-?\d{1,3},\d{3}$/.test(s)) euro++;
    if (/^-?\d{1,3}(,\d{3})+(\.\d+)?$/.test(s)) anglo += 2;
    else if (/^-?\d*\.\d+$/.test(s) && !/^-?\d{1,3}\.\d{3}$/.test(s)) anglo++;
  }
  return euro > anglo;
}
function parseNumber(raw, comma) {
  let s = raw.replace(/\s+/g, "").replace(/[−‒–]/g, "-");
  let pct = false, neg = false;
  if (s.endsWith("%")) { pct = true; s = s.slice(0, -1); }
  if (/^\(.*\)$/.test(s)) { neg = true; s = s.slice(1, -1); }
  s = s.replace(/^([+-]?)[$€£¥₩₹]+/, "$1").replace(/[$€£¥₩₹]+$/, "").replace(/^([+-]?)(USD|EUR|GBP|JPY|KRW|BDT|INR|CNY|CAD|AUD|CHF)(?=\d)/i, "$1");
  if (s.startsWith("-")) { neg = true; s = s.slice(1); } else if (s.startsWith("+")) s = s.slice(1);
  if (!/^[\d.,]+([eE][+-]?\d+)?$/.test(s)) return null;
  if (PLAIN.test(s)) { /* already plain */ }
  else if (comma) {
    if (!/^\d{1,3}(\.\d{3})*(,\d+)?$/.test(s) && !/^\d+(,\d+)?$/.test(s)) return null;
    s = s.replace(/\./g, "").replace(",", ".");
  } else {
    if (!/^\d{1,3}(,\d{3})*(\.\d+)?$/.test(s)) return null;
    s = s.replace(/,/g, "");
  }
  let v = Number(s);
  if (!Number.isFinite(v)) return null;
  return {v: neg ? -v : v, pct};
}

// ------------------------------------------------------------------ dates
const pad = n => String(n).padStart(2, "0");
const okDate = (y, m, d) => y >= 1000 && y <= 2999 && m >= 1 && m <= 12 && d >= 1 && d <= 31;
function dateFamily(s) {          // classify one value; returns {fam, ...parts} or null
  let m;
  if ((m = /^(\d{4})[-\/.](\d{1,2})[-\/.](\d{1,2})$/.exec(s)) && okDate(+m[1], +m[2], +m[3])) return {fam: "ymd", y: +m[1], m: +m[2], d: +m[3]};
  if ((m = /^(\d{1,2})[-\/.](\d{1,2})[-\/.](\d{4})$/.exec(s))) return {fam: "xy", a: +m[1], b: +m[2], y: +m[3]};
  if ((m = /^(\d{4})\s*[-_ ]?\s*q([1-4])$/i.exec(s)) || (m = /^q([1-4])\s*[-_ ]?\s*(\d{4})$/i.exec(s))) {
    const y = m[1].length === 4 ? +m[1] : +m[2], q = m[1].length === 4 ? +m[2] : +m[1];
    return {fam: "q", y, q};
  }
  if ((m = /^(\d{4})-(\d{1,2})$/.exec(s)) && +m[2] >= 1 && +m[2] <= 12) return {fam: "ym", y: +m[1], m: +m[2]};
  if ((m = /^(\d{4})\s*[-_ ]?\s*m(\d{1,2})$/i.exec(s)) && +m[2] >= 1 && +m[2] <= 12) return {fam: "ym", y: +m[1], m: +m[2]};
  if ((m = /^([a-z]{3,9})[\s\-\/,.']*(\d{4})$/i.exec(s))) {
    const mo = MONTHS[m[1].toLowerCase()] || MONTHS[m[1].slice(0, 3).toLowerCase()];
    if (mo) return {fam: "ym", y: +m[2], m: mo};
  }
  if ((m = /^(\d{4})(\d{2})(\d{2})$/.exec(s)) && okDate(+m[1], +m[2], +m[3])) return {fam: "ymd", y: +m[1], m: +m[2], d: +m[3], compact: true};
  return null;
}
function dateColumn(vals) {       // vals: non-blank strings. Returns {conv(s)->string, note, ambiguous} or null
  const parsed = vals.map(dateFamily);
  const ok = parsed.filter(Boolean);
  if (!vals.length || ok.length / vals.length < 0.95) return null;
  const fams = new Set(ok.map(p => p.fam));
  if (fams.size > 1) return null;
  const fam = [...fams][0];
  if (fam === "xy") {
    const dmy = ok.some(p => p.a > 12), mdy = ok.some(p => p.b > 12);
    if (dmy === mdy) return {ambiguous: true};
    return {conv: s => { const p = dateFamily(s); if (!p) return s; const [d, m] = dmy ? [p.a, p.b] : [p.b, p.a]; return okDate(p.y, m, d) ? `${p.y}-${pad(m)}-${pad(d)}` : s; },
            note: dmy ? "day/month/year" : "month/day/year"};
  }
  if (fam === "ymd") return {conv: s => { const p = dateFamily(s); return p ? `${p.y}-${pad(p.m)}-${pad(p.d)}` : s; }, note: "year-month-day"};
  if (fam === "q") return {conv: s => { const p = dateFamily(s); return p ? `${p.y}Q${p.q}` : s; }, note: "quarters (2020Q1)"};
  return {conv: s => { const p = dateFamily(s); return p ? `${p.y}-${pad(p.m)}` : s; }, note: "months (2020-01)"};
}
function timeKey(s) {             // sortable number for a converted time value
  let m;
  if ((m = /^(\d{4})Q([1-4])$/.exec(s))) return +m[1] * 4 + (+m[2] - 1);
  if ((m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(s))) return Date.UTC(+m[1], +m[2] - 1, +m[3]);
  if ((m = /^(\d{4})-(\d{2})$/.exec(s))) return Date.UTC(+m[1], +m[2] - 1, 1);
  const n = Number(s);
  return Number.isFinite(n) ? n : NaN;
}

// ------------------------------------------------------------------ main
function prepare(text, opts) {
  opts = opts || {};
  const changes = [], warnings = [];
  const note = (t) => changes.push(t), warn = (t) => warnings.push(t);
  text = String(text == null ? "" : text);
  if (text.charCodeAt(0) === 0xFEFF) { text = text.slice(1); note("Removed a byte-order mark at the start of the file."); }
  if (!text.trim()) throw new Error("The file is empty.");

  const delim = detectDelimiter(text);
  const label = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}[delim];
  if (delim !== ",") note(`Read the file as ${label}-separated and saved it as a standard comma-separated file.`);
  let raw = tokenize(text, delim).map(r => r.map(clean));
  const before = raw.length;
  raw = raw.filter(r => r.some(v => v !== ""));
  if (raw.length < before) note(`Removed ${before - raw.length} blank row${before - raw.length > 1 ? "s" : ""}.`);
  if (raw.length < 2) throw new Error("The file needs a header row and at least one data row.");

  let width = Math.max(...raw.map(r => r.length));
  const headerWidth = raw[0].length;
  let ragged = 0;
  const body0 = raw.slice(1).map(r => { if (r.length !== headerWidth) ragged++; const o = r.slice(0, headerWidth); while (o.length < headerWidth) o.push(""); return o; });
  if (ragged) warn(`${ragged} row${ragged > 1 ? "s have" : " has"} a different number of cells than the header; short rows were padded with blanks and extra cells were cut. Check that no text contains an unquoted separator.`);
  width = headerWidth;

  // columns as arrays
  let names = raw[0].slice(), cols = [];
  for (let j = 0; j < width; j++) cols.push(body0.map(r => r[j]));

  // drop empty columns and a leftover row-index column
  const keep = [];
  for (let j = 0; j < width; j++) {
    if (cols[j].every(v => v === "")) { note(`Removed the empty column "${names[j] || "(no name)"}".`); continue; }
    if (j === 0 && NO_INDEX.has(names[j].toLowerCase()) && cols[j].every((v, i) => /^\d+$/.test(v) && (+v === i || +v === i + 1) && +v === +cols[j][0] + i)) {
      note(`Removed "${names[j] || "(no name)"}", a row-number column left by the program that saved the file.`); continue;
    }
    keep.push(j);
  }
  names = keep.map(j => names[j]); cols = keep.map(j => cols[j]);
  if (cols.length < 2) throw new Error("After cleaning, fewer than two columns remain. Check the file layout.");

  // names
  const orig = names.slice(), used = new Set(), final = [];
  names.forEach((n, i) => {
    const base = cleanName(n, i); let s = base, k = 2;
    while (used.has(s)) s = base + "_" + k++;
    used.add(s); final.push(s);
  });
  const renamed = final.map((n, i) => [orig[i], n]).filter(([a, b]) => a !== b);
  if (renamed.length) note(`Renamed ${renamed.length} column${renamed.length > 1 ? "s" : ""} to simple names: ` +
    renamed.slice(0, 8).map(([a, b]) => `"${a || "(blank)"}" → ${b}`).join(", ") + (renamed.length > 8 ? `, and ${renamed.length - 8} more` : "") + ".");
  names = final;

  // values: classify and convert each column
  const info = names.map(n => ({name: n, type: "text", missing: 0, notes: []}));
  const n0 = body0.length;
  cols.forEach((vals, j) => {
    const col = info[j], nonblank = vals.filter(v => v !== "");
    const real = nonblank.filter(v => !MISSING.has(v.toLowerCase()));
    if (!real.length) { col.type = "empty"; return; }
    // numeric?
    const comma = commaStyle(real);
    const parsed = real.map(v => parseNumber(v, comma));
    const okN = parsed.filter(Boolean).length;
    if (okN / real.length >= 0.95) {
      let pctCount = 0, badCount = 0, tokenCount = 0;
      const out = vals.map(v => {
        if (v === "") return "";
        if (MISSING.has(v.toLowerCase())) { tokenCount++; return ""; }
        const p = parseNumber(v, comma);
        if (!p) { badCount++; return ""; }
        if (p.pct) pctCount++;
        return String(p.v);
      });
      const changed = vals.some((v, i) => v !== out[i] && !(v !== "" && out[i] !== "" && Number(v) === Number(out[i]) && PLAIN.test(v)));
      cols[j] = out; col.type = "numeric";
      const bits = [];
      if (comma) bits.push("decimal commas");
      if (pctCount) bits.push("% signs (kept as percentage points, 12% → 12)");
      if (changed && !comma && !pctCount && vals.some((v, i) => out[i] !== "" && !PLAIN.test(v))) bits.push("thousands separators, currency symbols or brackets");
      if (bits.length) note(`Converted "${names[j]}" to numbers: removed ${bits.join(", ")}.`);
      if (tokenCount) note(`"${names[j]}": ${tokenCount} missing-value code${tokenCount > 1 ? "s" : ""} (NA, ".", "-", #N/A ...) set to blank.`);
      if (badCount) warn(`"${names[j]}": ${badCount} value${badCount > 1 ? "s" : ""} that are not numbers were set to blank (for example "${vals.find((v, i) => v !== "" && out[i] === "" && !MISSING.has(v.toLowerCase()))}").`);
      return;
    }
    if (okN / real.length >= 0.5) warn(`"${names[j]}" mixes numbers and text (${okN} of ${real.length} values are numbers), so it was left as text. Fix or remove the text entries if it is a variable.`);
    // yes/no
    const lower = [...new Set(real.map(v => v.toLowerCase()))];
    const pairs = [["yes", "no"], ["y", "n"], ["true", "false"], ["t", "f"]];
    const pair = lower.length <= 2 && pairs.find(p => lower.every(v => p.includes(v)));
    if (pair) {
      cols[j] = vals.map(v => v === "" ? "" : (v.toLowerCase() === pair[0] ? "1" : "0")); col.type = "numeric";
      note(`Recoded "${names[j]}" (${pair[0]}/${pair[1]}) as 1/0.`); return;
    }
    // dates
    const d = dateColumn(real);
    if (d && d.ambiguous) { warn(`"${names[j]}" looks like dates written day/month/year or month/day/year and every value could be either. It was left unchanged: write the dates as 2020-12-31 if the order matters.`); col.type = "text"; return; }
    if (d) {
      const out = vals.map(v => v === "" ? "" : d.conv(v));
      if (out.some((v, i) => v !== vals[i])) note(`Standardised the dates in "${names[j]}" (${d.note}).`);
      cols[j] = out; col.type = "date"; return;
    }
    col.type = "text";
  });

  // duplicate rows
  let rows = [];
  for (let i = 0; i < n0; i++) rows.push(cols.map(c => c[i]));
  const seenRow = new Set(), unique = [];
  for (const r of rows) { const k = JSON.stringify(r); if (!seenRow.has(k)) { seenRow.add(k); unique.push(r); } }
  const dupRows = rows.length - unique.length;
  if (dupRows && opts.dropDuplicates) { note(`Removed ${dupRows} exact duplicate row${dupRows > 1 ? "s" : ""}.`); rows = unique; }
  else if (dupRows) warn(`${dupRows} row${dupRows > 1 ? "s are" : " is"} identical to another row. They were kept, because identical rows are normal in survey data; remove them yourself if the file was pasted twice by mistake.`);
  if (!rows.length) throw new Error("No data rows remain after cleaning.");

  // structure: time and unit columns
  const colVals = j => rows.map(r => r[j]);
  let time = info.findIndex(c => c.type === "date");
  if (time < 0) time = info.findIndex((c, j) => {
    if (c.type !== "numeric" || !TIME_NAMES.test(c.name)) return false;
    const v = colVals(j);
    if (!v.every(x => /^-?\d+$/.test(x))) return false;
    if (/year|yr|fyear/i.test(c.name)) return v.every(x => +x >= 1500 && +x <= 2300);
    if (/^(t|time|obs)$/i.test(c.name)) return new Set(v).size === v.length;      // a bare "time" can be survival time: accept only a running index
    return true;
  });
  const pick = n => n ? names.indexOf(n) : -1;
  if (opts.time !== undefined) { time = pick(opts.time); if (time >= 0 && info[time].type === "text") warn(`"${names[time]}" was chosen as the time column but its values are not dates or numbers, so rows cannot be ordered by it.`); }
  if (time >= 0 && new Set(colVals(time)).size < 2) time = -1;       // one period only: nothing to order
  let unit = -1, best = -1;
  const tvAll = time >= 0 ? colVals(time) : null, nPer = tvAll ? new Set(tvAll).size : 0;
  info.forEach((c, j) => {
    if (j === time || c.type === "empty" || c.type === "date") return;
    const v = colVals(j), distinct = new Set(v.filter(x => x !== "")).size;
    if (distinct < 2 || v.some(x => x === "") || distinct > rows.length / 2) return;
    if (c.type === "numeric" && !v.every(x => /^-?\d+$/.test(x))) return;
    let score;
    const named = ID_NAMES.test(c.name);
    if (tvAll) {            // with a time column: the unit should repeat over the periods and (unit, time) should identify a row
      const unique = new Set(v.map((x, i) => x + "\u0001" + tvAll[i])).size === rows.length;
      const coverage = Math.min(1, rows.length / (distinct * nPer));
      if (!(named || (unique && coverage >= 0.9 && distinct >= 3)) || coverage < 0.5) return;
      score = 10 * coverage + (named ? 2 : 0) + (c.type === "text" ? 1 : 0) + (unique ? 1 : 0);
    } else {                // without one: only a clearly named identifier repeated over rows
      if (!named || rows.length / distinct < 2) return;
      score = 1 + (c.type === "text" ? 1 : 0);
    }
    if (score > best) { best = score; unit = j; }
  });
  if (opts.unit !== undefined) unit = pick(opts.unit);
  if (unit >= 0 && unit === time) unit = -1;
  const structure = {unit: unit >= 0 ? names[unit] : null, time: time >= 0 ? names[time] : null, kind: "cross-section"};
  const tk = time >= 0 ? colVals(time).map(timeKey) : [];
  if (time >= 0 && tk.some(Number.isNaN)) { structure.time = null; time = -1; }
  const periodDup = time >= 0 && unit < 0 && tk.length - new Set(tk).size > 0;
  if (periodDup) {
    structure.kind = "unclear";
    if (info[time].type === "date" || /year|yr|date|period|quarter|month|wave/i.test(names[time])) warn(`Several rows share the same "${names[time]}" and no unit column (country, firm, id ...) was identified, so the rows were not reordered. If this is a panel, choose the unit and time columns yourself in step 4.`);
    time = -1;
  }
  if (time >= 0) {
    const order = rows.map((_, i) => i);
    const uk = unit >= 0 ? colVals(unit) : null;
    const numUnit = unit >= 0 && info[unit].type === "numeric";
    order.sort((a, b) => {
      if (uk) { const c = numUnit ? uk[a] - uk[b] : uk[a].localeCompare(uk[b], undefined, {numeric: true}); if (c) return c; }
      return tk[a] - tk[b];
    });
    const sorted = order.map(i => rows[i]), moved = order.some((o, i) => o !== i);
    rows = sorted;
    if (moved) note(uk ? `Sorted the rows by "${names[unit]}" and then "${names[time]}".` : `Sorted the rows by "${names[time]}".`);
    const keys = rows.map(r => (unit >= 0 ? r[unit] + "\u0001" : "") + r[time]);
    const dup = keys.length - new Set(keys).size;
    if (dup) warn(`${dup} row${dup > 1 ? "s share" : " shares"} the same ${unit >= 0 ? `"${names[unit]}" and "${names[time]}"` : `"${names[time]}"`} as another row. Panel and time-series models need one row per ${unit >= 0 ? "unit and period" : "period"}; check or aggregate these rows.`);
    structure.kind = unit >= 0 ? "panel" : "time series";
    const tvals = rows.map(r => timeKey(r[time]));
    if (unit >= 0) {
      const units = new Map();
      rows.forEach((r, i) => { const u = r[unit]; if (!units.has(u)) units.set(u, []); units.get(u).push(tvals[i]); });
      const periods = new Set(tvals), lens = [...units.values()].map(a => new Set(a).size);
      structure.units = units.size; structure.periods = periods.size;
      structure.balanced = lens.every(l => l === periods.size);
      structure.tMin = Math.min(...lens); structure.tMax = Math.max(...lens);
      if (!structure.balanced) warn(`The panel is unbalanced: ${units.size} units, ${periods.size} periods in total, but each unit appears ${structure.tMin === structure.tMax ? structure.tMin : `${structure.tMin} to ${structure.tMax}`} times. Most panel estimators accept this; difference GMM, panel VAR and mean-group methods need enough consecutive periods per unit.`);
    } else structure.periods = new Set(tvals).size;
    if (info[time].type === "numeric") {
      const per = [...new Set(tvals)].sort((a, b) => a - b);
      let gaps = 0; for (let i = 1; i < per.length; i++) if (per[i] - per[i - 1] > 1) gaps += per[i] - per[i - 1] - 1;
      if (gaps) warn(`"${names[time]}" skips ${gaps} value${gaps > 1 ? "s" : ""} between its first and last period (for example a missing year), so lags and differences may not mean what you expect.`);
    }
  } else if (!periodDup) {
    structure.kind = "cross-section";
  }

  // per-column checks
  let numericVars = 0;
  info.forEach((c, j) => {
    const v = colVals(j), blank = v.filter(x => x === "").length;
    c.missing = blank; c.pctMissing = v.length ? blank / v.length : 0; c.distinct = new Set(v.filter(x => x !== "")).size;
    if (c.type === "numeric") {
      numericVars++;
      const nums = v.filter(x => x !== "").map(Number);
      if (new Set(nums).size <= 1 && nums.length) { c.notes.push("constant"); warn(`"${c.name}" has the same value in every row, so it cannot explain anything and will be dropped by most models.`); }
      for (const s of SENTINELS) {
        const hit = nums.filter(x => x === s).length;
        if (hit) {
          const others = nums.filter(x => x !== s).map(Math.abs).sort((a, b) => a - b), med = others.length ? others[others.length >> 1] : 0;
          if (med < Math.abs(s) / 20) { c.notes.push("code " + s); warn(`"${c.name}" contains ${hit} value${hit > 1 ? "s" : ""} equal to ${s}, which is often a code for "missing" in survey data. They were NOT changed; replace them with blanks if that is what they mean.`); }
        }
      }
    }
    if (c.pctMissing > 0.4 && c.type !== "empty") warn(`"${c.name}" is ${(100 * c.pctMissing).toFixed(0)}% missing; including it would remove most of your observations.`);
  });
  const numIdx = info.map((c, j) => c.type === "numeric" ? j : -1).filter(j => j >= 0);
  const completeRows = rows.filter(r => numIdx.every(j => r[j] !== "")).length;

  const csv = [names.map(csvCell).join(",")].concat(rows.map(r => r.map(csvCell).join(","))).join("\n") + "\n";
  return {csv, rows: rows.length, cols: names.length, rowsIn: n0, changes, warnings, columns: info, structure,
          ready: {numericVars, completeRows, totalRows: rows.length}};
}


// ------------------------------------------------------------------ optional cleaning (the user chooses each action)
function quantile(sorted, p) { const x = (sorted.length - 1) * p, lo = Math.floor(x), hi = Math.ceil(x); return sorted[lo] + (sorted[hi] - sorted[lo]) * (x - lo); }
const ord = n => { const k = n % 100, m = n % 10; return n + (k >= 11 && k <= 13 ? "th" : m === 1 ? "st" : m === 2 ? "nd" : m === 3 ? "rd" : "th"); };
const tidy = v => String(Number(v.toPrecision(12)));
function summarize(names, rows) {
  const columns = names.map((n, j) => {
    const v = rows.map(r => r[j]), blank = v.filter(x => x === "").length, filled = v.filter(x => x !== "");
    const type = !filled.length ? "empty" : filled.every(x => Number.isFinite(Number(x))) ? "numeric" : "text";
    return {name: n, type, missing: blank, pctMissing: v.length ? blank / v.length : 0, distinct: new Set(filled).size, notes: []};
  });
  const num = columns.map((c, j) => c.type === "numeric" ? j : -1).filter(j => j >= 0);
  return {columns, ready: {numericVars: num.length, completeRows: rows.filter(r => num.every(j => r[j] !== "")).length, totalRows: rows.length}};
}
/* advanced(csv, opts, structure): opts = {drop:[names], duplicates:bool, fill:{name:{method,value}}, flag:bool,
 *   outlier:{method:"winsor"|"trim", lo, hi, cols:[names]}, dropMissing:bool}.  Order: drop, duplicates, fill, outliers, dropMissing.
 * fill methods: mean, median, mode, constant, ffill, bfill, interp (the last three run within each unit of a panel). */
function advanced(csv, o, structure) {
  o = o || {};
  const log = [], warnings = [];
  let rows = tokenize(csv, ","), names = rows.shift();
  rows = rows.filter(r => r.some(v => v !== ""));
  const n0 = rows.length;
  const isNum = j => rows.some(r => r[j] !== "") && rows.every(r => r[j] === "" || Number.isFinite(Number(r[j])));

  if (o.drop && o.drop.length) {
    const keep = names.map((n, j) => o.drop.includes(n) ? -1 : j).filter(j => j >= 0);
    if (keep.length < 2) throw new Error("Keep at least two variables.");
    rows = rows.map(r => keep.map(j => r[j])); names = keep.map(j => names[j]);
    log.push(`Dropped ${o.drop.length} variable${o.drop.length > 1 ? "s" : ""}: ${o.drop.join(", ")}.`);
  }
  if (o.duplicates) {
    const seen = new Set(); const before = rows.length;
    rows = rows.filter(r => { const k = JSON.stringify(r); if (seen.has(k)) return false; seen.add(k); return true; });
    log.push(`Removed ${before - rows.length} exact duplicate row${before - rows.length === 1 ? "" : "s"}.`);
  }
  // fill missing values
  const flags = [];
  const unitJ = structure && structure.unit ? names.indexOf(structure.unit) : -1;
  const groups = () => { const g = new Map(); rows.forEach((r, i) => { const k = unitJ >= 0 ? r[unitJ] : ""; if (!g.has(k)) g.set(k, []); g.get(k).push(i); }); return [...g.values()]; };
  for (const [col, spec] of Object.entries(o.fill || {})) {
    const j = names.indexOf(col), method = spec && spec.method;
    if (j < 0 || !method || method === "none") continue;
    const miss = rows.map(r => r[j] === "");
    const nMiss = miss.filter(Boolean).length;
    if (!nMiss) continue;
    const numeric = isNum(j);
    if (nMiss / rows.length > 0.3) warnings.push(`"${col}" was ${(100 * nMiss / rows.length).toFixed(0)}% missing before filling. Filled values will dominate the estimate; consider whether the filling assumption is believable.`);
    if (["mean", "median", "interp"].includes(method) && !numeric) { warnings.push(`"${col}" is not numeric, so ${method} filling was skipped.`); continue; }
    let desc = method;
    if (["mean", "median", "mode", "constant"].includes(method)) {
      let val;
      const vals = rows.map(r => r[j]).filter(x => x !== "");
      if (method === "constant") { val = String(spec.value == null ? "" : spec.value).trim(); if (val === "" || (numeric && !Number.isFinite(Number(val)))) { warnings.push(`"${col}": the constant to fill with is empty or not a number, so it was skipped.`); continue; } }
      else if (method === "mode") { const c = new Map(); vals.forEach(v => c.set(v, (c.get(v) || 0) + 1)); val = [...c.entries()].sort((a, b) => b[1] - a[1])[0][0]; }
      else { const nums = vals.map(Number).sort((a, b) => a - b); val = tidy(method === "mean" ? nums.reduce((a, b) => a + b, 0) / nums.length : quantile(nums, 0.5)); }
      rows.forEach(r => { if (r[j] === "") r[j] = val; });
      desc = `${method === "constant" ? "the constant" : "the " + method} (${val})`;
    } else {
      for (const idx of groups()) {
        if (method === "ffill") { let last = ""; idx.forEach(i => { if (rows[i][j] === "") rows[i][j] = last; else last = rows[i][j]; }); }
        else if (method === "bfill") { let next = ""; for (let k = idx.length - 1; k >= 0; k--) { const i = idx[k]; if (rows[i][j] === "") rows[i][j] = next; else next = rows[i][j]; } }
        else if (method === "interp") {
          const known = idx.filter(i => rows[i][j] !== "");
          for (let q = 0; q + 1 < known.length; q++) {
            const a = known[q], b = known[q + 1], pa = idx.indexOf(a), pb = idx.indexOf(b), ya = Number(rows[a][j]), yb = Number(rows[b][j]);
            for (let k = pa + 1; k < pb; k++) rows[idx[k]][j] = tidy(ya + (yb - ya) * (k - pa) / (pb - pa));
          }
        }
      }
      desc = {ffill: "the previous value", bfill: "the next value", interp: "linear interpolation"}[method] + (unitJ >= 0 ? ` within each ${names[unitJ]}` : "");
    }
    const left = rows.filter(r => r[j] === "").length, done = nMiss - left;
    log.push(`Filled ${done} of ${nMiss} missing value${nMiss > 1 ? "s" : ""} in "${col}" with ${desc}.` + (left ? ` ${left} could not be filled (no earlier or later value) and stay blank.` : ""));
    if (o.flag && done) flags.push([col, miss]);
  }
  for (const [col, miss] of flags) {
    let nm = col + "_imputed"; while (names.includes(nm)) nm += "_flag";
    names.push(nm); rows.forEach((r, i) => r.push(miss[i] && r[names.indexOf(col)] !== "" ? "1" : "0"));
  }
  if (flags.length) log.push(`Added ${flags.length} 0/1 column${flags.length > 1 ? "s" : ""} marking which values were filled.`);

  // outliers
  const ou = o.outlier;
  if (ou && ou.method && ou.method !== "none" && (ou.cols || []).length) {
    const lo = Number(ou.lo), hi = Number(ou.hi);
    if (!(lo >= 0 && lo < hi && hi <= 100)) throw new Error("The outlier percentiles must satisfy 0 ≤ lower < upper ≤ 100.");
    const out = new Set();
    for (const col of ou.cols) {
      const j = names.indexOf(col); if (j < 0 || !isNum(j)) continue;
      const vals = rows.map(r => r[j]).filter(x => x !== "").map(Number).sort((a, b) => a - b);
      if (vals.length < 4) continue;
      const a = quantile(vals, lo / 100), b = quantile(vals, hi / 100); let n = 0;
      rows.forEach((r, i) => {
        if (r[j] === "") return; const v = Number(r[j]);
        if (v < a || v > b) { n++; if (ou.method === "winsor") r[j] = tidy(Math.min(Math.max(v, a), b)); else out.add(i); }
      });
      log.push(`"${col}": ${n} value${n === 1 ? "" : "s"} outside the ${ord(lo)} to ${ord(hi)} percentile (${tidy(a)} to ${tidy(b)}) ${ou.method === "winsor" ? "set to the nearest bound (winsorized)" : "marked for removal"}.`);
    }
    if (ou.method === "trim" && out.size) { rows = rows.filter((_, i) => !out.has(i)); log.push(`Trimmed ${out.size} row${out.size > 1 ? "s" : ""} containing such values.`); warnings.push(`Trimming removed ${out.size} observations. This changes the sample, so say so in your paper.`); }
  }
  if (o.dropMissing) {
    const before = rows.length; rows = rows.filter(r => r.every(v => v !== ""));
    log.push(`Dropped ${before - rows.length} row${before - rows.length === 1 ? "" : "s"} with a missing value in any column.`);
    if (before && (before - rows.length) / before > 0.2) warnings.push(`Dropping rows with any missing value removed ${(100 * (before - rows.length) / before).toFixed(0)}% of the observations. Models already skip incomplete rows among the columns you choose, so this is usually not needed.`);
  }
  if (!rows.length) throw new Error("No rows remain with these options.");
  const out = [names.map(csvCell).join(",")].concat(rows.map(r => r.map(csvCell).join(","))).join("\n") + "\n";
  return Object.assign({csv: out, rows: rows.length, cols: names.length, rowsIn: n0, log, warnings}, summarize(names, rows));
}

const api = {prepare, advanced, parseNumber, dateFamily};
if (typeof module !== "undefined" && module.exports) module.exports = api; else root.OESPrepare = api;
})(typeof self !== "undefined" ? self : this);
