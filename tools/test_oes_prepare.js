// node tools/test_oes_prepare.js : checks the OES auto-prepare module on deliberately messy files
const assert = require("assert"), P = require("../oes/auto-prepare.js");
const t = (name, fn) => { fn(); console.log("ok  " + name); };

t("panel with messy numbers, codes, dates and sorting", () => {
  const csv = "﻿Country Name;Year;GDP (USD);Inflation %;Open?\r\n" +
    "Korea;2001;\"1.234,5\";3,2;yes\nKorea;2000;\"1.100,0\";NA;no\nJapan;2001;\"2.000,0\";-0,5;yes\nJapan;2000;\"1.900,5\";.;no\nJapan;2000;\"1.900,5\";.;no\n;;;;\n";
  const r = P.prepare(csv);
  assert.deepStrictEqual(r.csv.split("\n")[0], "country_name,Year,gdp_usd,inflation_pct,open");
  assert.strictEqual(r.rows, 5); assert.strictEqual(r.rowsIn, 5); assert.ok(r.warnings.some(w => w.includes("identical")));
  assert.strictEqual(P.prepare(csv, {dropDuplicates: true}).rows, 4);
  assert.strictEqual(r.structure.kind, "panel"); assert.strictEqual(r.structure.unit, "country_name"); assert.strictEqual(r.structure.time, "Year");
  assert.strictEqual(r.csv.split("\n")[1], "japan,2000".replace("japan", "Japan") + ",1900.5,,0");
  assert.ok(r.csv.includes("Korea,2001,1234.5,3.2,1"));
  assert.ok(r.structure.balanced);
});
t("US-style numbers, currency, brackets, percent", () => {
  assert.strictEqual(P.parseNumber("$1,234.50", false).v, 1234.5);
  assert.strictEqual(P.parseNumber("(250)", false).v, -250);
  assert.strictEqual(P.parseNumber("12.5%", false).v, 12.5);
  assert.strictEqual(P.parseNumber("1,5", false), null);
  assert.strictEqual(P.parseNumber("−3", false).v, -3);
  assert.strictEqual(P.parseNumber("1e-3", false).v, 0.001);
});
t("quarterly dates and time-series order", () => {
  const r = P.prepare("date,y,x\nQ2 2020,2,5\n2020Q1,1,4\n2020-Q3,3,6\n");
  assert.ok(r.csv.startsWith("date,y,x\n2020Q1,1,4\n2020Q2,2,5\n2020Q3,3,6"));
  assert.strictEqual(r.structure.kind, "time series");
});
t("day/month order: certain vs ambiguous", () => {
  const a = P.prepare("d,v\n31/12/2020,1\n01/01/2021,2\n");
  assert.ok(a.csv.includes("2020-12-31") && a.csv.includes("2021-01-01"));
  const b = P.prepare("d,v\n01/02/2020,1\n03/04/2020,2\n");
  assert.ok(b.csv.includes("01/02/2020")); assert.ok(b.warnings.some(w => w.includes("every value could be either")));
});
t("ISO country code NA stays text; NA in numeric column is blank", () => {
  const r = P.prepare("iso,year,y\nNA,2000,1\nNA,2001,NA\nUS,2000,3\nUS,2001,4\n");
  assert.ok(r.csv.includes("NA,2000,1") && r.csv.includes("NA,2001,"));
  assert.strictEqual(r.columns[0].type, "text");
});
t("row-index column, duplicate headers, empty column", () => {
  const r = P.prepare(",a,a,b,empty\n0,1,2,3,\n1,4,5,6,\n2,7,8,9,\n");
  assert.strictEqual(r.csv.split("\n")[0], "a,a_2,b");
});
t("missing-value codes, constants and heavy missingness are flagged, not changed", () => {
  const r = P.prepare("id,year,y,k,z\n1,2000,5,-999,1\n1,2001,6,3,1\n2,2000,7,4,\n2,2001,8,5,\n");
  assert.ok(r.warnings.some(w => w.includes("-999")));
  assert.ok(r.csv.includes("-999"));
  assert.ok(r.warnings.some(w => w.startsWith('"z" has the same value')));
  assert.ok(r.columns.find(c => c.name === "z").missing === 2);
});
t("unbalanced panel and duplicate keys", () => {
  const r = P.prepare("id,year,y\nA,2000,1\nA,2001,2\nA,2001,3\nB,2000,4\n");
  assert.ok(r.warnings.some(w => w.includes("unbalanced")));
  assert.ok(r.warnings.some(w => w.includes("share")));
});
t("Korean headers survive", () => {
  const r = P.prepare("국가,연도,GDP\n한국,2000,1\n한국,2001,2\n일본,2000,3\n일본,2001,4\n");
  assert.ok(r.csv.startsWith("국가,연도,GDP"));
});
t("year-month ISO is kept; month names are standardised", () => {
  const a = P.prepare("date,y\n1985-01,1\n1985-02,2\n"); assert.ok(a.csv.includes("1985-01,1")); assert.strictEqual(a.changes.length, 0); assert.strictEqual(a.structure.kind, "time series");
  const b = P.prepare("m,y\nFeb 2020,2\nJan 2020,1\n"); assert.ok(b.csv.includes("2020-01,1\n2020-02,2"));
});
t("numeric panel id named nr is found by structure and panel order is kept", () => {
  const r = P.prepare("nr,year,w\n13,1980,1\n13,1981,2\n17,1980,3\n17,1981,4\n21,1980,5\n21,1981,6\n");
  assert.strictEqual(r.structure.unit, "nr"); assert.strictEqual(r.structure.kind, "panel");
});
t("repeated periods without a unit are not reordered", () => {
  const r = P.prepare("year,y,x\n2001,1,1\n2000,2,2\n2001,3,3\n2000,4,4\n");
  assert.ok(r.csv.startsWith("year,y,x\n2001,1,1\n2000,2,2")); assert.strictEqual(r.structure.kind, "unclear");
});
t("advanced: fill (median, ffill/interp within unit, flag), outliers, drop", () => {
  const csv = "id,year,y,x,z\nA,1,1,5,a\nA,2,,6,\nA,3,3,,a\nB,1,10,7,b\nB,2,,8,b\nB,3,30,1000,b\n";
  const st = {unit: "id", time: "year"};
  const a = P.advanced(csv, {fill: {y: {method: "interp"}, x: {method: "median"}, z: {method: "mode"}}, flag: true}, st);
  const row = i => a.csv.split("\n")[i + 1].split(",");
  assert.deepStrictEqual(row(1).slice(2, 5), ["2", "6", "b"]);     // A year 2 (mode of z is b): y interpolated 1..3
  assert.deepStrictEqual(row(4).slice(2, 4), ["20", "8"]);          // B year 2: y interpolated 10..30
  assert.strictEqual(row(2)[3], "7");                               // x median of 5,6,7,8,1000 = 7
  assert.ok(a.csv.split("\n")[0].includes("y_imputed") && row(1)[5] === "1" && row(0)[5] === "0");
  const f = P.advanced(csv, {fill: {y: {method: "ffill"}}}, st); assert.strictEqual(f.csv.split("\n")[2].split(",")[2], "1");
  const w = P.advanced(csv, {outlier: {method: "winsor", lo: 0, hi: 80, cols: ["x"]}}, st); assert.ok(Number(w.csv.split("\n")[6].split(",")[3]) < 1000);
  const tr = P.advanced(csv, {outlier: {method: "trim", lo: 0, hi: 80, cols: ["x"]}}, st); assert.strictEqual(tr.rows, 5);
  const dm = P.advanced(csv, {dropMissing: true}, st); assert.strictEqual(dm.rows, 3);
  const dr = P.advanced(csv, {drop: ["z"]}, st); assert.strictEqual(dr.cols, 4);
  assert.throws(() => P.advanced(csv, {outlier: {method: "winsor", lo: 50, hi: 10, cols: ["x"]}}, st));
  assert.strictEqual(P.advanced("a,b\n1,2\n1,2\n", {duplicates: true}, {}).rows, 1);
});
t("errors", () => { assert.throws(() => P.prepare("")); assert.throws(() => P.prepare("a,b\n")); });
