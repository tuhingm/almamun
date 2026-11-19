/* OES publication report: renders a report produced by the engine (result.report) and exports it.
   Everything runs in the browser: copy buttons (text, HTML for Word, LaTeX, CSV), figure downloads (SVG, PNG at 300 dpi, PDF),
   the Word-compatible document and the "Download everything" ZIP. No server and no external library.
   Usage: OESReport.render(containerElement, report). */
(function () {
  "use strict";
  const esc = s => String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const enc = new TextEncoder();
  const b64ToBytes = b => Uint8Array.from(atob(b), c => c.charCodeAt(0));
  const bytesToB64 = u => { let s = ""; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(s); };

  // ------------------------------------------------------------------ CRC32, PNG metadata, ZIP (stored, no compression)
  const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
  function crc32(u, c = 0xFFFFFFFF) { for (let i = 0; i < u.length; i++) c = CRC[(c ^ u[i]) & 255] ^ (c >>> 8); return c >>> 0; }
  const crcFinal = c => (c ^ 0xFFFFFFFF) >>> 0;

  function pngWithDpi(png, dpi) {                       // insert a pHYs chunk right after IHDR so the file is labelled with its resolution
    const ppm = Math.round(dpi / 0.0254), chunk = new Uint8Array(21), dv = new DataView(chunk.buffer);
    dv.setUint32(0, 9); chunk.set([0x70, 0x48, 0x59, 0x73], 4); dv.setUint32(8, ppm); dv.setUint32(12, ppm); chunk[16] = 1;
    dv.setUint32(17, crcFinal(crc32(chunk.subarray(4, 17))));
    const out = new Uint8Array(png.length + 21); out.set(png.subarray(0, 33)); out.set(chunk, 33); out.set(png.subarray(33), 54);
    return out;
  }

  function makeZip(files) {                             // files: [{name, data: Uint8Array}]
    const parts = [], central = []; let offset = 0; const now = new Date();
    const dosTime = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1), dosDate = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
    for (const f of files) {
      const name = enc.encode(f.name), crc = crcFinal(crc32(f.data)), h = new DataView(new ArrayBuffer(30));
      h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(6, 0x0800, true); h.setUint16(8, 0, true); h.setUint16(10, dosTime, true); h.setUint16(12, dosDate, true);
      h.setUint32(14, crc, true); h.setUint32(18, f.data.length, true); h.setUint32(22, f.data.length, true); h.setUint16(26, name.length, true); h.setUint16(28, 0, true);
      parts.push(new Uint8Array(h.buffer), name, f.data);
      const c = new DataView(new ArrayBuffer(46));
      c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0x0800, true); c.setUint16(10, 0, true); c.setUint16(12, dosTime, true); c.setUint16(14, dosDate, true);
      c.setUint32(16, crc, true); c.setUint32(20, f.data.length, true); c.setUint32(24, f.data.length, true); c.setUint16(28, name.length, true); c.setUint32(42, offset, true);
      central.push(new Uint8Array(c.buffer), name);
      offset += 30 + name.length + f.data.length;
    }
    const csize = central.reduce((a, b) => a + b.length, 0), e = new DataView(new ArrayBuffer(22));
    e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true); e.setUint16(10, files.length, true); e.setUint32(12, csize, true); e.setUint32(16, offset, true);
    return new Blob([...parts, ...central, new Uint8Array(e.buffer)], {type: "application/zip"});
  }

  // ------------------------------------------------------------------ figures to PNG
  function svgToPng(fig, dpi = 300) {
    return new Promise((resolve, reject) => {
      const w = Math.round(fig.width_in * dpi), h = Math.round(fig.height_in * dpi);
      const img = new Image();
      img.onload = () => {
        const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
        const cx = cv.getContext("2d"); cx.fillStyle = "#fff"; cx.fillRect(0, 0, w, h); cx.drawImage(img, 0, 0, w, h);
        cv.toBlob(async b => b ? resolve(pngWithDpi(new Uint8Array(await b.arrayBuffer()), dpi)) : reject(new Error("PNG export failed")), "image/png");
      };
      img.onerror = () => reject(new Error("The figure could not be drawn for PNG export."));
      img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(fig.svg);
    });
  }

  // ------------------------------------------------------------------ downloads and clipboard
  function save(name, data, type) {
    const a = document.createElement("a"); a.href = URL.createObjectURL(data instanceof Blob ? data : new Blob([data], {type})); a.download = name;
    document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }
  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); return true; } catch (e) { /* fall through */ }
    const ta = document.createElement("textarea"); ta.value = text; ta.style.cssText = "position:fixed;left:-9999px"; document.body.appendChild(ta); ta.select();
    let ok = false; try { ok = document.execCommand("copy"); } catch (e) { ok = false; } ta.remove(); return ok;
  }
  async function copyHtml(html, plain) {   // rich copy: Word, Google Docs and Outlook read text/html; editors read text/plain
    try {
      if (navigator.clipboard && window.ClipboardItem) {
        await navigator.clipboard.write([new ClipboardItem({"text/html": new Blob([html], {type: "text/html"}), "text/plain": new Blob([plain], {type: "text/plain"})})]);
        return true;
      }
    } catch (e) { /* fall through */ }
    const d = document.createElement("div"); d.contentEditable = "true"; d.style.cssText = "position:fixed;left:-9999px;background:#fff"; d.innerHTML = html; document.body.appendChild(d);
    const r = document.createRange(); r.selectNodeContents(d); const s = getSelection(); s.removeAllRanges(); s.addRange(r);
    let ok = false; try { ok = document.execCommand("copy"); } catch (e) { ok = false; } s.removeAllRanges(); d.remove(); return ok;
  }

  // ------------------------------------------------------------------ documents
  function inlineHtml(rep) {                // images were written as cid: references for Word; the browser version embeds the SVG itself
    let h = rep.exports.html_images;
    for (const s of rep.steps) for (const f of s.figures) h = h.replace(new RegExp(`<img class="fig" src="cid:${f.name}"[^>]*>`), () => f.svg);
    return h;
  }
  async function wordDoc(rep, pngs) {       // MHTML: opens in Word with the figures embedded, and keeps the inline-styled tables
    const html = rep.exports.html_images.replace('<html lang="en">', '<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word" xmlns="http://www.w3.org/TR/REC-html40" lang="en">')
      .replace("<head>", '<head><!--[if gte mso 9]><xml><w:WordDocument><w:View>Print</w:View><w:Zoom>100</w:Zoom></w:WordDocument></xml><![endif]--><style>@page{size:8.5in 11in;margin:1in}</style>');
    const B = "----=_OES_" + Date.now().toString(36), L = [];
    const chunk = s => s.replace(/(.{76})/g, "$1\r\n");
    L.push("MIME-Version: 1.0", `Content-Type: multipart/related; boundary="${B}"; type="text/html"`, "", `--${B}`, 'Content-Type: text/html; charset="utf-8"', "Content-Transfer-Encoding: base64", "Content-Location: file:///report.htm", "", chunk(bytesToB64(enc.encode(html))));
    for (const s of rep.steps) for (const f of s.figures) {
      L.push(`--${B}`, "Content-Type: image/png", "Content-Transfer-Encoding: base64", `Content-ID: <${f.name}>`, `Content-Location: cid:${f.name}`, "", chunk(bytesToB64(pngs[f.name])));
    }
    L.push(`--${B}--`, "");
    return new Blob([L.join("\r\n")], {type: "application/msword"});
  }

  async function allFigures(rep, dpi = 300) {
    const out = {};
    for (const s of rep.steps) for (const f of s.figures) out[f.name] = await svgToPng(f, dpi);
    return out;
  }

  async function everything(rep, note) {
    note("Drawing the figures (PNG, 300 dpi)…");
    const pngs = await allFigures(rep);
    const F = [], add = (name, text) => F.push({name, data: typeof text === "string" ? enc.encode(text) : text});
    for (const s of rep.steps) {
      for (const t of s.tables) { add(`tables/${t.name}.csv`, "﻿" + t.csv); add(`tables/${t.name}.tex`, t.latex); add(`tables/${t.name}.html`, t.html); add(`tables/${t.name}.txt`, t.text); }
      for (const f of s.figures) { add(`figures/${f.name}.svg`, f.svg); add(`figures/${f.name}.pdf`, b64ToBytes(f.pdf_b64)); add(`figures/${f.name}.png`, pngs[f.name]); }
    }
    add("report.html", inlineHtml(rep));
    add("report.tex", rep.exports.tex);
    add("manuscript.txt", rep.manuscript);
    add("captions.txt", rep.steps.flatMap(s => [...s.tables.map(t => t.caption), ...s.figures.map(f => f.full_caption)]).join("\n"));
    note("Building the Word document…");
    const doc = await wordDoc(rep, pngs); F.push({name: "report.doc", data: new Uint8Array(await doc.arrayBuffer())});
    add("README.txt", `${rep.title}\nOpen Econometric Systems report. Every number comes from the model run on the data supplied.\n\nreport.html   full report (open in any browser)\nreport.doc    Word-compatible document with the figures (open in Word)\nreport.tex    LaTeX source; compile with the PDF files in figures/ (packages: booktabs, threeparttable, graphicx, times)\nmanuscript.txt  methods, results text, limitations and references as plain text\ntables/       every table as CSV (UTF-8, opens in Excel), LaTeX (booktabs), HTML (pastes into Word) and plain text\nfigures/      every figure as SVG, PDF (vector) and PNG (300 dpi)\ncaptions.txt  table and figure captions\n`);
    return makeZip(F);
  }

  // ------------------------------------------------------------------ page
  const CSS = `
  .oesr{margin:6px 0 18px;color:var(--ink,#1b2437)}
  .oesr .bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0 12px}
  .oesr .toc{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 14px;padding:0;list-style:none}
  .oesr .toc a{display:inline-flex;gap:6px;align-items:center;border:1px solid var(--line,#d5dcea);background:#f5f7fc;border-radius:30px;padding:5px 12px;font-size:12px;font-weight:700;color:#41506a;text-decoration:none}
  .oesr .toc a.off{opacity:.45;text-decoration:line-through}
  .oesr .toc input{margin:0}
  .oesr .rstep{border:1px solid var(--line,#d5dcea);border-radius:12px;margin:0 0 14px;background:#fff}
  .oesr .rhead{display:flex;gap:10px;align-items:center;justify-content:space-between;padding:12px 16px;cursor:pointer;background:#f7f9fd;border-radius:12px}
  .oesr .rhead h3{margin:0;font:800 16px Manrope,sans-serif}
  .oesr .rhead small{color:var(--muted,#55617a);font-size:12px}
  .oesr .rbody{padding:4px 16px 14px}
  .oesr .purpose{color:var(--muted,#55617a);font-size:13px;margin:8px 0}
  .oesr .badge{display:inline-block;font:700 10px "DM Sans",sans-serif;letter-spacing:.06em;text-transform:uppercase;border-radius:20px;padding:3px 9px;margin:2px 0 8px}
  .oesr .badge.ok{background:#e4f5ec;color:#0f7a4f}.oesr .badge.no{background:#fdeaea;color:#a12a2a}
  .oesr .valnote{font-size:11.5px;color:var(--muted,#55617a);margin:0 0 8px}
  .oesr .tbl{margin:14px 0}.oesr .tbl .tw{overflow-x:auto;background:#fff;padding:2px 0}
  .oesr .tbl .tw p{margin:6px 0 !important}
  .oesr .fig{margin:16px 0}.oesr .fig .fw{overflow-x:auto}.oesr .fig svg{max-width:100%;height:auto;display:block}
  .oesr .cap{font:13px/1.5 "Times New Roman",Times,serif;margin:4px 0}
  .oesr .acts{display:flex;flex-wrap:wrap;gap:5px;margin:6px 0}
  .oesr .acts button{border:1px solid var(--line,#d5dcea);background:#fff;border-radius:6px;padding:4px 9px;font:700 10.5px "DM Sans",sans-serif;letter-spacing:.04em;color:var(--navy,#0033a0);cursor:pointer}
  .oesr .acts button:hover{background:#e9eef8}.oesr .acts button.done{background:#e4f5ec;color:#0f7a4f;border-color:#b8e3cc}
  .oesr .para{position:relative;font:15px/1.6 "Times New Roman",Times,serif;margin:8px 0;padding:6px 8px;border-radius:6px;background:#fafbfe}
  .oesr .para.warning{background:#fff5f2;border-left:3px solid #b3392a}
  .oesr .para .acts{margin:4px 0 0}
  .oesr h4{font:800 13px Manrope,sans-serif;margin:14px 0 4px;letter-spacing:.02em;text-transform:uppercase;color:var(--navy,#0033a0)}
  .oesr .refs p{font:13px/1.5 "Times New Roman",Times,serif;margin:3px 0;padding-left:1.6em;text-indent:-1.6em}
  .oesr .msg{font-size:12px;color:var(--muted,#55617a)}
  @media (max-width:640px){.oesr .rbody{padding:2px 10px 12px}.oesr .rhead{padding:10px 12px}}`;

  function render(el, rep) {
    if (!document.getElementById("oesr-css")) { const st = document.createElement("style"); st.id = "oesr-css"; st.textContent = CSS; document.head.appendChild(st); }
    const nT = rep.steps.reduce((a, s) => a + s.tables.length, 0), nF = rep.steps.reduce((a, s) => a + s.figures.length, 0);
    const act = (label, cls, extra) => `<button type="button" data-a="${cls}" ${extra || ""}>${label}</button>`;
    let h = `<div class="oesr" id="oesr-root"><h3 style="margin-top:0">${esc(rep.title)}: publication report</h3>
      <p class="sub" style="margin-bottom:6px">${rep.steps.length} steps, ${nT} tables and ${nF} figures. Every number comes from the model run on your data; tables, figures and text can be copied or downloaded and pasted into a manuscript.</p>
      <div class="bar">
        <button class="btn" type="button" data-g="zip">Download everything (ZIP)</button>
        <button class="btn2" type="button" data-g="html">Full report (HTML)</button>
        <button class="btn2" type="button" data-g="doc">Word document</button>
        <button class="btn2" type="button" data-g="tex">LaTeX</button>
        <button class="btn2" type="button" data-g="ms">Copy all text</button>
        <span class="msg" id="oesr-msg" role="status" aria-live="polite"></span></div>
      <ul class="toc" aria-label="Steps of the report">${rep.steps.map((s, i) => `<li><a href="#oesr-s-${s.key}" data-toc="${s.key}">${i + 1}. ${esc(s.title)}</a></li>`).join("")}<li><a href="#oesr-text">Text for the paper</a></li></ul>`;
    for (const [i, s] of rep.steps.entries()) {
      h += `<section class="rstep" id="oesr-s-${s.key}" data-step="${s.key}"><div class="rhead" data-toggle="${s.key}" role="button" tabindex="0" aria-expanded="true"><h3>${i + 1}. ${esc(s.title)}</h3><small>${s.tables.length} table${s.tables.length === 1 ? "" : "s"} · ${s.figures.length} figure${s.figures.length === 1 ? "" : "s"} ▾</small></div><div class="rbody">`;
      if (s.purpose) h += `<p class="purpose">${esc(s.purpose)}</p>`;
      h += s.validated === "not validated" ? `<span class="badge no">Not validated</span>` : `<span class="badge ok">Validated</span><p class="valnote">Checked against: ${esc(s.validated)}</p>`;
      s.tables.forEach((t, ti) => { h += `<div class="tbl" data-t="${s.key}:${ti}"><div class="tw">${t.html}</div><div class="acts">${["Copy text:txt", "Copy for Word:html", "Copy LaTeX:tex", "Copy CSV:csv", "Download CSV:dl"].map(x => { const [l, k] = x.split(":"); return act(l, k); }).join("")}</div></div>`; });
      s.figures.forEach((f, fi) => { h += `<div class="fig" data-f="${s.key}:${fi}"><p class="cap"><b>Figure ${f.number}.</b> ${esc(f.title)}</p><div class="fw">${f.svg}</div><p class="cap" style="font-size:12px"><i>Note:</i> ${esc(f.note)} ${esc(f.source)}</p><div class="acts">${["PNG (300 dpi):png", "SVG:svg", "PDF:pdf", "Copy caption:cap"].map(x => { const [l, k] = x.split(":"); return act(l, k); }).join("")}</div></div>`; });
      s.paragraphs.forEach((p, pi) => { h += `<div class="para ${p.kind}" data-p="${s.key}:${pi}">${esc(p.text)}<div class="acts">${act("Copy text", "ptxt")}</div></div>`; });
      h += "</div></section>";
    }
    h += `<section class="rstep" id="oesr-text"><div class="rhead" data-toggle="text" role="button" tabindex="0" aria-expanded="true"><h3>Text for the paper</h3><small>methods, limitations, references ▾</small></div><div class="rbody">
      <h4>Methods paragraph</h4>${rep.methods.map((m, i) => `<div class="para" data-m="${i}">${esc(m)}<div class="acts">${act("Copy text", "mtxt")}</div></div>`).join("")}
      <h4>Notes on limits</h4>${rep.limitations.map((m, i) => `<div class="para" data-l="${i}">${esc(m)}<div class="acts">${act("Copy text", "ltxt")}</div></div>`).join("")}
      ${rep.references.length ? `<h4>References for this method</h4><div class="refs">${rep.references.map(r => `<p>${esc(r)}</p>`).join("")}</div><div class="acts">${act("Copy references", "refs")}</div>` : ""}
      <p class="valnote">${rep.unvalidated.length ? "Steps not validated against a reference: " + esc(rep.unvalidated.join("; ")) + "." : "Every step of this report is validated against a reference implementation or closed form in the project's automated tests."}</p></div></section></div>`;
    el.innerHTML = h;
    wire(el, rep);
  }

  function wire(el, rep) {
    const msg = t => { const m = el.querySelector("#oesr-msg"); if (m) m.textContent = t; };
    const flash = (b, ok) => { const o = b.textContent; b.classList.toggle("done", ok); b.textContent = ok ? "Copied ✓" : "Copy failed"; setTimeout(() => { b.textContent = o; b.classList.remove("done"); }, 1500); };
    const step = k => rep.steps.find(s => s.key === k);
    el.addEventListener("click", async ev => {
      const b = ev.target.closest("button"), tog = ev.target.closest("[data-toggle]"), toc = ev.target.closest("[data-toc]");
      if (toc && ev.target.closest("a")) {
        const sec = el.querySelector(`#oesr-s-${toc.dataset.toc}`), body = sec.querySelector(".rbody");
        if (ev.altKey) { ev.preventDefault(); body.hidden = !body.hidden; toc.classList.toggle("off", body.hidden); }
        return;
      }
      if (!b) { if (tog) { const body = tog.nextElementSibling; body.hidden = !body.hidden; tog.setAttribute("aria-expanded", String(!body.hidden)); const a = el.querySelector(`[data-toc="${tog.dataset.toggle}"]`); if (a) a.classList.toggle("off", body.hidden); } return; }
      const a = b.dataset.a, g = b.dataset.g;
      try {
        if (g) {
          if (g === "ms") return flash(b, await copyText(rep.manuscript));
          if (g === "tex") return save("report.tex", rep.exports.tex, "application/x-tex");
          if (g === "html") return save("report.html", inlineHtml(rep), "text/html");
          b.disabled = true;
          try {
            if (g === "zip") save(`oes_${rep.model_id}_report.zip`, await everything(rep, msg));
            if (g === "doc") { msg("Drawing the figures…"); save(`oes_${rep.model_id}_report.doc`, await wordDoc(rep, await allFigures(rep))); }
          } finally { b.disabled = false; msg(""); }
          return;
        }
        const host = b.closest("[data-t],[data-f],[data-p],[data-m],[data-l]");
        if (a === "refs") return flash(b, await copyText(rep.references.join("\n\n")));
        if (a === "mtxt") return flash(b, await copyText(rep.methods[+host.dataset.m]));
        if (a === "ltxt") return flash(b, await copyText(rep.limitations[+host.dataset.l]));
        if (a === "ptxt") { const [k, i] = host.dataset.p.split(":"); return flash(b, await copyText(step(k).paragraphs[+i].text)); }
        if (host && host.dataset.t) {
          const [k, i] = host.dataset.t.split(":"), t = step(k).tables[+i];
          if (a === "txt") return flash(b, await copyText(t.text));
          if (a === "html") return flash(b, await copyHtml(t.html, t.text));
          if (a === "tex") return flash(b, await copyText(t.latex));
          if (a === "csv") return flash(b, await copyText(t.csv));
          if (a === "dl") return save(`${t.name}.csv`, "﻿" + t.csv, "text/csv");
        }
        if (host && host.dataset.f) {
          const [k, i] = host.dataset.f.split(":"), f = step(k).figures[+i];
          if (a === "svg") return save(`${f.name}.svg`, f.svg, "image/svg+xml");
          if (a === "pdf") return save(`${f.name}.pdf`, new Blob([b64ToBytes(f.pdf_b64)], {type: "application/pdf"}));
          if (a === "png") return save(`${f.name}.png`, new Blob([await svgToPng(f)], {type: "image/png"}));
          if (a === "cap") return flash(b, await copyText(f.full_caption));
        }
      } catch (e) { msg(e.message || String(e)); }
    });
    el.addEventListener("keydown", ev => { const t = ev.target.closest && ev.target.closest("[data-toggle]"); if (t && (ev.key === "Enter" || ev.key === " ")) { ev.preventDefault(); t.click(); } });
  }

  window.OESReport = {render, makeZip, crc32: u => crcFinal(crc32(u)), pngWithDpi, svgToPng, everything, wordDoc, copyHtml, copyText, inlineHtml};
})();
