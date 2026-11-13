/* "Ask this site": answers questions by quoting the best passages of the site, in the visitor's browser.
   No server, no tracking, no external AI. The index (index.json) is built by tools/site_ai/build_index.py. */
(function () {
  var script = document.currentScript;
  if (!script || window.__siteAskLoaded) return;
  window.__siteAskLoaded = true;
  var BASE = new URL(".", script.src).href;                    // .../assets/ai/
  var SITE = new URL("../../", BASE).href;                      // site root
  var ISSUES = "https://github.com/tuhingm/almamun/issues/new";
  var EXAMPLES = ["Which model do I use for non-stationary series?", "How do I run a threshold regression?", "What is Open Econometric Systems?"];
  var idx = null, loading = null, S = null;

  var css = document.createElement("style");
  css.textContent =
    ".ask-btn{position:fixed;right:16px;bottom:16px;z-index:9998;background:#0b2a6f;color:#fff;border:0;border-radius:999px;padding:11px 16px;font:700 13px system-ui,sans-serif;box-shadow:0 4px 16px rgba(0,0,0,.25);cursor:pointer}" +
    ".ask-btn:focus-visible,.ask-box button:focus-visible,.ask-box input:focus-visible{outline:3px solid #f2a900;outline-offset:2px}" +
    ".ask-box{position:fixed;right:16px;bottom:64px;z-index:9999;width:min(420px,calc(100vw - 32px));max-height:min(70vh,560px);display:none;flex-direction:column;background:#fff;color:#14213d;border:1px solid #cfd6e4;border-radius:14px;box-shadow:0 10px 40px rgba(0,0,0,.28);font:14px/1.45 system-ui,sans-serif}" +
    ".ask-box.open{display:flex}.ask-head{padding:12px 14px;border-bottom:1px solid #e3e8f2;display:flex;justify-content:space-between;align-items:center;font-weight:800}" +
    ".ask-head button{background:none;border:0;font-size:20px;line-height:1;cursor:pointer;color:#14213d}" +
    ".ask-form{display:flex;gap:8px;padding:10px 14px}.ask-form input{flex:1;min-width:0;border:1px solid #b9c3d8;border-radius:8px;padding:9px 10px;font:inherit}" +
    ".ask-form button{background:#0b2a6f;color:#fff;border:0;border-radius:8px;padding:9px 14px;font:700 13px system-ui;cursor:pointer}" +
    ".ask-out{overflow:auto;padding:0 14px 12px}.ask-note{font-size:12px;color:#55607a;padding:0 14px 10px}" +
    ".ask-chip{display:block;width:100%;text-align:left;background:#eef2fb;border:0;border-radius:8px;padding:8px 10px;margin:0 0 6px;font:inherit;cursor:pointer;color:#0b2a6f}" +
    ".ask-card{border:1px solid #dfe5f1;border-radius:10px;padding:10px 12px;margin:0 0 8px}.ask-card.answer{border-color:#0b2a6f;background:#f4f7ff}" +
    ".ask-card a.t{font-weight:800;color:#0b2a6f;text-decoration:none}.ask-card a.t:hover{text-decoration:underline}" +
    ".ask-card .h{font-size:12px;color:#55607a;margin:0 0 4px}.ask-card p{margin:4px 0 0}.ask-card mark{background:#ffe9a8;color:inherit;border-radius:3px;padding:0 1px}" +
    ".ask-more{font-size:12px}.ask-more a{color:#0b2a6f}@media print{.ask-btn,.ask-box{display:none!important}}" +
    "@media (prefers-reduced-motion:no-preference){.ask-box.open{animation:askin .15s ease-out}@keyframes askin{from{opacity:0;transform:translateY(6px)}}}";
  document.head.appendChild(css);

  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

  var btn = el("button", "ask-btn", "Ask this site");
  btn.type = "button"; btn.setAttribute("aria-expanded", "false"); btn.setAttribute("aria-controls", "ask-box");
  var box = el("div", "ask-box"); box.id = "ask-box"; box.setAttribute("role", "dialog"); box.setAttribute("aria-label", "Ask this site");
  var head = el("div", "ask-head"); head.appendChild(el("span", "", "Ask this site"));
  var close = el("button", "", "×"); close.type = "button"; close.setAttribute("aria-label", "Close"); head.appendChild(close);
  var form = el("form", "ask-form"); var input = el("input"); input.type = "search"; input.placeholder = "Ask about a model, guide or course"; input.setAttribute("aria-label", "Your question"); input.autocomplete = "off";
  var go = el("button", "", "Ask"); go.type = "submit"; form.appendChild(input); form.appendChild(go);
  var out = el("div", "ask-out"); out.setAttribute("aria-live", "polite");
  var note = el("div", "ask-note", "Answers are quotes from this site, found in your browser. Nothing you type is sent anywhere.");
  box.appendChild(head); box.appendChild(form); box.appendChild(out); box.appendChild(note);
  document.body.appendChild(btn); document.body.appendChild(box);

  function load() {
    if (loading) return loading;
    out.textContent = "Loading the site index…";
    loading = Promise.all([
      fetch(BASE + "index.json").then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); }),
      new Promise(function (ok, bad) { if (window.SiteSearch) return ok(); var s = document.createElement("script"); s.src = BASE + "search.js"; s.onload = ok; s.onerror = bad; document.head.appendChild(s); })
    ]).then(function (r) { idx = r[0]; S = window.SiteSearch; showExamples(); })
      .catch(function () { loading = null; out.textContent = "The search index could not be loaded. Try again later."; });
    return loading;
  }

  function showExamples() {
    out.textContent = "";
    out.appendChild(el("p", "ask-more", "Try one of these:"));
    EXAMPLES.forEach(function (q) {
      var c = el("button", "ask-chip", q); c.type = "button";
      c.addEventListener("click", function () { input.value = q; run(q); });
      out.appendChild(c);
    });
  }

  function highlight(node, text, qtok) {
    var q = new Set(qtok);
    text.split(/(\s+)/).forEach(function (w) {
      if (!w.trim()) { node.appendChild(document.createTextNode(w)); return; }
      var t = S.tokens(w.replace(/[^\w.\-]/g, ""));
      if (t.length && q.has(t[0])) node.appendChild(el("mark", "", w)); else node.appendChild(document.createTextNode(w));
    });
  }

  function card(h, query, first) {
    var c = el("div", "ask-card" + (h.faq ? " answer" : ""));
    var a = el("a", "t", h.faq ? "Answer" : h.title); a.href = new URL(h.url || "", SITE).href;
    c.appendChild(a);
    if (!h.faq && h.heading) c.appendChild(el("div", "h", h.heading));
    if (h.faq) c.appendChild(el("div", "h", h.heading));
    var p = el("p"); var qt = S.tokens(query);
    if (h.faq) highlight(p, h.text, qt); else highlight(p, S.bestSentences(h.text, query, first ? 3 : 2).join(" "), qt);
    c.appendChild(p);
    return c;
  }

  function run(q) {
    q = q.trim();
    if (!q) return;
    load().then(function () {
      if (!idx) return;
      var hits = S.search(idx, q, 4);
      out.textContent = "";
      if (!hits.length) out.appendChild(el("p", "", "I could not find that on this site. Try fewer or different words."));
      hits.forEach(function (h, i) { out.appendChild(card(h, q, i === 0)); });
      var more = el("p", "ask-more");
      var a = el("a", "", "Not what you wanted? Send this question to the author.");
      a.href = ISSUES + "?title=" + encodeURIComponent("Question: " + q.slice(0, 80)) + "&body=" + encodeURIComponent("Question from the site search box:\n\n" + q + "\n\nPage: " + location.href);
      a.target = "_blank"; a.rel = "noopener";
      more.appendChild(a); out.appendChild(more);
      out.scrollTop = 0;
    });
  }

  function open(on) {
    box.classList.toggle("open", on); btn.setAttribute("aria-expanded", on ? "true" : "false");
    if (on) { load(); setTimeout(function () { input.focus(); }, 0); } else btn.focus();
  }
  btn.addEventListener("click", function () { open(!box.classList.contains("open")); });
  close.addEventListener("click", function () { open(false); });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape" && box.classList.contains("open")) open(false); });
  form.addEventListener("submit", function (e) { e.preventDefault(); run(input.value); });
})();
