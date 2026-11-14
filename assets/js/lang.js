// Language switch: English (default) / 한국어 / বাংলা.
// Pages with a hand-written translation file in assets/i18n/<lang>/ use it; other pages fall back to Google Translate.
(function () {
  var LANGS = [['en', 'English'], ['ko', '한국어'], ['bn', 'বাংলা']];
  var me = document.currentScript || document.querySelector('script[src*="lang.js"]');
  var base = new URL('../../', me.src);              // site root
  var rel = location.pathname.slice(base.pathname.length).replace(/index\.html$/, '').replace(/\.html$/, '').replace(/\/$/, '');
  var slug = rel ? rel.replace(/\//g, '__') : 'home';
  function getLang() { try { return localStorage.getItem('lang') || 'en'; } catch (e) { return 'en'; } }
  function clearG() {
    var host = location.hostname, exp = 'Thu, 01 Jan 1970 00:00:00 GMT';
    ['', '; domain=' + host, '; domain=.' + host].forEach(function (d) { document.cookie = 'googtrans=; expires=' + exp + '; path=/' + d; });
  }
  function setLang(l) { try { localStorage.setItem('lang', l); } catch (e) {} clearG(); location.reload(); }
  var cur = getLang(); if (cur !== 'ko' && cur !== 'bn') cur = 'en';

  var css = document.createElement('style');
  css.textContent =
    '#lang-switch{display:flex;justify-content:flex-end;align-items:center;gap:4px;background:#002a85;padding:4px 14px;font:12px/1 system-ui,sans-serif}' +
    '#lang-switch button{border:0;background:transparent;padding:5px 10px;border-radius:999px;cursor:pointer;color:#cfe0ff;font:inherit}' +
    '#lang-switch button[aria-pressed=true]{background:#6EC1E4;color:#002a85;font-weight:700}' +
    '.goog-te-banner-frame,#goog-gt-tt,.goog-te-balloon-frame,.skiptranslate iframe{display:none!important}body{top:0!important}#google_translate_element{display:none}';
  document.head.appendChild(css);
  var box = document.createElement('div');
  box.id = 'lang-switch'; box.className = 'notranslate'; box.setAttribute('translate', 'no');
  box.setAttribute('role', 'group'); box.setAttribute('aria-label', 'Language');
  LANGS.forEach(function (x) {
    var b = document.createElement('button');
    b.type = 'button'; b.textContent = x[1]; b.lang = x[0];
    b.setAttribute('aria-pressed', String(x[0] === cur));
    b.addEventListener('click', function () { if (x[0] !== cur) setLang(x[0]); });
    box.appendChild(b);
  });
  document.body.insertBefore(box, document.body.firstChild);
  if (cur === 'en') { clearG(); return; }
  document.documentElement.lang = cur;

  function google() {
    document.cookie = 'googtrans=/en/' + cur + '; path=/';
    var d = document.createElement('div'); d.id = 'google_translate_element'; document.body.appendChild(d);
    window.__gtInit = function () { new google.translate.TranslateElement({ pageLanguage: 'en', includedLanguages: 'ko,bn', autoDisplay: false }, 'google_translate_element'); };
    var s = document.createElement('script'); s.src = 'https://translate.google.com/translate_a/element.js?cb=__gtInit'; document.body.appendChild(s);
  }
  function get(u) { return fetch(u).then(function (r) { if (!r.ok) throw 0; return r.json(); }); }
  var dir = new URL('assets/i18n/' + cur + '/', base).href;
  Promise.all([get(dir + slug + '.json'), get(dir + 'common.json').catch(function () { return {}; })]).then(function (r) {
    run(Object.assign({}, r[1], r[0]));
  }, google);

  function norm(s) { return s.replace(/\s+/g, ' ').trim(); }
  function run(D) {
    var ATTR = ['alt', 'aria-label', 'title', 'placeholder'];
    var busy = false;
    function skip(el) { return !el || /^(SCRIPT|STYLE|NOSCRIPT|CODE|PRE|TEXTAREA|SVG)$/i.test(el.tagName) || el.closest('.notranslate,#lang-switch,[translate=no]'); }
    function walk(el) {
      if (skip(el)) return;
      if (el.children.length && D['html:' + norm(el.textContent)] !== undefined) { el.innerHTML = D['html:' + norm(el.textContent)]; return; }
      ATTR.forEach(function (a) { var v = el.getAttribute && el.getAttribute(a); if (v && D[norm(v)] !== undefined) el.setAttribute(a, D[norm(v)]); });
      for (var n = el.firstChild; n; n = n.nextSibling) {
        if (n.nodeType === 3) {
          var k = norm(n.nodeValue);
          if (k && D[k] !== undefined) n.nodeValue = (/^\s/.test(n.nodeValue) ? ' ' : '') + D[k] + (/\s$/.test(n.nodeValue) ? ' ' : '');
        } else if (n.nodeType === 1) walk(n);
      }
    }
    function all() {
      busy = true; walk(document.body);
      var t = norm(document.title); if (D[t] !== undefined) document.title = D[t];
      busy = false;
    }
    all();
    new MutationObserver(function () { if (!busy) all(); }).observe(document.body, { childList: true, subtree: true, characterData: true });
  }
})();
