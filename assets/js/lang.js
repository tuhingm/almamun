// Optional language switch (English / 한국어 / বাংলা). English is the default; other languages use Google Translate on demand.
(function () {
  var LANGS = [['en', 'English'], ['ko', '한국어'], ['bn', 'বাংলা']];
  function current() {
    var m = document.cookie.match(/(?:^|; )googtrans=\/en\/([a-z]+)/);
    return m ? m[1] : 'en';
  }
  function set(l) {
    var host = location.hostname, exp = 'Thu, 01 Jan 1970 00:00:00 GMT';
    ['', '; domain=' + host, '; domain=.' + host].forEach(function (d) {
      document.cookie = 'googtrans=; expires=' + exp + '; path=/' + d;
    });
    if (l !== 'en') document.cookie = 'googtrans=/en/' + l + '; path=/';
    try { localStorage.setItem('lang', l); } catch (e) {}
    location.reload();
  }
  var cur = current();
  var css = document.createElement('style');
  css.textContent =
    '#lang-switch{display:flex;justify-content:flex-end;align-items:center;gap:4px;background:#002a85;padding:4px 14px;font:12px/1 system-ui,sans-serif}' +
    '#lang-switch button{border:0;background:transparent;padding:5px 10px;border-radius:999px;cursor:pointer;color:#cfe0ff;font:inherit}' +
    '#lang-switch button[aria-pressed=true]{background:#6EC1E4;color:#002a85;font-weight:700}' +
    '.goog-te-banner-frame,#goog-gt-tt,.goog-te-balloon-frame,.skiptranslate iframe{display:none!important}body{top:0!important}' +
    '#google_translate_element{display:none}';
  document.head.appendChild(css);
  var box = document.createElement('div');
  box.id = 'lang-switch'; box.className = 'notranslate'; box.setAttribute('translate', 'no');
  box.setAttribute('role', 'group'); box.setAttribute('aria-label', 'Language');
  LANGS.forEach(function (x) {
    var b = document.createElement('button');
    b.type = 'button'; b.textContent = x[1]; b.lang = x[0];
    b.setAttribute('aria-pressed', String(x[0] === cur));
    b.addEventListener('click', function () { if (x[0] !== cur) set(x[0]); });
    box.appendChild(b);
  });
  document.body.insertBefore(box, document.body.firstChild);
  if (cur !== 'en') {
    var d = document.createElement('div'); d.id = 'google_translate_element'; document.body.appendChild(d);
    window.__gtInit = function () {
      new google.translate.TranslateElement({ pageLanguage: 'en', includedLanguages: 'ko,bn', autoDisplay: false }, 'google_translate_element');
    };
    var s = document.createElement('script');
    s.src = 'https://translate.google.com/translate_a/element.js?cb=__gtInit';
    document.body.appendChild(s);
  }
})();
