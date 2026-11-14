// Deterministic site-wide language switcher.
// All translations come from version-controlled JSON dictionaries.
// No machine-translation service is loaded at runtime.
(function () {
  'use strict';

  var LANGS = [['en', 'English'], ['ko', '한국어'], ['bn', 'বাংলা']];
  var script = document.currentScript || document.querySelector('script[src*="lang.js"]');
  var base = new URL('../../', script.src);
  var relativePath = location.pathname.slice(base.pathname.length)
    .replace(/index\.html$/, '')
    .replace(/\.html$/, '')
    .replace(/\/$/, '');
  var slug = relativePath ? relativePath.replace(/\//g, '__') : 'home';
  var STORAGE_KEY = 'lang';

  function readLanguage() {
    try {
      var value = localStorage.getItem(STORAGE_KEY);
      return LANGS.some(function (item) { return item[0] === value; }) ? value : 'en';
    } catch (error) {
      return 'en';
    }
  }

  function clearGoogleTranslateCookie() {
    var host = location.hostname;
    var expired = 'Thu, 01 Jan 1970 00:00:00 GMT';
    ['', '; domain=' + host, '; domain=.' + host].forEach(function (domain) {
      document.cookie = 'googtrans=; expires=' + expired + '; path=/' + domain;
    });
  }

  function changeLanguage(language) {
    try { localStorage.setItem(STORAGE_KEY, language); } catch (error) {}
    clearGoogleTranslateCookie();
    location.reload();
  }

  var currentLanguage = readLanguage();
  clearGoogleTranslateCookie();
  document.documentElement.lang = currentLanguage;

  var style = document.createElement('style');
  style.textContent =
    '#lang-switch{display:flex;justify-content:flex-end;align-items:center;gap:4px;background:#002a85;padding:4px 14px;font:12px/1 system-ui,sans-serif}' +
    '#lang-switch button{border:0;background:transparent;padding:5px 10px;border-radius:999px;cursor:pointer;color:#cfe0ff;font:inherit}' +
    '#lang-switch button[aria-pressed=true]{background:#6EC1E4;color:#002a85;font-weight:700}' +
    '#lang-switch button:focus-visible{outline:2px solid #fff;outline-offset:2px}' +
    '.goog-te-banner-frame,#goog-gt-tt,.goog-te-balloon-frame,.skiptranslate iframe{display:none!important}body{top:0!important}';
  document.head.appendChild(style);

  var switcher = document.createElement('div');
  switcher.id = 'lang-switch';
  switcher.className = 'notranslate';
  switcher.setAttribute('translate', 'no');
  switcher.setAttribute('role', 'group');
  switcher.setAttribute('aria-label', 'Language');
  LANGS.forEach(function (item) {
    var button = document.createElement('button');
    button.type = 'button';
    button.textContent = item[1];
    button.lang = item[0];
    button.setAttribute('aria-pressed', String(item[0] === currentLanguage));
    button.addEventListener('click', function () {
      if (item[0] !== currentLanguage) changeLanguage(item[0]);
    });
    switcher.appendChild(button);
  });
  document.body.insertBefore(switcher, document.body.firstChild);

  if (currentLanguage === 'en') return;

  var translationRoot = new URL('assets/i18n/' + currentLanguage + '/', base);
  var pageUrl = new URL(slug + '.json', translationRoot);
  var commonUrl = new URL('common.json', translationRoot);

  function fetchDictionary(url, optional) {
    return fetch(url.href, { credentials: 'same-origin' }).then(function (response) {
      if (!response.ok) {
        if (optional && response.status === 404) return {};
        throw new Error('Translation dictionary unavailable: ' + url.pathname);
      }
      return response.json();
    });
  }

  Promise.all([
    fetchDictionary(pageUrl, true),
    fetchDictionary(commonUrl, true)
  ]).then(function (dictionaries) {
    var dictionary = Object.assign({}, dictionaries[1], dictionaries[0]);
    applyTranslations(dictionary);
    document.documentElement.dataset.translationStatus =
      Object.keys(dictionaries[0]).length ? 'page-dictionary-loaded' : 'common-only';
    if (!Object.keys(dictionaries[0]).length) {
      console.warn('[i18n] Missing page dictionary:', pageUrl.pathname);
    }
  }).catch(function (error) {
    document.documentElement.dataset.translationStatus = 'dictionary-error';
    console.error('[i18n] Could not load local translations:', error);
  });

  function normalize(value) {
    return value.replace(/\s+/g, ' ').trim();
  }

  function applyTranslations(dictionary) {
    var attributes = ['alt', 'aria-label', 'title', 'placeholder'];
    var applying = false;

    function shouldSkip(element) {
      return !element ||
        /^(SCRIPT|STYLE|NOSCRIPT|CODE|PRE|TEXTAREA|SVG)$/i.test(element.tagName) ||
        element.closest('.notranslate, #lang-switch, [translate="no"]');
    }

    function translateElement(element) {
      if (shouldSkip(element)) return;

      var fullText = normalize(element.textContent || '');
      var htmlKey = 'html:' + fullText;
      if (element.children.length && Object.prototype.hasOwnProperty.call(dictionary, htmlKey)) {
        element.innerHTML = dictionary[htmlKey];
        return;
      }

      attributes.forEach(function (attribute) {
        var original = element.getAttribute(attribute);
        var key = original ? normalize(original) : '';
        if (key && Object.prototype.hasOwnProperty.call(dictionary, key)) {
          element.setAttribute(attribute, dictionary[key]);
        }
      });

      Array.prototype.forEach.call(element.childNodes, function (node) {
        if (node.nodeType === Node.TEXT_NODE) {
          var key = normalize(node.nodeValue);
          if (key && Object.prototype.hasOwnProperty.call(dictionary, key)) {
            node.nodeValue =
              (/^\s/.test(node.nodeValue) ? ' ' : '') +
              dictionary[key] +
              (/\s$/.test(node.nodeValue) ? ' ' : '');
          }
        } else if (node.nodeType === Node.ELEMENT_NODE) {
          translateElement(node);
        }
      });
    }

    function translatePage() {
      if (applying) return;
      applying = true;
      translateElement(document.body);
      var titleKey = normalize(document.title);
      if (Object.prototype.hasOwnProperty.call(dictionary, titleKey)) {
        document.title = dictionary[titleKey];
      }
      applying = false;
    }

    translatePage();

    // Translate content inserted later by interactive components, without
    // repeatedly reprocessing the same mutation batch.
    var observer = new MutationObserver(function (records) {
      if (applying || !records.length) return;
      applying = true;
      records.forEach(function (record) {
        if (record.type === 'characterData') {
          var key = normalize(record.target.nodeValue);
          if (key && Object.prototype.hasOwnProperty.call(dictionary, key)) {
            record.target.nodeValue = dictionary[key];
          }
        } else {
          Array.prototype.forEach.call(record.addedNodes, function (node) {
            if (node.nodeType === Node.ELEMENT_NODE) translateElement(node);
            else if (node.nodeType === Node.TEXT_NODE) {
              var key = normalize(node.nodeValue);
              if (key && Object.prototype.hasOwnProperty.call(dictionary, key)) {
                node.nodeValue = dictionary[key];
              }
            }
          });
        }
      });
      applying = false;
    });
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      characterData: true
    });
  }
})();
