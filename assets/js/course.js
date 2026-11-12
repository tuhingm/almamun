// Course pages: consistent site-wide navigation, mobile course menu and page utilities.
(function () {
  var menu = document.getElementById('menu'), nav = document.getElementById('nav');

  // Keep the navigation on every course page consistent with the main website.
  // Derive the project base path so this also works if the site moves to a custom domain.
  if (nav) {
    var marker = '/teaching/';
    var pathname = window.location.pathname;
    var markerIndex = pathname.indexOf(marker);
    var base = markerIndex >= 0 ? pathname.slice(0, markerIndex + 1) : '/';
    var items = [
      ['Home', 'index.html'],
      ['Teaching Courses', 'teaching/index.html', true],
      ['Working Papers', 'working-papers/index.html'],
      ['Time-Series Guide', 'research-guide/index.html'],
      ['Panel-Data Guide', 'panel-data-guide/index.html'],
      ['Causal Inference', 'causal-inference-guide/index.html'],
      ['DSGE Guide', 'dsge-guide/index.html'],
      ['CV', 'Tuhin%20Al%20Mamun_CV.pdf']
    ];

    nav.setAttribute('aria-label', 'Main navigation');
    nav.replaceChildren();
    items.forEach(function (item) {
      var link = document.createElement('a');
      link.href = base + item[1];
      link.textContent = item[0];
      if (item[2]) {
        link.className = 'current';
        link.setAttribute('aria-current', 'page');
      }
      nav.appendChild(link);
    });
  }

  if (menu && nav) {
    menu.addEventListener('click', function () {
      var open = nav.classList.toggle('show');
      menu.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }

  var side = document.querySelector('.course-side');
  var btn = document.querySelector('.side-toggle');
  if (side && btn) {
    btn.addEventListener('click', function () {
      var open = side.classList.toggle('open');
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }

  var toc = document.getElementById('toc');
  if (!toc) return;
  var heads = document.querySelectorAll('.notes h2[id]');
  heads.forEach(function (h) {
    var li = document.createElement('li');
    var a = document.createElement('a');
    a.href = '#' + h.id;
    a.textContent = h.textContent.replace(/^\d+\.\s*/, '');
    li.appendChild(a);
    toc.appendChild(li);
  });
  if (!heads.length) { toc.parentNode.style.display = 'none'; return; }
  if ('IntersectionObserver' in window) {
    var links = toc.querySelectorAll('a');
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          links.forEach(function (l) { l.classList.toggle('on', l.getAttribute('href') === '#' + e.target.id); });
        }
      });
    }, { rootMargin: '0px 0px -75% 0px' });
    heads.forEach(function (h) { obs.observe(h); });
  }
})();

// Copy buttons on code blocks
document.querySelectorAll('.notes pre').forEach(function (pre) {
  var b = document.createElement('button');
  b.className = 'copy-btn'; b.type = 'button'; b.textContent = 'Copy';
  b.addEventListener('click', function () {
    var code = pre.querySelector('code') || pre;
    navigator.clipboard.writeText(code.innerText).then(function () {
      b.textContent = 'Copied'; setTimeout(function () { b.textContent = 'Copy'; }, 1500);
    });
  });
  pre.appendChild(b);
});
