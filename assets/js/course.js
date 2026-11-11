// Course pages: mobile course menu and "On this page" table of contents.
(function () {
  var menu = document.getElementById('menu'), nav = document.getElementById('nav');
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
