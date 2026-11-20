// "More Guides" dropdown in the top menu: hover or focus opens it on a wide screen, a tap toggles it everywhere.
(function () {
  var groups = document.querySelectorAll('header.site-header .dd');
  if (!groups.length) return;
  function closeAll(except) {
    groups.forEach(function (g) {
      if (g === except) return;
      g.classList.remove('open');
      var b = g.querySelector('.dd-btn');
      if (b) b.setAttribute('aria-expanded', 'false');
    });
  }
  groups.forEach(function (g) {
    var b = g.querySelector('.dd-btn');
    b.addEventListener('click', function (e) {
      e.stopPropagation();
      var open = g.classList.toggle('open');
      b.setAttribute('aria-expanded', String(open));
      closeAll(g);
    });
  });
  document.addEventListener('click', function () { closeAll(null); });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeAll(null); });
})();
