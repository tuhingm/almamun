// Shared navigation dropdowns. Tools is injected into every page using the shared menu.
(function () {
  var nav = document.querySelector('header.site-header nav#nav');
  if (!nav) return;

  var groups = nav.querySelectorAll('.dd');
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
    if (!b || b.dataset.dropdownBound === 'true') return;
    b.dataset.dropdownBound = 'true';
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
