// Shared mobile navigation for all non-course pages.
(function () {
  var nav = document.getElementById('nav');
  var menu = document.getElementById('menu');
  if (!nav || !menu) return;

  function closeMenu() {
    nav.classList.remove('show');
    menu.setAttribute('aria-expanded', 'false');
  }

  menu.addEventListener('click', function () {
    var open = nav.classList.toggle('show');
    menu.setAttribute('aria-expanded', String(open));
  });

  nav.querySelectorAll('a').forEach(function (link) {
    link.addEventListener('click', closeMenu);
  });

  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') closeMenu();
  });

  window.addEventListener('resize', function () {
    if (window.innerWidth > 650) closeMenu();
  });
})();
