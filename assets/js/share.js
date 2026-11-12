// Share bar at the end of every page: LinkedIn, Facebook, X, copy link (and the phone's own share sheet where available).
(function () {
  if (document.querySelector('.share-bar')) return;
  var canon = document.querySelector('link[rel="canonical"]');
  var url = canon ? canon.href : location.href.split('#')[0];
  var title = (document.querySelector('meta[property="og:title"]') || {}).content || document.title;
  var u = encodeURIComponent(url), t = encodeURIComponent(title);
  var I = {
    li: '<path d="M4.98 3.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5zM3 9h4v12H3zM9 9h3.8v1.7h.05c.53-1 1.83-2.05 3.77-2.05 4.03 0 4.78 2.65 4.78 6.1V21h-4v-5.5c0-1.3-.03-3-1.83-3-1.84 0-2.12 1.43-2.12 2.9V21H9z"/>',
    fb: '<path d="M13.5 21v-7.5h2.5l.4-3h-2.9V8.6c0-.87.25-1.46 1.5-1.46h1.55V4.47A20 20 0 0 0 14.3 4.3c-2.2 0-3.7 1.34-3.7 3.8v2.4H8.1v3h2.5V21z"/>',
    x: '<path d="M17.75 3h3.07l-6.7 7.66L22 21h-6.17l-4.83-6.32L5.47 21H2.4l7.17-8.2L2 3h6.33l4.37 5.77zm-1.08 16.2h1.7L7.4 4.73H5.58z"/>',
    ln: '<path d="M10.6 13.4a1 1 0 0 1 0-1.42l3.54-3.53a3 3 0 1 1 4.24 4.24l-1.77 1.77a1 1 0 1 1-1.41-1.41l1.77-1.77a1 1 0 0 0-1.42-1.42l-3.53 3.54a1 1 0 0 1-1.42 0zm2.8-2.8a1 1 0 0 1 0 1.42l-3.54 3.53a3 3 0 1 1-4.24-4.24l1.77-1.77a1 1 0 0 1 1.41 1.41l-1.77 1.77a1 1 0 0 0 1.42 1.42l3.53-3.54a1 1 0 0 1 1.42 0z"/>',
    sh: '<path d="M18 16a3 3 0 0 0-2.4 1.2l-6.7-3.4a3 3 0 0 0 0-1.6l6.7-3.4A3 3 0 1 0 15 7l-6.7 3.4a3 3 0 1 0 0 3.2L15 17a3 3 0 1 0 3-1z"/>'
  };
  var svg = function (k) { return '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">' + I[k] + '</svg>'; };
  var a = function (k, label, href) {
    return '<a class="share-btn share-' + k + '" href="' + href + '" target="_blank" rel="noopener noreferrer" aria-label="Share on ' + label + '">' + svg(k) + '<span>' + label + '</span></a>';
  };
  var bar = document.createElement('div');
  bar.className = 'share-bar';
  bar.setAttribute('role', 'group');
  bar.setAttribute('aria-label', 'Share this page');
  bar.innerHTML = '<span class="share-label">Share this page</span>' +
    a('li', 'LinkedIn', 'https://www.linkedin.com/sharing/share-offsite/?url=' + u) +
    a('fb', 'Facebook', 'https://www.facebook.com/sharer/sharer.php?u=' + u) +
    a('x', 'X', 'https://twitter.com/intent/tweet?url=' + u + '&text=' + t) +
    '<button type="button" class="share-btn share-ln" aria-label="Copy link to this page">' + svg('ln') + '<span>Copy link</span></button>' +
    (navigator.share ? '<button type="button" class="share-btn share-sh" aria-label="More sharing options">' + svg('sh') + '<span>More</span></button>' : '');
  var main = document.querySelector('main');
  var foot = document.querySelector('footer');
  if (main && main.clientWidth >= document.documentElement.clientWidth - 2 && !main.classList.contains('wrap')) {
    // full-width <main> (homepage): keep the bar inside the page's content width
    var box = document.createElement('div');
    box.className = document.querySelector('.wrap') ? 'wrap' : 'share-wrap';
    box.appendChild(bar);
    main.appendChild(box);
  } else if (main) main.appendChild(bar); else if (foot) foot.parentNode.insertBefore(bar, foot); else document.body.appendChild(bar);

  var copy = bar.querySelector('.share-ln');
  copy.addEventListener('click', function () {
    var done = function () {
      var s = copy.querySelector('span'); s.textContent = 'Link copied';
      copy.classList.add('ok'); setTimeout(function () { s.textContent = 'Copy link'; copy.classList.remove('ok'); }, 2000);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(url).then(done, function () { prompt('Copy this link:', url); });
    else prompt('Copy this link:', url);
  });
  var more = bar.querySelector('.share-sh');
  if (more) more.addEventListener('click', function () { navigator.share({ title: title, url: url }).catch(function () {}); });
})();
