/* App chrome for pages using assets/app.css.
   <body data-app="hub|jumpshot|lineup|predictor" data-title="Screen title" data-depth="../">
   Renders: translucent top bar (large-title collapse on scroll), bottom tab bar on mobile,
   inline nav on desktop. Optional: <template id="top-actions"> content is placed in the top bar.
   Also registers the service worker and exposes window.App.toast(msg). */
(function () {
  var b = document.body;
  var app = b.getAttribute('data-app') || 'hub';
  var depth = b.getAttribute('data-depth');
  if (depth == null) depth = app === 'hub' ? '' : '../';
  var title = b.getAttribute('data-title') || document.title;

  var I = {
    hub: '<path d="M3.5 10.5 12 4l8.5 6.5"/><path d="M5.5 9.5V20h13V9.5"/>',
    jumpshot: '<circle cx="12" cy="12" r="8.5"/><path d="M3.5 12h17"/><path d="M12 3.5c-2.6 2.3-3.9 5.1-3.9 8.5s1.3 6.2 3.9 8.5"/><path d="M12 3.5c2.6 2.3 3.9 5.1 3.9 8.5s-1.3 6.2-3.9 8.5"/>',
    lineup: '<circle cx="8" cy="8" r="2.6"/><circle cx="16" cy="8" r="2.6"/><path d="M3.5 19c.5-3 2.3-4.6 4.5-4.6s4 1.6 4.5 4.6"/><path d="M11.5 19c.5-3 2.3-4.6 4.5-4.6s4 1.6 4.5 4.6"/>',
    predictor: '<path d="M4 19V5"/><path d="M4 19h16"/><path d="M7.5 15l3.5-4 3 2.5 5-6"/>'
  };
  var tabs = [
    { id: 'hub', label: 'Home', href: depth + 'index.html' },
    { id: 'jumpshot', label: 'Jumpshot', href: depth + 'jumpshot/index.html' },
    { id: 'lineup', label: 'Lineups', href: depth + 'lineup/index.html' },
    { id: 'predictor', label: 'Games', href: depth + 'predictor/index.html' }
  ];
  function svg(id) { return '<svg viewBox="0 0 24 24" aria-hidden="true">' + I[id] + '</svg>'; }

  var top = document.createElement('header');
  top.className = 'ap-top';
  top.innerHTML =
    '<a class="ap-home" href="' + depth + 'index.html" aria-label="NBA Projects home">' +
      '<svg viewBox="0 0 26 26" aria-hidden="true"><rect width="26" height="26" rx="7" fill="var(--brand)"/>' +
      '<g fill="none" stroke="var(--on-brand)" stroke-width="1.7" stroke-linecap="round"><circle cx="13" cy="13" r="7"/>' +
      '<path d="M6 13h14M13 6c-2 1.9-3 4.2-3 7s1 5.1 3 7M13 6c2 1.9 3 4.2 3 7s-1 5.1-3 7"/></g></svg>' +
      '<span class="ap-brandname">NBA Projects</span></a>' +
    '<nav class="ap-deskNav" aria-label="Apps">' + tabs.map(function (t) {
      return '<a href="' + t.href + '"' + (t.id === app ? ' class="on" aria-current="page"' : '') + '>' + t.label + '</a>';
    }).join('') + '</nav>' +
    '<span class="ap-title">' + title + '</span>' +
    '<div class="ap-actions"></div>';
  b.insertBefore(top, b.firstChild);
  var tpl = document.getElementById('top-actions');
  if (tpl && tpl.content) top.querySelector('.ap-actions').appendChild(tpl.content.cloneNode(true));

  var nav = document.createElement('nav');
  nav.className = 'ap-tabs';
  nav.setAttribute('aria-label', 'Apps');
  nav.innerHTML = tabs.map(function (t) {
    return '<a href="' + t.href + '"' + (t.id === app ? ' class="on" aria-current="page"' : '') + '>' + svg(t.id) + '<span>' + t.label + '</span></a>';
  }).join('');
  b.appendChild(nav);


  var ticking = false;
  function onScroll() {
    if (ticking) return; ticking = true;
    requestAnimationFrame(function () { top.classList.toggle('scrolled', scrollY > 44); ticking = false; });
  }
  addEventListener('scroll', onScroll, { passive: true }); onScroll();

  var toastEl;
  window.App = window.App || {};
  window.App.toast = function (msg) {
    if (!toastEl) { toastEl = document.createElement('div'); toastEl.className = 'toast'; toastEl.setAttribute('role', 'status'); b.appendChild(toastEl); }
    toastEl.textContent = msg; toastEl.classList.add('show');
    clearTimeout(toastEl._t); toastEl._t = setTimeout(function () { toastEl.classList.remove('show'); }, 2200);
  };

  if ('serviceWorker' in navigator && location.protocol !== 'file:') {
    try { navigator.serviceWorker.register(depth + 'sw.js').catch(function () {}); } catch (e) {}
  }
})();
