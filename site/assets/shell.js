/* Shared shell: renders the top bar, registers the service worker. Pass data-app on <body>. */
(function () {
  var app = document.body.getAttribute('data-app') || 'hub';
  var depth = document.body.getAttribute('data-depth') || (app === 'hub' ? '' : '../');
  var apps = [
    { id: 'hub', href: depth + 'index.html', label: 'Home',
      svg: '<path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 10v9h13v-9"/>' },
    { id: 'jumpshot', href: depth + 'jumpshot/index.html', label: 'Jumpshot',
      svg: '<circle cx="12" cy="5" r="2"/><path d="M12 7v5l-3 6"/><path d="M12 12l4 2 2 4"/><path d="M8 11l4-2 4 1"/>' },
    { id: 'lineup', href: depth + 'lineup/index.html', label: 'Lineup',
      svg: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 12h18M12 4v16"/><circle cx="12" cy="12" r="2.5"/>' },
    { id: 'predictor', href: depth + 'predictor/index.html', label: 'Predictor',
      svg: '<path d="M4 18 10 11l4 4 6-8"/><path d="M15 7h5v5"/>' }
  ];
  var title = document.body.getAttribute('data-title') || 'NBA Projects';
  var mark = document.body.getAttribute('data-mark') || 'NP';
  var bar = document.createElement('header'); bar.className = 'topbar';
  bar.innerHTML =
    '<a class="brand" href="' + depth + 'index.html"><span class="mark">' + mark + '</span><span>' + title + '</span></a>' +
    '<span class="chip">Sample data</span>' +
    '<nav>' + apps.map(function (a) {
      return '<a href="' + a.href + '" class="' + (a.id === app ? 'on' : '') + '" aria-label="' + a.label + '" title="' + a.label + '">' +
        '<svg viewBox="0 0 24 24">' + a.svg + '</svg></a>';
    }).join('') + '</nav>';
  document.body.insertBefore(bar, document.body.firstChild);

  if ('serviceWorker' in navigator && location.protocol !== 'file:') {
    try { navigator.serviceWorker.register(depth + 'sw.js').catch(function () {}); } catch (e) {}
  }
})();
