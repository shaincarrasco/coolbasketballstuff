/* NBA Projects service worker.
 *  - Pages + JSON data: network-first (daily data refresh shows up), cache fallback offline.
 *  - CSS/JS/icons: stale-while-revalidate.
 *  - vendor/ (MediaPipe wasm + pose models, ~30 MB, immutable): cache-first in a separate
 *    long-lived cache that survives app version bumps, filled on first use.
 *  - Range requests (video seeking) bypass the cache entirely.
 */
var VERSION = 'np-v3';
var CACHE = VERSION;
var VENDOR = 'np-vendor-mp-1.2.2';
var PRECACHE = [
  './', './index.html', './404.html', './manifest.webmanifest',
  './assets/shell.css', './assets/shell.js',
  './icons/icon-192.png', './icons/icon-512.png',
  './icons/icon-maskable-192.png', './icons/icon-maskable-512.png', './icons/apple-touch-icon.png',
  './jumpshot/', './jumpshot/index.html', './jumpshot/sample/', './jumpshot/sample/index.html', './lineup/', './lineup/index.html', './predictor/', './predictor/index.html'
];

self.addEventListener('install', function (e) {
  e.waitUntil(
    caches.open(CACHE).then(function (c) {
      return Promise.all(PRECACHE.map(function (u) {
        return fetch(u, { cache: 'no-cache' }).then(function (r) {
          if (r && r.ok && r.type === 'basic') return c.put(u, r);
        }).catch(function () {});
      }));
    }).then(function () { return self.skipWaiting(); })
  );
});

self.addEventListener('activate', function (e) {
  e.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.filter(function (k) { return k !== CACHE && k !== VENDOR; })
        .map(function (k) { return caches.delete(k); }));
    }).then(function () { return self.clients.claim(); })
  );
});

function okToCache(res) { return res && res.status === 200 && res.type === 'basic'; }

function put(cacheName, req, res) {
  var copy = res.clone();
  caches.open(cacheName).then(function (c) { return c.put(req, copy); }).catch(function () {});
}

function networkFirst(req) {
  return fetch(req).then(function (res) {
    if (okToCache(res)) put(CACHE, req, res);
    return res;
  }).catch(function () {
    return caches.match(req).then(function (hit) {
      if (hit) return hit;
      if (req.mode === 'navigate') return caches.match('./index.html');
      return Response.error();
    });
  });
}

function cacheFirst(req, cacheName) {
  return caches.open(cacheName).then(function (c) {
    return c.match(req).then(function (hit) {
      if (hit) return hit;
      return fetch(req).then(function (res) {
        if (okToCache(res)) c.put(req, res.clone()).catch(function () {});
        return res;
      });
    });
  });
}

function staleWhileRevalidate(req) {
  return caches.match(req).then(function (hit) {
    var net = fetch(req).then(function (res) {
      if (okToCache(res)) put(CACHE, req, res);
      return res;
    }).catch(function () { return hit || Response.error(); });
    return hit || net;
  });
}

self.addEventListener('fetch', function (e) {
  var req = e.request;
  if (req.method !== 'GET') return;
  var url = new URL(req.url);
  if (url.origin !== self.location.origin) return;     // never touch cross-origin (e.g. live ESPN scores)
  if (req.headers.has('range')) return;                 // let the browser stream/seek video natively

  var p = url.pathname;
  if (p.indexOf('/vendor/') !== -1) { e.respondWith(cacheFirst(req, VENDOR)); return; }
  if (req.mode === 'navigate' || /\.(html|json)$/.test(p) || p.endsWith('/')) { e.respondWith(networkFirst(req)); return; }
  if (/\.(mp4|mov|webm)$/.test(p)) return;              // don't bloat the cache with videos
  e.respondWith(staleWhileRevalidate(req));
});
