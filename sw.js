// PulseTrade AI — Service Worker
// Enables offline support and installability

const CACHE = "pulsetrade-v1";
const OFFLINE_ASSETS = [
  "/",
  "/index.html",
  "/manifest.json",
  "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap",
  "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"
];

// Install — cache core assets
self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE).then(cache => {
      return cache.addAll(OFFLINE_ASSETS).catch(() => {
        // Silently fail on external resources — that's fine
      });
    })
  );
  self.skipWaiting();
});

// Activate — clean up old caches
self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(k => k !== CACHE).map(k => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

// Fetch — network first, fall back to cache
self.addEventListener("fetch", event => {
  const url = new URL(event.request.url);

  // Always fetch API calls fresh — never cache live data
  if (url.pathname.startsWith("/api/")) {
    event.respondWith(fetch(event.request));
    return;
  }

  // For everything else: try network, fall back to cache
  event.respondWith(
    fetch(event.request)
      .then(response => {
        // Cache a copy of successful responses
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE).then(cache => cache.put(event.request, clone));
        }
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
