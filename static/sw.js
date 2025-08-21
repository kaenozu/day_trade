
// Service Worker - Day Trade Personal PWA
const CACHE_NAME = 'day-trade-v1.0.0';
const urlsToCache = [
    '/',
    '/static/css/mobile.css',
    '/static/js/mobile.js',
    '/static/js/offline.js',
    '/static/icons/icon-192x192.png',
    '/api/stocks/favorites',
    '/api/ai/quick-analysis'
];

// インストール
self.addEventListener('install', event => {
    console.log('[SW] Installing...');
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                console.log('[SW] Caching app shell');
                return cache.addAll(urlsToCache);
            })
    );
});

// アクティベート
self.addEventListener('activate', event => {
    console.log('[SW] Activating...');
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cacheName => {
                    if (cacheName !== CACHE_NAME) {
                        console.log('[SW] Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
});

// フェッチ（ネットワーク優先、フォールバックでキャッシュ）
self.addEventListener('fetch', event => {
    event.respondWith(
        fetch(event.request)
            .then(response => {
                // レスポンスのクローンを作成
                const responseClone = response.clone();

                // 成功レスポンスのみキャッシュ
                if (response.status === 200) {
                    caches.open(CACHE_NAME)
                        .then(cache => {
                            cache.put(event.request, responseClone);
                        });
                }

                return response;
            })
            .catch(() => {
                // ネットワークエラー時はキャッシュから返す
                return caches.match(event.request)
                    .then(response => {
                        if (response) {
                            return response;
                        }

                        // オフラインページを返す
                        if (event.request.destination === 'document') {
                            return caches.match('/offline.html');
                        }

                        return new Response('オフラインです', {
                            status: 503,
                            statusText: 'Service Unavailable'
                        });
                    });
            })
    );
});

// プッシュ通知
self.addEventListener('push', event => {
    console.log('[SW] Push received');

    const options = {
        body: event.data ? event.data.text() : '新しい市場情報があります',
        icon: '/static/icons/icon-192x192.png',
        badge: '/static/icons/badge-72x72.png',
        vibrate: [100, 50, 100],
        data: {
            dateOfArrival: Date.now(),
            primaryKey: 1
        },
        actions: [
            {
                action: 'explore',
                title: '詳細を見る',
                icon: '/static/icons/checkmark.png'
            },
            {
                action: 'close',
                title: '閉じる',
                icon: '/static/icons/xmark.png'
            }
        ]
    };

    event.waitUntil(
        self.registration.showNotification('Day Trade Personal Mobile', options)
    );
});

// 通知クリック処理
self.addEventListener('notificationclick', event => {
    console.log('[SW] Notification clicked');
    event.notification.close();

    if (event.action === 'explore') {
        event.waitUntil(
            clients.openWindow('/dashboard')
        );
    }
});

// バックグラウンド同期
self.addEventListener('sync', event => {
    if (event.tag === 'background-sync') {
        console.log('[SW] Background sync triggered');
        event.waitUntil(doBackgroundSync());
    }
});

function doBackgroundSync() {
    return fetch('/api/sync')
        .then(response => response.json())
        .then(data => {
            console.log('[SW] Background sync completed:', data);
        })
        .catch(err => {
            console.error('[SW] Background sync failed:', err);
        });
}
