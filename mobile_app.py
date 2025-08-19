#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mobile Application - PWA対応モバイルアプリ
Issue #951対応: Progressive Web App + モバイル最適化 + オフライン対応
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging


class DeviceType(Enum):
    """デバイスタイプ"""
    MOBILE = "MOBILE"
    TABLET = "TABLET"
    DESKTOP = "DESKTOP"


class ScreenOrientation(Enum):
    """画面向き"""
    PORTRAIT = "PORTRAIT"
    LANDSCAPE = "LANDSCAPE"


@dataclass
class MobileConfig:
    """モバイル設定"""
    app_name: str = "Day Trade Personal"
    app_short_name: str = "DayTrade"
    app_description: str = "AI-powered financial analysis mobile app"
    theme_color: str = "#1976d2"
    background_color: str = "#ffffff"
    display_mode: str = "standalone"  # standalone, fullscreen, minimal-ui
    orientation: str = "any"  # any, natural, landscape, portrait
    start_url: str = "/"
    scope: str = "/"
    offline_enabled: bool = True
    push_notifications: bool = True


class MobileAppGenerator:
    """モバイルアプリ生成器"""
    
    def __init__(self, config: MobileConfig = None):
        self.config = config or MobileConfig()
        self.static_dir = "static"
        self.templates_dir = "templates"
        
        os.makedirs(self.static_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)
    
    def generate_pwa_manifest(self) -> str:
        """PWAマニフェスト生成"""
        manifest = {
            "name": self.config.app_name,
            "short_name": self.config.app_short_name,
            "description": self.config.app_description,
            "start_url": self.config.start_url,
            "scope": self.config.scope,
            "display": self.config.display_mode,
            "orientation": self.config.orientation,
            "theme_color": self.config.theme_color,
            "background_color": self.config.background_color,
            "icons": [
                {
                    "src": "/static/icons/icon-72x72.png",
                    "sizes": "72x72",
                    "type": "image/png"
                },
                {
                    "src": "/static/icons/icon-96x96.png",
                    "sizes": "96x96",
                    "type": "image/png"
                },
                {
                    "src": "/static/icons/icon-128x128.png",
                    "sizes": "128x128",
                    "type": "image/png"
                },
                {
                    "src": "/static/icons/icon-144x144.png",
                    "sizes": "144x144",
                    "type": "image/png"
                },
                {
                    "src": "/static/icons/icon-152x152.png",
                    "sizes": "152x152",
                    "type": "image/png"
                },
                {
                    "src": "/static/icons/icon-192x192.png",
                    "sizes": "192x192",
                    "type": "image/png",
                    "purpose": "any maskable"
                },
                {
                    "src": "/static/icons/icon-384x384.png",
                    "sizes": "384x384",
                    "type": "image/png"
                },
                {
                    "src": "/static/icons/icon-512x512.png",
                    "sizes": "512x512",
                    "type": "image/png"
                }
            ]
        }
        
        manifest_path = os.path.join(self.static_dir, "manifest.json")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        return manifest_path
    
    def generate_service_worker(self) -> str:
        """Service Worker生成"""
        service_worker_js = f"""
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
self.addEventListener('install', event => {{
    console.log('[SW] Installing...');
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {{
                console.log('[SW] Caching app shell');
                return cache.addAll(urlsToCache);
            }})
    );
}});

// アクティベート
self.addEventListener('activate', event => {{
    console.log('[SW] Activating...');
    event.waitUntil(
        caches.keys().then(cacheNames => {{
            return Promise.all(
                cacheNames.map(cacheName => {{
                    if (cacheName !== CACHE_NAME) {{
                        console.log('[SW] Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }}
                }})
            );
        }})
    );
}});

// フェッチ（ネットワーク優先、フォールバックでキャッシュ）
self.addEventListener('fetch', event => {{
    event.respondWith(
        fetch(event.request)
            .then(response => {{
                // レスポンスのクローンを作成
                const responseClone = response.clone();
                
                // 成功レスポンスのみキャッシュ
                if (response.status === 200) {{
                    caches.open(CACHE_NAME)
                        .then(cache => {{
                            cache.put(event.request, responseClone);
                        }});
                }}
                
                return response;
            }})
            .catch(() => {{
                // ネットワークエラー時はキャッシュから返す
                return caches.match(event.request)
                    .then(response => {{
                        if (response) {{
                            return response;
                        }}
                        
                        // オフラインページを返す
                        if (event.request.destination === 'document') {{
                            return caches.match('/offline.html');
                        }}
                        
                        return new Response('オフラインです', {{
                            status: 503,
                            statusText: 'Service Unavailable'
                        }});
                    }});
            }})
    );
}});

// プッシュ通知
self.addEventListener('push', event => {{
    console.log('[SW] Push received');
    
    const options = {{
        body: event.data ? event.data.text() : '新しい市場情報があります',
        icon: '/static/icons/icon-192x192.png',
        badge: '/static/icons/badge-72x72.png',
        vibrate: [100, 50, 100],
        data: {{
            dateOfArrival: Date.now(),
            primaryKey: 1
        }},
        actions: [
            {{
                action: 'explore',
                title: '詳細を見る',
                icon: '/static/icons/checkmark.png'
            }},
            {{
                action: 'close',
                title: '閉じる',
                icon: '/static/icons/xmark.png'
            }}
        ]
    }};
    
    event.waitUntil(
        self.registration.showNotification('{self.config.app_name}', options)
    );
}});

// 通知クリック処理
self.addEventListener('notificationclick', event => {{
    console.log('[SW] Notification clicked');
    event.notification.close();
    
    if (event.action === 'explore') {{
        event.waitUntil(
            clients.openWindow('/dashboard')
        );
    }}
}});

// バックグラウンド同期
self.addEventListener('sync', event => {{
    if (event.tag === 'background-sync') {{
        console.log('[SW] Background sync triggered');
        event.waitUntil(doBackgroundSync());
    }}
}});

function doBackgroundSync() {{
    return fetch('/api/sync')
        .then(response => response.json())
        .then(data => {{
            console.log('[SW] Background sync completed:', data);
        }})
        .catch(err => {{
            console.error('[SW] Background sync failed:', err);
        }});
}}
"""
        
        sw_path = os.path.join(self.static_dir, "sw.js")
        with open(sw_path, 'w', encoding='utf-8') as f:
            f.write(service_worker_js)
        
        return sw_path
    
    def generate_mobile_css(self) -> str:
        """モバイル用CSS生成"""
        mobile_css = """
/* Day Trade Personal - Mobile Styles */

/* ベースレイアウト */
* {
    box-sizing: border-box;
    -webkit-tap-highlight-color: transparent;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    margin: 0;
    padding: 0;
    background: #f5f5f5;
    overflow-x: hidden;
}

/* ヘッダー */
.mobile-header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 60px;
    background: #1976d2;
    color: white;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    z-index: 1000;
}

.mobile-header h1 {
    font-size: 18px;
    margin: 0;
    font-weight: 600;
}

.hamburger-menu {
    width: 24px;
    height: 24px;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    justify-content: space-around;
}

.hamburger-menu span {
    width: 100%;
    height: 3px;
    background: white;
    border-radius: 2px;
    transition: all 0.3s;
}

/* メインコンテンツ */
.mobile-main {
    margin-top: 60px;
    padding: 16px;
    min-height: calc(100vh - 120px);
}

/* カード */
.mobile-card {
    background: white;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border: 1px solid #e0e0e0;
}

.mobile-card h3 {
    margin: 0 0 12px 0;
    font-size: 16px;
    color: #333;
}

/* グリッドレイアウト */
.mobile-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin: 16px 0;
}

.mobile-grid-item {
    background: white;
    padding: 12px;
    border-radius: 8px;
    text-align: center;
    border: 1px solid #e0e0e0;
    min-height: 80px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

/* ボタン */
.mobile-btn {
    background: #1976d2;
    color: white;
    border: none;
    padding: 14px 24px;
    border-radius: 8px;
    font-size: 16px;
    cursor: pointer;
    width: 100%;
    margin: 8px 0;
    font-weight: 600;
    transition: all 0.3s;
    min-height: 48px; /* タッチターゲット最小サイズ */
}

.mobile-btn:hover {
    background: #1565c0;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(25, 118, 210, 0.3);
}

.mobile-btn:active {
    transform: translateY(0);
}

.mobile-btn.secondary {
    background: #f5f5f5;
    color: #333;
}

.mobile-btn.secondary:hover {
    background: #e0e0e0;
}

/* フォーム */
.mobile-input {
    width: 100%;
    padding: 14px;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    font-size: 16px; /* iOSでのズームを防ぐ */
    margin: 8px 0;
    transition: border-color 0.3s;
}

.mobile-input:focus {
    outline: none;
    border-color: #1976d2;
}

.mobile-select {
    width: 100%;
    padding: 14px;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    font-size: 16px;
    background: white;
    margin: 8px 0;
}

/* 銘柄リスト */
.stock-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.stock-item {
    background: white;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 8px;
    border: 1px solid #e0e0e0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.stock-info h4 {
    margin: 0 0 4px 0;
    font-size: 16px;
    color: #333;
}

.stock-info p {
    margin: 0;
    font-size: 14px;
    color: #666;
}

.stock-price {
    text-align: right;
}

.price-value {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 4px;
}

.price-change {
    font-size: 14px;
    padding: 4px 8px;
    border-radius: 4px;
}

.price-change.positive {
    background: #e8f5e8;
    color: #2e7d32;
}

.price-change.negative {
    background: #ffebee;
    color: #d32f2f;
}

/* チャート */
.mobile-chart {
    height: 200px;
    margin: 16px 0;
    background: white;
    border-radius: 8px;
    padding: 16px;
}

/* タブ */
.mobile-tabs {
    display: flex;
    background: white;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 16px;
    border: 1px solid #e0e0e0;
}

.mobile-tab {
    flex: 1;
    padding: 14px;
    text-align: center;
    background: #f5f5f5;
    border: none;
    cursor: pointer;
    font-size: 14px;
    transition: background-color 0.3s;
}

.mobile-tab.active {
    background: #1976d2;
    color: white;
}

/* ローディング */
.mobile-loading {
    text-align: center;
    padding: 40px 20px;
}

.mobile-spinner {
    width: 40px;
    height: 40px;
    border: 4px solid #f3f3f3;
    border-top: 4px solid #1976d2;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 16px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* ボトムナビゲーション */
.mobile-bottom-nav {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    height: 60px;
    background: white;
    border-top: 1px solid #e0e0e0;
    display: flex;
    z-index: 1000;
}

.nav-item {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-decoration: none;
    color: #666;
    font-size: 12px;
    padding: 8px;
    transition: color 0.3s;
}

.nav-item.active {
    color: #1976d2;
}

.nav-item i {
    font-size: 20px;
    margin-bottom: 4px;
}

/* レスポンシブ対応 */
@media (max-width: 480px) {
    .mobile-main {
        padding: 12px;
    }
    
    .mobile-grid {
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 8px;
    }
    
    .mobile-card {
        padding: 12px;
    }
    
    .stock-item {
        padding: 12px;
    }
}

/* ランドスケープモード */
@media (orientation: landscape) and (max-height: 500px) {
    .mobile-header {
        height: 48px;
    }
    
    .mobile-main {
        margin-top: 48px;
        padding: 12px;
    }
    
    .mobile-bottom-nav {
        height: 48px;
    }
}

/* ダークモード対応 */
@media (prefers-color-scheme: dark) {
    body {
        background: #121212;
        color: #ffffff;
    }
    
    .mobile-card {
        background: #1e1e1e;
        border-color: #333;
        color: #ffffff;
    }
    
    .mobile-input {
        background: #1e1e1e;
        color: #ffffff;
        border-color: #333;
    }
    
    .stock-item {
        background: #1e1e1e;
        border-color: #333;
    }
    
    .mobile-bottom-nav {
        background: #1e1e1e;
        border-top-color: #333;
    }
    
    .mobile-tabs {
        background: #1e1e1e;
        border-color: #333;
    }
    
    .mobile-tab {
        background: #2e2e2e;
        color: #ffffff;
    }
}

/* プルトゥリフレッシュ */
.ptr-wrapper {
    position: relative;
}

.ptr-element {
    position: absolute;
    top: -50px;
    left: 50%;
    transform: translateX(-50%);
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: #1976d2;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    transition: all 0.3s;
}

.ptr-element.pulling {
    top: 10px;
}

/* オフライン表示 */
.offline-banner {
    position: fixed;
    top: 60px;
    left: 0;
    right: 0;
    background: #ff9800;
    color: white;
    padding: 8px 16px;
    font-size: 14px;
    text-align: center;
    z-index: 999;
    transform: translateY(-100%);
    transition: transform 0.3s;
}

.offline-banner.show {
    transform: translateY(0);
}

/* タッチフィードバック */
@keyframes tap-feedback {
    0% { transform: scale(1); }
    50% { transform: scale(0.95); }
    100% { transform: scale(1); }
}

.tap-feedback {
    animation: tap-feedback 0.2s ease;
}
"""
        
        css_path = os.path.join(self.static_dir, "css", "mobile.css")
        os.makedirs(os.path.dirname(css_path), exist_ok=True)
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write(mobile_css)
        
        return css_path
    
    def generate_mobile_js(self) -> str:
        """モバイル用JavaScript生成"""
        mobile_js = f"""
// Day Trade Personal - Mobile JavaScript

class MobileApp {{
    constructor() {{
        this.isOnline = navigator.onLine;
        this.currentPage = 'dashboard';
        this.init();
    }}
    
    init() {{
        this.registerServiceWorker();
        this.setupEventListeners();
        this.setupOfflineHandling();
        this.setupPullToRefresh();
        this.setupTouchFeedback();
        this.loadInitialData();
    }}
    
    // Service Worker登録
    async registerServiceWorker() {{
        if ('serviceWorker' in navigator) {{
            try {{
                const registration = await navigator.serviceWorker.register('/static/sw.js');
                console.log('SW registered:', registration);
                
                // アップデート検出
                registration.addEventListener('updatefound', () => {{
                    this.showUpdateNotification();
                }});
                
            }} catch (error) {{
                console.error('SW registration failed:', error);
            }}
        }}
    }}
    
    // イベントリスナー設定
    setupEventListeners() {{
        // ハンバーガーメニュー
        const hamburger = document.querySelector('.hamburger-menu');
        if (hamburger) {{
            hamburger.addEventListener('click', this.toggleMenu.bind(this));
        }}
        
        // ボトムナビゲーション
        document.querySelectorAll('.nav-item').forEach(item => {{
            item.addEventListener('click', this.handleNavigation.bind(this));
        }});
        
        // プッシュ通知許可
        const notificationBtn = document.querySelector('.enable-notifications');
        if (notificationBtn) {{
            notificationBtn.addEventListener('click', this.requestNotificationPermission.bind(this));
        }}
        
        // オンライン/オフライン検出
        window.addEventListener('online', this.handleOnline.bind(this));
        window.addEventListener('offline', this.handleOffline.bind(this));
        
        // ビューポート変更対応
        window.addEventListener('resize', this.handleViewportChange.bind(this));
        window.addEventListener('orientationchange', this.handleOrientationChange.bind(this));
    }}
    
    // オフライン処理設定
    setupOfflineHandling() {{
        this.updateOnlineStatus();
        
        // オフラインデータキャッシュ
        this.setupOfflineCache();
    }}
    
    updateOnlineStatus() {{
        const banner = document.querySelector('.offline-banner');
        if (banner) {{
            banner.classList.toggle('show', !this.isOnline);
        }}
        
        // オフライン時のUI調整
        document.body.classList.toggle('offline-mode', !this.isOnline);
    }}
    
    // プルトゥリフレッシュ設定
    setupPullToRefresh() {{
        let startY = 0;
        let currentY = 0;
        let isPulling = false;
        
        const wrapper = document.querySelector('.ptr-wrapper');
        const element = document.querySelector('.ptr-element');
        
        if (!wrapper || !element) return;
        
        wrapper.addEventListener('touchstart', (e) => {{
            if (window.scrollY === 0) {{
                startY = e.touches[0].pageY;
                isPulling = true;
            }}
        }});
        
        wrapper.addEventListener('touchmove', (e) => {{
            if (!isPulling) return;
            
            currentY = e.touches[0].pageY;
            const pullDistance = currentY - startY;
            
            if (pullDistance > 0) {{
                e.preventDefault();
                
                if (pullDistance > 100) {{
                    element.classList.add('pulling');
                }} else {{
                    element.classList.remove('pulling');
                }}
                
                element.style.transform = `translateX(-50%) translateY(${{Math.min(pullDistance - 50, 50)}}px)`;
            }}
        }});
        
        wrapper.addEventListener('touchend', () => {{
            if (!isPulling) return;
            
            const pullDistance = currentY - startY;
            
            if (pullDistance > 100) {{
                this.refreshData();
            }}
            
            element.style.transform = 'translateX(-50%) translateY(-50px)';
            element.classList.remove('pulling');
            isPulling = false;
        }});
    }}
    
    // タッチフィードバック設定
    setupTouchFeedback() {{
        document.querySelectorAll('.mobile-btn, .stock-item, .nav-item').forEach(element => {{
            element.addEventListener('touchstart', () => {{
                element.classList.add('tap-feedback');
            }});
            
            element.addEventListener('touchend', () => {{
                setTimeout(() => {{
                    element.classList.remove('tap-feedback');
                }}, 200);
            }});
        }});
    }}
    
    // メニュー切り替え
    toggleMenu() {{
        const menu = document.querySelector('.mobile-menu');
        if (menu) {{
            menu.classList.toggle('open');
        }}
    }}
    
    // ナビゲーション処理
    handleNavigation(event) {{
        event.preventDefault();
        const target = event.currentTarget.getAttribute('data-page');
        
        if (target && target !== this.currentPage) {{
            this.navigateTo(target);
        }}
    }}
    
    navigateTo(page) {{
        // アクティブタブ更新
        document.querySelectorAll('.nav-item').forEach(item => {{
            item.classList.remove('active');
        }});
        
        document.querySelector(`[data-page="${{page}}"]`).classList.add('active');
        
        // ページコンテンツ切り替え
        this.loadPage(page);
        this.currentPage = page;
    }}
    
    async loadPage(page) {{
        const main = document.querySelector('.mobile-main');
        if (!main) return;
        
        // ローディング表示
        main.innerHTML = `
            <div class="mobile-loading">
                <div class="mobile-spinner"></div>
                <p>読み込み中...</p>
            </div>
        `;
        
        try {{
            const response = await fetch(`/api/mobile/${{page}}`);
            const html = await response.text();
            main.innerHTML = html;
            
            // ページ固有の初期化
            this.initializePage(page);
            
        }} catch (error) {{
            console.error('Page load failed:', error);
            
            if (!this.isOnline) {{
                // オフラインキャッシュからロード
                const cachedData = this.getOfflineData(page);
                this.renderOfflinePage(page, cachedData);
            }} else {{
                main.innerHTML = `
                    <div class="mobile-card">
                        <h3>エラー</h3>
                        <p>ページの読み込みに失敗しました。</p>
                        <button class="mobile-btn" onclick="location.reload()">再読み込み</button>
                    </div>
                `;
            }}
        }}
    }}
    
    initializePage(page) {{
        switch (page) {{
            case 'dashboard':
                this.initDashboard();
                break;
            case 'stocks':
                this.initStocks();
                break;
            case 'ai-analysis':
                this.initAIAnalysis();
                break;
            case 'settings':
                this.initSettings();
                break;
        }}
    }}
    
    initDashboard() {{
        // ダッシュボード初期化
        this.loadStockSummary();
        this.loadAIRecommendations();
        this.startRealTimeUpdates();
    }}
    
    initStocks() {{
        // 銘柄リスト初期化
        this.loadWatchlist();
        this.setupStockSearch();
    }}
    
    initAIAnalysis() {{
        // AI分析初期化
        this.loadAISignals();
        this.setupAnalysisFilters();
    }}
    
    initSettings() {{
        // 設定画面初期化
        this.loadUserSettings();
        this.setupSettingsForm();
    }}
    
    // データ更新
    async refreshData() {{
        console.log('Refreshing data...');
        
        try {{
            await this.loadPage(this.currentPage);
            this.showToast('データを更新しました');
        }} catch (error) {{
            this.showToast('更新に失敗しました', 'error');
        }}
    }}
    
    // 初期データ読み込み
    async loadInitialData() {{
        if (this.isOnline) {{
            await this.loadPage('dashboard');
        }} else {{
            this.renderOfflinePage('dashboard');
        }}
    }}
    
    // プッシュ通知許可要求
    async requestNotificationPermission() {{
        if ('Notification' in window) {{
            const permission = await Notification.requestPermission();
            
            if (permission === 'granted') {{
                this.showToast('通知を有効にしました');
                this.subscribeToNotifications();
            }} else {{
                this.showToast('通知が拒否されました');
            }}
        }}
    }}
    
    async subscribeToNotifications() {{
        try {{
            const registration = await navigator.serviceWorker.ready;
            const subscription = await registration.pushManager.subscribe({{
                userVisibleOnly: true,
                applicationServerKey: this.urlBase64ToUint8Array('{self.config.app_name}')
            }});
            
            // サブスクリプションをサーバーに送信
            await fetch('/api/push/subscribe', {{
                method: 'POST',
                body: JSON.stringify(subscription),
                headers: {{
                    'Content-Type': 'application/json'
                }}
            }});
            
        }} catch (error) {{
            console.error('Push subscription failed:', error);
        }}
    }}
    
    urlBase64ToUint8Array(base64String) {{
        const padding = '='.repeat((4 - base64String.length % 4) % 4);
        const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
        const rawData = window.atob(base64);
        const outputArray = new Uint8Array(rawData.length);
        
        for (let i = 0; i < rawData.length; ++i) {{
            outputArray[i] = rawData.charCodeAt(i);
        }}
        return outputArray;
    }}
    
    // オフライン処理
    handleOffline() {{
        this.isOnline = false;
        this.updateOnlineStatus();
        this.showToast('オフラインになりました', 'warning');
    }}
    
    handleOnline() {{
        this.isOnline = true;
        this.updateOnlineStatus();
        this.showToast('オンラインに復帰しました', 'success');
        
        // 同期処理
        this.syncOfflineData();
    }}
    
    // オフラインキャッシュ管理
    setupOfflineCache() {{
        this.offlineCache = {{
            dashboard: null,
            stocks: null,
            aiAnalysis: null,
            lastUpdate: null
        }};
        
        this.loadOfflineCache();
    }}
    
    saveOfflineData(page, data) {{
        this.offlineCache[page] = data;
        this.offlineCache.lastUpdate = new Date().toISOString();
        localStorage.setItem('dayTradeOfflineCache', JSON.stringify(this.offlineCache));
    }}
    
    getOfflineData(page) {{
        return this.offlineCache[page];
    }}
    
    loadOfflineCache() {{
        const cached = localStorage.getItem('dayTradeOfflineCache');
        if (cached) {{
            this.offlineCache = JSON.parse(cached);
        }}
    }}
    
    renderOfflinePage(page, data = null) {{
        const main = document.querySelector('.mobile-main');
        if (!main) return;
        
        if (data) {{
            // キャッシュデータを使用
            main.innerHTML = this.generateOfflinePageHTML(page, data);
        }} else {{
            // オフライン専用ページ
            main.innerHTML = `
                <div class="mobile-card">
                    <h3>🔌 オフラインモード</h3>
                    <p>現在オフラインです。インターネット接続を確認してください。</p>
                    <p>一部の機能は制限されています。</p>
                    <button class="mobile-btn" onclick="location.reload()">再試行</button>
                </div>
            `;
        }}
    }}
    
    // ユーティリティ
    showToast(message, type = 'info') {{
        const toast = document.createElement('div');
        toast.className = `toast toast-${{type}}`;
        toast.textContent = message;
        
        const style = {{
            position: 'fixed',
            bottom: '80px',
            left: '16px',
            right: '16px',
            background: type === 'error' ? '#f44336' : type === 'warning' ? '#ff9800' : type === 'success' ? '#4caf50' : '#2196f3',
            color: 'white',
            padding: '12px',
            borderRadius: '8px',
            zIndex: '10000',
            textAlign: 'center',
            fontSize: '14px'
        }};
        
        Object.assign(toast.style, style);
        document.body.appendChild(toast);
        
        setTimeout(() => {{
            toast.style.opacity = '0';
            setTimeout(() => document.body.removeChild(toast), 300);
        }}, 3000);
    }}
    
    handleViewportChange() {{
        // ビューポート変更対応（キーボード表示など）
        const vh = window.innerHeight * 0.01;
        document.documentElement.style.setProperty('--vh', `${{vh}}px`);
    }}
    
    handleOrientationChange() {{
        // 画面回転対応
        setTimeout(() => {{
            this.handleViewportChange();
        }}, 100);
    }}
    
    showUpdateNotification() {{
        this.showToast('アプリの更新があります。再読み込みしてください。', 'info');
    }}
}}

// アプリ初期化
document.addEventListener('DOMContentLoaded', () => {{
    window.mobileApp = new MobileApp();
}});

// PWAインストールプロンプト
let deferredPrompt;

window.addEventListener('beforeinstallprompt', (e) => {{
    e.preventDefault();
    deferredPrompt = e;
    
    // インストールボタン表示
    const installBtn = document.querySelector('.install-app');
    if (installBtn) {{
        installBtn.style.display = 'block';
        installBtn.addEventListener('click', showInstallPrompt);
    }}
}});

function showInstallPrompt() {{
    if (deferredPrompt) {{
        deferredPrompt.prompt();
        
        deferredPrompt.userChoice.then((choiceResult) => {{
            if (choiceResult.outcome === 'accepted') {{
                console.log('PWA installed');
            }}
            deferredPrompt = null;
        }});
    }}
}}
"""
        
        js_path = os.path.join(self.static_dir, "js", "mobile.js")
        os.makedirs(os.path.dirname(js_path), exist_ok=True)
        with open(js_path, 'w', encoding='utf-8') as f:
            f.write(mobile_js)
        
        return js_path
    
    def generate_mobile_html(self) -> str:
        """モバイル用HTML生成"""
        mobile_html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <meta name="theme-color" content="{self.config.theme_color}">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="{self.config.app_short_name}">
    
    <title>{self.config.app_name}</title>
    
    <!-- PWA Manifest -->
    <link rel="manifest" href="/static/manifest.json">
    
    <!-- Icons -->
    <link rel="icon" type="image/png" sizes="32x32" href="/static/icons/icon-32x32.png">
    <link rel="icon" type="image/png" sizes="16x16" href="/static/icons/icon-16x16.png">
    <link rel="apple-touch-icon" href="/static/icons/icon-152x152.png">
    
    <!-- Styles -->
    <link rel="stylesheet" href="/static/css/mobile.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    
    <!-- iOS Safari specific -->
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <link rel="apple-touch-startup-image" href="/static/icons/icon-512x512.png">
</head>

<body>
    <!-- オフラインバナー -->
    <div class="offline-banner">
        <i class="fas fa-wifi"></i> オフラインです
    </div>
    
    <!-- ヘッダー -->
    <header class="mobile-header">
        <div class="hamburger-menu">
            <span></span>
            <span></span>
            <span></span>
        </div>
        <h1>{self.config.app_short_name}</h1>
        <div></div>
    </header>
    
    <!-- プルトゥリフレッシュ -->
    <div class="ptr-wrapper">
        <div class="ptr-element">
            <i class="fas fa-sync-alt"></i>
        </div>
        
        <!-- メインコンテンツ -->
        <main class="mobile-main">
            <!-- 初期ローディング -->
            <div class="mobile-loading">
                <div class="mobile-spinner"></div>
                <p>読み込み中...</p>
            </div>
        </main>
    </div>
    
    <!-- ボトムナビゲーション -->
    <nav class="mobile-bottom-nav">
        <a href="#dashboard" class="nav-item active" data-page="dashboard">
            <i class="fas fa-chart-line"></i>
            <span>ダッシュボード</span>
        </a>
        <a href="#stocks" class="nav-item" data-page="stocks">
            <i class="fas fa-list"></i>
            <span>銘柄</span>
        </a>
        <a href="#ai-analysis" class="nav-item" data-page="ai-analysis">
            <i class="fas fa-robot"></i>
            <span>AI分析</span>
        </a>
        <a href="#settings" class="nav-item" data-page="settings">
            <i class="fas fa-cog"></i>
            <span>設定</span>
        </a>
    </nav>
    
    <!-- Scripts -->
    <script src="/static/js/mobile.js"></script>
    
    <script>
        // PWA関連の初期化
        if ('serviceWorker' in navigator) {{
            window.addEventListener('load', () => {{
                navigator.serviceWorker.register('/static/sw.js');
            }});
        }}
        
        // ビューポート高さ対応（モバイルブラウザ対応）
        const setViewportHeight = () => {{
            const vh = window.innerHeight * 0.01;
            document.documentElement.style.setProperty('--vh', `${{vh}}px`);
        }};
        
        setViewportHeight();
        window.addEventListener('resize', setViewportHeight);
        window.addEventListener('orientationchange', () => {{
            setTimeout(setViewportHeight, 100);
        }});
    </script>
</body>
</html>
"""
        
        html_path = os.path.join(self.templates_dir, "mobile.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(mobile_html)
        
        return html_path
    
    def generate_offline_page(self) -> str:
        """オフラインページ生成"""
        offline_html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>オフライン - {self.config.app_name}</title>
    <link rel="stylesheet" href="/static/css/mobile.css">
</head>

<body>
    <div class="mobile-main" style="margin-top: 0; display: flex; align-items: center; justify-content: center; min-height: 100vh;">
        <div class="mobile-card" style="text-align: center; max-width: 400px;">
            <div style="font-size: 64px; margin-bottom: 20px;">📱</div>
            <h2>オフライン</h2>
            <p>インターネット接続が利用できません。</p>
            <p>接続が復旧したら自動的に同期されます。</p>
            
            <button class="mobile-btn" onclick="location.reload()">
                <i class="fas fa-sync-alt"></i>
                再試行
            </button>
            
            <div style="margin-top: 20px;">
                <small>キャッシュされたデータは引き続き利用できます。</small>
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        offline_path = os.path.join(self.templates_dir, "offline.html")
        with open(offline_path, 'w', encoding='utf-8') as f:
            f.write(offline_html)
        
        return offline_path
    
    def generate_app_icons(self):
        """アプリアイコン生成（プレースホルダー）"""
        icons_dir = os.path.join(self.static_dir, "icons")
        os.makedirs(icons_dir, exist_ok=True)
        
        # SVGアイコンを生成（実際のプロジェクトでは画像ファイルを使用）
        icon_sizes = [16, 32, 72, 96, 128, 144, 152, 192, 384, 512]
        
        for size in icon_sizes:
            svg_icon = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
  <rect width="{size}" height="{size}" fill="{self.config.theme_color}" rx="{size // 8}"/>
  <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" 
        fill="white" font-size="{size // 3}" font-weight="bold" font-family="Arial">DT</text>
</svg>
"""
            icon_path = os.path.join(icons_dir, f"icon-{size}x{size}.png")
            
            # 実際の実装では、SVGをPNGに変換するライブラリを使用
            # ここではプレースホルダーとしてSVGを保存
            with open(icon_path.replace('.png', '.svg'), 'w', encoding='utf-8') as f:
                f.write(svg_icon)
        
        return icons_dir
    
    def build_mobile_app(self) -> Dict[str, str]:
        """モバイルアプリビルド"""
        results = {}
        
        logging.info("Building mobile PWA application...")
        
        try:
            # 1. PWAマニフェスト生成
            results['manifest'] = self.generate_pwa_manifest()
            logging.info("Generated PWA manifest")
            
            # 2. Service Worker生成
            results['service_worker'] = self.generate_service_worker()
            logging.info("Generated service worker")
            
            # 3. CSS生成
            results['css'] = self.generate_mobile_css()
            logging.info("Generated mobile CSS")
            
            # 4. JavaScript生成
            results['js'] = self.generate_mobile_js()
            logging.info("Generated mobile JavaScript")
            
            # 5. HTML生成
            results['html'] = self.generate_mobile_html()
            logging.info("Generated mobile HTML")
            
            # 6. オフラインページ生成
            results['offline'] = self.generate_offline_page()
            logging.info("Generated offline page")
            
            # 7. アプリアイコン生成
            results['icons'] = self.generate_app_icons()
            logging.info("Generated app icons")
            
            logging.info("Mobile PWA application build completed!")
            
        except Exception as e:
            logging.error(f"Mobile app build failed: {e}")
            raise
        
        return results


# グローバルインスタンス
mobile_app_generator = MobileAppGenerator()


def build_mobile_app(config: MobileConfig = None) -> Dict[str, str]:
    """モバイルアプリビルド"""
    if config:
        global mobile_app_generator
        mobile_app_generator = MobileAppGenerator(config)
    
    return mobile_app_generator.build_mobile_app()


def get_mobile_config() -> MobileConfig:
    """モバイル設定取得"""
    return mobile_app_generator.config


if __name__ == "__main__":
    print("=== Mobile Application Build Test ===")
    
    # カスタム設定
    config = MobileConfig(
        app_name="Day Trade Personal Mobile",
        app_short_name="DayTradeMobile",
        theme_color="#1976d2",
        offline_enabled=True,
        push_notifications=True
    )
    
    # モバイルアプリビルド
    results = build_mobile_app(config)
    
    print("Mobile application build results:")
    for component, path in results.items():
        print(f"  {component}: {path}")
    
    print(f"\\nMobile PWA features:")
    print(f"  - Offline support: {config.offline_enabled}")
    print(f"  - Push notifications: {config.push_notifications}")
    print(f"  - Responsive design: Yes")
    print(f"  - Touch gestures: Yes")
    print(f"  - Pull-to-refresh: Yes")
    print(f"  - Install prompt: Yes")
    
    print("\\nMobile application build test completed!")