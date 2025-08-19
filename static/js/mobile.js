
// Day Trade Personal - Mobile JavaScript

class MobileApp {
    constructor() {
        this.isOnline = navigator.onLine;
        this.currentPage = 'dashboard';
        this.init();
    }

    init() {
        this.registerServiceWorker();
        this.setupEventListeners();
        this.setupOfflineHandling();
        this.setupPullToRefresh();
        this.setupTouchFeedback();
        this.loadInitialData();
    }

    // Service Worker登録
    async registerServiceWorker() {
        if ('serviceWorker' in navigator) {
            try {
                const registration = await navigator.serviceWorker.register('/static/sw.js');
                console.log('SW registered:', registration);

                // アップデート検出
                registration.addEventListener('updatefound', () => {
                    this.showUpdateNotification();
                });

            } catch (error) {
                console.error('SW registration failed:', error);
            }
        }
    }

    // イベントリスナー設定
    setupEventListeners() {
        // ハンバーガーメニュー
        const hamburger = document.querySelector('.hamburger-menu');
        if (hamburger) {
            hamburger.addEventListener('click', this.toggleMenu.bind(this));
        }

        // ボトムナビゲーション
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', this.handleNavigation.bind(this));
        });

        // プッシュ通知許可
        const notificationBtn = document.querySelector('.enable-notifications');
        if (notificationBtn) {
            notificationBtn.addEventListener('click', this.requestNotificationPermission.bind(this));
        }

        // オンライン/オフライン検出
        window.addEventListener('online', this.handleOnline.bind(this));
        window.addEventListener('offline', this.handleOffline.bind(this));

        // ビューポート変更対応
        window.addEventListener('resize', this.handleViewportChange.bind(this));
        window.addEventListener('orientationchange', this.handleOrientationChange.bind(this));
    }

    // オフライン処理設定
    setupOfflineHandling() {
        this.updateOnlineStatus();

        // オフラインデータキャッシュ
        this.setupOfflineCache();
    }

    updateOnlineStatus() {
        const banner = document.querySelector('.offline-banner');
        if (banner) {
            banner.classList.toggle('show', !this.isOnline);
        }

        // オフライン時のUI調整
        document.body.classList.toggle('offline-mode', !this.isOnline);
    }

    // プルトゥリフレッシュ設定
    setupPullToRefresh() {
        let startY = 0;
        let currentY = 0;
        let isPulling = false;

        const wrapper = document.querySelector('.ptr-wrapper');
        const element = document.querySelector('.ptr-element');

        if (!wrapper || !element) return;

        wrapper.addEventListener('touchstart', (e) => {
            if (window.scrollY === 0) {
                startY = e.touches[0].pageY;
                isPulling = true;
            }
        });

        wrapper.addEventListener('touchmove', (e) => {
            if (!isPulling) return;

            currentY = e.touches[0].pageY;
            const pullDistance = currentY - startY;

            if (pullDistance > 0) {
                e.preventDefault();

                if (pullDistance > 100) {
                    element.classList.add('pulling');
                } else {
                    element.classList.remove('pulling');
                }

                element.style.transform = `translateX(-50%) translateY(${Math.min(pullDistance - 50, 50)}px)`;
            }
        });

        wrapper.addEventListener('touchend', () => {
            if (!isPulling) return;

            const pullDistance = currentY - startY;

            if (pullDistance > 100) {
                this.refreshData();
            }

            element.style.transform = 'translateX(-50%) translateY(-50px)';
            element.classList.remove('pulling');
            isPulling = false;
        });
    }

    // タッチフィードバック設定
    setupTouchFeedback() {
        document.querySelectorAll('.mobile-btn, .stock-item, .nav-item').forEach(element => {
            element.addEventListener('touchstart', () => {
                element.classList.add('tap-feedback');
            });

            element.addEventListener('touchend', () => {
                setTimeout(() => {
                    element.classList.remove('tap-feedback');
                }, 200);
            });
        });
    }

    // メニュー切り替え
    toggleMenu() {
        const menu = document.querySelector('.mobile-menu');
        if (menu) {
            menu.classList.toggle('open');
        }
    }

    // ナビゲーション処理
    handleNavigation(event) {
        event.preventDefault();
        const target = event.currentTarget.getAttribute('data-page');

        if (target && target !== this.currentPage) {
            this.navigateTo(target);
        }
    }

    navigateTo(page) {
        // アクティブタブ更新
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });

        document.querySelector(`[data-page="${page}"]`).classList.add('active');

        // ページコンテンツ切り替え
        this.loadPage(page);
        this.currentPage = page;
    }

    async loadPage(page) {
        const main = document.querySelector('.mobile-main');
        if (!main) return;

        // ローディング表示
        main.innerHTML = `
            <div class="mobile-loading">
                <div class="mobile-spinner"></div>
                <p>読み込み中...</p>
            </div>
        `;

        try {
            const response = await fetch(`/api/mobile/${page}`);
            const html = await response.text();
            main.innerHTML = html;

            // ページ固有の初期化
            this.initializePage(page);

        } catch (error) {
            console.error('Page load failed:', error);

            if (!this.isOnline) {
                // オフラインキャッシュからロード
                const cachedData = this.getOfflineData(page);
                this.renderOfflinePage(page, cachedData);
            } else {
                main.innerHTML = `
                    <div class="mobile-card">
                        <h3>エラー</h3>
                        <p>ページの読み込みに失敗しました。</p>
                        <button class="mobile-btn" onclick="location.reload()">再読み込み</button>
                    </div>
                `;
            }
        }
    }

    initializePage(page) {
        switch (page) {
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
        }
    }

    initDashboard() {
        // ダッシュボード初期化
        this.loadStockSummary();
        this.loadAIRecommendations();
        this.startRealTimeUpdates();
    }

    initStocks() {
        // 銘柄リスト初期化
        this.loadWatchlist();
        this.setupStockSearch();
    }

    initAIAnalysis() {
        // AI分析初期化
        this.loadAISignals();
        this.setupAnalysisFilters();
    }

    initSettings() {
        // 設定画面初期化
        this.loadUserSettings();
        this.setupSettingsForm();
    }

    // データ更新
    async refreshData() {
        console.log('Refreshing data...');

        try {
            await this.loadPage(this.currentPage);
            this.showToast('データを更新しました');
        } catch (error) {
            this.showToast('更新に失敗しました', 'error');
        }
    }

    // 初期データ読み込み
    async loadInitialData() {
        if (this.isOnline) {
            await this.loadPage('dashboard');
        } else {
            this.renderOfflinePage('dashboard');
        }
    }

    // プッシュ通知許可要求
    async requestNotificationPermission() {
        if ('Notification' in window) {
            const permission = await Notification.requestPermission();

            if (permission === 'granted') {
                this.showToast('通知を有効にしました');
                this.subscribeToNotifications();
            } else {
                this.showToast('通知が拒否されました');
            }
        }
    }

    async subscribeToNotifications() {
        try {
            const registration = await navigator.serviceWorker.ready;
            const subscription = await registration.pushManager.subscribe({
                userVisibleOnly: true,
                applicationServerKey: this.urlBase64ToUint8Array('Day Trade Personal Mobile')
            });

            // サブスクリプションをサーバーに送信
            await fetch('/api/push/subscribe', {
                method: 'POST',
                body: JSON.stringify(subscription),
                headers: {
                    'Content-Type': 'application/json'
                }
            });

        } catch (error) {
            console.error('Push subscription failed:', error);
        }
    }

    urlBase64ToUint8Array(base64String) {
        const padding = '='.repeat((4 - base64String.length % 4) % 4);
        const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
        const rawData = window.atob(base64);
        const outputArray = new Uint8Array(rawData.length);

        for (let i = 0; i < rawData.length; ++i) {
            outputArray[i] = rawData.charCodeAt(i);
        }
        return outputArray;
    }

    // オフライン処理
    handleOffline() {
        this.isOnline = false;
        this.updateOnlineStatus();
        this.showToast('オフラインになりました', 'warning');
    }

    handleOnline() {
        this.isOnline = true;
        this.updateOnlineStatus();
        this.showToast('オンラインに復帰しました', 'success');

        // 同期処理
        this.syncOfflineData();
    }

    // オフラインキャッシュ管理
    setupOfflineCache() {
        this.offlineCache = {
            dashboard: null,
            stocks: null,
            aiAnalysis: null,
            lastUpdate: null
        };

        this.loadOfflineCache();
    }

    saveOfflineData(page, data) {
        this.offlineCache[page] = data;
        this.offlineCache.lastUpdate = new Date().toISOString();
        localStorage.setItem('dayTradeOfflineCache', JSON.stringify(this.offlineCache));
    }

    getOfflineData(page) {
        return this.offlineCache[page];
    }

    loadOfflineCache() {
        const cached = localStorage.getItem('dayTradeOfflineCache');
        if (cached) {
            this.offlineCache = JSON.parse(cached);
        }
    }

    renderOfflinePage(page, data = null) {
        const main = document.querySelector('.mobile-main');
        if (!main) return;

        if (data) {
            // キャッシュデータを使用
            main.innerHTML = this.generateOfflinePageHTML(page, data);
        } else {
            // オフライン専用ページ
            main.innerHTML = `
                <div class="mobile-card">
                    <h3>🔌 オフラインモード</h3>
                    <p>現在オフラインです。インターネット接続を確認してください。</p>
                    <p>一部の機能は制限されています。</p>
                    <button class="mobile-btn" onclick="location.reload()">再試行</button>
                </div>
            `;
        }
    }

    // ユーティリティ
    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;

        const style = {
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
        };

        Object.assign(toast.style, style);
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => document.body.removeChild(toast), 300);
        }, 3000);
    }

    handleViewportChange() {
        // ビューポート変更対応（キーボード表示など）
        const vh = window.innerHeight * 0.01;
        document.documentElement.style.setProperty('--vh', `${vh}px`);
    }

    handleOrientationChange() {
        // 画面回転対応
        setTimeout(() => {
            this.handleViewportChange();
        }, 100);
    }

    showUpdateNotification() {
        this.showToast('アプリの更新があります。再読み込みしてください。', 'info');
    }
}

// アプリ初期化
document.addEventListener('DOMContentLoaded', () => {
    window.mobileApp = new MobileApp();
});

// PWAインストールプロンプト
let deferredPrompt;

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;

    // インストールボタン表示
    const installBtn = document.querySelector('.install-app');
    if (installBtn) {
        installBtn.style.display = 'block';
        installBtn.addEventListener('click', showInstallPrompt);
    }
});

function showInstallPrompt() {
    if (deferredPrompt) {
        deferredPrompt.prompt();

        deferredPrompt.userChoice.then((choiceResult) => {
            if (choiceResult.outcome === 'accepted') {
                console.log('PWA installed');
            }
            deferredPrompt = null;
        });
    }
}
