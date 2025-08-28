#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTML Templates - HTMLテンプレート

ウェブダッシュボードのHTMLテンプレートとフロントエンドコードの管理
"""

from typing import Dict, Any


class DashboardTemplates:
    """ダッシュボードテンプレート管理"""
    
    @staticmethod
    def get_main_dashboard_template() -> str:
        """メインダッシュボードのHTMLテンプレート"""
        return """
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enhanced Trading Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                :root {
                    --primary-color: #2563eb;
                    --secondary-color: #64748b;
                    --success-color: #10b981;
                    --danger-color: #ef4444;
                    --warning-color: #f59e0b;
                    --info-color: #06b6d4;
                    --dark-bg: #1e293b;
                    --card-bg: #ffffff;
                    --border-color: #e2e8f0;
                }

                .dashboard-card { 
                    margin-bottom: 20px; 
                    border: 1px solid var(--border-color);
                    border-radius: 8px;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                }
                
                .price-up { 
                    color: var(--success-color); 
                    font-weight: bold;
                }
                
                .price-down { 
                    color: var(--danger-color); 
                    font-weight: bold;
                }
                
                .alert-critical { 
                    border-left: 4px solid var(--danger-color); 
                    background-color: rgba(239, 68, 68, 0.05);
                }
                
                .alert-warning { 
                    border-left: 4px solid var(--warning-color); 
                    background-color: rgba(245, 158, 11, 0.05);
                }
                
                .alert-info { 
                    border-left: 4px solid var(--info-color); 
                    background-color: rgba(6, 182, 212, 0.05);
                }

                .navbar-brand {
                    font-weight: bold;
                    color: var(--primary-color) !important;
                }

                .card-header {
                    background-color: #f8fafc;
                    border-bottom: 1px solid var(--border-color);
                    font-weight: 600;
                }

                .status-indicator {
                    display: inline-block;
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    margin-right: 8px;
                }

                .status-online {
                    background-color: var(--success-color);
                }

                .status-offline {
                    background-color: var(--danger-color);
                }

                .loading-spinner {
                    display: none;
                    text-align: center;
                    padding: 20px;
                }

                .symbol-badge {
                    font-size: 0.8em;
                    padding: 2px 6px;
                    border-radius: 4px;
                    background-color: var(--secondary-color);
                    color: white;
                }

                #realtime-prices {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 15px;
                }

                .price-card {
                    flex: 1;
                    min-width: 200px;
                    padding: 15px;
                    border: 1px solid var(--border-color);
                    border-radius: 6px;
                    background-color: var(--card-bg);
                }

                .price-symbol {
                    font-weight: bold;
                    font-size: 1.1em;
                    margin-bottom: 5px;
                }

                .price-value {
                    font-size: 1.3em;
                    font-weight: bold;
                }

                .price-change {
                    font-size: 0.9em;
                    margin-top: 5px;
                }

                .controls-panel {
                    background-color: #f8fafc;
                    padding: 15px;
                    border-radius: 6px;
                    margin-bottom: 20px;
                }

                .btn-custom {
                    margin-right: 10px;
                    margin-bottom: 5px;
                }

                .dark-theme {
                    --card-bg: #374151;
                    --dark-bg: #111827;
                    --border-color: #4b5563;
                    background-color: var(--dark-bg);
                    color: white;
                }

                .dark-theme .card {
                    background-color: var(--card-bg);
                    border-color: var(--border-color);
                    color: white;
                }

                .dark-theme .card-header {
                    background-color: #4b5563;
                    border-color: var(--border-color);
                }
            </style>
        </head>
        <body>
            <!-- ナビゲーションバー -->
            <nav class="navbar navbar-expand-lg navbar-light bg-light">
                <div class="container-fluid">
                    <a class="navbar-brand" href="#">
                        <i class="fas fa-chart-line"></i>
                        Enhanced Trading Dashboard
                    </a>
                    <div class="navbar-nav ms-auto">
                        <span class="nav-item">
                            <span class="status-indicator" id="connection-status"></span>
                            <span id="connection-text">接続中...</span>
                        </span>
                    </div>
                </div>
            </nav>

            <div class="container-fluid">
                <!-- 制御パネル -->
                <div class="row mt-3">
                    <div class="col-12">
                        <div class="controls-panel">
                            <div class="row align-items-center">
                                <div class="col-md-6">
                                    <label for="symbol-select" class="form-label">銘柄選択:</label>
                                    <select id="symbol-select" class="form-select" style="width: 200px; display: inline-block;">
                                        <option value="7203">7203 - トヨタ自動車</option>
                                        <option value="8306">8306 - 三菱UFJ銀行</option>
                                        <option value="9984">9984 - ソフトバンクグループ</option>
                                        <option value="6758">6758 - ソニー</option>
                                        <option value="4755">4755 - 楽天グループ</option>
                                    </select>
                                </div>
                                <div class="col-md-6 text-end">
                                    <button class="btn btn-primary btn-custom" id="refresh-btn">
                                        <i class="fas fa-sync-alt"></i> 更新
                                    </button>
                                    <button class="btn btn-secondary btn-custom" id="theme-toggle">
                                        <i class="fas fa-moon"></i> ダークモード
                                    </button>
                                    <button class="btn btn-success btn-custom" id="export-btn">
                                        <i class="fas fa-download"></i> エクスポート
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- リアルタイム価格表示 -->
                <div class="row">
                    <div class="col-12">
                        <div class="card dashboard-card">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <h5 class="mb-0">
                                    <i class="fas fa-chart-line"></i>
                                    リアルタイム価格
                                </h5>
                                <small class="text-muted" id="last-update">最終更新: --</small>
                            </div>
                            <div class="card-body">
                                <div id="realtime-prices">
                                    <!-- リアルタイム価格がここに表示される -->
                                    <div class="text-center text-muted">
                                        <i class="fas fa-spinner fa-spin"></i>
                                        データを読み込み中...
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- チャート表示エリア -->
                <div class="row">
                    <div class="col-lg-8">
                        <div class="card dashboard-card">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <h5 class="mb-0">
                                    <i class="fas fa-candlestick-chart"></i>
                                    価格チャート
                                </h5>
                                <div>
                                    <button class="btn btn-sm btn-outline-primary" id="candlestick-btn">ローソク足</button>
                                    <button class="btn btn-sm btn-outline-primary" id="line-btn">ライン</button>
                                    <button class="btn btn-sm btn-outline-primary" id="area-btn">エリア</button>
                                </div>
                            </div>
                            <div class="card-body">
                                <div class="loading-spinner" id="chart-loading">
                                    <i class="fas fa-spinner fa-spin"></i>
                                    チャートを読み込み中...
                                </div>
                                <div id="main-chart"></div>
                            </div>
                        </div>
                    </div>

                    <div class="col-lg-4">
                        <div class="card dashboard-card">
                            <div class="card-header">
                                <h5 class="mb-0">
                                    <i class="fas fa-bell"></i>
                                    アラート
                                </h5>
                            </div>
                            <div class="card-body" id="alerts-panel">
                                <div class="text-center text-muted">
                                    アラートはありません
                                </div>
                            </div>
                        </div>

                        <!-- 銘柄情報パネル -->
                        <div class="card dashboard-card">
                            <div class="card-header">
                                <h5 class="mb-0">
                                    <i class="fas fa-info-circle"></i>
                                    銘柄情報
                                </h5>
                            </div>
                            <div class="card-body" id="symbol-info">
                                <div class="text-center text-muted">
                                    銘柄を選択してください
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 予測とパフォーマンス -->
                <div class="row">
                    <div class="col-lg-6">
                        <div class="card dashboard-card">
                            <div class="card-header">
                                <h5 class="mb-0">
                                    <i class="fas fa-brain"></i>
                                    ML予測
                                </h5>
                            </div>
                            <div class="card-body">
                                <div id="prediction-chart"></div>
                            </div>
                        </div>
                    </div>

                    <div class="col-lg-6">
                        <div class="card dashboard-card">
                            <div class="card-header">
                                <h5 class="mb-0">
                                    <i class="fas fa-tachometer-alt"></i>
                                    システムパフォーマンス
                                </h5>
                            </div>
                            <div class="card-body">
                                <div id="performance-chart"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 統計情報 -->
                <div class="row">
                    <div class="col-12">
                        <div class="card dashboard-card">
                            <div class="card-header">
                                <h5 class="mb-0">
                                    <i class="fas fa-chart-bar"></i>
                                    統計情報
                                </h5>
                            </div>
                            <div class="card-body">
                                <div class="row" id="statistics-panel">
                                    <div class="col-md-3 text-center">
                                        <h4 class="text-primary" id="active-symbols">0</h4>
                                        <small class="text-muted">アクティブ銘柄</small>
                                    </div>
                                    <div class="col-md-3 text-center">
                                        <h4 class="text-success" id="total-alerts">0</h4>
                                        <small class="text-muted">総アラート数</small>
                                    </div>
                                    <div class="col-md-3 text-center">
                                        <h4 class="text-info" id="prediction-accuracy">0%</h4>
                                        <small class="text-muted">予測精度</small>
                                    </div>
                                    <div class="col-md-3 text-center">
                                        <h4 class="text-warning" id="update-frequency">--</h4>
                                        <small class="text-muted">更新頻度</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- フッター -->
            <footer class="bg-light text-center text-muted py-3 mt-5">
                <small>&copy; 2024 Enhanced Trading Dashboard. All rights reserved.</small>
            </footer>

            <script>
                // ダッシュボードJavaScriptコード
                {javascript_code}
            </script>
        </body>
        </html>
        """

    @staticmethod
    def get_javascript_code() -> str:
        """ダッシュボードのJavaScriptコード"""
        return """
        // グローバル変数
        let socket;
        let currentSymbol = '7203';
        let isDarkTheme = false;
        let chartUpdateInterval;

        // Socket.IO接続
        function initializeSocket() {
            socket = io();

            socket.on('connect', function() {
                console.log('サーバーに接続しました');
                updateConnectionStatus(true);
            });

            socket.on('disconnect', function() {
                console.log('サーバーから切断されました');
                updateConnectionStatus(false);
            });

            // リアルタイム価格更新
            socket.on('price_update', function(data) {
                updateRealtimePrice(data);
            });

            // アラート表示
            socket.on('alert_triggered', function(alert) {
                showAlert(alert);
            });

            // チャートデータ受信
            socket.on('chart_data', function(data) {
                updateChart(data);
            });

            // 分析結果受信
            socket.on('analysis_result', function(data) {
                displayAnalysisResult(data);
            });

            // システムメッセージ受信
            socket.on('system_message', function(data) {
                showSystemMessage(data.message, data.type);
            });
        }

        // 接続ステータス更新
        function updateConnectionStatus(isConnected) {
            const indicator = document.getElementById('connection-status');
            const text = document.getElementById('connection-text');
            
            if (isConnected) {
                indicator.className = 'status-indicator status-online';
                text.textContent = '接続済み';
            } else {
                indicator.className = 'status-indicator status-offline';
                text.textContent = '切断中';
            }
        }

        // リアルタイム価格更新
        function updateRealtimePrice(data) {
            const pricesContainer = document.getElementById('realtime-prices');
            
            // 初回の場合、既存の読み込み表示を削除
            if (pricesContainer.innerHTML.includes('データを読み込み中')) {
                pricesContainer.innerHTML = '';
            }

            let priceCard = document.getElementById('price-' + data.symbol);
            if (!priceCard) {
                priceCard = createPriceCard(data.symbol);
                pricesContainer.appendChild(priceCard);
            }

            updatePriceCardData(priceCard, data);
            updateLastUpdateTime();
        }

        // 価格カード作成
        function createPriceCard(symbol) {
            const card = document.createElement('div');
            card.className = 'price-card';
            card.id = 'price-' + symbol;
            
            card.innerHTML = `
                <div class="price-symbol">
                    <span class="symbol-badge">${symbol}</span>
                </div>
                <div class="price-value" id="price-value-${symbol}">--</div>
                <div class="price-change" id="price-change-${symbol}">--</div>
            `;
            
            return card;
        }

        // 価格カードデータ更新
        function updatePriceCardData(card, data) {
            const valueElement = card.querySelector(`#price-value-${data.symbol}`);
            const changeElement = card.querySelector(`#price-change-${data.symbol}`);
            
            valueElement.textContent = '¥' + data.price.toFixed(2);
            
            const changeText = `${data.change >= 0 ? '+' : ''}${data.change.toFixed(2)} (${data.change_percent >= 0 ? '+' : ''}${data.change_percent.toFixed(2)}%)`;
            changeElement.innerHTML = `<span class="${data.change >= 0 ? 'price-up' : 'price-down'}">${changeText}</span>`;
        }

        // 最終更新時刻更新
        function updateLastUpdateTime() {
            const now = new Date();
            const timeString = now.toLocaleTimeString('ja-JP');
            document.getElementById('last-update').textContent = `最終更新: ${timeString}`;
        }

        // アラート表示
        function showAlert(alert) {
            const alertsPanel = document.getElementById('alerts-panel');
            
            // 「アラートはありません」メッセージを削除
            if (alertsPanel.innerHTML.includes('アラートはありません')) {
                alertsPanel.innerHTML = '';
            }

            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${alert.severity} alert-dismissible fade show`;
            alertDiv.innerHTML = `
                <strong>${alert.symbol}</strong> ${alert.message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                <small class="d-block mt-1 text-muted">${new Date(alert.timestamp).toLocaleTimeString('ja-JP')}</small>
            `;
            
            alertsPanel.insertBefore(alertDiv, alertsPanel.firstChild);
            
            // 古いアラートを制限
            const alerts = alertsPanel.querySelectorAll('.alert');
            if (alerts.length > 5) {
                alerts[alerts.length - 1].remove();
            }
        }

        // システムメッセージ表示
        function showSystemMessage(message, type) {
            // トースト通知的な表示（簡易実装）
            const toast = document.createElement('div');
            toast.className = `alert alert-${type} position-fixed top-0 end-0 m-3`;
            toast.style.zIndex = '9999';
            toast.innerHTML = `
                ${message}
                <button type="button" class="btn-close" onclick="this.parentElement.remove()"></button>
            `;
            
            document.body.appendChild(toast);
            
            setTimeout(() => {
                if (toast.parentElement) {
                    toast.remove();
                }
            }, 5000);
        }

        // チャート読み込み
        function loadChart(symbol, chartType = 'candlestick') {
            const chartContainer = document.getElementById('main-chart');
            const loading = document.getElementById('chart-loading');
            
            loading.style.display = 'block';
            chartContainer.innerHTML = '';

            fetch(`/api/chart/${symbol}?type=${chartType}&indicators=RSI,MACD,SMA_20,SMA_50`)
                .then(response => response.json())
                .then(data => {
                    loading.style.display = 'none';
                    if (data.success) {
                        Plotly.newPlot('main-chart', data.chart);
                    } else {
                        chartContainer.innerHTML = '<div class="alert alert-danger">チャート読み込みエラー: ' + data.error + '</div>';
                    }
                })
                .catch(error => {
                    loading.style.display = 'none';
                    console.error('チャート読み込みエラー:', error);
                    chartContainer.innerHTML = '<div class="alert alert-danger">チャート読み込みに失敗しました</div>';
                });
        }

        // 予測チャート読み込み
        function loadPredictionChart(symbol) {
            fetch(`/api/prediction/${symbol}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.chart) {
                        Plotly.newPlot('prediction-chart', data.chart);
                    }
                })
                .catch(error => {
                    console.error('予測チャート読み込みエラー:', error);
                });
        }

        // パフォーマンスダッシュボード読み込み
        function loadPerformanceDashboard() {
            fetch('/api/performance')
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.chart) {
                        Plotly.newPlot('performance-chart', data.chart);
                    }
                })
                .catch(error => {
                    console.error('パフォーマンスダッシュボード読み込みエラー:', error);
                });
        }

        // テーマ切り替え
        function toggleTheme() {
            isDarkTheme = !isDarkTheme;
            const body = document.body;
            const button = document.getElementById('theme-toggle');
            
            if (isDarkTheme) {
                body.classList.add('dark-theme');
                button.innerHTML = '<i class="fas fa-sun"></i> ライトモード';
            } else {
                body.classList.remove('dark-theme');
                button.innerHTML = '<i class="fas fa-moon"></i> ダークモード';
            }
        }

        // 統計情報更新
        function updateStatistics() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('active-symbols').textContent = data.active_subscriptions || 0;
                    document.getElementById('update-frequency').textContent = data.config?.update_frequency || '--';
                })
                .catch(error => {
                    console.error('統計情報取得エラー:', error);
                });
        }

        // イベントリスナー設定
        document.addEventListener('DOMContentLoaded', function() {
            // Socket.IO初期化
            initializeSocket();

            // 銘柄選択変更
            document.getElementById('symbol-select').addEventListener('change', function() {
                currentSymbol = this.value;
                loadChart(currentSymbol);
                loadPredictionChart(currentSymbol);
                socket.emit('subscribe', {symbol: currentSymbol});
            });

            // ボタンイベント
            document.getElementById('refresh-btn').addEventListener('click', function() {
                loadChart(currentSymbol);
                loadPredictionChart(currentSymbol);
                loadPerformanceDashboard();
            });

            document.getElementById('theme-toggle').addEventListener('click', toggleTheme);

            document.getElementById('candlestick-btn').addEventListener('click', function() {
                loadChart(currentSymbol, 'candlestick');
            });

            document.getElementById('line-btn').addEventListener('click', function() {
                loadChart(currentSymbol, 'line');
            });

            document.getElementById('area-btn').addEventListener('click', function() {
                loadChart(currentSymbol, 'area');
            });

            // 初期データ読み込み
            setTimeout(() => {
                // デフォルト銘柄を購読
                const defaultSymbols = ['7203', '8306', '9984', '6758', '4755'];
                defaultSymbols.forEach(symbol => {
                    socket.emit('subscribe', {symbol: symbol});
                });

                // 初期チャート読み込み
                loadChart(currentSymbol);
                loadPredictionChart(currentSymbol);
                loadPerformanceDashboard();

                // 統計情報定期更新
                updateStatistics();
                setInterval(updateStatistics, 30000);
            }, 1000);
        });
        """

    @classmethod
    def render_dashboard(cls, **kwargs) -> str:
        """ダッシュボードのレンダリング"""
        template = cls.get_main_dashboard_template()
        javascript_code = cls.get_javascript_code()
        
        # JavaScriptコードを挿入
        return template.format(javascript_code=javascript_code)

    @staticmethod
    def get_mobile_optimized_css() -> str:
        """モバイル最適化CSS"""
        return """
        @media (max-width: 768px) {
            .container-fluid {
                padding: 10px;
            }
            
            .dashboard-card {
                margin-bottom: 15px;
            }
            
            .controls-panel {
                padding: 10px;
            }
            
            .btn-custom {
                margin-bottom: 10px;
                width: 100%;
            }
            
            #realtime-prices {
                flex-direction: column;
            }
            
            .price-card {
                min-width: 100%;
            }
            
            .navbar-brand {
                font-size: 1rem;
            }
        }
        """