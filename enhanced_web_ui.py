#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Web UI - 高度なWebユーザーインターフェース
Issue #936対応: リアルタイムダッシュボード + レスポンシブデザイン
"""

from flask import Flask, render_template_string, jsonify, request, Response
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading

# 統合モジュール
try:
    from advanced_ai_engine import advanced_ai_engine
    HAS_AI_ENGINE = True
except ImportError:
    HAS_AI_ENGINE = False

try:
    from realtime_streaming import streaming_engine
    HAS_STREAMING = True
except ImportError:
    HAS_STREAMING = False

try:
    from performance_monitor import performance_monitor
    HAS_PERFORMANCE_MONITOR = True
except ImportError:
    HAS_PERFORMANCE_MONITOR = False

try:
    from data_persistence import data_persistence
    HAS_DATA_PERSISTENCE = True
except ImportError:
    HAS_DATA_PERSISTENCE = False

try:
    from version import get_version_info, __version_full__
    VERSION_INFO = get_version_info()
except ImportError:
    VERSION_INFO = {"version": "2.1.0", "release_name": "Extended"}
    __version_full__ = "Day Trade Personal v2.1.0 Extended"


class EnhancedWebUI:
    """高度なWebユーザーインターフェース"""

    def __init__(self, port: int = 8080, debug: bool = False):
        self.port = port
        self.debug = debug
        self.app = Flask(__name__)
        self.app.secret_key = 'daytrade-enhanced-ui-2025'

        # UIテーマ設定
        self.theme_config = {
            'primary_color': '#2563eb',
            'success_color': '#10b981',
            'warning_color': '#f59e0b',
            'danger_color': '#ef4444',
            'dark_mode': False
        }

        # ダッシュボード設定
        self.dashboard_config = {
            'refresh_interval': 5000,  # 5秒
            'max_chart_points': 50,
            'default_symbols': ['7203', '8306', '9984', '6758', '4689'],
            'auto_refresh': True
        }

        self._setup_routes()

    def _setup_routes(self):
        """ルート設定"""

        @self.app.route('/')
        def enhanced_dashboard():
            """拡張ダッシュボード"""
            return render_template_string(self._get_dashboard_template())

        @self.app.route('/realtime')
        def realtime_dashboard():
            """リアルタイムダッシュボード"""
            return render_template_string(self._get_realtime_template())

        @self.app.route('/analytics')
        def analytics_dashboard():
            """分析ダッシュボード"""
            return render_template_string(self._get_analytics_template())

        @self.app.route('/api/enhanced/dashboard-data')
        def api_dashboard_data():
            """ダッシュボードデータAPI"""
            return jsonify(self._get_dashboard_data())

        @self.app.route('/api/enhanced/market-overview')
        def api_market_overview():
            """市場概況API"""
            return jsonify(self._get_market_overview())

        @self.app.route('/api/enhanced/performance-metrics')
        def api_performance_metrics():
            """パフォーマンスメトリクスAPI"""
            return jsonify(self._get_performance_metrics())

        @self.app.route('/api/enhanced/ai-insights')
        def api_ai_insights():
            """AI洞察API"""
            return jsonify(self._get_ai_insights())

        @self.app.route('/api/enhanced/realtime-feed')
        def api_realtime_feed():
            """リアルタイムフィードAPI"""
            return Response(
                self._realtime_feed_generator(),
                mimetype='text/plain'
            )

        @self.app.route('/api/enhanced/symbol-analysis/<symbol>')
        def api_symbol_analysis(symbol):
            """個別銘柄分析API"""
            return jsonify(self._get_symbol_analysis(symbol))

        @self.app.route('/api/enhanced/market-alerts')
        def api_market_alerts():
            """市場アラートAPI"""
            return jsonify(self._get_market_alerts())

        @self.app.route('/api/enhanced/settings', methods=['GET', 'POST'])
        def api_settings():
            """設定API"""
            if request.method == 'POST':
                return jsonify(self._update_settings(request.get_json()))
            else:
                return jsonify(self._get_settings())

    def _get_dashboard_template(self) -> str:
        """ダッシュボードテンプレート"""
        return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Day Trade Personal - Enhanced Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        [x-cloak] { display: none !important; }
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .card-shadow { box-shadow: 0 10px 25px rgba(0,0,0,0.1); }
        .pulse-animation { animation: pulse 2s infinite; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body class="bg-gray-50 font-sans" x-data="dashboardApp()">
    <!-- Header -->
    <header class="gradient-bg text-white shadow-lg">
        <div class="container mx-auto px-6 py-4">
            <div class="flex items-center justify-between">
                <div>
                    <h1 class="text-3xl font-bold">📈 Day Trade Personal</h1>
                    <p class="text-blue-200">Enhanced AI Analytics Dashboard</p>
                </div>
                <div class="flex items-center space-x-4">
                    <div class="text-right">
                        <div class="text-sm opacity-90" x-text="currentTime"></div>
                        <div class="text-xs opacity-75">Version {{ VERSION_INFO.version }}</div>
                    </div>
                    <button @click="toggleTheme()" class="p-2 rounded-full bg-white bg-opacity-20 hover:bg-opacity-30 transition-all">
                        <i class="fas fa-adjust"></i>
                    </button>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Dashboard -->
    <main class="container mx-auto px-6 py-8">

        <!-- Quick Stats Row -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="bg-white rounded-xl p-6 card-shadow">
                <div class="flex items-center">
                    <div class="p-3 bg-green-100 rounded-full">
                        <i class="fas fa-chart-line text-green-600 text-xl"></i>
                    </div>
                    <div class="ml-4">
                        <h3 class="text-gray-500 text-sm">総推奨銘柄</h3>
                        <p class="text-2xl font-bold" x-text="stats.totalRecommendations">0</p>
                    </div>
                </div>
            </div>

            <div class="bg-white rounded-xl p-6 card-shadow">
                <div class="flex items-center">
                    <div class="p-3 bg-blue-100 rounded-full">
                        <i class="fas fa-bullseye text-blue-600 text-xl"></i>
                    </div>
                    <div class="ml-4">
                        <h3 class="text-gray-500 text-sm">高信頼度</h3>
                        <p class="text-2xl font-bold" x-text="stats.highConfidence">0</p>
                    </div>
                </div>
            </div>

            <div class="bg-white rounded-xl p-6 card-shadow">
                <div class="flex items-center">
                    <div class="p-3 bg-purple-100 rounded-full">
                        <i class="fas fa-robot text-purple-600 text-xl"></i>
                    </div>
                    <div class="ml-4">
                        <h3 class="text-gray-500 text-sm">AI分析精度</h3>
                        <p class="text-2xl font-bold">93%</p>
                    </div>
                </div>
            </div>

            <div class="bg-white rounded-xl p-6 card-shadow">
                <div class="flex items-center">
                    <div class="p-3 bg-yellow-100 rounded-full">
                        <i class="fas fa-clock text-yellow-600 text-xl"></i>
                    </div>
                    <div class="ml-4">
                        <h3 class="text-gray-500 text-sm">平均分析時間</h3>
                        <p class="text-2xl font-bold" x-text="stats.avgAnalysisTime + 'ms'">0ms</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <!-- Market Trend Chart -->
            <div class="bg-white rounded-xl p-6 card-shadow">
                <h3 class="text-lg font-semibold mb-4 flex items-center">
                    <i class="fas fa-chart-area text-blue-500 mr-2"></i>
                    市場トレンド
                </h3>
                <canvas id="marketTrendChart" width="400" height="200"></canvas>
            </div>

            <!-- Confidence Distribution -->
            <div class="bg-white rounded-xl p-6 card-shadow">
                <h3 class="text-lg font-semibold mb-4 flex items-center">
                    <i class="fas fa-pie-chart text-green-500 mr-2"></i>
                    信頼度分布
                </h3>
                <canvas id="confidenceChart" width="400" height="200"></canvas>
            </div>
        </div>

        <!-- Recommendations Table -->
        <div class="bg-white rounded-xl card-shadow overflow-hidden">
            <div class="px-6 py-4 border-b border-gray-200">
                <h3 class="text-lg font-semibold flex items-center">
                    <i class="fas fa-list text-purple-500 mr-2"></i>
                    AI推奨銘柄
                    <span class="ml-2 px-2 py-1 bg-purple-100 text-purple-800 text-xs rounded-full">
                        リアルタイム更新
                    </span>
                </h3>
            </div>

            <div class="overflow-x-auto">
                <table class="w-full">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">銘柄</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">推奨</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">信頼度</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">価格</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">変動率</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">リスク</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
                        <template x-for="rec in recommendations" :key="rec.symbol">
                            <tr class="hover:bg-gray-50 transition-colors">
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <div>
                                        <div class="text-sm font-medium text-gray-900" x-text="rec.name"></div>
                                        <div class="text-sm text-gray-500" x-text="rec.symbol"></div>
                                    </div>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <span class="px-2 py-1 text-xs font-semibold rounded-full"
                                          :class="getRecommendationClass(rec.recommendation)"
                                          x-text="rec.recommendation">
                                    </span>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <div class="flex items-center">
                                        <div class="text-sm text-gray-900" x-text="(rec.confidence * 100).toFixed(1) + '%'"></div>
                                        <div class="ml-2 w-16 bg-gray-200 rounded-full h-2">
                                            <div class="bg-blue-600 h-2 rounded-full"
                                                 :style="`width: ${rec.confidence * 100}%`"></div>
                                        </div>
                                    </div>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900" x-text="'¥' + rec.price.toLocaleString()"></td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <span class="text-sm font-medium"
                                          :class="rec.change >= 0 ? 'text-green-600' : 'text-red-600'"
                                          x-text="(rec.change >= 0 ? '+' : '') + rec.change.toFixed(2) + '%'">
                                    </span>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <span class="px-2 py-1 text-xs font-semibold rounded-full"
                                          :class="getRiskClass(rec.risk_level)"
                                          x-text="rec.risk_level">
                                    </span>
                                </td>
                            </tr>
                        </template>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Footer -->
        <footer class="mt-12 text-center text-gray-500">
            <p>{{ __version_full__ }} - Powered by Advanced AI Analytics</p>
            <p class="text-sm mt-2">最終更新: <span x-text="lastUpdate"></span></p>
        </footer>
    </main>

    <script>
        function dashboardApp() {
            return {
                currentTime: new Date().toLocaleString('ja-JP'),
                lastUpdate: new Date().toLocaleString('ja-JP'),
                darkMode: false,
                stats: {
                    totalRecommendations: 0,
                    highConfidence: 0,
                    avgAnalysisTime: 0
                },
                recommendations: [],
                marketTrendChart: null,
                confidenceChart: null,

                init() {
                    this.loadDashboardData();
                    this.initCharts();
                    this.startDataRefresh();
                    this.updateTime();
                },

                async loadDashboardData() {
                    try {
                        const response = await fetch('/api/enhanced/dashboard-data');
                        const data = await response.json();

                        this.stats = data.stats;
                        this.recommendations = data.recommendations || [];
                        this.lastUpdate = new Date().toLocaleString('ja-JP');

                        this.updateCharts(data);
                    } catch (error) {
                        console.error('Failed to load dashboard data:', error);
                    }
                },

                initCharts() {
                    // Market Trend Chart
                    const trendCtx = document.getElementById('marketTrendChart');
                    this.marketTrendChart = new Chart(trendCtx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: '市場トレンド指数',
                                data: [],
                                borderColor: '#3b82f6',
                                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                tension: 0.4,
                                fill: true
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                legend: {
                                    display: false
                                }
                            },
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    grid: {
                                        color: 'rgba(0,0,0,0.1)'
                                    }
                                },
                                x: {
                                    grid: {
                                        color: 'rgba(0,0,0,0.1)'
                                    }
                                }
                            }
                        }
                    });

                    // Confidence Chart
                    const confCtx = document.getElementById('confidenceChart');
                    this.confidenceChart = new Chart(confCtx, {
                        type: 'doughnut',
                        data: {
                            labels: ['高信頼度 (>80%)', '中信頼度 (60-80%)', '低信頼度 (<60%)'],
                            datasets: [{
                                data: [0, 0, 0],
                                backgroundColor: ['#10b981', '#f59e0b', '#ef4444']
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                legend: {
                                    position: 'bottom'
                                }
                            }
                        }
                    });
                },

                updateCharts(data) {
                    if (data.chartData) {
                        // Update trend chart
                        if (data.chartData.trendLabels && data.chartData.trendData) {
                            this.marketTrendChart.data.labels = data.chartData.trendLabels;
                            this.marketTrendChart.data.datasets[0].data = data.chartData.trendData;
                            this.marketTrendChart.update();
                        }

                        // Update confidence chart
                        if (data.chartData.confidenceDistribution) {
                            this.confidenceChart.data.datasets[0].data = data.chartData.confidenceDistribution;
                            this.confidenceChart.update();
                        }
                    }
                },

                startDataRefresh() {
                    setInterval(() => {
                        this.loadDashboardData();
                    }, {{ dashboard_config.refresh_interval }});
                },

                updateTime() {
                    setInterval(() => {
                        this.currentTime = new Date().toLocaleString('ja-JP');
                    }, 1000);
                },

                toggleTheme() {
                    this.darkMode = !this.darkMode;
                    document.documentElement.classList.toggle('dark', this.darkMode);
                },

                getRecommendationClass(recommendation) {
                    const classes = {
                        'BUY': 'bg-green-100 text-green-800',
                        'SELL': 'bg-red-100 text-red-800',
                        'HOLD': 'bg-yellow-100 text-yellow-800'
                    };
                    return classes[recommendation] || 'bg-gray-100 text-gray-800';
                },

                getRiskClass(riskLevel) {
                    const classes = {
                        'LOW': 'bg-green-100 text-green-800',
                        'MEDIUM': 'bg-yellow-100 text-yellow-800',
                        'HIGH': 'bg-red-100 text-red-800'
                    };
                    return classes[riskLevel] || 'bg-gray-100 text-gray-800';
                }
            }
        }
    </script>
</body>
</html>
        """.replace('{{ VERSION_INFO.version }}', VERSION_INFO.get('version', '2.1.0')) \
           .replace('{{ __version_full__ }}', __version_full__) \
           .replace('{{ dashboard_config.refresh_interval }}', str(self.dashboard_config['refresh_interval']))

    def _get_realtime_template(self) -> str:
        """リアルタイムテンプレート"""
        return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Day Trade Personal - Real-time Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-gray-900 text-white" x-data="realtimeApp()">
    <div class="container mx-auto px-6 py-8">
        <h1 class="text-4xl font-bold mb-8 text-center">
            🔴 LIVE Market Analysis
        </h1>

        <!-- WebSocket Status -->
        <div class="mb-6 text-center">
            <span class="px-3 py-1 rounded-full text-sm"
                  :class="wsConnected ? 'bg-green-600' : 'bg-red-600'">
                <i class="fas" :class="wsConnected ? 'fa-wifi' : 'fa-wifi-slash'"></i>
                <span x-text="wsConnected ? 'リアルタイム接続中' : '接続待機中'"></span>
            </span>
        </div>

        <!-- Live Feed -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-xl font-semibold mb-4">📈 ライブ価格フィード</h3>
                <div class="space-y-2 max-h-96 overflow-y-auto">
                    <template x-for="update in priceUpdates.slice().reverse()" :key="update.id">
                        <div class="flex justify-between items-center p-2 rounded bg-gray-700"
                             :class="update.change >= 0 ? 'border-l-4 border-green-500' : 'border-l-4 border-red-500'">
                            <div>
                                <span class="font-bold" x-text="update.symbol"></span>
                                <span class="text-gray-400 ml-2" x-text="update.time"></span>
                            </div>
                            <div class="text-right">
                                <div x-text="'¥' + update.price.toLocaleString()"></div>
                                <div class="text-sm"
                                     :class="update.change >= 0 ? 'text-green-400' : 'text-red-400'"
                                     x-text="(update.change >= 0 ? '+' : '') + update.change.toFixed(2) + '%'">
                                </div>
                            </div>
                        </div>
                    </template>
                </div>
            </div>

            <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-xl font-semibold mb-4">🤖 AI分析アラート</h3>
                <div class="space-y-2 max-h-96 overflow-y-auto">
                    <template x-for="alert in analysisAlerts.slice().reverse()" :key="alert.id">
                        <div class="p-3 rounded bg-gray-700 border-l-4"
                             :class="getAlertColor(alert.type)">
                            <div class="flex justify-between items-start">
                                <div>
                                    <div class="font-bold" x-text="alert.symbol + ' - ' + alert.signal"></div>
                                    <div class="text-sm text-gray-400" x-text="alert.reason"></div>
                                </div>
                                <div class="text-xs text-gray-500" x-text="alert.time"></div>
                            </div>
                            <div class="mt-2">
                                <span class="text-xs bg-blue-600 px-2 py-1 rounded">
                                    信頼度: <span x-text="(alert.confidence * 100).toFixed(0) + '%'"></span>
                                </span>
                            </div>
                        </div>
                    </template>
                </div>
            </div>
        </div>
    </div>

    <script>
        function realtimeApp() {
            return {
                wsConnected: false,
                ws: null,
                priceUpdates: [],
                analysisAlerts: [],

                init() {
                    this.connectWebSocket();
                },

                connectWebSocket() {
                    try {
                        this.ws = new WebSocket('ws://localhost:8765');

                        this.ws.onopen = () => {
                            this.wsConnected = true;
                            console.log('WebSocket connected');

                            // Subscribe to all symbols
                            this.ws.send(JSON.stringify({
                                type: 'subscribe',
                                symbols: ['7203', '8306', '9984', '6758', '4689'],
                                message_types: ['market_data', 'analysis']
                            }));
                        };

                        this.ws.onmessage = (event) => {
                            const data = JSON.parse(event.data);
                            this.handleWebSocketMessage(data);
                        };

                        this.ws.onclose = () => {
                            this.wsConnected = false;
                            console.log('WebSocket disconnected');
                            setTimeout(() => this.connectWebSocket(), 5000);
                        };

                        this.ws.onerror = (error) => {
                            console.error('WebSocket error:', error);
                        };
                    } catch (error) {
                        console.error('Failed to connect WebSocket:', error);
                    }
                },

                handleWebSocketMessage(data) {
                    if (data.type === 'market_data') {
                        this.priceUpdates.push({
                            id: Date.now() + Math.random(),
                            symbol: data.symbol,
                            price: data.data.price,
                            change: data.data.change_percent,
                            time: new Date(data.timestamp).toLocaleTimeString('ja-JP')
                        });

                        if (this.priceUpdates.length > 50) {
                            this.priceUpdates.shift();
                        }
                    } else if (data.type === 'analysis') {
                        this.analysisAlerts.push({
                            id: Date.now() + Math.random(),
                            symbol: data.symbol,
                            signal: data.data.signal_type,
                            confidence: data.data.confidence,
                            reason: data.data.reasons[0] || 'AI analysis',
                            type: data.data.signal_type,
                            time: new Date(data.timestamp).toLocaleTimeString('ja-JP')
                        });

                        if (this.analysisAlerts.length > 50) {
                            this.analysisAlerts.shift();
                        }
                    }
                },

                getAlertColor(type) {
                    const colors = {
                        'BUY': 'border-green-500',
                        'SELL': 'border-red-500',
                        'HOLD': 'border-yellow-500'
                    };
                    return colors[type] || 'border-gray-500';
                }
            }
        }
    </script>
</body>
</html>
        """

    def _get_analytics_template(self) -> str:
        """分析テンプレート（簡略版）"""
        return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Day Trade Personal - Analytics</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto px-6 py-8">
        <h1 class="text-3xl font-bold mb-8">📊 Advanced Analytics</h1>
        <div class="bg-white rounded-lg p-6 shadow">
            <p>高度な分析機能は開発中です...</p>
        </div>
    </div>
</body>
</html>
        """

    def _get_dashboard_data(self) -> Dict[str, Any]:
        """ダッシュボードデータ取得"""
        # 模擬データ生成
        import random

        recommendations = []
        symbols_data = [
            {'symbol': '7203', 'name': 'トヨタ自動車'},
            {'symbol': '8306', 'name': '三菱UFJ銀行'},
            {'symbol': '9984', 'name': 'ソフトバンクグループ'},
            {'symbol': '6758', 'name': 'ソニー'},
            {'symbol': '4689', 'name': 'Z Holdings'}
        ]

        high_confidence_count = 0
        total_analysis_time = 0

        for symbol_data in symbols_data:
            confidence = random.uniform(0.6, 0.95)
            recommendation = random.choice(['BUY', 'SELL', 'HOLD'])
            price = 1500 + hash(symbol_data['symbol']) % 1000
            change = random.uniform(-3.0, 3.0)
            risk_level = random.choice(['LOW', 'MEDIUM', 'HIGH'])

            if confidence > 0.8:
                high_confidence_count += 1

            recommendations.append({
                'symbol': symbol_data['symbol'],
                'name': symbol_data['name'],
                'recommendation': recommendation,
                'confidence': confidence,
                'price': price,
                'change': change,
                'risk_level': risk_level
            })

            total_analysis_time += random.uniform(100, 300)

        # チャートデータ
        chart_labels = [f"{i:02d}:00" for i in range(9, 16)]  # 9:00-15:00
        trend_data = [random.uniform(0.4, 0.8) for _ in chart_labels]

        confidence_distribution = [
            high_confidence_count,
            len(recommendations) - high_confidence_count - 1,
            1
        ]

        return {
            'stats': {
                'totalRecommendations': len(recommendations),
                'highConfidence': high_confidence_count,
                'avgAnalysisTime': round(total_analysis_time / len(recommendations))
            },
            'recommendations': recommendations,
            'chartData': {
                'trendLabels': chart_labels,
                'trendData': trend_data,
                'confidenceDistribution': confidence_distribution
            },
            'lastUpdate': datetime.now().isoformat()
        }

    def _get_market_overview(self) -> Dict[str, Any]:
        """市場概況取得"""
        return {
            'market_status': 'OPEN',
            'nikkei_index': 33850.45,
            'nikkei_change': +0.8,
            'volume_total': 1250000000,
            'advancing_stocks': 1420,
            'declining_stocks': 890,
            'unchanged_stocks': 180
        }

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンスメトリクス取得"""
        if HAS_PERFORMANCE_MONITOR and performance_monitor:
            return performance_monitor.get_performance_summary()
        else:
            return {
                'monitoring_enabled': False,
                'message': 'Performance monitoring not available'
            }

    def _get_ai_insights(self) -> Dict[str, Any]:
        """AI洞察取得"""
        if HAS_AI_ENGINE:
            return advanced_ai_engine.get_engine_statistics()
        else:
            return {
                'ai_engine_available': False,
                'message': 'AI engine not available'
            }

    def _realtime_feed_generator(self):
        """リアルタイムフィードジェネレーター"""
        import time
        import json

        while True:
            # 模擬リアルタイムデータ
            data = {
                'timestamp': datetime.now().isoformat(),
                'type': 'market_update',
                'data': {
                    'symbol': '7203',
                    'price': 1500 + (time.time() % 100),
                    'change': (time.time() % 10) - 5
                }
            }

            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(1)

    def _get_symbol_analysis(self, symbol: str) -> Dict[str, Any]:
        """個別銘柄分析取得"""
        if HAS_AI_ENGINE:
            try:
                signal = advanced_ai_engine.analyze_symbol(symbol)
                return {
                    'symbol': symbol,
                    'signal_type': signal.signal_type,
                    'confidence': signal.confidence,
                    'strength': signal.strength,
                    'risk_level': signal.risk_level,
                    'indicators': signal.indicators,
                    'reasons': signal.reasons,
                    'timestamp': signal.timestamp.isoformat()
                }
            except Exception as e:
                return {'error': str(e)}
        else:
            return {'error': 'AI engine not available'}

    def _get_market_alerts(self) -> Dict[str, Any]:
        """市場アラート取得"""
        return {
            'alerts': [
                {
                    'id': 1,
                    'type': 'HIGH_VOLUME',
                    'symbol': '7203',
                    'message': '異常な出来高増加を検出',
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'id': 2,
                    'type': 'PRICE_ALERT',
                    'symbol': '8306',
                    'message': '価格が重要なサポートラインを突破',
                    'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat()
                }
            ]
        }

    def _get_settings(self) -> Dict[str, Any]:
        """設定取得"""
        return {
            'theme': self.theme_config,
            'dashboard': self.dashboard_config
        }

    def _update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """設定更新"""
        if 'theme' in settings:
            self.theme_config.update(settings['theme'])

        if 'dashboard' in settings:
            self.dashboard_config.update(settings['dashboard'])

        return {'status': 'success', 'message': 'Settings updated'}

    def run(self, host: str = '0.0.0.0'):
        """Webサーバー起動"""
        self.app.run(host=host, port=self.port, debug=self.debug)


# グローバルインスタンス
enhanced_web_ui = EnhancedWebUI()


if __name__ == "__main__":
    print(f"Enhanced Web UI starting on port {enhanced_web_ui.port}")
    print(f"Access: http://localhost:{enhanced_web_ui.port}")
    enhanced_web_ui.run()