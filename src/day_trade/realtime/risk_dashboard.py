#!/usr/bin/env python3
"""
生成AI統合リスク管理ダッシュボード
Generative AI Risk Management Dashboard

リアルタイム監視・アラート・分析レポート生成
"""

import asyncio
import json
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

# Web フレームワーク
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# 可視化
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.utils import PlotlyJSONEncoder
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# プロジェクト内インポート
from ..utils.logging_config import get_context_logger
from ..risk.generative_ai_engine import GenerativeAIRiskEngine, RiskAnalysisRequest, RiskAnalysisResult
from ..risk.fraud_detection_engine import FraudDetectionEngine, FraudDetectionRequest
from .alert_system import AlertManager, Alert

logger = get_context_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

class RiskDashboardManager:
    """生成AI統合リスク管理ダッシュボード"""

    def __init__(self, port: int = 8080):
        self.port = port

        # FastAPI アプリケーション
        self.app = FastAPI(
            title="Generative AI Risk Management Dashboard",
            description="生成AI統合リスク管理システム監視ダッシュボード",
            version="2.0.0"
        )

        # WebSocket接続管理
        self.active_connections: List[WebSocket] = []

        # リスク管理コンポーネント
        self.risk_engine = GenerativeAIRiskEngine()
        self.fraud_engine = FraudDetectionEngine()
        self.alert_manager = AlertManager()

        # ダッシュボードデータ
        self.dashboard_data = {
            'system_status': 'initializing',
            'last_update': datetime.now().isoformat(),
            'risk_metrics': {},
            'fraud_metrics': {},
            'active_alerts': [],
            'analysis_history': []
        }

        # 統計データ
        self.metrics_history: List[Dict[str, Any]] = []
        self.risk_analysis_log: List[RiskAnalysisResult] = []
        self.fraud_detection_log: List[Dict[str, Any]] = []

        # 更新タスク
        self.update_task: Optional[asyncio.Task] = None
        self.is_running = False

        # ルート設定
        self._setup_routes()

        logger.info("リスク管理ダッシュボード初期化完了")

    def _setup_routes(self):
        """APIルート設定"""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            return await self._render_risk_dashboard()

        @self.app.get("/api/risk/status")
        async def get_risk_status():
            """リスクシステム状況"""
            return await self._get_risk_system_status()

        @self.app.get("/api/risk/analysis/{analysis_id}")
        async def get_risk_analysis(analysis_id: str):
            """特定リスク分析結果取得"""
            return await self._get_risk_analysis(analysis_id)

        @self.app.post("/api/risk/analyze")
        async def create_risk_analysis(request: dict):
            """新規リスク分析実行"""
            return await self._execute_risk_analysis(request)

        @self.app.get("/api/fraud/metrics")
        async def get_fraud_metrics():
            """不正検知メトリクス"""
            return await self._get_fraud_metrics()

        @self.app.get("/api/charts/risk-timeline")
        async def risk_timeline_chart():
            """リスクスコア時系列チャート"""
            return await self._generate_risk_timeline_chart()

        @self.app.get("/api/charts/fraud-heatmap")
        async def fraud_heatmap():
            """不正検知ヒートマップ"""
            return await self._generate_fraud_heatmap()

        @self.app.get("/api/charts/ai-performance")
        async def ai_performance_chart():
            """AI性能チャート"""
            return await self._generate_ai_performance_chart()

        @self.app.get("/api/alerts/active")
        async def get_active_alerts():
            """アクティブアラート一覧"""
            return await self._get_active_alerts()

        @self.app.post("/api/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str):
            """アラート承認"""
            return await self._acknowledge_alert(alert_id)

        @self.app.get("/api/reports/daily")
        async def daily_report():
            """日次レポート生成"""
            return await self._generate_daily_report()

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self._handle_websocket(websocket)

    async def _render_risk_dashboard(self) -> HTMLResponse:
        """リスク管理ダッシュボードHTML生成"""

        html_content = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 生成AI統合リスク管理ダッシュボード</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/axios/dist/axios.min.js"></script>
    <style>
        .risk-critical { background: linear-gradient(145deg, #fee2e2, #fca5a5); color: #991b1b; }
        .risk-high { background: linear-gradient(145deg, #fef3c7, #fbbf24); color: #92400e; }
        .risk-medium { background: linear-gradient(145deg, #dbeafe, #60a5fa); color: #1e3a8a; }
        .risk-low { background: linear-gradient(145deg, #dcfce7, #4ade80); color: #166534; }
        .metric-card {
            background: white;
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border: 1px solid #e5e7eb;
        }
        .ai-indicator { display: inline-flex; align-items: center; gap: 0.5rem; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; }
        .status-healthy { background: #10b981; }
        .status-warning { background: #f59e0b; }
        .status-error { background: #ef4444; }
        .chart-container { height: 350px; margin: 1rem 0; }
    </style>
</head>
<body class="bg-gray-50">
    <div class="container mx-auto px-6 py-8">

        <!-- ヘッダー -->
        <header class="mb-8">
            <div class="bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg p-6">
                <h1 class="text-4xl font-bold mb-2">
                    🤖 生成AI統合リスク管理ダッシュボード
                </h1>
                <p class="text-lg opacity-90">GPT-4/Claude統合 × 深層学習 × リアルタイム監視</p>
                <div class="mt-4 flex items-center space-x-6">
                    <div id="systemStatus" class="ai-indicator">
                        <span class="status-dot status-healthy"></span>
                        <span>システム稼働中</span>
                    </div>
                    <div id="lastUpdate" class="opacity-75">
                        最終更新: 読み込み中...
                    </div>
                </div>
            </div>
        </header>

        <!-- KPI メトリクス -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">

            <div class="metric-card">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-500 text-sm">総リスク分析</p>
                        <p id="totalAnalyses" class="text-3xl font-bold text-indigo-600">--</p>
                    </div>
                    <div class="text-4xl">📊</div>
                </div>
            </div>

            <div class="metric-card">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-500 text-sm">不正検知件数</p>
                        <p id="fraudDetections" class="text-3xl font-bold text-red-600">--</p>
                    </div>
                    <div class="text-4xl">🚨</div>
                </div>
            </div>

            <div class="metric-card">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-500 text-sm">AI信頼度</p>
                        <p id="aiConfidence" class="text-3xl font-bold text-green-600">--%</p>
                    </div>
                    <div class="text-4xl">🎯</div>
                </div>
            </div>

            <div class="metric-card">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-500 text-sm">処理時間</p>
                        <p id="avgProcessingTime" class="text-3xl font-bold text-blue-600">--ms</p>
                    </div>
                    <div class="text-4xl">⚡</div>
                </div>
            </div>

        </div>

        <!-- リアルタイムチャート -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">

            <!-- リスクスコア時系列 -->
            <div class="metric-card">
                <h3 class="text-xl font-semibold mb-4">📈 リスクスコア推移</h3>
                <div id="riskTimelineChart" class="chart-container"></div>
            </div>

            <!-- AI性能モニター -->
            <div class="metric-card">
                <h3 class="text-xl font-semibold mb-4">🧠 AI性能モニター</h3>
                <div id="aiPerformanceChart" class="chart-container"></div>
            </div>

        </div>

        <!-- 不正検知ヒートマップ -->
        <div class="metric-card mb-8">
            <h3 class="text-xl font-semibold mb-4">🔥 不正検知ヒートマップ</h3>
            <div id="fraudHeatmapChart" class="chart-container" style="height: 400px;"></div>
        </div>

        <!-- アクティブアラート -->
        <div class="metric-card mb-8">
            <h3 class="text-xl font-semibold mb-4">🚨 アクティブアラート</h3>
            <div id="activeAlerts">
                <div class="text-center py-8 text-gray-500">
                    アラート読み込み中...
                </div>
            </div>
        </div>

        <!-- 最新分析結果 -->
        <div class="metric-card mb-8">
            <h3 class="text-xl font-semibold mb-4">🔍 最新リスク分析</h3>
            <div id="recentAnalyses" class="space-y-4">
                <div class="text-center py-8 text-gray-500">
                    分析結果読み込み中...
                </div>
            </div>
        </div>

        <!-- AI モデル状況 -->
        <div class="metric-card">
            <h3 class="text-xl font-semibold mb-4">🤖 AIモデル状況</h3>
            <div id="aiModelStatus" class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <!-- 動的生成 -->
            </div>
        </div>

    </div>

    <script>
        // WebSocket接続
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws`);

        // 接続状態管理
        ws.onopen = () => {
            console.log('WebSocket connected');
            updateConnectionStatus(true);
        };

        ws.onclose = () => {
            console.log('WebSocket disconnected');
            updateConnectionStatus(false);
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateConnectionStatus(false);
        };

        // メッセージ処理
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };

        // 接続状態更新
        function updateConnectionStatus(connected) {
            const statusElement = document.getElementById('systemStatus');
            const dot = statusElement.querySelector('.status-dot');
            const text = statusElement.querySelector('span:last-child');

            if (connected) {
                dot.className = 'status-dot status-healthy';
                text.textContent = 'システム稼働中';
            } else {
                dot.className = 'status-dot status-error';
                text.textContent = '接続エラー';
            }
        }

        // ダッシュボード更新
        function updateDashboard(data) {
            // KPIメトリクス更新
            updateKPIMetrics(data);

            // チャート更新
            updateCharts(data);

            // アラート更新
            updateAlerts(data.alerts || []);

            // 最新分析結果更新
            updateRecentAnalyses(data.recent_analyses || []);

            // AIモデル状況更新
            updateAIModelStatus(data.ai_models || {});

            // 最終更新時刻
            document.getElementById('lastUpdate').textContent =
                `最終更新: ${new Date(data.timestamp || Date.now()).toLocaleTimeString()}`;
        }

        function updateKPIMetrics(data) {
            const metrics = data.metrics || {};

            document.getElementById('totalAnalyses').textContent =
                metrics.total_analyses || 0;
            document.getElementById('fraudDetections').textContent =
                metrics.fraud_detections || 0;
            document.getElementById('aiConfidence').textContent =
                Math.round((metrics.ai_confidence || 0) * 100) + '%';
            document.getElementById('avgProcessingTime').textContent =
                Math.round(metrics.avg_processing_time || 0) + 'ms';
        }

        function updateCharts(data) {
            // チャートデータがある場合のみ更新
            if (data.charts) {
                updateRiskTimelineChart(data.charts.risk_timeline);
                updateAIPerformanceChart(data.charts.ai_performance);
                updateFraudHeatmap(data.charts.fraud_heatmap);
            }
        }

        function updateRiskTimelineChart(chartData) {
            if (!chartData) return;

            const trace = {
                x: chartData.timestamps,
                y: chartData.risk_scores,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'リスクスコア',
                line: { color: '#ef4444', width: 3 },
                marker: { size: 6, color: '#dc2626' }
            };

            const layout = {
                title: 'リスクスコア推移',
                xaxis: { title: '時刻' },
                yaxis: { title: 'リスクスコア', range: [0, 1] },
                showlegend: false,
                margin: { l: 50, r: 20, t: 40, b: 40 }
            };

            Plotly.newPlot('riskTimelineChart', [trace], layout, {responsive: true});
        }

        function updateAIPerformanceChart(chartData) {
            if (!chartData) return;

            const data = [
                {
                    values: [chartData.gpt4_calls, chartData.claude_calls, chartData.heuristic_calls],
                    labels: ['GPT-4', 'Claude', 'ヒューリスティック'],
                    type: 'pie',
                    marker: {
                        colors: ['#3b82f6', '#8b5cf6', '#10b981']
                    }
                }
            ];

            const layout = {
                title: 'AI モデル使用状況',
                showlegend: true,
                margin: { l: 20, r: 20, t: 40, b: 20 }
            };

            Plotly.newPlot('aiPerformanceChart', data, layout, {responsive: true});
        }

        function updateFraudHeatmap(chartData) {
            if (!chartData) return;

            const data = [{
                z: chartData.data,
                x: chartData.hours,
                y: chartData.days,
                type: 'heatmap',
                colorscale: 'Reds',
                hoverimplate: '時刻: %{x}時<br>曜日: %{y}<br>検知件数: %{z}<extra></extra>'
            }];

            const layout = {
                title: '不正検知頻度 (時間帯別)',
                xaxis: { title: '時刻' },
                yaxis: { title: '曜日' },
                margin: { l: 60, r: 20, t: 40, b: 40 }
            };

            Plotly.newPlot('fraudHeatmapChart', data, layout, {responsive: true});
        }

        function updateAlerts(alerts) {
            const container = document.getElementById('activeAlerts');

            if (alerts.length === 0) {
                container.innerHTML = `
                    <div class="text-center py-8 text-green-600">
                        ✅ 現在アクティブなアラートはありません
                    </div>
                `;
                return;
            }

            const alertsHTML = alerts.map(alert => `
                <div class="risk-${alert.level} rounded-lg p-4 mb-3">
                    <div class="flex justify-between items-start">
                        <div>
                            <h4 class="font-bold">${alert.title}</h4>
                            <p class="mt-1">${alert.message}</p>
                            <p class="text-sm opacity-75 mt-2">
                                ${new Date(alert.timestamp).toLocaleString()}
                            </p>
                        </div>
                        <button
                            onclick="acknowledgeAlert('${alert.id}')"
                            class="bg-white bg-opacity-20 hover:bg-opacity-30 px-3 py-1 rounded text-sm transition-colors"
                        >
                            承認
                        </button>
                    </div>
                </div>
            `).join('');

            container.innerHTML = alertsHTML;
        }

        function updateRecentAnalyses(analyses) {
            const container = document.getElementById('recentAnalyses');

            if (analyses.length === 0) {
                container.innerHTML = `
                    <div class="text-center py-8 text-gray-500">
                        最近の分析結果はありません
                    </div>
                `;
                return;
            }

            const analysesHTML = analyses.map(analysis => `
                <div class="border rounded-lg p-4 bg-gray-50">
                    <div class="flex justify-between items-start mb-2">
                        <span class="font-semibold">分析ID: ${analysis.id}</span>
                        <span class="risk-${analysis.risk_level} px-2 py-1 rounded text-sm">
                            ${analysis.risk_level.toUpperCase()}
                        </span>
                    </div>
                    <p class="text-sm text-gray-600 mb-2">${analysis.explanation}</p>
                    <div class="flex justify-between text-xs text-gray-500">
                        <span>リスクスコア: ${analysis.risk_score.toFixed(3)}</span>
                        <span>${new Date(analysis.timestamp).toLocaleString()}</span>
                    </div>
                </div>
            `).join('');

            container.innerHTML = analysesHTML;
        }

        function updateAIModelStatus(models) {
            const container = document.getElementById('aiModelStatus');

            const modelsHTML = Object.entries(models).map(([name, status]) => `
                <div class="text-center p-3 bg-gray-50 rounded-lg">
                    <div class="ai-indicator justify-center mb-2">
                        <span class="status-dot status-${status.status}"></span>
                        <span class="font-semibold">${name}</span>
                    </div>
                    <div class="text-sm text-gray-600">
                        呼び出し: ${status.calls || 0}<br>
                        成功率: ${Math.round((status.success_rate || 0) * 100)}%
                    </div>
                </div>
            `).join('');

            container.innerHTML = modelsHTML;
        }

        // アラート承認
        async function acknowledgeAlert(alertId) {
            try {
                const response = await axios.post(`/api/alerts/${alertId}/acknowledge`);
                if (response.data.success) {
                    console.log('Alert acknowledged:', alertId);
                    // 即座にUIを更新
                    const alertElement = document.querySelector(`[onclick="acknowledgeAlert('${alertId}')"]`);
                    if (alertElement) {
                        alertElement.textContent = '承認済み';
                        alertElement.disabled = true;
                        alertElement.className = alertElement.className.replace('hover:bg-opacity-30', '');
                    }
                }
            } catch (error) {
                console.error('Alert acknowledgment failed:', error);
                alert('アラート承認に失敗しました');
            }
        }

        // 初期化
        document.addEventListener('DOMContentLoaded', () => {
            // 初期データ取得
            fetchDashboardData();

            // 定期更新（WebSocketの補完）
            setInterval(fetchDashboardData, 10000); // 10秒間隔
        });

        async function fetchDashboardData() {
            try {
                const response = await axios.get('/api/risk/status');
                updateDashboard(response.data);
            } catch (error) {
                console.error('Failed to fetch dashboard data:', error);
            }
        }

    </script>
</body>
</html>
        """

        return HTMLResponse(content=html_content)

    async def _handle_websocket(self, websocket: WebSocket):
        """WebSocket接続処理"""

        await websocket.accept()
        self.active_connections.append(websocket)

        logger.info(f"WebSocket接続: {len(self.active_connections)} 接続中")

        try:
            while True:
                # リアルタイムデータ送信
                dashboard_data = await self._collect_realtime_data()
                await websocket.send_json(dashboard_data)

                # 1秒間隔で更新
                await asyncio.sleep(1)

        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket切断: {len(self.active_connections)} 接続中")

    async def _collect_realtime_data(self) -> Dict[str, Any]:
        """リアルタイムデータ収集"""

        # リスクエンジン統計
        risk_stats = self.risk_engine.get_performance_stats()
        fraud_stats = self.fraud_engine.get_stats()

        # サンプルデータ生成（実際の実装では実データを使用）
        current_time = datetime.now()
        risk_score = np.random.beta(2, 5)  # 0-1の範囲でリスクスコア

        return {
            'timestamp': current_time.isoformat(),
            'metrics': {
                'total_analyses': risk_stats.get('total_analyses', 0),
                'fraud_detections': fraud_stats.get('fraud_detected', 0),
                'ai_confidence': risk_stats.get('success_rate', 0.85),
                'avg_processing_time': risk_stats.get('avg_processing_time', 0.5) * 1000
            },
            'charts': {
                'risk_timeline': {
                    'timestamps': [(current_time - timedelta(minutes=i)).isoformat()
                                 for i in range(30, 0, -1)],
                    'risk_scores': [np.random.beta(2, 5) for _ in range(30)]
                },
                'ai_performance': {
                    'gpt4_calls': risk_stats.get('gpt4_calls', 0),
                    'claude_calls': risk_stats.get('claude_calls', 0),
                    'heuristic_calls': max(0, risk_stats.get('total_analyses', 0) -
                                         risk_stats.get('gpt4_calls', 0) -
                                         risk_stats.get('claude_calls', 0))
                },
                'fraud_heatmap': {
                    'data': np.random.poisson(2, (7, 24)).tolist(),
                    'hours': list(range(24)),
                    'days': ['月', '火', '水', '木', '金', '土', '日']
                }
            },
            'alerts': await self._get_current_alerts(),
            'recent_analyses': await self._get_recent_analyses(5),
            'ai_models': {
                'GPT-4': {
                    'status': 'healthy' if risk_stats.get('models_available', {}).get('gpt4') else 'error',
                    'calls': risk_stats.get('gpt4_calls', 0),
                    'success_rate': 0.95
                },
                'Claude': {
                    'status': 'healthy' if risk_stats.get('models_available', {}).get('claude') else 'error',
                    'calls': risk_stats.get('claude_calls', 0),
                    'success_rate': 0.93
                },
                'LSTM': {
                    'status': 'healthy' if fraud_stats.get('models_loaded') else 'warning',
                    'calls': fraud_stats.get('total_detections', 0),
                    'success_rate': 0.96
                },
                'Transformer': {
                    'status': 'healthy' if fraud_stats.get('models_loaded') else 'warning',
                    'calls': fraud_stats.get('total_detections', 0),
                    'success_rate': 0.92
                }
            }
        }

    async def _get_current_alerts(self) -> List[Dict[str, Any]]:
        """現在のアクティブアラート取得"""

        # サンプルアラート（実際の実装では alert_manager から取得）
        sample_alerts = [
            {
                'id': 'ALERT_001',
                'title': '高リスク取引検知',
                'message': '異常な取引パターンを検知しました',
                'level': 'high',
                'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat()
            },
            {
                'id': 'ALERT_002',
                'title': 'AI信頼度低下',
                'message': 'モデル予測精度が閾値を下回りました',
                'level': 'medium',
                'timestamp': (datetime.now() - timedelta(minutes=15)).isoformat()
            }
        ]

        return sample_alerts

    async def _get_recent_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """最近のリスク分析結果取得"""

        # サンプル分析結果（実際の実装では分析ログから取得）
        sample_analyses = []
        for i in range(limit):
            risk_score = np.random.beta(2, 5)
            risk_level = 'low' if risk_score < 0.3 else 'medium' if risk_score < 0.7 else 'high'

            sample_analyses.append({
                'id': f'ANALYSIS_{i+1:03d}',
                'risk_score': risk_score,
                'risk_level': risk_level,
                'explanation': f'リスク分析#{i+1}: {risk_level}レベルのリスクを検出',
                'timestamp': (datetime.now() - timedelta(minutes=i*2)).isoformat()
            })

        return sample_analyses

    async def _get_risk_system_status(self) -> Dict[str, Any]:
        """リスクシステム状況取得"""
        return await self._collect_realtime_data()

    async def run_dashboard(self):
        """ダッシュボード起動"""

        logger.info(f"生成AIリスク管理ダッシュボード起動: ポート{self.port}")
        logger.info("URL: http://localhost:{}/".format(self.port))

        self.is_running = True

        # バックグラウンドでサンプルアラート生成
        async def generate_test_alerts():
            while self.is_running:
                await asyncio.sleep(60)  # 1分ごと
                # テストアラート生成ロジック
                pass

        # uvicorn サーバー設定
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)

        try:
            # サーバー起動
            await asyncio.gather(
                server.serve(),
                generate_test_alerts()
            )
        except Exception as e:
            logger.error(f"ダッシュボードエラー: {e}")
        finally:
            self.is_running = False

# 使用例・テスト
async def test_risk_dashboard():
    """リスクダッシュボードテスト"""

    print("🖥️ 生成AI統合リスク管理ダッシュボード起動中...")
    print("📊 URL: http://localhost:8080")
    print("🤖 GPT-4/Claude統合 + 深層学習監視")
    print("⚡ リアルタイム更新・アラート対応")

    dashboard = RiskDashboardManager(port=8080)
    await dashboard.run_dashboard()

if __name__ == "__main__":
    asyncio.run(test_risk_dashboard())
