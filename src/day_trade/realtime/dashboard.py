#!/usr/bin/env python3
"""
Next-Gen AI Trading Engine - リアルタイムWebダッシュボード
統合監視・制御Webインターフェース

AI予測・市場データ・システム状況のリアルタイム可視化
"""

import asyncio
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn

# Web フレームワーク
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# プロジェクト内インポート
from ..utils.logging_config import get_context_logger
from .alert_system import AlertManager
from .live_prediction_engine import LivePredictionEngine
from .performance_monitor import RealTimePerformanceMonitor
from .websocket_stream import RealTimeStreamManager

logger = get_context_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class DashboardManager:
    """ダッシュボード管理システム"""

    def __init__(self):
        # FastAPI アプリケーション
        self.app = FastAPI(title="Next-Gen AI Trading Dashboard", version="1.0")

        # WebSocket接続管理
        self.active_connections: List[WebSocket] = []

        # システムコンポーネント（外部から注入）
        self.prediction_engine: Optional[LivePredictionEngine] = None
        self.performance_monitor: Optional[RealTimePerformanceMonitor] = None
        self.alert_manager: Optional[AlertManager] = None
        self.stream_manager: Optional[RealTimeStreamManager] = None

        # ダッシュボードデータ
        self.dashboard_data = {
            "system_status": "Starting",
            "last_update": datetime.now().isoformat(),
            "active_predictions": [],
            "performance_metrics": {},
            "recent_alerts": [],
            "market_data": {},
        }

        # 更新タスク
        self.update_task: Optional[asyncio.Task] = None
        self.is_running = False

        # ルート設定
        self._setup_routes()
        self._setup_static_files()

        logger.info("Dashboard Manager initialized")

    def _setup_routes(self):
        """ルート設定"""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """メインダッシュボード"""
            return await self._render_dashboard(request)

        @self.app.get("/api/status")
        async def get_system_status():
            """システム状況API"""
            return JSONResponse(self._get_system_status())

        @self.app.get("/api/predictions")
        async def get_predictions():
            """予測結果API"""
            return JSONResponse(self._get_recent_predictions())

        @self.app.get("/api/performance")
        async def get_performance():
            """パフォーマンス指標API"""
            return JSONResponse(self._get_performance_metrics())

        @self.app.get("/api/alerts")
        async def get_alerts():
            """アラート一覧API"""
            return JSONResponse(self._get_recent_alerts())

        @self.app.get("/api/market")
        async def get_market_data():
            """市場データAPI"""
            return JSONResponse(self._get_market_data())

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket エンドポイント"""
            await self._handle_websocket(websocket)

        @self.app.post("/api/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str):
            """アラート確認API"""
            if self.alert_manager:
                success = self.alert_manager.acknowledge_alert(alert_id)
                return {"success": success, "alert_id": alert_id}
            return {"success": False, "error": "Alert manager not available"}

        @self.app.post("/api/system/restart")
        async def restart_system():
            """システム再起動API"""
            try:
                await self._restart_components()
                return {"success": True, "message": "System restart initiated"}
            except Exception as e:
                return {"success": False, "error": str(e)}

    def _setup_static_files(self):
        """静的ファイル設定"""

        # 静的ファイルディレクトリ作成
        static_dir = Path("static")
        static_dir.mkdir(exist_ok=True)

        # テンプレートディレクトリ作成
        templates_dir = Path("templates")
        templates_dir.mkdir(exist_ok=True)

        # Jinja2 テンプレート設定
        self.templates = Jinja2Templates(directory="templates")

        # 静的ファイルマウント
        self.app.mount("/static", StaticFiles(directory="static"), name="static")

    async def _render_dashboard(self, request: Request) -> HTMLResponse:
        """ダッシュボードHTML描画"""

        dashboard_html = """
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Next-Gen AI Trading Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                .status-healthy { color: #10B981; }
                .status-warning { color: #F59E0B; }
                .status-critical { color: #EF4444; }
                .metric-card {
                    background: linear-gradient(145deg, #f0f9ff, #e0f2fe);
                    border: 1px solid #e5e7eb;
                    border-radius: 0.5rem;
                    padding: 1.5rem;
                    margin-bottom: 1rem;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                }
                .prediction-card {
                    background: linear-gradient(145deg, #fef3c7, #fbbf24);
                    color: #92400e;
                }
                .alert-card {
                    background: linear-gradient(145deg, #fee2e2, #fca5a5);
                    color: #991b1b;
                }
                .chart-container {
                    position: relative;
                    height: 300px;
                    margin: 1rem 0;
                }
            </style>
        </head>
        <body class="bg-gray-100">
            <div class="container mx-auto px-4 py-6">

                <!-- ヘッダー -->
                <header class="mb-8">
                    <h1 class="text-4xl font-bold text-gray-800 mb-2">
                        🤖 Next-Gen AI Trading Dashboard
                    </h1>
                    <div class="flex items-center space-x-4">
                        <div id="systemStatus" class="status-healthy">
                            ● System Active
                        </div>
                        <div id="lastUpdate" class="text-gray-600">
                            Last Update: Loading...
                        </div>
                    </div>
                </header>

                <!-- メインダッシュボード -->
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">

                    <!-- システム状況 -->
                    <div class="metric-card">
                        <h3 class="text-xl font-semibold mb-4">📊 System Status</h3>
                        <div id="systemMetrics">
                            <div class="mb-2">CPU: <span id="cpuUsage">--</span>%</div>
                            <div class="mb-2">Memory: <span id="memoryUsage">--</span>%</div>
                            <div class="mb-2">Active Tasks: <span id="activeTasks">--</span></div>
                        </div>
                    </div>

                    <!-- AI 予測 -->
                    <div class="prediction-card metric-card">
                        <h3 class="text-xl font-semibold mb-4">🎯 AI Predictions</h3>
                        <div id="predictionMetrics">
                            <div class="mb-2">Total: <span id="totalPredictions">--</span></div>
                            <div class="mb-2">Success Rate: <span id="successRate">--</span>%</div>
                            <div class="mb-2">Avg Latency: <span id="avgLatency">--</span>ms</div>
                        </div>
                    </div>

                    <!-- アラート -->
                    <div class="alert-card metric-card">
                        <h3 class="text-xl font-semibold mb-4">🚨 Active Alerts</h3>
                        <div id="alertMetrics">
                            <div class="mb-2">Active: <span id="activeAlerts">--</span></div>
                            <div class="mb-2">Unread: <span id="unreadAlerts">--</span></div>
                            <div class="mb-2">Today: <span id="todayAlerts">--</span></div>
                        </div>
                    </div>

                </div>

                <!-- チャートセクション -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">

                    <!-- 予測パフォーマンス チャート -->
                    <div class="metric-card">
                        <h3 class="text-xl font-semibold mb-4">📈 Prediction Performance</h3>
                        <div class="chart-container">
                            <canvas id="performanceChart"></canvas>
                        </div>
                    </div>

                    <!-- システムメトリクス チャート -->
                    <div class="metric-card">
                        <h3 class="text-xl font-semibold mb-4">⚡ System Metrics</h3>
                        <div class="chart-container">
                            <canvas id="systemChart"></canvas>
                        </div>
                    </div>

                </div>

                <!-- 最近の予測結果 -->
                <div class="metric-card mt-8">
                    <h3 class="text-xl font-semibold mb-4">🔮 Recent Predictions</h3>
                    <div id="recentPredictions" class="overflow-x-auto">
                        <table class="min-w-full table-auto">
                            <thead>
                                <tr class="bg-gray-50">
                                    <th class="px-4 py-2">Time</th>
                                    <th class="px-4 py-2">Symbol</th>
                                    <th class="px-4 py-2">Action</th>
                                    <th class="px-4 py-2">Confidence</th>
                                    <th class="px-4 py-2">Target Price</th>
                                </tr>
                            </thead>
                            <tbody id="predictionsTableBody">
                                <tr><td colspan="5" class="text-center py-4">Loading...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- アラート履歴 -->
                <div class="metric-card mt-8">
                    <h3 class="text-xl font-semibold mb-4">🔔 Recent Alerts</h3>
                    <div id="recentAlertsContainer">
                        Loading alerts...
                    </div>
                </div>

            </div>

            <script>
                // WebSocket 接続
                const ws = new WebSocket(`ws://${window.location.host}/ws`);

                // Chart.js 設定
                let performanceChart, systemChart;

                // WebSocket メッセージ処理
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };

                ws.onopen = function() {
                    console.log('WebSocket connected');
                };

                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                    document.getElementById('systemStatus').className = 'status-critical';
                    document.getElementById('systemStatus').textContent = '● Connection Error';
                };

                // ダッシュボード更新
                function updateDashboard(data) {
                    // システム状況更新
                    updateSystemStatus(data.system || {});

                    // AI予測更新
                    updatePredictions(data.ai || {});

                    // アラート更新
                    updateAlerts(data.alerts || []);

                    // 最終更新時刻
                    document.getElementById('lastUpdate').textContent =
                        `Last Update: ${new Date(data.timestamp).toLocaleTimeString()}`;

                    // チャート更新
                    updateCharts(data);
                }

                function updateSystemStatus(system) {
                    document.getElementById('cpuUsage').textContent =
                        Math.round(system.cpu_percent || 0);
                    document.getElementById('memoryUsage').textContent =
                        Math.round(system.memory_percent || 0);
                    document.getElementById('activeTasks').textContent =
                        system.active_tasks || 0;

                    // ステータス色更新
                    const status = system.status || 'unknown';
                    const statusElement = document.getElementById('systemStatus');
                    statusElement.className = `status-${status}`;
                    statusElement.textContent = `● System ${status.charAt(0).toUpperCase() + status.slice(1)}`;
                }

                function updatePredictions(ai) {
                    document.getElementById('totalPredictions').textContent =
                        ai.total_predictions || 0;
                    document.getElementById('successRate').textContent =
                        Math.round((ai.success_rate || 0) * 100);
                    document.getElementById('avgLatency').textContent =
                        Math.round(ai.average_latency || 0);
                }

                function updateAlerts(alerts) {
                    const activeCount = alerts.filter(a => !a.resolved).length;
                    const unreadCount = alerts.filter(a => !a.acknowledged).length;
                    const todayCount = alerts.filter(a => {
                        const alertDate = new Date(a.timestamp);
                        const today = new Date();
                        return alertDate.toDateString() === today.toDateString();
                    }).length;

                    document.getElementById('activeAlerts').textContent = activeCount;
                    document.getElementById('unreadAlerts').textContent = unreadCount;
                    document.getElementById('todayAlerts').textContent = todayCount;
                }

                function updateCharts(data) {
                    // チャート更新実装（簡略化）
                    // 実際の実装ではリアルタイムデータを使用
                }

                // 初期化
                document.addEventListener('DOMContentLoaded', function() {
                    // Chart.js初期化
                    initializeCharts();

                    // 定期的なデータ取得
                    setInterval(fetchDashboardData, 5000);

                    // 初回データ取得
                    fetchDashboardData();
                });

                function initializeCharts() {
                    // パフォーマンス チャート
                    const perfCtx = document.getElementById('performanceChart').getContext('2d');
                    performanceChart = new Chart(perfCtx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Success Rate',
                                data: [],
                                borderColor: '#10B981',
                                backgroundColor: 'rgba(16, 185, 129, 0.1)'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: { beginAtZero: true, max: 100 }
                            }
                        }
                    });

                    // システムメトリクス チャート
                    const sysCtx = document.getElementById('systemChart').getContext('2d');
                    systemChart = new Chart(sysCtx, {
                        type: 'doughnut',
                        data: {
                            labels: ['CPU', 'Memory', 'Available'],
                            datasets: [{
                                data: [0, 0, 100],
                                backgroundColor: ['#F59E0B', '#EF4444', '#10B981']
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false
                        }
                    });
                }

                async function fetchDashboardData() {
                    try {
                        const response = await fetch('/api/status');
                        const data = await response.json();
                        updateDashboard(data);
                    } catch (error) {
                        console.error('Failed to fetch dashboard data:', error);
                    }
                }

            </script>
        </body>
        </html>
        """

        return HTMLResponse(content=dashboard_html)

    async def _handle_websocket(self, websocket: WebSocket):
        """WebSocket接続処理"""

        await websocket.accept()
        self.active_connections.append(websocket)

        logger.info(
            f"WebSocket connected. Active connections: {len(self.active_connections)}"
        )

        try:
            while True:
                # クライアントからのメッセージ待機
                data = await websocket.receive_text()

                # メッセージ処理（必要に応じて）
                logger.debug(f"Received WebSocket message: {data}")

        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
            logger.info(
                f"WebSocket disconnected. Active connections: {len(self.active_connections)}"
            )

    async def broadcast_update(self, data: Dict[str, Any]):
        """WebSocket ブロードキャスト"""

        if not self.active_connections:
            return

        message = json.dumps(data)

        # 無効な接続を削除するためのリスト
        invalid_connections = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")
                invalid_connections.append(connection)

        # 無効な接続を削除
        for connection in invalid_connections:
            if connection in self.active_connections:
                self.active_connections.remove(connection)

    def inject_components(
        self,
        prediction_engine: Optional[LivePredictionEngine] = None,
        performance_monitor: Optional[RealTimePerformanceMonitor] = None,
        alert_manager: Optional[AlertManager] = None,
        stream_manager: Optional[RealTimeStreamManager] = None,
    ):
        """システムコンポーネント注入"""

        self.prediction_engine = prediction_engine
        self.performance_monitor = performance_monitor
        self.alert_manager = alert_manager
        self.stream_manager = stream_manager

        logger.info("System components injected into dashboard")

    async def start_dashboard(self, host: str = "0.0.0.0", port: int = 8000):
        """ダッシュボード開始"""

        self.is_running = True

        # データ更新タスク開始
        self.update_task = asyncio.create_task(self._update_loop())

        logger.info(f"Starting dashboard server on {host}:{port}")

        # uvicorn サーバー起動
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)

        try:
            await server.serve()
        except Exception as e:
            logger.error(f"Dashboard server error: {e}")
        finally:
            await self.stop_dashboard()

    async def stop_dashboard(self):
        """ダッシュボード停止"""

        self.is_running = False

        # 更新タスク停止
        if self.update_task and not self.update_task.done():
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass

        # WebSocket接続終了
        for connection in self.active_connections.copy():
            try:
                await connection.close()
            except Exception:
                pass

        self.active_connections.clear()
        logger.info("Dashboard stopped")

    async def _update_loop(self):
        """データ更新ループ"""

        logger.info("Dashboard update loop started")

        while self.is_running:
            try:
                # ダッシュボードデータ収集
                updated_data = await self._collect_dashboard_data()

                # WebSocket ブロードキャスト
                if updated_data:
                    await self.broadcast_update(updated_data)

                # 5秒間隔で更新
                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
                await asyncio.sleep(1)

    async def _collect_dashboard_data(self) -> Dict[str, Any]:
        """ダッシュボードデータ収集"""

        data = {
            "timestamp": datetime.now().isoformat(),
            "system": {},
            "ai": {},
            "trading": {},
            "alerts": [],
        }

        try:
            # システム状況
            if self.performance_monitor:
                comprehensive_status = (
                    self.performance_monitor.get_comprehensive_status()
                )
                system_summary = comprehensive_status.get("system_summary", {})
                data["system"] = {
                    "cpu_percent": system_summary.get("cpu", {}).get("current", 0),
                    "memory_percent": system_summary.get("memory", {}).get(
                        "current", 0
                    ),
                    "status": (
                        "healthy"
                        if system_summary.get("cpu", {}).get("current", 0) < 70
                        else "warning"
                    ),
                    "active_tasks": comprehensive_status.get(
                        "monitoring_stats", {}
                    ).get("total_monitoring_cycles", 0),
                }

                # AI性能
                ai_summary = comprehensive_status.get("ai_summary", {})
                data["ai"] = {
                    "total_predictions": ai_summary.get("total_predictions", 0),
                    "success_rate": ai_summary.get("success_rate", 0),
                    "average_latency": ai_summary.get("average_latency_ms", 0),
                    "status": (
                        "healthy"
                        if ai_summary.get("error_rate", 0) < 0.1
                        else "warning"
                    ),
                }

                # 取引パフォーマンス
                trading_summary = comprehensive_status.get("trading_summary", {})
                data["trading"] = {
                    "total_signals": trading_summary.get("total_signals", 0),
                    "virtual_return": trading_summary.get("virtual_return", 0),
                    "virtual_drawdown": trading_summary.get("virtual_drawdown", 0),
                    "status": (
                        "healthy"
                        if trading_summary.get("virtual_drawdown", 0) < 0.05
                        else "warning"
                    ),
                }

            # アラート情報
            if self.alert_manager:
                recent_alerts = self.alert_manager.get_alert_history(hours=24)
                data["alerts"] = [
                    {
                        "id": alert.id,
                        "timestamp": alert.timestamp.isoformat(),
                        "level": alert.level.value,
                        "title": alert.title,
                        "message": alert.message,
                        "acknowledged": alert.acknowledged,
                        "resolved": alert.resolved,
                    }
                    for alert in recent_alerts[-10:]  # 最新10件
                ]

        except Exception as e:
            logger.error(f"Dashboard data collection error: {e}")

        return data

    def _get_system_status(self) -> Dict[str, Any]:
        """システム状況取得"""

        if self.performance_monitor:
            return self.performance_monitor.get_comprehensive_status()

        return {"status": "unknown", "message": "Performance monitor not available"}

    def _get_recent_predictions(self) -> List[Dict]:
        """最近の予測取得"""

        # 簡略化実装（実際のLivePredictionEngineから取得）
        return []

    def _get_performance_metrics(self) -> Dict:
        """パフォーマンス指標取得"""

        if self.performance_monitor:
            return self.performance_monitor.get_dashboard_data()

        return {}

    def _get_recent_alerts(self) -> List[Dict]:
        """最近のアラート取得"""

        if self.alert_manager:
            alerts = self.alert_manager.get_alert_history(hours=24)
            return [alert.to_dict() for alert in alerts[-20:]]  # 最新20件

        return []

    def _get_market_data(self) -> Dict:
        """市場データ取得"""

        if self.stream_manager:
            return self.stream_manager.get_latest_data()

        return {}

    async def _restart_components(self):
        """システムコンポーネント再起動"""

        logger.info("Restarting system components...")

        # 各コンポーネントの再起動ロジック実装
        # 実際の実装では各コンポーネントの再起動メソッドを呼び出し

        await asyncio.sleep(2)  # 模擬再起動時間

        logger.info("System components restarted")


# 便利関数
def create_dashboard_manager() -> DashboardManager:
    """ダッシュボード管理システム作成"""

    return DashboardManager()


async def start_dashboard_server(
    prediction_engine: Optional[LivePredictionEngine] = None,
    performance_monitor: Optional[RealTimePerformanceMonitor] = None,
    alert_manager: Optional[AlertManager] = None,
    stream_manager: Optional[RealTimeStreamManager] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """ダッシュボードサーバー起動"""

    dashboard = create_dashboard_manager()

    # システムコンポーネント注入
    dashboard.inject_components(
        prediction_engine=prediction_engine,
        performance_monitor=performance_monitor,
        alert_manager=alert_manager,
        stream_manager=stream_manager,
    )

    # サーバー起動
    await dashboard.start_dashboard(host=host, port=port)


if __name__ == "__main__":
    # ダッシュボードテスト
    async def test_dashboard():
        print("=== Dashboard System Test ===")

        try:
            # ダッシュボード作成
            dashboard = create_dashboard_manager()

            print("Starting dashboard server...")
            print("Open http://localhost:8000 in your browser")

            # サーバー起動（テスト用に短時間）
            await asyncio.wait_for(
                dashboard.start_dashboard(host="0.0.0.0", port=8000),
                timeout=30,  # 30秒でタイムアウト
            )

        except asyncio.TimeoutError:
            print("Dashboard test completed (timeout)")
        except Exception as e:
            print(f"Test error: {e}")
            import traceback

            traceback.print_exc()

    # テスト実行
    asyncio.run(test_dashboard())
