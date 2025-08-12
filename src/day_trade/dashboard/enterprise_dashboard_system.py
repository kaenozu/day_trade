#!/usr/bin/env python3
"""
エンタープライズ級可視化ダッシュボードシステム
Issue #332: エンタープライズ級完全統合システム - Phase 2

統合システムの包括的可視化・リアルタイム監視・インタラクティブ分析
- リアルタイムシステム状態監視
- エンタープライズ級可視化
- インタラクティブ分析インターフェース
- 包括的レポート生成
"""

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# 可視化ライブラリ
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import uvicorn

# Web フレームワーク
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..core.enterprise_integration_orchestrator import (
    EnterpriseIntegrationOrchestrator,
)
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class DashboardTheme(Enum):
    """ダッシュボードテーマ"""

    LIGHT = "light"
    DARK = "dark"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class VisualizationType(Enum):
    """可視化タイプ"""

    SYSTEM_STATUS = "system_status"
    REAL_TIME_DATA = "real_time_data"
    PERFORMANCE_METRICS = "performance_metrics"
    ANALYSIS_RESULTS = "analysis_results"
    COMPONENT_HEALTH = "component_health"
    DATA_QUALITY = "data_quality"


@dataclass
class DashboardConfig:
    """ダッシュボード設定"""

    # 基本設定
    theme: DashboardTheme = DashboardTheme.ENTERPRISE
    auto_refresh_interval_seconds: int = 10
    enable_real_time_updates: bool = True

    # サーバー設定
    host: str = "localhost"
    port: int = 8000
    debug: bool = False

    # 可視化設定
    chart_width: int = 1200
    chart_height: int = 600
    enable_interactive_charts: bool = True
    max_data_points: int = 1000

    # 監視設定
    enable_system_monitoring: bool = True
    enable_performance_tracking: bool = True
    enable_alert_notifications: bool = True

    # データ保持設定
    data_retention_hours: int = 24
    max_websocket_connections: int = 100


class EnterpriseDashboardSystem:
    """エンタープライズダッシュボードシステム"""

    def __init__(
        self,
        orchestrator: EnterpriseIntegrationOrchestrator,
        config: Optional[DashboardConfig] = None,
    ):
        self.orchestrator = orchestrator
        self.config = config or DashboardConfig()

        # FastAPI アプリケーション
        self.app = FastAPI(
            title="Day Trade Enterprise Dashboard",
            description="エンタープライズ級投資分析ダッシュボード",
            version="1.0.0",
        )

        # WebSocket接続管理
        self.websocket_connections: List[WebSocket] = []

        # データバッファ
        self.system_metrics_buffer: List[Dict[str, Any]] = []
        self.performance_data_buffer: List[Dict[str, Any]] = []
        self.analysis_results_buffer: Dict[str, List[Dict[str, Any]]] = {}

        # 可視化キャッシュ
        self.chart_cache: Dict[str, str] = {}
        self.last_cache_update = {}

        # 設定・初期化
        self._setup_routes()
        self._setup_static_files()

        logger.info("エンタープライズダッシュボードシステム初期化完了")

    def _setup_routes(self) -> None:
        """ルート設定"""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """ダッシュボードホーム"""
            return await self._render_dashboard_template("dashboard.html", request)

        @self.app.get("/api/system/overview")
        async def get_system_overview():
            """システム概要API"""
            return self.orchestrator.get_system_overview()

        @self.app.get("/api/components/details")
        async def get_component_details():
            """コンポーネント詳細API"""
            return self.orchestrator.get_component_details()

        @self.app.get("/api/analysis/{symbol}")
        async def get_symbol_analysis(symbol: str):
            """銘柄分析API"""
            analysis_report = await self.orchestrator.get_integrated_analysis_report(
                [symbol]
            )
            return analysis_report["analysis_results"].get(symbol, {})

        @self.app.get("/api/charts/system_status")
        async def get_system_status_chart():
            """システム状態チャート"""
            chart_html = await self._generate_system_status_chart()
            return {"chart_html": chart_html}

        @self.app.get("/api/charts/performance")
        async def get_performance_chart():
            """パフォーマンスチャート"""
            chart_html = await self._generate_performance_chart()
            return {"chart_html": chart_html}

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket エンドポイント"""
            await self._handle_websocket_connection(websocket)

    def _setup_static_files(self) -> None:
        """静的ファイル設定"""
        # 静的ファイル用ディレクトリ作成
        static_dir = Path(__file__).parent / "static"
        static_dir.mkdir(exist_ok=True)

        templates_dir = Path(__file__).parent / "templates"
        templates_dir.mkdir(exist_ok=True)

        # FastAPI 設定
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        self.templates = Jinja2Templates(directory=str(templates_dir))

    async def _render_dashboard_template(
        self, template_name: str, request: Request
    ) -> HTMLResponse:
        """ダッシュボードテンプレート描画"""

        # テンプレートが存在しない場合は動的生成
        template_path = Path(self.templates.directory) / template_name
        if not template_path.exists():
            await self._create_dashboard_template()

        context = {
            "request": request,
            "system_overview": self.orchestrator.get_system_overview(),
            "config": self.config,
            "theme": self.config.theme.value,
        }

        return self.templates.TemplateResponse(template_name, context)

    async def _create_dashboard_template(self) -> None:
        """ダッシュボードテンプレート作成"""
        template_content = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Day Trade Enterprise Dashboard</title>

    <!-- CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

    <style>
        .dashboard-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 1rem;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-running { background-color: #28a745; }
        .status-degraded { background-color: #ffc107; }
        .status-error { background-color: #dc3545; }
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>
<body style="background-color: #f8f9fa;">
    <!-- ヘッダー -->
    <div class="dashboard-header">
        <div class="container-fluid">
            <h1><i class="fas fa-chart-line"></i> Day Trade Enterprise Dashboard</h1>
            <p class="mb-0">リアルタイム統合分析システム監視</p>
        </div>
    </div>

    <div class="container-fluid">
        <!-- システム概要 -->
        <div class="row">
            <div class="col-md-3">
                <div class="metric-card">
                    <h5><i class="fas fa-server"></i> システム状態</h5>
                    <div id="system-status">
                        <span class="status-indicator status-running"></span>
                        <span id="status-text">Running</span>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h5><i class="fas fa-clock"></i> 稼働時間</h5>
                    <div id="uptime">0 秒</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h5><i class="fas fa-puzzle-piece"></i> コンポーネント</h5>
                    <div id="components">0/0 健全</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h5><i class="fas fa-tachometer-alt"></i> パフォーマンス</h5>
                    <div id="performance">監視中...</div>
                </div>
            </div>
        </div>

        <!-- チャート -->
        <div class="row">
            <div class="col-md-6">
                <div class="chart-container">
                    <h5>システム状態推移</h5>
                    <div id="system-status-chart"></div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="chart-container">
                    <h5>パフォーマンス指標</h5>
                    <div id="performance-chart"></div>
                </div>
            </div>
        </div>

        <!-- コンポーネント詳細 -->
        <div class="row">
            <div class="col-12">
                <div class="chart-container">
                    <h5>コンポーネント詳細</h5>
                    <div id="component-details"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- JavaScript -->
    <script>
        // WebSocket接続
        let ws;

        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);

            ws.onopen = function(event) {
                console.log('WebSocket接続成功');
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };

            ws.onclose = function(event) {
                console.log('WebSocket接続切断、再接続試行...');
                setTimeout(connectWebSocket, 3000);
            };

            ws.onerror = function(error) {
                console.error('WebSocket エラー:', error);
            };
        }

        function updateDashboard(data) {
            // システム状態更新
            if (data.system_overview) {
                const overview = data.system_overview;

                document.getElementById('status-text').textContent = overview.system_status;
                document.getElementById('uptime').textContent = `${Math.round(overview.uptime_seconds)} 秒`;
                document.getElementById('components').textContent =
                    `${overview.components.healthy}/${overview.components.total} 健全`;

                // ステータス色更新
                const indicator = document.querySelector('.status-indicator');
                indicator.className = `status-indicator status-${overview.system_status}`;
            }

            // チャート更新
            if (data.charts) {
                if (data.charts.system_status) {
                    document.getElementById('system-status-chart').innerHTML = data.charts.system_status;
                }
                if (data.charts.performance) {
                    document.getElementById('performance-chart').innerHTML = data.charts.performance;
                }
            }

            // コンポーネント詳細更新
            if (data.component_details) {
                updateComponentDetails(data.component_details);
            }
        }

        function updateComponentDetails(details) {
            const container = document.getElementById('component-details');
            let html = '<div class="row">';

            for (const [name, component] of Object.entries(details)) {
                const statusClass = component.healthy ? 'success' : 'danger';
                const statusIcon = component.healthy ? 'check-circle' : 'exclamation-triangle';

                html += `
                    <div class="col-md-4 mb-3">
                        <div class="card border-${statusClass}">
                            <div class="card-body">
                                <h6 class="card-title">
                                    <i class="fas fa-${statusIcon} text-${statusClass}"></i>
                                    ${name}
                                </h6>
                                <p class="card-text">
                                    <small>タイプ: ${component.type}</small><br>
                                    <small>処理時間: ${component.performance.processing_time_ms}ms</small><br>
                                    <small>メモリ使用量: ${component.performance.memory_usage_mb}MB</small>
                                </p>
                            </div>
                        </div>
                    </div>
                `;
            }

            html += '</div>';
            container.innerHTML = html;
        }

        // ページ読み込み時の初期化
        document.addEventListener('DOMContentLoaded', function() {
            connectWebSocket();

            // 定期更新
            setInterval(async function() {
                try {
                    const response = await fetch('/api/system/overview');
                    const data = await response.json();
                    updateDashboard({system_overview: data});
                } catch (error) {
                    console.error('データ更新エラー:', error);
                }
            }, 10000); // 10秒ごと
        });
    </script>
</body>
</html>
        """

        template_path = Path(self.templates.directory) / "dashboard.html"
        with open(template_path, "w", encoding="utf-8") as f:
            f.write(template_content.strip())

    async def _handle_websocket_connection(self, websocket: WebSocket) -> None:
        """WebSocket接続処理"""
        await websocket.accept()

        if len(self.websocket_connections) >= self.config.max_websocket_connections:
            await websocket.send_text(json.dumps({"error": "最大接続数に達しました"}))
            await websocket.close()
            return

        self.websocket_connections.append(websocket)
        logger.info(f"WebSocket接続追加: {len(self.websocket_connections)} 接続")

        try:
            # 初期データ送信
            initial_data = {
                "system_overview": self.orchestrator.get_system_overview(),
                "component_details": self.orchestrator.get_component_details(),
            }
            await websocket.send_text(json.dumps(initial_data, default=str))

            # 接続維持
            while True:
                await websocket.receive_text()

        except WebSocketDisconnect:
            if websocket in self.websocket_connections:
                self.websocket_connections.remove(websocket)
                logger.info(
                    f"WebSocket接続削除: {len(self.websocket_connections)} 接続"
                )
        except Exception as e:
            logger.error(f"WebSocket処理エラー: {e}")
            if websocket in self.websocket_connections:
                self.websocket_connections.remove(websocket)

    async def _generate_system_status_chart(self) -> str:
        """システム状態チャート生成"""
        try:
            # データ準備
            timestamps = [
                datetime.now() - timedelta(minutes=i) for i in range(30, 0, -1)
            ]
            health_scores = [
                np.random.uniform(0.8, 1.0) for _ in timestamps
            ]  # 模擬データ

            # Plotly チャート作成
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=health_scores,
                    mode="lines+markers",
                    name="システム健全性",
                    line=dict(color="#28a745", width=3),
                    marker=dict(size=6),
                )
            )

            fig.update_layout(
                title="システム状態推移 (30分間)",
                xaxis_title="時刻",
                yaxis_title="健全性スコア",
                height=400,
                showlegend=True,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            return fig.to_html(include_plotlyjs=False, div_id="system-status-chart")

        except Exception as e:
            logger.error(f"システム状態チャート生成エラー: {e}")
            return f"<div>チャート生成エラー: {e}</div>"

    async def _generate_performance_chart(self) -> str:
        """パフォーマンスチャート生成"""
        try:
            # 模擬パフォーマンスデータ
            timestamps = [
                datetime.now() - timedelta(minutes=i) for i in range(30, 0, -1)
            ]
            cpu_usage = [np.random.uniform(20, 60) for _ in timestamps]
            memory_usage = [np.random.uniform(100, 500) for _ in timestamps]

            # サブプロット作成
            fig = sp.make_subplots(
                rows=2,
                cols=1,
                subplot_titles=["CPU使用率 (%)", "メモリ使用量 (MB)"],
                vertical_spacing=0.1,
            )

            # CPU使用率
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=cpu_usage,
                    mode="lines+markers",
                    name="CPU使用率",
                    line=dict(color="#007bff"),
                ),
                row=1,
                col=1,
            )

            # メモリ使用量
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=memory_usage,
                    mode="lines+markers",
                    name="メモリ使用量",
                    line=dict(color="#28a745"),
                ),
                row=2,
                col=1,
            )

            fig.update_layout(
                height=500,
                showlegend=True,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            return fig.to_html(include_plotlyjs=False, div_id="performance-chart")

        except Exception as e:
            logger.error(f"パフォーマンスチャート生成エラー: {e}")
            return f"<div>チャート生成エラー: {e}</div>"

    async def start_dashboard_server(self) -> None:
        """ダッシュボードサーバー開始"""
        try:
            # リアルタイム更新タスク開始
            if self.config.enable_real_time_updates:
                update_task = asyncio.create_task(self._real_time_update_loop())

            logger.info(
                f"🌐 エンタープライズダッシュボード開始: http://{self.config.host}:{self.config.port}"
            )

            # サーバー開始
            config = uvicorn.Config(
                self.app,
                host=self.config.host,
                port=self.config.port,
                log_level="info" if self.config.debug else "warning",
            )
            server = uvicorn.Server(config)
            await server.serve()

        except Exception as e:
            logger.error(f"ダッシュボードサーバー開始エラー: {e}")

    async def _real_time_update_loop(self) -> None:
        """リアルタイム更新ループ"""
        while True:
            try:
                if self.websocket_connections:
                    # 最新データ取得
                    update_data = {
                        "system_overview": self.orchestrator.get_system_overview(),
                        "component_details": self.orchestrator.get_component_details(),
                        "timestamp": datetime.now().isoformat(),
                    }

                    # チャート更新
                    update_data["charts"] = {
                        "system_status": await self._generate_system_status_chart(),
                        "performance": await self._generate_performance_chart(),
                    }

                    # 全WebSocket接続に送信
                    message = json.dumps(update_data, default=str)
                    disconnected = []

                    for websocket in self.websocket_connections:
                        try:
                            await websocket.send_text(message)
                        except Exception:
                            disconnected.append(websocket)

                    # 切断された接続を削除
                    for ws in disconnected:
                        if ws in self.websocket_connections:
                            self.websocket_connections.remove(ws)

                await asyncio.sleep(self.config.auto_refresh_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"リアルタイム更新エラー: {e}")
                await asyncio.sleep(5)

    async def generate_comprehensive_report(self, symbols: List[str]) -> Dict[str, Any]:
        """包括的レポート生成"""
        report = {
            "report_id": f"enterprise_report_{int(time.time())}",
            "generated_at": datetime.now().isoformat(),
            "system_overview": self.orchestrator.get_system_overview(),
            "component_details": self.orchestrator.get_component_details(),
            "analysis_results": {},
            "charts": {},
            "summary": {},
        }

        # 銘柄別分析
        for symbol in symbols:
            analysis_report = await self.orchestrator.get_integrated_analysis_report(
                [symbol]
            )
            report["analysis_results"][symbol] = analysis_report[
                "analysis_results"
            ].get(symbol, {})

        # チャート生成
        report["charts"]["system_status"] = await self._generate_system_status_chart()
        report["charts"]["performance"] = await self._generate_performance_chart()

        # サマリー計算
        report["summary"] = {
            "total_symbols_analyzed": len(symbols),
            "healthy_components_ratio": self.orchestrator.get_system_overview()[
                "components"
            ]["health_ratio"],
            "system_uptime_hours": self.orchestrator.get_system_overview()[
                "uptime_seconds"
            ]
            / 3600,
            "overall_system_health": (
                "healthy"
                if report["summary"]["healthy_components_ratio"] > 0.8
                else "degraded"
            ),
        }

        return report


# 使用例・統合テスト関数


async def setup_enterprise_dashboard(
    orchestrator: EnterpriseIntegrationOrchestrator,
) -> EnterpriseDashboardSystem:
    """エンタープライズダッシュボードセットアップ"""
    config = DashboardConfig(
        theme=DashboardTheme.ENTERPRISE,
        enable_real_time_updates=True,
        auto_refresh_interval_seconds=10,
        port=8080,
    )

    dashboard = EnterpriseDashboardSystem(orchestrator, config)
    return dashboard


async def test_dashboard_integration():
    """ダッシュボード統合テスト"""
    try:
        from ..core.enterprise_integration_orchestrator import setup_enterprise_system

        print("🚀 エンタープライズダッシュボード統合テスト開始")

        # オーケストレーター初期化
        orchestrator = await setup_enterprise_system()
        await orchestrator.start_enterprise_operations()

        # ダッシュボード初期化
        dashboard = await setup_enterprise_dashboard(orchestrator)

        # 包括レポート生成テスト
        test_symbols = ["7203", "8306", "9984"]
        report = await dashboard.generate_comprehensive_report(test_symbols)

        print("\n📊 包括レポート生成結果:")
        print(f"  レポートID: {report['report_id']}")
        print(f"  分析銘柄数: {report['summary']['total_symbols_analyzed']}")
        print(f"  システム健全性: {report['summary']['healthy_components_ratio']:.1%}")
        print(f"  稼働時間: {report['summary']['system_uptime_hours']:.2f} 時間")

        print("\n✅ ダッシュボード統合テスト完了")

        # NOTE: 実際のサーバー起動はテスト環境では行わない
        # await dashboard.start_dashboard_server()

        return True

    except Exception as e:
        print(f"❌ ダッシュボード統合テストエラー: {e}")
        return False

    finally:
        if "orchestrator" in locals():
            await orchestrator.stop_enterprise_operations()


if __name__ == "__main__":
    asyncio.run(test_dashboard_integration())
