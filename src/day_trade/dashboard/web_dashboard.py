#!/usr/bin/env python3
"""
Webベースダッシュボード

Issue #324: プロダクション運用監視ダッシュボード構築
Flask+WebSocket リアルタイムダッシュボードUI
"""

import os
import secrets
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit

from ..utils.logging_config import get_context_logger
from .dashboard_core import ProductionDashboard
from .visualization_engine import DashboardVisualizationEngine

logger = get_context_logger(__name__)


class WebDashboard:
    """Webダッシュボード"""

    def __init__(self, port: int = 5000, debug: bool = False):
        """
        初期化

        Args:
            port: サーバーポート
            debug: デバッグモード
        """
        self.port = port
        self.debug = debug

        # Flask アプリケーション設定
        self.app = Flask(
            __name__,
            template_folder=str(Path(__file__).parent / "templates"),
            static_folder=str(Path(__file__).parent / "static"),
        )
        # セキュリティ強化: 環境変数からSECRET_KEYを取得、なければランダム生成
        secret_key = os.environ.get("FLASK_SECRET_KEY")
        if not secret_key:
            secret_key = secrets.token_urlsafe(32)
            logger.warning("FLASK_SECRET_KEY環境変数が未設定です。ランダムキーを生成しました。")
            logger.info(
                f"本番環境では環境変数を設定してください: export FLASK_SECRET_KEY='{secret_key}'"
            )

        self.app.config["SECRET_KEY"] = secret_key

        # WebSocket設定 - CORS制限
        cors_origins = os.environ.get(
            "DASHBOARD_CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000"
        )
        allowed_origins = [origin.strip() for origin in cors_origins.split(",")]

        if debug:
            # デバッグモードでは開発用オリジンを許可
            allowed_origins.extend(["http://localhost:3000", "http://127.0.0.1:3000"])

        self.socketio = SocketIO(self.app, cors_allowed_origins=allowed_origins)
        logger.info(f"CORS許可オリジン: {allowed_origins}")

        # コアコンポーネント初期化
        self.dashboard_core = ProductionDashboard()
        self.visualization_engine = DashboardVisualizationEngine()

        # 更新スレッド制御
        self.update_thread = None
        self.running = False

        # ルート設定
        self._setup_routes()
        self._setup_websocket_events()
        self._setup_security_headers()

        # テンプレートフォルダ作成
        self._create_templates()
        self._create_static_files()

        logger.info(f"Webダッシュボード初期化完了 (ポート: {port})")

    def _setup_security_headers(self):
        """セキュリティヘッダー設定"""

        @self.app.after_request
        def set_security_headers(response):
            # XSS攻撃対策
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"

            # HTTPS強制（本番環境）
            if not self.debug:
                response.headers["Strict-Transport-Security"] = (
                    "max-age=31536000; includeSubDomains"
                )

            # CSP（Content Security Policy）
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' cdn.jsdelivr.net cdnjs.cloudflare.com; "
                "style-src 'self' 'unsafe-inline' cdn.jsdelivr.net cdnjs.cloudflare.com; "
                "font-src 'self' cdnjs.cloudflare.com; "
                "img-src 'self' data:; "
                "connect-src 'self'"
            )
            response.headers["Content-Security-Policy"] = csp

            return response

    def _sanitize_error_message(self, error: Exception) -> str:
        """エラーメッセージのサニタイズ"""
        # セキュリティ: 詳細なエラー情報を隠蔽し、一般的なメッセージを返す
        error_str = str(error).lower()

        # 機密情報が含まれる可能性のあるエラーパターン
        sensitive_patterns = [
            "password",
            "secret",
            "key",
            "token",
            "credential",
            "database",
            "connection",
            "path",
            "file",
            "directory",
        ]

        for pattern in sensitive_patterns:
            if pattern in error_str:
                return "システム内部エラーが発生しました。管理者にお問い合わせください。"

        # デバッグモード時のみ詳細表示
        if self.debug:
            return str(error)

        return "処理中にエラーが発生しました。"

    def _validate_metric_type(self, metric_type: str) -> bool:
        """メトリクスタイプの検証"""
        allowed_metrics = [
            "portfolio",
            "system",
            "trading",
            "risk",
            "performance",
            "alerts",
            "status",
        ]
        return metric_type in allowed_metrics

    def _validate_chart_type(self, chart_type: str) -> bool:
        """チャートタイプの検証"""
        allowed_charts = ["portfolio", "system", "trading", "risk", "comprehensive"]
        return chart_type in allowed_charts

    def _validate_hours_parameter(self, hours: int) -> bool:
        """時間パラメータの検証"""
        # 1時間から30日間（720時間）までを許可
        return 1 <= hours <= 720

    def _create_secure_file(self, file_path: Path, content: str, permissions: int = 0o644):
        """セキュアなファイル作成"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            # ファイル権限設定（Unix系OS）
            if os.name != "nt":  # Windows以外
                os.chmod(file_path, permissions)
                logger.info(f"ファイル権限設定: {file_path} -> {oct(permissions)}")
            else:
                logger.info(f"ファイル作成: {file_path} (Windows環境のため権限設定スキップ)")

        except Exception as e:
            logger.error(f"セキュアファイル作成エラー {file_path}: {e}")
            raise

    def _setup_routes(self):
        """HTTPルート設定"""

        @self.app.route("/")
        def index():
            """メインダッシュボードページ"""
            return render_template("dashboard.html")

        @self.app.route("/api/status")
        def get_status():
            """現在のステータス取得API"""
            try:
                status = self.dashboard_core.get_current_status()
                return jsonify(
                    {
                        "success": True,
                        "data": status,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            except Exception as e:
                # セキュアなエラーハンドリング
                error_message = self._sanitize_error_message(e)
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": error_message,
                            "timestamp": datetime.now().isoformat(),
                        }
                    ),
                    500,
                )

        @self.app.route("/api/history/<metric_type>")
        def get_history(metric_type):
            """過去データ取得API"""
            try:
                # 入力値検証
                if not self._validate_metric_type(metric_type):
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": "無効なメトリクスタイプです。",
                                "timestamp": datetime.now().isoformat(),
                            }
                        ),
                        400,
                    )

                hours = request.args.get("hours", 24, type=int)
                if not self._validate_hours_parameter(hours):
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": "時間パラメータが無効です。1-720時間の範囲で指定してください。",
                                "timestamp": datetime.now().isoformat(),
                            }
                        ),
                        400,
                    )

                data = self.dashboard_core.get_historical_data(metric_type, hours)
                return jsonify(
                    {
                        "success": True,
                        "data": data,
                        "metric_type": metric_type,
                        "hours": hours,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            except Exception as e:
                # セキュアなエラーハンドリング
                error_message = self._sanitize_error_message(e)
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": error_message,
                            "timestamp": datetime.now().isoformat(),
                        }
                    ),
                    500,
                )

        @self.app.route("/api/chart/<chart_type>")
        def get_chart(chart_type):
            """チャート生成API"""
            try:
                # 入力値検証
                if not self._validate_chart_type(chart_type):
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": "無効なチャートタイプです。",
                                "timestamp": datetime.now().isoformat(),
                            }
                        ),
                        400,
                    )

                hours = request.args.get("hours", 12, type=int)
                if not self._validate_hours_parameter(hours):
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": "時間パラメータが無効です。1-720時間の範囲で指定してください。",
                                "timestamp": datetime.now().isoformat(),
                            }
                        ),
                        400,
                    )

                if chart_type == "portfolio":
                    data = self.dashboard_core.get_historical_data("portfolio", hours)
                    chart_path = self.visualization_engine.create_portfolio_value_chart(data)
                elif chart_type == "system":
                    data = self.dashboard_core.get_historical_data("system", hours)
                    chart_path = self.visualization_engine.create_system_metrics_chart(data)
                elif chart_type == "trading":
                    data = self.dashboard_core.get_historical_data("trading", hours)
                    chart_path = self.visualization_engine.create_trading_performance_chart(data)
                elif chart_type == "risk":
                    data = self.dashboard_core.get_historical_data("risk", hours)
                    chart_path = self.visualization_engine.create_risk_metrics_heatmap(data)
                elif chart_type == "comprehensive":
                    portfolio_data = self.dashboard_core.get_historical_data("portfolio", hours)
                    system_data = self.dashboard_core.get_historical_data("system", hours)
                    trading_data = self.dashboard_core.get_historical_data("trading", hours)
                    risk_data = self.dashboard_core.get_historical_data("risk", hours)

                    # 現在のポジション情報取得
                    current_status = self.dashboard_core.get_current_status()
                    positions_data = current_status.get("portfolio", {}).get("positions", {})

                    chart_path = self.visualization_engine.create_comprehensive_dashboard(
                        portfolio_data,
                        system_data,
                        trading_data,
                        risk_data,
                        positions_data,
                    )
                else:
                    return (
                        jsonify({"success": False, "error": f"Unknown chart type: {chart_type}"}),
                        400,
                    )

                # Base64エンコード
                chart_base64 = self.visualization_engine.chart_to_base64(chart_path)

                return jsonify(
                    {
                        "success": True,
                        "chart_data": chart_base64,
                        "chart_type": chart_type,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            except Exception as e:
                # セキュアなエラーハンドリング
                error_message = self._sanitize_error_message(e)
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": error_message,
                            "timestamp": datetime.now().isoformat(),
                        }
                    ),
                    500,
                )

        @self.app.route("/api/report")
        def get_report():
            """ステータスレポート取得API"""
            try:
                report = self.dashboard_core.generate_status_report()
                return jsonify(
                    {
                        "success": True,
                        "report": report,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            except Exception as e:
                # セキュアなエラーハンドリング
                error_message = self._sanitize_error_message(e)
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": error_message,
                            "timestamp": datetime.now().isoformat(),
                        }
                    ),
                    500,
                )

    def _setup_websocket_events(self):
        """WebSocketイベント設定"""

        @self.socketio.on("connect")
        def handle_connect():
            """クライアント接続"""
            logger.info(f"クライアント接続: {request.sid}")
            emit("status", {"message": "ダッシュボードに接続されました"})

        @self.socketio.on("disconnect")
        def handle_disconnect():
            """クライアント切断"""
            logger.info(f"クライアント切断: {request.sid}")

        @self.socketio.on("request_update")
        def handle_request_update():
            """手動更新要求"""
            try:
                status = self.dashboard_core.get_current_status()
                emit("dashboard_update", status)
            except Exception as e:
                emit("error", {"message": str(e)})

    def _create_templates(self):
        """HTMLテンプレート作成"""
        templates_dir = Path(self.app.template_folder)
        templates_dir.mkdir(exist_ok=True)

        # メインダッシュボードテンプレート
        dashboard_html = """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>プロダクション運用監視ダッシュボード</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="{{ url_for('static', filename='dashboard.css') }}" rel="stylesheet">
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <nav class="navbar navbar-dark bg-primary">
        <div class="container-fluid">
            <a class="navbar-brand" href="/">
                <i class="fas fa-chart-line me-2"></i>
                プロダクション運用監視ダッシュボード
            </a>
            <div class="d-flex align-items-center">
                <span id="connection-status" class="badge bg-success me-3">
                    <i class="fas fa-circle"></i> 接続中
                </span>
                <button class="btn btn-outline-light" id="refresh-btn">
                    <i class="fas fa-sync-alt"></i> 更新
                </button>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-3">
        <!-- アラート表示エリア -->
        <div id="alerts-container"></div>

        <!-- サマリー情報 -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card bg-primary text-white">
                    <div class="card-body">
                        <div class="d-flex justify-content-between">
                            <div>
                                <h5>ポートフォリオ価値</h5>
                                <h3 id="portfolio-value">--</h3>
                            </div>
                            <i class="fas fa-wallet fa-2x"></i>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-success text-white">
                    <div class="card-body">
                        <div class="d-flex justify-content-between">
                            <div>
                                <h5>日次リターン</h5>
                                <h3 id="daily-return">--</h3>
                            </div>
                            <i class="fas fa-chart-line fa-2x"></i>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-info text-white">
                    <div class="card-body">
                        <div class="d-flex justify-content-between">
                            <div>
                                <h5>本日取引数</h5>
                                <h3 id="trades-today">--</h3>
                            </div>
                            <i class="fas fa-exchange-alt fa-2x"></i>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-warning text-dark">
                    <div class="card-body">
                        <div class="d-flex justify-content-between">
                            <div>
                                <h5>システム状態</h5>
                                <h3 id="system-status">--</h3>
                            </div>
                            <i class="fas fa-server fa-2x"></i>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- チャートエリア -->
        <div class="row">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-chart-area me-2"></i>統合ダッシュボード</h5>
                        <div class="float-end">
                            <button class="btn btn-sm btn-outline-primary" onclick="updateChart('comprehensive')">
                                <i class="fas fa-sync-alt"></i> 更新
                            </button>
                        </div>
                    </div>
                    <div class="card-body">
                        <div id="comprehensive-chart">
                            <div class="text-center p-5">
                                <div class="spinner-border text-primary" role="status">
                                    <span class="visually-hidden">読み込み中...</span>
                                </div>
                                <p class="mt-2">チャートを読み込み中...</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-list me-2"></i>ステータスレポート</h5>
                    </div>
                    <div class="card-body">
                        <pre id="status-report" class="small">読み込み中...</pre>
                    </div>
                </div>
            </div>
        </div>

        <!-- 詳細チャート -->
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-chart-line me-2"></i>ポートフォリオ</h5>
                        <button class="btn btn-sm btn-outline-primary float-end" onclick="updateChart('portfolio')">
                            <i class="fas fa-sync-alt"></i>
                        </button>
                    </div>
                    <div class="card-body">
                        <div id="portfolio-chart">チャート読み込み中...</div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-cogs me-2"></i>システムメトリクス</h5>
                        <button class="btn btn-sm btn-outline-primary float-end" onclick="updateChart('system')">
                            <i class="fas fa-sync-alt"></i>
                        </button>
                    </div>
                    <div class="card-body">
                        <div id="system-chart">チャート読み込み中...</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-exchange-alt me-2"></i>取引パフォーマンス</h5>
                        <button class="btn btn-sm btn-outline-primary float-end" onclick="updateChart('trading')">
                            <i class="fas fa-sync-alt"></i>
                        </button>
                    </div>
                    <div class="card-body">
                        <div id="trading-chart">チャート読み込み中...</div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-shield-alt me-2"></i>リスクメトリクス</h5>
                        <button class="btn btn-sm btn-outline-primary float-end" onclick="updateChart('risk')">
                            <i class="fas fa-sync-alt"></i>
                        </button>
                    </div>
                    <div class="card-body">
                        <div id="risk-chart">チャート読み込み中...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="{{ url_for('static', filename='dashboard.js') }}"></script>
</body>
</html>"""

        # セキュアなファイル作成
        self._create_secure_file(templates_dir / "dashboard.html", dashboard_html, 0o644)

    def _create_static_files(self):
        """静的ファイル作成"""
        static_dir = Path(self.app.static_folder)
        static_dir.mkdir(exist_ok=True)

        # CSS ファイル
        css_content = """
body {
    background-color: #f8f9fa;
}

.card {
    box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    border: 1px solid rgba(0, 0, 0, 0.125);
}

.card-header {
    background-color: #fff;
    border-bottom: 1px solid rgba(0, 0, 0, 0.125);
}

#status-report {
    font-family: 'Courier New', monospace;
    font-size: 11px;
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 0.375rem;
    padding: 0.75rem;
    max-height: 400px;
    overflow-y: auto;
}

.chart-container {
    position: relative;
    height: 300px;
}

.chart-container img {
    max-width: 100%;
    height: auto;
}

#connection-status.disconnected {
    background-color: #dc3545 !important;
}

.alert-custom {
    border-radius: 0.5rem;
    border-left: 4px solid;
}

.alert-custom.alert-danger {
    border-left-color: #dc3545;
    background-color: #f8d7da;
}

.alert-custom.alert-warning {
    border-left-color: #ffc107;
    background-color: #fff3cd;
}

.spinner-border {
    width: 3rem;
    height: 3rem;
}
        """

        # セキュアなファイル作成
        self._create_secure_file(static_dir / "dashboard.css", css_content, 0o644)

        # JavaScript ファイル
        js_content = """
// WebSocket接続
const socket = io();

// 接続状態管理
let isConnected = false;

// 初期化
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
});

function initializeDashboard() {
    // WebSocketイベントリスナー
    socket.on('connect', function() {
        isConnected = true;
        updateConnectionStatus(true);
        console.log('WebSocketに接続されました');

        // 初期データ要求
        requestUpdate();
    });

    socket.on('disconnect', function() {
        isConnected = false;
        updateConnectionStatus(false);
        console.log('WebSocketから切断されました');
    });

    socket.on('dashboard_update', function(data) {
        updateDashboardData(data);
    });

    socket.on('error', function(data) {
        showAlert('エラー: ' + data.message, 'danger');
    });

    // 手動更新ボタン
    document.getElementById('refresh-btn').addEventListener('click', function() {
        requestUpdate();
        updateAllCharts();
    });

    // 定期更新設定 (30秒間隔)
    setInterval(function() {
        if (isConnected) {
            requestUpdate();
        }
    }, 30000);

    // 初回チャート読み込み
    setTimeout(updateAllCharts, 1000);
}

function updateConnectionStatus(connected) {
    const statusElement = document.getElementById('connection-status');
    if (connected) {
        statusElement.className = 'badge bg-success me-3';
        statusElement.innerHTML = '<i class="fas fa-circle"></i> 接続中';
    } else {
        statusElement.className = 'badge bg-danger me-3';
        statusElement.innerHTML = '<i class="fas fa-circle"></i> 切断';
    }
}

function requestUpdate() {
    if (isConnected) {
        socket.emit('request_update');
    }

    // ステータスレポート更新
    fetch('/api/report')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('status-report').textContent = data.report;
            }
        })
        .catch(error => console.error('レポート取得エラー:', error));
}

function updateDashboardData(data) {
    try {
        // ポートフォリオデータ
        if (data.portfolio) {
            const portfolioValue = data.portfolio.total_value;
            const dailyReturn = data.portfolio.daily_return;

            document.getElementById('portfolio-value').textContent =
                new Intl.NumberFormat('ja-JP', {
                    style: 'currency',
                    currency: 'JPY',
                    maximumFractionDigits: 0
                }).format(portfolioValue);

            const returnElement = document.getElementById('daily-return');
            const returnPercent = (dailyReturn * 100).toFixed(2) + '%';
            returnElement.textContent = returnPercent;

            // リターンの色分け
            if (dailyReturn > 0) {
                returnElement.parentElement.parentElement.className = 'card bg-success text-white';
            } else if (dailyReturn < 0) {
                returnElement.parentElement.parentElement.className = 'card bg-danger text-white';
            } else {
                returnElement.parentElement.parentElement.className = 'card bg-secondary text-white';
            }
        }

        // 取引データ
        if (data.trading) {
            document.getElementById('trades-today').textContent = data.trading.trades_today + '回';
        }

        // システムデータ
        if (data.system) {
            const cpuUsage = data.system.cpu_usage;
            let systemStatus = '正常';
            let statusClass = 'bg-success';

            if (cpuUsage > 80) {
                systemStatus = '高負荷';
                statusClass = 'bg-danger';
            } else if (cpuUsage > 60) {
                systemStatus = '注意';
                statusClass = 'bg-warning text-dark';
            }

            document.getElementById('system-status').textContent = systemStatus;
            document.getElementById('system-status').parentElement.parentElement.className =
                'card ' + statusClass + ' text-white';
        }

    } catch (error) {
        console.error('データ更新エラー:', error);
    }
}

function updateChart(chartType) {
    const chartContainer = document.getElementById(chartType + '-chart');

    // ローディング表示
    chartContainer.innerHTML = `
        <div class="text-center p-3">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">読み込み中...</span>
            </div>
            <p class="mt-2">チャートを更新中...</p>
        </div>
    `;

    fetch('/api/chart/' + chartType)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                chartContainer.innerHTML = '<img src="data:image/png;base64,' + data.chart_data + '" class="img-fluid" alt="' + chartType + ' chart">';
            } else {
                chartContainer.innerHTML = '<div class="alert alert-danger">Chart Error: ' + data.error + '</div>';
            }
        })
        .catch(error => {
            console.error(chartType + ' chart error:', error);
            chartContainer.innerHTML = '<div class="alert alert-danger">チャートの読み込みに失敗しました</div>';
        });
}

function updateAllCharts() {
    const chartTypes = ['comprehensive', 'portfolio', 'system', 'trading', 'risk'];
    chartTypes.forEach(chartType => {
        updateChart(chartType);
    });
}

function showAlert(message, type = 'info') {
    const alertsContainer = document.getElementById('alerts-container');
    const alertId = 'alert-' + Date.now();

    const alertHTML = '<div id="' + alertId + '" class="alert alert-' + type + ' alert-custom alert-dismissible fade show" role="alert">' +
        '<strong><i class="fas fa-exclamation-triangle me-2"></i></strong>' +
        message +
        '<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>' +
        '</div>';

    alertsContainer.insertAdjacentHTML('beforeend', alertHTML);

    // 5秒後に自動削除
    setTimeout(function() {
        const alert = document.getElementById(alertId);
        if (alert) {
            alert.remove();
        }
    }, 5000);
}
        """

        # セキュアなファイル作成
        self._create_secure_file(static_dir / "dashboard.js", js_content, 0o644)

    def start_monitoring(self):
        """監視開始"""
        logger.info("Webダッシュボード監視開始")
        self.dashboard_core.start_monitoring()
        self.running = True

        # リアルタイム更新スレッド開始
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

    def stop_monitoring(self):
        """監視停止"""
        logger.info("Webダッシュボード監視停止")
        self.running = False
        self.dashboard_core.stop_monitoring()

    def _update_loop(self):
        """リアルタイム更新ループ"""
        while self.running:
            try:
                # 現在のステータス取得
                status = self.dashboard_core.get_current_status()

                # WebSocket経由でクライアントに送信
                self.socketio.emit("dashboard_update", status)

            except Exception as e:
                logger.error(f"更新ループエラー: {e}")

            time.sleep(10)  # 10秒間隔で更新

    def run(self):
        """サーバー開始"""
        try:
            self.start_monitoring()
            logger.info(f"Webダッシュボードサーバー開始: http://localhost:{self.port}")
            self.socketio.run(self.app, host="0.0.0.0", port=self.port, debug=self.debug)
        except KeyboardInterrupt:
            logger.info("\nサーバー停止中...")
            self.stop_monitoring()
        except Exception as e:
            logger.error(f"サーバーエラー: {e}")
            self.stop_monitoring()


def main():
    """メイン実行"""
    logger.info("Webダッシュボード起動")
    logger.info("=" * 50)

    dashboard = WebDashboard(port=5000, debug=False)
    dashboard.run()


if __name__ == "__main__":
    main()
