#!/usr/bin/env python3
"""
Webダッシュボード メインダッシュボードテンプレート生成モジュール

メインダッシュボードHTMLテンプレート生成機能
"""

from pathlib import Path

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class DashboardTemplateGenerator:
    """メインダッシュボードテンプレート生成クラス"""

    def create_dashboard_template(self, templates_dir: Path, security_manager):
        """メインダッシュボードテンプレート作成"""
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
                <a href="/analysis" class="btn btn-outline-light me-2">
                    <i class="fas fa-chart-line"></i> 分析ダッシュボード
                </a>
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
        security_manager.create_secure_file(
            templates_dir / "dashboard.html", dashboard_html, 0o644
        )