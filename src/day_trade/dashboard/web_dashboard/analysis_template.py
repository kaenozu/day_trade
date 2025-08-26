#!/usr/bin/env python3
"""
Webダッシュボード 分析ダッシュボードテンプレート生成モジュール

分析ダッシュボードHTMLテンプレート生成機能
"""

from pathlib import Path

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class AnalysisTemplateGenerator:
    """分析ダッシュボードテンプレート生成クラス"""

    def create_analysis_template(self, templates_dir: Path, security_manager):
        """分析ダッシュボードテンプレート作成"""
        analysis_html = """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>分析ダッシュボード</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="{{ url_for('static', filename='dashboard.css') }}" rel="stylesheet">
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
</head>
<body>
    <nav class="navbar navbar-dark bg-success">
        <div class="container-fluid">
            <a class="navbar-brand" href="/analysis">
                <i class="fas fa-chart-line me-2"></i>
                分析ダッシュボード
            </a>
            <div class="d-flex align-items-center">
                <a href="/" class="btn btn-outline-light me-2">
                    <i class="fas fa-tachometer-alt"></i> メインダッシュボード
                </a>
                <span id="connection-status" class="badge bg-light text-dark me-3">
                    <i class="fas fa-circle"></i> 接続中
                </span>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-3">
        <!-- 分析制御パネル -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-play me-2"></i>分析実行</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-8">
                                <label for="symbol-select" class="form-label">銘柄選択:</label>
                                <select id="symbol-select" class="form-select" multiple size="5">
                                    <option value="loading">銘柄情報を読み込み中...</option>
                                </select>
                                <div class="mt-2">
                                    <button class="btn btn-sm btn-secondary" onclick="selectTierSymbols(1)">Tier1</button>
                                    <button class="btn btn-sm btn-secondary" onclick="selectTierSymbols(2)">Tier2</button>
                                    <button class="btn btn-sm btn-secondary" onclick="selectTierSymbols(3)">Tier3</button>
                                    <button class="btn btn-sm btn-secondary" onclick="clearSelection()">クリア</button>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <button id="analyze-btn" class="btn btn-success btn-lg w-100" onclick="runAnalysis()">
                                    <i class="fas fa-play me-2"></i>分析開始
                                </button>
                                <div class="mt-3">
                                    <small class="text-muted">選択した銘柄に対してAI分析を実行します</small>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 進捗表示 -->
        <div id="progress-container" class="row mb-4" style="display: none;">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-spinner me-2"></i>分析進捗</h5>
                    </div>
                    <div class="card-body">
                        <div class="progress mb-2">
                            <div id="progress-bar" class="progress-bar" role="progressbar" style="width: 0%"></div>
                        </div>
                        <div id="progress-text">準備中...</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 結果表示エリア -->
        <div id="results-container" class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-chart-bar me-2"></i>分析結果</h5>
                    </div>
                    <div class="card-body">
                        <div id="results-content" class="text-center text-muted">
                            分析を開始してください
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 分析履歴 -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-history me-2"></i>分析履歴</h5>
                    </div>
                    <div class="card-body">
                        <div id="history-content" class="text-center text-muted">
                            分析履歴はありません
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 統計情報 -->
        <div class="row mt-4">
            <div class="col-md-4">
                <div class="card bg-info text-white">
                    <div class="card-body">
                        <div class="d-flex justify-content-between">
                            <div>
                                <h5>本日分析数</h5>
                                <h3 id="today-analysis-count">0</h3>
                            </div>
                            <i class="fas fa-calculator fa-2x"></i>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card bg-success text-white">
                    <div class="card-body">
                        <div class="d-flex justify-content-between">
                            <div>
                                <h5>買い推奨</h5>
                                <h3 id="buy-recommendations">0</h3>
                            </div>
                            <i class="fas fa-arrow-up fa-2x"></i>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card bg-danger text-white">
                    <div class="card-body">
                        <div class="d-flex justify-content-between">
                            <div>
                                <h5>売り推奨</h5>
                                <h3 id="sell-recommendations">0</h3>
                            </div>
                            <i class="fas fa-arrow-down fa-2x"></i>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="{{ url_for('static', filename='analysis.js') }}"></script>
</body>
</html>"""

        # セキュアなファイル作成
        security_manager.create_secure_file(
            templates_dir / "analysis.html", analysis_html, 0o644
        )