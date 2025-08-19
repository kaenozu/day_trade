#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template Service - Issue #959 リファクタリング対応
HTMLテンプレート管理サービス
"""

class TemplateService:
    """HTMLテンプレート管理サービス"""

    @staticmethod
    def get_dashboard_template() -> str:
        """メインダッシュボードテンプレート"""
        return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏠 Day Trade Personal</h1>
            <p>プロダクション対応 - 個人投資家専用版</p>
        </div>

        <div class="dashboard">
            <div class="card">
                <h3>📊 システム状態</h3>
                <div class="status">
                    <div class="status-dot"></div>
                    <span>正常運行中</span>
                </div>
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-number">93%</div>
                        <div class="stat-label">AI精度</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">A+</div>
                        <div class="stat-label">品質評価</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>🎯 分析機能</h3>
                <p>主要銘柄の即座分析が可能です</p>
                <button class="btn" onclick="runAnalysis()">単一分析実行</button>
                <button class="btn" onclick="loadRecommendations()" style="margin-left: 10px;">推奨銘柄表示</button>
                <div id="analysisResult" style="margin-top: 15px; padding: 10px; background: #f7fafc; border-radius: 6px; display: none;"></div>
            </div>
        </div>

        <!-- 拡張推奨銘柄セクション -->
        <div class="recommendations-section" style="margin-top: 30px;">
            <h2 style="color: white; text-align: center; margin-bottom: 20px;">📈 推奨銘柄一覧 (35銘柄)</h2>
            <div id="recommendationsContainer" style="display: none;">
                <div class="recommendations-summary" style="background: white; padding: 20px; border-radius: 12px; margin-bottom: 20px; text-align: center;">
                    <div id="summaryStats" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px;"></div>
                </div>
                <div id="recommendationsList" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;"></div>
            </div>
        </div>

        <div class="footer">
            <p>🤖 Issue #959 リファクタリング対応 - モジュール化完了</p>
            <p>Generated with Claude Code</p>
        </div>
    </div>

    <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>
        """
