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
                <h3>🎯 売買判断システム</h3>
                <p>リアルタイム株価分析による具体的な売買提案</p>
                <button class="btn" onclick="loadRecommendations()">売買推奨を取得</button>
                <div id="recommendationsResult" style="margin-top: 15px;"></div>
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
