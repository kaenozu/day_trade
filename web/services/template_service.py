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
            <!-- 機能メニューカード -->
            <div class="card">
                <h3>🔗 投資管理メニュー</h3>
                <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">
                    <button class="nav-btn" onclick="location.href='/portfolio'" style="background: #3b82f6; color: white; border: none; padding: 15px 25px; border-radius: 8px; font-weight: bold; cursor: pointer;">💼 ポートフォリオ追跡</button>
                </div>
            </div>
        </div>

        

        <!-- 拡張推奨銘柄セクション -->
        <div class="recommendations-section" style="margin-top: 30px;">
            <h2 style="color: white; text-align: center; margin-bottom: 20px;">📈 推奨銘柄一覧 (35銘柄)</h2>
            <div id="progressIndicator" style="background: white; padding: 20px; border-radius: 12px; margin-bottom: 20px; display: none; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"></div>
            <div id="filterButtons" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; margin-bottom: 15px;">
                <!-- フィルタリングボタンがJavaScriptで動的生成される -->
            </div>
            <div id="recommendationsContainer">
                <div id="recommendationsList" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;"></div>
            </div>
        </div>

        <div class="footer">
            <p>© 2025 Day Trade Personal - プロダクション対応版</p>
        </div>
    </div>

    <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    <script>
        async function initializeData() {
            const recommendationsList = document.getElementById('recommendationsList');
            const progressIndicator = document.getElementById('progressIndicator');
            progressIndicator.style.display = 'block';
            progressIndicator.innerHTML = '<div style="display: flex; align-items: center; gap: 15px;"><div style="width: 24px; height: 24px; border: 3px solid #e5e7eb; border-top: 3px solid #3b82f6; border-radius: 50%; animation: spin 1s linear infinite;"></div><div>推奨銘柄データを取得中...</div></div>';

            try {
                const response = await fetch('/api/recommendations');
                const data = await response.json();
                const recommendations = data.recommendations || [];
                displayRecommendationsList(recommendations);
            } catch (error) {
                console.error('データ取得エラー:', error);
                recommendationsList.innerHTML = '<p style="color: red; text-align: center;">データの読み込みに失敗しました。</p>';
            } finally {
                progressIndicator.style.display = 'none';
            }
        }

        function displayRecommendationsList(recommendations) {
            const recommendationsList = document.getElementById('recommendationsList');
            recommendationsList.innerHTML = recommendations.map(rec => `
                <div class="recommendation-card" style="border-left: 5px solid ${rec.recommendation === 'BUY' ? '#10b981' : rec.recommendation === 'SELL' ? '#ef4444' : '#f59e0b'}; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <h3 style="margin: 0; font-size: 1.4rem; color: #1f2937;">${rec.symbol} (${rec.name})</h3>
                        <span style="padding: 6px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; background: ${rec.recommendation === 'BUY' ? '#dcfdf7; color: #047857' : rec.recommendation === 'SELL' ? '#fee2e2; color: #dc2626' : '#fef3c7; color: #d97706'};">${rec.recommendation}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px; border-top: 1px solid #eee; padding-top: 15px;">
                        <div style="text-align: center;">
                            <div style="font-size: 0.8rem; color: #6b7280;">信頼度</div>
                            <div style="font-weight: 600; font-size: 1.1rem;">${rec.confidence ? (rec.confidence * 100).toFixed(0) : '-'}%</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.8rem; color: #6b7280;">現在価格</div>
                            <div style="font-weight: 600; font-size: 1.1rem;">¥${rec.price ? rec.price.toLocaleString() : '-'}</div>
                        </div>
                    </div>
                    <div style="background: #f8fafc; padding: 15px; border-radius: 8px;">
                        <div style="margin-bottom: 10px;">
                            <strong style="color: #10b981;">購入タイミング:</strong>
                            <p style="margin: 5px 0 0 0; white-space: pre-wrap;">${rec.buy_timing || '分析中'}</p>
                        </div>
                        <div>
                            <strong style="color: #ef4444;">売却タイミング:</strong>
                            <p style="margin: 5px 0 0 0; white-space: pre-wrap;">${rec.sell_timing || '分析中'}</p>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        document.addEventListener('DOMContentLoaded', initializeData);
    </script>
</body>
</html>
        """

    @staticmethod
    def get_recommendations_template() -> str:
        """推奨銘柄フィルタリングテンプレート"""
        return """"""

    @staticmethod
    def get_portfolio_template() -> str:
        """ポートフォリオ追跡テンプレート（改善版）"""
        return """"""

    @staticmethod
    def get_scheduler_template() -> str:
        """スケジューラー管理テンプレート"""
        return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: sans-serif; background-color: #f8f9fa; color: #343a40; text-align: center; padding: 50px; }
        h1 { font-size: 48px; margin-bottom: 10px; }
        p { font-size: 18px; }
        a { color: #007bff; text-decoration: none; }
    </style>
</head>
<body>
    <h1>エラー {{ error_code }}</h1>
    <p>{{ error_message }}</p>
    <p>{{ error_description }}</p>
    <a href="/">ホームに戻る</a>
</body>
</html>
"""
