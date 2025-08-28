#!/usr/bin/env python3
"""
Webダッシュボード ビューレート モジュール

HTMLページレンダリングルート実装
"""

from flask import render_template

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


def setup_view_routes(app):
    """ビューレート設定"""

    @app.route("/")
    def index():
        """メインダッシュボードページ"""
        try:
            return render_template("dashboard.html")
        except Exception as e:
            logger.error(f"メインダッシュボード表示エラー: {e}")
            return f"ダッシュボードの読み込み中にエラーが発生しました: {str(e)}", 500

    @app.route("/analysis")
    def analysis():
        """分析ダッシュボードページ"""
        try:
            return render_template("analysis.html")
        except Exception as e:
            logger.error(f"分析ダッシュボード表示エラー: {e}")
            return f"分析ダッシュボードの読み込み中にエラーが発生しました: {str(e)}", 500

    @app.route("/health")
    def health_check():
        """ヘルスチェックエンドポイント"""
        return {
            "status": "healthy",
            "service": "day-trade-web-dashboard",
            "version": "1.0.0"
        }

    @app.route("/info")
    def info():
        """システム情報エンドポイント"""
        return {
            "service": "日本株取引支援システム Webダッシュボード",
            "description": "プロダクション運用監視ダッシュボード",
            "features": [
                "リアルタイム監視",
                "WebSocket通信",
                "チャート生成",
                "分析機能",
                "アラート表示"
            ],
            "endpoints": {
                "/": "メインダッシュボード",
                "/analysis": "分析ダッシュボード",
                "/api/status": "現在のステータス",
                "/api/history/<type>": "履歴データ",
                "/api/chart/<type>": "チャート生成",
                "/api/report": "ステータスレポート",
                "/api/symbols": "銘柄マスタ",
                "/api/analysis": "分析実行",
                "/health": "ヘルスチェック"
            }
        }