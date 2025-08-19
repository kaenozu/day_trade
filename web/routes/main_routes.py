#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Routes Module - Issue #959 リファクタリング対応
メインルート定義モジュール
Issue #939対応: Gunicorn/Application Factory対応
"""

from flask import Flask, render_template_string, jsonify, request, g
from datetime import datetime
import time

def setup_main_routes(app: Flask) -> None:
    """メインルート設定 (Application Factory対応)"""

    @app.route('/')
    def index():
        """メインダッシュボード"""
        # gオブジェクトからtemplate_serviceを取得
        template_content = g.template_service.get_dashboard_template()

        response = render_template_string(
            template_content,
            title="Day Trade Personal - メインダッシュボード"
        )

        return response

    @app.route('/portfolio')
    def portfolio_page():
        """ポートフォリオ追跡ページ"""
        template_content = g.template_service.get_portfolio_template()
        return render_template_string(
            template_content,
            title="ポートフォリオ追跡 - 購入銘柄管理"
        )



    @app.route('/health')
    def health_check():
        """ヘルスチェック"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'service': 'daytrade-web'
        })