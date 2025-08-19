#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Routes Module - Issue #959 リファクタリング対応
メインルート定義モジュール
"""

from flask import Flask, render_template_string, jsonify, request
from datetime import datetime
import time
from typing import Optional

def setup_main_routes(app: Flask, web_server_instance) -> None:
    """メインルート設定"""
    
    @app.route('/')
    def index():
        """メインダッシュボード"""
        start_time = time.time()
        
        response = render_template_string(
            web_server_instance._get_dashboard_template(), 
            title="Day Trade Personal - メインダッシュボード"
        )
        
        return response
    
    @app.route('/health')
    def health_check():
        """ヘルスチェック"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'service': 'daytrade-web'
        })
