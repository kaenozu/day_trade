#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Routes Module - Issue #959 リファクタリング対応
API エンドポイント定義モジュール
"""

from flask import Flask, jsonify, request
from datetime import datetime
import time
from typing import Dict, Any, List

def setup_api_routes(app: Flask, web_server_instance) -> None:
    """APIルート設定"""
    
    @app.route('/api/status')
    def api_status():
        """システム状態API"""
        return jsonify({
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'version': getattr(web_server_instance, 'version_info', {}).get('version', '2.1.0'),
            'features': [
                'Real-time Analysis',
                'Security Enhanced',
                'Performance Optimized'
            ]
        })
    
    @app.route('/api/recommendations')
    def api_recommendations():
        """推奨銘柄API"""
        try:
            # 35銘柄の推奨システム
            recommendations = web_server_instance._get_recommendations()
            
            # 統計計算
            total_count = len(recommendations)
            high_confidence_count = len([r for r in recommendations if r.get('confidence', 0) > 0.8])
            buy_count = len([r for r in recommendations if r.get('recommendation') == 'BUY'])
            sell_count = len([r for r in recommendations if r.get('recommendation') == 'SELL'])
            hold_count = len([r for r in recommendations if r.get('recommendation') == 'HOLD'])
            
            return jsonify({
                'total_count': total_count,
                'high_confidence_count': high_confidence_count,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'hold_count': hold_count,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/analysis/<symbol>')
    def api_single_analysis(symbol):
        """個別銘柄分析API"""
        try:
            result = web_server_instance._analyze_single_symbol(symbol)
            return jsonify(result)
        except Exception as e:
            return jsonify({
                'error': str(e),
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }), 500
