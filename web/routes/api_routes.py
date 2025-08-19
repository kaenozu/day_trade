#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Routes Module - Issue #959 リファクタリング対応
API エンドポイント定義モジュール
Issue #939対応: Gunicorn/Application Factory対応
"""

from flask import Flask, jsonify, request, g
from datetime import datetime

def setup_api_routes(app: Flask) -> None:
    """APIルート設定 (Application Factory対応)"""

    @app.route('/api/status')
    def api_status():
        """システム状態API"""
        return jsonify({
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'version': app.config.get('VERSION_INFO', {}).get('version_extended', '2.4.0'),
            'features': [
                'Real-time Analysis',
                'Security Enhanced',
                'Performance Optimized',
                'Async Task Execution',
                'Production Ready (Gunicorn)'
            ]
        })

    @app.route('/api/recommendations')
    def api_recommendations():
        """推奨銘柄API"""
        try:
            recommendations = g.recommendation_service.get_recommendations()
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
            return jsonify({'error': str(e), 'timestamp': datetime.now().isoformat()}), 500

    @app.route('/api/analyze/<symbol>', methods=['POST'])
    def api_start_analysis(symbol):
        """個別銘柄分析の非同期タスクを開始するAPI"""
        try:
            result = app.start_analysis_task(symbol)
            return jsonify(result), 202 # 202 Accepted
        except Exception as e:
            return jsonify({
                'error': str(e),
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/result/<task_id>', methods=['GET'])
    def api_get_result(task_id):
        """非同期タスクの結果を取得するAPI"""
        try:
            result = app.get_task_status(task_id)
            return jsonify(result)
        except Exception as e:
            return jsonify({
                'error': str(e),
                'task_id': task_id,
                'timestamp': datetime.now().isoformat()
            }), 500
