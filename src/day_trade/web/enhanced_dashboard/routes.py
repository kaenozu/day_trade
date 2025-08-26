#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Routes - Flaskルート設定

REST APIエンドポイントとウェブルートの定義
"""

import logging
from flask import Flask, render_template, jsonify, request, send_from_directory
from typing import Any, Dict, List


class DashboardRoutes:
    """ダッシュボードルート管理"""
    
    def __init__(self, app: Flask, dashboard_instance):
        self.app = app
        self.dashboard = dashboard_instance
        self.logger = logging.getLogger(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """APIルートの設定"""

        @self.app.route('/')
        def dashboard():
            """メインダッシュボード"""
            return self.dashboard._render_enhanced_dashboard()

        @self.app.route('/api/symbols')
        def api_symbols():
            """銘柄一覧API"""
            return jsonify(self.dashboard._get_available_symbols())

        @self.app.route('/api/chart/<symbol>')
        def api_chart(symbol):
            """高度なチャートAPI"""
            chart_type = request.args.get('type', 'candlestick')
            indicators = request.args.getlist('indicators')
            return jsonify(self.dashboard._get_enhanced_chart_data(symbol, chart_type, indicators))

        @self.app.route('/api/prediction/<symbol>')
        def api_prediction(symbol):
            """予測データAPI"""
            return jsonify(self.dashboard._get_prediction_data(symbol))

        @self.app.route('/api/performance')
        def api_performance():
            """システム性能API"""
            return jsonify(self.dashboard._get_system_performance())

        @self.app.route('/api/alerts/<user_id>')
        def api_alerts(user_id):
            """ユーザーアラートAPI"""
            return jsonify(self.dashboard._get_user_alerts(user_id))

        @self.app.route('/api/alerts/<user_id>', methods=['POST'])
        def api_create_alert(user_id):
            """アラート作成API"""
            alert_data = request.json
            return jsonify(self.dashboard._create_user_alert(user_id, alert_data))

        @self.app.route('/api/alerts/<user_id>/<alert_id>', methods=['PUT'])
        def api_update_alert(user_id, alert_id):
            """アラート更新API"""
            alert_data = request.json
            return jsonify(self.dashboard._update_user_alert(user_id, alert_id, alert_data))

        @self.app.route('/api/alerts/<user_id>/<alert_id>', methods=['DELETE'])
        def api_delete_alert(user_id, alert_id):
            """アラート削除API"""
            return jsonify(self.dashboard._delete_user_alert(user_id, alert_id))

        @self.app.route('/api/preferences/<user_id>')
        def api_preferences(user_id):
            """ユーザー設定API"""
            return jsonify(self.dashboard._get_user_preferences(user_id))

        @self.app.route('/api/preferences/<user_id>', methods=['POST'])
        def api_save_preferences(user_id):
            """ユーザー設定保存API"""
            preferences = request.json
            return jsonify(self.dashboard._save_user_preferences(user_id, preferences))

        @self.app.route('/api/export/<format>')
        def api_export(format):
            """データエクスポートAPI"""
            symbols = request.args.getlist('symbols')
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            return self.dashboard._export_data(symbols, start_date, end_date, format)

        @self.app.route('/api/status')
        def api_status():
            """システムステータスAPI"""
            return jsonify({
                'status': 'running',
                'version': '1.0.0',
                'active_subscriptions': len(self.dashboard.real_time_manager.active_subscriptions),
                'cache_size': len(self.dashboard.real_time_manager.data_cache),
                'config': {
                    'theme': self.dashboard.config.theme.value,
                    'update_frequency': self.dashboard.config.update_frequency.value,
                    'alerts_enabled': self.dashboard.config.alerts_enabled
                }
            })

        @self.app.route('/api/symbols/<symbol>/data')
        def api_symbol_data(symbol):
            """個別銘柄データAPI"""
            period = request.args.get('period', '1d')
            interval = request.args.get('interval', '1m')
            return jsonify(self.dashboard._get_symbol_data(symbol, period, interval))

        @self.app.route('/api/symbols/<symbol>/indicators')
        def api_symbol_indicators(symbol):
            """技術指標API"""
            indicators = request.args.getlist('indicators')
            period = request.args.get('period', '1d')
            return jsonify(self.dashboard._get_symbol_indicators(symbol, indicators, period))

        @self.app.route('/api/watchlist/<user_id>')
        def api_watchlist(user_id):
            """ウォッチリストAPI"""
            return jsonify(self.dashboard._get_user_watchlist(user_id))

        @self.app.route('/api/watchlist/<user_id>', methods=['POST'])
        def api_add_to_watchlist(user_id):
            """ウォッチリスト追加API"""
            data = request.json
            symbol = data.get('symbol')
            return jsonify(self.dashboard._add_to_watchlist(user_id, symbol))

        @self.app.route('/api/watchlist/<user_id>/<symbol>', methods=['DELETE'])
        def api_remove_from_watchlist(user_id, symbol):
            """ウォッチリスト削除API"""
            return jsonify(self.dashboard._remove_from_watchlist(user_id, symbol))

        @self.app.route('/api/analysis/<symbol>')
        def api_analysis(symbol):
            """分析結果API"""
            analysis_type = request.args.get('type', 'technical')
            return jsonify(self.dashboard._get_analysis_result(symbol, analysis_type))

        @self.app.route('/api/alerts/history/<user_id>')
        def api_alert_history(user_id):
            """アラート履歴API"""
            limit = request.args.get('limit', 50, type=int)
            return jsonify(self.dashboard._get_alert_history(user_id, limit))

        @self.app.route('/api/dashboard/config', methods=['GET'])
        def api_dashboard_config():
            """ダッシュボード設定API"""
            return jsonify({
                'theme': self.dashboard.config.theme.value,
                'update_frequency': self.dashboard.config.update_frequency.value,
                'default_symbols': self.dashboard.config.default_symbols,
                'charts_per_row': self.dashboard.config.charts_per_row,
                'show_volume': self.dashboard.config.show_volume,
                'show_indicators': self.dashboard.config.show_indicators,
                'alerts_enabled': self.dashboard.config.alerts_enabled,
                'ml_predictions_enabled': self.dashboard.config.ml_predictions_enabled,
                'performance_monitoring_enabled': self.dashboard.config.performance_monitoring_enabled
            })

        @self.app.route('/api/dashboard/config', methods=['POST'])
        def api_update_dashboard_config():
            """ダッシュボード設定更新API"""
            config_data = request.json
            return jsonify(self.dashboard._update_dashboard_config(config_data))

        # エラーハンドラー
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Not found'}), 404

        @self.app.errorhandler(500)
        def internal_error(error):
            self.logger.error(f"Internal server error: {error}")
            return jsonify({'error': 'Internal server error'}), 500

        @self.app.errorhandler(400)
        def bad_request(error):
            return jsonify({'error': 'Bad request'}), 400

        # CORS対応
        @self.app.after_request
        def after_request(response):
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
            return response

        self.logger.info("ルート設定が完了しました")

    def register_custom_route(self, rule: str, endpoint: str, view_func, methods: List[str] = None):
        """カスタムルートの登録"""
        if methods is None:
            methods = ['GET']
        
        self.app.add_url_rule(rule, endpoint, view_func, methods=methods)
        self.logger.info(f"カスタムルート登録: {rule} -> {endpoint}")

    def get_route_list(self) -> List[Dict[str, Any]]:
        """登録済みルートの一覧取得"""
        routes = []
        for rule in self.app.url_map.iter_rules():
            routes.append({
                'endpoint': rule.endpoint,
                'methods': list(rule.methods),
                'rule': rule.rule
            })
        return routes