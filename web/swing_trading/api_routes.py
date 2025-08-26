#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swing Trading API Routes
スイングトレード用APIルート定義
"""

from flask import request, jsonify


def setup_api_routes(app, swing_trading_scheduler):
    """
    APIルートを設定
    
    Args:
        app: Flaskアプリケーションインスタンス
        swing_trading_scheduler: スイングトレードスケジューラ
    """
    
    @app.route('/api/portfolio_summary')
    def api_portfolio_summary():
        """ポートフォリオサマリーAPI"""
        try:
            if not swing_trading_scheduler:
                return jsonify({"error": "Swing Trading Scheduler not available"}), 500

            summary = swing_trading_scheduler.get_portfolio_summary()
            return jsonify(summary)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/monitoring_list')
    def api_monitoring_list():
        """監視対象一覧API"""
        try:
            if not swing_trading_scheduler:
                return jsonify({"error": "Swing Trading Scheduler not available"}), 500

            from swing_trading_scheduler import HoldingStatus
            
            status_filter = request.args.get('status')
            status_enum = HoldingStatus(status_filter) if status_filter else None

            monitoring_list = swing_trading_scheduler.get_monitoring_list(status_enum)
            return jsonify(monitoring_list)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/alerts')
    def api_alerts():
        """アラートAPI"""
        try:
            if not swing_trading_scheduler:
                return jsonify({"error": "Swing Trading Scheduler not available"}), 500

            unread_only = request.args.get('unread_only', 'false').lower() == 'true'
            limit = int(request.args.get('limit', 50))

            alerts_list = swing_trading_scheduler.get_alerts(limit=limit, unread_only=unread_only)
            return jsonify(alerts_list)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/evaluate_sell/<purchase_id>')
    def evaluate_sell(purchase_id):
        """売りタイミング評価API"""
        try:
            if not swing_trading_scheduler:
                return jsonify({"error": "Swing Trading Scheduler not available"}), 500

            # 売りタイミング評価
            monitoring_schedule = swing_trading_scheduler.evaluate_sell_timing(purchase_id)

            if monitoring_schedule:
                return jsonify({
                    "success": True,
                    "evaluation": {
                        "symbol": monitoring_schedule.symbol,
                        "current_price": monitoring_schedule.current_price,
                        "change_percent": monitoring_schedule.current_change_percent,
                        "signal_strength": monitoring_schedule.sell_signal_strength.value,
                        "confidence_score": monitoring_schedule.confidence_score,
                        "status": monitoring_schedule.status.value,
                        "alert_level": monitoring_schedule.alert_level,
                        "signals": monitoring_schedule.sell_signal_reasons,
                        "updated_at": monitoring_schedule.updated_at.isoformat()
                    }
                })
            else:
                return jsonify({"success": False, "error": "Evaluation failed"}), 400

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/mark_alert_read/<alert_id>')
    def mark_alert_read(alert_id):
        """アラート既読処理API"""
        try:
            if not swing_trading_scheduler:
                return jsonify({"error": "Swing Trading Scheduler not available"}), 500

            success = swing_trading_scheduler.mark_alert_as_read(alert_id)

            return jsonify({
                "success": success,
                "message": "アラートを既読にしました" if success else "アラートが見つかりません"
            })

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500