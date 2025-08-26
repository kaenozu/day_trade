#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swing Trading Routes
スイングトレード用ルート定義
"""

from flask import render_template_string, request, jsonify
from typing import Callable

# テンプレートインポート
from .templates.dashboard import DASHBOARD_TEMPLATE
from .templates.purchase_form import PURCHASE_FORM_TEMPLATE
from .templates.sell_form import PARTIAL_SELL_FORM_TEMPLATE  
from .templates.monitoring import MONITORING_TEMPLATE
from .templates.alerts import ALERTS_TEMPLATE

# APIルートインポート
from .api_routes import setup_api_routes


def setup_routes(app, swing_trading_scheduler, version_module, audit_logger):
    """
    Flaskアプリケーションにルートを設定
    
    Args:
        app: Flaskアプリケーションインスタンス
        swing_trading_scheduler: スイングトレードスケジューラ
        version_module: バージョン情報モジュール
        audit_logger: 監査ログ
    """
    
    # APIルートを設定
    setup_api_routes(app, swing_trading_scheduler)
    
    @app.route('/')
    def dashboard():
        """メインダッシュボード"""
        try:
            if not swing_trading_scheduler:
                return "Swing Trading Scheduler not available", 500

            # ポートフォリオサマリー
            summary = swing_trading_scheduler.get_portfolio_summary()

            # 監視対象一覧
            monitoring_list = swing_trading_scheduler.get_monitoring_list()

            # 最新アラート
            alerts = swing_trading_scheduler.get_alerts(limit=10, unread_only=True)

            # バージョン情報
            version_info = version_module.get_version_info() if version_module else {"version": "Unknown"}

            return render_template_string(DASHBOARD_TEMPLATE,
                                        summary=summary,
                                        monitoring_list=monitoring_list,
                                        alerts=alerts,
                                        version_info=version_info)

        except Exception as e:
            error_msg = f"Dashboard error: {str(e)}"
            print(error_msg)
            if audit_logger:
                audit_logger.log_error_with_context(e, {"context": "swing_trading_dashboard"})
            return f"Error: {error_msg}", 500

    @app.route('/purchase_form')
    def purchase_form():
        """購入記録フォーム"""
        try:
            from swing_trading_scheduler import PurchaseStrategy
            strategies = [strategy.value for strategy in PurchaseStrategy]
            return render_template_string(PURCHASE_FORM_TEMPLATE, strategies=strategies)
        except Exception as e:
            return f"Error: {str(e)}", 500

    @app.route('/record_purchase', methods=['POST'])
    def record_purchase():
        """購入記録処理"""
        try:
            if not swing_trading_scheduler:
                return jsonify({"error": "Swing Trading Scheduler not available"}), 500

            from swing_trading_scheduler import PurchaseStrategy
            
            # フォームデータ取得
            symbol = request.form['symbol']
            symbol_name = request.form['symbol_name']
            purchase_price = float(request.form['purchase_price'])
            shares = int(request.form['shares'])
            strategy = PurchaseStrategy(request.form['strategy'])
            purchase_reason = request.form['purchase_reason']
            target_profit_percent = float(request.form.get('target_profit_percent', 20.0))
            stop_loss_percent = float(request.form.get('stop_loss_percent', -10.0))
            expected_hold_days = int(request.form.get('expected_hold_days', 30))

            # 購入記録
            purchase_id = swing_trading_scheduler.record_purchase(
                symbol=symbol,
                symbol_name=symbol_name,
                purchase_price=purchase_price,
                shares=shares,
                strategy=strategy,
                purchase_reason=purchase_reason,
                target_profit_percent=target_profit_percent,
                stop_loss_percent=stop_loss_percent,
                expected_hold_days=expected_hold_days
            )

            return jsonify({
                "success": True,
                "purchase_id": purchase_id,
                "message": f"{symbol_name}の購入を記録しました"
            })

        except Exception as e:
            error_msg = str(e)
            print(f"Purchase record error: {error_msg}")
            if audit_logger:
                audit_logger.log_error_with_context(e, {"context": "record_purchase"})
            return jsonify({"success": False, "error": error_msg}), 400

    @app.route('/monitoring')
    def monitoring():
        """監視画面"""
        try:
            if not swing_trading_scheduler:
                return "Swing Trading Scheduler not available", 500

            from swing_trading_scheduler import HoldingStatus
            
            # フィルター取得
            status_filter = request.args.get('status')
            status_enum = HoldingStatus(status_filter) if status_filter else None

            # 監視対象一覧
            monitoring_list = swing_trading_scheduler.get_monitoring_list(status_enum)

            # ステータス選択肢
            status_options = [status.value for status in HoldingStatus]

            return render_template_string(MONITORING_TEMPLATE,
                                        monitoring_list=monitoring_list,
                                        status_options=status_options,
                                        current_filter=status_filter)

        except Exception as e:
            return f"Error: {str(e)}", 500

    @app.route('/partial_sell_form/<purchase_id>')
    def partial_sell_form(purchase_id):
        """部分売却フォーム"""
        try:
            if not swing_trading_scheduler:
                return "Swing Trading Scheduler not available", 500

            # 購入記録取得
            purchase_record = swing_trading_scheduler.get_purchase_record(purchase_id)
            if not purchase_record:
                return "Purchase record not found", 404

            # 現在の保有株数
            current_shares = swing_trading_scheduler._get_current_shares(purchase_id)

            return render_template_string(PARTIAL_SELL_FORM_TEMPLATE,
                                        purchase_record=purchase_record,
                                        current_shares=current_shares)

        except Exception as e:
            return f"Error: {str(e)}", 500

    @app.route('/record_partial_sell', methods=['POST'])
    def record_partial_sell():
        """部分売却記録処理"""
        try:
            if not swing_trading_scheduler:
                return jsonify({"error": "Swing Trading Scheduler not available"}), 500

            # フォームデータ取得
            purchase_id = request.form['purchase_id']
            sell_price = float(request.form['sell_price'])
            shares_sold = int(request.form['shares_sold'])
            sell_reason = request.form['sell_reason']

            # 部分売却記録
            sell_id = swing_trading_scheduler.record_partial_sell(
                purchase_id=purchase_id,
                sell_price=sell_price,
                shares_sold=shares_sold,
                sell_reason=sell_reason
            )

            return jsonify({
                "success": True,
                "sell_id": sell_id,
                "message": f"{shares_sold}株の部分売却を記録しました"
            })

        except Exception as e:
            error_msg = str(e)
            print(f"Partial sell error: {error_msg}")
            return jsonify({"success": False, "error": error_msg}), 400

    @app.route('/alerts')
    def alerts():
        """アラート一覧"""
        try:
            if not swing_trading_scheduler:
                return "Swing Trading Scheduler not available", 500

            # フィルター取得
            unread_only = request.args.get('unread_only', 'false').lower() == 'true'
            limit = int(request.args.get('limit', 50))

            # アラート一覧
            alerts_list = swing_trading_scheduler.get_alerts(limit=limit, unread_only=unread_only)

            return render_template_string(ALERTS_TEMPLATE,
                                        alerts=alerts_list,
                                        unread_only=unread_only)

        except Exception as e:
            return f"Error: {str(e)}", 500

