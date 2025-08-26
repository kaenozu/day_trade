#!/usr/bin/env python3
"""
Webダッシュボード チャートAPIモジュール

チャート生成専用APIエンドポイント
"""

from datetime import datetime

from flask import jsonify, request

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


def setup_chart_routes(app, dashboard_core, visualization_engine, security_manager):
    """チャートAPIルート設定"""

    @app.route("/api/chart/<chart_type>")
    def get_chart(chart_type):
        """チャート生成API"""
        try:
            if not security_manager.validate_chart_type(chart_type):
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "無効なチャートタイプです。",
                            "timestamp": datetime.now().isoformat(),
                        }
                    ),
                    400,
                )

            hours = request.args.get("hours", 12, type=int)
            if not security_manager.validate_hours_parameter(hours):
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "時間パラメータが無効です。1-720時間の範囲で指定してください。",
                            "timestamp": datetime.now().isoformat(),
                        }
                    ),
                    400,
                )

            # チャートタイプ別処理
            chart_path = _generate_chart(
                chart_type, hours, dashboard_core, visualization_engine
            )

            # Base64エンコード
            chart_base64 = visualization_engine.chart_to_base64(chart_path)

            return jsonify(
                {
                    "success": True,
                    "chart_data": chart_base64,
                    "chart_type": chart_type,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        except Exception as e:
            error_message = security_manager.sanitize_error_message(e)
            return (
                jsonify(
                    {
                        "success": False,
                        "error": error_message,
                        "timestamp": datetime.now().isoformat(),
                    }
                ),
                500,
            )


def _generate_chart(chart_type, hours, dashboard_core, visualization_engine):
    """チャート生成ヘルパー関数"""
    if chart_type == "portfolio":
        data = dashboard_core.get_historical_data("portfolio", hours)
        chart_path = visualization_engine.create_portfolio_value_chart(data)
    elif chart_type == "system":
        data = dashboard_core.get_historical_data("system", hours)
        chart_path = visualization_engine.create_system_metrics_chart(data)
    elif chart_type == "trading":
        data = dashboard_core.get_historical_data("trading", hours)
        chart_path = visualization_engine.create_trading_performance_chart(data)
    elif chart_type == "risk":
        data = dashboard_core.get_historical_data("risk", hours)
        chart_path = visualization_engine.create_risk_metrics_heatmap(data)
    elif chart_type == "comprehensive":
        portfolio_data = dashboard_core.get_historical_data("portfolio", hours)
        system_data = dashboard_core.get_historical_data("system", hours)
        trading_data = dashboard_core.get_historical_data("trading", hours)
        risk_data = dashboard_core.get_historical_data("risk", hours)

        # 現在のポジション情報取得
        current_status = dashboard_core.get_current_status()
        positions_data = current_status.get("portfolio", {}).get("positions", {})

        chart_path = visualization_engine.create_comprehensive_dashboard(
            portfolio_data, system_data, trading_data, risk_data, positions_data,
        )
    else:
        raise ValueError(f"Unknown chart type: {chart_type}")

    return chart_path