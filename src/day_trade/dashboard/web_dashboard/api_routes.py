#!/usr/bin/env python3
"""
Webダッシュボード APIルートモジュール

API エンドポイント実装
"""

import threading
from datetime import datetime

from flask import jsonify, request

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


def setup_api_routes(app, dashboard_core, visualization_engine, security_manager, 
                    socketio, debug=False):
    """APIルート設定"""

    @app.route("/api/status")
    def get_status():
        """現在のステータス取得API"""
        try:
            status = dashboard_core.get_current_status()
            return jsonify(
                {
                    "success": True,
                    "data": status,
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

    @app.route("/api/history/<metric_type>")
    def get_history(metric_type):
        """過去データ取得API"""
        try:
            if not security_manager.validate_metric_type(metric_type):
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "無効なメトリクスタイプです。",
                            "timestamp": datetime.now().isoformat(),
                        }
                    ),
                    400,
                )

            hours = request.args.get("hours", 24, type=int)
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

            data = dashboard_core.get_historical_data(metric_type, hours)
            return jsonify(
                {
                    "success": True,
                    "data": data,
                    "metric_type": metric_type,
                    "hours": hours,
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


    @app.route("/api/report")
    def get_report():
        """ステータスレポート取得API"""
        try:
            report = dashboard_core.generate_status_report()
            return jsonify(
                {
                    "success": True,
                    "report": report,
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

    @app.route("/api/symbols")
    def get_symbols():
        """銘柄マスタ取得API"""
        try:
            # 遅延インポート
            from ...data.tokyo_stock_symbols import tse

            # 設定ファイルから銘柄情報を取得
            symbols = []
            tier_symbols = {
                1: tse.get_tier1_symbols(),
                2: tse.get_extended_symbols(),
                3: tse.get_comprehensive_symbols(),
                4: tse.get_all_tse_symbols()
            }

            # 基本銘柄リスト（Tier1）
            for symbol in tse.get_tier1_symbols():
                company_info = tse.get_company_info(symbol)
                symbols.append({
                    'code': symbol,
                    'name': company_info['name'] if company_info else symbol
                })

            return jsonify({
                "success": True,
                "symbols": symbols,
                "tier_symbols": tier_symbols,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            error_message = security_manager.sanitize_error_message(e)
            return jsonify({
                "success": False,
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }), 500

    @app.route("/api/tier-symbols/<int:tier>")
    def get_tier_symbols(tier):
        """ティア別銘柄取得API"""
        try:
            if not security_manager.validate_tier_parameter(tier):
                return jsonify({
                    "success": False,
                    "error": "無効なティア番号です（1-4）",
                    "timestamp": datetime.now().isoformat()
                }), 400

            # 遅延インポート
            from ...data.tokyo_stock_symbols import tse

            if tier == 1:
                symbols = tse.get_tier1_symbols()
            elif tier == 2:
                symbols = tse.get_extended_symbols()
            elif tier == 3:
                symbols = tse.get_comprehensive_symbols()
            elif tier == 4:
                symbols = tse.get_all_tse_symbols()

            return jsonify({
                "success": True,
                "symbols": symbols,
                "tier": tier,
                "count": len(symbols),
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            error_message = security_manager.sanitize_error_message(e)
            return jsonify({
                "success": False,
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }), 500

    @app.route("/api/analysis", methods=['POST'])
    def run_analysis():
        """分析実行API"""
        try:
            data = request.get_json()
            symbols = data.get('symbols', [])
            mode = data.get('mode', 'web')

            if not symbols:
                return jsonify({
                    "success": False,
                    "error": "分析する銘柄を指定してください",
                    "timestamp": datetime.now().isoformat()
                }), 400

            # バックグラウンドで分析実行
            def run_background_analysis():
                try:
                    # 分析アプリを遅延初期化
                    from ....core.application import StockAnalysisApplication
                    analysis_app = StockAnalysisApplication(debug=debug)

                    # 進捗通知
                    socketio.emit('analysis_progress', {
                        'current': 0,
                        'total': len(symbols),
                        'currentSymbol': ''
                    })

                    results = []
                    for i, symbol in enumerate(symbols):
                        # 進捗更新
                        socketio.emit('analysis_progress', {
                            'current': i,
                            'total': len(symbols),
                            'currentSymbol': symbol
                        })

                        # 個別銘柄分析
                        result = analysis_app._analyze_symbol_with_ai(symbol)
                        results.append(result)

                    # 分析完了通知
                    socketio.emit('analysis_complete', {
                        'results': results,
                        'timestamp': datetime.now().isoformat()
                    })

                except Exception as e:
                    logger.error(f"バックグラウンド分析エラー: {e}")
                    socketio.emit('analysis_error', {
                        'message': str(e),
                        'timestamp': datetime.now().isoformat()
                    })

            # バックグラウンドスレッド開始
            analysis_thread = threading.Thread(
                target=run_background_analysis, daemon=True
            )
            analysis_thread.start()

            return jsonify({
                "success": True,
                "message": f"{len(symbols)}銘柄の分析を開始しました",
                "symbol_count": len(symbols),
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            error_message = security_manager.sanitize_error_message(e)
            return jsonify({
                "success": False,
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }), 500


