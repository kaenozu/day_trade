#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realtime API Routes Module - リアルタイムAPI
リアルタイム機能関連のAPIエンドポイント定義
"""

from flask import Flask, jsonify, request
from datetime import datetime
from typing import Dict, Any, List

# リアルタイム・リスク・レポートサービスのインポート
try:
    from web.services.realtime_service import RealtimeService, PriceUpdate
    REALTIME_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"リアルタイムサービス読み込みエラー: {e}")
    REALTIME_SERVICE_AVAILABLE = False

try:
    from web.services.risk_management_service import RiskManagementService, RiskLevel, PositionType
    RISK_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"リスク管理サービス読み込みエラー: {e}")
    RISK_SERVICE_AVAILABLE = False

try:
    from web.services.report_service import ReportService, ReportFormat, ReportType
    REPORT_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"レポートサービス読み込みエラー: {e}")
    REPORT_SERVICE_AVAILABLE = False

def setup_realtime_routes(app: Flask) -> None:
    """リアルタイムAPIルート設定"""

    # サービス初期化
    realtime_service = RealtimeService() if REALTIME_SERVICE_AVAILABLE else None
    risk_service = RiskManagementService() if RISK_SERVICE_AVAILABLE else None
    report_service = ReportService() if REPORT_SERVICE_AVAILABLE else None

    # リアルタイムサービス自動開始
    if realtime_service:
        realtime_service.start_service()

    # ===== リアルタイム価格配信 =====

    @app.route('/api/realtime/subscribe', methods=['POST'])
    def api_realtime_subscribe():
        """リアルタイム価格購読"""
        if not REALTIME_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'リアルタイムサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503

        try:
            data = request.get_json()
            symbols = data.get('symbols', [])

            if not symbols:
                return jsonify({
                    'error': 'シンボルが指定されていません',
                    'timestamp': datetime.now().isoformat()
                }), 400

            realtime_service.subscribe_symbols(symbols)

            return jsonify({
                'success': True,
                'subscribed_symbols': symbols,
                'message': f'{len(symbols)}銘柄の価格配信を開始しました',
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/realtime/snapshot')
    def api_realtime_snapshot():
        """市場スナップショット取得"""
        if not REALTIME_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'リアルタイムサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503

        try:
            snapshot = realtime_service.get_market_snapshot()
            return jsonify(snapshot)

        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/realtime/status')
    def api_realtime_status():
        """リアルタイムサービス状態"""
        if not REALTIME_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'リアルタイムサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503

        try:
            status = realtime_service.get_service_status()
            return jsonify(status)

        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    # ===== リスク管理 =====

    @app.route('/api/risk/analyze-trade', methods=['POST'])
    def api_risk_analyze_trade():
        """取引リスク分析"""
        if not RISK_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'リスク管理サービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503

        try:
            data = request.get_json()

            required_fields = ['symbol', 'current_price', 'quantity']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        'error': f'必須フィールドが不足: {field}',
                        'timestamp': datetime.now().isoformat()
                    }), 400

            analysis = risk_service.analyze_trade_risk(
                symbol=data['symbol'],
                current_price=float(data['current_price']),
                quantity=int(data['quantity']),
                stop_loss_price=data.get('stop_loss_price'),
                account_balance=data.get('account_balance', 1000000)
            )

            return jsonify(analysis)

        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/risk/position-sizing', methods=['POST'])
    def api_risk_position_sizing():
        """ポジションサイジング計算"""
        if not RISK_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'リスク管理サービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503

        try:
            data = request.get_json()

            required_fields = ['symbol', 'current_price', 'account_balance']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        'error': f'必須フィールドが不足: {field}',
                        'timestamp': datetime.now().isoformat()
                    }), 400

            # リスクレベル変換
            risk_level_str = data.get('risk_level', 'MODERATE').upper()
            try:
                risk_level = RiskLevel(risk_level_str)
            except ValueError:
                risk_level = RiskLevel.MODERATE

            sizing = risk_service.calculate_position_sizing(
                symbol=data['symbol'],
                current_price=float(data['current_price']),
                account_balance=float(data['account_balance']),
                risk_level=risk_level,
                support_level=data.get('support_level')
            )

            return jsonify(sizing)

        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    # ===== レポート生成 =====

    @app.route('/api/reports/generate', methods=['POST'])
    def api_reports_generate():
        """レポート生成"""
        if not REPORT_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'レポートサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503

        try:
            data = request.get_json()

            report_type = data.get('report_type', 'PORTFOLIO_SUMMARY')
            report_format = data.get('format', 'HTML')

            # フォーマット変換
            try:
                format_enum = ReportFormat(report_format.upper())
            except ValueError:
                format_enum = ReportFormat.HTML

            if report_type == 'PORTFOLIO_SUMMARY':
                # ポートフォリオデータ取得（サンプル）
                portfolio_data = {
                    'total_value': 1250000,
                    'total_pnl': 25000,
                    'total_pnl_pct': 2.04,
                    'positions_count': 5,
                    'positions': [
                        {
                            'symbol': '7203',
                            'quantity': 100,
                            'current_price': 2500,
                            'market_value': 250000,
                            'unrealized_pnl': 12000,
                            'unrealized_pnl_pct': 5.04,
                            'weight': 20.0
                        },
                        {
                            'symbol': '9984',
                            'quantity': 50,
                            'current_price': 8000,
                            'market_value': 400000,
                            'unrealized_pnl': 8000,
                            'unrealized_pnl_pct': 2.04,
                            'weight': 32.0
                        }
                    ],
                    'sector_allocation': {
                        'Technology': 40.0,
                        'Automotive': 30.0,
                        'Financial': 30.0
                    },
                    'value_history': [
                        {'date': '2025-08-01', 'value': 1200000},
                        {'date': '2025-08-10', 'value': 1230000},
                        {'date': '2025-08-19', 'value': 1250000}
                    ],
                    'monthly_returns': {
                        '2025-07': 1.5,
                        '2025-08': 2.1
                    },
                    'top_holdings': [
                        {'symbol': '9984', 'weight': 32.0},
                        {'symbol': '7203', 'weight': 20.0}
                    ]
                }

                result = report_service.generate_portfolio_report(
                    portfolio_data=portfolio_data,
                    format=format_enum
                )

            elif report_type == 'TRADING_JOURNAL':
                # 取引データ取得（サンプル）
                transactions = [
                    {
                        'date': '2025-08-15',
                        'symbol': '7203',
                        'action': 'BUY',
                        'quantity': 100,
                        'price': 2400,
                        'pnl': 0
                    },
                    {
                        'date': '2025-08-18',
                        'symbol': '7203',
                        'action': 'SELL',
                        'quantity': 50,
                        'price': 2500,
                        'pnl': 5000
                    }
                ]

                result = report_service.generate_trading_journal(
                    transactions=transactions,
                    period_days=data.get('period_days', 30)
                )

            else:
                return jsonify({
                    'error': f'未対応のレポートタイプ: {report_type}',
                    'timestamp': datetime.now().isoformat()
                }), 400

            return jsonify(result)

        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/reports/list')
    def api_reports_list():
        """レポート一覧取得"""
        if not REPORT_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'レポートサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503

        try:
            report_type_str = request.args.get('type')
            report_type = None

            if report_type_str:
                try:
                    report_type = ReportType(report_type_str.upper())
                except ValueError:
                    pass

            reports = report_service.list_reports(report_type)

            return jsonify({
                'reports': reports,
                'count': len(reports),
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/reports/<report_id>')
    def api_reports_get(report_id: str):
        """レポート内容取得"""
        if not REPORT_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'レポートサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503

        try:
            content = report_service.get_report_content(report_id)

            if content is None:
                return jsonify({
                    'error': 'レポートが見つかりません',
                    'timestamp': datetime.now().isoformat()
                }), 404

            # ファイル拡張子に基づいてレスポンス形式を決定
            if report_id.endswith('.html'):
                from flask import Response
                return Response(content, mimetype='text/html')
            elif report_id.endswith('.json'):
                from flask import Response
                return Response(content, mimetype='application/json')
            else:
                return jsonify({
                    'content': content,
                    'timestamp': datetime.now().isoformat()
                })

        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    # ===== 統合機能 =====

    @app.route('/api/integrated/market-analysis')
    def api_integrated_market_analysis():
        """統合市場分析"""
        try:
            analysis = {}

            # リアルタイムデータ
            if REALTIME_SERVICE_AVAILABLE:
                try:
                    analysis['market_snapshot'] = realtime_service.get_market_snapshot()
                except Exception as e:
                    analysis['market_snapshot'] = {'error': str(e)}

            # サンプルポートフォリオリスク
            if RISK_SERVICE_AVAILABLE:
                try:
                    # サンプルデータでリスク分析
                    sample_analysis = risk_service.analyze_trade_risk(
                        symbol='7203',
                        current_price=2500,
                        quantity=100,
                        account_balance=1000000
                    )
                    analysis['sample_risk_analysis'] = sample_analysis
                except Exception as e:
                    analysis['sample_risk_analysis'] = {'error': str(e)}

            # 利用可能サービス
            analysis['available_services'] = {
                'realtime': REALTIME_SERVICE_AVAILABLE,
                'risk_management': RISK_SERVICE_AVAILABLE,
                'reports': REPORT_SERVICE_AVAILABLE
            }

            analysis['timestamp'] = datetime.now().isoformat()

            return jsonify(analysis)

        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/integrated/trading-dashboard')
    def api_integrated_trading_dashboard():
        """統合取引ダッシュボード"""
        try:
            dashboard = {
                'timestamp': datetime.now().isoformat(),
                'services_status': {
                    'realtime': REALTIME_SERVICE_AVAILABLE,
                    'risk_management': RISK_SERVICE_AVAILABLE,
                    'reports': REPORT_SERVICE_AVAILABLE
                }
            }

            # リアルタイム価格（利用可能な場合）
            if REALTIME_SERVICE_AVAILABLE:
                try:
                    # デフォルト銘柄を購読
                    default_symbols = ['7203', '9984', '8306']
                    realtime_service.subscribe_symbols(default_symbols)

                    # 少し待機してデータを取得
                    import time
                    time.sleep(1)

                    dashboard['realtime_prices'] = realtime_service.get_market_snapshot()
                except Exception as e:
                    dashboard['realtime_prices'] = {'error': str(e)}

            # リスク評価サンプル
            if RISK_SERVICE_AVAILABLE:
                try:
                    dashboard['risk_recommendations'] = [
                        "現在のリスク水準は適切です",
                        "ポートフォリオの分散を継続してください",
                        "市場ボラティリティに注意してください"
                    ]
                except Exception as e:
                    dashboard['risk_recommendations'] = []

            # 最新レポート
            if REPORT_SERVICE_AVAILABLE:
                try:
                    recent_reports = report_service.list_reports()[:3]
                    dashboard['recent_reports'] = recent_reports
                except Exception as e:
                    dashboard['recent_reports'] = []

            return jsonify(dashboard)

        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    # サービス停止用（デバッグ）
    @app.route('/api/realtime/stop', methods=['POST'])
    def api_realtime_stop():
        """リアルタイムサービス停止"""
        if REALTIME_SERVICE_AVAILABLE and realtime_service:
            realtime_service.stop_service()
            return jsonify({
                'success': True,
                'message': 'リアルタイムサービスを停止しました',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'error': 'リアルタイムサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503