#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio API Routes Module - ポートフォリオ管理API
ポートフォリオ関連のAPIエンドポイント定義
"""

from flask import Flask, jsonify, request
from datetime import datetime
from typing import Dict, Any, List

# ポートフォリオサービスのインポート
try:
    from web.services.portfolio_service import PortfolioService
    PORTFOLIO_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"ポートフォリオサービス読み込みエラー: {e}")
    PORTFOLIO_SERVICE_AVAILABLE = False

def setup_portfolio_routes(app: Flask) -> None:
    """ポートフォリオAPIルート設定"""

    # ポートフォリオサービス初期化
    portfolio_service = PortfolioService() if PORTFOLIO_SERVICE_AVAILABLE else None

    @app.route('/api/portfolio/summary')
    def api_portfolio_summary():
        """ポートフォリオサマリーAPI"""
        if not PORTFOLIO_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'ポートフォリオサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503

        try:
            summary = portfolio_service.get_portfolio_summary()
            return jsonify({
                'summary': {
                    'total_value': summary.total_value,
                    'total_cost': summary.total_cost,
                    'total_pnl': summary.total_pnl,
                    'total_pnl_pct': summary.total_pnl_pct,
                    'cash_balance': summary.cash_balance,
                    'total_assets': summary.total_assets,
                    'positions_count': summary.positions_count,
                    'sectors_exposure': summary.sectors_exposure,
                    'top_holdings': summary.top_holdings,
                    'daily_change': summary.daily_change,
                    'daily_change_pct': summary.daily_change_pct,
                    'last_updated': summary.last_updated
                },
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/portfolio/positions')
    def api_portfolio_positions():
        """ポートフォリオポジションAPI"""
        if not PORTFOLIO_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'ポートフォリオサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503

        try:
            positions = portfolio_service.get_positions()
            positions_data = []

            for pos in positions:
                positions_data.append({
                    'symbol': pos.symbol,
                    'name': pos.name,
                    'quantity': pos.quantity,
                    'average_price': pos.average_price,
                    'current_price': pos.current_price,
                    'total_value': pos.total_value,
                    'total_cost': pos.total_cost,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                    'sector': pos.sector,
                    'category': pos.category,
                    'holding_days': pos.holding_days,
                    'purchase_date': pos.purchase_date,
                    'last_updated': pos.last_updated
                })

            return jsonify({
                'positions': positions_data,
                'count': len(positions_data),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/portfolio/transactions')
    def api_portfolio_transactions():
        """取引履歴API"""
        if not PORTFOLIO_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'ポートフォリオサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503

        try:
            limit = int(request.args.get('limit', 50))
            transactions = portfolio_service.get_transactions(limit)

            transactions_data = []
            for txn in transactions:
                transactions_data.append({
                    'id': txn.id,
                    'symbol': txn.symbol,
                    'name': txn.name,
                    'action': txn.action,
                    'quantity': txn.quantity,
                    'price': txn.price,
                    'total_amount': txn.total_amount,
                    'commission': txn.commission,
                    'net_amount': txn.net_amount,
                    'date': txn.date,
                    'notes': txn.notes
                })

            return jsonify({
                'transactions': transactions_data,
                'count': len(transactions_data),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/portfolio/buy', methods=['POST'])
    def api_portfolio_buy():
        """株式購入API"""
        if not PORTFOLIO_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'ポートフォリオサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503

        try:
            data = request.get_json()

            required_fields = ['symbol', 'name', 'quantity', 'price']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        'error': f'必須フィールドが不足: {field}',
                        'timestamp': datetime.now().isoformat()
                    }), 400

            success = portfolio_service.add_position(
                symbol=data['symbol'],
                name=data['name'],
                quantity=int(data['quantity']),
                price=float(data['price']),
                sector=data.get('sector', 'Unknown'),
                category=data.get('category', 'Stock')
            )

            if success:
                return jsonify({
                    'success': True,
                    'message': f"{data['symbol']} を {data['quantity']}株 購入しました",
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'error': '購入処理に失敗しました',
                    'timestamp': datetime.now().isoformat()
                }), 500

        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/portfolio/sell', methods=['POST'])
    def api_portfolio_sell():
        """株式売却API"""
        if not PORTFOLIO_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'ポートフォリオサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503

        try:
            data = request.get_json()

            required_fields = ['symbol', 'quantity', 'price']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        'error': f'必須フィールドが不足: {field}',
                        'timestamp': datetime.now().isoformat()
                    }), 400

            success = portfolio_service.sell_position(
                symbol=data['symbol'],
                quantity=int(data['quantity']),
                price=float(data['price'])
            )

            if success:
                return jsonify({
                    'success': True,
                    'message': f"{data['symbol']} を {data['quantity']}株 売却しました",
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'error': '売却処理に失敗しました',
                    'timestamp': datetime.now().isoformat()
                }), 500

        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/portfolio/performance')
    def api_portfolio_performance():
        """パフォーマンス分析API"""
        if not PORTFOLIO_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'ポートフォリオサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503

        try:
            metrics = portfolio_service.get_performance_metrics()
            return jsonify({
                'performance': metrics,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/portfolio/risk')
    def api_portfolio_risk():
        """リスク分析API"""
        if not PORTFOLIO_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'ポートフォリオサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503

        try:
            risk_analysis = portfolio_service.get_risk_analysis()
            return jsonify({
                'risk_analysis': risk_analysis,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/portfolio/update-prices', methods=['POST'])
    def api_portfolio_update_prices():
        """ポートフォリオ価格更新API"""
        if not PORTFOLIO_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'ポートフォリオサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503

        try:
            data = request.get_json()

            if 'prices' not in data:
                return jsonify({
                    'error': '価格データが不足しています',
                    'timestamp': datetime.now().isoformat()
                }), 400

            success = portfolio_service.update_prices(data['prices'])

            return jsonify({
                'success': success,
                'message': '価格を更新しました' if success else '価格更新に失敗しました',
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500