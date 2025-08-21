#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Routes Module - Issues #953-956対応 クリーンアップ版
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
                'Production Ready (Gunicorn)',
                'Portfolio Management',
                'Alert System',
                'Backtest Engine',
                'Database Integration',
                'Risk Management',
                'Report Generation',
                'Real-time Data Feed',
                'Position Sizing',
                'Filtering & Search',
                'Swing Trading Support',
                'Scheduler Integration'
            ]
        })

    @app.route('/api/recommendations')
    def api_recommendations():
        """推奨銘柄API（フィルタリング対応）"""
        try:
            # フィルターパラメータ取得
            category_filter = request.args.get('category', '').lower()
            confidence_filter = request.args.get('confidence', '').lower()
            recommendation_filter = request.args.get('recommendation', '').lower()

            recommendations = g.recommendation_service.get_recommendations()

            # フィルタリング適用
            filtered_recommendations = recommendations

            if category_filter:
                filtered_recommendations = [r for r in filtered_recommendations
                                         if r.get('category', '').lower() == category_filter]

            if confidence_filter == 'high':
                filtered_recommendations = [r for r in filtered_recommendations
                                         if r.get('confidence', 0) > 0.8]
            elif confidence_filter == 'medium':
                filtered_recommendations = [r for r in filtered_recommendations
                                         if 0.6 <= r.get('confidence', 0) <= 0.8]
            elif confidence_filter == 'low':
                filtered_recommendations = [r for r in filtered_recommendations
                                         if r.get('confidence', 0) < 0.6]

            if recommendation_filter == 'buy':
                filtered_recommendations = [r for r in filtered_recommendations
                                         if r.get('recommendation', '') in ['BUY', 'STRONG_BUY']]
            elif recommendation_filter == 'sell':
                filtered_recommendations = [r for r in filtered_recommendations
                                         if r.get('recommendation', '') in ['SELL', 'STRONG_SELL']]
            elif recommendation_filter == 'hold':
                filtered_recommendations = [r for r in filtered_recommendations
                                         if r.get('recommendation', '') == 'HOLD']

            # 統計計算（全体とフィルタ後）
            total_count = len(recommendations)
            filtered_count = len(filtered_recommendations)
            high_confidence_count = len([r for r in recommendations if r.get('confidence', 0) > 0.8])
            buy_count = len([r for r in recommendations if r.get('recommendation') in ['BUY', 'STRONG_BUY']])
            sell_count = len([r for r in recommendations if r.get('recommendation') in ['SELL', 'STRONG_SELL']])
            hold_count = len([r for r in recommendations if r.get('recommendation') == 'HOLD'])

            return jsonify({
                'total_count': total_count,
                'filtered_count': filtered_count,
                'high_confidence_count': high_confidence_count,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'hold_count': hold_count,
                'recommendations': filtered_recommendations,
                'applied_filters': {
                    'category': category_filter or None,
                    'confidence': confidence_filter or None,
                    'recommendation': recommendation_filter or None
                },
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            return jsonify({'error': str(e), 'timestamp': datetime.now().isoformat()}), 500

    @app.route('/api/portfolio/add', methods=['POST'])
    def api_portfolio_add():
        """ポートフォリオに銘柄を追加するAPI"""
        try:
            data = request.get_json()

            if not data or 'symbol' not in data:
                return jsonify({
                    'error': '銘柄コードが必要です',
                    'timestamp': datetime.now().isoformat()
                }), 400

            # 簡易的なファイルベース保存（実際のプロダクションではデータベースを使用）
            import json
            import os
            portfolio_file = 'data/portfolio.json'

            # ディレクトリ作成
            os.makedirs(os.path.dirname(portfolio_file), exist_ok=True)

            # 既存のポートフォリオを読み込み
            portfolio = []
            if os.path.exists(portfolio_file):
                with open(portfolio_file, 'r', encoding='utf-8') as f:
                    portfolio = json.load(f)

            # 重複チェック
            if any(item['symbol'] == data['symbol'] for item in portfolio):
                return jsonify({
                    'error': 'この銘柄は既にポートフォリオに追加されています',
                    'symbol': data['symbol'],
                    'timestamp': datetime.now().isoformat()
                }), 409

            # 新しい銘柄を追加
            portfolio_item = {
                'id': len(portfolio) + 1,
                'symbol': data['symbol'],
                'name': data['name'],
                'price': data['price'],
                'buy_timing': data.get('buy_timing', ''),
                'sell_timing': data.get('sell_timing', ''),
                'added_date': datetime.now().isoformat(),
                'status': 'watching',  # watching, bought, sold
                'notes': ''
            }

            portfolio.append(portfolio_item)

            # ファイルに保存
            with open(portfolio_file, 'w', encoding='utf-8') as f:
                json.dump(portfolio, f, ensure_ascii=False, indent=2)

            return jsonify({
                'message': 'ポートフォリオに追加しました',
                'item': portfolio_item,
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/portfolio', methods=['GET'])
    def api_portfolio_list():
        """ポートフォリオ一覧を取得するAPI"""
        try:
            import json
            import os
            portfolio_file = 'data/portfolio.json'

            portfolio = []
            if os.path.exists(portfolio_file):
                with open(portfolio_file, 'r', encoding='utf-8') as f:
                    portfolio = json.load(f)

            return jsonify({
                'portfolio': portfolio,
                'count': len(portfolio),
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
                'timestamp': datetime.now().isoformat()
            }), 500
    @app.route('/api/portfolio/<int:item_id>', methods=['PUT'])
    def api_portfolio_update(item_id):
        """ポートフォリオの銘柄情報を更新するAPI"""
        try:
            data = request.get_json()

            import json
            import os
            portfolio_file = 'data/portfolio.json'

            if not os.path.exists(portfolio_file):
                return jsonify({
                    'error': 'ポートフォリオが見つかりません',
                    'timestamp': datetime.now().isoformat()
                }), 404

            # ポートフォリオを読み込み
            with open(portfolio_file, 'r', encoding='utf-8') as f:
                portfolio = json.load(f)

            # 対象のアイテムを検索
            item_index = None
            for i, item in enumerate(portfolio):
                if item['id'] == item_id:
                    item_index = i
                    break

            if item_index is None:
                return jsonify({
                    'error': '指定された銘柄が見つかりません',
                    'timestamp': datetime.now().isoformat()
                }), 404

            # データを更新
            if 'sell_timing' in data:
                portfolio[item_index]['sell_timing'] = data['sell_timing']
            if 'buy_timing' in data:
                portfolio[item_index]['buy_timing'] = data['buy_timing']
            if 'status' in data:
                portfolio[item_index]['status'] = data['status']
            if 'notes' in data:
                portfolio[item_index]['notes'] = data['notes']

            portfolio[item_index]['updated_date'] = datetime.now().isoformat()

            # ファイルに保存
            with open(portfolio_file, 'w', encoding='utf-8') as f:
                json.dump(portfolio, f, ensure_ascii=False, indent=2)

            return jsonify({
                'message': '銘柄情報を更新しました',
                'item': portfolio[item_index],
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/portfolio/<int:item_id>', methods=['DELETE'])
    def api_portfolio_delete(item_id):
        """ポートフォリオから銘柄を削除するAPI"""
        try:
            import json
            import os
            portfolio_file = 'data/portfolio.json'

            if not os.path.exists(portfolio_file):
                return jsonify({
                    'error': 'ポートフォリオが見つかりません',
                    'timestamp': datetime.now().isoformat()
                }), 404

            # ポートフォリオを読み込み
            with open(portfolio_file, 'r', encoding='utf-8') as f:
                portfolio = json.load(f)

            # 対象のアイテムを削除
            portfolio = [item for item in portfolio if item['id'] != item_id]

            # ファイルに保存
            with open(portfolio_file, 'w', encoding='utf-8') as f:
                json.dump(portfolio, f, ensure_ascii=False, indent=2)

            return jsonify({
                'message': '銘柄をポートフォリオから削除しました',
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            return jsonify({
                'error': str(e),
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
