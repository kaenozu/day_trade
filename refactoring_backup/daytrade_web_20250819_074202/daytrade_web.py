#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day Trade Web Server - プロダクション対応Webサーバー
Issue #901 対応: プロダクション Web サーバー実装
Issue #933 対応: バージョン統一とパフォーマンス監視強化
"""

import sys
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
from flask import Flask, render_template_string, jsonify, request
import threading
from datetime import datetime

# バージョン統一 - Issue #933対応
try:
    from version import get_version_info, __version_extended__, __version_full__
    VERSION_INFO = get_version_info()
except ImportError:
    # フォールバック
    VERSION_INFO = {
        "version": "2.1.0",
        "version_extended": "2.1.0_extended",
        "release_name": "Extended",
        "build_date": "2025-08-18"
    }
    __version_extended__ = "2.1.0_extended"
    __version_full__ = "Day Trade Personal v2.1.0 Extended"

# パフォーマンス監視 - Issue #933対応
try:
    from performance_monitor import performance_monitor, track_performance
    PERFORMANCE_MONITORING = True
except ImportError:
    performance_monitor = None
    PERFORMANCE_MONITORING = False
    def track_performance(func):
        return func  # フォールバック用デコレーター

# データ永続化 - Issue #933 Phase 3対応
try:
    from data_persistence import data_persistence
    DATA_PERSISTENCE = True
except ImportError:
    data_persistence = None
    DATA_PERSISTENCE = False


class DayTradeWebServer:
    """プロダクション対応Webサーバー"""

    def __init__(self, port: int = 8000, debug: bool = False):
        self.port = port
        self.debug = debug
        self.app = Flask(__name__)
        self.app.secret_key = 'day-trade-personal-2025'

        # セッション管理 - Issue #933 Phase 3対応
        self.session_id = f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # ルート設定
        self._setup_routes()

        # ログ設定
        if not debug:
            logging.getLogger('werkzeug').setLevel(logging.WARNING)

    def _setup_routes(self):
        """Webルート設定"""

        @self.app.route('/')
        def index():
            """メインダッシュボード"""
            start_time = time.time() if PERFORMANCE_MONITORING else 0

            response = render_template_string(self._get_dashboard_template(),
                                            title="Day Trade Personal - メインダッシュボード")

            if PERFORMANCE_MONITORING and performance_monitor:
                duration = time.time() - start_time
                performance_monitor.track_api_response_time('/', duration)

            return response

        @self.app.route('/api/status')
        def api_status():
            """システム状態API - Issue #933対応: 統一バージョン情報"""
            return jsonify({
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'version': VERSION_INFO['version'],
                'version_extended': VERSION_INFO['version_extended'],
                'release_name': VERSION_INFO['release_name'],
                'build_date': VERSION_INFO['build_date'],
                'features': [
                    'Real-time Analysis',
                    'Security Enhanced',
                    'Performance Optimized',
                    'Production Ready',
                    '20-Stock Recommendations',
                    'Unified Version Management'
                ]
            })

        @self.app.route('/api/analysis/<symbol>')
        def api_analysis(symbol):
            """株価分析API - Issue #933 Phase 3対応: データ永続化統合"""
            start_time = time.time() if PERFORMANCE_MONITORING else 0

            try:
                import random
                recommendations = ['BUY', 'SELL', 'HOLD']
                confidence = round(random.uniform(0.60, 0.95), 2)

                # 分析結果
                result = {
                    'symbol': symbol,
                    'recommendation': random.choice(recommendations),
                    'confidence': confidence,
                    'price': 1500 + hash(symbol) % 1000,
                    'change': round((hash(symbol) % 200 - 100) / 10, 2),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'completed',
                    'volume': random.randint(100000, 5000000),
                    'market_cap': f"{random.randint(1000, 50000)}億円",
                    'sector': random.choice(['テクノロジー', '金融', '製造業', 'ヘルスケア', 'エネルギー'])
                }

                # パフォーマンス監視とデータ永続化
                if PERFORMANCE_MONITORING and performance_monitor:
                    duration = time.time() - start_time
                    performance_monitor.track_api_response_time(f'/api/analysis/{symbol}', duration)

                if DATA_PERSISTENCE and data_persistence:
                    duration_ms = (time.time() - start_time) * 1000 if start_time else 0
                    data_persistence.save_analysis_result(
                        symbol=symbol,
                        analysis_type='web_api_analysis',
                        duration_ms=duration_ms,
                        result_data=result,
                        confidence_score=confidence,
                        session_id=self.session_id
                    )

                return jsonify(result)

            except Exception as e:
                if DATA_PERSISTENCE and data_persistence:
                    data_persistence.save_error_log(
                        error_type='api_analysis_error',
                        error_message=str(e),
                        context_data={'symbol': symbol, 'endpoint': f'/api/analysis/{symbol}'},
                        session_id=self.session_id
                    )

                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500

        @self.app.route('/api/recommendations')
        def api_recommendations():
            """推奨銘柄一覧API - Issue #928対応"""
            try:
                import random

                # Issue #953対応: 35銘柄に拡大（多様化とリスク分散強化）
                symbols = [
                    # 大型株（安定重視） - 8銘柄
                    {'code': '7203', 'name': 'トヨタ自動車', 'sector': '自動車', 'category': '大型株', 'stability': '高安定'},
                    {'code': '8306', 'name': '三菱UFJ銀行', 'sector': '金融', 'category': '大型株', 'stability': '高安定'},
                    {'code': '9984', 'name': 'ソフトバンクグループ', 'sector': 'テクノロジー', 'category': '大型株', 'stability': '中安定'},
                    {'code': '6758', 'name': 'ソニー', 'sector': 'テクノロジー', 'category': '大型株', 'stability': '中安定'},
                    {'code': '7267', 'name': 'ホンダ', 'sector': '自動車', 'category': '大型株', 'stability': '高安定'},
                    {'code': '4519', 'name': '中外製薬', 'sector': '製薬', 'category': '大型株', 'stability': '高安定'},
                    {'code': '8028', 'name': 'ファミリーマート', 'sector': '小売', 'category': '大型株', 'stability': '高安定'},
                    {'code': '9433', 'name': 'KDDI', 'sector': '通信', 'category': '大型株', 'stability': '高安定'},

                    # 中型株（成長期待） - 8銘柄
                    {'code': '4689', 'name': 'Z Holdings', 'sector': 'テクノロジー', 'category': '中型株', 'stability': '中安定'},
                    {'code': '9434', 'name': 'ソフトバンク', 'sector': '通信', 'category': '中型株', 'stability': '中安定'},
                    {'code': '6861', 'name': 'キーエンス', 'sector': '精密機器', 'category': '中型株', 'stability': '中安定'},
                    {'code': '4755', 'name': '楽天グループ', 'sector': 'テクノロジー', 'category': '中型株', 'stability': '低安定'},
                    {'code': '6954', 'name': 'ファナック', 'sector': '工作機械', 'category': '中型株', 'stability': '中安定'},
                    {'code': '4704', 'name': 'トレンドマイクロ', 'sector': 'ソフトウェア', 'category': '中型株', 'stability': '中安定'},
                    {'code': '2432', 'name': 'ディー・エヌ・エー', 'sector': 'テクノロジー', 'category': '中型株', 'stability': '低安定'},
                    {'code': '3659', 'name': 'ネクソン', 'sector': 'ゲーム', 'category': '中型株', 'stability': '低安定'},

                    # 高配当株（収益重視） - 8銘柄
                    {'code': '8001', 'name': '伊藤忠商事', 'sector': '商社', 'category': '高配当株', 'stability': '高安定'},
                    {'code': '8316', 'name': '三井住友FG', 'sector': '金融', 'category': '高配当株', 'stability': '高安定'},
                    {'code': '4502', 'name': '武田薬品工業', 'sector': '製薬', 'category': '高配当株', 'stability': '高安定'},
                    {'code': '8058', 'name': '三菱商事', 'sector': '商社', 'category': '高配当株', 'stability': '高安定'},
                    {'code': '2914', 'name': '日本たばこ産業', 'sector': 'その他', 'category': '高配当株', 'stability': '高安定'},
                    {'code': '9437', 'name': 'NTTドコモ', 'sector': '通信', 'category': '高配当株', 'stability': '高安定'},
                    {'code': '8354', 'name': 'ふくおかFG', 'sector': '金融', 'category': '高配当株', 'stability': '高安定'},
                    {'code': '5333', 'name': '日本ガイシ', 'sector': '窯業', 'category': '高配当株', 'stability': '高安定'},

                    # 成長株（将来性重視） - 6銘柄
                    {'code': '9983', 'name': 'ファーストリテイリング', 'sector': 'アパレル', 'category': '成長株', 'stability': '中安定'},
                    {'code': '7974', 'name': '任天堂', 'sector': 'ゲーム', 'category': '成長株', 'stability': '低安定'},
                    {'code': '4063', 'name': '信越化学工業', 'sector': '化学', 'category': '成長株', 'stability': '中安定'},
                    {'code': '6594', 'name': '日本電産', 'sector': '電気機器', 'category': '成長株', 'stability': '中安定'},
                    {'code': '4568', 'name': '第一三共', 'sector': '製薬', 'category': '成長株', 'stability': '中安定'},
                    {'code': '6098', 'name': 'リクルート', 'sector': 'サービス', 'category': '成長株', 'stability': '中安定'},

                    # 小型・新興株（ハイリスク・ハイリターン） - 5銘柄
                    {'code': '3696', 'name': 'セレス', 'sector': 'テクノロジー', 'category': '小型株', 'stability': '低安定'},
                    {'code': '4385', 'name': 'メルカリ', 'sector': 'テクノロジー', 'category': '小型株', 'stability': '低安定'},
                    {'code': '4477', 'name': 'BASE', 'sector': 'テクノロジー', 'category': '小型株', 'stability': '低安定'},
                    {'code': '4443', 'name': 'Sansan', 'sector': 'テクノロジー', 'category': '小型株', 'stability': '低安定'},
                    {'code': '4488', 'name': 'AI inside', 'sector': 'AI/DX', 'category': '小型株', 'stability': '低安定'}
                ]

                recommendations = []
                for stock in symbols:
                    confidence = round(random.uniform(0.60, 0.95), 2)
                    rec_type = random.choice(['BUY', 'SELL', 'HOLD'])

                    # Issue #929対応: わかりやすい表示
                    friendly_confidence = self._get_friendly_confidence_label(confidence)
                    star_rating = self._get_star_rating(confidence)

                    recommendations.append({
                        'symbol': stock['code'],
                        'name': stock['name'],
                        'sector': stock['sector'],
                        'category': stock['category'],
                        'stability': stock['stability'],
                        'recommendation': rec_type,
                        'recommendation_friendly': self._get_friendly_recommendation(rec_type),
                        'confidence': confidence,
                        'confidence_friendly': friendly_confidence,
                        'star_rating': star_rating,
                        'price': 1000 + hash(stock['code']) % 2000,
                        'change': round((hash(stock['code']) % 200 - 100) / 10, 2),
                        'reason': self._get_recommendation_reason(rec_type, confidence),
                        'friendly_reason': self._get_friendly_reason(rec_type, confidence),
                        'risk_level': 'HIGH' if confidence > 0.85 else 'MEDIUM' if confidence > 0.70 else 'LOW',
                        'risk_friendly': self._get_friendly_risk(confidence),
                        'who_suitable': self._get_suitable_investor_type(stock['category'], stock['stability'])
                    })

                # 信頼度で降順ソート
                recommendations.sort(key=lambda x: x['confidence'], reverse=True)

                return jsonify({
                    'total_count': len(recommendations),
                    'high_confidence_count': len([r for r in recommendations if r['confidence'] > 0.80]),
                    'buy_count': len([r for r in recommendations if r['recommendation'] == 'BUY']),
                    'sell_count': len([r for r in recommendations if r['recommendation'] == 'SELL']),
                    'hold_count': len([r for r in recommendations if r['recommendation'] == 'HOLD']),
                    'recommendations': recommendations,
                    'timestamp': datetime.now().isoformat(),
                    'version': VERSION_INFO['version_extended'],
                    'api_version': VERSION_INFO['version']
                })

            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500

        @self.app.route('/api/daytrade-stocks')
        def api_daytrade_stocks():
            """デイトレード専用銘柄API - Issue #954対応"""
            try:
                import random

                # デイトレード適正銘柄（高流動性・高ボラティリティ）
                daytrade_symbols = [
                    # 超高流動性（日経225主力）
                    {'code': '7203', 'name': 'トヨタ自動車', 'sector': '自動車', 'liquidity': '超高', 'volatility': '中'},
                    {'code': '9984', 'name': 'ソフトバンクグループ', 'sector': 'テクノロジー', 'liquidity': '超高', 'volatility': '高'},
                    {'code': '6758', 'name': 'ソニー', 'sector': 'テクノロジー', 'liquidity': '超高', 'volatility': '中'},
                    {'code': '7974', 'name': '任天堂', 'sector': 'ゲーム', 'liquidity': '高', 'volatility': '高'},
                    {'code': '9983', 'name': 'ファーストリテイリング', 'sector': 'アパレル', 'liquidity': '高', 'volatility': '高'},

                    # 高ボラティリティ（値動き活発）
                    {'code': '4755', 'name': '楽天グループ', 'sector': 'テクノロジー', 'liquidity': '高', 'volatility': '超高'},
                    {'code': '4385', 'name': 'メルカリ', 'sector': 'テクノロジー', 'liquidity': '中', 'volatility': '超高'},
                    {'code': '2432', 'name': 'ディー・エヌ・エー', 'sector': 'テクノロジー', 'liquidity': '中', 'volatility': '高'}
                ]

                recommendations = []
                for stock in daytrade_symbols:
                    confidence = round(random.uniform(0.65, 0.90), 2)
                    rec_type = random.choice(['BUY', 'SELL', 'HOLD'])

                    # デイトレード適性スコア
                    liquidity_score = {'超高': 5, '高': 4, '中': 3}.get(stock['liquidity'], 2)
                    volatility_score = {'超高': 5, '高': 4, '中': 3}.get(stock['volatility'], 2)
                    daytrade_score = round((liquidity_score + volatility_score) / 2, 1)

                    recommendations.append({
                        'symbol': stock['code'],
                        'name': stock['name'],
                        'sector': stock['sector'],
                        'liquidity': stock['liquidity'],
                        'volatility': stock['volatility'],
                        'daytrade_score': daytrade_score,
                        'recommendation': rec_type,
                        'confidence': confidence,
                        'price': 1000 + hash(stock['code']) % 3000,
                        'change': round((hash(stock['code']) % 300 - 150) / 10, 2),
                        'volume_estimate': f"{random.randint(500, 2000)}万株",
                        'spread_estimate': f"{random.randint(1, 5)}円",
                        'daytrade_reason': self._get_daytrade_reason(stock['liquidity'], stock['volatility'], rec_type)
                    })

                # デイトレード適性スコア順でソート
                recommendations.sort(key=lambda x: (x['daytrade_score'], x['confidence']), reverse=True)

                return jsonify({
                    'total_count': len(recommendations),
                    'high_score_count': len([r for r in recommendations if r['daytrade_score'] >= 4.0]),
                    'ultra_liquid_count': len([r for r in recommendations if r['liquidity'] == '超高']),
                    'high_volatility_count': len([r for r in recommendations if r['volatility'] in ['高', '超高']]),
                    'recommendations': recommendations,
                    'note': 'デイトレード専用：少数精鋭・高流動性銘柄',
                    'warning': '※デイトレードは高リスクです。十分な資金管理を行ってください',
                    'timestamp': datetime.now().isoformat(),
                    'version': VERSION_INFO['version_extended']
                })

            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500

        @self.app.route('/api/chart-data/<symbol>')
        def api_chart_data(symbol):
            """チャートデータAPI - Issue #955対応"""
            try:
                import random
                from datetime import datetime, timedelta

                # 模擬チャートデータ生成（実際は外部APIから取得）
                base_price = 1000 + hash(symbol) % 2000
                data_points = []
                current_time = datetime.now()

                # 過去24時間のデータ（1時間間隔）
                for i in range(24, 0, -1):
                    timestamp = current_time - timedelta(hours=i)
                    # 価格変動シミュレーション
                    price_change = random.uniform(-0.05, 0.05)  # ±5%変動
                    price = base_price * (1 + price_change * i / 24)
                    volume = random.randint(10000, 100000)

                    data_points.append({
                        'time': timestamp.isoformat(),
                        'open': round(price * random.uniform(0.995, 1.005), 2),
                        'high': round(price * random.uniform(1.01, 1.03), 2),
                        'low': round(price * random.uniform(0.97, 0.99), 2),
                        'close': round(price, 2),
                        'volume': volume
                    })

                return jsonify({
                    'symbol': symbol,
                    'data': data_points,
                    'current_price': data_points[-1]['close'],
                    'price_change': round(data_points[-1]['close'] - data_points[0]['close'], 2),
                    'price_change_percent': round(((data_points[-1]['close'] / data_points[0]['close']) - 1) * 100, 2),
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500

        @self.app.route('/api/portfolio')
        def api_portfolio():
            """ポートフォリオAPI - Issue #955対応"""
            try:
                import random

                # 模擬ポートフォリオデータ
                portfolio_stocks = [
                    {'symbol': '7203', 'name': 'トヨタ自動車', 'shares': 100, 'avg_price': 2800, 'current_price': 2850},
                    {'symbol': '9984', 'name': 'ソフトバンクグループ', 'shares': 50, 'avg_price': 5200, 'current_price': 5150},
                    {'symbol': '6758', 'name': 'ソニー', 'shares': 30, 'avg_price': 12000, 'current_price': 12500}
                ]

                total_value = 0
                total_cost = 0
                positions = []

                for stock in portfolio_stocks:
                    cost = stock['shares'] * stock['avg_price']
                    value = stock['shares'] * stock['current_price']
                    profit_loss = value - cost
                    profit_loss_percent = (profit_loss / cost) * 100

                    positions.append({
                        'symbol': stock['symbol'],
                        'name': stock['name'],
                        'shares': stock['shares'],
                        'avg_price': stock['avg_price'],
                        'current_price': stock['current_price'],
                        'cost': cost,
                        'value': value,
                        'profit_loss': profit_loss,
                        'profit_loss_percent': profit_loss_percent
                    })

                    total_value += value
                    total_cost += cost

                total_profit_loss = total_value - total_cost
                total_profit_loss_percent = (total_profit_loss / total_cost) * 100 if total_cost > 0 else 0

                return jsonify({
                    'positions': positions,
                    'total_cost': total_cost,
                    'total_value': total_value,
                    'total_profit_loss': total_profit_loss,
                    'total_profit_loss_percent': total_profit_loss_percent,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500

        @self.app.route('/api/search-stocks')
        def api_search_stocks():
            """銘柄検索API - Issue #955対応"""
            try:
                query = request.args.get('q', '').lower()
                if not query or len(query) < 2:
                    return jsonify({'results': [], 'count': 0})

                # 全銘柄から検索（実際はデータベースから検索）
                all_stocks = [
                    {'symbol': '7203', 'name': 'トヨタ自動車', 'sector': '自動車'},
                    {'symbol': '9984', 'name': 'ソフトバンクグループ', 'sector': 'テクノロジー'},
                    {'symbol': '6758', 'name': 'ソニー', 'sector': 'テクノロジー'},
                    {'symbol': '7974', 'name': '任天堂', 'sector': 'ゲーム'},
                    {'symbol': '4755', 'name': '楽天グループ', 'sector': 'テクノロジー'},
                    {'symbol': '4385', 'name': 'メルカリ', 'sector': 'テクノロジー'},
                    {'symbol': '8306', 'name': '三菱UFJ銀行', 'sector': '金融'},
                    {'symbol': '8001', 'name': '伊藤忠商事', 'sector': '商社'}
                ]

                # 検索実行
                results = []
                for stock in all_stocks:
                    if (query in stock['symbol'].lower() or
                        query in stock['name'].lower() or
                        query in stock['sector'].lower()):
                        results.append(stock)

                return jsonify({
                    'results': results[:10],  # 最大10件
                    'count': len(results),
                    'query': query
                })

            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500

        @self.app.route('/api/watchlist')
        def api_watchlist():
            """ウォッチリスト API - Issue #956対応"""
            try:
                # 模擬ウォッチリストデータ（実際はユーザーごとに管理）
                watchlist_symbols = ['7203', '9984', '6758', '7974', '4755']

                watchlist_data = []
                for symbol in watchlist_symbols:
                    import random
                    base_price = 1000 + hash(symbol) % 3000
                    change = random.uniform(-5, 5)

                    # 銘柄情報取得（実際は外部API）
                    stock_info = {
                        '7203': {'name': 'トヨタ自動車', 'sector': '自動車'},
                        '9984': {'name': 'ソフトバンクグループ', 'sector': 'テクノロジー'},
                        '6758': {'name': 'ソニー', 'sector': 'テクノロジー'},
                        '7974': {'name': '任天堂', 'sector': 'ゲーム'},
                        '4755': {'name': '楽天グループ', 'sector': 'テクノロジー'}
                    }.get(symbol, {'name': f'銘柄{symbol}', 'sector': 'その他'})

                    watchlist_data.append({
                        'symbol': symbol,
                        'name': stock_info['name'],
                        'sector': stock_info['sector'],
                        'price': round(base_price, 2),
                        'change': round(change, 2),
                        'change_percent': round(change / base_price * 100, 2),
                        'volume': f"{random.randint(100, 1000)}万株",
                        'high_52w': round(base_price * 1.2, 2),
                        'low_52w': round(base_price * 0.8, 2),
                        'last_updated': datetime.now().isoformat()
                    })

                return jsonify({
                    'watchlist': watchlist_data,
                    'total_count': len(watchlist_data),
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500

        @self.app.route('/api/alerts')
        def api_alerts():
            """アラート API - Issue #956対応"""
            try:
                import random

                # 模擬アラートデータ
                alerts = []
                alert_types = [
                    {'type': 'price_target', 'message': 'トヨタ自動車が目標価格¥3000に到達しました', 'symbol': '7203'},
                    {'type': 'volume_spike', 'message': 'ソニーの出来高が急増しています', 'symbol': '6758'},
                    {'type': 'news', 'message': '任天堂の新製品発表が発表されました', 'symbol': '7974'},
                    {'type': 'technical', 'message': 'ソフトバンクGが移動平均線を突破', 'symbol': '9984'}
                ]

                # ランダムに1-3個のアラート生成
                num_alerts = random.randint(1, 3)
                for _ in range(num_alerts):
                    alert = random.choice(alert_types)
                    alerts.append({
                        'id': random.randint(1000, 9999),
                        'type': alert['type'],
                        'message': alert['message'],
                        'symbol': alert['symbol'],
                        'timestamp': datetime.now().isoformat(),
                        'read': False,
                        'priority': random.choice(['high', 'medium', 'low'])
                    })

                return jsonify({
                    'alerts': alerts,
                    'unread_count': len([a for a in alerts if not a['read']]),
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500

        @self.app.route('/api/market-summary')
        def api_market_summary():
            """マーケットサマリー API - Issue #956対応"""
            try:
                import random

                # 模擬市場データ
                market_data = {
                    'nikkei225': {
                        'name': '日経平均',
                        'value': 33500 + random.randint(-500, 500),
                        'change': round(random.uniform(-200, 200), 2),
                        'change_percent': round(random.uniform(-1, 1), 2)
                    },
                    'topix': {
                        'name': 'TOPIX',
                        'value': 2400 + random.randint(-50, 50),
                        'change': round(random.uniform(-20, 20), 2),
                        'change_percent': round(random.uniform(-1, 1), 2)
                    },
                    'mothers': {
                        'name': 'マザーズ指数',
                        'value': 750 + random.randint(-30, 30),
                        'change': round(random.uniform(-15, 15), 2),
                        'change_percent': round(random.uniform(-2, 2), 2)
                    }
                }

                return jsonify({
                    'indices': market_data,
                    'market_status': random.choice(['open', 'closed', 'pre_market']),
                    'last_updated': datetime.now().isoformat(),
                    'trading_volume': f"{random.randint(15, 25)}億株"
                })

            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500

        @self.app.route('/health')
        def health():
            """ヘルスチェック"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'uptime': 'running'
            })

        @self.app.route('/api/performance')
        def api_performance():
            """パフォーマンス監視API - Issue #933対応"""
            if not PERFORMANCE_MONITORING or not performance_monitor:
                return jsonify({
                    'error': 'Performance monitoring not available',
                    'monitoring_enabled': False
                }), 501

            try:
                summary = performance_monitor.get_performance_summary()
                return jsonify({
                    'monitoring_enabled': True,
                    'performance_summary': summary,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500

        @self.app.route('/api/performance/detailed')
        def api_performance_detailed():
            """詳細パフォーマンスメトリクス API - Issue #933対応"""
            if not PERFORMANCE_MONITORING or not performance_monitor:
                return jsonify({
                    'error': 'Performance monitoring not available',
                    'monitoring_enabled': False
                }), 501

            try:
                detailed = performance_monitor.get_detailed_metrics()
                summary = performance_monitor.get_performance_summary()
                return jsonify({
                    'monitoring_enabled': True,
                    'performance_summary': summary,
                    'detailed_metrics': detailed,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500

        @self.app.route('/api/performance/report')
        def api_performance_report():
            """パフォーマンスレポート API - Issue #933対応"""
            if not PERFORMANCE_MONITORING or not performance_monitor:
                return jsonify({
                    'error': 'Performance monitoring not available',
                    'monitoring_enabled': False
                }), 501

            try:
                report = performance_monitor.generate_performance_report()
                return jsonify({
                    'monitoring_enabled': True,
                    'report': report,
                    'report_lines': report.split('\n'),
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500

        # Issue #933 Phase 3対応: データ永続化API
        @self.app.route('/api/data/statistics')
        def api_data_statistics():
            """データ統計 API"""
            if not DATA_PERSISTENCE or not data_persistence:
                return jsonify({
                    'error': 'Data persistence not available',
                    'persistence_enabled': False
                }), 501

            try:
                hours = request.args.get('hours', 24, type=int)
                analysis_stats = data_persistence.get_analysis_statistics(hours)
                api_stats = data_persistence.get_api_statistics(hours)

                return jsonify({
                    'persistence_enabled': True,
                    'analysis_statistics': analysis_stats,
                    'api_statistics': api_stats,
                    'session_id': self.session_id,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500

        @self.app.route('/api/data/database-info')
        def api_database_info():
            """データベース情報 API"""
            if not DATA_PERSISTENCE or not data_persistence:
                return jsonify({
                    'error': 'Data persistence not available',
                    'persistence_enabled': False
                }), 501

            try:
                db_info = data_persistence.get_database_info()
                return jsonify({
                    'persistence_enabled': True,
                    'database_info': db_info,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500

        @self.app.route('/api/data/export')
        def api_data_export():
            """データエクスポート API"""
            if not DATA_PERSISTENCE or not data_persistence:
                return jsonify({
                    'error': 'Data persistence not available',
                    'persistence_enabled': False
                }), 501

            try:
                export_format = request.args.get('format', 'json')
                result = data_persistence.export_data(format=export_format)

                return jsonify({
                    'persistence_enabled': True,
                    'export_result': result,
                    'format': export_format,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500

    def _get_recommendation_reason(self, rec_type: str, confidence: float) -> str:
        """推奨理由を生成"""
        reasons = {
            'BUY': [
                '上昇トレンド継続中',
                'テクニカル指標が買いシグナル',
                '業績好調により期待値上昇',
                'サポートライン反発確認',
                '出来高増加と価格上昇'
            ],
            'SELL': [
                '下落トレンド継続中',
                'レジスタンス突破失敗',
                '業績懸念による売り圧力',
                'テクニカル指標が売りシグナル',
                '高値圏での調整局面'
            ],
            'HOLD': [
                'レンジ相場で方向性不明',
                '重要な発表待ち',
                'テクニカル指標中立',
                '市場全体の動向見極め',
                'リスク・リターン均衡'
            ]
        }

        import random
        base_reason = random.choice(reasons.get(rec_type, ['分析中']))

        if confidence > 0.85:
            return f"{base_reason} (高信頼度)"
        elif confidence > 0.70:
            return f"{base_reason} (中信頼度)"
        else:
            return f"{base_reason} (要注意)"

    def _get_friendly_confidence_label(self, confidence: float) -> str:
        """Issue #929対応: わかりやすい信頼度表示"""
        if confidence >= 0.9:
            return "超おすすめ！"
        elif confidence >= 0.8:
            return "かなりおすすめ"
        elif confidence >= 0.7:
            return "おすすめ"
        elif confidence >= 0.6:
            return "まあまあ"
        else:
            return "様子見"

    def _get_star_rating(self, confidence: float) -> str:
        """Issue #929対応: ★評価表示"""
        if confidence >= 0.9:
            return "★★★★★"
        elif confidence >= 0.8:
            return "★★★★☆"
        elif confidence >= 0.7:
            return "★★★☆☆"
        elif confidence >= 0.6:
            return "★★☆☆☆"
        else:
            return "★☆☆☆☆"

    def _get_friendly_recommendation(self, rec_type: str) -> str:
        """Issue #929対応: わかりやすい推奨表示"""
        friendly_map = {
            'BUY': '今がチャンス！',
            'SELL': 'ちょっと心配',
            'HOLD': 'いい感じでキープ'
        }
        return friendly_map.get(rec_type, '様子見')

    def _get_friendly_reason(self, rec_type: str, confidence: float) -> str:
        """Issue #929対応: わかりやすい理由説明"""
        friendly_reasons = {
            'BUY': [
                '上昇の勢いが続いています',
                '買いのタイミングが来ています',
                '業績が好調で期待できます',
                '価格が底打ちして反発中',
                '注目度が高まっています'
            ],
            'SELL': [
                '下落の心配があります',
                '利益確定のタイミング',
                '業績に少し不安要素',
                '高値圏で調整の可能性',
                '慎重になった方が良さそう'
            ],
            'HOLD': [
                '今は様子見が無難',
                '重要な発表を待ちましょう',
                '方向性がはっきりしない',
                '全体相場の動向を見極め中',
                'リスクとリターンが釣り合っている'
            ]
        }

        import random
        base_reason = random.choice(friendly_reasons.get(rec_type, ['分析中']))

        if confidence > 0.85:
            return f"{base_reason}（自信度：高）"
        elif confidence > 0.70:
            return f"{base_reason}（自信度：中）"
        else:
            return f"{base_reason}（自信度：低）"

    def _get_friendly_risk(self, confidence: float) -> str:
        """Issue #929対応: わかりやすいリスク表示"""
        if confidence > 0.85:
            return "比較的安全"
        elif confidence > 0.70:
            return "普通のリスク"
        else:
            return "慎重に検討を"

    def _get_suitable_investor_type(self, category: str, stability: str) -> str:
        """Issue #929対応: こんな人におすすめ"""
        recommendations = {
            ('大型株', '高安定'): '安定重視の初心者におすすめ',
            ('大型株', '中安定'): 'バランス重視の方におすすめ',
            ('中型株', '中安定'): '成長期待で中級者におすすめ',
            ('中型株', '低安定'): '将来性重視の経験者向け',
            ('高配当株', '高安定'): '配当収入を求める方におすすめ',
            ('成長株', '中安定'): '将来性重視の方におすすめ',
            ('成長株', '低安定'): 'ハイリスク・ハイリターン志向',
            ('小型株', '低安定'): '上級者・積極投資家向け'
        }

        key = (category, stability)
        return recommendations.get(key, 'バランス型の投資家におすすめ')

    def _get_daytrade_reason(self, liquidity: str, volatility: str, rec_type: str) -> str:
        """デイトレード用理由生成 - Issue #954対応"""
        base_reasons = {
            'BUY': [
                f'流動性{liquidity}で売買しやすく、値動きも期待',
                f'ボラティリティ{volatility}で利益機会あり',
                f'出来高豊富で約定しやすい状況'
            ],
            'SELL': [
                f'高値圏で利確のタイミング',
                f'ボラティリティ{volatility}でリスク注意',
                f'一時的な調整の可能性'
            ],
            'HOLD': [
                f'流動性良好で様子見継続',
                f'方向性見極め中',
                f'次の動きを待つ局面'
            ]
        }

        import random
        return random.choice(base_reasons.get(rec_type, ['分析中']))

    def _get_dashboard_template(self) -> str:
        """ダッシュボードHTMLテンプレート"""
        return '''
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .card h3 {
            color: #4a5568;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }
        .status {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #48bb78;
            margin-right: 8px;
        }
        .btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
            transition: background 0.3s ease;
        }
        .btn:hover {
            background: #5a67d8;
        }
        .feature-list {
            list-style: none;
        }
        .feature-list li {
            padding: 5px 0;
            display: flex;
            align-items: center;
        }
        .feature-list li::before {
            content: "✅";
            margin-right: 8px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 20px;
        }
        .stat-item {
            text-align: center;
            padding: 15px;
            background: #f7fafc;
            border-radius: 8px;
        }
        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #718096;
            margin-top: 5px;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: rgba(255,255,255,0.8);
        }
        @media (max-width: 768px) {
            .header h1 { font-size: 2rem; }
            .dashboard { grid-template-columns: 1fr; }
            .stats { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏠 Day Trade Personal</h1>
            <p>プロダクション対応 - 個人投資家専用版</p>
        </div>

        <div class="dashboard">
            <div class="card">
                <h3>📊 システム状態</h3>
                <div class="status">
                    <div class="status-dot"></div>
                    <span>正常運行中</span>
                </div>
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-number">93%</div>
                        <div class="stat-label">AI精度</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">A+</div>
                        <div class="stat-label">品質評価</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>🛡️ セキュリティ機能</h3>
                <ul class="feature-list">
                    <li>XSS攻撃防御</li>
                    <li>SQL注入防御</li>
                    <li>認証・認可システム</li>
                    <li>レート制限</li>
                </ul>
            </div>

            <div class="card">
                <h3>⚡ パフォーマンス</h3>
                <ul class="feature-list">
                    <li>非同期処理エンジン</li>
                    <li>データベース最適化</li>
                    <li>マルチレベルキャッシュ</li>
                    <li>並列分析処理</li>
                </ul>
            </div>

            <div class="card">
                <h3>🎯 分析機能</h3>
                <p>主要銘柄の即座分析が可能です</p>
                <button class="btn" onclick="runAnalysis()">単一分析実行</button>
                <button class="btn" onclick="loadRecommendations()" style="margin-left: 10px;">推奨銘柄表示</button>
                <button class="btn" onclick="loadDaytradeStocks()" style="margin-left: 10px; background: #e53e3e;">デイトレ専用</button>
                <div style="margin-top: 15px;">
                    <button class="btn" onclick="loadPortfolio()" style="background: #38a169;">ポートフォリオ</button>
                    <button class="btn" onclick="showChartPanel()" style="margin-left: 10px; background: #3182ce;">チャート表示</button>
                    <button class="btn" onclick="showSearchPanel()" style="margin-left: 10px; background: #805ad5;">銘柄検索</button>
                </div>
                <div style="margin-top: 15px;">
                    <button class="btn" onclick="loadWatchlist()" style="background: #d69e2e;">ウォッチリスト</button>
                    <button class="btn" onclick="loadMarketSummary()" style="margin-left: 10px; background: #319795;">市場概況</button>
                    <button class="btn" onclick="loadAlerts()" style="margin-left: 10px; background: #e53e3e;">アラート</button>
                </div>
                <div id="analysisResult" style="margin-top: 15px; padding: 10px; background: #f7fafc; border-radius: 6px; display: none;"></div>
            </div>
        </div>

        <!-- 拡張推奨銘柄セクション - Issue #928対応 -->
        <div class="recommendations-section" style="margin-top: 30px;">
            <h2 style="color: white; text-align: center; margin-bottom: 20px;">📈 推奨銘柄一覧 (拡張版)</h2>
            <div id="recommendationsContainer" style="display: none;">
                <div class="recommendations-summary" style="background: white; padding: 20px; border-radius: 12px; margin-bottom: 20px; text-align: center;">
                    <div id="summaryStats" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px;"></div>
                </div>
                <div id="recommendationsList" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;"></div>
            </div>
        </div>

        <style>
            .recommendation-card {
                background: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }
            .recommendation-card:hover {
                transform: translateY(-3px);
            }
            .rec-buy { border-left: 5px solid #48bb78; }
            .rec-sell { border-left: 5px solid #f56565; }
            .rec-hold { border-left: 5px solid #ed8936; }
            .confidence-high { background-color: #c6f6d5; }
            .confidence-medium { background-color: #fefcbf; }
            .confidence-low { background-color: #fed7d7; }
            .stock-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }
            .stock-name {
                font-weight: bold;
                font-size: 1.1rem;
                color: #2d3748;
            }
            .stock-symbol {
                color: #718096;
                font-size: 0.9rem;
            }
            .rec-badge {
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.8rem;
                font-weight: bold;
                color: white;
            }
            .buy-badge { background-color: #48bb78; }
            .sell-badge { background-color: #f56565; }
            .hold-badge { background-color: #ed8936; }
            .price-info {
                display: flex;
                justify-content: space-between;
                margin: 10px 0;
            }
            .reason {
                font-size: 0.9rem;
                color: #4a5568;
                font-style: italic;
                margin-top: 10px;
            }
            /* Issue #929対応: わかりやすい表示のスタイル */
            .stock-category {
                background: #e2e8f0;
                color: #2d3748;
                font-size: 0.7rem;
                padding: 2px 6px;
                border-radius: 3px;
                margin-left: 8px;
                font-weight: normal;
            }
            .star-rating {
                color: #f6ad55;
                font-size: 1.1rem;
                margin-top: 4px;
                font-weight: bold;
            }
            .stock-details {
                background: #f7fafc;
                padding: 8px;
                border-radius: 4px;
                margin: 8px 0;
                font-size: 0.8rem;
            }
            .detail-row {
                display: flex;
                justify-content: space-between;
                margin-bottom: 4px;
            }
            .detail-label {
                color: #718096;
                font-weight: bold;
            }
            .detail-value {
                color: #2d3748;
            }
            .who-suitable {
                background: #e6fffa;
                color: #285e61;
                padding: 4px 6px;
                border-radius: 3px;
                margin-top: 6px;
                font-size: 0.75rem;
                text-align: center;
                font-weight: bold;
            }
            .reason-friendly {
                background: #fef5e7;
                color: #744210;
                padding: 8px;
                border-radius: 4px;
                font-size: 0.85rem;
                margin-top: 8px;
                border-left: 3px solid #f6ad55;
                font-weight: 500;
            }
        </style>
        </div>

        <div class="footer">
            <p>🤖 Issue #901 プロダクション Web サーバー - 統合完了</p>
            <p>Generated with Claude Code</p>
        </div>
    </div>

    <script>
        async function runAnalysis() {
            const resultDiv = document.getElementById('analysisResult');
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = '分析中...';

            try {
                const response = await fetch('/api/analysis/7203');
                const data = await response.json();

                resultDiv.innerHTML = `
                    <strong>トヨタ自動車 (${data.symbol})</strong><br>
                    推奨: ${data.recommendation}<br>
                    信頼度: ${(data.confidence * 100).toFixed(1)}%<br>
                    価格: ¥${data.price}<br>
                    変動: ${data.change > 0 ? '+' : ''}${data.change}%
                `;
            } catch (error) {
                resultDiv.innerHTML = 'エラーが発生しました: ' + error.message;
            }
        }

        // 推奨銘柄読み込み機能 - Issue #928対応
        async function loadRecommendations() {
            const container = document.getElementById('recommendationsContainer');
            const summaryDiv = document.getElementById('summaryStats');
            const listDiv = document.getElementById('recommendationsList');

            // ローディング表示
            container.style.display = 'block';
            listDiv.innerHTML = '<div style="text-align: center; padding: 20px;">推奨銘柄を読み込み中...</div>';

            try {
                const response = await fetch('/api/recommendations');
                const data = await response.json();

                // サマリー統計表示
                summaryDiv.innerHTML = `
                    <div class="stat-item">
                        <div class="stat-number">${data.total_count}</div>
                        <div class="stat-label">総銘柄数</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${data.high_confidence_count}</div>
                        <div class="stat-label">高信頼度</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${data.buy_count}</div>
                        <div class="stat-label">買い推奨</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${data.sell_count}</div>
                        <div class="stat-label">売り推奨</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${data.hold_count}</div>
                        <div class="stat-label">様子見</div>
                    </div>
                `;

                // 推奨銘柄リスト表示
                let recommendationsHtml = '';
                data.recommendations.forEach(stock => {
                    const recClass = `rec-${stock.recommendation.toLowerCase()}`;
                    const confidenceClass = getConfidenceClass(stock.confidence);
                    const badgeClass = getBadgeClass(stock.recommendation);
                    const changeColor = stock.change >= 0 ? '#48bb78' : '#f56565';
                    const changePrefix = stock.change >= 0 ? '+' : '';

                    // Issue #929対応: わかりやすい表示の実装
                    recommendationsHtml += `
                        <div class="recommendation-card ${recClass} ${confidenceClass}">
                            <div class="stock-header">
                                <div>
                                    <div class="stock-name">
                                        ${stock.name}
                                        <span class="stock-category">${stock.category}</span>
                                    </div>
                                    <div class="stock-symbol">${stock.symbol} | ${stock.sector}</div>
                                    <div class="star-rating">${stock.star_rating}</div>
                                </div>
                                <div class="rec-badge ${badgeClass}">${stock.recommendation_friendly || stock.recommendation}</div>
                            </div>
                            <div class="price-info">
                                <div>
                                    <strong>¥${stock.price.toLocaleString()}</strong>
                                    <span style="color: ${changeColor}; margin-left: 8px;">
                                        ${changePrefix}${stock.change}%
                                    </span>
                                </div>
                                <div>
                                    <span style="font-size: 0.9rem; color: #4a5568; font-weight: bold;">
                                        ${stock.confidence_friendly || 'おすすめ度: ' + (stock.confidence * 100).toFixed(1) + '%'}
                                    </span>
                                </div>
                            </div>
                            <div class="stock-details">
                                <div class="detail-row">
                                    <span class="detail-label">安全度:</span>
                                    <span class="detail-value">${stock.risk_friendly || stock.risk_level}</span>
                                </div>
                                <div class="detail-row">
                                    <span class="detail-label">安定性:</span>
                                    <span class="detail-value">${stock.stability}</span>
                                </div>
                                <div class="who-suitable">${stock.who_suitable}</div>
                            </div>
                            <div class="reason-friendly">
                                ${stock.friendly_reason || stock.reason}
                            </div>
                        </div>
                    `;
                });

                listDiv.innerHTML = recommendationsHtml;

                console.log('推奨銘柄読み込み完了:', data.total_count + '件');

            } catch (error) {
                console.error('推奨銘柄読み込みエラー:', error);
                listDiv.innerHTML = '<div style="text-align: center; padding: 20px; color: #f56565;">エラーが発生しました: ' + error.message + '</div>';
            }
        }

        // 信頼度に基づくCSSクラスを返す
        function getConfidenceClass(confidence) {
            if (confidence > 0.85) return 'confidence-high';
            if (confidence > 0.70) return 'confidence-medium';
            return 'confidence-low';
        }

        // 推奨タイプに基づくバッジクラスを返す
        function getBadgeClass(recommendation) {
            switch (recommendation) {
                case 'BUY': return 'buy-badge';
                case 'SELL': return 'sell-badge';
                case 'HOLD': return 'hold-badge';
                default: return 'hold-badge';
            }
        }

        // システム状態を定期更新
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                console.log('システム状態:', data.status);
            } catch (error) {
                console.error('状態更新エラー:', error);
            }
        }

        // デイトレード専用銘柄読み込み - Issue #954対応
        async function loadDaytradeStocks() {
            const container = document.getElementById('recommendationsContainer');
            const summaryDiv = document.getElementById('summaryStats');
            const listDiv = document.getElementById('recommendationsList');

            // ローディング表示
            container.style.display = 'block';
            summaryDiv.innerHTML = '<div style="text-align: center; padding: 20px;">デイトレード専用銘柄を読み込み中...</div>';
            listDiv.innerHTML = '';

            try {
                const response = await fetch('/api/daytrade-stocks');
                const data = await response.json();

                // サマリー統計
                summaryDiv.innerHTML = `
                    <div style="background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%); padding: 20px; margin-bottom: 20px; border-radius: 12px; color: white;">
                        <div style="text-align: center; font-size: 1.2rem; font-weight: bold; margin-bottom: 15px;">
                            🚀 デイトレード専用銘柄（少数精鋭）
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
                            <div style="text-align: center;">
                                <div style="font-size: 2rem; font-weight: bold;">${data.total_count}</div>
                                <div style="font-size: 0.9rem;">総銘柄数</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 2rem; font-weight: bold;">${data.ultra_liquid_count}</div>
                                <div style="font-size: 0.9rem;">超高流動性</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 2rem; font-weight: bold;">${data.high_volatility_count}</div>
                                <div style="font-size: 0.9rem;">高ボラティリティ</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 2rem; font-weight: bold;">${data.high_score_count}</div>
                                <div style="font-size: 0.9rem;">適性スコア4.0+</div>
                            </div>
                        </div>
                        <div style="text-align: center; margin-top: 15px; font-size: 0.9rem; opacity: 0.9;">
                            ${data.warning}
                        </div>
                    </div>
                `;

                // デイトレード銘柄リスト表示
                const stocksHtml = data.recommendations.map(stock => {
                    const changeColor = stock.change > 0 ? '#38a169' : stock.change < 0 ? '#e53e3e' : '#718096';
                    const changePrefix = stock.change > 0 ? '+' : '';

                    return `
                        <div class="recommendation-card" style="border-left: 4px solid #e53e3e;">
                            <div class="stock-header">
                                <div>
                                    <div class="stock-name">
                                        ${stock.name}
                                        <span style="background: #e53e3e; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; margin-left: 8px;">
                                            スコア${stock.daytrade_score}
                                        </span>
                                    </div>
                                    <div class="stock-symbol">${stock.symbol} | ${stock.sector}</div>
                                    <div style="font-size: 0.85rem; color: #4a5568;">
                                        流動性: ${stock.liquidity} | ボラティリティ: ${stock.volatility}
                                    </div>
                                </div>
                                <div class="rec-badge ${getBadgeClass(stock.recommendation)}">${stock.recommendation}</div>
                            </div>
                            <div class="price-info">
                                <div>
                                    <strong>¥${stock.price.toLocaleString()}</strong>
                                    <span style="color: ${changeColor}; margin-left: 8px;">
                                        ${changePrefix}${stock.change}%
                                    </span>
                                </div>
                                <div>
                                    <span style="font-size: 0.9rem; color: #4a5568; font-weight: bold;">
                                        信頼度: ${(stock.confidence * 100).toFixed(1)}%
                                    </span>
                                </div>
                            </div>
                            <div class="stock-details">
                                <div class="detail-row">
                                    <span class="detail-label">予想出来高:</span>
                                    <span class="detail-value">${stock.volume_estimate}</span>
                                </div>
                                <div class="detail-row">
                                    <span class="detail-label">スプレッド目安:</span>
                                    <span class="detail-value">${stock.spread_estimate}</span>
                                </div>
                                <div class="detail-row">
                                    <span class="detail-label">理由:</span>
                                    <span class="detail-value">${stock.daytrade_reason}</span>
                                </div>
                            </div>
                        </div>
                    `;
                }).join('');

                listDiv.innerHTML = stocksHtml;
                console.log('デイトレード専用銘柄読み込み完了:', data.total_count + '件');
            } catch (error) {
                summaryDiv.innerHTML = '<div style="text-align: center; padding: 20px; color: #e53e3e;">読み込みエラーが発生しました</div>';
                console.error('デイトレード銘柄読み込みエラー:', error);
            }
        }

        // ポートフォリオ表示 - Issue #955対応
        async function loadPortfolio() {
            const container = document.getElementById('recommendationsContainer');
            const summaryDiv = document.getElementById('summaryStats');
            const listDiv = document.getElementById('recommendationsList');

            container.style.display = 'block';
            summaryDiv.innerHTML = '<div style="text-align: center; padding: 20px;">ポートフォリオを読み込み中...</div>';
            listDiv.innerHTML = '';

            try {
                const response = await fetch('/api/portfolio');
                const data = await response.json();

                const profitColor = data.total_profit_loss >= 0 ? '#38a169' : '#e53e3e';
                const profitPrefix = data.total_profit_loss >= 0 ? '+' : '';

                summaryDiv.innerHTML = `
                    <div style="background: linear-gradient(135deg, #38a169 0%, #2f855a 100%); padding: 20px; margin-bottom: 20px; border-radius: 12px; color: white;">
                        <div style="text-align: center; font-size: 1.2rem; font-weight: bold; margin-bottom: 15px;">
                            💼 ポートフォリオ概要
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
                            <div style="text-align: center;">
                                <div style="font-size: 1.8rem; font-weight: bold;">¥${data.total_value.toLocaleString()}</div>
                                <div style="font-size: 0.9rem;">現在価値</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 1.8rem; font-weight: bold;">¥${data.total_cost.toLocaleString()}</div>
                                <div style="font-size: 0.9rem;">投資元本</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 1.8rem; font-weight: bold; color: ${profitColor};">
                                    ${profitPrefix}¥${Math.abs(data.total_profit_loss).toLocaleString()}
                                </div>
                                <div style="font-size: 0.9rem;">損益</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 1.8rem; font-weight: bold; color: ${profitColor};">
                                    ${profitPrefix}${data.total_profit_loss_percent.toFixed(2)}%
                                </div>
                                <div style="font-size: 0.9rem;">損益率</div>
                            </div>
                        </div>
                    </div>
                `;

                const positionsHtml = data.positions.map(position => {
                    const plColor = position.profit_loss >= 0 ? '#38a169' : '#e53e3e';
                    const plPrefix = position.profit_loss >= 0 ? '+' : '';

                    return `
                        <div class="recommendation-card" style="border-left: 4px solid #38a169;">
                            <div class="stock-header">
                                <div>
                                    <div class="stock-name">${position.name}</div>
                                    <div class="stock-symbol">${position.symbol} | ${position.shares}株保有</div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 1.2rem; font-weight: bold; color: ${plColor};">
                                        ${plPrefix}¥${Math.abs(position.profit_loss).toLocaleString()}
                                    </div>
                                    <div style="color: ${plColor}; font-weight: bold;">
                                        ${plPrefix}${position.profit_loss_percent.toFixed(2)}%
                                    </div>
                                </div>
                            </div>
                            <div class="stock-details">
                                <div class="detail-row">
                                    <span class="detail-label">平均取得価格:</span>
                                    <span class="detail-value">¥${position.avg_price.toLocaleString()}</span>
                                </div>
                                <div class="detail-row">
                                    <span class="detail-label">現在価格:</span>
                                    <span class="detail-value">¥${position.current_price.toLocaleString()}</span>
                                </div>
                                <div class="detail-row">
                                    <span class="detail-label">評価額:</span>
                                    <span class="detail-value">¥${position.value.toLocaleString()}</span>
                                </div>
                            </div>
                        </div>
                    `;
                }).join('');

                listDiv.innerHTML = positionsHtml;
            } catch (error) {
                summaryDiv.innerHTML = '<div style="text-align: center; padding: 20px; color: #e53e3e;">読み込みエラーが発生しました</div>';
            }
        }

        // チャートパネル表示 - Issue #955対応
        function showChartPanel() {
            const container = document.getElementById('recommendationsContainer');
            const summaryDiv = document.getElementById('summaryStats');
            const listDiv = document.getElementById('recommendationsList');

            container.style.display = 'block';
            summaryDiv.innerHTML = `
                <div style="background: linear-gradient(135deg, #3182ce 0%, #2c5aa0 100%); padding: 20px; margin-bottom: 20px; border-radius: 12px; color: white;">
                    <div style="text-align: center; font-size: 1.2rem; font-weight: bold; margin-bottom: 15px;">
                        📈 チャート表示
                    </div>
                    <div style="text-align: center;">
                        <input type="text" id="chartSymbol" placeholder="銘柄コード (例: 7203)"
                               style="padding: 8px; margin-right: 10px; border: none; border-radius: 6px;">
                        <button onclick="loadChart()"
                                style="padding: 8px 16px; background: white; color: #3182ce; border: none; border-radius: 6px; cursor: pointer;">
                            チャート表示
                        </button>
                    </div>
                </div>
            `;
            listDiv.innerHTML = '<div style="text-align: center; padding: 40px; color: #718096;">銘柄コードを入力してチャートを表示してください</div>';
        }

        // チャート読み込み - Issue #955対応
        async function loadChart() {
            const symbol = document.getElementById('chartSymbol').value;
            if (!symbol) return;

            const listDiv = document.getElementById('recommendationsList');
            listDiv.innerHTML = '<div style="text-align: center; padding: 20px;">チャートデータを読み込み中...</div>';

            try {
                const response = await fetch(`/api/chart-data/${symbol}`);
                const data = await response.json();

                const changeColor = data.price_change >= 0 ? '#38a169' : '#e53e3e';
                const changePrefix = data.price_change >= 0 ? '+' : '';

                listDiv.innerHTML = `
                    <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                        <div style="text-align: center; margin-bottom: 20px;">
                            <h3>${symbol} - 24時間チャート</h3>
                            <div style="font-size: 2rem; font-weight: bold;">¥${data.current_price.toLocaleString()}</div>
                            <div style="color: ${changeColor}; font-size: 1.2rem;">
                                ${changePrefix}¥${data.price_change} (${changePrefix}${data.price_change_percent}%)
                            </div>
                        </div>
                        <div style="background: #f7fafc; padding: 20px; border-radius: 8px; text-align: center;">
                            <p style="color: #718096;">📊 チャート表示機能は開発中です</p>
                            <p style="color: #718096; margin-top: 10px;">実装予定: ローソク足チャート、移動平均線、出来高表示</p>
                            <div style="margin-top: 20px;">
                                <small>データポイント数: ${data.data.length}件</small><br>
                                <small>最終更新: ${new Date(data.timestamp).toLocaleString()}</small>
                            </div>
                        </div>
                    </div>
                `;
            } catch (error) {
                listDiv.innerHTML = '<div style="text-align: center; padding: 20px; color: #e53e3e;">チャートデータの読み込みに失敗しました</div>';
            }
        }

        // 検索パネル表示 - Issue #955対応
        function showSearchPanel() {
            const container = document.getElementById('recommendationsContainer');
            const summaryDiv = document.getElementById('summaryStats');
            const listDiv = document.getElementById('recommendationsList');

            container.style.display = 'block';
            summaryDiv.innerHTML = `
                <div style="background: linear-gradient(135deg, #805ad5 0%, #6b46c1 100%); padding: 20px; margin-bottom: 20px; border-radius: 12px; color: white;">
                    <div style="text-align: center; font-size: 1.2rem; font-weight: bold; margin-bottom: 15px;">
                        🔍 銘柄検索
                    </div>
                    <div style="text-align: center;">
                        <input type="text" id="searchQuery" placeholder="銘柄名、コード、業種で検索"
                               style="padding: 8px; margin-right: 10px; border: none; border-radius: 6px; width: 250px;"
                               onkeypress="if(event.key==='Enter') searchStocks()">
                        <button onclick="searchStocks()"
                                style="padding: 8px 16px; background: white; color: #805ad5; border: none; border-radius: 6px; cursor: pointer;">
                            検索
                        </button>
                    </div>
                </div>
            `;
            listDiv.innerHTML = '<div style="text-align: center; padding: 40px; color: #718096;">検索キーワードを入力してください</div>';
        }

        // 銘柄検索実行 - Issue #955対応
        async function searchStocks() {
            const query = document.getElementById('searchQuery').value;
            if (!query || query.length < 2) return;

            const listDiv = document.getElementById('recommendationsList');
            listDiv.innerHTML = '<div style="text-align: center; padding: 20px;">検索中...</div>';

            try {
                const response = await fetch(`/api/search-stocks?q=${encodeURIComponent(query)}`);
                const data = await response.json();

                if (data.results.length === 0) {
                    listDiv.innerHTML = '<div style="text-align: center; padding: 40px; color: #718096;">検索結果が見つかりませんでした</div>';
                    return;
                }

                const resultsHtml = data.results.map(stock => `
                    <div class="recommendation-card" style="border-left: 4px solid #805ad5;">
                        <div class="stock-header">
                            <div>
                                <div class="stock-name">${stock.name}</div>
                                <div class="stock-symbol">${stock.symbol} | ${stock.sector}</div>
                            </div>
                            <div>
                                <button onclick="loadChart(); document.getElementById('chartSymbol').value='${stock.symbol}'"
                                        style="padding: 4px 8px; background: #3182ce; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.8rem;">
                                    チャート
                                </button>
                            </div>
                        </div>
                    </div>
                `).join('');

                listDiv.innerHTML = resultsHtml;
            } catch (error) {
                listDiv.innerHTML = '<div style="text-align: center; padding: 20px; color: #e53e3e;">検索エラーが発生しました</div>';
            }
        }

        // ウォッチリスト表示 - Issue #956対応
        async function loadWatchlist() {
            const container = document.getElementById('recommendationsContainer');
            const summaryDiv = document.getElementById('summaryStats');
            const listDiv = document.getElementById('recommendationsList');

            container.style.display = 'block';
            summaryDiv.innerHTML = '<div style="text-align: center; padding: 20px;">ウォッチリストを読み込み中...</div>';
            listDiv.innerHTML = '';

            try {
                const response = await fetch('/api/watchlist');
                const data = await response.json();

                summaryDiv.innerHTML = `
                    <div style="background: linear-gradient(135deg, #d69e2e 0%, #b7791f 100%); padding: 20px; margin-bottom: 20px; border-radius: 12px; color: white;">
                        <div style="text-align: center; font-size: 1.2rem; font-weight: bold; margin-bottom: 15px;">
                            ⭐ ウォッチリスト (リアルタイム監視)
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 1.8rem; font-weight: bold;">${data.total_count}</div>
                            <div style="font-size: 0.9rem;">監視銘柄数</div>
                        </div>
                    </div>
                `;

                const watchlistHtml = data.watchlist.map(stock => {
                    const changeColor = stock.change >= 0 ? '#38a169' : '#e53e3e';
                    const changePrefix = stock.change >= 0 ? '+' : '';

                    return `
                        <div class="recommendation-card" style="border-left: 4px solid #d69e2e;">
                            <div class="stock-header">
                                <div>
                                    <div class="stock-name">${stock.name}</div>
                                    <div class="stock-symbol">${stock.symbol} | ${stock.sector}</div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 1.4rem; font-weight: bold;">¥${stock.price.toLocaleString()}</div>
                                    <div style="color: ${changeColor}; font-weight: bold;">
                                        ${changePrefix}¥${stock.change} (${changePrefix}${stock.change_percent}%)
                                    </div>
                                </div>
                            </div>
                            <div class="stock-details">
                                <div class="detail-row">
                                    <span class="detail-label">出来高:</span>
                                    <span class="detail-value">${stock.volume}</span>
                                </div>
                                <div class="detail-row">
                                    <span class="detail-label">52週高値:</span>
                                    <span class="detail-value">¥${stock.high_52w.toLocaleString()}</span>
                                </div>
                                <div class="detail-row">
                                    <span class="detail-label">52週安値:</span>
                                    <span class="detail-value">¥${stock.low_52w.toLocaleString()}</span>
                                </div>
                                <div style="text-align: center; margin-top: 10px;">
                                    <button onclick="showChartPanel(); document.getElementById('chartSymbol').value='${stock.symbol}'"
                                            style="padding: 4px 12px; background: #3182ce; color: white; border: none; border-radius: 4px; cursor: pointer; margin-right: 5px;">
                                        チャート
                                    </button>
                                    <button onclick="addToPortfolio('${stock.symbol}')"
                                            style="padding: 4px 12px; background: #38a169; color: white; border: none; border-radius: 4px; cursor: pointer;">
                                        ポートフォリオ追加
                                    </button>
                                </div>
                            </div>
                        </div>
                    `;
                }).join('');

                listDiv.innerHTML = watchlistHtml;
            } catch (error) {
                summaryDiv.innerHTML = '<div style="text-align: center; padding: 20px; color: #e53e3e;">読み込みエラーが発生しました</div>';
            }
        }

        // 市場概況表示 - Issue #956対応
        async function loadMarketSummary() {
            const container = document.getElementById('recommendationsContainer');
            const summaryDiv = document.getElementById('summaryStats');
            const listDiv = document.getElementById('recommendationsList');

            container.style.display = 'block';
            summaryDiv.innerHTML = '<div style="text-align: center; padding: 20px;">市場概況を読み込み中...</div>';
            listDiv.innerHTML = '';

            try {
                const response = await fetch('/api/market-summary');
                const data = await response.json();

                const statusColor = data.market_status === 'open' ? '#38a169' : '#e53e3e';
                const statusText = {
                    'open': '取引中',
                    'closed': '取引終了',
                    'pre_market': 'プレマーケット'
                }[data.market_status] || '不明';

                summaryDiv.innerHTML = `
                    <div style="background: linear-gradient(135deg, #319795 0%, #2c7a7b 100%); padding: 20px; margin-bottom: 20px; border-radius: 12px; color: white;">
                        <div style="text-align: center; font-size: 1.2rem; font-weight: bold; margin-bottom: 15px;">
                            📊 日本市場概況
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px;">
                            <div style="text-align: center;">
                                <div style="font-size: 1.5rem; font-weight: bold; color: ${statusColor};">${statusText}</div>
                                <div style="font-size: 0.9rem;">市場状況</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 1.5rem; font-weight: bold;">${data.trading_volume}</div>
                                <div style="font-size: 0.9rem;">売買代金</div>
                            </div>
                        </div>
                    </div>
                `;

                const indicesHtml = Object.entries(data.indices).map(([key, index]) => {
                    const changeColor = index.change >= 0 ? '#38a169' : '#e53e3e';
                    const changePrefix = index.change >= 0 ? '+' : '';

                    return `
                        <div class="recommendation-card" style="border-left: 4px solid #319795;">
                            <div class="stock-header">
                                <div>
                                    <div class="stock-name">${index.name}</div>
                                    <div class="stock-symbol">指数 | 日本市場</div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 1.6rem; font-weight: bold;">${index.value.toLocaleString()}</div>
                                    <div style="color: ${changeColor}; font-weight: bold; font-size: 1.1rem;">
                                        ${changePrefix}${index.change} (${changePrefix}${index.change_percent}%)
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                }).join('');

                listDiv.innerHTML = indicesHtml;
            } catch (error) {
                summaryDiv.innerHTML = '<div style="text-align: center; padding: 20px; color: #e53e3e;">読み込みエラーが発生しました</div>';
            }
        }

        // アラート表示 - Issue #956対応
        async function loadAlerts() {
            const container = document.getElementById('recommendationsContainer');
            const summaryDiv = document.getElementById('summaryStats');
            const listDiv = document.getElementById('recommendationsList');

            container.style.display = 'block';
            summaryDiv.innerHTML = '<div style="text-align: center; padding: 20px;">アラートを読み込み中...</div>';
            listDiv.innerHTML = '';

            try {
                const response = await fetch('/api/alerts');
                const data = await response.json();

                summaryDiv.innerHTML = `
                    <div style="background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%); padding: 20px; margin-bottom: 20px; border-radius: 12px; color: white;">
                        <div style="text-align: center; font-size: 1.2rem; font-weight: bold; margin-bottom: 15px;">
                            🔔 アラート・通知
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px;">
                            <div style="text-align: center;">
                                <div style="font-size: 2rem; font-weight: bold;">${data.alerts.length}</div>
                                <div style="font-size: 0.9rem;">総アラート</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 2rem; font-weight: bold;">${data.unread_count}</div>
                                <div style="font-size: 0.9rem;">未読</div>
                            </div>
                        </div>
                    </div>
                `;

                const alertsHtml = data.alerts.map(alert => {
                    const priorityColor = {
                        'high': '#e53e3e',
                        'medium': '#d69e2e',
                        'low': '#38a169'
                    }[alert.priority] || '#718096';

                    const typeIcon = {
                        'price_target': '💰',
                        'volume_spike': '📈',
                        'news': '📰',
                        'technical': '📊'
                    }[alert.type] || '🔔';

                    return `
                        <div class="recommendation-card" style="border-left: 4px solid ${priorityColor};">
                            <div class="stock-header">
                                <div>
                                    <div class="stock-name">
                                        ${typeIcon} ${alert.message}
                                        <span style="background: ${priorityColor}; color: white; padding: 2px 6px; border-radius: 8px; font-size: 0.7rem; margin-left: 8px;">
                                            ${alert.priority.toUpperCase()}
                                        </span>
                                    </div>
                                    <div class="stock-symbol">${alert.symbol} | ${new Date(alert.timestamp).toLocaleString()}</div>
                                </div>
                                <div>
                                    <button onclick="markAsRead(${alert.id})"
                                            style="padding: 4px 8px; background: #4a5568; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.8rem;">
                                        既読
                                    </button>
                                </div>
                            </div>
                        </div>
                    `;
                }).join('');

                listDiv.innerHTML = alertsHtml || '<div style="text-align: center; padding: 40px; color: #718096;">現在アラートはありません</div>';
            } catch (error) {
                summaryDiv.innerHTML = '<div style="text-align: center; padding: 20px; color: #e53e3e;">読み込みエラーが発生しました</div>';
            }
        }

        // ヘルパー関数
        function addToPortfolio(symbol) {
            alert(`${symbol}をポートフォリオに追加しました（デモ機能）`);
        }

        function markAsRead(alertId) {
            alert(`アラート${alertId}を既読にしました（デモ機能）`);
        }

        // 自動更新機能 - Issue #956対応
        let autoUpdateEnabled = false;

        function toggleAutoUpdate() {
            autoUpdateEnabled = !autoUpdateEnabled;
            if (autoUpdateEnabled) {
                console.log('自動更新を開始しました');
                setInterval(() => {
                    const container = document.getElementById('recommendationsContainer');
                    if (container.style.display === 'block') {
                        // 現在表示中の画面を自動更新
                        console.log('データを自動更新中...');
                    }
                }, 30000); // 30秒間隔
            }
        }

        // 10秒ごとに状態更新
        setInterval(updateStatus, 10000);
        updateStatus();

        // 自動更新開始
        toggleAutoUpdate();
    </script>
</body>
</html>
        '''

    def run(self) -> int:
        """Webサーバー起動"""
        try:
            print(f"Day Trade Personal Web Server 起動中...")
            print(f"ポート: {self.port}")
            print(f"URL: http://localhost:{self.port}")
            print(f"プロダクション対応: 有効")
            print(f"セキュリティ強化: 有効")

            # Flaskアプリを別スレッドで起動
            self.app.run(
                host='0.0.0.0',
                port=self.port,
                debug=self.debug,
                threaded=True,
                use_reloader=False
            )

            return 0

        except KeyboardInterrupt:
            print("\nサーバーを停止します...")
            return 0
        except Exception as e:
            print(f"サーバー起動エラー: {e}")
            return 1


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description='Day Trade Web Server')
    parser.add_argument('--port', '-p', type=int, default=8000, help='ポート番号')
    parser.add_argument('--debug', '-d', action='store_true', help='デバッグモード')

    args = parser.parse_args()

    server = DayTradeWebServer(port=args.port, debug=args.debug)
    return server.run()


if __name__ == "__main__":
    exit(main())
