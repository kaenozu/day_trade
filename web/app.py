#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Application Entry Point - Day Trade Web
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, jsonify
from datetime import datetime
from web.services.recommendation_service import RecommendationService
from web.services.trading_timing_service import TradingTimingService
from web.api.portfolio_api import portfolio_api

# 最適化システム統合
try:
    from src.day_trade.optimization.integrated_optimization_system import integrated_optimizer
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("⚠️  最適化システムが利用できません")

def create_app():
    """Flaskアプリケーションの作成"""
    app = Flask(__name__)
    # セキュアなsecret key設定
    import secrets
    secret_key = os.environ.get('FLASK_SECRET_KEY')
    if not secret_key:
        secret_key = secrets.token_urlsafe(32)
        print("⚠️  本番環境では環境変数FLASK_SECRET_KEYを設定してください")
    app.secret_key = secret_key

    # 静的ファイルとテンプレートのパス設定
    app.static_folder = str(project_root / 'static')
    app.template_folder = str(project_root / 'templates')

    # サービス初期化
    recommendation_service = RecommendationService()
    timing_service = TradingTimingService()

    # ブループリント登録
    app.register_blueprint(portfolio_api)

    @app.route('/')
    def index():
        """メインダッシュボード"""
        return render_template('index.html')

    @app.route('/health')
    def health_check():
        """ヘルスチェック"""
        return jsonify({
            'status': 'healthy',
            'timestamp': str(datetime.now()),
            'service': 'Day Trade Web'
        })

    @app.route('/api/status')
    def api_status():
        """APIステータス"""
        return jsonify({
            'status': 'ok',
            'timestamp': str(datetime.now()),
            'api_version': '2.1.0'
        })

    @app.route('/api/recommendations')
    def api_recommendations():
        """推奨銘柄API"""
        try:
            recommendations = recommendation_service.get_recommendations()

            # 統計情報の計算
            total_count = len(recommendations)
            high_confidence_count = len([r for r in recommendations if r['confidence'] > 0.8])
            buy_count = len([r for r in recommendations if r['recommendation'] in ['BUY', 'STRONG_BUY']])
            sell_count = len([r for r in recommendations if r['recommendation'] in ['SELL', 'STRONG_SELL']])
            hold_count = len([r for r in recommendations if r['recommendation'] == 'HOLD'])

            return jsonify({
                'recommendations': recommendations,
                'total_count': total_count,
                'high_confidence_count': high_confidence_count,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'hold_count': hold_count,
                'timestamp': str(datetime.now())
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/recommendations-with-timing')
    def api_recommendations_with_timing():
        """売買タイミング付き推奨銘柄API"""
        try:
            recommendations = recommendation_service.get_recommendations()

            # 各推奨銘柄にタイミング情報を追加
            enhanced_recommendations = []
            for rec in recommendations:
                # タイミング予想の取得
                timing_strategy = timing_service.predict_trading_timing(
                    symbol=rec['symbol'],
                    current_price=rec['price'],
                    recommendation=rec['recommendation'],
                    confidence=rec['confidence']
                )

                # レコメンデーションにタイミング情報を追加
                rec['buy_timing'] = timing_strategy.buy_timing.predicted_time
                rec['sell_timing'] = timing_strategy.sell_timing.predicted_time
                rec['strategy_type'] = timing_strategy.strategy_type
                rec['expected_return'] = timing_strategy.expected_return
                rec['time_horizon'] = timing_strategy.time_horizon

                enhanced_recommendations.append(rec)

            # 統計情報の計算
            total_count = len(enhanced_recommendations)
            high_confidence_count = len([r for r in enhanced_recommendations if r['confidence'] > 0.8])
            buy_count = len([r for r in enhanced_recommendations if r['recommendation'] in ['BUY', 'STRONG_BUY']])
            sell_count = len([r for r in enhanced_recommendations if r['recommendation'] in ['SELL', 'STRONG_SELL']])
            hold_count = len([r for r in enhanced_recommendations if r['recommendation'] == 'HOLD'])

            return jsonify({
                'recommendations': enhanced_recommendations,
                'total_count': total_count,
                'high_confidence_count': high_confidence_count,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'hold_count': hold_count,
                'timing_enabled': True,
                'timestamp': str(datetime.now())
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/trading-timing/<symbol>')
    def api_trading_timing(symbol):
        """個別銘柄の詳細売買タイミング"""
        try:
            # 銘柄の基本情報を取得
            stock_analysis = recommendation_service.analyze_single_symbol(symbol)

            # 詳細タイミング予想
            timing_strategy = timing_service.predict_trading_timing(
                symbol=symbol,
                current_price=stock_analysis['price'],
                recommendation=stock_analysis['recommendation'],
                confidence=stock_analysis['confidence']
            )

            # レスポンス構築（Enumを文字列に変換）
            response_data = {
                'symbol': symbol,
                'current_price': stock_analysis['price'],
                'buy_timing': {
                    'action': timing_strategy.buy_timing.action,
                    'timing_type': timing_strategy.buy_timing.timing_type.value,
                    'predicted_time': timing_strategy.buy_timing.predicted_time,
                    'confidence': timing_strategy.buy_timing.confidence,
                    'reasoning': timing_strategy.buy_timing.reasoning,
                    'market_condition': timing_strategy.buy_timing.market_condition.value,
                    'price_target': timing_strategy.buy_timing.price_target,
                    'risk_level': timing_strategy.buy_timing.risk_level
                },
                'sell_timing': {
                    'action': timing_strategy.sell_timing.action,
                    'timing_type': timing_strategy.sell_timing.timing_type.value,
                    'predicted_time': timing_strategy.sell_timing.predicted_time,
                    'confidence': timing_strategy.sell_timing.confidence,
                    'reasoning': timing_strategy.sell_timing.reasoning,
                    'market_condition': timing_strategy.sell_timing.market_condition.value,
                    'price_target': timing_strategy.sell_timing.price_target,
                    'stop_loss_target': timing_strategy.sell_timing.stop_loss_target,
                    'holding_period': timing_strategy.sell_timing.holding_period,
                    'risk_level': timing_strategy.sell_timing.risk_level
                },
                'strategy_type': timing_strategy.strategy_type,
                'expected_return': timing_strategy.expected_return,
                'max_risk': timing_strategy.max_risk,
                'time_horizon': timing_strategy.time_horizon,
                'market_events': timing_strategy.market_events,
                'timestamp': str(datetime.now())
            }

            return jsonify(response_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/profit-analysis')
    def api_profit_analysis():
        """収益分析API"""
        try:
            # 推奨銘柄を取得
            recommendations = recommendation_service.get_recommendations()
            buy_recommendations = [r for r in recommendations if r['recommendation'] in ['BUY', 'STRONG_BUY']]

            # 投資シナリオ
            scenarios = [
                {"name": "少額投資", "capital": 500000, "risk": "低"},
                {"name": "標準投資", "capital": 1000000, "risk": "中"},
                {"name": "積極投資", "capital": 2000000, "risk": "高"}
            ]

            analysis_results = []

            for scenario in scenarios:
                total_investment = 0
                total_expected = 0
                positions = []

                for stock in buy_recommendations:
                    price = stock.get('price', 0)
                    expected_return = stock.get('expected_return', 10.0)
                    confidence = stock.get('confidence', 0.7)

                    if price <= 0:
                        continue

                    # リスクレベル別の配分率
                    if scenario['risk'] == "低":
                        allocation_rate = min(0.05, confidence * 0.08)
                    elif scenario['risk'] == "中":
                        allocation_rate = min(0.10, confidence * 0.12)
                    else:  # 高
                        allocation_rate = min(0.15, confidence * 0.15)

                    investment_amount = scenario['capital'] * allocation_rate
                    shares = int(investment_amount / price / 100) * 100

                    if shares > 0:
                        actual_investment = shares * price
                        expected_profit = actual_investment * (expected_return / 100)

                        total_investment += actual_investment
                        total_expected += expected_profit

                        positions.append({
                            'symbol': stock['symbol'],
                            'name': stock['name'],
                            'price': price,
                            'shares': shares,
                            'investment': actual_investment,
                            'expected_profit': expected_profit,
                            'expected_return': expected_return,
                            'confidence': confidence
                        })

                # 期待利回り計算
                expected_rate = (total_expected / total_investment * 100) if total_investment > 0 else 0

                analysis_results.append({
                    'scenario_name': scenario['name'],
                    'capital': scenario['capital'],
                    'total_investment': total_investment,
                    'total_expected': total_expected,
                    'expected_rate': expected_rate,
                    'positions': positions,
                    'time_projections': {
                        '1_month': total_expected * 0.3,
                        '3_months': total_expected * 0.7,
                        '6_months': total_expected * 1.0
                    }
                })

            # 高期待銘柄ランキング
            top_stocks = sorted(buy_recommendations,
                              key=lambda x: x.get('expected_return', 0) * x.get('confidence', 0),
                              reverse=True)[:5]

            return jsonify({
                'success': True,
                'analysis_time': str(datetime.now()),
                'total_buy_recommendations': len(buy_recommendations),
                'scenarios': analysis_results,
                'top_stocks': top_stocks,
                'summary': {
                    'best_scenario_return': max([s['expected_rate'] for s in analysis_results]) if analysis_results else 0,
                    'total_opportunities': len(buy_recommendations),
                    'avg_expected_return': sum([s.get('expected_return', 0) for s in buy_recommendations]) / len(buy_recommendations) if buy_recommendations else 0
                }
            })

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    # 最適化システム API エンドポイント
    @app.route('/api/optimization/status')
    def optimization_status():
        """最適化システム状態API"""
        if not OPTIMIZATION_AVAILABLE:
            return jsonify({
                'available': False,
                'message': 'Optimization system not available'
            })
            
        try:
            status = integrated_optimizer.get_system_status()
            return jsonify({
                'available': True,
                'status': status,
                'timestamp': str(datetime.now())
            })
        except Exception as e:
            return jsonify({
                'available': False,
                'error': str(e)
            }), 500

    @app.route('/api/optimization/run', methods=['POST'])
    def run_optimization():
        """最適化実行API"""
        if not OPTIMIZATION_AVAILABLE:
            return jsonify({
                'success': False,
                'message': 'Optimization system not available'
            })
            
        try:
            import asyncio
            
            # 非同期実行のためのヘルパー関数
            async def run_async_optimization():
                return await integrated_optimizer.run_comprehensive_optimization()
            
            # イベントループで実行
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                report = loop.run_until_complete(run_async_optimization())
                return jsonify({
                    'success': True,
                    'report': {
                        'timestamp': report.timestamp.isoformat(),
                        'overall_score': report.overall_score,
                        'recommendations': report.recommendations,
                        'prediction_accuracy_status': report.prediction_accuracy.get('status'),
                        'performance_status': report.performance_metrics.get('status'),
                        'model_improvements_status': report.model_improvements.get('status'),
                        'response_speed_status': report.response_speed.get('status'),
                        'memory_efficiency_status': report.memory_efficiency.get('status')
                    }
                })
            finally:
                loop.close()
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/optimization/health')
    def optimization_health():
        """最適化システムヘルス API"""
        if not OPTIMIZATION_AVAILABLE:
            return jsonify({
                'available': False,
                'message': 'Optimization system not available'
            })
            
        try:
            import asyncio
            
            async def get_health_async():
                return await integrated_optimizer.check_system_health()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                health = loop.run_until_complete(get_health_async())
                return jsonify({
                    'available': True,
                    'health': {
                        'timestamp': health.timestamp.isoformat(),
                        'cpu_usage': health.cpu_usage,
                        'memory_usage': health.memory_usage,
                        'disk_usage': health.disk_usage,
                        'response_time': health.response_time,
                        'overall_health': health.overall_health
                    }
                })
            finally:
                loop.close()
                
        except Exception as e:
            return jsonify({
                'available': False,
                'error': str(e)
            }), 500

    @app.route('/api/optimization/history')
    def optimization_history():
        """最適化履歴API"""
        if not OPTIMIZATION_AVAILABLE:
            return jsonify({
                'available': False,
                'message': 'Optimization system not available'
            })
            
        try:
            history = integrated_optimizer.get_optimization_history(limit=10)
            
            formatted_history = []
            for report in history:
                formatted_history.append({
                    'timestamp': report.timestamp.isoformat(),
                    'overall_score': report.overall_score,
                    'recommendations_count': len(report.recommendations),
                    'status_summary': {
                        'prediction_accuracy': report.prediction_accuracy.get('status'),
                        'performance': report.performance_metrics.get('status'),
                        'model_improvements': report.model_improvements.get('status'),
                        'response_speed': report.response_speed.get('status'),
                        'memory_efficiency': report.memory_efficiency.get('status')
                    }
                })
            
            return jsonify({
                'available': True,
                'history': formatted_history,
                'count': len(formatted_history)
            })
            
        except Exception as e:
            return jsonify({
                'available': False,
                'error': str(e)
            }), 500

    return app

# Flask CLIで使用するためのアプリケーションインスタンス
app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)