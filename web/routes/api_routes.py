#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Routes Module - Issue #959 リファクタリング対応
API エンドポイント定義モジュール
"""

from flask import Flask, jsonify, request
from datetime import datetime
import time
from typing import Dict, Any, List

# ポートフォリオサービスのインポート
try:
    from web.services.portfolio_service import PortfolioService
    PORTFOLIO_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"ポートフォリオサービス読み込みエラー: {e}")
    PORTFOLIO_SERVICE_AVAILABLE = False

def setup_api_routes(app: Flask, web_server_instance) -> None:
    """APIルート設定"""
    
    # ポートフォリオ機能の統合
    try:
        from web.routes.api_routes_portfolio import setup_portfolio_routes
        setup_portfolio_routes(app)
        print("ポートフォリオAPIルートが設定されました")
    except ImportError as e:
        print(f"ポートフォリオAPI設定エラー: {e}")
    
    # アラート機能の統合
    try:
        from web.routes.api_routes_alerts import setup_alert_routes
        setup_alert_routes(app)
        print("アラートAPIルートが設定されました")
    except ImportError as e:
        print(f"アラートAPI設定エラー: {e}")
    
    # バックテスト機能の統合
    try:
        from web.routes.api_routes_backtest import setup_backtest_routes
        setup_backtest_routes(app)
        print("バックテストAPIルートが設定されました")
    except ImportError as e:
        print(f"バックテストAPI設定エラー: {e}")
    
    @app.route('/api/status')
    def api_status():
        """システム状態API"""
        return jsonify({
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'version': getattr(web_server_instance, 'version_info', {}).get('version', '2.1.0'),
            'features': [
                'Real-time Analysis',
                'Portfolio Management',
                'Alert System',
                'Backtest Engine',
                'Database Integration'
            ]
        })
    
    @app.route('/api/recommendations')
    def api_recommendations():
        """推奨銘柄API"""
        try:
            # 35銘柄の推奨システム
            recommendations = web_server_instance._get_recommendations()
            
            # 統計計算
            total_count = len(recommendations)
            high_confidence_count = len([r for r in recommendations if r.get('confidence', 0) > 0.8])
            buy_count = len([r for r in recommendations if r.get('recommendation') == 'BUY'])
            sell_count = len([r for r in recommendations if r.get('recommendation') == 'SELL'])
            hold_count = len([r for r in recommendations if r.get('recommendation') == 'HOLD'])
            
            return jsonify({
                'total_count': total_count,
                'high_confidence_count': high_confidence_count,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'hold_count': hold_count,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/analysis/<symbol>')
    def api_single_analysis(symbol):
        """個別銘柄分析API"""
        try:
            result = web_server_instance._analyze_single_symbol(symbol)
            return jsonify(result)
        except Exception as e:
            return jsonify({
                'error': str(e),
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/realtime/<symbol>')
    def api_realtime_price(symbol):
        """リアルタイム価格取得API"""
        try:
            import yfinance as yf
            
            # 日本株式のシンボル形式に変換
            jp_symbol = f"{symbol}.T"
            ticker = yf.Ticker(jp_symbol)
            
            # リアルタイム情報取得
            info = ticker.info
            hist = ticker.history(period="2d")  # 最新2日分
            
            if len(hist) > 0:
                current_price = info.get('currentPrice') or info.get('regularMarketPrice') or hist['Close'].iloc[-1]
                prev_close = info.get('previousClose') or hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                
                price_change = current_price - prev_close
                price_change_pct = (price_change / prev_close) * 100 if prev_close > 0 else 0
                
                return jsonify({
                    'symbol': symbol,
                    'name': info.get('longName', f'株式会社{symbol}'),
                    'current_price': round(current_price, 2),
                    'previous_close': round(prev_close, 2),
                    'price_change': round(price_change, 2),
                    'price_change_pct': round(price_change_pct, 2),
                    'day_high': round(info.get('dayHigh', current_price), 2),
                    'day_low': round(info.get('dayLow', current_price), 2),
                    'volume': info.get('volume', 0),
                    'market_cap': info.get('marketCap', 0),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                })
            else:
                return jsonify({
                    'error': 'データを取得できませんでした',
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat()
                }), 404
                
        except ImportError:
            return jsonify({
                'error': 'Yahoo Finance APIが利用できません',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }), 503
        except Exception as e:
            return jsonify({
                'error': str(e),
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/realtime/batch')
    def api_realtime_batch():
        """複数銘柄のリアルタイム価格一括取得API"""
        try:
            symbols = request.args.get('symbols', '').split(',')
            symbols = [s.strip() for s in symbols if s.strip()]
            
            if not symbols:
                return jsonify({
                    'error': 'シンボルが指定されていません',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            import yfinance as yf
            
            results = []
            for symbol in symbols[:10]:  # 最大10銘柄まで
                try:
                    jp_symbol = f"{symbol}.T"
                    ticker = yf.Ticker(jp_symbol)
                    info = ticker.info
                    hist = ticker.history(period="2d")
                    
                    if len(hist) > 0:
                        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or hist['Close'].iloc[-1]
                        prev_close = info.get('previousClose') or hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        
                        price_change = current_price - prev_close
                        price_change_pct = (price_change / prev_close) * 100 if prev_close > 0 else 0
                        
                        results.append({
                            'symbol': symbol,
                            'current_price': round(current_price, 2),
                            'price_change': round(price_change, 2),
                            'price_change_pct': round(price_change_pct, 2),
                            'volume': info.get('volume', 0),
                            'status': 'success'
                        })
                    else:
                        results.append({
                            'symbol': symbol,
                            'error': 'データ取得失敗',
                            'status': 'error'
                        })
                        
                except Exception as e:
                    results.append({
                        'symbol': symbol,
                        'error': str(e),
                        'status': 'error'
                    })
            
            return jsonify({
                'results': results,
                'total_requested': len(symbols),
                'successful': len([r for r in results if r.get('status') == 'success']),
                'timestamp': datetime.now().isoformat()
            })
            
        except ImportError:
            return jsonify({
                'error': 'Yahoo Finance APIが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500