#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest API Routes Module - バックテストAPI
バックテスト関連のAPIエンドポイント定義
"""

from flask import Flask, jsonify, request
from datetime import datetime
from typing import Dict, Any, List

# バックテストサービスのインポート
try:
    from web.services.backtest_service import BacktestService
    BACKTEST_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"バックテストサービス読み込みエラー: {e}")
    BACKTEST_SERVICE_AVAILABLE = False

def setup_backtest_routes(app: Flask) -> None:
    """バックテストAPIルート設定"""
    
    # バックテストサービス初期化
    backtest_service = BacktestService() if BACKTEST_SERVICE_AVAILABLE else None
    
    @app.route('/api/backtest/strategies')
    def api_backtest_strategies():
        """利用可能な戦略一覧API"""
        if not BACKTEST_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'バックテストサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        try:
            strategies = backtest_service.get_strategy_templates()
            return jsonify({
                'strategies': strategies,
                'count': len(strategies),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/backtest/run', methods=['POST'])
    def api_backtest_run():
        """バックテスト実行API"""
        if not BACKTEST_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'バックテストサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        try:
            data = request.get_json()
            
            # 必須フィールドチェック
            required_fields = ['strategy_name', 'symbol']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        'error': f'必須フィールドが不足: {field}',
                        'timestamp': datetime.now().isoformat()
                    }), 400
            
            # バックテスト実行
            result = backtest_service.run_backtest(
                strategy_name=data['strategy_name'],
                strategy_params=data.get('strategy_params', {}),
                symbol=data['symbol'],
                start_date=data.get('start_date'),
                end_date=data.get('end_date')
            )
            
            if 'error' in result:
                return jsonify({
                    'error': result['error'],
                    'timestamp': datetime.now().isoformat()
                }), 500
            
            return jsonify({
                'success': True,
                'result': result,
                'message': 'バックテストが完了しました',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/backtest/results')
    def api_backtest_results():
        """バックテスト結果一覧API"""
        if not BACKTEST_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'バックテストサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        try:
            limit = int(request.args.get('limit', 20))
            results = backtest_service.get_saved_results(limit)
            
            return jsonify({
                'results': results,
                'count': len(results),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/backtest/quick-test', methods=['POST'])
    def api_backtest_quick_test():
        """クイックバックテストAPI"""
        if not BACKTEST_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'バックテストサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        try:
            data = request.get_json()
            symbol = data.get('symbol', '7203')
            
            # 複数戦略での簡単比較
            strategies_to_test = [
                {
                    'name': 'sma',
                    'params': {'short_window': 5, 'long_window': 20},
                    'display_name': 'SMA(5,20)'
                },
                {
                    'name': 'sma',
                    'params': {'short_window': 10, 'long_window': 30},
                    'display_name': 'SMA(10,30)'
                },
                {
                    'name': 'rsi',
                    'params': {'rsi_period': 14, 'oversold': 30, 'overbought': 70},
                    'display_name': 'RSI(14,30,70)'
                }
            ]
            
            results = []
            for strategy in strategies_to_test:
                try:
                    result = backtest_service.run_backtest(
                        strategy_name=strategy['name'],
                        strategy_params=strategy['params'],
                        symbol=symbol,
                        start_date=data.get('start_date'),
                        end_date=data.get('end_date')
                    )
                    
                    if 'error' not in result:
                        # 要約データのみ抽出
                        summary = {
                            'strategy_name': strategy['display_name'],
                            'total_return_pct': result.get('total_return_pct', 0),
                            'win_rate': result.get('win_rate', 0),
                            'max_drawdown_pct': result.get('max_drawdown_pct', 0),
                            'sharpe_ratio': result.get('sharpe_ratio', 0),
                            'total_trades': result.get('total_trades', 0)
                        }
                        results.append(summary)
                    
                except Exception as strategy_error:
                    results.append({
                        'strategy_name': strategy['display_name'],
                        'error': str(strategy_error)
                    })
            
            return jsonify({
                'success': True,
                'symbol': symbol,
                'comparison_results': results,
                'message': f'{symbol} のクイック戦略比較が完了しました',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/backtest/optimize', methods=['POST'])
    def api_backtest_optimize():
        """戦略最適化API"""
        if not BACKTEST_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'バックテストサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        try:
            data = request.get_json()
            
            required_fields = ['strategy_name', 'symbol', 'parameter_ranges']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        'error': f'必須フィールドが不足: {field}',
                        'timestamp': datetime.now().isoformat()
                    }), 400
            
            strategy_name = data['strategy_name']
            symbol = data['symbol']
            parameter_ranges = data['parameter_ranges']
            
            # パラメータ組み合わせ生成
            optimization_results = []
            
            if strategy_name == 'sma':
                # SMA戦略の最適化
                short_windows = parameter_ranges.get('short_window', [5, 10, 15])
                long_windows = parameter_ranges.get('long_window', [20, 30, 50])
                
                for short in short_windows:
                    for long in long_windows:
                        if short < long:  # 短期 < 長期の条件
                            try:
                                result = backtest_service.run_backtest(
                                    strategy_name=strategy_name,
                                    strategy_params={'short_window': short, 'long_window': long},
                                    symbol=symbol,
                                    start_date=data.get('start_date'),
                                    end_date=data.get('end_date')
                                )
                                
                                if 'error' not in result:
                                    optimization_results.append({
                                        'parameters': {'short_window': short, 'long_window': long},
                                        'total_return_pct': result.get('total_return_pct', 0),
                                        'sharpe_ratio': result.get('sharpe_ratio', 0),
                                        'max_drawdown_pct': result.get('max_drawdown_pct', 0),
                                        'win_rate': result.get('win_rate', 0)
                                    })
                                    
                            except Exception as param_error:
                                continue
            
            elif strategy_name == 'rsi':
                # RSI戦略の最適化
                rsi_periods = parameter_ranges.get('rsi_period', [10, 14, 20])
                oversolds = parameter_ranges.get('oversold', [20, 30, 40])
                overboughts = parameter_ranges.get('overbought', [60, 70, 80])
                
                for period in rsi_periods:
                    for oversold in oversolds:
                        for overbought in overboughts:
                            if oversold < overbought:  # oversold < overbought の条件
                                try:
                                    result = backtest_service.run_backtest(
                                        strategy_name=strategy_name,
                                        strategy_params={
                                            'rsi_period': period,
                                            'oversold': oversold,
                                            'overbought': overbought
                                        },
                                        symbol=symbol,
                                        start_date=data.get('start_date'),
                                        end_date=data.get('end_date')
                                    )
                                    
                                    if 'error' not in result:
                                        optimization_results.append({
                                            'parameters': {
                                                'rsi_period': period,
                                                'oversold': oversold,
                                                'overbought': overbought
                                            },
                                            'total_return_pct': result.get('total_return_pct', 0),
                                            'sharpe_ratio': result.get('sharpe_ratio', 0),
                                            'max_drawdown_pct': result.get('max_drawdown_pct', 0),
                                            'win_rate': result.get('win_rate', 0)
                                        })
                                        
                                except Exception as param_error:
                                    continue
            
            # 結果をソート（リターン順）
            optimization_results.sort(key=lambda x: x['total_return_pct'], reverse=True)
            
            # 最適パラメータ
            best_params = optimization_results[0] if optimization_results else None
            
            return jsonify({
                'success': True,
                'symbol': symbol,
                'strategy_name': strategy_name,
                'optimization_results': optimization_results[:10],  # 上位10件
                'best_parameters': best_params,
                'total_combinations_tested': len(optimization_results),
                'message': f'{strategy_name} 戦略の最適化が完了しました',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500