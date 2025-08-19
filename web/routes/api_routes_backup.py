#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Routes Module - Issue #959 リファクタリング対応
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
                'Position Sizing'
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

    @app.route('/api/analyze/<symbol>', methods=['POST'])
    def api_start_analysis(symbol):
        """個別銘柄分析の非同期タスクを開始するAPI"""
        try:
            result = app.start_analysis_task(symbol)
            return jsonify(result), 202 # 202 Accepted
        except Exception as e:
            return jsonify({
                'error': str(e),
                'symbol': symbol,
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
    
    @app.route('/api/search-stocks')
    def api_search_stocks():
        """銘柄検索・フィルタリングAPI - Issues #955対応"""
        try:
            query = request.args.get('q', '').lower()
            category_filter = request.args.get('category', '').lower()
            
            if not query and not category_filter:
                return jsonify({'results': [], 'count': 0})
            
            # 35銘柄の拡張リスト
            all_stocks = [
                # 大型株（8銘柄）
                {'symbol': '7203', 'name': 'トヨタ自動車', 'sector': '自動車', 'category': '大型株'},
                {'symbol': '9984', 'name': 'ソフトバンクグループ', 'sector': 'テクノロジー', 'category': '大型株'},
                {'symbol': '6758', 'name': 'ソニー', 'sector': 'テクノロジー', 'category': '大型株'},
                {'symbol': '7974', 'name': '任天堂', 'sector': 'ゲーム', 'category': '大型株'},
                {'symbol': '8306', 'name': '三菱UFJ銀行', 'sector': '金融', 'category': '大型株'},
                {'symbol': '8001', 'name': '伊藤忠商事', 'sector': '商社', 'category': '大型株'},
                {'symbol': '4502', 'name': '武田薬品工業', 'sector': '医薬品', 'category': '大型株'},
                {'symbol': '6501', 'name': '日立製作所', 'sector': 'テクノロジー', 'category': '大型株'},
                
                # 中型株（8銘柄）
                {'symbol': '4755', 'name': '楽天グループ', 'sector': 'テクノロジー', 'category': '中型株'},
                {'symbol': '4385', 'name': 'メルカリ', 'sector': 'テクノロジー', 'category': '中型株'},
                {'symbol': '4689', 'name': 'Z Holdings', 'sector': 'テクノロジー', 'category': '中型株'},
                {'symbol': '9437', 'name': 'NTTドコモ', 'sector': '通信', 'category': '中型株'},
                {'symbol': '2914', 'name': '日本たばこ産業', 'sector': '食品・タバコ', 'category': '中型株'},
                {'symbol': '4704', 'name': 'トレンドマイクロ', 'sector': 'セキュリティ', 'category': '中型株'},
                {'symbol': '4751', 'name': 'サイバーエージェント', 'sector': 'テクノロジー', 'category': '中型株'},
                {'symbol': '3659', 'name': 'ネクソン', 'sector': 'ゲーム', 'category': '中型株'},
                
                # 高配当株（8銘柄）
                {'symbol': '8593', 'name': '三菱HCキャピタル', 'sector': '金融', 'category': '高配当株'},
                {'symbol': '8316', 'name': 'みずほフィナンシャルグループ', 'sector': '金融', 'category': '高配当株'},
                {'symbol': '5020', 'name': 'JXTGホールディングス', 'sector': 'エネルギー', 'category': '高配当株'},
                {'symbol': '8411', 'name': 'みずほ銀行', 'sector': '金融', 'category': '高配当株'},
                {'symbol': '9201', 'name': '日本航空', 'sector': '航空', 'category': '高配当株'},
                {'symbol': '9432', 'name': '日本電信電話', 'sector': '通信', 'category': '高配当株'},
                {'symbol': '8604', 'name': '野村ホールディングス', 'sector': '金融', 'category': '高配当株'},
                {'symbol': '7751', 'name': 'キヤノン', 'sector': '精密機器', 'category': '高配当株'},
                
                # 成長株（6銘柄）
                {'symbol': '4503', 'name': 'アステラス製薬', 'sector': '医薬品', 'category': '成長株'},
                {'symbol': '6981', 'name': '村田製作所', 'sector': '電子部品', 'category': '成長株'},
                {'symbol': '8035', 'name': '東京エレクトロン', 'sector': '半導体', 'category': '成長株'},
                {'symbol': '4704', 'name': 'トレンドマイクロ', 'sector': 'セキュリティ', 'category': '成長株'},
                {'symbol': '2491', 'name': 'バリューコマース', 'sector': 'テクノロジー', 'category': '成長株'},
                {'symbol': '3900', 'name': 'クラウドワークス', 'sector': 'テクノロジー', 'category': '成長株'},
                
                # 小型株（5銘柄）
                {'symbol': '4478', 'name': 'フリー', 'sector': 'テクノロジー', 'category': '小型株'},
                {'symbol': '4375', 'name': 'セーフィー', 'sector': 'テクノロジー', 'category': '小型株'},
                {'symbol': '4420', 'name': 'イーソル', 'sector': 'テクノロジー', 'category': '小型株'},
                {'symbol': '4475', 'name': 'HENNGE', 'sector': 'テクノロジー', 'category': '小型株'},
                {'symbol': '4478', 'name': 'フリー', 'sector': 'テクノロジー', 'category': '小型株'}
            ]
            
            # 検索・フィルタリング実行
            results = []
            for stock in all_stocks:
                match = False
                
                # カテゴリフィルター
                if category_filter and stock['category'].lower() != category_filter:
                    continue
                
                # テキスト検索
                if query:
                    if (query in stock['symbol'].lower() or 
                        query in stock['name'].lower() or 
                        query in stock['sector'].lower() or
                        query in stock['category'].lower()):
                        match = True
                else:
                    match = True  # カテゴリフィルターのみの場合
                
                if match:
                    results.append(stock)
            
            return jsonify({
                'results': results[:20],
                'count': len(results),
                'query': query or None,
                'category_filter': category_filter or None,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/scheduler/tasks')
    def api_scheduler_tasks():
        """スケジューラー状態API"""
        try:
            # ExecutionSchedulerからタスク情報を取得（模擬データ）
            scheduled_tasks = [
                {
                    'task_id': 'daily_analysis',
                    'name': '日次分析実行',
                    'schedule_type': 'daily',
                    'schedule_time': '09:00',
                    'status': 'ready',
                    'last_execution': '2025-08-18T09:00:00',
                    'next_execution': '2025-08-19T09:00:00',
                    'success_rate': 95.2
                },
                {
                    'task_id': 'market_data_update',
                    'name': '市場データ更新',
                    'schedule_type': 'hourly',
                    'interval_minutes': 30,
                    'status': 'running',
                    'last_execution': '2025-08-19T14:30:00',
                    'next_execution': '2025-08-19T15:00:00',
                    'success_rate': 98.7
                },
                {
                    'task_id': 'risk_monitoring',
                    'name': 'リスク監視',
                    'schedule_type': 'continuous',
                    'status': 'ready',
                    'last_execution': '2025-08-19T14:59:30',
                    'next_execution': '2025-08-19T15:00:00',
                    'success_rate': 99.1
                },
                {
                    'task_id': 'swing_analysis',
                    'name': 'スイングトレード分析',
                    'schedule_type': 'daily',
                    'schedule_time': '16:00',
                    'status': 'ready',
                    'last_execution': '2025-08-18T16:00:00',
                    'next_execution': '2025-08-19T16:00:00',
                    'success_rate': 92.8
                }
            ]
            
            return jsonify({
                'tasks': scheduled_tasks,
                'total_tasks': len(scheduled_tasks),
                'active_tasks': len([t for t in scheduled_tasks if t['status'] in ['running', 'ready']]),
                'avg_success_rate': round(sum(t['success_rate'] for t in scheduled_tasks) / len(scheduled_tasks), 1),
                'scheduler_status': 'running',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/scheduler/start/<task_id>', methods=['POST'])
    def api_scheduler_start_task(task_id):
        """スケジュールタスク手動実行API"""
        try:
            # 実際にはExecutionSchedulerのstart_task()を呼び出す
            # ここでは模擬応答
            
            task_names = {
                'daily_analysis': '日次分析実行',
                'market_data_update': '市場データ更新',
                'risk_monitoring': 'リスク監視',
                'swing_analysis': 'スイングトレード分析'
            }
            
            if task_id not in task_names:
                return jsonify({
                    'error': f'Task not found: {task_id}',
                    'status': 'error',
                    'timestamp': datetime.now().isoformat()
                }), 404
            
            return jsonify({
                'task_id': task_id,
                'task_name': task_names[task_id],
                'status': 'started',
                'message': f'{task_names[task_id]}を手動実行しました',
                'execution_id': f'manual_{task_id}_{int(datetime.now().timestamp())}',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/swing-trade-stocks')
    def api_swing_trade_stocks():
        """スイングトレード向け銘柄API"""
        try:
            swing_stocks = [
                {
                    'symbol': '7203', 'name': 'トヨタ自動車',
                    'trend_strength': 85, 'volatility': 65, 'liquidity': 95,
                    'holding_period': '1-4週間', 'risk_level': '低',
                    'target_gain': '5-15%', 'stop_loss': '3-5%',
                    'technical_setup': 'アップトレンド継続中'
                },
                {
                    'symbol': '6758', 'name': 'ソニー',
                    'trend_strength': 80, 'volatility': 70, 'liquidity': 88,
                    'holding_period': '2-6週間', 'risk_level': '中',
                    'target_gain': '8-20%', 'stop_loss': '4-6%',
                    'technical_setup': '支持線からの反発'
                },
                {
                    'symbol': '8035', 'name': '東京エレクトロン',
                    'trend_strength': 90, 'volatility': 85, 'liquidity': 78,
                    'holding_period': '2-8週間', 'risk_level': '中',
                    'target_gain': '10-25%', 'stop_loss': '5-8%',
                    'technical_setup': 'ブレイクアウト継続'
                },
                {
                    'symbol': '4755', 'name': '楽天グループ',
                    'trend_strength': 75, 'volatility': 90, 'liquidity': 85,
                    'holding_period': '1-3週間', 'risk_level': '高',
                    'target_gain': '15-30%', 'stop_loss': '6-10%',
                    'technical_setup': 'ボラティリティ拡大'
                }
            ]
            
            return jsonify({
                'stocks': swing_stocks,
                'count': len(swing_stocks),
                'trading_style': 'スイングトレード',
                'tips': [
                    'トレンドの方向に従ってエントリー',
                    'リスクは資金の2-5%に制限',
                    '明確な損切りラインを設定',
                    '利確目標は損切り幅の2-3倍'
                ],
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/scheduler/tasks')
    def api_scheduler_tasks():
        """スケジューラー状態API"""
        try:
            # ExecutionSchedulerからタスク情報を取得（模擬データ）
            scheduled_tasks = [
                {
                    'task_id': 'daily_analysis',
                    'name': '日次分析実行',
                    'schedule_type': 'daily',
                    'schedule_time': '09:00',
                    'status': 'ready',
                    'last_execution': '2025-08-18T09:00:00',
                    'next_execution': '2025-08-19T09:00:00',
                    'success_rate': 95.2
                },
                {
                    'task_id': 'market_data_update',
                    'name': '市場データ更新',
                    'schedule_type': 'hourly',
                    'interval_minutes': 30,
                    'status': 'running',
                    'last_execution': '2025-08-19T14:30:00',
                    'next_execution': '2025-08-19T15:00:00',
                    'success_rate': 98.7
                },
                {
                    'task_id': 'risk_monitoring',
                    'name': 'リスク監視',
                    'schedule_type': 'continuous',
                    'status': 'ready',
                    'last_execution': '2025-08-19T14:59:30',
                    'next_execution': '2025-08-19T15:00:00',
                    'success_rate': 99.1
                },
                {
                    'task_id': 'swing_analysis',
                    'name': 'スイングトレード分析',
                    'schedule_type': 'daily',
                    'schedule_time': '16:00',
                    'status': 'ready',
                    'last_execution': '2025-08-18T16:00:00',
                    'next_execution': '2025-08-19T16:00:00',
                    'success_rate': 92.8
                }
            ]
            
            return jsonify({
                'tasks': scheduled_tasks,
                'total_tasks': len(scheduled_tasks),
                'active_tasks': len([t for t in scheduled_tasks if t['status'] in ['running', 'ready']]),
                'avg_success_rate': round(sum(t['success_rate'] for t in scheduled_tasks) / len(scheduled_tasks), 1),
                'scheduler_status': 'running',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/scheduler/start/<task_id>', methods=['POST'])
    def api_scheduler_start_task(task_id):
        """スケジュールタスク手動実行API"""
        try:
            # 実際にはExecutionSchedulerのstart_task()を呼び出す
            # ここでは模擬応答
            
            task_names = {
                'daily_analysis': '日次分析実行',
                'market_data_update': '市場データ更新',
                'risk_monitoring': 'リスク監視',
                'swing_analysis': 'スイングトレード分析'
            }
            
            if task_id not in task_names:
                return jsonify({
                    'error': f'Task not found: {task_id}',
                    'status': 'error',
                    'timestamp': datetime.now().isoformat()
                }), 404
            
            return jsonify({
                'task_id': task_id,
                'task_name': task_names[task_id],
                'status': 'started',
                'message': f'{task_names[task_id]}を手動実行しました',
                'execution_id': f'manual_{task_id}_{int(datetime.now().timestamp())}',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }), 500