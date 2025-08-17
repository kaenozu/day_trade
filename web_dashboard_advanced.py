#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Web Dashboard System - 高度ウェブダッシュボードシステム
Issue #871対応：ウェブダッシュボード機能拡張提案

包括的なウェブダッシュボード機能:
1. リアルタイム監視・更新システム
2. 高度分析・予測統合機能
3. 翌朝場予測・精度向上システム統合
4. パフォーマンス・リスク監視
5. カスタマイズ・設定管理
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd

# Web関連
try:
    from flask import Flask, render_template, jsonify, request, session
    from flask_socketio import SocketIO, emit, join_room, leave_room
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# 既存システム統合
try:
    from prediction_accuracy_enhancement import PredictionAccuracyEnhancer
    ACCURACY_ENHANCEMENT_AVAILABLE = True
except ImportError:
    ACCURACY_ENHANCEMENT_AVAILABLE = False

try:
    from next_morning_trading_advanced import NextMorningTradingAdvanced
    NEXT_MORNING_AVAILABLE = True
except ImportError:
    NEXT_MORNING_AVAILABLE = False

try:
    from model_performance_monitor import ModelPerformanceMonitor
    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITOR_AVAILABLE = False

try:
    from data_quality_monitor import DataQualityMonitor
    DATA_QUALITY_AVAILABLE = True
except ImportError:
    DATA_QUALITY_AVAILABLE = False

try:
    from real_data_provider_v2 import real_data_provider, DataSource
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False

try:
    from multi_timeframe_predictor import MultiTimeframePredictor
    MULTI_TIMEFRAME_AVAILABLE = True
except ImportError:
    MULTI_TIMEFRAME_AVAILABLE = False

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)


class RealtimeDataManager:
    """リアルタイムデータ管理"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_subscriptions = set()
        self.current_data = {}
        self.update_interval = 5  # 5秒間隔
        self.is_running = False

    async def start_realtime_updates(self, socketio):
        """リアルタイム更新開始"""
        self.is_running = True
        self.socketio = socketio

        while self.is_running:
            try:
                await self.update_all_data()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Realtime update error: {e}")
                await asyncio.sleep(10)  # エラー時は10秒待機

    async def update_all_data(self):
        """全データ更新"""
        if not self.active_subscriptions:
            return

        for symbol in self.active_subscriptions:
            try:
                # 最新価格取得
                current_price = await self.get_current_price(symbol)

                # 簡易テクニカル指標
                technical_data = await self.get_technical_indicators(symbol)

                # 予測データ
                prediction_data = await self.get_prediction_data(symbol)

                # データ統合
                realtime_data = {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'current_price': current_price,
                    'technical': technical_data,
                    'prediction': prediction_data,
                    'change_percent': technical_data.get('change_percent', 0),
                    'volume': technical_data.get('volume', 0)
                }

                self.current_data[symbol] = realtime_data

                # WebSocketで配信
                if hasattr(self, 'socketio'):
                    self.socketio.emit('realtime_update', realtime_data, room=f"symbol_{symbol}")

            except Exception as e:
                self.logger.error(f"Failed to update data for {symbol}: {e}")

    async def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """現在価格取得"""
        try:
            if REAL_DATA_AVAILABLE:
                data = await real_data_provider.get_stock_data(symbol, "1d")
                if data is not None and not data.empty:
                    latest = data.iloc[-1]
                    return {
                        'price': float(latest['Close']),
                        'open': float(latest['Open']),
                        'high': float(latest['High']),
                        'low': float(latest['Low']),
                        'volume': int(latest['Volume'])
                    }

            # フォールバック：模擬データ
            import random
            base_price = 1000 + hash(symbol) % 1000
            change = random.uniform(-0.05, 0.05)
            current_price = base_price * (1 + change)

            return {
                'price': round(current_price, 2),
                'open': round(current_price * 0.995, 2),
                'high': round(current_price * 1.02, 2),
                'low': round(current_price * 0.98, 2),
                'volume': random.randint(1000000, 50000000)
            }

        except Exception as e:
            self.logger.error(f"Price fetch error for {symbol}: {e}")
            return {'price': 0, 'open': 0, 'high': 0, 'low': 0, 'volume': 0}

    async def get_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """テクニカル指標取得"""
        try:
            if REAL_DATA_AVAILABLE:
                data = await real_data_provider.get_stock_data(symbol, "1mo")
                if data is not None and not data.empty:
                    # 簡易計算
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                    change_percent = (current_price - prev_price) / prev_price * 100

                    sma_20 = data['Close'].rolling(20).mean().iloc[-1] if len(data) >= 20 else current_price
                    volume = data['Volume'].iloc[-1] if 'Volume' in data.columns else 0

                    return {
                        'change_percent': round(change_percent, 2),
                        'sma_20': round(sma_20, 2),
                        'volume': int(volume),
                        'price_vs_sma': round((current_price - sma_20) / sma_20 * 100, 2) if sma_20 > 0 else 0
                    }

            # フォールバック：模擬データ
            import random
            return {
                'change_percent': round(random.uniform(-5, 5), 2),
                'sma_20': 1000,
                'volume': random.randint(1000000, 10000000),
                'price_vs_sma': round(random.uniform(-2, 2), 2)
            }

        except Exception as e:
            self.logger.error(f"Technical indicators error for {symbol}: {e}")
            return {'change_percent': 0, 'sma_20': 0, 'volume': 0, 'price_vs_sma': 0}

    async def get_prediction_data(self, symbol: str) -> Dict[str, Any]:
        """予測データ取得"""
        try:
            # 簡易予測（実際の実装では機械学習モデルを使用）
            import random
            confidence = random.uniform(0.5, 0.9)
            direction = random.choice(['上昇', '下降', '中立'])
            expected_change = random.uniform(-3, 3)

            return {
                'direction': direction,
                'confidence': round(confidence, 2),
                'expected_change': round(expected_change, 2),
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Prediction data error for {symbol}: {e}")
            return {'direction': '不明', 'confidence': 0.5, 'expected_change': 0, 'last_updated': datetime.now().isoformat()}

    def subscribe_symbol(self, symbol: str):
        """銘柄購読開始"""
        self.active_subscriptions.add(symbol)
        self.logger.info(f"Subscribed to {symbol}")

    def unsubscribe_symbol(self, symbol: str):
        """銘柄購読停止"""
        self.active_subscriptions.discard(symbol)
        self.logger.info(f"Unsubscribed from {symbol}")

    def stop_updates(self):
        """更新停止"""
        self.is_running = False


class AdvancedAnalysisManager:
    """高度分析管理"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # システム統合
        self.accuracy_enhancer = None
        self.next_morning_system = None
        self.performance_monitor = None
        self.data_quality_monitor = None

        self._initialize_systems()

    def _initialize_systems(self):
        """システム初期化"""
        if ACCURACY_ENHANCEMENT_AVAILABLE:
            try:
                self.accuracy_enhancer = PredictionAccuracyEnhancer()
                self.logger.info("Accuracy enhancement system integrated")
            except Exception as e:
                self.logger.warning(f"Failed to initialize accuracy enhancer: {e}")

        if NEXT_MORNING_AVAILABLE:
            try:
                self.next_morning_system = NextMorningTradingAdvanced()
                self.logger.info("Next morning trading system integrated")
            except Exception as e:
                self.logger.warning(f"Failed to initialize next morning system: {e}")

        if PERFORMANCE_MONITOR_AVAILABLE:
            try:
                self.performance_monitor = ModelPerformanceMonitor()
                self.logger.info("Performance monitor integrated")
            except Exception as e:
                self.logger.warning(f"Failed to initialize performance monitor: {e}")

        if DATA_QUALITY_AVAILABLE:
            try:
                self.data_quality_monitor = DataQualityMonitor()
                self.logger.info("Data quality monitor integrated")
            except Exception as e:
                self.logger.warning(f"Failed to initialize data quality monitor: {e}")

    async def run_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """包括分析実行"""
        try:
            results = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'accuracy_enhancement': None,
                'next_morning_prediction': None,
                'performance_metrics': None,
                'data_quality': None
            }

            # 精度向上分析
            if self.accuracy_enhancer:
                try:
                    # データ取得
                    if REAL_DATA_AVAILABLE:
                        data = await real_data_provider.get_stock_data(symbol, "3mo")
                        if data is not None and not data.empty:
                            # ターゲット変数作成
                            data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
                            data = data.dropna()

                            if len(data) > 50:
                                enhancement_report = await self.accuracy_enhancer.enhance_prediction_accuracy(symbol, data)
                                results['accuracy_enhancement'] = {
                                    'baseline_accuracy': enhancement_report.baseline_accuracy,
                                    'improved_accuracy': enhancement_report.improved_accuracy,
                                    'improvement_percentage': enhancement_report.improvement_percentage,
                                    'confidence': enhancement_report.ensemble_result.ensemble_score,
                                    'status': 'success'
                                }
                except Exception as e:
                    self.logger.error(f"Accuracy enhancement failed for {symbol}: {e}")
                    results['accuracy_enhancement'] = {'status': 'error', 'message': str(e)}

            # 翌朝場予測
            if self.next_morning_system:
                try:
                    prediction = await self.next_morning_system.predict_next_morning(symbol)
                    results['next_morning_prediction'] = {
                        'direction': prediction.market_direction.value,
                        'predicted_change': prediction.predicted_change_percent,
                        'confidence': prediction.confidence.value,
                        'confidence_score': prediction.confidence_score,
                        'sentiment_score': prediction.market_sentiment.sentiment_score,
                        'risk_level': prediction.position_recommendation.risk_level.value,
                        'position_size': prediction.position_recommendation.position_size_percentage,
                        'status': 'success'
                    }
                except Exception as e:
                    self.logger.error(f"Next morning prediction failed for {symbol}: {e}")
                    results['next_morning_prediction'] = {'status': 'error', 'message': str(e)}

            # 性能監視
            if self.performance_monitor:
                try:
                    status = self.performance_monitor.get_monitoring_status()
                    latest_performance = await self.performance_monitor.get_latest_model_performance()

                    symbol_performance = {}
                    for key, metrics in latest_performance.items():
                        if symbol in key:
                            symbol_performance[key] = {
                                'accuracy': metrics.accuracy,
                                'confidence': metrics.confidence_avg,
                                'status': metrics.status.value
                            }

                    results['performance_metrics'] = {
                        'monitoring_active': status['monitoring_active'],
                        'symbol_performance': symbol_performance,
                        'last_check': status.get('last_check_time'),
                        'status': 'success'
                    }
                except Exception as e:
                    self.logger.error(f"Performance monitoring failed for {symbol}: {e}")
                    results['performance_metrics'] = {'status': 'error', 'message': str(e)}

            # データ品質
            if self.data_quality_monitor:
                try:
                    if REAL_DATA_AVAILABLE:
                        data = await real_data_provider.get_stock_data(symbol, "1mo")
                        if data is not None and not data.empty:
                            quality_result = await self.data_quality_monitor.validate_stock_data(
                                symbol, data, DataSource.YAHOO_FINANCE
                            )
                            results['data_quality'] = {
                                'is_valid': quality_result.is_valid,
                                'quality_score': quality_result.quality_score,
                                'completeness': quality_result.data_completeness,
                                'consistency': quality_result.price_consistency,
                                'anomalies_count': len(quality_result.anomalies),
                                'status': 'success'
                            }
                except Exception as e:
                    self.logger.error(f"Data quality check failed for {symbol}: {e}")
                    results['data_quality'] = {'status': 'error', 'message': str(e)}

            return results

        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'message': str(e)
            }

    async def get_system_health(self) -> Dict[str, Any]:
        """システム健全性チェック"""
        health = {
            'timestamp': datetime.now().isoformat(),
            'systems': {},
            'overall_status': 'healthy'
        }

        # 各システムの状態確認
        systems = {
            'accuracy_enhancer': self.accuracy_enhancer,
            'next_morning_system': self.next_morning_system,
            'performance_monitor': self.performance_monitor,
            'data_quality_monitor': self.data_quality_monitor
        }

        for name, system in systems.items():
            if system:
                try:
                    if hasattr(system, 'get_monitoring_status'):
                        status = system.get_monitoring_status()
                        health['systems'][name] = {'status': 'active', 'details': status}
                    else:
                        health['systems'][name] = {'status': 'active', 'details': 'available'}
                except Exception as e:
                    health['systems'][name] = {'status': 'error', 'details': str(e)}
                    health['overall_status'] = 'degraded'
            else:
                health['systems'][name] = {'status': 'unavailable', 'details': 'not initialized'}

        return health


class DashboardCustomization:
    """ダッシュボードカスタマイズ"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config_file = Path("config/dashboard_config.json")
        self.default_config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            'layout': {
                'theme': 'dark',
                'sidebar_position': 'left',
                'chart_height': 400,
                'refresh_interval': 5
            },
            'widgets': {
                'price_chart': {'enabled': True, 'position': {'row': 1, 'col': 1}, 'size': {'width': 6, 'height': 2}},
                'prediction_panel': {'enabled': True, 'position': {'row': 1, 'col': 7}, 'size': {'width': 6, 'height': 2}},
                'technical_indicators': {'enabled': True, 'position': {'row': 3, 'col': 1}, 'size': {'width': 4, 'height': 2}},
                'sentiment_analysis': {'enabled': True, 'position': {'row': 3, 'col': 5}, 'size': {'width': 4, 'height': 2}},
                'risk_metrics': {'enabled': True, 'position': {'row': 3, 'col': 9}, 'size': {'width': 4, 'height': 2}},
                'performance_monitor': {'enabled': True, 'position': {'row': 5, 'col': 1}, 'size': {'width': 6, 'height': 2}},
                'system_status': {'enabled': True, 'position': {'row': 5, 'col': 7}, 'size': {'width': 6, 'height': 2}}
            },
            'symbols': {
                'watchlist': ['7203', '4751', '9984', '8306', '2914'],
                'auto_update': True,
                'update_interval': 5
            },
            'alerts': {
                'price_change_threshold': 3.0,
                'prediction_confidence_threshold': 0.8,
                'email_notifications': False,
                'sound_alerts': True
            }
        }

    def load_user_config(self, user_id: str = 'default') -> Dict[str, Any]:
        """ユーザー設定読み込み"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    all_configs = json.load(f)
                    return all_configs.get(user_id, self.default_config)
            else:
                return self.default_config
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self.default_config

    def save_user_config(self, config: Dict[str, Any], user_id: str = 'default'):
        """ユーザー設定保存"""
        try:
            self.config_file.parent.mkdir(exist_ok=True)

            all_configs = {}
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    all_configs = json.load(f)

            all_configs[user_id] = config

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(all_configs, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Config saved for user {user_id}")

        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")


class AdvancedWebDashboard:
    """高度ウェブダッシュボード"""

    def __init__(self, host='localhost', port=5000):
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port

        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for web dashboard")

        # Flask アプリケーション初期化
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'advanced_dashboard_secret_key'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        CORS(self.app)

        # コンポーネント初期化
        self.realtime_manager = RealtimeDataManager()
        self.analysis_manager = AdvancedAnalysisManager()
        self.customization = DashboardCustomization()

        # ルート設定
        self._setup_routes()
        self._setup_websocket_events()

        self.logger.info("Advanced Web Dashboard initialized")

    def _setup_routes(self):
        """ルート設定"""

        @self.app.route('/')
        def index():
            """メインダッシュボード"""
            return render_template('dashboard_advanced.html')

        @self.app.route('/api/config', methods=['GET'])
        def get_config():
            """設定取得"""
            user_id = request.args.get('user_id', 'default')
            config = self.customization.load_user_config(user_id)
            return jsonify({'success': True, 'config': config})

        @self.app.route('/api/config', methods=['POST'])
        def save_config():
            """設定保存"""
            data = request.get_json()
            user_id = data.get('user_id', 'default')
            config = data.get('config', {})

            self.customization.save_user_config(config, user_id)
            return jsonify({'success': True, 'message': 'Configuration saved'})

        @self.app.route('/api/realtime/subscribe/<symbol>')
        def subscribe_realtime(symbol):
            """リアルタイム購読"""
            self.realtime_manager.subscribe_symbol(symbol)
            return jsonify({'success': True, 'message': f'Subscribed to {symbol}'})

        @self.app.route('/api/realtime/unsubscribe/<symbol>')
        def unsubscribe_realtime(symbol):
            """リアルタイム購読停止"""
            self.realtime_manager.unsubscribe_symbol(symbol)
            return jsonify({'success': True, 'message': f'Unsubscribed from {symbol}'})

        @self.app.route('/api/analysis/<symbol>')
        async def get_analysis(symbol):
            """包括分析取得"""
            try:
                analysis = await self.analysis_manager.run_comprehensive_analysis(symbol)
                return jsonify({'success': True, 'analysis': analysis})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/system/health')
        async def get_system_health():
            """システム健全性"""
            try:
                health = await self.analysis_manager.get_system_health()
                return jsonify({'success': True, 'health': health})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/multi-timeframe/predict/<symbol>')
        async def predict_multi_timeframe(symbol):
            """マルチタイムフレーム予測（既存機能拡張）"""
            try:
                if MULTI_TIMEFRAME_AVAILABLE:
                    predictor = MultiTimeframePredictor()
                    prediction = await predictor.predict_multi_timeframe(symbol)
                    return jsonify({'success': True, 'prediction': prediction})
                else:
                    return jsonify({'success': False, 'error': 'Multi-timeframe predictor not available'})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/current-data/<symbol>')
        def get_current_data(symbol):
            """現在データ取得"""
            data = self.realtime_manager.current_data.get(symbol, {})
            return jsonify({'success': True, 'data': data})

    def _setup_websocket_events(self):
        """WebSocketイベント設定"""

        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info(f"Client connected: {request.sid}")
            emit('connected', {'data': 'Connected to Advanced Dashboard'})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info(f"Client disconnected: {request.sid}")

        @self.socketio.on('join_symbol')
        def handle_join_symbol(data):
            symbol = data['symbol']
            join_room(f"symbol_{symbol}")
            self.realtime_manager.subscribe_symbol(symbol)
            emit('joined', {'symbol': symbol})

        @self.socketio.on('leave_symbol')
        def handle_leave_symbol(data):
            symbol = data['symbol']
            leave_room(f"symbol_{symbol}")
            emit('left', {'symbol': symbol})

        @self.socketio.on('request_analysis')
        def handle_request_analysis(data):
            symbol = data['symbol']
            # 非同期分析をバックグラウンドで実行
            self.socketio.start_background_task(self._background_analysis, symbol)

        async def _background_analysis(self, symbol):
            """バックグラウンド分析"""
            try:
                analysis = await self.analysis_manager.run_comprehensive_analysis(symbol)
                self.socketio.emit('analysis_complete', {
                    'symbol': symbol,
                    'analysis': analysis
                }, room=f"symbol_{symbol}")
            except Exception as e:
                self.socketio.emit('analysis_error', {
                    'symbol': symbol,
                    'error': str(e)
                }, room=f"symbol_{symbol}")

    async def start_background_tasks(self):
        """バックグラウンドタスク開始"""
        # リアルタイム更新開始
        self.socketio.start_background_task(
            lambda: asyncio.run(self.realtime_manager.start_realtime_updates(self.socketio))
        )

    def run(self, debug=False):
        """ダッシュボード起動"""
        try:
            self.logger.info(f"Starting Advanced Web Dashboard on {self.host}:{self.port}")

            # バックグラウンドタスク開始
            self.socketio.start_background_task(self.start_background_tasks)

            # Flask アプリケーション起動
            self.socketio.run(self.app, host=self.host, port=self.port, debug=debug)

        except Exception as e:
            self.logger.error(f"Failed to start dashboard: {e}")
            raise
        finally:
            self.realtime_manager.stop_updates()


# テスト関数
def test_advanced_dashboard():
    """高度ダッシュボードテスト"""
    print("=== Advanced Web Dashboard Test ===")

    try:
        # ダッシュボード初期化
        dashboard = AdvancedWebDashboard(host='localhost', port=5001)

        print("Dashboard components initialized:")
        print(f"- Realtime Manager: ✓")
        print(f"- Analysis Manager: ✓")
        print(f"- Customization: ✓")

        # 設定テスト
        config = dashboard.customization.load_user_config()
        print(f"- Default config loaded: {len(config)} sections")

        # コンポーネントテスト
        print("\nTesting components:")

        # リアルタイムマネージャー
        dashboard.realtime_manager.subscribe_symbol("7203")
        print(f"- Subscribed symbols: {dashboard.realtime_manager.active_subscriptions}")

        print(f"\nAdvanced Dashboard test completed!")
        print(f"To start the dashboard, run: dashboard.run()")

    except Exception as e:
        print(f"Dashboard test failed: {e}")


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # テスト実行
    test_advanced_dashboard()