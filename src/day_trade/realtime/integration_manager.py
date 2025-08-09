#!/usr/bin/env python3
"""
Next-Gen AI Trading Engine - リアルタイム統合管理システム
全システムコンポーネントの統合・制御・監視

WebSocket + AI推論 + パフォーマンス監視 + アラート + ダッシュボードの統合運用
"""

import asyncio
import time
import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path

# プロジェクト内インポート
from ..utils.logging_config import get_context_logger
from .websocket_stream import RealTimeStreamManager, MarketTick, NewsItem, SocialPost
from .live_prediction_engine import LivePredictionEngine, LivePrediction, PredictionConfig
from .performance_monitor import RealTimePerformanceMonitor, PerformanceConfig
from .alert_system import AlertManager, TradingAlertGenerator, AlertConfig
from .dashboard import DashboardManager

logger = get_context_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

@dataclass
class IntegrationConfig:
    """統合システム設定"""
    # 基本設定
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"])
    update_interval: float = 1.0  # 1秒間隔

    # コンポーネント有効化
    enable_streaming: bool = True
    enable_prediction: bool = True
    enable_monitoring: bool = True
    enable_alerts: bool = True
    enable_dashboard: bool = True

    # パフォーマンス設定
    max_concurrent_tasks: int = 10
    system_timeout: float = 30.0

    # ダッシュボード設定
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8000

    # ログ設定
    detailed_logging: bool = True
    performance_logging_interval: int = 60  # 60秒間隔

class RealTimeIntegrationManager:
    """リアルタイム統合管理システム"""

    def __init__(self, config: IntegrationConfig):
        self.config = config

        # システムコンポーネント
        self.stream_manager: Optional[RealTimeStreamManager] = None
        self.prediction_engine: Optional[LivePredictionEngine] = None
        self.performance_monitor: Optional[RealTimePerformanceMonitor] = None
        self.alert_manager: Optional[AlertManager] = None
        self.trading_alert_generator: Optional[TradingAlertGenerator] = None
        self.dashboard_manager: Optional[DashboardManager] = None

        # 実行状態
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.main_task: Optional[asyncio.Task] = None
        self.component_tasks: List[asyncio.Task] = []

        # データ統合
        self.latest_market_data: Dict[str, List[MarketTick]] = {}
        self.latest_predictions: Dict[str, LivePrediction] = {}

        # 統計
        self.stats = {
            'uptime_seconds': 0,
            'total_predictions': 0,
            'total_alerts': 0,
            'system_errors': 0,
            'last_performance_log': datetime.now()
        }

        logger.info("Real-Time Integration Manager initialized")

    async def initialize_system(self):
        """システム初期化"""

        try:
            logger.info("Initializing system components...")

            # 1. ストリーミングシステム
            if self.config.enable_streaming:
                logger.info("Initializing streaming system...")
                from .websocket_stream import create_realtime_stream_manager
                self.stream_manager = await create_realtime_stream_manager(self.config.symbols)
                # データコールバック設定
                self.stream_manager.add_data_callback(self._handle_stream_data)

            # 2. ライブ予測エンジン
            if self.config.enable_prediction:
                logger.info("Initializing prediction engine...")
                from .live_prediction_engine import create_live_prediction_engine
                self.prediction_engine = await create_live_prediction_engine(self.config.symbols)

            # 3. パフォーマンス監視
            if self.config.enable_monitoring:
                logger.info("Initializing performance monitor...")
                from .performance_monitor import create_performance_monitor
                self.performance_monitor = create_performance_monitor()

            # 4. アラートシステム
            if self.config.enable_alerts:
                logger.info("Initializing alert system...")
                from .alert_system import create_alert_system
                self.alert_manager, self.trading_alert_generator = create_alert_system()

            # 5. ダッシュボード
            if self.config.enable_dashboard:
                logger.info("Initializing dashboard...")
                from .dashboard import create_dashboard_manager
                self.dashboard_manager = create_dashboard_manager()

                # システムコンポーネント注入
                self.dashboard_manager.inject_components(
                    prediction_engine=self.prediction_engine,
                    performance_monitor=self.performance_monitor,
                    alert_manager=self.alert_manager,
                    stream_manager=self.stream_manager
                )

            # 6. システム統合設定
            if self.prediction_engine and self.performance_monitor:
                # 予測結果をパフォーマンス監視に登録
                self.prediction_engine.add_prediction_callback(self._handle_prediction_result)

            logger.info("System initialization completed successfully")

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise

    async def start_system(self):
        """システム開始"""

        if self.is_running:
            logger.warning("System already running")
            return

        try:
            # 初期化
            await self.initialize_system()

            self.is_running = True
            self.start_time = datetime.now()

            logger.info("Starting integrated real-time trading system...")

            # メインタスク開始
            self.main_task = asyncio.create_task(self._main_system_loop())

            # コンポーネント別タスク開始
            await self._start_component_tasks()

            logger.info("All system components started successfully")

            # システム監視
            await self._monitor_system()

        except Exception as e:
            logger.error(f"System startup failed: {e}")
            await self.stop_system()
            raise

    async def stop_system(self):
        """システム停止"""

        if not self.is_running:
            logger.warning("System not running")
            return

        self.is_running = False

        logger.info("Stopping integrated real-time trading system...")

        # コンポーネントタスク停止
        await self._stop_component_tasks()

        # メインタスク停止
        if self.main_task and not self.main_task.done():
            self.main_task.cancel()
            try:
                await self.main_task
            except asyncio.CancelledError:
                pass

        # 各コンポーネント停止
        await self._stop_components()

        # 統計出力
        self._log_final_statistics()

        logger.info("System stopped successfully")

    async def _start_component_tasks(self):
        """コンポーネントタスク開始"""

        tasks = []

        # ストリーミング
        if self.stream_manager:
            stream_task = asyncio.create_task(
                self.stream_manager.start_all_streams()
            )
            tasks.append(('streaming', stream_task))

        # パフォーマンス監視
        if self.performance_monitor:
            monitoring_task = asyncio.create_task(
                self.performance_monitor.start_monitoring()
            )
            tasks.append(('monitoring', monitoring_task))

        # ダッシュボード（別プロセスで起動）
        if self.dashboard_manager:
            dashboard_task = asyncio.create_task(
                self.dashboard_manager.start_dashboard(
                    host=self.config.dashboard_host,
                    port=self.config.dashboard_port
                )
            )
            tasks.append(('dashboard', dashboard_task))

        # タスク記録
        for name, task in tasks:
            self.component_tasks.append(task)
            logger.info(f"Started component task: {name}")

    async def _stop_component_tasks(self):
        """コンポーネントタスク停止"""

        for task in self.component_tasks:
            if not task.done():
                task.cancel()

        if self.component_tasks:
            await asyncio.gather(*self.component_tasks, return_exceptions=True)

        self.component_tasks.clear()

    async def _stop_components(self):
        """個別コンポーネント停止"""

        try:
            # ストリーミング停止
            if self.stream_manager:
                await self.stream_manager.stop_all_streams()

            # パフォーマンス監視停止
            if self.performance_monitor:
                await self.performance_monitor.stop_monitoring()

            # 予測エンジン停止
            if self.prediction_engine:
                await self.prediction_engine.cleanup()

            # ダッシュボード停止
            if self.dashboard_manager:
                await self.dashboard_manager.stop_dashboard()

        except Exception as e:
            logger.error(f"Component shutdown error: {e}")

    async def _main_system_loop(self):
        """メインシステムループ"""

        logger.info("Main system loop started")

        while self.is_running:
            try:
                loop_start_time = time.time()

                # 1. 市場データ収集・更新
                await self._update_market_data()

                # 2. AI予測実行
                await self._run_predictions()

                # 3. システム統計更新
                self._update_system_stats()

                # 4. パフォーマンスログ
                await self._log_performance_if_needed()

                # 処理時間調整
                loop_time = time.time() - loop_start_time
                sleep_time = max(0, self.config.update_interval - loop_time)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Main system loop error: {e}")
                self.stats['system_errors'] += 1
                await asyncio.sleep(1)

    async def _update_market_data(self):
        """市場データ更新"""

        if not self.stream_manager:
            return

        try:
            # 最新データ取得
            latest_data = self.stream_manager.get_latest_data()
            market_ticks = latest_data.get('market_ticks', [])

            if market_ticks:
                # データをシンボル別に整理
                for tick in market_ticks:
                    symbol = tick.symbol
                    if symbol not in self.latest_market_data:
                        self.latest_market_data[symbol] = []

                    self.latest_market_data[symbol].append(tick)

                    # 履歴サイズ制限（最新100件）
                    if len(self.latest_market_data[symbol]) > 100:
                        self.latest_market_data[symbol] = self.latest_market_data[symbol][-50:]

                # 予測エンジンに市場データ更新
                if self.prediction_engine:
                    await self.prediction_engine.update_market_data(market_ticks)

            # ニュース・ソーシャルデータ処理
            news_items = latest_data.get('news_items', [])
            social_posts = latest_data.get('social_posts', [])

            if self.prediction_engine and (news_items or social_posts):
                self.prediction_engine.update_news_data(news_items)
                self.prediction_engine.update_social_data(social_posts)

        except Exception as e:
            logger.error(f"Market data update error: {e}")

    async def _run_predictions(self):
        """AI予測実行"""

        if not self.prediction_engine or not self.latest_market_data:
            return

        try:
            # ニュース・ソーシャルデータ取得
            latest_data = self.stream_manager.get_latest_data() if self.stream_manager else {}
            news_items = latest_data.get('news_items', [])
            social_posts = latest_data.get('social_posts', [])

            # 予測実行
            predictions = await self.prediction_engine.generate_predictions(
                news_items=news_items,
                social_posts=social_posts
            )

            if predictions:
                self.latest_predictions.update(predictions)
                self.stats['total_predictions'] += len(predictions)

                # 高信頼度予測をログ出力
                for symbol, prediction in predictions.items():
                    if prediction.action_confidence > 0.8:
                        logger.info(
                            f"High-confidence prediction: {symbol} {prediction.final_action} "
                            f"({prediction.action_confidence:.2%} confidence, "
                            f"${prediction.predicted_price:.2f} target)"
                        )

        except Exception as e:
            logger.error(f"Prediction execution error: {e}")

    async def _handle_stream_data(self, stream_data: Dict[str, Any]):
        """ストリームデータ処理コールバック"""

        try:
            market_ticks = stream_data.get('market_ticks', [])
            news_items = stream_data.get('news_items', [])

            # 高頻度ログ抑制
            if len(market_ticks) > 0:
                logger.debug(f"Received {len(market_ticks)} market ticks, {len(news_items)} news items")

        except Exception as e:
            logger.error(f"Stream data handling error: {e}")

    def _handle_prediction_result(self, prediction: LivePrediction):
        """予測結果処理コールバック"""

        try:
            # パフォーマンス監視に記録
            if self.performance_monitor:
                self.performance_monitor.record_prediction(prediction)

            # アラート生成
            if self.trading_alert_generator and prediction.action_confidence > 0.8:
                asyncio.create_task(self._generate_prediction_alert(prediction))

        except Exception as e:
            logger.error(f"Prediction result handling error: {e}")

    async def _generate_prediction_alert(self, prediction: LivePrediction):
        """予測アラート生成"""

        try:
            alert = await self.trading_alert_generator.generate_trading_signal_alert(prediction)

            if alert and self.alert_manager:
                success = await self.alert_manager.send_alert(alert)
                if success:
                    self.stats['total_alerts'] += 1

        except Exception as e:
            logger.error(f"Prediction alert generation error: {e}")

    def _update_system_stats(self):
        """システム統計更新"""

        if self.start_time:
            self.stats['uptime_seconds'] = (datetime.now() - self.start_time).total_seconds()

    async def _log_performance_if_needed(self):
        """必要に応じてパフォーマンスログ出力"""

        if not self.config.detailed_logging:
            return

        now = datetime.now()
        time_since_last_log = (now - self.stats['last_performance_log']).total_seconds()

        if time_since_last_log >= self.config.performance_logging_interval:
            await self._log_system_performance()
            self.stats['last_performance_log'] = now

    async def _log_system_performance(self):
        """システムパフォーマンスログ"""

        try:
            # システム統計
            uptime_minutes = self.stats['uptime_seconds'] / 60
            pred_per_minute = self.stats['total_predictions'] / max(uptime_minutes, 1)

            performance_log = {
                'timestamp': datetime.now().isoformat(),
                'uptime_minutes': uptime_minutes,
                'total_predictions': self.stats['total_predictions'],
                'predictions_per_minute': pred_per_minute,
                'total_alerts': self.stats['total_alerts'],
                'system_errors': self.stats['system_errors'],
                'active_symbols': len(self.latest_market_data),
                'latest_predictions': len(self.latest_predictions)
            }

            # コンポーネント別統計
            if self.performance_monitor:
                comprehensive_status = self.performance_monitor.get_comprehensive_status()
                performance_log['system_status'] = comprehensive_status

            logger.info(f"System Performance: {json.dumps(performance_log, indent=2)}")

        except Exception as e:
            logger.error(f"Performance logging error: {e}")

    async def _monitor_system(self):
        """システム監視"""

        logger.info("System monitoring started")

        try:
            # メインタスクの完了待機
            if self.main_task:
                await self.main_task

        except Exception as e:
            logger.error(f"System monitoring error: {e}")
        finally:
            logger.info("System monitoring ended")

    def _log_final_statistics(self):
        """最終統計出力"""

        final_stats = {
            'total_uptime_minutes': self.stats['uptime_seconds'] / 60,
            'total_predictions': self.stats['total_predictions'],
            'total_alerts': self.stats['total_alerts'],
            'total_errors': self.stats['system_errors'],
            'symbols_processed': len(self.latest_market_data),
            'final_predictions': len(self.latest_predictions)
        }

        logger.info(f"Final System Statistics: {json.dumps(final_stats, indent=2)}")

    def get_system_status(self) -> Dict[str, Any]:
        """システム状況取得"""

        status = {
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': self.stats['uptime_seconds'],
            'statistics': self.stats.copy(),
            'components': {
                'streaming': self.stream_manager is not None,
                'prediction': self.prediction_engine is not None,
                'monitoring': self.performance_monitor is not None,
                'alerts': self.alert_manager is not None,
                'dashboard': self.dashboard_manager is not None
            }
        }

        return status

    def get_latest_predictions(self) -> Dict[str, LivePrediction]:
        """最新予測取得"""
        return self.latest_predictions.copy()

    def get_market_data_summary(self) -> Dict[str, Any]:
        """市場データ要約"""

        summary = {}

        for symbol, ticks in self.latest_market_data.items():
            if ticks:
                latest_tick = ticks[-1]
                summary[symbol] = {
                    'current_price': latest_tick.price,
                    'volume': latest_tick.volume,
                    'timestamp': latest_tick.timestamp.isoformat(),
                    'data_points': len(ticks)
                }

        return summary

# 便利関数
def create_integration_manager(symbols: List[str] = None,
                             dashboard_port: int = 8000) -> RealTimeIntegrationManager:
    """統合管理システム作成"""

    config = IntegrationConfig(
        symbols=symbols or ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
        enable_streaming=True,
        enable_prediction=True,
        enable_monitoring=True,
        enable_alerts=True,
        enable_dashboard=True,
        dashboard_port=dashboard_port,
        detailed_logging=True
    )

    return RealTimeIntegrationManager(config)

async def start_complete_trading_system(symbols: List[str] = None,
                                      dashboard_port: int = 8000):
    """完全なトレーディングシステム起動"""

    logger.info("Starting Next-Gen AI Trading Engine Complete System...")

    # 統合管理システム作成
    integration_manager = create_integration_manager(symbols, dashboard_port)

    try:
        # システム開始
        await integration_manager.start_system()

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        # システム停止
        await integration_manager.stop_system()

if __name__ == "__main__":
    # 統合システムテスト
    async def test_integration_system():
        print("=== Real-Time Integration System Test ===")

        try:
            # 統合管理システム作成
            manager = create_integration_manager(["AAPL", "MSFT"], 8001)

            print("Starting integrated system...")
            print("Dashboard will be available at: http://localhost:8001")

            # システム開始（テスト用に短時間）
            start_task = asyncio.create_task(manager.start_system())

            # 30秒間実行
            await asyncio.sleep(30)

            print("Stopping integrated system...")
            await manager.stop_system()

            # 最終状況
            status = manager.get_system_status()
            print(f"Final system status: {status}")

            print("Integration system test completed")

        except Exception as e:
            print(f"Test error: {e}")
            import traceback
            traceback.print_exc()

    # テスト実行
    asyncio.run(test_integration_system())
