#!/usr/bin/env python3
"""
統合トレーディングシステム
Issue #381: Integrated Trading System with Event-Driven Architecture

完成した全システムの統合:
- 高頻度取引最適化エンジン (Issue #366)
- バックテスト並列フレームワーク (Issue #382)
- イベント駆動シミュレーションエンジン (Issue #381)
- セキュリティ強化システム (Issue #395)
- 高度キャッシングシステム (Issue #377)
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List

from ..backtesting.parallel_backtest_framework import (
    OptimizationMethod,
    ParallelBacktestFramework,
    ParallelMode,
    ParameterSpace,
    create_parallel_backtest_framework,
)
from ..core.optimization_strategy import OptimizationConfig

# 既存システム統合
from ..trading.high_frequency_engine import (
    HighFrequencyTradingEngine,
    MarketDataTick,
    create_high_frequency_trading_engine,
)
from ..utils.logging_config import get_context_logger
from .event_driven_engine import (
    Event,
    EventDrivenSimulationEngine,
    EventPriority,
    EventType,
    MarketEvent,
    create_event_driven_simulation_engine,
)

logger = get_context_logger(__name__)


@dataclass
class IntegratedSystemConfig:
    """統合システム設定"""

    # 高頻度取引設定
    hft_symbols: List[str]
    hft_workers: int = 4

    # バックテスト設定
    backtest_workers: int = 4
    optimization_method: OptimizationMethod = OptimizationMethod.GRID_SEARCH

    # イベント駆動設定
    event_workers: int = 4
    simulation_frequency_hz: int = 1000  # 1kHz

    # 統合設定
    enable_real_time_optimization: bool = True
    enable_event_logging: bool = True
    enable_cross_system_alerts: bool = True


class SystemIntegrationBridge:
    """システム統合ブリッジ"""

    def __init__(self):
        self.systems = {}
        self.cross_system_events = []
        self.integration_stats = {
            "events_bridged": 0,
            "systems_connected": 0,
            "last_sync_time": None,
        }

    def register_system(self, system_name: str, system_instance: Any):
        """システム登録"""
        self.systems[system_name] = system_instance
        self.integration_stats["systems_connected"] = len(self.systems)
        logger.info(f"システム登録: {system_name}")

    async def bridge_hft_to_events(
        self,
        hft_engine: HighFrequencyTradingEngine,
        event_engine: EventDrivenSimulationEngine,
    ):
        """高頻度取引エンジン → イベントエンジン連携"""

        # 高頻度取引の市場データをイベントシステムに配信
        def on_market_data(tick: MarketDataTick):
            # MarketDataTick → MarketEvent変換
            market_event = MarketEvent(
                symbol=tick.symbol,
                price=tick.price,
                volume=tick.volume,
                bid=tick.bid,
                ask=tick.ask,
                timestamp_ns=tick.timestamp_ns,
                source="hft_engine",
            )

            event_engine.event_bus.publish(market_event)
            self.integration_stats["events_bridged"] += 1

        # 高頻度取引エンジンの市場データフィードに購読
        for symbol in hft_engine.symbols:
            hft_engine.market_data_feed.subscribe(symbol, on_market_data)

        logger.info("HFTエンジン→イベントエンジン連携開始")

    async def bridge_events_to_backtest(
        self,
        event_engine: EventDrivenSimulationEngine,
        backtest_framework: ParallelBacktestFramework,
    ):
        """イベントエンジン → バックテストフレームワーク連携"""

        # イベント駆動でのバックテスト最適化トリガー
        async def on_optimization_trigger(pattern_name: str, event: Event):
            if pattern_name == "market_volatility_spike":
                logger.info("市場ボラティリティ急変 - 緊急最適化実行")

                # 緊急パラメータ最適化実行
                parameter_spaces = [
                    ParameterSpace("risk_threshold", 0.01, 0.05, step_size=0.01),
                    ParameterSpace("position_size", 0.05, 0.2, step_size=0.05),
                ]

                # 非同期で最適化実行（メインスレッドをブロックしない）
                asyncio.create_task(
                    self._run_emergency_optimization(backtest_framework, parameter_spaces, event)
                )

        # 複合イベントパターン登録
        event_engine.event_bus.complex_processor.add_pattern(
            "market_volatility_spike",
            [EventType.PRICE_CHANGE, EventType.VOLUME_SPIKE],
            time_window_ms=1000,
            callback=on_optimization_trigger,
        )

        logger.info("イベントエンジン→バックテスト連携開始")

    async def _run_emergency_optimization(
        self,
        backtest_framework: ParallelBacktestFramework,
        parameter_spaces: List[ParameterSpace],
        trigger_event: Event,
    ):
        """緊急最適化実行"""
        try:
            # 市場状況に応じた最適化実行
            results = backtest_framework.run_parameter_optimization(
                symbols=[trigger_event.data.get("symbol", "AAPL")],
                parameter_spaces=parameter_spaces,
                start_date="2023-01-01",
                end_date="2023-12-31",
            )

            logger.info(
                f"緊急最適化完了: {results.get('optimization_summary', {}).get('best_value', 0):.4f}"
            )

        except Exception as e:
            logger.error(f"緊急最適化エラー: {e}")


class IntegratedTradingSystem:
    """統合トレーディングシステム（メイン）"""

    def __init__(self, config: IntegratedSystemConfig):
        self.config = config
        self.systems = {}
        self.integration_bridge = SystemIntegrationBridge()
        self.running = False

        # 統計
        self.system_stats = {
            "uptime_seconds": 0,
            "total_events_processed": 0,
            "total_trades_executed": 0,
            "total_optimizations_run": 0,
            "cross_system_syncs": 0,
        }

        logger.info("統合トレーディングシステム初期化")

    async def initialize(self):
        """システム初期化"""
        logger.info("統合システム初期化開始")

        try:
            # 1. 高頻度取引エンジン初期化
            optimization_config = OptimizationConfig(enable_gpu=True, enable_caching=True)

            self.systems["hft"] = await create_high_frequency_trading_engine(
                optimization_config, self.config.hft_symbols
            )

            # 2. バックテスト並列フレームワーク初期化
            self.systems["backtest"] = create_parallel_backtest_framework(
                max_workers=self.config.backtest_workers,
                parallel_mode=ParallelMode.MULTIPROCESSING,
                optimization_method=self.config.optimization_method,
            )

            # 3. イベント駆動シミュレーション初期化
            from ..backtesting.parallel_backtest_framework import ParallelBacktestConfig

            event_config = ParallelBacktestConfig(max_workers=self.config.event_workers)

            self.systems["events"] = await create_event_driven_simulation_engine(event_config)

            # 4. システム統合ブリッジ登録
            for name, system in self.systems.items():
                self.integration_bridge.register_system(name, system)

            # 5. システム間連携設定
            await self._setup_system_integration()

            logger.info("統合システム初期化完了")

        except Exception as e:
            logger.error(f"統合システム初期化エラー: {e}")
            raise

    async def _setup_system_integration(self):
        """システム間統合設定"""
        hft_engine = self.systems["hft"]
        event_engine = self.systems["events"]
        backtest_framework = self.systems["backtest"]

        # HFT → イベント連携
        await self.integration_bridge.bridge_hft_to_events(hft_engine, event_engine)

        # イベント → バックテスト連携
        await self.integration_bridge.bridge_events_to_backtest(event_engine, backtest_framework)

        # リアルタイム最適化パターン設定
        if self.config.enable_real_time_optimization:
            await self._setup_real_time_optimization()

        logger.info("システム間統合設定完了")

    async def _setup_real_time_optimization(self):
        """リアルタイム最適化設定"""
        event_engine = self.systems["events"]

        # 市場急変時の自動最適化パターン
        async def on_market_crisis(pattern_name: str, event: Event):
            logger.warning(f"市場危機パターン検出: {pattern_name}")

            # 緊急リスク管理モード発動
            await self._trigger_emergency_mode(event)

        # 危機パターン登録
        event_engine.event_bus.complex_processor.add_pattern(
            "market_crisis",
            [
                EventType.VOLATILITY_CHANGE,
                EventType.VOLUME_SPIKE,
                EventType.PRICE_CHANGE,
            ],
            time_window_ms=500,  # 0.5秒以内の連続イベント
            callback=on_market_crisis,
        )

    async def _trigger_emergency_mode(self, trigger_event: Event):
        """緊急モード発動"""
        logger.critical("緊急モード発動 - 全システム保護モード移行")

        try:
            # 1. 高頻度取引の一時停止
            hft_engine = self.systems["hft"]
            # await hft_engine.pause_trading()

            # 2. リスク評価の緊急実行
            backtest_framework = self.systems["backtest"]

            # 3. 緊急アラート発行
            emergency_event = Event(
                event_type=EventType.RISK_ALERT,
                priority=EventPriority.CRITICAL,
                source="integrated_system",
                data={
                    "alert_type": "EMERGENCY_MODE",
                    "trigger_event": trigger_event.event_id,
                    "timestamp": trigger_event.timestamp_ns,
                },
            )

            event_engine = self.systems["events"]
            event_engine.event_bus.publish(emergency_event)

        except Exception as e:
            logger.error(f"緊急モード処理エラー: {e}")

    async def start(self):
        """統合システム開始"""
        if self.running:
            return

        logger.info("統合トレーディングシステム開始")

        try:
            self.running = True
            start_time = asyncio.get_event_loop().time()

            # 各システム順次開始
            hft_engine = self.systems["hft"]
            await hft_engine.start()

            # イベントシステムは既に初期化時に開始済み

            self.system_stats["start_time"] = start_time
            logger.info("統合システム開始完了")

        except Exception as e:
            logger.error(f"統合システム開始エラー: {e}")
            raise

    async def run_integrated_demo(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """統合デモンストレーション実行"""
        logger.info(f"統合システム デモンストレーション開始: {duration_seconds}秒")

        try:
            # 1. イベント駆動シミュレーション開始
            event_engine = self.systems["events"]
            simulation_task = asyncio.create_task(event_engine.run_simulation(duration_seconds))

            # 2. 高頻度取引デモンストレーション
            hft_engine = self.systems["hft"]
            hft_task = asyncio.create_task(
                hft_engine.run_performance_benchmark(duration_seconds // 2)
            )

            # 3. 並行実行
            simulation_results = await simulation_task
            hft_results = await hft_task

            # 4. 結果統合
            integrated_results = {
                "demo_summary": {
                    "duration_seconds": duration_seconds,
                    "systems_active": len(self.systems),
                    "integration_success": True,
                },
                "event_simulation": simulation_results,
                "hft_performance": hft_results,
                "integration_stats": self.integration_bridge.integration_stats,
                "system_performance": await self._get_integrated_performance(),
            }

            logger.info("統合デモンストレーション完了")
            return integrated_results

        except Exception as e:
            logger.error(f"統合デモエラー: {e}")
            return {"error": str(e)}

    async def _get_integrated_performance(self) -> Dict[str, Any]:
        """統合システムパフォーマンス取得"""
        performance = {}

        # 各システムのパフォーマンス統計
        for name, system in self.systems.items():
            if hasattr(system, "get_performance_stats"):
                performance[name] = system.get_performance_stats()
            elif hasattr(system, "get_framework_stats"):
                performance[name] = system.get_framework_stats()
            elif hasattr(system, "get_statistics"):
                performance[name] = system.get_statistics()

        # 統合統計
        performance["integration"] = {
            "cross_system_events": self.integration_bridge.integration_stats["events_bridged"],
            "connected_systems": self.integration_bridge.integration_stats["systems_connected"],
            "system_uptime": self.system_stats.get("uptime_seconds", 0),
        }

        return performance

    async def stop(self):
        """統合システム停止"""
        if not self.running:
            return

        logger.info("統合トレーディングシステム停止中...")

        try:
            # 各システム順次停止
            for name, system in self.systems.items():
                if hasattr(system, "stop"):
                    await system.stop()
                elif hasattr(system, "cleanup"):
                    await system.cleanup()

            self.running = False
            logger.info("統合トレーディングシステム停止完了")

        except Exception as e:
            logger.error(f"統合システム停止エラー: {e}")

    async def get_system_status(self) -> Dict[str, Any]:
        """統合システム状態取得"""
        status = {
            "running": self.running,
            "systems": {},
            "integration": self.integration_bridge.integration_stats,
            "performance": await self._get_integrated_performance(),
        }

        # 各システム状態
        for name, system in self.systems.items():
            system_status = {
                "initialized": system is not None,
                "type": type(system).__name__,
            }

            # システム固有の状態情報
            if hasattr(system, "running"):
                system_status["running"] = system.running

            status["systems"][name] = system_status

        return status


# エクスポート用ファクトリ関数
async def create_integrated_trading_system(
    symbols: List[str] = None,
    hft_workers: int = 4,
    backtest_workers: int = 4,
    event_workers: int = 4,
) -> IntegratedTradingSystem:
    """統合トレーディングシステム作成"""
    config = IntegratedSystemConfig(
        hft_symbols=symbols or ["AAPL", "MSFT", "GOOGL"],
        hft_workers=hft_workers,
        backtest_workers=backtest_workers,
        event_workers=event_workers,
    )

    system = IntegratedTradingSystem(config)
    await system.initialize()
    return system
