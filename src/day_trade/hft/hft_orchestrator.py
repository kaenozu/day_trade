#!/usr/bin/env python3
"""
HFT統合オーケストレーター
Issue #366: 高頻度取引最適化エンジン - 統合管理システム

全HFTコンポーネントを統合し、<50μs end-to-end執行を実現する
超高速取引システムの中央制御システム
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional

# プロジェクトモジュール
try:
    from ..cache.advanced_cache_system import AdvancedCacheSystem
    from ..distributed.distributed_computing_manager import DistributedComputingManager
    from ..utils.logging_config import get_context_logger, log_performance_metric
    from .market_data_processor import (
        MarketUpdate,
        UltraFastMarketDataProcessor,
        create_market_data_processor,
    )
    from .microsecond_monitor import MicrosecondMonitor, create_microsecond_monitor
    from .realtime_decision_engine import (
        DecisionResult,
        RealtimeDecisionEngine,
        TradingAction,
        create_decision_engine,
    )
    from .ultra_fast_executor import (
        ExecutionResult,
        OrderEntry,
        UltraFastExecutor,
        create_ultra_fast_executor,
    )
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    def log_performance_metric(*args, **kwargs):
        pass

    # モッククラス
    class UltraFastExecutor:
        def __init__(self, **kwargs):
            pass

        def start_executor_thread(self):
            pass

        def stop_executor_thread(self):
            pass

        async def execute_order_async(self, order):
            return type("MockResult", (), {"success": True, "latency_us": 50.0})()

        def get_performance_stats(self):
            return {}

        async def cleanup(self):
            pass

    class MarketUpdate:
        def __init__(self, symbol_id=0, **kwargs):
            self.symbol_id = symbol_id

    class UltraFastMarketDataProcessor:
        def __init__(self, **kwargs):
            pass

        def add_signal_callback(self, callback):
            pass

        def start_udp_processing(self):
            pass

        def stop_udp_processing(self):
            pass

        async def process_market_update_async(self, update):
            pass

        def get_comprehensive_stats(self):
            return {}

        def get_order_book(self, symbol_id):
            return None

        async def cleanup(self):
            pass

    class RealtimeDecisionEngine:
        def __init__(self, **kwargs):
            pass

        async def make_decision(self, update, **kwargs):
            return type(
                "MockResult",
                (),
                {
                    "decision": type("Action", (), {"HOLD": 0})().HOLD,
                    "suggested_orders": [],
                },
            )()

        def get_comprehensive_stats(self):
            return {}

        async def cleanup(self):
            pass

    class DistributedComputingManager:
        def __init__(self):
            pass

        async def initialize(self, config):
            return {}

        async def cleanup(self):
            pass

    class AdvancedCacheSystem:
        def __init__(self):
            pass

    class MicrosecondMonitor:
        def __init__(self, **kwargs):
            pass

        def start_monitoring(self):
            pass

        def stop_monitoring(self):
            pass

        def time_operation(self, name):
            class MockContext:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

            return MockContext()

        def get_current_stats(self):
            return {}

        async def cleanup(self):
            pass

    def create_ultra_fast_executor(**kwargs):
        return UltraFastExecutor(**kwargs)

    def create_market_data_processor(**kwargs):
        return UltraFastMarketDataProcessor(**kwargs)

    def create_decision_engine(**kwargs):
        return RealtimeDecisionEngine(**kwargs)

    def create_microsecond_monitor(**kwargs):
        return MicrosecondMonitor(**kwargs)


logger = get_context_logger(__name__)


class HFTStatus(IntEnum):
    """HFTシステムステータス"""

    OFFLINE = 0
    INITIALIZING = 1
    READY = 2
    ACTIVE = 3
    PAUSED = 4
    ERROR = 5
    SHUTTING_DOWN = 6


class HFTMode(Enum):
    """HFT動作モード"""

    SIMULATION = "simulation"  # シミュレーション
    PAPER_TRADING = "paper_trading"  # ペーパートレード
    LIVE_TRADING = "live_trading"  # 実取引


@dataclass
class HFTConfig:
    """HFT設定"""

    # Core performance targets
    target_execution_latency_us: int = 50
    max_execution_latency_us: int = 100
    target_decision_latency_us: int = 1000

    # Trading parameters
    max_position_size: int = 10000
    max_order_value: float = 1_000_000.0  # $1M
    risk_limit_daily_loss: float = 100_000.0  # $100K

    # System configuration
    enable_ai_prediction: bool = True
    enable_distributed_processing: bool = True
    enable_ultra_fast_cache: bool = True

    # Market data settings
    enable_udp_market_data: bool = False  # Disabled for safety
    market_data_symbols: List[str] = field(
        default_factory=lambda: ["AAPL", "GOOGL", "MSFT"]
    )

    # Performance monitoring
    enable_nanosecond_monitoring: bool = True
    performance_alert_threshold_us: int = 200

    # Safety settings
    trading_mode: HFTMode = HFTMode.SIMULATION
    enable_kill_switch: bool = True
    max_orders_per_second: int = 1000


@dataclass
class HFTStrategy:
    """HFT戦略定義"""

    strategy_id: str
    strategy_name: str
    target_symbols: List[str]

    # Strategy parameters
    min_signal_confidence: float = 0.7
    position_sizing_method: str = "fixed"
    risk_per_trade: float = 0.01  # 1% risk

    # Execution parameters
    execution_urgency: int = 3  # 1-5 scale
    preferred_execution_venue: str = "primary"

    # Performance targets
    target_sharpe_ratio: float = 2.0
    max_drawdown_pct: float = 0.05  # 5%

    # Status
    is_active: bool = True
    last_update_time: float = field(default_factory=time.time)


class HFTPerformanceTracker:
    """HFTパフォーマンス追跡"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size

        # Latency tracking
        self.execution_latencies = []
        self.decision_latencies = []
        self.end_to_end_latencies = []

        # Throughput tracking
        self.orders_per_second = 0
        self.decisions_per_second = 0

        # P&L tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0

        # Risk metrics
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.var_95 = 0.0

        # System health
        self.error_count = 0
        self.success_rate = 1.0

        self._lock = threading.RLock()

    def record_execution(
        self, latency_us: float, pnl: float = 0.0, success: bool = True
    ):
        """執行記録"""
        with self._lock:
            if len(self.execution_latencies) >= self.window_size:
                self.execution_latencies.pop(0)
            self.execution_latencies.append(latency_us)

            if success:
                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1
                self.realized_pnl += pnl
            else:
                self.error_count += 1

            self._update_success_rate()
            self._update_drawdown()

    def record_decision(self, latency_us: float):
        """決定記録"""
        with self._lock:
            if len(self.decision_latencies) >= self.window_size:
                self.decision_latencies.pop(0)
            self.decision_latencies.append(latency_us)

    def record_end_to_end(self, latency_us: float):
        """End-to-end記録"""
        with self._lock:
            if len(self.end_to_end_latencies) >= self.window_size:
                self.end_to_end_latencies.pop(0)
            self.end_to_end_latencies.append(latency_us)

    def _update_success_rate(self):
        """成功率更新"""
        total = self.total_trades + self.error_count
        if total > 0:
            self.success_rate = self.total_trades / total

    def _update_drawdown(self):
        """ドローダウン更新"""
        # Simplified drawdown calculation
        if self.realized_pnl < 0:
            self.current_drawdown = abs(self.realized_pnl)
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        else:
            self.current_drawdown = 0.0

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンスサマリー"""
        with self._lock:
            import numpy as np

            return {
                # Latency metrics
                "avg_execution_latency_us": (
                    np.mean(self.execution_latencies) if self.execution_latencies else 0
                ),
                "p95_execution_latency_us": (
                    np.percentile(self.execution_latencies, 95)
                    if self.execution_latencies
                    else 0
                ),
                "p99_execution_latency_us": (
                    np.percentile(self.execution_latencies, 99)
                    if self.execution_latencies
                    else 0
                ),
                "avg_decision_latency_us": (
                    np.mean(self.decision_latencies) if self.decision_latencies else 0
                ),
                "avg_end_to_end_latency_us": (
                    np.mean(self.end_to_end_latencies)
                    if self.end_to_end_latencies
                    else 0
                ),
                # Trading metrics
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "win_rate": self.winning_trades / max(self.total_trades, 1),
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": self.unrealized_pnl,
                "total_pnl": self.realized_pnl + self.unrealized_pnl,
                # Risk metrics
                "max_drawdown": self.max_drawdown,
                "current_drawdown": self.current_drawdown,
                "success_rate": self.success_rate,
                "error_count": self.error_count,
            }


class HFTKillSwitch:
    """HFTキルスイッチ（緊急停止）"""

    def __init__(self):
        self.enabled = True
        self.triggered = False
        self.trigger_reasons = []
        self.trigger_time = None

        # Kill switch conditions
        self.max_loss_threshold = 50000.0  # $50K
        self.max_latency_threshold_us = 5000  # 5ms
        self.min_success_rate = 0.8  # 80%

        self._lock = threading.Lock()

    def check_conditions(self, performance_tracker: HFTPerformanceTracker) -> bool:
        """キルスイッチ条件チェック"""
        if not self.enabled or self.triggered:
            return self.triggered

        with self._lock:
            reasons = []

            # Loss limit check
            if performance_tracker.current_drawdown > self.max_loss_threshold:
                reasons.append(
                    f"Drawdown exceeded: ${performance_tracker.current_drawdown:,.2f}"
                )

            # Latency check
            if performance_tracker.execution_latencies:
                recent_latency = performance_tracker.execution_latencies[-1]
                if recent_latency > self.max_latency_threshold_us:
                    reasons.append(f"Latency exceeded: {recent_latency:.1f}μs")

            # Success rate check
            if performance_tracker.success_rate < self.min_success_rate:
                reasons.append(
                    f"Success rate too low: {performance_tracker.success_rate:.1%}"
                )

            if reasons:
                self.triggered = True
                self.trigger_reasons = reasons
                self.trigger_time = time.time()
                logger.critical(f"KILL SWITCH TRIGGERED: {', '.join(reasons)}")
                return True

            return False

    def reset(self):
        """キルスイッチリセット"""
        with self._lock:
            self.triggered = False
            self.trigger_reasons = []
            self.trigger_time = None

    def disable(self):
        """キルスイッチ無効化（注意して使用）"""
        with self._lock:
            self.enabled = False
            logger.warning("Kill switch disabled - USE WITH EXTREME CAUTION")


class HFTOrchestrator:
    """
    HFT統合オーケストレーター

    全HFTコンポーネントを統合管理し、<50μs end-to-end執行を実現
    """

    def __init__(
        self,
        config: HFTConfig,
        distributed_manager: Optional[DistributedComputingManager] = None,
        cache_system: Optional[AdvancedCacheSystem] = None,
    ):
        """
        初期化

        Args:
            config: HFT設定
            distributed_manager: 分散処理マネージャー
            cache_system: キャッシュシステム
        """
        self.config = config
        self.status = HFTStatus.OFFLINE

        # External systems
        self.distributed_manager = distributed_manager or DistributedComputingManager()
        self.cache_system = cache_system or AdvancedCacheSystem()

        # Core HFT components
        self.executor = None
        self.market_data_processor = None
        self.decision_engine = None

        # Management systems
        self.performance_tracker = HFTPerformanceTracker()
        self.kill_switch = HFTKillSwitch() if config.enable_kill_switch else None

        # Microsecond monitoring
        self.microsecond_monitor = create_microsecond_monitor(
            distributed_manager=self.distributed_manager,
            cache_system=self.cache_system,
            monitoring_interval_ms=50,
        )

        # Strategy management
        self.active_strategies: Dict[str, HFTStrategy] = {}
        self.symbol_strategies: Dict[str, List[str]] = {}  # symbol -> strategy_ids

        # Position tracking
        self.positions: Dict[str, int] = {}  # symbol -> position
        self.pending_orders: Dict[str, List[OrderEntry]] = {}  # symbol -> orders

        # Monitoring
        self.monitoring_thread = None
        self.monitoring_running = False

        # Performance statistics
        self.total_orders_processed = 0
        self.total_decisions_made = 0
        self.system_start_time = None

        logger.info("HFTOrchestrator初期化完了")

    async def initialize_system(self) -> bool:
        """
        HFTシステム初期化

        Returns:
            bool: 初期化成功
        """
        logger.info("HFTシステム初期化開始")
        self.status = HFTStatus.INITIALIZING

        try:
            # 1. Distributed processing initialization
            distributed_config = {
                "dask": {
                    "enable_distributed": self.config.enable_distributed_processing,
                    "n_workers": 4,
                    "memory_limit": "2GB",
                },
                "ray": {"num_cpus": 8, "log_to_driver": False},
            }

            await self.distributed_manager.initialize(distributed_config)

            # 2. Ultra-fast executor initialization
            self.executor = create_ultra_fast_executor(
                distributed_manager=self.distributed_manager,
                cache_system=self.cache_system,
                enable_cpu_affinity=True,
                order_queue_size=100000,  # Large queue for HFT
            )

            self.executor.start_executor_thread()

            # 3. Market data processor initialization
            self.market_data_processor = create_market_data_processor(
                distributed_manager=self.distributed_manager,
                cache_system=self.cache_system,
                max_symbols=len(self.config.market_data_symbols),
                enable_udp_receiver=self.config.enable_udp_market_data,
            )

            # Set up market data callback
            self.market_data_processor.add_signal_callback(self._on_market_data_signal)

            # 4. Decision engine initialization
            self.decision_engine = create_decision_engine(
                distributed_manager=self.distributed_manager,
                enable_ai_prediction=self.config.enable_ai_prediction,
            )

            # 5. Start monitoring
            if self.config.enable_nanosecond_monitoring:
                self._start_monitoring_thread()

            # 6. Start microsecond monitoring
            self.microsecond_monitor.start_monitoring()

            self.status = HFTStatus.READY
            self.system_start_time = time.perf_counter_ns()

            logger.info("HFTシステム初期化完了")
            return True

        except Exception as e:
            logger.error(f"HFTシステム初期化失敗: {e}")
            self.status = HFTStatus.ERROR
            return False

    def add_strategy(self, strategy: HFTStrategy):
        """HFT戦略追加"""
        self.active_strategies[strategy.strategy_id] = strategy

        # Update symbol -> strategy mapping
        for symbol in strategy.target_symbols:
            if symbol not in self.symbol_strategies:
                self.symbol_strategies[symbol] = []
            self.symbol_strategies[symbol].append(strategy.strategy_id)

        logger.info(f"HFT戦略追加: {strategy.strategy_name} ({strategy.strategy_id})")

    def remove_strategy(self, strategy_id: str):
        """HFT戦略削除"""
        if strategy_id in self.active_strategies:
            strategy = self.active_strategies[strategy_id]

            # Remove from symbol mapping
            for symbol in strategy.target_symbols:
                if symbol in self.symbol_strategies:
                    self.symbol_strategies[symbol] = [
                        sid
                        for sid in self.symbol_strategies[symbol]
                        if sid != strategy_id
                    ]

            del self.active_strategies[strategy_id]
            logger.info(f"HFT戦略削除: {strategy_id}")

    async def start_trading(self):
        """取引開始"""
        if self.status != HFTStatus.READY:
            logger.error("System not ready for trading")
            return

        logger.info("HFT取引開始")
        self.status = HFTStatus.ACTIVE

        # Start market data processing if UDP enabled
        if self.config.enable_udp_market_data:
            self.market_data_processor.start_udp_processing()

    async def stop_trading(self):
        """取引停止"""
        logger.info("HFT取引停止")
        self.status = HFTStatus.PAUSED

        # Stop market data processing
        if self.market_data_processor:
            self.market_data_processor.stop_udp_processing()

    async def emergency_stop(self, reason: str = "Manual stop"):
        """緊急停止"""
        logger.critical(f"HFT EMERGENCY STOP: {reason}")
        self.status = HFTStatus.ERROR

        # Immediate stop all trading
        await self.stop_trading()

        # Cancel all pending orders (would implement with real exchange connections)
        self.pending_orders.clear()

        # Trigger kill switch
        if self.kill_switch:
            with self.kill_switch._lock:
                self.kill_switch.triggered = True
                self.kill_switch.trigger_reasons.append(reason)
                self.kill_switch.trigger_time = time.time()

    def _on_market_data_signal(self, market_update: MarketUpdate):
        """市場データシグナル処理"""
        if self.status != HFTStatus.ACTIVE:
            return

        # Check kill switch
        if self.kill_switch and self.kill_switch.check_conditions(
            self.performance_tracker
        ):
            asyncio.create_task(self.emergency_stop("Kill switch triggered"))
            return

        # Process signal asynchronously
        asyncio.create_task(self._process_market_signal(market_update))

    async def _process_market_signal(self, market_update: MarketUpdate):
        """
        市場シグナル処理 (メインHFTループ)

        目標: <50μs end-to-end処理
        """
        # マイクロ秒監視でタイミング計測
        with self.microsecond_monitor.time_operation("hft_end_to_end_processing"):
            start_time = time.perf_counter_ns()

            try:
                # Get symbol and check if we have strategies for it
                symbol_id = market_update.symbol_id
                symbol = f"SYMBOL_{symbol_id}"  # Would map from ID to symbol

                if symbol not in self.symbol_strategies:
                    return  # No strategies for this symbol

                # Stage 1: Decision making (<1ms)
                with self.microsecond_monitor.time_operation("hft_decision_making"):
                    decision_start = time.perf_counter_ns()

                    current_position = self.positions.get(symbol, 0)
                    order_book = self.market_data_processor.get_order_book(symbol_id)

                    decision_result = await self.decision_engine.make_decision(
                        market_update, order_book, current_position
                    )

                    decision_time_us = (time.perf_counter_ns() - decision_start) / 1000
                    self.performance_tracker.record_decision(decision_time_us)
                    self.total_decisions_made += 1

                # Stage 2: Order execution (if actionable)
                if decision_result.suggested_orders:
                    with self.microsecond_monitor.time_operation("hft_order_execution"):
                        execution_start = time.perf_counter_ns()

                        for order in decision_result.suggested_orders:
                            # Execute order
                            execution_result = await self.executor.execute_order_async(
                                order
                            )

                            # Record execution
                            execution_time_us = (
                                time.perf_counter_ns() - execution_start
                            ) / 1000
                            success = (
                                execution_result.status.name == "COMPLETED"
                                if hasattr(execution_result, "status")
                                else True
                            )

                            self.performance_tracker.record_execution(
                                execution_time_us, 0.0, success
                            )
                            self.total_orders_processed += 1

                            # Update position
                            if success:
                                side_multiplier = (
                                    1 if order.side == 1 else -1
                                )  # Assuming BUY=1, SELL=-1
                                self.positions[symbol] = current_position + (
                                    order.quantity * side_multiplier
                                )

                # Stage 3: End-to-end latency tracking
                end_time = time.perf_counter_ns()
                total_latency_us = (end_time - start_time) / 1000
                self.performance_tracker.record_end_to_end(total_latency_us)

                # Performance logging
                if total_latency_us > self.config.performance_alert_threshold_us:
                    logger.warning(f"High latency detected: {total_latency_us:.1f}μs")

                log_performance_metric(
                    "hft_end_to_end_latency", total_latency_us, "microseconds"
                )

            except Exception as e:
                logger.error(f"市場シグナル処理エラー: {e}")
                self.performance_tracker.error_count += 1

    def _start_monitoring_thread(self):
        """監視スレッド開始"""
        if self.monitoring_running:
            return

        self.monitoring_running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, name="HFTMonitoring", daemon=True
        )
        self.monitoring_thread.start()
        logger.info("HFT監視スレッド開始")

    def _monitoring_loop(self):
        """監視ループ"""
        while self.monitoring_running:
            try:
                # Health check
                self._perform_health_check()

                # Performance check
                self._check_performance_targets()

                # Kill switch check
                if self.kill_switch:
                    if self.kill_switch.check_conditions(self.performance_tracker):
                        asyncio.create_task(
                            self.emergency_stop("Kill switch conditions met")
                        )

                time.sleep(0.1)  # 100ms monitoring interval

            except Exception as e:
                logger.error(f"監視ループエラー: {e}")

    def _perform_health_check(self):
        """ヘルスチェック"""
        # Check component health
        components_healthy = True

        if self.executor:
            executor_stats = self.executor.get_performance_stats()
            if not executor_stats.get("executor_running", True):
                components_healthy = False

        if not components_healthy and self.status == HFTStatus.ACTIVE:
            logger.warning("Component health degraded")

    def _check_performance_targets(self):
        """パフォーマンス目標チェック"""
        summary = self.performance_tracker.get_performance_summary()

        # Latency target check
        if summary["avg_execution_latency_us"] > self.config.max_execution_latency_us:
            logger.warning(
                f"Execution latency target exceeded: "
                f"{summary['avg_execution_latency_us']:.1f}μs > "
                f"{self.config.max_execution_latency_us}μs"
            )

    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        uptime_ns = 0
        if self.system_start_time:
            uptime_ns = time.perf_counter_ns() - self.system_start_time

        return {
            "status": self.status.name,
            "uptime_seconds": uptime_ns / 1_000_000_000,
            "trading_mode": self.config.trading_mode.value,
            "active_strategies": len(self.active_strategies),
            "total_orders_processed": self.total_orders_processed,
            "total_decisions_made": self.total_decisions_made,
            "current_positions": self.positions.copy(),
            "performance_summary": self.performance_tracker.get_performance_summary(),
            "kill_switch_status": {
                "enabled": self.kill_switch.enabled if self.kill_switch else False,
                "triggered": self.kill_switch.triggered if self.kill_switch else False,
                "trigger_reasons": (
                    self.kill_switch.trigger_reasons if self.kill_switch else []
                ),
            },
        }

    def get_detailed_performance(self) -> Dict[str, Any]:
        """詳細パフォーマンス取得"""
        detailed_stats = {}

        if self.executor:
            detailed_stats["executor"] = self.executor.get_performance_stats()

        if self.market_data_processor:
            detailed_stats["market_data"] = (
                self.market_data_processor.get_comprehensive_stats()
            )

        if self.decision_engine:
            detailed_stats["decision_engine"] = (
                self.decision_engine.get_comprehensive_stats()
            )

        # Microsecond monitoring stats
        detailed_stats["microsecond_monitoring"] = (
            self.microsecond_monitor.get_current_stats()
        )

        detailed_stats["orchestrator"] = self.get_system_status()

        return detailed_stats

    async def cleanup(self):
        """システムクリーンアップ"""
        logger.info("HFTOrchestrator クリーンアップ開始")
        self.status = HFTStatus.SHUTTING_DOWN

        # Stop monitoring
        self.monitoring_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)

        # Stop trading
        await self.stop_trading()

        # Cleanup components
        if self.executor:
            await self.executor.cleanup()

        if self.market_data_processor:
            await self.market_data_processor.cleanup()

        if self.decision_engine:
            await self.decision_engine.cleanup()

        # Cleanup microsecond monitoring
        await self.microsecond_monitor.cleanup()

        # Cleanup distributed processing
        await self.distributed_manager.cleanup()

        self.status = HFTStatus.OFFLINE
        logger.info("クリーンアップ完了")


# Factory function
def create_hft_orchestrator(
    config: HFTConfig,
    distributed_manager: Optional[DistributedComputingManager] = None,
    cache_system: Optional[AdvancedCacheSystem] = None,
) -> HFTOrchestrator:
    """HFTOrchestratorファクトリ関数"""
    return HFTOrchestrator(
        config=config,
        distributed_manager=distributed_manager,
        cache_system=cache_system,
    )


if __name__ == "__main__":
    # テスト実行
    async def main():
        print("=== Issue #366 HFT統合オーケストレーターテスト ===")

        orchestrator = None
        try:
            # HFT設定
            config = HFTConfig(
                target_execution_latency_us=50,
                trading_mode=HFTMode.SIMULATION,
                enable_kill_switch=True,
                market_data_symbols=["AAPL", "GOOGL", "MSFT"],
            )

            # オーケストレーター初期化
            orchestrator = create_hft_orchestrator(config)

            # システム初期化
            print("\n1. システム初期化")
            init_success = await orchestrator.initialize_system()
            print(f"初期化結果: {'成功' if init_success else '失敗'}")

            if init_success:
                # 戦略追加
                print("\n2. 戦略追加")
                test_strategy = HFTStrategy(
                    strategy_id="momentum_1",
                    strategy_name="Momentum Trading v1",
                    target_symbols=["AAPL", "GOOGL"],
                    min_signal_confidence=0.7,
                )
                orchestrator.add_strategy(test_strategy)

                # システム状態確認
                print("\n3. システム状態")
                status = orchestrator.get_system_status()
                print(f"状態: {status['status']}")
                print(f"アクティブ戦略: {status['active_strategies']}")
                print(f"取引モード: {status['trading_mode']}")

                # 取引開始テスト
                print("\n4. 取引開始テスト")
                await orchestrator.start_trading()

                # 短時間実行（シミュレーション）
                print("5秒間の動作シミュレーション...")
                await asyncio.sleep(5)

                # パフォーマンス確認
                print("\n5. パフォーマンス統計")
                performance = orchestrator.get_detailed_performance()

                orchestrator_stats = performance.get("orchestrator", {})
                perf_summary = orchestrator_stats.get("performance_summary", {})

                print(
                    f"処理された注文: {orchestrator_stats.get('total_orders_processed', 0)}"
                )
                print(f"決定回数: {orchestrator_stats.get('total_decisions_made', 0)}")
                print(
                    f"システム稼働時間: {orchestrator_stats.get('uptime_seconds', 0):.1f}秒"
                )

                if perf_summary:
                    print("\nレイテンシー統計:")
                    print(
                        f"  平均執行レイテンシー: {perf_summary.get('avg_execution_latency_us', 0):.1f}μs"
                    )
                    print(
                        f"  平均決定レイテンシー: {perf_summary.get('avg_decision_latency_us', 0):.1f}μs"
                    )
                    print(
                        f"  平均E2Eレイテンシー: {perf_summary.get('avg_end_to_end_latency_us', 0):.1f}μs"
                    )
                    print(f"  成功率: {perf_summary.get('success_rate', 0):.1%}")

                # レイテンシー目標確認
                print("\n6. レイテンシー目標確認")
                target_latency = config.target_execution_latency_us
                actual_latency = perf_summary.get("avg_execution_latency_us", 0)

                if actual_latency > 0 and actual_latency <= target_latency:
                    print(
                        f"✓ レイテンシー目標達成: {actual_latency:.1f}μs ≤ {target_latency}μs"
                    )
                elif actual_latency > target_latency:
                    print(
                        f"⚠ レイテンシー目標未達: {actual_latency:.1f}μs > {target_latency}μs"
                    )
                else:
                    print("レイテンシーデータ不足")

                # 取引停止
                await orchestrator.stop_trading()

        except Exception as e:
            print(f"テスト実行エラー: {e}")

        finally:
            if orchestrator:
                await orchestrator.cleanup()

        print("\n=== HFT統合オーケストレーターテスト完了 ===")

    asyncio.run(main())
