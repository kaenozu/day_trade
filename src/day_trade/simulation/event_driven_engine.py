#!/usr/bin/env python3
"""
イベント駆動シミュレーションエンジン
Issue #381: Event-Driven Simulation System

既存システムの技術資産を統合したイベント駆動型シミュレーション
- 高頻度取引エンジンのマイクロ秒精度技術活用
- 並列バックテストのタスク管理統合
- リアルタイム市場イベント処理
- 複合イベント処理(CEP)機能
"""

import asyncio
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..backtesting.parallel_backtest_framework import (
    ParallelBacktestConfig,
)

# 既存システムから技術を流用
from ..trading.high_frequency_engine import (
    HighSpeedOrderQueue,
    MarketDataTick,
    MemoryPool,
    MicrosecondTimer,
)
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class EventType(Enum):
    """イベント種別"""

    # 市場イベント
    MARKET_DATA = "market_data"  # 市場データ更新
    PRICE_CHANGE = "price_change"  # 価格変動
    VOLUME_SPIKE = "volume_spike"  # 出来高急増
    VOLATILITY_CHANGE = "volatility"  # ボラティリティ変化

    # 取引イベント
    ORDER_CREATED = "order_created"  # 注文生成
    ORDER_EXECUTED = "order_executed"  # 注文執行
    ORDER_CANCELLED = "order_cancelled"  # 注文取消
    POSITION_OPENED = "position_opened"  # ポジション開設
    POSITION_CLOSED = "position_closed"  # ポジション決済

    # システムイベント
    STRATEGY_SIGNAL = "strategy_signal"  # 戦略シグナル
    RISK_ALERT = "risk_alert"  # リスクアラート
    ERROR_OCCURRED = "error_occurred"  # エラー発生
    SYSTEM_HEALTH = "system_health"  # システム状態

    # カスタムイベント
    CUSTOM = "custom"  # カスタムイベント


class EventPriority(Enum):
    """イベント優先度"""

    CRITICAL = 0  # 緊急（システムエラー、リスクアラート）
    HIGH = 1  # 高優先度（取引実行、価格変動）
    NORMAL = 2  # 通常（戦略シグナル、分析結果）
    LOW = 3  # 低優先度（ログ、統計更新）


@dataclass
class Event:
    """基本イベントクラス"""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type: EventType = EventType.CUSTOM
    priority: EventPriority = EventPriority.NORMAL
    timestamp_ns: int = field(default_factory=MicrosecondTimer.now_ns)
    source: str = "unknown"
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 処理統計
    processed: bool = False
    processing_time_us: Optional[int] = None
    handlers_called: int = 0


@dataclass
class MarketEvent(Event):
    """市場イベント"""

    symbol: str = ""
    price: float = 0.0
    volume: int = 0
    bid: Optional[float] = None
    ask: Optional[float] = None

    def __post_init__(self):
        self.event_type = EventType.MARKET_DATA
        self.priority = EventPriority.HIGH


@dataclass
class TradingEvent(Event):
    """取引イベント"""

    symbol: str = ""
    action: str = ""  # BUY, SELL, CANCEL
    quantity: float = 0.0
    price: float = 0.0
    order_id: Optional[str] = None

    def __post_init__(self):
        if self.action in ["BUY", "SELL"]:
            self.event_type = EventType.ORDER_EXECUTED
            self.priority = EventPriority.HIGH
        elif self.action == "CANCEL":
            self.event_type = EventType.ORDER_CANCELLED
            self.priority = EventPriority.NORMAL


class EventHandler(ABC):
    """イベントハンドラー抽象基底クラス"""

    def __init__(self, handler_id: str = None):
        self.handler_id = handler_id or f"handler_{uuid.uuid4().hex[:8]}"
        self.processed_events = 0
        self.total_processing_time_us = 0
        self.last_error = None

    @abstractmethod
    async def handle_event(self, event: Event) -> bool:
        """イベント処理（抽象メソッド）"""
        pass

    def can_handle(self, event: Event) -> bool:
        """イベント処理可否判定"""
        return True

    def get_stats(self) -> Dict[str, Any]:
        """ハンドラー統計取得"""
        avg_time = (
            self.total_processing_time_us / self.processed_events
            if self.processed_events > 0
            else 0
        )
        return {
            "handler_id": self.handler_id,
            "processed_events": self.processed_events,
            "avg_processing_time_us": avg_time,
            "total_processing_time_us": self.total_processing_time_us,
            "last_error": str(self.last_error) if self.last_error else None,
        }


class MarketDataHandler(EventHandler):
    """市場データ処理ハンドラー"""

    def __init__(self):
        super().__init__("market_data_handler")
        self.latest_prices = {}
        self.price_history = defaultdict(deque)
        self.max_history_size = 1000

    async def handle_event(self, event: Event) -> bool:
        """市場データイベント処理"""
        if not isinstance(event, MarketEvent):
            return False

        try:
            start_time = MicrosecondTimer.now_ns()

            # 価格データ更新
            self.latest_prices[event.symbol] = event.price

            # 履歴管理
            price_history = self.price_history[event.symbol]
            price_history.append((event.timestamp_ns, event.price))

            # 履歴サイズ制限
            if len(price_history) > self.max_history_size:
                price_history.popleft()

            # 統計更新
            processing_time = MicrosecondTimer.elapsed_us(start_time)
            self.processed_events += 1
            self.total_processing_time_us += processing_time
            event.processing_time_us = processing_time

            return True

        except Exception as e:
            self.last_error = e
            logger.error(f"市場データ処理エラー: {e}")
            return False

    def can_handle(self, event: Event) -> bool:
        """市場データイベントのみ処理"""
        return event.event_type == EventType.MARKET_DATA


class TradingSignalHandler(EventHandler):
    """取引シグナル処理ハンドラー"""

    def __init__(self, strategy_function: Callable = None):
        super().__init__("trading_signal_handler")
        self.strategy_function = strategy_function
        self.signals_generated = 0
        self.signal_history = deque(maxlen=1000)

    async def handle_event(self, event: Event) -> bool:
        """取引シグナル生成処理"""
        if event.event_type != EventType.STRATEGY_SIGNAL:
            return False

        try:
            start_time = MicrosecondTimer.now_ns()

            # 戦略実行（カスタム関数がある場合）
            if self.strategy_function:
                signal_data = await self._execute_strategy(event)
                if signal_data:
                    # シグナル履歴記録
                    self.signal_history.append(
                        {
                            "timestamp": event.timestamp_ns,
                            "signal": signal_data,
                            "source_event": event.event_id,
                        }
                    )
                    self.signals_generated += 1

            # 統計更新
            processing_time = MicrosecondTimer.elapsed_us(start_time)
            self.processed_events += 1
            self.total_processing_time_us += processing_time

            return True

        except Exception as e:
            self.last_error = e
            logger.error(f"取引シグナル処理エラー: {e}")
            return False

    async def _execute_strategy(self, event: Event) -> Optional[Dict[str, Any]]:
        """戦略実行"""
        if not self.strategy_function:
            return None

        try:
            # 非同期で戦略関数実行
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.strategy_function, event)
            return result
        except Exception as e:
            logger.error(f"戦略実行エラー: {e}")
            return None


class ComplexEventProcessor:
    """複合イベント処理エンジン"""

    def __init__(self):
        self.patterns = {}
        self.pattern_states = defaultdict(dict)
        self.detected_patterns = deque(maxlen=1000)

    def add_pattern(
        self,
        pattern_name: str,
        event_sequence: List[EventType],
        time_window_ms: int = 5000,
        callback: Callable = None,
    ):
        """パターン追加"""
        self.patterns[pattern_name] = {
            "sequence": event_sequence,
            "time_window_ms": time_window_ms,
            "callback": callback,
            "active": True,
        }
        logger.info(f"パターン追加: {pattern_name}, シーケンス: {event_sequence}")

    async def process_event(self, event: Event) -> List[str]:
        """イベント処理とパターンマッチング"""
        detected_patterns = []

        for pattern_name, pattern_config in self.patterns.items():
            if not pattern_config["active"]:
                continue

            # パターンマッチング実行
            if await self._match_pattern(event, pattern_name, pattern_config):
                detected_patterns.append(pattern_name)

                # コールバック実行
                if pattern_config["callback"]:
                    try:
                        await pattern_config["callback"](pattern_name, event)
                    except Exception as e:
                        logger.error(f"パターンコールバックエラー: {e}")

        return detected_patterns

    async def _match_pattern(
        self, event: Event, pattern_name: str, pattern_config: Dict[str, Any]
    ) -> bool:
        """パターンマッチング"""
        sequence = pattern_config["sequence"]
        time_window_ms = pattern_config["time_window_ms"]

        # 現在の状態取得
        state = self.pattern_states[pattern_name]
        if "events" not in state:
            state["events"] = []

        # イベント追加
        state["events"].append(event)

        # 時間窓での古いイベント除去
        cutoff_time = event.timestamp_ns - (time_window_ms * 1_000_000)  # ns単位
        state["events"] = [e for e in state["events"] if e.timestamp_ns >= cutoff_time]

        # シーケンスマッチング
        if len(state["events"]) >= len(sequence):
            recent_types = [e.event_type for e in state["events"][-len(sequence) :]]
            if recent_types == sequence:
                # パターン検出！
                self.detected_patterns.append(
                    {
                        "pattern": pattern_name,
                        "timestamp": event.timestamp_ns,
                        "events": state["events"][-len(sequence) :].copy(),
                    }
                )

                # 状態リセット
                state["events"] = []
                return True

        return False


class EventBus:
    """統合イベントバス"""

    def __init__(self, config: ParallelBacktestConfig = None):
        self.config = config or ParallelBacktestConfig()

        # 高頻度取引エンジンの技術を活用
        self.memory_pool = MemoryPool(200)  # 200MB
        self.event_queue = HighSpeedOrderQueue()

        # イベント処理
        self.handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self.global_handlers: List[EventHandler] = []
        self.complex_processor = ComplexEventProcessor()

        # 統計・監視
        self.running = False
        self.processed_events = 0
        self.failed_events = 0
        self.avg_processing_time_us = 0.0
        self.event_stats = defaultdict(int)

        # ワーカースレッド
        self.worker_threads = []
        self.num_workers = min(4, self.config.max_workers)

        logger.info(f"イベントバス初期化完了: {self.num_workers}ワーカー")

    def subscribe(self, handler: EventHandler, event_types: List[EventType] = None):
        """イベントハンドラー登録"""
        if event_types is None:
            # 全イベント対象
            self.global_handlers.append(handler)
            logger.info(f"グローバルハンドラー登録: {handler.handler_id}")
        else:
            # 特定イベントタイプ対象
            for event_type in event_types:
                self.handlers[event_type].append(handler)
            logger.info(f"ハンドラー登録: {handler.handler_id}, タイプ: {event_types}")

    def publish(self, event: Event) -> bool:
        """イベント発行"""
        try:
            # イベントキューに追加（高頻度取引エンジンの技術活用）
            if hasattr(self.event_queue, "enqueue"):
                success = self.event_queue.enqueue(event)
                if success:
                    self.event_stats[event.event_type] += 1
                return success
            else:
                # フォールバック: 直接処理
                asyncio.create_task(self._process_event_async(event))
                return True

        except Exception as e:
            logger.error(f"イベント発行エラー: {e}")
            return False

    def publish_market_data(self, tick: MarketDataTick) -> bool:
        """市場データイベント発行（高頻度取引エンジン連携）"""
        market_event = MarketEvent(
            symbol=tick.symbol,
            price=tick.price,
            volume=tick.volume,
            bid=tick.bid,
            ask=tick.ask,
            timestamp_ns=tick.timestamp_ns,
            source="market_data_feed",
        )
        return self.publish(market_event)

    def publish_trading_event(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        order_id: str = None,
    ) -> bool:
        """取引イベント発行"""
        trading_event = TradingEvent(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            order_id=order_id,
            source="trading_engine",
        )
        return self.publish(trading_event)

    async def start(self):
        """イベントバス開始"""
        if self.running:
            return

        self.running = True

        # ワーカースレッド開始
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop, name=f"EventWorker_{i}", daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)

        logger.info(f"イベントバス開始: {self.num_workers}ワーカー")

    async def stop(self):
        """イベントバス停止"""
        self.running = False

        # ワーカー終了待機
        for worker in self.worker_threads:
            worker.join(timeout=1.0)

        logger.info("イベントバス停止")

    def _worker_loop(self):
        """ワーカーループ（専用スレッド）"""
        while self.running:
            try:
                # イベント取得（高頻度取引エンジンの技術活用）
                if hasattr(self.event_queue, "dequeue"):
                    event = self.event_queue.dequeue(timeout=0.1)
                    if event:
                        # 同期処理でイベント処理
                        asyncio.run(self._process_event_async(event))
                else:
                    # フォールバック: 短時間待機
                    time.sleep(0.001)

            except Exception as e:
                logger.error(f"ワーカーループエラー: {e}")
                time.sleep(0.01)

    async def _process_event_async(self, event: Event):
        """非同期イベント処理"""
        start_time = MicrosecondTimer.now_ns()

        try:
            # 複合イベント処理
            detected_patterns = await self.complex_processor.process_event(event)
            if detected_patterns:
                logger.debug(f"パターン検出: {detected_patterns}")

            # ハンドラー実行
            handlers_to_call = []

            # 特定イベントタイプハンドラー
            if event.event_type in self.handlers:
                handlers_to_call.extend(self.handlers[event.event_type])

            # グローバルハンドラー
            handlers_to_call.extend(self.global_handlers)

            # ハンドラー実行
            for handler in handlers_to_call:
                if handler.can_handle(event):
                    try:
                        success = await handler.handle_event(event)
                        if success:
                            event.handlers_called += 1
                    except Exception as e:
                        logger.error(f"ハンドラー実行エラー: {handler.handler_id}, {e}")
                        self.failed_events += 1

            # 統計更新
            processing_time_us = MicrosecondTimer.elapsed_us(start_time)
            event.processed = True
            event.processing_time_us = processing_time_us

            self.processed_events += 1
            self.avg_processing_time_us = (
                self.avg_processing_time_us * 0.9 + processing_time_us * 0.1
            )

        except Exception as e:
            logger.error(f"イベント処理エラー: {e}")
            self.failed_events += 1

    def get_statistics(self) -> Dict[str, Any]:
        """イベントバス統計取得"""
        handler_stats = []

        # 個別ハンドラー統計
        for event_type, handler_list in self.handlers.items():
            for handler in handler_list:
                stats = handler.get_stats()
                stats["subscribed_event_type"] = event_type.value
                handler_stats.append(stats)

        # グローバルハンドラー統計
        for handler in self.global_handlers:
            stats = handler.get_stats()
            stats["subscribed_event_type"] = "ALL"
            handler_stats.append(stats)

        return {
            "event_bus": {
                "running": self.running,
                "workers": self.num_workers,
                "processed_events": self.processed_events,
                "failed_events": self.failed_events,
                "success_rate": (
                    (self.processed_events - self.failed_events) / max(self.processed_events, 1)
                ),
                "avg_processing_time_us": self.avg_processing_time_us,
            },
            "event_stats": dict(self.event_stats),
            "handler_stats": handler_stats,
            "complex_patterns": {
                "registered_patterns": len(self.complex_processor.patterns),
                "detected_patterns": len(self.complex_processor.detected_patterns),
            },
            "memory_pool": (
                {
                    "size_mb": 200,
                    "allocated_blocks": len(self.memory_pool.allocated_blocks),
                    "free_blocks": len(self.memory_pool.free_blocks),
                }
                if self.memory_pool
                else None
            ),
        }


class EventDrivenSimulationEngine:
    """イベント駆動シミュレーションエンジン（メイン）"""

    def __init__(self, config: ParallelBacktestConfig = None):
        self.config = config or ParallelBacktestConfig()

        # コンポーネント初期化
        self.event_bus = EventBus(config)
        self.simulation_state = {
            "running": False,
            "start_time": None,
            "events_processed": 0,
            "simulations_completed": 0,
        }

        # デフォルトハンドラー登録
        self._register_default_handlers()

        logger.info("イベント駆動シミュレーションエンジン初期化完了")

    def _register_default_handlers(self):
        """デフォルトハンドラー登録"""
        # 市場データハンドラー
        market_handler = MarketDataHandler()
        self.event_bus.subscribe(market_handler, [EventType.MARKET_DATA])

        # 取引シグナルハンドラー
        signal_handler = TradingSignalHandler()
        self.event_bus.subscribe(signal_handler, [EventType.STRATEGY_SIGNAL])

    async def initialize(self):
        """エンジン初期化"""
        await self.event_bus.start()
        logger.info("イベント駆動シミュレーションエンジン初期化完了")

    async def run_simulation(
        self, duration_seconds: int = 60, market_data_source: Callable = None
    ) -> Dict[str, Any]:
        """シミュレーション実行"""
        logger.info(f"イベント駆動シミュレーション開始: {duration_seconds}秒")

        self.simulation_state["running"] = True
        self.simulation_state["start_time"] = MicrosecondTimer.now_ns()

        try:
            # 市場データ生成・配信（シミュレーション）
            if market_data_source:
                await self._run_with_data_source(market_data_source, duration_seconds)
            else:
                await self._run_simulated_market(duration_seconds)

            # 結果集計
            results = self._collect_simulation_results()

            self.simulation_state["simulations_completed"] += 1
            logger.info(f"シミュレーション完了: {results['total_events']}イベント処理")

            return results

        finally:
            self.simulation_state["running"] = False

    async def _run_simulated_market(self, duration_seconds: int):
        """シミュレーション市場データ生成"""
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        base_prices = {symbol: 100.0 + i * 10 for i, symbol in enumerate(symbols)}

        end_time = MicrosecondTimer.now_ns() + (duration_seconds * 1_000_000_000)

        while MicrosecondTimer.now_ns() < end_time and self.simulation_state["running"]:
            # 各銘柄の市場データ生成
            for symbol in symbols:
                # 価格変動シミュレーション
                base_price = base_prices[symbol]
                price_change = np.random.normal(0, 0.01)  # 1%標準偏差
                new_price = base_price * (1 + price_change)
                base_prices[symbol] = new_price

                # 市場データイベント発行
                tick = MarketDataTick(
                    symbol=symbol,
                    price=new_price,
                    volume=np.random.randint(100, 1000),
                    bid=new_price * 0.999,
                    ask=new_price * 1.001,
                )

                self.event_bus.publish_market_data(tick)
                self.simulation_state["events_processed"] += 1

            # 取引シグナルイベント発行（ランダム）
            if np.random.random() < 0.1:  # 10%確率
                signal_event = Event(
                    event_type=EventType.STRATEGY_SIGNAL,
                    priority=EventPriority.NORMAL,
                    source="simulation_engine",
                    data={
                        "signal_type": "momentum",
                        "strength": np.random.uniform(0.1, 0.9),
                    },
                )
                self.event_bus.publish(signal_event)

            # 100μs間隔（10,000 Hz）
            await asyncio.sleep(0.0001)

    async def _run_with_data_source(self, data_source: Callable, duration_seconds: int):
        """外部データソース使用シミュレーション"""
        # 外部データソースからのデータでシミュレーション
        # （実装は簡略化）
        await self._run_simulated_market(duration_seconds)

    def _collect_simulation_results(self) -> Dict[str, Any]:
        """シミュレーション結果収集"""
        runtime_ns = MicrosecondTimer.now_ns() - self.simulation_state["start_time"]
        runtime_seconds = runtime_ns / 1_000_000_000

        bus_stats = self.event_bus.get_statistics()

        return {
            "simulation_summary": {
                "runtime_seconds": runtime_seconds,
                "total_events": self.simulation_state["events_processed"],
                "events_per_second": self.simulation_state["events_processed"] / runtime_seconds,
                "simulations_completed": self.simulation_state["simulations_completed"],
            },
            "event_bus_stats": bus_stats,
            "performance": {
                "avg_event_processing_us": bus_stats["event_bus"]["avg_processing_time_us"],
                "event_success_rate": bus_stats["event_bus"]["success_rate"],
                "throughput_events_per_sec": bus_stats["event_bus"]["processed_events"]
                / runtime_seconds,
            },
        }

    async def cleanup(self):
        """リソースクリーンアップ"""
        await self.event_bus.stop()
        logger.info("イベント駆動シミュレーションエンジン終了")


# エクスポート用ファクトリ関数
async def create_event_driven_simulation_engine(
    config: ParallelBacktestConfig = None,
) -> EventDrivenSimulationEngine:
    """イベント駆動シミュレーションエンジン作成"""
    engine = EventDrivenSimulationEngine(config)
    await engine.initialize()
    return engine
