#!/usr/bin/env python3
"""
イベント駆動型バックテストエンジン
Issue #381: イベント駆動型シミュレーションの導入

価格変動やシグナル発生などのイベントをトリガーとする高速バックテストシステム
従来の日次ループ処理を置き換え、不要な計算をスキップして大幅な高速化を実現
"""

import heapq
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

try:
    from ..trading.high_frequency_engine import MicrosecondTimer
    from ..utils.performance_monitor import get_performance_monitor
    from ..utils.structured_logging import get_structured_logger

    logger = get_structured_logger()
    perf_monitor = get_performance_monitor()
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

    class SimpleMonitor:
        def monitor(self, name):
            class Context:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

            return Context()

    perf_monitor = SimpleMonitor()

    class MicrosecondTimer:
        @staticmethod
        def now_ns():
            return time.time_ns()

        @staticmethod
        def elapsed_us(start):
            return (time.time_ns() - start) / 1000


class EventType(Enum):
    """イベント種別"""

    MARKET_DATA = "market_data"  # 市場データ更新
    PRICE_UPDATE = "price_update"  # 価格更新
    SIGNAL_GENERATED = "signal_generated"  # シグナル発生
    ORDER_PLACED = "order_placed"  # 注文発注
    ORDER_FILLED = "order_filled"  # 注文約定
    PORTFOLIO_UPDATE = "portfolio_update"  # ポートフォリオ更新
    REBALANCE = "rebalance"  # リバランス
    STOP_LOSS = "stop_loss"  # ストップロス
    TAKE_PROFIT = "take_profit"  # 利確


@dataclass
class Event:
    """イベントデータ"""

    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    priority: int = 0  # 低い値が高優先度
    event_id: str = field(default="")
    source: str = field(default="")

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"{self.event_type.value}_{id(self)}"

    def __lt__(self, other):
        """heapq用の比較演算子"""
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.priority < other.priority


class EventHandler(ABC):
    """イベントハンドラ基底クラス"""

    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """イベント処理可否判定"""
        pass

    @abstractmethod
    def handle(self, event: Event, context: "EventDrivenContext") -> List[Event]:
        """イベント処理（新しいイベントを返す可能性）"""
        pass


@dataclass
class MarketDataSnapshot:
    """市場データスナップショット"""

    timestamp: datetime
    prices: Dict[str, float]
    volumes: Dict[str, float] = field(default_factory=dict)
    bid_ask: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class Position:
    """ポジション情報"""

    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price


@dataclass
class Portfolio:
    """ポートフォリオ状態"""

    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    total_value: float = 0.0
    daily_return: float = 0.0
    cumulative_return: float = 0.0


class EventDrivenContext:
    """イベント駆動実行コンテキスト"""

    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.portfolio = Portfolio(cash=initial_capital)
        self.current_market_data: Optional[MarketDataSnapshot] = None
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.strategy_function: Optional[Callable] = None

        # パフォーマンス追跡
        self.portfolio_history: List[Portfolio] = []
        self.trades: List[Dict[str, Any]] = []
        self.events_processed = 0
        self.last_rebalance_time: Optional[datetime] = None
        self.rebalance_frequency_days = 5

        # 取引コスト
        self.commission_rate = 0.001
        self.slippage_rate = 0.0005

    def update_market_data(self, snapshot: MarketDataSnapshot):
        """市場データ更新"""
        self.current_market_data = snapshot

        # ポジション評価更新
        for symbol, position in self.portfolio.positions.items():
            if symbol in snapshot.prices:
                position.current_price = snapshot.prices[symbol]
                position.unrealized_pnl = (
                    position.current_price - position.avg_price
                ) * position.quantity

        # ポートフォリオ価値計算
        positions_value = sum(
            pos.market_value for pos in self.portfolio.positions.values()
        )
        self.portfolio.total_value = self.portfolio.cash + positions_value

    def execute_trade(
        self, symbol: str, quantity: int, price: float, timestamp: datetime
    ):
        """取引実行"""
        if quantity == 0:
            return

        # 取引コスト計算
        gross_amount = abs(quantity) * price
        commission = gross_amount * self.commission_rate
        slippage = gross_amount * self.slippage_rate
        total_cost = commission + slippage

        if quantity > 0:  # 買い注文
            total_amount = gross_amount + total_cost
            if total_amount > self.portfolio.cash:
                # 買える分だけ購入
                affordable_quantity = int(
                    self.portfolio.cash
                    / (price * (1 + self.commission_rate + self.slippage_rate))
                )
                if affordable_quantity <= 0:
                    return
                quantity = affordable_quantity
                total_amount = (
                    quantity * price * (1 + self.commission_rate + self.slippage_rate)
                )

            self.portfolio.cash -= total_amount

            # ポジション更新
            if symbol in self.portfolio.positions:
                position = self.portfolio.positions[symbol]
                total_quantity = position.quantity + quantity
                total_cost_basis = (
                    position.quantity * position.avg_price + quantity * price
                )
                new_avg_price = (
                    total_cost_basis / total_quantity if total_quantity > 0 else price
                )

                position.quantity = total_quantity
                position.avg_price = new_avg_price
                position.current_price = price
            else:
                self.portfolio.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=price,
                    current_price=price,
                )

        else:  # 売り注文
            if symbol not in self.portfolio.positions:
                return

            position = self.portfolio.positions[symbol]
            sell_quantity = min(abs(quantity), position.quantity)

            if sell_quantity <= 0:
                return

            # 売却収益
            gross_proceeds = sell_quantity * price
            net_proceeds = gross_proceeds - total_cost
            self.portfolio.cash += net_proceeds

            # 実現損益
            realized_pnl = (price - position.avg_price) * sell_quantity
            position.realized_pnl += realized_pnl

            # ポジション更新
            position.quantity -= sell_quantity
            if position.quantity <= 0:
                del self.portfolio.positions[symbol]

        # 取引記録
        self.trades.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "commission": commission,
                "total_cost": total_cost,
            }
        )


class MarketDataHandler(EventHandler):
    """市場データイベントハンドラ"""

    def can_handle(self, event: Event) -> bool:
        return event.event_type == EventType.MARKET_DATA

    def handle(self, event: Event, context: EventDrivenContext) -> List[Event]:
        """市場データ処理"""
        snapshot = MarketDataSnapshot(
            timestamp=event.timestamp,
            prices=event.data.get("prices", {}),
            volumes=event.data.get("volumes", {}),
        )

        context.update_market_data(snapshot)

        # 次のイベント生成
        new_events = []

        # 価格更新イベント生成
        for symbol, price in snapshot.prices.items():
            new_events.append(
                Event(
                    event_type=EventType.PRICE_UPDATE,
                    timestamp=event.timestamp,
                    data={"symbol": symbol, "price": price},
                    priority=1,
                )
            )

        # リバランスチェック
        if self._should_rebalance(event.timestamp, context):
            new_events.append(
                Event(
                    event_type=EventType.REBALANCE,
                    timestamp=event.timestamp,
                    data={},
                    priority=2,
                )
            )

        return new_events

    def _should_rebalance(
        self, current_time: datetime, context: EventDrivenContext
    ) -> bool:
        """リバランス必要性判定"""
        if context.last_rebalance_time is None:
            return True

        days_since_rebalance = (current_time - context.last_rebalance_time).days
        return days_since_rebalance >= context.rebalance_frequency_days


class RebalanceHandler(EventHandler):
    """リバランスイベントハンドラ"""

    def can_handle(self, event: Event) -> bool:
        return event.event_type == EventType.REBALANCE

    def handle(self, event: Event, context: EventDrivenContext) -> List[Event]:
        """リバランス処理"""
        if not context.strategy_function or not context.current_market_data:
            return []

        # 戦略シグナル生成
        lookback_data = self._get_lookback_data(event.timestamp, context)

        if not lookback_data:
            return []

        try:
            signals = context.strategy_function(
                lookback_data, context.current_market_data.prices
            )

            if not signals:
                return []

            # シグナル処理イベント生成
            new_events = []
            for symbol, weight in signals.items():
                new_events.append(
                    Event(
                        event_type=EventType.SIGNAL_GENERATED,
                        timestamp=event.timestamp,
                        data={"symbol": symbol, "weight": weight},
                        priority=3,
                    )
                )

            context.last_rebalance_time = event.timestamp
            return new_events

        except Exception as e:
            logger.error(f"戦略実行エラー: {e}")
            return []

    def _get_lookback_data(
        self, current_time: datetime, context: EventDrivenContext, window: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """ルックバックデータ取得"""
        lookback_data = {}

        for symbol, data in context.historical_data.items():
            mask = data.index < current_time
            recent_data = data[mask].tail(window)

            if len(recent_data) >= window // 2:
                lookback_data[symbol] = recent_data

        return lookback_data


class SignalHandler(EventHandler):
    """シグナルイベントハンドラ"""

    def can_handle(self, event: Event) -> bool:
        return event.event_type == EventType.SIGNAL_GENERATED

    def handle(self, event: Event, context: EventDrivenContext) -> List[Event]:
        """シグナル処理"""
        symbol = event.data["symbol"]
        target_weight = event.data["weight"]

        if (
            not context.current_market_data
            or symbol not in context.current_market_data.prices
        ):
            return []

        current_price = context.current_market_data.prices[symbol]
        target_value = context.portfolio.total_value * target_weight
        target_quantity = int(target_value / current_price) if current_price > 0 else 0

        # 現在のポジション
        current_quantity = context.portfolio.positions.get(
            symbol, Position(symbol, 0, 0, 0)
        ).quantity

        # 取引必要数量
        trade_quantity = target_quantity - current_quantity

        if abs(trade_quantity) > 0:
            return [
                Event(
                    event_type=EventType.ORDER_PLACED,
                    timestamp=event.timestamp,
                    data={
                        "symbol": symbol,
                        "quantity": trade_quantity,
                        "price": current_price,
                    },
                    priority=4,
                )
            ]

        return []


class OrderHandler(EventHandler):
    """注文イベントハンドラ"""

    def can_handle(self, event: Event) -> bool:
        return event.event_type == EventType.ORDER_PLACED

    def handle(self, event: Event, context: EventDrivenContext) -> List[Event]:
        """注文処理"""
        symbol = event.data["symbol"]
        quantity = event.data["quantity"]
        price = event.data["price"]

        # 即座に約定と仮定（バックテストのため）
        context.execute_trade(symbol, quantity, price, event.timestamp)

        return [
            Event(
                event_type=EventType.ORDER_FILLED,
                timestamp=event.timestamp,
                data=event.data,
                priority=5,
            )
        ]


class EventQueue:
    """イベントキュー（優先度付きキュー）"""

    def __init__(self):
        self.heap: List[Event] = []
        self.event_count = 0

    def push(self, event: Event):
        """イベント追加"""
        heapq.heappush(self.heap, event)
        self.event_count += 1

    def pop(self) -> Optional[Event]:
        """次のイベント取得"""
        if self.heap:
            return heapq.heappop(self.heap)
        return None

    def push_multiple(self, events: List[Event]):
        """複数イベント追加"""
        for event in events:
            self.push(event)

    def is_empty(self) -> bool:
        """キュー空判定"""
        return len(self.heap) == 0

    def size(self) -> int:
        """キューサイズ"""
        return len(self.heap)


class EventDrivenBacktestEngine:
    """イベント駆動バックテストエンジン"""

    def __init__(self, initial_capital: float = 1000000):
        self.context = EventDrivenContext(initial_capital)
        self.event_queue = EventQueue()
        self.handlers: List[EventHandler] = [
            MarketDataHandler(),
            RebalanceHandler(),
            SignalHandler(),
            OrderHandler(),
        ]

        # パフォーマンスメトリクス
        self.start_time_ns: Optional[int] = None
        self.events_processed = 0
        self.total_execution_time_ms = 0.0

    def load_historical_data(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """過去データ読み込み"""
        logger.info(
            f"過去データ読み込み: {len(symbols)}銘柄, {start_date} - {end_date}"
        )

        historical_data = {}

        for symbol in symbols:
            try:
                import yfinance as yf

                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)

                if not data.empty and len(data) >= 30:
                    data["Returns"] = data["Close"].pct_change()
                    data["Volume_Avg"] = data["Volume"].rolling(20).mean()
                    historical_data[symbol] = data

            except Exception as e:
                logger.warning(f"データ読み込みエラー {symbol}: {e}")

        self.context.historical_data = historical_data
        logger.info(f"データ読み込み完了: {len(historical_data)}銘柄")
        return historical_data

    def execute_event_driven_backtest(
        self,
        historical_data: Dict[str, pd.DataFrame],
        strategy_function: Callable,
        rebalance_frequency: int = 5,
    ) -> Dict[str, Any]:
        """イベント駆動バックテスト実行"""
        logger.info("🚀 イベント駆動バックテスト開始")
        self.start_time_ns = MicrosecondTimer.now_ns()

        self.context.strategy_function = strategy_function
        self.context.rebalance_frequency_days = rebalance_frequency

        # 初期イベント生成（市場データイベント）
        self._generate_market_data_events(historical_data)

        logger.info(f"初期イベント数: {self.event_queue.size()}")

        # イベントループ実行
        processed_events = 0
        last_progress_time = time.time()

        while not self.event_queue.is_empty():
            event = self.event_queue.pop()
            if event is None:
                break

            # イベント処理
            new_events = self._process_event(event)

            # 新しいイベントをキューに追加
            if new_events:
                self.event_queue.push_multiple(new_events)

            processed_events += 1
            self.context.events_processed = processed_events

            # 進捗表示
            if time.time() - last_progress_time > 2.0:  # 2秒ごと
                logger.info(
                    f"処理済みイベント: {processed_events}, 残り: {self.event_queue.size()}"
                )
                last_progress_time = time.time()

        # 実行時間計算
        if self.start_time_ns:
            self.total_execution_time_ms = (
                MicrosecondTimer.elapsed_us(self.start_time_ns) / 1000
            )

        # 結果生成
        results = self._generate_results()

        logger.info(
            f"✅ イベント駆動バックテスト完了: {processed_events}イベント処理, "
            f"{self.total_execution_time_ms:.0f}ms"
        )

        return results

    def _generate_market_data_events(self, historical_data: Dict[str, pd.DataFrame]):
        """市場データイベント生成"""
        if not historical_data:
            return

        # 共通日付取得
        common_dates = None
        for data in historical_data.values():
            if common_dates is None:
                common_dates = data.index
            else:
                common_dates = common_dates.intersection(data.index)

        if common_dates is None or len(common_dates) == 0:
            return

        common_dates = common_dates.sort_values()

        # 各日付で市場データイベント生成
        for date in common_dates:
            prices = {}
            volumes = {}

            for symbol, data in historical_data.items():
                if date in data.index:
                    prices[symbol] = data.loc[date, "Close"]
                    volumes[symbol] = data.loc[date, "Volume"]

            if prices:  # データがある場合のみイベント生成
                event = Event(
                    event_type=EventType.MARKET_DATA,
                    timestamp=date.to_pydatetime()
                    if hasattr(date, "to_pydatetime")
                    else date,
                    data={"prices": prices, "volumes": volumes},
                    priority=0,
                )
                self.event_queue.push(event)

    def _process_event(self, event: Event) -> List[Event]:
        """イベント処理"""
        new_events = []

        for handler in self.handlers:
            if handler.can_handle(event):
                try:
                    handler_events = handler.handle(event, self.context)
                    if handler_events:
                        new_events.extend(handler_events)
                except Exception as e:
                    logger.error(f"イベント処理エラー {event.event_type}: {e}")

        return new_events

    def _generate_results(self) -> Dict[str, Any]:
        """結果生成"""
        final_value = self.context.portfolio.total_value
        total_return = (
            final_value - self.context.initial_capital
        ) / self.context.initial_capital

        return {
            "execution_summary": {
                "engine_type": "event_driven",
                "initial_capital": self.context.initial_capital,
                "final_value": final_value,
                "total_return": total_return,
                "events_processed": self.context.events_processed,
                "total_execution_time_ms": self.total_execution_time_ms,
                "trades_executed": len(self.context.trades),
            },
            "portfolio_final": {
                "cash": self.context.portfolio.cash,
                "positions": {
                    symbol: {
                        "quantity": pos.quantity,
                        "avg_price": pos.avg_price,
                        "current_price": pos.current_price,
                        "market_value": pos.market_value,
                        "unrealized_pnl": pos.unrealized_pnl,
                    }
                    for symbol, pos in self.context.portfolio.positions.items()
                },
                "total_value": final_value,
            },
            "performance_metrics": {
                "events_per_second": (
                    self.context.events_processed
                    / (self.total_execution_time_ms / 1000)
                    if self.total_execution_time_ms > 0
                    else 0
                ),
                "avg_event_processing_time_us": (
                    (self.total_execution_time_ms * 1000)
                    / self.context.events_processed
                    if self.context.events_processed > 0
                    else 0
                ),
            },
            "trades": self.context.trades,
        }


# エクスポート用ファクトリ関数
def create_event_driven_engine(
    initial_capital: float = 1000000,
) -> EventDrivenBacktestEngine:
    """イベント駆動エンジン作成"""
    return EventDrivenBacktestEngine(initial_capital)


if __name__ == "__main__":
    # テスト実行
    print("=== イベント駆動バックテストエンジン テスト ===")

    def simple_strategy(
        lookback_data: Dict[str, pd.DataFrame], current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """簡単な戦略"""
        signals = {}

        for symbol, data in lookback_data.items():
            if len(data) >= 20:
                returns_20d = data["Close"].iloc[-1] / data["Close"].iloc[-20] - 1

                if returns_20d > 0.05:
                    signals[symbol] = 0.3
                elif returns_20d < -0.05:
                    signals[symbol] = 0.0
                else:
                    signals[symbol] = 0.1

        # 正規化
        total_weight = sum(signals.values())
        if total_weight > 0:
            signals = {k: v / total_weight for k, v in signals.items()}

        return signals

    # エンジン初期化
    engine = create_event_driven_engine(1000000)

    # テストデータ読み込み
    test_symbols = ["7203.T", "8306.T", "9984.T"]

    from datetime import datetime, timedelta

    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    historical_data = engine.load_historical_data(
        test_symbols, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    )

    if historical_data:
        # バックテスト実行
        results = engine.execute_event_driven_backtest(
            historical_data, simple_strategy, rebalance_frequency=5
        )

        # 結果表示
        summary = results["execution_summary"]
        print("\n【実行結果】")
        print(f"初期資本: {summary['initial_capital']:,.0f}円")
        print(f"最終価値: {summary['final_value']:,.0f}円")
        print(f"総リターン: {summary['total_return']:.2%}")
        print(f"処理イベント数: {summary['events_processed']:,}")
        print(f"実行時間: {summary['total_execution_time_ms']:,.0f}ms")
        print(f"取引回数: {summary['trades_executed']}")

        perf = results["performance_metrics"]
        print("\n【パフォーマンス】")
        print(f"イベント処理速度: {perf['events_per_second']:,.0f} イベント/秒")
        print(f"平均イベント処理時間: {perf['avg_event_processing_time_us']:.1f}μs")

    else:
        print("テストデータが取得できませんでした")
