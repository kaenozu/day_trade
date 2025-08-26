"""
高度なバックテストエンジン - イベント処理システム

市場データ、シグナル、注文、約定イベントの処理を管理。
"""

import warnings
from collections import deque
from datetime import datetime
from typing import Callable, Deque, Dict, List, Optional

import pandas as pd

from day_trade.analysis.events import (
    Event, EventType, MarketDataEvent, SignalEvent, OrderEvent, FillEvent, Order
)
from day_trade.utils.logging_config import get_context_logger
from .order_management import OrderManager
from .position_management import PositionManager

warnings.filterwarnings("ignore")
logger = get_context_logger(__name__)


class EventHandler:
    """イベント処理システム"""

    def __init__(
        self, 
        order_manager: OrderManager,
        position_manager: PositionManager,
        max_position_size: float = 0.1,
        position_sizing: str = "fixed"
    ):
        """イベントハンドラの初期化"""
        self.order_manager = order_manager
        self.position_manager = position_manager
        self.max_position_size = max_position_size
        self.position_sizing = position_sizing

        # 各シンボルの最新市場データを保存
        self.current_market_data_by_symbol: Dict[str, MarketDataEvent] = {}
        self._current_sim_time: Optional[datetime] = None

        # イベントキュー
        self.events: Deque[Event] = deque()

        # イベントハンドラマッピング
        self._event_handlers: Dict[EventType, List[Callable]] = {
            EventType.MARKET_DATA: [self._handle_market_data],
            EventType.SIGNAL: [self._handle_signal],
            EventType.ORDER: [self._handle_order],
            EventType.FILL: [self._handle_fill],
        }

    def process_event(self, event: Event):
        """単一イベントの処理"""
        self._current_sim_time = event.timestamp

        try:
            if isinstance(event, MarketDataEvent):
                self._handle_market_data(event)
            elif isinstance(event, SignalEvent):
                self._handle_signal(event)
            elif isinstance(event, OrderEvent):
                self._handle_order(event)
            elif isinstance(event, FillEvent):
                self._handle_fill(event)
            else:
                logger.warning(f"未処理のイベントタイプ: {event.type.value}")

        except Exception as e:
            logger.warning(
                f"イベント処理エラー: {event.type.value} at {event.timestamp} - {e}",
                section="event_processing",
                error=str(e),
            )

    def _handle_market_data(self, event: MarketDataEvent) -> None:
        """市場データイベントを処理"""
        # 最新の市場データを保存
        self.current_market_data_by_symbol[event.symbol] = event

        # 待機中注文の処理
        fill_events = self.order_manager.process_pending_orders(
            event.timestamp, event, enable_slippage=True, enable_market_impact=True
        )
        
        # 生成された約定イベントをキューに追加
        for fill_event in fill_events:
            self.events.append(fill_event)

        # ポジション時価の更新
        self.position_manager.update_position_market_value(
            event.symbol, event.close, event.timestamp
        )

    def _handle_signal(self, event: SignalEvent) -> None:
        """シグナルイベントを処理"""
        symbol = event.symbol
        signal_action = event.action
        confidence = event.strength

        # 現在の市場データを取得
        current_market_data_event = self.current_market_data_by_symbol.get(symbol)
        if current_market_data_event is None:
            logger.warning(
                f"シグナル処理に十分な市場データがありません: {symbol} at {event.timestamp}"
            )
            return

        current_price_for_signal = current_market_data_event.close

        if signal_action in ["buy", "sell"] and confidence > 50.0:
            # ポジションサイズ計算
            position_size = self._calculate_position_size(
                symbol, current_price_for_signal, confidence
            )

            if position_size > 0:
                # OrderEventを生成し、キューにプッシュ
                order_id = f"{symbol}_{self._current_sim_time.strftime('%Y%m%d_%H%M%S')}"
                order_event = OrderEvent(
                    type=EventType.ORDER,
                    order_id=order_id,
                    symbol=symbol,
                    action=signal_action,
                    quantity=position_size,
                    order_type="market",
                    price=current_price_for_signal,
                    timestamp=self._current_sim_time,
                )
                self.events.append(order_event)

                logger.debug(
                    f"シグナルから注文イベント生成: {signal_action} {position_size} {symbol}",
                    section="signal_processing",
                    confidence=confidence,
                )

    def _handle_order(self, event: OrderEvent) -> None:
        """注文イベントを処理"""
        order = self.order_manager.create_order_from_event(event)
        logger.debug(f"新しい注文をキューに追加: {order.order_id}")

    def _handle_fill(self, event: FillEvent) -> None:
        """約定イベントを処理"""
        order = self.order_manager.get_order_by_id(event.order_id)
        if not order:
            logger.warning(f"未知の注文IDの約定イベント: {event.order_id}")
            return

        # 注文情報を更新
        order.status = Order.OrderStatus.FILLED
        order.filled_quantity = event.quantity
        order.filled_price = event.price
        order.commission = event.commission
        order.slippage = event.slippage

        # ポジションを更新
        realized_pnl = self.position_manager.update_position_from_order(order)

        # TradeRecordの処理
        self.position_manager.process_trade_record(
            order, event.fill_time, realized_pnl
        )

        logger.debug(f"注文約定を処理: {order.order_id}")

    def _calculate_position_size(
        self, symbol: str, price: float, confidence: float
    ) -> int:
        """ポジションサイズ計算"""
        if self.position_sizing == "fixed":
            # 固定金額
            target_value = self.position_manager.initial_capital * self.max_position_size
            return int(target_value / price)

        elif self.position_sizing == "percent":
            # ポートフォリオ比率
            current_portfolio_value = self.position_manager.get_portfolio_value()
            target_value = current_portfolio_value * self.max_position_size
            return int(target_value / price)

        elif self.position_sizing == "volatility":
            # ボラティリティ調整
            base_target = self.position_manager.get_portfolio_value() * self.max_position_size
            confidence_multiplier = confidence / 100.0
            adjusted_target = base_target * confidence_multiplier
            return int(adjusted_target / price)

        return 0

    def create_event_queue_from_data(
        self, data: pd.DataFrame, strategy_signals: pd.DataFrame
    ) -> Deque[Event]:
        """データからイベントキューを生成"""
        events = []
        all_timestamps = sorted(data.index.union(strategy_signals.index).unique())

        for timestamp in all_timestamps:
            if timestamp in data.index:
                # MarketDataEventを生成
                row = data.loc[timestamp]
                symbol = (data.columns.levels[0][0] if isinstance(data.columns, pd.MultiIndex) 
                         else "UNKNOWN")
                market_event = MarketDataEvent(
                    type=EventType.MARKET_DATA,
                    timestamp=timestamp,
                    symbol=symbol,
                    open=row[(symbol, "Open")] if isinstance(data.columns, pd.MultiIndex) else row["Open"],
                    high=row[(symbol, "High")] if isinstance(data.columns, pd.MultiIndex) else row["High"],
                    low=row[(symbol, "Low")] if isinstance(data.columns, pd.MultiIndex) else row["Low"],
                    close=row[(symbol, "Close")] if isinstance(data.columns, pd.MultiIndex) else row["Close"],
                    volume=row.get((symbol, "Volume")) if isinstance(data.columns, pd.MultiIndex) else row.get("Volume"),
                )
                events.append(market_event)

            if timestamp in strategy_signals.index:
                # SignalEventを生成
                signal_row = strategy_signals.loc[timestamp]
                symbol = (strategy_signals.columns.levels[0][0] if isinstance(strategy_signals.columns, pd.MultiIndex) 
                         else "UNKNOWN")
                signal_event = SignalEvent(
                    type=EventType.SIGNAL,
                    timestamp=timestamp,
                    symbol=symbol,
                    action=signal_row[(symbol, "signal")] if isinstance(strategy_signals.columns, pd.MultiIndex) else signal_row["signal"],
                    strength=signal_row.get((symbol, "confidence"), 0.0) if isinstance(strategy_signals.columns, pd.MultiIndex) else signal_row.get("confidence", 0.0),
                    data={"price_target": signal_row.get((symbol, "price_target"))} if isinstance(strategy_signals.columns, pd.MultiIndex) else {"price_target": signal_row.get("price_target")},
                )
                events.append(signal_event)

        # イベントキューを時系列順にソート
        return deque(sorted(events, key=lambda event: event.timestamp))

    def has_pending_events(self) -> bool:
        """保留中のイベントがあるかチェック"""
        return len(self.events) > 0

    def get_next_event(self) -> Optional[Event]:
        """次のイベントを取得"""
        return self.events.popleft() if self.events else None