"""
高度なバックテストエンジン - 注文管理システム

注文の処理、約定判定、価格計算を管理。
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from day_trade.analysis.events import (
    Event, EventType, MarketDataEvent, OrderEvent, FillEvent, 
    Order, OrderType, OrderStatus
)
from day_trade.utils.logging_config import get_context_logger
from .data_structures import TradingCosts

warnings.filterwarnings("ignore")
logger = get_context_logger(__name__)


@dataclass
class Order:
    """注文情報"""

    order_id: str
    symbol: str
    order_type: OrderType
    side: str  # "buy" or "sell"
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0


class OrderManager:
    """注文管理システム"""

    def __init__(self, trading_costs: TradingCosts):
        """注文管理システムの初期化"""
        self.trading_costs = trading_costs
        self.pending_orders: Dict[str, Order] = {}

    def create_order_from_event(self, event: OrderEvent) -> Order:
        """OrderEventからOrderオブジェクトを生成"""
        order_type_enum = OrderType[event.order_type.upper()]
        order = Order(
            order_id=event.order_id,
            symbol=event.symbol,
            order_type=order_type_enum,
            side=event.action,
            quantity=event.quantity,
            price=event.price,
            stop_price=event.stop_price,
            timestamp=event.timestamp,
        )
        self.pending_orders[order.order_id] = order
        return order

    def should_fill_order(self, order: Order, market_data: pd.Series) -> bool:
        """注文約定判定"""
        if order.order_type == OrderType.MARKET:
            return True

        elif order.order_type == OrderType.LIMIT:
            if (
                order.side == "buy"
                and market_data["Low"] <= order.price
                or order.side == "sell"
                and market_data["High"] >= order.price
            ):
                return True

        elif order.order_type == OrderType.STOP and (
            (order.side == "buy" and market_data["High"] >= order.stop_price)
            or (order.side == "sell" and market_data["Low"] <= order.stop_price)
        ):
            return True

        return False

    def calculate_fill_price(
        self, order: Order, market_data: pd.Series, 
        enable_slippage: bool = True, enable_market_impact: bool = True
    ) -> float:
        """約定価格計算"""
        base_price = market_data["Close"]

        if order.order_type == OrderType.MARKET:
            # マーケット注文：現在価格 + スプレッド + スリッページ
            spread_impact = self.trading_costs.bid_ask_spread_rate / 2
            if order.side == "buy":
                base_price *= 1 + spread_impact
            else:
                base_price *= 1 - spread_impact

        elif order.order_type == OrderType.LIMIT:
            base_price = order.price

        elif order.order_type == OrderType.STOP:
            base_price = order.stop_price

        # スリッページ適用
        if enable_slippage:
            slippage = np.random.normal(0, self.trading_costs.slippage_rate)
            if order.side == "buy":
                base_price *= 1 + abs(slippage)
            else:
                base_price *= 1 - abs(slippage)

        # マーケットインパクト適用
        if enable_market_impact:
            volume_ratio = order.quantity / market_data.get("Volume", 1000000)
            impact = volume_ratio * self.trading_costs.market_impact_rate
            if order.side == "buy":
                base_price *= 1 + impact
            else:
                base_price *= 1 - impact

        return base_price

    def calculate_commission(self, order: Order, fill_price: float) -> float:
        """手数料計算"""
        commission = order.quantity * fill_price * self.trading_costs.commission_rate
        
        # 最小・最大手数料の制限を適用
        commission = max(commission, self.trading_costs.min_commission)
        commission = min(commission, self.trading_costs.max_commission)
        
        return commission

    def process_pending_orders(
        self, current_time: datetime, market_data_event: MarketDataEvent,
        enable_slippage: bool = True, enable_market_impact: bool = True
    ) -> List[FillEvent]:
        """待機中注文の処理"""
        fill_events = []
        orders_to_remove = []

        for order_id, order in list(self.pending_orders.items()):
            if order.status != OrderStatus.PENDING:
                continue

            # このシンボルの市場データに対してのみ処理
            if order.symbol != market_data_event.symbol:
                continue

            # MarketDataEventをpd.Seriesに変換
            symbol_market_data_series = pd.Series({
                "Open": market_data_event.open,
                "High": market_data_event.high,
                "Low": market_data_event.low,
                "Close": market_data_event.close,
                "Volume": market_data_event.volume
            })
            symbol_market_data_series.name = market_data_event.symbol

            if self.should_fill_order(order, symbol_market_data_series):
                filled_price = self.calculate_fill_price(
                    order, symbol_market_data_series, 
                    enable_slippage, enable_market_impact
                )
                commission = self.calculate_commission(order, filled_price)

                # 約定イベントを生成
                fill_event = FillEvent(
                    type=EventType.FILL,
                    order_id=order.order_id,
                    symbol=order.symbol,
                    action=order.side,
                    quantity=order.quantity,
                    price=filled_price,
                    commission=commission,
                    slippage=abs(filled_price - (order.price or filled_price)),
                    fill_time=current_time,
                )
                
                fill_events.append(fill_event)
                orders_to_remove.append(order_id)

                logger.debug(
                    f"注文約定: {order.order_id} at {filled_price}",
                    section="order_processing"
                )

        # 処理済みの注文を削除
        for order_id in orders_to_remove:
            del self.pending_orders[order_id]

        return fill_events

    def get_pending_order_count(self) -> int:
        """待機中注文数を取得"""
        return len([order for order in self.pending_orders.values() 
                   if order.status == OrderStatus.PENDING])

    def cancel_order(self, order_id: str) -> bool:
        """注文キャンセル"""
        if order_id in self.pending_orders:
            self.pending_orders[order_id].status = OrderStatus.CANCELLED
            del self.pending_orders[order_id]
            logger.debug(f"注文キャンセル: {order_id}")
            return True
        return False

    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        """注文ID別取得"""
        return self.pending_orders.get(order_id)