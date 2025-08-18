"""
取引ドメインのエンティティ

ドメイン駆動設計に基づく核となるビジネスオブジェクト
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Protocol
from uuid import UUID, uuid4

from ..common.value_objects import Money, Quantity, Symbol, Price
from ..common.domain_events import DomainEvent, TradeExecutedEvent, PositionOpenedEvent, PositionClosedEvent


class TradeId:
    """取引ID値オブジェクト"""
    
    def __init__(self, value: Optional[UUID] = None):
        self._value = value or uuid4()
    
    @property
    def value(self) -> UUID:
        return self._value
    
    def __eq__(self, other) -> bool:
        return isinstance(other, TradeId) and self._value == other._value
    
    def __hash__(self) -> int:
        return hash(self._value)
    
    def __str__(self) -> str:
        return str(self._value)


class PositionId:
    """ポジションID値オブジェクト"""
    
    def __init__(self, value: Optional[UUID] = None):
        self._value = value or uuid4()
    
    @property
    def value(self) -> UUID:
        return self._value
    
    def __eq__(self, other) -> bool:
        return isinstance(other, PositionId) and self._value == other._value
    
    def __hash__(self) -> int:
        return hash(self._value)


class TradeExecuted(DomainEvent):
    """取引実行ドメインイベント"""
    
    def __init__(self, trade_id: TradeId, symbol: Symbol, quantity: Quantity, price: Price):
        super().__init__()
        self.trade_id = trade_id
        self.symbol = symbol
        self.quantity = quantity
        self.price = price


class PositionOpened(DomainEvent):
    """ポジション開始ドメインイベント"""
    
    def __init__(self, position_id: PositionId, symbol: Symbol, initial_quantity: Quantity):
        super().__init__()
        self.position_id = position_id
        self.symbol = symbol
        self.initial_quantity = initial_quantity


class PositionClosed(DomainEvent):
    """ポジション終了ドメインイベント"""
    
    def __init__(self, position_id: PositionId, realized_pnl: Money):
        super().__init__()
        self.position_id = position_id
        self.realized_pnl = realized_pnl


class Trade:
    """取引エンティティ
    
    不変のビジネスルールを持つ取引の核となるオブジェクト
    """
    
    def __init__(
        self,
        trade_id: TradeId,
        symbol: Symbol,
        quantity: Quantity,
        price: Price,
        direction: str,  # "buy" or "sell"
        executed_at: datetime,
        commission: Money
    ):
        self._id = trade_id
        self._symbol = symbol
        self._quantity = quantity
        self._price = price
        self._direction = direction.lower()
        self._executed_at = executed_at
        self._commission = commission
        self._domain_events: List[DomainEvent] = []
        
        # ビジネスルール検証
        self._validate()
        
        # ドメインイベント発行
        self._domain_events.append(
            TradeExecutedEvent(
                trade_id=trade_id.value,
                symbol=symbol.code,
                quantity=quantity.value,
                price=str(price.value),
                direction=direction,
                commission=str(commission.amount)
            )
        )
    
    def _validate(self):
        """ビジネスルール検証"""
        if self._quantity.value <= 0:
            raise ValueError("取引数量は正の値である必要があります")
        
        if self._price.value <= Decimal('0'):
            raise ValueError("取引価格は正の値である必要があります")
        
        if self._direction not in ['buy', 'sell']:
            raise ValueError("取引方向は'buy'または'sell'である必要があります")
        
        if self._commission.amount < Decimal('0'):
            raise ValueError("手数料は非負の値である必要があります")
    
    @property
    def id(self) -> TradeId:
        return self._id
    
    @property
    def symbol(self) -> Symbol:
        return self._symbol
    
    @property
    def quantity(self) -> Quantity:
        return self._quantity
    
    @property
    def price(self) -> Price:
        return self._price
    
    @property
    def direction(self) -> str:
        return self._direction
    
    @property
    def executed_at(self) -> datetime:
        return self._executed_at
    
    @property
    def commission(self) -> Money:
        return self._commission
    
    @property
    def is_buy(self) -> bool:
        return self._direction == 'buy'
    
    @property
    def is_sell(self) -> bool:
        return self._direction == 'sell'
    
    @property
    def total_value(self) -> Money:
        """取引総額（価格 × 数量）"""
        return Money(self._price.value * self._quantity.value)
    
    @property
    def net_value(self) -> Money:
        """手数料控除後価値"""
        total = self.total_value
        if self.is_buy:
            return Money(total.amount + self._commission.amount)
        else:
            return Money(total.amount - self._commission.amount)
    
    def get_domain_events(self) -> List[DomainEvent]:
        """ドメインイベント取得"""
        return self._domain_events.copy()
    
    def clear_domain_events(self):
        """ドメインイベントクリア"""
        self._domain_events.clear()


class Position:
    """ポジションエンティティ
    
    特定銘柄の保有状況を管理するドメインオブジェクト
    """
    
    def __init__(self, position_id: PositionId, symbol: Symbol):
        self._id = position_id
        self._symbol = symbol
        self._trades: List[Trade] = []
        self._is_closed = False
        self._domain_events: List[DomainEvent] = []
    
    @property
    def id(self) -> PositionId:
        return self._id
    
    @property
    def symbol(self) -> Symbol:
        return self._symbol
    
    @property
    def is_closed(self) -> bool:
        return self._is_closed
    
    @property
    def trades(self) -> List[Trade]:
        return self._trades.copy()
    
    def add_trade(self, trade: Trade):
        """取引追加"""
        if self._is_closed:
            raise ValueError("クローズされたポジションに取引を追加できません")
        
        if trade.symbol != self._symbol:
            raise ValueError("異なる銘柄の取引を追加できません")
        
        self._trades.append(trade)
        
        # 初回取引でポジション開始イベント
        if len(self._trades) == 1 and trade.is_buy:
            self._domain_events.append(
                PositionOpenedEvent(
                    position_id=self._id.value,
                    symbol=self._symbol.code,
                    initial_quantity=trade.quantity.value
                )
            )
    
    def calculate_current_quantity(self) -> Quantity:
        """現在保有数量計算"""
        total_bought = sum(
            trade.quantity.value for trade in self._trades if trade.is_buy
        )
        total_sold = sum(
            trade.quantity.value for trade in self._trades if trade.is_sell
        )
        return Quantity(total_bought - total_sold)
    
    def calculate_average_cost(self) -> Optional[Price]:
        """平均取得単価計算"""
        buy_trades = [trade for trade in self._trades if trade.is_buy]
        if not buy_trades:
            return None
        
        total_cost = sum(trade.net_value.amount for trade in buy_trades)
        total_quantity = sum(trade.quantity.value for trade in buy_trades)
        
        if total_quantity == 0:
            return None
        
        return Price(total_cost / total_quantity)
    
    def calculate_realized_pnl(self) -> Money:
        """実現損益計算（FIFO法）"""
        buy_trades = [trade for trade in self._trades if trade.is_buy]
        sell_trades = [trade for trade in self._trades if trade.is_sell]
        
        realized_pnl = Decimal('0')
        remaining_buys = [(trade.quantity.value, trade.price.value, trade.commission.amount) 
                         for trade in buy_trades]
        
        for sell_trade in sell_trades:
            remaining_sell_qty = sell_trade.quantity.value
            sell_price = sell_trade.price.value
            sell_commission = sell_trade.commission.amount
            
            while remaining_sell_qty > 0 and remaining_buys:
                buy_qty, buy_price, buy_commission = remaining_buys[0]
                
                matched_qty = min(remaining_sell_qty, buy_qty)
                
                # 損益計算
                buy_cost = (buy_price + buy_commission / buy_qty) * matched_qty
                sell_proceeds = (sell_price - sell_commission / sell_trade.quantity.value) * matched_qty
                pnl = sell_proceeds - buy_cost
                
                realized_pnl += pnl
                remaining_sell_qty -= matched_qty
                
                # 買い建て玉更新
                if matched_qty == buy_qty:
                    remaining_buys.pop(0)
                else:
                    remaining_buys[0] = (buy_qty - matched_qty, buy_price, buy_commission)
        
        return Money(realized_pnl)
    
    def calculate_unrealized_pnl(self, current_price: Price) -> Money:
        """未実現損益計算"""
        current_qty = self.calculate_current_quantity()
        if current_qty.value <= 0:
            return Money(Decimal('0'))
        
        avg_cost = self.calculate_average_cost()
        if avg_cost is None:
            return Money(Decimal('0'))
        
        market_value = current_price.value * current_qty.value
        cost_basis = avg_cost.value * current_qty.value
        
        return Money(market_value - cost_basis)
    
    def close_position(self) -> Money:
        """ポジションクローズ"""
        if self._is_closed:
            raise ValueError("既にクローズされたポジションです")
        
        current_qty = self.calculate_current_quantity()
        if current_qty.value != 0:
            raise ValueError("保有数量が0でないためクローズできません")
        
        self._is_closed = True
        realized_pnl = self.calculate_realized_pnl()
        
        self._domain_events.append(
            PositionClosedEvent(
                position_id=self._id.value,
                symbol=self._symbol.code,
                realized_pnl=str(realized_pnl.amount),
                total_trades=len(self._trades)
            )
        )
        
        return realized_pnl
    
    def get_domain_events(self) -> List[DomainEvent]:
        """ドメインイベント取得"""
        return self._domain_events.copy()
    
    def clear_domain_events(self):
        """ドメインイベントクリア"""
        self._domain_events.clear()


class Portfolio:
    """ポートフォリオ集約ルート
    
    複数ポジションを管理する集約ルート
    """
    
    def __init__(self, portfolio_id: UUID):
        self._id = portfolio_id
        self._positions: dict[Symbol, Position] = {}
        self._domain_events: List[DomainEvent] = []
    
    @property
    def id(self) -> UUID:
        return self._id
    
    def get_position(self, symbol: Symbol) -> Optional[Position]:
        """ポジション取得"""
        return self._positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        """全ポジション取得"""
        return list(self._positions.values())
    
    def execute_trade(self, trade: Trade) -> Position:
        """取引実行"""
        symbol = trade.symbol
        
        # 既存ポジション取得または新規作成
        if symbol not in self._positions:
            position_id = PositionId()
            self._positions[symbol] = Position(position_id, symbol)
        
        position = self._positions[symbol]
        position.add_trade(trade)
        
        # ポジションのドメインイベントを収集
        position_events = position.get_domain_events()
        self._domain_events.extend(position_events)
        position.clear_domain_events()
        
        # 取引のドメインイベントを収集
        trade_events = trade.get_domain_events()
        self._domain_events.extend(trade_events)
        trade.clear_domain_events()
        
        return position
    
    def calculate_total_value(self, current_prices: dict[Symbol, Price]) -> Money:
        """ポートフォリオ総価値計算"""
        total_value = Decimal('0')
        
        for position in self._positions.values():
            if position.symbol in current_prices:
                current_price = current_prices[position.symbol]
                qty = position.calculate_current_quantity()
                position_value = current_price.value * qty.value
                total_value += position_value
        
        return Money(total_value)
    
    def calculate_total_pnl(self, current_prices: dict[Symbol, Price]) -> Money:
        """総損益計算"""
        total_pnl = Decimal('0')
        
        for position in self._positions.values():
            # 実現損益
            realized = position.calculate_realized_pnl()
            total_pnl += realized.amount
            
            # 未実現損益
            if position.symbol in current_prices:
                current_price = current_prices[position.symbol]
                unrealized = position.calculate_unrealized_pnl(current_price)
                total_pnl += unrealized.amount
        
        return Money(total_pnl)
    
    def get_domain_events(self) -> List[DomainEvent]:
        """ドメインイベント取得"""
        return self._domain_events.copy()
    
    def clear_domain_events(self):
        """ドメインイベントクリア"""
        self._domain_events.clear()