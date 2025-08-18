"""
型安全な取引システム

Protocol、TypedDict、ジェネリクスを使用した堅牢な型定義
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from datetime import datetime
from typing import (
    Any, Dict, Generic, List, Literal, Optional, Protocol, 
    TypedDict, TypeVar, Union, runtime_checkable
)
from typing_extensions import TypeGuard
from typing_extensions import NotRequired

# 基本型エイリアス
SymbolCode = str
Price = Union[Decimal, float, int]
Quantity = int
Timestamp = datetime


# リテラル型
TradeDirection = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop", "stop_limit"]
TimeInForce = Literal["day", "gtc", "ioc", "fok"]  # good_till_cancel, immediate_or_cancel, fill_or_kill
MarketType = Literal["domestic", "foreign", "crypto"]


class MarketDataDict(TypedDict):
    """マーケットデータの型定義"""
    symbol: SymbolCode
    timestamp: Timestamp
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Quantity
    market_type: NotRequired[MarketType]


class OrderDict(TypedDict):
    """注文の型定義"""
    order_id: str
    symbol: SymbolCode
    direction: TradeDirection
    order_type: OrderType
    quantity: Quantity
    price: NotRequired[Price]  # 成行注文では不要
    stop_price: NotRequired[Price]
    time_in_force: TimeInForce
    timestamp: Timestamp


class TradeDict(TypedDict):
    """取引の型定義"""
    trade_id: str
    order_id: str
    symbol: SymbolCode
    direction: TradeDirection
    quantity: Quantity
    price: Price
    commission: Price
    timestamp: Timestamp
    market_type: MarketType


class PositionDict(TypedDict):
    """ポジションの型定義"""
    symbol: SymbolCode
    quantity: Quantity
    average_price: Price
    unrealized_pnl: Price
    realized_pnl: Price
    total_commission: Price
    last_updated: Timestamp


class PortfolioSummaryDict(TypedDict):
    """ポートフォリオサマリーの型定義"""
    total_value: Price
    cash_balance: Price
    total_unrealized_pnl: Price
    total_realized_pnl: Price
    total_commission: Price
    position_count: int
    last_updated: Timestamp


# ジェネリック型変数
T = TypeVar('T')
P = TypeVar('P', bound=Price)
Q = TypeVar('Q', bound=Quantity)


@runtime_checkable
class Priceable(Protocol):
    """価格を持つオブジェクトのプロトコル"""
    
    @property
    def price(self) -> Price:
        """価格取得"""
        ...


@runtime_checkable
class Tradeable(Protocol):
    """取引可能なオブジェクトのプロトコル"""
    
    @property
    def symbol(self) -> SymbolCode:
        """銘柄コード"""
        ...
    
    @property
    def current_price(self) -> Price:
        """現在価格"""
        ...
    
    def is_tradeable(self) -> bool:
        """取引可能かどうか"""
        ...


@runtime_checkable
class MarketDataProvider(Protocol):
    """マーケットデータプロバイダーのプロトコル"""
    
    def get_current_price(self, symbol: SymbolCode) -> Optional[Price]:
        """現在価格取得"""
        ...
    
    def get_market_data(
        self, 
        symbol: SymbolCode, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[MarketDataDict]:
        """マーケットデータ取得"""
        ...
    
    def is_market_open(self, market_type: MarketType = "domestic") -> bool:
        """市場オープン状況"""
        ...


@runtime_checkable
class OrderExecutor(Protocol):
    """注文実行者のプロトコル"""
    
    def place_order(self, order: OrderDict) -> str:
        """注文発注"""
        ...
    
    def cancel_order(self, order_id: str) -> bool:
        """注文キャンセル"""
        ...
    
    def get_order_status(self, order_id: str) -> Optional[str]:
        """注文状況取得"""
        ...


@runtime_checkable
class PortfolioManager(Protocol):
    """ポートフォリオ管理者のプロトコル"""
    
    def get_position(self, symbol: SymbolCode) -> Optional[PositionDict]:
        """ポジション取得"""
        ...
    
    def get_all_positions(self) -> List[PositionDict]:
        """全ポジション取得"""
        ...
    
    def get_portfolio_summary(self) -> PortfolioSummaryDict:
        """ポートフォリオサマリー取得"""
        ...
    
    def calculate_unrealized_pnl(self, symbol: SymbolCode) -> Price:
        """未実現損益計算"""
        ...


class TradingResult(Generic[T]):
    """取引結果のジェネリッククラス"""
    
    def __init__(self, success: bool, data: Optional[T] = None, error: Optional[str] = None):
        self.success = success
        self.data = data
        self.error = error
        self.timestamp = datetime.now()
    
    @property
    def is_success(self) -> bool:
        """成功判定"""
        return self.success
    
    @property
    def is_error(self) -> bool:
        """エラー判定"""
        return not self.success
    
    def get_data(self) -> T:
        """データ取得（失敗時は例外発生）"""
        if not self.success or self.data is None:
            raise ValueError(f"Operation failed: {self.error}")
        return self.data
    
    def get_data_or_default(self, default: T) -> T:
        """データ取得（失敗時はデフォルト値）"""
        return self.data if self.success and self.data is not None else default


class NumericRange(Generic[P]):
    """数値範囲クラス"""
    
    def __init__(self, min_value: P, max_value: P):
        if min_value > max_value:
            raise ValueError("min_value must be less than or equal to max_value")
        self.min_value = min_value
        self.max_value = max_value
    
    def contains(self, value: P) -> bool:
        """値が範囲内かチェック"""
        return self.min_value <= value <= self.max_value
    
    def clamp(self, value: P) -> P:
        """値を範囲内に制限"""
        return max(self.min_value, min(value, self.max_value))


class TradingConstraints:
    """取引制約クラス"""
    
    def __init__(
        self,
        min_order_size: NumericRange[Quantity],
        price_range: NumericRange[Price],
        max_position_size: Optional[Quantity] = None,
        allowed_symbols: Optional[List[SymbolCode]] = None
    ):
        self.min_order_size = min_order_size
        self.price_range = price_range
        self.max_position_size = max_position_size
        self.allowed_symbols = allowed_symbols or []
    
    def validate_order(self, order: OrderDict) -> TradingResult[OrderDict]:
        """注文バリデーション"""
        # 数量チェック
        if not self.min_order_size.contains(order["quantity"]):
            return TradingResult(
                success=False,
                error=f"Order quantity {order['quantity']} is outside allowed range"
            )
        
        # 価格チェック（指値注文の場合）
        if "price" in order and not self.price_range.contains(order["price"]):
            return TradingResult(
                success=False,
                error=f"Order price {order['price']} is outside allowed range"
            )
        
        # 銘柄チェック
        if self.allowed_symbols and order["symbol"] not in self.allowed_symbols:
            return TradingResult(
                success=False,
                error=f"Symbol {order['symbol']} is not allowed"
            )
        
        return TradingResult(success=True, data=order)


class TypeSafeTradeManager(Generic[T]):
    """型安全な取引マネージャー"""
    
    def __init__(
        self,
        data_provider: MarketDataProvider,
        order_executor: OrderExecutor,
        portfolio_manager: PortfolioManager,
        constraints: TradingConstraints
    ):
        self.data_provider = data_provider
        self.order_executor = order_executor
        self.portfolio_manager = portfolio_manager
        self.constraints = constraints
    
    def place_market_order(
        self,
        symbol: SymbolCode,
        direction: TradeDirection,
        quantity: Quantity
    ) -> TradingResult[str]:
        """成行注文発注"""
        order: OrderDict = {
            "order_id": f"order_{datetime.now().timestamp()}",
            "symbol": symbol,
            "direction": direction,
            "order_type": "market",
            "quantity": quantity,
            "time_in_force": "day",
            "timestamp": datetime.now()
        }
        
        # バリデーション
        validation_result = self.constraints.validate_order(order)
        if not validation_result.is_success:
            return TradingResult(success=False, error=validation_result.error)
        
        # 市場オープンチェック
        if not self.data_provider.is_market_open():
            return TradingResult(success=False, error="Market is closed")
        
        # 注文実行
        try:
            order_id = self.order_executor.place_order(order)
            return TradingResult(success=True, data=order_id)
        except Exception as e:
            return TradingResult(success=False, error=str(e))
    
    def place_limit_order(
        self,
        symbol: SymbolCode,
        direction: TradeDirection,
        quantity: Quantity,
        price: Price
    ) -> TradingResult[str]:
        """指値注文発注"""
        order: OrderDict = {
            "order_id": f"order_{datetime.now().timestamp()}",
            "symbol": symbol,
            "direction": direction,
            "order_type": "limit",
            "quantity": quantity,
            "price": price,
            "time_in_force": "gtc",
            "timestamp": datetime.now()
        }
        
        # バリデーション
        validation_result = self.constraints.validate_order(order)
        if not validation_result.is_success:
            return TradingResult(success=False, error=validation_result.error)
        
        # 注文実行
        try:
            order_id = self.order_executor.place_order(order)
            return TradingResult(success=True, data=order_id)
        except Exception as e:
            return TradingResult(success=False, error=str(e))
    
    def get_safe_portfolio_value(self) -> TradingResult[Price]:
        """型安全なポートフォリオ価値取得"""
        try:
            summary = self.portfolio_manager.get_portfolio_summary()
            return TradingResult(success=True, data=summary["total_value"])
        except Exception as e:
            return TradingResult(success=False, error=str(e))


def validate_trading_data(data: Dict[str, Any]) -> bool:
    """取引データの型安全バリデーション"""
    required_fields = ["symbol", "direction", "quantity", "timestamp"]
    
    for field in required_fields:
        if field not in data:
            return False
    
    # 型チェック
    if not isinstance(data["symbol"], str):
        return False
    if data["direction"] not in ["buy", "sell"]:
        return False
    if not isinstance(data["quantity"], int) or data["quantity"] <= 0:
        return False
    if not isinstance(data["timestamp"], datetime):
        return False
    
    return True


# 型ガード関数
def is_market_data(data: Any) -> TypeGuard[MarketDataDict]:
    """マーケットデータの型ガード"""
    if not isinstance(data, dict):
        return False
    
    required_fields = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    return all(field in data for field in required_fields)


def is_order_data(data: Any) -> TypeGuard[OrderDict]:
    """注文データの型ガード"""
    if not isinstance(data, dict):
        return False
    
    required_fields = ["order_id", "symbol", "direction", "order_type", "quantity", "time_in_force", "timestamp"]
    return all(field in data for field in required_fields)


# 型変換ユーティリティ
def ensure_decimal(value: Price) -> Decimal:
    """Price型をDecimalに確実に変換"""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def ensure_symbol(value: Any) -> SymbolCode:
    """値をSymbolCodeに変換（バリデーション付き）"""
    if not isinstance(value, str):
        raise TypeError(f"Symbol must be string, got {type(value)}")
    
    if not value or len(value.strip()) == 0:
        raise ValueError("Symbol cannot be empty")
    
    return value.strip().upper()