from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

class EventType(Enum):
    """イベントタイプ"""
    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    TICK = "tick" # 将来のリアルタイム・高頻度トレード用
    CUSTOM = "custom" # 汎用カスタムイベント

@dataclass
class Event:
    """イベント基底クラス"""
    type: EventType # デフォルト値なし
    timestamp: datetime = field(default_factory=datetime.now, kw_only=True) # デフォルト値あり、かつキーワード専用
    data: Dict[str, Any] = field(default_factory=dict, kw_only=True) # デフォルト値あり、かつキーワード専用

@dataclass
class MarketDataEvent(Event):
    """市場データイベント"""
    # Eventから'type'がデフォルト値なしで継承されるため、ここでは再定義しない
    # デフォルト値を持たないフィールド
    symbol: str
    open: float
    high: float
    low: float
    close: float
    # デフォルト値を持つフィールド (kw_only=True の影響を受けない)
    volume: Optional[int] = None

    def __post_init__(self):
        # super().__post_init__() を削除
        self.type = EventType.MARKET_DATA # ここでタイプを設定

@dataclass
class SignalEvent(Event):
    """シグナルイベント"""
    # Eventから'type'がデフォルト値なしで継承されるため、ここでは再定義しない
    # デフォルト値を持たないフィールド
    symbol: str
    action: str  # "buy", "sell", "hold"
    # デフォルト値を持つフィールド
    strength: float = 0.0

    def __post_init__(self):
        # super().__post_init__() を削除
        self.type = EventType.SIGNAL

@dataclass
class OrderEvent(Event):
    """注文イベント"""
    # Eventから'type'がデフォルト値なしで継承されるため、ここでは再定義しない
    # デフォルト値を持たないフィールド
    order_id: str
    symbol: str
    action: str # "buy", "sell"
    quantity: int
    order_type: "OrderType" # OrderType Enumを使用
    # デフォルト値を持つフィールド
    price: Optional[float] = None
    stop_price: Optional[float] = None

    def __post_init__(self):
        # super().__post_init__() を削除
        self.type = EventType.ORDER

@dataclass
class FillEvent(Event):
    """約定イベント"""
    # Eventから'type'がデフォルト値なしで継承されるため、ここでは再定義しない
    # デフォルト値を持たないフィールド
    order_id: str
    symbol: str
    action: str # "buy", "sell"
    quantity: int
    price: float
    commission: float
    slippage: float
    fill_time: datetime

    def __post_init__(self):
        # super().__post_init__() を削除
        self.type = EventType.FILL

class OrderType(Enum):
    """注文タイプ"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    """注文状態"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """注文情報"""
    # デフォルト値を持たないフィールド
    order_id: str
    symbol: str
    order_type: OrderType
    side: str  # "buy" or "sell"
    quantity: int
    # デフォルト値を持つフィールド
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class TradeRecord:
    """個々の取引記録（エントリーからエグジットまで）"""

    trade_id: str # 取引を一意に識別するID
    symbol: str
    entry_time: datetime
    entry_price: float
    entry_quantity: int
    entry_commission: float
    entry_slippage: float
    
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_quantity: Optional[int] = None
    exit_commission: Optional[float] = None
    exit_slippage: Optional[float] = None
    
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    is_closed: bool = False

# TODO: 必要に応じて他のイベントタイプを追加 (例: NewsEvent, EconomicEvent)