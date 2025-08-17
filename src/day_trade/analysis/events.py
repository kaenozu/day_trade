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

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
        # super().__post_init__() を削除
        self.type = EventType.FILL

class OrderType(Enum):
    """注文タイプ"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop"
    BUY = "buy" # backtest_engine から統合
    SELL = "sell" # backtest_engine から統合

class OrderStatus(Enum):
    """注文状態"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXECUTED = "executed" # backtest_engine から統合

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
    execution_price: Optional[float] = None # backtest_engine から統合
    execution_time: Optional[datetime] = None # backtest_engine から統合

@dataclass
class Position:
    """ポジション情報"""

    symbol: str
    quantity: int = 0
    average_price: float = 0.0 # backtest_engine の avg_price と統合
    current_price: float = 0.0 # backtest_engine から統合
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class Portfolio:
    """ポートフォリオデータ"""
    cash: float
    positions: Dict[str, Position]
    total_value: float
    daily_return: float
    cumulative_return: float

@dataclass
class BacktestResults:
    """バックテスト結果"""
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    daily_returns: List[float]
    portfolio_values: List[float]
    positions_history: List[Dict[str, Any]]
