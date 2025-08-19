"""
取引関連データモデル

trade_manager.py からのリファクタリング抽出
モデルクラス: Trade, Position, BuyLot, RealizedPnL, TradeStatus
"""

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Union

from ...models.enums import TradeType


class TradeStatus(Enum):
    """取引ステータス"""

    PENDING = "pending"  # 注文中
    EXECUTED = "executed"  # 約定済み
    CANCELLED = "cancelled"  # キャンセル
    PARTIAL = "partial"  # 一部約定


@dataclass
class Trade:
    """
    取引記録データクラス（強化版）

    実際の売買取引に関する詳細情報を管理する。
    実装上、immutableなデータ構造として設計されている。
    """

    symbol: str
    trade_type: TradeType
    quantity: int
    price: Decimal
    timestamp: datetime
    commission: Decimal = field(default_factory=lambda: Decimal('0'))
    trade_id: Optional[str] = None
    order_id: Optional[str] = None
    market: str = "domestic"
    currency: str = "JPY"
    status: TradeStatus = TradeStatus.PENDING

    # パフォーマンス追跡
    execution_time_ms: Optional[float] = None
    slippage: Optional[Decimal] = None

    def __post_init__(self):
        """初期化後の検証"""
        if self.quantity <= 0:
            raise ValueError("取引数量は正の値である必要があります")
        if self.price <= 0:
            raise ValueError("取引価格は正の値である必要があります")
        if self.commission < 0:
            raise ValueError("手数料は非負の値である必要があります")

    @property
    def total_cost(self) -> Decimal:
        """総コスト（価格 × 数量 + 手数料）"""
        return self.price * self.quantity + self.commission

    @property
    def is_buy(self) -> bool:
        """買い注文かどうか"""
        return self.trade_type == TradeType.BUY

    @property
    def is_sell(self) -> bool:
        """売り注文かどうか"""
        return self.trade_type == TradeType.SELL


@dataclass
class BuyLot:
    """
    買い建て玉管理用データクラス

    FIFO（先入先出）法による売却処理のために、
    買い建て時の詳細情報を保持する。
    """

    quantity: int
    price: Decimal
    commission: Decimal
    timestamp: datetime
    remaining_quantity: int
    commission: Decimal = field(default_factory=lambda: Decimal('0'))

    def __post_init__(self):
        """初期化後の検証"""
        if self.remaining_quantity > self.quantity:
            raise ValueError("残数量が取引数量を超えています")
        if self.remaining_quantity < 0:
            raise ValueError("残数量は非負である必要があります")

    @property
    def is_fully_sold(self) -> bool:
        """完全に売却済みかどうか"""
        return self.remaining_quantity == 0

    @property
    def average_price_with_commission(self) -> Decimal:
        """手数料込みの平均取得単価"""
        return (self.price * self.quantity + self.commission) / self.quantity


@dataclass
class Position:
    """
    ポジション情報データクラス（強化版）

    特定銘柄の現在の保有状況を管理する。
    買い建て玉リストと売却履歴を追跡し、
    正確な損益計算を可能にする。
    """

    symbol: str
    buy_lots: List[BuyLot] = field(default_factory=list)
    total_quantity: int = 0
    average_price: Decimal = field(default_factory=lambda: Decimal('0'))

    # パフォーマンス指標
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal('0'))
    realized_pnl: Decimal = field(default_factory=lambda: Decimal('0'))
    total_commission: Decimal = field(default_factory=lambda: Decimal('0'))

    def update_metrics(self, current_price: Optional[Decimal] = None):
        """メトリクスの更新"""
        if current_price and self.total_quantity > 0:
            market_value = current_price * self.total_quantity
            cost_basis = sum(lot.price * lot.remaining_quantity for lot in self.buy_lots)
            self.unrealized_pnl = market_value - cost_basis - self.total_commission

    @property
    def is_empty(self) -> bool:
        """ポジションが空かどうか"""
        return self.total_quantity == 0


@dataclass
class RealizedPnL:
    """
    実現損益データクラス（詳細版）

    売却による実現損益の詳細情報を記録する。
    税務申告や運用レポート作成に必要な情報を含む。
    """
    symbol: str
    quantity: int
    buy_price: Decimal
    sell_price: Decimal
    buy_commission: Decimal
    sell_commission: Decimal
    net_pnl: Decimal

    # 追加メタデータ
    holding_period_days: int = 0
    currency: str = "JPY"

    def __post_init__(self):
        """保有期間の計算"""
        if self.holding_period_days == 0:
            delta = self.sell_timestamp - self.buy_timestamp
            self.holding_period_days = delta.days

    @property
    def return_rate(self) -> float:
        """収益率（%）"""
        if self.buy_price == 0:
            return 0.0
        return float(self.net_pnl / (self.buy_price * self.quantity) * 100)

    @property
    def is_profit(self) -> bool:
        """利益かどうか"""
        return self.net_pnl > 0

    @property
    def total_commission(self) -> Decimal:
        """総手数料"""
        return self.buy_commission + self.sell_commission
