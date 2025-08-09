"""
取引管理システム型定義

データクラス・Enum・型定義の統合管理
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict

from ...models.enums import TradeType


class TradeStatus(Enum):
    """取引ステータス"""

    PENDING = "pending"      # 注文中
    EXECUTED = "executed"    # 約定済み
    CANCELLED = "cancelled"  # キャンセル
    PARTIAL = "partial"      # 一部約定


@dataclass
class Trade:
    """取引記録"""

    id: str
    symbol: str
    trade_type: TradeType
    quantity: int
    price: Decimal
    timestamp: datetime
    commission: Decimal = Decimal("0")
    status: TradeStatus = TradeStatus.EXECUTED
    notes: str = ""

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        data = asdict(self)
        data["trade_type"] = self.trade_type.value
        data["status"] = self.status.value
        data["price"] = str(self.price)
        data["commission"] = str(self.commission)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "Trade":
        """辞書から復元"""
        return cls(
            id=data["id"],
            symbol=data["symbol"],
            trade_type=TradeType(data["trade_type"]),
            quantity=data["quantity"],
            price=Decimal(data["price"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            commission=Decimal(data["commission"]),
            status=TradeStatus(data["status"]),
            notes=data.get("notes", ""),
        )


@dataclass
class Position:
    """ポジション情報"""

    symbol: str
    quantity: int
    average_price: Decimal
    total_cost: Decimal
    current_price: Decimal = Decimal("0")

    @property
    def market_value(self) -> Decimal:
        """時価総額"""
        return self.current_price * Decimal(self.quantity)

    @property
    def unrealized_pnl(self) -> Decimal:
        """含み損益"""
        return self.market_value - self.total_cost

    @property
    def unrealized_pnl_percentage(self) -> Decimal:
        """含み損益率(%)"""
        if self.total_cost == 0:
            return Decimal("0")
        return (self.unrealized_pnl / self.total_cost) * Decimal("100")

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_price": str(self.average_price),
            "total_cost": str(self.total_cost),
            "current_price": str(self.current_price),
            "market_value": str(self.market_value),
            "unrealized_pnl": str(self.unrealized_pnl),
            "unrealized_pnl_percentage": str(self.unrealized_pnl_percentage),
        }


@dataclass
class RealizedPnL:
    """実現損益記録"""

    symbol: str
    quantity: int
    buy_price: Decimal
    sell_price: Decimal
    buy_date: datetime
    sell_date: datetime
    realized_pnl: Decimal
    commission: Decimal = Decimal("0")

    @property
    def holding_days(self) -> int:
        """保有日数"""
        return (self.sell_date - self.buy_date).days

    @property
    def net_pnl(self) -> Decimal:
        """手数料控除後損益"""
        return self.realized_pnl - self.commission

    @property
    def return_percentage(self) -> Decimal:
        """リターン率(%)"""
        if self.buy_price == 0:
            return Decimal("0")
        total_buy_cost = self.buy_price * Decimal(self.quantity)
        return (self.net_pnl / total_buy_cost) * Decimal("100")

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "buy_price": str(self.buy_price),
            "sell_price": str(self.sell_price),
            "buy_date": self.buy_date.isoformat(),
            "sell_date": self.sell_date.isoformat(),
            "realized_pnl": str(self.realized_pnl),
            "commission": str(self.commission),
            "holding_days": self.holding_days,
            "net_pnl": str(self.net_pnl),
            "return_percentage": str(self.return_percentage),
        }
