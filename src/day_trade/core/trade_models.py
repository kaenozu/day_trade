from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Deque, Dict
from collections import deque # 追加

from ..models.enums import TradeType
from .trade_utils import safe_decimal_conversion, validate_positive_decimal


class TradeStatus(Enum):
    """取引ステータス"""

    PENDING = "pending"  # 注文中
    EXECUTED = "executed"  # 約定済み
    CANCELLED = "cancelled"  # キャンセル
    PARTIAL = "partial"  # 一部約定


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
        """辞書から復元（安全なDecimal変換）"""
        try:
            # 必須フィールドの検証
            required_fields = [
                "id",
                "symbol",
                "trade_type",
                "quantity",
                "price",
                "timestamp",
            ]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"必須フィールド '{field}' が不足しています")

            # 安全なDecimal変換
            price = safe_decimal_conversion(data["price"], "価格")
            commission = safe_decimal_conversion(data.get("commission", "0"), "手数料")

            # 正数検証
            validate_positive_decimal(price, "価格")
            validate_positive_decimal(commission, "手数料", allow_zero=True)

            # 数量の検証
            quantity = int(data["quantity"])
            if quantity <= 0:
                raise ValueError(f"数量は正数である必要があります: {quantity}")

            return cls(
                id=str(data["id"]),
                symbol=str(data["symbol"]),
                trade_type=TradeType(data["trade_type"]),
                quantity=quantity,
                price=price,
                timestamp=datetime.fromisoformat(data["timestamp"]),
                commission=commission,
                status=TradeStatus(data.get("status", TradeStatus.EXECUTED.value)),
                notes=str(data.get("notes", "")),
            )
        except Exception as e:
            raise ValueError(f"取引データの復元に失敗しました: {str(e)}")


@dataclass
class BuyLot:
    """
    FIFO会計のための買いロット情報
    個別の買い取引を管理し、正確な売却対応を可能にする
    """

    quantity: int
    price: Decimal
    commission: Decimal
    timestamp: datetime
    trade_id: str

    def total_cost_per_share(self) -> Decimal:
        """1株あたりの総コスト（買い価格 + 手数料按分）"""
        if self.quantity == 0:
            return Decimal("0")
        return self.price + (self.commission / Decimal(self.quantity))


@dataclass
class Position:
    """
    ポジション情報（FIFO会計対応強化版）

    買いロットキューを使用してFIFO原則を厳密に適用
    """

    symbol: str
    quantity: int
    average_price: Decimal
    total_cost: Decimal
    current_price: Decimal = Decimal("0")

    # FIFO会計のための買いロットキュー
    buy_lots: Deque[BuyLot] = None

    def __post_init__(self):
        """初期化後の処理"""
        if self.buy_lots is None:
            self.buy_lots = deque()

    @property
    def market_value(self) -> Decimal:
        """時価総額"""
        return self.current_price * Decimal(self.quantity)

    @property
    def unrealized_pnl(self) -> Decimal:
        """含み損益"""
        return self.market_value - self.total_cost

    @property
    def unrealized_pnl_percent(self) -> Decimal:
        """含み損益率"""
        if self.total_cost == 0:
            return Decimal("0")
        return (self.unrealized_pnl / self.total_cost) * 100

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
            "unrealized_pnl_percent": str(
                self.unrealized_pnl_percent.quantize(Decimal("0.01"))
            ),
        }


@dataclass
class RealizedPnL:
    """実現損益"""

    symbol: str
    quantity: int
    buy_price: Decimal
    sell_price: Decimal
    buy_commission: Decimal
    sell_commission: Decimal
    pnl: Decimal
    pnl_percent: Decimal
    buy_date: datetime
    sell_date: datetime

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "buy_price": str(self.buy_price),
            "sell_price": str(self.sell_price),
            "buy_commission": str(self.buy_commission),
            "sell_commission": str(self.sell_commission),
            "pnl": str(self.pnl),
            "pnl_percent": str(self.pnl_percent.quantize(Decimal("0.01"))),
            "buy_date": self.buy_date.isoformat(),
            "sell_date": self.sell_date.isoformat(),
        }