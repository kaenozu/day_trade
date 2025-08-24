"""
取引関連の型定義
"""

from enum import Enum


class TradeType(Enum):
    """取引タイプ"""

    BUY = "buy"
    SELL = "sell"


class TradeStatus(Enum):
    """取引ステータス"""

    PENDING = "pending"  # 注文中
    EXECUTED = "executed"  # 約定済み
    CANCELLED = "cancelled"  # キャンセル
    PARTIAL = "partial"  # 一部約定