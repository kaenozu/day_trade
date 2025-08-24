"""
基本型定義

売買シグナルシステムの基本的な型定義（Enum、dataclass）
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


class SignalType(Enum):
    """シグナルタイプ"""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class SignalStrength(Enum):
    """シグナル強度"""

    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"


@dataclass
class TradingSignal:
    """売買シグナル情報"""

    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0-100
    reasons: List[str]
    conditions_met: Dict[str, bool]
    timestamp: datetime
    price: Decimal
    symbol: Optional[str] = None  # 銘柄コード

    def __post_init__(self):
        """データクラス初期化後の検証"""
        if not 0 <= self.confidence <= 100:
            raise ValueError(f"信頼度は0-100の範囲である必要があります: {self.confidence}")
        
        if self.price <= 0:
            raise ValueError(f"価格は正の値である必要があります: {self.price}")
        
        if not self.reasons:
            raise ValueError("シグナル理由は空であってはいけません")

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "signal_type": self.signal_type.value,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "conditions_met": self.conditions_met,
            "timestamp": self.timestamp.isoformat(),
            "price": float(self.price),
            "symbol": self.symbol,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingSignal":
        """辞書から作成"""
        return cls(
            signal_type=SignalType(data["signal_type"]),
            strength=SignalStrength(data["strength"]),
            confidence=data["confidence"],
            reasons=data["reasons"],
            conditions_met=data["conditions_met"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            price=Decimal(str(data["price"])),
            symbol=data.get("symbol"),
        )