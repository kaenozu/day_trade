"""
シグナル関連の基本データ型定義
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional


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