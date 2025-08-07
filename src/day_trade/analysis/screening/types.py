"""
スクリーニング関連の型定義とデータクラス

3つの重複したScreenerファイルから共通部分を統合
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ScreenerCondition(Enum):
    """スクリーニング条件の種類"""

    RSI_OVERSOLD = "rsi_oversold"
    RSI_OVERBOUGHT = "rsi_overbought"
    MACD_BULLISH = "macd_bullish"
    MACD_BEARISH = "macd_bearish"
    GOLDEN_CROSS = "golden_cross"
    DEAD_CROSS = "dead_cross"
    BOLLINGER_SQUEEZE = "bollinger_squeeze"
    BOLLINGER_BREAKOUT = "bollinger_breakout"
    VOLUME_SPIKE = "volume_spike"
    PRICE_NEAR_SUPPORT = "price_near_support"
    PRICE_NEAR_RESISTANCE = "price_near_resistance"
    STRONG_MOMENTUM = "strong_momentum"
    REVERSAL_PATTERN = "reversal_pattern"


@dataclass
class ScreenerCriteria:
    """スクリーニング基準"""

    condition: ScreenerCondition
    threshold: Optional[float] = None
    lookback_days: int = 20
    weight: float = 1.0
    description: str = ""


@dataclass
class ScreenerResult:
    """スクリーニング結果"""

    symbol: str
    score: float
    matched_conditions: List[ScreenerCondition]
    technical_data: Dict[str, Any]
    signal_data: Optional[Dict[str, Any]] = None
    last_price: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[float] = None


@dataclass
class ScreeningReport:
    """スクリーニングレポート"""

    total_screened: int
    passed_criteria: int
    results: List[ScreenerResult]
    screening_time: float
    criteria_used: List[ScreenerCriteria]
    summary: Dict[str, Any]
