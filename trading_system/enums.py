from enum import Enum


class PredictionConfidence(Enum):
    """予測信頼度"""
    VERY_HIGH = "非常に高い"    # 90%以上
    HIGH = "高い"             # 80-89%
    MEDIUM = "中程度"         # 60-79%
    LOW = "低い"              # 40-59%
    VERY_LOW = "非常に低い"    # 40%未満


class MarketDirection(Enum):
    """市場方向"""
    STRONG_BULLISH = "強い上昇"      # +2%以上
    BULLISH = "上昇"               # +0.5-2%
    NEUTRAL = "中立"               # -0.5-+0.5%
    BEARISH = "下降"               # -2--0.5%
    STRONG_BEARISH = "強い下降"     # -2%未満


class RiskLevel(Enum):
    """リスクレベル"""
    VERY_LOW = "極低"     # 1-3%
    LOW = "低"           # 3-5%
    MEDIUM = "中"        # 5-8%
    HIGH = "高"          # 8-12%
    VERY_HIGH = "極高"    # 12%以上
