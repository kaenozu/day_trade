#!/usr/bin/env python3
"""
投資機会アラートシステム - 列挙型定義
"""

from enum import Enum


class OpportunityType(Enum):
    """投資機会タイプ"""

    TECHNICAL_BREAKOUT = "technical_breakout"
    MOMENTUM_SIGNAL = "momentum_signal"
    REVERSAL_PATTERN = "reversal_pattern"
    VOLUME_ANOMALY = "volume_anomaly"
    PRICE_UNDERVALUATION = "price_undervaluation"
    EARNINGS_SURPRISE = "earnings_surprise"
    DIVIDEND_OPPORTUNITY = "dividend_opportunity"
    SECTOR_ROTATION = "sector_rotation"
    PAIRS_TRADING = "pairs_trading"
    ARBITRAGE_OPPORTUNITY = "arbitrage_opportunity"
    NEWS_SENTIMENT = "news_sentiment"
    VOLATILITY_SQUEEZE = "volatility_squeeze"


class OpportunitySeverity(Enum):
    """機会重要度"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TradingAction(Enum):
    """トレーディングアクション"""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REDUCE = "reduce"
    INCREASE = "increase"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class TimeHorizon(Enum):
    """投資期間"""

    INTRADAY = "intraday"
    SHORT_TERM = "short_term"  # 1週間以内
    MEDIUM_TERM = "medium_term"  # 1ヶ月以内
    LONG_TERM = "long_term"  # 3ヶ月以上