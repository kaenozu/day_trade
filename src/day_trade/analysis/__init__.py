# Technical analysis module
from .indicators import TechnicalIndicators
from .patterns import ChartPatternRecognizer
from .signals import (
    SignalStrength,
    SignalType,
    TradingSignal,
    TradingSignalGenerator,
    VolumeSpikeBuyRule,
)

# Issue #937: 売買判断システムの実装
from .technical_indicators import (
    TechnicalIndicators as TechnicalIndicatorsV2,
    RiskManager,
    TechnicalSignal,
    SignalType as SignalTypeV2,
    create_trading_recommendation_pl as create_trading_recommendation
)

__all__ = [
    "TechnicalIndicators",
    "ChartPatternRecognizer",
    "TradingSignalGenerator",
    "SignalType",
    "SignalStrength",
    "TradingSignal",
    "VolumeSpikeBuyRule",
    # Issue #937対応
    "TechnicalIndicatorsV2",
    "RiskManager",
    "TechnicalSignal",
    "SignalTypeV2",
    "create_trading_recommendation"
]
