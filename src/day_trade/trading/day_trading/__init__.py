#!/usr/bin/env python3
"""
デイトレードエンジンパッケージ

後方互換性のために元の day_trading_engine.py の全てのクラスと関数を再エクスポートします。
"""

# 各モジュールから必要なクラス・関数をインポート
from .enums import DayTradingSignal, TradingSession
from .recommendation import DayTradingRecommendation
from .data_handlers import SymbolManager, MarketDataHandler, FeaturePreparator
from .signal_analysis import TradingSessionAnalyzer, SignalAnalyzer, RealDataConverter
from .forecaster import TomorrowForecaster
from .core import PersonalDayTradingEngine, create_day_trading_engine

# 後方互換性のためのエクスポート
__all__ = [
    # Enums
    'DayTradingSignal',
    'TradingSession',
    
    # Data classes
    'DayTradingRecommendation',
    
    # Data handlers
    'SymbolManager',
    'MarketDataHandler',
    'FeaturePreparator',
    
    # Signal analysis
    'TradingSessionAnalyzer',
    'SignalAnalyzer',
    'RealDataConverter',
    
    # Forecaster
    'TomorrowForecaster',
    
    # Core engine
    'PersonalDayTradingEngine',
    'create_day_trading_engine',
]

# バージョン情報
__version__ = "2.0.0"

# パッケージ情報
__description__ = "デイトレードエンジン - モジュール化版"
__author__ = "Day Trading Engine Team"