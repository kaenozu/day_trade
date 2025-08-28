#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Technical Analysis Module
技術分析モジュール - 後方互換性付きインポート
"""

# 基本クラス
from .base import (
    TechnicalIndicatorType,
    SignalStrength,
    TechnicalSignal,
    AdvancedAnalysis
)

# 各種指標クラス
from .trend_indicators import TrendIndicators
from .momentum_indicators import MomentumIndicators
from .volatility_indicators import VolatilityIndicators
from .volume_indicators import VolumeIndicators
from .statistical_analysis import StatisticalAnalysis
from .ml_prediction import MLPredictor
from .data_provider import TechnicalDataProvider
from .signal_generator import SignalGenerator

# メインアナライザークラス
from .analyzer import AdvancedTechnicalAnalyzer

# 後方互換性のために全てのクラスを公開
__all__ = [
    # 基本クラス
    'TechnicalIndicatorType',
    'SignalStrength', 
    'TechnicalSignal',
    'AdvancedAnalysis',
    
    # 各種指標クラス
    'TrendIndicators',
    'MomentumIndicators',
    'VolatilityIndicators',
    'VolumeIndicators',
    'StatisticalAnalysis',
    'MLPredictor',
    'TechnicalDataProvider',
    'SignalGenerator',
    
    # メインクラス
    'AdvancedTechnicalAnalyzer',
]

# 後方互換性のためのエイリアス（元のクラス名でもアクセス可能）
# 元のファイルから直接インポートした場合と同じ動作を保証