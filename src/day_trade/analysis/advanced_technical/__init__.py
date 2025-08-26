#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Technical Analysis Package - 高度技術分析パッケージ

後方互換性を保つため、元のクラス・関数を再エクスポート
"""

# メインクラスのインポート
from .main import AdvancedTechnicalAnalyzer, test_advanced_technical_analyzer

# データタイプ・列挙型のインポート
from .types_and_enums import (
    TechnicalIndicatorType,
    SignalStrength,
    TechnicalSignal,
    AdvancedAnalysis
)

# 各種計算器のインポート（必要に応じて使用可能）
from .data_provider import AdvancedDataProvider
from .trend_indicators import TrendIndicatorCalculator
from .momentum_indicators import MomentumIndicatorCalculator
from .volatility_volume_indicators import VolatilityVolumeCalculator
from .analysis_engine import AnalysisEngine
from .pattern_ml_features import PatternMLProcessor

# 後方互換性のため、元のクラス名でもアクセス可能
__all__ = [
    # メインクラス
    'AdvancedTechnicalAnalyzer',
    'test_advanced_technical_analyzer',
    
    # データタイプ
    'TechnicalIndicatorType',
    'SignalStrength', 
    'TechnicalSignal',
    'AdvancedAnalysis',
    
    # 計算器クラス（オプション）
    'AdvancedDataProvider',
    'TrendIndicatorCalculator',
    'MomentumIndicatorCalculator',
    'VolatilityVolumeCalculator',
    'AnalysisEngine',
    'PatternMLProcessor',
]