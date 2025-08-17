#!/usr/bin/env python3
"""
advanced_technical_analysis.py - AnalysisType

リファクタリングにより分割されたモジュール
"""

class AnalysisType(Enum):
    """分析タイプ"""
    TREND_ANALYSIS = "trend_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    VOLUME_ANALYSIS = "volume_analysis"
    VOLATILITY_ANALYSIS = "volatility_analysis"
    MOMENTUM_ANALYSIS = "momentum_analysis"
    CYCLE_ANALYSIS = "cycle_analysis"
    FRACTAL_ANALYSIS = "fractal_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
