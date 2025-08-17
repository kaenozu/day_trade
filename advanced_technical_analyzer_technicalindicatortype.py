#!/usr/bin/env python3
"""
advanced_technical_analyzer.py - TechnicalIndicatorType

リファクタリングにより分割されたモジュール
"""

class TechnicalIndicatorType(Enum):
    """技術指標タイプ"""
    TREND = "トレンド系"
    MOMENTUM = "モメンタム系"
    VOLATILITY = "ボラティリティ系"
    VOLUME = "出来高系"
    COMPOSITE = "複合指標"
    STATISTICAL = "統計系"
    PATTERN = "パターン認識"
    ML_BASED = "機械学習系"
