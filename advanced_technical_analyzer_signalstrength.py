#!/usr/bin/env python3
"""
advanced_technical_analyzer.py - SignalStrength

リファクタリングにより分割されたモジュール
"""

class SignalStrength(Enum):
    """シグナル強度"""
    VERY_STRONG = "非常に強い"
    STRONG = "強い"
    MODERATE = "中程度"
    WEAK = "弱い"
    NEUTRAL = "中立"
