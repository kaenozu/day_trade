#!/usr/bin/env python3
"""
advanced_technical_analysis.py - TechnicalSignal

リファクタリングにより分割されたモジュール
"""

class TechnicalSignal:
    """技術シグナル"""
    indicator_name: str
    signal_type: str  # "BUY", "SELL", "HOLD"
    strength: float  # 0-100
    confidence: float  # 0-1
    timeframe: str
    description: str
    timestamp: datetime
