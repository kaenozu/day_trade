#!/usr/bin/env python3
"""
advanced_technical_analyzer.py - TechnicalSignal

リファクタリングにより分割されたモジュール
"""

class TechnicalSignal:
    """技術シグナル"""
    indicator_name: str
    signal_type: str           # BUY, SELL, HOLD
    strength: SignalStrength
    confidence: float          # 0-100
    price_level: float        # シグナル発生価格
    timestamp: datetime

    # 詳細情報
    indicator_value: float
    threshold_upper: Optional[float] = None
    threshold_lower: Optional[float] = None
    trend_direction: str = "NEUTRAL"
    momentum: float = 0.0

    # 予測情報
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    time_horizon: str = "短期"
    risk_level: str = "中"
