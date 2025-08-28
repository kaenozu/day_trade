#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Technical Analysis Base Components
基本データクラスとEnum定義
"""

from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)


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


class SignalStrength(Enum):
    """シグナル強度"""
    VERY_STRONG = "非常に強い"
    STRONG = "強い"
    MODERATE = "中程度"
    WEAK = "弱い"
    NEUTRAL = "中立"


@dataclass
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


@dataclass
class AdvancedAnalysis:
    """高度分析結果"""
    symbol: str
    analysis_time: datetime

    # 基本情報
    current_price: float
    price_change: float
    volume_ratio: float

    # 技術指標群
    trend_indicators: Dict[str, float]
    momentum_indicators: Dict[str, float]
    volatility_indicators: Dict[str, float]
    volume_indicators: Dict[str, float]

    # 複合分析
    composite_score: float      # 総合スコア(0-100)
    trend_strength: float       # トレンド強度(-100~100)
    momentum_score: float       # モメンタムスコア(-100~100)
    volatility_regime: str      # ボラティリティ局面

    # シグナル
    primary_signals: List[TechnicalSignal]
    secondary_signals: List[TechnicalSignal]

    # 統計分析
    statistical_profile: Dict[str, float]
    anomaly_score: float       # 異常度スコア

    # 機械学習予測
    ml_prediction: Optional[Dict[str, Any]] = None
    pattern_recognition: Optional[Dict[str, Any]] = None