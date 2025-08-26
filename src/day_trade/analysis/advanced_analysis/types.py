#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Technical Analysis Types - 高度技術分析型定義

データクラス、Enum、型定義を含む
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


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


@dataclass
class TechnicalSignal:
    """技術シグナル"""
    indicator_name: str
    signal_type: str  # "BUY", "SELL", "HOLD"
    strength: float  # 0-100
    confidence: float  # 0-1
    timeframe: str
    description: str
    timestamp: datetime


@dataclass
class PatternMatch:
    """パターンマッチ結果"""
    pattern_name: str
    match_score: float
    start_index: int
    end_index: int
    pattern_type: str  # "continuation", "reversal"
    reliability: float
    target_price: Optional[float] = None


@dataclass
class TechnicalAnalysisResult:
    """技術分析結果"""
    symbol: str
    analysis_type: AnalysisType
    signals: List[TechnicalSignal]
    patterns: List[PatternMatch]
    indicators: dict[str, float]
    overall_sentiment: str
    confidence_score: float
    risk_level: str
    recommendations: List[str]
    timestamp: datetime