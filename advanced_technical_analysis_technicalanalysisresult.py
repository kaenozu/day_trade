#!/usr/bin/env python3
"""
advanced_technical_analysis.py - TechnicalAnalysisResult

リファクタリングにより分割されたモジュール
"""

class TechnicalAnalysisResult:
    """技術分析結果"""
    symbol: str
    analysis_type: AnalysisType
    signals: List[TechnicalSignal]
    patterns: List[PatternMatch]
    indicators: Dict[str, float]
    overall_sentiment: str
    confidence_score: float
    risk_level: str
    recommendations: List[str]
    timestamp: datetime
