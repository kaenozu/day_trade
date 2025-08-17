#!/usr/bin/env python3
"""
advanced_technical_analyzer.py - AdvancedAnalysis

リファクタリングにより分割されたモジュール
"""

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
