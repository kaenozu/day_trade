#!/usr/bin/env python3
"""
高度テクニカル指標システム データ構造定義
Issue #315: 高度テクニカル指標・ML機能拡張

データクラス定義:
- BollingerBandsAnalysis: Bollinger Bands分析結果
- IchimokuAnalysis: 一目均衡表分析結果  
- ComplexMAAnalysis: 複合移動平均分析結果
- FibonacciAnalysis: フィボナッチ分析結果
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class BollingerBandsAnalysis:
    """Bollinger Bands分析結果"""

    upper_band: float
    middle_band: float  # SMA
    lower_band: float
    current_price: float
    bb_position: float  # 0-1での位置 (0=下限, 1=上限)
    squeeze_ratio: float  # バンド幅比率（低い=スクイーズ）
    volatility_regime: str  # "low", "normal", "high"
    breakout_probability: float  # ブレイクアウト確率
    trend_strength: float  # トレンド強度
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 信頼度
    performance_score: float  # パフォーマンススコア


@dataclass
class IchimokuAnalysis:
    """一目均衡表分析結果"""

    tenkan_sen: float  # 転換線
    kijun_sen: float  # 基準線
    senkou_span_a: float  # 先行スパンA
    senkou_span_b: float  # 先行スパンB
    chikou_span: float  # 遅行スパン
    current_price: float
    cloud_thickness: float  # 雲の厚さ
    cloud_color: str  # "bullish", "bearish"
    price_vs_cloud: str  # "above", "in", "below"
    tk_cross: str  # "bullish", "bearish", "neutral"
    chikou_signal: str  # "bullish", "bearish", "neutral"
    overall_signal: str  # 総合判定
    trend_strength: float
    confidence: float
    performance_score: float


@dataclass
class ComplexMAAnalysis:
    """複合移動平均分析結果"""

    ma_5: float
    ma_25: float
    ma_75: float
    ma_200: float
    current_price: float
    ma_alignment: str  # "bullish", "bearish", "mixed"
    golden_cross: bool  # ゴールデンクロス
    death_cross: bool  # デッドクロス
    support_resistance: Dict[str, float]  # サポート・レジスタンスレベル
    trend_phase: str  # "accumulation", "markup", "distribution", "markdown"
    momentum_score: float  # モメンタムスコア
    signal: str
    confidence: float
    performance_score: float


@dataclass
class FibonacciAnalysis:
    """フィボナッチ分析結果"""

    retracement_levels: Dict[str, float]  # リトレースメントレベル
    extension_levels: Dict[str, float]  # エクステンションレベル
    current_level: str  # 現在の位置
    support_level: float  # 直近サポート
    resistance_level: float  # 直近レジスタンス
    signal: str
    confidence: float
    performance_score: float