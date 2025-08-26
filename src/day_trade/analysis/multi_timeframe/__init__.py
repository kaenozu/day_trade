#!/usr/bin/env python3
"""
マルチタイムフレーム分析システム パッケージ
Issue #315: 高度テクニカル指標・ML機能拡張

複数時間軸（日足・週足・月足）を統合した包括的トレンド分析システム
"""

# 後方互換性のためのメインクラス再エクスポート
from .main import MultiTimeframeAnalyzer

# 個別コンポーネントのエクスポート
from .integrated_analyzer import IntegratedAnalyzer
from .risk_evaluator import RiskEvaluator
from .technical_calculator import TechnicalCalculator
from .timeframe_resampler import TimeframeResampler
from .trend_analyzer import TrendAnalyzer

__all__ = [
    "MultiTimeframeAnalyzer",
    "TimeframeResampler", 
    "TechnicalCalculator",
    "TrendAnalyzer",
    "IntegratedAnalyzer",
    "RiskEvaluator",
]

# パッケージ情報
__version__ = "1.0.0"
__author__ = "Day Trading System"
__description__ = "マルチタイムフレーム分析システム - 複数時間軸統合トレンド分析"