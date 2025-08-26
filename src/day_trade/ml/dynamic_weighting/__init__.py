#!/usr/bin/env python3
"""
Dynamic Weighting System Package

動的重み調整システムの統合パッケージ
後方互換性を確保するため、元のクラス・関数を再エクスポート
"""

# 主要クラスのインポート
from .dynamic_weighting_system import DynamicWeightingSystem
from .core import (
    DynamicWeightingConfig,
    MarketRegime,
    PerformanceWindow,
    get_default_regime_adjustments,
    create_scoring_explanation
)
from .weighting_algorithms import WeightingAlgorithms
from .performance_manager import PerformanceManager
from .weight_constraints import WeightConstraintManager
from .market_regime_detector import MarketRegimeDetector

# 後方互換性のため、元の名前でエクスポート
__all__ = [
    # メインシステム
    'DynamicWeightingSystem',
    
    # 設定とデータクラス
    'DynamicWeightingConfig',
    'MarketRegime', 
    'PerformanceWindow',
    
    # アルゴリズムクラス
    'WeightingAlgorithms',
    'PerformanceManager',
    'WeightConstraintManager',
    'MarketRegimeDetector',
    
    # ユーティリティ関数
    'get_default_regime_adjustments',
    'create_scoring_explanation'
]

# バージョン情報
__version__ = "2.0.0"
__author__ = "Day Trade System Team"
__description__ = "動的重み調整システム - モジュラー化版"