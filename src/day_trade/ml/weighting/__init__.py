#!/usr/bin/env python3
"""
Dynamic Weighting System Package

動的重み調整システムのメインパッケージ
元の dynamic_weighting_system.py の機能を分割して提供

バックワード互換性:
- DynamicWeightingSystem
- DynamicWeightingConfig  
- MarketRegime
- PerformanceWindow

新しいモジュラー構成:
- system: メインシステムクラス
- core: 設定とデータ構造
- weight_calculator: 重み計算アルゴリズム
- weight_optimizer: 重み制約・最適化
- market_regime: 市場状態検出
- performance: パフォーマンス管理
"""

# バックワード互換性のための主要クラスのインポート
from .system import DynamicWeightingSystem
from .core import (
    DynamicWeightingConfig,
    MarketRegime, 
    PerformanceWindow,
    WeightingState
)

# 各種サブシステムクラス
from .market_regime import MarketRegimeDetector
from .weight_calculator import WeightCalculator  
from .weight_optimizer import WeightOptimizer
from .constraint_manager import WeightConstraintManager
from .performance import PerformanceManager
from .data_validator import DataValidator
from .visualization import WeightVisualization

# 便利な関数とユーティリティ
def create_default_config(**kwargs) -> DynamicWeightingConfig:
    """
    デフォルト設定を作成

    Args:
        **kwargs: 設定オーバーライドパラメータ

    Returns:
        DynamicWeightingConfig インスタンス
    """
    return DynamicWeightingConfig(**kwargs)

def create_system(model_names: list, 
                 config: DynamicWeightingConfig = None) -> DynamicWeightingSystem:
    """
    システムを簡単に作成するファクトリ関数

    Args:
        model_names: モデル名のリスト
        config: 設定（省略時はデフォルト）

    Returns:
        DynamicWeightingSystem インスタンス
    """
    if config is None:
        config = create_default_config()
    
    return DynamicWeightingSystem(model_names, config)

def validate_model_names(model_names) -> bool:
    """
    モデル名の妥当性チェック

    Args:
        model_names: チェック対象のモデル名リスト

    Returns:
        妥当性（True/False）
    """
    if not isinstance(model_names, (list, tuple)):
        return False
    
    if len(model_names) < 2:
        return False
        
    if len(set(model_names)) != len(model_names):
        return False  # 重複チェック
        
    return all(isinstance(name, str) and len(name.strip()) > 0 
               for name in model_names)

# パッケージメタ情報
__version__ = "1.0.0"
__author__ = "Day Trade Sub Team"
__description__ = "Dynamic weighting system for ensemble learning"

# モジュール公開リスト
__all__ = [
    # メインクラス（バックワード互換性）
    'DynamicWeightingSystem',
    'DynamicWeightingConfig', 
    'MarketRegime',
    'PerformanceWindow',
    'WeightingState',
    
    # サブシステムクラス
    'MarketRegimeDetector',
    'WeightCalculator',
    'WeightOptimizer', 
    'WeightConstraintManager',
    'PerformanceManager',
    'DataValidator',
    'WeightVisualization',
    
    # ユーティリティ関数
    'create_default_config',
    'create_system',
    'validate_model_names'
]

# パッケージ初期化時のチェック
def _check_dependencies():
    """必要な依存関係をチェック"""
    try:
        import numpy
        import pandas  
        from collections import deque
        from dataclasses import dataclass
        from enum import Enum
        return True
    except ImportError as e:
        import warnings
        warnings.warn(f"Dynamic Weighting System: 依存関係の問題 - {e}")
        return False

# 依存関係チェック実行
_dependencies_ok = _check_dependencies()

if not _dependencies_ok:
    import warnings
    warnings.warn(
        "Dynamic Weighting System: いくつかの機能が制限される可能性があります"
    )