#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Monitor Package
モデル性能監視パッケージ

後方互換性を保持するため、元の大きなファイルと同じインターフェースを提供します。
"""

# 新しいモジュール化されたクラスをインポート
from .enums_and_models import (
    PerformanceStatus,
    AlertLevel,
    RetrainingScope,
    PerformanceMetrics,
    PerformanceAlert,
    RetrainingTrigger,
    RetrainingResult
)

from .config_manager import EnhancedPerformanceConfigManager
from .symbol_manager import DynamicSymbolManager
from .retraining_manager import GranularRetrainingManager
from .database_manager import DatabaseManager
from .performance_evaluator import PerformanceEvaluator
from .alert_manager import AlertManager
from .main_monitor import (
    ModelPerformanceMonitor,
    create_enhanced_performance_monitor,
    test_model_performance_monitor
)

# 後方互換性のために元のクラス名でエクスポート
__all__ = [
    # Enums and Models
    'PerformanceStatus',
    'AlertLevel', 
    'RetrainingScope',
    'PerformanceMetrics',
    'PerformanceAlert',
    'RetrainingTrigger',
    'RetrainingResult',
    
    # Core Classes
    'EnhancedPerformanceConfigManager',
    'DynamicSymbolManager',
    'GranularRetrainingManager', 
    'DatabaseManager',
    'PerformanceEvaluator',
    'AlertManager',
    'ModelPerformanceMonitor',
    
    # Utility Functions
    'create_enhanced_performance_monitor',
    'test_model_performance_monitor'
]

# バージョン情報
__version__ = "2.0.0"
__description__ = "Modularized Model Performance Monitor"