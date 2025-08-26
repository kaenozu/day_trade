#!/usr/bin/env python3
"""
Advanced ML Engine Package

高度なML予測エンジンのモジュラー化パッケージ
後方互換性を保つため、元のクラス・関数を再エクスポート
"""

# 設定とデータクラス
from .config import ModelConfig, PredictionResult

# コアエンジン機能
from .core_engine import AdvancedMLEngineCore

# 特徴量エンジニアリング
from .feature_engineering import FeatureEngineer

# メインのMLエンジン
from .ml_engine import AdvancedMLEngine

# PyTorchモデル（条件付きインポート）
try:
    from .models import LSTMTransformerHybrid, PositionalEncoding
except ImportError:
    # PyTorchが利用できない場合はダミークラス
    class LSTMTransformerHybrid:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("PyTorchが必要です")
    
    class PositionalEncoding:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("PyTorchが必要です")

# 次世代AIエンジン
from .next_gen_engine import NextGenAITradingEngine

# 性能評価器
from .performance_evaluator import PerformanceEvaluator

# テクニカル指標計算器
from .technical_indicators import TechnicalIndicatorCalculator

# ユーティリティ関数
from .utils import (
    benchmark_model_performance,
    create_advanced_ml_engine,
    create_next_gen_engine,
    extract_fft_features_async,
    get_recommended_config,
    measure_inference_time_async,
    optimize_config_for_hardware,
    validate_model_config,
)

# 後方互換性のため、元のファイルからのインポートを模倣
__all__ = [
    # 設定とデータクラス
    "ModelConfig",
    "PredictionResult",
    
    # PyTorchモデル
    "LSTMTransformerHybrid",
    "PositionalEncoding",
    
    # コアエンジン
    "AdvancedMLEngineCore",
    
    # メインのMLエンジン
    "AdvancedMLEngine",
    
    # 次世代AIエンジン
    "NextGenAITradingEngine",
    
    # 特徴量エンジニアリング
    "FeatureEngineer",
    
    # 性能評価器
    "PerformanceEvaluator",
    
    # テクニカル指標計算器
    "TechnicalIndicatorCalculator",
    
    # ファクトリ関数
    "create_advanced_ml_engine",
    "create_next_gen_engine",
    
    # ユーティリティ関数
    "validate_model_config",
    "optimize_config_for_hardware", 
    "get_recommended_config",
    "benchmark_model_performance",
    "extract_fft_features_async",
    "measure_inference_time_async",
]

# バージョン情報
__version__ = "2.0.0"
__author__ = "Day Trade Sub Team"
__description__ = "Advanced ML Engine - モジュラー化版"

# パッケージメタデータ
PACKAGE_INFO = {
    "name": "advanced_ml",
    "version": __version__,
    "description": __description__,
    "modules": {
        "config": "設定とデータクラス",
        "models": "PyTorchモデル（LSTM-Transformer）",
        "core_engine": "コアML予測エンジン機能",
        "ml_engine": "メインML予測エンジン",
        "next_gen_engine": "次世代ハイブリッドAIエンジン",
        "feature_engineering": "特徴量エンジニアリング",
        "performance_evaluator": "性能評価器",
        "technical_indicators": "テクニカル指標計算器",
        "utils": "ユーティリティ関数群",
    },
    "total_files": 9,
    "original_file": "advanced_ml_engine.py (1792行)",
    "modularized": "9ファイルに分割、各300行以下",
}