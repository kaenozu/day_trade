#!/usr/bin/env python3
"""
Advanced ML Engine - Modular Version
後方互換性維持のためのメインモジュール

元のadvanced_ml_engine.pyとの完全な後方互換性を提供します。
すべての機能は新しいモジュラー構造からインポートされます。
"""

# 分割されたモジュールから必要なクラス・関数をすべてインポート
from .ml_engine import (
    # 設定とデータ構造
    ModelConfig,
    PredictionResult,
    
    # PyTorchモデル
    LSTMTransformerHybrid,
    PositionalEncoding,
    PYTORCH_AVAILABLE,
    
    # 特徴量エンジニアリング
    engineer_features,
    create_sequences,
    extract_fft_features,
    extract_fft_features_async,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    
    # メインエンジン
    AdvancedMLEngine,
    AdvancedMLEngineHelpers,
    
    # 次世代エンジン
    NextGenAITradingEngine,
    
    # ファクトリ関数
    create_advanced_ml_engine,
    create_next_gen_engine,
)

# 元のadvanced_ml_engine.pyから使用されていたすべてのシンボルを公開
__all__ = [
    # データクラス
    "ModelConfig",
    "PredictionResult",
    
    # PyTorchモデル
    "LSTMTransformerHybrid",
    "PositionalEncoding",
    
    # エンジンクラス
    "AdvancedMLEngine",
    "NextGenAITradingEngine",
    
    # ヘルパークラス
    "AdvancedMLEngineHelpers",
    
    # ファクトリ関数
    "create_advanced_ml_engine",
    "create_next_gen_engine",
    
    # 特徴量エンジニアリング関数
    "engineer_features",
    "create_sequences",
    "extract_fft_features",
    "extract_fft_features_async",
    
    # テクニカル指標計算関数
    "calculate_rsi",
    "calculate_macd", 
    "calculate_bollinger_bands",
    
    # 定数
    "PYTORCH_AVAILABLE",
]

# バージョン情報
__version__ = "2.0.0-modular"
__description__ = "Advanced ML Engine - Modular Version for Backward Compatibility"

# ログ出力
from .utils.logging_config import get_context_logger

logger = get_context_logger(__name__)
logger.info(f"Advanced ML Engine Modular Version {__version__} loaded successfully")
logger.info("All original functionality is available through the modular architecture")