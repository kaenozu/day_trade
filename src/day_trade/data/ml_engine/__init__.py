#!/usr/bin/env python3
"""
ML Engine Module - 機械学習エンジンパッケージ
Advanced ML Engine を機能別に分割したモジュール群

このパッケージは元々のadvanced_ml_engine.pyを
以下の機能別モジュールに分割して提供します：

- advanced_config: 設定クラスとデータクラス
- pytorch_models: PyTorchニューラルネットワークモデル  
- feature_engineering: 特徴量エンジニアリング機能
- advanced_engine: メインMLエンジン
- advanced_engine_helpers: MLエンジンヘルパーメソッド
- next_gen_trading_engine: 次世代AI取引エンジン
- factory_functions: ファクトリ関数

バックワード互換性のため、元のクラスと関数を
全て同一インターフェースで提供します。
"""

# 基本設定とデータ構造
from .advanced_config import (
    ModelConfig,
    PredictionResult,
)

# PyTorchモデル
from .pytorch_models import (
    LSTMTransformerHybrid,
    PositionalEncoding,
    PYTORCH_AVAILABLE,
)

# 特徴量エンジニアリング
from .feature_engineering import (
    engineer_features,
    create_sequences,
    extract_fft_features,
    extract_fft_features_async,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
)

# メインエンジン
from .advanced_engine import (
    AdvancedMLEngine,
)

# ヘルパーメソッド
from .advanced_engine_helpers import (
    AdvancedMLEngineHelpers,
)

# 次世代エンジン
from .next_gen_trading_engine import (
    NextGenAITradingEngine,
)

# ファクトリ関数
from .factory_functions import (
    create_advanced_ml_engine,
    create_next_gen_engine,
)

# パッケージ情報
__version__ = "2.0.0"
__author__ = "Day Trade System"
__description__ = "Advanced ML Engine - 機械学習エンジンパッケージ"

# パッケージレベルのエクスポート
__all__ = [
    # 設定関連
    "ModelConfig",
    "PredictionResult",
    
    # モデル関連
    "LSTMTransformerHybrid",
    "PositionalEncoding",
    "PYTORCH_AVAILABLE",
    
    # 特徴量エンジニアリング関連
    "engineer_features",
    "create_sequences",
    "extract_fft_features",
    "extract_fft_features_async",
    "calculate_rsi",
    "calculate_macd",
    "calculate_bollinger_bands",
    
    # エンジン関連
    "AdvancedMLEngine",
    "AdvancedMLEngineHelpers",
    "NextGenAITradingEngine",
    "create_advanced_ml_engine",
    "create_next_gen_engine",
]


# 初期化時の警告
import warnings
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# PyTorchの可用性チェック
if not PYTORCH_AVAILABLE:
    logger.warning(
        "PyTorch が利用できません。機械学習機能は制限モードで動作します。"
    )
    warnings.warn(
        "PyTorch is not available. ML features will run in limited mode.",
        UserWarning
    )

# 初期化完了ログ
logger.info(f"ML Engine Package 初期化完了 (version: {__version__})")